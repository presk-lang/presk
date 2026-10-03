//! The reference's requirement "No recursion": no function calls itself,
//! directly or through other functions, and no module includes itself,
//! directly or through other modules (`E2016`). Checked once the calls are
//! resolved, before anything walks them: the analyses on PSK-Core and the
//! parameter trees of `presk dump --stage elab` recurse along calls and
//! `use` members, and overflowed the stack on such a program.

use std::collections::BTreeSet;

use presk_syntax::{Diagnostic, Span};

use crate::elab::{ElabProgram, FnId};
use crate::ty::{ModRef, ModuleId};

/// The modules a module reference names (through `stack` and `∘`).
fn named(r: &ModRef, out: &mut BTreeSet<u32>) {
    match r {
        ModRef::Inst { def, .. } => {
            out.insert(def.0);
        }
        ModRef::Stack { inner, .. } => named(inner, out),
        ModRef::Compose(parts) => parts.iter().for_each(|p| named(p, out)),
    }
}

/// The cycles of a graph of `n` nodes whose edges are `(from, to, span)`:
/// for each, the edges along it from its least node, the cycle closing at
/// that node. One cycle per strongly connected component.
fn cycles(n: usize, edges: &[(usize, usize, Span)]) -> Vec<Vec<(usize, usize, Span)>> {
    let out: Vec<Vec<&(usize, usize, Span)>> = (0..n).map(|v| edges.iter().filter(|e| e.0 == v).collect()).collect();
    // Tarjan's components, iteratively over an explicit stack.
    let (mut index, mut low, mut on) = (vec![usize::MAX; n], vec![0; n], vec![false; n]);
    let (mut next, mut stack, mut comps) = (0, Vec::new(), Vec::new());
    for root in 0..n {
        if index[root] != usize::MAX {
            continue;
        }
        let mut work = vec![(root, 0usize)];
        while let Some(&mut (v, ref mut i)) = work.last_mut() {
            if *i == 0 {
                index[v] = next;
                low[v] = next;
                next += 1;
                stack.push(v);
                on[v] = true;
            }
            if let Some(e) = out[v].get(*i) {
                *i += 1;
                let w = e.1;
                if index[w] == usize::MAX {
                    work.push((w, 0));
                } else if on[w] {
                    low[v] = low[v].min(index[w]);
                }
                continue;
            }
            work.pop();
            if let Some(&(u, _)) = work.last() {
                low[u] = low[u].min(low[v]);
            }
            if low[v] == index[v] {
                let mut comp = BTreeSet::new();
                while let Some(w) = stack.pop() {
                    on[w] = false;
                    comp.insert(w);
                    if w == v {
                        break;
                    }
                }
                comps.push(comp);
            }
        }
    }
    let mut found = Vec::new();
    for comp in comps {
        let start = *comp.iter().next().unwrap();
        if comp.len() == 1 && !out[start].iter().any(|e| e.1 == start) {
            continue;
        }
        // A path back to `start` inside the component, by breadth first.
        let mut prev: Vec<Option<&(usize, usize, Span)>> = vec![None; n];
        let mut queue = std::collections::VecDeque::from([start]);
        let mut seen = BTreeSet::from([start]);
        let mut last = None;
        'search: while let Some(v) = queue.pop_front() {
            for e in &out[v] {
                if !comp.contains(&e.1) {
                    continue;
                }
                if e.1 == start {
                    last = Some(*e);
                    break 'search;
                }
                if seen.insert(e.1) {
                    prev[e.1] = Some(*e);
                    queue.push_back(e.1);
                }
            }
        }
        let mut path = vec![*last.unwrap()];
        while let Some(e) = prev[path.last().unwrap().0] {
            path.push(*e);
        }
        path.reverse();
        found.push(path);
    }
    found
}

fn through(names: &[String]) -> String {
    match names {
        [] => String::new(),
        [one] => format!(" through `{one}`"),
        _ => format!(" through {}", names.iter().map(|n| format!("`{n}`")).collect::<Vec<_>>().join(", ")),
    }
}

/// A module that includes itself, directly or through other modules.
pub fn module_cycles(prog: &ElabProgram) -> Vec<Diagnostic> {
    let mut edges = Vec::new();
    for (i, m) in prog.modules.iter().enumerate() {
        for u in &m.uses {
            let mut to = BTreeSet::new();
            named(&u.module, &mut to);
            edges.extend(to.into_iter().map(|t| (i, t as usize, u.span)));
        }
    }
    cycles(prog.modules.len(), &edges)
        .into_iter()
        .map(|path| {
            let name = |m: usize| prog.module(ModuleId(m as u32)).name.clone();
            let first = name(path[0].0);
            let others: Vec<String> = path[1..].iter().map(|e| name(e.0)).collect();
            let mut d = Diagnostic::error("E2016", format!("module `{first}` includes itself{}", through(&others)));
            for (k, e) in path.iter().enumerate() {
                let msg = if k + 1 == path.len() { format!("`{}` includes `{first}` here", name(e.0)) } else { format!("`{}` includes `{}` here", name(e.0), name(e.1)) };
                d = d.label(e.2, msg);
            }
            d.note("a module may not include itself, directly or through other modules: repetition is `stack{N}(M)`")
        })
        .collect()
}

/// A function that calls itself, directly or through other functions; the
/// calls are `(caller, callee, call span)` as the checker resolved them.
pub(crate) fn fn_cycles(prog: &ElabProgram, calls: &[(FnId, FnId, Span)]) -> Vec<Diagnostic> {
    let edges: Vec<(usize, usize, Span)> = calls.iter().map(|(a, b, s)| (a.0 as usize, b.0 as usize, *s)).collect();
    let name = |f: usize| {
        let sig = prog.function(FnId(f as u32));
        match sig.owner {
            Some(m) => format!("{}.{}", prog.module(m).name, sig.name),
            None => sig.name.clone(),
        }
    };
    cycles(prog.fns.len(), &edges)
        .into_iter()
        .map(|path| {
            let first = name(path[0].0);
            let others: Vec<String> = path[1..].iter().map(|e| name(e.0)).collect();
            let mut d = Diagnostic::error("E2016", format!("`{first}` calls itself{}", through(&others)));
            for e in &path {
                d = d.label(e.2, format!("`{}` calls `{}` here", name(e.0), name(e.1)));
            }
            d.note("a function may not call itself, directly or through other functions: iteration is `stack{N}(M)`")
        })
        .collect()
}
