//! The provenance table of one compilation (plan 0.13.0 step 1, D134), put
//! together from what each stage returned beside its value: the lowering's
//! rules, differentiation's and the optimizer's links, the fusion groups,
//! the launches the kernel backend logged, the C export's kernel names. The
//! stages run as the commands run them; the table is what `presk explain`
//! reads, and draws.
//!
//! Every stage is the part of its program the entry reaches. A link whose
//! child the table does not have, a function the entry does not reach, is
//! left out; a link whose parent it does not have is kept, and the test of
//! statement P2 (`Table::dangling`) finds it.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use presk_core::ir::Rhs;
use presk_core::{FuncId, Program};
use presk_precision::envelope::Envelope;
use presk_precision::solve::{Role, Roles};
use presk_provenance::{End, How, Id, Link, Stage, Table};
use presk_syntax::{SourceMap, Span};
use presk_types::check::TypedProgram;
use presk_types::elab::ElabProgram;

/// The precision stages' values, as `presk explain` holds them.
pub struct Precision<'a> {
    /// The links of the differentiation the precision stages read, which
    /// keeps every fusion group, where the kernels' program keeps them only
    /// for a realization that can see them (D94): a site it alone has, a
    /// residual computed apart to keep a group, links through them to the
    /// `let` of PSK-Core it copies.
    pub links: &'a [Link],
    pub env: Option<&'a Envelope>,
    pub roles: &'a Roles,
    pub realization: &'a presk_rt::Realization,
    pub solved: Option<&'a crate::precision::Solved>,
}

/// The layout plan: each nest's chosen point and its space's size.
pub struct Layouts {
    pub nests: Vec<(String, String, usize)>,
}

/// What every stage returned, for one entry.
pub struct Stages<'a> {
    pub sm: &'a SourceMap,
    pub prog: &'a ElabProgram,
    pub typed: &'a TypedProgram,
    /// PSK-Core as lowered, and the lowering's rules.
    pub core: &'a Program,
    pub lowered: &'a presk_core::lower::Trace,
    /// The entry's name.
    pub entry: &'a str,
    /// The program after differentiation, before the optimizer, and
    /// differentiation's links.
    pub ad: &'a Program,
    pub ad_links: &'a [Link],
    pub precision: Option<Precision<'a>>,
    /// The program after the optimizer, its links and ends, under `--opt`.
    pub opt: Option<(&'a Program, &'a [Link], &'a [End])>,
    pub layouts: Option<Layouts>,
    /// The kernels and the launches of a call, when the kernels were listed.
    pub kernels: Option<(&'a [presk_kernel::Kernel], &'a [presk_kernel::Launched])>,
    /// The C program's kernels, when it was exported.
    pub c: Option<CKernels<'a>>,
    /// Each listed kernel's machine code, by its place in the listing, when
    /// it was asked for (plan 0.13.0 step 5).
    pub native: Option<&'a [presk_cranelift::MachineCode]>,
}

/// The C program's kernels: `(listing place, k<n>)`, and each `k<n>`'s
/// first and last lines in `kernels.c`.
pub type CKernels<'a> = (&'a [(usize, usize)], &'a [(usize, usize)]);

/// The key of a span: `<file>:<start>-<end>`.
pub fn span_key(sp: Span) -> String {
    format!("{}:{}-{}", sp.file.0, sp.start, sp.end)
}

/// The site `<function>/%<n>`.
fn site(f: &presk_core::Func, v: presk_core::VarId) -> String {
    format!("{}/%{}", f.name, v.0)
}

/// The source text of a span, its first line, at most 72 characters.
fn snippet(sm: &SourceMap, sp: Span) -> String {
    let text = sm.file(sp.file).text.get(sp.start as usize..sp.end as usize).unwrap_or("");
    let line = text.lines().next().unwrap_or("").trim();
    let mut out: String = line.chars().take(72).collect();
    if line.chars().count() > 72 || text.lines().count() > 1 {
        out.push('…');
    }
    out
}

/// The functions `entry` reaches in `p`, itself first, then in the
/// program's order.
pub fn reached(p: &Program, entry: &str) -> Vec<FuncId> {
    let Some(e) = p.find(entry) else { return Vec::new() };
    let mut seen: BTreeSet<FuncId> = BTreeSet::new();
    let mut stack = vec![e];
    while let Some(f) = stack.pop() {
        if !seen.insert(f) {
            continue;
        }
        for l in &p.func(f).body.lets {
            match &l.rhs {
                Rhs::Call { f: g, .. } | Rhs::Closure { f: g, .. } | Rhs::Fold { body: g, .. } => stack.push(*g),
                _ => {}
            }
        }
    }
    seen.into_iter().collect()
}

/// The `let`s of a program's reached functions as elements of `stage`, each
/// labelled by its line.
fn program_elements(t: &mut Table, stage: Stage, p: &Program, prog: &ElabProgram, entry: &str) {
    for f in reached(p, entry) {
        let func = p.func(f);
        t.element(Id::new(stage, func.name.clone()), format!("fn {}", func.name));
        for l in &func.body.lets {
            t.element(Id::new(stage, site(func, l.var)), presk_core::pretty::pretty_let(func, l, p, prog));
        }
    }
}

/// Adds the links whose child the table has.
fn links_of(t: &mut Table, links: &[Link]) {
    for l in links {
        if t.has(&l.child) {
            t.link(l.child.clone(), l.parents.clone(), l.how.clone());
        }
    }
}

/// Why a let that is no nest has no kernel.
fn no_kernel(rhs: &Rhs) -> &'static str {
    match rhs {
        Rhs::Call { .. } => "a call: its callee's nests compute it",
        Rhs::Fold { .. } => "a fold: its body's nests compute it",
        Rhs::Proj(..) | Rhs::Field(..) | Rhs::Tuple(_) | Rhs::Record(_) => "carried by the host: no kernel",
        Rhs::Lit(_) | Rhs::Closure { .. } | Rhs::Apply { .. } | Rhs::Grad(_) => "the host's: no kernel",
        Rhs::Prim { .. } => "the host computes it",
    }
}

/// Why a fusion group ends at its root, or a primitive is alone, in words a
/// link carries: the stop's kind and the `let` it names.
fn stop_word(func: &presk_core::Func, stop: &presk_core::fusion::Stop) -> String {
    use presk_core::fusion::Stop;
    let at = |v: presk_core::VarId| site(func, v);
    match stop {
        Stop::Result => "the function's result".into(),
        Stop::Unread => "nothing reads it".into(),
        Stop::Reader(r) => format!("read by {}", at(*r)),
        Stop::Shape(r) => format!("read by {}, of another shape", at(*r)),
        Stop::Shared(rs) => format!("read by {}, not all in one group", rs.iter().map(|r| at(*r)).collect::<Vec<_>>().join(", ")),
        Stop::Size(r) => format!("its reader {}'s expression too large", at(*r)),
    }
}

/// The table of one compilation.
pub fn table(s: &Stages) -> Table {
    let mut t = Table::new();
    let source = |t: &mut Table, sp: Span| -> Id {
        let id = Id::new(Stage::Source, span_key(sp));
        if !t.has(&id) {
            t.element(id.clone(), snippet(s.sm, sp));
        }
        id
    };

    // ----- the elaborated program and PSK-Core -----------------------------
    program_elements(&mut t, Stage::Core, s.core, s.prog, s.entry);
    for inst in &s.prog.insts {
        let name = presk_types::ty::ModRef::Inst { def: inst.def, args: inst.args.clone() }.display(&s.prog.names());
        let id = Id::new(Stage::Elab, format!("inst {name}"));
        t.element(id.clone(), format!("instance {name}"));
        let src = source(&mut t, s.prog.module(inst.def).span);
        t.link(id, vec![src], How::Instantiated);
    }
    for f in reached(s.core, s.entry) {
        let func = s.core.func(f);
        // The function's own element of the elaborated program.
        let mut root = f;
        while let presk_core::ir::Origin::Lambda { parent } = s.core.func(root).origin {
            root = parent;
        }
        let (elab, label, kind, span) = match &s.core.func(root).origin {
            presk_core::ir::Origin::Source(id) => {
                let sig = s.prog.function(*id);
                (format!("fn {}", presk_core::lower::fn_name(s.prog, *id)), format!("fn {}", presk_core::lower::fn_name(s.prog, *id)), "function", sig.span)
            }
            presk_core::ir::Origin::Init(r) => {
                let m = match r {
                    presk_types::ty::ModRef::Inst { def, .. } => s.prog.module(*def),
                    _ => s.prog.module(presk_types::ty::ModuleId(0)),
                };
                (format!("module {}", m.name), format!("module {}", m.name), "init", m.span)
            }
            presk_core::ir::Origin::Optimizer { name, .. } => (format!("run {name}"), format!("training run, optimizer {name}"), "optimizer", s.core.func(root).span),
            presk_core::ir::Origin::Lambda { .. } => unreachable!("the root of a lambda is not a lambda"),
        };
        let eid = Id::new(Stage::Elab, elab);
        if !t.has(&eid) {
            t.element(eid.clone(), label);
            let src = source(&mut t, span);
            t.link(eid.clone(), vec![src], How::Defined);
        }
        let kind = if root == f { kind.to_string() } else { "lambda".to_string() };
        t.link(Id::new(Stage::Core, func.name.clone()), vec![eid.clone()], How::LoweredFn { kind });
        let rules = s.lowered.rules.get(&f);
        for l in &func.body.lets {
            let rule = rules.and_then(|r| r.get(&l.var)).cloned().unwrap_or_else(|| match &s.core.func(root).origin {
                presk_core::ir::Origin::Init(_) => "init".into(),
                presk_core::ir::Origin::Optimizer { .. } => "optimizer".into(),
                _ => "expression".into(),
            });
            let parent = if s.typed.ty_of.contains_key(&l.span) {
                let x = Id::new(Stage::Elab, format!("expr {}", span_key(l.span)));
                if !t.has(&x) {
                    t.element(x.clone(), snippet(s.sm, l.span));
                    let src = source(&mut t, l.span);
                    t.link(x.clone(), vec![src], How::Typed);
                }
                x
            } else {
                eid.clone()
            };
            t.link(Id::new(Stage::Core, site(func, l.var)), vec![parent], How::Lowered { rule });
        }
    }

    // ----- differentiation ------------------------------------------------------
    program_elements(&mut t, Stage::Ad, s.ad, s.prog, s.entry);
    links_of(&mut t, s.ad_links);

    // ----- precision -------------------------------------------------------------
    if let Some(p) = &s.precision {
        let mut named: BTreeSet<String> = BTreeSet::new();
        if let Some(e) = p.env {
            named.extend(e.sites.keys().cloned());
        }
        named.extend(p.realization.storage.keys().cloned());
        let mut params: BTreeSet<String> = p.realization.params.keys().cloned().collect();
        if let Some(sv) = p.solved {
            for c in &sv.breakdown {
                if c.param {
                    params.insert(c.name.clone());
                } else {
                    named.insert(c.name.clone());
                }
            }
        }
        for n in &named {
            let role = match p.roles.0.get(n) {
                Some(Role::Decision) => "decision",
                Some(Role::Follows(_)) => "follows",
                Some(Role::Carried) => "carried",
                None => "analysed",
            };
            let id = Id::new(Stage::Precision, n.clone());
            let ty = p.realization.dtype_at(n);
            let mag = p.env.and_then(|e| e.get(n)).map(|m| format!(", |v| ≤ {m:.3e}")).unwrap_or_default();
            t.element(id.clone(), format!("{n}: {}, {role}{mag}", ty.name()));
            let at = Id::new(Stage::Ad, n.clone());
            let parents = if t.has(&at) { vec![at] } else { p.links.iter().find(|l| l.child == at).map(|l| l.parents.clone()).unwrap_or(vec![at]) };
            t.link(id, parents, How::Site { role: role.into() });
        }
        for path in &params {
            let id = Id::new(Stage::Precision, format!("param {path}"));
            let ty = p.realization.params.get(path).map(|d| d.name()).unwrap_or(p.realization.default.name());
            t.element(id.clone(), format!("parameter {path}: {ty}"));
            t.link(id, vec![Id::new(Stage::Ad, s.entry.to_string())], How::Parameter);
        }
    }

    // ----- the optimizer ------------------------------------------------------------
    let (xprog, xstage) = match s.opt {
        Some((p, links, ends)) => {
            program_elements(&mut t, Stage::Opt, p, s.prog, s.entry);
            links_of(&mut t, links);
            for e in ends {
                if t.has(&e.id) {
                    t.end(e.id.clone(), e.why.clone());
                }
            }
            (p, Stage::Opt)
        }
        None => (s.ad, Stage::Ad),
    };

    // ----- fusion: one nest per group, or per primitive launched alone -----------
    let launched: BTreeSet<String> = s.kernels.map(|(_, l)| l.iter().flat_map(|x| std::iter::once(x.nest.clone()).chain(x.epilogue.clone())).collect()).unwrap_or_default();
    let mut nest_stage: BTreeMap<String, Stage> = BTreeMap::new();
    for f in reached(xprog, s.entry) {
        let func = xprog.func(f);
        let plan = presk_core::fusion::plan(func);
        let stops: BTreeMap<presk_core::VarId, presk_core::fusion::Stop> = presk_core::fusion::stops(func, &plan).into_iter().collect();
        let roots: HashSet<presk_core::VarId> = plan.groups.keys().copied().collect();
        // A nest is a group or a primitive alone that the call launches, when
        // the kernels were listed: a function the entry reaches only through
        // a closure it never applies (the lambda a `grad` was taken of)
        // launches nothing.
        let runs = |name: &str| s.kernels.is_none() || launched.contains(name);
        let root_of: BTreeMap<presk_core::VarId, presk_core::VarId> = plan.groups.values().flat_map(|g| g.members.iter().map(move |m| (*m, g.root))).collect();
        for l in &func.body.lets {
            let name = site(func, l.var);
            let from = Id::new(xstage, name.clone());
            if plan.interior.contains(&l.var) {
                if let Some(r) = root_of.get(&l.var)
                    && !runs(&site(func, *r))
                {
                    t.end(from, "a member of a group the call never launches");
                }
                continue;
            }
            let stop = stops.get(&l.var).map(|x| stop_word(func, x)).unwrap_or_else(|| "not element-wise".into());
            let candidate = roots.contains(&l.var) || matches!(l.rhs, Rhs::Prim { .. });
            if !candidate || !runs(&name) {
                t.end(from, if roots.contains(&l.var) { "a group the call never launches" } else { no_kernel(&l.rhs) });
                continue;
            }
            let id = Id::new(Stage::Fusion, name.clone());
            match plan.groups.get(&l.var) {
                Some(g) => {
                    let mut parents = vec![from];
                    parents.extend(g.members.iter().filter(|m| **m != l.var).map(|m| Id::new(xstage, site(func, *m))));
                    t.element(id.clone(), format!("{name}: a group of {}", g.members.len()));
                    t.link(id, parents, How::Fused { members: g.members.len(), stop });
                }
                None => {
                    t.element(id.clone(), format!("{name}: alone"));
                    t.link(id, vec![from], How::Alone { stop });
                }
            }
            nest_stage.insert(name, Stage::Fusion);
        }
    }

    // ----- the layout plan --------------------------------------------------------------
    if let Some(ls) = &s.layouts {
        for (nest, point, points) in &ls.nests {
            let id = Id::new(Stage::Layout, nest.clone());
            if t.has(&id) {
                continue;
            }
            t.element(id.clone(), format!("{nest}: {point}"));
            t.link(id, vec![Id::new(Stage::Fusion, nest.clone())], How::LaidOut { point: point.clone(), points: *points });
            nest_stage.insert(nest.clone(), Stage::Layout);
        }
    }
    let nest_id = |n: &str| Id::new(nest_stage.get(n).copied().unwrap_or(Stage::Fusion), n.to_string());

    // ----- the kernels and the launches -------------------------------------------------
    if let Some((kernels, log)) = s.kernels {
        let mut nests_of: BTreeMap<usize, Vec<String>> = BTreeMap::new();
        let mut counts: BTreeMap<(String, usize), (usize, Option<String>)> = BTreeMap::new();
        for x in log {
            let ns = nests_of.entry(x.kernel).or_default();
            for n in std::iter::once(&x.nest).chain(x.epilogue.as_ref()) {
                if !ns.contains(n) {
                    ns.push(n.clone());
                }
            }
            let c = counts.entry((x.nest.clone(), x.kernel)).or_insert((0, x.epilogue.clone()));
            c.0 += 1;
        }
        for (j, ns) in &nests_of {
            let id = Id::new(Stage::Kernel, format!("k{j}"));
            let head = kernels.get(*j).map(|k| k.to_string().lines().next().unwrap_or("").trim_end_matches(" {").to_string()).unwrap_or_default();
            t.element(id.clone(), head);
            t.link(id.clone(), ns.iter().map(|n| nest_id(n)).collect(), How::Kernel { shared_by: ns.len() });
            // Its machine code, every byte with the kernel's line it
            // compiles (statement P4).
            if let Some(m) = s.native.and_then(|n| n.get(*j)) {
                let nid = Id::new(Stage::Native, format!("native k{j}"));
                t.element(nid.clone(), format!("k{j}: {} bytes of machine code", m.bytes.len()));
                t.link(nid, vec![id], How::Code);
            }
        }
        for ((nest, j), (count, epilogue)) in &counts {
            let id = Id::new(Stage::Launch, format!("{nest} k{j}"));
            t.element(id.clone(), format!("{nest} at k{j}, {count} a call"));
            let mut parents = vec![Id::new(Stage::Kernel, format!("k{j}")), nest_id(nest)];
            parents.extend(epilogue.iter().map(|a| nest_id(a)));
            t.link(id, parents, How::Launched { count: *count });
        }
    }

    // ----- the C program ------------------------------------------------------------------
    if let Some((of, lines)) = s.c {
        let mut by_n: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (j, n) in of {
            by_n.entry(*n).or_default().push(*j);
        }
        for (n, js) in &by_n {
            let (first, last) = lines.get(*n).copied().unwrap_or((0, 0));
            let id = Id::new(Stage::C, format!("kernels.c k{n}"));
            t.element(id.clone(), format!("k{n} in kernels.c, lines {first}–{last}"));
            t.link(id, js.iter().map(|j| Id::new(Stage::Kernel, format!("k{j}"))).collect(), How::Function { first, last });
        }
    }
    t
}
