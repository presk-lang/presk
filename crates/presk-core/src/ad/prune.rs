//! Dead code after automatic differentiation (plan 0.10.0 §3.2, part one).
//!
//! A `.vjp` copies its function's forward body before its reverse sweep, and
//! a `.grad` reads one projection of the cotangents a `.vjp` returns. Much of
//! that is never read: forward values the sweep does not need (the primal
//! loss, a callee's result that the callee's own `.vjp` computes again), the
//! cotangents of arguments no caller asks for (a loss's data, the mask a
//! block receives), and forward products whose only reader is a `sum_to`
//! between equal types. This pass removes them from the functions one
//! differentiation generated, and touches no other function:
//!
//! - **Within a function**, `sum_to x, t` and `broadcast_like x, t` between
//!   equal types are `x` (`presk-opt`'s rule); a primitive that reads an
//!   operand for its shape only (the template of `sum_to`, `mean_vjp`,
//!   `dropout_mask`, `zeros_like`, …) reads another value of that type the
//!   function computes anyway, when there is one; then every `let` nothing
//!   reads goes.
//! - **Across calls**, the components of a generated function's result that
//!   no caller reads go from its result: a `.vjp`'s unread cotangents, a
//!   backward fold's unread accumulators. A function whose result is read
//!   outside the program keeps it whole: a `.grad`, which the program applies
//!   as a function value, and the functions a training loop calls.
//!
//! What stays is computed as it was: every fusion group of what the result
//! reads is kept (see `keeps_groups`), since a value inside a group stays
//! in the kernel's arithmetic type, `f32` for `bf16`, where one outside
//! reaches memory in its storage type; what would change one is held.
//! Unless no realization can see a group (`groups` false, D94): then
//! nothing is held.
//! Bindings keep their numbers and functions their names, since a nest is
//! named `<function>/%<let>` (a plan's layouts and T3's nests). PSK-Core is
//! pure: `sample` draws from its key since plan 0.10.0 step 7b, so a `let`
//! nothing reads goes, a draw among them.

use std::collections::{BTreeSet, HashMap, HashSet};

use presk_types::ty::Ty;

use crate::fusion;
use crate::ir::{Func, FuncId, PrimOp, Program, Rhs, VarId};

/// What is read of a value: all of it, or some components of a tuple.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Need {
    Whole,
    Parts(BTreeSet<usize>),
}

impl Need {
    fn nothing() -> Need {
        Need::Parts(BTreeSet::new())
    }

    /// Adds what `other` reads; whether that reads more.
    fn add(&mut self, other: &Need) -> bool {
        match (&mut *self, other) {
            (Need::Whole, _) => false,
            (s, Need::Whole) => {
                *s = Need::Whole;
                true
            }
            (Need::Parts(a), Need::Parts(b)) => {
                let n = a.len();
                a.extend(b);
                a.len() > n
            }
        }
    }

    fn reads(&self) -> bool {
        !matches!(self, Need::Parts(s) if s.is_empty())
    }

    fn has(&self, i: usize) -> bool {
        match self {
            Need::Whole => true,
            Need::Parts(s) => s.contains(&i),
        }
    }
}

/// What the pass removed, for a report.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Pruned {
    /// `let`s removed, the identities among them.
    pub lets: usize,
    /// `sum_to` and `broadcast_like` between equal types, replaced by their operand.
    pub identities: usize,
    /// Templates read from another value of their type.
    pub templates: usize,
    /// Components removed from the results of functions no caller read them of.
    pub components: usize,
    /// `let`s and components kept although nothing read them, since removing
    /// them would have changed a fusion group (`keeps_groups`).
    pub held: usize,
}

/// What one function keeps that the pass would otherwise remove, since
/// removing it would change a fusion group of what survives.
#[derive(Clone, Debug, Default)]
struct Hold {
    /// Identities not applied.
    identities: HashSet<VarId>,
    /// Templates, `(let, operand)`, left where they were.
    templates: HashSet<(VarId, usize)>,
    /// `let`s kept, read or not.
    lets: HashSet<VarId>,
    /// Components of its result kept, read or not.
    components: BTreeSet<usize>,
}

/// Removes the dead code of the functions in `generated`, which one
/// differentiation added to `prog`. The results of those in `fixed` are read
/// outside the program and stay whole. With `groups`, every fusion group of
/// what is kept stays as it was ([`super::Options::groups`]).
pub(crate) fn prune(prog: &mut Program, generated: &[FuncId], fixed: &[FuncId], groups: bool) -> Pruned {
    let (narrow, acc) = narrowable(prog, generated, fixed);
    let original: HashMap<FuncId, Func> = generated.iter().map(|g| (*g, prog.func(*g).clone())).collect();
    let mut hold: HashMap<FuncId, Hold> = HashMap::new();
    // Remove what can go, check that every fusion group of what is kept is
    // as it was, hold what would change one, and start again: what is held
    // only grows, so this ends.
    loop {
        let (out, live) = prune_once(prog, generated, &narrow, &acc, &original, &hold);
        if !groups {
            return out;
        }
        let mut more = false;
        for g in generated {
            for fix in keeps_groups(&original[g], prog.func(*g), &live[g]) {
                let h = hold.entry(*g).or_default();
                match fix {
                    Fix::Identity(v) => more |= h.identities.insert(v),
                    Fix::Template(l, k) => more |= h.templates.insert((l, k)),
                    Fix::Let(d) => more |= h.lets.insert(d),
                    Fix::Result(i) => more |= h.components.insert(i),
                    Fix::Accumulator(body, i) => more |= hold.entry(body).or_default().components.insert(i),
                }
            }
        }
        if !more {
            let held = hold.values().map(|h| h.identities.len() + h.templates.len() + h.lets.len() + h.components.len()).sum();
            return Pruned { held, ..out };
        }
        for g in generated {
            prog.funcs[g.0 as usize] = original[g].clone();
        }
    }
}

/// One pass over the original bodies, holding what `hold` holds. Returns
/// what it removed and, for each function, the `let`s whose value is read.
fn prune_once(prog: &mut Program, generated: &[FuncId], narrow: &HashSet<FuncId>, acc: &HashMap<FuncId, usize>, original: &HashMap<FuncId, Func>, hold: &HashMap<FuncId, Hold>) -> (Pruned, HashMap<FuncId, HashSet<VarId>>) {
    let none = Hold::default();
    let held = |g: &FuncId| hold.get(g).unwrap_or(&none);
    let mut out = Pruned::default();
    let mut read: HashMap<FuncId, HashSet<VarId>> = HashMap::new();
    for g in generated {
        out.identities += identities(&mut prog.funcs[g.0 as usize], &held(g).identities);
    }
    // What is read for its value, templates not counted: a template reads
    // another such value of its type instead, when the function has one.
    let by_value = demands(prog, generated, narrow, acc, hold, false);
    for g in generated {
        let d = by_value.get(g).cloned().unwrap_or(Need::Whole);
        let lv = live(prog.func(*g), &d, &by_value, &held(g).lets, false);
        let read: HashSet<VarId> = lv.need.iter().filter(|(_, n)| n.reads()).map(|(v, _)| *v).collect();
        out.templates += retarget(&mut prog.funcs[g.0 as usize], &read, &held(g).templates);
    }
    let demand = demands(prog, generated, narrow, acc, hold, true);
    for g in generated {
        let d = demand.get(g).cloned().unwrap_or(Need::Whole);
        let lv = live(prog.func(*g), &d, &demand, &HashSet::new(), true);
        read.insert(*g, lv.need.iter().filter(|(_, n)| n.reads()).map(|(v, _)| *v).collect());
        let lv = live(prog.func(*g), &d, &demand, &held(g).lets, true);
        let func = &mut prog.funcs[g.0 as usize];
        let result = func.body.result;
        func.body.lets.retain(|l| l.var == result || lv.need.get(&l.var).is_some_and(Need::reads) || held(g).lets.contains(&l.var));
        out.lets += original[g].body.lets.len() - func.body.lets.len();
        // Its own result, narrowed to what its callers read.
        if let Need::Parts(keep) = &d {
            let rl = func.body.lets.iter_mut().find(|l| l.var == result).expect("the result is a let");
            let Rhs::Tuple(vs) = &rl.rhs else { unreachable!("a narrowable function returns a tuple") };
            out.components += vs.len() - keep.len();
            rl.rhs = Rhs::Tuple(vs.iter().enumerate().filter(|(i, _)| keep.contains(i)).map(|(_, v)| *v).collect());
            rl.ty = narrowed(&rl.ty, keep);
            func.ret = narrowed(&func.ret, keep);
            if let Some(pos) = acc.get(g) {
                let a = func.params[*pos].var;
                func.params[*pos].ty = narrowed(&func.params[*pos].ty, keep);
                renumber(&mut func.body.lets, a, keep);
            }
        }
        // The calls and folds of narrowed functions, and what reads them.
        let mut remap: Vec<(VarId, BTreeSet<usize>)> = vec![];
        let mut inits: Vec<(VarId, BTreeSet<usize>)> = vec![];
        for l in func.body.lets.iter_mut() {
            let (callee, init) = match &l.rhs {
                Rhs::Call { f, .. } => (*f, None),
                Rhs::Fold { body, init, .. } => (*body, Some(*init)),
                _ => continue,
            };
            if let Some(Need::Parts(keep)) = demand.get(&callee) {
                l.ty = narrowed(&l.ty, keep);
                remap.push((l.var, keep.clone()));
                if let Some(i) = init {
                    inits.push((i, keep.clone()));
                }
            }
        }
        for (v, keep) in &remap {
            renumber(&mut func.body.lets, *v, keep);
        }
        for (v, keep) in &inits {
            let l = func.body.lets.iter_mut().find(|l| l.var == *v).expect("a narrowed fold's accumulator is a let of its function");
            let Rhs::Tuple(vs) = &l.rhs else { unreachable!("checked by `narrowable`") };
            l.rhs = Rhs::Tuple(vs.iter().enumerate().filter(|(i, _)| keep.contains(i)).map(|(_, v)| *v).collect());
            l.ty = narrowed(&l.ty, keep);
        }
    }
    (out, read)
}

/// What callers read of each narrowable function, to a fixpoint: a call read
/// more makes its callee read more of its own callees. With `templates`
/// false, an operand read for its shape only is not read.
fn demands(prog: &Program, generated: &[FuncId], narrow: &HashSet<FuncId>, acc: &HashMap<FuncId, usize>, hold: &HashMap<FuncId, Hold>, templates: bool) -> HashMap<FuncId, Need> {
    let none = Hold::default();
    let held = |g: &FuncId| hold.get(g).unwrap_or(&none);
    let mut demand: HashMap<FuncId, Need> = narrow.iter().map(|f| (*f, Need::Parts(held(f).components.clone()))).collect();
    loop {
        let mut more = false;
        for g in generated {
            let d = demand.get(g).cloned().unwrap_or(Need::Whole);
            let lv = live(prog.func(*g), &d, &demand, &held(g).lets, templates);
            for (callee, n) in &lv.asks {
                more |= demand.get_mut(callee).expect("asks name narrowable functions").add(n);
            }
            // A backward fold's body reads its accumulator from the previous
            // iteration: what it reads of it, it must return.
            if let Some(pos) = acc.get(g) {
                let a = prog.func(*g).params[*pos].var;
                if let Some(n) = lv.need.get(&a) {
                    more |= demand.get_mut(g).expect("a fold body is narrowable").add(n);
                }
            }
        }
        if !more {
            return demand;
        }
    }
}

/// The operands a primitive reads for their shape only: a template.
fn templates(op: &PrimOp) -> &'static [usize] {
    match op {
        PrimOp::SumTo | PrimOp::BroadcastLike | PrimOp::MeanVjp { .. } | PrimOp::ReshapeLike => &[1],
        PrimOp::ZerosLike | PrimOp::ScatterIndex { .. } | PrimOp::SliceVjp { .. } | PrimOp::ScatterParams { .. } | PrimOp::EmbedVjp => &[0],
        PrimOp::DropoutMask => &[2],
        _ => &[],
    }
}

/// Points a template that nothing else reads, `t`, at another value of
/// `t`'s type that the function reads anyway: a parameter, else the last
/// `let` before it that no fusion group holds (not elementwise, so reading
/// it moves no group). Then `t` can go: the forward product only
/// `dropout_mask` read the shape of, in a block's `.vjp`. The type must name
/// every axis and have no open prefix, since a template gives its axes'
/// names to the result, and two values of such a type have the same axes.
/// `read` is what the function reads, templates not counted.
fn retarget(func: &mut Func, read: &HashSet<VarId>, held: &HashSet<(VarId, usize)>) -> usize {
    let named = |t: &Ty| matches!(t, Ty::Real(s) if s.prefix.is_none() && s.axes.iter().all(|a| a.name.is_some()));
    let mut ty: HashMap<VarId, Ty> = func.params.iter().map(|p| (p.var, p.ty.clone())).collect();
    let params: Vec<VarId> = func.params.iter().map(|p| p.var).collect();
    let mut typed: HashSet<VarId> = params.iter().copied().collect();
    let mut plain: Vec<VarId> = vec![];
    let mut n = 0;
    for l in func.body.lets.iter_mut() {
        if let Rhs::Prim { op, args } = &mut l.rhs {
            for k in templates(op) {
                let t = args[*k];
                if read.contains(&t) || params.contains(&t) || held.contains(&(l.var, *k)) || !typed.contains(&t) || !named(&ty[&t]) {
                    continue;
                }
                let found = params.iter().find(|p| ty[p] == ty[&t]).or_else(|| plain.iter().rev().find(|v| **v != t && read.contains(v) && typed.contains(v) && ty[v] == ty[&t]));
                if let Some(u) = found {
                    args[*k] = *u;
                    n += 1;
                }
            }
        }
        ty.insert(l.var, l.ty.clone());
        if shape_typed(l) {
            typed.insert(l.var);
        }
        if !matches!(&l.rhs, Rhs::Prim { op, .. } if op.elementwise()) {
            plain.push(l.var);
        }
    }
    n
}

/// What a fusible value needs held so that its fusion group stays.
#[derive(Debug)]
enum Fix {
    /// The identity that moved its readers.
    Identity(VarId),
    /// A template, `(let, operand)`, moved to it or from it.
    Template(VarId, usize),
    /// A `let` that read it and was removed.
    Let(VarId),
    /// A component of the function's result that it was.
    Result(usize),
    /// A component of the accumulator of a fold whose body this is.
    Accumulator(FuncId, usize),
}

/// What to hold so that `after`, the pass's rewrite of `before`, computes
/// every value it reads as `before` did: nothing when it does. A value inside
/// a fusion group stays in the kernel's arithmetic type (`f32` for `bf16`),
/// and a value outside every group reaches memory in its storage type; the
/// partition ([`fusion::plan`]) grows a group backwards from its last `let`
/// over the elementwise `let`s of its type whose every reader the group
/// holds. So each `let` of `live` must be inside a group in both or in
/// neither, and each group they root have the same members. The pass only
/// removes readers, and gives the readers of an identity to its operand: a
/// group changes because some fusible `let` is read by other `let`s than
/// before. For those near a change, what is held restores their readers:
/// the identities that moved them, the readers removed, the components of a
/// result or of a fold's accumulator they were.
fn keeps_groups(before: &Func, after: &Func, live: &HashSet<VarId>) -> Vec<Fix> {
    let (p0, p1) = (fusion::plan(before), fusion::plan(after));
    let members = |p: &fusion::Plan, v: VarId| p.groups.get(&v).map(|g| g.members.clone());
    let bad: Vec<VarId> = live.iter().copied().filter(|v| p0.interior.contains(v) != p1.interior.contains(v) || members(&p0, *v) != members(&p1, *v)).collect();
    if bad.is_empty() {
        return vec![];
    }
    let readers = |f: &Func| -> HashMap<VarId, BTreeSet<VarId>> {
        let mut r: HashMap<VarId, BTreeSet<VarId>> = HashMap::new();
        for l in &f.body.lets {
            for v in l.rhs.vars() {
                r.entry(v).or_default().insert(l.var);
            }
        }
        r
    };
    let fusible = |l: &crate::ir::Let| matches!(&l.rhs, Rhs::Prim { op, .. } if op.elementwise()) && matches!(l.ty, Ty::Real(_) | Ty::Params(_));
    let identity = |l: &crate::ir::Let| matches!(&l.rhs, Rhs::Prim { op: PrimOp::SumTo | PrimOp::BroadcastLike, .. });
    let (r0, r1) = (readers(before), readers(after));
    let old: HashMap<VarId, &crate::ir::Let> = before.body.lets.iter().map(|l| (l.var, l)).collect();
    let now: HashMap<VarId, &crate::ir::Let> = after.body.lets.iter().map(|l| (l.var, l)).collect();
    let empty = BTreeSet::new();
    // The fusible `let`s whose readers changed, and those near a change: in
    // a group, or an input of a group, that differs.
    let changed: Vec<&crate::ir::Let> = after.body.lets.iter().filter(|l| fusible(l) && r0.get(&l.var).unwrap_or(&empty) != r1.get(&l.var).unwrap_or(&empty)).collect();
    let mut near: HashSet<VarId> = bad.iter().copied().collect();
    for p in [&p0, &p1] {
        for g in p.groups.values().filter(|g| g.members.iter().any(|m| bad.contains(m))) {
            near.extend(g.members.iter().chain(&g.inputs));
        }
    }
    let restore = |l: &crate::ir::Let, fixes: &mut Vec<Fix>| {
        let (was, is) = (r0.get(&l.var).unwrap_or(&empty), r1.get(&l.var).unwrap_or(&empty));
        let positions = |vs: &[VarId]| vs.iter().enumerate().filter(|(_, v)| **v == l.var).map(|(i, _)| i).collect::<Vec<_>>();
        // A template moved from it or to it: a reader that stays.
        let moved = |d: &VarId, fixes: &mut Vec<Fix>| -> bool {
            let (Some(Rhs::Prim { op, args: a0 }), Some(Rhs::Prim { args: a1, .. })) = (old.get(d).map(|x| &x.rhs), now.get(d).map(|x| &x.rhs)) else { return false };
            let ks: Vec<usize> = templates(op).iter().copied().filter(|k| (a0[*k] == l.var) != (a1[*k] == l.var)).collect();
            fixes.extend(ks.iter().map(|k| Fix::Template(*d, *k)));
            !ks.is_empty()
        };
        for d in was.difference(is) {
            if moved(d, fixes) {
                continue;
            }
            let dl = old[d];
            match &dl.rhs {
                _ if identity(dl) => fixes.extend([Fix::Identity(*d), Fix::Let(*d)]),
                Rhs::Tuple(vs) if *d == before.body.result => fixes.extend(positions(vs).into_iter().map(Fix::Result)),
                Rhs::Tuple(vs) => match before.body.lets.iter().find_map(|f| match &f.rhs {
                    Rhs::Fold { body, init, .. } if init == d => Some(*body),
                    _ => None,
                }) {
                    Some(body) => fixes.extend(positions(vs).into_iter().map(|i| Fix::Accumulator(body, i))),
                    None => fixes.push(Fix::Let(*d)),
                },
                _ => fixes.push(Fix::Let(*d)),
            }
        }
        let gained: Vec<VarId> = is.difference(was).copied().collect();
        for d in &gained {
            moved(d, fixes);
        }
        if !gained.is_empty() {
            // Readers gained: an identity whose operand it is gave it its own.
            fixes.extend(before.body.lets.iter().filter(|v| identity(v) && v.rhs.vars()[0] == l.var).map(|v| Fix::Identity(v.var)));
        }
    };
    // The changed `let`s that moved into a group or out of one, else those
    // whose group changed, else those near a change, else all: the last
    // ends, at worst, at the body as it was.
    let moved: HashSet<VarId> = bad.iter().copied().filter(|v| p0.interior.contains(v) != p1.interior.contains(v)).collect();
    let mut fixes = vec![];
    for set in [&moved, &bad.iter().copied().collect::<HashSet<VarId>>(), &near] {
        for l in changed.iter().filter(|l| set.contains(&l.var)) {
            restore(l, &mut fixes);
        }
        if !fixes.is_empty() {
            return fixes;
        }
    }
    for l in &changed {
        restore(l, &mut fixes);
    }
    fixes
}

/// Whether a `let`'s type is its value's shape. Automatic differentiation
/// types a few reductions as the tensor they reduce, and broadcasts them back
/// at once: softmax's rule `sum axis a` of the full shape, an `ein` adjoint
/// whose operand has an axis no other factor carries. Their type is not
/// their shape, and neither an identity nor a template may lean on it.
fn shape_typed(l: &crate::ir::Let) -> bool {
    let names = |t: &Ty| match t {
        Ty::Real(s) => s.axes.iter().map(|a| a.name.clone()).collect::<Vec<_>>(),
        _ => vec![],
    };
    match &l.rhs {
        Rhs::Prim { op: PrimOp::Sum { axis } | PrimOp::Mean { axis }, .. } => !names(&l.ty).contains(&Some(axis.clone())),
        Rhs::Prim { op: PrimOp::Ein { output, .. }, .. } => matches!(&l.ty, Ty::Real(s) if s.axes.len() == output.len()),
        _ => true,
    }
}

/// `sum_to x, t` and `broadcast_like x, t` where `x`, `t` and the result
/// have one type are `x`: every read of the result reads `x`. `t` is then
/// read no more, and a forward value kept only as its template can go (the
/// forward `dot` of `Linear.apply.vjp`). The types agree axis names
/// included, since the runtime tensor carries the template's names
/// (question 44), as in `presk-opt`'s rule, and are the shapes of `x` and
/// `t` ([`shape_typed`]).
fn identities(func: &mut Func, held: &HashSet<VarId>) -> usize {
    let mut ty: HashMap<VarId, Ty> = func.params.iter().map(|p| (p.var, p.ty.clone())).collect();
    let mut typed: HashSet<VarId> = func.params.iter().map(|p| p.var).collect();
    let mut subst: HashMap<VarId, VarId> = HashMap::new();
    for l in func.body.lets.iter_mut() {
        l.rhs = l.rhs.map_vars(|v| *subst.get(&v).unwrap_or(&v));
        ty.insert(l.var, l.ty.clone());
        if shape_typed(l) {
            typed.insert(l.var);
        }
        if let Rhs::Prim { op: PrimOp::SumTo | PrimOp::BroadcastLike, args } = &l.rhs
            && !held.contains(&l.var)
            && typed.contains(&args[0])
            && typed.contains(&args[1])
            && ty[&args[0]] == l.ty
            && ty[&args[1]] == l.ty
        {
            subst.insert(l.var, args[0]);
        }
    }
    let r = func.body.result;
    func.body.result = *subst.get(&r).unwrap_or(&r);
    subst.len()
}

/// The generated functions whose result may be narrowed, and for those that
/// are a backward fold's body, the position of the accumulator among their
/// parameters. A function qualifies when it returns a tuple built by its
/// last `let`, is not in `fixed`, and is referenced only by the calls and
/// folds of generated functions, each fold's accumulator a tuple built for
/// it alone.
fn narrowable(prog: &Program, generated: &[FuncId], fixed: &[FuncId]) -> (HashSet<FuncId>, HashMap<FuncId, usize>) {
    let is_gen: HashSet<FuncId> = generated.iter().copied().collect();
    let returns_tuple = |f: &Func| f.body.lets.iter().any(|l| l.var == f.body.result && matches!(l.rhs, Rhs::Tuple(_)));
    let mut narrow: HashSet<FuncId> = generated.iter().copied().filter(|g| !fixed.contains(g) && returns_tuple(prog.func(*g))).collect();
    let mut acc: HashMap<FuncId, usize> = HashMap::new();
    for (i, func) in prog.funcs.iter().enumerate() {
        let caller = FuncId(i as u32);
        let uses: HashMap<VarId, usize> = func.body.lets.iter().flat_map(|l| l.rhs.vars()).chain([func.body.result]).fold(HashMap::new(), |mut m, v| {
            *m.entry(v).or_default() += 1;
            m
        });
        for l in &func.body.lets {
            match &l.rhs {
                Rhs::Call { .. } if is_gen.contains(&caller) => {}
                Rhs::Fold { body, init, stacked, index, .. } if is_gen.contains(&caller) => {
                    let pos = usize::from(*index) + stacked.len();
                    let own_tuple = func.body.lets.iter().any(|t| t.var == *init && matches!(t.rhs, Rhs::Tuple(_))) && uses.get(init) == Some(&1);
                    let b = prog.func(*body);
                    let same = b.params.get(pos).is_some_and(|p| p.ty == b.ret && p.ty == l.ty);
                    if !own_tuple || !same || acc.insert(*body, pos).is_some_and(|p| p != pos) {
                        narrow.remove(body);
                    }
                }
                Rhs::Call { f, .. } | Rhs::Closure { f, .. } | Rhs::Fold { body: f, .. } => {
                    narrow.remove(f);
                }
                _ => {}
            }
        }
    }
    acc.retain(|f, _| narrow.contains(f));
    (narrow, acc)
}

/// Liveness in one function, reading `demand` of its result.
struct Live {
    /// What is read of each variable; absent, or nothing, is dead.
    need: HashMap<VarId, Need>,
    /// What its calls and folds read of narrowable functions.
    asks: Vec<(FuncId, Need)>,
}

fn live(func: &Func, demand: &Need, demands: &HashMap<FuncId, Need>, held: &HashSet<VarId>, shapes: bool) -> Live {
    let mut need: HashMap<VarId, Need> = HashMap::from([(func.body.result, demand.clone())]);
    let mut asks = vec![];
    let add = |need: &mut HashMap<VarId, Need>, v: VarId, n: &Need| {
        need.entry(v).or_insert_with(Need::nothing).add(n);
    };
    // The tuples the rewrite narrows: the result, and the accumulators of
    // the folds whose body it narrows. Another tuple that survives keeps
    // every component, so reading one of it reads all.
    let narrowed: HashSet<VarId> = func
        .body
        .lets
        .iter()
        .filter_map(|l| match &l.rhs {
            Rhs::Fold { body, init, .. } if demands.contains_key(body) => Some(*init),
            _ => None,
        })
        .chain([func.body.result])
        .collect();
    for l in func.body.lets.iter().rev() {
        let n = match need.get(&l.var) {
            Some(n) if n.reads() => n.clone(),
            _ if held.contains(&l.var) => Need::Whole,
            _ => continue,
        };
        let n = if matches!(l.rhs, Rhs::Tuple(_)) && !narrowed.contains(&l.var) { Need::Whole } else { n };
        match &l.rhs {
            Rhs::Tuple(vs) => {
                for (i, v) in vs.iter().enumerate() {
                    if n.has(i) {
                        add(&mut need, *v, &Need::Whole);
                    }
                }
            }
            Rhs::Proj(x, i) => add(&mut need, *x, &Need::Parts(BTreeSet::from([*i]))),
            Rhs::Call { f, args, .. } => {
                args.iter().for_each(|a| add(&mut need, *a, &Need::Whole));
                if demands.contains_key(f) {
                    asks.push((*f, n));
                }
            }
            Rhs::Fold { body, stacked, init, extra, .. } => {
                stacked.iter().chain(extra).for_each(|a| add(&mut need, *a, &Need::Whole));
                match demands.get(body) {
                    Some(d) => {
                        let mut i = d.clone();
                        i.add(&n);
                        add(&mut need, *init, &i);
                        asks.push((*body, n));
                    }
                    None => add(&mut need, *init, &Need::Whole),
                }
            }
            Rhs::Prim { op, args } if !shapes => {
                for (k, a) in args.iter().enumerate() {
                    if !templates(op).contains(&k) {
                        add(&mut need, *a, &Need::Whole);
                    }
                }
            }
            other => other.vars().into_iter().for_each(|a| add(&mut need, a, &Need::Whole)),
        }
    }
    Live { need, asks }
}

/// A tuple type without the components `keep` leaves out.
fn narrowed(t: &Ty, keep: &BTreeSet<usize>) -> Ty {
    match t {
        Ty::Tuple(ts) => Ty::Tuple(ts.iter().enumerate().filter(|(i, _)| keep.contains(i)).map(|(_, t)| t.clone()).collect()),
        other => unreachable!("a narrowed result is a tuple, not {other:?}"),
    }
}

/// Every projection of `v` reads its component's new position.
fn renumber(lets: &mut [crate::ir::Let], v: VarId, keep: &BTreeSet<usize>) {
    for l in lets.iter_mut() {
        if let Rhs::Proj(x, i) = &mut l.rhs
            && *x == v
        {
            debug_assert!(keep.contains(i), "a projection that survives reads a kept component");
            *i = keep.range(..*i).count();
        }
    }
}
