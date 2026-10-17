//! Residuals (plan 0.10.0 §3.2, part two): a gradient keeps what its forward
//! pass computed instead of computing it again.
//!
//! A `.vjp` copies its function's forward body before its reverse sweep, so a
//! gradient computes each callee's forward pass twice, once for the value its
//! caller reads and once more inside the callee's `.vjp`, at every level of
//! calls. This pass splits each `.vjp` that [`super::prune`] left into two
//! functions:
//!
//! - **`g.fwd`** runs `g`'s body once and returns its value and its
//!   residuals, the values `g`'s sweep reads that `g` computes (the inputs of
//!   the primitives' rules, and each callee's residuals). Its bindings are
//!   `g`'s, numbered as in `g`, so its nests are `g`'s under another name, and
//!   a realization or a plan that states a site of `g` states it of `g.fwd`.
//! - **`g.bwd`** takes the parameters of `g` its sweep reads, the residuals,
//!   and the cotangent, and runs the sweep, calling each callee's `.bwd` on
//!   that callee's residuals. Its bindings are the `.vjp`'s, numbered as in
//!   the `.vjp`, so the sweep's nests keep their numbers (`Linear.apply.vjp/%11`
//!   is `Linear.apply.bwd/%11`).
//!
//! `g.vjp` becomes `g.bwd` after `g.fwd`. A fold whose body is split stacks
//! each layer's residuals: its forward pass scans with the residuals in the
//! accumulator, so that the stack of the accumulators before each iteration
//! holds, at slot `i + 1`, the residuals of layer `i`, and the last layer's
//! are in the final accumulator; the backward fold carries the residuals of
//! the layer it differentiates in its accumulator and hands the next one its
//! own from the stack. The first slot holds zeros, which nothing reads.
//!
//! **Exactness.** Every gradient is bit-identical to the recomputing one's in
//! every type, `bf16` included, where a fusion group keeps its intermediates
//! in `f32`: the values `g.bwd` reads must be computed as the `.vjp`'s
//! forward copy computed them, in the same fusion groups, and `g.fwd`'s value
//! as `g` computes it. The two partitions differ where the sweep reads a
//! value that `g`'s own groups keep inside (a normalization's intermediate):
//! there `g.fwd` computes the residual apart, a copy of the `.vjp`'s group,
//! and `g`'s value as `g` does. Values the sweep's own groups hold inside, and
//! projections, tuples and key splits, which compute nothing, are computed
//! again in `g.bwd`, never stored. What cannot be made exact is left as it
//! was: that `.vjp` recomputes. When no realization can see a fusion group
//! (`groups` false, [`super::Options::groups`], D94), every group computes
//! what its members would alone, so `g.fwd` shares every residual with `g`'s
//! computation, reading it where `g`'s groups kept it inside, and nothing
//! is computed apart or compared.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use presk_types::dim::{DimExpr, DimVar};
use presk_types::ty::{PrefixVar, Shape, Ty};

use super::{Built, identity_dims};
use crate::fusion;
use crate::ir::{stack_ty, Block, Func, FuncId, Let, Origin, Param, PrimOp, Program, Rhs, VarId};

/// Leaves a `.vjp` recomputing; `PRESK_STORE_TRACE` says why.
macro_rules! keep {
    ($name:expr, $why:expr) => {{
        if std::env::var_os("PRESK_STORE_TRACE").is_some() {
            eprintln!("store: {} recomputes: {}", $name, $why);
        }
        return None;
    }};
}

/// How a `.vjp` was split.
#[derive(Clone, Debug)]
pub(crate) struct Split {
    /// `g.fwd`; absent when `g.bwd` reads no value `g` computes, only its
    /// parameters: a caller then calls `g` itself for the value.
    pub fwd: Option<FuncId>,
    pub bwd: FuncId,
    /// The parameters of `g` that `g.bwd` takes, by position.
    pub params: Vec<usize>,
    /// The tuple of residuals `g.fwd` returns, in `g`'s dimensions.
    pub r_ty: Ty,
}

/// What the pass did, for a report.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stored {
    /// `.vjp`s split into `.fwd` and `.bwd`.
    pub split: usize,
    /// `.vjp`s left recomputing, their split not exact or not expressible.
    pub kept: usize,
    /// Residuals computed apart in a `.fwd`, as the `.vjp` computed them.
    pub copies: usize,
    /// Folds that stack their layers' residuals.
    pub folds: usize,
}

/// [`split_with`]'s splits alone.
pub(crate) fn split_all(prog: &mut Program, built: &HashMap<FuncId, Built>, generated: &mut Vec<FuncId>, groups: bool) -> HashMap<FuncId, Split> {
    split_with(prog, built, generated, groups).1
}

/// Splits every `.vjp` of `built` that can be split exactly, callees first,
/// and rewrites its callers to call `.fwd` in their forward pass and `.bwd`
/// in their sweep. New functions are appended to `generated`. Returns what
/// was done and how each `.vjp` split was split.
pub(crate) fn split_with(prog: &mut Program, built: &HashMap<FuncId, Built>, generated: &mut Vec<FuncId>, groups: bool) -> (Stored, HashMap<FuncId, Split>) {
    let order = post_order(prog, built);
    let mut splits: HashMap<FuncId, Split> = HashMap::new();
    let mut steps: HashMap<FuncId, FuncId> = HashMap::new();
    let mut out = Stored::default();
    for v in order {
        let Some(v) = v else { continue };
        match split_one(prog, v, &built[&v], built, &splits, &mut steps, generated, groups) {
            Some((s, copies, folds)) => {
                splits.insert(v, s);
                out.split += 1;
                out.copies += copies;
                out.folds += folds;
            }
            None => out.kept += 1,
        }
    }
    (out, splits)
}

/// `f.value_vjp(params…, ct)`: `f.fwd`'s value, then `f.bwd`'s cotangents
/// (`f`'s own value when `f.bwd` reads no residual). `vjp`, split as `s`,
/// is `f.vjp`.
pub(crate) fn value_wrapper(prog: &mut Program, f: FuncId, vjp: FuncId, value_vjp: FuncId, s: &Split) {
    let vv = prog.func(value_vjp).clone();
    let src = prog.func(f).clone();
    let n = src.params.len();
    let span = vv.span;
    let ps: Vec<VarId> = vv.params[..n].iter().map(|p| p.var).collect();
    let ct = vv.params[n].var;
    let mut w = Fresh(n as u32 + 1);
    let mut lets = vec![];
    let mut a: Vec<VarId> = s.params.iter().map(|k| ps[*k]).collect();
    let value = match s.fwd {
        Some(fwd) => {
            let (pv, v, r) = (w.next(), w.next(), w.next());
            lets.push(mk(pv, prog.func(fwd).ret.clone(), Rhs::Call { f: fwd, dim_args: identity_dims(&src), args: ps.clone() }, span));
            lets.push(mk(v, src.ret.clone(), Rhs::Proj(pv, 0), span));
            lets.push(mk(r, s.r_ty.clone(), Rhs::Proj(pv, 1), span));
            a.push(r);
            v
        }
        None => {
            let v = w.next();
            lets.push(mk(v, src.ret.clone(), Rhs::Call { f, dim_args: identity_dims(&src), args: ps.clone() }, span));
            v
        }
    };
    a.push(ct);
    let cts_ty = prog.func(vjp).ret.clone();
    let cts = w.next();
    lets.push(mk(cts, cts_ty.clone(), Rhs::Call { f: s.bwd, dim_args: identity_dims(&src), args: a }, span));
    let Ty::Tuple(ts) = &cts_ty else { unreachable!("a .vjp returns a tuple") };
    let mut outs = vec![value];
    for (i, t) in ts.iter().enumerate() {
        let c = w.next();
        lets.push(mk(c, t.clone(), Rhs::Proj(cts, i), span));
        outs.push(c);
    }
    let out = w.next();
    lets.push(mk(out, vv.ret.clone(), Rhs::Tuple(outs), span));
    prog.funcs[value_vjp.0 as usize].body = Block { lets, result: out };
}

/// The `.vjp`s callees first; `None` for one on a cycle of calls (a function
/// that reaches itself through a closure), which is not split.
fn post_order(prog: &Program, built: &HashMap<FuncId, Built>) -> Vec<Option<FuncId>> {
    let deps = |v: FuncId| -> Vec<FuncId> {
        let mut d = vec![];
        let bwds: HashSet<FuncId> = built[&v].folds.values().map(|f| f.bwd).collect();
        for l in &prog.func(v).body.lets {
            match &l.rhs {
                Rhs::Call { f, .. } if built.contains_key(f) => d.push(*f),
                Rhs::Fold { body, .. } if bwds.contains(body) => {
                    for m in &prog.func(*body).body.lets {
                        if let Rhs::Call { f, .. } = &m.rhs
                            && built.contains_key(f)
                        {
                            d.push(*f);
                        }
                    }
                }
                _ => {}
            }
        }
        d
    };
    let mut state: HashMap<FuncId, u8> = HashMap::new();
    let mut cyclic: HashSet<FuncId> = HashSet::new();
    let mut order = vec![];
    let mut roots: Vec<FuncId> = built.keys().copied().collect();
    roots.sort_by_key(|f| f.0);
    fn visit(v: FuncId, deps: &dyn Fn(FuncId) -> Vec<FuncId>, state: &mut HashMap<FuncId, u8>, cyclic: &mut HashSet<FuncId>, order: &mut Vec<FuncId>) {
        state.insert(v, 1);
        for d in deps(v) {
            match state.get(&d) {
                None => visit(d, deps, state, cyclic, order),
                Some(1) => {
                    cyclic.insert(d);
                    cyclic.insert(v);
                }
                _ => {}
            }
            if cyclic.contains(&d) {
                cyclic.insert(v);
            }
        }
        state.insert(v, 2);
        order.push(v);
    }
    for r in roots {
        if !state.contains_key(&r) {
            visit(r, &deps, &mut state, &mut cyclic, &mut order);
        }
    }
    order.into_iter().map(|v| (!cyclic.contains(&v)).then_some(v)).collect()
}

/// Lets that compute nothing a kernel would: recomputed in `.bwd` rather
/// than stored.
fn free(rhs: &Rhs) -> bool {
    matches!(rhs, Rhs::Lit(_) | Rhs::Proj(..) | Rhs::Field(..) | Rhs::Tuple(_) | Rhs::Record(_) | Rhs::Closure { .. } | Rhs::Prim { op: PrimOp::SplitKey { .. } | PrimOp::Real | PrimOp::Pi, .. })
}

fn fusible(l: &Let) -> bool {
    matches!(&l.rhs, Rhs::Prim { op, .. } if op.elementwise()) && matches!(l.ty, Ty::Real(_) | Ty::Params(_))
}

/// How a materialized value is computed under a partition: the members and
/// inputs of the group it roots, or itself and its operands; `None` when it
/// is inside a group.
fn comp(p: &fusion::Plan, l: &Let) -> Option<(Vec<VarId>, Vec<VarId>)> {
    if p.interior.contains(&l.var) {
        return None;
    }
    Some(match p.groups.get(&l.var) {
        Some(g) => (g.members.clone(), g.inputs.clone()),
        None => (vec![l.var], l.rhs.vars()),
    })
}

/// A residual's type is one layer's; zeros of it stand for the residuals of
/// the layer before the first: every leaf a real tensor of named length.
fn zeroable(t: &Ty) -> bool {
    match t {
        Ty::Real(s) => s.prefix.is_none(),
        Ty::Tuple(ts) => ts.iter().all(zeroable),
        _ => false,
    }
}

/// A callee's residual type at a call: its dimensions and its open prefixes
/// bound as the call binds them, a prefix to the leading axes of the
/// argument whose parameter has it. The type-checker did the same to type
/// the call; the values carry those axes, so zeros of the type are the
/// shape of the residuals.
fn at_call(callee: &Func, r_ty: &Ty, dim_args: &BTreeMap<DimVar, DimExpr>, args: &[Ty]) -> Ty {
    fn bind(p: &Ty, a: &Ty, out: &mut HashMap<PrefixVar, Shape>) {
        match (p, a) {
            (Ty::Real(ps) | Ty::Idx { shape: ps, .. } | Ty::Truth(ps), Ty::Real(xs) | Ty::Idx { shape: xs, .. } | Ty::Truth(xs)) => {
                if let Some(v) = ps.prefix
                    && xs.axes.len() >= ps.axes.len()
                {
                    out.entry(v).or_insert_with(|| Shape { prefix: xs.prefix, axes: xs.axes[..xs.axes.len() - ps.axes.len()].to_vec() });
                }
            }
            (Ty::Tuple(ps), Ty::Tuple(xs)) => ps.iter().zip(xs).for_each(|(p, a)| bind(p, a, out)),
            _ => {}
        }
    }
    fn inst(t: &Ty, b: &HashMap<PrefixVar, Shape>) -> Ty {
        let sh = |s: &Shape| match s.prefix.and_then(|v| b.get(&v)) {
            Some(pre) => Shape { prefix: pre.prefix, axes: pre.axes.iter().chain(&s.axes).cloned().collect() },
            None => s.clone(),
        };
        match t {
            Ty::Real(s) => Ty::Real(sh(s)),
            Ty::Idx { bound, shape } => Ty::Idx { bound: bound.clone(), shape: sh(shape) },
            Ty::Truth(shape) => Ty::Truth(sh(shape)),
            Ty::Tuple(ts) => Ty::Tuple(ts.iter().map(|x| inst(x, b)).collect()),
            other => other.clone(),
        }
    }
    let mut b = HashMap::new();
    for (p, a) in callee.params.iter().zip(args) {
        bind(&p.ty.subst(dim_args), a, &mut b);
    }
    inst(&r_ty.subst(dim_args), &b)
}

/// Fresh bindings above those of a function.
struct Fresh(u32);

impl Fresh {
    fn above(f: &Func) -> Fresh {
        let m = f.params.iter().map(|p| p.var.0).chain(f.body.lets.iter().map(|l| l.var.0)).max().map_or(0, |m| m + 1);
        Fresh(m)
    }
    fn next(&mut self) -> VarId {
        let v = VarId(self.0);
        self.0 += 1;
        v
    }
}

fn mk(var: VarId, ty: Ty, rhs: Rhs, span: presk_syntax::Span) -> Let {
    Let { var, name: None, ty, rhs, span }
}

/// A residual of a call or a fold of the sweep that `g.fwd` makes apart from
/// the `.vjp`'s forward copy: the callee's residuals, or a fold's stacks.
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
enum Made {
    /// The residuals of the call that `let` of `g` makes, on the
    /// arguments the sweep differentiates it at (the `.vjp`'s bindings).
    Call { at: VarId, split: Split, args: Vec<VarId>, dim_args: BTreeMap<DimVar, DimExpr> },
    /// The stack of the accumulators and residuals before each layer of the
    /// fold that `let` of `g` makes.
    Stack { at: VarId },
    /// The last layer's residuals of that fold.
    Last { at: VarId },
}

#[allow(clippy::too_many_arguments)]
fn split_one(prog: &mut Program, v: FuncId, b: &Built, built: &HashMap<FuncId, Built>, splits: &HashMap<FuncId, Split>, steps: &mut HashMap<FuncId, FuncId>, generated: &mut Vec<FuncId>, groups: bool) -> Option<(Split, usize, usize)> {
    let vf = prog.func(v).clone();
    let src = prog.func(b.src).clone();
    let n = src.params.len();
    let ct = vf.params.get(n)?.var;
    let span = vf.span;
    let is_fwd = |x: VarId| x.0 < b.fwd_end && x != ct;
    let fwd_lets: HashMap<VarId, &Let> = vf.body.lets.iter().filter(|l| is_fwd(l.var)).map(|l| (l.var, l)).collect();
    let back: HashMap<VarId, VarId> = b.rename.iter().map(|(s, c)| (*c, *s)).collect();
    let mut fresh = Fresh::above(&vf);
    // Types of the `.vjp`'s bindings and of `g`'s.
    let vtys: HashMap<VarId, Ty> = vf.params.iter().map(|q| (q.var, q.ty.clone())).chain(vf.body.lets.iter().map(|l| (l.var, l.ty.clone()))).collect();
    let stys: HashMap<VarId, Ty> = src.params.iter().map(|q| (q.var, q.ty.clone())).chain(src.body.lets.iter().map(|l| (l.var, l.ty.clone()))).collect();
    let vty = |x: VarId| vtys.get(&x).cloned().unwrap_or(Ty::Tuple(vec![]));
    let sty = |x: VarId| stys.get(&x).cloned().unwrap_or(Ty::Tuple(vec![]));

    // 1. The sweep, its calls of split `.vjp`s read the callee's `.bwd`, its
    // backward folds over a split body carry the residuals.
    let mut sweep: Vec<Let> = vf.body.lets.iter().filter(|l| !is_fwd(l.var)).cloned().collect();
    let mut made: BTreeMap<VarId, (Made, Ty)> = BTreeMap::new();
    let mut must_share: Vec<VarId> = vec![];
    let mut calls_made: Vec<VarId> = vec![];
    let mut carried: HashMap<VarId, (Split, VarId, VarId, Ty)> = HashMap::new(); // fold let of g → (body split, pre2, last, residual type)
    let mut scanned: HashSet<VarId> = HashSet::new(); // fold lets of g whose forward copy's scan is kept
    let mut fold_bwds: Vec<(FuncId, Split, Option<Ty>)> = vec![]; // fold_bwd, body split, carried residual type
    for i in 0..sweep.len() {
        let l = sweep[i].clone();
        match &l.rhs {
            Rhs::Call { f, dim_args, args } if splits.contains_key(f) => {
                // A closure's call has no `let` of `g` to take residuals
                // from: it keeps calling the closure's `.vjp`.
                let Some(&at) = b.calls.get(&l.var) else { continue };
                let s = splits[f].clone();
                let (fwd_args, ct_arg) = args.split_at(args.len() - 1);
                let mut a: Vec<VarId> = s.params.iter().map(|k| fwd_args[*k]).collect();
                if s.fwd.is_some() {
                    let r = fresh.next();
                    let arg_tys: Vec<Ty> = fwd_args.iter().map(|x| vty(*x)).collect();
                    let r_at = at_call(prog.func(built[f].src), &s.r_ty, dim_args, &arg_tys);
                    made.insert(r, (Made::Call { at, split: s.clone(), args: fwd_args.to_vec(), dim_args: dim_args.clone() }, r_at));
                    // Its arguments are in the cone: when one is computed
                    // apart, so is the call, for its residuals.
                    calls_made.extend(fwd_args.iter().copied().filter(|x| is_fwd(*x)));
                    a.push(r);
                }
                a.push(ct_arg[0]);
                sweep[i].rhs = Rhs::Call { f: s.bwd, dim_args: dim_args.clone(), args: a };
            }
            Rhs::Fold { body, n: count, stacked, init, extra, .. } => {
                let Some((at, bf)) = b.folds.iter().find(|(_, x)| x.out == l.var).map(|(a, x)| (*a, x.clone())) else { continue };
                let Some(Rhs::Fold { body: fbody, dim_args, .. }) = src.body.lets.iter().find(|x| x.var == at).map(|x| &x.rhs) else { return None };
                let body_vjp = prog.func(*body).body.lets.iter().find_map(|m| match &m.rhs {
                    Rhs::Call { f, .. } if built.contains_key(f) && built[f].src == *fbody => Some(*f),
                    _ => None,
                });
                let split = body_vjp.and_then(|bv| splits.get(&bv).cloned());
                let Some(s) = split else {
                    scanned.insert(at);
                    continue;
                };
                let Some(Rhs::Fold { stacked: sst, init: sinit, extra: sex, .. }) = src.body.lets.iter().find(|x| x.var == at).map(|x| x.rhs.clone()) else { return None };
                let mut arg_tys: Vec<Ty> = sst.iter().map(|x| crate::ir::unstack_ty(&sty(*x))).collect();
                arg_tys.push(sty(sinit));
                arg_tys.extend(sex.iter().map(|x| sty(*x)));
                let r_f = at_call(prog.func(*fbody), &s.r_ty, dim_args, &arg_tys);
                if s.fwd.is_some() {
                    if !zeroable(&r_f) {
                        keep!(src.name, "a fold's residuals are not all tensors of named shape");
                    }
                    let (pre2, last) = (fresh.next(), fresh.next());
                    let acc_ty = src.body.lets.iter().find(|x| x.var == at)?.ty.clone();
                    let t2 = Ty::Tuple(vec![acc_ty, r_f.clone()]);
                    made.insert(pre2, (Made::Stack { at }, stack_ty(&t2, count)));
                    made.insert(last, (Made::Last { at }, r_f.clone()));
                    must_share.extend(stacked.iter().chain(extra).copied().filter(|x| is_fwd(*x) && *x != bf.pre));
                    must_share.push(b.rename[&src.body.lets.iter().find(|x| x.var == at)?.rhs.vars()[src_fold_init_pos(&src, at)?]]);
                    // The backward fold reads the stack, and starts from the
                    // last layer's residuals.
                    let mut st = stacked.clone();
                    *st.last_mut()? = pre2;
                    if let Rhs::Fold { stacked, .. } = &mut sweep[i].rhs {
                        *stacked = st;
                    }
                    if let Ty::Tuple(ts) = &mut sweep[i].ty {
                        ts.push(r_f.clone());
                    }
                    let a0 = sweep.iter_mut().find(|x| x.var == *init)?;
                    let Rhs::Tuple(vs) = &mut a0.rhs else { return None };
                    vs.push(last);
                    if let Ty::Tuple(ts) = &mut a0.ty {
                        ts.push(r_f.clone());
                    }
                    carried.insert(at, (s.clone(), pre2, last, r_f.clone()));
                    fold_bwds.push((*body, s, Some(r_f)));
                } else {
                    scanned.insert(at);
                    fold_bwds.push((*body, s, None));
                }
            }
            _ => {}
        }
    }

    // 2. What the sweep reads of the forward pass: recomputed when it
    // computes nothing or lies inside one of the sweep's fusion groups,
    // stored otherwise.
    let p = fusion::plan(&vf);
    let mut again: BTreeSet<VarId> = BTreeSet::new();
    let mut params: BTreeSet<usize> = BTreeSet::new();
    let mut stored: BTreeSet<VarId> = BTreeSet::new();
    // The value of a fold that stacks its residuals is `g.fwd`'s, as they are.
    let fold_values: HashSet<VarId> = carried.keys().map(|at| b.rename[at]).collect();
    let mut work: Vec<VarId> = sweep.iter().flat_map(|l| l.rhs.vars()).chain([vf.body.result]).collect();
    while let Some(x) = work.pop() {
        if let Some(k) = vf.params[..n].iter().position(|q| q.var == x) {
            params.insert(k);
            continue;
        }
        if made.contains_key(&x) || fold_values.contains(&x) {
            stored.insert(x);
            continue;
        }
        let Some(l) = fwd_lets.get(&x) else { continue };
        if free(&l.rhs) || p.interior.contains(&x) {
            if again.insert(x) {
                work.extend(l.rhs.vars());
            }
        } else {
            stored.insert(x);
        }
    }
    // A fold whose stack the sweep reads keeps its scan in `g.fwd`.
    for (at, f) in &b.folds {
        if stored.contains(&f.pair) || stored.contains(&f.pre) {
            scanned.insert(*at);
        }
    }

    // 3. `g.fwd`: `g`'s body, its value as `g` computes it; the residuals
    // the sweep reads, shared with it where `g` computes them as the
    // `.vjp`'s forward copy does, computed apart otherwise.
    let pg = fusion::plan(&src);
    let src_lets: HashMap<VarId, &Let> = src.body.lets.iter().map(|l| (l.var, l)).collect();
    // The `.vjp`'s forward values the residuals are made from.
    let mut cone: BTreeSet<VarId> = BTreeSet::new();
    let mut work: Vec<VarId> = stored.iter().copied().filter(|x| fwd_lets.contains_key(x)).collect();
    work.extend(must_share.iter().copied());
    work.extend(calls_made.iter().copied());
    while let Some(x) = work.pop() {
        let Some(l) = fwd_lets.get(&x) else { continue };
        if !cone.insert(x) || fold_values.contains(&x) {
            continue;
        }
        match comp(&p, l) {
            Some((_, inputs)) => work.extend(inputs),
            None => keep!(src.name, format!("%{} is read but inside a group", x.0)),
        }
    }
    // Which of them `g.fwd` shares with `g`'s own computation, in order.
    let pair_of: HashMap<VarId, VarId> = b.folds.iter().flat_map(|(at, f)| [(f.pair, *at), (f.pre, *at)]).collect();
    let mut copied: BTreeSet<VarId> = BTreeSet::new();
    for x in &cone {
        let l = fwd_lets[x];
        let inputs_shared = comp(&p, l).is_some_and(|(_, ins)| ins.iter().all(|i| !copied.contains(i)));
        let same = match back.get(x).and_then(|s| src_lets.get(s)) {
            // No group changes a bit: `g`'s value is the copy's.
            Some(_) if !groups => true,
            Some(sl) if fusible(l) => {
                let map = |vs: Vec<VarId>| vs.into_iter().map(|q| back.get(&q).copied().unwrap_or(q)).collect::<Vec<_>>();
                comp(&p, l).map(|(m, i)| (map(m), map(i))) == comp(&pg, sl)
            }
            Some(_) => true,
            // The scan of a fold kept in `g.fwd`.
            None => pair_of.get(x).is_some_and(|at| scanned.contains(at) && !carried.contains_key(at)),
        };
        if !(same && inputs_shared) {
            copied.insert(*x);
        }
    }
    if let Some(x) = must_share.iter().find(|x| copied.contains(x)) {
        keep!(src.name, format!("%{} (`{}`), an operand of a fold whose residuals are stacked, is computed apart", x.0, fwd_lets[x].name.clone().unwrap_or_default()));
    }
    // A fold's scan, a closure, an application: not computed apart.
    if let Some(x) = copied.iter().find(|x| !matches!(fwd_lets[x].rhs, Rhs::Prim { .. } | Rhs::Call { .. } | Rhs::Proj(..) | Rhs::Field(..) | Rhs::Tuple(_) | Rhs::Record(_) | Rhs::Lit(_)) || pair_of.contains_key(x)) {
        keep!(src.name, format!("%{} would be computed apart, and is a scan, a closure or an application", x.0));
    }

    let r_order: Vec<VarId> = stored.iter().copied().collect();
    let mut anchors: Vec<VarId> = vec![];
    let (fwd_func, r_ty) = loop {
        let (f, img, r_ty) = build_fwd(prog, &src, &vf, b, &made, &carried, &scanned, &copied, &r_order, &anchors, steps, generated);
        if !groups {
            break (f, r_ty);
        }
        let pf = fusion::plan(&f);
        // `g`'s value as `g` computes it.
        for l in src.body.lets.iter().filter(|l| fusible(l)) {
            let fl = f.body.lets.iter().find(|x| x.var == l.var).expect("g.fwd has every let of g");
            if comp(&pg, l) != comp(&pf, fl) {
                keep!(src.name, format!("g.fwd changes the group of %{}", l.var.0));
            }
        }
        // Each residual as the `.vjp` computed it.
        let map = |q: VarId| img.get(&q).copied().unwrap_or(q);
        let mut more = false;
        // (A value no group can hold, a call's or a contraction's, is the
        // `.vjp`'s when its operands are, which the cone holds.)
        for x in cone.iter().filter(|x| fusible(fwd_lets[x])) {
            let l = fwd_lets[x];
            let want = comp(&p, l).map(|(m, i)| (m.into_iter().map(map).collect::<Vec<_>>(), i.into_iter().map(map).collect::<Vec<_>>()));
            let fl = f.body.lets.iter().find(|y| y.var == map(*x)).expect("g.fwd has every residual's computation");
            let got = comp(&pf, fl);
            if want == got {
                continue;
            }
            // A value computed apart that a group of the copies took in:
            // returned with the residuals, it stays in memory as it was.
            if copied.contains(x) && got.is_none() && !anchors.contains(x) {
                anchors.push(*x);
                more = true;
            } else {
                keep!(src.name, format!("residual %{} not computed as the .vjp did: want {:?} got {:?}", x.0, want, got));
            }
        }
        if !more {
            break (f, r_ty);
        }
    };
    let copies = copied.len();

    // 4. `g.bwd`: the residuals, then what the sweep computes again, then the
    // sweep.
    let fwd_needed = !r_order.is_empty();
    let r_param = fresh.next();
    let mut bparams: Vec<Param> = params.iter().map(|k| vf.params[*k].clone()).collect();
    let mut blets: Vec<Let> = vec![];
    let type_of = |x: VarId| -> Ty { made.get(&x).map(|(_, t)| t.clone()).or_else(|| fwd_lets.get(&x).map(|l| l.ty.clone())).expect("a residual has a type") };
    if fwd_needed {
        bparams.push(Param { var: r_param, name: "residuals".into(), ty: r_ty.clone() });
        for (j, x) in r_order.iter().enumerate() {
            blets.push(mk(*x, type_of(*x), Rhs::Proj(r_param, j), span));
        }
    }
    bparams.push(vf.params[n].clone());
    blets.extend(vf.body.lets.iter().filter(|l| again.contains(&l.var)).cloned());
    blets.extend(sweep);
    let bwd_id = FuncId(prog.funcs.len() as u32);
    prog.funcs.push(Func { name: format!("{}.bwd", src.name), origin: Origin::Lambda { parent: b.src }, dims: vf.dims.clone(), params: bparams, ret: vf.ret.clone(), effects: vf.effects.clone(), body: Block { lets: blets, result: vf.body.result }, span });
    generated.push(bwd_id);
    // Its fusion groups are the `.vjp`'s sweep's.
    let pb = fusion::plan(prog.func(bwd_id));
    let moved = groups.then(|| prog.func(bwd_id).body.lets.iter().filter(|l| fusible(l)).find(|l| vf.body.lets.iter().find(|x| x.var == l.var).is_none_or(|orig| comp(&p, orig) != comp(&pb, l))).map(|l| l.var)).flatten();
    if let Some(m) = moved {
        prog.funcs.pop();
        generated.pop();
        keep!(src.name, format!("g.bwd changes the group of %{}", m.0));
    }
    let fwd_id = if fwd_needed {
        let id = FuncId(prog.funcs.len() as u32);
        prog.funcs.push(fwd_func);
        generated.push(id);
        Some(id)
    } else {
        None
    };
    let split = Split { fwd: fwd_id, bwd: bwd_id, params: params.iter().copied().collect(), r_ty: if fwd_needed { r_ty } else { Ty::Tuple(vec![]) } };

    // 5. The backward folds' bodies read the body's `.bwd`.
    let folds = carried.len();
    for (fb, s, carry) in fold_bwds {
        rewrite_fold_bwd(prog, fb, built, &s, carry);
    }

    // 6. `g.vjp` is `g.bwd` after `g.fwd`.
    let mut w = Fresh(n as u32 + 1);
    let mut lets = vec![];
    let mut a: Vec<VarId> = split.params.iter().map(|k| vf.params[*k].var).collect();
    if let Some(fwd) = split.fwd {
        let pv = w.next();
        let rv = w.next();
        lets.push(mk(pv, prog.func(fwd).ret.clone(), Rhs::Call { f: fwd, dim_args: identity_dims(&vf), args: vf.params[..n].iter().map(|q| q.var).collect() }, span));
        lets.push(mk(rv, split.r_ty.clone(), Rhs::Proj(pv, 1), span));
        a.push(rv);
    }
    a.push(ct);
    let cts = w.next();
    lets.push(mk(cts, vf.ret.clone(), Rhs::Call { f: bwd_id, dim_args: identity_dims(&vf), args: a }, span));
    prog.funcs[v.0 as usize].body = Block { lets, result: cts };
    if std::env::var_os("PRESK_STORE_TRACE").is_some() {
        let apart: Vec<String> = copied.iter().map(|x| format!("%{}", x.0)).collect();
        eprintln!("store: {} split: {} residuals, {} parameters, apart [{}], {} folds stacking", src.name, r_order.len(), split.params.len(), apart.join(" "), folds);
    }
    Some((split, copies, folds))
}

/// The position among a fold's operands of its accumulator.
fn src_fold_init_pos(src: &Func, at: VarId) -> Option<usize> {
    match &src.body.lets.iter().find(|x| x.var == at)?.rhs {
        Rhs::Fold { stacked, .. } => Some(stacked.len()),
        _ => None,
    }
}

/// Builds `g.fwd`. Returns it, where each forward value of the `.vjp` is in
/// it, and the type of its residuals.
#[allow(clippy::too_many_arguments)]
fn build_fwd(prog: &mut Program, src: &Func, vf: &Func, b: &Built, made: &BTreeMap<VarId, (Made, Ty)>, carried: &HashMap<VarId, (Split, VarId, VarId, Ty)>, scanned: &HashSet<VarId>, copied: &BTreeSet<VarId>, r_order: &[VarId], anchors: &[VarId], steps: &mut HashMap<FuncId, FuncId>, generated: &mut Vec<FuncId>) -> (Func, HashMap<VarId, VarId>, Ty) {
    let span = src.span;
    let mut fresh = Fresh::above(src);
    let mut img: HashMap<VarId, VarId> = b.rename.iter().map(|(s, c)| (*c, *s)).collect();
    // Calls whose residuals are made from arguments computed apart: the
    // value's call is `g`'s own, the residuals come from a call on the
    // arguments as the `.vjp` computed them.
    let apart_sites: HashSet<VarId> = made.values().filter_map(|(m, _)| match m {
        Made::Call { at, args, .. } if args.iter().any(|a| copied.contains(a)) => Some(*at),
        _ => None,
    }).collect();
    let mut lets: Vec<Let> = vec![];
    let made_at: HashMap<VarId, Vec<(VarId, &Made)>> = made.iter().fold(HashMap::new(), |mut m, (x, (k, _))| {
        let at = match k {
            Made::Call { at, .. } | Made::Stack { at } | Made::Last { at } => *at,
        };
        m.entry(at).or_default().push((*x, k));
        m
    });
    for l in &src.body.lets {
        match &l.rhs {
            Rhs::Call { f: _, dim_args, args } if made_at.contains_key(&l.var) && !apart_sites.contains(&l.var) => {
                let (x, Made::Call { split, .. }) = made_at[&l.var][0] else { unreachable!("a call makes a call's residuals") };
                let r_ty = made[&x].1.clone();
                let pair = fresh.next();
                lets.push(mk(pair, Ty::Tuple(vec![l.ty.clone(), r_ty.clone()]), Rhs::Call { f: split.fwd.expect("residuals come from a .fwd"), dim_args: dim_args.clone(), args: args.clone() }, l.span));
                lets.push(Let { rhs: Rhs::Proj(pair, 0), ..l.clone() });
                let r = fresh.next();
                lets.push(mk(r, r_ty, Rhs::Proj(pair, 1), l.span));
                img.insert(x, r);
            }
            Rhs::Fold { body, dim_args, n, stacked, init, extra, .. } if carried.contains_key(&l.var) => {
                let (s, pre2, last, r_f) = &carried[&l.var];
                let r_f = r_f.clone();
                let z = zeros(&r_f, &mut fresh, &mut lets, l.span);
                let t2 = Ty::Tuple(vec![l.ty.clone(), r_f.clone()]);
                let init2 = fresh.next();
                lets.push(mk(init2, t2.clone(), Rhs::Tuple(vec![*init, z]), l.span));
                let step = fwd_step(prog, *body, stacked.len(), s, steps, generated);
                let pair_ty = Ty::Tuple(vec![t2.clone(), stack_ty(&t2, n)]);
                let pair = fresh.next();
                lets.push(mk(pair, pair_ty, Rhs::Fold { body: step, dim_args: dim_args.clone(), n: n.clone(), stacked: stacked.clone(), init: init2, extra: extra.clone(), index: false, reverse: false, scan: true }, l.span));
                let fin = fresh.next();
                lets.push(mk(fin, t2.clone(), Rhs::Proj(pair, 0), l.span));
                lets.push(Let { rhs: Rhs::Proj(fin, 0), ..l.clone() });
                let (p2, l2) = (fresh.next(), fresh.next());
                lets.push(mk(l2, r_f, Rhs::Proj(fin, 1), l.span));
                lets.push(mk(p2, stack_ty(&t2, n), Rhs::Proj(pair, 1), l.span));
                img.insert(*pre2, p2);
                img.insert(*last, l2);
            }
            Rhs::Fold { body, dim_args, n, stacked, init, extra, index, reverse, .. } if scanned.contains(&l.var) => {
                let bf = &b.folds[&l.var];
                let pre_ty = stack_ty(&l.ty, n);
                let pair = fresh.next();
                lets.push(mk(pair, Ty::Tuple(vec![l.ty.clone(), pre_ty.clone()]), Rhs::Fold { body: *body, dim_args: dim_args.clone(), n: n.clone(), stacked: stacked.clone(), init: *init, extra: extra.clone(), index: *index, reverse: *reverse, scan: true }, l.span));
                let pre = fresh.next();
                lets.push(mk(pre, pre_ty, Rhs::Proj(pair, 1), l.span));
                lets.push(Let { rhs: Rhs::Proj(pair, 0), ..l.clone() });
                img.insert(bf.pair, pair);
                img.insert(bf.pre, pre);
            }
            _ => lets.push(l.clone()),
        }
    }
    // The residuals the `.vjp` computes otherwise than `g`, apart.
    let vlets: HashMap<VarId, &Let> = vf.body.lets.iter().map(|l| (l.var, l)).collect();
    let p = fusion::plan(vf);
    let mut apart: Vec<VarId> = vec![];
    for x in copied {
        match p.groups.get(x) {
            Some(g) => apart.extend(g.members.iter().copied()),
            None => apart.push(*x),
        }
    }
    let order: HashMap<VarId, usize> = vf.body.lets.iter().enumerate().map(|(i, l)| (l.var, i)).collect();
    apart.sort_by_key(|x| order[x]);
    apart.dedup();
    for x in &apart {
        img.insert(*x, fresh.next());
    }
    let back: HashMap<VarId, VarId> = b.rename.iter().map(|(s, c)| (*c, *s)).collect();
    let src_ty: HashMap<VarId, Ty> = src.body.lets.iter().map(|l| (l.var, l.ty.clone())).collect();
    // One call of the callee's `.fwd` for a site apart: its value is the
    // `.vjp`'s copy when the `.vjp` had one, its residuals the site's.
    let site_call = |at: VarId, value: Option<VarId>, img: &mut HashMap<VarId, VarId>, fresh: &mut Fresh, lets: &mut Vec<Let>| {
        let (r, Made::Call { split, args, dim_args, .. }) = made_at[&at][0] else { unreachable!("a call makes a call's residuals") };
        let r_ty = made[&r].1.clone();
        let vty = src_ty[&at].clone();
        let pair = fresh.next();
        let a: Vec<VarId> = args.iter().map(|q| img.get(q).copied().unwrap_or(*q)).collect();
        lets.push(mk(pair, Ty::Tuple(vec![vty.clone(), r_ty.clone()]), Rhs::Call { f: split.fwd.expect("residuals come from a .fwd"), dim_args: dim_args.clone(), args: a }, span));
        if let Some(v) = value {
            lets.push(mk(v, vty, Rhs::Proj(pair, 0), span));
        }
        let rv = fresh.next();
        lets.push(mk(rv, r_ty, Rhs::Proj(pair, 1), span));
        img.insert(r, rv);
    };
    let mut done: HashSet<VarId> = HashSet::new();
    for x in &apart {
        let l = vlets[x];
        match back.get(x) {
            Some(at) if apart_sites.contains(at) => {
                let v = img[x];
                site_call(*at, Some(v), &mut img, &mut fresh, &mut lets);
                done.insert(*at);
            }
            _ => {
                let rhs = l.rhs.map_vars(|q| img.get(&q).copied().unwrap_or(q));
                lets.push(Let { var: img[x], name: None, ty: l.ty.clone(), rhs, span: l.span });
            }
        }
    }
    let mut rest: Vec<VarId> = apart_sites.difference(&done).copied().collect();
    rest.sort_by_key(|x| x.0);
    for at in rest {
        site_call(at, None, &mut img, &mut fresh, &mut lets);
    }
    let all: Vec<VarId> = r_order.iter().chain(anchors).copied().collect();
    let tys: Vec<Ty> = all.iter().map(|x| made.get(x).map(|(_, t)| t.clone()).unwrap_or_else(|| vlets[x].ty.clone())).collect();
    let r_ty = Ty::Tuple(tys);
    let res = fresh.next();
    lets.push(mk(res, r_ty.clone(), Rhs::Tuple(all.iter().map(|x| img.get(x).copied().unwrap_or(*x)).collect()), span));
    let ret = Ty::Tuple(vec![src.ret.clone(), r_ty.clone()]);
    let out = fresh.next();
    lets.push(mk(out, ret.clone(), Rhs::Tuple(vec![src.body.result, res]), span));
    let f = Func { name: format!("{}.fwd", src.name), origin: Origin::Lambda { parent: b.src }, dims: src.dims.clone(), params: src.params.clone(), ret, effects: src.effects.clone(), body: Block { lets, result: out }, span };
    (f, img, r_ty)
}

/// Zeros of every leaf of `t`, as a value of `t`.
fn zeros(t: &Ty, fresh: &mut Fresh, lets: &mut Vec<Let>, span: presk_syntax::Span) -> VarId {
    let v = fresh.next();
    match t {
        Ty::Tuple(ts) => {
            let vs: Vec<VarId> = ts.iter().map(|x| zeros(x, fresh, lets, span)).collect();
            lets.push(mk(v, t.clone(), Rhs::Tuple(vs), span));
        }
        leaf => lets.push(mk(v, leaf.clone(), Rhs::Prim { op: PrimOp::Zeros { ty: leaf.clone() }, args: vec![] }, span)),
    }
    v
}

/// `body.fwd_step(slices…, (acc, _), extra…) = body.fwd(slices…, acc,
/// extra…)`: one layer of a fold that stacks its residuals, the residuals
/// of the layer before in the accumulator it receives, its own in the one it
/// returns.
fn fwd_step(prog: &mut Program, body: FuncId, acc_pos: usize, s: &Split, steps: &mut HashMap<FuncId, FuncId>, generated: &mut Vec<FuncId>) -> FuncId {
    if let Some(f) = steps.get(&body) {
        return *f;
    }
    let bf = prog.func(body).clone();
    let fwd = s.fwd.expect("a fold stacks the residuals of a .fwd");
    let mut fresh = Fresh::above(&bf);
    let mut params: Vec<Param> = bf.params.iter().map(|p| Param { var: fresh.next(), name: p.name.clone(), ty: p.ty.clone() }).collect();
    let t2 = Ty::Tuple(vec![bf.params[acc_pos].ty.clone(), s.r_ty.clone()]);
    params[acc_pos].ty = t2;
    let acc = fresh.next();
    let mut lets = vec![mk(acc, bf.params[acc_pos].ty.clone(), Rhs::Proj(params[acc_pos].var, 0), bf.span)];
    let mut args: Vec<VarId> = params.iter().map(|p| p.var).collect();
    args[acc_pos] = acc;
    let out = fresh.next();
    let ret = prog.func(fwd).ret.clone();
    lets.push(mk(out, ret.clone(), Rhs::Call { f: fwd, dim_args: identity_dims(&bf), args }, bf.span));
    let id = FuncId(prog.funcs.len() as u32);
    prog.funcs.push(Func { name: format!("{}.fwd_step", bf.name), origin: Origin::Lambda { parent: body }, dims: bf.dims.clone(), params, ret, effects: bf.effects.clone(), body: Block { lets, result: out }, span: bf.span });
    generated.push(id);
    steps.insert(body, id);
    id
}

/// A backward fold's body calls its layer's `.bwd`: on the layer's
/// residuals when the fold stacks them (`carry`), carried in the
/// accumulator, the layer's slice of the stack handed on.
fn rewrite_fold_bwd(prog: &mut Program, fb: FuncId, built: &HashMap<FuncId, Built>, s: &Split, carry: Option<Ty>) {
    let mut f = prog.func(fb).clone();
    let mut fresh = Fresh::above(&f);
    let Some(ci) = f.body.lets.iter().position(|l| matches!(&l.rhs, Rhs::Call { f, .. } if built.contains_key(f))) else { return };
    let Rhs::Call { dim_args, args, .. } = f.body.lets[ci].rhs.clone() else { unreachable!() };
    let (fwd_args, ct_arg) = args.split_at(args.len() - 1);
    let mut a: Vec<VarId> = s.params.iter().map(|k| fwd_args[*k]).collect();
    if let Some(r_f) = carry {
        // Parameters: `i`, the slices, the accumulator before the layer (now
        // with the residuals of the layer before), the backward accumulator
        // (now with this layer's residuals), the extras.
        let (acc_pos, a_pos) = pre_positions(&f);
        let acc_i = f.params[acc_pos].clone();
        let pre_i = fresh.next();
        f.params[acc_pos] = Param { var: pre_i, name: "pre".into(), ty: Ty::Tuple(vec![acc_i.ty.clone(), r_f.clone()]) };
        let av = f.params[a_pos].var;
        let Ty::Tuple(ts) = &mut f.params[a_pos].ty else { return };
        let last = ts.len();
        ts.push(r_f.clone());
        let (r_prev, r_cur) = (fresh.next(), fresh.next());
        let head = vec![
            mk(acc_i.var, acc_i.ty.clone(), Rhs::Proj(pre_i, 0), f.span),
            mk(r_prev, r_f.clone(), Rhs::Proj(pre_i, 1), f.span),
            mk(r_cur, r_f.clone(), Rhs::Proj(av, last), f.span),
        ];
        a.push(r_cur);
        let result = f.body.result;
        for l in f.body.lets.iter_mut() {
            if l.var == result
                && let Rhs::Tuple(vs) = &mut l.rhs
            {
                vs.push(r_prev);
                if let Ty::Tuple(ts) = &mut l.ty {
                    ts.push(r_f.clone());
                }
            }
        }
        if let Ty::Tuple(ts) = &mut f.ret {
            ts.push(r_f);
        }
        f.body.lets.splice(0..0, head);
    }
    a.push(ct_arg[0]);
    let ci = f.body.lets.iter().position(|l| matches!(&l.rhs, Rhs::Call { f, .. } if built.contains_key(f))).expect("found above");
    f.body.lets[ci].rhs = Rhs::Call { f: s.bwd, dim_args, args: a };
    prog.funcs[fb.0 as usize] = f;
}

/// In a backward fold's body, `(i, slices…, acc_i, ct, extra…)`: the
/// positions of the accumulator before the layer and of the backward
/// accumulator, the two after the slices of the forward fold's stacked
/// values.
fn pre_positions(f: &Func) -> (usize, usize) {
    let acc = f.params.iter().position(|p| p.name == "acc").expect("a backward fold's body names its accumulator `acc`");
    (acc, acc + 1)
}
