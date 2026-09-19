//! Automatic differentiation: reverse mode, source to source on PSK-Core.
//!
//! `grad v` where `v = closure f[captured…]` is rewritten to
//! `closure f.grad[captured…]`, where `f.grad(captured…, θ)` calls
//! `f.vjp(captured…, θ, 1)` and projects the adjoint of `θ`. `f.vjp` copies
//! the forward body and appends the reverse sweep: one adjoint accumulator
//! per differentiable variable, updated by the rules of [`rules`] for
//! primitives and by the (memoized, on-demand) VJP of every callee. The
//! result is ordinary PSK-Core, so `grad(grad f)` composes and `verify`
//! applies. Error code `E6xxx`.

pub mod rules;

use std::collections::{BTreeMap, HashMap};

use presk_syntax::{Diagnostic, Span};
use presk_types::dim::DimExpr;
use presk_types::ty::Ty;

use crate::ir::{stack_ty, unstack_ty, Block, Func, FuncId, Let, Lit, Origin, Param, PrimOp, Program, Rhs, VarId};

/// Rewrite every `grad` in the program; returns the diagnostics.
pub fn transform(prog: &mut Program) -> Vec<Diagnostic> {
    let mut ad = Ad { vjps: HashMap::new(), grads: HashMap::new(), diags: vec![] };
    // Rewrite `grad` nodes until none is left (generated functions may contain some).
    loop {
        let mut target = None;
        'outer: for (fi, f) in prog.funcs.iter().enumerate() {
            for (li, l) in f.body.lets.iter().enumerate() {
                if let Rhs::Grad(v) = &l.rhs {
                    target = Some((FuncId(fi as u32), li, *v));
                    break 'outer;
                }
            }
        }
        let Some((fid, li, v)) = target else { break };
        let closure = prog.func(fid).body.lets.iter().find_map(|l| match (&l.rhs, l.var == v) {
            (Rhs::Closure { f, captured }, true) => Some((*f, captured.clone())),
            _ => None,
        });
        let span = prog.func(fid).body.lets[li].span;
        match closure {
            Some((f, captured)) => {
                let n = captured.len();
                let g = ad.grad_fn(prog, f, n, span);
                prog.funcs[fid.0 as usize].body.lets[li].rhs = Rhs::Closure { f: g, captured };
            }
            None => {
                ad.diags.push(Diagnostic::error("E6001", "`grad` of a function value that is not a lambda").label(span, "only `grad(λ θ. …)` is differentiated in 0.1.0"));
                // Leave a zero-cost placeholder so the loop terminates.
                let ty = prog.func(fid).body.lets[li].ty.clone();
                prog.funcs[fid.0 as usize].body.lets[li].rhs = Rhs::Prim { op: PrimOp::ZerosLike, args: vec![v] };
                let _ = ty;
            }
        }
    }
    ad.diags
}

/// `f.vjp` for a named function (the training loop's gradient of the
/// objective), generated on demand after [`transform`].
pub fn vjp_function(prog: &mut Program, f: FuncId) -> (FuncId, Vec<Diagnostic>) {
    let mut ad = Ad { vjps: HashMap::new(), grads: HashMap::new(), diags: vec![] };
    let id = ad.vjp_of(prog, f);
    (id, ad.diags)
}

struct Ad {
    vjps: HashMap<FuncId, FuncId>,
    grads: HashMap<(FuncId, usize), FuncId>,
    diags: Vec<Diagnostic>,
}

/// Emits lets with fresh variables.
pub(crate) struct Emit {
    pub(crate) lets: Vec<Let>,
    next: u32,
}

impl Emit {
    fn new() -> Self {
        Emit { lets: vec![], next: 0 }
    }
    fn fresh(&mut self) -> VarId {
        let v = VarId(self.next);
        self.next += 1;
        v
    }
    pub(crate) fn emit(&mut self, ty: Ty, rhs: Rhs, span: Span) -> VarId {
        let var = self.fresh();
        self.lets.push(Let { var, name: None, ty, rhs, span });
        var
    }
    pub(crate) fn prim(&mut self, ty: Ty, op: PrimOp, args: Vec<VarId>, span: Span) -> VarId {
        self.emit(ty, Rhs::Prim { op, args }, span)
    }
}

/// Whether values of this type carry an adjoint.
pub(crate) fn differentiable(t: &Ty) -> bool {
    match t {
        Ty::Real(_) | Ty::Params(_) => true,
        Ty::Tuple(ts) => ts.iter().any(differentiable),
        _ => false,
    }
}

fn identity_dims(f: &Func) -> BTreeMap<presk_types::dim::DimVar, DimExpr> {
    f.dims.iter().map(|d| (*d, DimExpr::var(*d))).collect()
}

impl Ad {
    /// The body of the backward fold of `fold n body(stacked…; acc; extra…)`:
    /// `(i, s_k[i]…, acc_i, (ct_acc, ct_s…, ct_e…), extra…)` ↦ the accumulator
    /// after applying `body.vjp` at iteration `i`: the accumulator's cotangent
    /// flows to the previous iteration, the stacked inputs' cotangents are
    /// written into slice `i`, the extras' are summed.
    #[allow(clippy::too_many_arguments)]
    fn fold_bwd(&mut self, prog: &mut Program, body: FuncId, dim_args: &BTreeMap<presk_types::dim::DimVar, DimExpr>, src: &Func, acc_ty: &Ty, s_tys: &[Ty], ex_tys: &[Ty], ex_d: &[usize], span: Span) -> FuncId {
        let vjp = self.vjp_of(prog, body);
        let bsrc = prog.func(body).clone();
        let id = FuncId(prog.funcs.len() as u32);
        let mut e = Emit::new();
        let i = e.fresh();
        let thetas: Vec<VarId> = s_tys.iter().map(|_| e.fresh()).collect();
        let acc_i = e.fresh();
        let a = e.fresh();
        let exs: Vec<VarId> = ex_tys.iter().map(|_| e.fresh()).collect();
        let mut acc_tys = vec![acc_ty.clone()];
        acc_tys.extend(s_tys.iter().cloned());
        acc_tys.extend(ex_d.iter().map(|j| ex_tys[*j].clone()));
        let acc_tuple = Ty::Tuple(acc_tys);
        let mut params = vec![Param { var: i, name: "i".into(), ty: Ty::Nat }];
        for (k, t) in thetas.iter().zip(s_tys) {
            params.push(Param { var: *k, name: "slice".into(), ty: unstack_ty(t) });
        }
        params.push(Param { var: acc_i, name: "acc".into(), ty: acc_ty.clone() });
        params.push(Param { var: a, name: "ct".into(), ty: acc_tuple.clone() });
        for (j, v) in exs.iter().enumerate() {
            params.push(Param { var: *v, name: format!("extra{j}"), ty: ex_tys[j].clone() });
        }
        let ct_acc = e.emit(acc_ty.clone(), Rhs::Proj(a, 0), span);
        let ct_s: Vec<VarId> = (0..s_tys.len()).map(|k| e.emit(s_tys[k].clone(), Rhs::Proj(a, 1 + k), span)).collect();
        let ct_e: Vec<VarId> = ex_d.iter().enumerate().map(|(j, jj)| e.emit(ex_tys[*jj].clone(), Rhs::Proj(a, 1 + s_tys.len() + j), span)).collect();
        let ptys: Vec<Ty> = bsrc.params.iter().map(|p| p.ty.subst(dim_args)).collect();
        let mut args: Vec<VarId> = thetas.clone();
        args.push(acc_i);
        args.extend(exs.iter().copied());
        args.push(ct_acc);
        let cts = e.emit(Ty::Tuple(ptys.clone()), Rhs::Call { f: vjp, dim_args: dim_args.clone(), args }, span);
        let k = s_tys.len();
        let mut outs = vec![e.emit(acc_ty.clone(), Rhs::Proj(cts, k), span)];
        for (kk, cs) in ct_s.iter().enumerate() {
            let ct_theta = e.emit(unstack_ty(&s_tys[kk]), Rhs::Proj(cts, kk), span);
            outs.push(e.prim(s_tys[kk].clone(), PrimOp::SetSlice, vec![*cs, i, ct_theta], span));
        }
        for (j, jj) in ex_d.iter().enumerate() {
            let c = e.emit(ex_tys[*jj].clone(), Rhs::Proj(cts, k + 1 + jj), span);
            outs.push(e.prim(ex_tys[*jj].clone(), PrimOp::Add, vec![ct_e[j], c], span));
        }
        let result = e.emit(acc_tuple.clone(), Rhs::Tuple(outs), span);
        prog.funcs.push(Func { name: format!("{}.fold_bwd", src.name), origin: Origin::Lambda { parent: body }, dims: src.dims.clone(), params, ret: acc_tuple, effects: vec![], body: Block { lets: e.lets, result }, span });
        id
    }

    /// `f.grad(captured…, θ…)`: the gradient of `f` with respect to its
    /// parameters after the first `n_captured`.
    fn grad_fn(&mut self, prog: &mut Program, f: FuncId, n_captured: usize, span: Span) -> FuncId {
        if let Some(g) = self.grads.get(&(f, n_captured)) {
            return *g;
        }
        let vjp = self.vjp_of(prog, f);
        let src = prog.func(f).clone();
        let mut e = Emit::new();
        let params: Vec<Param> = src.params.iter().map(|p| Param { var: e.fresh(), name: p.name.clone(), ty: p.ty.clone() }).collect();
        let one_n = e.emit(Ty::Nat, Rhs::Lit(Lit::Nat(1)), span);
        let one = e.prim(Ty::scalar_real(), PrimOp::Real, vec![one_n], span);
        let ret_tuple = Ty::Tuple(src.params.iter().map(|p| p.ty.clone()).collect());
        let mut args: Vec<VarId> = params.iter().map(|p| p.var).collect();
        args.push(one);
        let cts = e.emit(ret_tuple, Rhs::Call { f: vjp, dim_args: identity_dims(&src), args }, span);
        let wrt: Vec<&Param> = src.params.iter().skip(n_captured).collect();
        let (result, ret) = if wrt.len() == 1 {
            (e.emit(wrt[0].ty.clone(), Rhs::Proj(cts, n_captured), span), wrt[0].ty.clone())
        } else {
            let vs: Vec<VarId> = wrt.iter().enumerate().map(|(i, p)| e.emit(p.ty.clone(), Rhs::Proj(cts, n_captured + i), span)).collect();
            let ty = Ty::Tuple(wrt.iter().map(|p| p.ty.clone()).collect());
            (e.emit(ty.clone(), Rhs::Tuple(vs), span), ty)
        };
        let g = FuncId(prog.funcs.len() as u32);
        prog.funcs.push(Func {
            name: format!("{}.grad", src.name),
            origin: Origin::Lambda { parent: f },
            dims: src.dims.clone(),
            params,
            ret,
            effects: src.effects.clone(),
            body: Block { lets: e.lets, result },
            span,
        });
        self.grads.insert((f, n_captured), g);
        g
    }

    /// `f.vjp(params…, ct) -> (adjoints of the params)`.
    fn vjp_of(&mut self, prog: &mut Program, f: FuncId) -> FuncId {
        if let Some(v) = self.vjps.get(&f) {
            return *v;
        }
        // Reserve the id first (a callee may be `f` itself through a closure).
        let id = FuncId(prog.funcs.len() as u32);
        self.vjps.insert(f, id);
        let src = prog.func(f).clone();
        prog.funcs.push(Func { name: format!("{}.vjp", src.name), origin: Origin::Lambda { parent: f }, dims: src.dims.clone(), params: vec![], ret: Ty::Nat, effects: src.effects.clone(), body: Block { lets: vec![], result: VarId(0) }, span: src.span });
        let mut e = Emit::new();
        // Forward pass, renamed.
        let mut rename: HashMap<VarId, VarId> = HashMap::new();
        let mut types: HashMap<VarId, Ty> = HashMap::new();
        let mut params = Vec::new();
        for p in &src.params {
            let nv = e.fresh();
            rename.insert(p.var, nv);
            types.insert(nv, p.ty.clone());
            params.push(Param { var: nv, name: p.name.clone(), ty: p.ty.clone() });
        }
        let ct_param = e.fresh();
        params.push(Param { var: ct_param, name: "ct".into(), ty: src.ret.clone() });
        let rn = |rename: &HashMap<VarId, VarId>, v: &VarId| *rename.get(v).unwrap_or(v);
        // Closures defined in the body, for `apply`.
        let mut closures: HashMap<VarId, (FuncId, Vec<VarId>)> = HashMap::new();
        // For every fold of the body: the stack of its accumulators before each
        // iteration (the forward copy scans), read by the backward fold.
        let mut fold_pre: HashMap<VarId, VarId> = HashMap::new();
        for l in &src.body.lets {
            let rhs = match &l.rhs {
                Rhs::Lit(x) => Rhs::Lit(x.clone()),
                Rhs::Prim { op, args } => Rhs::Prim { op: op.clone(), args: args.iter().map(|a| rn(&rename, a)).collect() },
                Rhs::Call { f, dim_args, args } => Rhs::Call { f: *f, dim_args: dim_args.clone(), args: args.iter().map(|a| rn(&rename, a)).collect() },
                Rhs::Tuple(vs) => Rhs::Tuple(vs.iter().map(|a| rn(&rename, a)).collect()),
                Rhs::Proj(x, i) => Rhs::Proj(rn(&rename, x), *i),
                Rhs::Record(fs) => Rhs::Record(fs.iter().map(|(n, a)| (n.clone(), rn(&rename, a))).collect()),
                Rhs::Field(x, n) => Rhs::Field(rn(&rename, x), n.clone()),
                Rhs::Closure { f, captured } => {
                    let c: Vec<VarId> = captured.iter().map(|a| rn(&rename, a)).collect();
                    closures.insert(l.var, (*f, c.clone()));
                    Rhs::Closure { f: *f, captured: c }
                }
                Rhs::Grad(_) => {
                    self.diags.push(Diagnostic::error("E6001", "nested `grad` inside a differentiated function is not supported in 0.1.0").label(l.span, "second-order differentiation"));
                    Rhs::Lit(Lit::Nat(0))
                }
                Rhs::Apply { fun, args } => Rhs::Apply { fun: rn(&rename, fun), args: args.iter().map(|a| rn(&rename, a)).collect() },
                Rhs::Fold { body, dim_args, n, stacked, init, extra, index, reverse, .. } => {
                    // Scanned, so that the reverse sweep can read the accumulator
                    // before each iteration.
                    let st: Vec<VarId> = stacked.iter().map(|a| rn(&rename, a)).collect();
                    let ex: Vec<VarId> = extra.iter().map(|a| rn(&rename, a)).collect();
                    let pre_ty = stack_ty(&l.ty, n);
                    let pair_ty = Ty::Tuple(vec![l.ty.clone(), pre_ty.clone()]);
                    let pair = e.emit(pair_ty.clone(), Rhs::Fold { body: *body, dim_args: dim_args.clone(), n: n.clone(), stacked: st, init: rn(&rename, init), extra: ex, index: *index, reverse: *reverse, scan: true }, l.span);
                    types.insert(pair, pair_ty);
                    let pre = e.emit(pre_ty.clone(), Rhs::Proj(pair, 1), l.span);
                    types.insert(pre, pre_ty);
                    fold_pre.insert(l.var, pre);
                    Rhs::Proj(pair, 0)
                }
            };
            let nv = e.emit(l.ty.clone(), rhs, l.span);
            e.lets.last_mut().unwrap().name = l.name.clone();
            rename.insert(l.var, nv);
            types.insert(nv, l.ty.clone());
        }
        // Reverse sweep.
        let mut adj: HashMap<VarId, VarId> = HashMap::new();
        adj.insert(rn(&rename, &src.body.result), ct_param);
        for l in src.body.lets.iter().rev() {
            let y = rn(&rename, &l.var);
            let Some(ct) = adj.get(&y).copied() else { continue };
            let span = l.span;
            let contributions: Vec<(VarId, VarId)> = match &l.rhs {
                Rhs::Lit(_) | Rhs::Closure { .. } | Rhs::Grad(_) => vec![],
                Rhs::Fold { body, dim_args, n, stacked, init, extra, index, reverse, .. } => {
                    if *index || *reverse {
                        self.diags.push(Diagnostic::error("E6001", "differentiating a backward fold is not supported").label(span, "second-order differentiation of a stacked module"));
                        continue;
                    }
                    let st: Vec<VarId> = stacked.iter().map(|a| rn(&rename, a)).collect();
                    let ex: Vec<VarId> = extra.iter().map(|a| rn(&rename, a)).collect();
                    let x0 = rn(&rename, init);
                    let pre = fold_pre[&l.var];
                    let acc_ty = l.ty.clone();
                    let s_tys: Vec<Ty> = st.iter().map(|v| types[v].clone()).collect();
                    let ex_tys: Vec<Ty> = ex.iter().map(|v| types[v].clone()).collect();
                    let ex_d: Vec<usize> = (0..ex.len()).filter(|j| differentiable(&ex_tys[*j])).collect();
                    let bwd = self.fold_bwd(prog, *body, dim_args, &src, &acc_ty, &s_tys, &ex_tys, &ex_d, span);
                    // The backward accumulator: (ct of the accumulator, ct of every stacked
                    // input, ct of every differentiable extra), from (ct, zeros…).
                    let mut acc_tys = vec![acc_ty.clone()];
                    acc_tys.extend(s_tys.iter().cloned());
                    acc_tys.extend(ex_d.iter().map(|j| ex_tys[*j].clone()));
                    let mut init_vs = vec![ct];
                    for v in &st {
                        init_vs.push(e.prim(types[v].clone(), PrimOp::ZerosLike, vec![*v], span));
                    }
                    for j in &ex_d {
                        init_vs.push(e.prim(ex_tys[*j].clone(), PrimOp::ZerosLike, vec![ex[*j]], span));
                    }
                    let acc_tuple = Ty::Tuple(acc_tys);
                    let acc0 = e.emit(acc_tuple.clone(), Rhs::Tuple(init_vs), span);
                    let mut st_bwd = st.clone();
                    st_bwd.push(pre);
                    let out = e.emit(acc_tuple, Rhs::Fold { body: bwd, dim_args: identity_dims(&src), n: n.clone(), stacked: st_bwd, init: acc0, extra: ex.clone(), index: true, reverse: true, scan: false }, span);
                    let mut cs = vec![(x0, e.emit(acc_ty.clone(), Rhs::Proj(out, 0), span))];
                    for (k, v) in st.iter().enumerate() {
                        cs.push((*v, e.emit(s_tys[k].clone(), Rhs::Proj(out, 1 + k), span)));
                    }
                    for (j, jj) in ex_d.iter().enumerate() {
                        cs.push((ex[*jj], e.emit(ex_tys[*jj].clone(), Rhs::Proj(out, 1 + st.len() + j), span)));
                    }
                    cs
                }
                Rhs::Prim { op, args } => {
                    let args: Vec<VarId> = args.iter().map(|a| rn(&rename, a)).collect();
                    let tys: Vec<Ty> = args.iter().map(|a| types[a].clone()).collect();
                    match rules::prim_vjp(&mut e, op, &args, &tys, y, &l.ty, ct, span) {
                        Ok(cs) => cs.into_iter().map(|(i, c)| (args[i], c)).collect(),
                        Err(msg) => {
                            self.diags.push(Diagnostic::error("E6002", format!("cannot differentiate `{}`: {msg}", op.name())).label(span, "in a differentiated function"));
                            vec![]
                        }
                    }
                }
                Rhs::Call { f: g, dim_args, args } => {
                    let args: Vec<VarId> = args.iter().map(|a| rn(&rename, a)).collect();
                    let gv = self.vjp_of(prog, *g);
                    let gsrc = prog.func(*g).clone();
                    let mut a = args.clone();
                    a.push(ct);
                    // Adjoint types are the callee's parameter types, instantiated by the call.
                    let ptys: Vec<Ty> = gsrc.params.iter().map(|p| p.ty.subst(dim_args)).collect();
                    let cts = e.emit(Ty::Tuple(ptys.clone()), Rhs::Call { f: gv, dim_args: dim_args.clone(), args: a }, span);
                    args.iter().enumerate().filter(|(i, _)| differentiable(&ptys[*i])).map(|(i, arg)| (*arg, e.emit(ptys[i].clone(), Rhs::Proj(cts, i), span))).collect()
                }
                Rhs::Tuple(vs) => vs.iter().enumerate().filter(|(_, v)| differentiable(&types[&rn(&rename, v)])).map(|(i, v)| (rn(&rename, v), e.emit(types[&rn(&rename, v)].clone(), Rhs::Proj(ct, i), span))).collect(),
                Rhs::Proj(x, i) => {
                    let x = rn(&rename, x);
                    let z = e.prim(types[&x].clone(), PrimOp::ZerosLike, vec![x], span);
                    vec![(x, e.prim(types[&x].clone(), PrimOp::SetElem { index: *i }, vec![z, ct], span))]
                }
                Rhs::Record(fields) => fields.iter().filter(|(_, v)| differentiable(&types[&rn(&rename, v)])).map(|(n, v)| (rn(&rename, v), e.emit(types[&rn(&rename, v)].clone(), Rhs::Field(ct, n.clone()), span))).collect(),
                Rhs::Field(x, name) => {
                    let x = rn(&rename, x);
                    let z = e.prim(types[&x].clone(), PrimOp::ZerosLike, vec![x], span);
                    vec![(x, e.prim(types[&x].clone(), PrimOp::SetField { name: name.clone() }, vec![z, ct], span))]
                }
                Rhs::Apply { fun, args } => {
                    let args: Vec<VarId> = args.iter().map(|a| rn(&rename, a)).collect();
                    let fun_r = rn(&rename, fun);
                    let Some((g, captured)) = closures.get(&fun_r).cloned() else {
                        self.diags.push(Diagnostic::error("E6001", "application of an unknown function value in a differentiated function").label(span, "the function value must be a lambda of the same body"));
                        continue;
                    };
                    let gv = self.vjp_of(prog, g);
                    let gsrc = prog.func(g).clone();
                    let mut a: Vec<VarId> = captured.clone();
                    a.extend(args.iter().copied());
                    a.push(ct);
                    let tuple_ty = Ty::Tuple(gsrc.params.iter().map(|p| p.ty.clone()).collect());
                    let cts = e.emit(tuple_ty, Rhs::Call { f: gv, dim_args: identity_dims(&src), args: a }, span);
                    let all: Vec<VarId> = captured.iter().chain(args.iter()).copied().collect();
                    all.iter().enumerate().filter(|(i, _)| differentiable(&gsrc.params[*i].ty)).map(|(i, v)| (*v, e.emit(gsrc.params[i].ty.clone(), Rhs::Proj(cts, i), span))).collect()
                }
            };
            for (var, c) in contributions {
                if !differentiable(&types[&var]) {
                    continue;
                }
                let acc = match adj.get(&var) {
                    Some(prev) => e.prim(types[&var].clone(), PrimOp::Add, vec![*prev, c], span),
                    None => c,
                };
                adj.insert(var, acc);
            }
        }
        // Result: the adjoint of every parameter (zeros when nothing flowed).
        let mut outs = Vec::new();
        for p in &params[..params.len() - 1] {
            let a = match adj.get(&p.var) {
                Some(a) => *a,
                None => e.prim(p.ty.clone(), PrimOp::ZerosLike, vec![p.var], src.span),
            };
            outs.push(a);
        }
        let ret = Ty::Tuple(params[..params.len() - 1].iter().map(|p| p.ty.clone()).collect());
        let result = e.emit(ret.clone(), Rhs::Tuple(outs), src.span);
        let func = &mut prog.funcs[id.0 as usize];
        func.params = params;
        func.ret = ret;
        func.body = Block { lets: e.lets, result };
        id
    }
}
