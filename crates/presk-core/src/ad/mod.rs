//! Automatic differentiation: reverse mode, source to source on PSK-Core.
//!
//! `grad v` where `v = closure f[captured…]` is rewritten to
//! `closure f.grad[captured…]`, where `f.grad(captured…, θ)` calls
//! `f.vjp(captured…, θ, 1)` and projects the adjoint of `θ`. `f.vjp` copies
//! the forward body and appends the reverse sweep: one adjoint accumulator
//! per differentiable variable, updated by the rules of [`rules`] for
//! primitives and by the (memoized, on-demand) VJP of every callee, and of
//! every lambda of the body that the body applies. The result is ordinary
//! PSK-Core, so `verify` applies. It does not compose: a `grad` inside a
//! differentiated function, a backward fold and the adjoint primitives are
//! refused (`E6001`, `E6002`), so `grad(grad f)` is not differentiated.
//! Error code `E6xxx`.
//!
//! Since 0.10.0 step 3 the functions a differentiation generates lose their
//! dead code before anything runs them ([`prune`]): what [`transform`] and
//! [`vjp_function`] return is the program without it. Since step 4 a gradient
//! keeps what its forward pass computed ([`store`], [`Remat::Store`], the
//! default): each `.vjp` is split into a `.fwd` that returns its value and
//! its residuals and a `.bwd` that reads them, so no forward pass runs twice;
//! [`Remat::Recompute`] keeps the recomputing `.vjp`s. Both keep every fusion
//! group of what a gradient computes as the recomputing `.vjp` had it, which
//! a `bf16` site needs for its bits, unless told that no realization run
//! can see a group ([`Options::groups`], step 4b).

pub mod prune;
pub mod rules;
pub mod store;

use std::collections::{BTreeMap, HashMap};

use presk_syntax::{Diagnostic, Span};
use presk_types::dim::DimExpr;
use presk_types::ty::Ty;

use crate::ir::{stack_ty, unstack_ty, Block, Func, FuncId, Let, Lit, Origin, Param, PrimOp, Program, Rhs, VarId};

/// What a gradient does with the values its forward pass computed (plan
/// 0.10.0 §3.2, question 76): keep them for its reverse sweep, or compute
/// them again inside each callee's `.vjp`. The two give the same bits; they
/// differ in time and memory. `presk train`, `run` and `export` take it as
/// `--remat store|recompute`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Remat {
    /// Each `.vjp` split into `.fwd` and `.bwd` ([`store`]).
    #[default]
    Store,
    /// Each `.vjp` copies its function's body and recomputes it: every
    /// version before 0.10.0 step 4.
    Recompute,
}

impl Remat {
    pub fn parse(s: &str) -> Result<Remat, String> {
        match s {
            "store" => Ok(Remat::Store),
            "recompute" => Ok(Remat::Recompute),
            other => Err(format!("`{other}` is not a policy: store or recompute")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Remat::Store => "store",
            Remat::Recompute => "recompute",
        }
    }
}

/// How a differentiation is made: what a gradient does with its forward
/// pass's values, and whether it keeps the fusion groups (plan 0.10.0 steps
/// 4 and 4b). The default stores and keeps them; a [`Remat`] alone keeps
/// them too.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    pub remat: Remat,
    /// Keep every fusion group of what a gradient computes as the
    /// recomputing `.vjp` had it (D91, D92): the dead code holds a reader
    /// that keeps a value outside a group, and the split computes apart a
    /// residual the function's own groups compute otherwise. A group
    /// computes its members in its root's arithmetic type and stores only
    /// the root, so its partition decides a value's bits under a `bf16`
    /// site, computed in `f32`, and under a site typed apart from its
    /// function (`presk_rt::Realization::groups_visible`). `false` when no
    /// realization the command runs is either: every group then computes
    /// what its members would alone, nothing is held, and the split shares
    /// every residual with the forward pass (D94).
    pub groups: bool,
}

impl Default for Options {
    fn default() -> Options {
        Options { remat: Remat::Store, groups: true }
    }
}

impl From<Remat> for Options {
    fn from(remat: Remat) -> Options {
        Options { remat, groups: true }
    }
}

/// Rewrite every `grad` in the program, then remove the dead code of what
/// that generated, the forward passes stored ([`Remat::Store`]); returns the
/// diagnostics.
pub fn transform(prog: &mut Program) -> Vec<Diagnostic> {
    transform_as(prog, Options::default())
}

/// [`transform`] under a policy, and with or without the fusion groups.
pub fn transform_as(prog: &mut Program, opts: impl Into<Options>) -> Vec<Diagnostic> {
    transform_opts(prog, true, opts.into())
}

/// [`transform`], recomputing, with the dead code of the generated functions
/// left in place when `remove_dead` is false: what every version before
/// 0.10.0 step 3 ran, kept to check that removing it changes no value.
pub fn transform_with(prog: &mut Program, remove_dead: bool) -> Vec<Diagnostic> {
    transform_opts(prog, remove_dead, Remat::Recompute.into())
}

/// Storing needs the dead code removed first: a residual is what the sweep
/// that survives reads.
fn transform_opts(prog: &mut Program, remove_dead: bool, opts: Options) -> Vec<Diagnostic> {
    let mut ad = Ad::new(opts.groups);
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
        let Some((fid, li, v)) = target else {
            if remove_dead && !ad.diags.iter().any(|d| d.is_error()) {
                // The program applies a `.grad` as a function value: its
                // result is read whole.
                let grads: Vec<FuncId> = ad.grads.values().copied().collect();
                prune::prune(prog, &ad.generated, &grads, ad.groups);
                if opts.remat == Remat::Store {
                    ad.store(prog, &grads);
                }
            }
            break;
        };
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

/// `f.vjp` for a named function (the gradient of a training objective, which
/// `presk solve --step` measures), generated on demand after [`transform`],
/// without its dead code; its result, read by its caller outside the
/// program, stays whole.
pub fn vjp_function(prog: &mut Program, f: FuncId) -> (FuncId, Vec<Diagnostic>) {
    vjp_function_as(prog, f, Options::default())
}

/// [`vjp_function`] under a policy, and with or without the fusion groups.
pub fn vjp_function_as(prog: &mut Program, f: FuncId, opts: impl Into<Options>) -> (FuncId, Vec<Diagnostic>) {
    let opts = opts.into();
    let mut ad = Ad::new(opts.groups);
    let id = ad.vjp_of(prog, f);
    if !ad.diags.iter().any(|d| d.is_error()) {
        prune::prune(prog, &ad.generated, &[id], ad.groups);
        if opts.remat == Remat::Store {
            ad.store(prog, &[id]);
        }
    }
    (id, ad.diags)
}

/// `f.vjp` and `f.value_vjp` for a training objective, generated together
/// after [`transform`], without their dead code (plan 0.10.0 step 3).
/// `f.value_vjp(params…, ct)` returns `f`'s value, then the cotangent of
/// every parameter, from one forward pass: a training step is one call.
/// Both results, read outside the program, stay whole.
pub fn value_vjp_function(prog: &mut Program, f: FuncId) -> ValueVjp {
    value_vjp_function_as(prog, f, Options::default())
}

/// [`value_vjp_function`] under a policy. Storing, `f.value_vjp` is
/// `f.fwd`'s value and `f.bwd`'s cotangents, one forward pass, and computes
/// nothing again for the value (`again` is empty): `f.fwd`'s bindings are
/// `f`'s, and a realization or a plan that states a site of `f` states it of
/// `f.fwd`.
pub fn value_vjp_function_as(prog: &mut Program, f: FuncId, opts: impl Into<Options>) -> ValueVjp {
    let opts = opts.into();
    let mut ad = Ad::new(opts.groups);
    let vjp = ad.vjp_of(prog, f);
    let value_vjp = ad.reserve(prog, f, "value_vjp");
    let mut again = ad.build_vjp(prog, f, value_vjp, true);
    if !ad.diags.iter().any(|d| d.is_error()) {
        prune::prune(prog, &ad.generated, &[vjp, value_vjp], ad.groups);
        if opts.remat == Remat::Store && ad.split_value_vjp(prog, f, vjp, value_vjp) {
            again = vec![];
        }
    }
    ValueVjp { vjp, value_vjp, again, diags: ad.diags }
}

/// What [`value_vjp_function`] generated.
pub struct ValueVjp {
    pub vjp: FuncId,
    pub value_vjp: FuncId,
    /// The `let`s of the objective's body that `value_vjp` computes again
    /// for the value, and the `let` of `value_vjp` that does (see
    /// `Ad::value_of`): a realization or a plan that states something of
    /// the objective's sites states it of these.
    pub again: Vec<(VarId, VarId)>,
    pub diags: Vec<Diagnostic>,
}

/// Where the adjoint of each forward value lives inside a generated `.vjp`:
/// `forward function → (its vjp, forward var → adjoint var in the vjp)`.
///
/// The reverse sweep already computes `∂output/∂v` for every intermediate —
/// that is what a reverse sweep *is* — and then throws all of them away but
/// the parameters'. This table keeps the correspondence so that the
/// sensitivity analysis can read them back (0.5.0 step 5) without a second
/// differentiation or a perturbation per site.
pub type AdjointMap = HashMap<FuncId, (FuncId, HashMap<VarId, VarId>)>;

/// The same as [`vjp_function`], also returning where each forward value's
/// adjoint ended up. Its dead code stays: the sensitivity analysis reads the
/// adjoint of every forward value, read by the sweep or not.
pub fn vjp_with_adjoints(prog: &mut Program, f: FuncId) -> (FuncId, AdjointMap, Vec<Diagnostic>) {
    let mut ad = Ad::new(true);
    let id = ad.vjp_of(prog, f);
    (id, ad.adjoints, ad.diags)
}

struct Ad {
    vjps: HashMap<FuncId, FuncId>,
    grads: HashMap<(FuncId, usize), FuncId>,
    diags: Vec<Diagnostic>,
    /// Filled as each vjp is built; see [`AdjointMap`].
    adjoints: AdjointMap,
    /// Every function this differentiation added, for [`prune`].
    generated: Vec<FuncId>,
    /// What each `.vjp` built needs to be split into `.fwd` and `.bwd`
    /// ([`store`]).
    built: HashMap<FuncId, Built>,
    /// [`Options::groups`].
    groups: bool,
}

/// How a `.vjp` was built from its function, for [`store`]: which of its
/// bindings are the forward copy's, and which `let` of the function each
/// call and fold of its sweep differentiates.
#[derive(Clone, Debug)]
pub(crate) struct Built {
    /// The function differentiated.
    pub src: FuncId,
    /// Its parameters and `let`s → their copies in the `.vjp`.
    pub rename: HashMap<VarId, VarId>,
    /// Every binding below it is a parameter, the cotangent, or the forward
    /// copy's; the sweep's are above.
    pub fwd_end: u32,
    /// A call of the sweep (the callee's `.vjp`) → the `let` of `src` whose
    /// call it differentiates.
    pub calls: HashMap<VarId, VarId>,
    /// A fold of `src` (its `let`) → its scan in the forward copy and its
    /// backward fold in the sweep.
    pub folds: HashMap<VarId, BuiltFold>,
}

/// One fold of a `.vjp`: the forward copy scans (`pair`, and `pre`, the
/// stack of the accumulators before each iteration), the sweep folds
/// backwards (`out`, over `bwd`).
#[derive(Clone, Debug)]
pub(crate) struct BuiltFold {
    pub pair: VarId,
    pub pre: VarId,
    pub out: VarId,
    pub bwd: FuncId,
}

/// Emits lets with fresh variables.
pub(crate) struct Emit {
    pub(crate) lets: Vec<Let>,
    pub(crate) next: u32,
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
    /// Splits the `.vjp`s this differentiation built ([`store`]), then removes
    /// the dead code of what that made; `fixed` as for [`prune::prune`].
    fn store(&mut self, prog: &mut Program, fixed: &[FuncId]) -> store::Stored {
        let built = std::mem::take(&mut self.built);
        let out = store::split(prog, &built, &mut self.generated, self.groups);
        self.built = built;
        prune::prune(prog, &self.generated, fixed, self.groups);
        out
    }

    /// Stores `f.vjp`'s forward pass, and makes `f.value_vjp` read `f.fwd`'s
    /// value: whether `f.vjp` was split.
    fn split_value_vjp(&mut self, prog: &mut Program, f: FuncId, vjp: FuncId, value_vjp: FuncId) -> bool {
        let built = std::mem::take(&mut self.built);
        let splits = store::split_all(prog, &built, &mut self.generated, self.groups);
        self.built = built;
        let done = match splits.get(&vjp) {
            Some(s) => {
                store::value_wrapper(prog, f, vjp, value_vjp, s);
                true
            }
            None => false,
        };
        prune::prune(prog, &self.generated, &[vjp, value_vjp], self.groups);
        done
    }

    fn new(groups: bool) -> Ad {
        Ad { vjps: HashMap::new(), grads: HashMap::new(), diags: vec![], adjoints: AdjointMap::new(), generated: vec![], built: HashMap::new(), groups }
    }

    /// Adds `f.<suffix>`, empty, to the program, to be filled once its id is
    /// known (a callee may be `f` itself through a closure).
    fn reserve(&mut self, prog: &mut Program, f: FuncId, suffix: &str) -> FuncId {
        let id = FuncId(prog.funcs.len() as u32);
        let src = prog.func(f);
        let func = Func { name: format!("{}.{suffix}", src.name), origin: Origin::Lambda { parent: f }, dims: src.dims.clone(), params: vec![], ret: Ty::Nat, effects: src.effects.clone(), body: Block { lets: vec![], result: VarId(0) }, span: src.span };
        prog.funcs.push(func);
        self.generated.push(id);
        id
    }

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
        self.generated.push(id);
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
        self.generated.push(g);
        self.grads.insert((f, n_captured), g);
        g
    }

    /// `f.vjp(params…, ct) -> (adjoints of the params)`.
    fn vjp_of(&mut self, prog: &mut Program, f: FuncId) -> FuncId {
        if let Some(v) = self.vjps.get(&f) {
            return *v;
        }
        let id = self.reserve(prog, f, "vjp");
        self.vjps.insert(f, id);
        let _ = self.build_vjp(prog, f, id, false);
        id
    }

    /// Fills `id`, reserved, with `f.vjp`, or with `f.value_vjp(params…, ct)
    /// -> (value, adjoints of the params)` when `value`. The two share their
    /// bindings' numbers up to the reverse sweep's end, and so their nests.
    /// Returns what [`Ad::value_of`] computed again.
    fn build_vjp(&mut self, prog: &mut Program, f: FuncId, id: FuncId, value: bool) -> Vec<(VarId, VarId)> {
        let src = prog.func(f).clone();
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
        let mut fold_pair: HashMap<VarId, VarId> = HashMap::new();
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
                    fold_pair.insert(l.var, pair);
                    Rhs::Proj(pair, 0)
                }
            };
            let nv = e.emit(l.ty.clone(), rhs, l.span);
            e.lets.last_mut().unwrap().name = l.name.clone();
            rename.insert(l.var, nv);
            types.insert(nv, l.ty.clone());
        }
        let fwd_end = e.next;
        let mut calls: HashMap<VarId, VarId> = HashMap::new();
        let mut folds: HashMap<VarId, BuiltFold> = HashMap::new();
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
                    folds.insert(l.var, BuiltFold { pair: fold_pair[&l.var], pre, out, bwd });
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
                    calls.insert(cts, l.var);
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
                    // `closures` is keyed by the source body's variables.
                    let Some((g, captured)) = closures.get(fun).cloned() else {
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
        let mut tys: Vec<Ty> = params[..params.len() - 1].iter().map(|p| p.ty.clone()).collect();
        let mut again = vec![];
        if value {
            let v;
            (v, again) = Self::value_of(&mut e, &src, &rename);
            outs.insert(0, v);
            tys.insert(0, src.ret.clone());
        }
        let ret = Ty::Tuple(tys);
        let result = e.emit(ret.clone(), Rhs::Tuple(outs), src.span);
        if !value {
            // Keep the correspondence before it is lost: which variable of
            // this vjp holds `∂output/∂v` for each `v` of the forward body.
            let table: HashMap<VarId, VarId> = src
                .body
                .lets
                .iter()
                .filter_map(|l| adj.get(&rn(&rename, &l.var)).map(|a| (l.var, *a)))
                .collect();
            self.adjoints.insert(f, (id, table));
        }
        if !value {
            self.built.insert(id, Built { src: f, rename: rename.clone(), fwd_end, calls, folds });
        }
        let func = &mut prog.funcs[id.0 as usize];
        func.params = params;
        func.ret = ret;
        func.body = Block { lets: e.lets, result };
        again
    }

    /// `f`'s value in `f.value_vjp`, computed as `f` computes it. The forward
    /// copy runs `f`'s operations in `f`'s order, but not as `f`: the reverse
    /// sweep reads some of its values, and a value read outside a fusion
    /// group leaves it (in `bf16`, whose groups keep their intermediates in
    /// `f32`, the copy's value could then differ from `f`'s in its last
    /// bits); and a realization may give the copy's sites, `f.vjp`'s, other
    /// types than `f`'s. So every primitive of `f`'s body is computed again
    /// for the value, with every `let` that reads one, read by nothing else,
    /// so that its fusion groups are `f`'s; the caller gives them `f`'s
    /// types. The calls and folds, whose kernels are their callees', and the
    /// projections, which compute nothing, are the forward copy's. Returns
    /// the value and what was computed again, `f`'s `let` and its copy's.
    fn value_of(e: &mut Emit, src: &Func, rename: &HashMap<VarId, VarId>) -> (VarId, Vec<(VarId, VarId)>) {
        let mut again: Vec<(VarId, VarId)> = vec![];
        let mut map: HashMap<VarId, VarId> = HashMap::new();
        for l in &src.body.lets {
            if !matches!(l.rhs, Rhs::Prim { .. }) && !l.rhs.vars().iter().any(|v| map.contains_key(v)) {
                continue;
            }
            let rhs = l.rhs.map_vars(|v| map.get(&v).or_else(|| rename.get(&v)).copied().unwrap_or(v));
            let nv = e.emit(l.ty.clone(), rhs, l.span);
            e.lets.last_mut().unwrap().name = l.name.clone();
            map.insert(l.var, nv);
            again.push((l.var, nv));
        }
        let r = src.body.result;
        (map.get(&r).or_else(|| rename.get(&r)).copied().unwrap_or(r), again)
    }
}
