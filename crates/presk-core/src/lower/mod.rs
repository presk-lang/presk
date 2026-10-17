//! Lowering of the typed program to PSK-Core.
//!
//! One Core function per source function (module members get the explicit
//! parameter `θ : params M`), one `init` function per module, one lifted
//! function per lambda. Module-typed parameters (`net : MLP{D, 4*D}`) are
//! compile-time and disappear. Error code `E5xxx` (lowering).

mod expr;
mod init;
pub mod optim;

use std::collections::{BTreeMap, HashMap};

use presk_syntax::{Diagnostic, Span};
use presk_types::check::TypedProgram;
use presk_types::dim::{DimExpr, DimVar};
use presk_types::elab::{ElabProgram, FnId};
use presk_types::ty::{ModRef, ModuleId, Ty};

use crate::ir::{Block, Func, FuncId, Let, Origin, Param, Program, Rhs, VarId};

/// Lowering options.
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// Unroll `stack{L}(M)` into `L` calls (0.1.0's form) instead of a `Fold`;
    /// a debugging aid, `L` must then be a constant.
    pub unroll_stack: bool,
}

pub fn lower(elab: &ElabProgram, typed: &TypedProgram) -> (Program, Vec<Diagnostic>) {
    lower_with(elab, typed, &Options::default())
}

pub fn lower_with(elab: &ElabProgram, typed: &TypedProgram, opts: &Options) -> (Program, Vec<Diagnostic>) {
    let (prog, diags, _) = lower_traced(elab, typed, opts);
    (prog, diags)
}

/// What the lowering made each `let` of a source function or a lambda from
/// (plan 0.13.0 step 1, D134): the rule, the kind of the typed expression it
/// lowered (`call`, `ein`, `index`, a builtin's name, …), whose span the
/// `let` carries. The `init` functions and the optimizers' functions have
/// none: their `let`s come from their module's parameters and their
/// training run as a whole.
#[derive(Clone, Debug, Default)]
pub struct Trace {
    pub rules: HashMap<FuncId, HashMap<VarId, String>>,
}

/// [`lower_with`], and the [`Trace`] of what it made: the program is the
/// same.
pub fn lower_traced(elab: &ElabProgram, typed: &TypedProgram, opts: &Options) -> (Program, Vec<Diagnostic>, Trace) {
    let mut l = Lowerer { elab, typed, opts: opts.clone(), funcs: vec![], diags: vec![], fn_ids: HashMap::new(), init_ids: HashMap::new(), rules: HashMap::new() };
    // Allocate ids first so calls can refer to functions lowered later.
    for (i, f) in elab.fns.iter().enumerate() {
        let id = FuncId(l.funcs.len() as u32);
        l.fn_ids.insert(FnId(i as u32), id);
        l.funcs.push(placeholder(fn_name(elab, FnId(i as u32)), Origin::Source(FnId(i as u32)), f.span));
    }
    for (i, m) in elab.modules.iter().enumerate() {
        let id = FuncId(l.funcs.len() as u32);
        l.init_ids.insert(ModuleId(i as u32), id);
        l.funcs.push(placeholder(format!("{}.init", m.name), Origin::Init(ModRef::Inst { def: ModuleId(i as u32), args: vec![] }), m.span));
    }
    for i in 0..elab.fns.len() {
        l.lower_fn(FnId(i as u32));
    }
    for i in 0..elab.modules.len() {
        l.lower_init(ModuleId(i as u32));
    }
    l.lower_optimizers();
    let trace = Trace { rules: std::mem::take(&mut l.rules) };
    (Program { funcs: l.funcs }, l.diags, trace)
}

fn placeholder(name: String, origin: Origin, span: Span) -> Func {
    Func { name, origin, dims: vec![], params: vec![], ret: Ty::Nat, effects: vec![], body: Block { lets: vec![], result: VarId(0) }, span }
}

pub fn fn_name(elab: &ElabProgram, id: FnId) -> String {
    let f = elab.function(id);
    match f.owner {
        Some(m) => format!("{}.{}", elab.module(m).name, f.name),
        None => f.name.clone(),
    }
}

pub(crate) struct Lowerer<'p> {
    pub(crate) elab: &'p ElabProgram,
    pub(crate) typed: &'p TypedProgram,
    pub(crate) opts: Options,
    pub(crate) funcs: Vec<Func>,
    pub(crate) diags: Vec<Diagnostic>,
    pub(crate) fn_ids: HashMap<FnId, FuncId>,
    pub(crate) init_ids: HashMap<ModuleId, FuncId>,
    /// [`Trace::rules`], filled as each function is finished.
    pub(crate) rules: HashMap<FuncId, HashMap<VarId, String>>,
}

/// The body under construction.
pub(crate) struct Builder {
    pub(crate) lets: Vec<Let>,
    next: u32,
    /// Source names in scope: variables, and how module members are reached.
    pub(crate) scope: Vec<(String, Bound)>,
    /// `θ` of the enclosing module function, if any.
    pub(crate) theta: Option<VarId>,
    pub(crate) owner: Option<ModuleId>,
    pub(crate) fn_id: Option<FnId>,
    /// The Core function id being built (for lambda naming).
    pub(crate) self_id: FuncId,
    pub(crate) lambdas: u32,
    /// The kind of the expression being lowered, which every `let` emitted
    /// meanwhile is recorded with ([`Trace`]).
    pub(crate) rule: String,
    pub(crate) rules: HashMap<VarId, String>,
}

#[derive(Clone, Debug)]
pub(crate) enum Bound {
    Var(VarId),
    /// A module-typed parameter or a `use` member: compile-time, never a value.
    Module,
}

impl Builder {
    pub(crate) fn new(self_id: FuncId) -> Self {
        Builder { lets: vec![], next: 0, scope: vec![], theta: None, owner: None, fn_id: None, self_id, lambdas: 0, rule: String::new(), rules: HashMap::new() }
    }

    pub(crate) fn fresh(&mut self) -> VarId {
        let v = VarId(self.next);
        self.next += 1;
        v
    }

    pub(crate) fn emit(&mut self, name: Option<String>, ty: Ty, rhs: Rhs, span: Span) -> VarId {
        let var = self.fresh();
        self.lets.push(Let { var, name, ty, rhs, span });
        if !self.rule.is_empty() {
            self.rules.insert(var, self.rule.clone());
        }
        var
    }

    pub(crate) fn lookup(&self, name: &str) -> Option<&Bound> {
        self.scope.iter().rev().find(|(n, _)| n == name).map(|(_, b)| b)
    }
}

impl<'p> Lowerer<'p> {
    pub(crate) fn error(&mut self, d: Diagnostic) {
        self.diags.push(d);
    }

    /// The dimension variables a function is generic in: its type parameters,
    /// implicit dimensions, the module's type parameters and the file's dims,
    /// and the dimensions its signature's types mention, another file's
    /// among them (`ctx : idx[m.V][batch: m.B, seq: m.T]`; the reference's
    /// chapter 5, "dimension parameters", plan 0.10.0 step 7b).
    pub(crate) fn fn_dims(&self, id: FnId) -> Vec<DimVar> {
        let f = self.elab.function(id);
        let mut dims: Vec<DimVar> = f.typarams.iter().chain(&f.implicit_dims).copied().collect();
        if let Some(m) = f.owner {
            dims.extend(self.elab.module(m).typarams.iter().copied());
        }
        dims.extend(self.elab.units[f.unit].dims.values().copied());
        for t in f.params.iter().filter_map(|p| p.ty.as_ref()).chain(f.ret.as_ref()) {
            ty_dims(t, &mut dims);
        }
        dims.sort();
        dims.dedup();
        dims
    }

    fn lower_fn(&mut self, id: FnId) {
        let f = self.elab.function(id).clone();
        let core_id = self.fn_ids[&id];
        let mut b = Builder::new(core_id);
        b.owner = f.owner;
        b.fn_id = Some(id);
        let mut params = Vec::new();
        if let Some(mid) = f.owner {
            let m = self.elab.module(mid);
            let theta_ty = Ty::Params(ModRef::Inst { def: mid, args: m.typarams.iter().map(|v| DimExpr::var(*v)).collect() });
            let theta = b.fresh();
            params.push(Param { var: theta, name: "θ".into(), ty: theta_ty });
            b.theta = Some(theta);
            for p in &m.params {
                b.scope.push((p.name.clone(), Bound::Var(theta))); // resolved as a field of θ in `var`
            }
            for _u in &m.uses {
                b.scope.push((_u.name.clone(), Bound::Module));
            }
        }
        for (i, p) in f.params.iter().enumerate() {
            match &p.ty {
                Some(Ty::Module(_)) => b.scope.push((p.name.clone(), Bound::Module)),
                _ => {
                    let Some(ty) = self.typed.param_ty.get(&(id, i)).cloned() else { continue };
                    let v = b.fresh();
                    params.push(Param { var: v, name: p.name.clone(), ty });
                    b.scope.push((p.name.clone(), Bound::Var(v)));
                }
            }
        }
        let Some(result) = self.expr(&f.body, &mut b) else {
            return;
        };
        let ret = self.typed.body_ty.get(&id).cloned().unwrap_or(Ty::Nat);
        let ret = f.ret.clone().unwrap_or(ret);
        let dims = self.fn_dims(id);
        let func = &mut self.funcs[core_id.0 as usize];
        func.dims = dims;
        func.params = params;
        func.ret = ret;
        func.effects = f.effects.clone();
        func.body = Block { lets: b.lets, result };
        self.rules.insert(core_id, b.rules);
    }

    /// Instantiation of a callee's dimensions for a call site.
    pub(crate) fn call_dims(&self, callee: FnId, span: Span) -> BTreeMap<DimVar, DimExpr> {
        let mut map = self.typed.call_subst.get(&span).cloned().unwrap_or_default();
        // Dimensions the checker left unmapped (already the caller's own) map to themselves.
        for d in self.fn_dims(callee) {
            map.entry(d).or_insert_with(|| DimExpr::var(d));
        }
        map
    }
}

/// The dimension variables a type mentions.
fn ty_dims(t: &Ty, out: &mut Vec<DimVar>) {
    fn modref(r: &ModRef, out: &mut Vec<DimVar>) {
        match r {
            ModRef::Inst { args, .. } => args.iter().for_each(|a| out.extend(a.vars())),
            ModRef::Stack { n, inner } => {
                out.extend(n.vars());
                modref(inner, out);
            }
            ModRef::Compose(parts) => parts.iter().for_each(|p| modref(p, out)),
        }
    }
    match t {
        Ty::Real(s) => s.axes.iter().for_each(|a| out.extend(a.len.vars())),
        Ty::Idx { bound, shape } => {
            out.extend(bound.vars());
            shape.axes.iter().for_each(|a| out.extend(a.len.vars()));
        }
        Ty::Truth(s) => s.axes.iter().for_each(|a| out.extend(a.len.vars())),
        Ty::Tuple(ts) => ts.iter().for_each(|t| ty_dims(t, out)),
        Ty::Params(r) | Ty::Module(r) => modref(r, out),
        Ty::Fn { params, ret } => {
            params.iter().for_each(|t| ty_dims(t, out));
            ty_dims(ret, out);
        }
        _ => {}
    }
}
