//! Shape typing of function bodies and checking of configuration blocks.
//!
//! Every function (top-level or module member) is checked once, generically:
//! its type parameters, the file's `dim`s and its `...` prefix are rigid.
//! At a call site the callee's signature is *instantiated* with fresh
//! inference dimensions and a fresh flexible prefix, which unification then
//! solves. Error codes `E2xxx`.

mod builtins;
mod config;
mod expr;
pub mod inference;
mod recursion;
pub mod training;
mod unify;

pub use recursion::module_cycles;

use std::collections::{BTreeMap, HashMap, HashSet};

use presk_syntax::ast;
use presk_syntax::{Diagnostic, SourceMap, Span};

use crate::dim::{DimExpr, DimVar};
use crate::elab::{ElabProgram, FnId};
use crate::ty::{Axis, ModRef, ModuleId, PrefixVar, Shape, Ty, TyVar};

/// What a call site resolved to.
#[derive(Clone, Debug)]
pub enum Callee {
    /// A user function; `module` is the instance for members, `explicit_theta`
    /// tells whether the parameter tree is the first argument at the call.
    Fn { id: FnId, module: Option<ModRef>, explicit_theta: bool },
    Builtin(String),
    /// A value of function type (a lambda, `grad(f)`).
    Value,
}

/// A typed `ein`: operands with their (relabeled) axes, the output axes and
/// the contracted names.
#[derive(Clone, Debug)]
pub struct EinInfo {
    pub operands: Vec<(Span, Vec<Axis>)>,
    pub output: Vec<Axis>,
    pub contracted: Vec<String>,
}

/// Side tables over the syntax trees, keyed by expression span.
#[derive(Default, Debug)]
pub struct TypedProgram {
    pub ty_of: HashMap<Span, Ty>,
    pub callees: HashMap<Span, Callee>,
    pub eins: HashMap<Span, EinInfo>,
    /// Inferred type of every function body.
    pub body_ty: HashMap<FnId, Ty>,
    /// `let` bindings in order: (function, name, type).
    pub lets: Vec<(FnId, String, Ty)>,
    /// For `axis a` arguments: the position of the axis in the tensor argument.
    pub axis_pos: HashMap<Span, usize>,
    /// Type of every function parameter (declared or inferred), by (function, index).
    pub param_ty: HashMap<(FnId, usize), Ty>,
    /// For every call site: the callee's dimension variables (type parameters,
    /// implicit and file-level dims, module type parameters) mapped to the
    /// caller's dimension expressions.
    pub call_subst: HashMap<Span, BTreeMap<DimVar, DimExpr>>,
    /// The `training run` blocks of the main file, resolved and typed.
    pub training: Vec<training::TrainingRun>,
    /// The `inference serve` blocks of the main file, resolved.
    pub inference: Vec<inference::InferenceServe>,
}

pub fn check(sm: &SourceMap, prog: &ElabProgram) -> (TypedProgram, Vec<Diagnostic>) {
    let mut c = Checker {
        prog,
        sm,
        diags: vec![],
        typed: TypedProgram::default(),
        dim_subst: BTreeMap::new(),
        prefix_subst: BTreeMap::new(),
        ty_subst: BTreeMap::new(),
        next_dim: prog.dims.len() as u32,
        next_prefix: prog.prefixes,
        next_tyvar: 0,
        flexible: HashSet::new(),
        infer_dims: HashSet::new(),
        quiet: 0,
        fn_id: None,
        unit: prog.main,
        calls: vec![],
    };
    c.diags.extend(recursion::module_cycles(prog));
    for (i, f) in prog.fns.iter().enumerate() {
        c.check_fn(FnId(i as u32), f);
    }
    let calls = std::mem::take(&mut c.calls);
    c.diags.extend(recursion::fn_cycles(prog, &calls));
    for (i, m) in prog.modules.iter().enumerate() {
        c.check_inits(ModuleId(i as u32), m);
    }
    for u in 0..prog.units.len() {
        c.check_configs(u);
    }
    c.zonk_tables();
    (c.typed, c.diags)
}

pub(crate) struct Checker<'p> {
    pub(crate) prog: &'p ElabProgram,
    #[allow(dead_code)]
    pub(crate) sm: &'p SourceMap,
    pub(crate) diags: Vec<Diagnostic>,
    pub(crate) typed: TypedProgram,
    pub(crate) dim_subst: BTreeMap<DimVar, DimExpr>,
    pub(crate) prefix_subst: BTreeMap<PrefixVar, Shape>,
    pub(crate) ty_subst: BTreeMap<TyVar, Ty>,
    next_dim: u32,
    next_prefix: u32,
    next_tyvar: u32,
    /// Prefixes created at call sites (solvable); the others are rigid.
    pub(crate) flexible: HashSet<PrefixVar>,
    /// Dimensions created at call sites (solvable); the others are rigid.
    pub(crate) infer_dims: HashSet<DimVar>,
    /// When > 0, diagnostics are suppressed (trial unification).
    pub(crate) quiet: u32,
    pub(crate) fn_id: Option<FnId>,
    pub(crate) unit: usize,
    /// Every call of a user function: (caller, callee, call span), for the
    /// requirement "No recursion" (`E2016`).
    pub(crate) calls: Vec<(FnId, FnId, Span)>,
}

/// A name in scope inside a function body.
#[derive(Clone, Debug)]
pub(crate) enum Binding {
    Val(Ty),
    /// A module value: a `use` member (its parameters are part of the
    /// enclosing θ, so calls pass no θ) or a module-typed parameter (calls
    /// pass θ explicitly).
    Module { r: ModRef, explicit_theta: bool },
    #[allow(dead_code)]
    Const(u64),
}

#[derive(Default)]
pub(crate) struct Env {
    scopes: Vec<Vec<(String, Binding)>>,
}

impl Env {
    pub(crate) fn push(&mut self) {
        self.scopes.push(vec![]);
    }
    pub(crate) fn pop(&mut self) {
        self.scopes.pop();
    }
    pub(crate) fn bind(&mut self, name: impl Into<String>, b: Binding) {
        if self.scopes.is_empty() {
            self.push();
        }
        self.scopes.last_mut().unwrap().push((name.into(), b));
    }
    pub(crate) fn lookup(&self, name: &str) -> Option<&Binding> {
        self.scopes.iter().rev().flat_map(|s| s.iter().rev()).find(|(n, _)| n == name).map(|(_, b)| b)
    }
}

impl<'p> Checker<'p> {
    pub(crate) fn error(&mut self, d: Diagnostic) {
        if self.quiet == 0 {
            self.diags.push(d);
        }
    }

    pub(crate) fn fresh_dim(&mut self) -> DimVar {
        let v = DimVar(self.next_dim);
        self.next_dim += 1;
        self.infer_dims.insert(v);
        v
    }

    pub(crate) fn fresh_prefix(&mut self) -> PrefixVar {
        let p = PrefixVar(self.next_prefix);
        self.next_prefix += 1;
        self.flexible.insert(p);
        p
    }

    pub(crate) fn fresh_ty(&mut self) -> Ty {
        let v = TyVar(self.next_tyvar);
        self.next_tyvar += 1;
        Ty::Infer(v)
    }

    /// Render a type in source terms after applying the current substitution.
    pub(crate) fn show(&self, t: &Ty) -> String {
        let z = self.zonk(t);
        z.display(&self.prog.names())
    }

    pub(crate) fn show_dim(&self, d: &DimExpr) -> String {
        self.resolve_dim(d).display(&|v| self.prog.dim_name(v))
    }

    pub(crate) fn record(&mut self, span: Span, ty: &Ty) {
        self.typed.ty_of.insert(span, ty.clone());
    }

    fn check_fn(&mut self, id: FnId, f: &crate::elab::FnSig) {
        self.fn_id = Some(id);
        self.unit = f.unit;
        let mut env = Env::default();
        env.push();
        // Module members see the module's parameters and sub-modules.
        if let Some(mid) = f.owner {
            let m = self.prog.module(mid);
            for p in &m.params {
                env.bind(p.name.clone(), Binding::Val(p.ty.clone()));
            }
            for u in &m.uses {
                env.bind(u.name.clone(), Binding::Module { r: u.module.clone(), explicit_theta: false });
            }
        }
        env.push();
        for (i, p) in f.params.iter().enumerate() {
            let b = match &p.ty {
                Some(Ty::Module(r)) => Binding::Module { r: r.clone(), explicit_theta: true },
                Some(t) => Binding::Val(t.clone()),
                None => Binding::Val(self.fresh_ty()),
            };
            if let Binding::Val(t) = &b {
                self.typed.param_ty.insert((id, i), t.clone());
            }
            if p.axis {
                // `axis i` parameters: not used by the examples; typed as an axis name.
                continue;
            }
            env.bind(p.name.clone(), b);
        }
        let Some(body_ty) = self.expr(&f.body, &mut env) else { return };
        if let Some(ret) = &f.ret {
            let ret = ret.clone();
            if !self.unify(&ret, &body_ty, f.body.span()) {
                let (e, a) = (self.show(&ret), self.show(&body_ty));
                self.error(
                    Diagnostic::error("E2011", format!("`{}` returns `{a}` but is declared `-> {e}`", f.name))
                        .label(f.body.span(), format!("this body has type `{a}`"))
                        .label(f.span, format!("declared `{e}`")),
                );
            }
        }
        self.typed.body_ty.insert(id, body_ty);
    }

    /// `init = xavier_uniform | zeros | ones | normal(std = f64)`.
    fn check_inits(&mut self, _mid: ModuleId, m: &crate::elab::ModuleDef) {
        for p in &m.params {
            let Some(init) = &p.init else { continue };
            let ok = match init {
                ast::Expr::Var { path } => matches!(path.dotted().as_str(), "xavier_uniform" | "zeros" | "ones"),
                ast::Expr::Call { callee, args, .. } => {
                    matches!(&**callee, ast::Expr::Var { path } if path.is_single("normal"))
                        && args.len() == 1
                        && matches!(&args[0], ast::Arg::Named { name, value } if name.name == "std" && matches!(value, ast::Expr::Lit { lit: ast::Lit::Float(_), .. }))
                }
                _ => false,
            };
            if !ok {
                self.error(
                    Diagnostic::error("E2013", format!("invalid initializer for `{}`", p.name))
                        .label(init.span(), "not a known initializer")
                        .help("initializers: `xavier_uniform`, `zeros`, `ones`, `normal(std = 0.02)`"),
                );
            }
        }
    }

    /// Apply the final substitution to every recorded type.
    fn zonk_tables(&mut self) {
        let keys: Vec<Span> = self.typed.ty_of.keys().copied().collect();
        for k in keys {
            let t = self.zonk(&self.typed.ty_of[&k]);
            self.typed.ty_of.insert(k, t);
        }
        let ids: Vec<FnId> = self.typed.body_ty.keys().copied().collect();
        for id in ids {
            let t = self.zonk(&self.typed.body_ty[&id]);
            self.typed.body_ty.insert(id, t);
        }
        let lets = std::mem::take(&mut self.typed.lets);
        self.typed.lets = lets.into_iter().map(|(f, n, t)| (f, n, self.zonk(&t))).collect();
        let params = std::mem::take(&mut self.typed.param_ty);
        self.typed.param_ty = params.into_iter().map(|(k, t)| (k, self.zonk(&t))).collect();
        let calls = std::mem::take(&mut self.typed.call_subst);
        self.typed.call_subst = calls.into_iter().map(|(k, m)| (k, m.into_iter().map(|(v, e)| (v, self.resolve_dim(&e))).collect())).collect();
        let eins = std::mem::take(&mut self.typed.eins);
        self.typed.eins = eins
            .into_iter()
            .map(|(k, e)| {
                let ops = e.operands.into_iter().map(|(s, axes)| (s, axes.iter().map(|a| self.zonk_axis(a)).collect())).collect();
                (k, EinInfo { operands: ops, output: e.output.iter().map(|a| self.zonk_axis(a)).collect(), contracted: e.contracted })
            })
            .collect();
    }

    pub(crate) fn zonk_axis(&self, a: &Axis) -> Axis {
        Axis { name: a.name.clone(), len: self.resolve_dim(&a.len) }
    }

    /// The instance behind a module reference (through `stack`), for member lookup.
    pub(crate) fn inst_of(&self, r: &ModRef) -> Option<(ModuleId, Vec<DimExpr>)> {
        match r {
            ModRef::Inst { def, args } => Some((*def, args.clone())),
            ModRef::Stack { inner, .. } => self.inst_of(inner),
            ModRef::Compose(_) => None,
        }
    }

}

/// `presk check --summary`: every function with its inferred body type and
/// the types of its `let` bindings.
pub fn summary(prog: &ElabProgram, typed: &TypedProgram) -> String {
    {
        let n = prog.names();
        let mut out = String::new();
        for (i, f) in prog.fns.iter().enumerate() {
            let id = FnId(i as u32);
            let owner = f.owner.map(|m| format!("{}.", prog.module(m).name)).unwrap_or_default();
            let unit = &prog.units[f.unit].path;
            let body = typed.body_ty.get(&id).map(|t| t.display(&n)).unwrap_or_else(|| "<error>".into());
            out.push_str(&format!("{} {}{} : {}\n", unit.file_name().and_then(|s| s.to_str()).unwrap_or(""), owner, f.name, body));
            for (fid, name, ty) in &typed.lets {
                if *fid == id {
                    out.push_str(&format!("    let {name} : {}\n", ty.display(&n)));
                }
            }
        }
        out
    }
}
