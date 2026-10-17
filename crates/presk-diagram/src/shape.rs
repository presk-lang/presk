//! A function's types at one of its calls: its dimension variables as
//! expressions in the entry's, and its open prefix `...` bound to the axes
//! of the argument it was given.
//!
//! One PSK-Core function serves every instance of its module, its types in
//! the module's dimensions (`ℝ[..., Out]`). The call writes the dimension
//! arguments (`call Linear.apply{In = 64, Out = 256}(up, x)`), and the
//! prefix is the caller's argument's leading axes, since the checker leaves
//! the dimensions it hides unsolved (`{B = ?17, T = ?18, Dm = 64}`).

use std::collections::BTreeMap;

use presk_types::dim::{DimExpr, DimVar};
use presk_types::elab::ElabProgram;
use presk_types::ty::{Axis, PrefixVar, Shape, Ty};

/// The types of one call of a function.
#[derive(Clone, Debug, Default)]
pub struct Ctx {
    /// Its dimension variables, as expressions in the entry's.
    pub dims: BTreeMap<DimVar, DimExpr>,
    /// Its prefixes, as axes in the entry's dimensions.
    pub prefixes: BTreeMap<PrefixVar, Vec<Axis>>,
}

impl Ctx {
    /// The entry's own: its dimensions are themselves, and its open prefix
    /// the axes `prefix` (`presk run --dim batch=3` for `x : ℝ[..., D]`).
    pub fn entry(func: &presk_core::Func, prefix: &[(Option<String>, u64)]) -> Ctx {
        let axes: Vec<Axis> = prefix.iter().map(|(name, n)| Axis { name: name.clone(), len: DimExpr::nat(*n) }).collect();
        let mut prefixes = BTreeMap::new();
        for p in &func.params {
            prefix_vars(&p.ty, &mut |v| {
                prefixes.entry(v).or_insert_with(|| axes.clone());
            });
        }
        Ctx { dims: func.dims.iter().map(|v| (*v, DimExpr::var(*v))).collect(), prefixes }
    }

    /// A shape in the entry's dimensions, its prefix expanded; `None` when
    /// the prefix is not bound.
    pub fn shape(&self, s: &Shape) -> Option<Shape> {
        let mut axes = match s.prefix {
            Some(p) => self.prefixes.get(&p)?.clone(),
            None => Vec::new(),
        };
        axes.extend(s.axes.iter().map(|a| Axis { name: a.name.clone(), len: a.len.subst(&self.dims) }));
        Some(Shape { prefix: None, axes })
    }

    /// A type in the entry's dimensions. A shape whose prefix is not bound
    /// keeps it.
    pub fn ty(&self, t: &Ty) -> Ty {
        let sh = |s: &Shape| self.shape(s).unwrap_or_else(|| Shape { prefix: s.prefix, axes: s.axes.iter().map(|a| Axis { name: a.name.clone(), len: a.len.subst(&self.dims) }).collect() });
        match t {
            Ty::Real(s) => Ty::Real(sh(s)),
            Ty::Idx { bound, shape } => Ty::Idx { bound: bound.subst(&self.dims), shape: sh(shape) },
            Ty::Truth(s) => Ty::Truth(sh(s)),
            Ty::Tuple(ts) => Ty::Tuple(ts.iter().map(|t| self.ty(t)).collect()),
            Ty::Params(m) => Ty::Params(m.subst(&self.dims)),
            Ty::Module(m) => Ty::Module(m.subst(&self.dims)),
            t => t.clone(),
        }
    }

    /// The types of a call of `callee` with `dim_args`, given the types of
    /// the arguments in this context: the callee's dimensions through the
    /// call's arguments, its prefix from the arguments' leading axes.
    pub fn call(&self, callee: &presk_core::Func, dim_args: &BTreeMap<DimVar, DimExpr>, args: &[Option<Ty>]) -> Ctx {
        let mut c = Ctx { dims: BTreeMap::new(), prefixes: BTreeMap::new() };
        for v in &callee.dims {
            if let Some(e) = dim_args.get(v) {
                c.dims.insert(*v, e.subst(&self.dims));
            }
        }
        for (p, a) in callee.params.iter().zip(args) {
            if let Some(a) = a {
                bind(&p.ty, &self.ty(a), &mut c.prefixes);
            }
        }
        c
    }

    /// A lifted lambda's, applied: it shares its caller's dimensions.
    pub fn apply(&self, callee: &presk_core::Func, args: &[Option<Ty>]) -> Ctx {
        let mut c = Ctx { dims: self.dims.clone(), prefixes: BTreeMap::new() };
        for v in &callee.dims {
            c.dims.entry(*v).or_insert_with(|| DimExpr::var(*v));
        }
        for (p, a) in callee.params.iter().zip(args) {
            if let Some(a) = a {
                bind(&p.ty, &self.ty(a), &mut c.prefixes);
            }
        }
        c
    }
}

/// The prefix variables of a type.
fn prefix_vars(t: &Ty, f: &mut dyn FnMut(PrefixVar)) {
    match t {
        Ty::Real(s) | Ty::Idx { shape: s, .. } | Ty::Truth(s) => {
            if let Some(v) = s.prefix {
                f(v);
            }
        }
        Ty::Tuple(ts) => ts.iter().for_each(|t| prefix_vars(t, f)),
        _ => {}
    }
}

/// Binds the prefix of `param` to the leading axes of `arg`, which has none
/// of its own: the axes `arg` has beyond those `param` names.
fn bind(param: &Ty, arg: &Ty, out: &mut BTreeMap<PrefixVar, Vec<Axis>>) {
    let shapes = match (param, arg) {
        (Ty::Real(p), Ty::Real(a)) | (Ty::Idx { shape: p, .. }, Ty::Idx { shape: a, .. }) | (Ty::Truth(p), Ty::Truth(a)) => Some((p, a)),
        (Ty::Tuple(ps), Ty::Tuple(as_)) => {
            for (p, a) in ps.iter().zip(as_) {
                bind(p, a, out);
            }
            None
        }
        _ => None,
    };
    if let Some((p, a)) = shapes
        && let Some(v) = p.prefix
        && a.prefix.is_none()
        && a.axes.len() >= p.axes.len()
    {
        out.entry(v).or_insert_with(|| a.axes[..a.axes.len() - p.axes.len()].to_vec());
    }
}

/// The entry's dimensions as constant expressions, to evaluate with.
pub fn assignment(dims: &BTreeMap<DimVar, u64>) -> BTreeMap<DimVar, DimExpr> {
    dims.iter().map(|(v, n)| (*v, DimExpr::nat(*n))).collect()
}

/// A length at the entry's dimensions.
pub fn eval(e: &DimExpr, at: &BTreeMap<DimVar, DimExpr>) -> Option<u64> {
    e.subst(at).as_nat()
}

/// The real elements and the other elements (indices, truth values) of a
/// value of type `t` (in the entry's dimensions), at them; `None` when a
/// length is not known.
pub fn elements(t: &Ty, at: &BTreeMap<DimVar, DimExpr>) -> Option<(u64, u64)> {
    let numel = |s: &Shape| -> Option<u64> {
        if s.prefix.is_some() {
            return None;
        }
        s.axes.iter().try_fold(1u64, |n, a| Some(n * eval(&a.len, at)?))
    };
    match t {
        Ty::Real(s) => Some((numel(s)?, 0)),
        Ty::Idx { shape, .. } | Ty::Truth(shape) => Some((0, numel(shape)?)),
        Ty::Tuple(ts) => ts.iter().try_fold((0, 0), |(r, o), t| {
            let (a, b) = elements(t, at)?;
            Some((r + a, o + b))
        }),
        Ty::F64 | Ty::Nat | Ty::Int | Ty::Rat | Ty::Bool | Ty::Key | Ty::Params(_) | Ty::Module(_) | Ty::Fn { .. } | Ty::Infer(_) => Some((0, 0)),
    }
}

/// Whether a value of type `t` is keys only.
pub fn keys_only(t: &Ty) -> bool {
    match t {
        Ty::Key => true,
        Ty::Tuple(ts) => !ts.is_empty() && ts.iter().all(keys_only),
        _ => false,
    }
}

/// A type as the program writes it, in the entry's dimension names.
pub fn display(t: &Ty, prog: &ElabProgram) -> String {
    t.display(&prog.names())
}

/// A type at the entry's dimensions: every length a number.
pub fn display_at(t: &Ty, prog: &ElabProgram, at: &BTreeMap<DimVar, DimExpr>) -> String {
    let at_ty = subst_all(t, at);
    at_ty.display(&prog.names())
}

fn subst_all(t: &Ty, at: &BTreeMap<DimVar, DimExpr>) -> Ty {
    let sh = |s: &Shape| Shape { prefix: s.prefix, axes: s.axes.iter().map(|a| Axis { name: a.name.clone(), len: a.len.subst(at) }).collect() };
    match t {
        Ty::Real(s) => Ty::Real(sh(s)),
        Ty::Idx { bound, shape } => Ty::Idx { bound: bound.subst(at), shape: sh(shape) },
        Ty::Truth(s) => Ty::Truth(sh(s)),
        Ty::Tuple(ts) => Ty::Tuple(ts.iter().map(|t| subst_all(t, at)).collect()),
        Ty::Params(m) => Ty::Params(m.subst(at)),
        t => t.clone(),
    }
}
