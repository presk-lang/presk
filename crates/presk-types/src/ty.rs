//! Shape-ready types: the result of elaborating the syntactic types.
//!
//! Dimensions are [`DimExpr`]s; `...` is a leading prefix of unknown rank
//! that shape typing instantiates per call (step 3).

use std::collections::BTreeMap;
use std::fmt;

use crate::dim::{DimExpr, DimVar};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ModuleId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct InstId(pub u32);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Axis {
    pub name: Option<String>,
    pub len: DimExpr,
}

/// The `...` of a signature: an unknown, fixed sequence of leading axes.
/// One variable per function signature (every `...` of the signature is the
/// same prefix); the checker instantiates it per call.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct PrefixVar(pub u32);

/// A type inference variable (untyped `fn` parameters, lambda parameters).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct TyVar(pub u32);

#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Shape {
    /// `...`: an open prefix of leading axes.
    pub prefix: Option<PrefixVar>,
    pub axes: Vec<Axis>,
}

/// A reference to a module: an instance, a stack of instances, or a
/// composition (spec-only `model` declarations).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ModRef {
    /// A module applied to dimension arguments (structural: substitution
    /// reaches the arguments).
    Inst { def: ModuleId, args: Vec<DimExpr> },
    Stack { n: DimExpr, inner: Box<ModRef> },
    Compose(Vec<ModRef>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ty {
    Real(Shape),
    Idx { bound: DimExpr, shape: Shape },
    Nat,
    Int,
    Rat,
    F64,
    Bool,
    Key,
    Tuple(Vec<Ty>),
    /// `params M`: the parameter tree of a module.
    Params(ModRef),
    /// A module used as a value (compile-time argument).
    Module(ModRef),
    /// A function value (`grad(f)`, a lambda).
    Fn { params: Vec<Ty>, ret: Box<Ty> },
    /// Not yet known (bound by unification).
    Infer(TyVar),
}

impl Ty {
    pub fn scalar_real() -> Ty {
        Ty::Real(Shape::default())
    }

    pub fn subst(&self, map: &BTreeMap<DimVar, DimExpr>) -> Ty {
        match self {
            Ty::Real(s) => Ty::Real(s.subst(map)),
            Ty::Idx { bound, shape } => Ty::Idx { bound: bound.subst(map), shape: shape.subst(map) },
            Ty::Tuple(ts) => Ty::Tuple(ts.iter().map(|t| t.subst(map)).collect()),
            Ty::Params(m) => Ty::Params(m.subst(map)),
            Ty::Module(m) => Ty::Module(m.subst(map)),
            Ty::Fn { params, ret } => Ty::Fn { params: params.iter().map(|t| t.subst(map)).collect(), ret: Box::new(ret.subst(map)) },
            t => t.clone(),
        }
    }

    pub fn is_tensor(&self) -> bool {
        matches!(self, Ty::Real(_) | Ty::Idx { .. })
    }

    pub fn shape(&self) -> Option<&Shape> {
        match self {
            Ty::Real(s) | Ty::Idx { shape: s, .. } => Some(s),
            _ => None,
        }
    }
}

impl Shape {
    pub fn subst(&self, map: &BTreeMap<DimVar, DimExpr>) -> Shape {
        Shape { prefix: self.prefix, axes: self.axes.iter().map(|a| Axis { name: a.name.clone(), len: a.len.subst(map) }).collect() }
    }

    pub fn closed(axes: Vec<Axis>) -> Shape {
        Shape { prefix: None, axes }
    }

    pub fn rank(&self) -> Option<usize> {
        if self.prefix.is_some() { None } else { Some(self.axes.len()) }
    }
}

impl ModRef {
    pub fn subst(&self, map: &BTreeMap<DimVar, DimExpr>) -> ModRef {
        match self {
            ModRef::Inst { def, args } => ModRef::Inst { def: *def, args: args.iter().map(|a| a.subst(map)).collect() },
            ModRef::Stack { n, inner } => ModRef::Stack { n: n.subst(map), inner: Box::new(inner.subst(map)) },
            ModRef::Compose(parts) => ModRef::Compose(parts.iter().map(|p| p.subst(map)).collect()),
        }
    }
}

/// The parameter tree of a module instance: what `params M` denotes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParamsTree {
    Leaf(Ty),
    Node(Vec<(String, ParamsTree)>),
    /// `stack{n}(M)`: every leaf gets a leading axis of length `n`.
    Stacked { n: DimExpr, inner: Box<ParamsTree> },
}

impl ParamsTree {
    /// Flatten to `(dotted path, type)` with stacked axes prepended.
    pub fn leaves(&self) -> Vec<(String, Ty)> {
        let mut out = Vec::new();
        self.collect("", &[], &mut out);
        out
    }

    fn collect(&self, prefix: &str, stacks: &[DimExpr], out: &mut Vec<(String, Ty)>) {
        match self {
            ParamsTree::Leaf(ty) => {
                let mut ty = ty.clone();
                if let Ty::Real(shape) = &mut ty {
                    for n in stacks.iter().rev() {
                        shape.axes.insert(0, Axis { name: None, len: n.clone() });
                    }
                }
                out.push((prefix.to_string(), ty));
            }
            ParamsTree::Node(fields) => {
                for (name, sub) in fields {
                    let p = if prefix.is_empty() { name.clone() } else { format!("{prefix}.{name}") };
                    sub.collect(&p, stacks, out);
                }
            }
            ParamsTree::Stacked { n, inner } => {
                let mut s = stacks.to_vec();
                s.push(n.clone());
                inner.collect(prefix, &s, out);
            }
        }
    }
}

/// Rendering in source terms; the closures name dimensions and instances.
pub struct Names<'a> {
    pub dim: Box<dyn Fn(DimVar) -> String + 'a>,
    pub module: Box<dyn Fn(ModuleId) -> String + 'a>,
}

impl Ty {
    pub fn display(&self, n: &Names) -> String {
        match self {
            Ty::Real(s) => format!("ℝ{}", s.display(n)),
            Ty::Idx { bound, shape } => format!("idx[{}]{}", bound.display(&*n.dim), shape.display(n)),
            Ty::Nat => "nat".into(),
            Ty::Int => "int".into(),
            Ty::Rat => "rat".into(),
            Ty::F64 => "f64".into(),
            Ty::Bool => "bool".into(),
            Ty::Key => "key".into(),
            Ty::Tuple(ts) => format!("({})", ts.iter().map(|t| t.display(n)).collect::<Vec<_>>().join(", ")),
            Ty::Params(m) => format!("params {}", m.display(n)),
            Ty::Module(m) => m.display(n),
            Ty::Fn { params, ret } => format!("({}) -> {}", params.iter().map(|t| t.display(n)).collect::<Vec<_>>().join(", "), ret.display(n)),
            Ty::Infer(v) => format!("?{}", v.0),
        }
    }
}

impl Shape {
    pub fn display(&self, n: &Names) -> String {
        let mut parts: Vec<String> = Vec::new();
        if self.prefix.is_some() {
            parts.push("...".into());
        }
        for a in &self.axes {
            let len = a.len.display(&*n.dim);
            parts.push(match &a.name {
                Some(name) => format!("{name}: {len}"),
                None => len,
            });
        }
        format!("[{}]", parts.join(", "))
    }
}

impl ModRef {
    pub fn display(&self, n: &Names) -> String {
        match self {
            ModRef::Inst { def, args } => {
                let name = (n.module)(*def);
                if args.is_empty() { name } else { format!("{name}{{{}}}", args.iter().map(|a| a.display(&*n.dim)).collect::<Vec<_>>().join(", ")) }
            }
            ModRef::Stack { n: count, inner } => format!("stack{{{}}}({})", count.display(&*n.dim), inner.display(n)),
            ModRef::Compose(parts) => parts.iter().map(|p| p.display(n)).collect::<Vec<_>>().join(" ∘ "),
        }
    }
}

impl fmt::Display for Ty {
    /// Debug-style rendering with raw variable ids (`d3`); prefer `display`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display(&Names { dim: Box::new(|v: DimVar| format!("{v:?}")), module: Box::new(|m: ModuleId| format!("{m:?}")) }))
    }
}
