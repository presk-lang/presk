//! The e-class analysis: every class carries the PSK-Core type of its
//! value (all the nodes of a class denote one value, hence one type) and,
//! for scalar constants, whether it is zero or one. Types come from the ANF
//! `let`s for the original nodes and are computed for the nodes the rules
//! create; a rule only fires when the types it needs are known, so the
//! rewritten program is typed like the original.

use std::rc::Rc;

use egg::{Analysis, DidMerge, EGraph, Id};
use presk_types::elab::ModuleDef;
use presk_types::ty::{ModRef, Shape, Ty};

use crate::lang::{Node, Table};

/// A scalar constant the rules care about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Konst {
    Zero,
    One,
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Data {
    pub ty: Option<Ty>,
    pub konst: Option<Konst>,
}

/// The module definitions of the program (for the types of record fields);
/// `egg` requires the analysis to own its data, so they are shared by `Rc`.
pub type Modules = Rc<Vec<ModuleDef>>;

pub struct Ana {
    pub modules: Modules,
    pub table: Table,
}

pub type Graph = EGraph<Node, Ana>;

pub fn ty_of(egraph: &Graph, id: Id) -> Option<&Ty> {
    egraph[id].data.ty.as_ref()
}

/// `ℝ[]`: a closed shape with no axis.
pub fn is_real_scalar(ty: &Ty) -> bool {
    matches!(ty, Ty::Real(Shape { prefix: None, axes }) if axes.is_empty())
}

/// A value the elementwise operations act on: a real tensor or a parameter tree.
pub fn is_tensorish(ty: &Ty) -> bool {
    matches!(ty, Ty::Real(_) | Ty::Params(_))
}

/// The type of `a ⊕ b` when it needs no broadcasting decision: equal types,
/// or a real scalar against a tensor or tree.
pub fn elementwise_ty(a: &Ty, b: &Ty) -> Option<Ty> {
    if !is_tensorish(a) || !is_tensorish(b) {
        return None;
    }
    if a == b {
        Some(a.clone())
    } else if is_real_scalar(a) {
        Some(b.clone())
    } else if is_real_scalar(b) {
        Some(a.clone())
    } else {
        None
    }
}

/// The type of the field `name` of a value of type `ty`.
pub fn field_ty(modules: &[ModuleDef], ty: &Ty, name: &str) -> Option<Ty> {
    let Ty::Params(ModRef::Inst { def, args }) = ty else { return None };
    let def = &modules[def.0 as usize];
    let map = def.typarams.iter().copied().zip(args.iter().cloned()).collect();
    if let Some(p) = def.params.iter().find(|p| p.name == name) {
        return Some(p.ty.subst(&map));
    }
    def.uses.iter().find(|u| u.name == name).map(|u| Ty::Params(u.module.subst(&map)))
}

/// The field names of a parameter tree of type `ty`, in declaration order.
pub fn field_names(modules: &[ModuleDef], ty: &Ty) -> Option<Vec<String>> {
    let Ty::Params(ModRef::Inst { def, .. }) = ty else { return None };
    let def = &modules[def.0 as usize];
    Some(def.params.iter().map(|p| p.name.clone()).chain(def.uses.iter().map(|u| u.name.clone())).collect())
}

impl Analysis<Node> for Ana {
    type Data = Data;

    fn make(egraph: &mut Graph, enode: &Node, _id: Id) -> Data {
        let t = |id: &Id| egraph[*id].data.ty.clone();
        let k = |id: &Id| egraph[*id].data.konst;
        let (ty, konst) = match enode {
            Node::Nat(n) => (Some(Ty::Nat), match n { 0 => Some(Konst::Zero), 1 => Some(Konst::One), _ => None }),
            Node::Add([a, b]) | Node::Sub([a, b]) | Node::Mul([a, b]) | Node::Div([a, b]) => (t(a).zip(t(b)).and_then(|(x, y)| elementwise_ty(&x, &y)), None),
            Node::Neg(x) => (t(x), k(x).filter(|c| *c == Konst::Zero)),
            Node::Exp(x) | Node::Log(x) | Node::Sqrt(x) | Node::ZerosLike(x) => (t(x), None),
            Node::Pow([_, x]) => (t(x), None),
            Node::Real(x) => (Some(Ty::scalar_real()), k(x)),
            Node::SumTo([_, t_]) | Node::BroadcastLike([_, t_]) => (t(t_), None),
            Node::SumAll(_) | Node::MeanAll(_) => (Some(Ty::scalar_real()), None),
            Node::Field([n, x]) => {
                let name = egraph[*n].nodes.iter().find_map(|m| if let Node::Sym(s) = m { Some(s.as_str()) } else { None });
                (name.and_then(|name| t(x).and_then(|ty| field_ty(&egraph.analysis.modules, &ty, name))), None)
            }
            Node::SetField([_, r, _]) => (t(r), None),
            Node::Proj([i, x]) => {
                let index = egraph[*i].nodes.iter().find_map(|m| if let Node::Nat(k) = m { Some(*k as usize) } else { None });
                let ty = match (index, t(x)) {
                    (Some(i), Some(Ty::Tuple(ts))) => ts.get(i).cloned(),
                    _ => None,
                };
                (ty, None)
            }
            Node::Tuple(xs) => (xs.iter().map(t).collect::<Option<Vec<_>>>().map(Ty::Tuple), None),
            // Records, parameters and opaque nodes are typed by whoever adds them.
            Node::Record(_) | Node::Param(_) | Node::Names(_) | Node::Sym(_) | Node::Other(..) => (None, None),
        };
        Data { ty, konst }
    }

    fn merge(&mut self, a: &mut Data, b: Data) -> DidMerge {
        let mut changed_a = false;
        let mut changed_b = false;
        if a.ty.is_none() && b.ty.is_some() {
            a.ty = b.ty.clone();
            changed_a = true;
        } else if a.ty != b.ty {
            changed_b = true;
        }
        if a.konst.is_none() && b.konst.is_some() {
            a.konst = b.konst;
            changed_a = true;
        } else if a.konst != b.konst {
            changed_b = true;
        }
        DidMerge(changed_a, changed_b)
    }
}
