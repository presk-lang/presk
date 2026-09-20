//! Work and traffic of a PSK-Core node from its types — the `W` and `Q` of
//! the cost-model document (§1.1), symbolic in the dimensions and evaluated
//! here with the dimensions the run binds (by name, as the interpreter binds
//! them) and a default for the rest. Every tensor primitive is one kernel of
//! PSK-Kernel v0, so the roofline `λ + max(W/π, Q/β)` of the profile applies
//! node by node; structural nodes (fields, tuples, records, calls) launch no
//! kernel and cost nothing but a tie-breaking epsilon.

use std::collections::BTreeMap;

use egg::{Id, Language};
use presk_core::PrimOp;
use presk_types::dim::{DimExpr, DimVar};
use presk_types::elab::ModuleDef;
use presk_types::ty::{ModRef, Shape, Ty};

use crate::analysis::Graph;
use crate::lang::{Node, Opaque};

/// Dimensions bound by name for the cost, and the default of the unbound ones.
#[derive(Clone, Debug)]
pub struct DimGuess {
    pub known: BTreeMap<DimVar, u64>,
    pub default: u64,
}

impl DimGuess {
    pub fn eval(&self, e: &DimExpr) -> f64 {
        let map: BTreeMap<DimVar, _> = e.vars().map(|v| (v, presk_types::dim::rat(*self.known.get(&v).unwrap_or(&self.default) as i64))).collect();
        match e.eval(&map) {
            Some(r) => num_traits::ToPrimitive::to_f64(&r).unwrap_or(0.0).max(0.0),
            None => self.default as f64,
        }
    }

    fn shape_elems(&self, s: &Shape) -> f64 {
        let mut n = s.axes.iter().map(|a| self.eval(&a.len)).product::<f64>();
        if s.prefix.is_some() {
            n *= self.default as f64;
        }
        n
    }

    /// Elements of a value of type `ty` (tensors and trees; 0 for the rest).
    pub fn elems(&self, modules: &[ModuleDef], ty: &Ty) -> f64 {
        match ty {
            Ty::Real(s) | Ty::Idx { shape: s, .. } => self.shape_elems(s),
            Ty::Tuple(ts) => ts.iter().map(|t| self.elems(modules, t)).sum(),
            Ty::Params(m) => self.params_elems(modules, m, &BTreeMap::new()),
            _ => 0.0,
        }
    }

    fn params_elems(&self, modules: &[ModuleDef], m: &ModRef, outer: &BTreeMap<DimVar, DimExpr>) -> f64 {
        match m {
            ModRef::Inst { def, args } => {
                let def = &modules[def.0 as usize];
                let map: BTreeMap<DimVar, DimExpr> = def.typarams.iter().copied().zip(args.iter().map(|a| a.subst(outer))).collect();
                def.params.iter().map(|p| self.elems(modules, &p.ty.subst(&map))).sum::<f64>() + def.uses.iter().map(|u| self.params_elems(modules, &u.module, &map)).sum::<f64>()
            }
            ModRef::Stack { n, inner } => self.eval(&n.subst(outer)) * self.params_elems(modules, inner, outer),
            ModRef::Compose(parts) => parts.iter().map(|p| self.params_elems(modules, p, outer)).sum(),
        }
    }

    /// The length of the last axis of a tensor type (contraction depth of `dot`).
    fn last_axis(&self, ty: &Ty) -> f64 {
        match ty.shape().and_then(|s| s.axes.last()) {
            Some(a) => self.eval(&a.len),
            None => 1.0,
        }
    }
}

/// Weighted operations per element for the transcendental primitives.
const TRANSCENDENTAL: f64 = 8.0;

/// `(work, bytes)` of one node: `bytes` counts every tensor operand and the
/// result once, in `real_bytes` per element; `work` is the weighted operation
/// count of the kernel the node launches. `(0, 0)` for structural nodes.
pub fn work_bytes(egraph: &Graph, node: &Node, guess: &DimGuess, real_bytes: f64) -> (f64, f64) {
    let modules = &egraph.analysis.modules;
    let ty_of = |id: Id| egraph[id].data.ty.clone();
    let elems = |id: Id| ty_of(id).map_or(0.0, |t| guess.elems(modules, &t));
    let out = egraph.lookup(node.clone()).and_then(ty_of);
    let n_out = out.as_ref().map_or(0.0, |t| guess.elems(modules, t));
    let in_bytes = |ids: &[Id]| ids.iter().map(|i| elems(*i)).sum::<f64>() * real_bytes;
    let tensor_children: Vec<Id> = node.children().iter().copied().filter(|c| egraph[*c].data.ty.as_ref().is_some_and(|t| matches!(t, Ty::Real(_) | Ty::Idx { .. } | Ty::Params(_) | Ty::Tuple(_)))).collect();
    let q = || in_bytes(&tensor_children) + n_out * real_bytes;
    match node {
        Node::Nat(_) | Node::Param(_) | Node::Names(_) | Node::Sym(_) => (0.0, 0.0),
        Node::Field(_) | Node::SetField(_) | Node::Proj(_) | Node::Tuple(_) | Node::Record(_) => (0.0, 0.0),
        Node::Add(_) | Node::Sub(_) | Node::Mul(_) | Node::Div(_) | Node::Neg(_) | Node::Real(_) | Node::Pow(_) => (n_out, q()),
        Node::Exp(_) | Node::Log(_) | Node::Sqrt(_) => (TRANSCENDENTAL * n_out, q()),
        Node::ZerosLike(_) => (0.0, n_out * real_bytes),
        Node::SumTo([x, _]) | Node::BroadcastLike([x, _]) => (elems(*x).max(n_out), (elems(*x) + n_out) * real_bytes),
        Node::SumAll(x) | Node::MeanAll(x) => (elems(*x), elems(*x) * real_bytes),
        Node::Other(k, ch) => match egraph.analysis.table.get_op(*k) {
            Opaque::Prim(op) => prim_work_bytes(op, ch, &elems, &ty_of, guess, n_out, q()),
            Opaque::Lit(_) | Opaque::Call { .. } | Opaque::Closure(_) | Opaque::Apply | Opaque::Fold { .. } => (0.0, 0.0),
        },
    }
}

fn prim_work_bytes(op: &PrimOp, ch: &[Id], elems: &dyn Fn(Id) -> f64, ty_of: &dyn Fn(Id) -> Option<Ty>, guess: &DimGuess, n_out: f64, q: f64) -> (f64, f64) {
    use PrimOp::*;
    let in0 = ch.first().map_or(0.0, |c| elems(*c));
    let w = match op {
        Add | Sub | Mul | Div | Neg | Real | Pi | Relu | Abs | Sign | Heaviside | Pow(_) => n_out,
        Exp | Log | Tanh | Sqrt | Sigmoid | Sin | Cos => TRANSCENDENTAL * n_out,
        Gelu | GeluDeriv => (TRANSCENDENTAL + 4.0) * n_out,
        // 2·(elements of the result)·(contraction depth).
        Dot | Matmul => 2.0 * n_out * ch.first().and_then(|c| ty_of(*c)).map_or(1.0, |t| guess.last_axis(&t)),
        DotVjpL | DotVjpR => 2.0 * n_out * ch.get(1).and_then(|c| ty_of(*c)).map_or(1.0, |t| guess.last_axis(&t)),
        Ein { operands, output } => {
            // 2 × the product of every distinct axis (output and contracted).
            let mut lens: BTreeMap<&str, f64> = BTreeMap::new();
            for (i, names) in operands.iter().enumerate() {
                if let Some(Ty::Real(s)) = ch.get(i).and_then(|c| ty_of(*c)) {
                    for (name, axis) in names.iter().zip(s.axes.iter().rev().take(names.len()).collect::<Vec<_>>().into_iter().rev()) {
                        lens.entry(name.as_str()).or_insert_with(|| guess.eval(&axis.len));
                    }
                }
            }
            let _ = output;
            2.0 * lens.values().product::<f64>().max(n_out)
        }
        SumAll | MeanAll | Sum { .. } | Mean { .. } | SumTo | SumSquares => in0.max(n_out),
        Softmax { .. } => (TRANSCENDENTAL + 3.0) * n_out,
        CrossEntropy | CrossEntropyVjp => (TRANSCENDENTAL + 3.0) * in0,
        Embed | OneHot | EmbedVjp | MeanVjp { .. } | BroadcastLike | ScatterIndex { .. } | ScatterParams { .. } | SetSlice => n_out,
        Argmax | Sample => in0,
        Zeros { .. } | Ones { .. } | ZerosLike | Normal { .. } | XavierUniform { .. } | DropoutMask | Positions { .. } | CausalMask { .. } => n_out,
        MergeAxes { .. } | SplitAxes { .. } | ReshapeLike | IndexConst { .. } | SplitHeads { .. } | MergeHeads { .. } | SliceParams { .. } | StackParams { .. } | SplitKey { .. } | SetField { .. } | SetElem { .. } => 0.0,
    };
    (w, q)
}
