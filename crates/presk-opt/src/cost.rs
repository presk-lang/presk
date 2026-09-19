//! Extraction costs. Without a profile, *operation counts*: every node costs
//! a weight by category (structural nodes almost nothing, elementwise
//! arithmetic one, transcendental functions two, contractions and loops
//! eight) plus the cost of its children. With a profile (`presk bench
//! --calibrate`), the fitted roofline of `profile`: predicted seconds per
//! node from its work and traffic (`model`), summed over the tree.

use egg::{CostFunction, Id, Language};
use presk_core::PrimOp;

use crate::analysis::Graph;
use crate::lang::{Node, Opaque};
use crate::model::{DimGuess, work_bytes};
use crate::profile::Profile;

/// Structural: field access, tuples, records, projections, closures.
pub const STRUCTURAL: f64 = 0.01;
pub const ELEMENTWISE: f64 = 1.0;
/// A power reads one operand where a product reads two: `x · x` is written `x²`.
pub const POWER: f64 = 0.9;
pub const TRANSCENDENTAL: f64 = 2.0;
pub const REDUCTION: f64 = 1.0;
pub const CONTRACTION: f64 = 8.0;
pub const CALL: f64 = 1.0;

/// The seconds a structural node costs under a profile: enough to prefer
/// fewer of them, far below any kernel.
pub const STRUCTURAL_SECONDS: f64 = 1e-9;

pub fn prim_weight(op: &PrimOp) -> f64 {
    use PrimOp::*;
    match op {
        Add | Sub | Mul | Div | Neg | Real | Pi | Relu | Abs | Sign | Heaviside => ELEMENTWISE,
        Pow(_) => POWER,
        Exp | Log | Tanh | Sqrt | Gelu | Sigmoid | Sin | Cos | GeluDeriv => TRANSCENDENTAL,
        Dot | Matmul | Ein { .. } | DotVjpL | DotVjpR => CONTRACTION,
        SumAll | MeanAll | Sum { .. } | Mean { .. } | Softmax { .. } | CrossEntropy | CrossEntropyVjp | Argmax | Sample | Embed | EmbedVjp | SumTo | BroadcastLike | MeanVjp { .. } | SumSquares | ZerosLike | Zeros { .. } | Ones { .. } | Normal { .. } | XavierUniform { .. } | Positions { .. } | CausalMask { .. } | ScatterIndex { .. } | ScatterParams { .. } => REDUCTION,
        IndexConst { .. } | SplitHeads { .. } | MergeHeads { .. } | SplitKey { .. } | SliceParams { .. } | StackParams { .. } | SetField { .. } | SetElem { .. } | SetSlice => STRUCTURAL,
    }
}

/// The operation-count weight of a kernel from its name (`PrimOp::name`),
/// for the correlation `presk bench --calibrate` reports.
pub fn weight_of_name(name: &str) -> f64 {
    let head = name.split_whitespace().next().unwrap_or("");
    let head = head.split('[').next().unwrap_or(head);
    match head {
        "add" | "sub" | "mul" | "div" | "neg" | "real" | "pi" | "relu" | "abs" | "sign" | "heaviside" => ELEMENTWISE,
        "pow" => POWER,
        "exp" | "log" | "tanh" | "sqrt" | "gelu" | "gelu'" | "sigmoid" | "sin" | "cos" => TRANSCENDENTAL,
        "dot" | "matmul" | "ein" | "dot_vjp_l" | "dot_vjp_r" => CONTRACTION,
        "index" | "split_heads" | "merge_heads" | "split_key" | "slice_params" | "stack_params" | "set_field" | "set_elem" | "set_slice" => STRUCTURAL,
        _ => REDUCTION,
    }
}

/// Which cost decides the extraction.
pub enum Model<'p> {
    Counts,
    Roofline { profile: &'p Profile, guess: &'p DimGuess },
}

pub struct Cost<'g> {
    pub egraph: &'g Graph,
    pub model: Model<'g>,
}

impl Cost<'_> {
    fn counts(&self, enode: &Node) -> f64 {
        match enode {
            Node::Nat(_) | Node::Param(_) | Node::Names(_) | Node::Sym(_) => 0.0,
            Node::Add(_) | Node::Sub(_) | Node::Mul(_) | Node::Div(_) | Node::Neg(_) | Node::Real(_) => ELEMENTWISE,
            Node::Pow(_) => POWER,
            Node::Exp(_) | Node::Log(_) | Node::Sqrt(_) => TRANSCENDENTAL,
            Node::ZerosLike(_) | Node::SumTo(_) | Node::BroadcastLike(_) | Node::SumAll(_) | Node::MeanAll(_) => REDUCTION,
            Node::Field(_) | Node::SetField(_) | Node::Proj(_) | Node::Tuple(_) | Node::Record(_) => STRUCTURAL,
            Node::Other(k, _) => match self.egraph.analysis.table.get_op(*k) {
                Opaque::Lit(_) => 0.0,
                Opaque::Prim(op) => prim_weight(op),
                Opaque::Call { .. } | Opaque::Apply => CALL,
                Opaque::Closure(_) => STRUCTURAL,
                Opaque::Fold { .. } => CONTRACTION,
            },
        }
    }

    fn seconds(&self, enode: &Node, profile: &Profile, guess: &DimGuess) -> f64 {
        match enode {
            Node::Nat(_) | Node::Param(_) | Node::Names(_) | Node::Sym(_) => 0.0,
            _ => {
                let (w, q) = work_bytes(self.egraph, enode, guess, profile.real_bytes());
                if w == 0.0 && q == 0.0 { STRUCTURAL_SECONDS } else { profile.predict(w, q) }
            }
        }
    }
}

impl CostFunction<Node> for Cost<'_> {
    type Cost = f64;

    fn cost<C>(&mut self, enode: &Node, mut costs: C) -> f64
    where
        C: FnMut(Id) -> f64,
    {
        let own = match &self.model {
            Model::Counts => self.counts(enode),
            Model::Roofline { profile, guess } => self.seconds(enode, profile, guess),
        };
        enode.children().iter().fold(own, |acc, id| acc + costs(*id))
    }
}
