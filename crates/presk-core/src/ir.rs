//! PSK-Core: a first-order functional IR in A-normal form over exact real
//! tensors. This is the seam between the front-end and everything that
//! consumes programs (the reference interpreter today; optimizers and
//! backends later).
//!
//! Functions are *generic in their dimensions*: types mention the rigid
//! dimension variables of the elaborated signature, and every call carries
//! the instantiation of the callee's variables by the caller's dimension
//! expressions (`dim_args`). Module functions take their parameter tree as an
//! explicit first parameter `θ : params M`; `stack{L}(M)` is unrolled.

use std::collections::BTreeMap;

use presk_syntax::Span;
use presk_types::dim::{DimExpr, DimVar};
use presk_types::elab::FnId;
use presk_types::ty::{ModRef, Ty};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct FuncId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct VarId(pub u32);

#[derive(Clone, Debug)]
pub struct Program {
    pub funcs: Vec<Func>,
}

impl Program {
    pub fn func(&self, id: FuncId) -> &Func {
        &self.funcs[id.0 as usize]
    }

    pub fn find(&self, name: &str) -> Option<FuncId> {
        self.funcs.iter().position(|f| f.name == name).map(|i| FuncId(i as u32))
    }
}

/// Where a Core function comes from.
#[derive(Clone, Debug)]
pub enum Origin {
    /// A source function (top-level or module member).
    Source(FnId),
    /// The initializer of a module's parameter tree: `init(key) -> params M`.
    Init(ModRef),
    /// A lambda lifted out of `parent` (closure conversion).
    Lambda { parent: FuncId },
}

#[derive(Clone, Debug)]
pub struct Func {
    /// `Module.fn`, `fn`, `Module.init`, `fn.lambda0`
    pub name: String,
    pub origin: Origin,
    /// Dimension variables the function is generic in (appear in its types).
    pub dims: Vec<DimVar>,
    pub params: Vec<Param>,
    pub ret: Ty,
    pub effects: Vec<String>,
    pub body: Block,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub var: VarId,
    pub name: String,
    pub ty: Ty,
}

#[derive(Clone, Debug)]
pub struct Block {
    pub lets: Vec<Let>,
    pub result: VarId,
}

#[derive(Clone, Debug)]
pub struct Let {
    pub var: VarId,
    /// Source name when the variable comes from a `let` or a parameter.
    pub name: Option<String>,
    pub ty: Ty,
    pub rhs: Rhs,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum Lit {
    Nat(u64),
    /// Decimal text, converted at the evaluator's precision.
    Float(String),
}

#[derive(Clone, Debug)]
pub enum Rhs {
    Lit(Lit),
    Prim { op: PrimOp, args: Vec<VarId> },
    /// Call of a Core function; `dim_args` maps the callee's dimension
    /// variables to expressions in the caller's dimensions.
    Call { f: FuncId, dim_args: BTreeMap<DimVar, DimExpr>, args: Vec<VarId> },
    Tuple(Vec<VarId>),
    Proj(VarId, usize),
    /// A parameter tree: named fields, in declaration order.
    Record(Vec<(String, VarId)>),
    Field(VarId, String),
    /// A function value: `f` partially applied to `captured` (the lambda's
    /// free variables come first in `f`'s parameters).
    Closure { f: FuncId, captured: Vec<VarId> },
    /// The gradient of a function value `(P) -> ℝ[]`, as a function value `(P) -> P`.
    Grad(VarId),
    /// Application of a function value.
    Apply { fun: VarId, args: Vec<VarId> },
}

/// Primitive operations. Axes are designated by *name*: runtime tensors
/// carry the axis names of their types.
#[derive(Clone, Debug)]
pub enum PrimOp {
    // elementwise, with broadcasting
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Pow(u64),
    Exp,
    Log,
    Tanh,
    Sqrt,
    Gelu,
    Relu,
    Abs,
    Sigmoid,
    Sin,
    Cos,
    /// Injection of a hyperparameter (`nat`, `f64`) into the real scalar.
    Real,
    /// Contraction of the last axis of the left operand with the first of the right.
    Dot,
    Matmul,
    /// Einstein summation: the (relabeled) axis names of each operand and of the output.
    Ein { operands: Vec<Vec<String>>, output: Vec<String> },
    SumAll,
    MeanAll,
    Sum { axis: String },
    Mean { axis: String },
    Softmax { axis: String },
    /// `embed(table, indices)`
    Embed,
    /// `cross_entropy(logits, targets)`: per-position losses.
    CrossEntropy,
    Argmax,
    /// `sample(logits)`: uses the `Rand` effect.
    Sample,
    /// `t[axis: index]` (negative indices count from the end).
    IndexConst { axis: String, index: i64 },
    SplitHeads { h: u64, hd: u64 },
    MergeHeads { h: u64, hd: u64 },
    Positions { axis: String, len: DimExpr },
    CausalMask { axis: String, len: DimExpr },
    // initializers (the shape comes from `ty`)
    Zeros { ty: Ty },
    Ones { ty: Ty },
    Normal { ty: Ty, std: String },
    XavierUniform { ty: Ty },
    /// Split a random key into `n` keys (a tuple).
    SplitKey { n: usize },
    /// The `index`-th slice of a stacked parameter tree.
    SliceParams { index: u64 },
    /// Stack `n` parameter trees along a new leading axis.
    StackParams { n: u64 },
    // ----- adjoint primitives (emitted by automatic differentiation) -----
    /// Zeros with the structure and shape of the argument (tensor or tree).
    ZerosLike,
    /// `(record, value)`: the record with one field replaced.
    SetField { name: String },
    /// `(tuple, value)`: the tuple with one element replaced.
    SetElem { index: usize },
    /// `(x, template)`: sum `x` down to the shape of the template (adjoint of broadcasting).
    SumTo,
    /// `(x, template)`: broadcast `x` up to the shape of the template.
    BroadcastLike,
    /// `(ct, template)`: adjoint of `mean` (all axes, or one axis): `ct / n` broadcast.
    MeanVjp { axis: Option<String> },
    /// Derivative of GELU, `Φ(x) + x·φ(x)`.
    GeluDeriv,
    /// Derivative of ReLU: `1` where `x > 0`, else `0`.
    Heaviside,
    Sign,
    /// `(ct, b)`: adjoint of the left operand of `dot`.
    DotVjpL,
    /// `(a, ct)`: adjoint of the right operand of `dot`.
    DotVjpR,
    /// `(table, indices, ct)`: scatter-add of `ct` into a zero table.
    EmbedVjp,
    /// `(logits, targets, ct)`: `ct · (softmax(logits) − onehot(targets))`.
    CrossEntropyVjp,
    /// `(template, ct)`: zeros like the template with the slice `axis: index` set to `ct`.
    ScatterIndex { axis: String, index: i64 },
    /// `(stacked, ct)`: zeros like the stacked tree with slice `index` set to `ct`.
    ScatterParams { index: u64 },
}

impl PrimOp {
    pub fn name(&self) -> String {
        match self {
            PrimOp::Add => "add".into(),
            PrimOp::Sub => "sub".into(),
            PrimOp::Mul => "mul".into(),
            PrimOp::Div => "div".into(),
            PrimOp::Neg => "neg".into(),
            PrimOp::Pow(n) => format!("pow {n}"),
            PrimOp::Exp => "exp".into(),
            PrimOp::Log => "log".into(),
            PrimOp::Tanh => "tanh".into(),
            PrimOp::Sqrt => "sqrt".into(),
            PrimOp::Gelu => "gelu".into(),
            PrimOp::Relu => "relu".into(),
            PrimOp::Abs => "abs".into(),
            PrimOp::Sigmoid => "sigmoid".into(),
            PrimOp::Sin => "sin".into(),
            PrimOp::Cos => "cos".into(),
            PrimOp::Real => "real".into(),
            PrimOp::Dot => "dot".into(),
            PrimOp::Matmul => "matmul".into(),
            PrimOp::Ein { operands, output } => {
                format!("ein[{}]({})", output.join(", "), operands.iter().map(|o| format!("[{}]", o.join(", "))).collect::<Vec<_>>().join(" * "))
            }
            PrimOp::SumAll => "sum".into(),
            PrimOp::MeanAll => "mean".into(),
            PrimOp::Sum { axis } => format!("sum axis {axis}"),
            PrimOp::Mean { axis } => format!("mean axis {axis}"),
            PrimOp::Softmax { axis } => format!("softmax axis {axis}"),
            PrimOp::Embed => "embed".into(),
            PrimOp::CrossEntropy => "cross_entropy".into(),
            PrimOp::Argmax => "argmax".into(),
            PrimOp::Sample => "sample".into(),
            PrimOp::IndexConst { axis, index } => format!("index {axis}: {index}"),
            PrimOp::SplitHeads { h, hd } => format!("split_heads {h}x{hd}"),
            PrimOp::MergeHeads { h, hd } => format!("merge_heads {h}x{hd}"),
            PrimOp::Positions { axis, .. } => format!("positions {axis}"),
            PrimOp::CausalMask { axis, .. } => format!("causal_mask {axis}"),
            PrimOp::Zeros { .. } => "zeros".into(),
            PrimOp::Ones { .. } => "ones".into(),
            PrimOp::Normal { std, .. } => format!("normal std={std}"),
            PrimOp::XavierUniform { .. } => "xavier_uniform".into(),
            PrimOp::SplitKey { n } => format!("split_key {n}"),
            PrimOp::SliceParams { index } => format!("slice_params {index}"),
            PrimOp::StackParams { n } => format!("stack_params {n}"),
            PrimOp::ZerosLike => "zeros_like".into(),
            PrimOp::SetField { name } => format!("set_field {name}"),
            PrimOp::SetElem { index } => format!("set_elem {index}"),
            PrimOp::SumTo => "sum_to".into(),
            PrimOp::BroadcastLike => "broadcast_like".into(),
            PrimOp::MeanVjp { axis: None } => "mean_vjp".into(),
            PrimOp::MeanVjp { axis: Some(a) } => format!("mean_vjp axis {a}"),
            PrimOp::GeluDeriv => "gelu'".into(),
            PrimOp::Heaviside => "heaviside".into(),
            PrimOp::Sign => "sign".into(),
            PrimOp::DotVjpL => "dot_vjp_l".into(),
            PrimOp::DotVjpR => "dot_vjp_r".into(),
            PrimOp::EmbedVjp => "embed_vjp".into(),
            PrimOp::CrossEntropyVjp => "cross_entropy_vjp".into(),
            PrimOp::ScatterIndex { axis, index } => format!("scatter_index {axis}: {index}"),
            PrimOp::ScatterParams { index } => format!("scatter_params {index}"),
        }
    }

    /// Whether the primitive draws randomness.
    pub fn uses_rand(&self) -> bool {
        matches!(self, PrimOp::Sample | PrimOp::Normal { .. } | PrimOp::XavierUniform { .. } | PrimOp::SplitKey { .. })
    }
}
