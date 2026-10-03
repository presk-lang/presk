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
    /// A function of the optimizer `name` of a `training run` for `model`
    /// (`adamw.init`, `adamw.lr`, `adamw.step`).
    Optimizer { name: String, model: ModRef },
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
    /// A loop over the leading axis of stacked values (`stack{n}(M)`):
    /// `acc ← body([i,] slice_i(stacked…), acc, extra…)` for `i` in `0..n`
    /// (`n−1..0` when `reverse`); `index` passes `i : nat` first. The result
    /// is the final accumulator, or with `scan` the pair of the final
    /// accumulator and the stack of the accumulators *before* each iteration
    /// (indexed by `i` whatever the direction), which differentiation needs.
    Fold { body: FuncId, dim_args: BTreeMap<DimVar, DimExpr>, n: DimExpr, stacked: Vec<VarId>, init: VarId, extra: Vec<VarId>, index: bool, reverse: bool, scan: bool },
}

impl Rhs {
    /// The variables it reads, in order, once per read.
    pub fn vars(&self) -> Vec<VarId> {
        match self {
            Rhs::Lit(_) => vec![],
            Rhs::Prim { args, .. } | Rhs::Tuple(args) | Rhs::Call { args, .. } | Rhs::Closure { captured: args, .. } => args.clone(),
            Rhs::Proj(x, _) | Rhs::Field(x, _) | Rhs::Grad(x) => vec![*x],
            Rhs::Record(fields) => fields.iter().map(|(_, v)| *v).collect(),
            Rhs::Apply { fun, args } => std::iter::once(*fun).chain(args.iter().copied()).collect(),
            Rhs::Fold { stacked, init, extra, .. } => stacked.iter().chain(std::iter::once(init)).chain(extra).copied().collect(),
        }
    }

    /// The same right-hand side reading `m(v)` wherever it reads `v`.
    pub fn map_vars(&self, m: impl Fn(VarId) -> VarId) -> Rhs {
        let all = |vs: &[VarId]| vs.iter().map(|v| m(*v)).collect::<Vec<_>>();
        match self {
            Rhs::Lit(x) => Rhs::Lit(x.clone()),
            Rhs::Prim { op, args } => Rhs::Prim { op: op.clone(), args: all(args) },
            Rhs::Call { f, dim_args, args } => Rhs::Call { f: *f, dim_args: dim_args.clone(), args: all(args) },
            Rhs::Tuple(vs) => Rhs::Tuple(all(vs)),
            Rhs::Proj(x, i) => Rhs::Proj(m(*x), *i),
            Rhs::Record(fs) => Rhs::Record(fs.iter().map(|(n, v)| (n.clone(), m(*v))).collect()),
            Rhs::Field(x, n) => Rhs::Field(m(*x), n.clone()),
            Rhs::Closure { f, captured } => Rhs::Closure { f: *f, captured: all(captured) },
            Rhs::Grad(x) => Rhs::Grad(m(*x)),
            Rhs::Apply { fun, args } => Rhs::Apply { fun: m(*fun), args: all(args) },
            Rhs::Fold { body, dim_args, n, stacked, init, extra, index, reverse, scan } => Rhs::Fold { body: *body, dim_args: dim_args.clone(), n: n.clone(), stacked: all(stacked), init: m(*init), extra: all(extra), index: *index, reverse: *reverse, scan: *scan },
        }
    }
}

/// The type of `n` stacked values of type `ty` along a new leading axis:
/// `params M` → `params stack{n}(M)`, a tensor gains a leading axis, a tuple
/// is stacked leaf by leaf. A shape with an open prefix `...` is unchanged:
/// the prefix already covers the new leading axis (and the interpreter's
/// retagging must not touch the axes it hides).
pub fn stack_ty(ty: &Ty, n: &DimExpr) -> Ty {
    use presk_types::ty::{Axis, Shape};
    let lead = |s: &Shape| {
        if s.prefix.is_some() {
            return s.clone();
        }
        let mut axes = vec![Axis { name: None, len: n.clone() }];
        axes.extend(s.axes.iter().cloned());
        Shape { prefix: None, axes }
    };
    match ty {
        Ty::Real(s) => Ty::Real(lead(s)),
        Ty::Idx { bound, shape } => Ty::Idx { bound: bound.clone(), shape: lead(shape) },
        Ty::Truth(shape) => Ty::Truth(lead(shape)),
        Ty::Tuple(ts) => Ty::Tuple(ts.iter().map(|t| stack_ty(t, n)).collect()),
        Ty::Params(r) => Ty::Params(ModRef::Stack { n: n.clone(), inner: Box::new(r.clone()) }),
        other => other.clone(),
    }
}

/// The type of one slice of a stacked value (inverse of [`stack_ty`]).
pub fn unstack_ty(ty: &Ty) -> Ty {
    use presk_types::ty::Shape;
    let drop = |s: &Shape| if s.prefix.is_some() { s.clone() } else { Shape { prefix: None, axes: s.axes.iter().skip(1).cloned().collect() } };
    match ty {
        Ty::Real(s) => Ty::Real(drop(s)),
        Ty::Idx { bound, shape } => Ty::Idx { bound: bound.clone(), shape: drop(shape) },
        Ty::Truth(shape) => Ty::Truth(drop(shape)),
        Ty::Tuple(ts) => Ty::Tuple(ts.iter().map(unstack_ty).collect()),
        Ty::Params(ModRef::Stack { inner, .. }) => Ty::Params((**inner).clone()),
        other => other.clone(),
    }
}

/// Primitive operations. Axes are designated by *name*: runtime tensors
/// carry the axis names of their types.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
    /// The constant π, correctly rounded at the evaluator's precision.
    Pi,
    /// Contraction of the last axis of the left operand with the first of the right.
    Dot,
    Matmul,
    /// Einstein summation: the (relabeled) axis names of each operand and of the output.
    Ein { operands: Vec<Vec<String>>, output: Vec<String> },
    SumAll,
    MeanAll,
    Sum { axis: String },
    Mean { axis: String },
    /// `(x)`, or `(x, mask)` with the mask a `bool[S2]` that broadcasts into
    /// `x` (plan 0.10.0 step 7b): along `axis`, over the positions the mask
    /// allows, `0` where it refuses.
    Softmax { axis: String },
    /// `embed(table, indices)`
    Embed,
    /// `one_hot(indices)`: the indicator vectors of an `idx[V]` tensor, along a
    /// new trailing axis of length `V`. `embed(w, x)` denotes `one_hot(x) · w`.
    OneHot,
    /// `cross_entropy(logits, targets)`: per-position losses.
    CrossEntropy,
    Argmax,
    /// `(key, logits)`: a token per position of the logits but the last,
    /// the least index whose cumulative probability exceeds the key's uniform
    /// draw of that position's rank (plan 0.10.0 step 7b; the reference's
    /// definition "Draws"). Uses the `Rand` effect.
    Sample,
    /// `t[axis: index]` (negative indices count from the end).
    IndexConst { axis: String, index: i64 },
    /// `merge(axis a, axis b, x)`: two adjacent axes become one, row-major
    /// (`(i, j) ↦ i·|b| + j`). The new axis is unnamed, as `embed`'s is. A
    /// pure retag: row-major buffers make this metadata-only.
    MergeAxes { a: String, b: String },
    /// `split(axis a, n1: L1, n2: L2, x)`: one axis becomes two, row-major
    /// (`k ↦ (k / L2, k % L2)`), so it is exactly the inverse of `MergeAxes`
    /// and the two halves are named at the split site. A pure retag, like
    /// `MergeAxes`: it reads the same buffer under a different shape.
    SplitAxes { a: String, n1: String, l1: DimExpr, n2: String, l2: DimExpr },
    /// `(x, template)`: `x`'s buffer under the template's axes, which must
    /// hold as many elements. The adjoint of `MergeAxes`, and a retag like it.
    ReshapeLike,
    SplitHeads { h: u64, hd: u64 },
    MergeHeads { h: u64, hd: u64 },
    Positions { axis: String, len: DimExpr },
    /// `bool[axis: len, axis2: len]`, true where the second index is at most
    /// the first (plan 0.10.0 step 7b).
    CausalMask { axis: String, len: DimExpr },
    // initializers (the shape comes from `ty`)
    Zeros { ty: Ty },
    Ones { ty: Ty },
    Normal { ty: Ty, std: String },
    XavierUniform { ty: Ty },
    /// `(rate, key, template)`: the inverted-dropout mask of the template's
    /// shape — `0` with probability `rate`, `1/(1 − rate)` otherwise, drawn
    /// from the key alone. `dropout(rate, key, x)` lowers to `x · mask`, so
    /// differentiation needs no rule of its own: the backward pass recomputes
    /// the mask *from the same key* and therefore gets the same one. Outside
    /// training the mask is all ones, which is the random function's
    /// expectation.
    DropoutMask,
    /// Split a random key into `n` keys (a tuple).
    SplitKey { n: usize },
    /// The `index`-th slice of a stacked parameter tree.
    SliceParams { index: u64 },
    /// Stack `n` parameter trees along a new leading axis.
    StackParams { n: u64 },
    // ----- adjoint primitives (emitted by automatic differentiation) -----
    /// Zeros with the structure and shape of the argument (tensor or tree).
    ZerosLike,
    /// Sum of the squares of every element of a tensor or a tree: `ℝ[]`.
    SumSquares,
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
    /// `(stacked, i : nat, value)`: the stacked value with slice `i` replaced.
    SetSlice,
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
            PrimOp::Pi => "pi".into(),
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
            PrimOp::OneHot => "one_hot".into(),
            PrimOp::CrossEntropy => "cross_entropy".into(),
            PrimOp::Argmax => "argmax".into(),
            PrimOp::Sample => "sample".into(),
            PrimOp::IndexConst { axis, index } => format!("index {axis}: {index}"),
            PrimOp::MergeAxes { a, b } => format!("merge {a}, {b}"),
            PrimOp::SplitAxes { a, n1, n2, .. } => format!("split {a} into {n1}, {n2}"),
            PrimOp::ReshapeLike => "reshape_like".into(),
            PrimOp::SplitHeads { h, hd } => format!("split_heads {h}x{hd}"),
            PrimOp::MergeHeads { h, hd } => format!("merge_heads {h}x{hd}"),
            PrimOp::Positions { axis, .. } => format!("positions {axis}"),
            PrimOp::CausalMask { axis, .. } => format!("causal_mask {axis}"),
            PrimOp::Zeros { .. } => "zeros".into(),
            PrimOp::Ones { .. } => "ones".into(),
            PrimOp::Normal { std, .. } => format!("normal std={std}"),
            PrimOp::XavierUniform { .. } => "xavier_uniform".into(),
            PrimOp::DropoutMask => "dropout_mask".into(),
            PrimOp::SplitKey { n } => format!("split_key {n}"),
            PrimOp::SliceParams { index } => format!("slice_params {index}"),
            PrimOp::StackParams { n } => format!("stack_params {n}"),
            PrimOp::ZerosLike => "zeros_like".into(),
            PrimOp::SumSquares => "sum_squares".into(),
            PrimOp::SetSlice => "set_slice".into(),
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

    /// Whether the primitive is an elementwise map over its operands, and so
    /// can share a loop nest with its neighbours (PSK-Realize's fusion
    /// groups). `real` and `pi` are excluded: they lift a hyperparameter, and
    /// the host produces them without a kernel. `pow n` expands to `n − 1`
    /// products of its argument, so a big exponent would duplicate the fused
    /// subtree; those stay kernels of their own.
    pub fn elementwise(&self) -> bool {
        use PrimOp::*;
        matches!(self, Add | Sub | Mul | Div | Neg | Exp | Log | Tanh | Sqrt | Gelu | Relu | Abs | Sigmoid | Sin | Cos | GeluDeriv | Heaviside | Sign) || matches!(self, Pow(n) if *n <= 2)
    }

    /// Whether the primitive draws randomness.
    pub fn uses_rand(&self) -> bool {
        matches!(self, PrimOp::Sample | PrimOp::Normal { .. } | PrimOp::XavierUniform { .. } | PrimOp::SplitKey { .. } | PrimOp::DropoutMask)
    }
}
