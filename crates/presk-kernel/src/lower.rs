//! From a PSK-Core tensor primitive at concrete operand shapes to a
//! [`Kernel`]: the loop nest that the reference interpreter's semantics
//! prescribe (its alignment rules for broadcasting, its contraction axes, its
//! adjoint primitives), with one kernel per primitive instance.

use presk_core::PrimOp;
use presk_realize::{Src, Step};
use presk_rt::Axis;

use presk_rt::layout::{BlockLoop, Form, KBlock};
use presk_rt::{Layout, NestKind, ReductionOrder};
use crate::ir::{BinOp, BufKind, BufSpec, CmpOp, Dtype, Expr, Kernel, Stmt, UnOp};

/// An operand: its axes and whether it holds reals or indices.
#[derive(Clone, Debug)]
pub struct Operand {
    pub axes: Vec<Axis>,
    pub kind: BufKind,
}

/// A lowered primitive: the kernel and the axes of its result.
#[derive(Clone, Debug)]
pub struct Lowered {
    pub kernel: Kernel,
    pub out_axes: Vec<Axis>,
    pub out_kind: BufKind,
    /// For index outputs: the bound.
    pub out_bound: u64,
}

pub type Result<T> = std::result::Result<T, String>;

// ----- alignment (the interpreter's rules) ------------------------------------

/// How the axes of `src` map into `target`: trailing positional alignment when
/// names agree, else alignment by name.
pub fn align(src: &[Axis], target: &[Axis]) -> Option<Vec<usize>> {
    if src.len() > target.len() {
        return None;
    }
    let off = target.len() - src.len();
    let positional = src.iter().enumerate().all(|(i, a)| {
        let t = &target[off + i];
        a.len == t.len && (a.name.is_none() || t.name.is_none() || a.name == t.name)
    });
    if positional {
        return Some((0..src.len()).map(|i| off + i).collect());
    }
    let mut out = Vec::new();
    for a in src {
        let name = a.name.as_ref()?;
        let p = target.iter().position(|t| t.name.as_deref() == Some(name) && t.len == a.len)?;
        out.push(p);
    }
    Some(out)
}

/// The result axes of an elementwise binary operation.
pub fn result_axes(a: &[Axis], b: &[Axis]) -> Option<Vec<Axis>> {
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let map = align(short, long)?;
    let mut out = long.to_vec();
    for (i, p) in map.iter().enumerate() {
        if out[*p].name.is_none() {
            out[*p].name = short[i].name.clone();
        }
    }
    Some(out)
}

// ----- builder ----------------------------------------------------------------

struct B {
    nloops: usize,
    nreal: usize,
    nint: usize,
    /// Vector variables and the lane width this kernel uses (1 = scalar).
    nvec: usize,
    lanes: usize,
}

impl B {
    fn new() -> B {
        B { nloops: 0, nreal: 0, nint: 0, nvec: 0, lanes: 1 }
    }
    fn iv(&mut self) -> usize {
        self.nloops += 1;
        self.nloops - 1
    }
    /// A vector variable of `lanes` reals. The first call fixes the kernel's
    /// lane width; every vector in one kernel shares it.
    fn vv(&mut self, lanes: usize) -> usize {
        self.lanes = lanes;
        self.nvec += 1;
        self.nvec - 1
    }
    fn sv(&mut self) -> usize {
        self.nreal += 1;
        self.nreal - 1
    }
    fn si(&mut self) -> usize {
        self.nint += 1;
        self.nint - 1
    }
    /// Fresh loop variables over `dims`, and the nest wrapping `body`.
    fn loops(&mut self, dims: &[usize]) -> Vec<usize> {
        dims.iter().map(|_| self.iv()).collect()
    }
}

fn nest(ivs: &[usize], dims: &[usize], body: Vec<Stmt>) -> Vec<Stmt> {
    let mut b = body;
    for (v, d) in ivs.iter().zip(dims).rev() {
        b = vec![Stmt::For { var: *v, bound: *d, body: b }];
    }
    b
}

fn spec(axes: &[Axis], kind: BufKind) -> BufSpec {
    BufSpec { dims: axes.iter().map(|a| a.len).collect(), kind, names: axes.iter().map(|a| a.name.clone()).collect() }
}

fn dims(axes: &[Axis]) -> Vec<usize> {
    axes.iter().map(|a| a.len).collect()
}

fn ivs(v: &[usize]) -> Vec<Expr> {
    v.iter().map(|i| Expr::Iv(*i)).collect()
}

fn f(x: f64) -> Expr {
    Expr::F(x)
}

fn bin(op: BinOp, a: Expr, b: Expr) -> Expr {
    Expr::Bin(op, Box::new(a), Box::new(b))
}

fn un(op: UnOp, a: Expr) -> Expr {
    Expr::Un(op, Box::new(a))
}

fn cmp(op: CmpOp, a: Expr, b: Expr) -> Expr {
    Expr::Cmp(op, Box::new(a), Box::new(b))
}

fn select(c: Expr, a: Expr, b: Expr) -> Expr {
    Expr::Select(Box::new(c), Box::new(a), Box::new(b))
}

fn load(buf: usize, index: Vec<Expr>) -> Expr {
    Expr::Load(buf, index)
}

fn kernel(name: impl Into<String>, dtype: Dtype, ins: &[Operand], out_axes: &[Axis], out_kind: BufKind, b: &B, body: Vec<Stmt>) -> Kernel {
    Kernel { name: name.into(), dtype, inputs: ins.iter().map(|o| spec(&o.axes, o.kind)).collect(), output: spec(out_axes, out_kind), nloops: b.nloops, nreal: b.nreal, nint: b.nint, nvec: b.nvec, lanes: b.lanes, body, threads: Default::default() }
}

fn done(kernel: Kernel, out_axes: Vec<Axis>) -> Result<Lowered> {
    Ok(Lowered { kernel, out_axes, out_kind: BufKind::Real, out_bound: 0 })
}

fn done_idx(kernel: Kernel, out_axes: Vec<Axis>, bound: u64) -> Result<Lowered> {
    Ok(Lowered { kernel, out_axes, out_kind: BufKind::Idx, out_bound: bound })
}

/// `x·Φ(x)`, `Φ(x) = (1 + erf(x/√2))/2`.
fn gelu(x: Expr) -> Expr {
    let phi = bin(BinOp::Div, bin(BinOp::Add, f(1.0), un(UnOp::Erf, bin(BinOp::Div, x.clone(), f(std::f64::consts::SQRT_2)))), f(2.0));
    bin(BinOp::Mul, x, phi)
}

/// `Φ(x) + x·φ(x)`, `φ(x) = exp(−x²/2)/√(2π)`.
fn gelu_deriv(x: Expr) -> Expr {
    let phi = bin(BinOp::Div, bin(BinOp::Add, f(1.0), un(UnOp::Erf, bin(BinOp::Div, x.clone(), f(std::f64::consts::SQRT_2)))), f(2.0));
    let dens = bin(BinOp::Div, un(UnOp::Exp, bin(BinOp::Div, un(UnOp::Neg, bin(BinOp::Mul, x.clone(), x.clone())), f(2.0))), f((2.0 * std::f64::consts::PI).sqrt()));
    bin(BinOp::Add, phi, bin(BinOp::Mul, x, dens))
}

/// The element of a binary elementwise primitive. `max` and `min` are a
/// comparison and a selection, the second operand at a tie (the
/// reference's realized `max`), so that every backend gives one bit
/// pattern, a zero's sign included.
fn binary_expr(op: &PrimOp, x: Expr, y: Expr) -> Option<Expr> {
    Some(match op {
        PrimOp::Add => bin(BinOp::Add, x, y),
        PrimOp::Sub => bin(BinOp::Sub, x, y),
        PrimOp::Mul => bin(BinOp::Mul, x, y),
        PrimOp::Div => bin(BinOp::Div, x, y),
        PrimOp::Max => select(cmp(CmpOp::Gt, x.clone(), y.clone()), x, y),
        PrimOp::Min => select(cmp(CmpOp::Lt, x.clone(), y.clone()), x, y),
        _ => return None,
    })
}

fn unary_expr(op: &PrimOp, x: Expr) -> Option<Expr> {
    Some(match op {
        PrimOp::Neg => un(UnOp::Neg, x),
        PrimOp::Exp => un(UnOp::Exp, x),
        PrimOp::Log => un(UnOp::Log, x),
        PrimOp::Tanh => un(UnOp::Tanh, x),
        PrimOp::Sqrt => un(UnOp::Sqrt, x),
        PrimOp::Sin => un(UnOp::Sin, x),
        PrimOp::Cos => un(UnOp::Cos, x),
        PrimOp::Abs => un(UnOp::Abs, x),
        PrimOp::Gelu => gelu(x),
        PrimOp::GeluDeriv => gelu_deriv(x),
        PrimOp::Relu => select(cmp(CmpOp::Lt, x.clone(), f(0.0)), f(0.0), x),
        PrimOp::Heaviside => select(cmp(CmpOp::Gt, x, f(0.0)), f(1.0), f(0.0)),
        PrimOp::Sign => select(cmp(CmpOp::Lt, x.clone(), f(0.0)), f(-1.0), select(cmp(CmpOp::Gt, x, f(0.0)), f(1.0), f(0.0))),
        PrimOp::Sigmoid => bin(BinOp::Div, f(1.0), bin(BinOp::Add, f(1.0), un(UnOp::Exp, un(UnOp::Neg, x)))),
        PrimOp::Pow(n) => {
            if *n == 0 {
                f(1.0)
            } else {
                let mut e = x.clone();
                for _ in 1..*n {
                    e = bin(BinOp::Mul, e, x.clone());
                }
                e
            }
        }
        PrimOp::Real => x,
        _ => return None,
    })
}

// ----- the table -------------------------------------------------------------

/// Lowers a tensor primitive; `None` when the primitive is a host operation
/// (structural, random, or on scalars).
pub fn lower(op: &PrimOp, ins: &[Operand], dtype: Dtype) -> Option<Result<Lowered>> {
    lower_ordered(op, ins, dtype, ReductionOrder::Sequential)
}

/// The same, under the realization's declared reduction order. An
/// `Interleaved` order selects a contraction that reassociates and says so
/// (D31); every other order keeps the exact forms of 0.4.0.
pub fn lower_ordered(op: &PrimOp, ins: &[Operand], dtype: Dtype, order: ReductionOrder) -> Option<Result<Lowered>> {
    lower_planned(op, ins, dtype, order, &Layout { lanes: Some(1), ..Layout::default() })
}

/// The same, under a nest's layout (0.7.0 step 2): a contraction nest
/// ([`Contraction::of`]) is built by [`emit_contraction`] in the layout's
/// form, tile, k-block, loop order and lanes, the decisions it does not
/// state taking the reference's default; every other primitive keeps its
/// lowering, which has no layout decision but the thread split. Every point
/// of the layout space is exact: the kernels differ in where the running
/// sums live and which memory they touch, never in the order of additions.
pub fn lower_planned(op: &PrimOp, ins: &[Operand], dtype: Dtype, order: ReductionOrder, layout: &Layout) -> Option<Result<Lowered>> {
    if let Some(c) = Contraction::of(op, ins, order) {
        let kind = NestKind::Contraction(c.kind(dtype, dtype.arith()));
        return Some(emit_contraction(&c, dtype, ins, &layout.resolve(&kind)));
    }
    Some(match op {
        PrimOp::Add | PrimOp::Sub | PrimOp::Mul | PrimOp::Div | PrimOp::Max | PrimOp::Min => elementwise_binary(op, ins, dtype),
        PrimOp::Neg | PrimOp::Exp | PrimOp::Log | PrimOp::Tanh | PrimOp::Sqrt | PrimOp::Sin | PrimOp::Cos | PrimOp::Abs | PrimOp::Gelu | PrimOp::GeluDeriv | PrimOp::Relu | PrimOp::Heaviside | PrimOp::Sign | PrimOp::Sigmoid | PrimOp::Pow(_) => unary(op, ins, dtype),
        PrimOp::Real if ins.len() == 1 && ins[0].kind == BufKind::Idx => unary(op, ins, dtype),
        PrimOp::Dot | PrimOp::Matmul => dot(ins, dtype, order),
        PrimOp::DotVjpL => dot_vjp_l(ins, dtype),
        PrimOp::DotVjpR => dot_vjp_r(ins, dtype),
        PrimOp::Ein { operands, output } => ein(ins, operands, output, dtype),
        PrimOp::SumAll | PrimOp::MeanAll => reduce_all(op, ins, dtype),
        PrimOp::Sum { axis } | PrimOp::Mean { axis } => reduce_axis(op, axis, ins, dtype),
        PrimOp::MeanVjp { axis } => mean_vjp(axis.as_deref(), ins, dtype),
        PrimOp::Softmax { axis } if ins.len() == 2 => masked_softmax(axis, ins, dtype, layout.passes == Some(2)),
        PrimOp::Softmax { axis } if layout.passes == Some(2) => softmax_two_pass(axis, ins, dtype),
        PrimOp::Softmax { axis } => softmax(axis, ins, dtype),
        PrimOp::Embed => embed(ins, dtype),
        PrimOp::EmbedVjp => embed_vjp(ins, dtype),
        PrimOp::CrossEntropy => cross_entropy(ins, dtype),
        PrimOp::CrossEntropyVjp => cross_entropy_vjp(ins, dtype),
        PrimOp::Argmax => argmax(ins, dtype),
        PrimOp::IndexConst { axis, index } => index_const(axis, *index, ins, dtype),
        PrimOp::ScatterIndex { axis, index } => scatter_index(axis, *index, ins, dtype),
        // A slice's bounds reach a kernel as numbers: the host resolves them
        // under the launch's dimensions, and the key of a compiled kernel
        // carries them.
        PrimOp::Slice { axis, lo, hi } => match (lo.as_nat(), hi.as_nat()) {
            (Some(l), Some(h)) => slice(axis, l as usize, h as usize, ins, dtype),
            _ => Err(format!("slice {axis}: bounds not resolved to numbers")),
        },
        PrimOp::SliceVjp { axis, lo, hi } => match (lo.as_nat(), hi.as_nat()) {
            (Some(l), Some(h)) => slice_vjp(axis, l as usize, h as usize, ins, dtype),
            _ => Err(format!("slice_vjp {axis}: bounds not resolved to numbers")),
        },
        PrimOp::Concat { axis } => concat(axis, ins, dtype),
        PrimOp::SplitHeads { h, hd } => split_heads(*h as usize, *hd as usize, ins, dtype),
        PrimOp::MergeHeads { h, hd } => merge_heads(*h as usize, *hd as usize, ins, dtype),
        PrimOp::SumTo => sum_to(ins, dtype),
        PrimOp::BroadcastLike => broadcast_like(ins, dtype),
        PrimOp::SumSquares if ins.len() == 1 && ins[0].kind == BufKind::Real => sum_squares(ins, dtype),
        _ => return None,
    })
}

fn tensor_in<'a>(ins: &'a [Operand], i: usize, what: &str) -> Result<&'a Operand> {
    ins.get(i).ok_or_else(|| format!("{what}: missing operand {i}"))
}

fn elementwise_binary(op: &PrimOp, ins: &[Operand], dtype: Dtype) -> Result<Lowered> {
    let (a, b) = (tensor_in(ins, 0, "binary")?, tensor_in(ins, 1, "binary")?);
    let out = result_axes(&a.axes, &b.axes).ok_or_else(|| format!("{}: cannot broadcast {} with {}", op.name(), show(&a.axes), show(&b.axes)))?;
    let ma = align(&a.axes, &out).unwrap();
    let mb = align(&b.axes, &out).unwrap();
    let mut bd = B::new();
    let iv = bd.loops(&dims(&out));
    let ia: Vec<Expr> = ma.iter().map(|p| Expr::Iv(iv[*p])).collect();
    let ib: Vec<Expr> = mb.iter().map(|p| Expr::Iv(iv[*p])).collect();
    let value = binary_expr(op, load(0, ia), load(1, ib)).ok_or_else(|| format!("{}: not a binary primitive", op.name()))?;
    let body = vec![Stmt::Store { buf: 2, index: ivs(&iv), value }];
    let kern = kernel(op.name(), dtype, ins, &out, BufKind::Real, &bd, nest(&iv, &dims(&out), body));
    done(kern, out)
}

/// One loop nest for a fusion group (`presk-realize`): each input is read
/// through its own broadcast map, each step's value stays in the expression
/// tree instead of reaching memory, and only the last one is stored. Every
/// interior step has exactly one use, so the tree cannot blow up.
pub fn fused(steps: &[Step], ins: &[Operand], out: &[Axis], dtype: Dtype) -> Result<Lowered> {
    let mut bd = B::new();
    let iv = bd.loops(&dims(out));
    let mut idx: Vec<Vec<Expr>> = Vec::with_capacity(ins.len());
    for (i, o) in ins.iter().enumerate() {
        let m = align(&o.axes, out).ok_or_else(|| format!("fused: operand {i} of shape {} does not broadcast into {}", show(&o.axes), show(out)))?;
        idx.push(m.iter().map(|p| Expr::Iv(iv[*p])).collect());
    }
    let mut vals: Vec<Expr> = Vec::with_capacity(steps.len());
    for s in steps {
        let a: Vec<Expr> = s
            .args
            .iter()
            .map(|src| match src {
                Src::Input(k) => match ins[*k].kind {
                    BufKind::Idx => Expr::ToF(Box::new(Expr::LoadI(*k, idx[*k].clone()))),
                    BufKind::Real | BufKind::Quant { .. } | BufKind::Wide(_) => load(*k, idx[*k].clone()),
                },
                Src::Step(k) => vals[*k].clone(),
            })
            .collect();
        let value = match a.as_slice() {
            [x] => unary_expr(&s.op, x.clone()).ok_or_else(|| format!("fused: `{}` is not a unary primitive", s.op.name()))?,
            [x, y] => binary_expr(&s.op, x.clone(), y.clone()).ok_or_else(|| format!("fused: `{}` is not a binary primitive", s.op.name()))?,
            _ => return Err(format!("fused: `{}` takes {} arguments", s.op.name(), a.len())),
        };
        vals.push(value);
    }
    let value = vals.pop().ok_or_else(|| "fused: empty group".to_string())?;
    let body = nest(&iv, &dims(out), vec![Stmt::Store { buf: ins.len(), index: ivs(&iv), value }]);
    let kern = kernel(format!("fused {} steps", steps.len()), dtype, ins, out, BufKind::Real, &bd, body);
    done(kern, out.to_vec())
}

fn unary(op: &PrimOp, ins: &[Operand], dtype: Dtype) -> Result<Lowered> {
    let a = tensor_in(ins, 0, "unary")?;
    let mut bd = B::new();
    let iv = bd.loops(&dims(&a.axes));
    let x = if a.kind == BufKind::Idx { Expr::ToF(Box::new(Expr::LoadI(0, ivs(&iv)))) } else { load(0, ivs(&iv)) };
    let value = unary_expr(op, x).ok_or_else(|| format!("{}: not a unary primitive", op.name()))?;
    let body = vec![Stmt::Store { buf: 1, index: ivs(&iv), value }];
    let kern = kernel(op.name(), dtype, ins, &a.axes, BufKind::Real, &bd, nest(&iv, &dims(&a.axes), body));
    done(kern, a.axes.clone())
}

/// `acc = 0; for k in contracted: acc += x[..]·y[..]; out[..] = acc` for the
/// given index builders.
#[allow(clippy::too_many_arguments)]
fn contraction(name: &str, dtype: Dtype, ins: &[Operand], out: Vec<Axis>, outer_dims: Vec<usize>, inner_dims: Vec<usize>, index_x: impl Fn(&[usize], &[usize]) -> Vec<Expr>, index_y: impl Fn(&[usize], &[usize]) -> Vec<Expr>) -> Result<Lowered> {
    contraction_ordered(name, dtype, ins, out, outer_dims, inner_dims, index_x, index_y, ReductionOrder::Sequential)
}

/// The same, under a declared reduction order.
///
/// `Interleaved { partials: n }` keeps `n` accumulators over one contracted
/// axis, each summing every `n`-th term, and adds them at the end. That
/// **reassociates** the sum, so it is a different realization owing a
/// declared order and a measured gap (D31) — which is exactly why the order
/// is a field of the realization and not a flag on the backend. It also
/// breaks the serial dependency of the accumulator chain, which is what
/// makes it worth having: `k` dependent additions become `k/n`.
#[allow(clippy::too_many_arguments)]
fn contraction_ordered(name: &str, dtype: Dtype, ins: &[Operand], out: Vec<Axis>, outer_dims: Vec<usize>, inner_dims: Vec<usize>, index_x: impl Fn(&[usize], &[usize]) -> Vec<Expr>, index_y: impl Fn(&[usize], &[usize]) -> Vec<Expr>, order: ReductionOrder) -> Result<Lowered> {
    // `Interleaved { partials: n }`: `n` accumulators over the contracted
    // axis, each summing every `n`-th term, added at the end. This
    // **reassociates** the sum, which is the whole reason the order is a
    // declared field of the realization and not a flag on the backend (D31)
    // — and it is also why it is worth having, since `k` dependent additions
    // become `k/n` and the accumulator chain stops being the bottleneck.
    //
    // The contracted index is `cb·n + t` for an unrolled `t`, which needs a
    // *loop* variable to be assigned rather than iterated: `Stmt::SetL`.
    if let ReductionOrder::Interleaved { partials } = order
        && inner_dims.len() == 1
        && partials >= 2
        && inner_dims[0] % partials as usize == 0
    {
        let n = partials as usize;
        let chunks = inner_dims[0] / n;
        let mut bd = B::new();
        let ov = bd.loops(&outer_dims);
        let cb = bd.iv();
        let terms: Vec<usize> = (0..n).map(|_| bd.iv()).collect();
        let accs: Vec<usize> = (0..n).map(|_| bd.sv()).collect();
        let mut inner = Vec::new();
        for t in 0..n {
            let idx = bin(BinOp::IAdd, bin(BinOp::IMul, Expr::Iv(cb), Expr::I(n as i64)), Expr::I(t as i64));
            inner.push(Stmt::SetL { var: terms[t], value: idx });
            inner.push(Stmt::Set {
                var: accs[t],
                value: bin(BinOp::Add, Expr::Sv(accs[t]), bin(BinOp::Mul, load(0, index_x(&ov, &terms[t..t + 1])), load(1, index_y(&ov, &terms[t..t + 1])))),
            });
        }
        let mut body: Vec<Stmt> = accs.iter().map(|a| Stmt::Set { var: *a, value: f(0.0) }).collect();
        body.push(Stmt::For { var: cb, bound: chunks, body: inner });
        // The partials are combined left to right, so the shape of the tree
        // is fixed and declared rather than left to the backend.
        let mut total = Expr::Sv(accs[0]);
        for a in &accs[1..] {
            total = bin(BinOp::Add, total, Expr::Sv(*a));
        }
        body.push(Stmt::Store { buf: 2, index: ivs(&ov), value: total });
        let kern = kernel(name, dtype, ins, &out, BufKind::Real, &bd, nest(&ov, &outer_dims, body));
        return done(kern, out);
    }
    let mut bd = B::new();
    let ov = bd.loops(&outer_dims);
    let cv = bd.loops(&inner_dims);
    let acc = bd.sv();
    let inner = nest(&cv, &inner_dims, vec![Stmt::Set { var: acc, value: bin(BinOp::Add, Expr::Sv(acc), bin(BinOp::Mul, load(0, index_x(&ov, &cv)), load(1, index_y(&ov, &cv)))) }]);
    let mut body = vec![Stmt::Set { var: acc, value: f(0.0) }];
    body.extend(inner);
    body.push(Stmt::Store { buf: 2, index: ivs(&ov), value: Expr::Sv(acc) });
    let kern = kernel(name, dtype, ins, &out, BufKind::Real, &bd, nest(&ov, &outer_dims, body));
    done(kern, out)
}

/// `x[..., K] · w[K, rest…]` that is not a contraction nest: a right operand
/// of rank one or above two, a left operand of rank one, or an interleaved
/// order. One accumulator per output element (or `n` partial ones for an
/// interleaved order). A contraction nest is built by [`emit_contraction`].
fn dot(ins: &[Operand], dtype: Dtype, order: ReductionOrder) -> Result<Lowered> {
    let (x, w) = (tensor_in(ins, 0, "dot")?, tensor_in(ins, 1, "dot")?);
    if x.axes.is_empty() || w.axes.is_empty() {
        return Err("dot: both operands need at least one axis".into());
    }
    let k = x.axes[x.axes.len() - 1].len;
    if w.axes[0].len != k {
        return Err(format!("dot: contracted axes differ ({k} vs {})", w.axes[0].len));
    }
    let lead = x.axes.len() - 1;
    let mut out = x.axes[..lead].to_vec();
    out.extend(w.axes[1..].iter().cloned());
    let outer: Vec<usize> = dims(&out);
    contraction_ordered(
        "dot",
        dtype,
        ins,
        out,
        outer,
        vec![k],
        move |ov, cv| {
            let mut ix: Vec<Expr> = ov[..lead].iter().map(|v| Expr::Iv(*v)).collect();
            ix.push(Expr::Iv(cv[0]));
            ix
        },
        move |ov, cv| {
            let mut ix = vec![Expr::Iv(cv[0])];
            ix.extend(ov[lead..].iter().map(|v| Expr::Iv(*v)));
            ix
        },
        order,
    )
}

/// `ct[..., N…] · b[K, N…] → a[..., K]` (`k = rank(b) − 1` trailing axes of `ct` contracted)
/// that is not a contraction nest: a `b` of rank above two, a `ct` of rank
/// one, or an interleaved order. One accumulator per output element. A
/// contraction nest is built by [`emit_contraction`] (0.10.0 step 1).
fn dot_vjp_l(ins: &[Operand], dtype: Dtype) -> Result<Lowered> {
    let (ct, b) = (tensor_in(ins, 0, "dot_vjp_l")?, tensor_in(ins, 1, "dot_vjp_l")?);
    let k = b.axes.len() - 1;
    if ct.axes.len() < k {
        return Err("dot_vjp_l: cotangent has too few axes".into());
    }
    let lead = ct.axes.len() - k;
    let mut out = ct.axes[..lead].to_vec();
    out.push(b.axes[0].clone());
    let outer = dims(&out);
    let inner: Vec<usize> = b.axes[1..].iter().map(|a| a.len).collect();
    contraction(
        "dot_vjp_l",
        dtype,
        ins,
        out,
        outer,
        inner,
        move |ov, cv| {
            let mut ix: Vec<Expr> = ov[..lead].iter().map(|v| Expr::Iv(*v)).collect();
            ix.extend(cv.iter().map(|v| Expr::Iv(*v)));
            ix
        },
        move |ov, cv| {
            let mut ix = vec![Expr::Iv(ov[lead])];
            ix.extend(cv.iter().map(|v| Expr::Iv(*v)));
            ix
        },
    )
}

/// `a[..., K], ct[..., N…] → b[K, N…]`, summing over the `rank(a) − 1` leading axes,
/// that is not a contraction nest: a `ct` of another rank than `a`, an `a` of
/// rank one, or an interleaved order. One accumulator per output element. A
/// contraction nest is built by [`emit_contraction`] (0.10.0 step 1).
fn dot_vjp_r(ins: &[Operand], dtype: Dtype) -> Result<Lowered> {
    let (a, ct) = (tensor_in(ins, 0, "dot_vjp_r")?, tensor_in(ins, 1, "dot_vjp_r")?);
    let lead = a.axes.len() - 1;
    if ct.axes.len() < lead {
        return Err("dot_vjp_r: cotangent has too few axes".into());
    }
    let mut out = vec![a.axes[lead].clone()];
    out.extend(ct.axes[lead..].iter().cloned());
    let outer = dims(&out);
    let inner: Vec<usize> = a.axes[..lead].iter().map(|x| x.len).collect();
    contraction(
        "dot_vjp_r",
        dtype,
        ins,
        out,
        outer,
        inner,
        move |ov, cv| {
            let mut ix: Vec<Expr> = cv.iter().map(|v| Expr::Iv(*v)).collect();
            ix.push(Expr::Iv(ov[0]));
            ix
        },
        move |ov, cv| {
            let mut ix: Vec<Expr> = cv.iter().map(|v| Expr::Iv(*v)).collect();
            ix.extend(ov[1..].iter().map(|v| Expr::Iv(*v)));
            ix
        },
    )
}

fn ein(ins: &[Operand], operands: &[Vec<String>], output: &[String], dtype: Dtype) -> Result<Lowered> {
    let mut names: Vec<(String, usize)> = Vec::new();
    for (t, ns) in ins.iter().zip(operands) {
        for (a, n) in t.axes.iter().zip(ns) {
            match names.iter().find(|(m, _)| m == n) {
                Some((_, l)) if *l != a.len => return Err(format!("ein: axis `{n}` has lengths {l} and {}", a.len)),
                Some(_) => {}
                None => names.push((n.clone(), a.len)),
            }
        }
    }
    for o in output {
        if !names.iter().any(|(n, _)| n == o) {
            return Err(format!("ein: output axis `{o}` appears in no operand"));
        }
    }
    let contracted: Vec<(String, usize)> = names.iter().filter(|(n, _)| !output.contains(n)).cloned().collect();
    let out: Vec<Axis> = output.iter().map(|o| Axis { name: Some(o.clone()), len: names.iter().find(|(n, _)| n == o).unwrap().1 }).collect();
    let all: Vec<String> = output.iter().cloned().chain(contracted.iter().map(|(n, _)| n.clone())).collect();
    let mut bd = B::new();
    let ov = bd.loops(&dims(&out));
    let cv = bd.loops(&contracted.iter().map(|(_, l)| *l).collect::<Vec<_>>());
    let all_v: Vec<usize> = ov.iter().chain(cv.iter()).copied().collect();
    let acc = bd.sv();
    let mut prod = f(1.0);
    for (i, ns) in operands.iter().enumerate() {
        let ix: Vec<Expr> = ns.iter().map(|n| Expr::Iv(all_v[all.iter().position(|m| m == n).unwrap()])).collect();
        prod = bin(BinOp::Mul, prod, load(i, ix));
    }
    let inner = nest(&cv, &contracted.iter().map(|(_, l)| *l).collect::<Vec<_>>(), vec![Stmt::Set { var: acc, value: bin(BinOp::Add, Expr::Sv(acc), prod) }]);
    let mut body = vec![Stmt::Set { var: acc, value: f(0.0) }];
    body.extend(inner);
    body.push(Stmt::Store { buf: ins.len(), index: ivs(&ov), value: Expr::Sv(acc) });
    let kern = kernel(format!("ein[{}]", output.join(", ")), dtype, ins, &out, BufKind::Real, &bd, nest(&ov, &dims(&out), body));
    done(kern, out)
}

// ----- contraction nests (0.7.0 step 2) ------------------------------------------

/// Where an axis of a contraction's operand comes from: an outer axis (its
/// position among them), `I`, `J` or `K`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    Outer(usize),
    I,
    J,
    K,
}

/// A contraction nest (language reference, definition "Contraction nest"): a
/// `dot` whose right operand has rank two and whose left has rank at least two,
/// or an `ein` whose output is the outer axes, then `I`, then `J`, every outer
/// axis carried by both operands, `I` by operand 0 only, `J` by operand 1 only,
/// over one contracted axis `K` — under an exact reduction order. Since 0.10.0
/// step 1 (D89), also the two derivatives of such a `dot`: `dot_vjp_l`, the
/// NT product `ct · bᵀ`, and `dot_vjp_r`, the product `aᵀ · ct` summed over
/// every leading axis of its operands, taken as one summed axis `K` of their
/// lengths' product iterated lexicographically ([`Contraction::ks`]). Its
/// output is always (outer…, `I`, `J`), so its kernel is one builder for
/// every primitive ([`emit_contraction`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Contraction {
    /// The kernel's name: `dot`, `dot_vjp_l`, `dot_vjp_r`, or `ein[…]` as
    /// 0.6.0 named it.
    pub name: String,
    /// The output's axes: the outer axes, `I`, `J`.
    pub out: Vec<Axis>,
    pub outer: Vec<usize>,
    pub i: usize,
    pub j: usize,
    /// The length of the summed axis `K`: the product of [`Contraction::ks`].
    pub k: usize,
    /// The axes `K` runs over, outermost first: one, except in a `dot_vjp_r`
    /// whose operands have several leading axes. Their iterations in
    /// lexicographic order are `K`'s in increasing order, since both operands
    /// carry them first, in this order, and row-major; so the slot `K` stands
    /// for all of them in an operand's index, and a k-block divides the
    /// innermost of them.
    pub ks: Vec<usize>,
    /// Operand 0's axes, and operand 1's, as slots.
    pub x_index: Vec<Slot>,
    pub w_index: Vec<Slot>,
    pub dot: bool,
    /// `K` is operand 1's innermost axis, so `J` is not contiguous in it.
    pub nt: bool,
    /// Where the tile holds one accumulator of one lane, operand 0 is read
    /// inside the product rather than into a variable first: how 0.9.0
    /// lowered the derivatives of `dot`, so that their default kernel is
    /// 0.9.0's byte for byte (D89). The two texts compute the same product.
    pub inline_x: bool,
}

impl Contraction {
    /// The contraction nest a primitive is at these operands, if it is one.
    pub fn of(op: &PrimOp, ins: &[Operand], order: ReductionOrder) -> Option<Contraction> {
        if !order.is_exact() || ins.len() != 2 {
            return None;
        }
        match op {
            PrimOp::Dot | PrimOp::Matmul => {
                let (x, w) = (&ins[0].axes, &ins[1].axes);
                if w.len() != 2 || x.len() < 2 || x[x.len() - 1].len != w[0].len {
                    return None;
                }
                let p = x.len() - 2;
                let mut out = x[..=p].to_vec();
                out.push(w[1].clone());
                let mut x_index: Vec<Slot> = (0..p).map(Slot::Outer).collect();
                x_index.extend([Slot::I, Slot::K]);
                Some(Contraction { name: "dot".into(), out, outer: x[..p].iter().map(|a| a.len).collect(), i: x[p].len, j: w[1].len, k: w[0].len, ks: vec![w[0].len], x_index, w_index: vec![Slot::K, Slot::J], dot: true, nt: false, inline_x: false })
            }
            // `ct[outer…, I, N] · b[J, N] → [outer…, I, J]`: the derivative of
            // a `dot` contraction nest in its left operand, NT, its summed axis
            // `N` innermost in both operands.
            PrimOp::DotVjpL => {
                let (ct, b) = (&ins[0].axes, &ins[1].axes);
                if b.len() != 2 || ct.len() < 2 || ct[ct.len() - 1].len != b[1].len {
                    return None;
                }
                let p = ct.len() - 2;
                let mut out = ct[..=p].to_vec();
                out.push(b[0].clone());
                let mut x_index: Vec<Slot> = (0..p).map(Slot::Outer).collect();
                x_index.extend([Slot::I, Slot::K]);
                Some(Contraction { name: "dot_vjp_l".into(), out, outer: ct[..p].iter().map(|a| a.len).collect(), i: ct[p].len, j: b[0].len, k: b[1].len, ks: vec![b[1].len], x_index, w_index: vec![Slot::J, Slot::K], dot: false, nt: true, inline_x: true })
            }
            // `a[lead…, I], ct[lead…, J] → [I, J]`, summed over the leading
            // axes: the derivative of a `dot` contraction nest in its right
            // operand, operand 0 transposed (`K` outermost).
            PrimOp::DotVjpR => {
                let (a, ct) = (&ins[0].axes, &ins[1].axes);
                let lead = a.len().checked_sub(1)?;
                if lead == 0 || ct.len() != a.len() || a[..lead].iter().zip(&ct[..lead]).any(|(x, y)| x.len != y.len) {
                    return None;
                }
                let ks: Vec<usize> = a[..lead].iter().map(|x| x.len).collect();
                let out = vec![a[lead].clone(), ct[lead].clone()];
                Some(Contraction { name: "dot_vjp_r".into(), out, outer: vec![], i: a[lead].len, j: ct[lead].len, k: ks.iter().product(), ks, x_index: vec![Slot::K, Slot::I], w_index: vec![Slot::K, Slot::J], dot: false, nt: false, inline_x: true })
            }
            PrimOp::Ein { operands, output } if operands.len() == 2 => {
                let n = output.len();
                let (a, b) = (&operands[0], &operands[1]);
                if n < 2 || a.len() != n || b.len() != n || ins[0].axes.len() != n || ins[1].axes.len() != n {
                    return None;
                }
                let (outer, i, j) = (&output[..n - 2], &output[n - 2], &output[n - 1]);
                let ks: Vec<&String> = a.iter().filter(|x| !output.contains(x)).collect();
                let [k] = ks.as_slice() else { return None };
                let slot = |x: &String, own: &String, own_slot: Slot| -> Option<Slot> {
                    if x == *k {
                        Some(Slot::K)
                    } else if x == own {
                        Some(own_slot)
                    } else {
                        outer.iter().position(|o| o == x).map(Slot::Outer)
                    }
                };
                let x_index: Vec<Slot> = a.iter().map(|x| slot(x, i, Slot::I)).collect::<Option<_>>()?;
                let w_index: Vec<Slot> = b.iter().map(|x| slot(x, j, Slot::J)).collect::<Option<_>>()?;
                // Each operand carries every slot it should exactly once.
                let wants = |idx: &[Slot], own: Slot| {
                    let mut want: Vec<Slot> = (0..outer.len()).map(Slot::Outer).collect();
                    want.extend([own, Slot::K]);
                    want.iter().all(|s| idx.iter().filter(|t| *t == s).count() == 1)
                };
                if !wants(&x_index, Slot::I) || !wants(&w_index, Slot::J) {
                    return None;
                }
                let len_x = |s: Slot| x_index.iter().position(|t| *t == s).map(|p| ins[0].axes[p].len);
                let len_w = |s: Slot| w_index.iter().position(|t| *t == s).map(|p| ins[1].axes[p].len);
                if len_x(Slot::K) != len_w(Slot::K) || (0..outer.len()).any(|p| len_x(Slot::Outer(p)) != len_w(Slot::Outer(p))) {
                    return None;
                }
                let outer_len: Vec<usize> = (0..outer.len()).map(|p| len_x(Slot::Outer(p))).collect::<Option<_>>()?;
                let (ni, nj, nk) = (len_x(Slot::I)?, len_w(Slot::J)?, len_x(Slot::K)?);
                let mut out: Vec<Axis> = outer.iter().zip(&outer_len).map(|(o, l)| Axis { name: Some(o.clone()), len: *l }).collect();
                out.push(Axis { name: Some(i.clone()), len: ni });
                out.push(Axis { name: Some(j.clone()), len: nj });
                let nt = w_index.last() == Some(&Slot::K);
                Some(Contraction { name: format!("ein[{}]", output.join(", ")), out, outer: outer_len, i: ni, j: nj, k: nk, ks: vec![nk], x_index, w_index, dot: false, nt, inline_x: false })
            }
            _ => None,
        }
    }

    /// The outer axes operand 1 carries, by their positions among the outer
    /// axes: every one in an `ein`, none in a `dot` or a `dot_vjp_l`, whose
    /// operand 1 is a matrix.
    pub fn w_outer(&self) -> Vec<usize> {
        (0..self.outer.len()).filter(|o| self.w_index.contains(&Slot::Outer(*o))).collect()
    }

    /// The axes of operand 1's panel under a pack: the outer axes it carries,
    /// `K`, `J`, named as operand 1 names them.
    pub fn panel_axes(&self, w: &Operand) -> Vec<Axis> {
        let at = |s: Slot| w.axes[self.w_index.iter().position(|t| *t == s).expect("operand 1 carries K and J")].clone();
        let mut axes: Vec<Axis> = self.w_outer().into_iter().map(|o| at(Slot::Outer(o))).collect();
        axes.push(at(Slot::K));
        axes.push(at(Slot::J));
        axes
    }

    /// The same contraction reading operand 1 from its panel, which is NN.
    pub fn packed(&self) -> Contraction {
        let mut w_index: Vec<Slot> = self.w_outer().into_iter().map(Slot::Outer).collect();
        w_index.extend([Slot::K, Slot::J]);
        Contraction { w_index, nt: false, ..self.clone() }
    }

    /// What a layout needs to know of this nest, in a storage type and an
    /// accumulation type.
    pub fn kind(&self, dtype: Dtype, accum: Dtype) -> presk_rt::layout::Contraction {
        presk_rt::layout::Contraction { dot: self.dot, i: self.i, j: self.j, k: self.k, k_inner: self.k_inner(), nt: self.nt, accum_is_storage: accum == dtype, width: dtype.natural_lanes() }
    }

    /// The innermost of the axes `K` runs over, which a k-block divides.
    pub fn k_inner(&self) -> usize {
        self.ks.last().copied().unwrap_or(self.k)
    }
}

/// An operand's index: its slots, given the outer loop variables and the
/// expressions of `i`, `j` and of each axis `K` runs over.
fn slots(idx: &[Slot], outer: &[usize], i: &Expr, j: &Expr, k: &[Expr]) -> Vec<Expr> {
    idx.iter()
        .flat_map(|s| match s {
            Slot::Outer(p) => vec![Expr::Iv(outer[*p])],
            Slot::I => vec![i.clone()],
            Slot::J => vec![j.clone()],
            Slot::K => k.to_vec(),
        })
        .collect()
}

/// The `add` a contraction nest's kernel performs once each output element's
/// sum is done (plan 0.10.0 step 7): its epilogue, a bias added in the
/// product's kernel. The operand added is the kernel's last input, read
/// through its broadcast into the output, as the `add`'s kernel reads it.
/// Exact: the separate nests round the sum to the storage type, then add in
/// the arithmetic type and round again, and so does the epilogue. Where the
/// two types are one (`f64`, `f32`) the sum is added to in its register
/// before its store; where storage is narrower (`bf16`), the sum is stored,
/// which rounds it, and the stored value is read back and added to.
#[derive(Clone, Debug, PartialEq)]
pub struct Epilogue {
    /// The product is the `add`'s first operand: `sum + b`; else `b + sum`.
    pub first: bool,
    /// The `add`'s result axes: the product's lengths, under the names the
    /// `add` gives them. The kernel's output.
    pub out: Vec<Axis>,
}

impl Epilogue {
    /// `sum + b`, or `b + sum`, as the `add` has it.
    fn add(&self, sum: Expr, b: Expr) -> Expr {
        if self.first { bin(BinOp::Add, sum, b) } else { bin(BinOp::Add, b, sum) }
    }

    fn vadd(&self, sum: Expr, b: Expr) -> Expr {
        let (x, y) = if self.first { (sum, b) } else { (b, sum) };
        Expr::VBin(BinOp::Add, Box::new(x), Box::new(y))
    }
}

/// `iv·n + t`: the index a block loop of width `n` gives its `t`-th offset,
/// or the loop's variable itself when the width is one.
fn blocked(iv: usize, n: usize, t: usize) -> Expr {
    if n == 1 { Expr::Iv(iv) } else { bin(BinOp::IAdd, bin(BinOp::IMul, Expr::Iv(iv), Expr::I(n as i64)), Expr::I(t as i64)) }
}

/// A contraction nest's kernel under a resolved layout (0.7.0 step 2; the
/// reference's §6.3 and the tile loops of §7.1). Two forms:
///
/// - **row-wise**: the outer loops and `I`; the output row zeroed; then for
///   each `k` one element of operand 0 swept along a row of operand 1 and of
///   the output, each running sum living in the output buffer from one `k` to
///   the next (0.4.0's form for a width that is not a multiple of eight);
/// - **tiled**: the outer loops, then the block loops over `I`, `J` and, with
///   a k-block, `K` in the stated order, then a micro-kernel holding `m_i ×
///   n_j` accumulators — scalar, or `n_j / lanes` vectors per row — over the
///   `k` of one block. When the loop over the blocks of `K` encloses the one
///   over `I` or `J`, the accumulators are stored at the end of a block and
///   reloaded at the start of the next (a spill).
///
/// Every loop counts upward and the within-block loop runs inside its block
/// loop, so every output element adds its terms in increasing `k` under every
/// tile, order and lane count: the forms differ in where a running sum lives,
/// never in the order of the additions. A spill stores and reloads in the
/// storage type, which is exact where the layout is well-formed (condition 7).
/// At the default layout the text is 0.6.0's `dot`, character for character,
/// and 0.9.0's `dot_vjp_l` and `dot_vjp_r` (D89).
///
/// When `K` runs over several axes ([`Contraction::ks`]), its loop is one
/// loop per axis, outermost first, and with a k-block the loop over the
/// blocks of `K` is one loop per leading axis, then the loop over the blocks
/// of the innermost: the same `k` in the same order, since the flat index of
/// row-major axes is lexicographic.
pub fn emit_contraction(c: &Contraction, dtype: Dtype, ins: &[Operand], layout: &Layout) -> Result<Lowered> {
    emit_contraction_with(c, dtype, ins, layout, None)
}

/// The same, with an epilogue when `epi` is given: the operand it adds is
/// `ins`'s last, after the contraction's two (plan 0.10.0 step 7). Its sum
/// is added to once the last block of `K` is done, in whichever form.
pub fn emit_contraction_with(c: &Contraction, dtype: Dtype, ins: &[Operand], layout: &Layout, epi: Option<&Epilogue>) -> Result<Lowered> {
    let lanes = layout.lanes.unwrap_or(1);
    let want = if epi.is_some() { 3 } else { 2 };
    if ins.len() != want {
        return Err(format!("{}: {} operands, {want} expected", c.name, ins.len()));
    }
    // Where the epilogue's operand reads each output position: the output's
    // positions its axes align with.
    let bias = match epi {
        Some(e) => {
            let b = &ins[2];
            if e.out.len() != c.out.len() || e.out.iter().zip(&c.out).any(|(a, o)| a.len != o.len) {
                return Err(format!("{}: the epilogue's output {} is not the product's {}", c.name, show(&e.out), show(&c.out)));
            }
            Some(align(&b.axes, &e.out).ok_or_else(|| format!("{}: the epilogue's operand {} does not broadcast into {}", c.name, show(&b.axes), show(&e.out)))?)
        }
        None => None,
    };
    if layout.form == Some(Form::RowWise) {
        if lanes != 1 {
            return Err(format!("{}: a row-wise contraction has one lane, not {lanes}", c.name));
        }
        return row_wise(c, dtype, ins, epi.zip(bias.as_deref()));
    }
    let (mi, nj) = layout.tile.unwrap_or((1, 1));
    let kblock = layout.kblock.unwrap_or(KBlock::None);
    let order = layout.order.clone().unwrap_or_else(|| match kblock {
        KBlock::None => vec![BlockLoop::I, BlockLoop::J],
        KBlock::Width(_) => vec![BlockLoop::I, BlockLoop::J, BlockLoop::K],
    });
    if mi == 0 || nj == 0 || c.i % mi != 0 || c.j % nj != 0 || nj % lanes != 0 {
        return Err(format!("{}: the tile [{mi}, {nj}] with {lanes} lanes does not fit I = {}, J = {}", c.name, c.i, c.j));
    }
    if lanes > 1 && c.nt {
        return Err(format!("{}: operand 1 has K innermost, so its J is not contiguous and cannot be loaded in lanes", c.name));
    }
    let b = match kblock {
        KBlock::None => None,
        KBlock::Width(b) if b > 0 && c.k_inner() % b == 0 => Some(b),
        KBlock::Width(b) if c.ks.len() > 1 => return Err(format!("{}: the k-block {b} does not divide the innermost summed axis, {}", c.name, c.k_inner())),
        KBlock::Width(b) => return Err(format!("{}: the k-block {b} does not divide K = {}", c.name, c.k)),
    };
    let mut sorted = order.clone();
    sorted.sort_by_key(|l| *l as u8);
    let want = if b.is_some() { vec![BlockLoop::I, BlockLoop::J, BlockLoop::K] } else { vec![BlockLoop::I, BlockLoop::J] };
    if sorted != want {
        return Err(format!("{}: the loop order does not match the k-block", c.name));
    }

    let mut bd = B::new();
    let ov = bd.loops(&c.outer);
    // The block loops' variables in loop order, then the loop over k (or,
    // with a k-block, the within-block loop and the `k` it sets): at the
    // default layout, the numbering of 0.6.0's blocked `dot`. The leading
    // axes `K` runs over, when there are several, take their variables where
    // the loop over the blocks of `K` is, or before the loop over the last.
    let lead_k = c.ks.len().saturating_sub(1);
    let mut var = [0usize; 3];
    let mut kouter: Vec<usize> = vec![];
    for l in &order {
        if *l == BlockLoop::K {
            kouter = bd.loops(&c.ks[..lead_k]);
        }
        var[*l as usize] = bd.iv();
    }
    let (kloops, kx): (Vec<usize>, Vec<Expr>) = match b {
        None => {
            let kv = bd.loops(&c.ks);
            (kv.clone(), ivs(&kv))
        }
        Some(_) => {
            let (kloop, kvar) = (bd.iv(), bd.iv());
            (vec![kloop, kvar], kouter.iter().chain([&kvar]).map(|v| Expr::Iv(*v)).collect())
        }
    };
    let (vi, vj, vk) = (var[BlockLoop::I as usize], var[BlockLoop::J as usize], var[BlockLoop::K as usize]);
    let vector = lanes > 1;
    let cols = nj / lanes;
    let inline = c.inline_x && mi == 1 && nj == 1;
    let fresh = |bd: &mut B| if vector { bd.vv(lanes) } else { bd.sv() };
    let accs: Vec<Vec<usize>> = (0..mi).map(|_| (0..cols).map(|_| fresh(&mut bd)).collect()).collect();
    let xs: Vec<usize> = if inline { vec![] } else { (0..mi).map(|_| bd.sv()).collect() };
    // With several rows, operand 1's columns are loaded once per k and shared.
    let ws: Vec<usize> = if mi > 1 { (0..cols).map(|_| fresh(&mut bd)).collect() } else { vec![] };

    let (ix, jx) = (|r: usize| blocked(vi, mi, r), |g: usize| blocked(vj, nj, g * lanes));
    let out_ix = |r: usize, g: usize| {
        let mut v = ivs(&ov);
        v.push(ix(r));
        v.push(jx(g));
        v
    };
    let x_at = |r: usize| slots(&c.x_index, &ov, &ix(r), &Expr::I(0), &kx);
    let w_at = |g: usize| slots(&c.w_index, &ov, &Expr::I(0), &jx(g), &kx);
    let x_of = |r: usize| if inline { load(0, x_at(r)) } else { Expr::Sv(xs[r]) };
    let buf_out = ins.len();

    let mut micro: Vec<Stmt> = xs.iter().enumerate().map(|(r, x)| Stmt::Set { var: *x, value: load(0, x_at(r)) }).collect();
    for (g, w) in ws.iter().enumerate() {
        micro.push(if vector { Stmt::VSet { var: *w, value: Expr::VLoad(1, w_at(g)) } } else { Stmt::Set { var: *w, value: load(1, w_at(g)) } });
    }
    for (r, row) in accs.iter().enumerate() {
        for (g, a) in row.iter().enumerate() {
            let wv = match (ws.get(g), vector) {
                (Some(w), true) => Expr::Vv(*w),
                (Some(w), false) => Expr::Sv(*w),
                (None, true) => Expr::VLoad(1, w_at(g)),
                (None, false) => load(1, w_at(g)),
            };
            micro.push(if vector {
                Stmt::VSet { var: *a, value: Expr::VBin(BinOp::Add, Box::new(Expr::Vv(*a)), Box::new(Expr::VBin(BinOp::Mul, Box::new(Expr::VSplat(Box::new(x_of(r)))), Box::new(wv)))) }
            } else {
                Stmt::Set { var: *a, value: bin(BinOp::Add, Expr::Sv(*a), bin(BinOp::Mul, x_of(r), wv)) }
            });
        }
    }
    let each = |f: &dyn Fn(usize, usize, usize) -> Stmt| -> Vec<Stmt> { accs.iter().enumerate().flat_map(|(r, row)| row.iter().enumerate().map(move |(g, a)| (r, g, *a))).map(|(r, g, a)| f(r, g, a)).collect() };
    let init = each(&|_, _, a| if vector { Stmt::VSet { var: a, value: Expr::VSplat(Box::new(f(0.0))) } } else { Stmt::Set { var: a, value: f(0.0) } });
    let store = each(&|r, g, a| if vector { Stmt::VStore { buf: buf_out, index: out_ix(r, g), value: Expr::Vv(a) } } else { Stmt::Store { buf: buf_out, index: out_ix(r, g), value: Expr::Sv(a) } });
    let reload = each(&|r, g, a| if vector { Stmt::VSet { var: a, value: Expr::VLoad(buf_out, out_ix(r, g)) } } else { Stmt::Set { var: a, value: load(buf_out, out_ix(r, g)) } });
    // The epilogue (step 7): the operand at output position (r, g), and what
    // the tile does with it once its sums are done.
    let narrow = dtype.arith() != dtype;
    let (epi_add, epi_rounded) = match (epi, &bias) {
        (Some(e), Some(map)) => {
            let rank = c.out.len();
            let at = |r: usize, g: usize| -> Vec<Expr> {
                map.iter()
                    .map(|p| if *p + 2 < rank { Expr::Iv(ov[*p]) } else if *p + 2 == rank { ix(r) } else { jx(g) })
                    .collect()
            };
            let operand = |r: usize, g: usize| -> Result<Expr> {
                if !vector {
                    return Ok(load(2, at(r, g)));
                }
                // Lanes read `lanes` columns of the operand: contiguous where
                // `J` is its innermost axis, one value where it has no `J`.
                match map.iter().position(|p| *p + 1 == rank) {
                    Some(q) if q + 1 == map.len() => Ok(Expr::VLoad(2, at(r, g))),
                    Some(_) => Err(format!("{}: the epilogue's operand {} has J, but not innermost, so its columns are not contiguous and cannot be loaded in lanes", c.name, show(&ins[2].axes))),
                    None => Ok(Expr::VSplat(Box::new(load(2, at(r, g))))),
                }
            };
            let mut adds = Vec::new();
            for (r, row) in accs.iter().enumerate() {
                for (g, a) in row.iter().enumerate() {
                    let b = operand(r, g)?;
                    adds.push(if narrow {
                        if vector {
                            return Err(format!("{}: lanes in a type narrower than its arithmetic", c.name));
                        }
                        // The sum was stored, rounded to the storage type: read
                        // it back, add, and store the rounded sum.
                        Stmt::Store { buf: buf_out, index: out_ix(r, g), value: e.add(load(buf_out, out_ix(r, g)), b) }
                    } else if vector {
                        Stmt::VSet { var: *a, value: e.vadd(Expr::Vv(*a), b) }
                    } else {
                        Stmt::Set { var: *a, value: e.add(Expr::Sv(*a), b) }
                    });
                }
            }
            (adds, narrow)
        }
        _ => (vec![], false),
    };

    // The loop over the k of one block (or of all of K).
    let over_k = match b {
        None => nest(&kloops, &c.ks, micro),
        Some(b) => {
            let (kloop, kvar) = (kloops[0], kloops[1]);
            let mut body = vec![Stmt::SetL { var: kvar, value: bin(BinOp::IAdd, bin(BinOp::IMul, Expr::Iv(vk), Expr::I(b as i64)), Expr::Iv(kloop)) }];
            body.extend(micro);
            vec![Stmt::For { var: kloop, bound: b, body }]
        }
    };
    // Each block loop's loops, outermost first: the loop over the blocks of
    // K is preceded by one per leading axis K runs over.
    let bounds = |l: BlockLoop| -> Vec<(usize, usize)> {
        match l {
            BlockLoop::I => vec![(vi, c.i / mi)],
            BlockLoop::J => vec![(vj, c.j / nj)],
            BlockLoop::K => kouter.iter().copied().zip(c.ks[..lead_k].iter().copied()).chain([(vk, c.k_inner() / b.unwrap_or(1))]).collect(),
        }
    };
    let wrap = |loops: &[BlockLoop], body: Vec<Stmt>| -> Vec<Stmt> {
        let mut body = body;
        for (v, n) in loops.iter().flat_map(|l| bounds(*l)).collect::<Vec<_>>().into_iter().rev() {
            body = vec![Stmt::For { var: v, bound: n, body }];
        }
        body
    };
    let spill = b.is_some() && order.last() != Some(&BlockLoop::K);
    // Not the first block of K: a block variable is not zero.
    let later_block = kouter.iter().rev().fold(Expr::Iv(vk), |e, v| bin(BinOp::IAdd, Expr::Iv(*v), e));
    // The last block of K: every block variable at its last value, so their
    // sum at the sum of those.
    let last_sum: usize = c.ks[..lead_k].iter().map(|n| n - 1).sum::<usize>() + (c.k_inner() / b.unwrap_or(1)).saturating_sub(1);
    let last_block = Expr::Cmp(CmpOp::Eq, Box::new(Expr::ToF(Box::new(later_block.clone()))), Box::new(f(last_sum as f64)));
    // A spilling tile stores its sums at the end of every block of K, and
    // the epilogue belongs to the last.
    let epilogue = |spill: bool| -> Vec<Stmt> {
        if spill && !epi_add.is_empty() { vec![Stmt::If { cond: last_block.clone(), then: epi_add.clone() }] } else { epi_add.clone() }
    };
    let tile = |k_part: Vec<Stmt>, spill: bool| -> Vec<Stmt> {
        let mut v = init.clone();
        if spill {
            // After the first block of K, the running sums come back from the
            // output, where the previous block left them (condition 7).
            v.push(Stmt::If { cond: later_block.clone(), then: reload.clone() });
        }
        v.extend(k_part);
        if !epi_rounded {
            v.extend(epilogue(spill));
        }
        v.extend(store.clone());
        if epi_rounded {
            v.extend(epilogue(spill));
        }
        v
    };
    let body = if spill {
        wrap(&order, tile(over_k, true))
    } else if b.is_some() {
        // K innermost: the blocks of K run inside the tile, in registers.
        let kb = wrap(&[BlockLoop::K], over_k);
        wrap(&order[..2], tile(kb, false))
    } else {
        wrap(&order, tile(over_k, false))
    };
    let out = epi.map_or(&c.out, |e| &e.out);
    let kern = kernel(epilogue_name(c, epi), dtype, ins, out, BufKind::Real, &bd, nest(&ov, &c.outer, body));
    done(kern, out.clone())
}

/// A contraction's kernel name, `dot+add` with an epilogue.
fn epilogue_name(c: &Contraction, epi: Option<&Epilogue>) -> String {
    match epi {
        Some(_) => format!("{}+add", c.name),
        None => c.name.clone(),
    }
}

/// The pack of an NT contraction (0.7.0 step 3): operand 1 copied, in its own
/// type, to a panel with the axes (the outer axes it carries…, `K`, `J`),
/// which the contraction then reads as NN. A copy moves no value's bits, so
/// the pack is layout; it is a kernel of its own, launched just before the
/// contraction's (spec question 66). Written along `J`, the panel's innermost
/// axis.
pub fn pack_panel(c: &Contraction, dtype: Dtype, w: &Operand) -> Result<Lowered> {
    if c.ks.len() > 1 {
        // Only an NT contraction is packed (condition 9), and `K` runs over
        // several axes in a `dot_vjp_r` only, which is NN.
        return Err(format!("{}: a pack of a sum over several axes", c.name));
    }
    let axes = c.panel_axes(w);
    let mut bd = B::new();
    let lv = bd.loops(&dims(&axes));
    // The panel's loop over each outer axis operand 1 carries, by the axis's
    // position among the outer axes.
    let carried = c.w_outer();
    let mut ov = vec![0; c.outer.len()];
    for (n, o) in carried.iter().enumerate() {
        ov[*o] = lv[n];
    }
    let p = carried.len();
    let (k, j) = (Expr::Iv(lv[p]), Expr::Iv(lv[p + 1]));
    let src = slots(&c.w_index, &ov, &Expr::I(0), &j, std::slice::from_ref(&k));
    let body = vec![Stmt::Store { buf: 1, index: ivs(&lv), value: load(0, src) }];
    let kern = kernel("pack", dtype, std::slice::from_ref(w), &axes, BufKind::Real, &bd, nest(&lv, &dims(&axes), body));
    done(kern, axes)
}

/// The row-wise form: for each output row, `out = 0`, then for each `k`,
/// `out[j] += x[k] · w[k, j]` along the row. 0.4.0's form for a `dot` whose
/// width is not a multiple of eight, and a point of every contraction's
/// layout space since 0.7.0. When `K` runs over several axes, its loop is
/// one loop per axis, outermost first.
fn row_wise(c: &Contraction, dtype: Dtype, ins: &[Operand], epi: Option<(&Epilogue, &[usize])>) -> Result<Lowered> {
    let mut bd = B::new();
    let mut rows = c.outer.clone();
    rows.push(c.i);
    let ov = bd.loops(&rows);
    let kv = bd.loops(&c.ks);
    let (jv, acc) = (bd.iv(), bd.sv());
    let (outer, iv) = (&ov[..ov.len() - 1], Expr::Iv(ov[ov.len() - 1]));
    let buf_out = ins.len();
    let cix = |j: usize| -> Vec<Expr> {
        let mut ix: Vec<Expr> = ov.iter().map(|v| Expr::Iv(*v)).collect();
        ix.push(Expr::Iv(j));
        ix
    };
    let zero = Stmt::For { var: jv, bound: c.j, body: vec![Stmt::Store { buf: buf_out, index: cix(jv), value: f(0.0) }] };
    let kx = ivs(&kv);
    let xix = slots(&c.x_index, outer, &iv, &Expr::I(0), &kx);
    let wix = slots(&c.w_index, outer, &iv, &Expr::Iv(jv), &kx);
    let sweep = Stmt::For {
        var: jv,
        bound: c.j,
        body: vec![Stmt::Store { buf: buf_out, index: cix(jv), value: bin(BinOp::Add, load(buf_out, cix(jv)), bin(BinOp::Mul, Expr::Sv(acc), load(1, wix))) }],
    };
    let mut body = vec![zero];
    body.extend(nest(&kv, &c.ks, vec![Stmt::Set { var: acc, value: load(0, xix) }, sweep]));
    // The epilogue: the row's sums are in the output, in its type, as the
    // `add` would read them; one more sweep adds the operand to them.
    if let Some((e, map)) = epi {
        let rank = c.out.len();
        let bix: Vec<Expr> = map.iter().map(|p| if *p + 1 == rank { Expr::Iv(jv) } else { Expr::Iv(ov[*p]) }).collect();
        body.push(Stmt::For { var: jv, bound: c.j, body: vec![Stmt::Store { buf: buf_out, index: cix(jv), value: e.add(load(buf_out, cix(jv)), load(2, bix)) }] });
    }
    let out = epi.map_or(&c.out, |(e, _)| &e.out);
    let kern = kernel(epilogue_name(c, epi.map(|(e, _)| e)), dtype, ins, out, BufKind::Real, &bd, nest(&ov, &rows, body));
    done(kern, out.clone())
}

fn reduce_all(op: &PrimOp, ins: &[Operand], dtype: Dtype) -> Result<Lowered> {
    let t = tensor_in(ins, 0, "sum")?;
    let mut bd = B::new();
    let iv = bd.loops(&dims(&t.axes));
    let acc = bd.sv();
    let mut body = vec![Stmt::Set { var: acc, value: f(0.0) }];
    body.extend(nest(&iv, &dims(&t.axes), vec![Stmt::Set { var: acc, value: bin(BinOp::Add, Expr::Sv(acc), load(0, ivs(&iv))) }]));
    let n = t.axes.iter().map(|a| a.len).product::<usize>() as f64;
    let value = if matches!(op, PrimOp::MeanAll) { bin(BinOp::Div, Expr::Sv(acc), f(n)) } else { Expr::Sv(acc) };
    body.push(Stmt::Store { buf: 1, index: vec![], value });
    let kern = kernel(op.name(), dtype, ins, &[], BufKind::Real, &bd, body);
    done(kern, vec![])
}

fn axis_pos(axes: &[Axis], name: &str) -> Result<usize> {
    axes.iter().position(|a| a.name.as_deref() == Some(name)).ok_or_else(|| format!("no axis `{name}` in {}", show(axes)))
}

fn reduce_axis(op: &PrimOp, axis: &str, ins: &[Operand], dtype: Dtype) -> Result<Lowered> {
    let t = tensor_in(ins, 0, "sum")?;
    let p = axis_pos(&t.axes, axis)?;
    let mut out = t.axes.clone();
    let n = out.remove(p).len;
    let mut bd = B::new();
    let ov = bd.loops(&dims(&out));
    let k = bd.iv();
    let acc = bd.sv();
    let mut full: Vec<Expr> = ivs(&ov);
    full.insert(p, Expr::Iv(k));
    let mut body = vec![Stmt::Set { var: acc, value: f(0.0) }, Stmt::For { var: k, bound: n, body: vec![Stmt::Set { var: acc, value: bin(BinOp::Add, Expr::Sv(acc), load(0, full)) }] }];
    let value = if matches!(op, PrimOp::Mean { .. }) { bin(BinOp::Div, Expr::Sv(acc), f(n as f64)) } else { Expr::Sv(acc) };
    body.push(Stmt::Store { buf: 1, index: ivs(&ov), value });
    let kern = kernel(op.name(), dtype, ins, &out, BufKind::Real, &bd, nest(&ov, &dims(&out), body));
    done(kern, out)
}

fn mean_vjp(axis: Option<&str>, ins: &[Operand], dtype: Dtype) -> Result<Lowered> {
    let (ct, tmpl) = (tensor_in(ins, 0, "mean_vjp")?, tensor_in(ins, 1, "mean_vjp")?);
    let n = match axis {
        None => tmpl.axes.iter().map(|a| a.len).product::<usize>(),
        Some(ax) => tmpl.axes.iter().find(|x| x.name.as_deref() == Some(ax)).map(|x| x.len).unwrap_or(1),
    };
    let map = align(&ct.axes, &tmpl.axes).ok_or_else(|| format!("mean_vjp: cannot broadcast {} to {}", show(&ct.axes), show(&tmpl.axes)))?;
    let mut bd = B::new();
    let iv = bd.loops(&dims(&tmpl.axes));
    let ic: Vec<Expr> = map.iter().map(|p| Expr::Iv(iv[*p])).collect();
    let body = vec![Stmt::Store { buf: 2, index: ivs(&iv), value: bin(BinOp::Div, load(0, ic), f(n as f64)) }];
    let kern = kernel("mean_vjp", dtype, ins, &tmpl.axes, BufKind::Real, &bd, nest(&iv, &dims(&tmpl.axes), body));
    done(kern, tmpl.axes.clone())
}

/// Row-wise `max`, `Σ exp(x − max)`, then the normalized exponentials, over
/// axis `p` for every index of the other axes.
fn softmax_rows(bd: &mut B, buf: usize, axes: &[Axis], p: usize) -> (Vec<usize>, usize, usize, Vec<Stmt>, Vec<usize>) {
    let mut outer = axes.to_vec();
    let n = outer.remove(p).len;
    let ov = bd.loops(&dims(&outer));
    let k = bd.iv();
    let m = bd.sv();
    let z = bd.sv();
    let full = |i: Expr| -> Vec<Expr> {
        let mut v = ivs(&ov);
        v.insert(p, i);
        v
    };
    let stmts = vec![
        Stmt::Set { var: m, value: load(buf, full(Expr::I(0))) },
        Stmt::For { var: k, bound: n, body: vec![Stmt::If { cond: cmp(CmpOp::Gt, load(buf, full(Expr::Iv(k))), Expr::Sv(m)), then: vec![Stmt::Set { var: m, value: load(buf, full(Expr::Iv(k))) }] }] },
        Stmt::Set { var: z, value: f(0.0) },
        Stmt::For { var: k, bound: n, body: vec![Stmt::Set { var: z, value: bin(BinOp::Add, Expr::Sv(z), un(UnOp::Exp, bin(BinOp::Sub, load(buf, full(Expr::Iv(k))), Expr::Sv(m)))) }] },
    ];
    (ov, m, z, stmts, vec![k, n])
}

fn softmax(axis: &str, ins: &[Operand], dtype: Dtype) -> Result<Lowered> {
    let t = tensor_in(ins, 0, "softmax")?;
    let p = axis_pos(&t.axes, axis)?;
    let mut bd = B::new();
    let (ov, m, z, mut body, kn) = softmax_rows(&mut bd, 0, &t.axes, p);
    let (k, n) = (kn[0], kn[1]);
    let mut full = ivs(&ov);
    full.insert(p, Expr::Iv(k));
    body.push(Stmt::For { var: k, bound: n, body: vec![Stmt::Store { buf: 1, index: full.clone(), value: bin(BinOp::Div, un(UnOp::Exp, bin(BinOp::Sub, load(0, full), Expr::Sv(m))), Expr::Sv(z)) }] });
    let mut outer = t.axes.clone();
    outer.remove(p);
    // Over an axis of length 0 the softmax is defined, and empty: no row
    // has a first element to start its maximum from, and none is stored.
    let body = if n == 0 { vec![] } else { nest(&ov, &dims(&outer), body) };
    let kern = kernel(format!("softmax axis {axis}"), dtype, ins, &t.axes, BufKind::Real, &bd, body);
    done(kern, t.axes.clone())
}

/// Softmax in two passes over `x` (0.7.0 step 3): the row maximum `m`, then
/// one loop storing `e = exp(x − m)` into the output while summing `z`, then a
/// loop dividing the output by `z` in place. Each exponential is computed
/// once, where the three-pass form computes it twice; the values are the
/// same, rounded at the same places — `e` is stored in the storage type, which
/// is the arithmetic type wherever the layout is well-formed (condition 7),
/// and a backend may not contract.
fn softmax_two_pass(axis: &str, ins: &[Operand], dtype: Dtype) -> Result<Lowered> {
    let t = tensor_in(ins, 0, "softmax")?;
    let p = axis_pos(&t.axes, axis)?;
    let mut outer = t.axes.clone();
    let n = outer.remove(p).len;
    let mut bd = B::new();
    let ov = bd.loops(&dims(&outer));
    let k = bd.iv();
    let (m, z, e) = (bd.sv(), bd.sv(), bd.sv());
    let full = |i: Expr| -> Vec<Expr> {
        let mut v = ivs(&ov);
        v.insert(p, i);
        v
    };
    let at = full(Expr::Iv(k));
    let body = vec![
        Stmt::Set { var: m, value: load(0, full(Expr::I(0))) },
        Stmt::For { var: k, bound: n, body: vec![Stmt::If { cond: cmp(CmpOp::Gt, load(0, at.clone()), Expr::Sv(m)), then: vec![Stmt::Set { var: m, value: load(0, at.clone()) }] }] },
        Stmt::Set { var: z, value: f(0.0) },
        Stmt::For {
            var: k,
            bound: n,
            body: vec![
                Stmt::Set { var: e, value: un(UnOp::Exp, bin(BinOp::Sub, load(0, at.clone()), Expr::Sv(m))) },
                Stmt::Store { buf: 1, index: at.clone(), value: Expr::Sv(e) },
                Stmt::Set { var: z, value: bin(BinOp::Add, Expr::Sv(z), Expr::Sv(e)) },
            ],
        },
        Stmt::For { var: k, bound: n, body: vec![Stmt::Store { buf: 1, index: at.clone(), value: bin(BinOp::Div, load(1, at), Expr::Sv(z)) }] },
    ];
    let body = if n == 0 { vec![] } else { nest(&ov, &dims(&outer), body) };
    let kern = kernel(format!("softmax axis {axis}, two passes"), dtype, ins, &t.axes, BufKind::Real, &bd, body);
    done(kern, t.axes.clone())
}

/// The softmax over the positions a mask allows, 0 where it refuses (plan
/// 0.10.0 step 7b): input 0 the scores, input 1 the mask, indices 0 or 1 that
/// broadcast into the scores. Per row, the maximum `m` of the allowed scores
/// from the first allowed one on, a strict `>` after it, as the unmasked
/// form takes it from the row's first; then, in three passes, `z` the sum of
/// `exp(x − m)` over the allowed positions in order and each output
/// `exp(x − m) / z`, or in two, each allowed exponential stored and summed
/// and the row divided by `z`. A refused position adds 0 to `z` and gives 0:
/// the bits the unmasked softmax gives a row whose refused positions hold
/// `−∞`, the additive mask's, since `exp(−∞) = 0` and `z + 0 = z`. The mask
/// is read in selections only, never a branch around a store, so the nest
/// stays a static-control part (the SCoP mode). A row the mask empties, which
/// has no value, gives zeros.
fn masked_softmax(axis: &str, ins: &[Operand], dtype: Dtype, two_passes: bool) -> Result<Lowered> {
    let (t, m) = (tensor_in(ins, 0, "softmax")?, tensor_in(ins, 1, "softmax")?);
    let p = axis_pos(&t.axes, axis)?;
    let map = align(&m.axes, &t.axes).ok_or_else(|| format!("softmax: the mask {} does not broadcast into {}", show(&m.axes), show(&t.axes)))?;
    let mut outer = t.axes.clone();
    let n = outer.remove(p).len;
    let mut bd = B::new();
    let ov = bd.loops(&dims(&outer));
    let k = bd.iv();
    let (mx, z, first, e) = (bd.sv(), bd.sv(), bd.sv(), bd.sv());
    let full = |i: Expr| -> Vec<Expr> {
        let mut v = ivs(&ov);
        v.insert(p, i);
        v
    };
    let at = full(Expr::Iv(k));
    let x = || load(0, at.clone());
    let allowed = || cmp(CmpOp::Gt, Expr::ToF(Box::new(Expr::LoadI(1, map.iter().map(|q| at[*q].clone()).collect()))), f(0.5));
    let ex = || un(UnOp::Exp, bin(BinOp::Sub, x(), Expr::Sv(mx)));
    let mut body = vec![
        Stmt::Set { var: first, value: f(1.0) },
        Stmt::Set { var: mx, value: f(0.0) },
        Stmt::For {
            var: k,
            bound: n,
            body: vec![
                Stmt::Set { var: mx, value: select(allowed(), select(cmp(CmpOp::Gt, Expr::Sv(first), f(0.5)), x(), select(cmp(CmpOp::Gt, x(), Expr::Sv(mx)), x(), Expr::Sv(mx))), Expr::Sv(mx)) },
                Stmt::Set { var: first, value: select(allowed(), f(0.0), Expr::Sv(first)) },
            ],
        },
        Stmt::Set { var: z, value: f(0.0) },
    ];
    if two_passes {
        body.push(Stmt::For {
            var: k,
            bound: n,
            body: vec![
                Stmt::Set { var: e, value: select(allowed(), ex(), f(0.0)) },
                Stmt::Store { buf: 2, index: at.clone(), value: Expr::Sv(e) },
                Stmt::Set { var: z, value: bin(BinOp::Add, Expr::Sv(z), Expr::Sv(e)) },
            ],
        });
        body.push(Stmt::For { var: k, bound: n, body: vec![Stmt::Store { buf: 2, index: at.clone(), value: select(allowed(), bin(BinOp::Div, load(2, at.clone()), Expr::Sv(z)), f(0.0)) }] });
    } else {
        body.push(Stmt::For { var: k, bound: n, body: vec![Stmt::Set { var: z, value: bin(BinOp::Add, Expr::Sv(z), select(allowed(), ex(), f(0.0))) }] });
        body.push(Stmt::For { var: k, bound: n, body: vec![Stmt::Store { buf: 2, index: at.clone(), value: select(allowed(), bin(BinOp::Div, ex(), Expr::Sv(z)), f(0.0)) }] });
    }
    let body = if n == 0 { vec![] } else { nest(&ov, &dims(&outer), body) };
    let name = if two_passes { format!("softmax axis {axis} where, two passes") } else { format!("softmax axis {axis} where") };
    let kern = kernel(name, dtype, ins, &t.axes, BufKind::Real, &bd, body);
    done(kern, t.axes.clone())
}

fn embed(ins: &[Operand], dtype: Dtype) -> Result<Lowered> {
    let (table, idx) = (tensor_in(ins, 0, "embed")?, tensor_in(ins, 1, "embed")?);
    if table.axes.len() != 2 {
        return Err("embed: the table must have rank 2".into());
    }
    let d = table.axes[1].len;
    let mut out = idx.axes.clone();
    out.push(Axis { name: None, len: d });
    let mut bd = B::new();
    let iv = bd.loops(&dims(&idx.axes));
    let j = bd.iv();
    let mut oix = ivs(&iv);
    oix.push(Expr::Iv(j));
    let body = vec![Stmt::For { var: j, bound: d, body: vec![Stmt::Store { buf: 2, index: oix, value: load(0, vec![Expr::LoadI(1, ivs(&iv)), Expr::Iv(j)]) }] }];
    let kern = kernel("embed", dtype, ins, &out, BufKind::Real, &bd, nest(&iv, &dims(&idx.axes), body));
    done(kern, out)
}

fn zero_fill(bd: &mut B, buf: usize, axes: &[Axis]) -> Vec<Stmt> {
    let iv = bd.loops(&dims(axes));
    nest(&iv, &dims(axes), vec![Stmt::Store { buf, index: ivs(&iv), value: f(0.0) }])
}

fn embed_vjp(ins: &[Operand], dtype: Dtype) -> Result<Lowered> {
    let (table, idx, _ct) = (tensor_in(ins, 0, "embed_vjp")?, tensor_in(ins, 1, "embed_vjp")?, tensor_in(ins, 2, "embed_vjp")?);
    let d = table.axes[1].len;
    let mut bd = B::new();
    let mut body = zero_fill(&mut bd, 3, &table.axes);
    let iv = bd.loops(&dims(&idx.axes));
    let j = bd.iv();
    let mut cix = ivs(&iv);
    cix.push(Expr::Iv(j));
    let oix = vec![Expr::LoadI(1, ivs(&iv)), Expr::Iv(j)];
    body.extend(nest(&iv, &dims(&idx.axes), vec![Stmt::For { var: j, bound: d, body: vec![Stmt::Store { buf: 3, index: oix.clone(), value: bin(BinOp::Add, load(3, oix), load(2, cix)) }] }]));
    let kern = kernel("embed_vjp", dtype, ins, &table.axes, BufKind::Real, &bd, body);
    done(kern, table.axes.clone())
}

fn cross_entropy(ins: &[Operand], dtype: Dtype) -> Result<Lowered> {
    let (logits, y) = (tensor_in(ins, 0, "cross_entropy")?, tensor_in(ins, 1, "cross_entropy")?);
    let p = logits.axes.len() - 1;
    let mut bd = B::new();
    let (ov, m, z, mut body, _) = softmax_rows(&mut bd, 0, &logits.axes, p);
    let mut at_y = ivs(&ov);
    at_y.push(Expr::LoadI(1, ivs(&ov)));
    // −ln softmax(x)[y] = ln z + m − x[y]
    body.push(Stmt::Store { buf: 2, index: ivs(&ov), value: bin(BinOp::Sub, bin(BinOp::Add, un(UnOp::Log, Expr::Sv(z)), Expr::Sv(m)), load(0, at_y)) });
    let kern = kernel("cross_entropy", dtype, ins, &y.axes, BufKind::Real, &bd, nest(&ov, &dims(&y.axes), body));
    done(kern, y.axes.clone())
}

fn cross_entropy_vjp(ins: &[Operand], dtype: Dtype) -> Result<Lowered> {
    let (logits, _y, _ct) = (tensor_in(ins, 0, "cross_entropy_vjp")?, tensor_in(ins, 1, "cross_entropy_vjp")?, tensor_in(ins, 2, "cross_entropy_vjp")?);
    let p = logits.axes.len() - 1;
    let v = logits.axes[p].len;
    let mut bd = B::new();
    let (ov, m, z, mut body, _) = softmax_rows(&mut bd, 0, &logits.axes, p);
    let k = bd.iv();
    let mut full = ivs(&ov);
    full.push(Expr::Iv(k));
    let onehot = select(cmp(CmpOp::Eq, Expr::ToF(Box::new(Expr::Iv(k))), Expr::ToF(Box::new(Expr::LoadI(1, ivs(&ov))))), f(1.0), f(0.0));
    let sm = bin(BinOp::Div, un(UnOp::Exp, bin(BinOp::Sub, load(0, full.clone()), Expr::Sv(m))), Expr::Sv(z));
    body.push(Stmt::For { var: k, bound: v, body: vec![Stmt::Store { buf: 3, index: full, value: bin(BinOp::Mul, bin(BinOp::Sub, sm, onehot), load(2, ivs(&ov))) }] });
    let mut outer = logits.axes.clone();
    outer.remove(p);
    let kern = kernel("cross_entropy_vjp", dtype, ins, &logits.axes, BufKind::Real, &bd, nest(&ov, &dims(&outer), body));
    done(kern, logits.axes.clone())
}

fn argmax(ins: &[Operand], dtype: Dtype) -> Result<Lowered> {
    let t = tensor_in(ins, 0, "argmax")?;
    if t.axes.is_empty() {
        return Err("argmax: scalar argument".into());
    }
    let p = t.axes.len() - 1;
    let v = t.axes[p].len;
    let out = t.axes[..p].to_vec();
    let mut bd = B::new();
    let ov = bd.loops(&dims(&out));
    let k = bd.iv();
    let m = bd.sv();
    let best = bd.si();
    let mut full = ivs(&ov);
    full.push(Expr::Iv(k));
    let mut first = ivs(&ov);
    first.push(Expr::I(0));
    let body = vec![
        Stmt::Set { var: m, value: load(0, first) },
        Stmt::SetI { var: best, value: Expr::I(0) },
        // the interpreter's `max_by` keeps the last of equal maxima
        Stmt::For { var: k, bound: v, body: vec![Stmt::If { cond: cmp(CmpOp::Ge, load(0, full.clone()), Expr::Sv(m)), then: vec![Stmt::Set { var: m, value: load(0, full) }, Stmt::SetI { var: best, value: Expr::Iv(k) }] }] },
        Stmt::StoreI { buf: 1, index: ivs(&ov), value: Expr::Si(best) },
    ];
    let kern = kernel("argmax", dtype, ins, &out, BufKind::Idx, &bd, nest(&ov, &dims(&out), body));
    done_idx(kern, out, v as u64)
}

fn index_const(axis: &str, index: i64, ins: &[Operand], dtype: Dtype) -> Result<Lowered> {
    let t = tensor_in(ins, 0, "index")?;
    let p = axis_pos(&t.axes, axis)?;
    let len = t.axes[p].len as i64;
    let i = if index < 0 { len + index } else { index };
    if i < 0 || i >= len {
        return Err(format!("index {index} out of range for axis `{axis}` of length {len}"));
    }
    let mut out = t.axes.clone();
    out.remove(p);
    let mut bd = B::new();
    let ov = bd.loops(&dims(&out));
    let mut full = ivs(&ov);
    full.insert(p, Expr::I(i));
    let body = vec![if t.kind == BufKind::Idx { Stmt::StoreI { buf: 1, index: ivs(&ov), value: Expr::LoadI(0, full) } } else { Stmt::Store { buf: 1, index: ivs(&ov), value: load(0, full) } }];
    let kern = kernel(format!("index {axis}: {index}"), dtype, ins, &out, t.kind, &bd, nest(&ov, &dims(&out), body));
    Ok(Lowered { kernel: kern, out_axes: out, out_kind: t.kind, out_bound: 0 })
}

fn scatter_index(axis: &str, index: i64, ins: &[Operand], dtype: Dtype) -> Result<Lowered> {
    let (tmpl, ct) = (tensor_in(ins, 0, "scatter_index")?, tensor_in(ins, 1, "scatter_index")?);
    let p = axis_pos(&tmpl.axes, axis)?;
    let len = tmpl.axes[p].len as i64;
    let i = if index < 0 { len + index } else { index };
    let mut bd = B::new();
    let mut body = zero_fill(&mut bd, 2, &tmpl.axes);
    let cv = bd.loops(&dims(&ct.axes));
    let mut full = ivs(&cv);
    full.insert(p, Expr::I(i));
    body.extend(nest(&cv, &dims(&ct.axes), vec![Stmt::Store { buf: 2, index: full, value: load(1, ivs(&cv)) }]));
    let kern = kernel(format!("scatter_index {axis}: {index}"), dtype, ins, &tmpl.axes, BufKind::Real, &bd, body);
    done(kern, tmpl.axes.clone())
}

/// `t[axis: lo..hi]`: a copy of the elements `lo` to `hi − 1` of the
/// axis, which the result keeps; for a real and an index tensor.
fn slice(axis: &str, lo: usize, hi: usize, ins: &[Operand], dtype: Dtype) -> Result<Lowered> {
    let t = tensor_in(ins, 0, "slice")?;
    let p = axis_pos(&t.axes, axis)?;
    if lo > hi || hi > t.axes[p].len {
        return Err(format!("slice {lo}..{hi} outside the axis `{axis}`, of length {}", t.axes[p].len));
    }
    let mut out = t.axes.clone();
    out[p].len = hi - lo;
    let mut bd = B::new();
    let ov = bd.loops(&dims(&out));
    let mut src = ivs(&ov);
    src[p] = bin(BinOp::IAdd, Expr::Iv(ov[p]), Expr::I(lo as i64));
    let body = vec![if t.kind == BufKind::Idx { Stmt::StoreI { buf: 1, index: ivs(&ov), value: Expr::LoadI(0, src) } } else { Stmt::Store { buf: 1, index: ivs(&ov), value: load(0, src) } }];
    let kind = if t.kind == BufKind::Idx { BufKind::Idx } else { BufKind::Real };
    let kern = kernel(format!("slice {axis}: {lo}..{hi}"), dtype, ins, &out, kind, &bd, nest(&ov, &dims(&out), body));
    Ok(Lowered { kernel: kern, out_axes: out, out_kind: kind, out_bound: 0 })
}

/// `(template, ct)`: zeros of the template's shape, `ct` at the elements
/// `lo` to `hi − 1` of the axis.
fn slice_vjp(axis: &str, lo: usize, hi: usize, ins: &[Operand], dtype: Dtype) -> Result<Lowered> {
    let (tmpl, ct) = (tensor_in(ins, 0, "slice_vjp")?, tensor_in(ins, 1, "slice_vjp")?);
    let p = axis_pos(&tmpl.axes, axis)?;
    if ct.axes.len() != tmpl.axes.len() || ct.axes[p].len != hi - lo {
        return Err(format!("slice_vjp {axis}: {lo}..{hi}: a cotangent of {} into {}", show(&ct.axes), show(&tmpl.axes)));
    }
    let mut bd = B::new();
    let mut body = zero_fill(&mut bd, 2, &tmpl.axes);
    let cv = bd.loops(&dims(&ct.axes));
    let mut dst = ivs(&cv);
    dst[p] = bin(BinOp::IAdd, Expr::Iv(cv[p]), Expr::I(lo as i64));
    body.extend(nest(&cv, &dims(&ct.axes), vec![Stmt::Store { buf: 2, index: dst, value: load(1, ivs(&cv)) }]));
    let kern = kernel(format!("slice_vjp {axis}: {lo}..{hi}"), dtype, ins, &tmpl.axes, BufKind::Real, &bd, body);
    done(kern, tmpl.axes.clone())
}

/// `concat(axis a, x, y)`: `x` into the first part of the result along
/// the axis, `y` into the rest, each element written once.
fn concat(axis: &str, ins: &[Operand], dtype: Dtype) -> Result<Lowered> {
    let (x, y) = (tensor_in(ins, 0, "concat")?, tensor_in(ins, 1, "concat")?);
    let (p, q) = (axis_pos(&x.axes, axis)?, axis_pos(&y.axes, axis)?);
    let same = p == q && x.axes.len() == y.axes.len() && x.axes.iter().zip(&y.axes).enumerate().all(|(k, (u, v))| k == p || u.len == v.len);
    if !same {
        return Err(format!("concat {axis}: {} and {}", show(&x.axes), show(&y.axes)));
    }
    let m = x.axes[p].len;
    let mut out = x.axes.clone();
    out[p].len = m + y.axes[p].len;
    let mut bd = B::new();
    let xv = bd.loops(&dims(&x.axes));
    let mut body = nest(&xv, &dims(&x.axes), vec![Stmt::Store { buf: 2, index: ivs(&xv), value: load(0, ivs(&xv)) }]);
    let yv = bd.loops(&dims(&y.axes));
    let mut dst = ivs(&yv);
    dst[p] = bin(BinOp::IAdd, Expr::Iv(yv[p]), Expr::I(m as i64));
    body.extend(nest(&yv, &dims(&y.axes), vec![Stmt::Store { buf: 2, index: dst, value: load(1, ivs(&yv)) }]));
    let kern = kernel(format!("concat {axis}"), dtype, ins, &out, BufKind::Real, &bd, body);
    done(kern, out)
}

/// `[..., seq, model] → [..., heads, seq, hd]`.
fn split_heads(h: usize, hd: usize, ins: &[Operand], dtype: Dtype) -> Result<Lowered> {
    let t = tensor_in(ins, 0, "split_heads")?;
    let r = t.axes.len();
    if r < 2 || t.axes[r - 1].len != h * hd {
        return Err(format!("split_heads: expected a trailing axis of length {}", h * hd));
    }
    let lead = &t.axes[..r - 2];
    let seq = t.axes[r - 2].clone();
    let mut out = lead.to_vec();
    out.push(Axis { name: Some("heads".into()), len: h });
    out.push(seq);
    out.push(Axis { name: Some("hd".into()), len: hd });
    let mut bd = B::new();
    let ov = bd.loops(&dims(&out));
    let n = ov.len();
    // out[lead…, hh, s, d] = in[lead…, s, hh·hd + d]
    let mut ix: Vec<Expr> = ov[..n - 3].iter().map(|v| Expr::Iv(*v)).collect();
    ix.push(Expr::Iv(ov[n - 2]));
    ix.push(bin(BinOp::IAdd, bin(BinOp::IMul, Expr::Iv(ov[n - 3]), Expr::I(hd as i64)), Expr::Iv(ov[n - 1])));
    let body = vec![Stmt::Store { buf: 1, index: ivs(&ov), value: load(0, ix) }];
    let kern = kernel(format!("split_heads {h}x{hd}"), dtype, ins, &out, BufKind::Real, &bd, nest(&ov, &dims(&out), body));
    done(kern, out)
}

/// `[..., heads, seq, hd] → [..., seq, model]`.
fn merge_heads(h: usize, hd: usize, ins: &[Operand], dtype: Dtype) -> Result<Lowered> {
    let t = tensor_in(ins, 0, "merge_heads")?;
    let r = t.axes.len();
    if r < 3 {
        return Err("merge_heads: expected [..., heads, seq, hd]".into());
    }
    let lead = &t.axes[..r - 3];
    let seq = t.axes[r - 2].clone();
    let mut out = lead.to_vec();
    out.push(seq);
    out.push(Axis { name: Some("model".into()), len: h * hd });
    let mut bd = B::new();
    let ov = bd.loops(&dims(&out));
    let n = ov.len();
    let hh = bd.iv();
    let d = bd.iv();
    // out[lead…, s, hh·hd + d] = in[lead…, hh, s, d]
    let mut oix: Vec<Expr> = ov[..n - 1].iter().map(|v| Expr::Iv(*v)).collect();
    oix.push(bin(BinOp::IAdd, bin(BinOp::IMul, Expr::Iv(hh), Expr::I(hd as i64)), Expr::Iv(d)));
    let mut ix: Vec<Expr> = ov[..n - 2].iter().map(|v| Expr::Iv(*v)).collect();
    ix.push(Expr::Iv(hh));
    ix.push(Expr::Iv(ov[n - 2]));
    ix.push(Expr::Iv(d));
    let outer_dims: Vec<usize> = dims(&out[..n - 1]);
    let outer_v: Vec<usize> = ov[..n - 1].to_vec();
    let body = nest(&outer_v, &outer_dims, vec![Stmt::For { var: hh, bound: h, body: vec![Stmt::For { var: d, bound: hd, body: vec![Stmt::Store { buf: 1, index: oix, value: load(0, ix) }] }] }]);
    let kern = kernel(format!("merge_heads {h}x{hd}"), dtype, ins, &out, BufKind::Real, &bd, body);
    done(kern, out)
}

/// Sum `x` down to the template's axes (adjoint of broadcasting).
fn sum_to(ins: &[Operand], dtype: Dtype) -> Result<Lowered> {
    let (x, tmpl) = (tensor_in(ins, 0, "sum_to")?, tensor_in(ins, 1, "sum_to")?);
    let map = align(&tmpl.axes, &x.axes).ok_or_else(|| format!("sum_to: cannot reduce {} to {}", show(&x.axes), show(&tmpl.axes)))?;
    let mut bd = B::new();
    let mut body = zero_fill(&mut bd, 2, &tmpl.axes);
    let iv = bd.loops(&dims(&x.axes));
    let oix: Vec<Expr> = map.iter().map(|p| Expr::Iv(iv[*p])).collect();
    body.extend(nest(&iv, &dims(&x.axes), vec![Stmt::Store { buf: 2, index: oix.clone(), value: bin(BinOp::Add, load(2, oix), load(0, ivs(&iv))) }]));
    let kern = kernel("sum_to", dtype, ins, &tmpl.axes, BufKind::Real, &bd, body);
    done(kern, tmpl.axes.clone())
}

fn broadcast_like(ins: &[Operand], dtype: Dtype) -> Result<Lowered> {
    let (x, tmpl) = (tensor_in(ins, 0, "broadcast_like")?, tensor_in(ins, 1, "broadcast_like")?);
    let map = align(&x.axes, &tmpl.axes).ok_or_else(|| format!("broadcast_like: cannot broadcast {} to {}", show(&x.axes), show(&tmpl.axes)))?;
    let mut bd = B::new();
    let iv = bd.loops(&dims(&tmpl.axes));
    let ix: Vec<Expr> = map.iter().map(|p| Expr::Iv(iv[*p])).collect();
    let body = vec![Stmt::Store { buf: 2, index: ivs(&iv), value: load(0, ix) }];
    let kern = kernel("broadcast_like", dtype, ins, &tmpl.axes, BufKind::Real, &bd, nest(&iv, &dims(&tmpl.axes), body));
    done(kern, tmpl.axes.clone())
}

fn sum_squares(ins: &[Operand], dtype: Dtype) -> Result<Lowered> {
    let t = tensor_in(ins, 0, "sum_squares")?;
    let mut bd = B::new();
    let iv = bd.loops(&dims(&t.axes));
    let acc = bd.sv();
    let mut body = vec![Stmt::Set { var: acc, value: f(0.0) }];
    body.extend(nest(&iv, &dims(&t.axes), vec![Stmt::Set { var: acc, value: bin(BinOp::Add, Expr::Sv(acc), bin(BinOp::Mul, load(0, ivs(&iv)), load(0, ivs(&iv)))) }]));
    body.push(Stmt::Store { buf: 1, index: vec![], value: Expr::Sv(acc) });
    let kern = kernel("sum_squares", dtype, ins, &[], BufKind::Real, &bd, body);
    done(kern, vec![])
}

/// `positions`: `0..n` along `axis`.
pub fn positions(axis: &str, n: usize, dtype: Dtype) -> Lowered {
    let out = vec![Axis { name: Some(axis.to_string()), len: n }];
    let mut bd = B::new();
    let i = bd.iv();
    let body = vec![Stmt::For { var: i, bound: n, body: vec![Stmt::StoreI { buf: 0, index: vec![Expr::Iv(i)], value: Expr::Iv(i) }] }];
    let kern = kernel(format!("positions {axis}"), dtype, &[], &out, BufKind::Idx, &bd, body);
    Lowered { kernel: kern, out_axes: out, out_kind: BufKind::Idx, out_bound: n as u64 }
}

/// `causal_mask`: a mask, 1 (true) on `j ≤ i` and 0 elsewhere, indices below
/// 2 (plan 0.10.0 step 7b; it was 0 and −∞, a real tensor added to the
/// scores).
pub fn causal_mask(axis: &str, n: usize, dtype: Dtype) -> Lowered {
    let out = vec![Axis { name: Some(axis.to_string()), len: n }, Axis { name: Some(format!("{axis}2")), len: n }];
    let mut bd = B::new();
    let (i, j) = (bd.iv(), bd.iv());
    let below = select(cmp(CmpOp::Le, Expr::ToF(Box::new(Expr::Iv(j))), Expr::ToF(Box::new(Expr::Iv(i)))), f(1.0), f(0.0));
    let body = nest(&[i, j], &[n, n], vec![Stmt::StoreI { buf: 0, index: vec![Expr::Iv(i), Expr::Iv(j)], value: Expr::ToI(Box::new(below)) }]);
    let kern = kernel(format!("causal_mask {axis}"), dtype, &[], &out, BufKind::Idx, &bd, body);
    Lowered { kernel: kern, out_axes: out, out_kind: BufKind::Idx, out_bound: 2 }
}

/// `one_hot`: 1 where the index equals the position on the new trailing axis.
/// `out` is the result's axes; the index tensor carries all but the last.
pub fn one_hot(out: &[Axis], dtype: Dtype) -> Lowered {
    let k = out.len() - 1;
    let (idx_axes, v) = (&out[..k], out[k].len);
    let ins = vec![Operand { axes: idx_axes.to_vec(), kind: BufKind::Idx }];
    let mut bd = B::new();
    let iv = bd.loops(&dims(idx_axes));
    let j = bd.iv();
    let mut oix = ivs(&iv);
    oix.push(Expr::Iv(j));
    let hit = cmp(CmpOp::Eq, Expr::ToF(Box::new(Expr::LoadI(0, ivs(&iv)))), Expr::ToF(Box::new(Expr::Iv(j))));
    let body = nest(&iv, &dims(idx_axes), vec![Stmt::For { var: j, bound: v, body: vec![Stmt::Store { buf: 1, index: oix, value: select(hit, f(1.0), f(0.0)) }] }]);
    let kern = kernel("one_hot", dtype, &ins, out, BufKind::Real, &bd, body);
    Lowered { kernel: kern, out_axes: out.to_vec(), out_kind: BufKind::Real, out_bound: 0 }
}

/// A constant fill (`zeros`, `ones`).
pub fn fill(name: &str, axes: &[Axis], value: f64, dtype: Dtype) -> Lowered {
    let mut bd = B::new();
    let iv = bd.loops(&dims(axes));
    let body = nest(&iv, &dims(axes), vec![Stmt::Store { buf: 0, index: ivs(&iv), value: f(value) }]);
    let kern = kernel(name, dtype, &[], axes, BufKind::Real, &bd, body);
    Lowered { kernel: kern, out_axes: axes.to_vec(), out_kind: BufKind::Real, out_bound: 0 }
}

pub fn show(axes: &[Axis]) -> String {
    let parts: Vec<String> = axes.iter().map(|a| match &a.name {
        Some(n) => format!("{n}: {}", a.len),
        None => a.len.to_string(),
    }).collect();
    format!("[{}]", parts.join(", "))
}
