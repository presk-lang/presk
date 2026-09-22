//! From a PSK-Core tensor primitive at concrete operand shapes to a
//! [`Kernel`]: the loop nest that the reference interpreter's semantics
//! prescribe (its alignment rules for broadcasting, its contraction axes, its
//! adjoint primitives), with one kernel per primitive instance.

use presk_core::PrimOp;
use presk_sched::{Src, Step};
use presk_rt::Axis;

use presk_rt::ReductionOrder;
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
    Kernel { name: name.into(), dtype, inputs: ins.iter().map(|o| spec(&o.axes, o.kind)).collect(), output: spec(out_axes, out_kind), nloops: b.nloops, nreal: b.nreal, nint: b.nint, nvec: b.nvec, lanes: b.lanes, body }
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
    lower_vec(op, ins, dtype, order, 1)
}

/// The same, emitting `lanes`-wide vectors where the shape allows (0.6.0
/// step 1). Lanes are independent output elements, so this is exact.
pub fn lower_vec(op: &PrimOp, ins: &[Operand], dtype: Dtype, order: ReductionOrder, lanes: usize) -> Option<Result<Lowered>> {
    Some(match op {
        PrimOp::Add | PrimOp::Sub | PrimOp::Mul | PrimOp::Div => elementwise_binary(op, ins, dtype),
        PrimOp::Neg | PrimOp::Exp | PrimOp::Log | PrimOp::Tanh | PrimOp::Sqrt | PrimOp::Sin | PrimOp::Cos | PrimOp::Abs | PrimOp::Gelu | PrimOp::GeluDeriv | PrimOp::Relu | PrimOp::Heaviside | PrimOp::Sign | PrimOp::Sigmoid | PrimOp::Pow(_) => unary(op, ins, dtype),
        PrimOp::Real if ins.len() == 1 && ins[0].kind == BufKind::Idx => unary(op, ins, dtype),
        PrimOp::Dot | PrimOp::Matmul => dot(ins, dtype, order, lanes),
        PrimOp::DotVjpL => dot_vjp_l(ins, dtype),
        PrimOp::DotVjpR => dot_vjp_r(ins, dtype),
        PrimOp::Ein { operands, output } => ein(ins, operands, output, dtype),
        PrimOp::SumAll | PrimOp::MeanAll => reduce_all(op, ins, dtype),
        PrimOp::Sum { axis } | PrimOp::Mean { axis } => reduce_axis(op, axis, ins, dtype),
        PrimOp::MeanVjp { axis } => mean_vjp(axis.as_deref(), ins, dtype),
        PrimOp::Softmax { axis } => softmax(axis, ins, dtype),
        PrimOp::Embed => embed(ins, dtype),
        PrimOp::EmbedVjp => embed_vjp(ins, dtype),
        PrimOp::CrossEntropy => cross_entropy(ins, dtype),
        PrimOp::CrossEntropyVjp => cross_entropy_vjp(ins, dtype),
        PrimOp::Argmax => argmax(ins, dtype),
        PrimOp::IndexConst { axis, index } => index_const(axis, *index, ins, dtype),
        PrimOp::ScatterIndex { axis, index } => scatter_index(axis, *index, ins, dtype),
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
    let k = match op {
        PrimOp::Add => BinOp::Add,
        PrimOp::Sub => BinOp::Sub,
        PrimOp::Mul => BinOp::Mul,
        _ => BinOp::Div,
    };
    let body = vec![Stmt::Store { buf: 2, index: ivs(&iv), value: bin(k, load(0, ia), load(1, ib)) }];
    let kern = kernel(op.name(), dtype, ins, &out, BufKind::Real, &bd, nest(&iv, &dims(&out), body));
    done(kern, out)
}

/// One loop nest for a fusion group (`presk-sched`): each input is read
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
                    BufKind::Real | BufKind::Quant { .. } => load(*k, idx[*k].clone()),
                },
                Src::Step(k) => vals[*k].clone(),
            })
            .collect();
        let value = match a.as_slice() {
            [x] => unary_expr(&s.op, x.clone()).ok_or_else(|| format!("fused: `{}` is not a unary primitive", s.op.name()))?,
            [x, y] => {
                let k = match s.op {
                    PrimOp::Add => BinOp::Add,
                    PrimOp::Sub => BinOp::Sub,
                    PrimOp::Mul => BinOp::Mul,
                    PrimOp::Div => BinOp::Div,
                    _ => return Err(format!("fused: `{}` is not a binary primitive", s.op.name())),
                };
                bin(k, x.clone(), y.clone())
            }
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

/// Columns of the result held in registers at once by the blocked
/// contraction. Eight fits the register files this targets and divides the
/// widths the corpus uses; a width that is not a multiple falls back to the
/// row-wise form, which handles any.
const GEMM_TILE: usize = 8;

/// `x[..., K] · w[K, rest…]`.
fn dot(ins: &[Operand], dtype: Dtype, order: ReductionOrder, lanes: usize) -> Result<Lowered> {
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
    // Row-wise form when the last output axis comes from `w`: hold one element
    // of `x` and sweep a *row* of `w` and of the result, both unit-stride,
    // instead of walking a column of `w` for each output element. Each output
    // accumulates `k` in the same increasing order as the column form, so the
    // additions happen in the same order and the result is bit-identical; only
    // the memory it touches changes.
    // Register-blocked form: hold a tile of the result in scalars across the
    // whole contraction, so the inner loop loads one element of `x` and a
    // contiguous run of `w` and stores nothing. Each accumulator still sums
    // `k` in increasing order, so this is bit-identical to the column form —
    // what changes is where the partial sums live, not their order.
    // The blocked and row-wise forms below are exact: every output still
    // accumulates `k` in increasing order (D32). An interleaved order asks
    // for a different sum and takes the contraction at the end instead.
    let exact = order.is_exact();
    if exact && w.axes.len() >= 2 && outer.len() >= 2 && *outer.last().unwrap() % GEMM_TILE == 0 {
        let (opre, jn) = (outer[..outer.len() - 1].to_vec(), *outer.last().unwrap());
        let mut bd = B::new();
        let ov = bd.loops(&opre);
        let (jb, kv) = (bd.iv(), bd.iv());
        // Lanes, when asked for. Each lane is a different output column, so
        // every output still accumulates `k` in increasing order and the
        // vector kernel is **bit-identical** to the scalar one (D42) — this
        // is a layout change, not a reassociation, and it is why it owes no
        // realization and no gap.
        if lanes > 1 && GEMM_TILE % lanes == 0 {
            let groups = GEMM_TILE / lanes;
            let accs: Vec<usize> = (0..groups).map(|_| bd.vv(lanes)).collect();
            let col0 = |g: usize| bin(BinOp::IAdd, bin(BinOp::IMul, Expr::Iv(jb), Expr::I(GEMM_TILE as i64)), Expr::I((g * lanes) as i64));
            let xs = bd.sv();
            let mut xix: Vec<Expr> = ov[..lead].iter().map(|v| Expr::Iv(*v)).collect();
            xix.push(Expr::Iv(kv));
            let mut inner = vec![Stmt::Set { var: xs, value: load(0, xix) }];
            for (g, a) in accs.iter().enumerate() {
                let mut wix = vec![Expr::Iv(kv)];
                wix.extend(ov[lead..].iter().map(|v| Expr::Iv(*v)));
                wix.push(col0(g));
                let prod = Expr::VBin(BinOp::Mul, Box::new(Expr::VSplat(Box::new(Expr::Sv(xs)))), Box::new(Expr::VLoad(1, wix)));
                inner.push(Stmt::VSet { var: *a, value: Expr::VBin(BinOp::Add, Box::new(Expr::Vv(*a)), Box::new(prod)) });
            }
            let mut body: Vec<Stmt> = accs.iter().map(|a| Stmt::VSet { var: *a, value: Expr::VSplat(Box::new(f(0.0))) }).collect();
            body.push(Stmt::For { var: kv, bound: k, body: inner });
            for (g, a) in accs.iter().enumerate() {
                let mut cix: Vec<Expr> = ov.iter().map(|v| Expr::Iv(*v)).collect();
                cix.push(col0(g));
                body.push(Stmt::VStore { buf: 2, index: cix, value: Expr::Vv(*a) });
            }
            let tiles = Stmt::For { var: jb, bound: jn / GEMM_TILE, body };
            let kern = kernel("dot", dtype, ins, &out, BufKind::Real, &bd, nest(&ov, &opre, vec![tiles]));
            return done(kern, out);
        }
        let acc: Vec<usize> = (0..GEMM_TILE).map(|_| bd.sv()).collect();
        // `jb · TILE + t`, the column this accumulator owns.
        let col = |t: usize| bin(BinOp::IAdd, bin(BinOp::IMul, Expr::Iv(jb), Expr::I(GEMM_TILE as i64)), Expr::I(t as i64));
        let mut xix: Vec<Expr> = ov[..lead].iter().map(|v| Expr::Iv(*v)).collect();
        xix.push(Expr::Iv(kv));
        let xs = bd.sv();
        let mut inner = vec![Stmt::Set { var: xs, value: load(0, xix) }];
        for (t, a) in acc.iter().enumerate() {
            let mut wix = vec![Expr::Iv(kv)];
            wix.extend(ov[lead..].iter().map(|v| Expr::Iv(*v)));
            wix.push(col(t));
            inner.push(Stmt::Set { var: *a, value: bin(BinOp::Add, Expr::Sv(*a), bin(BinOp::Mul, Expr::Sv(xs), load(1, wix))) });
        }
        let mut body: Vec<Stmt> = acc.iter().map(|a| Stmt::Set { var: *a, value: f(0.0) }).collect();
        body.push(Stmt::For { var: kv, bound: k, body: inner });
        for (t, a) in acc.iter().enumerate() {
            let mut cix: Vec<Expr> = ov.iter().map(|v| Expr::Iv(*v)).collect();
            cix.push(col(t));
            body.push(Stmt::Store { buf: 2, index: cix, value: Expr::Sv(*a) });
        }
        let tiles = Stmt::For { var: jb, bound: jn / GEMM_TILE, body };
        let kern = kernel("dot", dtype, ins, &out, BufKind::Real, &bd, nest(&ov, &opre, vec![tiles]));
        return done(kern, out);
    }
    if exact && w.axes.len() >= 2 && outer.len() >= 2 {
        let (opre, jn) = (outer[..outer.len() - 1].to_vec(), *outer.last().unwrap());
        let mut bd = B::new();
        let ov = bd.loops(&opre);
        let (kv, jv, acc) = (bd.iv(), bd.iv(), bd.sv());
        let cix = |j: usize| -> Vec<Expr> {
            let mut ix: Vec<Expr> = ov.iter().map(|v| Expr::Iv(*v)).collect();
            ix.push(Expr::Iv(j));
            ix
        };
        let zero = Stmt::For { var: jv, bound: jn, body: vec![Stmt::Store { buf: 2, index: cix(jv), value: f(0.0) }] };
        let mut xix: Vec<Expr> = ov[..lead].iter().map(|v| Expr::Iv(*v)).collect();
        xix.push(Expr::Iv(kv));
        let mut wix = vec![Expr::Iv(kv)];
        wix.extend(ov[lead..].iter().map(|v| Expr::Iv(*v)));
        wix.push(Expr::Iv(jv));
        let sweep = Stmt::For {
            var: jv,
            bound: jn,
            body: vec![Stmt::Store { buf: 2, index: cix(jv), value: bin(BinOp::Add, load(2, cix(jv)), bin(BinOp::Mul, Expr::Sv(acc), load(1, wix))) }],
        };
        let over_k = Stmt::For { var: kv, bound: k, body: vec![Stmt::Set { var: acc, value: load(0, xix) }, sweep] };
        let kern = kernel("dot", dtype, ins, &out, BufKind::Real, &bd, nest(&ov, &opre, vec![zero, over_k]));
        return done(kern, out);
    }
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

/// `ct[..., N…] · b[K, N…] → a[..., K]` (`k = rank(b) − 1` trailing axes of `ct` contracted).
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

/// `a[..., K], ct[..., N…] → b[K, N…]`, summing over the `rank(a) − 1` leading axes.
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
    let kern = kernel(format!("softmax axis {axis}"), dtype, ins, &t.axes, BufKind::Real, &bd, nest(&ov, &dims(&outer), body));
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

/// `causal_mask`: 0 on `j ≤ i`, −∞ elsewhere.
pub fn causal_mask(axis: &str, n: usize, dtype: Dtype) -> Lowered {
    let out = vec![Axis { name: Some(axis.to_string()), len: n }, Axis { name: Some(format!("{axis}2")), len: n }];
    let mut bd = B::new();
    let (i, j) = (bd.iv(), bd.iv());
    let body = nest(&[i, j], &[n, n], vec![Stmt::Store { buf: 0, index: vec![Expr::Iv(i), Expr::Iv(j)], value: select(cmp(CmpOp::Le, Expr::ToF(Box::new(Expr::Iv(j))), Expr::ToF(Box::new(Expr::Iv(i)))), f(0.0), f(f64::NEG_INFINITY)) }]);
    let kern = kernel(format!("causal_mask {axis}"), dtype, &[], &out, BufKind::Real, &bd, body);
    Lowered { kernel: kern, out_axes: out, out_kind: BufKind::Real, out_bound: 0 }
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
