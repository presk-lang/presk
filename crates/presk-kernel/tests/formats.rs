//! What a nest computes in (plan 0.18.0 step 3, D206): a kernel carries its
//! arithmetic, product and accumulation formats, and the kernel interpreter
//! rounds each operation once to its own. Every operation of a kernel of
//! several formats is held to its exact result rounded once, computed apart
//! with MPFR; the path of several formats gives the one-format path's bits
//! where the formats are one; a sum's products and additions round to their
//! formats, and a chunked order sums in chunks.

use std::cmp::Ordering;

use presk_core::PrimOp;
use presk_format::{Float, Format, Tie, BF16, E4M3FN, E4M3_SATURATING, E5M2, E8M15, F16, F32, F64};
use presk_kernel::exec::{run_emulated, run_kernel, run_kernel_in, run_kernel_mixed, Counts};
use presk_kernel::ir::{Arith, BinOp, BufSpec, Expr, Stmt, UnOp};
use presk_kernel::lower::{lower, lower_ordered, Operand};
use presk_kernel::{BufKind, Dtype, Kernel, Threads};
use presk_rt::{Axis, Buf, ReductionOrder};
use rug::float::Round;
use rug::ops::AssignRound;
use rug::{Float as R, Integer};

const PREC: u32 = 1200;

fn fmt(s: &str) -> Format {
    Format::parse(s).unwrap()
}

/// A xorshift stream.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A value of `f`, its codes drawn uniformly: specials and subnormals
    /// included.
    fn value(&mut self, f: Float) -> f64 {
        f.decode(self.next() & ((1u64 << f.bits()) - 1))
    }
}

/// The real `x` rounded once to `f`, apart from the `f64` path. A float:
/// `x` cut to 120 bits toward zero, a last bit set for what was cut or for
/// a rest (`inexact`), then rounded by the format's integer rounding. An
/// integer or a fixed-point format: `x`'s units rounded by MPFR, ties to
/// even, then wrapped or clamped as integers (`x` exact: a sum, a
/// difference, a product).
fn rounded(f: Format, x: &R, inexact: bool) -> (f64, bool) {
    if let Format::Int(_) | Format::Fixed(_) = f {
        let (int, frac) = match f {
            Format::Int(i) => (i, 0),
            Format::Fixed(q) => (q.int, q.frac),
            Format::Float(_) | Format::Affine(_) => unreachable!(),
        };
        assert!(!inexact, "a grid divides nothing");
        let units = R::with_val(PREC, x << frac);
        let (n, _) = units.to_integer_round(Round::Nearest).expect("finite");
        let fitted = match int.overflow {
            presk_format::IntOverflow::Wrap if int.signed => n.clone().keep_signed_bits(int.bits),
            presk_format::IntOverflow::Wrap => n.clone().keep_bits(int.bits),
            presk_format::IntOverflow::Clamp => n.clone().clamp(&Integer::from(int.min()), &Integer::from(int.max())),
        };
        let y = fitted.to_f64() * 2f64.powi(-frac);
        return (y, fitted != n);
    }
    let (m, e) = x.to_integer_exp().expect("finite");
    let negative = m < 0;
    let mut a = m.abs();
    let mut e = e;
    let mut sticky = inexact;
    let bits = a.significant_bits();
    if bits > 120 {
        let s = bits - 120;
        sticky |= !a.is_divisible_2pow(s);
        a >>= s;
        e += s as i32;
    }
    let a: Integer = (a << 1u32) | Integer::from(sticky as u32);
    let m = a.to_i128().expect("121 bits");
    f.round_scaled(if negative { -m } else { m }, e - 1, Tie::Even)
}

/// `a op b` (or `√a`) rounded once to `f`: from MPFR's exact sum, product
/// and quotient (cut toward zero at 1 200 bits, its rest kept), or, where
/// an operand or the result is not a finite non-zero number, the `f64`
/// operation's value, which IEEE 754 fixes.
fn reference(op: Option<BinOp>, a: f64, b: f64, f: Format) -> (f64, bool) {
    let y = match op {
        Some(BinOp::Accum) => a + b,
        Some(BinOp::Sub) => a - b,
        Some(BinOp::Product) => a * b,
        Some(BinOp::Div) => a / b,
        None => a.sqrt(),
        Some(other) => unreachable!("{other:?}"),
    };
    if !a.is_finite() || !b.is_finite() || !y.is_finite() || y == 0.0 {
        // A NaN rounded to a format without NaNs saturates with the sign of
        // the first NaN operand, else +.
        let y = match f {
            Format::Float(g) if y.is_nan() && !g.has_nan() => [a, b].into_iter().find(|v| v.is_nan()).unwrap_or(f64::NAN),
            _ => y,
        };
        return f.round_noted(y, Tie::Even);
    }
    let (ra, rb) = (R::with_val(PREC, a), R::with_val(PREC, b));
    let mut x = R::new(PREC);
    let o = match op {
        Some(BinOp::Accum) => x.assign_round(&ra + &rb, Round::Zero),
        Some(BinOp::Sub) => x.assign_round(&ra - &rb, Round::Zero),
        Some(BinOp::Product) => x.assign_round(&ra * &rb, Round::Zero),
        Some(BinOp::Div) => x.assign_round(&ra / &rb, Round::Zero),
        _ => x.assign_round(ra.sqrt_ref(), Round::Zero),
    };
    rounded(f, &x, o != Ordering::Equal)
}

fn spec(n: usize) -> BufSpec {
    BufSpec { dims: vec![n], kind: BufKind::Real, names: vec![None] }
}

/// `%2[i] = %0[i] op %1[i]` (or `%1[i] = √%0[i]`) over `n` elements, stored
/// in `f64`, in the formats `arith`.
fn op_kernel(op: Option<BinOp>, n: usize, arith: Arith) -> Kernel {
    let at = || vec![Expr::Iv(0)];
    let (inputs, value) = match op {
        Some(op) => (vec![spec(n), spec(n)], Expr::Bin(op, Box::new(Expr::Load(0, at())), Box::new(Expr::Load(1, at())))),
        None => (vec![spec(n)], Expr::Un(UnOp::Sqrt, Box::new(Expr::Load(0, at())))),
    };
    let out = inputs.len();
    Kernel { name: "op".into(), dtype: Dtype::F64, arith, inputs, output: spec(n), nloops: 1, nreal: 0, nint: 0, nvec: 0, lanes: 1, body: vec![Stmt::For { var: 0, bound: n, body: vec![Stmt::Store { buf: out, index: at(), value }] }], threads: Threads::One }
}

/// The formats that route `op` to `f` on the path of several formats: a
/// sum's addition to the accumulation format, a product of terms to the
/// product format, any other operation to the arithmetic format, with
/// another format beside it.
fn routed(op: Option<BinOp>, f: Format) -> Arith {
    let other = Format::Float(F64);
    match op {
        Some(BinOp::Accum) => Arith { arith: other, product: other, accum: f, store: None },
        Some(BinOp::Product) => Arith { arith: other, product: f, accum: other, store: None },
        _ => Arith { arith: f, product: other, accum: other, store: None },
    }
}

/// A result rounded once with its overflow flag, and its operands.
type Rounded = ((f64, bool), (f64, f64));

/// The counts a run of results `ys` gives, each with its overflow flag.
fn counted(f: Format, ys: &[Rounded]) -> Counts {
    let mut c = Counts::default();
    for ((y, o), (a, b)) in ys {
        let (inf_in, nan_in) = (a.is_infinite() || b.is_infinite(), a.is_nan() || b.is_nan());
        match f {
            Format::Float(_) if *o && y.is_finite() => c.saturations += 1,
            Format::Float(_) if y.is_infinite() && !inf_in => c.infinities += 1,
            Format::Float(_) if y.is_nan() && !nan_in => c.nans += 1,
            Format::Int(i) | Format::Fixed(presk_format::Fixed { int: i, .. }) if *o => {
                if i.overflow == presk_format::IntOverflow::Wrap && !inf_in && !nan_in {
                    c.wraps += 1
                } else {
                    c.saturations += 1
                }
            }
            _ => {}
        }
    }
    c
}

/// Every operation of a kernel of several formats is its exact result
/// rounded once to the operation's format, whatever its operands' formats:
/// `+` as a sum's addition, `×` as a product of terms, `−`, `÷` and `√` as
/// the arithmetic's, on random codes of five operand formats (specials and
/// subnormals among them) and on the cases where the `f64` result is a
/// target's midpoint and only the rest decides, into eleven formats; the
/// counts are the results' overflows.
#[test]
fn every_operation_rounds_once_to_its_format() {
    let operands = [F16, BF16, E4M3_SATURATING, E4M3FN, F32];
    let targets = [Format::Float(F16), Format::Float(BF16), Format::Float(E4M3_SATURATING), Format::Float(E4M3FN), Format::Float(E5M2), Format::Float(F32), Format::Float(E8M15), fmt("i16"), fmt("i8/sat"), fmt("fixed(i32,frac=8)"), fmt("fixed(i16/sat,frac=4)")];
    let ops = [Some(BinOp::Accum), Some(BinOp::Sub), Some(BinOp::Product), Some(BinOp::Div), None];
    let mut rng = Rng(0x5eed_0018_0003);
    // A midpoint of each float target near 1 and 1 + 1/4, and of the
    // grids, with rests far below `f64`'s last bit, of both signs.
    let mut hard: Vec<(f64, f64)> = Vec::new();
    for p in [2, 3, 4, 8, 11, 16] {
        for base in [1.0, 1.25, -3.0] {
            let m: f64 = base * (1.0 + 2f64.powi(-p));
            for rest in [2f64.powi(-70), -2f64.powi(-70), 2f64.powi(-30), -2f64.powi(-30), 0.0] {
                hard.push((m, rest * base.abs()));
            }
        }
    }
    for (m, rest) in [(2.5, 1e-20), (2.5, -1e-20), (-3.5, 1e-20), (1.0 + 1.0 / 512.0, 1e-25), (1.0 + 1.0 / 512.0, -1e-25), (32767.5, 1e-15), (480.0 + 16.0, -1e-18), (480.0 + 16.0, 1e-18)] {
        hard.push((m, rest));
    }
    let mut checked = 0;
    for &fa in &operands {
        for &fb in &operands {
            let mut pairs: Vec<(f64, f64)> = (0..1500).map(|_| (rng.value(fa), rng.value(fb))).collect();
            if fa == F32 && fb == F32 {
                pairs.extend(hard.iter().copied());
            }
            for &f in &targets {
                for &op in &ops {
                    let exact = matches!(f, Format::Int(_) | Format::Fixed(_));
                    if exact && matches!(op, Some(BinOp::Div) | None) {
                        continue;
                    }
                    let n = pairs.len();
                    let k = op_kernel(op, n, routed(op, f));
                    let a = Buf::Format(Format::Float(fa), pairs.iter().map(|p| p.0).collect());
                    let b = Buf::Format(Format::Float(fb), pairs.iter().map(|p| p.1).collect());
                    let mut out = Buf::F64(vec![0.0; n].into());
                    let ins: Vec<&Buf> = if op.is_some() { vec![&a, &b] } else { vec![&a] };
                    let counts = run_emulated(&k, &ins, &mut out).unwrap();
                    let got = out.to_f64();
                    let want: Vec<Rounded> = pairs.iter().map(|&(x, y)| (reference(op, x, if op.is_some() { y } else { 0.0 }, f), (x, if op.is_some() { y } else { 0.0 }))).collect();
                    for (i, ((w, _), (x, y))) in want.iter().enumerate() {
                        assert!(got[i].to_bits() == w.to_bits() || (got[i].is_nan() && w.is_nan()), "{fa} {op:?} {fb} into {f}: {x:e} and {y:e} gave {:e}, rounded once {w:e}", got[i]);
                    }
                    assert_eq!(counts, counted(f, &want), "{fa} {op:?} {fb} into {f}");
                    checked += n;
                }
            }
        }
    }
    assert!(checked > 1_000_000, "{checked}");
}

fn bits(b: &Buf) -> Vec<u64> {
    b.to_f64().into_iter().map(f64::to_bits).collect()
}

fn ax(name: &str, len: usize) -> Axis {
    Axis { name: if name.is_empty() { None } else { Some(name.into()) }, len }
}

fn real(axes: Vec<Axis>) -> Operand {
    Operand { axes, kind: BufKind::Real }
}

fn kernel(op: PrimOp, ins: &[Operand], d: Dtype) -> Kernel {
    lower(&op, ins, d).unwrap_or_else(|| panic!("{} is a host operation", op.name())).unwrap_or_else(|e| panic!("{}: {e}", op.name())).kernel
}

/// Where a kernel's formats are one, its path of several formats gives
/// the one-format path's bits and counts: binary16's primitives, the
/// functions, the sums and the contractions included, on random binary16
/// operands (step 2 holds the one-format path to the machine's).
#[test]
fn the_path_of_several_formats_is_the_one_format_path_in_one_format() {
    let mut rng = Rng(0x0018_0003_0001);
    let n = 6;
    let (m, v) = (ax("model", n), ax("rows", 4));
    let x = real(vec![v.clone(), m.clone()]);
    let w = real(vec![m.clone(), ax("out", 5)]);
    let row = real(vec![m.clone()]);
    let cases: Vec<(PrimOp, Vec<Operand>)> = vec![
        (PrimOp::Add, vec![x.clone(), x.clone()]),
        (PrimOp::Sub, vec![x.clone(), row.clone()]),
        (PrimOp::Mul, vec![x.clone(), x.clone()]),
        (PrimOp::Div, vec![x.clone(), x.clone()]),
        (PrimOp::Exp, vec![x.clone()]),
        (PrimOp::Tanh, vec![x.clone()]),
        (PrimOp::Sqrt, vec![x.clone()]),
        (PrimOp::Gelu, vec![x.clone()]),
        (PrimOp::Sigmoid, vec![x.clone()]),
        (PrimOp::SumAll, vec![x.clone()]),
        (PrimOp::Mean { axis: "model".into() }, vec![x.clone()]),
        (PrimOp::Softmax { axis: "model".into() }, vec![x.clone()]),
        (PrimOp::SumSquares, vec![x.clone()]),
        (PrimOp::Dot, vec![x.clone(), w.clone()]),
    ];
    let f16 = Format::Float(F16);
    for (op, ins) in cases {
        let mut k = kernel(op.clone(), &ins, Dtype::new(F16));
        k.arith = Arith::one(f16);
        for trial in 0..200 {
            let bufs: Vec<Buf> = ins.iter().map(|o| Buf::in_format(f16, &(0..o.axes.iter().map(|a| a.len).product::<usize>()).map(|_| if trial % 2 == 0 { rng.value(F16) } else { (rng.value(F16) / 1024.0).clamp(-8.0, 8.0) }).collect::<Vec<_>>())).collect();
            let refs: Vec<&Buf> = bufs.iter().collect();
            let mut one = Buf::in_format(f16, &vec![0.0; k.output.numel()]);
            let mut several = one.clone();
            let c1 = run_kernel_in(&k, f16, &refs, &mut one).unwrap();
            let c2 = run_kernel_mixed(&k, &refs, &mut several).unwrap();
            let (a, b) = (one.to_f64(), several.to_f64());
            assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits() || (p.is_nan() && q.is_nan())), "{}: {a:?} and {b:?}", op.name());
            assert_eq!(c1, c2, "{}", op.name());
        }
    }
}

/// A contraction's products round to its product format and its sums to
/// its accumulation format, then the sum to its storage type: E4M3
/// operands (their products exact in binary16) summed in `f32`, stored in
/// binary16, against the sum in `f32` computed natively; the same summed in
/// binary16; and in a fixed-point format, where the sum is exact.
#[test]
fn a_sum_rounds_its_products_and_its_additions_to_their_formats() {
    let mut rng = Rng(0x0018_0003_0002);
    let (i, kk, j) = (3, 40, 4);
    let x = real(vec![ax("i", i), ax("k", kk)]);
    let w = real(vec![ax("k", kk), ax("j", j)]);
    let e4m3 = Format::Float(E4M3_SATURATING);
    let draw = |rng: &mut Rng, n: usize| -> Vec<f64> { (0..n).map(|_| rng.value(E4M3_SATURATING)).map(|v| v / 16.0).map(|v| E4M3_SATURATING.round(v, Tie::Even)).collect() };
    for _ in 0..50 {
        let (xs, ws) = (draw(&mut rng, i * kk), draw(&mut rng, kk * j));
        let (bx, bw) = (Buf::in_format(e4m3, &xs), Buf::in_format(e4m3, &ws));
        let product = |a: f64, b: f64| F16.round(a * b, Tie::Even);
        for (accum, sum) in [
            (Format::Float(F32), Box::new(|terms: &[f64]| terms.iter().fold(0f32, |s, t| s + *t as f32) as f64) as Box<dyn Fn(&[f64]) -> f64>),
            (Format::Float(F16), Box::new(|terms: &[f64]| terms.iter().fold(0.0, |s, t| F16.round(s + t, Tie::Even)))),
            (fmt("fixed(i32,frac=18)"), Box::new(|terms: &[f64]| terms.iter().sum::<f64>())),
        ] {
            let mut k = kernel(PrimOp::Dot, &[x.clone(), w.clone()], Dtype::new(F16));
            k.arith = Arith { arith: Format::Float(F16), product: Format::Float(F16), accum, store: None };
            let mut out = Buf::in_format(Format::Float(F16), &vec![0.0; i * j]);
            run_emulated(&k, &[&bx, &bw], &mut out).unwrap();
            for r in 0..i {
                for c in 0..j {
                    let terms: Vec<f64> = (0..kk).map(|t| product(xs[r * kk + t], ws[t * j + c])).collect();
                    let want = F16.round(sum(&terms), Tie::Even);
                    assert_eq!(out.to_f64()[r * j + c].to_bits(), want.to_bits(), "accumulated in {accum}");
                }
            }
        }
    }
}

/// `chunked(c)`: sums of `c` consecutive terms each from zero, then the
/// chunks' sums from zero, the last chunk shorter: in binary16 against the
/// sum written out, at chunks of 1, 3, 8, the whole axis and more; a chunk
/// of the whole axis or more is the sequential sum; native kernels run the
/// order natively with the emulation's bits; `ein` and `dot_vjp_l` take it,
/// and a sum over two axes refuses it.
#[test]
fn a_chunked_order_sums_in_chunks() {
    let mut rng = Rng(0x0018_0003_0003);
    let (i, kk, j) = (2, 20, 3);
    let x = real(vec![ax("i", i), ax("k", kk)]);
    let w = real(vec![ax("k", kk), ax("j", j)]);
    let f16 = Format::Float(F16);
    let xs: Vec<f64> = (0..i * kk).map(|_| F16.round(rng.value(F16) / 4096.0, Tie::Even)).collect();
    let ws: Vec<f64> = (0..kk * j).map(|_| F16.round(rng.value(F16) / 4096.0, Tie::Even)).collect();
    let (bx, bw) = (Buf::in_format(f16, &xs), Buf::in_format(f16, &ws));
    let add = |s: f64, t: f64| F16.round(s + t, Tie::Even);
    let sequential = {
        let mut k = lower_ordered(&PrimOp::Dot, &[x.clone(), w.clone()], Dtype::new(F16), ReductionOrder::Sequential).unwrap().unwrap().kernel;
        k.arith = Arith::one(f16);
        let mut out = Buf::in_format(f16, &vec![0.0; i * j]);
        run_emulated(&k, &[&bx, &bw], &mut out).unwrap();
        out.to_f64()
    };
    for c in [1, 3, 8, 20, 32] {
        let mut k = lower_ordered(&PrimOp::Dot, &[x.clone(), w.clone()], Dtype::new(F16), ReductionOrder::Chunked { size: c }).unwrap().unwrap().kernel;
        k.arith = Arith::one(f16);
        let mut out = Buf::in_format(f16, &vec![0.0; i * j]);
        run_emulated(&k, &[&bx, &bw], &mut out).unwrap();
        let got = out.to_f64();
        for r in 0..i {
            for col in 0..j {
                let terms: Vec<f64> = (0..kk).map(|t| F16.round(xs[r * kk + t] * ws[t * j + col], Tie::Even)).collect();
                let want = terms.chunks(c as usize).map(|ch| ch.iter().fold(0.0, |s, t| add(s, *t))).fold(0.0, add);
                assert_eq!(got[r * j + col].to_bits(), want.to_bits(), "chunked({c})");
            }
        }
        if c as usize >= kk {
            assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), sequential.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "chunked({c}) over {kk} terms");
        }
        // Natively, the order is the kernel's text: `f32`'s bits.
        let native = lower_ordered(&PrimOp::Dot, &[x.clone(), w.clone()], Dtype::F32, ReductionOrder::Chunked { size: c }).unwrap().unwrap().kernel;
        let (nx, nw) = (Buf::of(Dtype::F32, &xs), Buf::of(Dtype::F32, &ws));
        let mut a = Buf::of(Dtype::F32, &vec![0.0; i * j]);
        let mut b = a.clone();
        run_kernel(&native, &[&nx, &nw], &mut a);
        run_emulated(&native, &[&nx, &nw], &mut b).unwrap();
        assert_eq!(bits(&a), bits(&b), "chunked({c}) in f32");
    }
    // `ein` over one summed axis and `dot_vjp_l` take the order; a sum over
    // two axes does not.
    let chunked = ReductionOrder::Chunked { size: 4 };
    let ein = PrimOp::Ein { operands: vec![vec!["i".into(), "k".into()], vec!["k".into(), "j".into()]], output: vec!["i".into(), "j".into()] };
    let e = lower_ordered(&ein, &[x.clone(), w.clone()], Dtype::F32, chunked).unwrap().unwrap().kernel;
    let d = lower_ordered(&PrimOp::Dot, &[x.clone(), w.clone()], Dtype::F32, chunked).unwrap().unwrap().kernel;
    let (nx, nw) = (Buf::of(Dtype::F32, &xs), Buf::of(Dtype::F32, &ws));
    let (mut a, mut b) = (Buf::of(Dtype::F32, &vec![0.0; i * j]), Buf::of(Dtype::F32, &vec![0.0; i * j]));
    run_kernel(&e, &[&nx, &nw], &mut a);
    run_kernel(&d, &[&nx, &nw], &mut b);
    assert_eq!(bits(&a), bits(&b), "ein and dot sum in the same chunks");
    let ct = real(vec![ax("i", i), ax("j", j)]);
    let wt = real(vec![ax("k", kk), ax("j", j)]);
    assert!(lower_ordered(&PrimOp::DotVjpL, &[ct.clone(), wt], Dtype::F32, chunked).unwrap().is_ok());
    let a3 = real(vec![ax("b", 2), ax("t", 3), ax("i", i)]);
    let c3 = real(vec![ax("b", 2), ax("t", 3), ax("j", j)]);
    // The derivative in the right operand of a `dot` of rank three: a sum
    // over two axes, not a contraction nest under a chunked order.
    let e = lower_ordered(&PrimOp::DotVjpR, &[a3, c3], Dtype::F32, chunked).unwrap().unwrap_err();
    assert!(e.contains("chunked(4)") && e.contains("over 2 axes"), "{e}");
}

/// What a kernel of several formats is refused: a format no `f64` holds,
/// beside another; a float of 26 to 52 bits; a division in an integer
/// arithmetic. One that the formats' one path takes is not.
#[test]
fn the_formats_a_kernel_cannot_be_emulated_in_are_refused() {
    let n = 4;
    let a = Buf::of(Dtype::F32, &[1.0, 2.0, 3.0, 4.0]);
    let mut out = Buf::F64(vec![0.0; n].into());
    let wide = fmt("fixed(i64,frac=18)");
    let k = op_kernel(Some(BinOp::Accum), n, Arith { arith: Format::Float(F32), product: Format::Float(F32), accum: wide, store: None });
    let e = run_emulated(&k, &[&a, &a], &mut out).unwrap_err();
    assert!(e.contains("fixed(i64,frac=18), which an f64 does not hold"), "{e}");
    // In one format that holds the operands, E4M3 values, the one-format
    // path sums it, exactly.
    let k = op_kernel(Some(BinOp::Accum), n, Arith::one(wide));
    let e4m3 = Buf::in_format(Format::Float(E4M3_SATURATING), &[1.0, 2.0, 3.0, 480.0]);
    assert_eq!(run_emulated(&k, &[&e4m3, &e4m3], &mut out), Ok(Counts::default()));
    assert_eq!(out.to_f64(), [2.0, 4.0, 6.0, 960.0]);
    let p30 = Format::Float(Float { p: 30, ..F64 });
    let k = op_kernel(Some(BinOp::Div), n, Arith { arith: p30, product: Format::Float(F32), accum: Format::Float(F32), store: None });
    assert!(run_emulated(&k, &[&a, &a], &mut out).unwrap_err().contains("2p + 2 = 62"));
    let k = op_kernel(Some(BinOp::Div), n, Arith { arith: fmt("i32"), product: Format::Float(F32), accum: Format::Float(F32), store: None });
    assert!(run_emulated(&k, &[&a, &a], &mut out).unwrap_err().contains("a division, which i32 does not have"));
}

