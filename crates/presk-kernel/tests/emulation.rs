//! The kernel interpreter's emulation of every format (plan 0.18.0 step 2,
//! D205): `f64` and `f32` emulated are the native kernels' bits; a float
//! format's operations are correctly rounded; the conversions, the integer
//! and fixed-point formats, and what an emulated run counts. The exhaustive
//! checks of MT1 are the ignored tests (`cargo test --release -p
//! presk-kernel --test emulation -- --ignored`): binary16's arithmetic and
//! conversions against the machine's own instructions on aarch64, and
//! `bf16`'s conversions against the runtime's functions.

use presk_core::PrimOp;
use presk_format::{Fixed, Float, Format, Int, BF16, E4M3FN, E4M3_SATURATING, F16, F32, F64};
use presk_kernel::exec::{launch_buf, output_buf, run_emulated, run_kernel, run_kernel_in, Counts};
use presk_kernel::lower::{lower, Operand};
use presk_kernel::{BufKind, Dtype, Kernel};
use presk_rt::{Axis, Buf};

fn ax(name: &str, len: usize) -> Axis {
    Axis { name: if name.is_empty() { None } else { Some(name.into()) }, len }
}

fn real(axes: Vec<Axis>) -> Operand {
    Operand { axes, kind: BufKind::Real }
}

/// `n` values in [−1, 1) from `seed`.
fn values(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        })
        .collect()
}

fn kernel(op: PrimOp, ins: &[Operand], d: Dtype) -> Kernel {
    lower(&op, ins, d).unwrap_or_else(|| panic!("{} is a host operation", op.name())).unwrap_or_else(|e| panic!("{}: {e}", op.name())).kernel
}

fn bits(b: &Buf) -> Vec<u64> {
    match b {
        Buf::I64(v) => v.iter().map(|x| *x as u64).collect(),
        Buf::Bf16(v) => v.iter().map(|x| *x as u64).collect(),
        other => other.to_f64().into_iter().map(f64::to_bits).collect(),
    }
}

/// A kernel's result emulated in `arith`, into a buffer of `out` of its
/// output's length, and the counts.
fn emulate(k: &Kernel, arith: Format, ins: &[&Buf], out: Format) -> (Vec<f64>, Counts) {
    let mut o = Buf::in_format(out, &vec![0.0; k.output.numel()]);
    let counts = run_kernel_in(k, arith, ins, &mut o).unwrap_or_else(|e| panic!("{}: {e}", k.name));
    (o.to_f64(), counts)
}

/// `f64` and `f32` emulated are the native kernels: every tensor primitive
/// of the conformance suite's that reads reals, lowered at `f64`, `f32` and
/// `bf16`, on random operands, gives `run_kernel`'s bits, and counts
/// nothing.
/// Every tensor primitive of the conformance suite's that reads reals,
/// with operands of small shapes.
fn suite() -> Vec<(PrimOp, Vec<Operand>)> {
    let x = || real(vec![ax("batch", 2), ax("model", 3)]);
    let row = || real(vec![ax("model", 3)]);
    let w = || real(vec![ax("", 3), ax("", 4)]);
    let q = || real(vec![ax("batch", 2), ax("heads", 2), ax("seq", 3), ax("hd", 2)]);
    let k2 = || real(vec![ax("batch", 2), ax("heads", 2), ax("seq2", 3), ax("hd", 2)]);
    let ct2 = || real(vec![ax("batch", 2)]);
    let sm = || real(vec![ax("batch", 2), ax("seq", 3), ax("model", 4)]);
    let logits = || real(vec![ax("batch", 2), ax("vocab", 5)]);
    let targets = || Operand { axes: vec![ax("batch", 2)], kind: BufKind::Idx };
    vec![
        (PrimOp::Add, vec![x(), x()]),
        (PrimOp::Sub, vec![x(), row()]),
        (PrimOp::Mul, vec![x(), x()]),
        (PrimOp::Div, vec![x(), ct2()]),
        (PrimOp::Neg, vec![x()]),
        (PrimOp::Pow(3), vec![x()]),
        (PrimOp::Exp, vec![x()]),
        (PrimOp::Tanh, vec![x()]),
        (PrimOp::Sin, vec![x()]),
        (PrimOp::Cos, vec![x()]),
        (PrimOp::Abs, vec![x()]),
        (PrimOp::Gelu, vec![x()]),
        (PrimOp::GeluDeriv, vec![x()]),
        (PrimOp::Relu, vec![x()]),
        (PrimOp::Sign, vec![x()]),
        (PrimOp::Sigmoid, vec![x()]),
        (PrimOp::Dot, vec![x(), w()]),
        (PrimOp::Ein { operands: vec![vec!["batch".into(), "heads".into(), "seq".into(), "hd".into()], vec!["batch".into(), "heads".into(), "seq2".into(), "hd".into()]], output: vec!["batch".into(), "heads".into(), "seq".into(), "seq2".into()] }, vec![q(), k2()]),
        (PrimOp::SumAll, vec![x()]),
        (PrimOp::Mean { axis: "model".into() }, vec![x()]),
        (PrimOp::Softmax { axis: "model".into() }, vec![sm()]),
        (PrimOp::CrossEntropy, vec![logits(), targets()]),
        (PrimOp::Max, vec![x(), row()]),
        (PrimOp::SumSquares, vec![x()]),
    ]
}

/// The operands of case `i` in the type `d`, as `Buf::of` rounds them.
fn operands(i: usize, ins: &[Operand], d: Dtype) -> Vec<Buf> {
    ins.iter()
        .enumerate()
        .map(|(j, o)| {
            let n = o.axes.iter().map(|a| a.len).product();
            match o.kind {
                BufKind::Idx => Buf::I64((0..n as i64).map(|t| (t * 3 + 1) % 5).collect()),
                _ => Buf::of(d, &values(n, (i * 8 + j) as u64).iter().map(|v| v * 3e4).collect::<Vec<_>>()),
            }
        })
        .collect()
}

#[test]
fn f64_and_f32_emulated_are_native() {
    let cases = suite();
    for d in [Dtype::F64, Dtype::F32, Dtype::BF16] {
        for (i, (op, ins)) in cases.iter().enumerate() {
            let k = kernel(op.clone(), ins, d);
            let bufs: Vec<Buf> = ins
                .iter()
                .enumerate()
                .map(|(j, o)| {
                    let n = o.axes.iter().map(|a| a.len).product();
                    match o.kind {
                        BufKind::Idx => Buf::I64((0..n as i64).map(|t| (t * 3 + 1) % 5).collect()),
                        _ => Buf::of(d, &values(n, (i * 8 + j) as u64)),
                    }
                })
                .collect();
            let refs: Vec<&Buf> = bufs.iter().collect();
            let mut native = output_buf(&k);
            run_kernel(&k, &refs, &mut native);
            let mut emulated = output_buf(&k);
            let counts = run_kernel_in(&k, Format::Float(d.arith().format()), &refs, &mut emulated).unwrap();
            assert_eq!(bits(&native), bits(&emulated), "{} at {d}", op.name());
            assert!(counts.is_zero(), "{} at {d}: {counts}", op.name());
        }
    }
}

/// binary16's and `e8m15`'s values held as their codes (plan 0.18.2 step
/// 1, D215) emulate as they did held as `f64`s ([`Buf::Format`], 0.18.1's
/// buffers): every case of the suite lowered in either type, on operands
/// of a range where binary16 overflows, gives the same values and counts
/// the same infinities and NaNs, the outputs of a launch and of
/// `output_buf` both codes.
#[test]
fn codes_emulate_as_the_values_held_as_f64s_did() {
    let mut counted = 0;
    for d in [Dtype::F16, Dtype::E8M15] {
        let f = Format::Float(d.format());
        for (i, (op, ins)) in suite().iter().enumerate() {
            let k = kernel(op.clone(), ins, d);
            let coded = operands(i, ins, d);
            assert!(coded.iter().any(|b| matches!(b, Buf::Codes(g, _) if Format::Float(*g) == f)), "{} at {d}", op.name());
            let held: Vec<Buf> = coded
                .iter()
                .map(|b| match b {
                    Buf::Codes(..) => Buf::Format(f, b.to_f64().into()),
                    other => other.clone(),
                })
                .collect();
            let mut by_codes = launch_buf(&k);
            assert!(matches!((&by_codes, output_buf(&k)), (Buf::Codes(..), Buf::Codes(..))), "{} at {d}", op.name());
            let mut by_values = Buf::Format(f, vec![0.0; k.output.numel()].into());
            let c = run_emulated(&k, &coded.iter().collect::<Vec<_>>(), &mut by_codes).unwrap();
            let v = run_emulated(&k, &held.iter().collect::<Vec<_>>(), &mut by_values).unwrap();
            for (x, y) in by_codes.to_f64().into_iter().zip(by_values.to_f64()) {
                assert!(if y.is_nan() { x.is_nan() } else { x.to_bits() == y.to_bits() }, "{} at {d}: {x:e} for {y:e}", op.name());
            }
            assert_eq!(c, v, "{} at {d}", op.name());
            counted += usize::from(!c.is_zero());
        }
    }
    assert!(counted > 4, "the operands overflow binary16 in a few cases: {counted}");
}

/// The pairs kernel: `out[a, b] = x[a, b] op y[b]`, the lowering's own
/// elementwise kernel with a broadcast, `x` holding each of its values along
/// `b` ([`rows`]).
fn pairs(op: PrimOp, na: usize, nb: usize) -> Kernel {
    let k = kernel(op, &[real(vec![ax("a", na), ax("b", nb)]), real(vec![ax("b", nb)])], Dtype::F32);
    assert_eq!(k.output.dims, [na, nb]);
    k
}

/// Each of `xs` `n` times, the first operand of [`pairs`].
fn rows(xs: &[f64], n: usize) -> Vec<f64> {
    xs.iter().flat_map(|x| std::iter::repeat_n(*x, n)).collect()
}

/// Every binary16 value but the NaNs, and a few NaNs.
fn f16_values() -> Vec<f64> {
    (0..1u64 << 16).map(|c| F16.decode(c)).filter(|x| !x.is_nan()).chain([f64::NAN, -f64::NAN]).collect()
}

/// binary16's +, −, × and ÷ are correctly rounded: computed in `f32` and
/// rounded once, against the operation in `f64` (exact for +, − and ×,
/// innocuous double rounding for ÷: 53 ≥ 2·11 + 2) rounded once, on every
/// value against 61 others, the edges among them.
#[test]
fn binary16_is_correctly_rounded() {
    let xs = f16_values();
    let mut ys: Vec<f64> = vec![0.0, -0.0, 1.0, -1.0, 65504.0, -65504.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 2f64.powi(-24), -2f64.powi(-24), 2f64.powi(-14), 3.0, 0.1f64, 1.0 / 3.0, 65519.0, 1e-7];
    ys.extend(values(44, 7).into_iter().map(|v| v * 300.0));
    let ys: Vec<f64> = ys.into_iter().map(|y| F16.round(y, presk_format::Tie::Even)).collect();
    let (bx, by) = (Buf::in_format(Format::Float(F16), &rows(&xs, ys.len())), Buf::in_format(Format::Float(F16), &ys));
    for op in [PrimOp::Add, PrimOp::Sub, PrimOp::Mul, PrimOp::Div] {
        let k = pairs(op.clone(), xs.len(), ys.len());
        let (got, counts) = emulate(&k, Format::Float(F16), &[&bx, &by], Format::Float(F16));
        let (mut infinities, mut nans) = (0, 0);
        for (i, x) in xs.iter().enumerate() {
            for (j, y) in ys.iter().enumerate() {
                let exact = match op {
                    PrimOp::Add => x + y,
                    PrimOp::Sub => x - y,
                    PrimOp::Mul => x * y,
                    _ => x / y,
                };
                let want = F16.round(exact, presk_format::Tie::Even);
                let g = got[i * ys.len() + j];
                assert!(g.to_bits() == want.to_bits() || (g.is_nan() && want.is_nan()), "{} {x:e} {y:e}: {g:e}, not {want:e}", op.name());
                infinities += (want.is_infinite() && x.is_finite() && y.is_finite()) as u64;
                nans += (want.is_nan() && !x.is_nan() && !y.is_nan()) as u64;
            }
        }
        assert_eq!(counts, Counts { infinities, nans, ..Counts::default() }, "{}", op.name());
        assert!(infinities > 0, "{}", op.name());
    }
}

/// A store converts, rounded once, ties to even, and counts what the
/// target's overflow rule did: every binary16 value into the saturating
/// E4M3 (an infinity, a NaN and anything from 496 up saturate), into OFP8's
/// E4M3 (they are NaN, from 464 up) and into `f32` (nothing).
#[test]
fn stores_convert_and_count() {
    let xs: Vec<f64> = (0..1u64 << 16).map(|c| F16.decode(c)).collect();
    let k = kernel(PrimOp::Neg, &[real(vec![ax("a", xs.len())])], Dtype::F32);
    let b = Buf::in_format(Format::Float(F16), &xs);
    for (target, sat, nan) in [(E4M3_SATURATING, true, false), (E4M3FN, false, true), (F32, false, false)] {
        let (got, counts) = emulate(&k, Format::Float(F16), &[&b], Format::Float(target));
        let mut want = Counts::default();
        for (x, g) in xs.iter().zip(&got) {
            let r = target.round(-x, presk_format::Tie::Even);
            assert!(r.to_bits() == g.to_bits() || (r.is_nan() && g.is_nan()), "{target} {x:e}");
            // 496 ties up to 512, past 480; 464 ties down to 448, so OFP8's
            // E4M3 overflows above it.
            let past = x.is_nan() || x.is_infinite() || if sat { x.abs() >= 496.0 } else { x.abs() > 464.0 };
            if target != F32 && past {
                if sat {
                    want.saturations += 1;
                } else if nan && !x.is_nan() {
                    want.nans += 1;
                }
            }
        }
        assert_eq!(counts, want, "{target}");
    }
    // The saturating E4M3's underflow is +0: −2^-11 is +0, not −0.
    let one = kernel(PrimOp::Neg, &[real(vec![ax("a", 1)])], Dtype::F32);
    let (got, _) = emulate(&one, Format::Float(F16), &[&Buf::in_format(Format::Float(F16), &[2f64.powi(-11)])], Format::Float(E4M3_SATURATING));
    assert_eq!(got[0].to_bits(), 0.0f64.to_bits());
}

/// A buffer holds the codes it is given: the saturating E4M3's −0 is a
/// code, which a conversion never gives (its underflow is +0) but a load
/// reads as −0, and binary16 keeps it.
#[test]
fn a_code_is_kept() {
    let e4 = Format::Float(E4M3_SATURATING);
    let b = Buf::in_format(e4, &[-0.0, 0.0, -480.0]);
    assert_eq!(b.to_f64().iter().map(|x| x.to_bits()).collect::<Vec<_>>(), [(-0.0f64).to_bits(), 0, (-480.0f64).to_bits()]);
    let k = kernel(PrimOp::Neg, &[real(vec![ax("a", 3)])], Dtype::F32);
    let (got, _) = emulate(&k, Format::Float(F16), &[&b], Format::Float(F16));
    assert_eq!(got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), [0, (-0.0f64).to_bits(), 480.0f64.to_bits()]);
    let (got, _) = emulate(&k, Format::Float(F16), &[&b], e4);
    assert_eq!(got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), [0, 0, 480.0f64.to_bits()]);
    assert_eq!(Buf::in_format(e4, &[1000.0]).to_f64(), [480.0]);
}

/// Loads convert into the arithmetic: an `f32` value read by a binary16
/// kernel is rounded once, and an overflow counted.
#[test]
fn loads_convert_and_count() {
    let xs = [1.0 + 2f64.powi(-11), 1.0 + 3.0 * 2f64.powi(-11), 65520.0, -1e6, 0.1];
    let k = kernel(PrimOp::Neg, &[real(vec![ax("a", xs.len())])], Dtype::F32);
    let (got, counts) = emulate(&k, Format::Float(F16), &[&Buf::of(Dtype::F32, &xs)], Format::Float(F32));
    let want: Vec<f64> = xs.iter().map(|x| -F16.round(*x as f32 as f64, presk_format::Tie::Even)).collect();
    assert_eq!(got, want);
    assert_eq!(counts, Counts { infinities: 2, ..Counts::default() });
}

/// `[m, k]·[k, n]` in the form the lowering gives a contraction whose
/// storage is narrower than its arithmetic (`bf16`'s): one accumulator per
/// element, in a variable, stored once. The `f32` form sums in its output
/// buffer, whose format would then hold every partial sum; where a sum
/// lives is the realization's `accum`, step 3's.
fn contraction(m: usize, k: usize, n: usize) -> Kernel {
    let kern = kernel(PrimOp::Dot, &[real(vec![ax("m", m), ax("k", k)]), real(vec![ax("", k), ax("", n)])], Dtype::BF16);
    assert!(kern.to_string().contains("s0 = 0"), "{kern}");
    kern
}

/// An integer contraction: `int8` operands, products exact and sums in a
/// wrapping `int32`, stored into an `int16` that clamps (the integer mode
/// of the accelerator the plan targets, as its simulator reads it) or into
/// the `int32` itself, against the sums computed exactly.
#[test]
fn integer_contractions_wrap_and_clamp() {
    let (m, kk, n) = (3usize, 64usize, 4usize);
    let i8s = Int::parse("i8").unwrap();
    let a: Vec<f64> = values(m * kk, 1).iter().map(|v| (v * 128.0).floor()).collect();
    let w: Vec<f64> = values(kk * n, 2).iter().map(|v| (v * 128.0).floor()).collect();
    // A row and a column of −128s, so that some sums leave int16.
    let a: Vec<f64> = a.iter().enumerate().map(|(i, v)| if i < kk { -128.0 } else { *v }).collect();
    let w: Vec<f64> = w.iter().enumerate().map(|(i, v)| if i % n == 0 { -128.0 } else { *v }).collect();
    let k = contraction(m, kk, n);
    let (ba, bw) = (Buf::in_format(Format::Int(i8s), &a), Buf::in_format(Format::Int(i8s), &w));
    let exact: Vec<i128> = (0..m).flat_map(|r| (0..n).map(move |c| (r, c))).map(|(r, c)| (0..kk).map(|t| a[r * kk + t] as i128 * w[t * n + c] as i128).sum()).collect();
    let i32w = Format::parse("i32").unwrap();
    let (got, counts) = emulate(&k, i32w, &[&ba, &bw], Format::parse("i16/sat").unwrap());
    let want: Vec<f64> = exact.iter().map(|s| (*s).clamp(-32768, 32767) as f64).collect();
    assert_eq!(got, want);
    let outside = exact.iter().filter(|s| !(-32768..=32767).contains(*s)).count() as u64;
    assert!(outside > 0);
    assert_eq!(counts, Counts { saturations: outside, ..Counts::default() });
    // A narrower wrapping sum: int16, every partial sum wrapped.
    let i16w = Format::parse("i16").unwrap();
    let (got, counts) = emulate(&k, i16w, &[&ba, &bw], i32w);
    let wrap = |s: i128| ((s + 32768).rem_euclid(65536) - 32768) as f64;
    assert_eq!(got, exact.iter().map(|s| wrap(*s)).collect::<Vec<_>>());
    assert!(counts.wraps > 0 && counts.saturations == 0);
    // Exp is no integer operation.
    let soft = kernel(PrimOp::Softmax { axis: "k".into() }, &[real(vec![ax("m", m), ax("k", kk)])], Dtype::F32);
    let err = run_kernel_in(&soft, i32w, &[&ba], &mut output_buf(&soft)).unwrap_err();
    assert!(err.contains("exp") && err.contains("i32"), "{err}");
}

/// An exact accumulator: E4M3 products summed in `fixed(i64,frac=18)` are
/// the exact sums, rounded once to binary16 at the store, where a binary16
/// sum rounds at every addition.
#[test]
fn a_fixed_point_sum_is_exact() {
    let (m, kk, n) = (2usize, 256usize, 3usize);
    let e4 = Format::Float(E4M3_SATURATING);
    // Codes of magnitude at most 15 (exponent field at most 10), so that
    // the sums stay inside binary16.
    let draw = |n: usize, seed: u64| -> Vec<f64> {
        values(n, seed)
            .iter()
            .map(|v| {
                let c = ((v + 1.0) * 128.0) as u64 & 0xff;
                E4M3_SATURATING.decode((c & 0x87) | ((c >> 3 & 15) % 11) << 3)
            })
            .collect()
    };
    let (a, w) = (draw(m * kk, 3), draw(kk * n, 4));
    let k = contraction(m, kk, n);
    let (ba, bw) = (Buf::in_format(e4, &a), Buf::in_format(e4, &w));
    let fixed = Format::Fixed(Fixed::parse("fixed(i64,frac=18)").unwrap());
    let (got, counts) = emulate(&k, fixed, &[&ba, &bw], Format::Float(F16));
    assert!(counts.is_zero(), "{counts}");
    let (mut differ, mut sixteen) = (0, Vec::new());
    for r in 0..m {
        for c in 0..n {
            // Exact: every product is a multiple of 2^-18 below 2^18.
            let units: i128 = (0..kk).map(|t| ((a[r * kk + t] * w[t * n + c]) * 2f64.powi(18)) as i128).sum();
            let want = F16.round_scaled(units, -18, presk_format::Tie::Even).0;
            assert_eq!(got[r * n + c].to_bits(), want.to_bits(), "{r} {c}");
            sixteen.push(want);
        }
    }
    let (got16, _) = emulate(&k, Format::Float(F16), &[&ba, &bw], Format::Float(F16));
    differ += got16.iter().zip(&sixteen).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    assert!(differ > 0, "a binary16 sum of 256 products rounds");
}

/// What has no exact emulation is refused: a float of 30 bits computes in
/// no native type with 62.
#[test]
fn a_format_without_an_exact_emulation_is_refused() {
    let f = Float::parse("float(p=30,emin=-126,emax=127)").unwrap();
    let k = kernel(PrimOp::Neg, &[real(vec![ax("a", 2)])], Dtype::F32);
    let err = run_kernel_in(&k, Format::Float(f), &[&Buf::in_format(Format::Float(f), &[1.0, 2.0])], &mut output_buf(&k)).unwrap_err();
    assert!(err.contains("62"), "{err}");
    assert_eq!(presk_kernel::exec::emulated_in(F16), Some(F32));
    assert_eq!(presk_kernel::exec::emulated_in(BF16), Some(F32));
    assert_eq!(presk_kernel::exec::emulated_in(presk_format::E8M15), Some(F64));
    assert_eq!(presk_kernel::exec::emulated_in(F64), Some(F64));
}

// ----- the exhaustive checks (MT1) --------------------------------------------

/// Runs `f(lo, hi)` over `0..n` in chunks of `chunk` on the machine's
/// threads, at most eight, and adds what they return.
fn chunks<T: Send + Default + std::ops::AddAssign>(n: usize, chunk: usize, f: impl Fn(usize, usize) -> T + Sync) -> T {
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get()).min(8);
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut acc = T::default();
                    loop {
                        let lo = next.fetch_add(chunk, std::sync::atomic::Ordering::Relaxed);
                        if lo >= n {
                            return acc;
                        }
                        acc += f(lo, (lo + chunk).min(n));
                    }
                })
            })
            .collect();
        let mut total = T::default();
        for h in hs {
            total += h.join().unwrap();
        }
        total
    })
}

#[derive(Default)]
struct Tally {
    pairs: u64,
    wrong: u64,
    first: Option<String>,
    counted: Counts,
    machine: Counts,
}

impl std::ops::AddAssign for Tally {
    fn add_assign(&mut self, o: Tally) {
        self.pairs += o.pairs;
        self.wrong += o.wrong;
        self.first = self.first.take().or(o.first);
        self.counted.add(o.counted);
        self.machine.add(o.machine);
    }
}

/// The aarch64 instructions on binary16 (FEAT_FP16), on raw codes, under
/// the process's default FPCR: to nearest even, no flush to zero.
#[cfg(target_arch = "aarch64")]
mod machine {
    use std::arch::asm;

    pub fn has_fp16() -> bool {
        std::arch::is_aarch64_feature_detected!("fp16")
    }

    macro_rules! binary {
        ($name:ident, $insn:literal) => {
            #[target_feature(enable = "fp16")]
            pub unsafe fn $name(a: u16, b: u16) -> u16 {
                let r: u16;
                unsafe { asm!(concat!($insn, " {r:h}, {a:h}, {b:h}"), a = in(vreg) a, b = in(vreg) b, r = lateout(vreg) r, options(pure, nomem, nostack)) };
                r
            }
        };
    }
    binary!(fadd, "fadd");
    binary!(fsub, "fsub");
    binary!(fmul, "fmul");
    binary!(fdiv, "fdiv");

    #[target_feature(enable = "fp16")]
    pub unsafe fn fsqrt(a: u16) -> u16 {
        let r: u16;
        unsafe { asm!("fsqrt {r:h}, {a:h}", a = in(vreg) a, r = lateout(vreg) r, options(pure, nomem, nostack)) };
        r
    }

    /// `fcvt h, s`: an `f32` to binary16.
    #[target_feature(enable = "fp16")]
    pub unsafe fn to_f16(a: f32) -> u16 {
        let r: u16;
        unsafe { asm!("fcvt {r:h}, {a:s}", a = in(vreg) a, r = lateout(vreg) r, options(pure, nomem, nostack)) };
        r
    }

    /// `fcvt s, h`: binary16 to an `f32`.
    #[target_feature(enable = "fp16")]
    pub unsafe fn from_f16(a: u16) -> f32 {
        let r: f32;
        unsafe { asm!("fcvt {r:s}, {a:h}", a = in(vreg) a, r = lateout(vreg) r, options(pure, nomem, nostack)) };
        r
    }
}

/// Whether a binary16 code is what the emulation gave: the same code, or
/// both NaN (the emulation keeps no payload).
#[cfg(target_arch = "aarch64")]
fn same_f16(code: u16, got: f64) -> bool {
    let want = F16.decode(code as u64);
    if want.is_nan() { got.is_nan() } else { F16.encode(got) == Some(code as u64) }
}

/// MT1: binary16's +, −, × and ÷ emulated by the kernel interpreter (each
/// operation in `f32`, rounded once to binary16) are the machine's `fadd`,
/// `fsub`, `fmul` and `fdiv` on all 2³² pairs of codes, and its √ the
/// machine's `fsqrt` on all 2¹⁶; the infinities and NaNs counted are those
/// the machine produced from finite or non-NaN operands. The lowering's own
/// kernels, on eight threads: about four minutes a release build.
#[cfg(target_arch = "aarch64")]
#[test]
#[ignore]
fn binary16_arithmetic_is_the_machine_s() {
    if !machine::has_fp16() {
        eprintln!("this machine has no binary16 arithmetic (FEAT_FP16): nothing checked");
        return;
    }
    let all: Vec<f64> = (0..1u64 << 16).map(|c| F16.decode(c)).collect();
    let ball = Buf::in_format(Format::Float(F16), &all);
    const CHUNK: usize = 32;
    for op in [PrimOp::Add, PrimOp::Sub, PrimOp::Mul, PrimOp::Div] {
        let k = pairs(op.clone(), CHUNK, all.len());
        let insn: unsafe fn(u16, u16) -> u16 = match op {
            PrimOp::Add => machine::fadd,
            PrimOp::Sub => machine::fsub,
            PrimOp::Mul => machine::fmul,
            _ => machine::fdiv,
        };
        let t = std::time::Instant::now();
        let tally = chunks(all.len(), CHUNK, |lo, hi| {
            let a = Buf::in_format(Format::Float(F16), &rows(&all[lo..hi], all.len()));
            let (got, counted) = emulate(&k, Format::Float(F16), &[&a, &ball], Format::Float(F16));
            let mut t = Tally { counted, ..Tally::default() };
            for (i, ca) in (lo..hi).enumerate() {
                let x = all[ca];
                for (cb, y) in all.iter().enumerate() {
                    let r = unsafe { insn(ca as u16, cb as u16) };
                    let g = got[i * all.len() + cb];
                    t.pairs += 1;
                    if !same_f16(r, g) {
                        t.wrong += 1;
                        t.first.get_or_insert_with(|| format!("{ca:#06x} {cb:#06x}: the machine {r:#06x}, the emulation {g:e}"));
                    }
                    let m = F16.decode(r as u64);
                    t.machine.infinities += (m.is_infinite() && x.is_finite() && y.is_finite()) as u64;
                    t.machine.nans += (m.is_nan() && !x.is_nan() && !y.is_nan()) as u64;
                }
            }
            t
        });
        eprintln!("binary16 {}: {} pairs, {} wrong, counted {}, the machine {} ({:.0} s)", op.name(), tally.pairs, tally.wrong, tally.counted, tally.machine, t.elapsed().as_secs_f64());
        assert_eq!(tally.pairs, 1 << 32);
        assert_eq!(tally.wrong, 0, "{}: {}", op.name(), tally.first.unwrap_or_default());
        assert_eq!(tally.counted, tally.machine, "{}", op.name());
    }
    let k = kernel(PrimOp::Sqrt, &[real(vec![ax("a", all.len())])], Dtype::F32);
    let (got, counted) = emulate(&k, Format::Float(F16), &[&ball], Format::Float(F16));
    let mut nans = 0;
    for (c, g) in got.iter().enumerate() {
        let r = unsafe { machine::fsqrt(c as u16) };
        assert!(same_f16(r, *g), "sqrt {c:#06x}: the machine {r:#06x}, the emulation {g:e}");
        nans += (F16.decode(r as u64).is_nan() && !all[c].is_nan()) as u64;
    }
    assert_eq!(counted, Counts { nans, ..Counts::default() });
    eprintln!("binary16 sqrt: 65536 codes, 0 wrong, counted {counted}");
}

/// MT1: the conversions to and from binary16 are the machine's `fcvt`: every
/// `f32` (2³²) to binary16, as a binary16 kernel loads it and as an `f32`
/// kernel stores it, and every binary16 code to `f32`.
#[cfg(target_arch = "aarch64")]
#[test]
#[ignore]
fn binary16_conversions_are_the_machine_s() {
    if !machine::has_fp16() {
        eprintln!("this machine has no binary16 arithmetic (FEAT_FP16): nothing checked");
        return;
    }
    const CHUNK: usize = 1 << 20;
    let k = kernel(PrimOp::Neg, &[real(vec![ax("a", CHUNK)])], Dtype::F32);
    let t = std::time::Instant::now();
    let tally = chunks(1 << 32, CHUNK, |lo, hi| {
        let xs: Vec<f64> = (lo..hi).map(|b| f32::from_bits(b as u32) as f64).collect();
        let b = Buf::of(Dtype::F32, &xs);
        // −x is exact in every format: the negation shows the conversion.
        let (loaded, c1) = emulate(&k, Format::Float(F16), &[&b], Format::Float(F16));
        let (stored, c2) = emulate(&k, Format::Float(F32), &[&b], Format::Float(F16));
        let mut t = Tally { counted: c1, ..Tally::default() };
        t.counted.add(c2);
        for (i, x) in xs.iter().enumerate() {
            let r = unsafe { machine::to_f16(-(*x as f32)) };
            t.pairs += 1;
            for g in [loaded[i], stored[i]] {
                if !same_f16(r, g) {
                    t.wrong += 1;
                    t.first.get_or_insert_with(|| format!("{:#010x}: the machine {r:#06x}, the emulation {g:e}", lo + i));
                }
            }
            t.machine.infinities += 2 * (F16.decode(r as u64).is_infinite() && x.is_finite()) as u64;
        }
        t
    });
    eprintln!("f32 to binary16: {} values, {} wrong, counted {}, the machine {} ({:.0} s)", tally.pairs, tally.wrong, tally.counted, tally.machine, t.elapsed().as_secs_f64());
    assert_eq!(tally.pairs, 1 << 32);
    assert_eq!(tally.wrong, 0, "{}", tally.first.unwrap_or_default());
    assert_eq!(tally.counted, tally.machine);
    let all: Vec<f64> = (0..1u64 << 16).map(|c| F16.decode(c)).collect();
    let k = kernel(PrimOp::Neg, &[real(vec![ax("a", all.len())])], Dtype::F32);
    let (got, counted) = emulate(&k, Format::Float(F32), &[&Buf::in_format(Format::Float(F16), &all)], Format::Float(F32));
    assert!(counted.is_zero());
    for (c, g) in got.iter().enumerate() {
        let r = unsafe { machine::from_f16(c as u16 ^ 0x8000) } as f64;
        assert!(r.to_bits() == g.to_bits() || (r.is_nan() && g.is_nan()), "{c:#06x}: the machine {r:e}, the emulation {g:e}");
    }
    eprintln!("binary16 to f32: 65536 codes, 0 wrong");
}

/// MT1: `bf16`'s conversions emulated are the runtime's of 0.17.1: every
/// `f32` stored by an `f32` kernel into a `bf16` buffer is
/// `presk_rt::f32_to_bf16`'s code (a NaN a NaN), and every `bf16` code
/// loaded is `bf16_to_f32`'s value.
#[test]
#[ignore]
fn bf16_conversions_are_the_runtime_s() {
    const CHUNK: usize = 1 << 20;
    let k = kernel(PrimOp::Neg, &[real(vec![ax("a", CHUNK)])], Dtype::F32);
    let tally = chunks(1 << 32, CHUNK, |lo, hi| {
        let xs: Vec<f64> = (lo..hi).map(|b| f32::from_bits(b as u32) as f64).collect();
        let mut out = Buf::Bf16(vec![0u16; hi - lo].into());
        let counted = run_kernel_in(&k, Format::Float(F32), &[&Buf::of(Dtype::F32, &xs)], &mut out).unwrap();
        let Buf::Bf16(got) = out else { unreachable!() };
        let mut t = Tally { counted, ..Tally::default() };
        for (i, x) in xs.iter().enumerate() {
            let want = presk_rt::f32_to_bf16(-(*x as f32));
            let nan = |c: u16| c & 0x7f80 == 0x7f80 && c & 0x7f != 0;
            t.pairs += 1;
            if !(got[i] == want || (nan(got[i]) && nan(want))) {
                t.wrong += 1;
                t.first.get_or_insert_with(|| format!("{:#010x}: {want:#06x}, the emulation {:#06x}", lo + i, got[i]));
            }
            t.machine.infinities += (BF16.decode(want as u64).is_infinite() && x.is_finite()) as u64;
        }
        t
    });
    eprintln!("f32 to bf16: {} values, {} wrong, counted {}, the runtime's overflows {}", tally.pairs, tally.wrong, tally.counted, tally.machine);
    assert_eq!(tally.pairs, 1 << 32);
    assert_eq!(tally.wrong, 0, "{}", tally.first.unwrap_or_default());
    assert_eq!(tally.counted, tally.machine);
    let all: Vec<f64> = (0..1u32 << 16).map(|c| presk_rt::bf16_to_f32(c as u16) as f64).collect();
    let k = kernel(PrimOp::Neg, &[real(vec![ax("a", all.len())])], Dtype::BF16);
    let codes = Buf::Bf16((0..1u32 << 16).map(|c| c as u16).collect());
    let mut out = Buf::F32(vec![0.0; all.len()].into());
    run_kernel_in(&k, Format::Float(F32), &[&codes], &mut out).unwrap();
    for (x, g) in all.iter().zip(out.to_f64()) {
        assert!((-x).to_bits() == g.to_bits() || (x.is_nan() && g.is_nan()), "{x:e}");
    }
    eprintln!("bf16 to f32: 65536 codes, 0 wrong");
}

