//! Quantized activations (plan 0.18.0 step 4, D207): a value stored in an
//! affine format is rounded into it once by its rescaling, every result held
//! to the exact quotient's rounding computed apart with GMP's integers, the
//! rescaling's other readings to binary32's own operations; a value of one is
//! read as it is; a sum of products of scaled values whose product and
//! accumulation formats are integers counts in their unit, the integer sum a
//! machine's integer unit computes, wrapped or clamped as its accumulator is;
//! the shift form is the rounding shift of every 16-bit integer.

use std::cmp::Ordering;

use presk_core::PrimOp;
use presk_format::{Affine, Float, Format, Int, IntOverflow, Tie, E4M3_SATURATING, F16, F32, F64};
use presk_kernel::exec::{run_emulated, run_kernel_mixed, Counts};
use presk_kernel::ir::{Arith, BinOp, BufSpec, Expr, Stmt};
use presk_kernel::lower::{lower, lower_ordered, Operand};
use presk_kernel::{BufKind, Dtype, Kernel, Threads};
use presk_rt::{Axis, Buf, QBuf, ReductionOrder};
use rug::Integer;

fn fmt(s: &str) -> Format {
    Format::parse(s).unwrap()
}

fn int(s: &str) -> Int {
    Int::parse(s).unwrap()
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

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// An integer of `lo..=hi`.
    fn int(&mut self, lo: i128, hi: i128) -> i128 {
        lo + (self.next() as u128 % (hi - lo + 1) as u128) as i128
    }

    /// A value of `f`, its codes drawn uniformly.
    fn value(&mut self, f: Float) -> f64 {
        f.decode(self.next() & (u64::MAX >> (64 - f.bits())))
    }

    /// A positive normal binary32 scale: a power of two, or any.
    fn scale(&mut self) -> f32 {
        let e = self.int(-40, 10) as i32;
        let m = if self.below(3) == 0 { 0 } else { self.next() as u32 & 0x7f_ffff };
        f32::from_bits((((e + 127) as u32) << 23) | m)
    }
}

/// `m·2^e` exactly, as GMP's integer and an exponent.
fn exact(x: f64) -> (Integer, i32) {
    let bits = x.abs().to_bits();
    let (be, frac) = ((bits >> 52) as i32, bits & ((1u64 << 52) - 1));
    let (m, e) = if be == 0 { (frac, -1074) } else { (frac | (1u64 << 52), be - 1075) };
    (if x < 0.0 { -Integer::from(m) } else { Integer::from(m) }, e)
}

/// The integer nearest `num/den` under `tie`, `den > 0`, plus `z0` where the
/// tie is decided on the sum.
fn nearest(num: Integer, den: &Integer, z0: i128, tie: Tie) -> Integer {
    let num = num + Integer::from(z0) * den;
    let (q, r) = num.div_rem_floor(den.clone());
    let up = match (Integer::from(&r * 2u32)).cmp(den) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => match tie {
            Tie::Even => q.is_odd(),
            Tie::Away => q >= 0,
        },
    };
    q + u32::from(up) - Integer::from(z0)
}

/// The code of the real `m·2^e` rescaled into `a`, exactly: the integer
/// nearest the quotient by GMP's integers, the zero point, the clamp.
fn rescaled_exact(a: Affine, m: Integer, e: i32) -> (i128, bool) {
    let (lo, hi, z) = (a.int.min(), a.int.max(), i128::from(a.zero));
    let z0 = if a.zero_rounded { z } else { 0 };
    let (sm, se) = exact(a.scale());
    let (mut num, mut den) = (m, sm);
    if e >= se {
        num <<= (e - se) as u32;
    } else {
        den <<= (se - e) as u32;
    }
    let n = nearest(num, &den, z0, a.tie) + Integer::from(z);
    let q = n.clone().clamp(&Integer::from(lo), &Integer::from(hi));
    (q.to_i128().unwrap(), q != n)
}

/// The code of `x` rescaled into `a`: a NaN the zero point, an infinity the
/// end of its sign; in binary32 as Rust's `f32` operations compute it where
/// the format says so.
fn rescaled(a: Affine, x: f64) -> (i128, bool) {
    let (lo, hi, z) = (a.int.min(), a.int.max(), i128::from(a.zero));
    if x.is_nan() {
        return (z, true);
    }
    if x.is_infinite() {
        return (if x > 0.0 { hi } else { lo }, true);
    }
    if let Some(f) = a.quotient {
        assert_eq!(f, F32);
        let t = x as f32 / a.scale() as f32;
        let t = if a.zero_rounded { t + z as f32 } else { t };
        if !t.is_finite() {
            return (if t > 0.0 { hi } else { lo }, true);
        }
        let n = match a.tie {
            Tie::Even => (t as f64).round_ties_even(),
            Tie::Away => (t as f64).round(),
        }
        .clamp(-1e30, 1e30) as i128;
        let q = if a.zero_rounded { n } else { n + z };
        let c = q.clamp(lo, hi);
        return (c, c != q);
    }
    let (m, e) = if x == 0.0 { (Integer::new(), 0) } else { exact(x) };
    rescaled_exact(a, m, e)
}

fn spec(n: usize) -> BufSpec {
    BufSpec { dims: vec![n], kind: BufKind::Real, names: vec![None] }
}

/// `%1[i] = %0[i]` over `n` elements, of storage `f64`, in `arith`.
fn copy(n: usize, arith: Arith) -> Kernel {
    let at = || vec![Expr::Iv(0)];
    Kernel { name: "copy".into(), dtype: Dtype::F64, arith, inputs: vec![spec(n)], output: spec(n), nloops: 1, nreal: 0, nint: 0, nvec: 0, lanes: 1, body: vec![Stmt::For { var: 0, bound: n, body: vec![Stmt::Store { buf: 1, index: at(), value: Expr::Load(0, at()) }] }], threads: Threads::One }
}

/// A random affine format: its integer, scale and zero point, its tie, where
/// the zero point is added, and the quotient's format.
fn draw_affine(rng: &mut Rng) -> Affine {
    let ints = ["i8/sat", "u8/sat", "i4/sat", "u4/sat", "i16/sat", "u16/sat", "i2/sat"];
    loop {
        let i = int(ints[rng.below(ints.len() as u64) as usize]);
        let zero = if rng.below(2) == 0 { 0 } else { rng.int(i.min(), i.max()) as i64 };
        let a = Affine { int: i, scale_bits: rng.scale().to_bits(), zero, tie: if rng.below(2) == 0 { Tie::Even } else { Tie::Away }, zero_rounded: rng.below(3) == 0, quotient: if rng.below(4) == 0 { Some(F32) } else { None } };
        if let Ok(a) = a.check() {
            return a;
        }
    }
}

/// The `f64` next to `x`, above it or below it.
fn beside(x: f64, up: bool) -> f64 {
    if x == 0.0 {
        return if up { f64::from_bits(1) } else { -f64::from_bits(1) };
    }
    let b = x.to_bits();
    f64::from_bits(if (x > 0.0) == up { b + 1 } else { b - 1 })
}

/// Values about `a`'s range: its grid's ties exactly and beside them, codes
/// of several formats scaled to it, and the values no number is.
fn draw_values(rng: &mut Rng, a: Affine, n: usize) -> Vec<f64> {
    let (s, z) = (a.scale(), a.zero as f64);
    let (lo, hi) = (a.int.min() as f64 - z, a.int.max() as f64 - z);
    (0..n)
        .map(|i| match i % 8 {
            // A tie of the grid, or of the grid shifted by the zero point.
            0 => (rng.int(lo as i128 - 2, hi as i128 + 2) as f64 + 0.5) * s,
            1 => beside((rng.int(lo as i128 - 2, hi as i128 + 2) as f64 + 0.5) * s, true),
            2 => beside((rng.int(lo as i128 - 2, hi as i128 + 2) as f64 + 0.5) * s, false),
            3 => rng.value(F64),
            4 => rng.value(F32) * s,
            5 => rng.value(F16) * s,
            6 => rng.value(E4M3_SATURATING) * s * 4.0,
            _ => [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0, f64::MIN_POSITIVE, 1e300, -1e-300][rng.below(8) as usize],
        })
        .collect()
}

/// A value stored in an affine format is rounded into it once by its
/// rescaling: on both of the emulation's paths, against the exact quotient's
/// rounding (the binary32 quotient's where the format says), every clamp
/// counted; and a value of one is read as it is.
#[test]
fn a_store_rounds_once_into_an_affine_format() {
    let mut rng = Rng(0x0018_0004_0001);
    let n = 400;
    for _ in 0..600 {
        let a = draw_affine(&mut rng);
        let xs = draw_values(&mut rng, a, n);
        let input = Buf::F64(xs.clone().into());
        let k = copy(n, Arith { store: Some(Format::Affine(a)), ..Arith::one(Format::Float(F64)) });
        let (mut one, mut several) = (Buf::Format(Format::Affine(a), vec![0.0; n].into()), Buf::Format(Format::Affine(a), vec![0.0; n].into()));
        let c1 = run_emulated(&k, &[&input], &mut one).unwrap();
        let c2 = run_kernel_mixed(&k, &[&input], &mut several).unwrap();
        let mut clamps = 0;
        for (i, x) in xs.iter().enumerate() {
            let (q, o) = rescaled(a, *x);
            clamps += u64::from(o);
            let want = a.value(q);
            assert_eq!(one.to_f64()[i].to_bits(), want.to_bits(), "{x:e} into {a}: one format");
            assert_eq!(several.to_f64()[i].to_bits(), want.to_bits(), "{x:e} into {a}: several");
        }
        assert_eq!(c1, Counts { saturations: clamps, ..Counts::default() }, "{a}");
        assert_eq!(c1, c2, "{a}");
        // Read as it is: its values copied into `f64` unchanged.
        let k = copy(n, Arith::one(Format::Float(F64)));
        let (mut back, mut back2) = (Buf::F64(vec![0.0; n].into()), Buf::F64(vec![0.0; n].into()));
        run_emulated(&k, &[&one], &mut back).unwrap();
        run_kernel_mixed(&k, &[&one], &mut back2).unwrap();
        assert_eq!(back.to_f64(), one.to_f64(), "{a}");
        assert_eq!(back2.to_f64(), one.to_f64(), "{a}");
    }
}

/// The shift form, a target's rescaling of a 16-bit integer to an 8-bit one
/// by a right shift of `k` bits, ties to even, clamped: an affine format of
/// scale `2^k`, on every `i16` and every shift from 0 to 15, against the
/// shift written with integers.
#[test]
fn the_shift_form_is_the_rounding_shift() {
    let xs: Vec<f64> = (i16::MIN..=i16::MAX).map(f64::from).collect();
    let i16w = fmt("i16");
    let input = Buf::in_format(i16w, &xs);
    for k in 0..16 {
        let a = Affine::new(int("i8/sat"), (1u32 << k) as f32, 0).unwrap();
        assert_eq!(a.name(), format!("affine(i8/sat,scale=2^{k})"));
        let kern = copy(xs.len(), Arith { store: Some(Format::Affine(a)), ..Arith::one(Format::Float(F32)) });
        let mut out = Buf::Format(Format::Affine(a), vec![0.0; xs.len()].into());
        let c = run_emulated(&kern, &[&input], &mut out).unwrap();
        let mut clamps = 0;
        for (i, x) in (i16::MIN..=i16::MAX).enumerate() {
            let x = i32::from(x);
            let r = if k == 0 { x } else { (x + (1 << (k - 1)) - 1 + ((x >> k) & 1)) >> k };
            let q = r.clamp(-128, 127);
            clamps += u64::from(q != r);
            assert_eq!(out.to_f64()[i], f64::from(q) * f64::from(1u32 << k), "{x} >> {k}");
        }
        assert_eq!(c.saturations, clamps, "shift {k}");
    }
}

fn ax(name: &str, len: usize) -> Axis {
    Axis { name: Some(name.into()), len }
}

/// Integer weights of `bits` over axis 0 of a `[k, n]` matrix, a scale per
/// column.
fn weights(q: &[i8], scales: &[f32], k: usize) -> Buf {
    Buf::Quant(std::sync::Arc::new(QBuf { bits: 8, over: 0, k, inner: scales.len(), data: q.iter().map(|x| *x as u8).collect(), scales: scales.to_vec() }))
}

/// A sum of products of scaled values counts in their unit: an activation
/// in an affine format of eight bits, its zero point drawn, by integer
/// weights with a scale per column, the products in `i32` and the sum in a
/// wrapping `i32`, a wrapping `i16`, a clamping one and a fixed-point
/// `i32` of four fractional bits, sequentially and in
/// chunks; the result `s_a·s_w·n` rounded once into binary16, or rescaled
/// into an affine format, against the integer sum written out; every wrap
/// and clamp counted.
#[test]
fn a_sum_of_scaled_products_counts_in_their_unit() {
    let mut rng = Rng(0x0018_0004_0002);
    let (m, kk, nn) = (3, 48, 5);
    let x = Operand { axes: vec![ax("i", m), ax("k", kk)], kind: BufKind::Real };
    let w = Operand { axes: vec![ax("k", kk), ax("j", nn)], kind: BufKind::Real };
    let mut seen = Counts::default();
    for trial in 0..300 {
        let ai = if trial % 2 == 0 { int("u8/sat") } else { int("i8/sat") };
        let a = Affine::new(ai, rng.scale(), rng.int(ai.min(), ai.max()) as i64).unwrap();
        let codes: Vec<i128> = (0..m * kk).map(|_| rng.int(ai.min(), ai.max())).collect();
        let xs: Vec<f64> = codes.iter().map(|q| a.value(*q)).collect();
        let qw: Vec<i8> = (0..kk * nn).map(|_| rng.int(-127, 127) as i8).collect();
        let sw: Vec<f32> = (0..nn).map(|_| rng.scale()).collect();
        let (bx, bw) = (Buf::Format(Format::Affine(a), xs.into()), weights(&qw, &sw, kk));
        let accum = [fmt("i32"), fmt("i16"), fmt("i16/sat"), fmt("fixed(i32,frac=4)")][trial % 4];
        let chunk = [None, Some(8u32), Some(20)][(trial / 3) % 3];
        let out_affine = Affine::absmax(int("i8/sat"), (rng.below(1 << 20) as f64 + 1.0) * a.scale() * sw[0] as f64 * 64.0).unwrap();
        let store = if (trial / 4) % 2 == 0 { None } else { Some(Format::Affine(out_affine)) };
        let mut k = match chunk {
            None => lower(&PrimOp::Dot, &[x.clone(), w.clone()], Dtype::new(F16)).unwrap().unwrap().kernel,
            Some(c) => lower_ordered(&PrimOp::Dot, &[x.clone(), w.clone()], Dtype::new(F16), ReductionOrder::Chunked { size: c }).unwrap().unwrap().kernel,
        };
        k.arith = Arith { arith: Format::Float(F32), product: fmt("i32"), accum, store };
        let mut out = match store {
            Some(f) => Buf::Format(f, vec![0.0; m * nn].into()),
            None => Buf::in_format(Format::Float(F16), &vec![0.0; m * nn]),
        };
        let c = run_emulated(&k, &[&bx, &bw], &mut out).unwrap();
        // A fixed-point accumulator counts sixteenths of the unit, never
        // past its range here: the sum is exact.
        let acc = match accum {
            Format::Int(i) => i,
            Format::Fixed(_) => int("i64"),
            _ => unreachable!(),
        };
        let span = 1i128 << acc.bits;
        let fit = |v: i128, counts: &mut Counts| -> i128 {
            let (lo, hi) = (acc.min(), acc.max());
            if (lo..=hi).contains(&v) {
                return v;
            }
            match acc.overflow {
                IntOverflow::Clamp => {
                    counts.saturations += 1;
                    v.clamp(lo, hi)
                }
                IntOverflow::Wrap => {
                    counts.wraps += 1;
                    (v - lo).rem_euclid(span) + lo
                }
            }
        };
        let mut want_counts = Counts::default();
        for r in 0..m {
            for col in 0..nn {
                let terms: Vec<i128> = (0..kk).map(|t| (codes[r * kk + t] - i128::from(a.zero)) * i128::from(qw[t * nn + col])).collect();
                let n = match chunk {
                    None => terms.iter().fold(0, |s, t| fit(s + t, &mut want_counts)),
                    Some(c) => terms.chunks(c as usize).map(|ch| ch.iter().fold(0, |s, t| fit(s + t, &mut want_counts))).collect::<Vec<_>>().into_iter().fold(0, |s, p| fit(s + p, &mut want_counts)),
                };
                // `s_a·s_w·n` exactly, as `m·2^e`: the two scales' product
                // is an `f64`.
                let (mu, eu) = exact(a.scale() * sw[col] as f64);
                let mm = Integer::from(n) * mu;
                let got = out.to_f64()[r * nn + col];
                match store {
                    None => {
                        let (y, o) = F16.round_scaled(mm.to_i128().unwrap(), eu, Tie::Even);
                        want_counts.infinities += u64::from(o);
                        assert_eq!(got.to_bits(), y.to_bits(), "trial {trial} [{r}, {col}]: n = {n}");
                    }
                    Some(_) => {
                        let (q, o) = rescaled_exact(out_affine, mm, eu);
                        want_counts.saturations += u64::from(o);
                        assert_eq!(got.to_bits(), out_affine.value(q).to_bits(), "trial {trial} [{r}, {col}]: n = {n} into {out_affine}");
                    }
                }
            }
        }
        assert_eq!(c, want_counts, "trial {trial}: {accum}, {chunk:?}");
        seen.add(c);
    }
    // Every rule met: wraps, clamps of the sum and of the rescaling,
    // binary16's overflows.
    assert!(seen.wraps > 0 && seen.saturations > 0 && seen.infinities > 0, "{seen}");
}

/// What a nest of scaled values is refused: an operation rounded to an
/// affine format; products counted in their unit summed in a float; a count
/// that an operation reads before it is stored, where an `f64` does not
/// hold its value; and an affine format whose values an `f64` does not hold.
#[test]
fn what_a_nest_of_scaled_values_is_refused() {
    let n = 4;
    let a = Affine::new(int("i8/sat"), 0.1, 0).unwrap();
    let xs: Vec<f64> = [100, -97, 113, 101].iter().map(|q| a.value(*q)).collect();
    let bx = Buf::Format(Format::Affine(a), xs.into());
    let mut out = Buf::F64(vec![0.0; n].into());
    let at = || vec![Expr::Iv(0)];
    let kernel = |value: Expr, arith: Arith| Kernel { name: "op".into(), dtype: Dtype::F64, arith, inputs: vec![spec(n), spec(n)], output: spec(n), nloops: 1, nreal: 0, nint: 0, nvec: 0, lanes: 1, body: vec![Stmt::For { var: 0, bound: n, body: vec![Stmt::Store { buf: 2, index: at(), value }] }], threads: Threads::One };
    let product = || Expr::Bin(BinOp::Product, Box::new(Expr::Load(0, at())), Box::new(Expr::Load(1, at())));
    let e = run_emulated(&kernel(product(), Arith::one(Format::Affine(a))), &[&bx, &bx], &mut out).unwrap_err();
    assert!(e.contains("a stored value's, not an operation's"), "{e}");
    let e = run_emulated(&kernel(product(), Arith { arith: Format::Float(F32), product: fmt("i32"), accum: Format::Float(F32), store: None }), &[&bx, &bx], &mut out).unwrap_err();
    assert!(e.contains("summed in an integer or a fixed-point format"), "{e}");
    // The product stored as it is: rounded once into `f64`.
    let counted = Arith { arith: Format::Float(F32), product: fmt("i32"), accum: fmt("i32"), store: None };
    assert_eq!(run_emulated(&kernel(product(), counted), &[&bx, &bx], &mut out), Ok(Counts::default()));
    let unit = 0.1f32 as f64 * 0.1f32 as f64;
    assert_eq!(out.to_f64()[0], F64.round_scaled(10000 * exact(unit).0.to_i128().unwrap(), exact(unit).1, Tie::Even).0);
    // Read by an addition first: 10 000 units of a 48-bit unit.
    let read = Expr::Bin(BinOp::Add, Box::new(product()), Box::new(Expr::F(0.5)));
    let e = run_emulated(&kernel(read, counted), &[&bx, &bx], &mut out).unwrap_err();
    assert!(e.contains("before it stores it"), "{e}");
    assert!(Format::parse("affine(i32/sat,scale=0.1)").unwrap_err().contains("53 bits"));
    // A kernel that reads back what it stores: the derivative of `embed`,
    // whose sums live in its output, which an affine format would hold as
    // codes.
    let table = Operand { axes: vec![ax("v", 5), ax("d", 3)], kind: BufKind::Real };
    let idx = Operand { axes: vec![ax("r", 2)], kind: BufKind::Idx };
    let ct = Operand { axes: vec![ax("r", 2), ax("d", 3)], kind: BufKind::Real };
    let mut k = lower(&PrimOp::EmbedVjp, &[table, idx, ct], Dtype::F32).unwrap().unwrap().kernel;
    k.arith = Arith { store: Some(Format::Affine(a)), ..Arith::one(Format::Float(F32)) };
    let mut q = Buf::Format(Format::Affine(a), vec![0.0; 15].into());
    let (bt, bi, bc) = (Buf::of(Dtype::F32, &[0.0; 15]), Buf::I64(vec![1, 3].into()), Buf::of(Dtype::F32, &[0.5; 6]));
    let e = run_emulated(&k, &[&bt, &bi, &bc], &mut q).unwrap_err();
    assert!(e.contains("reads back what it stores"), "{e}");
}
