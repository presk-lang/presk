//! FC1 (plan 0.18.2 step 1, D215): the conversions between `f32` and
//! binary16 and between `f32` and `e8m15`, and from `f64` to each, are the
//! formats' rounding (`presk_format`'s, which MT1 pinned to the machine's
//! for binary16, D205), on every code and every input checked here; a
//! buffer built from `f64`s holds what the emulation's `Float::store` gives.
//!
//! The exhaustive checks are the ignored tests (`cargo test --release -p
//! presk-rt --test codes -- --ignored --nocapture`): every `f32` (2³²)
//! narrowed to each format, every `e8m15` code (2²⁴) widened, every
//! midpoint between two consecutive numbers of each format with the `f64`s
//! on either side narrowed from `f64`, and 2³² random `f64`s; and on
//! aarch64, binary16's conversions against `fcvt` bit for bit, NaNs'
//! payloads included. The tests that run by default take the same checks
//! on a sample.

use presk_format::{Float, Tie, E8M15, F16};
use presk_rt::codes::{e8m15_to_f32, f16_to_f32, f32_to_e8m15, f32_to_f16, f64_to_e8m15, f64_to_f16, overflows};
use presk_rt::{Buf, Dtype};

/// The code the format's rounding gives `x`, to nearest, ties to even; for
/// a NaN, the quiet NaN of its sign with the top bits of its payload, `t`
/// of the code's trailing bits, `t − 1` of them the payload's.
fn want(f: Float, x: f64) -> u64 {
    if x.is_nan() {
        let (t, b) = (f.p - 1, x.to_bits());
        let sign = (b >> 63) << (f.bits() - 1);
        return sign | ((1u64 << f.exp_bits()) - 1) << t | 1 << (t - 1) | (b >> (52 - t)) & ((1 << (t - 1)) - 1);
    }
    f.encode(f.round(x, Tie::Even)).expect("a rounded value is a value of the format")
}

/// The bits of the `f64` equal to an `f32`, a NaN's moved without a
/// conversion.
fn wide(x: f32) -> f64 {
    let b = u64::from(x.to_bits());
    if x.is_nan() { f64::from_bits((b >> 31) << 63 | 0x7ff << 52 | (b & 0x7f_ffff) << 29) } else { f64::from(x) }
}

/// `check(i)` for every `i` in `lo..hi` on every core; the inputs that
/// failed (the first eight) and how many did.
fn every(lo: u64, hi: u64, check: impl Fn(u64) -> bool + Sync) -> (Vec<u64>, u64) {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()) as u64;
    let per = (hi - lo).div_ceil(threads);
    let parts: Vec<(Vec<u64>, u64)> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let check = &check;
                s.spawn(move || {
                    let (mut first, mut n) = (Vec::new(), 0u64);
                    for i in (lo + t * per)..(lo + (t + 1) * per).min(hi) {
                        if !check(i) {
                            n += 1;
                            if first.len() < 8 {
                                first.push(i);
                            }
                        }
                    }
                    (first, n)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("a checking thread")).collect()
    });
    let n = parts.iter().map(|p| p.1).sum();
    (parts.into_iter().flat_map(|p| p.0).take(8).collect(), n)
}

fn assert_none(what: &str, (first, n): (Vec<u64>, u64), shown: impl Fn(u64) -> String) {
    assert!(n == 0, "{what}: {n} wrong, the first {}", first.iter().map(|i| shown(*i)).collect::<Vec<_>>().join(", "));
}

/// Narrowing an `f32` of bits `b` to both formats is their rounding.
fn f32_narrows(b: u32) -> bool {
    let x = f32::from_bits(b);
    u64::from(f32_to_f16(x)) == want(F16, wide(x)) && u64::from(f32_to_e8m15(x)) == want(E8M15, wide(x))
}

/// Narrowing an `f64` to both formats is their rounding, once.
fn f64_narrows(x: f64) -> bool {
    u64::from(f64_to_f16(x)) == want(F16, x) && u64::from(f64_to_e8m15(x)) == want(E8M15, x)
}

/// The `f32`s of a sample of bit patterns: a stride through all 2³², and
/// every pattern within 64 of each binary16 and `e8m15` rounding boundary
/// that an `f32` meets: each number of either format, the midpoints, the
/// overflow thresholds.
fn f32_sample() -> Vec<u32> {
    let mut v: Vec<u32> = (0..1u64 << 32).step_by(65_521).map(|b| b as u32).collect();
    for c in (0..1u32 << 16).step_by(7) {
        let x = f16_to_f32(c as u16);
        if x.is_finite() {
            let b = x.to_bits();
            v.extend((b.saturating_sub(64)..=b.saturating_add(64)).chain([b ^ 0x8000_0000]));
            // The midpoint above, in `f32`: 13 bits below binary16's last.
            v.extend([b + 0x1000, b + 0x0fff, b + 0x1001]);
        }
    }
    for c in (0..1u32 << 24).step_by(4093) {
        let b = c << 8;
        v.extend([b, b + 0x7f, b + 0x80, b + 0x81, b.wrapping_sub(1)]);
    }
    // 65 504 + half an ulp, the binary16 overflow threshold, and `e8m15`'s.
    for t in [65_520f32.to_bits(), 0x7f7f_ff80] {
        v.extend((t - 64..=t + 64).flat_map(|b| [b, b | 0x8000_0000]));
    }
    v
}

#[test]
fn binary16_widens_every_code_exactly() {
    for c in 0..=u16::MAX {
        let (got, w) = (f16_to_f32(c), F16.decode(u64::from(c)));
        if w.is_nan() {
            // `fcvt s, h` quiets a NaN and keeps its payload.
            let want = u32::from(c & 0x8000) << 16 | 0x7fc0_0000 | u32::from(c & 0x3ff) << 13;
            assert_eq!(got.to_bits(), want, "{c:#06x}");
        } else {
            assert_eq!(f64::from(got).to_bits(), w.to_bits(), "{c:#06x}");
        }
    }
}

#[test]
fn e8m15_widens_every_code_exactly() {
    let r = every(0, 1 << 24, |c| {
        let (got, w) = (e8m15_to_f32(c as u32), E8M15.decode(c));
        // A NaN keeps its 24 bits, as `bf16`'s does.
        if w.is_nan() { got.is_nan() && got.to_bits() == (c as u32) << 8 } else { f64::from(got).to_bits() == w.to_bits() }
    });
    assert_none("e8m15 widened", r, |c| format!("{c:#08x}"));
}

#[test]
fn narrowing_an_f32_is_the_format_s_rounding_on_a_sample() {
    let v = f32_sample();
    let r = every(0, v.len() as u64, |i| f32_narrows(v[i as usize]));
    assert_none("f32 narrowed", r, |i| format!("{:#010x}", v[i as usize]));
}

/// Every midpoint between two consecutive numbers of `f` of one sign, the
/// zero and the least subnormal's and the largest finite's and the
/// overflow threshold's included, and the `f64`s on either side of it:
/// codes `lo..hi` of the positive ones, each with its sign flipped too.
fn midpoints(f: Float, c: u64) -> [f64; 6] {
    let (a, b) = (f.decode(c), if c + 1 == f.encode(f64::INFINITY).unwrap() { 2f64.powi(f.emax + 1) } else { f.decode(c + 1) });
    let m = (a + b) / 2.0;
    let (below, above) = (f64::from_bits(m.to_bits() - 1), f64::from_bits(m.to_bits() + 1));
    [m, below, above, -m, -below, -above]
}

fn midpoints_round_once(f: Float, codes: impl Iterator<Item = u64>) -> (Vec<u64>, u64) {
    let codes: Vec<u64> = codes.collect();
    every(0, codes.len() as u64, |i| midpoints(f, codes[i as usize]).into_iter().all(f64_narrows))
}

#[test]
fn narrowing_an_f64_rounds_once_at_every_binary16_midpoint() {
    let top = F16.encode(f64::INFINITY).unwrap();
    let r = midpoints_round_once(F16, 0..top);
    assert_none("binary16's midpoints", r, |c| format!("{c:#06x}"));
}

#[test]
fn narrowing_an_f64_rounds_once_on_a_sample() {
    let top = E8M15.encode(f64::INFINITY).unwrap();
    let r = midpoints_round_once(E8M15, (0..top).step_by(257).chain(top - 64..top));
    assert_none("e8m15's midpoints", r, |c| format!("{c:#08x}"));
    let r = every(0, 1 << 18, |i| random(i).into_iter().all(f64_narrows));
    assert_none("random f64s", r, |i| format!("{:?}", random(i)));
}

/// Two random `f64`s of seed `i`: one of uniform bits, and one of a random
/// sign and significand and an exponent where the formats' numbers lie,
/// from below half their least subnormal to past their largest.
fn random(i: u64) -> [f64; 2] {
    let mix = |mut z: u64| {
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    let (a, b) = (mix(i.wrapping_mul(0x9e37_79b9_7f4a_7c15)), mix(i ^ 0x5851_f42d_4c95_7f2d));
    let e = (b >> 52) % 300;
    [f64::from_bits(a), f64::from_bits((b & (1 << 63)) | (1023 - 160 + e) << 52 | (a & ((1 << 52) - 1)))]
}

/// A buffer of binary16 or `e8m15` built from `f64`s holds its codes, each
/// value the one the emulation's `Float::store` gives (D206), so that a
/// native buffer and the emulation's hold the same numbers.
#[test]
fn a_buffer_from_f64s_holds_what_store_gives() {
    let xs: Vec<f64> = (0..1u64 << 12).flat_map(random).chain(midpoints(F16, 0x3c00)).chain(midpoints(E8M15, 0x3f8000)).chain([0.0, -0.0, f64::INFINITY, -f64::INFINITY, 65_520.0, 1e-8]).collect();
    for (d, f) in [(Dtype::F16, F16), (Dtype::E8M15, E8M15)] {
        let b = Buf::of(d, &xs);
        assert!(matches!(&b, Buf::Codes(g, e) if *g == f && e.len() == xs.len() * d.bytes()), "{d}: {}", b.dtype());
        assert_eq!(b.len(), xs.len());
        for (x, y) in xs.iter().zip(b.to_f64()) {
            if x.is_nan() {
                assert!(y.is_nan(), "{d}: {x:e}");
            } else {
                assert_eq!(y.to_bits(), f.store(*x).to_bits(), "{d}: {x:e}");
            }
        }
        assert_eq!(Buf::in_format(presk_format::Format::Float(f), &xs), b, "{d}");
        assert_eq!(Buf::zeros(d, 3).to_f64(), [0.0; 3]);
        assert_eq!(b.dtype(), d.name());
    }
    // A NaN is made quiet, its sign and its payload's top bits kept: a
    // signalling one of payload bit 50 is binary16's 0xff00.
    let nan = f64::from_bits(0xfff4_0000_0000_0001);
    assert_eq!(presk_rt::codes::code(F16, match &Buf::of(Dtype::F16, &[nan]) { Buf::Codes(_, e) => e, _ => unreachable!() }, 0), 0xff00);
}

/// The overflow guard's test (plan 0.18.2 step 4, D218) on an `f64`: it
/// holds exactly when `x` is finite and its code in the format is an
/// infinity's, for both formats.
fn guard_is_the_overflow(x: f64) -> bool {
    let inf = |f: Float, c: u64| c & !(1 << (f.bits() - 1)) == f.encode(f64::INFINITY).unwrap();
    overflows(F16, x) == (x.is_finite() && inf(F16, u64::from(f64_to_f16(x)))) && overflows(E8M15, x) == (x.is_finite() && inf(E8M15, u64::from(f64_to_e8m15(x))))
}

#[test]
fn the_guard_s_test_is_the_overflow_on_a_sample() {
    let v = f32_sample();
    let r = every(0, v.len() as u64, |i| guard_is_the_overflow(wide(f32::from_bits(v[i as usize]))));
    assert_none("the guard on f32s", r, |i| format!("{:#010x}", v[i as usize]));
    let mids = |f: Float, c: u64| midpoints(f, c);
    let top = F16.encode(f64::INFINITY).unwrap();
    for f in [F16, E8M15] {
        let near = f.encode(f64::INFINITY).unwrap() - 3;
        for c in (0..top).chain(near..near + 3) {
            for x in mids(f, c) {
                assert!(guard_is_the_overflow(x), "the guard at {x:e}, near code {c:#x} of {f}");
            }
        }
    }
    assert!(overflows(F16, 65520.0) && !overflows(F16, 65519.996) && !overflows(F16, f64::INFINITY) && !overflows(F16, f64::NAN));
}

#[test]
#[ignore = "the guard on every f32 and every binary16 and e8m15 midpoint: seconds a release build on 20 cores"]
fn the_guard_s_test_is_the_overflow_on_every_input() {
    let r = every(0, 1 << 32, |b| guard_is_the_overflow(wide(f32::from_bits(b as u32))));
    assert_none("the guard on every f32", r, |b| format!("{b:#010x}"));
    for f in [F16, E8M15] {
        let top = f.encode(f64::INFINITY).unwrap();
        let r = every(0, top, |c| midpoints(f, c).into_iter().all(guard_is_the_overflow));
        assert_none(&format!("the guard at {f}'s midpoints"), r, |c| format!("{c:#08x}"));
    }
    println!("the guard: every f32 and every midpoint of binary16 and e8m15 with the f64s on either side, of both signs: overflows exactly where the rounding gives an infinity");
}

#[test]
#[ignore = "FC1 on every f32: seconds a release build on 20 cores"]
fn narrowing_an_f32_is_the_format_s_rounding_on_every_f32() {
    let r = every(0, 1 << 32, |b| f32_narrows(b as u32));
    assert_none("f32 narrowed", r, |b| format!("{b:#010x}"));
    println!("FC1: 4294967296 f32s narrowed to binary16 and to e8m15, 0 wrong");
}

#[test]
#[ignore = "FC1 at every e8m15 midpoint: seconds a release build on 20 cores"]
fn narrowing_an_f64_rounds_once_at_every_e8m15_midpoint() {
    let top = E8M15.encode(f64::INFINITY).unwrap();
    let r = midpoints_round_once(E8M15, 0..top);
    assert_none("e8m15's midpoints", r, |c| format!("{c:#08x}"));
    println!("FC1: the {} midpoints of e8m15 and the f64s on either side, of both signs, narrowed from f64 to both formats, 0 wrong", top);
}

#[test]
#[ignore = "FC1 on 2^32 random f64s: seconds a release build on 20 cores"]
fn narrowing_an_f64_rounds_once_on_random_f64s() {
    let r = every(0, 1 << 31, |i| random(i).into_iter().all(f64_narrows));
    assert_none("random f64s", r, |i| format!("{:?}", random(i)));
    println!("FC1: 4294967296 random f64s narrowed to binary16 and to e8m15, 0 wrong");
}

/// aarch64's conversions between binary16 and `f32` and `f64`, on raw
/// codes, under the process's default FPCR: to nearest even, no flush to
/// zero, NaNs propagated (base Armv8-A).
#[cfg(target_arch = "aarch64")]
mod machine {
    use std::arch::asm;

    /// `fcvt h, s`.
    pub fn from_f32(a: f32) -> u16 {
        let r: u16;
        unsafe { asm!("fcvt {r:h}, {a:s}", a = in(vreg) a, r = lateout(vreg) r, options(pure, nomem, nostack)) };
        r
    }

    /// `fcvt h, d`.
    pub fn from_f64(a: f64) -> u16 {
        let r: u16;
        unsafe { asm!("fcvt {r:h}, {a:d}", a = in(vreg) a, r = lateout(vreg) r, options(pure, nomem, nostack)) };
        r
    }

    /// `fcvt s, h`.
    pub fn to_f32(a: u16) -> f32 {
        let r: f32;
        unsafe { asm!("fcvt {r:s}, {a:h}", a = in(vreg) a, r = lateout(vreg) r, options(pure, nomem, nostack)) };
        r
    }
}

#[cfg(target_arch = "aarch64")]
fn machine_narrows(x: f32) -> bool {
    f32_to_f16(x) == machine::from_f32(x) && f64_to_f16(wide(x)) == machine::from_f64(wide(x))
}

/// binary16's conversions are `fcvt`'s, bit for bit, NaNs included.
#[cfg(target_arch = "aarch64")]
#[test]
fn binary16_s_conversions_are_the_machine_s() {
    for c in 0..=u16::MAX {
        assert_eq!(f16_to_f32(c).to_bits(), machine::to_f32(c).to_bits(), "{c:#06x}");
    }
    let v = f32_sample();
    let nans = (0..1u32 << 12).map(|i| 0x7f80_0001 | i << 11 | (i & 1) << 31);
    let r = every(0, v.len() as u64, |i| machine_narrows(f32::from_bits(v[i as usize])));
    assert_none("f32s against fcvt", r, |i| format!("{:#010x}", v[i as usize]));
    assert!(nans.map(f32::from_bits).all(machine_narrows), "a NaN's payload");
    let r = every(0, 1 << 16, |i| random(i).into_iter().all(|x| f64_to_f16(x) == machine::from_f64(x)));
    assert_none("random f64s against fcvt", r, |i| format!("{:?}", random(i)));
}

#[cfg(target_arch = "aarch64")]
#[test]
#[ignore = "FC1 against fcvt on every f32 and every binary16 midpoint: seconds a release build on 20 cores"]
fn binary16_s_conversions_are_the_machine_s_on_every_input() {
    let r = every(0, 1 << 32, |b| machine_narrows(f32::from_bits(b as u32)));
    assert_none("f32s against fcvt", r, |b| format!("{b:#010x}"));
    let top = F16.encode(f64::INFINITY).unwrap();
    let r = every(0, top, |c| midpoints(F16, c).into_iter().all(|x| f64_to_f16(x) == machine::from_f64(x)));
    assert_none("binary16's midpoints against fcvt", r, |c| format!("{c:#06x}"));
    let r = every(0, 1 << 31, |i| random(i).into_iter().all(|x| f64_to_f16(x) == machine::from_f64(x)));
    assert_none("random f64s against fcvt", r, |i| format!("{:?}", random(i)));
    println!("FC1: binary16 against fcvt h,s on 4294967296 f32s, fcvt h,d on wide f32s, the {top} midpoints and 4294967296 random f64s, 0 wrong");
}
