//! The codes of binary16 and `e8m15`, the two formats a machine buffer
//! holds as its codes (plan 0.18.2 step 1, D215).
//!
//! Both are formats `f32` holds, so a backend computes a kernel of either
//! in `f32` (`Dtype::arith`), as it computes one of `bf16`, and converts on
//! the way to and from memory. A buffer of either is [`crate::Buf::Codes`]:
//! each element's code, little-endian, in the fewest whole bytes that hold
//! it, 2 for binary16 and 3 for `e8m15`, as a `.psk.d` version 3 file lays
//! them out. `e8m15`'s code is the top 24 bits of the `f32` of equal value.
//!
//! **The conversions.** A widening is exact. A narrowing rounds once, to
//! nearest, ties to even, from an `f32` or directly from an `f64`, so that
//! a buffer built from `f64`s holds exactly what the emulation's
//! [`presk_format::Float::store`] gives (D206). A finite value past the
//! largest finite number becomes an infinity, as IEEE 754 says. A NaN stays
//! a NaN of its sign, made quiet, with the top bits of its payload, as
//! aarch64's `fcvt` does under the default FPCR; widening a binary16 NaN
//! quiets it as `fcvt` does, and widening an `e8m15` code is the shift by
//! 8, which keeps a NaN's bits, as `bf16`'s does. The tests hold each
//! conversion to `presk_format`'s rounding on every `f32` and every code
//! (FC1), and binary16's to the machine's `fcvt`.

use presk_format::{Float, E8M15, F16};

/// The code of the IEEE format of `t` trailing bits and `w` exponent bits
/// nearest the `f64` of bits `b`, ties to even; a NaN quiet, with the top
/// `t − 1` bits of its payload. An `f64` subnormal is below half of every
/// such format's least subnormal here (`w` ≤ 8), so it rounds to a zero.
fn narrow(b: u64, t: u32, w: u32) -> u64 {
    let sign = (b >> 63) << (t + w);
    let inf = ((1u64 << w) - 1) << t;
    let abs = b & !(1u64 << 63);
    if abs > 0x7ff << 52 {
        return sign | inf | 1 << (t - 1) | (abs >> (52 - t)) & ((1 << (t - 1)) - 1);
    }
    let field = (abs >> 52) as i32;
    if field == 0 {
        return sign;
    }
    // `|x| = m · 2^(lead − 52)`, rounded to a multiple of `2^(lead − t)`
    // where it is normal and of the least subnormal below.
    let m = abs & ((1 << 52) - 1) | 1 << 52;
    let lead = field - 1023;
    let bias = (1i32 << (w - 1)) - 1;
    let emin = 1 - bias;
    let shift = 52 - t as i32 + (emin - lead).max(0);
    let r = if shift > 63 {
        0
    } else {
        let s = shift as u32;
        let (q, low, half) = (m >> s, m & ((1 << s) - 1), 1u64 << (s - 1));
        q + u64::from(low > half || low == half && q & 1 == 1)
    };
    // The exponent field and the significand's leading one add; a carry
    // out of the significand moves the exponent, and past the largest
    // finite number it reaches the infinity's code.
    let code = if lead >= emin { (((lead + bias - 1) as u64) << t) + r } else { r };
    sign | code.min(inf)
}

/// The bits of the `f64` equal to `x`; a NaN's sign and payload moved into
/// the `f64`'s, quiet bit included, without a conversion that may change
/// them.
fn wide_bits(x: f32) -> u64 {
    let b = u64::from(x.to_bits());
    if x.is_nan() { (b >> 31) << 63 | 0x7ff << 52 | (b & 0x7f_ffff) << 29 } else { f64::from(x).to_bits() }
}

/// The binary16 code nearest an `f32`, ties to even (`fcvt h, s`).
pub fn f32_to_f16(x: f32) -> u16 {
    narrow(wide_bits(x), 10, 5) as u16
}

/// The binary16 code nearest an `f64`, rounded once (`fcvt h, d`).
pub fn f64_to_f16(x: f64) -> u16 {
    narrow(x.to_bits(), 10, 5) as u16
}

/// The `f32` a binary16 code denotes, exactly; a NaN made quiet
/// (`fcvt s, h`).
pub fn f16_to_f32(c: u16) -> f32 {
    let sign = u32::from(c & 0x8000) << 16;
    let (field, frac) = (u32::from(c >> 10 & 0x1f), u32::from(c & 0x3ff));
    let mag = match field {
        0x1f if frac == 0 => 0x7f80_0000,
        0x1f => 0x7fc0_0000 | frac << 13,
        // `frac · 2^−24`, a normal `f32`, exactly.
        0 => (frac as f32 * f32::from_bits(0x3380_0000)).to_bits(),
        _ => (field + 112) << 23 | frac << 13,
    };
    f32::from_bits(sign | mag)
}

/// The `e8m15` code nearest an `f32`, ties to even: `bf16`'s rounding at
/// bit 8, the code in the low 24 bits.
pub fn f32_to_e8m15(x: f32) -> u32 {
    let bits = x.to_bits();
    if x.is_nan() {
        return bits >> 8 | 0x4000;
    }
    (bits + 0x7f + (bits >> 8 & 1)) >> 8
}

/// The `e8m15` code nearest an `f64`, rounded once.
pub fn f64_to_e8m15(x: f64) -> u32 {
    narrow(x.to_bits(), 15, 8) as u32
}

/// The `f32` an `e8m15` code denotes, exactly: its 24 bits shifted into the
/// top of an `f32`'s.
pub fn e8m15_to_f32(c: u32) -> f32 {
    f32::from_bits(c << 8)
}

/// The least magnitude a narrowing to `f`, binary16 or `e8m15`, rounds to
/// an infinity: the midpoint between the format's largest finite number
/// and the next power of two, which rounds up, the largest finite
/// number's significand being odd. 65 520 for binary16, `(2 − 2⁻¹⁶)·2¹²⁷`
/// for `e8m15`; both are `f32`s.
pub fn overflow_threshold(f: Float) -> f64 {
    if f == F16 {
        65520.0
    } else if f == E8M15 {
        f64::from(f32::from_bits(0x7f7f_ff80))
    } else {
        panic!("{f} is no format a buffer holds as its codes")
    }
}

/// Whether rounding `x` to `f` turns a finite value into an infinity: the
/// overflow guard's test (plan 0.18.2 step 4, D218), which every executor
/// makes at a kernel's store into binary16 or `e8m15`. An infinity or a
/// NaN is not an overflow of the store.
pub fn overflows(f: Float, x: f64) -> bool {
    x.is_finite() && x.abs() >= overflow_threshold(f)
}

/// The largest finite number of `f`, as the guard's message names it.
pub fn largest(f: Float) -> &'static str {
    if f == F16 { "65504" } else { "(2 − 2⁻¹⁵)·2¹²⁷" }
}

/// Whether a buffer holds the format's numbers as its codes: binary16 and
/// `e8m15`.
pub fn coded(f: Float) -> bool {
    f == F16 || f == E8M15
}

/// Bytes of one code of a format [`coded`] names.
pub fn width(f: Float) -> usize {
    if f == F16 {
        2
    } else if f == E8M15 {
        3
    } else {
        panic!("{f} is no format a buffer holds as its codes")
    }
}

/// Element `e`'s code in `b`, a buffer of codes of `f`.
pub fn code(f: Float, b: &[u8], e: usize) -> u32 {
    if f == F16 {
        u32::from(u16::from_le_bytes([b[2 * e], b[2 * e + 1]]))
    } else {
        let w = width(f);
        u32::from_le_bytes([b[w * e], b[w * e + 1], b[w * e + 2], 0])
    }
}

/// Element `e`'s value in `b`, a buffer of codes of `f`, as an `f32`.
pub fn get(f: Float, b: &[u8], e: usize) -> f32 {
    let c = code(f, b, e);
    if f == F16 { f16_to_f32(c as u16) } else { e8m15_to_f32(c) }
}

/// Sets element `e` of `b`, a buffer of codes of `f`, to `x` rounded once.
pub fn put(f: Float, b: &mut [u8], e: usize, x: f64) {
    if f == F16 {
        b[2 * e..2 * e + 2].copy_from_slice(&f64_to_f16(x).to_le_bytes());
    } else {
        let w = width(f);
        b[w * e..w * e + 3].copy_from_slice(&f64_to_e8m15(x).to_le_bytes()[..3]);
    }
}

/// The codes of `v` in `f`, each rounded once.
pub fn encode(f: Float, v: &[f64]) -> Vec<u8> {
    let mut b = vec![0u8; v.len() * width(f)];
    for (e, x) in v.iter().enumerate() {
        put(f, &mut b, e, *x);
    }
    b
}

/// The values of `b`, a buffer of codes of `f`, elements `lo..hi`, as
/// `f64`s (exactly).
pub fn decode(f: Float, b: &[u8], lo: usize, hi: usize) -> Vec<f64> {
    (lo..hi).map(|e| f64::from(get(f, b, e))).collect()
}
