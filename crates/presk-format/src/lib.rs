//! Machine number formats as data (plan 0.18.0 step 1, D204).
//!
//! A format is described by its parameters, not named after a machine:
//!
//! - a **float** ([`Float`]) by its significand's bits, its exponent range,
//!   whether it has subnormals, its special values, what a result past the
//!   largest finite gives, and the sign a result that rounds to zero keeps;
//! - an **integer** ([`Int`]) by its width, its sign and what an overflow
//!   does;
//! - a **scaled format** ([`Scaled`]) by an integer, the part of a tensor one
//!   scale covers and the scale's own format.
//!
//! One generic rounding takes a real held as an `f64` to any float format
//! under a tie rule ([`Float::round`]), with the format's limits and special
//! values; a conversion is a path of such roundings ([`convert`]), and the
//! runtime's own path into a format is [`Float::store`]. Each format has an
//! encoding, its codes as the bits a buffer or a `.psk.d` file holds
//! ([`Float::encode`], [`Float::decode`]).
//!
//! `f64`, `f32` and `bf16` are three descriptions among the others and keep
//! their bits: [`F32`]'s rounding is Rust's `as f32`, and [`BF16`]'s of an
//! `f32` is the runtime's `f32_to_bf16`, both checked on every `f32`. The
//! formats the field names have their names ([`ALIASES`]); every other
//! format is written by its parameters, `float(p=4,emin=-6,emax=8,
//! specials=none,underflow=+0)`, which is also how a format is serialized.
//!
//! The crate depends on nothing but `serde`, so that the data container,
//! the reference interpreter and the runtime can all read a format.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt;

/// The values a float format holds besides its finite numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Specials {
    /// ±∞ and NaN, the top exponent code reserved for them (IEEE 754).
    Ieee,
    /// A NaN of each sign and no infinity: the top code of each sign is
    /// NaN and every other code finite (OFP8's E4M3).
    Nan,
    /// None: every code is a finite number.
    None,
}

/// What a result past the largest finite number of a float format gives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Overflow {
    /// An infinity of its sign (IEEE 754).
    Infinity,
    /// A NaN.
    Nan,
    /// The largest finite number of its sign. In a format with no NaN, a
    /// NaN converts to it too, with its own sign.
    Saturate,
}

/// The sign of a result that rounds to zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Underflow {
    /// The value's own: a negative value that rounds to zero is −0.
    Signed,
    /// +0 whatever the value's sign, −0 included. The format may still
    /// have a code for −0; a conversion never produces it.
    Positive,
}

/// The NaN an operation gives in a format with NaNs, or a conversion into
/// it (plan 0.18.0 step 6, D209). IEEE 754 leaves its sign and payload to
/// the machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Nans {
    /// An operand's NaN, quieted, with its sign: IEEE 754's propagation,
    /// what falicon's own instructions do and the emulation computes.
    Operand,
    /// The format's default NaN, positive and quiet, whatever the operands:
    /// a machine's default-NaN mode (an ARM core's `FPCR.DN`), the binary16
    /// adder of the accelerator plan 0.18.0 checks (MT2).
    Default,
}

/// How a value halfway between two neighbours is rounded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Tie {
    /// To the neighbour whose last significand bit is 0 (IEEE 754's
    /// default).
    Even,
    /// To the neighbour of larger magnitude.
    Away,
}

/// A binary floating-point format, by its parameters.
///
/// Its finite numbers are ±m·2^(e−p+1) for an integer m < 2^p and e in
/// `emin..=emax`, the normal ones with m ≥ 2^(p−1), and, when it has
/// subnormals, ±m·2^(emin−p+1) for m < 2^(p−1). The encoding is IEEE 754's
/// layout: a sign bit, an exponent field biased by 1 − `emin` whose code 0
/// holds zero and the subnormals, and the p − 1 trailing bits of the
/// significand. [`Float::check`] says which descriptions are formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Float {
    /// Bits of significand, the implicit leading one included: 53 for
    /// `f64`, 24 for `f32`, 8 for `bf16`.
    pub p: u32,
    /// The exponent of the smallest normal number, 2^emin.
    pub emin: i32,
    /// The exponent of the largest finite number's leading bit.
    pub emax: i32,
    pub subnormals: bool,
    pub specials: Specials,
    pub overflow: Overflow,
    pub underflow: Underflow,
    pub nans: Nans,
}

const fn ieee(p: u32, emin: i32, emax: i32) -> Float {
    Float { p, emin, emax, subnormals: true, specials: Specials::Ieee, overflow: Overflow::Infinity, underflow: Underflow::Signed, nans: Nans::Operand }
}

/// IEEE 754 binary64.
pub const F64: Float = ieee(53, -1022, 1023);
/// IEEE 754 binary32.
pub const F32: Float = ieee(24, -126, 127);
/// `bfloat16`: binary32's exponent and 8 bits of significand.
pub const BF16: Float = ieee(8, -126, 127);
/// IEEE 754 binary16.
pub const F16: Float = ieee(11, -14, 15);
/// Binary32's exponent and 16 bits of significand, three bytes: the width
/// SR6 certified the course GPT's weights at (D196).
pub const E8M15: Float = ieee(16, -126, 127);
/// OFP8's E5M2: binary16's exponent and 3 bits of significand.
pub const E5M2: Float = ieee(3, -14, 15);
/// OFP8's E4M3: no infinity, a NaN at the top code of each sign, so the
/// largest finite is 448; a result past it is NaN, as OFP8 converts without
/// saturation.
pub const E4M3FN: Float = Float { p: 4, emin: -6, emax: 8, subnormals: true, specials: Specials::Nan, overflow: Overflow::Nan, underflow: Underflow::Signed, nans: Nans::Operand };
/// An E4M3 of bias 7 with no special value: every code finite, the largest
/// 480, an overflow, an infinity or a NaN saturating to ±480, and a result
/// that rounds to zero +0.
pub const E4M3_SATURATING: Float = Float { p: 4, emin: -6, emax: 8, subnormals: true, specials: Specials::None, overflow: Overflow::Saturate, underflow: Underflow::Positive, nans: Nans::Operand };
/// OCP MX's FP4 E2M1: 0, 0.5, 1, 1.5, 2, 3, 4 and 6 of each sign, no special
/// value, saturating.
pub const E2M1: Float = Float { p: 2, emin: 0, emax: 2, subnormals: true, specials: Specials::None, overflow: Overflow::Saturate, underflow: Underflow::Signed, nans: Nans::Operand };

/// The formats the field names, by their names. Every other format is
/// written by its parameters.
pub const ALIASES: &[(&str, Float)] = &[("f64", F64), ("f32", F32), ("bf16", BF16), ("f16", F16), ("e8m15", E8M15), ("e5m2", E5M2), ("e4m3fn", E4M3FN), ("e2m1", E2M1)];

/// A comma-separated list of names split at the commas outside
/// parentheses, each trimmed: `f16,float(p=4,emin=-6,emax=8)` is two
/// formats.
pub fn split_names(s: &str) -> Vec<&str> {
    let (mut out, mut depth, mut start) = (Vec::new(), 0i32, 0);
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(s[start..].trim());
    out
}

/// `2^k` as an `f64`, for `k` in `-1074..=1023`.
fn exp2(k: i32) -> f64 {
    if k >= -1022 {
        f64::from_bits(((k + 1023) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (k + 1074))
    }
}

/// `|x| = m·2^e` with `m` an integer below 2^53, for a finite non-zero `x`.
fn decompose(x: f64) -> (u64, i32) {
    let bits = x.abs().to_bits();
    let (be, frac) = ((bits >> 52) as i32, bits & ((1u64 << 52) - 1));
    if be == 0 { (frac, -1074) } else { (frac | (1u64 << 52), be - 1075) }
}

/// `m·2^-s` rounded to an integer under `tie`.
fn shift_round(m: u64, s: i32, tie: Tie) -> u64 {
    if s <= 0 {
        return m << -s;
    }
    if s >= 64 {
        return 0;
    }
    let (r, rem, half) = (m >> s, m & ((1u64 << s) - 1), 1u64 << (s - 1));
    let up = match tie {
        Tie::Even => rem > half || (rem == half && r & 1 == 1),
        Tie::Away => rem >= half,
    };
    r + up as u64
}

/// [`shift_round`] of a 128-bit integer.
fn shift_round_wide(m: u128, s: i32, tie: Tie) -> u128 {
    if s <= 0 {
        return m << -s;
    }
    if s >= 128 {
        return u128::from(s == 128 && tie == Tie::Away && m >> 127 == 1);
    }
    let (r, rem, half) = (m >> s, m & ((1u128 << s) - 1), 1u128 << (s - 1));
    let up = match tie {
        Tie::Even => rem > half || (rem == half && r & 1 == 1),
        Tie::Away => rem >= half,
    };
    r + up as u128
}

impl Float {
    /// Whether the description is a format this crate can round to and
    /// encode: 2 ≤ p ≤ 53, an exponent range inside `f64`'s whose codes fill
    /// a whole exponent field, and an overflow rule the special values allow.
    pub fn check(self) -> Result<Float, String> {
        if !(2..=53).contains(&self.p) {
            return Err(format!("a float format has 2 to 53 bits of significand, not {}", self.p));
        }
        if self.emin > self.emax || self.emin < -1022 || self.emax > 1023 {
            return Err(format!("the exponent range {}..={} is not inside f64's, -1022..=1023", self.emin, self.emax));
        }
        let codes = (self.emax - self.emin + 2) as u64 + u64::from(self.specials == Specials::Ieee);
        if !codes.is_power_of_two() {
            return Err(format!("the exponent range {}..={} with {} does not fill an exponent field: it needs {codes} codes", self.emin, self.emax, self.specials_name()));
        }
        if self.nans == Nans::Default && !self.has_nan() {
            return Err("a format with no NaN gives no default NaN: nans=default needs specials=ieee or specials=nan".into());
        }
        match (self.specials, self.overflow) {
            (Specials::Ieee, _) | (Specials::Nan, Overflow::Nan | Overflow::Saturate) | (Specials::None, Overflow::Saturate) => Ok(self),
            _ => Err(format!("a format with {} cannot overflow to {}", self.specials_name(), self.overflow_name())),
        }
    }

    /// Bits of the exponent field.
    pub fn exp_bits(self) -> u32 {
        let codes = (self.emax - self.emin + 2) as u64 + u64::from(self.specials == Specials::Ieee);
        codes.trailing_zeros()
    }

    /// Bits of one code: the sign, the exponent field and the trailing
    /// significand.
    pub fn bits(self) -> u32 {
        1 + self.exp_bits() + self.p - 1
    }

    /// The exponent field's bias, 1 − emin.
    pub fn bias(self) -> i32 {
        1 - self.emin
    }

    pub fn has_infinity(self) -> bool {
        self.specials == Specials::Ieee
    }

    pub fn has_nan(self) -> bool {
        self.specials != Specials::None
    }

    /// The largest finite number. Under [`Specials::Nan`] the top
    /// significand of the top exponent is NaN, so it is one step below.
    pub fn max_finite(self) -> f64 {
        let top = (1u64 << self.p) - if self.specials == Specials::Nan { 2 } else { 1 };
        top as f64 * exp2(self.emax - self.p as i32 + 1)
    }

    /// The smallest positive number: the least subnormal, or 2^emin.
    pub fn min_positive(self) -> f64 {
        if self.subnormals { exp2(self.emin - self.p as i32 + 1) } else { exp2(self.emin) }
    }

    /// The unit roundoff `u = 2^-p`, half an ulp of 1.
    pub fn unit_roundoff(self) -> f64 {
        exp2(-(self.p as i32))
    }

    /// Whether every number of `other`, and each special value it has, is
    /// one of this format: the inclusion 𝔽_other ⊆ 𝔽_self of the
    /// reference's chapter 4.
    pub fn contains(self, other: Float) -> bool {
        other.p <= self.p
            && other.max_finite() <= self.max_finite()
            && other.min_positive() >= self.min_positive()
            && other.emin - other.p as i32 >= self.emin - self.p as i32
            && (!other.has_infinity() || self.has_infinity())
            && (!other.has_nan() || self.has_nan())
    }

    fn zero(self, negative: bool) -> f64 {
        if negative && self.underflow == Underflow::Signed { -0.0 } else { 0.0 }
    }

    fn overflowed(self, negative: bool) -> f64 {
        let v = match self.overflow {
            Overflow::Infinity => f64::INFINITY,
            Overflow::Nan => return f64::NAN,
            Overflow::Saturate => self.max_finite(),
        };
        if negative { -v } else { v }
    }

    /// The number of this format nearest `x` under `tie`, as an `f64`
    /// (exactly: every format is inside `f64`).
    ///
    /// The significand is rounded with an unbounded exponent above and
    /// `emin` below, so a value under the smallest normal rounds among the
    /// subnormals (or, without them, to 0 or 2^emin), and a result above
    /// the largest finite is an overflow, which gives what [`Overflow`]
    /// says: the IEEE rule, under which 65 520 is an infinity in binary16.
    /// An infinity converts to an infinity when the format has one and as
    /// an overflow otherwise; a NaN to a NaN when it has one and, in a
    /// format with none, to its saturation with the NaN's sign. A result
    /// that rounds to zero takes the sign [`Underflow`] says.
    pub fn round(self, x: f64, tie: Tie) -> f64 {
        self.round_noted(x, tie).0
    }

    /// [`Float::round`], and whether the format's overflow rule gave the
    /// result: a finite value past the largest finite number, or a special
    /// value the format does not have. The kernel interpreter's emulation
    /// counts what such a rule produced (plan 0.18.0 step 2, D205).
    pub fn round_noted(self, x: f64, tie: Tie) -> (f64, bool) {
        let negative = x.is_sign_negative();
        if x.is_nan() {
            return match (self.has_nan(), self.nans) {
                (true, Nans::Operand) => (x, false),
                (true, Nans::Default) => (f64::NAN, false),
                (false, _) => (self.overflowed(negative), true),
            };
        }
        if x.is_infinite() {
            return if self.has_infinity() { (x, false) } else { (self.overflowed(negative), true) };
        }
        if x == 0.0 {
            return (self.zero(negative), false);
        }
        let (m, e) = decompose(x);
        let lead = e + 63 - m.leading_zeros() as i32;
        let q = if lead >= self.emin {
            lead - (self.p as i32 - 1)
        } else if self.subnormals {
            self.emin - (self.p as i32 - 1)
        } else {
            self.emin
        };
        let r = shift_round(m, q - e, tie);
        if r == 0 {
            return (self.zero(negative), false);
        }
        let v = r as f64 * exp2(q);
        if v > self.max_finite() {
            return (self.overflowed(negative), true);
        }
        (if negative { -v } else { v }, false)
    }

    /// The number of this format nearest `m·2^e` under `tie`, and whether
    /// the overflow rule gave it: [`Float::round_noted`] of a value no `f64`
    /// may hold, an integer of up to 127 bits scaled by a power of two
    /// (an integer sum, a fixed-point one), rounded once.
    pub fn round_scaled(self, m: i128, e: i32, tie: Tie) -> (f64, bool) {
        let negative = m < 0;
        if m == 0 {
            return (self.zero(false), false);
        }
        let a = m.unsigned_abs();
        let lead = e + 127 - a.leading_zeros() as i32;
        if lead > self.emax {
            return (self.overflowed(negative), true);
        }
        let q = if lead >= self.emin {
            lead - (self.p as i32 - 1)
        } else if self.subnormals {
            self.emin - (self.p as i32 - 1)
        } else {
            self.emin
        };
        let r = shift_round_wide(a, q - e, tie);
        if r == 0 {
            return (self.zero(negative), false);
        }
        let v = r as f64 * exp2(q);
        if v > self.max_finite() {
            return (self.overflowed(negative), true);
        }
        (if negative { -v } else { v }, false)
    }

    /// The number the runtime stores a real held as an `f64` as, ties to
    /// even: rounded once, but `bf16` first to `f32`, as a `bf16` buffer
    /// always was (`f32_to_bf16(x as f32)`, D38). `f64` is the identity
    /// and `f32` the machine's conversion, which the exhaustive checks show
    /// is the rounding (`f32_on_every_midpoint`). Until plan 0.18.0 step 3
    /// every format `f32` holds went through `f32`; a conversion is one
    /// rounding of the value's own (D206), and `bf16` keeps its path so that
    /// no native bit moves.
    pub fn store(self, x: f64) -> f64 {
        if self == F64 {
            x
        } else if self == F32 {
            x as f32 as f64
        } else if self == BF16 {
            self.round(x as f32 as f64, Tie::Even)
        } else {
            self.round(x, Tie::Even)
        }
    }

    /// Whether `x` is a number or a special value of this format.
    pub fn holds(self, x: f64) -> bool {
        if x.is_nan() {
            return self.has_nan();
        }
        if x.is_infinite() {
            return self.has_infinity();
        }
        x == 0.0 || self.round(x, Tie::Even) == x
    }

    /// The code of `x`, a number or a special value of the format; `None`
    /// for any other `x`. A NaN is the quiet NaN of its sign (IEEE) or the
    /// top code of its sign (OFP8's E4M3).
    pub fn encode(self, x: f64) -> Option<u64> {
        if !self.holds(x) {
            return None;
        }
        let t = self.p - 1;
        let sign = u64::from(x.is_sign_negative()) << (self.bits() - 1);
        let top = (1u64 << self.exp_bits()) - 1;
        if x.is_nan() {
            return Some(match self.specials {
                Specials::Ieee => sign | top << t | 1u64 << (t - 1),
                _ => sign | top << t | ((1u64 << t) - 1),
            });
        }
        if x.is_infinite() {
            return Some(sign | top << t);
        }
        if x == 0.0 {
            return Some(sign);
        }
        let (m, e) = decompose(x);
        let lead = e + 63 - m.leading_zeros() as i32;
        if lead < self.emin {
            return Some(sign | shift_round(m, self.emin - t as i32 - e, Tie::Even));
        }
        let significand = shift_round(m, lead - t as i32 - e, Tie::Even);
        Some(sign | ((lead + self.bias()) as u64) << t | (significand - (1u64 << t)))
    }

    /// The number or special value a code denotes; the bits above
    /// [`Float::bits`] are ignored. In a format without subnormals, a code
    /// of exponent field 0 is a zero.
    pub fn decode(self, code: u64) -> f64 {
        let t = self.p - 1;
        let negative = code >> (self.bits() - 1) & 1 == 1;
        let top = (1u64 << self.exp_bits()) - 1;
        let (field, trailing) = (code >> t & top, code & ((1u64 << t) - 1));
        let v = if self.specials == Specials::Ieee && field == top {
            if trailing == 0 { f64::INFINITY } else { f64::NAN }
        } else if self.specials == Specials::Nan && field == top && trailing == (1u64 << t) - 1 {
            f64::NAN
        } else if field == 0 {
            if self.subnormals { trailing as f64 * exp2(self.emin - t as i32) } else { 0.0 }
        } else {
            ((1u64 << t) | trailing) as f64 * exp2(field as i32 - self.bias() - t as i32)
        };
        if negative { -v } else { v }
    }

    /// The name the field gives the format, if it has one.
    pub fn alias(self) -> Option<&'static str> {
        ALIASES.iter().find(|(_, f)| *f == self).map(|(n, _)| *n)
    }

    /// The overflow a format's special values make the default: an
    /// infinity, a NaN, or saturation when it has neither.
    fn default_overflow(specials: Specials) -> Overflow {
        match specials {
            Specials::Ieee => Overflow::Infinity,
            Specials::Nan => Overflow::Nan,
            Specials::None => Overflow::Saturate,
        }
    }

    fn specials_name(self) -> &'static str {
        match self.specials {
            Specials::Ieee => "ieee",
            Specials::Nan => "nan",
            Specials::None => "none",
        }
    }

    fn overflow_name(self) -> &'static str {
        match self.overflow {
            Overflow::Infinity => "inf",
            Overflow::Nan => "nan",
            Overflow::Saturate => "saturate",
        }
    }

    /// The spelling of the parameters: `p`, `emin` and `emax`, then every
    /// other parameter that differs from its default (subnormals, IEEE's
    /// special values, the overflow they make the default, the sign kept).
    pub fn spelling(self) -> String {
        let mut s = format!("float(p={},emin={},emax={}", self.p, self.emin, self.emax);
        if !self.subnormals {
            s.push_str(",subnormals=no");
        }
        if self.specials != Specials::Ieee {
            s.push_str(&format!(",specials={}", self.specials_name()));
        }
        if self.overflow != Float::default_overflow(self.specials) {
            s.push_str(&format!(",overflow={}", self.overflow_name()));
        }
        if self.underflow == Underflow::Positive {
            s.push_str(",underflow=+0");
        }
        if self.nans == Nans::Default {
            s.push_str(",nans=default");
        }
        s.push(')');
        s
    }

    /// Its alias, or the spelling of its parameters.
    pub fn name(self) -> Cow<'static, str> {
        match self.alias() {
            Some(n) => Cow::Borrowed(n),
            None => Cow::Owned(self.spelling()),
        }
    }

    /// A format from its alias or the spelling of its parameters, in any
    /// order of the parameters after `p`, `emin` and `emax`, which are
    /// required. A spelling that describes an aliased format is that
    /// format: `float(p=8,emin=-126,emax=127)` is `bf16`.
    pub fn parse(s: &str) -> Result<Float, String> {
        if let Some((_, f)) = ALIASES.iter().find(|(n, _)| *n == s) {
            return Ok(*f);
        }
        let Some(body) = s.strip_prefix("float(").and_then(|b| b.strip_suffix(')')) else {
            return Err(format!("unknown format `{s}`: expected {} or float(p=…,emin=…,emax=…)", ALIASES.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ")));
        };
        let (mut p, mut emin, mut emax) = (None, None, None);
        let (mut subnormals, mut specials, mut overflow, mut underflow, mut nans) = (None, None, None, None, None);
        for item in body.split(',') {
            let Some((key, value)) = item.split_once('=') else {
                return Err(format!("`{item}` in `{s}` is not a parameter `key=value`"));
            };
            let bad = || format!("`{value}` is not a value of `{key}` in `{s}`");
            let dup = |set: bool| if set { Err(format!("`{key}` is given twice in `{s}`")) } else { Ok(()) };
            match key {
                "p" => {
                    dup(p.is_some())?;
                    p = Some(value.parse::<u32>().map_err(|_| bad())?);
                }
                "emin" => {
                    dup(emin.is_some())?;
                    emin = Some(value.parse::<i32>().map_err(|_| bad())?);
                }
                "emax" => {
                    dup(emax.is_some())?;
                    emax = Some(value.parse::<i32>().map_err(|_| bad())?);
                }
                "subnormals" => {
                    dup(subnormals.is_some())?;
                    subnormals = Some(match value {
                        "yes" => true,
                        "no" => false,
                        _ => return Err(bad()),
                    });
                }
                "specials" => {
                    dup(specials.is_some())?;
                    specials = Some(match value {
                        "ieee" => Specials::Ieee,
                        "nan" => Specials::Nan,
                        "none" => Specials::None,
                        _ => return Err(bad()),
                    });
                }
                "overflow" => {
                    dup(overflow.is_some())?;
                    overflow = Some(match value {
                        "inf" => Overflow::Infinity,
                        "nan" => Overflow::Nan,
                        "saturate" => Overflow::Saturate,
                        _ => return Err(bad()),
                    });
                }
                "underflow" => {
                    dup(underflow.is_some())?;
                    underflow = Some(match value {
                        "signed" => Underflow::Signed,
                        "+0" => Underflow::Positive,
                        _ => return Err(bad()),
                    });
                }
                "nans" => {
                    dup(nans.is_some())?;
                    nans = Some(match value {
                        "operand" => Nans::Operand,
                        "default" => Nans::Default,
                        _ => return Err(bad()),
                    });
                }
                _ => return Err(format!("unknown parameter `{key}` in `{s}`: expected p, emin, emax, subnormals, specials, overflow, underflow or nans")),
            }
        }
        let (Some(p), Some(emin), Some(emax)) = (p, emin, emax) else {
            return Err(format!("`{s}` does not give p, emin and emax"));
        };
        let specials = specials.unwrap_or(Specials::Ieee);
        Float { p, emin, emax, subnormals: subnormals.unwrap_or(true), specials, overflow: overflow.unwrap_or(Float::default_overflow(specials)), underflow: underflow.unwrap_or(Underflow::Signed), nans: nans.unwrap_or(Nans::Operand) }.check()
    }

    /// The key formats are ordered by: the most precise first, so that
    /// `f64` < `f32` < `bf16`, the order every table and every map keyed by
    /// a format has always printed them in; then the widest range.
    fn key(self) -> (std::cmp::Reverse<u32>, std::cmp::Reverse<i32>, i32, bool, Specials, Overflow, Underflow, Nans) {
        (std::cmp::Reverse(self.p), std::cmp::Reverse(self.emax), self.emin, !self.subnormals, self.specials, self.overflow, self.underflow, self.nans)
    }
}

impl PartialOrd for Float {
    fn partial_cmp(&self, other: &Float) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Float {
    fn cmp(&self, other: &Float) -> Ordering {
        self.key().cmp(&other.key())
    }
}

impl fmt::Display for Float {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name())
    }
}

impl serde::Serialize for Float {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.name())
    }
}

impl<'de> serde::Deserialize<'de> for Float {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Float, D::Error> {
        struct Name;
        impl serde::de::Visitor<'_> for Name {
            type Value = Float;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a format's name")
            }
            fn visit_str<E: serde::de::Error>(self, s: &str) -> Result<Float, E> {
                Float::parse(s).map_err(E::custom)
            }
        }
        d.deserialize_str(Name)
    }
}

/// `x` converted along `path`, each step rounded under `tie`: a conversion
/// between two formats is a path and a tie rule (the reference's chapter 4,
/// once the formats are no longer a chain).
pub fn convert(x: f64, path: &[Float], tie: Tie) -> f64 {
    path.iter().fold(x, |y, f| f.round(y, tie))
}

/// What an integer format does with a result outside its range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum IntOverflow {
    /// Modulo 2^bits, two's complement.
    Wrap,
    /// To the nearest end of the range.
    Clamp,
}

/// A binary integer format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Int {
    /// 1 to 64.
    pub bits: u32,
    pub signed: bool,
    pub overflow: IntOverflow,
}

impl Int {
    pub fn min(self) -> i128 {
        if self.signed { -(1i128 << (self.bits - 1)) } else { 0 }
    }

    pub fn max(self) -> i128 {
        if self.signed { (1i128 << (self.bits - 1)) - 1 } else { (1i128 << self.bits) - 1 }
    }

    /// An exact integer brought into the range as [`IntOverflow`] says.
    pub fn fit(self, n: i128) -> i128 {
        match self.overflow {
            IntOverflow::Clamp => n.clamp(self.min(), self.max()),
            IntOverflow::Wrap => {
                let m = 1i128 << self.bits;
                let r = n.rem_euclid(m);
                if r > self.max() { r - m } else { r }
            }
        }
    }

    /// `x` rounded to an integer under `tie`, then fitted. A NaN is 0, an
    /// infinity the end of its sign.
    pub fn round(self, x: f64, tie: Tie) -> i128 {
        self.round_noted(x, tie).0
    }

    /// [`Int::round`], and whether the value was outside the range (fitted
    /// by the overflow rule, or a NaN or an infinity).
    pub fn round_noted(self, x: f64, tie: Tie) -> (i128, bool) {
        if x.is_nan() {
            return (0, true);
        }
        if x.is_infinite() {
            return (if x > 0.0 { self.max() } else { self.min() }, true);
        }
        let r = match tie {
            Tie::Even => x.round_ties_even(),
            Tie::Away => x.round(),
        };
        let n = r.clamp(-1.7e38, 1.7e38) as i128;
        let y = self.fit(n);
        (y, y != n)
    }

    /// `m·2^e` rounded to an integer under `tie`, then fitted, and whether
    /// it was outside the range: an integer, a fixed-point value, converted
    /// exactly.
    pub fn round_scaled(self, m: i128, e: i32, tie: Tie) -> (i128, bool) {
        let n = if e >= 0 {
            let exact = if m == 0 { Some(0) } else if e < 127 { m.checked_mul(1i128 << e) } else { None };
            match exact {
                Some(n) => n,
                None => {
                    return match self.overflow {
                        IntOverflow::Clamp => (if m > 0 { self.max() } else { self.min() }, true),
                        IntOverflow::Wrap => (self.fit(if e < 128 { m.wrapping_shl(e as u32) } else { 0 }), true),
                    };
                }
            }
        } else {
            // At most 2^126 + 1 once shifted by one bit or more.
            let r = shift_round_wide(m.unsigned_abs(), -e, tie) as i128;
            if m < 0 { -r } else { r }
        };
        let y = self.fit(n);
        (y, y != n)
    }

    /// `i8`, `u16`, `i32`, …; `/sat` when an overflow clamps.
    pub fn name(self) -> String {
        format!("{}{}{}", if self.signed { "i" } else { "u" }, self.bits, if self.overflow == IntOverflow::Clamp { "/sat" } else { "" })
    }

    pub fn parse(s: &str) -> Result<Int, String> {
        let (body, overflow) = match s.strip_suffix("/sat") {
            Some(b) => (b, IntOverflow::Clamp),
            None => (s, IntOverflow::Wrap),
        };
        let (signed, digits) = match (body.strip_prefix('i'), body.strip_prefix('u')) {
            (Some(d), _) => (true, d),
            (_, Some(d)) => (false, d),
            _ => return Err(format!("unknown integer format `{s}`: expected i8, u16, i32/sat, …")),
        };
        match digits.parse::<u32>() {
            Ok(bits @ 1..=64) => Ok(Int { bits, signed, overflow }),
            _ => Err(format!("unknown integer format `{s}`: an integer has 1 to 64 bits")),
        }
    }
}

impl fmt::Display for Int {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name())
    }
}

/// A binary fixed-point format: an integer `m` of the format `int` that
/// denotes `m·2^-frac`. Its sums are exact while they stay in the range,
/// which is what an exact accumulator is: the products of two E4M3 values
/// are multiples of 2^-18, so `fixed(i64,frac=18)` sums them exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Fixed {
    pub int: Int,
    /// 0 to 255.
    pub frac: i32,
}

impl Fixed {
    pub fn check(self) -> Result<Fixed, String> {
        if !(0..=255).contains(&self.frac) {
            return Err(format!("a fixed-point format has 0 to 255 fractional bits, not {}", self.frac));
        }
        Ok(self)
    }

    /// The units, `m`, of `x` rounded to the format under `tie`, and
    /// whether it was outside the range.
    pub fn round_noted(self, x: f64, tie: Tie) -> (i128, bool) {
        self.int.round_noted(x * exp2(self.frac), tie)
    }

    /// The units of `m·2^e` rounded to the format, and whether it was
    /// outside the range.
    pub fn round_scaled(self, m: i128, e: i32, tie: Tie) -> (i128, bool) {
        self.int.round_scaled(m, e + self.frac, tie)
    }

    /// The value of `m` units, rounded to `f64` when it has more than 53
    /// significant bits.
    pub fn value(self, m: i128) -> f64 {
        F64.round_scaled(m, -self.frac, Tie::Even).0
    }

    /// `fixed(i64,frac=18)`: the integer's name and the fractional bits.
    pub fn name(self) -> String {
        format!("fixed({},frac={})", self.int.name(), self.frac)
    }

    pub fn parse(s: &str) -> Result<Fixed, String> {
        let bad = || format!("unknown fixed-point format `{s}`: expected fixed(i64,frac=18), fixed(i32/sat,frac=8), …");
        let body = s.strip_prefix("fixed(").and_then(|b| b.strip_suffix(')')).ok_or_else(bad)?;
        let (int, frac) = body.split_once(",frac=").ok_or_else(bad)?;
        Fixed { int: Int::parse(int)?, frac: frac.parse().map_err(|_| bad())? }.check()
    }
}

impl fmt::Display for Fixed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name())
    }
}

/// An affine format (plan 0.18.0 step 4, D207): an integer code `q` of a
/// saturating integer format that denotes `(q − z)·s`, its scale `s` and
/// its zero point `z` fixed. A quantized activation is stored in one, its
/// scale computed once from the magnitudes the envelope measured at its site
/// ([`Affine::absmax`], [`Affine::pow2`], [`Affine::minmax`]). Its values are
/// a grid of step `s` through zero: `z = 0` centres it, and a zero point
/// shifts the codes along it, the asymmetric form.
///
/// A value is rounded into it by its *rescaling*, `q = clamp(⌊x/s⌉ + z)`:
/// the integer nearest the exact quotient under `tie`, the zero point added
/// after the rounding, then the range's clamp. Two machines' readings differ
/// from it and are parameters (D203): the zero point added before the
/// rounding, `clamp(⌊x/s + z⌉)` ([`Affine::zero_rounded`]), and the quotient
/// computed in a float ([`Affine::quotient`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Affine {
    /// The code's format, which saturates (`i8/sat`): a rescaling clamps.
    pub int: Int,
    /// The scale's bits, a positive normal binary32 number.
    pub scale_bits: u32,
    /// An integer of `int`.
    pub zero: i64,
    pub tie: Tie,
    /// Whether the zero point is added before the rounding.
    pub zero_rounded: bool,
    /// The float the quotient is computed in, `f32` or `f64`: `x` rounded to
    /// it, then `x/s`, then the sum with `z` where it is rounded with it,
    /// each rounded to it, ties to even. `None` for the exact quotient.
    pub quotient: Option<Float>,
}

impl Affine {
    /// The format of codes `int` with the scale `scale` and the zero point
    /// `zero`, the quotient exact, ties to even, the zero point added after.
    pub fn new(int: Int, scale: f32, zero: i64) -> Result<Affine, String> {
        Affine { int, scale_bits: scale.to_bits(), zero, tie: Tie::Even, zero_rounded: false, quotient: None }.check()
    }

    pub fn check(self) -> Result<Affine, String> {
        if self.int.overflow != IntOverflow::Clamp {
            return Err(format!("an affine format's integer saturates, as a rescaling does: write {}/sat", self.int.name()));
        }
        if self.int.bits > 32 {
            return Err(format!("an affine format's integer has at most 32 bits, not {}", self.int.bits));
        }
        let s = f32::from_bits(self.scale_bits);
        if !(s.is_normal() && s > 0.0) {
            return Err(format!("an affine format's scale is a positive normal binary32 number, not {s}"));
        }
        if !(self.int.min()..=self.int.max()).contains(&i128::from(self.zero)) {
            return Err(format!("the zero point {} is not an integer of {}", self.zero, self.int.name()));
        }
        // What the emulation holds a value in.
        if !Format::Affine(self).exact_in_f64() {
            return Err(format!("an affine format's values are held in an f64, and those of {} codes at the scale {s} take more than its 53 bits: a scale of fewer significant bits, a power of two", self.int.name()));
        }
        if let Some(q) = self.quotient
            && !q.contains(F32)
        {
            return Err(format!("an affine format's quotient is computed in f32 or f64, which hold its scale, not in {q}"));
        }
        Ok(self)
    }

    pub fn scale(self) -> f64 {
        f32::from_bits(self.scale_bits) as f64
    }

    /// The value of the code `q`, `(q − z)·s`: exact where
    /// [`Format::exact_in_f64`].
    pub fn value(self, q: i128) -> f64 {
        (q - i128::from(self.zero)) as f64 * self.scale()
    }

    /// The code of `m·2^e`, rescaled, and whether the clamp gave it.
    pub fn round_scaled(self, m: i128, e: i32) -> (i128, bool) {
        let z = i128::from(self.zero);
        let z0 = if self.zero_rounded { z } else { 0 };
        let n = match self.quotient {
            Some(f) => {
                // As the float computes it: each step rounded to it once.
                let x = f.round_scaled(m, e, Tie::Even).0;
                let t = f.round(x / self.scale(), Tie::Even);
                let t = if z0 == 0 {
                    t
                } else {
                    let y = t + z0 as f64;
                    let bv = y - t;
                    Format::Float(f).round_exact(y, (t - (y - bv)) + (z0 as f64 - bv), Tie::Even).0
                };
                let (n, o) = Int { bits: 64, signed: true, overflow: IntOverflow::Clamp }.round_noted(t, self.tie);
                if o {
                    return self.ends(t);
                }
                n - z0
            }
            None => match quotient_round(m, e, self.scale(), z0, self.tie) {
                Some(n) => n,
                None => return self.ends(m as f64),
            },
        };
        let q = n + z;
        let y = self.int.fit(q);
        (y, y != q)
    }

    /// The end of the range a value of the sign of `x` saturates to, or the
    /// zero point for a NaN.
    fn ends(self, x: f64) -> (i128, bool) {
        if x.is_nan() {
            (i128::from(self.zero), true)
        } else if x > 0.0 {
            (self.int.max(), true)
        } else {
            (self.int.min(), true)
        }
    }

    /// The code of `x`, rescaled, and whether the clamp gave it. A NaN is
    /// the zero point, the value 0; an infinity the end of its sign.
    pub fn round_noted(self, x: f64) -> (i128, bool) {
        if !x.is_finite() {
            return self.ends(x);
        }
        if x == 0.0 {
            return self.round_scaled(0, 0);
        }
        let (m, e) = decompose(x);
        self.round_scaled(if x < 0.0 { -(m as i128) } else { m as i128 }, e)
    }

    /// The symmetric format whose largest code holds `m`, the largest `|x|`
    /// measured at a site: no zero point, `s = m / qmax` rounded to
    /// binary32, the practice and the tightest (the weights' `absmax`, D44).
    /// A site whose values are all zero takes `s = 1`.
    pub fn absmax(int: Int, m: f64) -> Result<Affine, String> {
        let s = if m > 0.0 && m.is_finite() { normal_f32(m / int.max() as f64) } else { 1.0 };
        Affine::new(int, s, 0)
    }

    /// The symmetric format whose scale is the smallest power of two with
    /// `m / s ≤ qmax`: up to twice as coarse as [`Affine::absmax`], and a
    /// code's value is then exact in every float that holds the code and the
    /// scale (the weights' `pow2`).
    pub fn pow2(int: Int, m: f64) -> Result<Affine, String> {
        let qmax = int.max() as f64;
        let mut k = 0i32;
        if m > 0.0 && m.is_finite() {
            while k > -126 && m / exp2(k - 1) <= qmax {
                k -= 1;
            }
            while k < 127 && m / exp2(k) > qmax {
                k += 1;
            }
        }
        Affine::new(int, exp2(k) as f32, 0)
    }

    /// The asymmetric format that spans `[lo, hi]`, the smallest and the
    /// largest value measured at a site, widened to hold zero exactly:
    /// `s = (hi − lo) / (qmax − qmin)` rounded to binary32 and the zero point
    /// `z = qmin − ⌊lo / s⌉`, in binary64, clamped to the range, as the
    /// practice computes them. A site whose values are all zero takes
    /// `s = 1` and the zero point 0.
    pub fn minmax(int: Int, lo: f64, hi: f64) -> Result<Affine, String> {
        let (lo, hi) = (lo.min(0.0), hi.max(0.0));
        let (qmin, qmax) = (int.min(), int.max());
        if !(hi > lo && lo.is_finite() && hi.is_finite()) {
            return Affine::new(int, 1.0, 0i128.clamp(qmin, qmax) as i64);
        }
        let s = normal_f32((hi - lo) / (qmax - qmin) as f64);
        let z = (qmin - (lo / s as f64).round_ties_even() as i128).clamp(qmin, qmax);
        Affine::new(int, s, z as i64)
    }

    /// `affine(i8/sat,scale=0.025,zero=-3)`, with `tie=away`,
    /// `rounds=x/s+zero` and `quotient=f32` where they are not the defaults;
    /// a power-of-two scale as `2^-7`.
    pub fn name(self) -> String {
        let s = f32::from_bits(self.scale_bits);
        let scale = if s.to_bits() & 0x7f_ffff == 0 {
            format!("2^{}", (s.to_bits() >> 23) as i32 - 127)
        } else if (1e-4..1e6).contains(&s) {
            format!("{s}")
        } else {
            format!("{s:e}")
        };
        let mut n = format!("affine({},scale={scale}", self.int.name());
        if self.zero != 0 {
            n.push_str(&format!(",zero={}", self.zero));
        }
        if self.tie == Tie::Away {
            n.push_str(",tie=away");
        }
        if self.zero_rounded {
            n.push_str(",rounds=x/s+zero");
        }
        if let Some(q) = self.quotient {
            n.push_str(&format!(",quotient={q}"));
        }
        n.push(')');
        n
    }

    pub fn parse(s: &str) -> Result<Affine, String> {
        let Some(body) = s.strip_prefix("affine(").and_then(|b| b.strip_suffix(')')) else {
            return Err(format!("unknown affine format `{s}`: expected affine(i8/sat,scale=0.025,zero=-3)"));
        };
        let mut items = body.split(',');
        let int = Int::parse(items.next().unwrap_or(""))?;
        let (mut scale, mut zero, mut tie, mut rounds, mut quotient) = (None, None, None, None, None);
        for item in items {
            let Some((key, value)) = item.split_once('=') else {
                return Err(format!("`{item}` in `{s}` is not a parameter `key=value`"));
            };
            let bad = || format!("`{value}` is not a value of `{key}` in `{s}`");
            let dup = |set: bool| if set { Err(format!("`{key}` is given twice in `{s}`")) } else { Ok(()) };
            match key {
                "scale" => {
                    dup(scale.is_some())?;
                    scale = Some(match value.strip_prefix("2^") {
                        Some(k) => k.parse::<i32>().ok().filter(|k| (-126..=127).contains(k)).map(|k| exp2(k) as f32).ok_or_else(bad)?,
                        None => value.parse::<f32>().map_err(|_| bad())?,
                    });
                }
                "zero" => {
                    dup(zero.is_some())?;
                    zero = Some(value.parse::<i64>().map_err(|_| bad())?);
                }
                "tie" => {
                    dup(tie.is_some())?;
                    tie = Some(match value {
                        "even" => Tie::Even,
                        "away" => Tie::Away,
                        _ => return Err(bad()),
                    });
                }
                "rounds" => {
                    dup(rounds.is_some())?;
                    rounds = Some(match value {
                        "x/s" => false,
                        "x/s+zero" => true,
                        _ => return Err(bad()),
                    });
                }
                "quotient" => {
                    dup(quotient.is_some())?;
                    quotient = Some(if value == "exact" { None } else { Some(Float::parse(value).map_err(|_| bad())?) });
                }
                _ => return Err(format!("unknown parameter `{key}` in `{s}`: expected scale, zero, tie, rounds or quotient")),
            }
        }
        let Some(scale) = scale else {
            return Err(format!("`{s}` does not give its scale"));
        };
        Affine { int, scale_bits: scale.to_bits(), zero: zero.unwrap_or(0), tie: tie.unwrap_or(Tie::Even), zero_rounded: rounds.unwrap_or(false), quotient: quotient.flatten() }.check()
    }
}

impl fmt::Display for Affine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name())
    }
}

impl serde::Serialize for Affine {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.name())
    }
}

/// By name, as [`Affine::parse`] reads it: a realization's `scaled` entry.
impl<'de> serde::Deserialize<'de> for Affine {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Affine, D::Error> {
        struct Name;
        impl serde::de::Visitor<'_> for Name {
            type Value = Affine;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("an affine format's name")
            }
            fn visit_str<E: serde::de::Error>(self, s: &str) -> Result<Affine, E> {
                Affine::parse(s).map_err(E::custom)
            }
        }
        d.deserialize_str(Name)
    }
}

/// `x` as a positive normal binary32 number: rounded to it, and brought
/// into its normal range.
fn normal_f32(x: f64) -> f32 {
    (x as f32).clamp(f32::MIN_POSITIVE, f32::MAX)
}

/// `⌊m·2^e / s + z0⌉ − z0` under `tie`, exactly, for a positive binary32
/// `s` and an integer `z0`: the integer nearest the quotient, a tie decided
/// on the integer `z0` is added to. `None` when the quotient is beyond
/// 2^62, past every range.
fn quotient_round(m: i128, e: i32, s: f64, z0: i128, tie: Tie) -> Option<i128> {
    if m == 0 {
        return Some(0);
    }
    let (ms, es) = decompose(s);
    let tz = ms.trailing_zeros();
    // `s = ms·2^es` with `ms` odd.
    let (ms, es) = (i128::from(ms >> tz), es + tz as i32);
    let k = e - es;
    let bits = 128 - m.unsigned_abs().leading_zeros() as i32;
    // The quotient's floor `f`, and how its fraction compares with ½.
    let (f, half) = if k >= 0 {
        if bits + k > 125 {
            return None;
        }
        let num = m << k;
        let (f, r) = (num.div_euclid(ms), num.rem_euclid(ms));
        (f, (2 * r).cmp(&ms))
    } else if -k > bits {
        // |x/s| < 1/2: the nearest integer to `z0 + x/s` is `z0`.
        return Some(0);
    } else {
        // `m / (ms·2^sh)` in two divisions, so that nothing overflows:
        // `m = a·2^sh + r`, `a = f·ms + b`, and the fraction is
        // `(b·2^sh + r) / (ms·2^sh)`, against ½ by the sign of
        // `(2b − ms)·2^sh + 2r`, `2b − ms` odd since `ms` is.
        let sh = -k as u32;
        let a = m >> sh;
        let r = m - (a << sh);
        let (f, b) = (a.div_euclid(ms), a.rem_euclid(ms));
        let d = 2 * b - ms;
        (f, if d >= 1 { Ordering::Greater } else if d <= -3 { Ordering::Less } else { r.cmp(&(1i128 << (sh - 1))) })
    };
    if f.unsigned_abs() > 1 << 62 {
        return None;
    }
    let up = match half {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => match tie {
            Tie::Even => (f + z0) & 1 == 1,
            Tie::Away => f + z0 >= 0,
        },
    };
    Some(f + up as i128)
}

/// A format a value is held or computed in: a float, an integer or a
/// fixed-point number. What a buffer the kernel interpreter emulates holds,
/// and what its operations are rounded to (plan 0.18.0 step 2, D205); an
/// affine format is a stored value's alone, never an operation's (step 4,
/// D207).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Format {
    Float(Float),
    Int(Int),
    Fixed(Fixed),
    Affine(Affine),
}

impl Format {
    /// Its name: a float's ([`Float::name`]), `i32`, `i16/sat`, or
    /// `fixed(i64,frac=18)`.
    pub fn name(self) -> Cow<'static, str> {
        match self {
            Format::Float(f) => f.name(),
            Format::Int(i) => Cow::Owned(i.name()),
            Format::Fixed(x) => Cow::Owned(x.name()),
            Format::Affine(a) => Cow::Owned(a.name()),
        }
    }

    pub fn parse(s: &str) -> Result<Format, String> {
        if s.starts_with("affine(") {
            return Affine::parse(s).map(Format::Affine);
        }
        if s.starts_with("fixed(") {
            return Fixed::parse(s).map(Format::Fixed);
        }
        if s.starts_with('i') || s.starts_with('u') {
            return Int::parse(s).map(Format::Int);
        }
        Float::parse(s).map(Format::Float)
    }

    /// Bits of one value.
    pub fn bits(self) -> u32 {
        match self {
            Format::Float(f) => f.bits(),
            Format::Int(i) => i.bits,
            Format::Fixed(x) => x.int.bits,
            Format::Affine(a) => a.int.bits,
        }
    }

    /// Whether an `f64` holds every value of the format exactly: every
    /// float, an integer or a fixed-point number of at most 53 bits, an
    /// affine format whose codes' distances to the zero point and scale's
    /// significand take 53 bits together.
    pub fn exact_in_f64(self) -> bool {
        match self {
            Format::Float(_) => true,
            Format::Int(i) => i.bits <= 53,
            Format::Fixed(x) => x.int.bits <= 53,
            Format::Affine(a) => {
                let z = i128::from(a.zero);
                let span = (a.int.max() - z).max(z - a.int.min()).unsigned_abs();
                let (m, _) = decompose(a.scale());
                (128 - span.leading_zeros()) + (64 - (m >> m.trailing_zeros()).leading_zeros()) <= 53
            }
        }
    }

    /// The value of the format nearest `x` under `tie`, as an `f64`, and
    /// whether the overflow rule gave it; exact when
    /// [`Format::exact_in_f64`].
    pub fn round_noted(self, x: f64, tie: Tie) -> (f64, bool) {
        match self {
            Format::Float(f) => f.round_noted(x, tie),
            Format::Int(i) => {
                let (n, o) = i.round_noted(x, tie);
                (n as f64, o)
            }
            Format::Fixed(q) => {
                let (m, o) = q.round_noted(x, tie);
                (q.value(m), o)
            }
            // The rescaling's own tie, which `tie` does not change.
            Format::Affine(a) => {
                let (q, o) = a.round_noted(x);
                (a.value(q), o)
            }
        }
    }

    /// [`Format::round_noted`] of the real `hi + lo`, rounded once, where
    /// `hi` is that real rounded to `f64` and `lo` the rest, at most half
    /// an ulp of `hi`, as an error-free transformation of an operation
    /// computed in `f64` gives them (plan 0.18.0 step 3, D206): what the
    /// operation is in any format, whatever its operands' formats.
    ///
    /// A float reads the rest's sign alone: its rounding position is at or
    /// above `hi`'s last bit, so the real rounds as `hi` unless `hi` is
    /// halfway between two of its numbers, where the rest decides; `lo` may
    /// then be any number of the rest's sign (a quotient's, a root's, which
    /// no transformation finds exactly). An integer or a fixed-point format
    /// reads the rest itself, an exact sum's or product's: below the grid's
    /// unit it decides the rounding, and above it, past the range, a wrap.
    pub fn round_exact(self, hi: f64, lo: f64, tie: Tie) -> (f64, bool) {
        if lo == 0.0 || !lo.is_finite() || !hi.is_finite() || hi == 0.0 || self == Format::Float(F64) {
            return self.round_noted(hi, tie);
        }
        // `hi` on a grid four times finer, and the rest one step of it: a
        // value of 55 bits that rounds as the real does wherever the
        // rounding position is at or above `hi`'s last bit.
        let sticky = |hi: f64, lo: f64| {
            let (m, e) = decompose(hi);
            let m = (m as i128) << 2;
            self.round_scaled(if hi < 0.0 { -m } else { m } + if lo > 0.0 { 1 } else { -1 }, e - 2, tie)
        };
        let frac = match self {
            Format::Float(_) => return sticky(hi, lo),
            Format::Int(_) => 0,
            Format::Fixed(x) => x.frac,
            // A grid of step `s`, which no binary grid finer than `hi`'s
            // last bit lines up with: the real itself, where its parts are
            // within 70 bits of each other; else the rest only decides a
            // tie of `hi`, since no other rounding boundary of a binary32
            // step lies that close to it. No operation rounds to an
            // affine format; this is a conversion's.
            Format::Affine(a) => {
                let ((mh, eh), (ml, el)) = (decompose(hi), decompose(lo));
                let (mh, ml) = (if hi < 0.0 { -(mh as i128) } else { mh as i128 }, if lo < 0.0 { -(ml as i128) } else { ml as i128 });
                let (m, e) = match eh - el {
                    d @ 0..=70 => ((mh << d) + ml, el),
                    d if d > 70 => ((mh << 72) + ml.signum(), eh - 72),
                    // A rest above `hi`'s last bit, which an error-free
                    // transformation never gives: the same, roles swapped.
                    d if d >= -70 => (mh + (ml << -d), eh),
                    _ => ((ml << 72) + mh.signum(), el - 72),
                };
                let (q, o) = a.round_scaled(m, e);
                return (a.value(q), o);
            }
        };
        // A part's units, `|x|·2^frac = m·2^e`: an integer of at most 126
        // bits (`Some`), a number with a fraction of a unit (`None`, and
        // below 2^53 units), or more units than that (`None`).
        let units = |x: f64| -> (Option<i128>, bool) {
            let (m, e) = decompose(x);
            let e = e + frac;
            let u = if e >= 0 {
                (e <= 73).then(|| (m as i128) << e)
            } else if -e <= 53 && m & ((1u64 << -e) - 1) == 0 {
                Some((m >> -e) as i128)
            } else {
                None
            };
            (u.map(|u| if x < 0.0 { -u } else { u }), e > 73)
        };
        match (units(hi), units(lo)) {
            ((Some(a), _), (Some(b), _)) => self.round_scaled(a + b, -frac, tie),
            // A whole number of units and a rest with a fraction: the rest
            // rounded beside it, a tie to the even total.
            ((Some(a), _), (None, _)) => {
                let x = lo * exp2(frac);
                let k = x.floor();
                let f = x - k;
                let k = k as i128;
                let up = if f != 0.5 {
                    f > 0.5
                } else {
                    match tie {
                        Tie::Even => (a + k) & 1 == 1,
                        Tie::Away => a + k >= 0,
                    }
                };
                self.round_scaled(a + k + up as i128, -frac, tie)
            }
            // A fraction of a unit in `hi`: the grid's unit is above its
            // last bit.
            ((None, false), _) => sticky(hi, lo),
            // More units than 2^126: past every range; a wrap of the rest,
            // since such a `hi` is a multiple of 2^bits units.
            ((None, true), _) => match self {
                Format::Int(Int { overflow: IntOverflow::Clamp, .. }) | Format::Fixed(Fixed { int: Int { overflow: IntOverflow::Clamp, .. }, .. }) => self.round_noted(hi, tie),
                _ => (self.round_exact(lo, 0.0, tie).0, true),
            },
        }
    }

    /// Whether every value of `other` is a value of this format, so that
    /// converting it changes nothing: a float of a float as
    /// [`Float::contains`] says; an integer or a fixed-point format, a grid
    /// of `m·2^-frac`, of a float without special values whose numbers all
    /// lie on its grid and in its range, or of another grid inside it; a
    /// float of a grid whose integers it holds, in its range.
    pub fn contains(self, other: Format) -> bool {
        // A grid as (frac, min, max): its values are m·2^-frac.
        let grid = |f: Format| match f {
            Format::Int(i) => Some((0, i.min(), i.max())),
            Format::Fixed(x) => Some((x.frac, x.int.min(), x.int.max())),
            Format::Float(_) | Format::Affine(_) => None,
        };
        match (self, other) {
            (Format::Float(a), Format::Float(b)) => a.contains(b),
            // The values `n·s`, `n` from `qmin − z` to `qmax − z`: their
            // significands, their last bit and their largest magnitude.
            (Format::Float(a), Format::Affine(o)) => {
                let z = i128::from(o.zero);
                let top = (o.int.max() - z).max(z - o.int.min()).unsigned_abs();
                let (m, e) = decompose(o.scale());
                let (m, e) = (u128::from(m >> m.trailing_zeros()), e + m.trailing_zeros() as i32);
                let finest = if a.subnormals { a.emin - (a.p as i32 - 1) } else { a.emin };
                top * m <= 1u128 << a.p && e >= finest && (top as f64) * o.scale() <= a.max_finite()
            }
            (Format::Affine(a), o) => Format::Affine(a) == o,
            (_, Format::Affine(_)) => false,
            (Format::Float(a), o) => {
                let (frac, lo, hi) = grid(o).expect("a grid");
                let top = lo.unsigned_abs().max(hi.unsigned_abs());
                top <= 1u128 << a.p && (a.subnormals && a.emin - (a.p as i32 - 1) <= -frac || -frac >= a.emin) && (top as f64) * exp2(-frac) <= a.max_finite()
            }
            (s, Format::Float(b)) => {
                let (frac, lo, hi) = grid(s).expect("a grid");
                // The last bit of the numbers at `emin`, the subnormals' too.
                let finest = b.emin - (b.p as i32 - 1);
                let limit = |m: i128| m as f64 * exp2(-frac);
                b.specials == Specials::None && finest >= -frac && b.max_finite() <= limit(hi) && -b.max_finite() >= limit(lo)
            }
            (s, o) => {
                let ((fs, ls, hs), (fo, lo, ho)) = (grid(s).expect("a grid"), grid(o).expect("a grid"));
                let up = fs - fo;
                (0..64).contains(&up) && lo.checked_shl(up as u32).is_some_and(|m| m >= ls) && ho.checked_shl(up as u32).is_some_and(|m| m <= hs)
            }
        }
    }

    /// [`Format::round_noted`] of `m·2^e`, rounded once.
    pub fn round_scaled(self, m: i128, e: i32, tie: Tie) -> (f64, bool) {
        match self {
            Format::Float(f) => f.round_scaled(m, e, tie),
            Format::Int(i) => {
                let (n, o) = i.round_scaled(m, e, tie);
                (n as f64, o)
            }
            Format::Fixed(q) => {
                let (u, o) = q.round_scaled(m, e, tie);
                (q.value(u), o)
            }
            Format::Affine(a) => {
                let (q, o) = a.round_scaled(m, e);
                (a.value(q), o)
            }
        }
    }
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name())
    }
}

impl serde::Serialize for Format {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.name())
    }
}

/// By name, as [`Format::parse`] reads it: a realization's `arith`,
/// `product` and `accum` (plan 0.18.0 step 3, D206).
impl<'de> serde::Deserialize<'de> for Format {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Format, D::Error> {
        struct Name;
        impl serde::de::Visitor<'_> for Name {
            type Value = Format;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a format's name")
            }
            fn visit_str<E: serde::de::Error>(self, s: &str) -> Result<Format, E> {
                Format::parse(s).map_err(E::custom)
            }
        }
        d.deserialize_str(Name)
    }
}

/// What a run's formats produced besides rounding (plan 0.18.0 step 2,
/// D205), the kernel interpreter's counterpart of a machine's
/// floating-point status flags. Each operation, conversion and constant
/// counts at most once.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    /// Results past the largest finite number, or an infinity or a NaN in a
    /// format without them, replaced by the largest finite number of their
    /// sign; integers clamped to their range.
    pub saturations: u64,
    /// Infinities from finite operands: an overflow, a division by zero.
    pub infinities: u64,
    /// NaNs from operands that were not: 0/0, ∞ − ∞, an overflow a format
    /// gives as NaN.
    pub nans: u64,
    /// Integers past their range brought back modulo 2^bits.
    pub wraps: u64,
}

impl Counts {
    pub fn add(&mut self, o: Counts) {
        self.saturations += o.saturations;
        self.infinities += o.infinities;
        self.nans += o.nans;
        self.wraps += o.wraps;
    }

    pub fn is_zero(&self) -> bool {
        *self == Counts::default()
    }

    /// Counts a result `y` of the format `f`, from operands among which an
    /// infinity (`inf_in`) or a NaN (`nan_in`), `o` whether the format's
    /// overflow rule gave it: an operation's, a conversion's.
    pub fn note(&mut self, f: Format, y: f64, o: bool, inf_in: bool, nan_in: bool) {
        match f {
            Format::Float(_) => {
                if o && y.is_finite() {
                    self.saturations += 1;
                } else if y.is_infinite() && !inf_in {
                    self.infinities += 1;
                } else if y.is_nan() && !nan_in {
                    self.nans += 1;
                }
            }
            Format::Int(i) | Format::Fixed(Fixed { int: i, .. }) => {
                if o {
                    if i.overflow == IntOverflow::Wrap && !inf_in && !nan_in { self.wraps += 1 } else { self.saturations += 1 }
                }
            }
            // A rescaling clamps: past the range, a NaN, an infinity.
            Format::Affine(_) => {
                if o {
                    self.saturations += 1;
                }
            }
        }
    }

    /// What these counts hold beyond `base`, counts taken earlier of the
    /// same tally.
    pub fn since(self, base: Counts) -> Counts {
        Counts { saturations: self.saturations - base.saturations, infinities: self.infinities - base.infinities, nans: self.nans - base.nans, wraps: self.wraps - base.wraps }
    }
}

impl fmt::Display for Counts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = |k: u64, one: &str, many: &str| format!("{k} {}", if k == 1 { one } else { many });
        write!(f, "{}, {}, {}, {}", n(self.saturations, "saturation", "saturations"), n(self.infinities, "infinity", "infinities"), n(self.nans, "NaN", "NaNs"), n(self.wraps, "wrap", "wraps"))
    }
}

/// The part of a tensor one scale covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Granularity {
    /// One scale for the whole tensor.
    Tensor,
    /// One per index of every axis but one, the axis the tensor is
    /// contracted over, so that the scale factors out of the sum.
    Channel,
    /// One per block of this many elements along an axis.
    Block(u32),
}

/// The format a scale is held in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ScaleFormat {
    /// A float of this format, the scale `m / qmax` rounded to it.
    Float(Float),
    /// A power of two, the smallest with `m / s ≤ qmax`: `q·s` is then
    /// exact in any float that holds `q` and `s`.
    Pow2,
}

/// A scaled format: an integer `q` that denotes `q·s` (`(q − z)·s` with a
/// zero point), the scale `s` shared by a part of the tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Scaled {
    pub int: Int,
    /// The range is symmetric, `−qmax..=qmax`, `qmax = 2^(b−1) − 1`, so
    /// that `−2^(b−1)` is never used and negation is exact.
    pub symmetric: bool,
    pub granularity: Granularity,
    pub scale: ScaleFormat,
    /// Whether an integer zero point shifts the range (the asymmetric
    /// form).
    pub zero_point: bool,
}

impl Scaled {
    /// The weights' integers of plan 0.6.0 (D44): a signed integer of
    /// `bits`, symmetric, one scale per channel in `f32` (absmax) or a
    /// power of two (`pow2`), no zero point.
    pub fn weights(bits: u32, pow2: bool) -> Scaled {
        Scaled { int: Int { bits, signed: true, overflow: IntOverflow::Clamp }, symmetric: true, granularity: Granularity::Channel, scale: if pow2 { ScaleFormat::Pow2 } else { ScaleFormat::Float(F32) }, zero_point: false }
    }

    /// The largest integer magnitude used.
    pub fn qmax(self) -> i128 {
        self.int.max()
    }

    /// The smallest integer used: `−qmax` when symmetric.
    pub fn qmin(self) -> i128 {
        if self.symmetric { -self.int.max() } else { self.int.min() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The runtime's `f32_to_bf16` followed by its decoding, as
    /// `presk-rt` and `presk-interp` compute it.
    fn bf16_of(x: f32) -> f32 {
        if x.is_nan() {
            return x;
        }
        let bits = x.to_bits();
        let lsb = (bits >> 16) & 1;
        f32::from_bits(((bits + 0x7fff + lsb) >> 16) << 16)
    }

    fn same(a: f64, b: f64) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    #[test]
    fn the_catalogue_is_well_formed() {
        for (name, f) in ALIASES.iter().copied().chain([("", E4M3_SATURATING)]) {
            assert_eq!(f.check(), Ok(f), "{name}");
        }
        let widths: Vec<u32> = [F64, F32, BF16, F16, E8M15, E5M2, E4M3FN, E4M3_SATURATING, E2M1].iter().map(|f| f.bits()).collect();
        assert_eq!(widths, [64, 32, 16, 16, 24, 8, 8, 8, 4]);
        assert_eq!(F64.max_finite(), f64::MAX);
        assert_eq!(F32.max_finite(), f32::MAX as f64);
        assert_eq!(F64.min_positive(), 5e-324);
        assert_eq!(F32.min_positive(), f32::from_bits(1) as f64);
        assert_eq!(F16.max_finite(), 65504.0);
        assert_eq!(E4M3FN.max_finite(), 448.0);
        assert_eq!(E4M3_SATURATING.max_finite(), 480.0);
        assert_eq!(E2M1.max_finite(), 6.0);
        assert_eq!(E5M2.max_finite(), 57344.0);
        assert_eq!(E4M3_SATURATING.min_positive(), 2f64.powi(-9));
    }

    #[test]
    fn the_unit_roundoffs_are_the_runtime_s() {
        assert_eq!(F64.unit_roundoff().to_bits(), (f64::EPSILON / 2.0).to_bits());
        assert_eq!(F32.unit_roundoff().to_bits(), (f32::EPSILON as f64 / 2.0).to_bits());
        assert_eq!(BF16.unit_roundoff().to_bits(), 2f64.powi(-8).to_bits());
        assert_eq!(F16.unit_roundoff(), 2f64.powi(-11));
    }

    #[test]
    fn names_round_trip_and_spellings_resolve_to_aliases() {
        for (name, f) in ALIASES {
            assert_eq!(Float::parse(name), Ok(*f));
            assert_eq!(f.name(), *name);
            assert_eq!(Float::parse(&f.spelling()), Ok(*f), "{}", f.spelling());
        }
        assert_eq!(E4M3_SATURATING.name(), "float(p=4,emin=-6,emax=8,specials=none,underflow=+0)");
        assert_eq!(Float::parse("float(p=4,emin=-6,emax=8,underflow=+0,specials=none)"), Ok(E4M3_SATURATING));
        assert_eq!(Float::parse("float(p=8,emin=-126,emax=127)"), Ok(BF16));
        let saturating = Float::parse("float(p=3,emin=-14,emax=15,overflow=saturate)").unwrap();
        assert_eq!(saturating.name(), "float(p=3,emin=-14,emax=15,overflow=saturate)");
        for bad in ["f8", "float(p=4)", "float(p=4,emin=-6,emax=8)", "float(p=4,emin=-6,emax=8,specials=none,overflow=inf)", "float(p=1,emin=0,emax=1,specials=none)", "float(p=4,p=4,emin=-6,emax=8,specials=none)", "float(p=4,emin=-6,emax=8,specials=none,colour=red)"] {
            assert!(Float::parse(bad).is_err(), "{bad}");
        }
    }

    /// A default-NaN format (D209): every NaN rounded into it the positive
    /// quiet one, the format otherwise binary16; refused without NaNs.
    #[test]
    fn a_default_nan_format_gives_the_positive_quiet_nan() {
        let f = Float::parse("float(p=11,emin=-14,emax=15,nans=default)").unwrap();
        assert_eq!(f, Float { nans: Nans::Default, ..F16 });
        assert_eq!(f.name(), "float(p=11,emin=-14,emax=15,nans=default)");
        assert_eq!(Float::parse("float(p=11,emin=-14,emax=15,nans=operand)"), Ok(F16));
        let (y, o) = f.round_noted(-f64::NAN, Tie::Even);
        assert!(y.is_nan() && y.is_sign_positive() && !o);
        assert_eq!(f.encode(y), Some(0x7e00));
        assert!(F16.round(-f64::NAN, Tie::Even).is_sign_negative());
        assert_eq!(f.round(-2.5, Tie::Even), -2.5);
        assert!(Float::parse("float(p=4,emin=-6,emax=8,specials=none,nans=default)").is_err());
    }

    #[test]
    fn serialized_as_its_name() {
        let m = std::collections::BTreeMap::from([(BF16, 3.0), (F64, 1.0), (F32, 2.0)]);
        assert_eq!(serde_json::to_string(&m).unwrap(), r#"{"f64":1.0,"f32":2.0,"bf16":3.0}"#);
        let back: std::collections::BTreeMap<Float, f64> = serde_json::from_str(r#"{"bf16":3.0,"f32":2.0,"f64":1.0}"#).unwrap();
        assert_eq!(back, m);
        assert_eq!(serde_json::to_string(&E4M3_SATURATING).unwrap(), r#""float(p=4,emin=-6,emax=8,specials=none,underflow=+0)""#);
        assert!(serde_json::from_str::<Float>(r#""f8""#).is_err());
    }

    #[test]
    fn ordered_most_precise_first() {
        let mut v = vec![E2M1, BF16, F16, F64, E4M3FN, F32, E8M15, E5M2];
        v.sort();
        assert_eq!(v, [F64, F32, E8M15, F16, BF16, E4M3FN, E5M2, E2M1]);
    }

    #[test]
    fn inclusions() {
        assert!(F64.contains(F32) && F32.contains(BF16) && F64.contains(BF16));
        assert!(!F32.contains(F64) && !BF16.contains(F32));
        assert!(F32.contains(F16) && F32.contains(E8M15) && F32.contains(E4M3FN) && F32.contains(E4M3_SATURATING) && F32.contains(E2M1));
        assert!(!BF16.contains(F16) && !F16.contains(BF16));
        assert!(F16.contains(E4M3_SATURATING) && F16.contains(E5M2) && F16.contains(E2M1));
        assert!(!E4M3FN.contains(E4M3_SATURATING) && !E4M3_SATURATING.contains(E4M3FN));
        assert!(BF16.contains(E4M3FN) && BF16.contains(E5M2) && !E5M2.contains(E4M3FN));
    }

    /// Edge values of every kind, positive and negative.
    fn edges() -> Vec<f64> {
        let mut v = vec![0.0, 1.0, 1.5, 2.0, 3.0, 0.1, 1.0 / 3.0, 65504.0, 65519.0, 65520.0, 65536.0, 448.0, 464.0, 480.0, 496.0, 500.0, 6.0, 5.0, 7.0, 0.25, 0.75, 1e-5, 6e-8, 3e-8, 2.98e-8, 1e-9, 1e-40, 1e-45, 1e-46, 3.4e38, 3.4028235677973366e38, 3.402823669209385e38, 1e300, f64::MAX, f64::MIN_POSITIVE, 5e-324, f64::INFINITY, f64::NAN];
        for k in -30..20 {
            for m in [1.0, 1.0625, 1.125, 1.5, 1.9375, 1.96875] {
                v.push(m * 2f64.powi(k));
            }
        }
        let mut s = 0x2545f4914f6cdd1du64;
        for _ in 0..20_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let x = f64::from_bits(s);
            if x.is_finite() {
                v.push(x);
                v.push(x.abs().log2().fract() * 2f64.powi((s % 80) as i32 - 40));
            }
        }
        let neg: Vec<f64> = v.iter().map(|x| -x).collect();
        v.extend(neg);
        v
    }

    #[test]
    fn f32_s_rounding_is_rust_s_cast() {
        for x in edges() {
            assert!(same(F32.round(x, Tie::Even), x as f32 as f64), "{x:e}: {:e} against {:e}", F32.round(x, Tie::Even), x as f32 as f64);
            assert!(same(F64.round(x, Tie::Even), x), "{x:e}");
        }
    }

    #[test]
    fn bf16_s_store_is_the_runtime_s() {
        for x in edges() {
            let runtime = bf16_of(x as f32) as f64;
            assert!(same(BF16.store(x), runtime), "{x:e}: {:e} against {runtime:e}", BF16.store(x));
        }
    }

    /// The rounding of every `f32` to `bf16` is the runtime's: 2³² inputs.
    /// About a minute in a release build (`cargo test --release -p
    /// presk-format -- --ignored`).
    #[test]
    #[ignore]
    fn bf16_on_every_f32() {
        for b in 0..=u32::MAX {
            let x = f32::from_bits(b);
            assert!(same(BF16.round(x as f64, Tie::Even), bf16_of(x) as f64), "{b:#010x}");
        }
    }

    /// The rounding to `f32` is Rust's cast on every double where it is
    /// hard: each midpoint between two neighbouring positive `f32`s, the
    /// double below it and the double above (3·2³¹ inputs, about two minutes
    /// in a release build).
    #[test]
    #[ignore]
    fn f32_on_every_midpoint() {
        for b in 0..0x7f80_0000u32 {
            let (lo, hi) = (f32::from_bits(b) as f64, f32::from_bits(b + 1) as f64);
            let mid = (lo + hi) / 2.0;
            for x in [mid, f64::from_bits(mid.to_bits() - 1), f64::from_bits(mid.to_bits() + 1)] {
                assert!(same(F32.round(x, Tie::Even), x as f32 as f64), "{b:#010x}: {x:e}");
            }
        }
    }

    #[test]
    fn binary16_by_its_rules() {
        assert_eq!(F16.round(65519.0, Tie::Even), 65504.0);
        assert_eq!(F16.round(65520.0, Tie::Even), f64::INFINITY);
        assert_eq!(F16.round(-65520.0, Tie::Away), f64::NEG_INFINITY);
        assert_eq!(F16.round(2f64.powi(-24), Tie::Even), 2f64.powi(-24));
        assert_eq!(F16.round(2f64.powi(-25), Tie::Even), 0.0);
        assert_eq!(F16.round(2f64.powi(-25), Tie::Away), 2f64.powi(-24));
        assert!(F16.round(-2f64.powi(-26), Tie::Even).is_sign_negative());
        assert_eq!(F16.round(1.0 + 2f64.powi(-11), Tie::Even), 1.0);
        assert_eq!(F16.round(1.0 + 2f64.powi(-11), Tie::Away), 1.0 + 2f64.powi(-10));
        assert_eq!(F16.round(1.0 + 3.0 * 2f64.powi(-11), Tie::Even), 1.0 + 2f64.powi(-9));
    }

    #[test]
    fn the_saturating_e4m3() {
        let f = E4M3_SATURATING;
        assert_eq!(f.round(1000.0, Tie::Even), 480.0);
        assert_eq!(f.round(f64::INFINITY, Tie::Even), 480.0);
        assert_eq!(f.round(f64::NEG_INFINITY, Tie::Even), -480.0);
        assert_eq!(f.round(-f64::NAN, Tie::Even), -480.0);
        assert_eq!(f.round(f64::NAN, Tie::Even), 480.0);
        assert_eq!(f.round(496.0, Tie::Even), 480.0);
        assert_eq!(f.round(-2f64.powi(-11), Tie::Even).to_bits(), 0.0f64.to_bits());
        assert_eq!(f.round(-0.0, Tie::Even).to_bits(), 0.0f64.to_bits());
        assert_eq!(f.round(3.0 * 2f64.powi(-10), Tie::Even), 2.0 * 2f64.powi(-9));
        assert!(f.holds(-0.0));
        assert_eq!(f.encode(-0.0), Some(0x80));
        assert_eq!(f.encode(480.0), Some(0x7f));
        assert_eq!(f.encode(-480.0), Some(0xff));
        assert_eq!(f.encode(2f64.powi(-9)), Some(0x01));
        assert_eq!(f.encode(2f64.powi(-6)), Some(0x08));
        assert_eq!(f.encode(f64::NAN), None);
        for c in 0..256u64 {
            let x = f.decode(c);
            let (e, m) = (c >> 3 & 15, c & 7);
            let v = if e == 0 { m as f64 * 2f64.powi(-9) } else { (8 + m) as f64 * 2f64.powi(e as i32 - 10) };
            assert_eq!(x, if c & 0x80 == 0 { v } else { -v }, "{c:#04x}");
            assert_eq!(f.encode(x), Some(c), "{c:#04x}");
        }
    }

    #[test]
    fn ofp8_and_fp4() {
        assert_eq!(E4M3FN.round(464.0, Tie::Even), 448.0);
        assert!(E4M3FN.round(465.0, Tie::Even).is_nan());
        assert!(E4M3FN.round(f64::INFINITY, Tie::Even).is_nan());
        assert!(E4M3FN.decode(0x7f).is_nan() && E4M3FN.decode(0xff).is_nan());
        assert_eq!(E4M3FN.decode(0x7e), 448.0);
        assert_eq!(E4M3FN.encode(f64::NAN), Some(0x7f));
        assert_eq!(E5M2.decode(0x7c), f64::INFINITY);
        assert!(E5M2.decode(0x7d).is_nan());
        assert_eq!(E5M2.encode(f64::NAN), Some(0x7e));
        let fp4: Vec<f64> = (0..8).map(|c| E2M1.decode(c)).collect();
        assert_eq!(fp4, [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]);
        assert_eq!(E2M1.round(5.0, Tie::Even), 4.0);
        assert_eq!(E2M1.round(5.0, Tie::Away), 6.0);
        assert_eq!(E2M1.round(100.0, Tie::Even), 6.0);
        assert_eq!(E2M1.round(0.25, Tie::Even), 0.0);
        assert_eq!(E2M1.round(0.25, Tie::Away), 0.5);
    }

    #[test]
    fn codes_round_trip() {
        for f in [F16, BF16, E5M2, E4M3FN, E4M3_SATURATING, E2M1] {
            for c in 0..(1u64 << f.bits()) {
                let x = f.decode(c);
                if x.is_nan() {
                    continue;
                }
                assert!(f.holds(x), "{f} {c:#x}");
                assert_eq!(f.encode(x), Some(c), "{f} {c:#x} = {x:e}");
            }
        }
        for x in edges() {
            let y = x as f32;
            if !y.is_nan() {
                assert_eq!(F32.encode(y as f64), Some(y.to_bits() as u64), "{y:e}");
            }
            if let Some(c) = F64.encode(x) {
                assert_eq!(c, x.to_bits(), "{x:e}");
            }
            let b = bf16_of(y);
            if !b.is_nan() {
                assert_eq!(BF16.encode(b as f64), Some((b.to_bits() >> 16) as u64), "{b:e}");
            }
        }
        assert_eq!(F16.encode(0.1), None);
        assert_eq!(F16.encode(1.0), Some(0x3c00));
        assert_eq!(F16.decode(0x7bff), 65504.0);
    }

    #[test]
    fn no_subnormals_flush_to_zero_or_the_least_normal() {
        let f = Float { subnormals: false, ..F16 }.check().unwrap();
        assert_eq!(f.round(2f64.powi(-15), Tie::Even), 0.0);
        assert_eq!(f.round(2f64.powi(-15), Tie::Away), 2f64.powi(-14));
        assert_eq!(f.round(1.5 * 2f64.powi(-15), Tie::Even), 2f64.powi(-14));
        assert_eq!(f.min_positive(), 2f64.powi(-14));
        assert_eq!(f.name(), "float(p=11,emin=-14,emax=15,subnormals=no)");
    }

    #[test]
    fn integers() {
        let w = Int::parse("i32").unwrap();
        assert_eq!(w, Int { bits: 32, signed: true, overflow: IntOverflow::Wrap });
        assert_eq!(w.fit(1i128 << 31), -(1i128 << 31));
        let s = Int::parse("i16/sat").unwrap();
        assert_eq!(s.fit(40000), 32767);
        assert_eq!(s.fit(-40000), -32768);
        assert_eq!(s.name(), "i16/sat");
        assert_eq!(Int::parse("u8").unwrap().max(), 255);
        assert_eq!(s.round(2.5, Tie::Even), 2);
        assert_eq!(s.round(2.5, Tie::Away), 3);
        assert_eq!(s.round(-2.5, Tie::Away), -3);
        assert!(Int::parse("i0").is_err() && Int::parse("f32").is_err());
        let q = Scaled::weights(8, false);
        assert_eq!((q.qmin(), q.qmax()), (-127, 127));
        assert_eq!(Scaled::weights(4, true).qmax(), 7);
    }

    #[test]
    fn overflows_are_noted() {
        assert_eq!(F16.round_noted(65519.0, Tie::Even), (65504.0, false));
        assert_eq!(F16.round_noted(65520.0, Tie::Even), (f64::INFINITY, true));
        assert_eq!(F16.round_noted(f64::NEG_INFINITY, Tie::Even), (f64::NEG_INFINITY, false));
        assert_eq!(E4M3_SATURATING.round_noted(496.0, Tie::Even), (480.0, true));
        assert_eq!(E4M3_SATURATING.round_noted(495.0, Tie::Even), (480.0, false));
        assert_eq!(E4M3_SATURATING.round_noted(-f64::NAN, Tie::Even), (-480.0, true));
        assert_eq!(E4M3_SATURATING.round_noted(f64::INFINITY, Tie::Even), (480.0, true));
        let (y, o) = E4M3FN.round_noted(465.0, Tie::Even);
        assert!(y.is_nan() && o);
        assert_eq!(Int::parse("i8/sat").unwrap().round_noted(127.5, Tie::Even), (127, true));
        assert_eq!(Int::parse("i8/sat").unwrap().round_noted(126.5, Tie::Even), (126, false));
        assert_eq!(Int::parse("i8").unwrap().round_noted(128.0, Tie::Even), (-128, true));
    }

    #[test]
    fn scaled_integers_round_once() {
        // Every finite double m·2^e rounds as the double does.
        for x in edges() {
            if !x.is_finite() || x == 0.0 {
                continue;
            }
            let (m, e) = decompose(x);
            let m = if x < 0.0 { -(m as i128) } else { m as i128 };
            for f in [F64, F32, BF16, F16, E8M15, E5M2, E4M3FN, E4M3_SATURATING, E2M1] {
                for tie in [Tie::Even, Tie::Away] {
                    let (a, b) = (f.round_scaled(m, e, tie), f.round_noted(x, tie));
                    assert!(same(a.0, b.0) && a.1 == b.1, "{f} {x:e}: {:e} against {:e}", a.0, b.0);
                }
            }
        }
        // A value no double holds, 1 + 2^-11 + 2^-60: once to binary16 it
        // is above the midpoint; through f64 it is the midpoint, and goes to
        // even.
        let m = (1i128 << 60) + (1i128 << 49) + 1;
        assert_eq!(F16.round_scaled(m, -60, Tie::Even), (1.0 + 2f64.powi(-10), false));
        assert_eq!(F16.round(F64.round_scaled(m, -60, Tie::Even).0, Tie::Even), 1.0);
        assert_eq!(F16.round_scaled(1i128 << 100, 0, Tie::Even), (f64::INFINITY, true));
        assert_eq!(F16.round_scaled(-(1i128 << 16), 0, Tie::Even), (f64::NEG_INFINITY, true));
        assert_eq!(F16.round_scaled(1, -25, Tie::Even), (0.0, false));
        assert_eq!(F16.round_scaled(3, -26, Tie::Even), (2f64.powi(-24), false));
        assert_eq!(F64.round_scaled(i128::MIN, 0, Tie::Even), (-(2f64.powi(127)), false));
        let i8s = Int::parse("i8/sat").unwrap();
        assert_eq!(i8s.round_scaled(5, -1, Tie::Even), (2, false));
        assert_eq!(i8s.round_scaled(5, -1, Tie::Away), (3, false));
        assert_eq!(i8s.round_scaled(-5, -1, Tie::Away), (-3, false));
        assert_eq!(i8s.round_scaled(-7, -1, Tie::Even), (-4, false));
        assert_eq!(i8s.round_scaled(255, -1, Tie::Even), (127, true));
        assert_eq!(i8s.round_scaled(3, 7, Tie::Even), (127, true));
        assert_eq!(Int::parse("i8").unwrap().round_scaled(3, 7, Tie::Even), (-128, true));
        assert_eq!(Int::parse("i8").unwrap().round_scaled(1, 200, Tie::Even), (0, true));
    }

    #[test]
    fn fixed_points_and_formats() {
        let x = Fixed::parse("fixed(i64,frac=18)").unwrap();
        assert_eq!(x, Fixed { int: Int { bits: 64, signed: true, overflow: IntOverflow::Wrap }, frac: 18 });
        assert_eq!(x.name(), "fixed(i64,frac=18)");
        // The least product of two E4M3 values, and the largest.
        assert_eq!(x.round_noted(2f64.powi(-18), Tie::Even), (1, false));
        assert_eq!(x.round_noted(-230_400.0, Tie::Even), (-230_400 << 18, false));
        assert_eq!(x.round_noted(2f64.powi(-19), Tie::Even), (0, false));
        assert_eq!(x.value(3), 3.0 * 2f64.powi(-18));
        assert_eq!(x.value((1i128 << 60) + 1), 2f64.powi(42));
        let s = Fixed::parse("fixed(i16/sat,frac=8)").unwrap();
        assert_eq!(s.round_noted(200.0, Tie::Even), (32767, true));
        assert_eq!(s.round_scaled(3, -9, Tie::Even), (2, false));
        for bad in ["fixed(i64)", "fixed(f32,frac=3)", "fixed(i64,frac=-1)", "fixed(i64,frac=256)"] {
            assert!(Fixed::parse(bad).is_err(), "{bad}");
        }
        for (name, f) in [("f16", Format::Float(F16)), ("i32", Format::Int(Int { bits: 32, signed: true, overflow: IntOverflow::Wrap })), ("i16/sat", Format::Int(Int { bits: 16, signed: true, overflow: IntOverflow::Clamp })), ("fixed(i64,frac=18)", Format::Fixed(x))] {
            assert_eq!(Format::parse(name), Ok(f));
            assert_eq!(f.name(), name);
        }
        assert_eq!(Format::parse(&E4M3_SATURATING.name()), Ok(Format::Float(E4M3_SATURATING)));
        assert!(Format::parse("int8").is_err() && Format::parse("f8").is_err());
        assert_eq!([Format::Float(E2M1), Format::Int(Int::parse("i32").unwrap()), Format::Fixed(x)].map(|f| f.bits()), [4, 32, 64]);
        assert!(Format::parse("i32").unwrap().exact_in_f64() && !Format::Fixed(x).exact_in_f64());
        assert_eq!(Format::parse("i16/sat").unwrap().round_noted(-40000.0, Tie::Even), (-32768.0, true));
        assert_eq!(Format::Fixed(s).round_noted(0.01, Tie::Even), (3.0 / 256.0, false));
    }

    #[test]
    fn an_affine_format_rescales_once() {
        let i8s = Int::parse("i8/sat").unwrap();
        let a = Affine::new(i8s, 0.25, 3).unwrap();
        assert_eq!(a.name(), "affine(i8/sat,scale=2^-2,zero=3)");
        assert_eq!(a.round_noted(1.0), (7, false));
        assert_eq!(a.value(7), 1.0);
        // −0.375 / 0.25 = −1.5: −2 to even, then the zero point; with the
        // zero point inside, 1.5 to even. The two readings of the target
        // note's affine form (D203).
        assert_eq!(a.round_noted(-0.375), (1, false));
        assert_eq!(a.value(1), -0.5);
        let inside = Affine { zero_rounded: true, ..a };
        assert_eq!(inside.round_noted(-0.375), (2, false));
        assert_eq!(inside.value(2), -0.25);
        let away = Affine { tie: Tie::Away, ..a };
        assert_eq!(away.round_noted(-0.375), (1, false));
        assert_eq!(Affine { tie: Tie::Away, ..inside }.round_noted(-0.375), (2, false));
        assert_eq!(away.round_noted(0.625), (6, false));
        assert_eq!(a.round_noted(0.625), (5, false));
        // The clamp, and the values no number is.
        assert_eq!(a.round_noted(1000.0), (127, true));
        assert_eq!(a.round_noted(-1000.0), (-128, true));
        assert_eq!(a.round_noted(f64::INFINITY), (127, true));
        assert_eq!(a.round_noted(f64::NAN), (3, true));
        assert_eq!(a.round_noted(0.0), (3, false));
        assert_eq!(a.round_noted(-0.0), (3, false));
        assert_eq!(a.round_noted(1e-300), (3, false));
        assert_eq!(a.round_noted(1e300), (127, true));
        // A scale that is no power of two, exactly: 0.1f32 is above 0.1,
        // so 0.25 / 0.1f32 is below 2.5 and rounds down.
        let t = Affine::new(i8s, 0.1, 0).unwrap();
        assert_eq!(t.round_noted(0.25), (2, false));
        assert_eq!(t.round_noted(0.1f32 as f64 * 2.5), (2, false));
        assert_eq!(t.round_noted(0.1f32 as f64 * 3.5), (4, false));
        assert_eq!(Format::Affine(t).round_scaled(5, -1, Tie::Even), (t.value(25), false));
        // In binary32 (`quotient=f32`): 2^24 + 1 is rounded to 2^24 first.
        let i32s = Int::parse("i32/sat").unwrap();
        let w = Affine::new(i32s, 1.0, 0).unwrap();
        assert_eq!(w.round_noted(16777217.0), (16777217, false));
        assert_eq!(Affine { quotient: Some(F32), ..w }.round_noted(16777217.0), (16777216, false));
        // The rest of an exact sum decides a tie the leading part makes.
        assert_eq!(Format::Affine(a).round_exact(-0.375, 1e-30, Tie::Even), (-0.25, false));
        assert_eq!(Format::Affine(a).round_exact(-0.375, -1e-30, Tie::Even), (-0.5, false));
        assert_eq!(Format::Affine(a).round_exact(-0.375, -0.0625, Tie::Even), (-0.5, false));
    }

    #[test]
    fn affine_scales_from_magnitudes() {
        let (i8s, u8s) = (Int::parse("i8/sat").unwrap(), Int::parse("u8/sat").unwrap());
        let a = Affine::absmax(i8s, 3.0).unwrap();
        assert_eq!((a.scale(), a.zero), ((3.0f64 / 127.0) as f32 as f64, 0));
        assert_eq!(a.round_noted(3.0).0, 127);
        assert_eq!(a.round_noted(-3.0).0, -127);
        let p = Affine::pow2(i8s, 3.0).unwrap();
        assert_eq!(p.name(), "affine(i8/sat,scale=2^-5)");
        assert_eq!(Affine::pow2(i8s, 127.0).unwrap().scale(), 1.0);
        assert_eq!(Affine::pow2(i8s, 127.5).unwrap().scale(), 2.0);
        let m = Affine::minmax(u8s, -1.0, 3.0).unwrap();
        assert_eq!((m.scale(), m.zero), ((4.0f64 / 255.0) as f32 as f64, 64));
        assert_eq!(m.value(m.round_noted(0.0).0), 0.0);
        // Widened to hold zero; all zero; past binary32's normal range.
        assert_eq!(Affine::minmax(u8s, 1.0, 3.0).unwrap().zero, 0);
        assert_eq!(Affine::minmax(i8s, 0.0, 0.0).unwrap(), Affine::new(i8s, 1.0, 0).unwrap());
        assert_eq!(Affine::absmax(i8s, 0.0).unwrap().scale(), 1.0);
        assert_eq!(Affine::absmax(i8s, 1e-60).unwrap().scale(), f32::MIN_POSITIVE as f64);
        assert_eq!(Affine::absmax(i8s, 1e60).unwrap().scale(), f32::MAX as f64);
        assert_eq!(Affine::pow2(i8s, 1e-60).unwrap().scale(), f32::MIN_POSITIVE as f64);
    }

    #[test]
    fn affine_names_and_what_holds_them() {
        for name in ["affine(i8/sat,scale=2^-2,zero=3)", "affine(u8/sat,scale=0.015686275,zero=64)", "affine(i8/sat,scale=0.25,zero=3,tie=away,rounds=x/s+zero,quotient=f32)", "affine(i16/sat,scale=1.5e-9)", "affine(i4/sat,scale=2^-126)"] {
            let f = Format::parse(name).unwrap();
            assert_eq!(Format::parse(&f.name()), Ok(f), "{name}");
        }
        assert_eq!(Affine::parse("affine(i8/sat,scale=0.25,zero=3,tie=away,rounds=x/s+zero,quotient=f32)").unwrap().name(), "affine(i8/sat,scale=2^-2,zero=3,tie=away,rounds=x/s+zero,quotient=f32)");
        assert_eq!(Affine::parse("affine(i8/sat,scale=0.5,zero=0,tie=even,rounds=x/s,quotient=exact)").unwrap().name(), "affine(i8/sat,scale=2^-1)");
        for bad in ["affine(i8,scale=0.25)", "affine(i8/sat)", "affine(i8/sat,scale=0)", "affine(i8/sat,scale=-1)", "affine(i8/sat,scale=1e-40)", "affine(i8/sat,scale=0.5,zero=200)", "affine(u8/sat,scale=0.5,zero=-1)", "affine(i64/sat,scale=0.5)", "affine(i8/sat,scale=0.5,quotient=f16)", "affine(i8/sat,scale=0.5,scale=1)", "affine(i8/sat,scale=0.5,tie=up)", "affine(i8/sat,scale=2^-127)", "affine(f32,scale=1)"] {
            assert!(Format::parse(bad).is_err(), "{bad}");
        }
        let a = |s: &str| Format::parse(s).unwrap();
        assert!(a("affine(i8/sat,scale=0.1,zero=-128)").exact_in_f64());
        assert!(Format::parse("affine(i32/sat,scale=0.1)").unwrap_err().contains("53 bits"));
        assert!(a("affine(i32/sat,scale=2^-5)").exact_in_f64());
        assert!(a("affine(i32/sat,scale=1.5)").exact_in_f64());
        assert_eq!(a("affine(i16/sat,scale=0.5)").bits(), 16);
        // n·0.1f32 takes up to 8 + 24 bits.
        assert!(!Format::Float(F32).contains(a("affine(i8/sat,scale=0.1,zero=3)")));
        assert!(Format::Float(F64).contains(a("affine(i8/sat,scale=0.1,zero=3)")));
        assert!(Format::Float(F32).contains(a("affine(i16/sat,scale=0.375)")));
        assert!(Format::Float(F16).contains(a("affine(i8/sat,scale=2^-5)")));
        assert!(!Format::Float(F16).contains(a("affine(i8/sat,scale=0.1)")));
        assert!(!Format::Float(BF16).contains(a("affine(i16/sat,scale=2^-5)")));
        assert!(!Format::Float(F16).contains(a("affine(i8/sat,scale=2^-30)")));
        assert!(!Format::Float(F16).contains(a("affine(i8/sat,scale=2^10)")));
        assert!(!a("affine(i8/sat,scale=0.5)").contains(Format::Float(F16)));
        assert!(!Format::Int(Int::parse("i32").unwrap()).contains(a("affine(i8/sat,scale=1)")));
    }

    #[test]
    fn conversions_are_paths() {
        // Rounded once to `bf16`, more than half a step up. Through binary16
        // first, exactly half a step: the tie the first rounding made goes
        // down to even, or up away from zero.
        let x = 1.0 + 2f64.powi(-8) + 2f64.powi(-12);
        assert_eq!(convert(x, &[BF16], Tie::Even), 1.0 + 2f64.powi(-7));
        assert_eq!(convert(x, &[F16], Tie::Even), 1.0 + 2f64.powi(-8));
        assert_eq!(convert(x, &[F16, BF16], Tie::Even), 1.0);
        assert_eq!(convert(x, &[F16, BF16], Tie::Away), 1.0 + 2f64.powi(-7));
    }
}
