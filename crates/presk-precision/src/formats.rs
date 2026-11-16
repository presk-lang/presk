//! What a format costs a value, and which values it holds (plan 0.18.0 step
//! 5, D208).
//!
//! The error analysis and the solver both charge a value the rounding it
//! takes where it is stored or computed. Until 0.18.0 that was `u(τ)·|x|`,
//! the unit roundoff of a type `f64`, `f32` or `bf16` times the value's
//! magnitude, which never leaves their normal range. A format of the plan
//! may: binary16's normal numbers stop at 2⁻¹⁴, E4M3's largest is 480, an
//! affine format's values are a grid of one step. So the rounding is bounded
//! here from what the envelope saw of a value ([`Seen`]), per kind of format
//! ([`rounding`]):
//! - **a float**: `u·|x|` in its normal range, `u·2^emin` under it, half the
//!   subnormals' step (`2^(emin−1)` without subnormals), and past its largest
//!   finite number an infinity, or under saturation the excess clipped;
//! - **an affine format**: half its step `s/2`, the quotient's own rounding
//!   where a machine computes it in a float, and past the grid's ends the
//!   excess its clamp cuts;
//! - **an integer or a fixed-point number**: half its grid's step, the excess
//!   clamped, and a wrap an infinity.
//!
//! A sum's operations are bounded by [`sum_rounding`]: Higham's `c·u·Σ|terms|`
//! in a float accumulator, nothing in an integer one whose largest count the
//! terms' codes bound, and its products' rounding unless they are exact.
//!
//! **The range rules** ([`refusal`]) say which formats a solver does not offer
//! a value at all, whatever its budget: one whose largest value, times a
//! margin, passes the format's largest finite number; one a divisor reads
//! whose smallest value is under the format's smallest normal number (an
//! affine format's step); one whose nest computes in a format a constant of
//! it rounds to zero in, or past its largest number (AdamW's ε = 10⁻⁸ in
//! binary16, presk-fp16 §5). An affine format's scale is computed from the
//! envelope, so it spans the values seen: the margin is not asked of it, and
//! what it clips out of sample is its term's.

use presk_format::{Affine, Float, Format, IntOverflow, Overflow};

/// What the envelope saw of a value: the largest `|x|`, the smallest
/// non-zero `|x|` (0 where it saw none, or did not record it), and the
/// smallest and largest value when `presk calibrate --range` recorded them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Seen {
    pub mag: f64,
    pub floor: f64,
    pub range: Option<(f64, f64)>,
}

impl Seen {
    /// A value of magnitude `mag`, nothing else known: its floor may be
    /// anywhere under it.
    pub fn of(mag: f64) -> Seen {
        Seen { mag, floor: 0.0, range: None }
    }

    /// Whether some value may lie under `x` in magnitude.
    fn under(self, x: f64) -> bool {
        self.floor == 0.0 || self.floor < x
    }
}

fn exp2(k: i32) -> f64 {
    2f64.powi(k)
}

/// The smallest normal number of a float format, `2^emin`.
pub fn normal(f: Float) -> f64 {
    exp2(f.emin)
}

/// A bound on `|fl(x) − x|` over the values seen, `fl` one rounding into
/// `f` to nearest (either tie): infinite where the format's overflow rule
/// gives an infinity or a NaN for a value seen.
pub fn rounding(f: Format, seen: Seen) -> f64 {
    if seen.mag == 0.0 {
        return 0.0;
    }
    if !seen.mag.is_finite() {
        return f64::INFINITY;
    }
    match f {
        Format::Float(f) => float_rounding(f, seen),
        Format::Affine(a) => affine_rounding(a, seen),
        Format::Int(i) => grid_rounding(1.0, i.min() as f64, i.max() as f64, i.overflow, seen),
        Format::Fixed(x) => grid_rounding(exp2(-x.frac), x.int.min() as f64 * exp2(-x.frac), x.int.max() as f64 * exp2(-x.frac), x.int.overflow, seen),
    }
}

fn float_rounding(f: Float, seen: Seen) -> f64 {
    let (u, max, normal) = (f.unit_roundoff(), f.max_finite(), normal(f));
    let inside = |m: f64| match (seen.under(normal), f.subnormals) {
        (false, _) => u * m,
        (true, true) => u * m.max(normal),
        (true, false) => (u * m).max(normal / 2.0),
    };
    if seen.mag <= max {
        return inside(seen.mag);
    }
    match f.overflow {
        Overflow::Saturate => inside(max).max(seen.mag - max),
        Overflow::Infinity | Overflow::Nan => f64::INFINITY,
    }
}

fn affine_rounding(a: Affine, seen: Seen) -> f64 {
    let (s, z) = (a.scale(), a.zero as f64);
    let (lo, hi) = seen.range.unwrap_or((-seen.mag, seen.mag));
    let (vlo, vhi) = ((a.int.min() as f64 - z) * s, (a.int.max() as f64 - z) * s);
    // A quotient computed in a float: `x` rounded to it, divided, and the
    // zero point added there, each within `u_q` of its operands; three of
    // them bound the integer's move, in units of the step.
    let quotient = a.quotient.map_or(0.0, |q| 3.0 * q.unit_roundoff() * (seen.mag + z.abs() * s));
    (s / 2.0 + quotient).max(hi - vhi).max(vlo - lo)
}

fn grid_rounding(step: f64, lo: f64, hi: f64, overflow: IntOverflow, seen: Seen) -> f64 {
    let (vlo, vhi) = seen.range.unwrap_or((-seen.mag, seen.mag));
    let excess = (vhi - hi).max(lo - vlo);
    if excess > step / 2.0 && overflow == IntOverflow::Wrap {
        return f64::INFINITY;
    }
    (step / 2.0).max(excess)
}

/// Why a format is not offered for a value (the range rules).
#[derive(Clone, Debug, PartialEq)]
pub enum Refusal {
    /// The values seen, times the margin, pass the format's largest finite
    /// number: it would saturate or overflow out of sample.
    Range { mag: f64, max: f64 },
    /// A divisor reads the value, and its smallest one is under the format's
    /// smallest normal number, or an affine format's step: rounded there, a
    /// divisor loses its digits or becomes zero.
    Divisor { floor: f64, normal: f64 },
    /// A constant its nest computes in the format rounds to zero there, or
    /// past its largest finite number.
    Constant { value: f64 },
}

impl Refusal {
    /// The rule's name, as the solver's report and `presk explain` print it.
    pub fn name(&self) -> &'static str {
        match self {
            Refusal::Range { .. } => "range",
            Refusal::Divisor { .. } => "divisor",
            Refusal::Constant { .. } => "constant",
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::Range { mag, max } => write!(f, "its largest value {mag:.3e} with the margin passes the format's largest {max:.3e}"),
            Refusal::Divisor { floor, normal } => write!(f, "a divisor reads it, and its smallest value {floor:.3e} is under the format's smallest normal number {normal:.3e}"),
            Refusal::Constant { value } => write!(f, "its nest computes the constant {value:e}, which the format rounds to zero or past its largest number"),
        }
    }
}

/// Whether the format holds `c` well enough to compute with it: not to
/// zero, nor past its largest finite number.
pub fn holds_constant(f: Format, c: f64) -> bool {
    if c == 0.0 || !c.is_finite() {
        return true;
    }
    let (y, overflowed) = f.round_noted(c, presk_format::Tie::Even);
    !overflowed && y != 0.0 && y.is_finite()
}

/// The range rules of a stored value in `f`: refused when its values seen,
/// times `margin`, pass a float's largest finite number, or when a divisor
/// reads it and its smallest value is under the float's smallest normal
/// number or an affine format's step. `None` when the format is offered.
pub fn refusal(f: Format, seen: Seen, margin: f64, divisor: bool) -> Option<Refusal> {
    match f {
        Format::Float(x) => {
            if seen.mag * margin > x.max_finite() {
                return Some(Refusal::Range { mag: seen.mag, max: x.max_finite() });
            }
            if divisor && seen.floor > 0.0 && seen.floor < normal(x) {
                return Some(Refusal::Divisor { floor: seen.floor, normal: normal(x) });
            }
            None
        }
        Format::Affine(a) => (divisor && seen.floor > 0.0 && seen.floor < a.scale()).then(|| Refusal::Divisor { floor: seen.floor, normal: a.scale() }),
        // No value is stored in an integer or a fixed-point format.
        Format::Int(_) | Format::Fixed(_) => None,
    }
}

/// A sum of `k` terms as an analysis or a solver knows it: what its terms
/// add up to without cancelling, its largest product, how many roundings
/// its order puts a term through, the products' roundings per term, and,
/// for a sum counted in its terms' unit, the largest count its codes allow.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sum {
    pub k: f64,
    /// `Σ|terms|`, a maximum over the sum's outputs.
    pub abs: f64,
    /// The largest `|product|` of one term, `Π max|operand|`.
    pub largest: f64,
    /// `c(order, k)`: the additions a term goes through.
    pub additions: f64,
    /// The roundings of a term's product, its operands less one (0 for a
    /// plain sum), or `None` where they are exact in the product format.
    pub products: Option<f64>,
    /// The largest `Σ|q|` the terms' codes allow, in the accumulator's
    /// grid, for a sum counted in its terms' unit; `None` where an operand
    /// is no scaled value, which no integer count reads.
    pub count: Option<f64>,
}

/// What a sum's own operations add, its products rounded to `product` and
/// its additions to `accum`, at a margin on their ranges. Infinite where a
/// float's partial sums or products, times the margin, pass its largest
/// finite number, or an integer count may leave its accumulator's range.
pub fn sum_rounding(product: Format, accum: Format, s: &Sum, margin: f64) -> f64 {
    let products = match (product, s.products) {
        (_, None) => 0.0,
        (_, Some(0.0)) => 0.0,
        (Format::Float(f), Some(n)) => {
            if s.largest * margin > f.max_finite() {
                return f64::INFINITY;
            }
            n * f.unit_roundoff() * (s.abs + s.k * normal(f))
        }
        // Integer codes multiply exactly; a product past the format's
        // range is the accumulator's count's to refuse.
        (Format::Int(_) | Format::Fixed(_), Some(_)) if s.count.is_some() => 0.0,
        _ => return f64::INFINITY,
    };
    let additions = match accum {
        Format::Float(f) => {
            if s.abs * margin > f.max_finite() {
                return f64::INFINITY;
            }
            s.additions * f.unit_roundoff() * (s.abs + normal(f))
        }
        Format::Int(i) => match s.count {
            Some(c) if c <= i.max() as f64 => 0.0,
            _ => return f64::INFINITY,
        },
        Format::Fixed(x) => match s.count {
            Some(c) if c * exp2(x.frac) <= x.int.max() as f64 => 0.0,
            _ => return f64::INFINITY,
        },
        Format::Affine(_) => return f64::INFINITY,
    };
    products + additions
}

/// Whether every product of a value of `a` by a value of `b` is a number of
/// `f`, but for an overflow: their significands' bits and their smallest
/// product's last bit fit it. Two E4M3 values multiply exactly in binary16
/// (D203).
pub fn exact_product(a: Float, b: Float, f: Float) -> bool {
    let last = |x: Float| x.emin - x.p as i32 + 1;
    a.p + b.p <= f.p && last(a) + last(b) >= last(f)
}

/// The largest `|q − z|` of an affine format's codes.
pub fn largest_code(a: Affine) -> f64 {
    let z = i128::from(a.zero);
    (a.int.max() - z).max(z - a.int.min()) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use presk_format::{Int, Tie};

    /// A tiny generator, enough for draws over the formats' ranges.
    struct Draw(u64);
    impl Draw {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        /// A value of magnitude up to `mag`, spread over its binades down
        /// to `2^-lo` of it, of either sign.
        fn value(&mut self, mag: f64, lo: i32) -> f64 {
            let e = -((self.next() % (lo as u64 + 1)) as i32);
            let m = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
            let x = mag * m * exp2(e);
            if self.next() & 1 == 0 { x } else { -x }
        }
    }

    /// Every value drawn is rounded within the bound, in every float of the
    /// catalogue, through its subnormals and past its largest number.
    #[test]
    fn a_float_rounds_within_its_term() {
        let mut d = Draw(0x9e37_79b9_7f4a_7c15);
        for f in [presk_format::F32, presk_format::BF16, presk_format::F16, presk_format::E5M2, presk_format::E4M3FN, presk_format::E4M3_SATURATING, presk_format::E2M1] {
            for mag in [1e-9, 1e-6, 3e-3, 0.7, 5.0, 300.0, 470.0, 6e4, 1e6] {
                let seen = Seen::of(mag);
                let bound = rounding(Format::Float(f), seen);
                for _ in 0..4000 {
                    let x = d.value(mag, 40);
                    for tie in [Tie::Even, Tie::Away] {
                        let y = f.round(x, tie);
                        let e = (y - x).abs();
                        assert!(e <= bound || (bound.is_infinite()), "{f}: {x} → {y}, error {e} over the bound {bound} at magnitude {mag}");
                    }
                }
                // Past the largest number: an infinity's format bounds
                // nothing, a saturating one the excess.
                if mag > f.max_finite() {
                    match f.overflow {
                        Overflow::Saturate => assert!(bound >= mag - f.max_finite() && bound.is_finite()),
                        _ => assert!(bound.is_infinite(), "{f} at {mag}"),
                    }
                }
            }
        }
        // A floor at or above the smallest normal number keeps `u·|x|`.
        let f = presk_format::F16;
        assert_eq!(rounding(Format::Float(f), Seen { mag: 2.0, floor: 1e-3, range: None }), 2.0 * f.unit_roundoff());
        assert_eq!(rounding(Format::Float(f), Seen { mag: 1e-6, floor: 1e-7, range: None }), f.unit_roundoff() * exp2(-14));
        assert_eq!(rounding(Format::Float(f), Seen::of(0.0)), 0.0);
    }

    /// An affine format: half its step, its quotient's rounding in a float,
    /// the excess its clamp cuts; every reading of the rescaling within it.
    #[test]
    fn an_affine_format_rounds_within_its_term() {
        let mut d = Draw(0x2545_f491_4f6c_dd1d);
        let i8s = Int { bits: 8, signed: true, overflow: IntOverflow::Clamp };
        let u8s = Int { bits: 8, signed: false, overflow: IntOverflow::Clamp };
        for (int, lo, hi) in [(i8s, -1.7, 1.7), (u8s, 0.0, 2.4), (u8s, -0.3, 5.0), (i8s, -0.01, 0.02)] {
            let base = if int.signed { Affine::absmax(int, f64::max(-lo, hi)).unwrap() } else { Affine::minmax(int, lo, hi).unwrap() };
            for a in [base, Affine { tie: Tie::Away, ..base }, Affine { zero_rounded: true, quotient: Some(presk_format::F32), ..base }, Affine { quotient: Some(presk_format::F32), ..base }] {
                // The range seen, then twice it: the clamp cuts the excess.
                for widen in [1.0, 2.0] {
                    let seen = Seen { mag: f64::max(-lo, hi) * widen, floor: 0.0, range: Some((lo * widen, hi * widen)) };
                    let bound = rounding(Format::Affine(a), seen);
                    for _ in 0..4000 {
                        let x = lo * widen + (hi - lo) * widen * ((d.next() >> 11) as f64 / (1u64 << 53) as f64);
                        let y = a.value(a.round_noted(x).0);
                        assert!((y - x).abs() <= bound, "{a}: {x} → {y} over {bound}");
                    }
                    if widen == 1.0 {
                        assert!(bound < a.scale() * 0.51, "{a}: {bound} for the step {}", a.scale());
                    }
                }
            }
        }
    }

    #[test]
    fn the_range_rules() {
        let (f16, e4m3) = (Format::Float(presk_format::F16), Format::Float(presk_format::E4M3_SATURATING));
        assert!(refusal(f16, Seen::of(3e4), 2.0, false).is_none());
        assert!(matches!(refusal(f16, Seen::of(4e4), 2.0, false), Some(Refusal::Range { .. })));
        assert!(matches!(refusal(e4m3, Seen::of(300.0), 2.0, false), Some(Refusal::Range { .. })));
        assert!(refusal(e4m3, Seen::of(200.0), 2.0, false).is_none());
        // A divisor under the smallest normal number, and one above it.
        let small = Seen { mag: 1.0, floor: 1e-5, range: None };
        assert!(matches!(refusal(f16, small, 2.0, true), Some(Refusal::Divisor { .. })));
        assert!(refusal(f16, small, 2.0, false).is_none());
        assert!(refusal(f16, Seen { floor: 1e-4, ..small }, 2.0, true).is_none());
        // AdamW's ε in binary16, LayerNorm's: the one rounds to zero.
        assert!(!holds_constant(f16, 1e-8));
        assert!(holds_constant(f16, 1e-5));
        assert!(!holds_constant(e4m3, 5e-4), "under half its least subnormal, 2^-10");
        assert!(holds_constant(e4m3, 1e-3));
        assert!(!holds_constant(f16, 1e5));
        assert!(holds_constant(Format::Float(presk_format::F32), 1e-8));
    }

    #[test]
    fn a_sum_s_operations() {
        let f16 = Format::Float(presk_format::F16);
        let s = Sum { k: 64.0, abs: 10.0, largest: 2.0, additions: 63.0, products: Some(1.0), count: None };
        let u = presk_format::F16.unit_roundoff();
        let want = u * (10.0 + 64.0 * exp2(-14)) + 63.0 * u * (10.0 + exp2(-14));
        assert_eq!(sum_rounding(f16, f16, &s, 2.0), want);
        assert_eq!(sum_rounding(f16, f16, &Sum { products: None, ..s }, 2.0), 63.0 * u * (10.0 + exp2(-14)));
        assert!(sum_rounding(f16, f16, &Sum { abs: 4e4, ..s }, 2.0).is_infinite());
        // Counted in an integer: exact under its precondition.
        let i32w = Format::Int(Int { bits: 32, signed: true, overflow: IntOverflow::Wrap });
        let i16w = Format::Int(Int { bits: 16, signed: true, overflow: IntOverflow::Wrap });
        let counted = Sum { count: Some(64.0 * 127.0 * 127.0), ..s };
        assert_eq!(sum_rounding(i32w, i32w, &counted, 2.0), 0.0);
        assert!(sum_rounding(i32w, i16w, &counted, 2.0).is_infinite());
        assert!(sum_rounding(i32w, i32w, &s, 2.0).is_infinite(), "an operand no scaled value");
        assert!(exact_product(presk_format::E4M3_SATURATING, presk_format::E4M3_SATURATING, presk_format::F16));
        assert!(exact_product(presk_format::F16, presk_format::F16, presk_format::F32));
        assert!(!exact_product(presk_format::F16, presk_format::F16, presk_format::F16));
        assert!(!exact_product(presk_format::BF16, presk_format::BF16, presk_format::F16), "bf16's exponents");
    }
}
