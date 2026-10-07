//! The arithmetic of the interpreter: a trait, so that the evaluator does not
//! depend on one number representation, and its implementation with
//! `rug::Float` (MPFR, correctly rounded, precision chosen in bits).

use std::cmp::Ordering;
use std::fmt::Debug;

use rug::Float;
use rug::float::Round;
use rug::ops::Pow;

pub trait Real: Clone + Debug + PartialEq {
    fn from_u64(n: u64, prec: u32) -> Self;
    /// π, correctly rounded at `prec` bits.
    fn pi(prec: u32) -> Self;
    fn from_i64(n: i64, prec: u32) -> Self;
    /// Exact injection of a machine double (every double is a dyadic rational).
    fn from_f64(x: f64, prec: u32) -> Self;
    /// A decimal literal, rounded once at `prec` bits.
    fn from_decimal(s: &str, prec: u32) -> Option<Self>;
    fn prec(&self) -> u32;

    fn add(&self, o: &Self) -> Self;
    fn sub(&self, o: &Self) -> Self;
    fn mul(&self, o: &Self) -> Self;
    fn div(&self, o: &Self) -> Self;
    fn neg(&self) -> Self;
    fn sqrt(&self) -> Self;
    fn exp(&self) -> Self;
    fn ln(&self) -> Self;
    fn tanh(&self) -> Self;
    fn sin(&self) -> Self;
    fn cos(&self) -> Self;
    fn erf(&self) -> Self;
    fn abs(&self) -> Self;
    fn pow_u(&self, n: u64) -> Self;

    fn is_zero(&self) -> bool;
    fn is_negative(&self) -> bool;
    /// Whether `x + (+0)` is `x` itself: not for `−0`, whose sum with `+0` is
    /// `+0`, and not for a NaN, which no rule should be asked about.
    fn plus_zero_is_self(&self) -> bool;
    fn cmp(&self, o: &Self) -> Ordering;
    fn to_f64(&self) -> f64;
    /// Decimal rendering with `digits` significant digits.
    fn to_string_sig(&self, digits: usize) -> String;
}

/// `rug::Float`: every value carries its precision; binary operations take
/// the precision of the left operand (all values of one run share it).
impl Real for Float {
    fn pi(prec: u32) -> Self {
        Float::with_val(prec, rug::float::Constant::Pi)
    }

    fn from_u64(n: u64, prec: u32) -> Self {
        Float::with_val(prec, n)
    }
    fn from_i64(n: i64, prec: u32) -> Self {
        Float::with_val(prec, n)
    }
    fn from_f64(x: f64, prec: u32) -> Self {
        Float::with_val(prec, x)
    }
    fn from_decimal(s: &str, prec: u32) -> Option<Self> {
        Float::parse(s).ok().map(|p| Float::with_val(prec, p))
    }
    fn prec(&self) -> u32 {
        Float::prec(self)
    }
    fn add(&self, o: &Self) -> Self {
        Float::with_val(self.prec(), self + o)
    }
    fn sub(&self, o: &Self) -> Self {
        Float::with_val(self.prec(), self - o)
    }
    fn mul(&self, o: &Self) -> Self {
        Float::with_val(self.prec(), self * o)
    }
    fn div(&self, o: &Self) -> Self {
        Float::with_val(self.prec(), self / o)
    }
    fn neg(&self) -> Self {
        Float::with_val(self.prec(), -self)
    }
    fn sqrt(&self) -> Self {
        self.clone().sqrt()
    }
    fn exp(&self) -> Self {
        self.clone().exp()
    }
    fn ln(&self) -> Self {
        self.clone().ln()
    }
    fn tanh(&self) -> Self {
        self.clone().tanh()
    }
    fn sin(&self) -> Self {
        self.clone().sin()
    }
    fn cos(&self) -> Self {
        self.clone().cos()
    }
    fn erf(&self) -> Self {
        self.clone().erf()
    }
    fn abs(&self) -> Self {
        self.clone().abs()
    }
    fn pow_u(&self, n: u64) -> Self {
        Float::with_val(self.prec(), self.pow(n as u32))
    }
    fn is_zero(&self) -> bool {
        Float::is_zero(self)
    }
    fn is_negative(&self) -> bool {
        self.is_sign_negative() && !Float::is_zero(self)
    }
    fn plus_zero_is_self(&self) -> bool {
        !self.is_nan() && !(Float::is_zero(self) && self.is_sign_negative())
    }
    fn cmp(&self, o: &Self) -> Ordering {
        self.partial_cmp(o).unwrap_or(Ordering::Equal)
    }
    fn to_f64(&self) -> f64 {
        Float::to_f64(self)
    }
    fn to_string_sig(&self, digits: usize) -> String {
        if self.is_infinite() {
            return if self.is_sign_negative() { "-inf".into() } else { "inf".into() };
        }
        if self.is_nan() {
            return "nan".into();
        }
        if Float::is_zero(self) {
            return "0".into();
        }
        // `to_string_radix_round` gives digits and an exponent; render as plain
        // decimal when the exponent is small, else in scientific notation.
        let (sign, mantissa, exp) = self.to_sign_string_exp_round(10, Some(digits), Round::Nearest);
        let mantissa = mantissa.trim_end_matches('0');
        let mantissa = if mantissa.is_empty() { "0" } else { mantissa };
        let exp = exp.unwrap_or(0);
        let body = if (-4..=digits as i32 + 1).contains(&exp) {
            if exp <= 0 {
                format!("0.{}{}", "0".repeat((-exp) as usize), mantissa)
            } else if (exp as usize) >= mantissa.len() {
                format!("{}{}", mantissa, "0".repeat(exp as usize - mantissa.len()))
            } else {
                format!("{}.{}", &mantissa[..exp as usize], &mantissa[exp as usize..])
            }
        } else {
            let (head, tail) = mantissa.split_at(1);
            if tail.is_empty() { format!("{head}e{}", exp - 1) } else { format!("{head}.{tail}e{}", exp - 1) }
        };
        if sign { format!("-{body}") } else { body }
    }
}
