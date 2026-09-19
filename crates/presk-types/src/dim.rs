//! Dimension expressions: linear forms over ℚ in symbolic dimensions.
//!
//! Every dimension that is not a literal is a [`DimVar`]: a `dim`
//! declaration, a module or function type parameter, or an implicit
//! dimension. Constants (`const`) are evaluated at elaboration and never
//! appear as variables. Rational coefficients arise from division by a
//! literal (`D / H` with `H` a known constant), which is legal only under a
//! divisibility obligation checked by the solver.

use std::collections::BTreeMap;
use std::fmt;

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

/// A symbolic dimension. Names and origins live in the elaborated program.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DimVar(pub u32);

impl fmt::Debug for DimVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "d{}", self.0)
    }
}

pub type Rational = BigRational;

pub fn rat(n: i64) -> Rational {
    Rational::from_integer(BigInt::from(n))
}

/// `Σ cᵢ·vᵢ + c₀`, coefficients in ℚ, variables ordered; zero coefficients
/// are never stored.
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct DimExpr {
    pub coeffs: BTreeMap<DimVar, Rational>,
    pub constant: Rational,
}

impl DimExpr {
    pub fn constant(c: impl Into<Rational>) -> Self {
        DimExpr { coeffs: BTreeMap::new(), constant: c.into() }
    }

    pub fn nat(n: u64) -> Self {
        Self::constant(Rational::from_integer(BigInt::from(n)))
    }

    pub fn var(v: DimVar) -> Self {
        let mut coeffs = BTreeMap::new();
        coeffs.insert(v, Rational::one());
        DimExpr { coeffs, constant: Rational::zero() }
    }

    pub fn is_const(&self) -> bool {
        self.coeffs.is_empty()
    }

    pub fn as_const(&self) -> Option<&Rational> {
        if self.is_const() { Some(&self.constant) } else { None }
    }

    /// The value as a natural number, when the expression is a constant
    /// non-negative integer.
    pub fn as_nat(&self) -> Option<u64> {
        let c = self.as_const()?;
        if c.is_integer() && !c.is_negative() { c.to_integer().to_u64() } else { None }
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty() && self.constant.is_zero()
    }

    pub fn vars(&self) -> impl Iterator<Item = DimVar> + '_ {
        self.coeffs.keys().copied()
    }

    pub fn coeff(&self, v: DimVar) -> Rational {
        self.coeffs.get(&v).cloned().unwrap_or_else(Rational::zero)
    }

    fn normalize(mut self) -> Self {
        self.coeffs.retain(|_, c| !c.is_zero());
        self
    }

    pub fn add(&self, other: &DimExpr) -> DimExpr {
        let mut out = self.clone();
        for (v, c) in &other.coeffs {
            *out.coeffs.entry(*v).or_insert_with(Rational::zero) += c;
        }
        out.constant += &other.constant;
        out.normalize()
    }

    pub fn neg(&self) -> DimExpr {
        DimExpr { coeffs: self.coeffs.iter().map(|(v, c)| (*v, -c)).collect(), constant: -&self.constant }
    }

    pub fn sub(&self, other: &DimExpr) -> DimExpr {
        self.add(&other.neg())
    }

    pub fn scale(&self, k: &Rational) -> DimExpr {
        if k.is_zero() {
            return DimExpr::default();
        }
        DimExpr { coeffs: self.coeffs.iter().map(|(v, c)| (*v, c * k)).collect(), constant: &self.constant * k }
    }

    /// Product, defined only when at least one side is constant.
    pub fn mul(&self, other: &DimExpr) -> Option<DimExpr> {
        if let Some(k) = other.as_const() {
            Some(self.scale(k))
        } else {
            self.as_const().map(|k| other.scale(k))
        }
    }

    /// Division by a non-zero constant (the divisibility obligation is the
    /// caller's business).
    pub fn div(&self, other: &DimExpr) -> Option<DimExpr> {
        let k = other.as_const()?;
        if k.is_zero() { None } else { Some(self.scale(&k.recip())) }
    }

    /// Replace variables by expressions.
    pub fn subst(&self, map: &BTreeMap<DimVar, DimExpr>) -> DimExpr {
        let mut out = DimExpr::constant(self.constant.clone());
        for (v, c) in &self.coeffs {
            match map.get(v) {
                Some(e) => out = out.add(&e.scale(c)),
                None => out = out.add(&DimExpr::var(*v).scale(c)),
            }
        }
        out.normalize()
    }

    /// Evaluate under a total assignment of the variables.
    pub fn eval(&self, assignment: &BTreeMap<DimVar, Rational>) -> Option<Rational> {
        let mut acc = self.constant.clone();
        for (v, c) in &self.coeffs {
            acc += c * assignment.get(v)?;
        }
        Some(acc)
    }

    /// Multiply by the least positive integer making every coefficient an
    /// integer; returns the multiplier and the integral form.
    pub fn integral(&self) -> (BigInt, DimExpr) {
        let mut m = BigInt::one();
        for c in self.coeffs.values().chain(std::iter::once(&self.constant)) {
            m = num_integer::lcm(m, c.denom().clone());
        }
        let k = Rational::from_integer(m.clone());
        (m, self.scale(&k))
    }

    /// Render in source terms using `name` for the variables, e.g. `4*D + 2`.
    pub fn display(&self, name: &dyn Fn(DimVar) -> String) -> String {
        let mut parts: Vec<String> = Vec::new();
        for (v, c) in &self.coeffs {
            let n = name(*v);
            let term = if c.is_one() {
                n
            } else if (-c).is_one() {
                format!("-{n}")
            } else if c.is_integer() {
                format!("{}*{n}", c.numer())
            } else if c.numer().is_one() {
                format!("{n}/{}", c.denom())
            } else {
                format!("{}*{n}/{}", c.numer(), c.denom())
            };
            parts.push(term);
        }
        if !self.constant.is_zero() || parts.is_empty() {
            parts.push(if self.constant.is_integer() { self.constant.numer().to_string() } else { self.constant.to_string() });
        }
        let mut s = String::new();
        for (i, p) in parts.iter().enumerate() {
            if i == 0 {
                s.push_str(p);
            } else if let Some(rest) = p.strip_prefix('-') {
                s.push_str(" - ");
                s.push_str(rest);
            } else {
                s.push_str(" + ");
                s.push_str(p);
            }
        }
        s
    }
}

impl fmt::Debug for DimExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display(&|v| format!("{v:?}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(i: u32) -> DimExpr {
        DimExpr::var(DimVar(i))
    }

    #[test]
    fn normal_forms() {
        // 4*D/2 ≡ 2*D
        let d = v(0);
        let four_d = d.scale(&rat(4));
        let half = four_d.div(&DimExpr::nat(2)).unwrap();
        assert_eq!(half, d.scale(&rat(2)));
        // D - D ≡ 0
        assert!(d.sub(&d).is_zero());
        // (D/12)*12 ≡ D
        let hd = d.div(&DimExpr::nat(12)).unwrap();
        assert_eq!(hd.mul(&DimExpr::nat(12)).unwrap(), d);
        assert!(d.mul(&v(1)).is_none());
    }

    #[test]
    fn display_in_source_terms() {
        let name = |v: DimVar| ["D", "H"][v.0 as usize].to_string();
        let e = v(0).scale(&rat(4)).add(&DimExpr::nat(2)).sub(&v(1));
        assert_eq!(e.display(&name), "4*D - H + 2");
        assert_eq!(v(0).div(&DimExpr::nat(12)).unwrap().display(&name), "D/12");
        assert_eq!(DimExpr::nat(0).display(&name), "0");
    }

    #[test]
    fn subst_and_eval() {
        let mut map = BTreeMap::new();
        map.insert(DimVar(0), DimExpr::nat(768));
        let e = v(0).scale(&rat(4)).add(&v(1));
        assert_eq!(e.subst(&map), v(1).add(&DimExpr::nat(3072)));
        let mut a = BTreeMap::new();
        a.insert(DimVar(0), rat(3));
        a.insert(DimVar(1), rat(5));
        assert_eq!(e.eval(&a), Some(rat(17)));
    }
}
