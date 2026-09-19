//! Decision procedure for the shape fragment: linear arithmetic over ℕ with
//! divisibility by literals.
//!
//! Variables range over ℕ (0 included). Facts come from `dim T <= N`
//! declarations, `static assert`s already discharged, and the implicit
//! `v ≥ 0` of every variable. The procedure is deliberately small:
//!
//! 1. equality facts are used as rewriting rules (Gaussian elimination),
//!    divisibility facts `k | v` as substitutions `v := k·v'`;
//! 2. the goal is checked syntactically on the rewritten form (an equality
//!    holds iff its difference is identically zero; `a ≤ b` holds if
//!    `b − a` has non-negative coefficients and constant; `k | e` holds if
//!    every coefficient and the constant are integers divisible by `k`);
//! 3. otherwise a bounded search for a counterexample in ℕ decides
//!    `Invalid`, and failing that the answer is `Unknown`.
//!
//! Soundness is what matters for the type checker (a `Valid` answer is never
//! wrong, see the property tests); completeness is only needed on the shapes
//! that occur in practice, and `Unknown` is reported to the user as
//! "cannot prove".

use std::collections::BTreeMap;

use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive, Zero};

use crate::dim::{DimExpr, DimVar, Rational, rat};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Constraint {
    /// `lhs = rhs`
    Eq(DimExpr, DimExpr),
    /// `lhs ≤ rhs`
    Le(DimExpr, DimExpr),
    /// `k | e`
    Divides(u64, DimExpr),
}

impl Constraint {
    pub fn holds(&self, a: &BTreeMap<DimVar, Rational>) -> Option<bool> {
        Some(match self {
            Constraint::Eq(l, r) => l.eval(a)? == r.eval(a)?,
            Constraint::Le(l, r) => l.eval(a)? <= r.eval(a)?,
            Constraint::Divides(k, e) => {
                let v = e.eval(a)?;
                v.is_integer() && (v.to_integer() % BigInt::from(*k)).is_zero()
            }
        })
    }

    pub fn vars(&self) -> Vec<DimVar> {
        let mut vs: Vec<DimVar> = match self {
            Constraint::Eq(l, r) | Constraint::Le(l, r) => l.vars().chain(r.vars()).collect(),
            Constraint::Divides(_, e) => e.vars().collect(),
        };
        vs.sort();
        vs.dedup();
        vs
    }

    pub fn subst(&self, map: &BTreeMap<DimVar, DimExpr>) -> Constraint {
        match self {
            Constraint::Eq(l, r) => Constraint::Eq(l.subst(map), r.subst(map)),
            Constraint::Le(l, r) => Constraint::Le(l.subst(map), r.subst(map)),
            Constraint::Divides(k, e) => Constraint::Divides(*k, e.subst(map)),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Facts {
    pub constraints: Vec<Constraint>,
}

impl Facts {
    pub fn push(&mut self, c: Constraint) {
        self.constraints.push(c);
    }

    pub fn holds(&self, a: &BTreeMap<DimVar, Rational>) -> bool {
        self.constraints.iter().all(|c| c.holds(a) == Some(true))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Validity {
    Valid,
    /// A counterexample: an assignment satisfying the facts and violating the goal.
    Invalid(BTreeMap<DimVar, Rational>),
    Unknown,
}

/// Does `facts` entail `goal` for all assignments in ℕ?
pub fn entails(facts: &Facts, goal: &Constraint) -> Validity {
    let (map, fresh) = rewriting(facts);
    let goal_r = goal.subst(&map);
    let facts_r: Vec<Constraint> = facts.constraints.iter().map(|c| c.subst(&map)).collect();
    if syntactic(&goal_r, &facts_r) {
        return Validity::Valid;
    }
    match counterexample(facts, goal, &fresh) {
        Some(a) => Validity::Invalid(a),
        None => Validity::Unknown,
    }
}

/// Build the substitution from the equality and divisibility facts. Returns
/// the map and the fresh variables introduced for divisibility.
fn rewriting(facts: &Facts) -> (BTreeMap<DimVar, DimExpr>, Vec<DimVar>) {
    let mut map: BTreeMap<DimVar, DimExpr> = BTreeMap::new();
    let mut fresh = Vec::new();
    let mut next = facts
        .constraints
        .iter()
        .flat_map(|c| c.vars())
        .map(|v| v.0 + 1)
        .max()
        .unwrap_or(0)
        .max(1_000_000);
    for c in &facts.constraints {
        match c.subst(&map) {
            Constraint::Eq(l, r) => {
                let d = l.sub(&r);
                // Solve for the variable with the largest index: v = (rest)/(-c).
                if let Some((&v, c)) = d.coeffs.iter().next_back() {
                    let mut rest = d.clone();
                    rest.coeffs.remove(&v);
                    let sol = rest.scale(&(-c.recip()));
                    // Apply to earlier substitutions too.
                    let mut single = BTreeMap::new();
                    single.insert(v, sol.clone());
                    for e in map.values_mut() {
                        *e = e.subst(&single);
                    }
                    map.insert(v, sol);
                }
            }
            Constraint::Divides(k, e) => {
                // Only the simple form `k | v` is used as a rewrite v := k·v'.
                if e.constant.is_zero() && e.coeffs.len() == 1 {
                    let (&v, c) = e.coeffs.iter().next().unwrap();
                    if c == &rat(1) {
                        let nv = DimVar(next);
                        next += 1;
                        fresh.push(nv);
                        let sol = DimExpr::var(nv).scale(&rat(k as i64));
                        let mut single = BTreeMap::new();
                        single.insert(v, sol.clone());
                        for e in map.values_mut() {
                            *e = e.subst(&single);
                        }
                        map.insert(v, sol);
                    }
                }
            }
            Constraint::Le(..) => {}
        }
    }
    (map, fresh)
}

/// The goal holds on the rewritten forms by inspection.
fn syntactic(goal: &Constraint, facts: &[Constraint]) -> bool {
    match goal {
        Constraint::Eq(l, r) => l.sub(r).is_zero(),
        Constraint::Le(l, r) => {
            let d = r.sub(l);
            if nonneg(&d) {
                return true;
            }
            // One upper-bound fact `v ≤ b` (b constant) may be needed: substitute the bound.
            for f in facts {
                if let Constraint::Le(fl, fr) = f
                    && fr.is_const()
                    && fl.coeffs.len() == 1
                    && fl.constant.is_zero()
                {
                    let (&v, c) = fl.coeffs.iter().next().unwrap();
                    if c == &rat(1) && d.coeff(v).is_negative() {
                        let mut m = BTreeMap::new();
                        m.insert(v, fr.clone());
                        if nonneg(&d.subst(&m)) {
                            return true;
                        }
                    }
                }
            }
            false
        }
        Constraint::Divides(k, e) => {
            let k = BigInt::from(*k);
            e.coeffs.values().chain(std::iter::once(&e.constant)).all(|c| c.is_integer() && (c.to_integer() % &k).is_zero())
        }
    }
}

fn nonneg(e: &DimExpr) -> bool {
    !e.constant.is_negative() && e.coeffs.values().all(|c| !c.is_negative())
}

/// Bounded search over small naturals for an assignment satisfying the facts
/// and violating the goal. Variables of the goal and facts are enumerated in
/// `0..=B` where `B` shrinks with the number of variables.
fn counterexample(facts: &Facts, goal: &Constraint, _fresh: &[DimVar]) -> Option<BTreeMap<DimVar, Rational>> {
    // Only the goal's variables and those connected to them through facts
    // are enumerated; facts on unrelated variables cannot constrain them.
    let mut vars: Vec<DimVar> = goal.vars();
    loop {
        let before = vars.len();
        for c in &facts.constraints {
            let cv = c.vars();
            if cv.iter().any(|v| vars.contains(v)) {
                vars.extend(cv);
            }
        }
        vars.sort();
        vars.dedup();
        if vars.len() == before {
            break;
        }
    }
    let all_facts = facts;
    let relevant: Vec<&Constraint> = facts.constraints.iter().filter(|c| c.vars().iter().all(|v| vars.contains(v))).collect();
    let facts = Facts { constraints: relevant.into_iter().cloned().collect() };
    let facts = &facts;
    if vars.is_empty() {
        let a = BTreeMap::new();
        return if goal.holds(&a) == Some(false) && facts.holds(&a) { complete(all_facts, a) } else { None };
    }
    let bound: u64 = match vars.len() {
        1 => 4096,
        2 => 256,
        3 => 48,
        4 => 16,
        _ => 6,
    };
    let mut a: BTreeMap<DimVar, Rational> = vars.iter().map(|v| (*v, rat(0))).collect();
    let mut idx = vec![0u64; vars.len()];
    loop {
        for (i, v) in vars.iter().enumerate() {
            a.insert(*v, rat(idx[i] as i64));
        }
        if facts.holds(&a) && goal.holds(&a) == Some(false) {
            return complete(all_facts, a);
        }
        // increment the odometer
        let mut i = 0;
        loop {
            if i == idx.len() {
                return None;
            }
            idx[i] += 1;
            if idx[i] > bound {
                idx[i] = 0;
                i += 1;
            } else {
                break;
            }
        }
    }
}

/// Extend a witness on the goal's variables to the variables of the facts it
/// does not mention (they are independent of the goal by construction), so
/// that the witness satisfies *every* fact. `None` when no small extension
/// works: the caller then answers `Unknown` rather than `Invalid`.
fn complete(facts: &Facts, mut a: BTreeMap<DimVar, Rational>) -> Option<BTreeMap<DimVar, Rational>> {
    let mut rest: Vec<DimVar> = facts.constraints.iter().flat_map(|c| c.vars()).filter(|v| !a.contains_key(v)).collect();
    rest.sort();
    rest.dedup();
    if rest.is_empty() {
        return if facts.holds(&a) { Some(a) } else { None };
    }
    let bound: u64 = match rest.len() {
        1 => 4096,
        2 => 256,
        3 => 48,
        _ => 8,
    };
    let mut idx = vec![0u64; rest.len()];
    loop {
        for (i, v) in rest.iter().enumerate() {
            a.insert(*v, rat(idx[i] as i64));
        }
        if facts.holds(&a) {
            return Some(a);
        }
        let mut i = 0;
        loop {
            if i == idx.len() {
                return None;
            }
            idx[i] += 1;
            if idx[i] > bound {
                idx[i] = 0;
                i += 1;
            } else {
                break;
            }
        }
    }
}

/// Convenience for error messages: render an assignment as `D = 768, H = 7`,
/// restricted to the variables of `goal` (the witness may also assign
/// variables of unrelated facts).
pub fn display_assignment(a: &BTreeMap<DimVar, Rational>, goal: &Constraint, name: &dyn Fn(DimVar) -> String) -> String {
    let relevant = goal.vars();
    a.iter()
        .filter(|(v, _)| v.0 < 1_000_000 && relevant.contains(v))
        .map(|(v, x)| format!("{} = {}", name(*v), x.to_integer().to_i64().map(|n| n.to_string()).unwrap_or_else(|| x.to_string())))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(i: u32) -> DimExpr {
        DimExpr::var(DimVar(i))
    }

    #[test]
    fn equalities() {
        let f = Facts::default();
        let d = v(0);
        assert_eq!(entails(&f, &Constraint::Eq(d.scale(&rat(4)).div(&DimExpr::nat(2)).unwrap(), d.scale(&rat(2)))), Validity::Valid);
        assert!(matches!(entails(&f, &Constraint::Eq(d.clone(), v(1))), Validity::Invalid(_)));
        let mut f = Facts::default();
        f.push(Constraint::Eq(v(1), d.scale(&rat(4))));
        assert_eq!(entails(&f, &Constraint::Eq(v(1).add(&d), d.scale(&rat(5)))), Validity::Valid);
    }

    #[test]
    fn bounds() {
        let mut f = Facts::default();
        f.push(Constraint::Le(v(0), DimExpr::nat(1024)));
        assert_eq!(entails(&f, &Constraint::Le(v(0), DimExpr::nat(4096))), Validity::Valid);
        assert_eq!(entails(&f, &Constraint::Le(DimExpr::nat(0), v(0).add(&v(1)))), Validity::Valid);
        assert!(matches!(entails(&f, &Constraint::Le(v(0), DimExpr::nat(100))), Validity::Invalid(_)));
        // 1 ≤ D is not valid: D may be 0.
        assert!(matches!(entails(&Facts::default(), &Constraint::Le(DimExpr::nat(1), v(0))), Validity::Invalid(_)));
    }

    #[test]
    fn divisibility() {
        let f = Facts::default();
        assert_eq!(entails(&f, &Constraint::Divides(12, DimExpr::nat(768))), Validity::Valid);
        assert!(matches!(entails(&f, &Constraint::Divides(7, DimExpr::nat(768))), Validity::Invalid(_)));
        assert!(matches!(entails(&f, &Constraint::Divides(12, v(0))), Validity::Invalid(_)));
        let mut f = Facts::default();
        f.push(Constraint::Divides(12, v(0)));
        assert_eq!(entails(&f, &Constraint::Divides(4, v(0))), Validity::Valid);
        assert_eq!(entails(&f, &Constraint::Divides(12, v(0).scale(&rat(3)))), Validity::Valid);
        // D/12 * 12 = D under 12 | D
        let hd = v(0).div(&DimExpr::nat(12)).unwrap();
        assert_eq!(entails(&f, &Constraint::Eq(hd.scale(&rat(12)), v(0))), Validity::Valid);
    }
}
