//! Property tests of the decision procedure: a `Valid` answer is never wrong
//! (soundness), and equal linear forms are always recognized (completeness on
//! the equality fragment).

use std::collections::BTreeMap;

use num_bigint::BigInt;
use num_rational::BigRational;
use presk_types::dim::{DimExpr, DimVar, rat};
use presk_types::presburger::{Constraint, Facts, Validity, entails};
use proptest::prelude::*;

fn form() -> impl Strategy<Value = DimExpr> {
    (prop::collection::vec(-4i64..=4, 3), -8i64..=8).prop_map(|(cs, c)| {
        let mut e = DimExpr::nat(0);
        for (i, k) in cs.iter().enumerate() {
            e = e.add(&DimExpr::var(DimVar(i as u32)).scale(&rat(*k)));
        }
        e.add(&DimExpr::constant(rat(c)))
    })
}

fn constraint() -> impl Strategy<Value = Constraint> {
    prop_oneof![
        (form(), form()).prop_map(|(a, b)| Constraint::Eq(a, b)),
        (form(), form()).prop_map(|(a, b)| Constraint::Le(a, b)),
        (1u64..=6, form()).prop_map(|(k, e)| Constraint::Divides(k, e)),
    ]
}

fn assignment() -> impl Strategy<Value = BTreeMap<DimVar, BigRational>> {
    prop::collection::vec(0i64..=12, 3).prop_map(|vs| vs.iter().enumerate().map(|(i, v)| (DimVar(i as u32), BigRational::from_integer(BigInt::from(*v)))).collect())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Whenever `entails` answers Valid, every assignment satisfying the
    /// facts satisfies the goal.
    #[test]
    fn valid_is_sound(facts in prop::collection::vec(constraint(), 0..3), goal in constraint(), samples in prop::collection::vec(assignment(), 50)) {
        let facts = Facts { constraints: facts };
        if entails(&facts, &goal) == Validity::Valid {
            for a in &samples {
                if facts.holds(a) {
                    prop_assert_eq!(goal.holds(a), Some(true), "counterexample {:?} to {:?} under {:?}", a, goal, facts);
                }
            }
        }
    }

    /// A counterexample really is one.
    #[test]
    fn invalid_has_witness(facts in prop::collection::vec(constraint(), 0..3), goal in constraint()) {
        let facts = Facts { constraints: facts };
        if let Validity::Invalid(a) = entails(&facts, &goal) {
            prop_assert!(facts.holds(&a));
            prop_assert_eq!(goal.holds(&a), Some(false));
        }
    }

    /// Rewritings of a form are recognized as equal without facts.
    #[test]
    fn equal_forms_are_valid(e in form(), k in 1i64..=5) {
        let facts = Facts::default();
        let doubled = e.add(&e).div(&DimExpr::nat(2)).unwrap();
        prop_assert_eq!(entails(&facts, &Constraint::Eq(e.clone(), doubled)), Validity::Valid);
        let scaled = e.scale(&rat(k)).div(&DimExpr::nat(k as u64)).unwrap();
        prop_assert_eq!(entails(&facts, &Constraint::Eq(e.clone(), scaled)), Validity::Valid);
        prop_assert_eq!(entails(&facts, &Constraint::Le(e.clone(), e.add(&DimExpr::nat(1)))), Validity::Valid);
    }

    /// Under `k | D`, `k` divides `c·D` and `D/k · k = D`.
    #[test]
    fn divisibility_under_fact(k in 2u64..=12, c in 1i64..=5) {
        let d = DimExpr::var(DimVar(0));
        let mut facts = Facts::default();
        facts.push(Constraint::Divides(k, d.clone()));
        prop_assert_eq!(entails(&facts, &Constraint::Divides(k, d.scale(&rat(c)))), Validity::Valid);
        let q = d.div(&DimExpr::nat(k)).unwrap();
        prop_assert_eq!(entails(&facts, &Constraint::Eq(q.scale(&rat(k as i64)), d.clone())), Validity::Valid);
    }
}
