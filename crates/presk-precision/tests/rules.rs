//! The two rules of 0.16.1 (D169): a value of one element is not offered,
//! nor a value whose first narrowing, `f64` to `f32`, saves less than θ of
//! the entry's all-`f64` traffic.

use presk_precision::envelope::Envelope;
use presk_precision::sensitivity::Sensitivity;
use presk_precision::solve::{self, Budget, Menu, Options, ParamSite, Roles, Rule};
use presk_rt::{Dtype, Realization};

/// Sites of magnitude 1 and sensitivity 1, with these elements, their
/// rounding the unit term's.
fn envelope(sites: &[(&str, u64)]) -> (Envelope, Sensitivity) {
    let rounds = [Dtype::F64.unit_roundoff(), Dtype::F32.unit_roundoff(), Dtype::Bf16.unit_roundoff()];
    let mut e = Envelope::new("p.psk", "f", 128, "2026-10-22");
    e.observe(&sites.iter().map(|(s, _)| (s.to_string(), 1.0)).collect());
    e.observe_elems(&sites.iter().map(|(s, n)| (s.to_string(), *n)).collect());
    e.observe_rounding(&sites.iter().map(|(s, _)| (s.to_string(), rounds)).collect());
    let sens = Sensitivity { sites: sites.iter().map(|(s, _)| (s.to_string(), 1.0)).collect() };
    (e, sens)
}

fn solve(env: &Envelope, sens: &Sensitivity, roles: &Roles, budget: f64, options: Options, verify: Option<&mut dyn FnMut(&Realization) -> f64>) -> solve::Solution {
    solve::solve_with(env, sens, roles, "f/%1", Budget::Absolute(budget), &Menu::default(), Dtype::F64, &solve::Traffic, verify, options)
}

/// A budget every value fits in `bf16`.
const LOOSE: f64 = 1.0;

fn excluded(s: &solve::Solution) -> Vec<(&str, Rule)> {
    s.offers.excluded.iter().map(|e| (e.name.as_str(), e.rule)).collect()
}

#[test]
fn a_stored_value_of_one_element_is_not_offered() {
    let (env, sens) = envelope(&[("f/%1", 300), ("f/%2", 200), ("f/%3", 1)]);
    let s = solve(&env, &sens, &Roles::default(), LOOSE, Options::default(), None);
    assert_eq!(excluded(&s), [("f/%3", Rule::OneElement)]);
    assert_eq!(s.realization.storage.get("f/%3"), None, "it keeps the default type");
    assert_eq!(s.realization.storage.get("f/%1"), Some(&Dtype::Bf16));
    assert_eq!(s.realization.storage.get("f/%2"), Some(&Dtype::Bf16));
    // No move names it, and the cost counts it at f64.
    assert!(s.moves.iter().all(|m| !matches!(m, solve::Move::Tried { site, .. } if site == "f/%3")));
    assert_eq!(s.bytes, (300.0 + 200.0) * 2.0 + 8.0);
    assert_eq!(s.offers.base, 501.0 * 8.0);
    // Rule 1 has no option: θ = 0 leaves it out all the same.
    let off = solve(&env, &sens, &Roles::default(), LOOSE, Options { min_gain: 0.0, ..Options::default() }, None);
    assert_eq!(excluded(&off), [("f/%3", Rule::OneElement)]);
}

#[test]
fn a_parameter_of_one_element_is_not_offered() {
    let (env, sens) = envelope(&[("f/%1", 300), ("f/%2", 200)]);
    let param = |path: &str, elems: f64| ParamSite { path: path.into(), elems, mag: 1.0, sens: 1.0, fields: Vec::new(), round: None, ints: Vec::new() };
    let roles = Roles::default().with_params(vec![param("θ.w", 400.0), param("θ.t", 1.0)]);
    let s = solve(&env, &sens, &roles, LOOSE, Options::default(), None);
    assert_eq!(excluded(&s), [("θ.t", Rule::OneElement)]);
    assert!(s.offers.excluded[0].param);
    assert_eq!(s.realization.params.get("θ.t"), None);
    assert_eq!(s.realization.params.get("θ.w"), Some(&Dtype::Bf16));
}

#[test]
fn rule_2_on_either_side_of_its_threshold() {
    // θ = 10⁻²: B = 8·(1000 + n) bytes, and the small value's first step
    // saves 4n. At n = 20, 80 < 81.6: not offered; at n = 21, 84 ≥ 81.68.
    let theta = Options { min_gain: 1e-2, ..Options::default() };
    let (env, sens) = envelope(&[("f/%1", 1000), ("f/%2", 20)]);
    let s = solve(&env, &sens, &Roles::default(), LOOSE, theta, None);
    assert_eq!(excluded(&s), [("f/%2", Rule::MinGain)]);
    let e = &s.offers.excluded[0];
    assert_eq!((e.elems, e.gain, e.forgone), (20.0, 80.0, 120.0));
    assert_eq!(s.offers.threshold(), 1e-2 * 8160.0);
    assert_eq!(s.realization.storage.get("f/%2"), None);
    assert_eq!(s.offers.forgone(), 120.0);
    assert_eq!(s.traffic, 1000.0 * 2.0 + 20.0 * 8.0);
    let (env, sens) = envelope(&[("f/%1", 1000), ("f/%2", 21)]);
    let s = solve(&env, &sens, &Roles::default(), LOOSE, theta, None);
    assert!(s.offers.excluded.is_empty());
    assert_eq!(s.realization.storage.get("f/%2"), Some(&Dtype::Bf16));
    // θ = 0 turns rule 2 off.
    let (env, sens) = envelope(&[("f/%1", 1000), ("f/%2", 20)]);
    let s = solve(&env, &sens, &Roles::default(), LOOSE, Options { min_gain: 0.0, ..Options::default() }, None);
    assert!(s.offers.excluded.is_empty());
    assert_eq!(s.realization.storage.get("f/%2"), Some(&Dtype::Bf16));
}

#[test]
fn no_value_is_offered_bf16_without_f32() {
    // n = 15: B = 8120, θ·B = 81.2; `f32` would save 60 and `bf16` 90. A
    // threshold per type would offer `bf16` alone; the rule offers neither.
    let theta = Options { min_gain: 1e-2, ..Options::default() };
    let (env, sens) = envelope(&[("f/%1", 1000), ("f/%2", 15)]);
    let s = solve(&env, &sens, &Roles::default(), LOOSE, theta, None);
    assert_eq!(excluded(&s), [("f/%2", Rule::MinGain)]);
    assert_eq!(s.realization.storage.get("f/%2"), None);
    // Every decision taken saves at least θ·B against f64.
    for (site, d) in &s.realization.storage {
        let n = if site == "f/%1" { 1000.0 } else { 15.0 };
        assert!(n * (8 - d.bytes()) as f64 >= s.offers.threshold(), "{site} in {d:?}");
    }
}

#[test]
fn the_measurement_never_widens_or_narrows_a_value_not_offered() {
    // A "measurement" over the budget until f/%1 is widened twice: the
    // tightening walks f/%1 alone, never the value of one element.
    let (env, sens) = envelope(&[("f/%1", 300), ("f/%2", 1)]);
    let mut m = |r: &Realization| match r.storage.get("f/%1") {
        Some(Dtype::Bf16) | Some(Dtype::F32) => 2.0,
        _ => 0.0,
    };
    let s = solve(&env, &sens, &Roles::default(), LOOSE, Options::default(), Some(&mut m));
    assert_eq!(s.rounds, 2);
    assert!(s.moves.iter().all(|mv| match mv {
        solve::Move::Widened { site, .. } | solve::Move::Narrowed { site, .. } => site == "f/%1",
        _ => true,
    }));
    assert_eq!(s.realization.storage.get("f/%2"), None);
    let mut m = |_: &Realization| 0.0;
    let s = solve(&env, &sens, &Roles::default(), LOOSE, Options { narrow: solve::Narrow::Measured, margin: 1.0, ..Options::default() }, Some(&mut m));
    assert!(s.moves.iter().all(|mv| !matches!(mv, solve::Move::Narrowed { site, .. } if site == "f/%2")));
}

#[test]
fn the_exhaustive_search_runs_over_the_values_offered() {
    let (env, sens) = envelope(&[("f/%1", 300), ("f/%2", 200), ("f/%3", 1)]);
    let (r, bytes) = solve::exhaustive_with(&env, &sens, &Roles::default(), "f/%1", Budget::Absolute(LOOSE), &Menu::default(), Dtype::F64, &solve::Traffic, 1000, Options::default()).unwrap();
    assert_eq!(r.storage.get("f/%3"), None);
    assert_eq!(bytes, 500.0 * 2.0 + 8.0);
    // Three values offered would be 27 states; two are 9.
    assert!(solve::exhaustive_with(&env, &sens, &Roles::default(), "f/%1", Budget::Absolute(LOOSE), &Menu::default(), Dtype::F64, &solve::Traffic, 9, Options::default()).is_some());
}

#[test]
fn without_a_value_the_rules_leave_out_the_solver_is_0_16_0_s() {
    // No value of one element, and every first step far above θ·B: the
    // rules change nothing, at the default θ or at 0.
    let (env, sens) = envelope(&[("f/%1", 300), ("f/%2", 200), ("f/%3", 100)]);
    for budget in [LOOSE, 5e-3, 1e-6] {
        let a = solve(&env, &sens, &Roles::default(), budget, Options::default(), None);
        let b = solve(&env, &sens, &Roles::default(), budget, Options { min_gain: 0.0, ..Options::default() }, None);
        assert!(a.offers.excluded.is_empty());
        assert_eq!(a.realization, b.realization);
        assert_eq!(a.moves, b.moves);
        assert_eq!((a.bytes, a.relaxation_bytes, a.estimated_error), (b.bytes, b.relaxation_bytes, b.estimated_error));
    }
}
