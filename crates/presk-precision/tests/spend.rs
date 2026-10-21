//! The three ways to spend the budget (plan 0.16.0 step 4, D161): the
//! margin as the narrowing's threshold, the root-sum-square estimate aimed
//! at the budget over the margin, and early stopping on held-out samples.

use presk_precision::envelope::Envelope;
use presk_precision::sensitivity::Sensitivity;
use presk_precision::solve::{self, Budget, Estimate, Menu, Narrow, Options, RoundingTerm};
use presk_rt::{Dtype, Realization};

/// Sites of magnitude 1 and sensitivity 1, with these elements, their
/// rounding the unit term's.
fn envelope(sites: &[(&str, u64)]) -> (Envelope, Sensitivity) {
    let rounds = [Dtype::F64.unit_roundoff(), Dtype::F32.unit_roundoff(), Dtype::Bf16.unit_roundoff()];
    let mut e = Envelope::new("p.psk", "f", 128, "2026-10-19");
    e.observe(&sites.iter().map(|(s, _)| (s.to_string(), 1.0)).collect());
    e.observe_elems(&sites.iter().map(|(s, n)| (s.to_string(), *n)).collect());
    e.observe_rounding(&sites.iter().map(|(s, _)| (s.to_string(), rounds)).collect());
    let sens = Sensitivity { sites: sites.iter().map(|(s, _)| (s.to_string(), 1.0)).collect() };
    (e, sens)
}

fn solve(env: &Envelope, sens: &Sensitivity, budget: f64, options: Options, verify: Option<&mut dyn FnMut(&Realization) -> f64>, heldout: Option<&mut dyn FnMut(&Realization) -> f64>) -> solve::Solution {
    solve::solve_with_heldout(env, sens, &solve::Roles::default(), "f/%2", Budget::Absolute(budget), &Menu::default(), Dtype::F64, &solve::Traffic, verify, heldout, options)
}

/// A "measurement" adding 1e-6 a site in `f32` and 1e-3 a site in `bf16`.
fn gap(r: &Realization) -> f64 {
    r.storage.values().map(|d| match d {
        Dtype::F64 => 0.0,
        Dtype::F32 => 1e-6,
        Dtype::Bf16 => 1e-3,
    }).sum()
}

const SITES: [(&str, u64); 3] = [("f/%1", 300), ("f/%2", 200), ("f/%3", 100)];

#[test]
fn the_margin_is_the_narrowing_s_threshold() {
    let (env, sens) = envelope(&SITES);
    let budget = 2.5e-3;
    let narrow = |margin: f64| {
        let mut m = |r: &Realization| gap(r);
        solve(&env, &sens, budget, Options { narrow: Narrow::Measured, margin, ..Options::default() }, Some(&mut m), None)
    };
    // D155's narrowing, a margin of 1: two sites in bf16, 2.0·10⁻³.
    assert_eq!(narrow(1.0).narrowed, 2);
    // Under 2.5·10⁻³ / 2.5 = 10⁻³ no bf16 fits beside the f32 ones.
    let k = narrow(2.5);
    assert_eq!(k.narrowed, 0);
    assert!(k.measured.unwrap() <= budget / 2.5);
}

#[test]
fn the_rss_estimate_aims_at_the_budget_over_the_margin() {
    let (env, sens) = envelope(&SITES);
    // A bf16 term is 2⁻⁸ ≈ 3.9·10⁻³: the sum fits one under 6·10⁻³, the
    // root-sum-square two (√2 · 3.9·10⁻³ = 5.5·10⁻³).
    let budget = 6e-3;
    let bf16 = |s: &solve::Solution| s.realization.storage.values().filter(|d| **d == Dtype::Bf16).count();
    let sum = solve(&env, &sens, budget, Options { rounding: RoundingTerm::Unit, ..Options::default() }, None, None);
    let rss1 = solve(&env, &sens, budget, Options { rounding: RoundingTerm::Unit, estimate: Estimate::Rss, margin: 1.0, ..Options::default() }, None, None);
    let rss = solve(&env, &sens, budget, Options { rounding: RoundingTerm::Unit, estimate: Estimate::Rss, ..Options::default() }, None, None);
    assert_eq!((bf16(&sum), bf16(&rss1), bf16(&rss)), (1, 2, 0));
    assert!(rss1.bytes < sum.bytes);
    assert!(rss1.estimated_error <= budget && rss1.estimated_error > 5e-3, "the estimate is the root-sum-square: {}", rss1.estimated_error);
    assert!(rss.estimated_error <= budget / 2.5);
}

#[test]
fn held_out_samples_undo_the_last_narrowings_until_they_fit() {
    let (env, sens) = envelope(&SITES);
    let budget = 2.5e-3;
    let mut m = |r: &Realization| gap(r);
    // The held-out samples see 1.5 times the envelope's gap.
    let mut held = 0usize;
    let mut h = |r: &Realization| {
        held += 1;
        1.5 * gap(r)
    };
    let s = solve(&env, &sens, budget, Options { narrow: Narrow::Heldout, margin: 1.0, ..Options::default() }, Some(&mut m), Some(&mut h));
    // Narrowed: %1 and %2 in bf16, 2.0·10⁻³ on the envelope, 3.0·10⁻³ held
    // out; %2 undone, 1.5·10⁻³ held out.
    assert_eq!(s.realization.storage.get("f/%1"), Some(&Dtype::Bf16));
    assert_eq!(s.realization.storage.get("f/%2"), Some(&Dtype::F32));
    assert_eq!((s.narrowed, s.undone), (1, 1));
    assert_eq!(held, 2);
    assert!((s.heldout.unwrap() - 1.5 * gap(&s.realization)).abs() < 1e-18);
    assert!((s.measured.unwrap() - gap(&s.realization)).abs() < 1e-18, "the envelope's gap of the realization returned");
    assert!(s.moves.iter().any(|m| matches!(m, solve::Move::Undone { site, .. } if site == "f/%2")));
    // Without held-out samples, `Heldout` is `Measured`.
    let mut m = |r: &Realization| gap(r);
    let alone = solve(&env, &sens, budget, Options { narrow: Narrow::Heldout, margin: 1.0, ..Options::default() }, Some(&mut m), None);
    assert_eq!((alone.narrowed, alone.undone, alone.heldout), (2, 0, None));
}
