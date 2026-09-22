//! The solver over a training step (plan 0.6.0 §3.7, step 7b): groups are
//! whole functions, their gaps are measured alone, the proposal combines
//! them as a root-sum-square, and the measurement of the combination has
//! the last word.

use std::collections::BTreeMap;

use presk_precision::solve::{Group, Menu, by_function, solve_step, step_cost};
use presk_rt::{Dtype, Realization};

fn group(name: &str, f64_cost: f64) -> Group {
    let cost = BTreeMap::from([(Dtype::F64, f64_cost), (Dtype::F32, f64_cost / 2.0), (Dtype::Bf16, f64_cost / 3.0)]);
    Group { name: name.into(), sites: 1, elems: 1, cost, gap: BTreeMap::new() }
}

/// A synthetic step: each function's isolated gap per dtype, and a
/// combination that is measured as the root-sum-square of its parts unless
/// `interact` says the two named functions together cost more.
fn measure_with(isolated: &BTreeMap<(String, Dtype), f64>, interact: bool) -> impl FnMut(&Realization) -> f64 + '_ {
    move |r: &Realization| {
        let parts: Vec<f64> = r.functions.iter().map(|(f, d)| isolated[&(f.clone(), *d)]).collect();
        let both = r.functions.contains_key("adamw.step") && r.functions.contains_key("Linear.apply.vjp");
        parts.iter().map(|x| x * x).sum::<f64>().sqrt() * if interact && both { 100.0 } else { 1.0 }
    }
}

fn isolated() -> BTreeMap<(String, Dtype), f64> {
    [
        (("adamw.step", Dtype::F32), 1e-4),
        (("adamw.step", Dtype::Bf16), 0.9),
        (("Linear.apply.vjp", Dtype::F32), 1e-6),
        (("Linear.apply.vjp", Dtype::Bf16), 1e-3),
        (("GPT.apply", Dtype::F32), 1e-7),
        (("GPT.apply", Dtype::Bf16), 5e-4),
    ]
    .into_iter()
    .map(|((f, d), g)| ((f.to_string(), d), g))
    .collect()
}

/// The optimizer's `bf16` is refused on its isolated gap alone; everything
/// else fits a 1e-2 budget, and the result is what was measured.
#[test]
fn the_proposal_fits_the_budget() {
    let iso = isolated();
    let groups = vec![group("adamw.step", 10.0), group("Linear.apply.vjp", 30.0), group("GPT.apply", 20.0)];
    let s = solve_step(groups.clone(), 1e-2, &Menu::default(), Dtype::F64, &mut measure_with(&iso, false), true);
    assert_eq!(s.chosen["adamw.step"], Dtype::F32);
    assert_eq!(s.chosen["Linear.apply.vjp"], Dtype::Bf16);
    assert_eq!(s.chosen["GPT.apply"], Dtype::Bf16);
    assert_eq!(s.rounds, 0);
    assert!((s.measured.unwrap() - s.estimate).abs() < 1e-15);
    assert!((step_cost(&s.groups, &s.realization) - s.cost).abs() < 1e-12);
    assert!(s.speedup() > 2.0);
}

/// When the combination measures outside although the parts fit, the group
/// with the largest isolated gap is widened one step of the menu, not
/// straight to `f64`.
#[test]
fn measurement_has_the_last_word() {
    let iso = isolated();
    let groups = vec![group("adamw.step", 10.0), group("Linear.apply.vjp", 30.0), group("GPT.apply", 20.0)];
    let s = solve_step(groups, 1e-2, &Menu::default(), Dtype::F64, &mut measure_with(&iso, true), true);
    assert!(s.measured.unwrap() <= 1e-2, "{:?}", s.measured);
    assert!(s.rounds >= 1);
    assert_eq!(s.chosen["Linear.apply.vjp"], Dtype::F32, "widened by one step: {:?}", s.chosen);
}

/// A budget below every isolated gap narrows nothing.
#[test]
fn a_tight_budget_is_all_f64() {
    let iso = isolated();
    let groups = vec![group("adamw.step", 10.0), group("Linear.apply.vjp", 30.0)];
    let s = solve_step(groups, 1e-9, &Menu::default(), Dtype::F64, &mut measure_with(&iso, false), true);
    assert!(s.realization.functions.is_empty());
    assert_eq!(s.realization, by_function(Dtype::F64, &BTreeMap::new()));
}
