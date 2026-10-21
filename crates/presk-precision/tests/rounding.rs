//! The estimate's rounding term measured on the envelope, and the narrowing
//! step decided by measurement (D155).

use presk_precision::envelope::Envelope;
use presk_precision::sensitivity::Sensitivity;
use presk_precision::solve::{self, Budget, Menu, Narrow, Options, RoundingTerm};
use presk_rt::{Dtype, Realization};

/// Sites of magnitude 1 and sensitivity 1, with these elements and these
/// rounding errors `[f64, f32, bf16]`.
fn envelope(sites: &[(&str, u64, [f64; 3])]) -> (Envelope, Sensitivity) {
    let mut e = Envelope::new("p.psk", "f", 128, "2026-10-19");
    e.observe(&sites.iter().map(|(s, _, _)| (s.to_string(), 1.0)).collect());
    e.observe_elems(&sites.iter().map(|(s, n, _)| (s.to_string(), *n)).collect());
    e.observe_rounding(&sites.iter().map(|(s, _, r)| (s.to_string(), *r)).collect());
    let sens = Sensitivity { sites: sites.iter().map(|(s, _, _)| (s.to_string(), 1.0)).collect() };
    (e, sens)
}

fn solve_with(env: &Envelope, sens: &Sensitivity, budget: f64, options: Options, verify: Option<&mut dyn FnMut(&Realization) -> f64>) -> solve::Solution {
    solve::solve_with(env, sens, &solve::Roles::default(), "f/%2", Budget::Absolute(budget), &Menu::default(), Dtype::F64, &solve::Traffic, verify, options)
}

#[test]
fn a_value_exact_in_bf16_costs_nothing_under_the_measured_term() {
    // A mask of ones (exact in every type) and a value that rounds as the
    // unit term says, under a budget `bf16`'s unit term exceeds.
    let rounds = [Dtype::F64.unit_roundoff(), Dtype::F32.unit_roundoff(), Dtype::Bf16.unit_roundoff()];
    let (env, sens) = envelope(&[("f/%1", 1000, [0.0; 3]), ("f/%2", 10, rounds)]);
    let budget = 1e-4;
    assert!(Dtype::Bf16.unit_roundoff() > budget);
    let unit = solve_with(&env, &sens, budget, Options { rounding: RoundingTerm::Unit, narrow: Narrow::Off, ..Options::default() }, None);
    assert_eq!(unit.realization.storage.get("f/%1"), Some(&Dtype::F32), "the unit term charges the ones");
    let measured = solve_with(&env, &sens, budget, Options { rounding: RoundingTerm::Measured, narrow: Narrow::Off, ..Options::default() }, None);
    assert_eq!(measured.realization.storage.get("f/%1"), Some(&Dtype::Bf16), "measured, they cost nothing in bf16");
    assert_eq!(measured.realization.storage.get("f/%2"), Some(&Dtype::F32), "and the other value is charged as before");
    assert!(measured.bytes < unit.bytes);
    // An envelope without the measured term keeps the unit one.
    let mut old = env.clone();
    old.rounding.clear();
    let again = solve_with(&old, &sens, budget, Options { rounding: RoundingTerm::Measured, narrow: Narrow::Off, ..Options::default() }, None);
    assert_eq!(again.realization, unit.realization);
}

#[test]
fn the_default_options_are_the_measured_term_without_narrowing() {
    // 0.16.0 (D157): the measured term by default, the narrowing a flag.
    let o = Options::default();
    assert_eq!((o.rounding, o.narrow), (RoundingTerm::Measured, Narrow::Off));
}

#[test]
fn narrowing_keeps_a_step_only_while_the_measurement_fits() {
    // Three sites whose `bf16` the estimate refuses; the "measurement" adds
    // 1e-6 a site in `f32` and 1e-3 a site in `bf16`.
    let rounds = [Dtype::F64.unit_roundoff(), Dtype::F32.unit_roundoff(), Dtype::Bf16.unit_roundoff()];
    let (env, sens) = envelope(&[("f/%1", 300, rounds), ("f/%2", 200, rounds), ("f/%3", 100, rounds)]);
    let budget = 2.5e-3;
    let gap = |r: &Realization| -> f64 {
        r.storage.values().map(|d| match d {
            Dtype::F64 => 0.0,
            Dtype::F32 => 1e-6,
            Dtype::Bf16 => 1e-3,
        }).sum()
    };
    let mut calls = 0usize;
    let mut measure = |r: &Realization| {
        calls += 1;
        gap(r)
    };
    let off = solve_with(&env, &sens, budget, Options { rounding: RoundingTerm::Unit, narrow: Narrow::Off, ..Options::default() }, Some(&mut measure));
    assert_eq!(off.realization.storage.values().filter(|d| **d == Dtype::F32).count(), 3, "the proposal: every site in f32");
    assert_eq!((off.narrowed, off.measurements), (0, 1));
    calls = 0;
    let mut measure = |r: &Realization| {
        calls += 1;
        gap(r)
    };
    let on = solve_with(&env, &sens, budget, Options { rounding: RoundingTerm::Unit, narrow: Narrow::Measured, margin: 1.0, ..Options::default() }, Some(&mut measure));
    // Largest saving first: %1 and %2 fit in bf16, %3 does not, twice.
    assert_eq!(on.realization.storage.get("f/%1"), Some(&Dtype::Bf16));
    assert_eq!(on.realization.storage.get("f/%2"), Some(&Dtype::Bf16));
    assert_eq!(on.realization.storage.get("f/%3"), Some(&Dtype::F32));
    assert_eq!(on.narrowed, 2);
    assert_eq!(on.narrow_passes, 2, "a second pass, since the first changed something, which changes nothing");
    assert_eq!(on.measurements, 1 + 3 + 1);
    assert_eq!(calls, on.measurements);
    assert!((on.measured.unwrap() - gap(&on.realization)).abs() < 1e-18, "the measurement of the realization returned");
    assert!(on.measured.unwrap() <= budget);
    let kept: Vec<_> = on.moves.iter().filter_map(|m| if let solve::Move::Narrowed { site, kept, .. } = m { Some((site.as_str(), *kept)) } else { None }).collect();
    assert_eq!(kept, vec![("f/%1", true), ("f/%2", true), ("f/%3", false), ("f/%3", false)]);
}

#[test]
fn the_narrowing_undoes_every_step_the_measurement_refuses() {
    let rounds = [Dtype::F64.unit_roundoff(), Dtype::F32.unit_roundoff(), Dtype::Bf16.unit_roundoff()];
    let (env, sens) = envelope(&[("f/%1", 300, rounds), ("f/%2", 200, rounds)]);
    // Every realization but all-f64 fails: the widening walks back to f64,
    // and every narrowing step from there is measured and undone.
    let mut measure = |r: &Realization| if r.storage.is_empty() { 0.0 } else { 1.0 };
    let s = solve_with(&env, &sens, 1e-3, Options { rounding: RoundingTerm::Unit, narrow: Narrow::Measured, margin: 1.0, ..Options::default() }, Some(&mut measure));
    assert!(s.realization.storage.is_empty());
    assert_eq!((s.narrowed, s.narrow_passes), (0, 1));
    assert_eq!(s.measured, Some(0.0), "the measurement of the realization returned, not of a step undone");
    assert_eq!(s.measurements, 1 + s.rounds + 2);
}

#[test]
fn the_interpreters_bf16_is_the_runtimes() {
    // Every rounding case: exact values, ties to even both ways, the largest
    // finite values, subnormals, infinities and both signs.
    let mut xs: Vec<f32> = vec![0.0, -0.0, 1.0, 32.0, 0.2, 1.0 / 3.0, f32::MAX, f32::MIN_POSITIVE, 1e-40, f32::INFINITY, -f32::INFINITY];
    for hi in [0x3f80u32, 0x3f81, 0x4000, 0x7f7f, 0x0001, 0x8000 | 0x3f80] {
        for lo in [0x0000u32, 0x7fff, 0x8000, 0x8001, 0xffff] {
            xs.push(f32::from_bits((hi << 16) | lo));
        }
    }
    for x in xs {
        let want = presk_rt::bf16_to_f32(presk_rt::f32_to_bf16(x));
        let got = presk_interp::eval::bf16_of(x);
        assert_eq!(got.to_bits(), want.to_bits(), "{x:e}");
    }
    // A double's rounding: exact where it is representable, at most the
    // unit term elsewhere.
    for x in [1.0f64, 32.0, 0.5, -2.0, 0.0] {
        assert_eq!(presk_interp::eval::rounding_of_f64(x), [0.0; 3], "{x}");
    }
    for x in [1.0f64 / 3.0, 0.2, 5.656854249492381, -1e-5, 123.456] {
        let r = presk_interp::eval::rounding_of_f64(x);
        assert!(r[1] > 0.0 && r[1] <= Dtype::F32.unit_roundoff() * x.abs(), "{x}: {r:?}");
        assert!(r[2] > 0.0 && r[2] <= Dtype::Bf16.unit_roundoff() * x.abs(), "{x}: {r:?}");
    }
}
