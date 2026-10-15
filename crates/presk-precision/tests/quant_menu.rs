//! The parameter menu (plan 0.6.0 step 5d): what it offers, how it proposes,
//! and that measurement, not the estimate, has the last word.

use std::collections::BTreeMap;

use presk_precision::quant::{candidates, solve};
use presk_rt::{Axis, Buf, Dtype, QuantScheme, Realization, Tensor, Value};

fn t(dims: &[usize], f: impl Fn(usize) -> f64) -> Value {
    let n: usize = dims.iter().product();
    Value::Real(Tensor { axes: dims.iter().map(|d| Axis { name: None, len: *d }).collect(), data: Buf::F64((0..n).map(f).collect()) })
}

/// A model with a matrix, a stacked matrix, a stacked vector and a vector.
fn model(scale: f64) -> Value {
    Value::Record(vec![
        ("head".into(), t(&[8, 6], |i| scale * ((i * 37 % 23) as f64 - 11.0) / 7.0)),
        ("blocks".into(), Value::Record(vec![("w".into(), t(&[2, 8, 6], |i| scale * ((i * 13 % 17) as f64 - 8.0) / 5.0)), ("b".into(), t(&[2, 6], |i| i as f64))])),
        ("γ".into(), t(&[6], |_| 1.0)),
    ])
}

fn declared() -> BTreeMap<String, usize> {
    [("head", 2), ("blocks.w", 2), ("blocks.b", 1), ("γ", 1)].into_iter().map(|(p, r)| (p.to_string(), r)).collect()
}

/// Matrices as declared: the stacked vector has a leaf of rank two and is
/// not offered, the stacked matrix is.
#[test]
fn the_menu_offers_matrices_as_declared() {
    let c = candidates(&model(1.0), &model(1.0), &declared(), &|_| 4, QuantScheme::Absmax);
    let paths: Vec<&str> = c.iter().map(|c| c.path.as_str()).collect();
    assert_eq!(paths, vec!["head", "blocks.w"]);
    for c in &c {
        assert!(c.est4 > c.est8 && c.est8 > 0.0, "{}: four bits cost more than eight", c.path);
        assert!(c.bytes[0] > c.bytes[1] && c.bytes[1] > c.bytes[2], "{}: and save more", c.path);
    }
}

/// A generous budget quantises everything to four bits; a zero budget
/// quantises nothing; in between, the estimate is respected.
#[test]
fn the_proposal_respects_the_estimate() {
    let (w, g) = (model(1.0), model(1.0));
    let base = Realization::uniform(Dtype::F32);
    let all = solve(candidates(&w, &g, &declared(), &|_| 4, QuantScheme::Absmax), f64::INFINITY, &base, QuantScheme::Absmax, None);
    assert!(all.chosen.values().all(|b| *b == Some(4)));
    assert_eq!(all.quant.len(), 2);
    let none = solve(candidates(&w, &g, &declared(), &|_| 4, QuantScheme::Absmax), 0.0, &base, QuantScheme::Absmax, None);
    assert!(none.quant.is_empty());
    let cands = candidates(&w, &g, &declared(), &|_| 4, QuantScheme::Absmax);
    let budget = cands.iter().map(|c| c.est8).sum::<f64>() * 1.01;
    let some = solve(cands, budget, &base, QuantScheme::Absmax, None);
    assert!(some.estimated <= budget);
    assert!(some.bytes_after < some.bytes_before);
}

/// The estimate proposes and the measurement disposes: a measurement that
/// refuses anything quantised to four bits makes the solver widen until it
/// accepts, and the result is what was measured.
#[test]
fn measurement_has_the_last_word() {
    // Small gradients: the estimate lets every matrix go to four bits.
    let (w, g) = (model(1.0), model(1e-6));
    let base = Realization::uniform(Dtype::F32);
    let mut calls = 0;
    let mut measure = |r: &Realization| -> f64 {
        calls += 1;
        if r.quant.values().any(|q| q.bits == 4) { 1.0 } else { 0.0 }
    };
    let s = solve(candidates(&w, &g, &declared(), &|_| 4, QuantScheme::Absmax), 0.5, &base, QuantScheme::Absmax, Some(&mut measure));
    assert_eq!(s.measured, Some(0.0));
    assert!(s.chosen.values().all(|b| *b == Some(8)), "{:?}", s.chosen);
    assert_eq!(s.rounds, 2);
    assert_eq!(calls, 3);
}
