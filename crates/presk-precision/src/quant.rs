//! The parameter menu: which weights to store as `int8` or `int4` (plan 0.6.0
//! step 5d, D44).
//!
//! The float solver assigns a machine type per *site*; quantisation is a
//! property of a *parameter*, so it is a second, smaller choice made on top
//! of the first, under the budget the first left over. Its shape is the float
//! solver's (D41): propose by a first-order estimate, **accept by
//! measurement** against the exact reference, widen while the measurement
//! says no.
//!
//! The estimate needs no calibration. Quantising a parameter moves each
//! weight by a known `δ = ŵ − w` — the weights are at hand, so the rounding
//! is not a bound but a number — and the reverse sweep the solver already
//! runs gives `g = ∂L/∂w`. To first order the output moves by `Σ g·δ`.
//!
//! Three readings of that sum are computed, and the measurement chose
//! between them (D49). The *bound* `Σ |g|·|δ|` assumes every rounding pushes
//! the same way and was 230× the measured gap on the corpus GPT — D39 again,
//! at parameter scale. The *signed* `Σ g·δ` predicted the measured gap within
//! ten per cent, which says the linearisation is right and only the worst
//! case is wrong. The proposal uses the *root-sum-square* `√Σ (g·δ)²`, the
//! size of the change if the roundings are independent: within a factor of
//! two of the measurement, never negative, and combined across parameters as
//! a root-sum-square, so a set of parameters cannot look free because their
//! signed predictions happen to cancel.
//!
//! The menu is every parameter declared of rank two or more — a matrix, not
//! a stacked vector — over its second-to-last axis. It does not know which
//! of them feed a contraction, which is an analysis through the `stack{L}`
//! fold's slicing that nothing here needs yet, so a gathered table (an
//! embedding) is offered too, where the scale is simply per column. The error rule holds on any axis; what is lost there is
//! only the scale's factoring out of a sum, which no kernel uses.
//!
//! This is the two-pass solver of `--quant`, kept for the record. Since
//! plan 0.17.0 step 4 (D177) `presk solve --menu` offers the integers in
//! the float solve itself, priced as the floats are, over the axis the
//! readers contract (`crate::solve`'s `Rung` and `contracted_axes`).

use std::collections::BTreeMap;

use presk_rt::quant::{QTensor, param_paths};
use presk_rt::{AxisRef, Quant, QuantScheme, Realization, Tensor, Value};

/// One parameter the menu offers, with what each rung would cost and save.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub path: String,
    pub elems: usize,
    /// `Σ |g|·|δ|` at `int8` and at `int4`: the first-order *bound*.
    pub est8: f64,
    pub est4: f64,
    /// `Σ g·δ`: the first-order *prediction*, signed.
    pub signed: [f64; 2],
    /// `√Σ (g·δ)²`: the size of the first-order change if the roundings are
    /// independent — the statistical model of question 58.
    pub rss: [f64; 2],
    /// Bytes stored at the realization's float type, at `int8` and at `int4`
    /// (integers plus `f32` scales).
    pub bytes: [usize; 3],
}

/// What the parameter solver chose.
#[derive(Clone, Debug)]
pub struct QuantSolution {
    pub quant: BTreeMap<String, Quant>,
    pub candidates: Vec<Candidate>,
    /// Bits chosen per candidate path (`None`: left as it was).
    pub chosen: BTreeMap<String, Option<u8>>,
    pub estimated: f64,
    pub measured: Option<f64>,
    pub rounds: usize,
    pub bytes_before: usize,
    pub bytes_after: usize,
}

fn leaves(v: &Value) -> BTreeMap<String, Tensor> {
    let mut out = BTreeMap::new();
    fn go(v: &Value, path: &str, out: &mut BTreeMap<String, Tensor>) {
        let join = |n: &str| if path.is_empty() { n.to_string() } else { format!("{path}.{n}") };
        match v {
            Value::Record(fs) => fs.iter().for_each(|(n, x)| go(x, &join(n), out)),
            Value::Tuple(xs) => xs.iter().enumerate().for_each(|(i, x)| go(x, &join(&i.to_string()), out)),
            Value::Real(t) => {
                out.insert(path.to_string(), t.clone());
            }
            _ => {}
        }
    }
    go(v, "", &mut out);
    out
}

fn quant(bits: u8, scheme: QuantScheme) -> Quant {
    Quant { bits, over: AxisRef::Pos(-2), scheme }
}

/// The candidates of `theta`, priced with the gradient `grad` (same tree).
/// `declared` gives each leaf's rank as its module declares it, without the
/// axes `stack{n}` prepends; a leaf it does not name is judged by its own
/// rank. `float_bytes` gives each leaf's bytes an element in the float
/// realization, its own type's for a parameter `params` types (plan 0.12.0).
pub fn candidates(theta: &Value, grad: &Value, declared: &BTreeMap<String, usize>, float_bytes: &dyn Fn(&str) -> usize, scheme: QuantScheme) -> Vec<Candidate> {
    let (w, g) = (leaves(theta), leaves(grad));
    param_paths(theta)
        .into_iter()
        .filter_map(|path| {
            let t = w.get(&path)?;
            // A matrix, as declared: a stacked bias has a leaf of rank two
            // and is a vector, and "over −2" would scale it across layers.
            if declared.get(&path).copied().unwrap_or(t.rank()) < 2 {
                return None;
            }
            let gv = g.get(&path).map(|g| g.data.to_f64()).unwrap_or_else(|| vec![0.0; t.numel()]);
            let wv = t.data.to_f64();
            let (mut est, mut signed, mut rss) = ([0.0; 2], [0.0; 2], [0.0; 2]);
            let mut bytes = [t.numel() * float_bytes(&path), 0, 0];
            for (i, bits) in [8u8, 4].into_iter().enumerate() {
                let q = QTensor::quantize(t, &quant(bits, scheme)).ok()?;
                let Tensor { data, .. } = q.dequantize();
                let terms: Vec<f64> = data.to_f64().iter().zip(&wv).zip(&gv).map(|((d, x), g)| g * (d - x)).collect();
                est[i] = terms.iter().map(|t| t.abs()).sum();
                signed[i] = terms.iter().sum();
                rss[i] = terms.iter().map(|t| t * t).sum::<f64>().sqrt();
                bytes[i + 1] = q.bytes();
            }
            Some(Candidate { path, elems: t.numel(), est8: est[0], est4: est[1], signed, rss, bytes })
        })
        .collect()
}

/// The proposal's estimate of a rung: the root-sum-square, whose squares add.
fn rung_est(c: &Candidate, bits: Option<u8>) -> f64 {
    match bits {
        None => 0.0,
        Some(8) => c.rss[0],
        Some(_) => c.rss[1],
    }
}

fn rung_bytes(c: &Candidate, bits: Option<u8>) -> usize {
    match bits {
        None => c.bytes[0],
        Some(8) => c.bytes[1],
        Some(_) => c.bytes[2],
    }
}

fn narrower(bits: Option<u8>) -> Option<Option<u8>> {
    match bits {
        None => Some(Some(8)),
        Some(8) => Some(Some(4)),
        Some(_) => None,
    }
}

fn wider(bits: Option<u8>) -> Option<Option<u8>> {
    match bits {
        Some(4) => Some(Some(8)),
        Some(_) => Some(None),
        None => None,
    }
}

/// Propose greedily under `budget` (absolute, on the output), then verify by
/// `measure` — the distance of `base` plus the proposal to the exact
/// reference — widening the parameter whose estimate is largest while the
/// measurement exceeds the budget.
pub fn solve(cands: Vec<Candidate>, budget: f64, base: &Realization, scheme: QuantScheme, measure: Option<&mut dyn FnMut(&Realization) -> f64>) -> QuantSolution {
    let mut chosen: BTreeMap<String, Option<u8>> = cands.iter().map(|c| (c.path.clone(), None)).collect();
    // The running sum of squared estimates: the combined estimate is its root.
    let mut total2 = 0.0;
    // Propose: the move saving the most bytes per unit of estimate, while
    // the combined estimate stays inside the budget.
    loop {
        let best = cands
            .iter()
            .filter_map(|c| {
                let now = chosen[&c.path];
                let next = narrower(now)?;
                let d2 = rung_est(c, next).powi(2) - rung_est(c, now).powi(2);
                let db = rung_bytes(c, now).saturating_sub(rung_bytes(c, next));
                ((total2 + d2).sqrt() <= budget && db > 0).then_some((c, next, d2, db as f64 / d2.max(f64::MIN_POSITIVE)))
            })
            .max_by(|a, b| a.3.total_cmp(&b.3));
        let Some((c, next, d2, _)) = best else { break };
        chosen.insert(c.path.clone(), next);
        total2 += d2;
    }
    let build = |chosen: &BTreeMap<String, Option<u8>>| -> Realization {
        let mut r = base.clone();
        for (p, b) in chosen {
            if let Some(bits) = b {
                // Integers replace a float type at rest: a parameter is
                // typed or quantised, not both.
                r.params.remove(p);
                r.quant.insert(p.clone(), quant(*bits, scheme));
            }
        }
        r
    };
    // Verify and tighten.
    let (mut rounds, mut measured) = (0usize, None);
    if let Some(measure) = measure {
        loop {
            let gap = measure(&build(&chosen));
            measured = Some(gap);
            if gap <= budget {
                break;
            }
            let worst = cands.iter().filter(|c| chosen[&c.path].is_some()).max_by(|a, b| rung_est(a, chosen[&a.path]).total_cmp(&rung_est(b, chosen[&b.path])));
            let Some(c) = worst else { break };
            let w = wider(chosen[&c.path]).expect("a quantised parameter can widen");
            chosen.insert(c.path.clone(), w);
            rounds += 1;
        }
    }
    let r = build(&chosen);
    let estimated = cands.iter().map(|c| rung_est(c, chosen[&c.path]).powi(2)).sum::<f64>().sqrt();
    let bytes_before = cands.iter().map(|c| c.bytes[0]).sum();
    let bytes_after = cands.iter().map(|c| rung_bytes(c, chosen[&c.path])).sum();
    QuantSolution { quant: r.quant, candidates: cands, chosen, estimated, measured, rounds, bytes_before, bytes_after }
}
