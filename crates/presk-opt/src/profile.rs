//! The machine profile: the roofline `T̂ = λ + max(W/π, Q/β)` of the
//! cost-model document (§1.1) with its three constants *fitted* on measured
//! kernel times (`presk bench --calibrate`) rather than declared — `π` the
//! sustained operation rate, `β` the sustained bandwidth, `λ` the launch
//! overhead. The fit minimizes the squared *relative* error, so that the
//! microsecond kernels of the examples weigh as much as the millisecond
//! ones; the quality metric is Spearman's rank correlation between predicted
//! and measured times, the roadmap's evaluation metric for cost models.

use std::collections::BTreeMap;

use presk_kernel::cost::{Machine, PlannedCost};
use serde::{Deserialize, Serialize};

/// One measured kernel.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Sample {
    /// The kernel's name and shapes, as `presk dump --stage kernel` prints its header.
    pub signature: String,
    /// The primitive (`add`, `softmax axis seq2`, …).
    pub name: String,
    /// Weighted operation count.
    pub work: f64,
    /// Bytes read and written.
    pub bytes: f64,
    /// Median time of a launch, in seconds.
    pub seconds: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Profile {
    pub version: u32,
    /// `f64` or `f32`: the realization the kernels were timed in.
    pub dtype: String,
    pub machine: String,
    /// Operations per second.
    pub pi: f64,
    /// Bytes per second.
    pub beta: f64,
    /// Seconds per launch.
    pub lambda: f64,
    /// Rank correlations of three predictors with the measured times.
    pub spearman: Correlations,
    pub kernels: Vec<Sample>,
    /// Constants fitted for *other* realizations on the same machine, keyed
    /// by realization label (0.6.0 step 3).
    ///
    /// A narrow type's `π` is not a wide type's — that is the whole reason
    /// vector lanes exist — so a cost model that means to compare machine
    /// types cannot have one `π`. The top-level fields stay the profile's
    /// own realization, so every file written before this step still loads
    /// and still means what it said.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub others: BTreeMap<String, Fit>,
    /// The machine the profile was measured on, as the layout cost model
    /// reads it (0.7.0 step 4): its cache sizes, vector registers and workers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub machine_desc: Option<Machine>,
    /// The layout cost model's constants, fitted on a GEMM under a spanning
    /// set of layouts (`presk bench --calibrate --space calib`, 0.7.0 step 4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<LayoutFit>,
}

/// One kernel timed under one layout: what the layout cost model is fitted on.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PlannedSample {
    /// The kernel's header, as `presk dump --stage kernel` prints it.
    pub signature: String,
    /// The layout it was lowered under (`Layout::key`).
    pub layout: String,
    /// `"{dtype}+lanes{n}+w{p}"`: the group whose `π` it is fitted with.
    pub group: String,
    pub cost: PlannedCost,
    /// Median time of a launch, in seconds.
    pub seconds: f64,
}

/// The layout cost model of plan 0.7.0 §3.5:
///
/// ```text
/// T̂ = λ + spawn·[p > 1] + max(W/π_g, Q_L2/β_L2, p·Q_L3/β_L3, p·Q_mem/β_mem)
/// ```
///
/// with `W` and `Q_c` one worker's work and traffic from each level
/// ([`PlannedCost`]), the shared levels seeing every worker's traffic, and `π`
/// fitted per group `g` = (machine type, lanes, workers) because the cores
/// are not equal and a per-core `π` would need a schedule. `u` is the median
/// absolute relative residual on the calibration set: a close pair whose
/// predicted ratio lies within `1 ± u` is one the model does not decide.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct LayoutFit {
    pub lambda: f64,
    pub spawn: f64,
    pub beta_l2: f64,
    pub beta_l3: f64,
    pub beta_mem: f64,
    pub pi: BTreeMap<String, f64>,
    pub u: f64,
    /// Spearman between predicted and measured times over the samples.
    pub spearman: f64,
    pub samples: Vec<PlannedSample>,
}

/// The group a kernel's `π` is fitted in.
pub fn group_key(dtype: &str, lanes: usize, workers: usize) -> String {
    format!("{dtype}+lanes{lanes}+w{workers}")
}

impl LayoutFit {
    /// Predicted seconds of a kernel of this planned cost, if its group was
    /// calibrated.
    pub fn predict(&self, c: &PlannedCost, group: &str) -> Option<f64> {
        let pi = *self.pi.get(group)?;
        Some(model(self.lambda, self.spawn, self.beta_l2, self.beta_l3, self.beta_mem, pi, c))
    }

    /// The terms of that prediction, for a reader asking which one decided
    /// it: `presk autotune` records them beside each point.
    pub fn terms(&self, c: &PlannedCost, group: &str) -> Option<Terms> {
        let pi = *self.pi.get(group)?;
        let p = c.workers as f64;
        Some(Terms {
            lambda: self.lambda,
            spawn: if c.workers > 1 { self.spawn } else { 0.0 },
            work: c.work / pi,
            l2: c.traffic[0] / self.beta_l2,
            l3: p * c.traffic[1] / self.beta_l3,
            mem: p * c.traffic[2] / self.beta_mem,
        })
    }
}

/// A prediction's terms in seconds: `λ + spawn + max(work, l2, l3, mem)`.
#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub struct Terms {
    pub lambda: f64,
    pub spawn: f64,
    pub work: f64,
    pub l2: f64,
    pub l3: f64,
    pub mem: f64,
}

impl Terms {
    pub fn total(&self) -> f64 {
        self.lambda + self.spawn + self.work.max(self.l2).max(self.l3).max(self.mem)
    }

    /// The bound the maximum takes: `work`, `l2`, `l3` or `mem`.
    pub fn bound(&self) -> &'static str {
        [("work", self.work), ("l2", self.l2), ("l3", self.l3), ("mem", self.mem)].into_iter().fold(("work", f64::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a }).0
    }
}

fn model(lambda: f64, spawn: f64, b2: f64, b3: f64, bm: f64, pi: f64, c: &PlannedCost) -> f64 {
    let p = c.workers as f64;
    let bound = (c.work / pi).max(c.traffic[0] / b2).max(p * c.traffic[1] / b3).max(p * c.traffic[2] / bm);
    lambda + if c.workers > 1 { spawn } else { 0.0 } + bound
}

/// Fits the layout cost model by coordinate descent on the squared relative
/// error: each constant in turn searched on a log grid around its value (a
/// grid holding zero too for `λ` and `spawn`), the grid narrowed each round,
/// until a round improves nothing.
pub fn fit_layout(samples: Vec<PlannedSample>) -> LayoutFit {
    let valid: Vec<&PlannedSample> = samples.iter().filter(|s| s.seconds > 0.0).collect();
    let groups: Vec<String> = {
        let mut g: Vec<String> = valid.iter().map(|s| s.group.clone()).collect();
        g.sort();
        g.dedup();
        g
    };
    // Parameters: λ, spawn, β_L2, β_L3, β_mem, then one π per group.
    let mut x: Vec<f64> = vec![0.0, 0.0];
    let max_rate = |f: &dyn Fn(&PlannedSample) -> f64| valid.iter().map(|s| f(s) / s.seconds).fold(1.0f64, f64::max);
    x.push(max_rate(&|s| s.cost.traffic[0]));
    x.push(max_rate(&|s| s.cost.workers as f64 * s.cost.traffic[1]));
    x.push(max_rate(&|s| s.cost.workers as f64 * s.cost.traffic[2]));
    for g in &groups {
        let r: Vec<f64> = valid.iter().filter(|s| &s.group == g).map(|s| s.cost.work / s.seconds).collect();
        x.push(r.iter().cloned().fold(1.0, f64::max));
    }
    let gi: Vec<usize> = valid.iter().map(|s| 5 + groups.iter().position(|g| g == &s.group).unwrap()).collect();
    let error = |x: &[f64]| -> f64 {
        valid.iter().zip(&gi).map(|(s, g)| {
            let r = model(x[0], x[1], x[2], x[3], x[4], x[*g], &s.cost) / s.seconds - 1.0;
            r * r
        }).sum()
    };
    let mut best = error(&x);
    let mut span = 1e3f64;
    for _round in 0..200 {
        let before = best;
        for i in 0..x.len() {
            let current = x[i];
            let centre = if current > 0.0 { current } else { 1e-6 };
            let mut candidates: Vec<f64> = (0..=40).map(|k| centre * span.powf(k as f64 / 20.0 - 1.0)).collect();
            if i < 2 {
                // λ and spawn: zero too, and nothing over a second.
                candidates.retain(|t| *t < 1.0);
                candidates.push(0.0);
            }
            let (mut kept, mut kept_err) = (current, best);
            for c in candidates {
                x[i] = c;
                let e = error(&x);
                if e < kept_err {
                    (kept, kept_err) = (c, e);
                }
            }
            x[i] = kept;
            best = kept_err;
        }
        // A round that improves nothing narrows the grid, down to 1 %.
        if before - best <= 1e-12 * before.max(1e-300) {
            if span <= 1.01 {
                break;
            }
            span = span.sqrt();
        }
    }
    let predicted: Vec<f64> = valid.iter().zip(&gi).map(|(s, g)| model(x[0], x[1], x[2], x[3], x[4], x[*g], &s.cost)).collect();
    let measured: Vec<f64> = valid.iter().map(|s| s.seconds).collect();
    let mut residuals: Vec<f64> = predicted.iter().zip(&measured).map(|(p, t)| (p / t - 1.0).abs()).collect();
    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let u = if residuals.is_empty() { 0.0 } else { residuals[residuals.len() / 2] };
    LayoutFit {
        lambda: x[0],
        spawn: x[1],
        beta_l2: x[2],
        beta_l3: x[3],
        beta_mem: x[4],
        pi: groups.iter().enumerate().map(|(k, g)| (g.clone(), x[5 + k])).collect(),
        u,
        spearman: spearman(&predicted, &measured),
        samples,
    }
}

/// One realization's fitted constants.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Fit {
    pub pi: f64,
    pub beta: f64,
    pub lambda: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Correlations {
    /// Operation counts by primitive (the cost without a profile).
    pub counts: f64,
    /// Work alone.
    pub work: f64,
    /// The fitted roofline.
    pub roofline: f64,
}

impl Profile {
    /// Predicted seconds of one kernel launch.
    pub fn predict(&self, work: f64, bytes: f64) -> f64 {
        self.lambda + (work / self.pi).max(bytes / self.beta)
    }

    /// The constants of a realization, its own or another's.
    pub fn fit_for(&self, label: &str) -> Fit {
        if label == self.dtype {
            return Fit { pi: self.pi, beta: self.beta, lambda: self.lambda };
        }
        self.others.get(label).copied().unwrap_or(Fit { pi: self.pi, beta: self.beta, lambda: self.lambda })
    }

    /// Predicted seconds for a realization the profile may have measured
    /// separately. Falls back to its own constants, which is the honest
    /// behaviour: a prediction for an unmeasured type is the wide type's and
    /// the caller should know it.
    pub fn predict_for(&self, label: &str, work: f64, bytes: f64) -> f64 {
        let f = self.fit_for(label);
        f.lambda + (work / f.pi).max(bytes / f.beta)
    }

    /// Bytes of one real in the profile's realization. This is the axis along
    /// which a narrower storage type buys anything at all: `Q` is counted in
    /// it, so halving it halves the traffic term of the roofline while the
    /// work term does not move (0.5.0 step 2). A mixed realization is priced
    /// by its default here; pricing it per site is the solver's business.
    pub fn real_bytes(&self) -> f64 {
        // The realization's label, whose first field is the default dtype
        // (`presk_rt::Realization::label`). Read as a string rather than
        // through `presk-rt`, which the optimizer does not otherwise need.
        match self.dtype.split('+').next().unwrap_or("") {
            "bf16" => 2.0,
            "f32" => 4.0,
            _ => 8.0,
        }
    }

    pub fn load(path: &std::path::Path) -> Result<Profile, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("{} is not a profile: {e}", path.display()))
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        std::fs::write(path, serde_json::to_string_pretty(self).unwrap()).map_err(|e| format!("cannot write {}: {e}", path.display()))
    }
}

/// Fit `λ + max(W/π, Q/β)` to the samples by squared relative error: a
/// log-spaced grid over `π` and `β`, refined twice around the best cell,
/// with `λ` in closed form for each cell (clipped at zero).
pub fn fit(samples: &[Sample], dtype: &str, machine: &str, counts: &dyn Fn(&Sample) -> f64) -> Profile {
    let valid: Vec<&Sample> = samples.iter().filter(|s| s.seconds > 0.0).collect();
    let lambda_for = |pi: f64, beta: f64| -> f64 {
        // minimize Σ ((λ + m_i − t_i)/t_i)²  ⇒  λ = Σ((t_i − m_i)/t_i²) / Σ(1/t_i²)
        let (mut num, mut den) = (0.0, 0.0);
        for s in &valid {
            let m = (s.work / pi).max(s.bytes / beta);
            num += (s.seconds - m) / (s.seconds * s.seconds);
            den += 1.0 / (s.seconds * s.seconds);
        }
        if den > 0.0 { (num / den).max(0.0) } else { 0.0 }
    };
    let error = |pi: f64, beta: f64, lambda: f64| -> f64 {
        valid.iter().map(|s| {
            let p = lambda + (s.work / pi).max(s.bytes / beta);
            let r = (p - s.seconds) / s.seconds;
            r * r
        }).sum::<f64>()
    };
    let (mut lo_pi, mut hi_pi, mut lo_beta, mut hi_beta) = (1e6f64, 1e13f64, 1e6f64, 1e13f64);
    let (mut best, mut best_err) = ((1e9, 1e9, 0.0), f64::INFINITY);
    for _round in 0..3 {
        let n = 60;
        for i in 0..=n {
            let pi = lo_pi * (hi_pi / lo_pi).powf(i as f64 / n as f64);
            for j in 0..=n {
                let beta = lo_beta * (hi_beta / lo_beta).powf(j as f64 / n as f64);
                let lambda = lambda_for(pi, beta);
                let e = error(pi, beta, lambda);
                if e < best_err {
                    best_err = e;
                    best = (pi, beta, lambda);
                }
            }
        }
        // Refine: a decade around the best on each axis.
        let step_pi = (hi_pi / lo_pi).powf(1.0 / n as f64);
        let step_beta = (hi_beta / lo_beta).powf(1.0 / n as f64);
        (lo_pi, hi_pi) = (best.0 / step_pi.powi(3), best.0 * step_pi.powi(3));
        (lo_beta, hi_beta) = (best.1 / step_beta.powi(3), best.1 * step_beta.powi(3));
    }
    let (pi, beta, lambda) = best;
    let measured: Vec<f64> = valid.iter().map(|s| s.seconds).collect();
    let spearman = Correlations {
        counts: spearman(&valid.iter().map(|s| counts(s)).collect::<Vec<_>>(), &measured),
        work: spearman(&valid.iter().map(|s| s.work).collect::<Vec<_>>(), &measured),
        roofline: spearman(&valid.iter().map(|s| lambda + (s.work / pi).max(s.bytes / beta)).collect::<Vec<_>>(), &measured),
    };
    Profile { version: 1, dtype: dtype.to_string(), machine: machine.to_string(), pi, beta, lambda, spearman, kernels: samples.to_vec(), others: BTreeMap::new(), machine_desc: None, layout: None }
}

/// Average ranks (ties share the mean rank).
fn ranks(xs: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..xs.len()).collect();
    idx.sort_by(|a, b| xs[*a].partial_cmp(&xs[*b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0; xs.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && xs[idx[j + 1]] == xs[idx[i]] {
            j += 1;
        }
        let mean = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            r[idx[k]] = mean;
        }
        i = j + 1;
    }
    r
}

/// Spearman's rank correlation of two series (0 when undefined).
pub fn spearman(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    let (rx, ry) = (ranks(x), ranks(y));
    let n = x.len() as f64;
    let (mx, my) = (rx.iter().sum::<f64>() / n, ry.iter().sum::<f64>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..x.len() {
        let (a, b) = (rx[i] - mx, ry[i] - my);
        sxy += a * b;
        sxx += a * a;
        syy += b * b;
    }
    if sxx == 0.0 || syy == 0.0 { 0.0 } else { sxy / (sxx * syy).sqrt() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_recovers_synthetic_constants() {
        // Three families so that every constant is identified: tiny kernels
        // (λ dominates), compute-bound ones (W/π) and memory-bound ones (Q/β).
        let (pi, beta, lambda) = (2.0e9, 8.0e9, 3.0e-6);
        let mut samples = Vec::new();
        for i in 0..36u32 {
            let (work, bytes) = match i % 3 {
                0 => (10.0 * (i + 1) as f64, 80.0 * (i + 1) as f64),
                1 => (1.0e5 * (1.4f64).powi((i / 3) as i32), 1.0e3),
                _ => (1.0e3, 1.0e5 * (1.4f64).powi((i / 3) as i32)),
            };
            let t = lambda + (work / pi).max(bytes / beta);
            let noise = 1.0 + 0.02 * ((i as f64 * 0.7).sin());
            samples.push(Sample { signature: format!("k{i}"), name: "add".into(), work, bytes, seconds: t * noise });
        }
        let p = fit(&samples, "f64", "test", &|_| 1.0);
        assert!((p.pi / pi - 1.0).abs() < 0.15, "pi {}", p.pi);
        assert!((p.beta / beta - 1.0).abs() < 0.15, "beta {}", p.beta);
        assert!((p.lambda / lambda - 1.0).abs() < 0.15, "lambda {}", p.lambda);
        for s in &samples {
            let pred = p.predict(s.work, s.bytes);
            assert!((pred / s.seconds - 1.0).abs() < 0.1, "{}: {pred} vs {}", s.signature, s.seconds);
        }
        assert!(p.spearman.roofline > 0.98, "{:?}", p.spearman);
    }

    #[test]
    fn the_layout_fit_recovers_synthetic_constants() {
        // Kernels bound by each term in turn — work, traffic from L2, L3 and
        // memory — over two groups and two worker counts, so that every
        // constant is identified.
        let (lambda, spawn, b2, b3, bm) = (4.0e-6, 2.0e-5, 1.6e11, 6.0e10, 2.0e10);
        let pi: BTreeMap<String, f64> = [("f32+lanes4+w1", 3.0e10), ("f32+lanes4+w10", 2.2e11), ("f32+lanes1+w1", 9.0e9)].iter().map(|(g, p)| (g.to_string(), *p)).collect();
        let mut samples = Vec::new();
        for (i, (g, _)) in pi.iter().enumerate() {
            let workers = if g.ends_with("w10") { 10 } else { 1 };
            for j in 0..24u32 {
                let scale = 1.5f64.powi((j / 4) as i32);
                let (work, traffic) = match j % 4 {
                    0 => (1.0e8 * scale, [1.0e5, 1.0e4, 1.0e3]),
                    1 => (1.0e5, [5.0e7 * scale, 1.0e4, 1.0e3]),
                    2 => (1.0e5, [1.0e5, 2.0e7 * scale / workers as f64, 1.0e3]),
                    _ => (1.0e5, [1.0e5, 1.0e4, 8.0e6 * scale / workers as f64]),
                };
                let cost = PlannedCost { work, bytes: 0.0, traffic, regs: 0, spilled: 0, workers, depth: 0 };
                let t = model(lambda, spawn, b2, b3, bm, pi[g], &cost) * (1.0 + 0.01 * ((i as f64 * 7.0 + j as f64) * 0.9).sin());
                samples.push(PlannedSample { signature: format!("k{i}{j}"), layout: String::new(), group: g.clone(), cost, seconds: t });
            }
        }
        let f = fit_layout(samples);
        let close = |a: f64, b: f64| (a / b - 1.0).abs() < 0.1;
        assert!(close(f.beta_l2, b2) && close(f.beta_l3, b3) && close(f.beta_mem, bm), "{f:?}");
        for (g, p) in &pi {
            assert!(close(f.pi[g], *p), "{g}: {} against {p}", f.pi[g]);
        }
        assert!(f.u < 0.02 && f.spearman > 0.99, "u {} spearman {}", f.u, f.spearman);
        // The terms add up to the prediction, and name the bound that set it.
        for (s, bound) in f.samples.iter().zip(["work", "l2", "l3", "mem"].iter().cycle()) {
            let t = f.terms(&s.cost, &s.group).unwrap();
            assert_eq!(t.total(), f.predict(&s.cost, &s.group).unwrap());
            assert_eq!(t.bound(), *bound, "{}", s.signature);
        }
    }

    #[test]
    fn spearman_of_monotone_series_is_one() {
        assert!((spearman(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 25.0, 100.0]) - 1.0).abs() < 1e-12);
        assert!((spearman(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]) + 1.0).abs() < 1e-12);
        assert_eq!(ranks(&[2.0, 1.0, 2.0]), vec![2.5, 1.0, 2.5]);
    }
}
