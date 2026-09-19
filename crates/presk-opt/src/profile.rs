//! The machine profile: the roofline `T̂ = λ + max(W/π, Q/β)` of the
//! cost-model document (§1.1) with its three constants *fitted* on measured
//! kernel times (`presk bench --calibrate`) rather than declared — `π` the
//! sustained operation rate, `β` the sustained bandwidth, `λ` the launch
//! overhead. The fit minimizes the squared *relative* error, so that the
//! microsecond kernels of the examples weigh as much as the millisecond
//! ones; the quality metric is Spearman's rank correlation between predicted
//! and measured times, the roadmap's evaluation metric for cost models.

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

    /// Bytes of one real in the profile's realization.
    pub fn real_bytes(&self) -> f64 {
        if self.dtype == "f32" { 4.0 } else { 8.0 }
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
    Profile { version: 1, dtype: dtype.to_string(), machine: machine.to_string(), pi, beta, lambda, spearman, kernels: samples.to_vec() }
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
    fn spearman_of_monotone_series_is_one() {
        assert!((spearman(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 25.0, 100.0]) - 1.0).abs() < 1e-12);
        assert!((spearman(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]) + 1.0).abs() < 1e-12);
        assert_eq!(ranks(&[2.0, 1.0, 2.0]), vec![2.5, 1.0, 2.5]);
    }
}
