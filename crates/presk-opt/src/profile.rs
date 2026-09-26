//! The machine profile: the roofline `T̂ = λ + max(W/π, Q/β)` of the
//! cost-model document (§1.1) with its three constants *fitted* on measured
//! kernel times (`presk bench --calibrate`) rather than declared — `π` the
//! sustained operation rate, `β` the sustained bandwidth, `λ` the launch
//! overhead. The fit minimizes the squared *relative* error, so that the
//! microsecond kernels of the examples weigh as much as the millisecond
//! ones; the quality metric is Spearman's rank correlation between predicted
//! and measured times, the roadmap's evaluation metric for cost models.

use std::collections::BTreeMap;

use presk_kernel::cost::{CostRules, Machine, PlannedCost};
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

/// The layout cost model of plan 0.7.0 §3.5, with plan 0.8.0's load term
/// (§3.1, step 2) and its combinations (§3.2, step 3):
///
/// ```text
/// T̂ = λ + spawn·[p > 1] + C(W/π_g, (N_ld + S)/ρ_g, Q_L2/β_L2, p·Q_L3/β_L3, p·Q_mem/β_mem)
/// ```
///
/// where `C` is the maximum, 0.7.0's, the sum of the in-core and the memory
/// terms, or a p-norm ([`Combine`]).
///
/// with `W`, `N_ld` and `Q_c` one worker's work, register loads and stores,
/// and traffic from each level ([`PlannedCost`]), the shared levels seeing
/// every worker's traffic, and `π` and `ρ` fitted per group `g` = (machine
/// type, lanes, workers) because the cores are not equal and a per-core rate
/// would need a schedule. `S` is the spilled registers' accesses. A model
/// without `ρ` is 0.7.0's: no load term, and `S` charged as work,
/// `(W + S)/π_g`, which is 0.7.0's prediction to the last bit. `u` is the
/// median absolute relative residual on the calibration set: a close pair
/// whose predicted ratio lies within `1 ± u` is one the model does not
/// decide.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct LayoutFit {
    pub lambda: f64,
    pub spawn: f64,
    pub beta_l2: f64,
    pub beta_l3: f64,
    pub beta_mem: f64,
    pub pi: BTreeMap<String, f64>,
    /// Register loads and stores per second, per group (0.8.0 step 2);
    /// empty in a model without the load term, 0.7.0's among them.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub rho: BTreeMap<String, f64>,
    /// How the terms combine (0.8.0 step 3); the maximum in a profile
    /// written before.
    #[serde(default, skip_serializing_if = "Combine::is_max")]
    pub combine: Combine,
    /// The rules of the planned cost its samples were costed under, and its
    /// predictions are (0.8.0 step 4); none in a profile written before.
    #[serde(default, skip_serializing_if = "CostRules::is_none")]
    pub rules: CostRules,
    pub u: f64,
    /// Spearman between predicted and measured times over the samples.
    pub spearman: f64,
    pub samples: Vec<PlannedSample>,
}

/// How the layout model's terms combine (plan 0.8.0 §3.2). `λ` and the
/// spawn cost are added in every form.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Combine {
    /// The largest term, 0.7.0's form: a core that overlaps everything, so
    /// that only the slowest resource counts.
    #[default]
    Max,
    /// The in-core term plus the memory term, `max(work, loads) + max(l2,
    /// l3, mem)`: a core that does not overlap its arithmetic with its
    /// misses.
    Sum,
    /// `(Σ tᵢ^q)^(1/q)` over the five terms, `q ≥ 1` fitted with the other
    /// constants: `q → ∞` is the maximum, `q = 1` the sum of all five. A fit
    /// searches `q` from 1 to 100, whatever it is given.
    Pnorm(f64),
}

impl Combine {
    pub fn is_max(&self) -> bool {
        *self == Combine::Max
    }

    /// `max`, `sum` or `pnorm` (its `q` to be fitted).
    pub fn parse(s: &str) -> Result<Combine, String> {
        match s {
            "max" => Ok(Combine::Max),
            "sum" => Ok(Combine::Sum),
            "pnorm" => Ok(Combine::Pnorm(4.0)),
            other => Err(format!("--combine {other}: `max`, `sum` or `pnorm`")),
        }
    }

    pub fn name(&self) -> String {
        match self {
            Combine::Max => "max".into(),
            Combine::Sum => "sum".into(),
            Combine::Pnorm(q) => format!("p-norm, q = {q:.2}"),
        }
    }

    /// The five terms combined, `loads` absent from a model without the
    /// load term. The maximum is computed as 0.7.0 computed it.
    fn apply(&self, work: f64, loads: Option<f64>, l2: f64, l3: f64, mem: f64) -> f64 {
        let core = match loads {
            Some(l) => work.max(l),
            None => work,
        };
        match self {
            Combine::Max => core.max(l2).max(l3).max(mem),
            Combine::Sum => core + l2.max(l3).max(mem),
            Combine::Pnorm(q) => {
                // Scaled by the largest term, so that no power under- or
                // overflows.
                let terms = [Some(work), loads, Some(l2), Some(l3), Some(mem)];
                let m = terms.iter().flatten().fold(0.0f64, |a, t| a.max(*t));
                if m == 0.0 { 0.0 } else { m * terms.iter().flatten().map(|t| (t / m).powf(*q)).sum::<f64>().powf(1.0 / q) }
            }
        }
    }
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
        Some(model(self.lambda, self.spawn, self.beta_l2, self.beta_l3, self.beta_mem, pi, self.rho(group)?, self.combine, c))
    }

    /// The group's load rate: `Some(None)` for a model without the load
    /// term, `None` for a group it has no rate for.
    fn rho(&self, group: &str) -> Option<Option<f64>> {
        if self.rho.is_empty() { Some(None) } else { self.rho.get(group).map(|r| Some(*r)) }
    }

    /// The terms of that prediction, for a reader asking which one decided
    /// it: `presk autotune` records them beside each point.
    pub fn terms(&self, c: &PlannedCost, group: &str) -> Option<Terms> {
        let pi = *self.pi.get(group)?;
        let rho = self.rho(group)?;
        let p = c.workers as f64;
        Some(Terms {
            lambda: self.lambda,
            spawn: if c.workers > 1 { self.spawn } else { 0.0 },
            work: work_term(pi, rho, c),
            loads: rho.map(|r| (c.loads + c.spill) / r),
            l2: c.traffic[0] / self.beta_l2,
            l3: p * c.traffic[1] / self.beta_l3,
            mem: p * c.traffic[2] / self.beta_mem,
            combine: self.combine,
        })
    }
}

/// A prediction's terms in seconds: `λ + spawn + C(work, loads, l2, l3,
/// mem)`, `loads` absent from a model without the load term, `C` the
/// model's combination (recorded when it is not the maximum).
#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub struct Terms {
    pub lambda: f64,
    pub spawn: f64,
    pub work: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loads: Option<f64>,
    pub l2: f64,
    pub l3: f64,
    pub mem: f64,
    #[serde(skip_serializing_if = "Combine::is_max")]
    pub combine: Combine,
}

impl Terms {
    pub fn total(&self) -> f64 {
        self.lambda + self.spawn + self.combine.apply(self.work, self.loads, self.l2, self.l3, self.mem)
    }

    /// The largest term: the one the maximum takes, and the one that weighs
    /// most in a sum or a p-norm. `work`, `loads`, `l2`, `l3` or `mem`.
    pub fn bound(&self) -> &'static str {
        let loads = self.loads.map(|l| ("loads", l));
        [("work", self.work)].into_iter().chain(loads).chain([("l2", self.l2), ("l3", self.l3), ("mem", self.mem)]).fold(("work", f64::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a }).0
    }
}

/// The work term: `W/π`, or 0.7.0's `(W + S)/π` in a model without the load
/// term, the spilled registers' accesses counted as operations. `W + S` is
/// the sum 0.7.0 formed, in the same order, so its predictions come back to
/// the last bit; a sample recorded before 0.8.0 step 2 has `S` zero and
/// holds it in `W`.
fn work_term(pi: f64, rho: Option<f64>, c: &PlannedCost) -> f64 {
    match rho {
        Some(_) => c.work / pi,
        None => (c.work + c.spill) / pi,
    }
}

#[allow(clippy::too_many_arguments)]
fn model(lambda: f64, spawn: f64, b2: f64, b3: f64, bm: f64, pi: f64, rho: Option<f64>, combine: Combine, c: &PlannedCost) -> f64 {
    let p = c.workers as f64;
    let bound = combine.apply(work_term(pi, rho, c), rho.map(|r| (c.loads + c.spill) / r), c.traffic[0] / b2, p * c.traffic[1] / b3, p * c.traffic[2] / bm);
    lambda + if c.workers > 1 { spawn } else { 0.0 } + bound
}

/// Whether a layout model has the load term, and how its rate is fitted
/// (plan 0.8.0 §3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadTerm {
    /// No load term: 0.7.0's model, the spilled registers charged as work.
    None,
    /// One `ρ` per group, as `π` is.
    Group,
    /// One `ρ` for every group, for a calibration that cannot tell the
    /// groups' load rates apart.
    Machine,
}

impl LoadTerm {
    pub fn parse(s: &str) -> Result<LoadTerm, String> {
        match s {
            "none" => Ok(LoadTerm::None),
            "group" => Ok(LoadTerm::Group),
            "machine" => Ok(LoadTerm::Machine),
            other => Err(format!("--load-term {other}: `none`, `group` or `machine`")),
        }
    }
}

/// Fits the layout cost model by coordinate descent on the squared relative
/// error: each constant in turn searched on a log grid around its value (a
/// grid holding zero too for `λ` and `spawn`), the grid narrowed each round
/// that improves the error by less than a millionth, until it is 1 % wide. `loads` says whether the model has the
/// load term and how many rates it fits, `combine` how the terms combine; a
/// p-norm's `q` is fitted too, from 1 to 100, whatever `combine` gives.
pub fn fit_layout(samples: Vec<PlannedSample>, loads: LoadTerm, combine: Combine) -> LayoutFit {
    let (mut fit, _) = match combine {
        Combine::Pnorm(_) => {
            // `q` trades off against every rate at once: a larger `q` lowers
            // the combination, and lower rates make it up. A coordinate at a
            // time cannot follow that, so `q` is searched on its own, the
            // rates fitted again at every `q` tried: ten values on log q from
            // 1 to 100, then ten between the best one's neighbours. The fits
            // of a stage are independent and run on as many threads, from
            // the maximum's fit, which is quick to make from nothing.
            let (start, _) = fit_fixed(&samples, loads, Combine::Max, None);
            let stage = |qs: &[f64]| -> Vec<(LayoutFit, f64)> {
                std::thread::scope(|sc| {
                    let runs: Vec<_> = qs.iter().map(|q| sc.spawn(|| fit_fixed(&samples, loads, Combine::Pnorm(*q), Some(&start)))).collect();
                    runs.into_iter().map(|h| h.join().expect("a fit does not panic")).collect()
                })
            };
            let log_grid = |lo: f64, hi: f64| -> Vec<f64> { (0..10).map(|k| (lo.ln() + (hi.ln() - lo.ln()) * k as f64 / 9.0).exp()).collect() };
            let best_of = |qs: &[f64], fits: Vec<(LayoutFit, f64)>| -> (usize, (LayoutFit, f64)) {
                let k = (0..qs.len()).min_by(|a, b| fits[*a].1.total_cmp(&fits[*b].1)).unwrap();
                (k, fits.into_iter().nth(k).unwrap())
            };
            let coarse = log_grid(1.0, 100.0);
            let (k, best) = best_of(&coarse, stage(&coarse));
            let fine = log_grid(coarse[k.saturating_sub(1)], coarse[(k + 1).min(coarse.len() - 1)]);
            let (_, refined) = best_of(&fine, stage(&fine));
            if refined.1 < best.1 { refined } else { best }
        }
        _ => fit_fixed(&samples, loads, combine, None),
    };
    fit.samples = samples;
    fit
}

/// [`fit_layout`] with the combination fixed, a p-norm's `q` included: the
/// fit, its samples left out, and its squared relative error. From `start`,
/// the search begins at its constants on a grid ten times either way,
/// instead of a thousand from the largest rate each sample shows.
fn fit_fixed(samples: &[PlannedSample], loads: LoadTerm, combine: Combine, start: Option<&LayoutFit>) -> (LayoutFit, f64) {
    let valid: Vec<&PlannedSample> = samples.iter().filter(|s| s.seconds > 0.0).collect();
    let groups: Vec<String> = {
        let mut g: Vec<String> = valid.iter().map(|s| s.group.clone()).collect();
        g.sort();
        g.dedup();
        g
    };
    // Parameters: λ, spawn, β_L2, β_L3, β_mem, one π per group, then the
    // load rates: one per group, one, or none.
    let mut x: Vec<f64> = vec![0.0, 0.0];
    let max_rate = |f: &dyn Fn(&PlannedSample) -> f64| valid.iter().map(|s| f(s) / s.seconds).fold(1.0f64, f64::max);
    x.push(max_rate(&|s| s.cost.traffic[0]));
    x.push(max_rate(&|s| s.cost.workers as f64 * s.cost.traffic[1]));
    x.push(max_rate(&|s| s.cost.workers as f64 * s.cost.traffic[2]));
    for g in &groups {
        let r: Vec<f64> = valid.iter().filter(|s| &s.group == g).map(|s| s.cost.work / s.seconds).collect();
        x.push(r.iter().cloned().fold(1.0, f64::max));
    }
    let n = groups.len();
    let load_rate = |f: &dyn Fn(&PlannedSample) -> bool| valid.iter().filter(|s| f(s)).map(|s| (s.cost.loads + s.cost.spill) / s.seconds).fold(1.0f64, f64::max);
    match loads {
        LoadTerm::None => {}
        LoadTerm::Group => groups.iter().for_each(|g| x.push(load_rate(&|s| &s.group == g))),
        LoadTerm::Machine => x.push(load_rate(&|_| true)),
    }
    let gi: Vec<usize> = valid.iter().map(|s| 5 + groups.iter().position(|g| g == &s.group).unwrap()).collect();
    let ri: Vec<Option<usize>> = gi.iter().map(|g| match loads {
        LoadTerm::None => None,
        LoadTerm::Group => Some(g + n),
        LoadTerm::Machine => Some(5 + n),
    }).collect();
    let error = |x: &[f64]| -> f64 {
        valid.iter().zip(&gi).zip(&ri).map(|((s, g), r)| {
            let r = model(x[0], x[1], x[2], x[3], x[4], x[*g], r.map(|r| x[r]), combine, &s.cost) / s.seconds - 1.0;
            r * r
        }).sum()
    };
    let mut span = 1e3f64;
    if let Some(f) = start {
        x[0] = f.lambda;
        x[1] = f.spawn;
        (x[2], x[3], x[4]) = (f.beta_l2, f.beta_l3, f.beta_mem);
        for (k, g) in groups.iter().enumerate() {
            if let Some(p) = f.pi.get(g) {
                x[5 + k] = *p;
            }
            match loads {
                LoadTerm::Group => {
                    if let Some(r) = f.rho.get(g) {
                        x[5 + n + k] = *r;
                    }
                }
                LoadTerm::Machine => {
                    if let Some(r) = f.rho.get(g) {
                        x[5 + n] = *r;
                    }
                }
                LoadTerm::None => {}
            }
        }
        span = 10.0;
    }
    let mut best = error(&x);
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
        // A round that improves the error by less than a millionth narrows
        // the grid, down to 1 %. (Until 0.8.0 step 3, by nothing: in the flat
        // valley along which a p-norm's rates trade off, a round always
        // improves a little, and every fit ran its 200 rounds.)
        if before - best <= 1e-6 * before.max(1e-300) {
            if span <= 1.01 {
                break;
            }
            span = span.sqrt();
        }
    }
    let predicted: Vec<f64> = valid.iter().zip(&gi).zip(&ri).map(|((s, g), r)| model(x[0], x[1], x[2], x[3], x[4], x[*g], r.map(|r| x[r]), combine, &s.cost)).collect();
    let measured: Vec<f64> = valid.iter().map(|s| s.seconds).collect();
    let mut residuals: Vec<f64> = predicted.iter().zip(&measured).map(|(p, t)| (p / t - 1.0).abs()).collect();
    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let u = if residuals.is_empty() { 0.0 } else { residuals[residuals.len() / 2] };
    let fit = LayoutFit {
        lambda: x[0],
        spawn: x[1],
        beta_l2: x[2],
        beta_l3: x[3],
        beta_mem: x[4],
        pi: groups.iter().enumerate().map(|(k, g)| (g.clone(), x[5 + k])).collect(),
        rho: groups.iter().enumerate().filter_map(|(k, g)| match loads {
            LoadTerm::None => None,
            LoadTerm::Group => Some((g.clone(), x[5 + n + k])),
            LoadTerm::Machine => Some((g.clone(), x[5 + n])),
        }).collect(),
        combine,
        rules: CostRules::default(),
        u,
        spearman: spearman(&predicted, &measured),
        samples: vec![],
    };
    (fit, best)
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
                let cost = PlannedCost { work, bytes: 0.0, traffic, regs: 0, spilled: 0, loads: 0.0, spill: 0.0, workers, depth: 0 };
                let t = model(lambda, spawn, b2, b3, bm, pi[g], None, Combine::Max, &cost) * (1.0 + 0.01 * ((i as f64 * 7.0 + j as f64) * 0.9).sin());
                samples.push(PlannedSample { signature: format!("k{i}{j}"), layout: String::new(), group: g.clone(), cost, seconds: t });
            }
        }
        let f = fit_layout(samples, LoadTerm::None, Combine::Max);
        let close = |a: f64, b: f64| (a / b - 1.0).abs() < 0.1;
        assert!(f.rho.is_empty());
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

    /// Plan 0.8.0 step 2's check: kernels bound by each term in turn, the
    /// load term among them, recover every constant, `ρ` per group within
    /// 10 %; and one `ρ` for the machine, where the groups share it.
    #[test]
    fn the_layout_fit_recovers_a_load_rate() {
        let (lambda, spawn, b2, b3, bm) = (4.0e-6, 2.0e-5, 1.6e11, 6.0e10, 2.0e10);
        let pi: BTreeMap<String, f64> = [("f32+lanes4+w1", 3.0e10), ("f32+lanes4+w10", 2.2e11), ("f32+lanes1+w1", 9.0e9)].iter().map(|(g, p)| (g.to_string(), *p)).collect();
        let rho: BTreeMap<String, f64> = [("f32+lanes4+w1", 5.0e9), ("f32+lanes4+w10", 4.0e10), ("f32+lanes1+w1", 3.5e9)].iter().map(|(g, r)| (g.to_string(), *r)).collect();
        let samples = |rho: &BTreeMap<String, f64>| {
            let mut samples = Vec::new();
            for (i, (g, _)) in pi.iter().enumerate() {
                let workers = if g.ends_with("w10") { 10 } else { 1 };
                for j in 0..30u32 {
                    let scale = 1.5f64.powi((j / 5) as i32);
                    // Work-bound, load-bound (a spill among its loads), then
                    // bound by the traffic from each level.
                    let (work, loads, spill, traffic) = match j % 5 {
                        0 => (1.0e8 * scale, 1.0e5, 0.0, [1.0e5, 1.0e4, 1.0e3]),
                        1 => (1.0e5, 2.0e7 * scale, 1.0e6 * scale, [1.0e5, 1.0e4, 1.0e3]),
                        2 => (1.0e5, 1.0e3, 0.0, [5.0e7 * scale, 1.0e4, 1.0e3]),
                        3 => (1.0e5, 1.0e3, 0.0, [1.0e5, 2.0e7 * scale / workers as f64, 1.0e3]),
                        _ => (1.0e5, 1.0e3, 0.0, [1.0e5, 1.0e4, 8.0e6 * scale / workers as f64]),
                    };
                    let cost = PlannedCost { work, bytes: 0.0, traffic, regs: 0, spilled: 0, loads, spill, workers, depth: 0 };
                    let t = model(lambda, spawn, b2, b3, bm, pi[g], Some(rho[g]), Combine::Max, &cost) * (1.0 + 0.01 * ((i as f64 * 7.0 + j as f64) * 0.9).sin());
                    samples.push(PlannedSample { signature: format!("k{i}{j}"), layout: String::new(), group: g.clone(), cost, seconds: t });
                }
            }
            samples
        };
        let close = |a: f64, b: f64| (a / b - 1.0).abs() < 0.1;
        let f = fit_layout(samples(&rho), LoadTerm::Group, Combine::Max);
        assert!(close(f.beta_l2, b2) && close(f.beta_l3, b3) && close(f.beta_mem, bm), "{f:?}");
        for (g, r) in &rho {
            assert!(close(f.pi[g], pi[g]), "{g}: π {} against {}", f.pi[g], pi[g]);
            assert!(close(f.rho[g], *r), "{g}: ρ {} against {r}", f.rho[g]);
        }
        assert!(f.u < 0.02 && f.spearman > 0.99, "u {} spearman {}", f.u, f.spearman);
        for (s, bound) in f.samples.iter().zip(["work", "loads", "l2", "l3", "mem"].iter().cycle()) {
            let t = f.terms(&s.cost, &s.group).unwrap();
            assert_eq!(t.total(), f.predict(&s.cost, &s.group).unwrap());
            assert_eq!(t.bound(), *bound, "{}", s.signature);
        }
        // One rate shared by every group.
        let shared: BTreeMap<String, f64> = rho.keys().map(|g| (g.clone(), 8.0e9)).collect();
        let f = fit_layout(samples(&shared), LoadTerm::Machine, Combine::Max);
        assert!(f.rho.values().all(|r| close(*r, 8.0e9)) && f.rho.len() == 3, "{:?}", f.rho);
    }

    /// A model without `ρ` is 0.7.0's: the spilled registers' accesses
    /// charged as work, the prediction to the last bit, on a sample recorded
    /// now (`W` and `S` apart) and on one 0.7.0 recorded (`S` inside `W`).
    #[test]
    fn a_model_without_a_load_rate_predicts_as_0_7_0() {
        let f = LayoutFit { lambda: 3.0e-6, spawn: 1.5e-5, beta_l2: 1.1e11, beta_l3: 7.0e10, beta_mem: 2.5e10, pi: [("f32+lanes1+w1".to_string(), 9.7e9)].into_iter().collect(), ..LayoutFit::default() };
        let (work, spill) = (3.3e7, 1.7e6);
        let now = PlannedCost { work, bytes: 0.0, traffic: [1.0e6, 1.0e5, 1.0e4], regs: 40, spilled: 8, loads: 9.0e6, spill, workers: 1, depth: 0 };
        let recorded: PlannedCost = serde_json::from_str(&format!(r#"{{ "work": {}, "bytes": 0.0, "traffic": [1.0e6, 1.0e5, 1.0e4], "regs": 40, "spilled": 8, "workers": 1, "depth": 0 }}"#, work + spill)).unwrap();
        assert_eq!((recorded.loads, recorded.spill), (0.0, 0.0));
        let v070 = |w: f64| 3.0e-6 + (w / 9.7e9).max(1.0e6 / 1.1e11).max(1.0e5 / 7.0e10).max(1.0e4 / 2.5e10);
        assert_eq!(f.predict(&now, "f32+lanes1+w1"), Some(v070(work + spill)));
        assert_eq!(f.predict(&recorded, "f32+lanes1+w1"), Some(v070(work + spill)));
        let t = f.terms(&now, "f32+lanes1+w1").unwrap();
        assert_eq!((t.loads, t.bound()), (None, "work"));
        assert!(!serde_json::to_string(&t).unwrap().contains("loads"), "0.7.0's terms, as recorded");
    }

    /// Plan 0.8.0 step 3's check: the sum and the p-norm are each recovered
    /// from samples they generated — every rate within 10 %, and `q`. The
    /// samples draw each term's seconds over two decades, so that the terms
    /// are often comparable, which is what identifies a sum or a `q`.
    #[test]
    fn each_combination_is_recovered() {
        let (lambda, spawn, b2, b3, bm) = (4.0e-6, 2.0e-5, 1.6e11, 6.0e10, 2.0e10);
        let pi: BTreeMap<String, f64> = [("f32+lanes4+w1", 3.0e10), ("f32+lanes4+w10", 2.2e11), ("f32+lanes1+w1", 9.0e9)].iter().map(|(g, p)| (g.to_string(), *p)).collect();
        let rho: BTreeMap<String, f64> = [("f32+lanes4+w1", 5.0e9), ("f32+lanes4+w10", 4.0e10), ("f32+lanes1+w1", 3.5e9)].iter().map(|(g, r)| (g.to_string(), *r)).collect();
        for truth in [Combine::Sum, Combine::Pnorm(3.0)] {
            let mut samples = Vec::new();
            for (i, g) in pi.keys().enumerate() {
                let workers = if g.ends_with("w10") { 10 } else { 1 };
                let p = workers as f64;
                for j in 0..80u32 {
                    // Each term's seconds, 10⁻⁴ to 10⁻², drawn apart from the
                    // others by a fixed hash (splitmix64).
                    let t = |k: u64| {
                        let mut z = ((i as u64) << 32 | (j as u64) << 8 | k).wrapping_add(0x9e37_79b9_7f4a_7c15);
                        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                        1.0e-4 * 100f64.powf(((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64)
                    };
                    let cost = PlannedCost { work: t(1) * pi[g], bytes: 0.0, traffic: [t(3) * b2, t(4) * b3 / p, t(5) * bm / p], regs: 0, spilled: 0, loads: t(2) * rho[g], spill: 0.0, workers, depth: 0 };
                    let seconds = model(lambda, spawn, b2, b3, bm, pi[g], Some(rho[g]), truth, &cost) * (1.0 + 0.005 * ((i as f64 * 7.0 + j as f64) * 0.9).sin());
                    samples.push(PlannedSample { signature: format!("k{i}{j}"), layout: String::new(), group: g.clone(), cost, seconds });
                }
            }
            let start = if truth == Combine::Sum { Combine::Sum } else { Combine::Pnorm(1.0) };
            let f = fit_layout(samples, LoadTerm::Group, start);
            let close = |a: f64, b: f64| (a / b - 1.0).abs() < 0.1;
            assert!(close(f.beta_l2, b2) && close(f.beta_l3, b3) && close(f.beta_mem, bm), "{truth:?}: {f:?}");
            for g in pi.keys() {
                assert!(close(f.pi[g], pi[g]) && close(f.rho[g], rho[g]), "{truth:?} {g}: π {} ρ {}", f.pi[g], f.rho[g]);
            }
            match (truth, f.combine) {
                (Combine::Sum, Combine::Sum) => {}
                (Combine::Pnorm(q), Combine::Pnorm(fq)) => assert!(close(fq, q), "q {fq} against {q}, u {}", f.u),
                (t, c) => panic!("{t:?} fitted as {c:?}"),
            }
            assert!(f.u < 0.02, "{truth:?}: u {}", f.u);
            for s in f.samples.iter().take(10) {
                assert_eq!(f.terms(&s.cost, &s.group).unwrap().total(), f.predict(&s.cost, &s.group).unwrap());
            }
        }
    }

    #[test]
    fn spearman_of_monotone_series_is_one() {
        assert!((spearman(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 25.0, 100.0]) - 1.0).abs() < 1e-12);
        assert!((spearman(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]) + 1.0).abs() < 1e-12);
        assert_eq!(ranks(&[2.0, 1.0, 2.0]), vec![2.5, 1.0, 2.5]);
    }
}
