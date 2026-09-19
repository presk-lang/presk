//! `presk-opt`: e-graph rewriting of PSK-Core programs (0.3.0 plan, step 6).
//!
//! Every function body becomes an e-graph (crate `egg`), saturates under a
//! set of rules that are exact over ℝ — algebra, reductions, and the
//! structural identities that clean the adjoints of automatic
//! differentiation — and the cheapest equivalent term is extracted and
//! written back in ANF. The cost is operation counts until the calibrated
//! model of step 7. The reference interpreter is the oracle: the rewritten
//! program must reproduce its goldens (tests of this crate).

pub mod analysis;
pub mod cost;
pub mod graph;
pub mod lang;
pub mod model;
pub mod profile;
pub mod rules;

use std::collections::BTreeMap;

use presk_core::Program;
use presk_types::dim::DimVar;
use presk_types::elab::ElabProgram;

pub use profile::Profile;

/// Bounds on saturation, per function.
#[derive(Clone, Debug)]
pub struct Limits {
    pub iterations: usize,
    pub nodes: usize,
    pub time_ms: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Limits { iterations: 30, nodes: 20_000, time_ms: 2_000 }
    }
}

/// How to optimize: the saturation limits, the cost (operation counts, or
/// the roofline of a calibrated profile) and, for the roofline, the
/// dimensions the run binds by name (the rest default to `default_dim`).
#[derive(Clone, Debug, Default)]
pub struct Options {
    pub limits: Limits,
    pub profile: Option<Profile>,
    pub dims: BTreeMap<String, u64>,
    pub default_dim: Option<u64>,
}

/// What happened to one function.
#[derive(Clone, Debug)]
pub enum Outcome {
    Rewritten { lets_before: usize, lets_after: usize, cost_before: f64, cost_after: f64, stop: String, iterations: usize, classes: usize, nodes: usize },
    /// Left as it was.
    Skipped(String),
}

#[derive(Clone, Debug)]
pub struct FuncReport {
    pub name: String,
    pub outcome: Outcome,
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub funcs: Vec<FuncReport>,
    /// `operation counts` or the profile's machine.
    pub model: String,
}

impl Report {
    pub fn lets_before(&self) -> usize {
        self.funcs.iter().map(|f| match &f.outcome { Outcome::Rewritten { lets_before, .. } => *lets_before, _ => 0 }).sum()
    }

    pub fn lets_after(&self) -> usize {
        self.funcs.iter().map(|f| match &f.outcome { Outcome::Rewritten { lets_after, .. } => *lets_after, _ => 0 }).sum()
    }

    pub fn cost_before(&self) -> f64 {
        self.funcs.iter().map(|f| match &f.outcome { Outcome::Rewritten { cost_before, .. } => *cost_before, _ => 0.0 }).sum()
    }

    pub fn cost_after(&self) -> f64 {
        self.funcs.iter().map(|f| match &f.outcome { Outcome::Rewritten { cost_after, .. } => *cost_after, _ => 0.0 }).sum()
    }

    /// Counts as numbers, seconds as microseconds.
    pub fn fmt_cost(&self, c: f64) -> String {
        if self.model == "operation counts" { format!("{c:.2}") } else { format!("{:.1} µs", c * 1e6) }
    }

    /// Functions whose saturation stopped on a limit.
    pub fn limited(&self) -> Vec<&FuncReport> {
        self.funcs.iter().filter(|f| matches!(&f.outcome, Outcome::Rewritten { stop, .. } if stop != "saturated")).collect()
    }

    /// One line per function that changed, and the totals.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        let mut changed = 0;
        for f in &self.funcs {
            match &f.outcome {
                Outcome::Rewritten { lets_before, lets_after, cost_before, cost_after, stop, iterations, .. } if lets_before != lets_after || cost_before != cost_after => {
                    changed += 1;
                    s.push_str(&format!("  {}: {} → {} lets, cost {} → {} ({stop}, {iterations} iterations)\n", f.name, lets_before, lets_after, self.fmt_cost(*cost_before), self.fmt_cost(*cost_after)));
                }
                Outcome::Rewritten { .. } => {}
                Outcome::Skipped(why) => s.push_str(&format!("  {}: left unchanged ({why})\n", f.name)),
            }
        }
        let limited = self.limited().len();
        let head = format!(
            "presk-opt: {} of {} functions rewritten, {} → {} lets, cost {} → {} ({}){}\n",
            changed,
            self.funcs.len(),
            self.lets_before(),
            self.lets_after(),
            self.fmt_cost(self.cost_before()),
            self.fmt_cost(self.cost_after()),
            self.model,
            if limited > 0 { format!(", {limited} stopped on a limit (extracted from the partial e-graph)") } else { String::new() }
        );
        head + &s
    }
}

/// Rewrite every function of `core` in place. A function whose rewritten
/// body does not verify is left as it was (and reported).
pub fn optimize(core: &mut Program, elab: &ElabProgram, opts: &Options) -> Report {
    let mut report = Report { funcs: vec![], model: match &opts.profile { Some(p) => format!("roofline, profile {} {}", p.machine, p.dtype), None => "operation counts".to_string() } };
    let modules: analysis::Modules = std::rc::Rc::new(elab.modules.clone());
    // The run's dimensions, by name, for every dimension variable of the program.
    let mut known = BTreeMap::new();
    for i in 0..elab.dims.len() {
        let v = DimVar(i as u32);
        if let Some(n) = opts.dims.get(&elab.dim_name(v)) {
            known.insert(v, *n);
        }
    }
    let guess = model::DimGuess { known, default: opts.default_dim.unwrap_or(64) };
    for i in 0..core.funcs.len() {
        let (block, mut outcome) = graph::rewrite_func(&core.funcs[i], &modules, opts, &guess);
        if let Some(block) = block {
            let saved = std::mem::replace(&mut core.funcs[i].body, block);
            if let Err(errs) = presk_core::verify(core) {
                core.funcs[i].body = saved;
                outcome = Outcome::Skipped(format!("the rewritten body does not verify: {}", errs.join("; ")));
            }
        }
        report.funcs.push(FuncReport { name: core.funcs[i].name.clone(), outcome });
    }
    report
}
