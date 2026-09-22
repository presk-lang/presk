//! The solver: a budget in, a realization out.
//!
//! # What a budget is a claim about
//!
//! The plan proposed that the solver *propose* with the sensitivity and
//! *verify* with the propagated bound. Step 4 established that the bound,
//! though sound, is vacuous on a real transformer — a relative bound of
//! 10²⁰ on a loss of 4.5 (D39) — so verifying against it would accept
//! everything and mean nothing.
//!
//! So the solver verifies by **measurement** against the exact reference
//! (D41). That is a stronger check than a bound, not a weaker one: it is the
//! error actually committed rather than an envelope around it, and presk is
//! the only compiler that can make it because only presk keeps an exact
//! denotation to measure against. The propagated bound is still computed and
//! reported beside the measurement, clearly labelled as the weaker artefact.
//!
//! # The shape of the search
//!
//! Choosing one dtype per site to minimise traffic subject to an error
//! constraint is a multiple-choice knapsack. This version takes the cheap,
//! honest route, as 0.4.0 did for extraction:
//!
//! 1. **Propose** greedily, narrowing sites in order of bytes saved per unit
//!    of sensitivity-weighted error added, while the first-order estimate
//!    stays under the budget.
//! 2. **Verify** by running the candidate and measuring its distance to the
//!    exact reference.
//! 3. **Tighten**: while the measured gap exceeds the budget, widen the site
//!    whose estimated contribution is largest, and measure again. This
//!    terminates, because all-`f64` is a feasible floor whenever anything is.
//! 4. **Report the gap** to the relaxation's lower bound — every site at its
//!    narrowest — and, on programs small enough, to exhaustive search.
//!
//! # Why traffic and not time
//!
//! The objective is bytes moved, because that is what a narrower type
//! actually reduces. It is deliberately *not* time: step 2 measured that
//! `bf16` is **slower** than `f32` on a compute-bound kernel, the conversion
//! being pure overhead there, so minimising traffic and minimising time are
//! different problems. Pricing time per site needs the calibrated roofline's
//! `W` as well as its `Q`, and that is named work rather than assumed.
//!
//! # Over a training step
//!
//! A budget in a `training run` binds one optimizer step's update (0.6.0
//! §3.7, D52), and most of a step — the reverse sweep and the optimizer — has
//! no first-order sensitivity to rank sites by. [`solve_step`] therefore
//! chooses per **function**: each is measured alone in every narrower type,
//! the proposal combines the isolated gaps as a root-sum-square (D49), and
//! the combination is measured and widened as above. The measurements are
//! the caller's, since the step's state and the exact reference live in
//! `presk-train` and the CLI.

use std::collections::BTreeMap;

use presk_rt::{Dtype, Realization};

use crate::envelope::Envelope;
use crate::sensitivity::Sensitivity;

/// The machine types a site may be assigned.
#[derive(Clone, Debug)]
pub struct Menu(pub Vec<Dtype>);

impl Default for Menu {
    /// Storage types 0.5.0 can emit. Integer quantisation is 0.6.0: it needs
    /// per-channel scales and a calibration semantics of its own.
    fn default() -> Menu {
        Menu(vec![Dtype::F64, Dtype::F32, Dtype::Bf16])
    }
}

/// What the solver found, and what it cost.
#[derive(Clone, Debug)]
pub struct Solution {
    pub realization: Realization,
    /// Bytes the assignment moves, and what all-`f64` would move.
    pub bytes: f64,
    /// Bytes the *proposal* moved, before measurement widened anything. This
    /// is what compares with exhaustive search, since both answer the same
    /// question — the best assignment under the first-order estimate — while
    /// the final figure has also survived a measurement the search never ran.
    pub proposed_bytes: f64,
    pub baseline_bytes: f64,
    /// Bytes the relaxation would move — every site at its narrowest, the
    /// constraint ignored. The distance to it is how much the constraint,
    /// rather than the heuristic, is costing.
    pub relaxation_bytes: f64,
    /// `Σ s_v · u(d) · mag(v)`: the first-order estimate that drove the
    /// proposal. It *ranks*; it does not guarantee.
    pub estimated_error: f64,
    /// The gap actually measured against the exact reference, if a verifier
    /// was supplied.
    pub measured: Option<f64>,
    /// The budget, in absolute terms.
    pub budget_abs: f64,
    /// How many widen-and-measure rounds the tightening took.
    pub rounds: usize,
    /// Sites the solver could narrow at all (the rest had no magnitude, no
    /// sensitivity or no element count and were left at the default).
    pub considered: usize,
}

impl Solution {
    pub fn speedup(&self) -> f64 {
        if self.bytes > 0.0 { self.baseline_bytes / self.bytes } else { 1.0 }
    }
}

/// One site's cost and error at one dtype.
struct Site {
    name: String,
    elems: f64,
    mag: f64,
    sens: f64,
    /// Weighted operations the site performs, for the roofline's `W`.
    work: f64,
}

/// What a site costs in a given machine type.
///
/// Traffic when no machine is given, and **time** when one is. The
/// difference is the point of 0.6.0 step 4: 0.5.0 minimised traffic, which
/// is what a narrow type reduces, and lost — on a compute-bound site
/// narrowing buys nothing and the conversions cost, so minimising bytes
/// picked assignments that were slower. A roofline priced per site declines
/// to narrow exactly there.
pub trait Machine {
    fn seconds(&self, dtype: Dtype, work: f64, bytes: f64) -> f64;
}

/// The traffic-only objective of 0.5.0, kept so the two can be compared.
pub struct Traffic;

impl Machine for Traffic {
    fn seconds(&self, _d: Dtype, _work: f64, bytes: f64) -> f64 {
        bytes
    }
}

/// `s_v · u(d) · mag(v)`: how much the output moves, to first order, when a
/// site is stored in `d`.
fn err_at(s: &Site, d: Dtype) -> f64 {
    s.sens * d.unit_roundoff() * s.mag
}

fn bytes_at(s: &Site, d: Dtype) -> f64 {
    s.elems * d.bytes() as f64
}

/// What a site costs under a machine: the roofline's own prediction, with
/// the dtype in `Q` and not in `W`.
fn cost_at(s: &Site, d: Dtype, m: &dyn Machine) -> f64 {
    m.seconds(d, s.work, bytes_at(s, d))
}

/// Solve for a realization under a relative error budget.
///
/// `verify` measures a candidate's distance to the exact reference. Passing
/// `None` gives the proposal without the check, which is what the tests use
/// when they want to compare the heuristic with exhaustive search on the same
/// footing.
#[allow(clippy::too_many_arguments)]
pub fn solve(
    env: &Envelope,
    sens: &Sensitivity,
    output_site: &str,
    budget_rel: f64,
    menu: &Menu,
    default: Dtype,
    machine: &dyn Machine,
    mut verify: Option<&mut dyn FnMut(&Realization) -> f64>,
) -> Solution {
    let out_mag = env.get(output_site).unwrap_or(1.0).max(f64::MIN_POSITIVE);
    let budget_abs = budget_rel * out_mag;

    let sites: Vec<Site> = env
        .sites
        .iter()
        .filter_map(|(name, mag)| {
            let elems = env.elems(name) as f64;
            // A site with **no** measured sensitivity is not a site that is
            // insensitive: it is one this run did not reach, and reading the
            // absence as zero would make it look maximally safe to narrow.
            // Unknown means leave it alone.
            let sens = sens.get(name)?;
            let work = elems * op_weight(env.op(name));
            (elems > 0.0 && *mag > 0.0).then(|| Site { name: name.clone(), elems, mag: *mag, sens, work })
        })
        .collect();

    let widest = menu.0.iter().copied().max_by_key(|d| d.bytes()).unwrap_or(default);
    let narrowest = menu.0.iter().copied().min_by_key(|d| d.bytes()).unwrap_or(default);
    let baseline_bytes: f64 = sites.iter().map(|s| cost_at(s, widest, machine)).sum();
    let relaxation_bytes: f64 = sites.iter().map(|s| cost_at(s, narrowest, machine)).sum();

    // Propose: every site at the widest type, then narrow in order of bytes
    // saved per unit of error added, while the estimate fits.
    let mut chosen: BTreeMap<String, Dtype> = sites.iter().map(|s| (s.name.clone(), widest)).collect();
    let mut spent: f64 = sites.iter().map(|s| err_at(s, widest)).sum();
    let mut moves: Vec<(f64, usize, Dtype)> = Vec::new();
    for (i, s) in sites.iter().enumerate() {
        for d in &menu.0 {
            if d.bytes() >= widest.bytes() {
                continue;
            }
            let saved = cost_at(s, widest, machine) - cost_at(s, *d, machine);
            let added = err_at(s, *d) - err_at(s, widest);
            if saved <= 0.0 {
                continue;
            }
            // Ratio, with a free move (no error added) sorted first.
            let ratio = if added > 0.0 { saved / added } else { f64::INFINITY };
            moves.push((ratio, i, *d));
        }
    }
    moves.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    for (_, i, d) in &moves {
        let s = &sites[*i];
        let cur = chosen[&s.name];
        if d.bytes() >= cur.bytes() {
            continue;
        }
        let delta = err_at(s, *d) - err_at(s, cur);
        if spent + delta <= budget_abs {
            spent += delta;
            chosen.insert(s.name.clone(), *d);
        }
    }

    let proposed_bytes: f64 = sites.iter().map(|s| cost_at(s, chosen[&s.name], machine)).sum();

    let build = |chosen: &BTreeMap<String, Dtype>| -> Realization {
        let mut r = Realization::uniform(default);
        r.storage = chosen.iter().filter(|(_, d)| **d != default).map(|(k, d)| (k.clone(), *d)).collect();
        r.envelope = Some(env.program.clone());
        r
    };

    // Verify, and tighten while the *measured* gap exceeds the budget.
    let mut rounds = 0usize;
    let mut measured = None;
    if let Some(v) = verify.as_mut() {
        loop {
            let r = build(&chosen);
            let gap = v(&r);
            measured = Some(gap);
            if gap <= budget_abs || rounds >= sites.len() * menu.0.len() {
                break;
            }
            // Widen the narrowed site whose estimated contribution is
            // largest — by **one step** of the menu, not straight to the
            // widest. Jumping to the top undoes in one round everything the
            // proposal found, and with a flat estimate (every sensitivity
            // zero, as happens on an untrained model) it walks the whole
            // assignment back to `f64` and reports no saving at all.
            let worst = sites
                .iter()
                .filter(|s| chosen[&s.name].bytes() < widest.bytes())
                .max_by(|a, b| {
                    let (ea, eb) = (err_at(a, chosen[&a.name]), err_at(b, chosen[&b.name]));
                    ea.partial_cmp(&eb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|s| s.name.clone());
            match worst {
                Some(name) => {
                    let cur = chosen[&name];
                    let next = menu
                        .0
                        .iter()
                        .copied()
                        .filter(|d| d.bytes() > cur.bytes())
                        .min_by_key(|d| d.bytes())
                        .unwrap_or(widest);
                    chosen.insert(name, next);
                    rounds += 1;
                }
                None => break,
            }
        }
    }

    let bytes: f64 = sites.iter().map(|s| cost_at(s, chosen[&s.name], machine)).sum();
    let estimated_error: f64 = sites.iter().map(|s| err_at(s, chosen[&s.name])).sum();
    Solution {
        realization: build(&chosen),
        bytes,
        proposed_bytes,
        baseline_bytes,
        relaxation_bytes,
        estimated_error,
        measured,
        budget_abs,
        rounds,
        considered: sites.len(),
    }
}

/// The best assignment by exhaustive search, for programs small enough that
/// it is affordable. Used to say how far the greedy proposal is from optimal
/// rather than to assert that it is optimal.
///
/// Returns `None` when the search space exceeds `max_states`.
#[allow(clippy::too_many_arguments)]
pub fn exhaustive(env: &Envelope, sens: &Sensitivity, output_site: &str, budget_rel: f64, menu: &Menu, default: Dtype, machine: &dyn Machine, max_states: u64) -> Option<(Realization, f64)> {
    let out_mag = env.get(output_site).unwrap_or(1.0).max(f64::MIN_POSITIVE);
    let budget_abs = budget_rel * out_mag;
    let sites: Vec<Site> = env
        .sites
        .iter()
        .filter_map(|(name, mag)| {
            let elems = env.elems(name) as f64;
            let sens = sens.get(name)?;
            let work = elems * op_weight(env.op(name));
            (elems > 0.0 && *mag > 0.0).then(|| Site { name: name.clone(), elems, mag: *mag, sens, work })
        })
        .collect();
    let m = menu.0.len() as u64;
    let states = m.checked_pow(sites.len() as u32)?;
    if states > max_states {
        return None;
    }
    let mut best: Option<(Vec<Dtype>, f64)> = None;
    for code in 0..states {
        let mut c = code;
        let mut pick = Vec::with_capacity(sites.len());
        for _ in 0..sites.len() {
            pick.push(menu.0[(c % m) as usize]);
            c /= m;
        }
        let err: f64 = sites.iter().zip(&pick).map(|(s, d)| err_at(s, *d)).sum();
        if err > budget_abs {
            continue;
        }
        let bytes: f64 = sites.iter().zip(&pick).map(|(s, d)| cost_at(s, *d, machine)).sum();
        if best.as_ref().is_none_or(|(_, b)| bytes < *b) {
            best = Some((pick, bytes));
        }
    }
    let (pick, bytes) = best?;
    let mut r = Realization::uniform(default);
    r.storage = sites.iter().zip(&pick).filter(|(_, d)| **d != default).map(|(s, d)| (s.name.clone(), *d)).collect();
    Some((r, bytes))
}


/// Weighted operations per element for a primitive, by name.
///
/// The same weights the kernel cost model uses, read from the name the
/// envelope recorded rather than from the kernel, because the solver works
/// before any kernel exists. A transcendental is dearer than an addition and
/// a contraction is dearer than either; anything unrecognised is one, which
/// under-states a contraction and is the conservative direction for an
/// objective that is being *minimised*.
fn op_weight(op: Option<&str>) -> f64 {
    match op.unwrap_or("") {
        "dot" | "matmul" | "ein" => 64.0,
        "exp" | "log" | "tanh" | "sigmoid" | "gelu" | "erf" | "sin" | "cos" | "sqrt" | "softmax" | "cross_entropy" => 8.0,
        "" => 1.0,
        _ => 1.0,
    }
}

/// One function of a training step, offered as a whole (0.6.0 step 7b,
/// §3.7): the unit at which the reverse sweep and the optimizer are chosen,
/// since they have no first-order sensitivity to rank sites by without a
/// second derivative.
#[derive(Clone, Debug)]
pub struct Group {
    /// The function's name: `Linear.apply.vjp`, `adamw.step`.
    pub name: String,
    /// Sites of the envelope in it, and their elements.
    pub sites: usize,
    pub elems: u64,
    /// Predicted cost of the group's sites per dtype of the menu (seconds
    /// with a machine profile, bytes otherwise).
    pub cost: BTreeMap<Dtype, f64>,
    /// The update gap with this group alone in each dtype, every other site
    /// at the default: measured, never estimated.
    pub gap: BTreeMap<Dtype, f64>,
}

/// The groups of a step's envelope, priced but not yet measured.
pub fn step_groups(env: &Envelope, menu: &Menu, machine: &dyn Machine) -> Vec<Group> {
    let mut by_fn: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
    for name in env.sites.keys() {
        let Some((f, _)) = name.rsplit_once('/') else { continue };
        let elems = env.elems(name) as f64;
        if elems > 0.0 {
            by_fn.entry(f.to_string()).or_default().push((elems, elems * op_weight(env.op(name))));
        }
    }
    by_fn
        .into_iter()
        .map(|(name, sites)| {
            let cost = menu.0.iter().map(|d| (*d, sites.iter().map(|(e, w)| machine.seconds(*d, *w, e * d.bytes() as f64)).sum())).collect();
            Group { name, sites: sites.len(), elems: sites.iter().map(|(e, _)| *e as u64).sum(), cost, gap: BTreeMap::new() }
        })
        .collect()
}

/// What the solver over a step found.
#[derive(Clone, Debug)]
pub struct StepSolution {
    pub realization: Realization,
    /// The groups, with their isolated gaps filled in.
    pub groups: Vec<Group>,
    /// The type each group was given (the default for those left alone).
    pub chosen: BTreeMap<String, Dtype>,
    /// `√Σ gap²` of the chosen groups' isolated gaps: what the greedy
    /// proposal predicted for the combination (D49's root-sum-square).
    pub estimate: f64,
    /// The combination's update gap, measured.
    pub measured: Option<f64>,
    /// Predicted cost of the solution and of all-default.
    pub cost: f64,
    pub baseline: f64,
    pub rounds: usize,
}

impl StepSolution {
    pub fn speedup(&self) -> f64 {
        if self.cost > 0.0 { self.baseline / self.cost } else { 1.0 }
    }
}

/// The step under a realization that gives each group of `chosen` its type.
pub fn by_function(default: Dtype, chosen: &BTreeMap<String, Dtype>) -> Realization {
    let mut r = Realization::uniform(default);
    r.functions = chosen.iter().filter(|(_, d)| **d != default).map(|(k, d)| (k.clone(), *d)).collect();
    r
}

/// Solve a training step under an update budget (§3.7):
///
/// 1. every group is measured alone in every narrower type of the menu
///    (`measure` returns the update gap of a realization);
/// 2. groups are narrowed greedily by predicted cost saved per unit of
///    isolated gap, while the gaps' root-sum-square stays inside the budget;
/// 3. the combination is measured, and while it is outside, the group with
///    the largest isolated gap at its current type is widened one step of
///    the menu (D41).
///
/// The budget is relative to the update, as the gaps are, so there is no
/// magnitude to scale it by.
pub fn solve_step(mut groups: Vec<Group>, budget: f64, menu: &Menu, default: Dtype, measure: &mut dyn FnMut(&Realization) -> f64, verify: bool) -> StepSolution {
    let narrower: Vec<Dtype> = menu.0.iter().copied().filter(|d| d.bytes() < default.bytes()).collect();
    for g in groups.iter_mut() {
        for d in &narrower {
            let r = by_function(default, &BTreeMap::from([(g.name.clone(), *d)]));
            g.gap.insert(*d, measure(&r));
        }
        g.gap.insert(default, 0.0);
    }
    let cost_of = |g: &Group, d: Dtype| g.cost.get(&d).copied().unwrap_or(0.0);
    let gap_of = |g: &Group, d: Dtype| g.gap.get(&d).copied().unwrap_or(f64::INFINITY);
    let mut chosen: BTreeMap<String, Dtype> = groups.iter().map(|g| (g.name.clone(), default)).collect();
    let mut moves: Vec<(f64, usize, Dtype)> = Vec::new();
    for (i, g) in groups.iter().enumerate() {
        for d in &narrower {
            let saved = cost_of(g, default) - cost_of(g, *d);
            let gap = gap_of(g, *d);
            if saved > 0.0 && gap.is_finite() {
                moves.push((if gap > 0.0 { saved / gap } else { f64::INFINITY }, i, *d));
            }
        }
    }
    moves.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let rss = |chosen: &BTreeMap<String, Dtype>| -> f64 { groups.iter().map(|g| gap_of(g, chosen[&g.name]).powi(2)).sum::<f64>().sqrt() };
    for (_, i, d) in &moves {
        let g = &groups[*i];
        if d.bytes() >= chosen[&g.name].bytes() {
            continue;
        }
        let mut next = chosen.clone();
        next.insert(g.name.clone(), *d);
        if rss(&next) <= budget {
            chosen = next;
        }
    }
    let mut rounds = 0;
    let mut measured = None;
    if verify {
        loop {
            let gap = measure(&by_function(default, &chosen));
            measured = Some(gap);
            if gap <= budget {
                break;
            }
            let worst = groups
                .iter()
                .filter(|g| chosen[&g.name].bytes() < default.bytes())
                .max_by(|a, b| gap_of(a, chosen[&a.name]).partial_cmp(&gap_of(b, chosen[&b.name])).unwrap_or(std::cmp::Ordering::Equal))
                .map(|g| g.name.clone());
            let Some(name) = worst else { break };
            let cur = chosen[&name];
            let wider = menu.0.iter().copied().filter(|d| d.bytes() > cur.bytes()).min_by_key(|d| d.bytes()).unwrap_or(default);
            chosen.insert(name, wider);
            rounds += 1;
        }
    }
    let cost = groups.iter().map(|g| cost_of(g, chosen[&g.name])).sum();
    let baseline = groups.iter().map(|g| cost_of(g, default)).sum();
    let estimate = rss(&chosen);
    StepSolution { realization: by_function(default, &chosen), groups, chosen, estimate, measured, cost, baseline, rounds }
}

/// The predicted cost of any realization of the step, over the same groups:
/// a group's cost at the type its function is given (sites named one by one
/// are not priced separately; the realizations measured here have none).
pub fn step_cost(groups: &[Group], r: &Realization) -> f64 {
    groups.iter().map(|g| g.cost.get(&r.functions.get(&g.name).copied().unwrap_or(r.default)).copied().unwrap_or(0.0)).sum()
}
