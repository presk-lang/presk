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
//! # What the solver chooses
//!
//! A type, in the reference's chapter 4, means something only where a value
//! is stored: a group's root or an unfused primitive. Those are the
//! solver's decisions ([`Role::Decision`]). A fused member is computed in its
//! root's type, and a parameter's field or an input's projection is rounded
//! by the nests that read it: their terms of the estimate follow those
//! decisions ([`Role::Follows`]). A call's or a fold's result, a tuple, a
//! parameter subtree carry another site's value and have none
//! ([`Role::Carried`]). [`Roles::of`] reads the roles off the program (plan
//! 0.11.0 step 2b, D112); until then every site was offered, and the solver
//! counted and priced types that changed no bit (D109).
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
    /// What the proposal and the measurement did, in order: each narrowing
    /// tried and whether the budget took it, each site widened after a
    /// measurement (plan 0.11.0 step 4, D114). A decision no move names
    /// saves nothing in a narrower type under the machine. `presk explain`
    /// gives each site's type its reason from these.
    pub moves: Vec<Move>,
}

/// One move of the solver.
#[derive(Clone, Debug, PartialEq)]
pub enum Move {
    /// Narrowing `site` to `to`, which saves `saved` (seconds under a
    /// profile, bytes without) and adds `added` to the estimate, which
    /// stood at `spent`: taken when `spent + added` fits the budget. The
    /// moves are tried by saving per unit of error added, the best first;
    /// `rank` is this one's place (1 the first).
    Tried { site: String, to: Dtype, saved: f64, added: f64, spent: f64, taken: bool, rank: usize },
    /// The measured `gap` exceeded the budget: `site`, the narrowed decision
    /// with the largest share of the estimate, widened from `from` to `to`
    /// at round `round` (1 the first).
    Widened { round: usize, site: String, from: Dtype, to: Dtype, gap: f64 },
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

/// The sites the solver offers: those of the envelope with elements, a
/// magnitude and a measured sensitivity. A site with **no** measured
/// sensitivity is not a site that is insensitive: it is one this run did not
/// reach, and reading the absence as zero would make it look maximally safe
/// to narrow. Unknown means leave it alone.
fn sites(env: &Envelope, sens: &Sensitivity) -> Vec<Site> {
    env.sites
        .iter()
        .filter_map(|(name, mag)| {
            let elems = env.elems(name) as f64;
            let sens = sens.get(name)?;
            let work = elems * op_weight(env.op(name));
            (elems > 0.0 && *mag > 0.0).then(|| Site { name: name.clone(), elems, mag: *mag, sens, work })
        })
        .collect()
}

/// What a site is to the solver (plan 0.11.0 step 2b, D112). The
/// reference's chapter 4 ("Realized value") gives a type a meaning only
/// where a value is stored: a group's root and an unfused primitive. A
/// fused member is computed in its root's type and stored nowhere; a field,
/// a projection, a call's or a fold's result, a tuple carry their value
/// unchanged; a nest converts each operand it reads to its own type. Until
/// 0.11.0 the solver offered every site, so that it counted, charged and
/// priced types that changed no bit (D109).
#[derive(Clone, Debug, PartialEq)]
pub enum Role {
    /// A value the program stores: its type is the solver's choice.
    Decision,
    /// A value rounded where others compute it: a fused member in its
    /// group's root (the one site named), a parameter's field or an
    /// input's projection in the nests that read it, at the narrowest of
    /// their types. Its term of the estimate is charged at that type; it
    /// has no type and no traffic of its own.
    Follows(Vec<String>),
    /// A value passed along unchanged: a call's or a fold's result, a
    /// tuple, a parameter subtree. Its value is another site's: no term, no
    /// cost.
    Carried,
}

/// The roles of a program's sites. A site it does not name is a decision,
/// which is what the solver took every site for until 0.11.0.
#[derive(Clone, Debug, Default)]
pub struct Roles(pub BTreeMap<String, Role>);

impl Roles {
    /// The roles of every site of `core`, from its `let`s and their fusion
    /// groups (`presk_core::fusion::plan`, the groups the kernel backends
    /// launch).
    pub fn of(core: &presk_core::Program) -> Roles {
        use presk_core::ir::{Rhs, VarId};
        use std::collections::{BTreeSet, HashMap};
        let mut roles = BTreeMap::new();
        for f in &core.funcs {
            let plan = presk_core::fusion::plan(f);
            let name = |v: VarId| format!("{}/%{}", f.name, v.0);
            let mut readers: HashMap<VarId, Vec<VarId>> = HashMap::new();
            for l in &f.body.lets {
                for v in l.rhs.vars() {
                    readers.entry(v).or_default().push(l.var);
                }
            }
            let mut root_of: HashMap<VarId, VarId> = HashMap::new();
            for g in plan.groups.values() {
                for m in &g.members {
                    root_of.insert(*m, g.root);
                }
            }
            let rhs: HashMap<VarId, &Rhs> = f.body.lets.iter().map(|l| (l.var, &l.rhs)).collect();
            for l in &f.body.lets {
                let role = if plan.interior.contains(&l.var) {
                    Role::Follows(vec![name(root_of[&l.var])])
                } else {
                    match &l.rhs {
                        Rhs::Prim { .. } | Rhs::Lit(_) => Role::Decision,
                        Rhs::Field(..) | Rhs::Proj(..) => {
                            // The nests that read it, each as the stored
                            // site it is computed under.
                            let leaders: BTreeSet<String> = readers
                                .get(&l.var)
                                .into_iter()
                                .flatten()
                                .filter(|r| matches!(rhs.get(*r), Some(Rhs::Prim { .. })))
                                .map(|r| name(*root_of.get(r).unwrap_or(r)))
                                .collect();
                            if leaders.is_empty() { Role::Carried } else { Role::Follows(leaders.into_iter().collect()) }
                        }
                        _ => Role::Carried,
                    }
                };
                roles.insert(name(l.var), role);
            }
        }
        Roles(roles)
    }

    pub fn role(&self, site: &str) -> Role {
        self.0.get(site).cloned().unwrap_or(Role::Decision)
    }
}

/// The sites the solver offers, split by role: the decisions, and each
/// follower with the decisions it follows (indices into the first) and
/// whether it follows a site outside them too, which stays in the
/// realization's default type. Carried sites are left out.
struct Split {
    decisions: Vec<Site>,
    followers: Vec<(Site, Vec<usize>, bool)>,
    /// For each decision, the followers it leads.
    led: Vec<Vec<usize>>,
}

fn split(env: &Envelope, sens: &Sensitivity, roles: &Roles) -> Split {
    let mut decisions = Vec::new();
    let mut pending = Vec::new();
    for s in sites(env, sens) {
        match roles.role(&s.name) {
            Role::Decision => decisions.push(s),
            Role::Follows(leaders) => pending.push((s, leaders)),
            Role::Carried => {}
        }
    }
    let index: BTreeMap<&str, usize> = decisions.iter().enumerate().map(|(i, s)| (s.name.as_str(), i)).collect();
    let mut led = vec![Vec::new(); decisions.len()];
    let mut followers = Vec::new();
    for (s, leaders) in pending {
        let ids: Vec<usize> = leaders.iter().filter_map(|l| index.get(l.as_str()).copied()).collect();
        let outside = ids.len() < leaders.len();
        for i in &ids {
            led[*i].push(followers.len());
        }
        followers.push((s, ids, outside));
    }
    Split { decisions, followers, led }
}

impl Split {
    /// A follower's type: the narrowest of its leaders', the default for a
    /// leader the solver does not offer.
    fn follower_type(&self, f: usize, chosen: &[Dtype], default: Dtype) -> Dtype {
        let (_, ids, outside) = &self.followers[f];
        ids.iter().map(|i| chosen[*i]).chain(outside.then_some(default)).min_by_key(|d| d.bytes()).unwrap_or(default)
    }

    fn follower_err(&self, f: usize, chosen: &[Dtype], default: Dtype) -> f64 {
        err_at(&self.followers[f].0, self.follower_type(f, chosen, default))
    }

    /// The estimate: every decision's term and every follower's.
    fn estimate(&self, chosen: &[Dtype], default: Dtype) -> f64 {
        let d: f64 = self.decisions.iter().zip(chosen).map(|(s, t)| err_at(s, *t)).sum();
        d + (0..self.followers.len()).map(|f| self.follower_err(f, chosen, default)).sum::<f64>()
    }

    /// What decision `i` and the followers it leads add to the estimate.
    fn share(&self, i: usize, chosen: &[Dtype], default: Dtype) -> f64 {
        err_at(&self.decisions[i], chosen[i]) + self.led[i].iter().map(|f| self.follower_err(*f, chosen, default)).sum::<f64>()
    }

    fn cost(&self, chosen: &[Dtype], machine: &dyn Machine) -> f64 {
        self.decisions.iter().zip(chosen).map(|(s, t)| cost_at(s, *t, machine)).sum()
    }
}

/// One site the solver offers, under a realization: what its type adds to
/// the estimate and costs, beside what the widest type of the menu would
/// (plan 0.11.0 step 2). The reason of each site's type, which `presk
/// explain` prints: the estimate is the sum of the `err`s.
#[derive(Clone, Debug)]
pub struct SiteCost {
    pub name: String,
    /// Its role (step 2b): a decision, a follower or a carried value.
    pub role: Role,
    /// Its type: the realization's for a decision, the type it is rounded
    /// to for a follower, its own for a carried value, which is not rounded
    /// there.
    pub dtype: Dtype,
    pub elems: f64,
    /// The envelope's magnitude.
    pub mag: f64,
    /// `‖∂L/∂v‖₁`.
    pub sens: f64,
    /// `s · u(dtype) · |v|`, its term of the estimate; 0 for a carried
    /// value.
    pub err: f64,
    /// What it costs under the machine, in its type; 0 but for a decision.
    pub cost: f64,
    /// The same in the widest type of the menu.
    pub widest_err: f64,
    pub widest_cost: f64,
}

/// Every site the solver offers, decisions first, then the followers and
/// the carried values, under `r`.
pub fn breakdown(env: &Envelope, sens: &Sensitivity, roles: &Roles, r: &Realization, menu: &Menu, default: Dtype, machine: &dyn Machine) -> Vec<SiteCost> {
    let widest = menu.0.iter().copied().max_by_key(|d| d.bytes()).unwrap_or(default);
    let sp = split(env, sens, roles);
    let chosen: Vec<Dtype> = sp.decisions.iter().map(|s| r.storage.get(&s.name).copied().unwrap_or(r.default)).collect();
    let mut out: Vec<SiteCost> = sp
        .decisions
        .iter()
        .zip(&chosen)
        .map(|(s, d)| SiteCost { name: s.name.clone(), role: Role::Decision, dtype: *d, elems: s.elems, mag: s.mag, sens: s.sens, err: err_at(s, *d), cost: cost_at(s, *d, machine), widest_err: err_at(s, widest), widest_cost: cost_at(s, widest, machine) })
        .collect();
    for (f, (s, _, _)) in sp.followers.iter().enumerate() {
        let d = sp.follower_type(f, &chosen, r.default);
        out.push(SiteCost { name: s.name.clone(), role: roles.role(&s.name), dtype: d, elems: s.elems, mag: s.mag, sens: s.sens, err: err_at(s, d), cost: 0.0, widest_err: err_at(s, widest), widest_cost: 0.0 });
    }
    for s in sites(env, sens) {
        if roles.role(&s.name) == Role::Carried {
            out.push(SiteCost { name: s.name.clone(), role: Role::Carried, dtype: r.default, elems: s.elems, mag: s.mag, sens: s.sens, err: 0.0, cost: 0.0, widest_err: 0.0, widest_cost: 0.0 });
        }
    }
    out
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
/// footing. `roles` says which sites are the solver's to choose (step 2b);
/// empty, every site is, as until 0.11.0.
#[allow(clippy::too_many_arguments)]
pub fn solve(
    env: &Envelope,
    sens: &Sensitivity,
    roles: &Roles,
    output_site: &str,
    budget_rel: f64,
    menu: &Menu,
    default: Dtype,
    machine: &dyn Machine,
    mut verify: Option<&mut dyn FnMut(&Realization) -> f64>,
) -> Solution {
    let out_mag = env.get(output_site).unwrap_or(1.0).max(f64::MIN_POSITIVE);
    let budget_abs = budget_rel * out_mag;
    let sp = split(env, sens, roles);
    let sites = &sp.decisions;

    let widest = menu.0.iter().copied().max_by_key(|d| d.bytes()).unwrap_or(default);
    let narrowest = menu.0.iter().copied().min_by_key(|d| d.bytes()).unwrap_or(default);
    let baseline_bytes: f64 = sites.iter().map(|s| cost_at(s, widest, machine)).sum();
    let relaxation_bytes: f64 = sites.iter().map(|s| cost_at(s, narrowest, machine)).sum();

    // Propose: every site at the widest type, then narrow in order of bytes
    // saved per unit of error added (its followers' with its own), while
    // the estimate fits.
    let mut chosen: Vec<Dtype> = vec![widest; sites.len()];
    let mut spent: f64 = sp.estimate(&chosen, default);
    let mut moves: Vec<(f64, usize, Dtype)> = Vec::new();
    for (i, s) in sites.iter().enumerate() {
        for d in &menu.0 {
            if d.bytes() >= widest.bytes() {
                continue;
            }
            let saved = cost_at(s, widest, machine) - cost_at(s, *d, machine);
            let mut moved = chosen.clone();
            moved[i] = *d;
            let added = sp.share(i, &moved, default) - sp.share(i, &chosen, default);
            if saved <= 0.0 {
                continue;
            }
            // Ratio, with a free move (no error added) sorted first.
            let ratio = if added > 0.0 { saved / added } else { f64::INFINITY };
            moves.push((ratio, i, *d));
        }
    }
    moves.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut log: Vec<Move> = Vec::new();
    for (rank, (_, i, d)) in moves.iter().enumerate() {
        let cur = chosen[*i];
        if d.bytes() >= cur.bytes() {
            continue;
        }
        let before = sp.share(*i, &chosen, default);
        chosen[*i] = *d;
        let delta = sp.share(*i, &chosen, default) - before;
        let saved = cost_at(&sites[*i], cur, machine) - cost_at(&sites[*i], *d, machine);
        let taken = spent + delta <= budget_abs;
        log.push(Move::Tried { site: sites[*i].name.clone(), to: *d, saved, added: delta, spent, taken, rank: rank + 1 });
        if taken {
            spent += delta;
        } else {
            chosen[*i] = cur;
        }
    }

    let proposed_bytes: f64 = sp.cost(&chosen, machine);

    let build = |chosen: &[Dtype]| -> Realization {
        let mut r = Realization::uniform(default);
        r.storage = sites.iter().zip(chosen).filter(|(_, d)| **d != default).map(|(s, d)| (s.name.clone(), *d)).collect();
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
            // Widen the narrowed site whose estimated contribution, its
            // followers' with its own, is largest — by **one step** of the
            // menu, not straight to the widest. Jumping to the top undoes in
            // one round everything the proposal found, and with a flat
            // estimate (every sensitivity zero, as happens on an untrained
            // model) it walks the whole assignment back to `f64` and reports
            // no saving at all.
            let worst = (0..sites.len()).filter(|i| chosen[*i].bytes() < widest.bytes()).max_by(|a, b| {
                let (ea, eb) = (sp.share(*a, &chosen, default), sp.share(*b, &chosen, default));
                ea.partial_cmp(&eb).unwrap_or(std::cmp::Ordering::Equal)
            });
            match worst {
                Some(i) => {
                    let cur = chosen[i];
                    let next = menu.0.iter().copied().filter(|d| d.bytes() > cur.bytes()).min_by_key(|d| d.bytes()).unwrap_or(widest);
                    chosen[i] = next;
                    rounds += 1;
                    log.push(Move::Widened { round: rounds, site: sites[i].name.clone(), from: cur, to: next, gap });
                }
                None => break,
            }
        }
    }

    let bytes: f64 = sp.cost(&chosen, machine);
    let estimated_error: f64 = sp.estimate(&chosen, default);
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
        moves: log,
    }
}

/// The best assignment by exhaustive search, for programs small enough that
/// it is affordable. Used to say how far the greedy proposal is from optimal
/// rather than to assert that it is optimal.
///
/// Returns `None` when the search space exceeds `max_states`.
#[allow(clippy::too_many_arguments)]
pub fn exhaustive(env: &Envelope, sens: &Sensitivity, roles: &Roles, output_site: &str, budget_rel: f64, menu: &Menu, default: Dtype, machine: &dyn Machine, max_states: u64) -> Option<(Realization, f64)> {
    let out_mag = env.get(output_site).unwrap_or(1.0).max(f64::MIN_POSITIVE);
    let budget_abs = budget_rel * out_mag;
    let sp = split(env, sens, roles);
    let sites = &sp.decisions;
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
        let err: f64 = sp.estimate(&pick, default);
        if err > budget_abs {
            continue;
        }
        let bytes: f64 = sp.cost(&pick, machine);
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
