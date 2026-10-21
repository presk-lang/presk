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
//! Two options change steps 1 and 3 (D155), both off by default, since every
//! row of record was solved without them:
//! - **`RoundingTerm::Measured`**: the estimate's rounding step is the
//!   envelope's measured `max |fl_d(x) − x|` at each site instead of `u(d) ·
//!   max|x|` ([`crate::envelope::Rounding`]). A value exact in a type, a mask
//!   of ones in `bf16`, then costs nothing there. The term is as conditional
//!   on the envelope as the magnitude is; the estimate was never a bound.
//! - **`Narrow::Measured`**, a step 3′ after a passing measurement: each
//!   decision not at the menu's narrowest, the largest saving first, is
//!   narrowed one step and kept if the measured gap stays inside the budget;
//!   up to three passes while one changes something. The solver then takes
//!   by its measurement a step only the proposal decided before, at one
//!   measurement a decision a pass.
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

use crate::envelope::{Envelope, Rounding};
use crate::sensitivity::Sensitivity;

/// A budget on an entry's output: relative to its largest magnitude in the
/// envelope, as every budget was until 0.12.0, or absolute (plan 0.12.0
/// step 4, the plan's question 5: ε\* from a model's margins is a number on
/// its logits).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Budget {
    Relative(f64),
    Absolute(f64),
}

impl Budget {
    /// The budget in absolute terms, for an output of largest magnitude
    /// `out_mag`.
    pub fn absolute(self, out_mag: f64) -> f64 {
        match self {
            Budget::Relative(r) => r * out_mag,
            Budget::Absolute(a) => a,
        }
    }
}

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

/// The estimate's rounding step at a site in a type (D155).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RoundingTerm {
    /// `u(d) · max|x|`, the unit roundoff times the magnitude: the solver
    /// of 0.12.0–0.15.0, kept for the record.
    Unit,
    /// The envelope's `max |fl_d(x) − x|`, measured on the sampled values; a
    /// site without it, from an envelope written before D155, keeps `Unit`.
    /// The default since 0.16.0 (D157, the user's decision after D155 and
    /// D156 measured it holding out of sample).
    #[default]
    Measured,
}

/// Whether the solver narrows by measurement after its measurement passes
/// (D155).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Narrow {
    /// No: what the proposal left wide stays wide. The default.
    #[default]
    Off,
    /// Step 3′: each decision narrowed one step, the largest saving first,
    /// kept while the measured gap stays inside the budget divided by
    /// [`Options::margin`] (D161; inside the budget itself in D155, a
    /// margin of 1); up to three passes.
    Measured,
    /// [`Narrow::Measured`], then early stopping on held-out samples
    /// (D161): the moves it kept are undone, the last first, until the gap
    /// on samples the envelope was not calibrated on is inside the budget
    /// divided by the margin. The held-out samples are measured by the
    /// second verifier of [`solve_with_heldout`].
    Heldout,
}

/// The estimate the proposal spends the budget by (D161).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Estimate {
    /// The sum of the sites' first-order terms, aimed at the budget: the
    /// solver of record. The default.
    #[default]
    Sum,
    /// Their root-sum-square, the roundings taken as independent (Higham
    /// and Mary's probabilistic analysis), aimed at the budget divided by
    /// [`Options::margin`].
    Rss,
}

/// The options of [`solve_with`]; the default is 0.16.0's solver: the
/// measured rounding term, the sum aimed at the budget, no narrowing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    pub rounding: RoundingTerm,
    pub narrow: Narrow,
    pub estimate: Estimate,
    /// The factor kept between the budget and what is spent in sample
    /// (D161): the target of [`Estimate::Rss`] and the narrowings'
    /// threshold, the budget divided by it. The solver of record (`Sum`,
    /// `Off`) does not read it. 2.5 by default, the largest gap out of
    /// sample over the gap in sample that D156 measured from four windows
    /// on, on the course GPT's logits at D = 64.
    pub margin: f64,
    /// Under `Narrow::Heldout`: how many samples are held out, the ones
    /// after the envelope's in its data file; `None`, as many as the
    /// envelope's.
    pub heldout: Option<usize>,
}

impl Default for Options {
    fn default() -> Options {
        Options { rounding: RoundingTerm::default(), narrow: Narrow::default(), estimate: Estimate::default(), margin: 2.5, heldout: None }
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
    /// How many decisions the narrowing step kept narrowed (D155), and in
    /// how many passes: 0 without `Narrow::Measured`.
    pub narrowed: usize,
    pub narrow_passes: usize,
    /// Under `Narrow::Heldout` (D161): the narrowings undone, and the gap
    /// of the final realization on the held-out samples.
    pub undone: usize,
    pub heldout: Option<f64>,
    /// How many realizations were measured against the exact reference: the
    /// verification and its widening rounds, then the narrowing's trials.
    pub measurements: usize,
    /// The options it was solved under.
    pub options: Options,
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
    /// The narrowing step (D155): `site` narrowed from `from` to `to` at
    /// pass `pass` (1 the first), the realization measured at `gap`, and
    /// `kept` when that is inside the budget, else undone.
    Narrowed { pass: usize, site: String, from: Dtype, to: Dtype, gap: f64, kept: bool },
    /// Early stopping (D161): the realization measured `heldout` on the
    /// held-out samples, outside the threshold, so the last narrowing kept,
    /// `site`'s from `from` to `to`, is undone.
    Undone { site: String, from: Dtype, to: Dtype, heldout: f64 },
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
    /// A parameter, decided by its path (plan 0.12.0), not a nest.
    param: bool,
    /// The envelope's measured rounding errors, or the parameter's own
    /// (D155): `None` keeps the unit-roundoff term.
    round: Option<Rounding>,
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
            // One sample's elements (D157), so that the parameters, counted
            // once, weigh the same against the activations at every N.
            let elems = env.cost_elems(name) as f64;
            let sens = sens.get(name)?;
            let work = elems * op_weight(env.op(name));
            (elems > 0.0 && *mag > 0.0).then(|| Site { name: name.clone(), elems, mag: *mag, sens, work, param: false, round: env.rounding.get(name).copied() })
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
/// which is what the solver took every site for until 0.11.0. Since 0.12.0
/// the parameters it offers come with them ([`Roles::with_params`]).
#[derive(Clone, Debug, Default)]
pub struct Roles(pub BTreeMap<String, Role>, pub Vec<ParamSite>);

/// A parameter the solver offers (plan 0.12.0 step 4, D123): a leaf of the
/// entry's parameter tree, decided by its path as the realization's
/// `params` keys it, rounded once at rest to its type and read by the
/// nests that read its field sites. Its term of the estimate is
/// `sens · u · mag` at the narrowest of its type and its readers' types:
/// the value a nest reads is rounded there, at rest or at the nest's edge.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamSite {
    pub path: String,
    pub elems: f64,
    /// Its largest magnitude.
    pub mag: f64,
    /// `‖∂L/∂θ‖₁` over the leaf, from the reverse sweep's cotangent.
    pub sens: f64,
    /// The field sites that read it ([`param_reads`]).
    pub fields: Vec<String>,
    /// Its largest rounding error in each type, from its values at rest
    /// (D155); `None` keeps the unit-roundoff term.
    pub round: Option<Rounding>,
}

/// The paths of the entry's parameter tree each field or projection site
/// reads, followed from the entry's first parameter through fields,
/// projections, calls, closures and folds (a stacked subtree sliced by a
/// `fold` keeps its path: a parameter's type is every layer's). A site of a
/// function called on several subtrees reads several paths: `Linear.apply`'s
/// `θ.w` reads every linear layer's matrix. A path may be a subtree's; the
/// caller keeps the leaves'.
pub fn param_reads(core: &presk_core::Program, entry: presk_core::FuncId) -> BTreeMap<String, std::collections::BTreeSet<String>> {
    use presk_core::ir::{Rhs, VarId};
    use std::collections::{BTreeSet, HashMap};
    let join = |p: &str, n: &str| if p.is_empty() { n.to_string() } else { format!("{p}.{n}") };
    let mut known: HashMap<(presk_core::FuncId, usize), BTreeSet<String>> = HashMap::new();
    let mut out: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    if core.func(entry).params.is_empty() {
        return out;
    }
    known.insert((entry, 0), BTreeSet::from([String::new()]));
    let mut work = vec![entry];
    while let Some(f) = work.pop() {
        let func = core.func(f);
        let mut vars: HashMap<VarId, BTreeSet<String>> = HashMap::new();
        for (i, p) in func.params.iter().enumerate() {
            if let Some(ps) = known.get(&(f, i)) {
                vars.insert(p.var, ps.clone());
            }
        }
        let mut closures: HashMap<VarId, (presk_core::FuncId, Vec<VarId>)> = HashMap::new();
        let mut pass: Vec<(presk_core::FuncId, usize, BTreeSet<String>)> = Vec::new();
        for l in &func.body.lets {
            let read = match &l.rhs {
                Rhs::Field(x, n) => vars.get(x).map(|ps| ps.iter().map(|p| join(p, n)).collect::<BTreeSet<_>>()),
                Rhs::Proj(x, i) => vars.get(x).map(|ps| ps.iter().map(|p| join(p, &i.to_string())).collect::<BTreeSet<_>>()),
                Rhs::Call { f: g, args, .. } => {
                    pass.extend(args.iter().enumerate().filter_map(|(k, a)| vars.get(a).map(|ps| (*g, k, ps.clone()))));
                    None
                }
                Rhs::Fold { body, stacked, init, extra, index, .. } => {
                    let off = usize::from(*index);
                    let ins = stacked.iter().chain(std::iter::once(init)).chain(extra);
                    pass.extend(ins.enumerate().filter_map(|(k, a)| vars.get(a).map(|ps| (*body, off + k, ps.clone()))));
                    None
                }
                Rhs::Closure { f: g, captured } => {
                    closures.insert(l.var, (*g, captured.clone()));
                    None
                }
                Rhs::Apply { fun, args } => {
                    if let Some((g, cap)) = closures.get(fun) {
                        pass.extend(cap.iter().chain(args).enumerate().filter_map(|(k, a)| vars.get(a).map(|ps| (*g, k, ps.clone()))));
                    }
                    None
                }
                _ => None,
            };
            if let Some(ps) = read {
                out.entry(format!("{}/%{}", func.name, l.var.0)).or_default().extend(ps.iter().cloned());
                vars.insert(l.var, ps);
            }
        }
        for (g, k, ps) in pass {
            let e = known.entry((g, k)).or_default();
            let before = e.len();
            e.extend(ps);
            if e.len() > before {
                work.push(g);
            }
        }
    }
    out
}

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
        Roles(roles, Vec::new())
    }

    pub fn role(&self, site: &str) -> Role {
        self.0.get(site).cloned().unwrap_or(Role::Decision)
    }

    /// These roles with the parameters the solver offers besides the sites.
    pub fn with_params(self, params: Vec<ParamSite>) -> Roles {
        Roles(self.0, params)
    }
}

/// The sites the solver offers, split by role: the decisions, the nests
/// first and then the parameters, and each follower with the decisions it
/// follows (indices into the first) and whether it follows a site outside
/// them too, which stays in the realization's default type. Carried sites
/// are left out, and so are the field sites that read an offered
/// parameter: their rounding is the parameter's.
struct Split {
    decisions: Vec<Site>,
    followers: Vec<(Site, Vec<usize>, bool)>,
    /// For each decision, the followers it leads.
    led: Vec<Vec<usize>>,
    /// For a parameter, the nests that read it (indices into `decisions`)
    /// and whether one is outside them, in the default type.
    readers: Vec<(Vec<usize>, bool)>,
    /// For a nest, the parameters it reads.
    reads: Vec<Vec<usize>>,
    /// The estimate's rounding step (D155).
    rounding: RoundingTerm,
}

fn split(env: &Envelope, sens: &Sensitivity, roles: &Roles, rounding: RoundingTerm) -> Split {
    let mut decisions = Vec::new();
    let mut pending = Vec::new();
    let param_fields: std::collections::BTreeSet<&str> = roles.1.iter().flat_map(|p| p.fields.iter().map(String::as_str)).collect();
    for s in sites(env, sens) {
        match roles.role(&s.name) {
            Role::Decision => decisions.push(s),
            Role::Follows(_) if param_fields.contains(s.name.as_str()) => {}
            Role::Follows(leaders) => pending.push((s, leaders)),
            Role::Carried => {}
        }
    }
    let nests = decisions.len();
    for p in &roles.1 {
        decisions.push(Site { name: p.path.clone(), elems: p.elems, mag: p.mag, sens: p.sens, work: 0.0, param: true, round: p.round });
    }
    let index: BTreeMap<&str, usize> = decisions[..nests].iter().enumerate().map(|(i, s)| (s.name.as_str(), i)).collect();
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
    // A parameter's readers: the leaders of its field sites. A field read by
    // a call rather than a nest, or a parameter with no field site, has a
    // reader the solver does not see: the default type.
    let mut readers = vec![(Vec::new(), false); decisions.len()];
    let mut reads = vec![Vec::new(); decisions.len()];
    for (k, p) in roles.1.iter().enumerate() {
        let i = nests + k;
        let mut outside = p.fields.is_empty();
        for f in &p.fields {
            match roles.role(f) {
                Role::Follows(leaders) => {
                    for l in &leaders {
                        match index.get(l.as_str()) {
                            Some(j) if !readers[i].0.contains(j) => {
                                readers[i].0.push(*j);
                                reads[*j].push(i);
                            }
                            Some(_) => {}
                            None => outside = true,
                        }
                    }
                }
                _ => outside = true,
            }
        }
        readers[i].1 = outside;
    }
    Split { decisions, followers, led, readers, reads, rounding }
}

impl Split {
    /// The type a decision's value is rounded to: a nest's own; a
    /// parameter's, or the narrowest of its readers' where that is
    /// narrower, since the value a nest reads is rounded at rest and again
    /// at the nest's edge.
    fn rounded(&self, i: usize, chosen: &[Dtype], default: Dtype) -> Dtype {
        if !self.decisions[i].param {
            return chosen[i];
        }
        let (ids, outside) = &self.readers[i];
        ids.iter().map(|j| chosen[*j]).chain(outside.then_some(default)).chain(std::iter::once(chosen[i])).min_by_key(|d| d.bytes()).unwrap_or(chosen[i])
    }

    fn decision_err(&self, i: usize, chosen: &[Dtype], default: Dtype) -> f64 {
        err_at(&self.decisions[i], self.rounded(i, chosen, default), self.rounding)
    }

    /// A follower's type: the narrowest of its leaders', the default for a
    /// leader the solver does not offer.
    fn follower_type(&self, f: usize, chosen: &[Dtype], default: Dtype) -> Dtype {
        let (_, ids, outside) = &self.followers[f];
        ids.iter().map(|i| chosen[*i]).chain(outside.then_some(default)).min_by_key(|d| d.bytes()).unwrap_or(default)
    }

    fn follower_err(&self, f: usize, chosen: &[Dtype], default: Dtype) -> f64 {
        err_at(&self.followers[f].0, self.follower_type(f, chosen, default), self.rounding)
    }

    /// The estimate: every decision's term and every follower's.
    fn estimate(&self, chosen: &[Dtype], default: Dtype) -> f64 {
        let d: f64 = (0..self.decisions.len()).map(|i| self.decision_err(i, chosen, default)).sum();
        d + (0..self.followers.len()).map(|f| self.follower_err(f, chosen, default)).sum::<f64>()
    }

    /// The squares of every decision's term and every follower's: the
    /// root-sum-square estimate squared (D161).
    fn estimate_sq(&self, chosen: &[Dtype], default: Dtype) -> f64 {
        let d: f64 = (0..self.decisions.len()).map(|i| self.decision_err(i, chosen, default).powi(2)).sum();
        d + (0..self.followers.len()).map(|f| self.follower_err(f, chosen, default).powi(2)).sum::<f64>()
    }

    /// What decision `i`, the followers it leads and the parameters it
    /// reads add to the estimate.
    fn share(&self, i: usize, chosen: &[Dtype], default: Dtype) -> f64 {
        self.decision_err(i, chosen, default) + self.led[i].iter().map(|f| self.follower_err(*f, chosen, default)).sum::<f64>() + self.reads[i].iter().map(|p| self.decision_err(*p, chosen, default)).sum::<f64>()
    }

    /// A realization of `default` with these choices: the nests' in
    /// `storage`, the parameters' in `params` (a parameter in the default
    /// type is named nowhere).
    fn realization(&self, chosen: &[Dtype], default: Dtype, envelope: &str) -> Realization {
        let mut r = Realization::uniform(default);
        for (s, d) in self.decisions.iter().zip(chosen).filter(|(_, d)| **d != default) {
            if s.param {
                r.params.insert(s.name.clone(), *d);
            } else {
                r.storage.insert(s.name.clone(), *d);
            }
        }
        r.envelope = Some(envelope.to_string());
        r
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
    /// A site's name, or a parameter's path.
    pub name: String,
    /// A parameter, decided by its path (plan 0.12.0), not a nest.
    pub param: bool,
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
    /// Its term of the estimate: `s · max|fl_d(x) − x|` under the measured
    /// rounding term, `s · u(dtype) · |v|` under the unit one; 0 for a
    /// carried value.
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
    breakdown_with(env, sens, roles, r, menu, default, machine, RoundingTerm::default())
}

/// [`breakdown`] under a rounding term (D155).
#[allow(clippy::too_many_arguments)]
pub fn breakdown_with(env: &Envelope, sens: &Sensitivity, roles: &Roles, r: &Realization, menu: &Menu, default: Dtype, machine: &dyn Machine, rounding: RoundingTerm) -> Vec<SiteCost> {
    let widest = menu.0.iter().copied().max_by_key(|d| d.bytes()).unwrap_or(default);
    let sp = split(env, sens, roles, rounding);
    let chosen: Vec<Dtype> = sp.decisions.iter().map(|s| if s.param { r.params.get(&s.name) } else { r.storage.get(&s.name) }.copied().unwrap_or(r.default)).collect();
    let mut out: Vec<SiteCost> = sp
        .decisions
        .iter()
        .zip(&chosen)
        .enumerate()
        .map(|(i, (s, d))| SiteCost { name: s.name.clone(), param: s.param, role: Role::Decision, dtype: *d, elems: s.elems, mag: s.mag, sens: s.sens, err: sp.decision_err(i, &chosen, r.default), cost: cost_at(s, *d, machine), widest_err: err_at(s, widest, rounding), widest_cost: cost_at(s, widest, machine) })
        .collect();
    for (f, (s, _, _)) in sp.followers.iter().enumerate() {
        let d = sp.follower_type(f, &chosen, r.default);
        out.push(SiteCost { name: s.name.clone(), param: false, role: roles.role(&s.name), dtype: d, elems: s.elems, mag: s.mag, sens: s.sens, err: err_at(s, d, rounding), cost: 0.0, widest_err: err_at(s, widest, rounding), widest_cost: 0.0 });
    }
    for s in sites(env, sens) {
        if roles.role(&s.name) == Role::Carried {
            out.push(SiteCost { name: s.name.clone(), param: false, role: Role::Carried, dtype: r.default, elems: s.elems, mag: s.mag, sens: s.sens, err: 0.0, cost: 0.0, widest_err: 0.0, widest_cost: 0.0 });
        }
    }
    out
}

/// `s_v · u(d) · mag(v)`: how much the output moves, to first order, when a
/// site is stored in `d`; under `RoundingTerm::Measured`, `s_v · max |fl_d(x)
/// − x|`, the rounding the site's sampled values actually take (D155).
fn err_at(s: &Site, d: Dtype, rounding: RoundingTerm) -> f64 {
    match (rounding, s.round) {
        (RoundingTerm::Measured, Some(r)) => s.sens * r.at(d),
        _ => s.sens * d.unit_roundoff() * s.mag,
    }
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
    budget: Budget,
    menu: &Menu,
    default: Dtype,
    machine: &dyn Machine,
    verify: Option<&mut dyn FnMut(&Realization) -> f64>,
) -> Solution {
    solve_with(env, sens, roles, output_site, budget, menu, default, machine, verify, Options::default())
}

/// [`solve`] under [`Options`] (D155): the measured rounding term, the
/// narrowing step.
#[allow(clippy::too_many_arguments)]
pub fn solve_with(
    env: &Envelope,
    sens: &Sensitivity,
    roles: &Roles,
    output_site: &str,
    budget: Budget,
    menu: &Menu,
    default: Dtype,
    machine: &dyn Machine,
    verify: Option<&mut dyn FnMut(&Realization) -> f64>,
    options: Options,
) -> Solution {
    solve_with_heldout(env, sens, roles, output_site, budget, menu, default, machine, verify, None, options)
}

/// [`solve_with`], with `heldout` measuring a candidate on samples the
/// envelope was not calibrated on, which `Narrow::Heldout` stops early by
/// (D161).
#[allow(clippy::too_many_arguments)]
pub fn solve_with_heldout(
    env: &Envelope,
    sens: &Sensitivity,
    roles: &Roles,
    output_site: &str,
    budget: Budget,
    menu: &Menu,
    default: Dtype,
    machine: &dyn Machine,
    mut verify: Option<&mut dyn FnMut(&Realization) -> f64>,
    mut heldout: Option<&mut dyn FnMut(&Realization) -> f64>,
    options: Options,
) -> Solution {
    let out_mag = env.get(output_site).unwrap_or(1.0).max(f64::MIN_POSITIVE);
    let budget_abs = budget.absolute(out_mag);
    let sp = split(env, sens, roles, options.rounding);
    let sites = &sp.decisions;

    let widest = menu.0.iter().copied().max_by_key(|d| d.bytes()).unwrap_or(default);
    let narrowest = menu.0.iter().copied().min_by_key(|d| d.bytes()).unwrap_or(default);
    let baseline_bytes: f64 = sites.iter().map(|s| cost_at(s, widest, machine)).sum();
    let relaxation_bytes: f64 = sites.iter().map(|s| cost_at(s, narrowest, machine)).sum();

    // Propose: every site at the widest type, then narrow in order of bytes
    // saved per unit of error added (its followers' with its own), while
    // the estimate fits.
    let mut chosen: Vec<Dtype> = vec![widest; sites.len()];
    // The proposal's target (D161): the budget under the sum; under the
    // root-sum-square the budget over the margin, compared on the squares,
    // which add as the sum's terms do.
    let rss = options.estimate == Estimate::Rss;
    let target = if rss { (budget_abs / options.margin).powi(2) } else { budget_abs };
    let mut spent: f64 = if rss { sp.estimate_sq(&chosen, default) } else { sp.estimate(&chosen, default) };
    let mut moves: Vec<(f64, usize, Dtype)> = Vec::new();
    for (i, s) in sites.iter().enumerate() {
        for d in &menu.0 {
            if d.bytes() >= widest.bytes() {
                continue;
            }
            let saved = cost_at(s, widest, machine) - cost_at(s, *d, machine);
            let mut moved = chosen.clone();
            moved[i] = *d;
            let added = if rss { sp.estimate_sq(&moved, default) - sp.estimate_sq(&chosen, default) } else { sp.share(i, &moved, default) - sp.share(i, &chosen, default) };
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
        let before = if rss { spent } else { sp.share(*i, &chosen, default) };
        chosen[*i] = *d;
        let delta = if rss { sp.estimate_sq(&chosen, default) - spent } else { sp.share(*i, &chosen, default) - before };
        let saved = cost_at(&sites[*i], cur, machine) - cost_at(&sites[*i], *d, machine);
        let taken = spent + delta <= target;
        // Logged in the estimate's own units: under the root-sum-square, what
        // the move adds to it, not to its square.
        let (added, at) = if rss { ((spent + delta).max(0.0).sqrt() - spent.sqrt(), spent.sqrt()) } else { (delta, spent) };
        log.push(Move::Tried { site: sites[*i].name.clone(), to: *d, saved, added, spent: at, taken, rank: rank + 1 });
        if taken {
            spent += delta;
        } else {
            chosen[*i] = cur;
        }
    }

    let proposed_bytes: f64 = sp.cost(&chosen, machine);

    let build = |chosen: &[Dtype]| -> Realization { sp.realization(chosen, default, &env.program) };

    // Verify, and tighten while the *measured* gap exceeds the budget.
    let mut rounds = 0usize;
    let mut measured = None;
    let mut measurements = 0usize;
    if let Some(v) = verify.as_mut() {
        loop {
            let r = build(&chosen);
            let gap = v(&r);
            measurements += 1;
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

    // Narrow by measurement (D155): only from a realization that holds,
    // each step kept while the gap is inside the budget over the margin
    // (D161). The steps kept, in order, for early stopping.
    let (mut narrowed, mut narrow_passes) = (0usize, 0usize);
    let limit = budget_abs / options.margin;
    let mut kept_steps: Vec<(usize, Dtype, Dtype, Option<f64>)> = Vec::new();
    if matches!(options.narrow, Narrow::Measured | Narrow::Heldout)
        && let Some(v) = verify.as_mut()
        && measured.is_some_and(|g| g <= budget_abs)
    {
        for pass in 1..=3 {
            // Each decision one step narrower, the largest saving first.
            let mut steps: Vec<(f64, usize, Dtype)> = (0..sites.len())
                .filter_map(|i| {
                    let cur = chosen[i];
                    let next = menu.0.iter().copied().filter(|d| d.bytes() < cur.bytes()).max_by_key(|d| d.bytes())?;
                    let saved = cost_at(&sites[i], cur, machine) - cost_at(&sites[i], next, machine);
                    (saved > 0.0).then_some((saved, i, next))
                })
                .collect();
            steps.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
            if steps.is_empty() {
                break;
            }
            narrow_passes = pass;
            let mut changed = false;
            for (_, i, next) in steps {
                let cur = chosen[i];
                chosen[i] = next;
                let gap = v(&build(&chosen));
                measurements += 1;
                let kept = gap <= limit;
                log.push(Move::Narrowed { pass, site: sites[i].name.clone(), from: cur, to: next, gap, kept });
                if kept {
                    kept_steps.push((i, cur, next, measured));
                    measured = Some(gap);
                    narrowed += 1;
                    changed = true;
                } else {
                    chosen[i] = cur;
                }
            }
            if !changed {
                break;
            }
        }
    }

    // Early stopping on the held-out samples (D161): the last step kept is
    // undone while their gap is outside the threshold; the envelope's gap
    // is then the one measured before that step.
    let (mut undone, mut heldout_gap) = (0usize, None);
    if options.narrow == Narrow::Heldout
        && let Some(h) = heldout.as_mut()
        && measured.is_some()
    {
        loop {
            let g = h(&build(&chosen));
            measurements += 1;
            heldout_gap = Some(g);
            if g <= limit {
                break;
            }
            let Some((i, from, to, before)) = kept_steps.pop() else { break };
            chosen[i] = from;
            narrowed -= 1;
            undone += 1;
            measured = before;
            log.push(Move::Undone { site: sites[i].name.clone(), from: to, to: from, heldout: g });
        }
    }

    let bytes: f64 = sp.cost(&chosen, machine);
    let estimated_error: f64 = if rss { sp.estimate_sq(&chosen, default).sqrt() } else { sp.estimate(&chosen, default) };
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
        narrowed,
        narrow_passes,
        undone,
        heldout: heldout_gap,
        measurements,
        options,
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
pub fn exhaustive(env: &Envelope, sens: &Sensitivity, roles: &Roles, output_site: &str, budget: Budget, menu: &Menu, default: Dtype, machine: &dyn Machine, max_states: u64) -> Option<(Realization, f64)> {
    exhaustive_with(env, sens, roles, output_site, budget, menu, default, machine, max_states, RoundingTerm::default())
}

/// [`exhaustive`] under a rounding term (D155): the best assignment under the
/// same estimate the proposal used.
#[allow(clippy::too_many_arguments)]
pub fn exhaustive_with(env: &Envelope, sens: &Sensitivity, roles: &Roles, output_site: &str, budget: Budget, menu: &Menu, default: Dtype, machine: &dyn Machine, max_states: u64, rounding: RoundingTerm) -> Option<(Realization, f64)> {
    let out_mag = env.get(output_site).unwrap_or(1.0).max(f64::MIN_POSITIVE);
    let budget_abs = budget.absolute(out_mag);
    let sp = split(env, sens, roles, rounding);
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
    Some((sp.realization(&pick, default, &env.program), bytes))
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
