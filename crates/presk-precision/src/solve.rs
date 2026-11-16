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
//! Two rules decide, before step 1, which values are offered at all (plan
//! 0.16.1, D169); a value not offered keeps the default type:
//! - **Rule 1**: a value of one element, as the solver prices it, has no
//!   traffic to save, so narrowing it can only add error. LayerNorm's ε in
//!   `bf16` took the course logits solved at ε\* to 1.90 ε\* on one window
//!   of the Code pénal (D168).
//! - **Rule 2**: a value whose first step down the menu saves less than
//!   [`Options::min_gain`] of the entry's all-default traffic is not offered
//!   ([`Offers`]). The course certificate's tail out of sample was a bias in
//!   `bf16` worth 4.6 KB of 5.79·10⁷ bytes (steps 0, 0b and 0d after
//!   0.16.0). The first step's gain is a value's smallest, so a value is
//!   offered its whole menu or none of it, and every decision taken saves at
//!   least the threshold.
//!
//! Two options change steps 1 and 3 (D155). The first has been the default
//! since 0.16.0 (D158); the second is off by default:
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
//! # One menu (plan 0.17.0 step 4, D177)
//!
//! A value is assigned a [`Rung`]: a float storage type, or for a parameter
//! of rank two or more as declared, `int8` or `int4` under `absmax` or
//! `pow2` scales ([`Menu`]'s second list, `presk solve --menu`). A
//! parameter's integer rung is priced at rest as its float types are
//! ([`IntRung`]): its term `sens · max |ŵ − w|`, the largest move of its
//! values quantized, computed exactly, D155's form; its bytes the integers
//! and their `f32` scales, which are constant along the axis its readers
//! contract ([`contracted_axes`]). Every move of the solver runs over one
//! list of rungs: the proposal by saving per unit of term, the
//! measurement's widening one rung at a time, the narrowing step, the
//! exhaustive search. Rule 2 reads the first float step. Without integer
//! rungs in the menu, nothing moves (QM0); `--quant` stays the two-pass
//! solver of [`crate::quant`].
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
use std::sync::Arc;

use presk_format::{Affine, Format};
use presk_rt::{AxisRef, Dtype, Quant, QuantScheme, Realization, Scale, ScaleRule};

use crate::envelope::{Envelope, Rounding};
use crate::formats::{self, Refusal, Seen};
use crate::sensitivity::Sensitivity;
use crate::target::{Kind, Target};

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

/// What a site may be assigned: the float storage types, and the integer
/// rungs, `(bits, scheme)`, a parameter of rank two or more is offered
/// besides them (plan 0.17.0 step 4, D177); and a machine's description
/// when the menu is its (`presk solve --target`, plan 0.18.0 step 5, D208),
/// which adds the affine formats an activation is offered, says which
/// floats a parameter is, and what every nest computes in.
#[derive(Clone, Debug, PartialEq)]
pub struct Menu(pub Vec<Dtype>, pub Vec<(u8, QuantScheme)>, pub Option<Arc<Target>>);

impl Default for Menu {
    /// The float storage types, the menu since 0.5.0, and no integer rung:
    /// `--quant` quantizes in a pass after the solve (0.6.0, D44, D49), and
    /// `presk solve --menu` offers the integers in the solve itself (D177).
    fn default() -> Menu {
        Menu(vec![Dtype::F64, Dtype::F32, Dtype::BF16], Vec::new(), None)
    }
}

impl Menu {
    /// `--menu`'s list, comma-separated: `f64`, `f32`, `bf16`, and `int8`
    /// or `int4`, each under both schemes, or one of them, `int8/absmax`,
    /// `int4/pow2`. The float types are put widest first and the integer
    /// rungs `int8` before `int4`, `absmax` before `pow2`, in whatever order
    /// they are written, so that one menu solves one way; it needs a float
    /// type. A float type is read by its format's name, any float format
    /// since plan 0.18.0 step 5 (D208; `E7004` until then): a value in a
    /// format no backend computes in is measured on the kernel interpreter.
    pub fn parse(s: &str) -> Result<Menu, String> {
        let (mut floats, mut ints) = (Vec::new(), Vec::new());
        for w in presk_format::split_names(s) {
            let (bits, schemes): (u8, &[QuantScheme]) = match w {
                "int8" => (8, &[QuantScheme::Absmax, QuantScheme::Pow2]),
                "int4" => (4, &[QuantScheme::Absmax, QuantScheme::Pow2]),
                "int8/absmax" => (8, &[QuantScheme::Absmax]),
                "int8/pow2" => (8, &[QuantScheme::Pow2]),
                "int4/absmax" => (4, &[QuantScheme::Absmax]),
                "int4/pow2" => (4, &[QuantScheme::Pow2]),
                other => match presk_format::Float::parse(other) {
                    Ok(f) => {
                        floats.push(Dtype::new(f));
                        continue;
                    }
                    Err(_) => return Err(format!("`{other}` is not a rung: a float format's name (f64, f32, bf16, f16, …), int8, int4, int8/absmax, int8/pow2, int4/absmax or int4/pow2")),
                },
            };
            ints.extend(schemes.iter().map(|s| (bits, *s)));
        }
        sort_floats(&mut floats);
        let order = |s: &QuantScheme| matches!(s, QuantScheme::Pow2);
        ints.sort_by_key(|(b, s)| (std::cmp::Reverse(*b), order(s)));
        ints.dedup();
        if floats.is_empty() {
            return Err("a menu needs a float type: the values that are not parameters of rank two or more have no other".into());
        }
        Ok(Menu(floats, ints, None))
    }

    /// A machine's menu (D208): its storage and its parameters' floats, its
    /// integer weights, and the description, which says the rest.
    pub fn of_target(t: Target) -> Result<Menu, String> {
        let mut floats: Vec<Dtype> = t.storage.iter().chain(t.param_floats()).copied().collect();
        sort_floats(&mut floats);
        let ints = t.weights()?;
        Ok(Menu(floats, ints, Some(Arc::new(t))))
    }

    /// The type a value the solver does not offer keeps: `f64`, or the
    /// target's widest storage format.
    pub fn default_type(&self) -> Dtype {
        self.2.as_ref().map_or(Dtype::F64, |t| t.widest())
    }

    /// Every rung, the float types first, then the integer weights, then
    /// the target's affine formats.
    pub fn rungs(&self) -> impl Iterator<Item = Rung> + '_ {
        let affine = self.2.iter().flat_map(|t| t.scaled.iter()).filter_map(|s| s.resolve(1.0, Some((-1.0, 1.0))).ok().map(|a| Rung::Affine { int: a.int, rule: s.scheme }));
        self.0.iter().map(|d| Rung::Float(*d)).chain(self.1.iter().map(|(bits, scheme)| Rung::Int { bits: *bits, scheme: *scheme })).chain(affine)
    }

    /// How many rungs it has.
    pub fn len(&self) -> usize {
        self.rungs().count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Floats widest first, each once.
fn sort_floats(floats: &mut Vec<Dtype>) {
    floats.sort_by_key(|d| (std::cmp::Reverse(d.bytes()), std::cmp::Reverse(d.format().p)));
    floats.dedup();
}

/// A step of a value's menu (plan 0.17.0 step 4, D177): a float storage
/// type, or a parameter's integers of `bits` bits under a scheme, with a
/// scale per index of every axis but the one its readers contract. The
/// integers are dequantized to `f32(q·s)` as a nest reads them
/// (`presk_rt::quant`): a storage type, as `bf16` is, with a scale. The
/// rungs live here, not in `Dtype`, since no value is computed in them.
/// Under a target (D208), an activation's affine format of `int` codes, its
/// scale and zero point computed by `rule` from the envelope at its site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rung {
    Float(Dtype),
    Int { bits: u8, scheme: QuantScheme },
    Affine { int: presk_format::Int, rule: ScaleRule },
}

impl Rung {
    /// Its name: a float type's, or the realization's for a `quant` entry,
    /// `int8`, `int4/pow2` ([`Quant::name`]), or `affine(u8/sat)/minmax`.
    pub fn name(self) -> String {
        match self {
            Rung::Float(d) => d.name().to_string(),
            Rung::Int { bits, scheme } => Quant { bits, over: AxisRef::Pos(-2), scheme }.name(),
            Rung::Affine { int, rule } => format!("affine({int})/{}", rule.name()),
        }
    }

    pub fn float(self) -> Option<Dtype> {
        match self {
            Rung::Float(d) => Some(d),
            Rung::Int { .. } | Rung::Affine { .. } => None,
        }
    }

    /// The type a nest reads the value in: a float type, or `f32` for the
    /// integers, which it dequantizes to. A machine prices the traffic of a
    /// rung under it.
    pub fn read_as(self) -> Dtype {
        self.float().unwrap_or(Dtype::F32)
    }
}

impl From<Dtype> for Rung {
    fn from(d: Dtype) -> Rung {
        Rung::Float(d)
    }
}

impl std::fmt::Display for Rung {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name())
    }
}

/// A parameter's integer rung, priced at rest (D177): the scale's axis, the
/// largest move of quantizing the values `max |ŵ − w|`, computed exactly,
/// `ŵ` the dequantized `f32(q·s)` a nest reads, and the bytes stored, the
/// integers and their `f32` scales. Its term of the estimate is `sens ·
/// round`, the form of a float type's at rest (D155).
#[derive(Clone, Debug, PartialEq)]
pub struct IntRung {
    pub bits: u8,
    pub scheme: QuantScheme,
    /// The axis the scale is constant along, counted from the end: the one
    /// every reader of the parameter contracts when `contracted`, else −2,
    /// the axis of `--quant`.
    pub axis: i64,
    pub contracted: bool,
    pub round: f64,
    pub bytes: f64,
}

impl IntRung {
    pub fn rung(&self) -> Rung {
        Rung::Int { bits: self.bits, scheme: self.scheme }
    }

    /// The realization's `quant` entry.
    pub fn quant(&self) -> Quant {
        Quant { bits: self.bits, over: AxisRef::Pos(self.axis), scheme: self.scheme }
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
    /// Rule 2's θ (plan 0.16.1, D169): a value is offered only when its
    /// first step down the menu saves at least this fraction of the entry's
    /// all-default traffic, in bytes whatever the machine. 10⁻³ by default;
    /// 0 turns rule 2 off. Rule 1 has no option.
    pub min_gain: f64,
    /// The range rules' margin (plan 0.18.0 step 5, D208): a float format is
    /// not offered a value whose largest magnitude, times it, passes the
    /// format's largest finite number, nor a sum whose terms or products do.
    /// 2 by default, a binade.
    pub range_margin: f64,
}

impl Default for Options {
    fn default() -> Options {
        Options { rounding: RoundingTerm::default(), narrow: Narrow::default(), estimate: Estimate::default(), margin: 2.5, heldout: None, min_gain: 1e-3, range_margin: 2.0 }
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
    /// The smallest gap any of them measured (D208): under a target, what
    /// the solver reports when no realization in the target's types is
    /// inside the budget.
    pub smallest: Option<f64>,
    /// Under a target, the values no realization in its types can hold
    /// (D208): a sum whose operands no mode takes, a format the range rules
    /// refuse and no other format replaces. None is measured then.
    pub infeasible: Vec<String>,
    /// The options it was solved under.
    pub options: Options,
    /// Sites the solver could narrow at all (the rest had no magnitude, no
    /// sensitivity or no element count and were left at the default).
    /// Since 0.16.1 the values the rules do not offer are among them, at
    /// the default type, and named in `offers`.
    pub considered: usize,
    /// What the two rules of 0.16.1 left out, and the base of rule 2.
    pub offers: Offers,
    /// The bytes the realization moves, at the same counts as the cost but
    /// in bytes under any machine: what `offers`' forgone traffic is a
    /// share of.
    pub traffic: f64,
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
    Tried { site: String, to: Rung, saved: f64, added: f64, spent: f64, taken: bool, rank: usize },
    /// The measured `gap` exceeded the budget: `site`, the narrowed decision
    /// with the largest share of the estimate, widened from `from` to `to`
    /// at round `round` (1 the first).
    Widened { round: usize, site: String, from: Rung, to: Rung, gap: f64 },
    /// The narrowing step (D155): `site` narrowed from `from` to `to` at
    /// pass `pass` (1 the first), the realization measured at `gap`, and
    /// `kept` when that is inside the budget, else undone.
    Narrowed { pass: usize, site: String, from: Rung, to: Rung, gap: f64, kept: bool },
    /// Early stopping (D161): the realization measured `heldout` on the
    /// held-out samples, outside the threshold, so the last narrowing kept,
    /// `site`'s from `from` to `to`, is undone.
    Undone { site: String, from: Rung, to: Rung, heldout: f64 },
    /// Under a target (D208), when no decision widens inside its modes:
    /// the contraction `site` put in `mode`, its operands with it, at round
    /// `round`, measured at `gap` and `kept` when that is smaller than
    /// before, else undone.
    Switched { round: usize, site: String, mode: String, gap: f64, kept: bool },
}

impl Solution {
    pub fn speedup(&self) -> f64 {
        if self.bytes > 0.0 { self.baseline_bytes / self.bytes } else { 1.0 }
    }
}

/// Why a value is not offered (plan 0.16.1, D169).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    /// Rule 1: it has one element.
    OneElement,
    /// Rule 2: its first step down the menu saves less than θ·B.
    MinGain,
}

/// A value the rules do not offer: it keeps the default type.
#[derive(Clone, Debug, PartialEq)]
pub struct Excluded {
    /// A site's name, or a parameter's path.
    pub name: String,
    pub param: bool,
    pub rule: Rule,
    /// Its elements, as the solver prices them (one sample's, D157).
    pub elems: f64,
    /// The bytes its first step down the menu would save: rule 2's gain.
    pub gain: f64,
    /// The bytes it would save in the menu's narrowest type: the most the
    /// rule can cost.
    pub forgone: f64,
}

/// Which values the solver offers (plan 0.16.1, D169).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Offers {
    /// B: the entry's all-default traffic, every decision's elements times
    /// the default type's bytes, the values not offered included.
    pub base: f64,
    /// θ, the options' `min_gain`; a value is offered when its first step
    /// saves at least θ·B.
    pub min_gain: f64,
    /// The values not offered, in the order of the decisions.
    pub excluded: Vec<Excluded>,
    /// The formats the range rules do not offer a value (plan 0.18.0 step
    /// 5, D208), by decision.
    pub refused: Vec<Refused>,
}

/// A format the range rules do not offer a value.
#[derive(Clone, Debug, PartialEq)]
pub struct Refused {
    pub name: String,
    pub param: bool,
    pub rung: Rung,
    pub why: Refusal,
}

impl Offers {
    /// θ·B, in bytes.
    pub fn threshold(&self) -> f64 {
        self.min_gain * self.base
    }

    /// The bytes the rules forgo at most: what the values not offered would
    /// save in the menu's narrowest type.
    pub fn forgone(&self) -> f64 {
        self.excluded.iter().map(|e| e.forgone).sum()
    }

    pub fn by(&self, rule: Rule) -> impl Iterator<Item = &Excluded> {
        self.excluded.iter().filter(move |e| e.rule == rule)
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
    /// A parameter's integer rungs, priced at rest (D177); none for a nest.
    ints: Vec<IntRung>,
    /// Its smallest non-zero magnitude and its range, as the envelope saw
    /// them, for the formats' terms and the range rules (D208).
    floor: f64,
    range: Option<(f64, f64)>,
}

impl Site {
    fn seen(&self) -> Seen {
        Seen { mag: self.mag, floor: self.floor, range: self.range }
    }
}

impl Site {
    fn int(&self, bits: u8, scheme: QuantScheme) -> Option<&IntRung> {
        self.ints.iter().find(|q| q.bits == bits && q.scheme == scheme)
    }
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
            let (floor, range) = (env.floor(name).unwrap_or(0.0), env.range.get(name).map(|r| (r[0], r[1])));
            (elems > 0.0 && *mag > 0.0).then(|| Site { name: name.clone(), elems, mag: *mag, sens, work, param: false, round: env.rounding.get(name).cloned(), ints: Vec::new(), floor, range })
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
/// the parameters it offers come with them ([`Roles::with_params`]), and
/// since 0.18.0 what the formats' terms and the range rules read of each
/// stored value ([`Facts`]).
#[derive(Clone, Debug, Default)]
pub struct Roles(pub BTreeMap<String, Role>, pub Vec<ParamSite>, pub BTreeMap<String, Facts>);

/// What the formats' terms and the range rules read of a stored value off
/// the program (plan 0.18.0 step 5, D208).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Facts {
    /// For a sum or a contraction, the terms it adds, read off the types at
    /// the envelope's shapes ([`crate::propagate::Analysis::sums`]).
    pub k: Option<f64>,
    /// The stored values a sum or a contraction reads its terms from, per
    /// operand in order: every value the operand may be, followed through
    /// the calls, folds and closures that pass it ([`value_sources`]), a
    /// function called from several places reading each caller's.
    pub operands: Vec<Vec<String>>,
    /// The roundings a term's product takes: a contraction's operands less
    /// one, one for a sum of squares, none for a plain sum.
    pub products: f64,
    /// Whether a division divides by the value as it is stored.
    pub divisor: bool,
    /// The divisors its nest computes, its group's members a division
    /// divides by, by site: the envelope has their smallest values, which
    /// its arithmetic format must hold (AdamW's `√v̂ + ε`).
    pub divides: Vec<String>,
    /// Whether the value is a constant: a literal, or a host's number made
    /// real (`real eps`).
    pub constant: bool,
    /// The constants its nest computes, its group's members that are, by
    /// site: the envelope has their values.
    pub consts: Vec<String>,
}

/// A parameter the solver offers (plan 0.12.0 step 4, D123): a leaf of the
/// entry's parameter tree, decided by its path as the realization's
/// `params` keys it, rounded once at rest to its type and read by the
/// nests that read its field sites. Its term of the estimate is
/// `sens · u · mag` at the narrowest of its type and its readers' types:
/// the value a nest reads is rounded there, at rest or at the nest's edge.
/// In integers (D177) the term is the larger of the integers' move and the
/// narrowest reader's rounding.
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
    /// Its integer rungs, those of the menu (D177): none for a parameter of
    /// rank under two as declared, or without `--menu`'s integers.
    pub ints: Vec<IntRung>,
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

/// The stored values each variable of the program may be (plan 0.18.0 step
/// 5, D208), by its site: a `let` that computes a value (a primitive, a
/// literal, a parameter's field) is its own; a tuple's projection the
/// component's where the tuple is in sight, else every component's; a
/// function's parameter every value its calls, folds (a fold's accumulator
/// its body's result too) and closures pass it; a call's result its
/// callee's. Context-insensitive, so a function called from several places
/// reads the values of all of them, as its sites have one realization.
pub fn value_sources(core: &presk_core::Program) -> BTreeMap<String, std::collections::BTreeSet<String>> {
    use presk_core::ir::{Rhs, VarId};
    use presk_core::FuncId;
    use std::collections::{BTreeSet, HashMap};
    type Set = BTreeSet<String>;
    let mut params: HashMap<(FuncId, usize), Set> = HashMap::new();
    let mut results: HashMap<FuncId, Set> = HashMap::new();
    let mut out: BTreeMap<String, Set> = BTreeMap::new();
    let grow = |into: &mut Set, from: &Set| -> bool {
        let before = into.len();
        into.extend(from.iter().cloned());
        into.len() > before
    };
    for _ in 0..64 {
        let mut changed = false;
        for (fi, f) in core.funcs.iter().enumerate() {
            let fid = FuncId(fi as u32);
            let name = |v: VarId| format!("{}/%{}", f.name, v.0);
            let mut env: HashMap<VarId, Set> = HashMap::new();
            for (k, p) in f.params.iter().enumerate() {
                env.insert(p.var, params.get(&(fid, k)).cloned().unwrap_or_default());
            }
            let mut tuples: HashMap<VarId, Vec<VarId>> = HashMap::new();
            let mut closures: HashMap<VarId, (FuncId, Vec<VarId>)> = HashMap::new();
            let mut pass: Vec<((FuncId, usize), Set)> = Vec::new();
            for l in &f.body.lets {
                let get = |env: &HashMap<VarId, Set>, v: &VarId| env.get(v).cloned().unwrap_or_default();
                let src: Set = match &l.rhs {
                    Rhs::Prim { .. } | Rhs::Lit(_) | Rhs::Field(..) => Set::from([name(l.var)]),
                    Rhs::Tuple(vs) => {
                        tuples.insert(l.var, vs.clone());
                        vs.iter().flat_map(|v| get(&env, v)).collect()
                    }
                    Rhs::Record(fs) => fs.iter().flat_map(|(_, v)| get(&env, v)).collect(),
                    Rhs::Proj(x, i) => match tuples.get(x).and_then(|vs| vs.get(*i)) {
                        Some(v) => get(&env, v),
                        None => get(&env, x),
                    },
                    Rhs::Call { f: g, args, .. } => {
                        pass.extend(args.iter().enumerate().map(|(k, a)| ((*g, k), get(&env, a))));
                        results.get(g).cloned().unwrap_or_default()
                    }
                    Rhs::Fold { body, stacked, init, extra, index, .. } => {
                        let off = usize::from(*index);
                        let ins = stacked.iter().chain(std::iter::once(init)).chain(extra);
                        pass.extend(ins.enumerate().map(|(k, a)| ((*body, off + k), get(&env, a))));
                        let carried = results.get(body).cloned().unwrap_or_default();
                        pass.push(((*body, off + stacked.len()), carried.clone()));
                        carried
                    }
                    Rhs::Closure { f: g, captured } => {
                        closures.insert(l.var, (*g, captured.clone()));
                        Set::new()
                    }
                    Rhs::Grad(x) => {
                        if let Some((g, cap)) = closures.get(x).cloned()
                            && let Some(gg) = core.find(&format!("{}.grad", core.func(g).name))
                        {
                            closures.insert(l.var, (gg, cap));
                        }
                        Set::new()
                    }
                    Rhs::Apply { fun, args } => match closures.get(fun).cloned() {
                        Some((g, cap)) => {
                            pass.extend(cap.iter().chain(args).enumerate().map(|(k, a)| ((g, k), get(&env, a))));
                            results.get(&g).cloned().unwrap_or_default()
                        }
                        None => Set::new(),
                    },
                };
                env.insert(l.var, src);
            }
            for (key, set) in pass {
                changed |= grow(params.entry(key).or_default(), &set);
            }
            let result = env.get(&f.body.result).cloned().unwrap_or_default();
            changed |= grow(results.entry(fid).or_default(), &result);
            for p in &f.params {
                out.insert(name(p.var), env.get(&p.var).cloned().unwrap_or_default());
            }
            for l in &f.body.lets {
                out.insert(name(l.var), env.get(&l.var).cloned().unwrap_or_default());
            }
        }
        if !changed {
            break;
        }
    }
    out
}

/// The axis each parameter's readers contract, counted from the end (plan
/// 0.17.0 step 4, D177), which its integers' scale is constant along so
/// that the scale factors out of the sum: a `dot`'s or `matmul`'s left
/// operand contracts its last axis and its right operand its first, an
/// `ein`'s operand its one label the output does not keep. A path is named
/// only when every reader of each field site reading it is one of those,
/// all along the same axis; a gather (`embed`), an elementwise reader, a
/// call, a returned field or two different axes leave it out, and its
/// integers then scale over −2, as `--quant`'s do. Counted from the end,
/// the axis is the same whether a fold slices the leaf's `stack{n}` axis
/// off or not.
pub fn contracted_axes(core: &presk_core::Program, entry: presk_core::FuncId) -> BTreeMap<String, i64> {
    use presk_core::ir::{PrimOp, Rhs};
    let funcs: BTreeMap<&str, &presk_core::Func> = core.funcs.iter().map(|f| (f.name.as_str(), f)).collect();
    // The axis a field site's readers contract, if one.
    let site_axis = |site: &str| -> Option<i64> {
        let (name, var) = site.rsplit_once("/%")?;
        let v: u32 = var.parse().ok()?;
        let f = funcs.get(name)?;
        let l = f.body.lets.iter().find(|l| l.var.0 == v)?;
        let rank = match &l.ty {
            presk_types::ty::Ty::Real(s) if s.prefix.is_none() => s.axes.len() as i64,
            _ => return None,
        };
        if f.body.result == l.var {
            return None;
        }
        let mut axis = None;
        for r in f.body.lets.iter().filter(|r| r.rhs.vars().contains(&l.var)) {
            let here: Vec<i64> = match &r.rhs {
                Rhs::Prim { op: PrimOp::Dot | PrimOp::Matmul, args } => args.iter().enumerate().filter(|(_, a)| **a == l.var).map(|(k, _)| if k == 0 { -1 } else { -rank }).collect(),
                Rhs::Prim { op: PrimOp::Ein { operands, output }, args } => {
                    let mut v = Vec::new();
                    for (k, _) in args.iter().enumerate().filter(|(_, a)| **a == l.var) {
                        let labels = operands.get(k)?;
                        let summed: Vec<usize> = labels.iter().enumerate().filter(|(_, n)| !output.contains(n)).map(|(j, _)| j).collect();
                        match summed[..] {
                            [j] => v.push(j as i64 - labels.len() as i64),
                            _ => return None,
                        }
                    }
                    v
                }
                _ => return None,
            };
            for a in here {
                match axis {
                    None => axis = Some(a),
                    Some(b) if b == a => {}
                    Some(_) => return None,
                }
            }
        }
        axis
    };
    let mut sites: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (site, paths) in param_reads(core, entry) {
        for p in paths {
            sites.entry(p).or_default().push(site.clone());
        }
    }
    sites
        .into_iter()
        .filter_map(|(path, ss)| {
            let axes: Option<Vec<i64>> = ss.iter().map(|s| site_axis(s)).collect();
            match axes?.as_slice() {
                [a, rest @ ..] if rest.iter().all(|b| b == a) => Some((path, *a)),
                _ => None,
            }
        })
        .collect()
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
        Roles(roles, Vec::new(), Roles::facts(core))
    }

    /// Every stored value's [`Facts`] but its sum's terms: a sum's or a
    /// contraction's operands, the values a divisor reads, and each group's
    /// literals, which its root computes.
    fn facts(core: &presk_core::Program) -> BTreeMap<String, Facts> {
        use presk_core::ir::{Lit, PrimOp, Rhs, VarId};
        use std::collections::HashMap;
        let mut out: BTreeMap<String, Facts> = BTreeMap::new();
        let sources = value_sources(core);
        for f in &core.funcs {
            let plan = presk_core::fusion::plan(f);
            let name = |v: VarId| format!("{}/%{}", f.name, v.0);
            // The stored values a variable may be: its own site for a let
            // that stores one, its sources otherwise.
            let of = |v: VarId| -> Vec<String> {
                let site = name(v);
                match sources.get(&site) {
                    Some(s) if !s.is_empty() => s.iter().cloned().collect(),
                    _ => vec![site],
                }
            };
            let rhs: HashMap<VarId, &Rhs> = f.body.lets.iter().map(|l| (l.var, &l.rhs)).collect();
            let mut root_of: HashMap<VarId, VarId> = HashMap::new();
            for g in plan.groups.values() {
                for m in &g.members {
                    root_of.insert(*m, g.root);
                }
            }
            for l in &f.body.lets {
                match &l.rhs {
                    Rhs::Prim { op: PrimOp::Dot | PrimOp::Matmul | PrimOp::Ein { .. } | PrimOp::DotVjpL | PrimOp::DotVjpR, args } => {
                        let e = out.entry(name(l.var)).or_default();
                        e.operands = args.iter().map(|a| of(*a)).collect();
                        e.products = (args.len() as f64 - 1.0).max(0.0);
                    }
                    Rhs::Prim { op: PrimOp::Sum { .. } | PrimOp::SumAll | PrimOp::SumTo | PrimOp::Mean { .. } | PrimOp::MeanAll | PrimOp::SumSquares, args } => {
                        let e = out.entry(name(l.var)).or_default();
                        e.operands = args.iter().take(1).map(|a| of(*a)).collect();
                        e.products = if matches!(l.rhs, Rhs::Prim { op: PrimOp::SumSquares, .. }) { 1.0 } else { 0.0 };
                    }
                    // A divisor stored, or computed in its group's arithmetic
                    // (LayerNorm's root, AdamW's `√v̂ + ε`).
                    Rhs::Prim { op: PrimOp::Div, args } if args.len() == 2 => match root_of.get(&args[1]).filter(|_| plan.interior.contains(&args[1]) && matches!(rhs.get(&args[1]), Some(Rhs::Prim { .. }))) {
                        Some(root) => out.entry(name(*root)).or_default().divides.push(name(args[1])),
                        None => {
                            for site in of(args[1]) {
                                out.entry(site).or_default().divisor = true;
                            }
                        }
                    },
                    Rhs::Lit(Lit::Float(_)) | Rhs::Prim { op: PrimOp::Real, .. } => match root_of.get(&l.var).filter(|_| plan.interior.contains(&l.var)) {
                        Some(root) => out.entry(name(*root)).or_default().consts.push(name(l.var)),
                        None => out.entry(name(l.var)).or_default().constant = true,
                    },
                    _ => {}
                }
            }
        }
        out
    }

    /// These roles with each sum's terms, by site.
    pub fn with_sums(mut self, sums: &BTreeMap<String, f64>) -> Roles {
        for (site, k) in sums {
            self.2.entry(site.clone()).or_default().k = Some(*k);
        }
        self
    }

    pub fn role(&self, site: &str) -> Role {
        self.0.get(site).cloned().unwrap_or(Role::Decision)
    }

    /// These roles with the parameters the solver offers besides the sites.
    pub fn with_params(self, params: Vec<ParamSite>) -> Roles {
        Roles(self.0, params, self.2)
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
    /// For each decision, whether the rules offer it ([`Split::offer`]);
    /// every one until then.
    offered: Vec<bool>,
    /// For each decision, whether a division divides by it, the smallest
    /// values of the divisors its nest computes, and whether it is a
    /// constant (D208).
    divisor: Vec<bool>,
    divides: Vec<Vec<f64>>,
    constant: Vec<bool>,
    /// For each decision, the rungs the range rules refuse it, and why.
    refusals: Vec<Vec<(Rung, Refusal)>>,
    /// Under a target, what its terms read of the program (D208).
    ctx: Option<Ctx>,
}

/// What a target's terms read of the program (plan 0.18.0 step 5, D208).
struct Ctx {
    t: Arc<Target>,
    /// The range rules' margin.
    margin: f64,
    /// For each decision that is a sum or a contraction, its terms.
    sums: Vec<Option<SumAt>>,
    /// For each decision, the sums whose operand it is.
    readers: Vec<Vec<usize>>,
    /// For each nest, its affine rungs, their formats computed from the
    /// envelope at its site.
    affines: Vec<Vec<(Rung, Affine)>>,
    /// For each decision, the constants its nest computes, their values.
    consts: Vec<Vec<f64>>,
    /// Every value the entry stores, which the realization gives the
    /// target's arithmetic.
    stored: Vec<String>,
}

/// A sum as the target's terms read it.
struct SumAt {
    k: f64,
    /// `Σ|terms|`, measured, or `k` times its largest product.
    abs: f64,
    /// Its largest product, its operands' largest magnitudes multiplied.
    largest: f64,
    products: f64,
    /// Per operand, every stored value it may be.
    operands: Vec<Vec<Operand>>,
}

/// Where a sum's operand is stored.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Operand {
    /// A decision of the solver.
    Decision(usize),
    /// A value the solver does not see, in the default type.
    Default,
    /// A member of the sum's own group, computed in its arithmetic.
    Member,
}

fn split(env: &Envelope, sens: &Sensitivity, roles: &Roles, rounding: RoundingTerm, menu: &Menu, margin: f64) -> Split {
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
        decisions.push(Site { name: p.path.clone(), elems: p.elems, mag: p.mag, sens: p.sens, work: 0.0, param: true, round: p.round.clone(), ints: p.ints.clone(), floor: 0.0, range: None });
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
    let n = decisions.len();
    let offered = vec![true; n];
    let divisor: Vec<bool> = decisions.iter().map(|s| !s.param && roles.2.get(&s.name).is_some_and(|f| f.divisor)).collect();
    let constant: Vec<bool> = decisions.iter().map(|s| !s.param && roles.2.get(&s.name).is_some_and(|f| f.constant)).collect();
    let divides: Vec<Vec<f64>> = decisions.iter().map(|s| roles.2.get(&s.name).filter(|_| !s.param).map(|f| f.divides.iter().filter_map(|d| env.floor(d)).collect()).unwrap_or_default()).collect();
    let ctx = menu.2.as_ref().map(|t| {
        // A field site and the parameters it reads, every one.
        let mut field_of: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for (k, p) in roles.1.iter().enumerate() {
            for f in &p.fields {
                field_of.entry(f.as_str()).or_default().push(nests + k);
            }
        }
        let affines = decisions.iter().map(|s| if s.param { Vec::new() } else { t.scaled.iter().filter_map(|sc| sc.resolve(s.mag, s.range).ok().map(|a| (Rung::Affine { int: a.int, rule: sc.scheme }, a))).collect() }).collect();
        let sums: Vec<Option<SumAt>> = decisions
            .iter()
            .map(|s| {
                let f = roles.2.get(&s.name).filter(|_| !s.param)?;
                let k = f.k?;
                // Each value an operand may be: a decision, the parameters a
                // field reads, a member of the sum's own group, a value the
                // solver does not see; one the envelope saw no real values
                // of (a key, an index) is none of them.
                let resolve = |o: &String| -> Vec<Operand> {
                    if let Some(j) = index.get(o.as_str()) {
                        return vec![Operand::Decision(*j)];
                    }
                    if let Some(js) = field_of.get(o.as_str()) {
                        return js.iter().map(|j| Operand::Decision(*j)).collect();
                    }
                    if env.elems(o) == 0 && env.get(o).is_none() {
                        return Vec::new();
                    }
                    match roles.role(o) {
                        Role::Follows(ls) if ls.len() == 1 && ls[0] == s.name => vec![Operand::Member],
                        Role::Follows(ls) if ls.len() == 1 => vec![index.get(ls[0].as_str()).map_or(Operand::Default, |j| Operand::Decision(*j))],
                        _ => vec![Operand::Default],
                    }
                };
                let operands: Vec<Vec<Operand>> = f.operands.iter().map(|os| {
                    let mut v: Vec<Operand> = os.iter().flat_map(resolve).collect();
                    v.dedup();
                    if v.is_empty() {
                        v.push(Operand::Default);
                    }
                    v
                }).collect();
                let mags: Vec<f64> = f.operands.iter().zip(&operands).map(|(os, ops)| {
                    let seen = os.iter().filter_map(|o| env.get(o)).fold(0.0, f64::max);
                    ops.iter().map(|op| match op {
                        Operand::Decision(j) => decisions[*j].mag,
                        _ => seen,
                    }).fold(seen, f64::max)
                }).collect();
                let mut largest: f64 = mags.iter().product();
                if f.products == 1.0 && mags.len() == 1 {
                    largest *= mags[0];
                }
                let abs = env.terms(&s.name).map(|t| t.abs).unwrap_or(k * largest);
                Some(SumAt { k, abs, largest, products: f.products, operands })
            })
            .collect();
        let mut readers = vec![Vec::new(); n];
        for (i, sum) in sums.iter().enumerate() {
            for o in sum.iter().flat_map(|x| x.operands.iter().flatten()) {
                if let Operand::Decision(d) = o
                    && !readers[*d].contains(&i)
                {
                    readers[*d].push(i);
                }
            }
        }
        let consts = decisions.iter().map(|s| roles.2.get(&s.name).map(|f| f.consts.iter().filter_map(|c| env.get(c)).collect()).unwrap_or_default()).collect();
        let stored = env.sites.keys().filter(|k| roles.role(k) == Role::Decision && !param_fields.contains(k.as_str())).cloned().collect();
        Ctx { t: t.clone(), margin, sums, readers, affines, consts, stored }
    });
    let mut sp = Split { decisions, followers, led, readers, reads, rounding, offered, divisor, divides, constant, refusals: Vec::new(), ctx };
    sp.refusals = (0..n).map(|i| menu.rungs().filter_map(|r| sp.refusal(i, r, margin).map(|w| (r, w))).collect()).collect();
    sp
}

/// Whether a product format counts a sum of scaled values in their unit.
fn counts(f: Format) -> bool {
    matches!(f, Format::Int(_) | Format::Fixed(_))
}

/// A site's term per unit of sensitivity in a format: the envelope's
/// measured rounding where it records it, under the measured term; the unit
/// roundoff times the magnitude in a native type, as until 0.18.0; any
/// other format's bound of [`formats::rounding`] (D208).
fn unit_err(s: &Site, f: Format, rounding: RoundingTerm) -> f64 {
    match f {
        Format::Float(x) => {
            let d = Dtype::new(x);
            match (rounding, s.round.as_ref().and_then(|r| r.at(d))) {
                (RoundingTerm::Measured, Some(r)) => r,
                _ if d.is_native() => d.unit_roundoff() * s.mag,
                _ => formats::rounding(f, s.seen()),
            }
        }
        other => formats::rounding(other, s.seen()),
    }
}

impl Split {
    /// The two rules of 0.16.1 (D169): a decision of one element is not
    /// offered, nor one whose first step down the menu, the widest float
    /// type narrower than `default`, saves less than `min_gain` of the
    /// all-default traffic. Both in bytes, at the elements the solver
    /// prices. The integer rungs (D177) do not change the first step, which
    /// is `f32`. The range rules' refusals (D208) are listed beside them.
    fn offer(&mut self, menu: &Menu, default: Dtype, min_gain: f64) -> Offers {
        let narrower = || menu.0.iter().copied().filter(|d| d.bytes() < default.bytes());
        let first = narrower().max_by_key(|d| d.bytes());
        let base: f64 = self.decisions.iter().map(|s| bytes_at(s, default.into())).sum();
        let mut excluded = Vec::new();
        for i in 0..self.decisions.len() {
            let Some(first) = first else { break };
            let s = &self.decisions[i];
            let gain = bytes_at(s, default.into()) - bytes_at(s, first.into());
            let rule = if s.elems <= 1.0 {
                Rule::OneElement
            } else if gain < min_gain * base {
                Rule::MinGain
            } else {
                continue;
            };
            let forgone = bytes_at(s, default.into()) - bytes_at(s, self.narrowest(i, menu));
            self.offered[i] = false;
            excluded.push(Excluded { name: s.name.clone(), param: s.param, rule, elems: s.elems, gain, forgone });
        }
        let refused = self.decisions.iter().zip(&self.refusals).flat_map(|(s, v)| v.iter().map(|(r, w)| Refused { name: s.name.clone(), param: s.param, rung: *r, why: w.clone() })).collect();
        Offers { base, min_gain, excluded, refused }
    }

    /// The format decision `i` stores its value in under a rung: a float's,
    /// an affine format computed at its site, none for integer weights.
    fn stored(&self, i: usize, r: Rung) -> Option<Format> {
        match r {
            Rung::Float(d) => Some(Format::Float(d.format())),
            Rung::Affine { .. } => self.ctx.as_ref()?.affines.get(i)?.iter().find(|(x, _)| *x == r).map(|(_, a)| Format::Affine(*a)),
            Rung::Int { .. } => None,
        }
    }

    /// The range rules (D208): why decision `i` is not offered a rung, if
    /// it is not. A float its values with the margin pass, a float or an
    /// affine format whose smallest normal number or step is above the
    /// smallest value a divisor reads, a format a constant of its nest
    /// rounds to zero or past its largest number in.
    fn refusal(&self, i: usize, r: Rung, margin: f64) -> Option<Refusal> {
        let s = &self.decisions[i];
        let f = self.stored(i, r)?;
        if let Some(w) = formats::refusal(f, s.seen(), margin, self.divisor[i]) {
            return Some(w);
        }
        // A constant stored in a format that does not hold it.
        if self.constant[i] && !formats::holds_constant(f, s.mag) {
            return Some(Refusal::Constant { value: s.mag });
        }
        // The format the nest computes in: the target's for it, else its
        // storage type's arithmetic type.
        let alpha = match (&self.ctx, f) {
            (Some(ctx), f) => ctx.t.arith_for(f)?,
            (None, Format::Float(x)) => Format::Float(Dtype::new(x).arith().format()),
            (None, _) => return None,
        };
        // A divisor the nest computes under its arithmetic's smallest
        // normal number.
        if let Format::Float(a) = alpha
            && let Some(floor) = self.divides[i].iter().copied().find(|x| *x < formats::normal(a))
        {
            return Some(Refusal::Divisor { floor, normal: formats::normal(a) });
        }
        self.ctx.as_ref()?.consts[i].iter().find(|c| !formats::holds_constant(alpha, **c)).map(|c| Refusal::Constant { value: *c })
    }

    /// The rungs decision `i` may take: the menu's float types, then for a
    /// parameter the menu's integer rungs it is priced in (D177); under a
    /// target, a nest's its storage formats and affine ones, a parameter's
    /// its parameters' (D208); none the range rules refuse.
    fn rungs(&self, i: usize, menu: &Menu) -> Vec<Rung> {
        let s = &self.decisions[i];
        menu.rungs()
            .filter(|r| match r {
                Rung::Float(d) => self.ctx.as_ref().is_none_or(|c| if s.param { c.t.param_floats().contains(d) } else { c.t.storage.contains(d) }),
                Rung::Int { bits, scheme } => s.int(*bits, *scheme).is_some(),
                Rung::Affine { .. } => !s.param && self.stored(i, *r).is_some(),
            })
            .filter(|r| !self.refusals.get(i).is_some_and(|v| v.iter().any(|(x, _)| x == r)))
            .collect()
    }

    /// The fewest bytes decision `i` can move: its narrowest rung.
    fn narrowest(&self, i: usize, menu: &Menu) -> Rung {
        let s = &self.decisions[i];
        self.rungs(i, menu).into_iter().min_by(|a, b| bytes_at(s, *a).total_cmp(&bytes_at(s, *b))).unwrap_or(Rung::Float(Dtype::F64))
    }

    /// The widest rung decision `i` may take, of two of the same bytes the
    /// one of the smaller term: where the proposal starts it under a target.
    fn widest(&self, i: usize, menu: &Menu, default: Dtype) -> Rung {
        let s = &self.decisions[i];
        self.rungs(i, menu).into_iter().min_by(|a, b| bytes_at(s, *b).total_cmp(&bytes_at(s, *a)).then(self.err_of(i, *a).total_cmp(&self.err_of(i, *b)))).unwrap_or(default.into())
    }

    /// The rung of decision `i` next to `cur` on the side `wider` says: the
    /// fewest bytes above its own, or the most below them, and of two rungs
    /// of the same bytes (`int8` under `absmax` and under `pow2`) the one of
    /// the smaller term. Under a target, the first such rung that leaves
    /// every sum it is an operand of in one of the target's modes.
    fn step(&self, i: usize, menu: &Menu, cur: Rung, wider: bool, chosen: &[Rung], default: Dtype) -> Option<Rung> {
        let s = &self.decisions[i];
        let b = bytes_at(s, cur);
        let key = |r: &Rung| {
            let x = bytes_at(s, *r);
            (if wider { x } else { -x }, self.err_of(i, *r))
        };
        let mut steps: Vec<Rung> = self.rungs(i, menu).into_iter().filter(|r| if wider { bytes_at(s, *r) > b } else { bytes_at(s, *r) < b }).collect();
        steps.sort_by(|x, y| {
            let (kx, ky) = (key(x), key(y));
            kx.0.total_cmp(&ky.0).then(kx.1.total_cmp(&ky.1))
        });
        if self.ctx.is_none() {
            return steps.first().copied();
        }
        steps.into_iter().find(|r| {
            let mut moved = chosen.to_vec();
            moved[i] = *r;
            self.share(i, &moved, default).is_finite()
        })
    }

    /// Decision `i`'s term per unit of sensitivity in a rung, alone: its
    /// rounding there.
    fn err_of(&self, i: usize, r: Rung) -> f64 {
        let s = &self.decisions[i];
        match self.stored(i, r) {
            Some(f @ Format::Affine(_)) => s.sens * formats::rounding(f, s.seen()),
            _ => err_at(s, r, self.rounding),
        }
    }

    /// The narrowest type the nests reading parameter `i` round it to, the
    /// default for a reader the solver does not see; `None` without one.
    fn edge(&self, i: usize, chosen: &[Rung], default: Dtype) -> Option<Dtype> {
        let (ids, outside) = &self.readers[i];
        ids.iter().map(|j| chosen[*j].read_as()).chain(outside.then_some(default)).min_by_key(|d| d.bytes())
    }

    /// A decision's term. A nest's in its own type; a parameter's in the
    /// narrowest of its type and its readers', since the value a nest reads
    /// is rounded at rest and again at the nest's edge. In integers, the
    /// larger of their move at rest and the narrowest reader's rounding of
    /// the parameter (D177): the dequantized `f32` is rounded again only by
    /// a narrower nest. Under a target, [`Split::target_err`].
    fn decision_err(&self, i: usize, chosen: &[Rung], default: Dtype) -> f64 {
        let s = &self.decisions[i];
        if self.ctx.is_some() {
            return self.target_err(i, chosen, default);
        }
        match chosen[i] {
            Rung::Float(d) if s.param => {
                let t = self.edge(i, chosen, default).into_iter().chain(std::iter::once(d)).min_by_key(|d| d.bytes()).unwrap_or(d);
                err_at(s, t.into(), self.rounding)
            }
            r @ Rung::Float(_) => err_at(s, r, self.rounding),
            r @ Rung::Int { .. } => err_at(s, r, self.rounding).max(self.edge(i, chosen, default).map_or(0.0, |d| err_at(s, d.into(), self.rounding))),
            r @ Rung::Affine { .. } => self.err_of(i, r),
        }
    }

    /// A decision's term under a target (D208). A parameter's at rest: a
    /// nest reads it as it is (D206). A nest's: one rounding of its value,
    /// in the format it is stored in, or in its result's format where that
    /// one holds it exactly (a sum's accumulator, a nest's arithmetic), a
    /// counted sum's count converted once; and a sum's own operations in
    /// the formats of the target's mode its operands' formats select, under
    /// the target's order. Infinite where its stored format has no
    /// arithmetic in the target, or its operands no mode, or the range rules
    /// refuse its sum's formats.
    fn target_err(&self, i: usize, chosen: &[Rung], default: Dtype) -> f64 {
        let s = &self.decisions[i];
        if s.param {
            return self.err_of(i, chosen[i]);
        }
        let ctx = self.ctx.as_ref().expect("a target");
        let Some(stored) = self.stored(i, chosen[i]) else { return f64::INFINITY };
        let Some(alpha) = ctx.t.arith_for(stored) else { return f64::INFINITY };
        let (result, ops) = match &ctx.sums[i] {
            None => (Some(alpha), 0.0),
            Some(sum) => match self.mode(i, chosen, default) {
                None => return f64::INFINITY,
                Some((mu, kappa)) => (if counts(mu) { None } else { Some(kappa) }, self.sum_err(sum, mu, kappa, chosen, default, ctx)),
            },
        };
        let at = match result {
            Some(f) if stored.contains(f) => f,
            _ => stored,
        };
        s.sens * (unit_err(s, at, self.rounding) + ops)
    }

    /// The formats a sum's products and additions are rounded to under the
    /// target: its mode's, selected by its operands' formats, or its
    /// arithmetic format for a plain sum or a target with no mode. `None`
    /// for no sum, or operands no mode takes.
    fn mode(&self, i: usize, chosen: &[Rung], default: Dtype) -> Option<(Format, Format)> {
        let ctx = self.ctx.as_ref()?;
        let sum = ctx.sums[i].as_ref()?;
        let alpha = ctx.t.arith_for(self.stored(i, chosen[i])?)?;
        if sum.operands.len() < 2 || ctx.t.contractions.is_empty() {
            return Some((alpha, alpha));
        }
        let kind = |o: &Operand| match o {
            Operand::Decision(d) => Some(match chosen[*d] {
                Rung::Float(x) => Kind::Float(x.format()),
                Rung::Int { .. } | Rung::Affine { .. } => Kind::Int,
            }),
            Operand::Default => Some(Kind::Float(default.format())),
            Operand::Member => match alpha {
                Format::Float(f) => Some(Kind::Float(f)),
                _ => None,
            },
        };
        // An operand's values all of one kind, which the mode takes.
        let kinds: Option<Vec<Kind>> = sum
            .operands
            .iter()
            .map(|os| {
                let ks: Option<Vec<Kind>> = os.iter().map(kind).collect();
                let ks = ks?;
                ks.windows(2).all(|w| w[0] == w[1]).then(|| ks[0])
            })
            .collect();
        ctx.t.mode_for(&kinds?).map(|m| (m.product, m.accum))
    }

    /// The name of the mode decision `i` sums in, for the report.
    fn mode_name(&self, i: usize, chosen: &[Rung], default: Dtype) -> Option<String> {
        let ctx = self.ctx.as_ref()?;
        let sum = ctx.sums[i].as_ref()?;
        if sum.operands.len() < 2 || ctx.t.contractions.is_empty() {
            return None;
        }
        let (mu, kappa) = self.mode(i, chosen, default)?;
        ctx.t.contractions.iter().find(|m| m.product == mu && m.accum == kappa).map(|m| m.label())
    }

    /// A sum's own operations' term per unit of sensitivity, in its formats
    /// ([`formats::sum_rounding`]): its products exact where its operands'
    /// formats multiply exactly in the product format, its count exact
    /// while its operands' codes keep it inside its accumulator.
    fn sum_err(&self, sum: &SumAt, mu: Format, kappa: Format, chosen: &[Rung], default: Dtype, ctx: &Ctx) -> f64 {
        let float = |o: &Operand| match o {
            Operand::Decision(d) => chosen[*d].float().map(|x| x.format()),
            Operand::Default => Some(default.format()),
            Operand::Member => None,
        };
        // An operand's float, when all its values are of one.
        let common = |os: &Vec<Operand>| -> Option<presk_format::Float> {
            let fs: Option<Vec<presk_format::Float>> = os.iter().map(float).collect();
            let fs = fs?;
            fs.windows(2).all(|w| w[0] == w[1]).then(|| fs[0])
        };
        let exact = match (mu, sum.operands.as_slice()) {
            (Format::Float(f), [a, b]) => common(a).zip(common(b)).is_some_and(|(a, b)| formats::exact_product(a, b, f)),
            _ => sum.products == 0.0,
        };
        let code = |o: &Operand| match o {
            Operand::Decision(d) => match chosen[*d] {
                r @ Rung::Affine { .. } => match self.stored(*d, r) {
                    Some(Format::Affine(a)) => Some(formats::largest_code(a)),
                    _ => None,
                },
                Rung::Int { bits, .. } => Some(((1i64 << (bits - 1)) - 1) as f64),
                Rung::Float(_) => None,
            },
            _ => None,
        };
        // An operand's largest code over all its values.
        let codes = |os: &Vec<Operand>| os.iter().map(code).try_fold(0.0f64, |m, c| c.map(|c| m.max(c)));
        let count = if counts(mu) { sum.operands.iter().map(codes).try_fold(sum.k, |n, c| c.map(|c| n * c)) } else { None };
        let s = formats::Sum { k: sum.k, abs: sum.abs, largest: sum.largest, additions: crate::propagate::order_constant(ctx.t.order, sum.k), products: (!exact).then_some(sum.products), count };
        formats::sum_rounding(mu, kappa, &s, ctx.margin)
    }

    /// Under a target, a start the proposal can narrow from: each decision
    /// at its widest rung, then each sum whose operands no mode takes given
    /// its operands the widest rungs of the mode that takes them of the
    /// smallest estimate, in three passes at most (D208).
    fn start(&self, menu: &Menu, default: Dtype, offered: &[bool]) -> Vec<Rung> {
        let n = self.decisions.len();
        let mut chosen: Vec<Rung> = (0..n).map(|i| if offered[i] || self.ctx.is_some() { self.widest(i, menu, default) } else { default.into() }).collect();
        let Some(ctx) = &self.ctx else { return chosen };
        for _ in 0..3 {
            let mut changed = false;
            for j in 0..n {
                let Some(sum) = &ctx.sums[j] else { continue };
                if self.mode(j, &chosen, default).is_some() {
                    continue;
                }
                let mut best: Option<(f64, Vec<Rung>)> = None;
                for m in &ctx.t.contractions {
                    let moved = self.put(sum, m, &chosen, menu);
                    if self.mode(j, &moved, default).is_some() {
                        let e = self.estimate(&moved, default);
                        if best.as_ref().is_none_or(|(b, _)| e < *b) {
                            best = Some((e, moved));
                        }
                    }
                }
                if let Some((_, moved)) = best {
                    chosen = moved;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        chosen
    }

    /// `chosen` with sum `j`'s operands at the widest rungs mode `m` takes.
    fn put(&self, sum: &SumAt, m: &crate::target::Mode, chosen: &[Rung], menu: &Menu) -> Vec<Rung> {
        let mut moved = chosen.to_vec();
        for (os, spec) in sum.operands.iter().zip(&m.operands) {
            let fits = |r: &Rung| match (spec.as_str(), r) {
                ("*", _) => true,
                ("int", Rung::Int { .. } | Rung::Affine { .. }) => true,
                ("int", Rung::Float(_)) => false,
                (name, Rung::Float(x)) => presk_format::Float::parse(name).is_ok_and(|g| g == x.format()),
                _ => false,
            };
            for o in os {
                let Operand::Decision(d) = *o else { continue };
                let s = &self.decisions[d];
                if let Some(r) = self.rungs(d, menu).into_iter().filter(fits).min_by(|a, b| bytes_at(s, *b).total_cmp(&bytes_at(s, *a)).then(self.err_of(d, *a).total_cmp(&self.err_of(d, *b)))) {
                    moved[d] = r;
                }
            }
        }
        moved
    }

    /// Under a target, every contraction put in another of the target's
    /// modes than its own, its operands at that mode's widest rungs, where
    /// the estimate stays finite: `(estimate, sum, mode, choices)`, the
    /// smallest estimate first (D208). The measurement's lateral moves.
    fn switches(&self, menu: &Menu, default: Dtype, chosen: &[Rung]) -> Vec<(f64, usize, String, Vec<Rung>)> {
        let Some(ctx) = &self.ctx else { return Vec::new() };
        let mut out = Vec::new();
        for (j, sum) in ctx.sums.iter().enumerate() {
            let Some(sum) = sum.as_ref().filter(|s| s.operands.len() >= 2) else { continue };
            let Some(now) = self.mode(j, chosen, default) else { continue };
            for m in ctx.t.contractions.iter().filter(|m| (m.product, m.accum) != now) {
                let moved = self.put(sum, m, chosen, menu);
                let e = self.estimate(&moved, default);
                if self.mode(j, &moved, default) == Some((m.product, m.accum)) && e.is_finite() {
                    out.push((e, j, m.label(), moved));
                }
            }
        }
        out.sort_by(|a, b| a.0.total_cmp(&b.0));
        out
    }

    /// A follower's type: the narrowest of its leaders', the default for a
    /// leader the solver does not offer. A leader is a nest, in a float
    /// type.
    fn follower_type(&self, f: usize, chosen: &[Rung], default: Dtype) -> Dtype {
        let (_, ids, outside) = &self.followers[f];
        ids.iter().map(|i| chosen[*i].read_as()).chain(outside.then_some(default)).min_by_key(|d| d.bytes()).unwrap_or(default)
    }

    /// A follower's term: at its type, or under a target at the
    /// arithmetic format of its leaders it is computed in, the largest
    /// (D208).
    fn follower_err(&self, f: usize, chosen: &[Rung], default: Dtype) -> f64 {
        let (s, ids, outside) = &self.followers[f];
        if let Some(ctx) = &self.ctx {
            let alphas = ids.iter().map(|i| self.stored(*i, chosen[*i]).and_then(|x| ctx.t.arith_for(x))).chain(outside.then(|| ctx.t.arith_for(Format::Float(default.format()))));
            return alphas.map(|a| a.map_or(f64::INFINITY, |a| s.sens * unit_err(s, a, self.rounding))).fold(0.0, f64::max);
        }
        err_at(s, self.follower_type(f, chosen, default).into(), self.rounding)
    }

    /// The estimate: every decision's term and every follower's.
    fn estimate(&self, chosen: &[Rung], default: Dtype) -> f64 {
        let d: f64 = (0..self.decisions.len()).map(|i| self.decision_err(i, chosen, default)).sum();
        d + (0..self.followers.len()).map(|f| self.follower_err(f, chosen, default)).sum::<f64>()
    }

    /// The squares of every decision's term and every follower's: the
    /// root-sum-square estimate squared (D161).
    fn estimate_sq(&self, chosen: &[Rung], default: Dtype) -> f64 {
        let d: f64 = (0..self.decisions.len()).map(|i| self.decision_err(i, chosen, default).powi(2)).sum();
        d + (0..self.followers.len()).map(|f| self.follower_err(f, chosen, default).powi(2)).sum::<f64>()
    }

    /// What decision `i`, the followers it leads and the parameters it
    /// reads add to the estimate; under a target, the sums whose operand it
    /// is too, whose mode its format selects.
    fn share(&self, i: usize, chosen: &[Rung], default: Dtype) -> f64 {
        let readers = self.ctx.as_ref().map_or(0.0, |c| c.readers[i].iter().map(|j| self.decision_err(*j, chosen, default)).sum::<f64>());
        self.decision_err(i, chosen, default) + self.led[i].iter().map(|f| self.follower_err(*f, chosen, default)).sum::<f64>() + self.reads[i].iter().map(|p| self.decision_err(*p, chosen, default)).sum::<f64>() + readers
    }

    /// A realization of `default` with these choices: the nests' in
    /// `storage`, the parameters' in `params` (a parameter in the default
    /// type is named nowhere), a parameter's integers in `quant` (D177).
    /// Under a target (D208), an activation's affine format in `scaled`,
    /// every value the entry stores computing in the target's arithmetic,
    /// each sum's products and additions in its mode's formats, and the
    /// target's order.
    fn realization(&self, chosen: &[Rung], default: Dtype, envelope: &str) -> Realization {
        let mut r = Realization::uniform(default);
        for (i, (s, rung)) in self.decisions.iter().zip(chosen).enumerate() {
            match *rung {
                Rung::Float(d) if d == default => {}
                Rung::Float(d) if s.param => {
                    r.params.insert(s.name.clone(), d);
                }
                Rung::Float(d) => {
                    r.storage.insert(s.name.clone(), d);
                }
                Rung::Int { bits, scheme } => {
                    if let Some(q) = s.int(bits, scheme) {
                        r.quant.insert(s.name.clone(), q.quant());
                    }
                }
                Rung::Affine { .. } => {
                    if let Some(Format::Affine(a)) = self.stored(i, *rung) {
                        r.scaled.insert(s.name.clone(), Scale::Static(a));
                    }
                }
            }
        }
        if let Some(ctx) = &self.ctx {
            r.order = ctx.t.order;
            if let Some(a) = ctx.t.arith_for(Format::Float(default.format())) {
                for site in &ctx.stored {
                    r.arith.insert(site.clone(), a);
                }
            }
            for (i, s) in self.decisions.iter().enumerate().filter(|(_, s)| !s.param) {
                if let Some(a) = self.stored(i, chosen[i]).and_then(|f| ctx.t.arith_for(f)) {
                    r.arith.insert(s.name.clone(), a);
                }
                if let Some((mu, kappa)) = self.mode(i, chosen, default) {
                    r.product.insert(s.name.clone(), mu);
                    r.accum.insert(s.name.clone(), kappa);
                }
            }
        }
        r.envelope = Some(envelope.to_string());
        r
    }

    fn cost(&self, chosen: &[Rung], machine: &dyn Machine) -> f64 {
        self.decisions.iter().zip(chosen).map(|(s, t)| cost_at(s, *t, machine)).sum()
    }

    fn traffic(&self, chosen: &[Rung]) -> f64 {
        self.decisions.iter().zip(chosen).map(|(s, t)| bytes_at(s, *t)).sum()
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
    /// Its rung: the realization's for a decision, the type it is rounded
    /// to for a follower, its own for a carried value, which is not rounded
    /// there. A float type but for a parameter in integers (D177).
    pub rung: Rung,
    /// The integer rung's pricing, its scale's axis among it, when `rung`
    /// is integers.
    pub int: Option<IntRung>,
    pub elems: f64,
    /// The envelope's magnitude.
    pub mag: f64,
    /// `‖∂L/∂v‖₁`.
    pub sens: f64,
    /// Its term of the estimate: `s · max|fl_d(x) − x|` under the measured
    /// rounding term, `s · u(dtype) · |v|` under the unit one, `s · max|ŵ −
    /// w|` in integers; 0 for a carried value.
    pub err: f64,
    /// What it costs under the machine, in its type; 0 but for a decision.
    pub cost: f64,
    /// The bytes it stores at rest in its rung: its elements times its
    /// type's bytes, the integers and their scales in integers.
    pub bytes: f64,
    /// The same in the widest type of the menu.
    pub widest_err: f64,
    pub widest_cost: f64,
    /// Under a target, the contraction mode its sum is computed in (D208).
    pub mode: Option<String>,
}

/// Every site the solver offers, decisions first, then the followers and
/// the carried values, under `r`.
pub fn breakdown(env: &Envelope, sens: &Sensitivity, roles: &Roles, r: &Realization, menu: &Menu, default: Dtype, machine: &dyn Machine) -> Vec<SiteCost> {
    breakdown_with(env, sens, roles, r, menu, default, machine, RoundingTerm::default())
}

/// [`breakdown`] under a rounding term (D155). A parameter `r` quantizes is
/// read in its integer rung when the solver priced it there (D177), and in
/// its float type otherwise: a `--quant` realization's breakdown is the
/// float solve's, as before.
#[allow(clippy::too_many_arguments)]
pub fn breakdown_with(env: &Envelope, sens: &Sensitivity, roles: &Roles, r: &Realization, menu: &Menu, default: Dtype, machine: &dyn Machine, rounding: RoundingTerm) -> Vec<SiteCost> {
    let widest = menu.0.iter().copied().max_by_key(|d| d.bytes()).unwrap_or(default);
    let sp = split(env, sens, roles, rounding, menu, Options::default().range_margin);
    let chosen: Vec<Rung> = sp
        .decisions
        .iter()
        .enumerate()
        .map(|(i, s)| match (r.quant.get(&s.name).filter(|_| s.param).and_then(|q| s.int(q.bits, q.scheme)), r.scaled.get(&s.name)) {
            (Some(q), _) => q.rung(),
            // An affine format the target's rungs computed at the site.
            (_, Some(Scale::Static(a))) if !s.param => sp.ctx.as_ref().and_then(|c| c.affines[i].iter().find(|(_, b)| b == a).map(|(x, _)| *x)).unwrap_or(Rung::Float(r.default)),
            _ => Rung::Float(if s.param { r.params.get(&s.name) } else { r.storage.get(&s.name) }.copied().unwrap_or(r.default)),
        })
        .collect();
    let mut out: Vec<SiteCost> = sp
        .decisions
        .iter()
        .zip(&chosen)
        .enumerate()
        .map(|(i, (s, d))| {
            let int = match d {
                Rung::Int { bits, scheme } => s.int(*bits, *scheme).cloned(),
                Rung::Float(_) | Rung::Affine { .. } => None,
            };
            SiteCost { name: s.name.clone(), param: s.param, role: Role::Decision, rung: *d, int, elems: s.elems, mag: s.mag, sens: s.sens, err: sp.decision_err(i, &chosen, r.default), cost: cost_at(s, *d, machine), bytes: bytes_at(s, *d), widest_err: err_at(s, widest.into(), rounding), widest_cost: cost_at(s, widest.into(), machine), mode: sp.mode_name(i, &chosen, r.default) }
        })
        .collect();
    for (f, (s, _, _)) in sp.followers.iter().enumerate() {
        let d = sp.follower_type(f, &chosen, r.default);
        out.push(SiteCost { name: s.name.clone(), param: false, role: roles.role(&s.name), rung: d.into(), int: None, elems: s.elems, mag: s.mag, sens: s.sens, err: sp.follower_err(f, &chosen, r.default), cost: 0.0, bytes: 0.0, widest_err: err_at(s, widest.into(), rounding), widest_cost: 0.0, mode: None });
    }
    for s in sites(env, sens) {
        if roles.role(&s.name) == Role::Carried {
            out.push(SiteCost { name: s.name.clone(), param: false, role: Role::Carried, rung: r.default.into(), int: None, elems: s.elems, mag: s.mag, sens: s.sens, err: 0.0, cost: 0.0, bytes: 0.0, widest_err: 0.0, widest_cost: 0.0, mode: None });
        }
    }
    out
}

/// `s_v · u(d) · mag(v)`: how much the output moves, to first order, when a
/// site is stored in `d`; under `RoundingTerm::Measured`, `s_v · max |fl_d(x)
/// − x|`, the rounding the site's sampled values actually take (D155). In
/// integers, `s_v · max |ŵ − w|` under either term, the move computed
/// exactly (D177).
fn err_at(s: &Site, d: Rung, rounding: RoundingTerm) -> f64 {
    match d {
        // A site is offered only the integer rungs it is priced in; one it
        // is not is charged its whole magnitude.
        Rung::Int { bits, scheme } => s.sens * s.int(bits, scheme).map_or(s.mag, |q| q.round),
        // A format no backend computes in, unmeasured: its bound (D208).
        Rung::Float(d) => s.sens * unit_err(s, Format::Float(d.format()), rounding),
        // Its format is the site's, which `Split::err_of` reads.
        Rung::Affine { .. } => s.sens * s.mag,
    }
}

fn bytes_at(s: &Site, d: Rung) -> f64 {
    match d {
        Rung::Float(d) => s.elems * d.bytes() as f64,
        Rung::Int { bits, scheme } => s.int(bits, scheme).map_or(s.elems * bits as f64 / 8.0, |q| q.bytes),
        Rung::Affine { int, .. } => s.elems * f64::from(int.bits.div_ceil(8)),
    }
}

/// What a site costs under a machine: the roofline's own prediction, with
/// the dtype in `Q` and not in `W`; integers priced as the `f32` they are
/// read in, at their own bytes.
fn cost_at(s: &Site, d: Rung, m: &dyn Machine) -> f64 {
    m.seconds(d.read_as(), s.work, bytes_at(s, d))
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
    verify: Option<&mut dyn FnMut(&Realization) -> f64>,
    mut heldout: Option<&mut dyn FnMut(&Realization) -> f64>,
    options: Options,
) -> Solution {
    let out_mag = env.get(output_site).unwrap_or(1.0).max(f64::MIN_POSITIVE);
    let budget_abs = budget.absolute(out_mag);
    let mut sp = split(env, sens, roles, options.rounding, menu, options.range_margin);
    let offers = sp.offer(menu, default, options.min_gain);
    let sp = sp;
    let sites = &sp.decisions;
    let offered = &sp.offered;
    let on_target = sp.ctx.is_some();

    let widest = menu.0.iter().copied().max_by_key(|d| d.bytes()).unwrap_or(default);
    let top = Rung::Float(widest);
    // Each decision's rungs: the float types, and a parameter's integers
    // (D177); under a target, its own and its affine formats (D208).
    let rungs: Vec<Vec<Rung>> = (0..sites.len()).map(|i| sp.rungs(i, menu)).collect();
    let baseline_bytes: f64 = sites.iter().map(|s| cost_at(s, if on_target { default.into() } else { top }, machine)).sum();
    let relaxation_bytes: f64 = sites.iter().zip(offered).enumerate().map(|(i, (s, o))| cost_at(s, if *o { sp.narrowest(i, menu) } else { default.into() }, machine)).sum();

    // Propose: every site offered at the widest type, the others at the
    // default (D169), then narrow in order of bytes saved per unit of error
    // added (its followers' with its own), while the estimate fits. Under a
    // target, each decision at its widest rung, its sums put in a mode
    // (D208).
    let start: Vec<Rung> = if on_target { sp.start(menu, default, offered) } else { offered.iter().map(|o| Rung::Float(if *o { widest } else { default })).collect() };
    let mut chosen = start.clone();
    // The proposal's target (D161): the budget under the sum; under the
    // root-sum-square the budget over the margin, compared on the squares,
    // which add as the sum's terms do.
    let rss = options.estimate == Estimate::Rss;
    let target = if rss { (budget_abs / options.margin).powi(2) } else { budget_abs };
    let mut spent: f64 = if rss { sp.estimate_sq(&chosen, default) } else { sp.estimate(&chosen, default) };
    let mut moves: Vec<(f64, usize, Rung)> = Vec::new();
    for (i, s) in sites.iter().enumerate().filter(|(i, _)| offered[*i]) {
        for d in &rungs[i] {
            if bytes_at(s, *d) >= bytes_at(s, start[i]) {
                continue;
            }
            let saved = cost_at(s, start[i], machine) - cost_at(s, *d, machine);
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
        if bytes_at(&sites[*i], *d) >= bytes_at(&sites[*i], cur) {
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

    let build = |chosen: &[Rung]| -> Realization { sp.realization(chosen, default, &env.program) };
    // Under a target, a value whose term stays infinite has no realization
    // in the target's types: nothing is measured (D208).
    let infeasible: Vec<String> = if on_target { (0..sites.len()).filter(|i| !sp.decision_err(*i, &chosen, default).is_finite()).map(|i| sites[i].name.clone()).collect() } else { Vec::new() };
    let mut verify = if infeasible.is_empty() { verify } else { None };

    // Verify, and tighten while the *measured* gap exceeds the budget.
    let mut rounds = 0usize;
    let mut measured = None;
    let mut measurements = 0usize;
    // The smallest gap measured, and its choices (D208).
    let mut best: Option<(f64, Vec<Rung>)> = None;
    let mut note = |gap: f64, chosen: &[Rung]| {
        if best.as_ref().is_none_or(|(g, _)| gap < *g) {
            best = Some((gap, chosen.to_vec()));
        }
    };
    if let Some(v) = verify.as_mut() {
        // Under a target, a decision no step widens inside the target's
        // modes is left where it is, and the contractions' other modes are
        // tried, each switch kept when it measures smaller (D208).
        let mut stuck = vec![false; sites.len()];
        let mut tried: std::collections::BTreeSet<(usize, String)> = Default::default();
        let mut pending: Option<(Vec<Rung>, f64, usize, String)> = None;
        let mut carried: Option<f64> = None;
        loop {
            let gap = match carried.take() {
                Some(g) => g,
                None => {
                    let g = v(&build(&chosen));
                    measurements += 1;
                    note(g, &chosen);
                    g
                }
            };
            let gap = match pending.take() {
                Some((prev, prev_gap, j, mode)) => {
                    let kept = gap < prev_gap;
                    log.push(Move::Switched { round: rounds, site: sites[j].name.clone(), mode, gap, kept });
                    if kept {
                        gap
                    } else {
                        chosen = prev;
                        prev_gap
                    }
                }
                None => gap,
            };
            measured = Some(gap);
            if gap <= budget_abs || rounds >= sites.len() * menu.len() {
                break;
            }
            // Widen the narrowed site whose estimated contribution, its
            // followers' with its own, is largest — by **one step** of the
            // menu, not straight to the widest. Jumping to the top undoes in
            // one round everything the proposal found, and with a flat
            // estimate (every sensitivity zero, as happens on an untrained
            // model) it walks the whole assignment back to `f64` and reports
            // no saving at all.
            let narrowed = |i: usize| if on_target { offered[i] && !stuck[i] && sp.step(i, menu, chosen[i], true, &chosen, default).is_some() } else { offered[i] && bytes_at(&sites[i], chosen[i]) < bytes_at(&sites[i], top) };
            let worst = (0..sites.len()).filter(|i| narrowed(*i)).max_by(|a, b| {
                let (ea, eb) = (sp.share(*a, &chosen, default), sp.share(*b, &chosen, default));
                ea.partial_cmp(&eb).unwrap_or(std::cmp::Ordering::Equal)
            });
            match worst {
                Some(i) => {
                    let cur = chosen[i];
                    let next = match sp.step(i, menu, cur, true, &chosen, default) {
                        Some(r) => r,
                        None if on_target => {
                            stuck[i] = true;
                            carried = Some(gap);
                            continue;
                        }
                        None => top,
                    };
                    chosen[i] = next;
                    rounds += 1;
                    log.push(Move::Widened { round: rounds, site: sites[i].name.clone(), from: cur, to: next, gap });
                }
                None if on_target => match sp.switches(menu, default, &chosen).into_iter().find(|(_, j, m, _)| !tried.contains(&(*j, m.clone()))) {
                    Some((_, j, mode, moved)) => {
                        tried.insert((j, mode.clone()));
                        pending = Some((chosen.clone(), gap, j, mode));
                        chosen = moved;
                        rounds += 1;
                    }
                    None => break,
                },
                None => break,
            }
        }
    }

    // Narrow by measurement (D155): only from a realization that holds,
    // each step kept while the gap is inside the budget over the margin
    // (D161). The steps kept, in order, for early stopping.
    let (mut narrowed, mut narrow_passes) = (0usize, 0usize);
    let limit = budget_abs / options.margin;
    let mut kept_steps: Vec<(usize, Rung, Rung, Option<f64>)> = Vec::new();
    if matches!(options.narrow, Narrow::Measured | Narrow::Heldout)
        && let Some(v) = verify.as_mut()
        && measured.is_some_and(|g| g <= budget_abs)
    {
        for pass in 1..=3 {
            // Each decision one step narrower, the largest saving first.
            let mut steps: Vec<(f64, usize, Rung)> = (0..sites.len())
                .filter(|i| offered[*i])
                .filter_map(|i| {
                    let cur = chosen[i];
                    let next = sp.step(i, menu, cur, false, &chosen, default)?;
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
                note(gap, &chosen);
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

    // Under a target, a realization measured outside its budget is not
    // returned when one inside was, nor when none was: then the one of the
    // smallest gap, which the report says is outside (D208).
    let smallest = best.as_ref().map(|(g, _)| *g);
    if on_target
        && measured.is_some_and(|g| g > budget_abs)
        && let Some((g, b)) = best.take()
    {
        chosen = b;
        measured = Some(g);
    }
    let bytes: f64 = sp.cost(&chosen, machine);
    let traffic: f64 = sp.traffic(&chosen);
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
        smallest,
        infeasible,
        options,
        considered: sites.len(),
        offers,
        traffic,
        moves: log,
    }
}

/// What the range rules offer a stored value (plan 0.18.0 step 5, D208):
/// the menu's rungs it is offered, and those refused with the rule and why;
/// what `presk solve --offers` writes, with what the rules read.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Offered {
    pub site: String,
    pub mag: f64,
    /// Its smallest non-zero magnitude, 0 where the envelope saw none.
    pub floor: f64,
    pub divisor: bool,
    /// The smallest values of the divisors its nest computes.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub divides: Vec<f64>,
    pub constant: bool,
    /// The values of the constants its nest computes.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub consts: Vec<f64>,
    pub offered: Vec<String>,
    /// `(rung, rule, why)`.
    pub refused: Vec<(String, String, String)>,
}

/// Every value the envelope saw stored, with a magnitude and elements, and
/// what the range rules offer it from the menu, without a sensitivity: the
/// rules read the envelope and the program alone.
pub fn offered(env: &Envelope, roles: &Roles, menu: &Menu, options: Options) -> Vec<Offered> {
    let sens = Sensitivity { sites: env.sites.keys().map(|k| (k.clone(), 1.0)).collect() };
    let roles = Roles(roles.0.clone(), Vec::new(), roles.2.clone());
    let sp = split(env, &sens, &roles, options.rounding, menu, options.range_margin);
    sp.decisions
        .iter()
        .enumerate()
        .map(|(i, s)| Offered {
            site: s.name.clone(),
            mag: s.mag,
            floor: s.floor,
            divisor: sp.divisor[i],
            divides: sp.divides[i].clone(),
            consts: sp.ctx.as_ref().map(|c| c.consts[i].clone()).unwrap_or_else(|| roles.2.get(&s.name).map(|f| f.consts.iter().filter_map(|c| env.get(c)).collect()).unwrap_or_default()),
            constant: sp.constant[i],
            offered: sp.rungs(i, menu).iter().map(|r| r.name()).collect(),
            refused: sp.refusals[i].iter().map(|(r, w)| (r.name(), w.name().to_string(), w.to_string())).collect(),
        })
        .collect()
}

/// The best assignment by exhaustive search, for programs small enough that
/// it is affordable. Used to say how far the greedy proposal is from optimal
/// rather than to assert that it is optimal.
///
/// Returns `None` when the search space exceeds `max_states`.
#[allow(clippy::too_many_arguments)]
pub fn exhaustive(env: &Envelope, sens: &Sensitivity, roles: &Roles, output_site: &str, budget: Budget, menu: &Menu, default: Dtype, machine: &dyn Machine, max_states: u64) -> Option<(Realization, f64)> {
    exhaustive_with(env, sens, roles, output_site, budget, menu, default, machine, max_states, Options::default())
}

/// [`exhaustive`] under the options' rounding term (D155) and rules (D169):
/// the best assignment under the same estimate the proposal used, over the
/// values it offers.
#[allow(clippy::too_many_arguments)]
pub fn exhaustive_with(env: &Envelope, sens: &Sensitivity, roles: &Roles, output_site: &str, budget: Budget, menu: &Menu, default: Dtype, machine: &dyn Machine, max_states: u64, options: Options) -> Option<(Realization, f64)> {
    let out_mag = env.get(output_site).unwrap_or(1.0).max(f64::MIN_POSITIVE);
    let budget_abs = budget.absolute(out_mag);
    let mut sp = split(env, sens, roles, options.rounding, menu, options.range_margin);
    sp.offer(menu, default, options.min_gain);
    let sites = &sp.decisions;
    let free: Vec<usize> = (0..sites.len()).filter(|i| sp.offered[*i]).collect();
    // Each free decision's rungs, a parameter's integers among them (D177):
    // the space is the product of their counts.
    let rungs: Vec<Vec<Rung>> = free.iter().map(|i| sp.rungs(*i, menu)).collect();
    let states = rungs.iter().try_fold(1u64, |n, r| n.checked_mul(r.len() as u64))?;
    if states > max_states {
        return None;
    }
    let mut best: Option<(Vec<Rung>, f64)> = None;
    for code in 0..states {
        let mut c = code;
        let mut pick = vec![Rung::Float(default); sites.len()];
        for (i, r) in free.iter().zip(&rungs) {
            let m = r.len() as u64;
            pick[*i] = r[(c % m) as usize];
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

/// The grain a training step is solved at (plan 0.17.0 step 5, D178).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Grain {
    /// One type per function (D52): J′'s solver, the default.
    #[default]
    Function,
    /// One type per stored value of the step, each value a group of its
    /// own, written in the realization's `storage` ([`step_value_groups`]).
    Value,
}

/// The groups of a step's envelope at the grain of its stored values (plan
/// 0.17.0 step 5, D178): one per site `offered` takes, of more than one
/// element (rule 1, D169). A value is priced with the members fused into it,
/// which are computed in its type (D112), as [`step_groups`] prices a
/// function's sites. The caller says which sites are stored values the step
/// may type: not a carried value or a fused member, nor the optimizer's
/// results, the master weights and the moments, which keep the default type
/// (the plan's question 6).
pub fn step_value_groups(env: &Envelope, menu: &Menu, machine: &dyn Machine, roles: &Roles, offered: &dyn Fn(&str) -> bool) -> Vec<Group> {
    // Each value's members: the sites that follow it alone.
    let mut members: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for name in env.sites.keys() {
        if let Role::Follows(leaders) = roles.role(name)
            && let [one] = leaders.as_slice()
            && let Some((k, _)) = env.sites.get_key_value(one.as_str())
        {
            members.entry(k.as_str()).or_default().push(name.as_str());
        }
    }
    let priced = |name: &str| (env.elems(name) as f64, op_weight(env.op(name)));
    env.sites
        .keys()
        .filter(|name| offered(name))
        .filter_map(|name| {
            let elems = env.elems(name) as f64;
            (elems > 1.0).then(|| {
                let sites: Vec<(f64, f64)> = std::iter::once(name.as_str()).chain(members.get(name.as_str()).into_iter().flatten().copied()).map(priced).filter(|(e, _)| *e > 0.0).collect();
                let cost = menu.0.iter().map(|d| (*d, sites.iter().map(|(e, w)| machine.seconds(*d, e * w, e * d.bytes() as f64)).sum())).collect();
                Group { name: name.clone(), sites: sites.len(), elems: elems as u64, cost, gap: BTreeMap::new() }
            })
        })
        .collect()
}

/// The predicted time of a whole step under any realization, by role (plan
/// 0.17.0 step 5, D178): each stored value at the type the realization gives
/// it (`Realization::dtype_at`: its own, else its function's, else the
/// default), each value that follows others at the narrowest of theirs, a
/// carried value at nothing, since it is another site's. The one model both
/// grains are compared in: [`step_cost`], D52's, prices every site of a
/// function at its function's type, carried values included, and cannot
/// price a realization of stored values.
pub fn step_time(env: &Envelope, machine: &dyn Machine, roles: &Roles, r: &Realization) -> f64 {
    env.sites
        .keys()
        .map(|name| {
            let elems = env.elems(name) as f64;
            let d = match roles.role(name) {
                Role::Decision => Some(r.dtype_at(name)),
                Role::Follows(leaders) => leaders.iter().map(|l| r.dtype_at(l)).min_by_key(|d| d.bytes()),
                Role::Carried => None,
            };
            match d {
                Some(d) if elems > 0.0 => machine.seconds(d, elems * op_weight(env.op(name)), elems * d.bytes() as f64),
                _ => 0.0,
            }
        })
        .sum()
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

/// The step under a realization that gives each stored value of `chosen`
/// its type, in `storage` (D178).
pub fn by_value(default: Dtype, chosen: &BTreeMap<String, Dtype>) -> Realization {
    let mut r = Realization::uniform(default);
    r.storage = chosen.iter().filter(|(_, d)| **d != default).map(|(k, d)| (k.clone(), *d)).collect();
    r
}

fn by_grain(grain: Grain, default: Dtype, chosen: &BTreeMap<String, Dtype>) -> Realization {
    match grain {
        Grain::Function => by_function(default, chosen),
        Grain::Value => by_value(default, chosen),
    }
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
pub fn solve_step(groups: Vec<Group>, budget: f64, menu: &Menu, default: Dtype, measure: &mut dyn FnMut(&Realization) -> f64, verify: bool) -> StepSolution {
    solve_step_with(groups, budget, menu, default, measure, verify, Grain::Function)
}

/// [`solve_step`] at a grain (plan 0.17.0 step 5, D178): its groups
/// functions, as D52 chose, or stored values ([`step_value_groups`]), the
/// same method at a finer grain, each value measured alone in every
/// narrower type and its realization written in `storage`.
#[allow(clippy::too_many_arguments)]
pub fn solve_step_with(mut groups: Vec<Group>, budget: f64, menu: &Menu, default: Dtype, measure: &mut dyn FnMut(&Realization) -> f64, verify: bool, grain: Grain) -> StepSolution {
    let narrower: Vec<Dtype> = menu.0.iter().copied().filter(|d| d.bytes() < default.bytes()).collect();
    for g in groups.iter_mut() {
        for d in &narrower {
            let r = by_grain(grain, default, &BTreeMap::from([(g.name.clone(), *d)]));
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
            let gap = measure(&by_grain(grain, default, &chosen));
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
    StepSolution { realization: by_grain(grain, default, &chosen), groups, chosen, estimate, measured, cost, baseline, rounds }
}

/// The predicted cost of any realization of the step, over the same groups:
/// a group's cost at the type its function is given (sites named one by one
/// are not priced separately; the realizations measured here have none).
pub fn step_cost(groups: &[Group], r: &Realization) -> f64 {
    groups.iter().map(|g| g.cost.get(&r.functions.get(&g.name).copied().unwrap_or(r.default)).copied().unwrap_or(0.0)).sum()
}
