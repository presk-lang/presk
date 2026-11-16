//! First-order error propagation over PSK-Core: the sound, pessimistic side.
//!
//! Each value carries a [`Bound`]: a ceiling on `|v|`, read from the measured
//! envelope, and a ceiling on `|executed(v) − exact(v)|`, computed here. The
//! two travel together because neither is usable alone — an absolute error
//! means nothing without a scale, and a relative one is unbounded wherever a
//! result passes near zero.
//!
//! # What makes this different from counting roundoffs
//!
//! A naive propagation multiplies the error by an operator norm at every
//! step and reaches a number with no information in it after a dozen layers.
//! The specification's instruction is to exploit *known conditioning*, and
//! three cases carry the transformer:
//!
//! - a **softmax** is 1-Lipschitz in the ∞ norm, so its output error is its
//!   input error and not `k` times it, whatever the length of the row;
//! - a **normaliser** divides by a quantity of the same scale as its
//!   numerator, so the relative error crosses it unchanged — which falls out
//!   of the quotient rule once the envelope supplies a floor for the divisor,
//!   rather than needing a case of its own;
//! - a **residual** adds errors instead of multiplying them, which is why
//!   depth costs a sum rather than a product.
//!
//! # Where the reduction order enters
//!
//! A contraction of `k` terms adds `c(order, k) · u` to the relative error,
//! with `c = k − 1` for a sequential or blocked sum and `c ≈ k/n + n` for `n`
//! interleaved partials. So the realization's declared order is an *argument*
//! of the analysis (D34), not bookkeeping: a vectorized kernel is provably
//! looser than a scalar one by a factor this prints.
//!
//! # Four rules for a sum
//!
//! What a sum does to the error its inputs carry is where the bound is won
//! or lost, and [`Model`] names four answers (0.6.0 step 6, D51): 0.5.0's
//! `k · max|w| · max|δx|`; a normwise rule with the measured `Σ|wᵢ|`; the
//! condition number, `(|δx|/|x|) · Σ|xᵢwᵢ|`, the default; and a
//! root-sum-square *estimate*. The last three read the sums of terms the
//! envelope records. The condition number takes the corpus GPT's bound from
//! 10³³ to 10¹⁰ times the error and no further: the rest is the roundings'
//! independence, which only the estimate sees.
//!
//! # What the result promises
//!
//! A bound here is conditional on the envelope it was computed from (D35),
//! and it is first-order: products of two errors are dropped, as is standard
//! in this literature, so it is an asymptotic statement about small
//! perturbations rather than an interval-arithmetic certificate. Both
//! qualifications are printed with the number.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use presk_core::ir::Lit;
use presk_core::{FuncId, PrimOp, Program, Rhs, VarId};
use presk_format::Format;
use presk_rt::{Realization, ReductionOrder};
use presk_types::elab::ElabProgram;

use crate::envelope::Envelope;
use crate::formats::{self, Seen};

/// A ceiling on a value's magnitude and on its distance to the exact value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bound {
    /// `max |v|` over the envelope's samples.
    pub mag: f64,
    /// The smallest non-zero `|v|` the envelope saw, or `0` when it saw none.
    /// A quotient's error grows as its divisor shrinks, so this travels with
    /// the value rather than being looked up at the division.
    pub floor: f64,
    /// `max |executed(v) − exact(v)|`, to first order.
    pub err: f64,
}

impl Bound {
    pub const EXACT_ZERO: Bound = Bound { mag: 0.0, floor: 0.0, err: 0.0 };

    pub fn exact(mag: f64) -> Bound {
        Bound { mag, floor: 0.0, err: 0.0 }
    }

    /// The relative error, with the magnitude as its scale. Infinite when the
    /// value can be zero and the error cannot, which is the honest answer:
    /// a relative bound on something that vanishes says nothing.
    pub fn relative(&self) -> f64 {
        if self.mag > 0.0 { self.err / self.mag } else if self.err == 0.0 { 0.0 } else { f64::INFINITY }
    }

    fn join(self, other: Bound) -> Bound {
        // The widest magnitude, the *lowest* floor and the largest error: a
        // join has to cover both, which for a floor means the smaller one.
        let floor = match (self.floor, other.floor) {
            (0.0, f) | (f, 0.0) => f,
            (a, b) => a.min(b),
        };
        Bound { mag: self.mag.max(other.mag), floor, err: self.err.max(other.err) }
    }
}

/// Which rule a sum or a contraction propagates its inputs' error by
/// (0.6.0 step 6, D51). Everything else — quotients, softmax, residuals — is
/// the same in all four.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Model {
    /// `k · max|w| · max|δx|`, the rule of 0.5.0. Sound, needs no
    /// measurement beyond the magnitudes, and vacuous on a transformer (D39).
    Worst,
    /// `max|δx| · Σ|wᵢ|`, with the sums measured. Still sound: it only
    /// replaces `k · max|w|` by what the weights actually add up to.
    Normwise,
    /// `(|δx|/|x|) · Σ|xᵢwᵢ|`: the condition number. Sound when every
    /// element's error is proportional to its magnitude, which holds of a
    /// value rounded once and is assumed after that.
    #[default]
    Conditioned,
    /// `(|δx|/|x|) · √Σ(xᵢwᵢ)²` and `√c` roundings: the terms' errors taken
    /// as independent, so they cancel as `√k`. An *estimate*, not a bound.
    Statistical,
}

impl Model {
    pub const ALL: [Model; 4] = [Model::Worst, Model::Normwise, Model::Conditioned, Model::Statistical];

    pub fn name(self) -> &'static str {
        match self {
            Model::Worst => "k",
            Model::Normwise => "normwise",
            Model::Conditioned => "cond",
            Model::Statistical => "rss",
        }
    }

    pub fn parse(s: &str) -> Option<Model> {
        Model::ALL.into_iter().find(|m| m.name() == s)
    }

    /// Whether what this model computes is a bound. The statistical one is
    /// an estimate and has to be printed as one.
    pub fn is_bound(self) -> bool {
        self != Model::Statistical
    }
}

/// What a propagation found.
#[derive(Clone, Debug)]
pub struct Analysis {
    /// The bound at every site, `"<function>/%<let>"`.
    pub sites: BTreeMap<String, Bound>,
    /// The bound on the entry's result.
    pub output: Bound,
    /// The envelope this is conditional on, and how many samples it saw.
    pub envelope: String,
    pub samples: usize,
    /// Sites the envelope did not cover. Their magnitude was taken from their
    /// inputs, which is a guess; a long list means the envelope was measured
    /// on a different entry and the bound should not be trusted.
    pub uncovered: Vec<String>,
    /// Sites where a divisor's floor was unknown, so the quotient's error
    /// could not be bounded at all.
    pub unbounded: Vec<String>,
    /// The rule sums and contractions were propagated by.
    pub model: Model,
    /// Sums and contractions whose terms the envelope did not record, so the
    /// model's rule fell back to the `k` one there. An envelope written
    /// before 0.6.0 step 6 has none recorded.
    pub fallback: Vec<String>,
    /// The terms each sum and contraction adds, read off the types at the
    /// envelope's shapes (plan 0.18.0 step 5): the `k` of its rule, which
    /// the solver's terms of a sum's formats read too.
    pub sums: BTreeMap<String, f64>,
}

impl Analysis {
    pub fn site(&self, s: &str) -> Option<Bound> {
        self.sites.get(s).copied()
    }
}

/// Whether the error analysis bounds a realization. Every format has its
/// terms since plan 0.18.0 step 5 (D208, [`crate::formats`]): a nest's
/// operations at its arithmetic format, a sum's products and additions at
/// theirs, a value at the format it is stored in, an affine one at half its
/// step and what its clamp cuts. A scale stated by a scheme and not computed
/// has no step to charge, and is refused (`E7006`'s message).
pub fn bounded_formats(realize: &Realization) -> Result<(), String> {
    realize.check_scales().map_err(|e| e.message)
}

/// Propagate the error of `entry` under `realize`, given `env`, by the
/// default rule, the condition number.
pub fn propagate(core: &Program, elab: &ElabProgram, entry: FuncId, dims: &BTreeMap<presk_types::dim::DimVar, u64>, env: &Envelope, realize: &Realization) -> Analysis {
    propagate_with(core, elab, entry, dims, env, realize, Model::default())
}

/// The same, by a chosen rule for sums and contractions.
pub fn propagate_with(core: &Program, elab: &ElabProgram, entry: FuncId, dims: &BTreeMap<presk_types::dim::DimVar, u64>, env: &Envelope, realize: &Realization, model: Model) -> Analysis {
    let mut a = Analysis {
        sites: BTreeMap::new(),
        output: Bound::EXACT_ZERO,
        envelope: format!("{} ({} samples)", env.program, env.samples),
        samples: env.samples,
        uncovered: Vec::new(),
        unbounded: Vec::new(),
        model,
        fallback: Vec::new(),
        sums: BTreeMap::new(),
    };
    let _ = elab;
    // A fused member is computed in its group's root's type and stored
    // nowhere (the reference's chapter 4, "Realized value"): its rounding is
    // charged at the root's type, whatever the realization states for the
    // member (plan 0.11.0 step 2b, D112).
    let mut roots: HashMap<String, String> = HashMap::new();
    for f in &core.funcs {
        let plan = presk_core::fusion::plan(f);
        for g in plan.groups.values() {
            for m in g.members.iter().filter(|m| **m != g.root) {
                roots.insert(format!("{}/%{}", f.name, m.0), format!("{}/%{}", f.name, g.root.0));
            }
        }
    }
    // The paths of the parameters each field site reads, for the integers
    // a quantized weight is counted in (plan 0.18.0 step 5).
    let reads = if realize.quant.is_empty() { BTreeMap::new() } else { crate::solve::param_reads(core, entry) };
    let mut p = Prop { core, env, realize, out: &mut a, depth: 0, model, roots, reads };
    // The entry's arguments arrive rounded once into their storage type,
    // which is what every backend does when a value crosses the seam. The
    // parameter tree is one bound, the join of its leaves: rounded at the
    // narrowest of the default and the types `params` stores its leaves in
    // (plan 0.12.0 step 4), and since plan 0.18.0 step 5 at half the step of
    // an integer weight's scale, read as `f32(q·s)` (D44).
    let args: Vec<Bound> = core
        .func(entry)
        .params
        .iter()
        .map(|param| {
            let site = format!("{}/%{}", core.func(entry).name, param.var.0);
            let mag = p.env.get(&site).unwrap_or(1.0);
            let seen = p.seen(&site, mag);
            let at = |d: presk_rt::Dtype| if d.is_native() { d.unit_roundoff() * mag } else { formats::rounding(Format::Float(d.format()), seen) };
            let mut err = at(p.realize.dtype_at(&site));
            if matches!(param.ty, presk_types::ty::Ty::Params(_)) {
                err = p.realize.params.values().map(|d| at(*d)).fold(err, f64::max);
                err = p.realize.quant.values().map(|q| quantized(q, mag)).fold(err, f64::max);
            }
            Bound { mag, floor: p.env.floor(&site).unwrap_or(0.0), err }
        })
        .collect();
    let out = p.func(entry, dims.clone(), &args);
    a.output = out;
    a
}

struct Prop<'a> {
    core: &'a Program,
    env: &'a Envelope,
    realize: &'a Realization,
    out: &'a mut Analysis,
    depth: usize,
    model: Model,
    /// A fused member's site → its group's root's.
    roots: HashMap<String, String>,
    /// A field site → the parameters' paths it reads, when the realization
    /// quantizes some.
    reads: BTreeMap<String, BTreeSet<String>>,
}

/// What a parameter in integers moves at most, read as a nest reads it: half
/// its scale's step, `max|w|/(2·qmax)` under `absmax` and up to `max|w|/qmax`
/// under `pow2`, then the `f32` the nest dequantizes it to.
fn quantized(q: &presk_rt::Quant, mag: f64) -> f64 {
    let qmax = ((1i64 << (q.bits - 1)) - 1) as f64;
    let half = match q.scheme {
        presk_rt::QuantScheme::Absmax => mag / (2.0 * qmax),
        presk_rt::QuantScheme::Pow2 => mag / qmax,
    };
    half + presk_rt::Dtype::F32.unit_roundoff() * mag
}

/// How deep the walk follows calls. A Core program's call graph is a DAG of
/// module functions, so this is a guard against a pathological program rather
/// than a limit anything real reaches.
const MAX_DEPTH: usize = 64;

impl Prop<'_> {
    fn func(&mut self, f: FuncId, dims: BTreeMap<presk_types::dim::DimVar, u64>, args: &[Bound]) -> Bound {
        if self.depth >= MAX_DEPTH {
            return Bound { mag: f64::INFINITY, floor: 0.0, err: f64::INFINITY };
        }
        self.depth += 1;
        let func = self.core.func(f);
        let mut env: HashMap<VarId, Bound> = HashMap::new();
        let mut tys: HashMap<VarId, presk_types::ty::Ty> = HashMap::new();
        for (p, a) in func.params.iter().zip(args) {
            env.insert(p.var, *a);
            tys.insert(p.var, p.ty.clone());
        }
        let mut closures: HashMap<VarId, (FuncId, Vec<Bound>)> = HashMap::new();
        for l in &func.body.lets {
            let site = format!("{}/%{}", func.name, l.var.0);
            let b = self.rhs(&l.rhs, l.var, &site, &l.ty, &dims, &env, &tys, &mut closures);
            // The envelope's floor belongs to the site, not to the rule that
            // produced it, so it is attached once here.
            let b = Bound { floor: self.env.floor(&site).unwrap_or(b.floor), ..b };
            // A function called from several places has one site and several
            // bounds. The table reports the worst, which is what a reader
            // wants; propagation uses the call's own, because feeding the
            // worst forward would compound a bound that belongs to another
            // call — five shared layer norms that way put a `f64` transformer
            // at 10³⁰.
            let e = self.out.sites.entry(site).or_insert(Bound::EXACT_ZERO);
            *e = e.join(b);
            env.insert(l.var, b);
            tys.insert(l.var, l.ty.clone());
        }
        let out = env.get(&func.body.result).copied().unwrap_or(Bound::EXACT_ZERO);
        self.depth -= 1;
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn rhs(
        &mut self,
        rhs: &Rhs,
        var: VarId,
        site: &str,
        out_ty: &presk_types::ty::Ty,
        dims: &BTreeMap<presk_types::dim::DimVar, u64>,
        env: &HashMap<VarId, Bound>,
        tys: &HashMap<VarId, presk_types::ty::Ty>,
        closures: &mut HashMap<VarId, (FuncId, Vec<Bound>)>,
    ) -> Bound {
        let get = |v: &VarId| -> Bound { env.get(v).copied().unwrap_or(Bound::EXACT_ZERO) };
        match rhs {
            Rhs::Lit(Lit::Nat(n)) => Bound::exact(*n as f64),
            Rhs::Lit(Lit::Float(s)) => {
                let x: f64 = s.parse().unwrap_or(0.0);
                // A decimal literal is rounded once into the storage type.
                Bound { mag: x.abs(), floor: x.abs(), err: self.rt(site, x.abs()) }
            }
            Rhs::Prim { op, args } => {
                let bs: Vec<Bound> = args.iter().map(get).collect();
                let atys: Vec<Option<presk_types::ty::Ty>> = args.iter().map(|v| tys.get(v).cloned()).collect();
                let k = terms(op, &atys, out_ty, dims);
                // The operands' sites, for the formats they are stored in.
                let func = site.rsplit_once("/%").map_or(site, |(f, _)| f);
                let operands: Vec<String> = args.iter().map(|v| format!("{func}/%{}", v.0)).collect();
                self.prim(op, &bs, site, k, &operands)
            }
            Rhs::Call { f, dim_args, args } => {
                let bs: Vec<Bound> = args.iter().map(get).collect();
                let inner = callee_dims(dim_args, dims);
                self.func(*f, inner, &bs)
            }
            Rhs::Tuple(vs) => vs.iter().map(get).fold(Bound::EXACT_ZERO, Bound::join),
            Rhs::Record(fs) => fs.iter().map(|(_, v)| get(v)).fold(Bound::EXACT_ZERO, Bound::join),
            // A projection cannot be resolved without tracking tuples field by
            // field; the join over the tuple is the sound over-approximation.
            Rhs::Proj(v, _) | Rhs::Field(v, _) => get(v),
            // A closure is a function value: remember what it captures, so
            // that the `apply` downstream walks the body instead of joining
            // its arguments. Forgetting this is how the first version of the
            // analysis returned a single rounding for a whole gradient, which
            // the soundness test caught at once.
            Rhs::Closure { f, captured } => {
                closures.insert(var, (*f, captured.iter().map(get).collect()));
                Bound::EXACT_ZERO
            }
            // `grad(f)` is a function value too, and its body is the vjp the
            // AD pass generated: `<f>.grad`. It captures what `f` captured.
            Rhs::Grad(v) => {
                if let Some((f, cap)) = closures.get(v).cloned() {
                    let name = format!("{}.grad", self.core.func(f).name);
                    if let Some(g) = self.core.find(&name) {
                        closures.insert(var, (g, cap));
                    }
                }
                Bound::EXACT_ZERO
            }
            Rhs::Apply { fun, args } => {
                let bs: Vec<Bound> = args.iter().map(get).collect();
                match closures.get(fun).cloned() {
                    Some((f, cap)) => {
                        let all: Vec<Bound> = cap.into_iter().chain(bs).collect();
                        self.func(f, dims.clone(), &all)
                    }
                    None => bs.into_iter().fold(Bound::EXACT_ZERO, Bound::join),
                }
            }
            // A fold is its body applied `n` times, the accumulator threaded.
            // Depth costs a sum here rather than a product, because the body
            // of a transformer block is a residual.
            Rhs::Fold { body, dim_args, n, stacked, init, extra, .. } => {
                let inner = callee_dims(dim_args, dims);
                let n = eval_dim(n, dims).unwrap_or(1).min(256);
                let st: Vec<Bound> = stacked.iter().map(get).collect();
                let ex: Vec<Bound> = extra.iter().map(get).collect();
                let mut acc = get(init);
                for _ in 0..n {
                    let mut all = st.clone();
                    all.push(acc);
                    all.extend(ex.iter().copied());
                    acc = self.func(*body, inner.clone(), &all);
                }
                acc
            }
        }
    }

    /// The site whose formats a site is computed in: its group's root's for
    /// a fused member.
    fn root<'s>(&'s self, site: &'s str) -> &'s str {
        self.roots.get(site).map(String::as_str).unwrap_or(site)
    }

    /// What the envelope saw of a site's value, `mag` its magnitude.
    fn seen(&self, site: &str, mag: f64) -> Seen {
        Seen { mag, floor: self.env.floor(site).unwrap_or(0.0), range: self.env.range.get(site).map(|r| (r[0], r[1])) }
    }

    /// The format a site's value is stored in: an affine one, or its
    /// storage type.
    fn stored(&self, root: &str) -> Format {
        match self.realize.scaled_at(root) {
            Ok(Some(a)) => Format::Affine(a),
            _ => Format::Float(self.realize.dtype_at(root).format()),
        }
    }

    /// Whether a nest computes as every nest did until 0.18.0: in its
    /// storage type's arithmetic type, its value stored in that type. Its
    /// rounding is then charged as it was, `u(σ)·|x|` at its storage type
    /// σ, the arithmetic type's own rounding not counted, so that no bound
    /// of record moves (MT0).
    fn plain(&self, root: &str) -> bool {
        let d = self.realize.dtype_at(root);
        self.realize.arith_at(root) == Format::Float(d.arith().format()) && !matches!(self.realize.scaled_at(root), Ok(Some(_)))
    }

    /// Whether a sum computes as every sum did until 0.18.0: a plain nest
    /// whose products are rounded to its arithmetic format and whose
    /// additions to a float, charged at the accumulator's unit roundoff.
    fn plain_sum(&self, root: &str) -> bool {
        self.plain(root) && self.realize.product_at(root) == self.realize.arith_at(root) && matches!(self.realize.accum_at(root), Format::Float(_))
    }

    /// The rounding a site's result takes (plan 0.18.0 step 5, D208): at a
    /// plain nest its storage type's, as until then (a native type's
    /// `u·|x|`, any other float's term of [`formats::rounding`]); otherwise
    /// its operation's at its nest's arithmetic format and, at a nest's
    /// root, its store's into the format it is stored in, unless that holds
    /// the arithmetic format's numbers.
    fn rt(&self, site: &str, mag: f64) -> f64 {
        let root = self.root(site);
        let seen = self.seen(site, mag);
        if self.plain(root) {
            let d = self.realize.dtype_at(root);
            return if d.is_native() { d.unit_roundoff() * mag } else { formats::rounding(Format::Float(d.format()), seen) };
        }
        let alpha = self.realize.arith_at(root);
        let op = formats::rounding(alpha, seen);
        if self.roots.contains_key(site) { op } else { op + self.store(root, alpha, seen) }
    }

    /// A value computed in `from` stored at `root`: one rounding into the
    /// format it is stored in, unless that holds `from`'s numbers.
    fn store(&self, root: &str, from: Format, seen: Seen) -> f64 {
        let to = self.stored(root);
        if to.contains(from) { 0.0 } else { formats::rounding(to, seen) }
    }

    /// The rounding a sum's result takes as it is stored: a plain sum's as
    /// [`Prop::rt`]'s; a member's none; a counted sum's one conversion of its
    /// count into the stored format; otherwise its accumulator's value
    /// stored.
    fn rt_sum(&self, site: &str, mag: f64) -> f64 {
        let root = self.root(site);
        if self.plain(root) {
            return self.rt(site, mag);
        }
        if self.roots.contains_key(site) {
            return 0.0;
        }
        let seen = self.seen(site, mag);
        if counts(self.realize.product_at(root)) { formats::rounding(self.stored(root), seen) } else { self.store(root, self.realize.accum_at(root), seen) }
    }

    /// The accumulator's unit roundoff, which is what a contraction's inner
    /// sum actually rounds in: a plain sum's float.
    fn u_acc(&self, site: &str) -> f64 {
        match self.realize.accum_at(site) {
            Format::Float(f) => f.unit_roundoff(),
            other => unreachable!("a plain sum accumulates in a float, not {other}"),
        }
    }

    /// The largest `|q − z|` of an operand counted in its unit: a value
    /// stored in an affine format, or a field reading parameters every one
    /// of which the realization quantizes. `None` for any other operand.
    fn codes(&self, operand: &str) -> Option<f64> {
        if let Ok(Some(a)) = self.realize.scaled_at(operand) {
            return Some(formats::largest_code(a));
        }
        let paths = self.reads.get(operand).filter(|p| !p.is_empty())?;
        paths.iter().map(|p| self.realize.quant.get(p).map(|q| ((1i64 << (q.bits - 1)) - 1) as f64)).try_fold(0.0f64, |m, q| q.map(|q| m.max(q)))
    }

    /// The format an operand is read in as a float: its stored one, a
    /// parameter's the `f32` it is dequantized to or its own type.
    fn operand_float(&self, operand: &str) -> Option<presk_format::Float> {
        if let Some(paths) = self.reads.get(operand) {
            return (!paths.iter().any(|p| self.realize.quant.contains_key(p))).then(|| self.realize.dtype_at(operand).format());
        }
        match self.stored(operand) {
            Format::Float(f) => Some(f),
            _ => None,
        }
    }

    /// What a sum's own operations add (plan 0.18.0 step 5): a plain sum's
    /// `c·u·Σ|terms|`, its products counted among its roundings as until
    /// then; otherwise [`formats::sum_rounding`] at its product and
    /// accumulation formats, its products exact where its operands' formats
    /// multiply exactly in the product format, and a counted sum exact while
    /// its operands' codes keep the count inside its accumulator. Infinite,
    /// and the site listed as unbounded, where it is neither.
    #[allow(clippy::too_many_arguments)]
    fn sum_ops(&mut self, site: &str, k: f64, c: f64, products: f64, abs: f64, largest: f64, operands: &[String], statistical: bool) -> f64 {
        let root = self.root(site).to_string();
        if self.plain_sum(&root) {
            let n = c + products;
            return if statistical { n.sqrt() } else { n } * self.u_acc(&root) * abs;
        }
        let (mu, kappa) = (self.realize.product_at(&root), self.realize.accum_at(&root));
        let count = if counts(mu) { operands.iter().map(|o| self.codes(o)).try_fold(k, |n, a| a.map(|a| n * a)) } else { None };
        let exact = match (mu, operands) {
            (Format::Float(f), [a, b]) => self.operand_float(a).zip(self.operand_float(b)).is_some_and(|(a, b)| formats::exact_product(a, b, f)),
            _ => false,
        };
        let root_n = |n: f64| if statistical { n.sqrt() } else { n };
        let s = formats::Sum { k, abs, largest, additions: root_n(c), products: (!exact).then_some(root_n(products)), count };
        let e = formats::sum_rounding(mu, kappa, &s, 1.0);
        if e.is_infinite() && !self.out.unbounded.iter().any(|x| x == site) {
            self.out.unbounded.push(site.to_string());
        }
        e
    }

    /// The measured magnitude of a site, or a fallback derived from its
    /// inputs when the envelope does not cover it.
    fn mag_of(&mut self, site: &str, fallback: f64) -> f64 {
        match self.env.get(site) {
            Some(m) => m,
            None => {
                if !self.out.uncovered.iter().any(|s| s == site) {
                    self.out.uncovered.push(site.to_string());
                }
                fallback
            }
        }
    }

    /// The shape of nearly every rule: the exact result's error, plus the
    /// rounding of the result ([`Prop::rt`]).
    fn rounded(&self, site: &str, err: f64, mag: f64) -> Bound {
        Bound { mag, floor: 0.0, err: err + self.rt(site, mag) }
    }

    /// A differentiable unary function of derivative bounded by `k`.
    fn lip(&mut self, site: &str, k: f64, x: Bound, fallback: f64) -> Bound {
        let m = self.mag_of(site, fallback);
        self.rounded(site, k * x.err, m)
    }

    fn prim(&mut self, op: &PrimOp, a: &[Bound], site: &str, k: f64, operands: &[String]) -> Bound {
        let x = a.first().copied().unwrap_or(Bound::EXACT_ZERO);
        let y = a.get(1).copied().unwrap_or(Bound::EXACT_ZERO);
        if matches!(op, PrimOp::SumAll | PrimOp::Sum { .. } | PrimOp::SumTo | PrimOp::SumSquares | PrimOp::MeanAll | PrimOp::Mean { .. } | PrimOp::Dot | PrimOp::Matmul | PrimOp::Ein { .. } | PrimOp::DotVjpL | PrimOp::DotVjpR) {
            self.out.sums.insert(site.to_string(), k);
        }
        match op {
            // ----- exact, or error-free -----
            PrimOp::Zeros { .. } | PrimOp::Ones { .. } | PrimOp::Positions { .. } | PrimOp::CausalMask { .. } | PrimOp::Pi | PrimOp::Normal { .. } | PrimOp::XavierUniform { .. } | PrimOp::SplitKey { .. } | PrimOp::Argmax | PrimOp::Sample | PrimOp::OneHot | PrimOp::Heaviside | PrimOp::Sign | PrimOp::ScatterIndex { .. } => {
                let m = self.mag_of(site, 1.0);
                Bound::exact(m)
            }
            // A mask is 0 or `1/(1−p)`: one rounding, no input error.
            PrimOp::DropoutMask => {
                let m = self.mag_of(site, 1.0);
                Bound { mag: m, floor: 0.0, err: self.rt(site, m) }
            }
            // ----- retags and selections: the data does not change -----
            PrimOp::MergeAxes { .. } | PrimOp::SplitAxes { .. } | PrimOp::ReshapeLike | PrimOp::SplitHeads { .. } | PrimOp::MergeHeads { .. } | PrimOp::IndexConst { .. } | PrimOp::SliceParams { .. } | PrimOp::StackParams { .. } | PrimOp::SetField { .. } | PrimOp::SetElem { .. } | PrimOp::SetSlice | PrimOp::BroadcastLike | PrimOp::Embed | PrimOp::EmbedVjp | PrimOp::ScatterParams { .. } | PrimOp::Real | PrimOp::Slice { .. } | PrimOp::Concat { .. } => {
                let m = self.mag_of(site, x.mag.max(y.mag));
                Bound { mag: m, floor: 0.0, err: a.iter().map(|b| b.err).fold(0.0, f64::max) }
            }
            // The cotangent placed into zeros: its error, the template read
            // for its shape.
            PrimOp::SliceVjp { .. } => {
                let m = self.mag_of(site, y.mag);
                Bound { mag: m, floor: 0.0, err: y.err }
            }
            PrimOp::ZerosLike => Bound::EXACT_ZERO,
            // ----- elementwise arithmetic -----
            PrimOp::Neg => Bound { mag: x.mag, floor: 0.0, err: x.err },
            PrimOp::Abs | PrimOp::Relu => Bound { mag: x.mag, floor: 0.0, err: x.err },
            // A selection of one operand, exact: the larger error of the two.
            PrimOp::Max | PrimOp::Min => Bound { mag: x.mag.max(y.mag), floor: 0.0, err: x.err.max(y.err) },
            PrimOp::Add | PrimOp::Sub => {
                // Errors add. This is the residual's rule, and the reason
                // depth is affordable at all.
                let m = self.mag_of(site, x.mag + y.mag);
                self.rounded(site, x.err + y.err, m)
            }
            PrimOp::Mul => {
                // A product needs no relative detour: the absolute form is
                // exact to first order and never divides by anything.
                let m = self.mag_of(site, x.mag * y.mag);
                self.rounded(site, x.mag * y.err + y.mag * x.err, m)
            }
            PrimOp::Div => {
                // In *relative* terms a quotient simply adds:
                // `d(a/b)/(a/b) = da/a − db/b`. Propagating that way is what
                // makes a normaliser cost nothing, which is the conditioning
                // the specification names — and it is not a special case for
                // `LayerNorm`, it is the quotient rule written in the right
                // variable.
                //
                // The absolute form, `da/b + a·db/b²`, is sound pointwise but
                // pairs the largest numerator with the smallest divisor, a
                // pairing that never occurs: written that way, five layer
                // norms in a row turned a `f64` bound into 10³⁰.
                let m = self.mag_of(site, if y.mag > 0.0 { x.mag / y.mag } else { x.mag });
                if y.mag > 0.0 && floor_of(y).is_none() {
                    // A divisor that reaches zero in the samples: the
                    // quotient has no relative bound, and saying so is the
                    // answer rather than a number.
                    if !self.out.unbounded.iter().any(|s| s == site) {
                        self.out.unbounded.push(site.to_string());
                    }
                    return Bound { mag: m, floor: 0.0, err: f64::INFINITY };
                }
                // Both forms are first-order bounds under their own
                // assumptions, so the tighter one is the honest report.
                let rel = m * (x.relative() + y.relative());
                let abs = floor_of(y).map(|lo| x.err / lo + x.mag * y.err / (lo * lo)).unwrap_or(f64::INFINITY);
                self.rounded(site, rel.min(abs), m)
            }
            PrimOp::Pow(n) => {
                let m = self.mag_of(site, x.mag.powi(*n as i32));
                self.rounded(site, (*n as f64) * x.mag.powi((*n as i32 - 1).max(0)) * x.err, m)
            }
            PrimOp::Sqrt => {
                // `d√x/√x = (dx/x)/2`: a square root *halves* relative error.
                let m = self.mag_of(site, x.mag.sqrt());
                let rel = m * x.relative() / 2.0;
                let abs = floor_of(x).map(|lo| x.err / (2.0 * lo.sqrt())).unwrap_or(f64::INFINITY);
                self.rounded(site, rel.min(abs), m)
            }
            // `d(eˣ) = eˣ dx`: the derivative *is* the output.
            PrimOp::Exp => {
                let m = self.mag_of(site, x.mag.exp());
                self.rounded(site, m * x.err, m)
            }
            PrimOp::Log => {
                // `d(ln x) = dx/x`: the *absolute* error of a logarithm is
                // the *relative* error of its argument. The two analyses meet
                // here, which is why the pair travels together.
                let m = self.mag_of(site, 1.0);
                self.rounded(site, x.relative(), m)
            }
            PrimOp::Tanh | PrimOp::Sin | PrimOp::Cos => self.lip(site, 1.0, x, 1.0),
            PrimOp::Sigmoid => self.lip(site, 0.25, x, 1.0),
            // |Φ(x) + x·φ(x)| ≤ 1.1290, attained near x ≈ 1.33.
            PrimOp::Gelu => self.lip(site, 1.1290, x, x.mag),
            PrimOp::GeluDeriv => self.lip(site, 1.0844, x, 1.1),
            // |erf′| = 2/√π at 0.
            // ----- reductions and contractions: the order enters here -----
            PrimOp::SumAll | PrimOp::Sum { .. } | PrimOp::SumTo | PrimOp::SumSquares => {
                let m = self.mag_of(site, x.mag * k);
                let c = self.order_constant(k);
                // A sum of squares multiplies its operand by itself.
                let (products, largest) = if matches!(op, PrimOp::SumSquares) { (1.0, x.mag * x.mag) } else { (0.0, x.mag) };
                let ops: Vec<String> = if matches!(op, PrimOp::SumSquares) { operands.iter().chain(operands).cloned().collect() } else { operands.to_vec() };
                if let Some(t) = self.terms(site) {
                    // A sum of squares has terms `xᵢ²`, whose relative error
                    // is twice `x`'s.
                    let rho = if matches!(op, PrimOp::SumSquares) { 2.0 * x.relative() } else { x.relative() };
                    let normwise = t.norms.first().copied().unwrap_or(k) * x.err;
                    // Until 0.18.0 a sum of squares' product was not counted
                    // among its roundings: a plain sum keeps that.
                    let products = if self.plain_sum(self.root(site)) { 0.0 } else { products };
                    return self.summed(site, normwise, rho, &t, 1.0, c, products, k, largest, &ops, m);
                }
                // Partial sums can be larger than the result when terms
                // cancel, so the rounding is counted against `k·|x|` rather
                // than against the output.
                let partial = (x.mag * k).max(m);
                let products = if self.plain_sum(self.root(site)) { 0.0 } else { products };
                Bound { mag: m, floor: 0.0, err: k * x.err + self.sum_ops(site, k, c, products, partial, largest, &ops, false) }
            }
            PrimOp::MeanAll | PrimOp::Mean { .. } | PrimOp::MeanVjp { .. } => {
                // A mean does **not** multiply the error by `k`. The sum of
                // `k` errors is `k·e`, and the division by `k` takes it
                // straight back to `e`: averaging is the one reduction that
                // does not accumulate input error at all. Treating it like a
                // sum cost a factor of the batch size in the first version
                // of these rules.
                let m = self.mag_of(site, x.mag);
                let c = self.order_constant(k);
                // `mean_vjp` spreads a cotangent back over `k` elements; it
                // sums nothing, and nothing is recorded for it.
                if !matches!(op, PrimOp::MeanVjp { .. })
                    && let Some(t) = self.terms(site)
                {
                    // The recorded sums are of the *mean*'s terms `xᵢ/k`
                    // already, except the root-sum-square, recorded as
                    // `√(Σxᵢ²/k)`: independent errors average down by `√k`.
                    return self.summed(site, x.err, x.relative(), &t, 1.0 / k.sqrt(), c, 0.0, k, x.mag, operands, m);
                }
                let partial = (x.mag * k).max(m);
                Bound { mag: m, floor: 0.0, err: x.err + self.sum_ops(site, k, c, 0.0, partial, x.mag, operands, false) / k + self.rt_sum(site, m) }
            }
            PrimOp::Dot | PrimOp::Matmul | PrimOp::Ein { .. } | PrimOp::DotVjpL | PrimOp::DotVjpR => {
                let m = self.mag_of(site, x.mag * y.mag * k);
                let c = self.order_constant(k);
                if let Some(t) = self.terms(site) {
                    let rho: f64 = a.iter().map(|b| b.relative()).sum();
                    let normwise = if t.norms.len() == a.len() {
                        a.iter().zip(&t.norms).map(|(b, n)| b.err * n).sum()
                    } else {
                        k * (x.mag * y.err + y.mag * x.err)
                    };
                    // Each term is a product of `n` operands, rounded
                    // `n − 1` times before it is added.
                    let products = (a.len() as f64 - 1.0).max(0.0);
                    let largest = a.iter().map(|b| b.mag).product();
                    return self.summed(site, normwise, rho, &t, 1.0, c, products, k, largest, operands, m);
                }
                let term = x.mag * y.mag;
                let term_err = x.mag * y.err + y.mag * x.err;
                let partial = (term * k).max(m);
                // Until 0.18.0 this rule counted no product's rounding, nor
                // the store: a plain sum keeps that.
                if self.plain_sum(self.root(site)) {
                    return Bound { mag: m, floor: 0.0, err: k * term_err + c * self.u_acc(site) * partial };
                }
                let products = (a.len() as f64 - 1.0).max(0.0);
                Bound { mag: m, floor: 0.0, err: k * term_err + self.sum_ops(site, k, c, products, partial, term, operands, false) + self.rt_sum(site, m) }
            }
            // ----- the conditioning the specification names -----
            PrimOp::Softmax { .. } => {
                // A softmax is 1-Lipschitz in the ∞ norm and its Jacobian has
                // ∞-norm at most 2, so the error crosses it with a constant
                // and *not* with the row length. This is the single rule that
                // keeps a transformer's bound finite.
                let m = self.mag_of(site, 1.0);
                Bound { mag: m.min(1.0), floor: 0.0, err: 2.0 * x.err + 4.0 * self.rt(site, 1.0) }
            }
            PrimOp::CrossEntropy | PrimOp::CrossEntropyVjp => {
                // `∂(−log softmax(z)_y)/∂z = p − e_y`, of ∞-norm at most 1.
                let m = self.mag_of(site, 1.0);
                self.rounded(site, 2.0 * x.err, m)
            }
        }
    }

    /// The sums of a site's terms, when the model uses them and the envelope
    /// has them. A site that should have them and does not is listed, since
    /// its rule silently reverts to the `k` one.
    fn terms(&mut self, site: &str) -> Option<crate::envelope::Terms> {
        if self.model == Model::Worst {
            return None;
        }
        match self.env.terms(site) {
            Some(t) => Some(t.clone()),
            None => {
                if !self.out.fallback.iter().any(|s| s == site) {
                    self.out.fallback.push(site.to_string());
                }
                None
            }
        }
    }

    /// The rule of a sum whose terms were measured: the error its inputs
    /// carry in, plus the rounding of `c` additions and of each term's
    /// `products`, plus the store.
    ///
    /// `normwise` is the sound propagated term, `Σⱼ errⱼ · Σ Π_{i≠j}|·|`;
    /// `rho` the operands' relative errors summed, which the conditioned and
    /// statistical rules multiply by `Σ|terms|` and `√Σ terms²`; `scale`
    /// corrects the root-sum-square of a mean, recorded per term of the sum.
    ///
    /// The rounding is Higham's `γ_c · Σ|terms|`: every partial sum is at most
    /// `Σ|terms|`, which is sound whatever the model. The statistical model
    /// takes `√c` for `c`, the independent roundings' own cancellation. A
    /// sum's own formats are [`Prop::sum_ops`]'s.
    #[allow(clippy::too_many_arguments)]
    fn summed(&mut self, site: &str, normwise: f64, rho: f64, t: &crate::envelope::Terms, scale: f64, c: f64, products: f64, k: f64, largest: f64, operands: &[String], m: f64) -> Bound {
        let prop = match self.model {
            Model::Worst | Model::Normwise => normwise,
            // Componentwise: `|Σ wᵢ δxᵢ| ≤ ρ Σ|xᵢwᵢ|` when every `|δxᵢ| ≤
            // ρ|xᵢ|`. That holds of a value rounded once and is *assumed*
            // after that — the soundness test is what checks it.
            Model::Conditioned => rho * t.abs,
            Model::Statistical => rho * t.rss * scale,
        };
        let ops = self.sum_ops(site, k, c, products, t.abs, largest, operands, self.model == Model::Statistical);
        Bound { mag: m, floor: 0.0, err: prop + ops + self.rt_sum(site, m) }
    }

    /// `c` in `c · u`, the rounding a sum of `k` terms accumulates.
    fn order_constant(&self, k: f64) -> f64 {
        order_constant(self.realize.order, k)
    }
}

/// `c` in `c · u`, the roundings a term of a sum of `k` terms goes through
/// under an order.
pub fn order_constant(order: ReductionOrder, k: f64) -> f64 {
    match order {
        // Every addition in index order: the classical `k − 1`.
        ReductionOrder::Sequential | ReductionOrder::Blocked { .. } => (k - 1.0).max(0.0),
        // `n` partials, each of about `k/n` terms, summed at the end.
        ReductionOrder::Interleaved { partials } => {
            let n = (partials as f64).max(1.0);
            (k / n - 1.0).max(0.0) + (n - 1.0)
        }
        // Chunks of `size` terms, then the chunks' sums: a term is in at
        // most `size − 1` additions of its chunk, and its chunk's sum in
        // one fewer than there are chunks (plan 0.18.0 step 3).
        ReductionOrder::Chunked { size } => {
            let s = (size as f64).max(1.0).min(k.max(1.0));
            (s - 1.0) + ((k / s).ceil() - 1.0).max(0.0)
        }
    }
}

/// Terms assumed for a reduction whose length could not be read off the
/// types. Deliberately large: a bound that guesses low is not a bound.
const DEFAULT_TERMS: f64 = 1024.0;

/// A value's floor, when the envelope recorded one.
fn floor_of(b: Bound) -> Option<f64> {
    (b.floor > 0.0 && b.floor.is_finite()).then_some(b.floor)
}

fn eval_dim(e: &presk_types::dim::DimExpr, dims: &BTreeMap<presk_types::dim::DimVar, u64>) -> Option<u64> {
    let map: BTreeMap<_, _> = dims.iter().map(|(v, n)| (*v, presk_types::dim::rat(*n as i64))).collect();
    let r = e.eval(&map)?;
    r.is_integer().then(|| num_traits::ToPrimitive::to_u64(&r.to_integer()))?
}

/// Whether a product format counts a sum of scaled values in their unit:
/// an integer or a fixed-point one (D207).
fn counts(f: Format) -> bool {
    matches!(f, Format::Int(_) | Format::Fixed(_))
}

/// The callee's dimension environment: each of its variables bound to the
/// value of the expression the call site gives it.
fn callee_dims(
    dim_args: &BTreeMap<presk_types::dim::DimVar, presk_types::dim::DimExpr>,
    dims: &BTreeMap<presk_types::dim::DimVar, u64>,
) -> BTreeMap<presk_types::dim::DimVar, u64> {
    dim_args.iter().filter_map(|(v, e)| eval_dim(e, dims).map(|n| (*v, n))).collect()
}

/// Elements of a type at concrete dimensions, when it is a tensor.
fn numel(ty: &presk_types::ty::Ty, dims: &BTreeMap<presk_types::dim::DimVar, u64>) -> Option<f64> {
    let s = ty.shape()?;
    if s.prefix.is_some() {
        return None;
    }
    let mut n = 1.0;
    for a in &s.axes {
        n *= eval_dim(&a.len, dims)? as f64;
    }
    Some(n)
}

/// How many terms a reduction or a contraction sums.
///
/// This is where `k` in the error rules comes from, so it is read off the
/// *types* at the dimensions the envelope was measured at rather than
/// guessed — and it has to be read per operation, because the ratio of
/// element counts is wrong whenever a shape has an open prefix `...`, which
/// is most of them. A contraction's length is the axis it contracts, and
/// that axis is nearly always in a closed operand: `dot x, w` contracts
/// `w`'s first axis, whatever the batch in front of `x`.
///
/// Getting this wrong is not a rounding matter. The first version of this
/// function fell back to 1024 for every `dot` under an open prefix and made
/// a two-layer network's bound a thousand times its own value.
fn terms(
    op: &PrimOp,
    atys: &[Option<presk_types::ty::Ty>],
    out_ty: &presk_types::ty::Ty,
    dims: &BTreeMap<presk_types::dim::DimVar, u64>,
) -> f64 {
    let ty = |i: usize| atys.get(i).and_then(|t| t.as_ref());
    let axis_len = |t: Option<&presk_types::ty::Ty>, name: &str| -> Option<f64> {
        let s = t?.shape()?;
        let a = s.axes.iter().find(|a| a.name.as_deref() == Some(name))?;
        eval_dim(&a.len, dims).map(|n| n as f64)
    };
    let nth_len = |t: Option<&presk_types::ty::Ty>, i: usize| -> Option<f64> {
        let s = t?.shape()?;
        eval_dim(&s.axes.get(i)?.len, dims).map(|n| n as f64)
    };
    let ratio = || match (ty(0).and_then(|t| numel(t, dims)), numel(out_ty, dims)) {
        (Some(i), Some(o)) if o > 0.0 && i >= o => Some((i / o).max(1.0)),
        _ => None,
    };
    let k = match op {
        // `dot`/`matmul` contract the last axis of the left operand with the
        // first of the right. The right operand is a weight, hence closed.
        PrimOp::Dot | PrimOp::Matmul => nth_len(ty(1), 0),
        // A reduction over a named axis: that axis, wherever it sits.
        PrimOp::Sum { axis } | PrimOp::Mean { axis } => axis_len(ty(0), axis),
        PrimOp::SumAll | PrimOp::MeanAll | PrimOp::SumSquares | PrimOp::SumTo => ratio(),
        PrimOp::Ein { .. } | PrimOp::DotVjpL | PrimOp::DotVjpR | PrimOp::MeanVjp { .. } => ratio(),
        _ => Some(1.0),
    };
    k.filter(|x| x.is_finite() && *x >= 1.0).unwrap_or(DEFAULT_TERMS)
}
