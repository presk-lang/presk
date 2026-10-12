//! The precision stages as values: the envelope (`presk calibrate
//! --precision`), the bound (`presk check --precision`), the sensitivities
//! (`presk dump --stage sensitivity`) and the solve (`presk solve`). Each
//! returns what its command prints, and its errors as values; the command
//! prints both, as before 0.11.0.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use presk_core::FuncId;
use presk_interp::RunOptions;
use presk_interp::eval::Dims;
use presk_precision::envelope::{DataSlice, Envelope};
use presk_precision::propagate::{Analysis, Model};
use presk_precision::quant::QuantSolution;
use presk_precision::sensitivity::Sensitivity;
use presk_precision::solve::{self, Machine, Menu, Solution};
use presk_rt::{Backend, Buf, Dtype, InterpBackend, QuantScheme, Realization, Tensor, Value};
use presk_syntax::SourceMap;
use presk_syntax::diag::has_errors;
use presk_types::elab::ElabProgram;

use crate::Report;
use crate::entry::{entry_args, entry_dims};
use crate::gap::{leaf_gap, leaf_max, params_cotangent};

/// Whether an entry returns a scalar `ℝ[]`, which a sensitivity `∂L/∂v`
/// needs.
pub fn returns_scalar(core: &presk_core::Program, f: FuncId) -> bool {
    matches!(core.func(f).ret, presk_types::ty::Ty::Real(ref s) if s.axes.is_empty() && s.prefix.is_none())
}

/// The cotangent `1` of a scalar output.
fn seed_one() -> Value {
    Value::Real(Tensor { axes: vec![], data: Buf::F64(vec![1.0].into()) })
}

/// What `presk calibrate --precision` measures an envelope of.
pub struct Calibrate<'a> {
    /// The program's file, as the envelope names it.
    pub file: &'a Path,
    pub entry: &'a str,
    /// `--dim` pairs.
    pub dims: Vec<(String, u64)>,
    /// `--const` as given, which the envelope keeps.
    pub consts: &'a [String],
    pub data: Option<&'a Path>,
    pub samples: Option<usize>,
    pub from: usize,
    pub seed: u64,
    pub bits: u32,
    pub weights: Option<&'a Path>,
}

/// The magnitude envelope of an entry (plan 0.5.0 §3.2): the entry run once
/// on the reference interpreter, the largest `|x|` at every site recorded,
/// with the floors, the element counts, the operations and the sums' terms.
/// The error is the message `presk calibrate: ` prefixes.
pub fn calibrate(core: &presk_core::Program, prog: &ElabProgram, c: &Calibrate) -> Result<Envelope, String> {
    let opts = RunOptions { entry: c.entry.into(), dims: c.dims.clone(), seed: c.seed, bits: c.bits, digits: 17 };
    let mut backend = InterpBackend::new(core, prog, c.bits);
    backend.interp.magnitudes = Some(Default::default());
    backend.interp.minima = Some(Default::default());
    backend.interp.elems = Some(Default::default());
    backend.interp.ops = Some(Default::default());
    backend.interp.terms = Some(Default::default());
    let today = presk_precision::envelope::today();
    let mut env = Envelope::new(&c.file.display().to_string(), c.entry, c.bits, &today);
    env.dims = opts.dims.clone();
    env.consts = c.consts.to_vec();
    env.weights = c.weights.map(|w| w.display().to_string());
    env.data = c.data.map(|d| DataSlice { path: d.display().to_string(), from: c.from, samples: c.samples });
    entry_args(core, prog, &opts, env.data.as_ref(), c.weights).and_then(|(func, fdims, args)| {
        let one = Realization::uniform(Dtype::F64);
        Backend::prepare(&mut backend, func, &fdims, &one).and_then(|p| Backend::call(&mut backend, &p, args)).map_err(|e| e.message)
    })?;
    env.observe(&backend.interp.magnitudes.take().unwrap_or_default());
    env.observe_minima(&backend.interp.minima.take().unwrap_or_default());
    env.observe_elems(&backend.interp.elems.take().unwrap_or_default());
    env.observe_ops(&backend.interp.ops.take().unwrap_or_default());
    env.observe_terms(&backend.interp.terms.take().unwrap_or_default());
    env.samples = c.samples.unwrap_or(1).max(1);
    Ok(env)
}

/// The bound of `presk check --precision`, by every rule asked, and the gap
/// measured beside it.
pub struct Bounded {
    pub entry: FuncId,
    /// The dimensions the bound is at: the envelope's, and a module entry's
    /// own arguments.
    pub dims: Dims,
    /// The realization run on the kernel interpreter against the exact
    /// reference, on the envelope's own inputs and weights (`--measure`).
    pub measured: Option<f64>,
    /// One analysis per rule, in the order asked.
    pub runs: Vec<Analysis>,
}

/// Why a bound was not given.
pub enum BoundError {
    /// The envelope's entry is not one of this program.
    NotEntry,
    /// The measurement failed; the message `presk check: ` prefixes.
    Measure(String),
}

/// The bound of the entry the envelope was measured on, under `realize`, by
/// each of `models`, at the envelope's shapes (`k` in a contraction's rule
/// is a shape, so a bound at other dimensions is about another program),
/// and with `measure` the gap first, since every rule is compared with it.
pub fn bound(core: &presk_core::Program, prog: &ElabProgram, env: &Envelope, realize: &Realization, models: &[Model], measure: bool) -> Result<Bounded, BoundError> {
    let Ok((entry, dims)) = entry_dims(core, prog, &env.entry, &env.dims) else {
        return Err(BoundError::NotEntry);
    };
    let measured = if measure {
        let opts = RunOptions { entry: env.entry.clone(), dims: env.dims.clone(), seed: 0, bits: env.bits, digits: 17 };
        let weights = env.weights.as_ref().map(PathBuf::from);
        let (func, fdims, args) = entry_args(core, prog, &opts, env.data.as_ref(), weights.as_deref()).map_err(BoundError::Measure)?;
        let one = Realization::uniform(Dtype::F64);
        let mut exact = InterpBackend::new(core, prog, env.bits);
        let reference = Backend::prepare(&mut exact, func, &fdims, &one).and_then(|p| Backend::call(&mut exact, &p, args.clone()));
        let mut b = presk_kernel::KernelBackend::new(core, prog, realize);
        let mut a: Vec<Value> = args.iter().map(|v| v.realize(realize.default)).collect();
        let got = realize.quantize_args(&mut a).and_then(|_| Backend::prepare(&mut b, func, &fdims, realize)).and_then(|p| Backend::call(&mut b, &p, a));
        match (reference, got) {
            (Ok(r), Ok(g)) => Some(leaf_gap(&r, &g)),
            (Err(e), _) | (_, Err(e)) => return Err(BoundError::Measure(e.message)),
        }
    } else {
        None
    };
    let runs = models.iter().map(|m| presk_precision::propagate::propagate_with(core, prog, entry, &dims, env, realize, *m)).collect();
    Ok(Bounded { entry, dims, measured, runs })
}

/// Why the sensitivities were not given; each is printed by the command.
pub enum SensitivityError {
    NotEntry,
    NotScalar,
    /// The reverse sweep's diagnostics, reported, had an error.
    Diagnostics,
    /// The inputs could not be bound.
    Inputs,
    /// The sweep or the weights failed: the message.
    Run(String),
}

/// `‖∂L/∂v‖₁` at every site of `entry`, read off one reverse sweep on the
/// reference interpreter at `bits` (plan 0.5.0 §3.3), on the inputs of
/// `seed` and, when given, the trained `weights`.
#[allow(clippy::too_many_arguments)]
pub fn sensitivity(core: &mut presk_core::Program, prog: &ElabProgram, sm: &SourceMap, entry: &str, dims: Vec<(String, u64)>, seed: u64, bits: u32, weights: Option<&Path>, report: &mut dyn Report) -> Result<Sensitivity, SensitivityError> {
    let Ok((f, _, _)) = presk_interp::run::resolve_entry(core, prog, entry) else {
        return Err(SensitivityError::NotEntry);
    };
    if !returns_scalar(core, f) {
        return Err(SensitivityError::NotScalar);
    }
    // The reverse sweep already computes the adjoint of every intermediate;
    // this only asks where each one ended up.
    let (vjp, adjoints, diags) = presk_core::ad::vjp_with_adjoints(core, f);
    report.diagnostics(&diags, sm);
    if has_errors(&diags) {
        return Err(SensitivityError::Diagnostics);
    }
    let opts = RunOptions { entry: entry.into(), dims, seed, bits, digits: 17 };
    let Ok((_, fdims, inputs)) = presk_interp::run::prepare_run(core, prog, &opts) else {
        return Err(SensitivityError::Inputs);
    };
    let mut backend = InterpBackend::new(core, prog, bits);
    backend.interp.sums = Some(Default::default());
    // The vjp takes the forward arguments and a cotangent; seeding it with 1
    // is what makes the adjoints `∂L/∂v` rather than a scaled multiple.
    let mut args: Vec<Value> = inputs.iter().map(|(_, v)| Value::from_interp(v).unwrap()).collect();
    if let Some(path) = weights {
        args[0] = presk_rt::load_params(path, &args[0]).map_err(|e| SensitivityError::Run(e.message))?;
    }
    args.push(seed_one());
    let one = Realization::uniform(Dtype::F64);
    let p = Backend::prepare(&mut backend, vjp, &fdims, &one).map_err(|e| SensitivityError::Run(e.message))?;
    Backend::call(&mut backend, &p, args).map_err(|e| SensitivityError::Run(e.message))?;
    let sums = backend.interp.sums.take().unwrap_or_default();
    Ok(presk_precision::sensitivity::from_run(core, f, &adjoints, &sums))
}

/// What `presk solve` solves under.
pub struct SolveArgs<'a> {
    /// The relative budget.
    pub budget: f64,
    pub weights: Option<&'a Path>,
    /// Verify by measurement (no `--no-verify`).
    pub verify: bool,
    pub bits: u32,
    pub seed: u64,
    /// A machine profile: the objective is then time, and traffic without.
    pub profile: Option<&'a Path>,
    /// Also offer the matrix parameters as `int8` or `int4`.
    pub quant: Option<QuantScheme>,
}

/// What `presk solve` found.
pub struct Solved {
    pub entry: FuncId,
    pub sens: Sensitivity,
    pub solution: Solution,
    /// The parameter menu's, with `--quant`.
    pub quant: Option<QuantSolution>,
    /// The largest `|x|` of the exact output.
    pub reference_max: f64,
    /// The site of the entry's result.
    pub output_site: String,
    /// The exhaustive search under the same estimate, when the space is
    /// small enough: its cost.
    pub optimal: Option<f64>,
    /// Every site the solver offered, its type, its term of the estimate
    /// and its cost, beside the widest type's (plan 0.11.0 step 2).
    pub breakdown: Vec<presk_precision::solve::SiteCost>,
    /// Whether the costs are times, a profile's roofline pricing them, or
    /// bytes.
    pub time: bool,
}

/// Why `presk solve` gave no realization; each is printed by the command.
pub enum SolveError {
    NotEntry,
    NotScalar,
    /// The inputs could not be bound: the message.
    Inputs(String),
    /// The reverse sweep failed: the message.
    NoGradient(String),
    /// The exact reference failed: the message.
    Reference(String),
    /// `--quant` without the parameters' gradient.
    QuantWithoutGradient,
}

/// A machine profile's roofline, pricing *time*: 0.5.0 minimised traffic
/// and lost, since on a compute-bound site narrowing buys nothing and the
/// conversions cost.
struct Roofline(presk_opt::Profile);

impl Machine for Roofline {
    fn seconds(&self, d: Dtype, work: f64, bytes: f64) -> f64 {
        self.0.predict_for(d.name(), work, bytes)
    }
}

/// Solve for a realization of the envelope's entry under a relative budget
/// (plan 0.5.0 §3.4): the sites ranked by one reverse sweep, a greedy
/// proposal under the first-order estimate, verified against the exact
/// reference and widened until it holds; with `--quant`, the parameter menu
/// on top.
pub fn solve(core: &mut presk_core::Program, prog: &ElabProgram, env: &Envelope, a: &SolveArgs) -> Result<Solved, SolveError> {
    let Ok((entry, _, _)) = presk_interp::run::resolve_entry(core, prog, &env.entry) else {
        return Err(SolveError::NotEntry);
    };
    // The sites are ranked by `∂L/∂v`, as `dump --stage sensitivity` ranks
    // them: refused here, not after the reverse sweep's forward half.
    if !returns_scalar(core, entry) {
        return Err(SolveError::NotScalar);
    }
    let opts = RunOptions { entry: env.entry.clone(), dims: env.dims.clone(), seed: a.seed, bits: a.bits, digits: 17 };
    // The envelope's own inputs: synthesized, or the data slice it was
    // measured on (0.6.0 step 6). A checkpoint that does not load is an
    // error, never a fallback to the initial parameters: 0.5.0 read a
    // `T = 256` checkpoint at `T = 32` without a word (D48).
    let (_, fdims, args) = entry_args(core, prog, &opts, env.data.as_ref(), a.weights).map_err(SolveError::Inputs)?;

    // Where the budget should *not* be spent: one reverse sweep.
    let (vjp, adjoints, _) = presk_core::ad::vjp_with_adjoints(core, entry);
    let core: &presk_core::Program = core;
    let mut sb = InterpBackend::new(core, prog, a.bits);
    sb.interp.sums = Some(Default::default());
    let mut vargs = args.clone();
    vargs.push(seed_one());
    let one = Realization::uniform(Dtype::F64);
    // The same sweep's cotangent of the parameters, `∂L/∂θ`, prices the
    // parameter menu (0.6.0 step 5d); kept only when `--quant` asks for it,
    // and the sweep's result dropped once read: it is parameter-sized (plan
    // 0.10.1 step 3).
    let (sens, grad_theta) = match Backend::prepare(&mut sb, vjp, &fdims, &one).and_then(|p| Backend::call(&mut sb, &p, vargs)) {
        Ok(g) => (presk_precision::sensitivity::from_run(core, entry, &adjoints, &sb.interp.sums.take().unwrap_or_default()), if a.quant.is_some() { params_cotangent(&g, &args[0]) } else { None }),
        Err(e) => return Err(SolveError::NoGradient(e.message)),
    };

    // The exact value, to measure candidates against.
    let mut exact = InterpBackend::new(core, prog, a.bits);
    let reference = Backend::prepare(&mut exact, entry, &fdims, &one).and_then(|p| Backend::call(&mut exact, &p, args.clone())).map_err(|e| SolveError::Reference(e.message))?;
    let reference_max = leaf_max(&reference);
    let output_site = format!("{}/%{}", core.func(entry).name, core.func(entry).body.result.0);

    let mut measure = |r: &Realization| -> f64 {
        let mut b = presk_kernel::KernelBackend::new(core, prog, r);
        let mut a: Vec<Value> = args.iter().map(|v| v.realize(r.default)).collect();
        if r.quantize_args(&mut a).is_err() {
            return f64::INFINITY;
        }
        match Backend::prepare(&mut b, entry, &fdims, r).and_then(|p| Backend::call(&mut b, &p, a)) {
            Ok(v) => leaf_gap(&reference, &v),
            Err(_) => f64::INFINITY,
        }
    };
    let menu = Menu::default();
    let (machine, time): (Box<dyn Machine>, bool) = match a.profile.map(presk_opt::Profile::load) {
        Some(Ok(p)) => (Box::new(Roofline(p)), true),
        _ => (Box::new(solve::Traffic), false),
    };
    // The sites that are the solver's to choose, and how the others are
    // rounded (plan 0.11.0 step 2b, D112).
    let roles = solve::Roles::of(core);
    let mut sol = if a.verify {
        solve::solve(env, &sens, &roles, &output_site, a.budget, &menu, Dtype::F64, machine.as_ref(), Some(&mut measure))
    } else {
        solve::solve(env, &sens, &roles, &output_site, a.budget, &menu, Dtype::F64, machine.as_ref(), None)
    };
    // The parameter menu, on top of the float assignment and under what it
    // left of the budget (0.6.0 step 5d). Verified, the measurement is of
    // the whole realization, so the budget is the whole budget.
    let quant = match (a.quant, &grad_theta) {
        (Some(scheme), Some(g)) => {
            // Ranks as declared, so a stacked vector is not taken for a matrix.
            let declared: BTreeMap<String, usize> = match &core.func(entry).params.first().map(|p| &p.ty) {
                Some(presk_types::ty::Ty::Params(m)) => prog.params_tree(m).declared_ranks().into_iter().collect(),
                _ => Default::default(),
            };
            let cands = presk_precision::quant::candidates(&args[0], g, &declared, sol.realization.default.bytes(), scheme);
            let q = if a.verify {
                presk_precision::quant::solve(cands, sol.budget_abs, &sol.realization, scheme, Some(&mut measure))
            } else {
                let left = (sol.budget_abs - sol.estimated_error).max(0.0);
                presk_precision::quant::solve(cands, left, &sol.realization, scheme, None)
            };
            sol.realization.quant = q.quant.clone();
            Some(q)
        }
        (Some(_), None) => return Err(SolveError::QuantWithoutGradient),
        _ => None,
    };
    let optimal = solve::exhaustive(env, &sens, &roles, &output_site, a.budget, &menu, Dtype::F64, machine.as_ref(), 200_000).map(|(_, bytes)| bytes);
    let breakdown = solve::breakdown(env, &sens, &roles, &sol.realization, &menu, Dtype::F64, machine.as_ref());
    Ok(Solved { entry, sens, solution: sol, quant, reference_max, output_site, optimal, breakdown, time })
}

/// The program of the sensitivities' sweep (plan 0.11.0 step 2): the
/// entry's `.vjp` that keeps the adjoint of every intermediate, which `presk
/// dump --stage sensitivity` and `presk solve` run and no stage printed. It
/// is added to `core`; the functions added are returned, the sweep first,
/// with differentiation's diagnostics.
pub fn sensitivity_program(core: &mut presk_core::Program, entry: FuncId) -> (Vec<FuncId>, Vec<presk_syntax::Diagnostic>) {
    let before = core.funcs.len();
    let (vjp, _, diags) = presk_core::ad::vjp_with_adjoints(core, entry);
    let mut added: Vec<FuncId> = vec![vjp];
    added.extend((before..core.funcs.len()).map(|i| FuncId(i as u32)).filter(|f| *f != vjp));
    (added, diags)
}

