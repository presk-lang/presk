//! One optimizer step as the thing a precision budget binds (plan 0.6.0
//! §3.7, step 7b, D52).
//!
//! A `training run` that declares `precision = budget(rel_error = ε)` makes a
//! claim about **one step's parameter update**: from an optimizer state
//! `s = (θ, m, v, t)` and a batch, the step under a realization, `Ŝ(s)`, and
//! the exact step `S(s)` — the objective's `vjp`, then `adamw.step`, over
//! the reals — satisfy
//!
//! ```text
//! ‖θ̂′ − θ′‖₂ / ‖θ′ − θ‖₂ ≤ ε
//! ```
//!
//! on the states and batches it was measured on. This module builds such a
//! state and runs the step; the measuring is the caller's, since the exact
//! reference and the native backends live in different crates.
//!
//! **The state** is a checkpoint's `θ` followed by `N` steps of the run's own
//! pipeline in `f64`, so that `v` is not the zero of initialisation and
//! AdamW's division by `√v` acts as it does in training.
//!
//! **"The same state"** is the state rounded to the realization's default
//! type ([`StepSetup::input`]). An `f32` run's parameters already live in
//! `f32`, so rounding them on the way in is not an error of its step; rounding
//! `θ′` on the way *out* is, and it stays in the measurement — it is why
//! `bf16` master weights fail.

use std::path::PathBuf;

use presk_core::Program;
use presk_data::{Pipeline, Reader};
use presk_interp::eval::Dims;
use presk_interp::key::Key;
use presk_rt::{Backend, Dtype, Realization, Value};
use presk_types::check::TypedProgram;
use presk_types::elab::ElabProgram;

use crate::{Result, TrainError, TrainingFns, batch_dim, batch_value, err, gradient, update};

/// Where the measured state comes from, and which batches the step sees.
#[derive(Clone, Debug)]
pub struct StepOptions {
    /// The checkpoint `θ` starts from.
    pub weights: PathBuf,
    /// Warm-up steps in `f64` before the measured one.
    pub warmup: u64,
    /// The data file of the measured batch (the run's own by default).
    pub data: Option<PathBuf>,
    /// The measured batch is samples `from..from + samples`; the second
    /// batch, for the sampling noise, is the next `samples`.
    pub from: usize,
    pub samples: usize,
    /// The schedule's horizon (the run's by default).
    pub steps: Option<u64>,
    pub seed: u64,
}

/// A state, a batch, and everything needed to take one step from it.
#[derive(Clone, Debug)]
pub struct StepSetup {
    pub fns: TrainingFns,
    /// The optimizer step's function name (`adamw.step`): where `θ′` is
    /// stored, for [`StepGap::floor`].
    pub opt_name: String,
    pub theta: Value,
    /// `(m, v)` after the warm-up.
    pub state: Value,
    /// The index of the measured step, `warmup + 1`, and the horizon.
    pub t: u64,
    pub total: u64,
    /// The per-step key, when the objective takes one.
    pub key: Option<Value>,
    pub batch: Value,
    /// A disjoint batch of the same size: the gradient's sampling noise.
    pub batch2: Value,
    /// The dimensions of the measured step (the batch axis bound to
    /// `samples`, not to the run's `B`).
    pub d_obj: Dims,
    pub d_opt: Dims,
    pub data: PathBuf,
}

/// The step's result: the new parameters and the gradient they came from.
#[derive(Clone, Debug)]
pub struct StepResult {
    pub theta: Value,
    pub grad: Value,
}

/// The two gaps a step budget is read against.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepGap {
    /// `‖θ̂′ − θ′‖₂ / ‖θ′ − θ‖₂`: what the budget binds.
    pub update: f64,
    /// `‖ĝ − g‖₂ / ‖g‖₂`: where the error enters, reported beside it.
    pub grad: f64,
    /// `‖round(θ′) − θ′‖₂ / ‖θ′ − θ‖₂`, `θ′` exact and rounded to the type
    /// the optimizer stores it in: the part of the update gap no arithmetic
    /// can remove. The update is a small difference of large numbers, so
    /// this is `u·‖θ‖/‖θ′ − θ‖` — at a warm-up learning rate it is large
    /// even in `f64`, and it is why `bf16` master weights cannot work.
    pub floor: f64,
}

/// The dimensions of `f` from the run's, with the batch axis's own
/// dimension rebound to `batch` when given.
fn dims_of(core: &Program, elab: &ElabProgram, run: &presk_types::check::training::TrainingRun, f: presk_core::FuncId, batch: Option<(String, u64)>) -> Result<Dims> {
    let mut d = Dims::new();
    for v in &core.func(f).dims {
        let name = elab.dim_name(*v);
        let n = match &batch {
            Some((b, n)) if *b == name => Some(*n),
            _ => run.dims.iter().find(|(w, _)| elab.dim_name(**w) == name).map(|(_, n)| *n),
        };
        match n {
            Some(n) => {
                d.insert(*v, n);
            }
            None => return err("E7001", format!("dimension `{name}` of `{}` is not fixed by the training run", core.func(f).name)),
        }
    }
    Ok(d)
}

/// The state and batches of a measured step: `θ` from the checkpoint, then
/// `opts.warmup` steps of the run's pipeline on `warm`, which should be a
/// native `f64` backend (the warm-up only has to be realistic and
/// reproducible, not exact).
pub fn setup(warm: &mut dyn Backend, core: &Program, elab: &ElabProgram, typed: &TypedProgram, fns: &TrainingFns, opts: &StepOptions) -> Result<StepSetup> {
    let run = typed.training.first().ok_or_else(|| TrainError { code: "E7001", message: "no `training run` in this file".into(), span: None })?;
    let Some(pipe) = &run.data else {
        return err("E7001", "the training run has no `data` field");
    };
    let mut reader = Reader::open(&pipe.path)?;
    let layout = reader.layout()?.clone();
    let count = reader.len();
    let pipeline = Pipeline { shuffle: pipe.shuffle, batch: pipe.batch, take: pipe.take };
    let per_epoch = pipeline.batches_per_epoch(count) as u64;
    if per_epoch == 0 {
        return err("E9003", format!("{}: {count} samples, fewer than one batch of {}", pipe.path.display(), pipe.batch));
    }
    let total = opts.steps.unwrap_or(run.epochs.unwrap_or(1) * per_epoch).max(opts.warmup + 1);
    let one = Realization::uniform(Dtype::F64);
    let d_run = dims_of(core, elab, run, fns.objective, None)?;
    let d_opt_run = dims_of(core, elab, run, fns.opt_step, None)?;
    let vjp = warm.prepare(fns.vjp, &d_run, &one)?;
    let opt_init = warm.prepare(fns.opt_init, &d_opt_run, &one)?;
    let opt_step = warm.prepare(fns.opt_step, &d_opt_run, &one)?;
    warm.set_training(true);
    let run_key = Key::from_seed(opts.seed);
    let mut theta = presk_rt::load_params(&opts.weights, &{
        // The checkpoint's tree is read against the initializer's, so a
        // checkpoint of other dimensions is refused (D48).
        let p = warm.prepare(fns.init, &dims_of(core, elab, run, fns.init, None)?, &one)?;
        warm.call(&p, vec![Value::Key(run_key.clone())])?
    })
    .map_err(|e| TrainError { code: e.code, message: format!("{}: {}", opts.weights.display(), e.message), span: None })?;
    let mut state = warm.call(&opt_init, vec![theta.clone()])?;
    let key_at = |t: u64| fns.key.then(|| Value::Key(run_key.child(t)));
    let mut t = 0u64;
    'warm: for epoch in 1.. {
        for group in pipeline.batches(count, epoch) {
            if t >= opts.warmup {
                break 'warm;
            }
            t += 1;
            let samples: Vec<presk_data::Value> = group.iter().map(|i| reader.get(*i)).collect::<std::result::Result<_, _>>()?;
            let refs: Vec<&presk_data::Value> = samples.iter().collect();
            let batch = batch_value(&run.batch_ty, &layout, &refs, &d_run)?;
            let g = gradient(warm, &vjp, vjp_args(&theta, key_at(t), batch))?;
            (theta, state) = update(warm, &opt_step, theta, state, g, t, total)?;
        }
    }
    // The measured batch, at its own size.
    let data = opts.data.clone().unwrap_or_else(|| pipe.path.clone());
    let mut reader = Reader::open(&data)?;
    let layout = reader.layout()?.clone();
    if opts.samples == 0 || opts.from + 2 * opts.samples > reader.len() {
        return err("E9003", format!("{}: {} samples, and two batches of {} from {} are needed", data.display(), reader.len(), opts.samples, opts.from));
    }
    let bname = batch_dim(&run.batch_ty, elab).map(|b| (b, opts.samples as u64));
    let d_obj = dims_of(core, elab, run, fns.objective, bname.clone())?;
    let d_opt = dims_of(core, elab, run, fns.opt_step, bname)?;
    let mut read = |from: usize| -> Result<Value> {
        let samples: Vec<presk_data::Value> = (from..from + opts.samples).map(|i| reader.get(i)).collect::<std::result::Result<_, _>>()?;
        let refs: Vec<&presk_data::Value> = samples.iter().collect();
        batch_value(&run.batch_ty, &layout, &refs, &d_obj)
    };
    let batch = read(opts.from)?;
    let batch2 = read(opts.from + opts.samples)?;
    let t = opts.warmup + 1;
    Ok(StepSetup { fns: fns.clone(), opt_name: core.func(fns.opt_step).name.clone(), theta, state, t, total, key: key_at(t), batch, batch2, d_obj, d_opt, data })
}

fn vjp_args(theta: &Value, key: Option<Value>, batch: Value) -> Vec<Value> {
    let mut a = vec![theta.clone()];
    a.extend(key);
    a.push(batch);
    a.push(Value::Real(presk_rt::Tensor::scalar_f64(1.0)));
    a
}

impl StepSetup {
    /// The state as a realization with default `d` holds it: every real
    /// leaf rounded once to `d`.
    pub fn input(&self, d: Dtype) -> (Value, Value) {
        (self.theta.realize(d), self.state.realize(d))
    }

    /// One step on `backend` under `realize`, from the state rounded to the
    /// realization's default. On the reference interpreter the realization
    /// is ignored and the step is exact from that rounded state.
    pub fn step(&self, backend: &mut dyn Backend, realize: &Realization) -> Result<StepResult> {
        self.step_on(backend, realize, realize.default, false)
    }

    /// The step under `realize` from the state rounded to `input` instead of
    /// the realization's default: an `f64` step standing in for the exact
    /// one where the exact one is not affordable (at course scale).
    pub fn step_from(&self, backend: &mut dyn Backend, realize: &Realization, input: Dtype) -> Result<StepResult> {
        self.step_on(backend, realize, input, false)
    }

    /// The same step on the second batch: its gradient against the first
    /// one's is the sampling noise.
    pub fn step_other_batch(&self, backend: &mut dyn Backend, realize: &Realization) -> Result<StepResult> {
        self.step_on(backend, realize, realize.default, true)
    }

    fn step_on(&self, backend: &mut dyn Backend, realize: &Realization, input: Dtype, other: bool) -> Result<StepResult> {
        let vjp = backend.prepare(self.fns.vjp, &self.d_obj, realize)?;
        let opt_step = backend.prepare(self.fns.opt_step, &self.d_opt, realize)?;
        backend.set_training(true);
        let (theta, state) = self.input(input);
        let batch = if other { self.batch2.clone() } else { self.batch.clone() };
        let grad = gradient(backend, &vjp, vjp_args(&theta, self.key.clone(), batch))?;
        let (theta, _) = update(backend, &opt_step, theta, state, grad.clone(), self.t, self.total)?;
        Ok(StepResult { theta, grad })
    }

    /// The gaps of `got`, run under `realize`, against `exact`, both from
    /// the state rounded to the realization's default.
    pub fn gap(&self, realize: &Realization, exact: &StepResult, got: &StepResult) -> StepGap {
        let theta_in = self.theta.realize(realize.default);
        let step = exact.theta.dist2_f64(&theta_in).sqrt().max(f64::MIN_POSITIVE);
        let gnorm = exact.grad.sum_squares_f64().sqrt().max(f64::MIN_POSITIVE);
        let stored = realize.functions.get(&self.opt_name).copied().unwrap_or(realize.default);
        StepGap {
            update: got.theta.dist2_f64(&exact.theta).sqrt() / step,
            grad: got.grad.dist2_f64(&exact.grad).sqrt() / gnorm,
            floor: exact.theta.realize(stored).dist2_f64(&exact.theta).sqrt() / step,
        }
    }
}

/// `‖g_B − g_B′‖₂ / ‖g_B‖₂`: how far two batches' gradients are apart, the
/// scale a rounding error in training has to be compared with.
pub fn sampling_noise(a: &StepResult, b: &StepResult) -> f64 {
    a.grad.dist2_f64(&b.grad).sqrt() / a.grad.sum_squares_f64().sqrt().max(f64::MIN_POSITIVE)
}
