//! `presk train`: the loop of a `training run` over the reference interpreter
//! (plan 0.2.0, step 5).
//!
//! ```text
//! θ₀ = M.init(key(seed))        s₀ = adamw.init(θ₀)        t = 0
//! for epoch in 1..=epochs:
//!   for batch in pipeline(epoch):
//!     t += 1
//!     ℓ = objective(θ, batch);   g = objective.vjp(θ, batch, 1).θ
//!     (θ, s) = adamw.step(θ, s, g, t, total)
//!     log; checkpoint every N steps
//! write ckpt/final.psk.d
//! ```
//!
//! Everything with a semantics is PSK-Core (the objective, its `vjp`, the
//! optimizer) or the documented data pipeline; this crate only sequences them
//! and converts between `.psk.d` samples and interpreter values.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use presk_core::{FuncId, PrimOp, Program};
use presk_data::{Layout, Leaf, Pipeline, Reader};
use presk_interp::eval::Dims;
use presk_interp::key::Key;
use presk_interp::value::Axis;
use presk_interp::{Interp, Tensor, Value};
use presk_syntax::Span;
use presk_types::check::TypedProgram;
use presk_types::elab::ElabProgram;
use presk_types::ty::{ModRef, Ty};
use rug::Float;

#[derive(Clone, Debug)]
pub struct TrainOptions {
    /// Overrides the run's `data` path.
    pub data: Option<PathBuf>,
    /// Stop after this many steps (the schedule's horizon becomes this too).
    pub steps: Option<u64>,
    /// Overrides the run's `epochs`.
    pub epochs: Option<u64>,
    pub seed: u64,
    pub bits: u32,
    /// Overrides the run's checkpoint directory.
    pub ckpt: Option<PathBuf>,
    /// Write checkpoints and `final.psk.d` at all (the conformance replay does not).
    pub checkpoints: bool,
}

#[derive(Clone, Debug)]
pub struct StepLog {
    pub step: u64,
    pub epoch: u64,
    pub loss: f64,
    pub lr: f64,
    pub grad_norm: f64,
}

#[derive(Debug)]
pub struct TrainOutput {
    pub model: String,
    pub logs: Vec<StepLog>,
    /// The initial parameters (from the initializer and `--seed`).
    pub theta0: Value<Float>,
    /// The sample indices of every batch, in step order (what the K-step
    /// conformance test replays).
    pub batches: Vec<Vec<usize>>,
    pub theta: Value<Float>,
    /// `ckpt/final.psk.d` when a checkpoint directory was known.
    pub final_path: Option<PathBuf>,
    pub checkpoints: Vec<PathBuf>,
    pub total: u64,
}

#[derive(Clone, Debug)]
pub struct TrainError {
    pub code: &'static str,
    pub message: String,
    pub span: Option<Span>,
}

pub type Result<T> = std::result::Result<T, TrainError>;

fn err<T>(code: &'static str, message: impl Into<String>) -> Result<T> {
    Err(TrainError { code, message: message.into(), span: None })
}

impl From<presk_interp::RunError> for TrainError {
    fn from(e: presk_interp::RunError) -> Self {
        TrainError { code: "E7001", message: e.message, span: e.span }
    }
}

impl From<presk_data::DataError> for TrainError {
    fn from(e: presk_data::DataError) -> Self {
        TrainError { code: e.code, message: e.message, span: None }
    }
}

/// Runs the (first) `training run` of the program. `core` must have gone
/// through `presk_core::ad::transform`; the objective's `vjp` is generated here.
pub fn train(core: &mut Program, elab: &ElabProgram, typed: &TypedProgram, opts: &TrainOptions, mut on_step: impl FnMut(&StepLog)) -> Result<TrainOutput> {
    let Some(run) = typed.training.first() else {
        return err("E7001", "no `training run` in this file");
    };
    let ModRef::Inst { def, .. } = &run.model else {
        return err("E7001", "the model of a training run must be a plain module");
    };
    let model_name = elab.module(*def).name.clone();
    // Functions.
    // By origin, not by name: an imported file may define a function of the same name.
    let objective = core
        .funcs
        .iter()
        .position(|f| matches!(f.origin, presk_core::ir::Origin::Source(id) if id == run.objective))
        .map(|i| FuncId(i as u32))
        .ok_or_else(|| TrainError { code: "E7001", message: "objective not lowered".into(), span: None })?;
    let (vjp, diags) = presk_core::ad::vjp_function(core, objective);
    if let Some(d) = diags.iter().find(|d| d.is_error()) {
        return Err(TrainError { code: "E6001", message: format!("cannot differentiate the objective: {}", d.message), span: d.primary_span() });
    }
    presk_core::verify(core).map_err(|e| TrainError { code: "E7001", message: format!("internal error after differentiation: {}", e.join("; ")), span: None })?;
    let init = core.find(&format!("{model_name}.init")).ok_or_else(|| TrainError { code: "E7001", message: "no initializer".into(), span: None })?;
    let opt = &run.optimizer;
    let (o_init, o_step, o_lr) = (core.find(&format!("{opt}.init")), core.find(&format!("{opt}.step")), core.find(&format!("{opt}.lr")));
    let (Some(o_init), Some(o_step), Some(o_lr)) = (o_init, o_step, o_lr) else {
        return err("E7001", format!("optimizer `{opt}` has no lowered functions (only `adamw` is implemented)"));
    };
    // Data.
    let Some(pipe) = &run.data else {
        return err("E7001", "the training run has no `data` field");
    };
    let path = opts.data.clone().unwrap_or_else(|| pipe.path.clone());
    let mut reader = Reader::open(&path)?;
    let layout = reader.layout()?.clone();
    let count = reader.len();
    let pipeline = Pipeline { shuffle: pipe.shuffle, batch: pipe.batch, take: pipe.take };
    let per_epoch = pipeline.batches_per_epoch(count) as u64;
    if per_epoch == 0 {
        return err("E9003", format!("{}: {count} samples, fewer than one batch of {}", path.display(), pipe.batch));
    }
    let epochs = opts.epochs.or(run.epochs).unwrap_or(1);
    let total = opts.steps.unwrap_or(epochs * per_epoch).max(1);
    // Dimensions, by name, for every function called.
    let names: BTreeMap<String, u64> = run.dims.iter().map(|(v, n)| (elab.dim_name(*v), *n)).collect();
    let dims_of = |core: &Program, f: FuncId| -> Result<Dims> {
        let mut d = Dims::new();
        for v in &core.func(f).dims {
            let name = elab.dim_name(*v);
            match names.get(&name) {
                Some(n) => {
                    d.insert(*v, *n);
                }
                None => return err("E7001", format!("dimension `{name}` of `{}` is not fixed by the training run (bind it in the data or the batch)", core.func(f).name)),
            }
        }
        Ok(d)
    };
    let (d_obj, d_init, d_opt) = (dims_of(core, objective)?, dims_of(core, init)?, dims_of(core, o_step)?);
    let core: &Program = core;
    let mut interp: Interp<Float> = Interp::new(core, elab, opts.bits, opts.seed);
    // Checkpoints.
    let ckpt_dir: Option<PathBuf> = if opts.checkpoints { opts.ckpt.clone().or_else(|| run.checkpoint.as_ref().map(|(_, dir)| run.dir.join(dir))) } else { None };
    let every = run.checkpoint.as_ref().map(|(k, _)| *k).filter(|k| *k > 0);
    if let Some(dir) = &ckpt_dir {
        std::fs::create_dir_all(dir).map_err(|e| TrainError { code: "E9003", message: format!("{}: {e}", dir.display()), span: None })?;
    }
    // θ₀, s₀.
    let key = Key::from_seed(opts.seed);
    let mut theta = interp.call(init, d_init, vec![Value::Key(key)])?;
    let theta0 = theta.clone();
    let mut batches: Vec<Vec<usize>> = Vec::new();
    let mut state = interp.call(o_init, d_opt.clone(), vec![theta.clone()])?;
    let one = Value::Real(Tensor { axes: vec![], data: vec![Float::with_val(opts.bits, 1)] });
    let mut logs = Vec::new();
    let mut checkpoints = Vec::new();
    let mut t = 0u64;
    'epochs: for epoch in 1..=epochs {
        for group in pipeline.batches(count, epoch) {
            t += 1;
            batches.push(group.clone());
            let samples: Vec<presk_data::Value> = group.iter().map(|i| reader.get(*i)).collect::<std::result::Result<_, _>>()?;
            let refs: Vec<&presk_data::Value> = samples.iter().collect();
            let batch = batch_value(&interp, &run.batch_ty, &layout, &refs, &d_obj)?;
            let loss = interp.call(objective, d_obj.clone(), vec![theta.clone(), batch.clone()])?;
            let cts = interp.call(vjp, d_obj.clone(), vec![theta.clone(), batch, one.clone()])?;
            let g = match cts {
                Value::Tuple(mut parts) if !parts.is_empty() => parts.swap_remove(0),
                other => return err("E7001", format!("vjp returned a {}", other.kind())),
            };
            let n2 = interp.prim_public(&PrimOp::SumSquares, std::slice::from_ref(&g))?;
            let lr = interp.call(o_lr, Dims::new(), vec![Value::Nat(t), Value::Nat(total)])?;
            let next = interp.call(o_step, d_opt.clone(), vec![theta, state, g, Value::Nat(t), Value::Nat(total)])?;
            match next {
                Value::Tuple(mut parts) if parts.len() == 2 => {
                    state = parts.pop().unwrap();
                    theta = parts.pop().unwrap();
                }
                other => return err("E7001", format!("optimizer step returned a {}", other.kind())),
            }
            let log = StepLog { step: t, epoch, loss: scalar_f64(&loss), lr: scalar_f64(&lr), grad_norm: scalar_f64(&n2).sqrt() };
            on_step(&log);
            logs.push(log);
            if let (Some(dir), Some(k)) = (&ckpt_dir, every)
                && t % k == 0
            {
                let p = dir.join(format!("step-{t}.psk.d"));
                save_params(&p, &model_name, &theta)?;
                checkpoints.push(p);
            }
            if t >= total {
                break 'epochs;
            }
        }
    }
    let final_path = match &ckpt_dir {
        Some(dir) => {
            let p = dir.join("final.psk.d");
            save_params(&p, &model_name, &theta)?;
            Some(p)
        }
        None => None,
    };
    Ok(TrainOutput { model: model_name, logs, theta0, batches, theta, final_path, checkpoints, total })
}

fn scalar_f64(v: &Value<Float>) -> f64 {
    match v {
        Value::Real(t) if t.rank() == 0 => t.data[0].to_f64(),
        _ => f64::NAN,
    }
}

/// `B` samples stacked under the `batch` axis of every leaf of the batch type.
fn batch_value(interp: &Interp<Float>, ty: &Ty, layout: &Layout, samples: &[&presk_data::Value], dims: &Dims) -> Result<Value<Float>> {
    let b = samples.len();
    match (ty, layout) {
        (Ty::Tuple(ts), Layout::Tuple(ls)) => {
            let mut out = Vec::new();
            for (k, (t, l)) in ts.iter().zip(ls).enumerate() {
                let subs: Vec<&presk_data::Value> = samples
                    .iter()
                    .map(|s| match s {
                        presk_data::Value::Tuple(vs) => Ok(&vs[k]),
                        _ => err("E9005", "sample is not a tuple"),
                    })
                    .collect::<Result<_>>()?;
                out.push(batch_value(interp, t, l, &subs, dims)?);
            }
            Ok(Value::Tuple(out))
        }
        (Ty::Real(shape), Layout::Leaf(Leaf::Real { .. })) => {
            let axes = interp.axes_of(shape, dims)?;
            let data = fill(&axes, b, |i| match samples[i] {
                presk_data::Value::Real(d) => Ok(d.iter().map(|x| Float::with_val(interp.prec, *x)).collect()),
                _ => err("E9005", "reals expected in the sample"),
            })?;
            Ok(Value::Real(Tensor { axes, data }))
        }
        (Ty::Idx { bound, shape }, Layout::Leaf(Leaf::Idx { .. })) => {
            let axes = interp.axes_of(shape, dims)?;
            let bound = interp.dim(bound, dims)?;
            let data = fill(&axes, b, |i| match samples[i] {
                presk_data::Value::Idx(d) => Ok(d.iter().map(|x| *x as i64).collect()),
                _ => err("E9005", "indices expected in the sample"),
            })?;
            Ok(Value::Idx { bound, t: Tensor { axes, data } })
        }
        _ => err("E9005", "the batch type and the file's layout disagree"),
    }
}

/// Lays `b` samples along the axis named `batch` (any position) of `axes`.
fn fill<T: Clone>(axes: &[Axis], b: usize, sample: impl Fn(usize) -> Result<Vec<T>>) -> Result<Vec<T>> {
    let p = axes.iter().position(|a| a.name.as_deref() == Some("batch")).ok_or_else(|| TrainError { code: "E7001", message: "batch type without a `batch` axis".into(), span: None })?;
    if axes[p].len != b {
        return err("E7001", format!("batch axis of length {}, {b} samples", axes[p].len));
    }
    let total: usize = axes.iter().map(|a| a.len).product();
    let inner: Vec<usize> = axes.iter().enumerate().filter(|(i, _)| *i != p).map(|(_, a)| a.len).collect();
    let n: usize = inner.iter().product();
    let mut out: Vec<Option<T>> = vec![None; total];
    for bi in 0..b {
        let s = sample(bi)?;
        if s.len() != n {
            return err("E9005", format!("sample of {} values, {n} expected", s.len()));
        }
        for (j, x) in s.into_iter().enumerate() {
            // multi-index of j over `inner`, then the flat index with `bi` at position p
            let mut rem = j;
            let mut idx: Vec<usize> = vec![0; inner.len()];
            for k in (0..inner.len()).rev() {
                idx[k] = rem % inner[k];
                rem /= inner[k];
            }
            let mut full = Vec::with_capacity(axes.len());
            let mut it = idx.into_iter();
            for (k, _) in axes.iter().enumerate() {
                full.push(if k == p { bi } else { it.next().unwrap() });
            }
            let mut flat = 0;
            for (k, a) in axes.iter().enumerate() {
                flat = flat * a.len + full[k];
            }
            out[flat] = Some(x);
        }
    }
    Ok(out.into_iter().map(|x| x.unwrap()).collect())
}

/// A parameter tree as a `.psk.d` sample: `f64` leaves, fields in order.
pub fn params_to_data(v: &Value<Float>) -> Result<(Layout, presk_data::Value)> {
    match v {
        Value::Record(fs) => {
            let mut ls = Vec::new();
            let mut vs = Vec::new();
            for (n, sub) in fs {
                let (l, d) = params_to_data(sub)?;
                ls.push((n.clone(), l));
                vs.push((n.clone(), d));
            }
            Ok((Layout::Record(ls), presk_data::Value::Record(vs)))
        }
        Value::Tuple(ts) => {
            let (ls, vs): (Vec<_>, Vec<_>) = ts.iter().map(params_to_data).collect::<Result<Vec<_>>>()?.into_iter().unzip();
            Ok((Layout::Tuple(ls), presk_data::Value::Tuple(vs)))
        }
        Value::Real(t) => Ok((Layout::Leaf(Leaf::Real { shape: t.axes.iter().map(|a| a.len).collect() }), presk_data::Value::Real(t.data.iter().map(|x| x.to_f64()).collect()))),
        other => err("E9005", format!("a parameter tree holds no {}", other.kind())),
    }
}

/// The parameter tree read from a `.psk.d` sample, with the axes of `template`.
pub fn params_from_data(template: &Value<Float>, d: &presk_data::Value, prec: u32) -> Result<Value<Float>> {
    match (template, d) {
        (Value::Record(fs), presk_data::Value::Record(ds)) if fs.len() == ds.len() => Ok(Value::Record(fs.iter().zip(ds).map(|((n, t), (_, x))| Ok((n.clone(), params_from_data(t, x, prec)?))).collect::<Result<_>>()?)),
        (Value::Tuple(ts), presk_data::Value::Tuple(ds)) if ts.len() == ds.len() => Ok(Value::Tuple(ts.iter().zip(ds).map(|(t, x)| params_from_data(t, x, prec)).collect::<Result<_>>()?)),
        (Value::Real(t), presk_data::Value::Real(xs)) if t.data.len() == xs.len() => Ok(Value::Real(Tensor { axes: t.axes.clone(), data: xs.iter().map(|x| Float::with_val(prec, *x)).collect() })),
        _ => err("E9005", "the checkpoint does not have the structure of the model's parameters"),
    }
}

/// Writes `θ` as a one-sample `.psk.d` of type `params <model>`.
pub fn save_params(path: &Path, model: &str, theta: &Value<Float>) -> Result<()> {
    let (layout, data) = params_to_data(theta)?;
    presk_data::write(path, &format!("params {model}"), &layout, std::slice::from_ref(&data))?;
    Ok(())
}

/// Reads a checkpoint written by [`save_params`], shaped like `template`.
pub fn load_params(path: &Path, template: &Value<Float>, prec: u32) -> Result<Value<Float>> {
    let (layout, _) = params_to_data(template)?;
    let mut r = Reader::open(path)?.with_layout(layout)?;
    if r.is_empty() {
        return err("E9003", format!("{}: no sample", path.display()));
    }
    let d = r.get(0)?;
    params_from_data(template, &d, prec)
}

/// One text line per step, for the terminal.
pub fn format_step(l: &StepLog, total: u64) -> String {
    format!("step {}/{total}  epoch {}  loss {:.6}  lr {:.3e}  |g| {:.4e}", l.step, l.epoch, l.loss, l.lr, l.grad_norm)
}
