//! `presk train`: the loop of a `training run` over a [`Backend`] (plan 0.2.0
//! step 5, rewritten over the runtime seam in 0.3.0 step 1).
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
//! through the backend and converts between `.psk.d` samples and runtime
//! values. With the interpreter behind the seam, exact values pass through
//! untouched and the loop reproduces the reference digit for digit.

use std::collections::BTreeMap;
use std::path::PathBuf;

use presk_core::{FuncId, Program};
use presk_data::{Layout, Leaf, Pipeline, Reader};
use presk_interp::eval::Dims;
use presk_interp::key::Key;
pub use presk_rt::{Backend, Buf, Realize, Tensor, Value, load_params, params_from_data, params_to_data, save_params};
use presk_rt::{Axis, Prepared, RtError};
use presk_syntax::Span;
use presk_types::check::TypedProgram;
use presk_types::check::training::TrainingRun;
use presk_types::elab::ElabProgram;
use presk_types::ty::{ModRef, Ty};

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
    /// The machine type of a native backend (ignored by the interpreter).
    pub realize: Realize,
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
    pub theta0: Value,
    /// The sample indices of every batch, in step order (what the K-step
    /// conformance test replays).
    pub batches: Vec<Vec<usize>>,
    pub theta: Value,
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

impl From<RtError> for TrainError {
    fn from(e: RtError) -> Self {
        TrainError { code: e.code, message: e.message, span: e.span }
    }
}

impl From<presk_data::DataError> for TrainError {
    fn from(e: presk_data::DataError) -> Self {
        TrainError { code: e.code, message: e.message, span: None }
    }
}

/// The Core functions a training run needs, resolved from the typed program.
/// `prepare_training` generates the objective's `vjp` into `core`; a backend
/// is then built over the final program.
#[derive(Clone, Debug)]
pub struct TrainingFns {
    pub model: String,
    /// The objective takes a per-step key between `θ` and the batch.
    pub key: bool,
    pub objective: FuncId,
    pub vjp: FuncId,
    pub init: FuncId,
    pub opt_init: FuncId,
    pub opt_step: FuncId,
    pub opt_lr: FuncId,
}

pub fn prepare_training(core: &mut Program, elab: &ElabProgram, typed: &TypedProgram) -> Result<TrainingFns> {
    let Some(run) = typed.training.first() else {
        return err("E7001", "no `training run` in this file");
    };
    let ModRef::Inst { def, .. } = &run.model else {
        return err("E7001", "the model of a training run must be a plain module");
    };
    let model = elab.module(*def).name.clone();
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
    let init = core.find(&format!("{model}.init")).ok_or_else(|| TrainError { code: "E7001", message: "no initializer".into(), span: None })?;
    let opt = &run.optimizer;
    let (Some(opt_init), Some(opt_step), Some(opt_lr)) = (core.find(&format!("{opt}.init")), core.find(&format!("{opt}.step")), core.find(&format!("{opt}.lr"))) else {
        return err("E7001", format!("optimizer `{opt}` has no lowered functions (only `adamw` is implemented)"));
    };
    Ok(TrainingFns {
        key: matches!(core.func(objective).params.get(1).map(|p| &p.ty), Some(Ty::Key)), model, objective, vjp, init, opt_init, opt_step, opt_lr })
}

/// Runs the (first) `training run` of the program over `backend`, built
/// over the program returned by [`prepare_training`].
pub fn train(backend: &mut dyn Backend, core: &Program, elab: &ElabProgram, typed: &TypedProgram, fns: &TrainingFns, opts: &TrainOptions, mut on_step: impl FnMut(&StepLog)) -> Result<TrainOutput> {
    let run: &TrainingRun = typed.training.first().ok_or_else(|| TrainError { code: "E7001", message: "no `training run` in this file".into(), span: None })?;
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
    let dims_of = |f: FuncId| -> Result<Dims> {
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
    let d_obj = dims_of(fns.objective)?;
    let objective: Prepared = backend.prepare(fns.objective, &d_obj, opts.realize)?;
    let vjp = backend.prepare(fns.vjp, &d_obj, opts.realize)?;
    let init = backend.prepare(fns.init, &dims_of(fns.init)?, opts.realize)?;
    let d_opt = dims_of(fns.opt_step)?;
    let opt_init = backend.prepare(fns.opt_init, &d_opt, opts.realize)?;
    let opt_step = backend.prepare(fns.opt_step, &d_opt, opts.realize)?;
    let opt_lr = backend.prepare(fns.opt_lr, &Dims::new(), opts.realize)?;
    // Checkpoints.
    let ckpt_dir: Option<PathBuf> = if opts.checkpoints { opts.ckpt.clone().or_else(|| run.checkpoint.as_ref().map(|(_, dir)| run.dir.join(dir))) } else { None };
    let every = run.checkpoint.as_ref().map(|(k, _)| *k).filter(|k| *k > 0);
    if let Some(dir) = &ckpt_dir {
        std::fs::create_dir_all(dir).map_err(|e| TrainError { code: "E9003", message: format!("{}: {e}", dir.display()), span: None })?;
    }
    // θ₀, s₀.
    let run_key = Key::from_seed(opts.seed);
    // A training run trains: `dropout` drops here and nowhere else.
    backend.set_training(true);
    let mut theta = backend.call(&init, vec![Value::Key(run_key.clone())])?;
    let theta0 = theta.clone();
    let mut batches: Vec<Vec<usize>> = Vec::new();
    let mut state = backend.call(&opt_init, vec![theta.clone()])?;
    let one = Value::Real(Tensor::scalar_f64(1.0));
    let mut logs = Vec::new();
    let mut checkpoints = Vec::new();
    let mut t = 0u64;
    'epochs: for epoch in 1..=epochs {
        for group in pipeline.batches(count, epoch) {
            t += 1;
            batches.push(group.clone());
            let samples: Vec<presk_data::Value> = group.iter().map(|i| reader.get(*i)).collect::<std::result::Result<_, _>>()?;
            let refs: Vec<&presk_data::Value> = samples.iter().collect();
            let batch = batch_value(&run.batch_ty, &layout, &refs, &d_obj)?;
            // A fresh key per step, derived from the run's seed: `dropout`
            // draws its mask from it, and the backward pass, which recomputes
            // the forward, recomputes the same mask from the same key.
            let mut obj_args = vec![theta.clone()];
            let mut vjp_args = vec![theta.clone()];
            if fns.key {
                let k = Value::Key(run_key.child(t));
                obj_args.push(k.clone());
                vjp_args.push(k);
            }
            obj_args.push(batch.clone());
            vjp_args.push(batch);
            vjp_args.push(one.clone());
            let loss = backend.call(&objective, obj_args)?;
            let cts = backend.call(&vjp, vjp_args)?;
            let g = match cts {
                Value::Tuple(mut parts) if !parts.is_empty() => parts.swap_remove(0),
                other => return err("E7001", format!("vjp returned a {}", other.kind())),
            };
            let grad_norm = g.sum_squares_f64().sqrt();
            let lr = backend.call(&opt_lr, vec![Value::Nat(t), Value::Nat(total)])?;
            let next = backend.call(&opt_step, vec![theta, state, g, Value::Nat(t), Value::Nat(total)])?;
            match next {
                Value::Tuple(mut parts) if parts.len() == 2 => {
                    state = parts.pop().unwrap();
                    theta = parts.pop().unwrap();
                }
                other => return err("E7001", format!("optimizer step returned a {}", other.kind())),
            }
            let log = StepLog { step: t, epoch, loss: loss.scalar_f64(), lr: lr.scalar_f64(), grad_norm };
            on_step(&log);
            logs.push(log);
            if let (Some(dir), Some(k)) = (&ckpt_dir, every)
                && t % k == 0
            {
                let p = dir.join(format!("step-{t}.psk.d"));
                save_params(&p, &fns.model, &theta)?;
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
            save_params(&p, &fns.model, &theta)?;
            Some(p)
        }
        None => None,
    };
    Ok(TrainOutput { model: fns.model.clone(), logs, theta0, batches, theta, final_path, checkpoints, total })
}

/// `presk run --data FILE.psk.d`: evaluate an entry with its **last** parameter
/// taken from a data file instead of synthesized, the others synthesized as
/// usual. The `batch` axis of that parameter is bound to the number of samples
/// read, so an objective can be evaluated on its own dataset and a plain
/// function on a corpus.
pub fn run_on_data(backend: &mut dyn Backend, core: &Program, elab: &ElabProgram, opts: &presk_interp::RunOptions, realize: Realize, path: &std::path::Path, samples: Option<usize>) -> Result<presk_interp::RunOutput> {
    let to_train = |e: presk_interp::RunError| TrainError { code: "E7001", message: e.message, span: e.span };
    let (entry_fn, _, _) = presk_interp::run::resolve_entry(core, elab, &opts.entry).map_err(to_train)?;
    let Some(last) = core.func(entry_fn).params.last() else {
        return err("E7001", format!("`{}` takes no argument to fill from a data file", opts.entry));
    };
    let last_ty = last.ty.clone();
    let mut reader = Reader::open(path)?;
    let count = samples.unwrap_or(usize::MAX).min(reader.len());
    if count == 0 {
        return err("E9003", format!("{}: no samples to read", path.display()));
    }
    let layout = reader.layout()?.clone();
    // The batch dimension of that parameter is the number of samples taken.
    let mut dims = opts.dims.clone();
    if let Some(name) = batch_dim(&last_ty, elab)
        && !dims.iter().any(|(n, _)| *n == name)
    {
        dims.push((name, count as u64));
    }
    let opts = presk_interp::RunOptions { entry: opts.entry.clone(), dims, seed: opts.seed, bits: opts.bits, digits: opts.digits };
    let (func, fdims, inputs) = presk_interp::run::prepare_run(core, elab, &opts).map_err(to_train)?;
    let values: Vec<presk_data::Value> = (0..count).map(|i| reader.get(i)).collect::<std::result::Result<_, _>>()?;
    let refs: Vec<&presk_data::Value> = values.iter().collect();
    let batch = batch_value(&last_ty, &layout, &refs, &fdims)?;
    let mut args: Vec<Value> = inputs.iter().map(|(_, v)| Value::from_interp(v)).collect::<std::result::Result<_, _>>()?;
    *args.last_mut().unwrap() = batch;
    let prepared = backend.prepare(func, &fdims, realize)?;
    let result = backend.call(&prepared, args)?;
    let dims_named: Vec<(String, u64)> = fdims.iter().map(|(v, n)| (elab.dim_name(*v), *n)).collect();
    Ok(presk_interp::RunOutput { entry: opts.entry.clone(), func, dims: dims_named, inputs, result: result.to_interp(opts.bits) })
}

/// The name of the dimension that lengths the `batch` axis of `ty`, when it is
/// a single dimension variable.
fn batch_dim(ty: &Ty, elab: &ElabProgram) -> Option<String> {
    match ty {
        Ty::Real(s) | Ty::Idx { shape: s, .. } => s.axes.iter().find(|a| a.name.as_deref() == Some("batch")).and_then(|a| {
            let vs: Vec<_> = a.len.vars().collect();
            if vs.len() == 1 && a.len == presk_types::dim::DimExpr::var(vs[0]) { Some(elab.dim_name(vs[0])) } else { None }
        }),
        Ty::Tuple(ts) => ts.iter().find_map(|t| batch_dim(t, elab)),
        _ => None,
    }
}

/// `B` samples stacked under the `batch` axis of every leaf of the batch type,
/// as `f64` buffers (the data file's precision).
fn batch_value(ty: &Ty, layout: &Layout, samples: &[&presk_data::Value], dims: &Dims) -> Result<Value> {
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
                out.push(batch_value(t, l, &subs, dims)?);
            }
            Ok(Value::Tuple(out))
        }
        (Ty::Real(shape), Layout::Leaf(Leaf::Real { .. })) => {
            let axes = presk_rt::axes_of(shape, dims)?;
            let data = fill(&axes, b, |i| match samples[i] {
                presk_data::Value::Real(d) => Ok(d.clone()),
                _ => err("E9005", "reals expected in the sample"),
            })?;
            Ok(Value::Real(Tensor { axes, data: Buf::F64(data) }))
        }
        (Ty::Idx { bound, shape }, Layout::Leaf(Leaf::Idx { .. })) => {
            let axes = presk_rt::axes_of(shape, dims)?;
            let map: BTreeMap<_, _> = dims.iter().map(|(v, n)| (*v, presk_types::dim::rat(*n as i64))).collect();
            let bound = bound.eval(&map).and_then(|r| r.to_integer().to_string().parse::<u64>().ok()).ok_or_else(|| TrainError { code: "E7001", message: "unbound index bound".into(), span: None })?;
            let data = fill(&axes, b, |i| match samples[i] {
                presk_data::Value::Idx(d) => Ok(d.iter().map(|x| *x as i64).collect()),
                _ => err("E9005", "indices expected in the sample"),
            })?;
            Ok(Value::Idx { bound, t: Tensor { axes, data: Buf::I64(data) } })
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
            let mut rem = j;
            let mut idx: Vec<usize> = vec![0; inner.len()];
            for k in (0..inner.len()).rev() {
                idx[k] = rem % inner[k];
                rem /= inner[k];
            }
            let mut full = Vec::with_capacity(axes.len());
            let mut it = idx.into_iter();
            for k in 0..axes.len() {
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

/// One text line per step, for the terminal.
pub fn format_step(l: &StepLog, total: u64) -> String {
    format!("step {}/{total}  epoch {}  loss {:.6}  lr {:.3e}  |g| {:.4e}", l.step, l.epoch, l.loss, l.lr, l.grad_norm)
}
