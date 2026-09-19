//! The `training run` block typed against its objective (plan 0.2.0, step 2).
//!
//! The objective's second parameter is the *batch* type; the *sample* type
//! is the batch type with the `batch` axis removed from every tensor leaf.
//! `batch(B = n)` binds the length of that axis; the data file, when it is
//! readable at check time, must hold samples of the sample type, and its
//! literal dimensions bind the objective's symbolic ones (`T = 8`). Error
//! code `E2010`; a missing data file is the warning `W0002`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use presk_syntax::ast::{Arg, BinOp, ConfigItem, Expr, Lit};
use presk_syntax::{Diagnostic, Span};

use super::Checker;
use crate::dim::{DimExpr, DimVar};
use crate::elab::{ElabProgram, FnId};
use crate::ty::{ModRef, Shape, Ty};

/// `stream "path" |> shuffle(seed = s) |> batch(B = n) |> take(n)`, resolved.
#[derive(Clone, Debug)]
pub struct DataPipeline {
    /// As written in the source.
    pub raw_path: String,
    /// Relative to the `.psk` file that declares the run.
    pub path: PathBuf,
    pub shuffle: Option<u64>,
    pub batch: usize,
    pub take: Option<usize>,
    /// The sample type read from the file's header, when the file was readable.
    pub file_ty: Option<String>,
}

/// One `training run`, resolved and typed.
#[derive(Clone, Debug)]
pub struct TrainingRun {
    pub span: Span,
    /// The directory of the `.psk` file declaring the run (paths of the
    /// block, `data` and `checkpoint … to`, are relative to it).
    pub dir: PathBuf,
    pub model: ModRef,
    pub objective: FnId,
    pub optimizer: String,
    pub data: Option<DataPipeline>,
    pub epochs: Option<u64>,
    /// `checkpoint every N steps to "dir"`.
    pub checkpoint: Option<(u64, String)>,
    /// The objective's batch parameter type.
    pub batch_ty: Ty,
    /// The batch type without its `batch` axis.
    pub sample_ty: Ty,
    /// Dimensions of the objective bound by the pipeline (`B`) and by the
    /// data file's type (`T`).
    pub dims: BTreeMap<DimVar, u64>,
}

impl Checker<'_> {
    /// Builds the record of a `training run` whose fields passed the schema;
    /// `model` is its resolved `model` field.
    pub(crate) fn training_run(&mut self, items: &[ConfigItem], model: Option<ModRef>, span: Span) -> Option<TrainingRun> {
        let mut objective: Option<(FnId, Span)> = None;
        let mut optimizer: Option<String> = None;
        let mut data: Option<(&Expr, Span)> = None;
        let mut epochs: Option<u64> = None;
        let mut checkpoint: Option<(u64, String)> = None;
        for item in items {
            match item {
                ConfigItem::Field { name, value, span, .. } => match (name.name.as_str(), value) {
                    ("objective", Expr::Var { path }) => {
                        let Some(&fid) = self.prog.units[self.unit].fns.get(&path.segs[0].name) else {
                            self.error(Diagnostic::error("E2010", format!("`objective`: `{}` is not a function of this file", path.dotted())).label(path.span, "expected a top-level `fn`"));
                            continue;
                        };
                        objective = Some((fid, path.span));
                    }
                    ("optimizer", Expr::Var { path }) => {
                        if !self.prog.units[self.unit].optimizers.contains_key(&path.segs[0].name) {
                            self.error(Diagnostic::error("E2010", format!("`optimizer`: no `optimizer {}` block in this file", path.dotted())).label(path.span, "expected the name of an optimizer block"));
                        } else if path.segs[0].name != "adamw" {
                            // The block's name is the algorithm (question 30).
                            self.error(Diagnostic::error("E2010", format!("`optimizer {}`: only `adamw` is implemented in 0.2.0", path.segs[0].name)).label(path.span, "unknown optimizer algorithm").help("name the block `optimizer adamw { … }`; the name selects the update rule"));
                        } else {
                            optimizer = Some(path.segs[0].name.clone());
                        }
                    }
                    ("data", e) => data = Some((e, *span)),
                    ("epochs", Expr::Lit { lit: Lit::Nat(n), .. }) => epochs = n.parse().ok(),
                    _ => {}
                },
                ConfigItem::Checkpoint { every, to, .. } => {
                    if let Expr::Lit { lit: Lit::Nat(n), .. } = every {
                        checkpoint = n.parse().ok().map(|n| (n, to.clone()));
                    }
                }
            }
        }
        let (model, (objective, obj_span)) = (model?, objective?);
        let f = self.prog.function(objective);
        // First parameter: the model's parameter tree.
        match f.params.first().and_then(|p| p.ty.clone()) {
            Some(Ty::Params(m)) => {
                let (m, model) = (m.clone(), model.clone());
                if !self.trial(|c| c.unify_modref(&m, &model, obj_span)) {
                    let n = self.prog.names();
                    self.error(Diagnostic::error("E2010", format!("`objective` `{}` takes `params {}` but the model is `{}`", f.name, m.display(&n), model.display(&n))).label(obj_span, "objective and model disagree"));
                    return None;
                }
            }
            _ => {
                self.error(Diagnostic::error("E2010", format!("`objective` `{}` must take the parameter tree first: `θ : params M`", f.name)).label(obj_span, "first parameter is not `params M`"));
                return None;
            }
        }
        // Second parameter: the batch.
        let Some(batch_ty) = f.params.get(1).and_then(|p| p.ty.clone()) else {
            self.error(Diagnostic::error("E2010", format!("`objective` `{}` must take the batch second: `batch : (idx[V][batch: B, seq: T], …)`", f.name)).label(obj_span, "no batch parameter"));
            return None;
        };
        if f.params.len() > 2 && f.params[2..].iter().any(|p| p.default.is_none()) {
            self.error(Diagnostic::error("E2010", format!("`objective` `{}` has parameters after the batch without defaults; the loop passes only `θ` and the batch", f.name)).label(obj_span, "too many parameters"));
            return None;
        }
        let mut dims = BTreeMap::new();
        let sample_ty = match strip_batch(&batch_ty) {
            Ok(t) => t,
            Err(msg) => {
                let n = self.prog.names();
                self.error(Diagnostic::error("E2010", format!("`objective` `{}`: {msg}", f.name)).label(obj_span, format!("batch type `{}`", batch_ty.display(&n))).help("every tensor of the batch needs an axis named `batch`, e.g. `idx[V][batch: B, seq: T]`"));
                return None;
            }
        };
        let batch_lens: Vec<DimExpr> = batch_axis_lens(&batch_ty);
        let unit_dir = self.prog.units[self.unit].path.parent().map(Path::to_path_buf).unwrap_or_default();
        let mut run = TrainingRun { span, dir: unit_dir, model, objective, optimizer: optimizer.unwrap_or_default(), data: None, epochs, checkpoint, batch_ty, sample_ty, dims: BTreeMap::new() };
        let Some((data_expr, data_span)) = data else { return Some(run) };
        let Some(mut pipeline) = self.resolve_pipeline(data_expr) else { return Some(run) };
        // `batch(B = n)` binds the batch axis of every leaf.
        for len in &batch_lens {
            if !bind(&mut dims, len, pipeline.batch as u64) {
                let n = self.prog.names();
                self.error(Diagnostic::error("E2010", format!("`batch(B = {})` disagrees with the batch axis `{}` of the objective", pipeline.batch, len.display(&*n.dim))).label(data_span, "batch size fixed here"));
            }
        }
        // The data file, when readable: its sample type against the objective's.
        pipeline.path = run.dir.join(&pipeline.raw_path);
        if pipeline.path.exists() {
            match presk_data::Reader::open(&pipeline.path) {
                Err(e) => {
                    self.error(Diagnostic::error("E2010", format!("`data`: {}", e.message)).label(data_span, "this file"));
                }
                Ok(reader) => {
                    let file_ty = reader.header().sample_ty.clone();
                    match reader.layout() {
                        Err(e) => self.error(Diagnostic::error("E2010", format!("`data`: {}", e.message)).label(data_span, "this file")),
                        Ok(layout) => {
                            if let Err(msg) = match_layout(&run.sample_ty, layout, &mut dims, self.prog) {
                                let n = self.prog.names();
                                self.error(
                                    Diagnostic::error("E2010", format!("`data`: the file holds samples of type `{file_ty}` but `{}` expects batches of `{}`", self.prog.function(run.objective).name, run.batch_ty.display(&n)))
                                        .label(data_span, format!("samples must have type `{}`", run.sample_ty.display(&n)))
                                        .note(msg),
                                );
                            }
                        }
                    }
                    pipeline.file_ty = Some(file_ty);
                }
            }
        } else {
            self.error(Diagnostic::warning("W0002", format!("data file `{}` not found; its sample type is checked by `presk train`", pipeline.path.display())).label(data_span, "relative to this file"));
        }
        run.data = Some(pipeline);
        run.dims = dims;
        Some(run)
    }

    /// The pipeline's fields (the schema has already validated the forms).
    fn resolve_pipeline(&mut self, e: &Expr) -> Option<DataPipeline> {
        let mut p = DataPipeline { raw_path: String::new(), path: PathBuf::new(), shuffle: None, batch: 0, take: None, file_ty: None };
        let mut cur = e;
        let mut stages: Vec<&Expr> = Vec::new();
        while let Expr::Binary { op: BinOp::Pipe, lhs, rhs, .. } = cur {
            stages.push(rhs);
            cur = lhs;
        }
        if let Expr::Call { args, .. } = cur
            && let Some(Arg::Pos(Expr::Lit { lit: Lit::Str(s), .. })) = args.first()
        {
            p.raw_path = s.clone();
        }
        for st in stages.iter().rev() {
            let Expr::Call { callee, args, .. } = st else { continue };
            let Expr::Var { path } = &**callee else { continue };
            let nat = |key: &str| -> Option<u64> {
                args.iter().find_map(|a| match a {
                    Arg::Named { name, value: Expr::Lit { lit: Lit::Nat(n), .. } } if name.name == key => n.parse().ok(),
                    _ => None,
                })
            };
            match path.segs[0].name.as_str() {
                "shuffle" => p.shuffle = nat("seed"),
                "batch" => p.batch = nat("B").or_else(|| nat("size")).unwrap_or(0) as usize,
                "take" => p.take = nat("n").map(|n| n as usize),
                _ => {}
            }
        }
        if p.batch == 0 {
            self.error(Diagnostic::error("E2010", "`data`: the pipeline needs `batch(B = n)` with `n ≥ 1`").label(e.span(), "no batch size"));
            return None;
        }
        Some(p)
    }
}

/// The batch type without its `batch` axis; an error message when a leaf has
/// none or is open.
fn strip_batch(t: &Ty) -> Result<Ty, String> {
    match t {
        Ty::Real(s) => Ok(Ty::Real(strip_shape(s)?)),
        Ty::Idx { bound, shape } => Ok(Ty::Idx { bound: bound.clone(), shape: strip_shape(shape)? }),
        Ty::Tuple(ts) => Ok(Ty::Tuple(ts.iter().map(strip_batch).collect::<Result<_, _>>()?)),
        _ => Err("the batch must be a tensor or a tuple of tensors".into()),
    }
}

fn strip_shape(s: &Shape) -> Result<Shape, String> {
    if s.prefix.is_some() {
        return Err("a batch type cannot have an open prefix `...`".into());
    }
    let Some(i) = s.axes.iter().position(|a| a.name.as_deref() == Some("batch")) else {
        return Err("a tensor of the batch has no axis named `batch`".into());
    };
    let mut axes = s.axes.clone();
    axes.remove(i);
    Ok(Shape { prefix: None, axes })
}

/// The lengths of the `batch` axes of every leaf.
fn batch_axis_lens(t: &Ty) -> Vec<DimExpr> {
    match t {
        Ty::Real(s) | Ty::Idx { shape: s, .. } => s.axes.iter().filter(|a| a.name.as_deref() == Some("batch")).map(|a| a.len.clone()).collect(),
        Ty::Tuple(ts) => ts.iter().flat_map(batch_axis_lens).collect(),
        _ => vec![],
    }
}

/// Binds a dimension expression to a value: a literal must equal it, a
/// single variable is bound (consistently); anything else is unsupported.
fn bind(dims: &mut BTreeMap<DimVar, u64>, e: &DimExpr, n: u64) -> bool {
    if let Some(k) = e.as_nat() {
        return k == n;
    }
    let vars: Vec<DimVar> = e.vars().collect();
    if vars.len() == 1 && *e == DimExpr::var(vars[0]) {
        match dims.get(&vars[0]) {
            Some(k) => *k == n,
            None => {
                dims.insert(vars[0], n);
                true
            }
        }
    } else {
        false
    }
}

/// The file's leaf layout against the sample type, binding symbolic dimensions.
fn match_layout(t: &Ty, l: &presk_data::Layout, dims: &mut BTreeMap<DimVar, u64>, prog: &ElabProgram) -> Result<(), String> {
    use presk_data::{Layout, Leaf};
    let names = prog.names();
    match (t, l) {
        (Ty::Tuple(ts), Layout::Tuple(ls)) => {
            if ts.len() != ls.len() {
                return Err(format!("a tuple of {} in the program, of {} in the file", ts.len(), ls.len()));
            }
            ts.iter().zip(ls).try_for_each(|(t, l)| match_layout(t, l, dims, prog))
        }
        (Ty::Real(s), Layout::Leaf(Leaf::Real { shape })) => match_shape(s, shape, dims, &names),
        (Ty::Idx { bound, shape: s }, Layout::Leaf(Leaf::Idx { bound: b, shape })) => {
            // Samples of a smaller vocabulary are valid samples of a larger one:
            // a literal program bound must be at least the file's; a symbolic
            // bound is bound to the file's.
            let ok = match bound.as_nat() {
                Some(k) => *b <= k,
                None => bind(dims, bound, *b),
            };
            if !ok {
                return Err(format!("index bound `{}` in the program, {b} in the file (the file's values may exceed the program's bound)", bound.display(&*names.dim)));
            }
            match_shape(s, shape, dims, &names)
        }
        (Ty::Real(_), Layout::Leaf(Leaf::Idx { .. })) => Err("reals expected, the file holds indices".into()),
        (Ty::Idx { .. }, Layout::Leaf(Leaf::Real { .. })) => Err("indices expected, the file holds reals".into()),
        (Ty::Tuple(_), _) => Err("a tuple expected, the file holds one tensor".into()),
        (_, Layout::Tuple(_)) => Err("one tensor expected, the file holds tuples".into()),
        _ => Err("the file's samples are not tensors".into()),
    }
}

fn match_shape(s: &Shape, shape: &[usize], dims: &mut BTreeMap<DimVar, u64>, names: &crate::ty::Names) -> Result<(), String> {
    if s.axes.len() != shape.len() {
        return Err(format!("rank {} in the program, {} in the file", s.axes.len(), shape.len()));
    }
    for (a, &k) in s.axes.iter().zip(shape) {
        if !bind(dims, &a.len, k as u64) {
            let what = a.name.as_deref().map(|n| format!("axis `{n}`")).unwrap_or_else(|| "an axis".into());
            return Err(format!("{what} has length `{}` in the program, {k} in the file{}", a.len.display(&*names.dim), bound_note(dims, &a.len, names)));
        }
    }
    Ok(())
}

fn bound_note(dims: &BTreeMap<DimVar, u64>, e: &DimExpr, names: &crate::ty::Names) -> String {
    let bound: Vec<String> = e.vars().filter_map(|v| dims.get(&v).map(|k| format!("{} = {k}", (names.dim)(v)))).collect();
    if bound.is_empty() { String::new() } else { format!(" (with {})", bound.join(", ")) }
}

/// `presk check --summary`: the training runs of the main file.
pub fn summary(prog: &ElabProgram, runs: &[TrainingRun]) -> String {
    let n = prog.names();
    let mut s = String::new();
    for r in runs {
        s.push_str(&format!("training run: model {}, objective {}, optimizer {}\n", r.model.display(&n), prog.function(r.objective).name, r.optimizer));
        s.push_str(&format!("    batch   : {}\n    sample  : {}\n", r.batch_ty.display(&n), r.sample_ty.display(&n)));
        match &r.data {
            Some(d) => {
                let mut ops = vec![format!("batch {}", d.batch)];
                if let Some(seed) = d.shuffle {
                    ops.insert(0, format!("shuffle {seed}"));
                }
                if let Some(t) = d.take {
                    ops.push(format!("take {t}"));
                }
                s.push_str(&format!("    data    : {} ({})", d.raw_path, ops.join(", ")));
                match &d.file_ty {
                    Some(t) => s.push_str(&format!(" — file holds {t}\n")),
                    None => s.push_str(" — file not found\n"),
                }
            }
            None => s.push_str("    data    : none\n"),
        }
        if let Some(e) = r.epochs {
            s.push_str(&format!("    epochs  : {e}\n"));
        }
        if let Some((k, dir)) = &r.checkpoint {
            s.push_str(&format!("    ckpt    : every {k} steps to {dir}\n"));
        }
        if !r.dims.is_empty() {
            s.push_str(&format!("    dims    : {}\n", r.dims.iter().map(|(v, k)| format!("{} = {k}", prog.dim_name(*v))).collect::<Vec<_>>().join(", ")));
        }
    }
    s
}
