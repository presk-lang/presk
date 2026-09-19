//! Configuration blocks (`optimizer`, `training run`, `inference serve`)
//! checked against a schema. The values are matched syntactically: their
//! semantics (schedules, parallelism, precision budgets) is not used in
//! 0.1.0. Error code `E2010`.

use presk_syntax::ast::{Arg, BinOp, ConfigItem, Expr, Lit, Path};
use presk_syntax::Diagnostic;

use super::Checker;
use crate::ty::ModRef;

impl Checker<'_> {
    pub(crate) fn check_configs(&mut self, unit: usize) {
        self.unit = unit;
        let u = &self.prog.units[unit];
        let optimizers: Vec<(String, Vec<ConfigItem>)> = u.optimizers.iter().map(|(n, i)| (n.clone(), i.clone())).collect();
        let training = u.training.clone();
        let inference = u.inference.clone();
        for (name, items) in &optimizers {
            self.block("optimizer", &format!("optimizer {name}"), items, &OPTIMIZER, &[]);
        }
        for (items, span) in &training {
            let model = self.block("training run", "training run", items, &TRAINING, &["model", "objective", "optimizer"]);
            if let Some(run) = self.training_run(items, model, *span)
                && unit == self.prog.main
            {
                self.typed.training.push(run);
            }
        }
        for (items, _) in &inference {
            self.block("inference serve", "inference serve", items, &INFERENCE, &["model"]);
        }
    }

    /// Check every item against the schema; returns the `model` module if any.
    fn block(&mut self, kind: &str, what: &str, items: &[ConfigItem], schema: &[(&str, Form)], required: &[&str]) -> Option<ModRef> {
        let mut seen: Vec<String> = Vec::new();
        let mut model = None;
        for item in items {
            match item {
                ConfigItem::Checkpoint { every, to: _, span } => {
                    if kind != "training run" {
                        self.error(Diagnostic::error("E2010", format!("`checkpoint` is not allowed in `{what}`")).label(*span, "only `training run` has checkpoints"));
                    } else if !matches!(every, Expr::Lit { lit: Lit::Nat(_), .. }) {
                        self.error(Diagnostic::error("E2010", "`checkpoint every N steps`: N must be a natural literal").label(every.span(), "not a natural"));
                    }
                }
                ConfigItem::Field { name, ty, value, span } => {
                    if seen.contains(&name.name) {
                        self.error(Diagnostic::error("E2010", format!("field `{}` given twice", name.name)).label(*span, "duplicate field"));
                        continue;
                    }
                    seen.push(name.name.clone());
                    let Some((_, form)) = schema.iter().find(|(n, _)| *n == name.name) else {
                        let known: Vec<String> = schema.iter().map(|(n, _)| format!("`{n}`")).collect();
                        self.error(Diagnostic::error("E2010", format!("unknown field `{}` in `{what}`", name.name)).label(name.span, "not a field of this block").help(format!("fields: {}", known.join(", "))));
                        continue;
                    };
                    if let Some(t) = ty {
                        // The declared type must agree with the schema's scalar form.
                        let ok = matches!((form, t), (Form::F64, presk_syntax::ast::Ty::Named { path, .. }) if path.is_single("f64"))
                            || matches!((form, t), (Form::Nat, presk_syntax::ast::Ty::Named { path, .. }) if path.is_single("nat"))
                            || matches!((form, t), (Form::F64Pair, presk_syntax::ast::Ty::Tuple { elems, .. }) if elems.len() == 2);
                        if !ok {
                            self.error(Diagnostic::error("E2010", format!("field `{}` has type {}", name.name, form.describe())).label(t.span(), "declared type does not match"));
                        }
                    }
                    if let Some(m) = self.form(form, value, &name.name) {
                        if name.name == "model" {
                            model = Some(m);
                        }
                    }
                }
            }
        }
        for r in required {
            if !seen.iter().any(|s| s == r) {
                let span = items.first().map(|i| match i {
                    ConfigItem::Field { span, .. } | ConfigItem::Checkpoint { span, .. } => *span,
                });
                let mut d = Diagnostic::error("E2010", format!("`{what}` needs a `{r}` field"));
                if let Some(s) = span {
                    d = d.label(s, format!("`{r}` is missing from this block"));
                }
                self.error(d);
            }
        }
        model
    }

    /// Match a value against a form; returns the module for `Form::Module`.
    fn form(&mut self, form: &Form, value: &Expr, field: &str) -> Option<ModRef> {
        let ok = match form {
            Form::F64 => matches!(value, Expr::Lit { lit: Lit::Float(_), .. } | Expr::Lit { lit: Lit::Nat(_), .. }),
            Form::Nat => matches!(value, Expr::Lit { lit: Lit::Nat(_), .. }),
            Form::Str => matches!(value, Expr::Lit { lit: Lit::Str(_), .. }),
            Form::F64Pair => matches!(value, Expr::Tuple { elems, .. } if elems.len() == 2 && elems.iter().all(|e| matches!(e, Expr::Lit { lit: Lit::Float(_) | Lit::Nat(_), .. }))),
            Form::Name => matches!(value, Expr::Var { path } if path.segs.len() == 1),
            Form::Module => {
                return match value {
                    Expr::Var { path } => match self.module_path(path) {
                        Some(m) => Some(m),
                        None => {
                            self.error(Diagnostic::error("E2010", format!("`{field}`: `{}` is not a module", path.dotted())).label(path.span, "expected a module without dimension arguments"));
                            None
                        }
                    },
                    _ => {
                        self.error(Diagnostic::error("E2010", format!("`{field}` expects a module name")).label(value.span(), "not a module"));
                        None
                    }
                };
            }
            Form::Schedule => self.call_form(value, &[("cosine", &[("warmup", Form::Nat)]), ("constant", &[]), ("linear", &[("warmup", Form::Nat)])]),
            Form::Data => self.pipeline(value),
            Form::Parallel => self.chain(value, BinOp::Tensor, &[("data_parallel", &[("replicas", Form::AutoOrNat)]), ("tensor_parallel", &[("axis", Form::Axis)]), ("pipeline", &[("stages", Form::Nat)])]),
            Form::Precision => self.chain(value, BinOp::Alt, &[("budget", &[("rel_error", Form::F64), ("abs_error", Form::F64)]), ("quantize", &[("_", Form::Name), ("calibration", Form::Str)])]),
            Form::Target => self.chain(value, BinOp::Alt, &[("gpu", &[("_", Form::Name)]), ("cpu", &[("_", Form::Name)])]),
            Form::Batching => self.call_form(value, &[("dynamic", &[("max_batch", Form::Nat), ("timeout", Form::Duration)]), ("static", &[("batch", Form::Nat)])]),
            Form::Cache => self.call_form(value, &[("kv_cache", &[("axis", Form::Axis)])]),
            Form::AutoOrNat => matches!(value, Expr::Var { path } if path.is_single("auto")) || matches!(value, Expr::Lit { lit: Lit::Nat(_), .. }),
            Form::Duration => matches!(value, Expr::Lit { lit: Lit::Quantity { .. }, .. }),
            Form::Axis => true,
        };
        if !ok {
            self.error(Diagnostic::error("E2010", format!("`{field}` expects {}", form.describe())).label(value.span(), "not of the expected form"));
        }
        None
    }

    /// `m.MiniGPT` or `MiniGPT`: a module definition without type parameters.
    fn module_path(&self, path: &Path) -> Option<ModRef> {
        let (unit, name) = match path.segs.len() {
            1 => (self.unit, &path.segs[0].name),
            2 => (*self.prog.units[self.unit].imports.get(&path.segs[0].name)?, &path.segs[1].name),
            _ => return None,
        };
        let mid = *self.prog.units[unit].modules.get(name)?;
        if !self.prog.module(mid).typarams.is_empty() {
            return None;
        }
        Some(ModRef::Inst { def: mid, args: vec![] })
    }

    /// `f(k = v, …)` against a table of constructors and their keyword forms;
    /// a key `_` accepts one positional bare name (`quantize(int8, …)`).
    fn call_form(&mut self, value: &Expr, ctors: &[(&str, &[(&str, Form)])]) -> bool {
        let Expr::Call { callee, args, span } = value else {
            return false;
        };
        let Expr::Var { path } = &**callee else { return false };
        let Some((_, kws)) = ctors.iter().find(|(n, _)| path.is_single(n)) else {
            let known: Vec<String> = ctors.iter().map(|(n, _)| format!("`{n}`")).collect();
            self.error(Diagnostic::error("E2010", format!("unknown constructor `{}`", path.dotted())).label(path.span, format!("expected {}", known.join(" or "))));
            return false;
        };
        let mut ok = true;
        for a in args {
            match a {
                Arg::Named { name, value } => match kws.iter().find(|(k, _)| *k == name.name) {
                    Some((_, f)) => {
                        let f = f.clone();
                        if self.form(&f, value, &name.name).is_none() && !form_is_module(&f) {
                            // errors already reported by `form`
                        }
                    }
                    None => {
                        self.error(Diagnostic::error("E2010", format!("`{}` has no argument `{}`", path.dotted(), name.name)).label(name.span, "unknown keyword"));
                        ok = false;
                    }
                },
                Arg::Axis(id) => {
                    if !kws.iter().any(|(k, f)| *k == "axis" && matches!(f, Form::Axis)) {
                        self.error(Diagnostic::error("E2010", format!("`{}` takes no `axis` argument", path.dotted())).label(id.span, "unexpected axis"));
                        ok = false;
                    }
                }
                Arg::Pos(Expr::Var { path: p }) if kws.iter().any(|(k, _)| *k == "_") && p.segs.len() == 1 => {}
                Arg::Pos(e) | Arg::NamedAxis { value: e, .. } => {
                    self.error(Diagnostic::error("E2010", format!("`{}` takes keyword arguments only", path.dotted())).label(e.span(), "unexpected positional argument"));
                    ok = false;
                }
            }
        }
        let _ = span;
        ok
    }

    /// `a OP b OP c` where every part is a constructor call.
    fn chain(&mut self, value: &Expr, op: BinOp, ctors: &[(&str, &[(&str, Form)])]) -> bool {
        match value {
            Expr::Binary { op: o, lhs, rhs, .. } if *o == op => self.chain(lhs, op, ctors) & self.chain(rhs, op, ctors),
            _ => self.call_form(value, ctors),
        }
    }

    /// `stream "path" |> shuffle(seed = N) |> batch(B = N)`
    fn pipeline(&mut self, value: &Expr) -> bool {
        match value {
            Expr::Binary { op: BinOp::Pipe, lhs, rhs, .. } => {
                let l = self.pipeline(lhs);
                let r = self.call_form(rhs, &[("shuffle", &[("seed", Form::Nat)]), ("batch", &[("B", Form::Nat), ("size", Form::Nat)]), ("take", &[("n", Form::Nat)])]);
                l & r
            }
            Expr::Call { callee, args, .. } => {
                matches!(&**callee, Expr::Var { path } if path.is_single("stream")) && args.len() == 1 && matches!(&args[0], Arg::Pos(Expr::Lit { lit: Lit::Str(_), .. }))
            }
            _ => false,
        }
    }
}

fn form_is_module(f: &Form) -> bool {
    matches!(f, Form::Module)
}

#[derive(Clone, Debug)]
enum Form {
    F64,
    Nat,
    Str,
    F64Pair,
    /// A bare name (`adamw`, `int8`, `sm_90`).
    Name,
    Module,
    Schedule,
    Data,
    Parallel,
    Precision,
    Target,
    Batching,
    Cache,
    AutoOrNat,
    Duration,
    Axis,
}

impl Form {
    fn describe(&self) -> &'static str {
        match self {
            Form::F64 => "`f64`",
            Form::Nat => "`nat`",
            Form::Str => "a string",
            Form::F64Pair => "`(f64, f64)`",
            Form::Name => "a name",
            Form::Module => "a module",
            Form::Schedule => "`cosine(warmup = N)`, `linear(warmup = N)` or `constant()`",
            Form::Data => "`stream \"path\" |> shuffle(seed = N) |> batch(B = N)`",
            Form::Parallel => "`data_parallel(replicas = auto|N) ⊗ tensor_parallel(axis NAME)`",
            Form::Precision => "`budget(rel_error|abs_error = ε) [| quantize(int8, calibration = \"path\")]`",
            Form::Target => "`gpu(NAME) | cpu(NAME)`",
            Form::Batching => "`dynamic(max_batch = N, timeout = 5ms)`",
            Form::Cache => "`kv_cache(axis NAME)`",
            Form::AutoOrNat => "`auto` or a natural",
            Form::Duration => "a duration such as `5ms`",
            Form::Axis => "an axis name",
        }
    }
}

const OPTIMIZER: [(&str, Form); 6] = [("lr", Form::F64), ("betas", Form::F64Pair), ("weight_decay", Form::F64), ("grad_clip", Form::F64), ("schedule", Form::Schedule), ("eps", Form::F64)];
const TRAINING: [(&str, Form); 7] =
    [("model", Form::Module), ("objective", Form::Name), ("optimizer", Form::Name), ("data", Form::Data), ("epochs", Form::Nat), ("parallel", Form::Parallel), ("precision", Form::Precision)];
const INFERENCE: [(&str, Form); 6] =
    [("model", Form::Module), ("weights", Form::Str), ("precision", Form::Precision), ("target", Form::Target), ("batching", Form::Batching), ("cache", Form::Cache)];
