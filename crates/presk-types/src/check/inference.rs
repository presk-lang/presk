//! The `inference serve` block resolved (plan 0.3.0, step 5): its model, the
//! weights file and the *generation function*, the top-level function of the
//! file whose signature is
//!
//! ```text
//! (θ : params M, ctx : idx[V][batch: B, seq: T], … with defaults) -> idx[V][batch: B]
//! ```
//!
//! (`next_token` in the specification's example). The field `generate =
//! name` names it (the reference's decision W4, plan 0.10.0 step 8): a name
//! that is not a top-level function of the file, or whose signature does
//! not fit, is an error (`E2010`); without the field, the first function of
//! the file whose signature fits, in declaration order. The other fields
//! (`precision`, `target`, `batching`, `cache`) pass the schema and are
//! recorded as ignored: the minimal `presk serve` runs the generation
//! function on the whole context, without cache or batching (question 37).
//! A file with the block but no generation function gets the warning
//! `W0003`; `presk serve --entry` then names the function.

use std::path::PathBuf;

use presk_syntax::ast::{ConfigItem, Expr, Lit};
use presk_syntax::{Diagnostic, Span};

use super::Checker;
use crate::elab::{ElabProgram, FnId};
use crate::ty::{ModRef, Ty};

/// One `inference serve`, resolved.
#[derive(Clone, Debug)]
pub struct InferenceServe {
    pub span: Span,
    /// The directory of the `.psk` file declaring the block (`weights` is
    /// relative to it).
    pub dir: PathBuf,
    pub model: ModRef,
    /// `weights = "ckpt/final"` as written.
    pub weights: Option<String>,
    /// The weights file: `dir/ckpt/final.psk.d`.
    pub weights_path: Option<PathBuf>,
    /// The generation function: the one `generate` names, or the first
    /// that fits, when the file has one.
    pub generate: Option<FnId>,
    /// Fields checked by the schema but without effect in this version.
    pub ignored: Vec<String>,
}

impl Checker<'_> {
    /// Builds the record of an `inference serve` whose fields passed the
    /// schema; `model` is its resolved `model` field.
    pub(crate) fn inference_serve(&mut self, items: &[ConfigItem], model: Option<ModRef>, span: Span) -> Option<InferenceServe> {
        let model = model?;
        let mut weights = None;
        let mut named: Option<&presk_syntax::ast::Path> = None;
        let mut ignored = Vec::new();
        for item in items {
            if let ConfigItem::Field { name, value, .. } = item {
                match (name.name.as_str(), value) {
                    ("weights", Expr::Lit { lit: Lit::Str(s), .. }) => weights = Some(s.clone()),
                    ("generate", Expr::Var { path }) => named = Some(path),
                    ("model", _) | ("weights", _) | ("generate", _) => {}
                    (other, _) => ignored.push(other.to_string()),
                }
            }
        }
        let unit_dir = self.prog.units[self.unit].path.parent().map(std::path::Path::to_path_buf).unwrap_or_default();
        let weights_path = weights.as_ref().map(|w| unit_dir.join(if w.ends_with(".psk.d") { w.clone() } else { format!("{w}.psk.d") }));
        // The generation function: the one `generate` names, or by
        // signature, in declaration order.
        let generate = match named {
            Some(path) => {
                let Some(&fid) = self.prog.units[self.unit].fns.get(&path.segs[0].name) else {
                    self.error(Diagnostic::error("E2010", format!("`generate`: `{}` is not a function of this file", path.dotted())).label(path.span, "expected a top-level `fn`"));
                    return Some(InferenceServe { span, dir: unit_dir, model, weights, weights_path, generate: None, ignored });
                };
                if !self.is_generation_fn(fid, &model) {
                    let m = model.display(&self.prog.names());
                    self.error(
                        Diagnostic::error("E2010", format!("`generate`: `{}` is not a generation function of `{m}`", path.dotted()))
                            .label(path.span, "its signature does not fit")
                            .label(self.prog.function(fid).span, "declared here")
                            .help(format!("a generation function is `fn (θ : params {m}, [key : key,] ctx : idx[V][batch: B, seq: T], …) -> idx[V][batch: B]`, its other parameters with defaults")),
                    );
                    return Some(InferenceServe { span, dir: unit_dir, model, weights, weights_path, generate: None, ignored });
                }
                Some(fid)
            }
            None => {
                let fns: Vec<FnId> = self.prog.units[self.unit].fns.values().copied().collect();
                fns.into_iter().find(|&fid| self.is_generation_fn(fid, &model))
            }
        };
        if generate.is_none() {
            self.error(
                Diagnostic::warning("W0003", "no generation function for this `inference serve`")
                    .label(span, "no top-level `fn (θ : params M, [key : key,] ctx : idx[V][batch: B, seq: T], …) -> idx[V][batch: B]` in this file")
                    .help("`presk serve` needs `--entry FN` to name the function that produces the next token"),
            );
        }
        Some(InferenceServe { span, dir: unit_dir, model, weights, weights_path, generate, ignored })
    }

    /// `(θ : params M, [key : key,] ctx : idx[V][batch: B, seq: T], …
    /// defaults) -> idx[V][batch: B]`: the reference's definition
    /// "Generation function", the key taking a key of its own at every token
    /// (plan 0.10.0 step 7b).
    fn is_generation_fn(&mut self, fid: FnId, model: &ModRef) -> bool {
        let f = self.prog.function(fid).clone();
        if f.owner.is_some() || f.params.len() < 2 {
            return false;
        }
        let Some(Ty::Params(m)) = &f.params[0].ty else { return false };
        let m = m.clone();
        if !self.trial(|c| c.unify_modref(&m, model, f.span)) {
            return false;
        }
        let c = if matches!(f.params[1].ty, Some(Ty::Key)) { 2 } else { 1 };
        let Some(Ty::Idx { shape, .. }) = f.params.get(c).and_then(|p| p.ty.as_ref()) else { return false };
        let has = |n: &str| shape.axes.iter().any(|a| a.name.as_deref() == Some(n));
        if shape.prefix.is_some() || shape.axes.len() != 2 || !has("batch") || !has("seq") {
            return false;
        }
        if f.params[c + 1..].iter().any(|p| p.default.is_none()) {
            return false;
        }
        match &f.ret {
            Some(Ty::Idx { shape, .. }) => shape.prefix.is_none() && shape.axes.len() == 1 && shape.axes[0].name.as_deref() == Some("batch"),
            _ => false,
        }
    }
}

/// The `--summary` lines of the inference blocks.
pub fn summary(prog: &ElabProgram, serves: &[InferenceServe]) -> String {
    let n = prog.names();
    let mut s = String::new();
    for r in serves {
        s.push_str(&format!("inference serve: model {}\n", r.model.display(&n)));
        match (&r.weights, &r.weights_path) {
            (Some(w), Some(p)) => s.push_str(&format!("    weights : {w} ({}{})\n", p.display(), if p.is_file() { "" } else { ", not found" })),
            _ => s.push_str("    weights : none (give --weights to `presk serve`)\n"),
        }
        match r.generate {
            Some(f) => {
                let f = prog.function(f);
                let params: Vec<String> = f.params.iter().map(|p| p.name.clone()).collect();
                s.push_str(&format!("    generate: {}({})\n", f.name, params.join(", ")));
            }
            None => s.push_str("    generate: none (W0003)\n"),
        }
        if !r.ignored.is_empty() {
            s.push_str(&format!("    ignored : {} (checked, without effect in this version)\n", r.ignored.join(", ")));
        }
    }
    s
}
