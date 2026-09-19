//! The optimizer of a `training run` as PSK-Core functions (plan 0.2.0,
//! step 3). For `optimizer adamw { … }` used with a model `M`
//! (`P = params M`):
//!
//! ```text
//! adamw.init(θ : P) -> (P, P)                          -- moments at zero
//! adamw.lr(t : nat, total : nat) -> ℝ[]               -- the schedule
//! adamw.step(θ : P, s : (P, P), g : P, t : nat, total : nat) -> (P, (P, P))
//! ```
//!
//! AdamW as PyTorch defines it: the gradient is first clipped to the global
//! norm `c` when `grad_clip = c` (`g · min(1, c / (‖g‖₂ + 1e-6))`), the
//! moments are `m' = β₁ m + (1 − β₁) g`, `v' = β₂ v + (1 − β₂) g²`, bias
//! corrected by `1 − βᵗ`, and `θ' = θ − lr(t) · (m̂ / (√v̂ + ε) + λ θ)` with a
//! decoupled weight decay `λ`. Schedules over `total = epochs × batches`
//! steps, `t` counted from 1: `constant()`, `linear(warmup = w)` (`lr · t/w`
//! then `lr · (total − t)/(total − w)`), `cosine(warmup = w)` (`lr · t/w`
//! then `lr · ½(1 + cos(π (t − w)/(total − w)))`). The piecewise forms are
//! written with `relu` (`min(1, x) = 1 − relu(1 − x)`), so the functions use
//! only the primitives of the language plus `pi`, the constant π at the
//! evaluator's precision (a decimal literal would pass through `f64` like a
//! hyperparameter and leave the cosine schedule at 4e-33 instead of 0 at its
//! end). Everything is ordinary PSK-Core: the same interpreter, export and
//! future code generation apply to it.

use presk_syntax::Span;
use presk_syntax::ast::{Arg, ConfigItem, Expr, Lit};
use presk_types::ty::{ModRef, Ty};

use super::{Builder, Lowerer};
use crate::ir::{Block, Func, FuncId, Lit as CLit, Origin, Param, PrimOp, Rhs, VarId};

/// The hyperparameters of an `optimizer` block, with the defaults of 0.2.0.
#[derive(Clone, Debug)]
pub struct AdamW {
    pub lr: String,
    pub beta1: String,
    pub beta2: String,
    pub eps: String,
    pub weight_decay: String,
    pub grad_clip: Option<String>,
    pub schedule: Schedule,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Schedule {
    Constant,
    Linear { warmup: u64 },
    Cosine { warmup: u64 },
}

impl AdamW {
    /// Reads the block (already validated against the schema by the checker).
    pub fn from_items(items: &[ConfigItem]) -> AdamW {
        let mut o = AdamW { lr: "1e-3".into(), beta1: "0.9".into(), beta2: "0.999".into(), eps: "1e-8".into(), weight_decay: "0".into(), grad_clip: None, schedule: Schedule::Constant };
        let text = |e: &Expr| -> Option<String> {
            match e {
                Expr::Lit { lit: Lit::Float(f), .. } | Expr::Lit { lit: Lit::Nat(f), .. } => Some(f.clone()),
                _ => None,
            }
        };
        for item in items {
            let ConfigItem::Field { name, value, .. } = item else { continue };
            match name.name.as_str() {
                "lr" => o.lr = text(value).unwrap_or(o.lr),
                "eps" => o.eps = text(value).unwrap_or(o.eps),
                "weight_decay" => o.weight_decay = text(value).unwrap_or(o.weight_decay),
                "grad_clip" => o.grad_clip = text(value),
                "betas" => {
                    if let Expr::Tuple { elems, .. } = value
                        && elems.len() == 2
                    {
                        o.beta1 = text(&elems[0]).unwrap_or(o.beta1);
                        o.beta2 = text(&elems[1]).unwrap_or(o.beta2);
                    }
                }
                "schedule" => {
                    if let Expr::Call { callee, args, .. } = value
                        && let Expr::Var { path } = &**callee
                    {
                        let warmup = args
                            .iter()
                            .find_map(|a| match a {
                                Arg::Named { name, value: Expr::Lit { lit: Lit::Nat(n), .. } } if name.name == "warmup" => n.parse().ok(),
                                _ => None,
                            })
                            .unwrap_or(0);
                        o.schedule = match path.segs[0].name.as_str() {
                            "linear" => Schedule::Linear { warmup },
                            "cosine" => Schedule::Cosine { warmup },
                            _ => Schedule::Constant,
                        };
                    }
                }
                _ => {}
            }
        }
        o
    }
}

impl Lowerer<'_> {
    /// One set of optimizer functions per (optimizer block, model) of the
    /// training runs of the main file.
    pub(crate) fn lower_optimizers(&mut self) {
        let mut done: Vec<(String, ModRef)> = Vec::new();
        for run in &self.typed.training {
            if run.optimizer.is_empty() || done.iter().any(|(n, m)| *n == run.optimizer && *m == run.model) {
                continue;
            }
            let Some(items) = self.elab.units[self.elab.main].optimizers.get(&run.optimizer).cloned() else { continue };
            done.push((run.optimizer.clone(), run.model.clone()));
            let o = AdamW::from_items(&items);
            self.lower_adamw(&run.optimizer, &run.model, &o, run.span);
        }
    }

    fn lower_adamw(&mut self, name: &str, model: &ModRef, o: &AdamW, span: Span) {
        let p = Ty::Params(model.clone());
        let dims = match model {
            ModRef::Inst { def, .. } => self.init_dims(*def),
            _ => vec![],
        };
        let origin = Origin::Optimizer { name: name.to_string(), model: model.clone() };
        let scalar = Ty::scalar_real();
        let state_ty = Ty::Tuple(vec![p.clone(), p.clone()]);

        // ----- init(θ) -> (m, v) -----
        let id = FuncId(self.funcs.len() as u32);
        let mut b = Builder::new(id);
        let theta = b.fresh();
        let m = b.emit(Some("m".into()), p.clone(), Rhs::Prim { op: PrimOp::ZerosLike, args: vec![theta] }, span);
        let v = b.emit(Some("v".into()), p.clone(), Rhs::Prim { op: PrimOp::ZerosLike, args: vec![theta] }, span);
        let r = b.emit(None, state_ty.clone(), Rhs::Tuple(vec![m, v]), span);
        self.funcs.push(Func { name: format!("{name}.init"), origin: origin.clone(), dims: dims.clone(), params: vec![Param { var: theta, name: "θ".into(), ty: p.clone() }], ret: state_ty.clone(), effects: vec![], body: Block { lets: b.lets, result: r }, span });

        // ----- lr(t, total) -> ℝ[] -----
        let id = FuncId(self.funcs.len() as u32);
        let mut b = Builder::new(id);
        let t = b.fresh();
        let total = b.fresh();
        let lr = {
            let mut e = Emit { b: &mut b, scalar: scalar.clone(), span };
            e.schedule(o, t, total)
        };
        self.funcs.push(Func { name: format!("{name}.lr"), origin: origin.clone(), dims: vec![], params: vec![Param { var: t, name: "t".into(), ty: Ty::Nat }, Param { var: total, name: "total".into(), ty: Ty::Nat }], ret: scalar.clone(), effects: vec![], body: Block { lets: b.lets, result: lr }, span });
        let lr_id = id;

        // ----- step(θ, s, g, t, total) -> (θ', s') -----
        let id = FuncId(self.funcs.len() as u32);
        let mut b = Builder::new(id);
        let (theta, s, g, t, total) = (b.fresh(), b.fresh(), b.fresh(), b.fresh(), b.fresh());
        let result = {
            let mut e = Emit { b: &mut b, scalar: scalar.clone(), span };
            let m = e.b.emit(Some("m".into()), p.clone(), Rhs::Proj(s, 0), span);
            let v = e.b.emit(Some("v".into()), p.clone(), Rhs::Proj(s, 1), span);
            // Clipping to the global norm, as `clip_grad_norm_`.
            let g = match &o.grad_clip {
                None => g,
                Some(c) => {
                    let n2 = e.prim(PrimOp::SumSquares, vec![g], scalar.clone());
                    let norm = e.prim(PrimOp::Sqrt, vec![n2], scalar.clone());
                    let tiny = e.lit("1e-6");
                    let denom = e.prim(PrimOp::Add, vec![norm, tiny], scalar.clone());
                    let c = e.lit(c);
                    let coef = e.prim(PrimOp::Div, vec![c, denom], scalar.clone());
                    let factor = e.min1(coef);
                    e.prim(PrimOp::Mul, vec![g, factor], p.clone())
                }
            };
            let one = e.lit("1");
            let (b1, b2) = (e.lit(&o.beta1), e.lit(&o.beta2));
            let omb1 = e.prim(PrimOp::Sub, vec![one, b1], scalar.clone());
            let omb2 = e.prim(PrimOp::Sub, vec![one, b2], scalar.clone());
            // m' = β₁ m + (1 − β₁) g
            let a = e.prim(PrimOp::Mul, vec![m, b1], p.clone());
            let c = e.prim(PrimOp::Mul, vec![g, omb1], p.clone());
            let m2 = e.named("m'", PrimOp::Add, vec![a, c], p.clone());
            // v' = β₂ v + (1 − β₂) g²
            let a = e.prim(PrimOp::Mul, vec![v, b2], p.clone());
            let gg = e.prim(PrimOp::Mul, vec![g, g], p.clone());
            let c = e.prim(PrimOp::Mul, vec![gg, omb2], p.clone());
            let v2 = e.named("v'", PrimOp::Add, vec![a, c], p.clone());
            // bias corrections 1 − βᵗ = 1 − exp(t · ln β)
            let tr = e.prim(PrimOp::Real, vec![t], scalar.clone());
            let bc1 = e.bias_correction(b1, tr, one);
            let bc2 = e.bias_correction(b2, tr, one);
            let mhat = e.prim(PrimOp::Div, vec![m2, bc1], p.clone());
            let vhat = e.prim(PrimOp::Div, vec![v2, bc2], p.clone());
            let sq = e.prim(PrimOp::Sqrt, vec![vhat], p.clone());
            let eps = e.lit(&o.eps);
            let denom = e.prim(PrimOp::Add, vec![sq, eps], p.clone());
            let update = e.prim(PrimOp::Div, vec![mhat, denom], p.clone());
            // + λ θ
            let lambda = e.lit(&o.weight_decay);
            let decay = e.prim(PrimOp::Mul, vec![theta, lambda], p.clone());
            let dir = e.prim(PrimOp::Add, vec![update, decay], p.clone());
            // × lr(t, total)
            let lr = e.b.emit(Some("lr".into()), scalar.clone(), Rhs::Call { f: lr_id, dim_args: Default::default(), args: vec![t, total] }, span);
            let scaled = e.prim(PrimOp::Mul, vec![dir, lr], p.clone());
            let theta2 = e.named("θ'", PrimOp::Sub, vec![theta, scaled], p.clone());
            let s2 = e.b.emit(Some("s'".into()), state_ty.clone(), Rhs::Tuple(vec![m2, v2]), span);
            e.b.emit(None, Ty::Tuple(vec![p.clone(), state_ty.clone()]), Rhs::Tuple(vec![theta2, s2]), span)
        };
        self.funcs.push(Func {
            name: format!("{name}.step"),
            origin,
            dims,
            params: vec![
                Param { var: theta, name: "θ".into(), ty: p.clone() },
                Param { var: s, name: "s".into(), ty: state_ty.clone() },
                Param { var: g, name: "g".into(), ty: p.clone() },
                Param { var: t, name: "t".into(), ty: Ty::Nat },
                Param { var: total, name: "total".into(), ty: Ty::Nat },
            ],
            ret: Ty::Tuple(vec![p, state_ty]),
            effects: vec![],
            body: Block { lets: b.lets, result },
            span,
        });
    }
}

/// Small emission helpers over a builder.
struct Emit<'b> {
    b: &'b mut Builder,
    scalar: Ty,
    span: Span,
}

impl Emit<'_> {
    fn prim(&mut self, op: PrimOp, args: Vec<VarId>, ty: Ty) -> VarId {
        self.b.emit(None, ty, Rhs::Prim { op, args }, self.span)
    }

    fn named(&mut self, name: &str, op: PrimOp, args: Vec<VarId>, ty: Ty) -> VarId {
        self.b.emit(Some(name.into()), ty, Rhs::Prim { op, args }, self.span)
    }

    /// A decimal literal as a real scalar (`f64` text injected exactly).
    fn lit(&mut self, text: &str) -> VarId {
        let f = self.b.emit(None, Ty::F64, Rhs::Lit(CLit::Float(text.into())), self.span);
        self.prim(PrimOp::Real, vec![f], self.scalar.clone())
    }

    /// `min(1, x) = 1 − relu(1 − x)`.
    fn min1(&mut self, x: VarId) -> VarId {
        let one = self.lit("1");
        let d = self.prim(PrimOp::Sub, vec![one, x], self.scalar.clone());
        let r = self.prim(PrimOp::Relu, vec![d], self.scalar.clone());
        self.prim(PrimOp::Sub, vec![one, r], self.scalar.clone())
    }

    /// `1 − βᵗ`.
    fn bias_correction(&mut self, beta: VarId, t: VarId, one: VarId) -> VarId {
        let lnb = self.prim(PrimOp::Log, vec![beta], self.scalar.clone());
        let tl = self.prim(PrimOp::Mul, vec![t, lnb], self.scalar.clone());
        let pw = self.prim(PrimOp::Exp, vec![tl], self.scalar.clone());
        self.prim(PrimOp::Sub, vec![one, pw], self.scalar.clone())
    }

    /// `lr(t, total)` for the schedule; `t`, `total` are `nat` variables.
    fn schedule(&mut self, o: &AdamW, t: VarId, total: VarId) -> VarId {
        let base = self.lit(&o.lr);
        let (warmup, cosine) = match o.schedule {
            Schedule::Constant => return base,
            Schedule::Linear { warmup } => (warmup, false),
            Schedule::Cosine { warmup } => (warmup, true),
        };
        let tr = self.prim(PrimOp::Real, vec![t], self.scalar.clone());
        let totr = self.prim(PrimOp::Real, vec![total], self.scalar.clone());
        let w = self.lit(&warmup.to_string());
        let one = self.lit("1");
        // warm-up factor min(1, t / w), or 1 without warm-up
        let warm = if warmup == 0 {
            one
        } else {
            let q = self.prim(PrimOp::Div, vec![tr, w], self.scalar.clone());
            self.min1(q)
        };
        // progress p = min(1, relu((t − w) / max(total − w, 1)))
        let num = self.prim(PrimOp::Sub, vec![tr, w], self.scalar.clone());
        let den0 = self.prim(PrimOp::Sub, vec![totr, w], self.scalar.clone());
        let den1 = self.prim(PrimOp::Sub, vec![den0, one], self.scalar.clone());
        let den2 = self.prim(PrimOp::Relu, vec![den1], self.scalar.clone());
        let den = self.prim(PrimOp::Add, vec![den2, one], self.scalar.clone());
        let q = self.prim(PrimOp::Div, vec![num, den], self.scalar.clone());
        let r = self.prim(PrimOp::Relu, vec![q], self.scalar.clone());
        let p = self.min1(r);
        let decay = if cosine {
            // ½ (1 + cos(π p))
            let pi = self.prim(PrimOp::Pi, vec![], self.scalar.clone());
            let x = self.prim(PrimOp::Mul, vec![pi, p], self.scalar.clone());
            let c = self.prim(PrimOp::Cos, vec![x], self.scalar.clone());
            let s = self.prim(PrimOp::Add, vec![one, c], self.scalar.clone());
            let half = self.lit("0.5");
            self.prim(PrimOp::Mul, vec![s, half], self.scalar.clone())
        } else {
            self.prim(PrimOp::Sub, vec![one, p], self.scalar.clone())
        };
        let f = self.prim(PrimOp::Mul, vec![warm, decay], self.scalar.clone());
        self.prim(PrimOp::Mul, vec![base, f], self.scalar.clone())
    }
}
