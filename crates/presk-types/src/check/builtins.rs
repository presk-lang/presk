//! The builtin functions of 0.1.0 and their typing rules.

use presk_syntax::{Diagnostic, Span};

use super::expr::TArg;
use super::{Checker, Env};
use crate::dim::DimExpr;
use crate::ty::{Axis, Shape, Ty};

const ELEMENTWISE: &[&str] = &["gelu", "relu", "exp", "tanh", "sqrt", "abs", "log", "sigmoid", "sin", "cos"];
const NOT_IMPLEMENTED: &[&str] = &["vjp", "jvp", "concat", "clamp", "max", "min"];
const ALL: &[&str] = &["mean", "sum", "softmax", "embed", "one_hot", "merge", "dropout", "split_key", "cross_entropy", "argmax", "sample", "real", "split_heads", "merge_heads", "positions", "causal_mask", "grad", "matmul"];

impl Checker<'_> {
    pub(crate) fn is_builtin(&self, name: &str) -> bool {
        ELEMENTWISE.contains(&name) || NOT_IMPLEMENTED.contains(&name) || ALL.contains(&name)
    }

    /// `Some(result)` if `name` is a builtin (result `None` on error).
    pub(crate) fn builtin(&mut self, name: &str, args: &[TArg], span: Span, _env: &Env) -> Option<Option<Ty>> {
        if !self.is_builtin(name) {
            return None;
        }
        Some(self.builtin_inner(name, args, span))
    }

    fn one_tensor(&mut self, name: &str, args: &[TArg], span: Span) -> Option<(Ty, Shape)> {
        let [TArg::Pos(t, s)] = args else {
            self.error(Diagnostic::error("E2004", format!("`{name}` takes one tensor argument")).label(span, format!("{} given", args.len())));
            return None;
        };
        match self.zonk(t) {
            Ty::Real(sh) => Some((Ty::Real(sh.clone()), sh)),
            other => {
                let st = self.show(&other);
                self.error(Diagnostic::error("E2002", format!("`{name}` expects a real tensor, found `{st}`")).label(*s, "not a real tensor"));
                None
            }
        }
    }

    /// `(axis a, x)`: the tensor's shape and the position of `a` in it.
    fn axis_and_tensor(&mut self, name: &str, args: &[TArg], span: Span) -> Option<(Shape, usize)> {
        let [TArg::Axis(a, aspan), TArg::Pos(t, s)] = args else {
            self.error(Diagnostic::error("E2004", format!("`{name}` takes `(axis NAME, tensor)`")).label(span, "wrong arguments"));
            return None;
        };
        let Ty::Real(sh) = self.zonk(t) else {
            let st = self.show(t);
            self.error(Diagnostic::error("E2002", format!("`{name}` expects a real tensor, found `{st}`")).label(*s, "not a real tensor"));
            return None;
        };
        let Some(pos) = sh.axes.iter().position(|x| x.name.as_deref() == Some(a)) else {
            self.no_axis(a, &sh, *aspan);
            return None;
        };
        self.typed.axis_pos.insert(*aspan, pos);
        Some((sh, pos))
    }

    fn builtin_inner(&mut self, name: &str, args: &[TArg], span: Span) -> Option<Ty> {
        if ELEMENTWISE.contains(&name) {
            return self.one_tensor(name, args, span).map(|(t, _)| t);
        }
        if NOT_IMPLEMENTED.contains(&name) {
            self.error(Diagnostic::error("E2012", format!("`{name}` is not implemented in 0.1.0")).label(span, "specification builtin without an implementation yet"));
            return None;
        }
        match name {
            "mean" | "sum" => match args {
                [TArg::Pos(..)] => {
                    self.one_tensor(name, args, span)?;
                    Some(Ty::scalar_real())
                }
                _ => {
                    let (mut sh, pos) = self.axis_and_tensor(name, args, span)?;
                    sh.axes.remove(pos);
                    Some(Ty::Real(sh))
                }
            },
            "softmax" => {
                let (sh, _) = self.axis_and_tensor(name, args, span)?;
                Some(Ty::Real(sh))
            }
            "real" => {
                let [TArg::Pos(t, s)] = args else {
                    self.error(Diagnostic::error("E2004", "`real` takes one argument").label(span, "wrong arguments"));
                    return None;
                };
                match self.zonk(t) {
                    Ty::Nat | Ty::Int | Ty::Rat | Ty::F64 => Some(Ty::scalar_real()),
                    Ty::Idx { shape, .. } => Some(Ty::Real(shape)),
                    other => {
                        let st = self.show(&other);
                        self.error(Diagnostic::error("E2002", format!("`real` expects a number or an index tensor, found `{st}`")).label(*s, "cannot be injected into ℝ"));
                        None
                    }
                }
            }
            "embed" => {
                let [TArg::Pos(w, ws), TArg::Pos(t, ts)] = args else {
                    self.error(Diagnostic::error("E2004", "`embed` takes `(table, indices)`").label(span, "wrong arguments"));
                    return None;
                };
                let Ty::Real(wsh) = self.zonk(w) else {
                    let st = self.show(w);
                    self.error(Diagnostic::error("E2002", format!("`embed` table must be a real tensor `ℝ[V, D]`, found `{st}`")).label(*ws, "not a real tensor"));
                    return None;
                };
                let Ty::Idx { bound, shape } = self.zonk(t) else {
                    let st = self.show(t);
                    self.error(Diagnostic::error("E2002", format!("`embed` indices must be an index tensor `idx[V][…]`, found `{st}`")).label(*ts, "not an index tensor"));
                    return None;
                };
                if wsh.prefix.is_some() || wsh.axes.len() != 2 {
                    let n = self.prog.names();
                    self.error(Diagnostic::error("E2003", "`embed` table must have rank 2").label(*ws, format!("this has shape `{}`", wsh.display(&n))));
                    return None;
                }
                self.unify_dim(&wsh.axes[0].len, &bound, *ts);
                let mut out = shape.clone();
                // The table's column name, when it has one, names the new axis;
                // otherwise it stays unnamed and the context names it.
                out.axes.push(Axis { name: wsh.axes[1].name.clone(), len: wsh.axes[1].len.clone() });
                Some(Ty::Real(out))
            }
            "cross_entropy" => {
                let [TArg::Pos(l, ls), TArg::Pos(y, ys)] = args else {
                    self.error(Diagnostic::error("E2004", "`cross_entropy` takes `(logits, targets)`").label(span, "wrong arguments"));
                    return None;
                };
                let Ty::Real(lsh) = self.zonk(l) else {
                    let st = self.show(l);
                    self.error(Diagnostic::error("E2002", format!("`cross_entropy` logits must be a real tensor, found `{st}`")).label(*ls, "not a real tensor"));
                    return None;
                };
                let Ty::Idx { bound, shape } = self.zonk(y) else {
                    let st = self.show(y);
                    self.error(Diagnostic::error("E2002", format!("`cross_entropy` targets must be an index tensor, found `{st}`")).label(*ys, "not an index tensor"));
                    return None;
                };
                let (lsh, shape) = (self.resolve_shape(&lsh), self.resolve_shape(&shape));
                if lsh.axes.is_empty() {
                    self.error(Diagnostic::error("E2003", "`cross_entropy` logits need a last axis of size the vocabulary").label(*ls, "scalar logits"));
                    return None;
                }
                let v = lsh.axes.last().unwrap().len.clone();
                self.unify_dim(&v, &bound, *ys);
                let lead = Shape { prefix: lsh.prefix, axes: lsh.axes[..lsh.axes.len() - 1].to_vec() };
                self.unify_shape(&lead, &shape, *ys);
                Some(Ty::Real(shape))
            }
            "split_key" => {
                let [TArg::Pos(k, ks)] = args else {
                    self.error(Diagnostic::error("E2004", "`split_key` takes one key").label(span, format!("{} given", args.len())));
                    return None;
                };
                if !matches!(self.zonk(k), Ty::Key) {
                    let st = self.show(k);
                    self.error(Diagnostic::error("E2002", format!("`split_key` expects a key, found `{st}`")).label(*ks, "not a key"));
                    return None;
                }
                Some(Ty::Tuple(vec![Ty::Key, Ty::Key]))
            }
            "dropout" => {
                let [TArg::Pos(r, rs), TArg::Pos(k, ks), TArg::Pos(t, ts)] = args else {
                    self.error(Diagnostic::error("E2004", "`dropout` takes `(rate, key, tensor)`").label(span, "wrong arguments"));
                    return None;
                };
                match self.zonk(r) {
                    Ty::F64 | Ty::Nat | Ty::Rat | Ty::Real(_) => {}
                    other => {
                        let st = self.show(&other);
                        self.error(Diagnostic::error("E2002", format!("`dropout` rate must be a number, found `{st}`")).label(*rs, "not a rate"));
                        return None;
                    }
                }
                if !matches!(self.zonk(k), Ty::Key) {
                    let st = self.show(k);
                    self.error(
                        Diagnostic::error("E2002", format!("`dropout` needs a key, found `{st}`"))
                            .label(*ks, "not a key")
                            .note("the mask is a function of the key alone, so differentiation recomputes the same one; a function that drops out takes a `key` parameter, as an initializer does"),
                    );
                    return None;
                }
                let Ty::Real(sh) = self.zonk(t) else {
                    let st = self.show(t);
                    self.error(Diagnostic::error("E2002", format!("`dropout` expects a real tensor, found `{st}`")).label(*ts, "not a real tensor"));
                    return None;
                };
                Some(Ty::Real(sh))
            }
            "merge" => {
                let [TArg::Axis(a, aspan), TArg::Axis(b, bspan), TArg::Pos(t, s)] = args else {
                    self.error(Diagnostic::error("E2004", "`merge` takes `(axis A, axis B, tensor)`").label(span, "wrong arguments"));
                    return None;
                };
                let Ty::Real(sh) = self.zonk(t) else {
                    let st = self.show(t);
                    self.error(Diagnostic::error("E2002", format!("`merge` expects a real tensor, found `{st}`")).label(*s, "not a real tensor"));
                    return None;
                };
                let sh = self.resolve_shape(&sh);
                let n = self.prog.names();
                let Some(p) = sh.axes.iter().position(|x| x.name.as_deref() == Some(a.as_str())) else {
                    self.no_axis(a, &sh, *aspan);
                    return None;
                };
                let Some(q) = sh.axes.iter().position(|x| x.name.as_deref() == Some(b.as_str())) else {
                    self.no_axis(b, &sh, *bspan);
                    return None;
                };
                if q != p + 1 {
                    self.error(
                        Diagnostic::error("E2003", format!("`merge` needs `{a}` immediately followed by `{b}`"))
                            .label(span, format!("`{a}` is axis {p} and `{b}` is axis {q} of `ℝ{}`", sh.display(&n)))
                            .note("merging axes that are not adjacent would also transpose; `merge` never moves data"),
                    );
                    return None;
                }
                let Some(len) = sh.axes[p].len.mul(&sh.axes[q].len) else {
                    self.error(
                        Diagnostic::error("E1007", format!("`merge` of `{a}` and `{b}` multiplies two symbolic dimensions"))
                            .label(span, format!("`{}` · `{}` is outside the decidable fragment", sh.axes[p].len.display(&*n.dim), sh.axes[q].len.display(&*n.dim)))
                            .help("at least one of the two axes must have a constant length"),
                    );
                    return None;
                };
                self.typed.axis_pos.insert(*aspan, p);
                self.typed.axis_pos.insert(*bspan, q);
                let mut axes: Vec<Axis> = sh.axes[..p].to_vec();
                axes.push(Axis { name: None, len });
                axes.extend_from_slice(&sh.axes[q + 1..]);
                Some(Ty::Real(Shape { prefix: sh.prefix, axes }))
            }
            "one_hot" => {
                let [TArg::Pos(t, s)] = args else {
                    self.error(Diagnostic::error("E2004", "`one_hot` takes one index tensor").label(span, format!("{} given", args.len())));
                    return None;
                };
                let Ty::Idx { bound, shape } = self.zonk(t) else {
                    let st = self.show(t);
                    self.error(Diagnostic::error("E2002", format!("`one_hot` expects an index tensor `idx[V][…]`, found `{st}`")).label(*s, "not an index tensor"));
                    return None;
                };
                let mut out = self.resolve_shape(&shape);
                out.axes.push(Axis { name: None, len: bound });
                Some(Ty::Real(out))
            }
            "argmax" | "sample" => {
                let (_, sh) = self.one_tensor(name, args, span)?;
                let sh = self.resolve_shape(&sh);
                let Some(last) = sh.axes.last() else {
                    self.error(Diagnostic::error("E2003", format!("`{name}` needs a last axis to reduce")).label(span, "scalar argument"));
                    return None;
                };
                let bound = last.len.clone();
                Some(Ty::Idx { bound, shape: Shape { prefix: sh.prefix, axes: sh.axes[..sh.axes.len() - 1].to_vec() } })
            }
            "split_heads" | "merge_heads" => {
                let (_, sh) = self.one_tensor(name, args, span)?;
                let sh = self.resolve_shape(&sh);
                let (Some(h), Some(hd)) = (self.const_named("H"), self.const_named("HD")) else {
                    self.error(Diagnostic::error("E2012", format!("`{name}` needs the constants `H` and `HD` in scope")).label(span, "0.1.0 reads the number of heads from `const H` and the head size from `const HD`"));
                    return None;
                };
                let n = self.prog.names();
                if name == "split_heads" {
                    let Some(last) = sh.axes.last() else {
                        self.error(Diagnostic::error("E2003", "`split_heads` needs a trailing model axis").label(span, "scalar argument"));
                        return None;
                    };
                    self.unify_dim(&last.len, &DimExpr::nat(h * hd), span);
                    if sh.axes.len() < 2 {
                        self.error(Diagnostic::error("E2003", "`split_heads` expects `ℝ[batch, seq, model]`").label(span, format!("this has shape `{}`", sh.display(&n))));
                        return None;
                    }
                    let k = sh.axes.len();
                    let mut axes = sh.axes[..k - 2].to_vec();
                    axes.push(Axis { name: Some("heads".into()), len: DimExpr::nat(h) });
                    axes.push(sh.axes[k - 2].clone());
                    axes.push(Axis { name: Some("hd".into()), len: DimExpr::nat(hd) });
                    Some(Ty::Real(Shape { prefix: sh.prefix, axes }))
                } else {
                    let k = sh.axes.len();
                    if k < 3 || sh.axes[k - 3].name.as_deref() != Some("heads") || sh.axes[k - 1].name.as_deref() != Some("hd") {
                        self.error(Diagnostic::error("E2003", "`merge_heads` expects `ℝ[batch, heads, seq, hd]`").label(span, format!("this has shape `{}`", sh.display(&n))));
                        return None;
                    }
                    self.unify_dim(&sh.axes[k - 3].len, &DimExpr::nat(h), span);
                    self.unify_dim(&sh.axes[k - 1].len, &DimExpr::nat(hd), span);
                    let mut axes = sh.axes[..k - 3].to_vec();
                    axes.push(sh.axes[k - 2].clone());
                    axes.push(Axis { name: Some("model".into()), len: DimExpr::nat(h * hd) });
                    Some(Ty::Real(Shape { prefix: sh.prefix, axes }))
                }
            }
            "positions" | "causal_mask" => {
                let [TArg::NamedAxis(a, len, _)] = args else {
                    self.error(Diagnostic::error("E2004", format!("`{name}` takes one named axis, e.g. `{name}(seq: T)`")).label(span, "wrong arguments"));
                    return None;
                };
                if name == "positions" {
                    Some(Ty::Idx { bound: len.clone(), shape: Shape::closed(vec![Axis { name: Some(a.clone()), len: len.clone() }]) })
                } else {
                    Some(Ty::Real(Shape::closed(vec![Axis { name: Some(a.clone()), len: len.clone() }, Axis { name: Some(format!("{a}2")), len: len.clone() }])))
                }
            }
            "grad" => {
                let [TArg::Pos(f, fs)] = args else {
                    self.error(Diagnostic::error("E2004", "`grad` takes one function").label(span, "wrong arguments"));
                    return None;
                };
                let Ty::Fn { params, ret } = self.zonk(f) else {
                    let st = self.show(f);
                    self.error(Diagnostic::error("E2002", format!("`grad` expects a function `λ θ. …`, found `{st}`")).label(*fs, "not a function"));
                    return None;
                };
                if params.len() != 1 {
                    self.error(Diagnostic::error("E2004", "`grad` differentiates a function of one argument").label(*fs, format!("{} parameters", params.len())));
                    return None;
                }
                self.unify(&Ty::scalar_real(), &ret, *fs);
                Some(Ty::Fn { params: params.clone(), ret: Box::new(params[0].clone()) })
            }
            "matmul" => {
                let [TArg::Pos(a, sa), TArg::Pos(b, sb)] = args else {
                    self.error(Diagnostic::error("E2004", "`matmul` takes two matrices").label(span, "wrong arguments"));
                    return None;
                };
                let (Ty::Real(x), Ty::Real(y)) = (self.zonk(a), self.zonk(b)) else {
                    self.error(Diagnostic::error("E2002", "`matmul` expects real tensors").label(span, "not real tensors"));
                    return None;
                };
                let (x, y) = (self.resolve_shape(&x), self.resolve_shape(&y));
                if x.rank() != Some(2) || y.rank() != Some(2) {
                    self.error(Diagnostic::error("E2003", "`matmul` expects rank-2 tensors").label(*sa, "left operand").label(*sb, "right operand"));
                    return None;
                }
                self.unify_dim(&x.axes[1].len, &y.axes[0].len, span);
                Some(Ty::Real(Shape::closed(vec![x.axes[0].clone(), y.axes[1].clone()])))
            }
            _ => unreachable!("builtin table"),
        }
    }

    fn const_named(&self, name: &str) -> Option<u64> {
        self.prog.units[self.unit].consts.get(name).map(|c| c.value)
    }
}
