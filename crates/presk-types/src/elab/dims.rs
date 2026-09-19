//! Dimension expressions, constants, `dim` declarations, `static assert`s.

use presk_syntax::ast::{self, BinOp, Expr, Lit, UnOp};
use presk_syntax::{Diagnostic, Span};

use super::{ConstDef, DimOrigin, Elaborator, Scope, UnitId};
use crate::dim::{DimExpr, DimVar};
use crate::presburger::{Constraint, Validity, display_assignment, entails};

/// What a name in a dimension position may resolve to.
pub(crate) enum DimName {
    Const(u64),
    Var(DimVar),
}

impl Elaborator<'_> {
    /// Look a (possibly qualified) name up as a dimension: a local type
    /// parameter, a `dim`, a `const`, or the same through an import alias.
    pub(crate) fn lookup_dim(&self, scope: &Scope, path: &ast::Path) -> Option<DimName> {
        let unit = if path.segs.len() == 2 {
            *self.prog.units[scope.unit].imports.get(&path.segs[0].name)?
        } else if path.segs.len() == 1 {
            for l in &scope.locals {
                if let Some(v) = l.dims.get(&path.segs[0].name) {
                    return Some(DimName::Var(*v));
                }
            }
            scope.unit
        } else {
            return None;
        };
        let name = &path.segs.last().unwrap().name;
        let u = &self.prog.units[unit];
        if let Some(c) = u.consts.get(name) {
            return Some(DimName::Const(c.value));
        }
        if let Some(v) = u.dims.get(name) {
            return Some(DimName::Var(*v));
        }
        None
    }

    /// Convert a syntactic expression in dimension position to a linear form.
    /// `implicit` receives undeclared single names (function signatures turn
    /// them into implicit dimensions); `None` makes them errors.
    pub(crate) fn dim_expr(&mut self, scope: &Scope, e: &Expr, implicit: &mut Option<&mut super::Locals>) -> Option<DimExpr> {
        match e {
            Expr::Lit { lit: Lit::Nat(n), span } => match n.parse::<u64>() {
                Ok(n) => Some(DimExpr::nat(n)),
                Err(_) => {
                    self.error(Diagnostic::error("E1005", format!("`{n}` is too large for a dimension")).label(*span, "must fit in 64 bits"));
                    None
                }
            },
            Expr::Var { path } => {
                if let Some(d) = self.lookup_dim(scope, path) {
                    return Some(match d {
                        DimName::Const(n) => DimExpr::nat(n),
                        DimName::Var(v) => DimExpr::var(v),
                    });
                }
                if path.segs.len() == 1
                    && let Some(locals) = implicit
                {
                    let name = &path.segs[0];
                    if let Some(v) = locals.dims.get(&name.name) {
                        return Some(DimExpr::var(*v));
                    }
                    let v = self.new_dim(&name.name, DimOrigin::Implicit, name.span);
                    locals.dims.insert(name.name.clone(), v);
                    self.diags.push(
                        Diagnostic::warning("W0001", format!("implicit dimension `{}`", name.name))
                            .label(name.span, "not declared; taken as a dimension of this function")
                            .help(format!("declare it with `dim {}` at file level or as a type parameter `{{{} : dim}}`", name.name, name.name)),
                    );
                    return Some(DimExpr::var(v));
                }
                self.error(Diagnostic::error("E1001", format!("unresolved name `{}` in a dimension", path.dotted())).label(path.span, "not a `const`, a `dim` or a type parameter in scope"));
                None
            }
            Expr::Unary { op: UnOp::Neg, expr, .. } => Some(self.dim_expr(scope, expr, implicit)?.neg()),
            Expr::Binary { op, lhs, rhs, span } => {
                let l = self.dim_expr(scope, lhs, implicit)?;
                let r = self.dim_expr(scope, rhs, implicit)?;
                match op {
                    BinOp::Add => Some(l.add(&r)),
                    BinOp::Sub => Some(l.sub(&r)),
                    BinOp::Mul => match l.mul(&r) {
                        Some(p) => Some(p),
                        None => {
                            self.error(
                                Diagnostic::error("E1007", "product of two symbolic dimensions")
                                    .label(*span, format!("`{}` times `{}` is outside the decidable shape fragment", self.show(&l), self.show(&r)))
                                    .note("dimension arithmetic is linear: one factor of a product must be a constant"),
                            );
                            None
                        }
                    },
                    BinOp::Div => {
                        let Some(k) = r.as_nat().filter(|k| *k > 0) else {
                            self.error(
                                Diagnostic::error("E1007", "division by a non-constant dimension")
                                    .label(rhs.span(), format!("`{}` is not a positive constant", self.show(&r)))
                                    .note("a divisor must be a literal or a `const` (ℕ includes 0, so a symbolic divisor could be zero)"),
                            );
                            return None;
                        };
                        let q = l.div(&r)?;
                        if !l.is_const() {
                            // Symbolic numerator: k must divide it.
                            let goal = Constraint::Divides(k, l.clone());
                            match entails(&self.prog.facts, &goal) {
                                Validity::Valid => {}
                                other => {
                                    let why = match other {
                                        Validity::Invalid(a) => format!("fails for {}", display_assignment(&a, &goal, &|v| self.prog.dim_name(v))),
                                        _ => "cannot be proved from the declared facts".into(),
                                    };
                                    self.error(
                                        Diagnostic::error("E1006", format!("`{}` may not be divisible by {k}", self.show(&l)))
                                            .label(*span, why)
                                            .help(format!("add `static assert {} % {k} == 0` before this use", self.show(&l))),
                                    );
                                    return None;
                                }
                            }
                        }
                        Some(q)
                    }
                    BinOp::Rem => {
                        self.error(Diagnostic::error("E1007", "`%` is only allowed in `static assert … % k == 0`").label(*span, "not a dimension expression"));
                        None
                    }
                    _ => {
                        self.error(Diagnostic::error("E1007", format!("`{}` is not a dimension operator", op.text())).label(*span, "expected `+`, `-`, `*` or `/`"));
                        None
                    }
                }
            }
            _ => {
                self.error(Diagnostic::error("E1007", "not a dimension expression").label(e.span(), "expected a literal, a name, or `+ - * /` of dimensions"));
                None
            }
        }
    }

    pub(crate) fn show(&self, e: &DimExpr) -> String {
        e.display(&|v| self.prog.dim_name(v))
    }

    pub(crate) fn const_decl(&mut self, unit: UnitId, name: &ast::Ident, ty: Option<&ast::Ty>, value: &Expr, span: Span, is_main: bool) {
        if let Some(t) = ty
            && !matches!(t, ast::Ty::Named { path, args, .. } if path.is_single("nat") && args.is_empty())
        {
            self.error(Diagnostic::error("E1009", "only `nat` constants are supported").label(t.span(), "expected `nat`"));
        }
        let scope = Scope { unit, locals: vec![] };
        // `--const` applies to every file that declares the constant (the
        // constants of a model usually live in an imported file).
        let _ = is_main;
        let overridden = self.opts.const_overrides.iter().find(|(n, _)| n == &name.name).map(|(_, v)| *v);
        let value = match overridden {
            Some(v) => Some(v),
            None => {
                let e = self.dim_expr(&scope, value, &mut None);
                match e {
                    None => None,
                    Some(e) => match e.as_nat() {
                        Some(n) => Some(n),
                        None => {
                            self.error(
                                Diagnostic::error("E1005", format!("`{}` is not a natural number", name.name))
                                    .label(value.span(), format!("evaluates to {}", self.show(&e)))
                                    .note("constants are evaluated exactly: a division must be exact and the result non-negative"),
                            );
                            None
                        }
                    },
                }
            }
        };
        if let Some(v) = value {
            let def = ConstDef { value: v, span, overridden: overridden.is_some() };
            if self.prog.units[unit].consts.insert(name.name.clone(), def).is_some() {
                self.duplicate("const", name);
            }
        }
    }

    pub(crate) fn dim_decl(&mut self, unit: UnitId, name: &ast::Ident, bound: Option<&Expr>, _span: Span) {
        let v = self.new_dim(&name.name, DimOrigin::Decl, name.span);
        if self.prog.units[unit].dims.insert(name.name.clone(), v).is_some() {
            self.duplicate("dim", name);
            return;
        }
        if let Some(b) = bound {
            let scope = Scope { unit, locals: vec![] };
            if let Some(b) = self.dim_expr(&scope, b, &mut None) {
                self.prog.facts.push(Constraint::Le(DimExpr::var(v), b));
            }
        }
    }

    /// Check a `static assert`; a symbolic assertion that holds is added to
    /// the facts. Returns whether it was discharged.
    pub(crate) fn static_assert(&mut self, unit: UnitId, cond: &Expr, span: Span) -> bool {
        let scope = Scope { unit, locals: vec![] };
        let Some(c) = self.assert_constraint(&scope, cond) else { return false };
        match entails(&self.prog.facts, &c) {
            Validity::Valid => {
                if !c.vars().is_empty() {
                    self.prog.facts.push(c);
                }
                true
            }
            Validity::Invalid(a) => {
                let values = display_assignment(&a, &c, &|v| self.prog.dim_name(v));
                let text = self.sm.slice(span).trim_start_matches("static assert").trim().to_string();
                let mut d = Diagnostic::error("E1004", format!("static assertion fails: {text}"));
                d = if values.is_empty() {
                    d.label(span, format!("`{text}` is false with the constants of this file"))
                } else {
                    d.label(span, format!("fails for {values}"))
                };
                self.error(d);
                false
            }
            Validity::Unknown => {
                self.error(Diagnostic::error("E1004", "static assertion cannot be proved").label(span, "the decision procedure gives up on this assertion").note("supported forms: `a == b`, `a <= b`, `a < b`, `a >= b`, `a > b`, `a % k == 0` with `k` a positive constant"));
                false
            }
        }
    }

    fn assert_constraint(&mut self, scope: &Scope, cond: &Expr) -> Option<Constraint> {
        let Expr::Binary { op, lhs, rhs, span } = cond else {
            self.error(Diagnostic::error("E1007", "unsupported assertion").label(cond.span(), "expected a comparison of dimension expressions"));
            return None;
        };
        // `a % k == 0`
        if let (BinOp::Eq, Expr::Binary { op: BinOp::Rem, lhs: a, rhs: k, .. }) = (op, &**lhs) {
            let a = self.dim_expr(scope, a, &mut None)?;
            let k = self.dim_expr(scope, k, &mut None)?;
            let zero = self.dim_expr(scope, rhs, &mut None)?;
            let Some(k) = k.as_nat().filter(|k| *k > 0) else {
                self.error(Diagnostic::error("E1007", "the modulus must be a positive constant").label(*span, format!("`{}` is not a positive constant", self.show(&k))));
                return None;
            };
            if !zero.is_zero() {
                self.error(Diagnostic::error("E1007", "only `… % k == 0` is supported").label(rhs.span(), "expected `0`"));
                return None;
            }
            return Some(Constraint::Divides(k, a));
        }
        let l = self.dim_expr(scope, lhs, &mut None)?;
        let r = self.dim_expr(scope, rhs, &mut None)?;
        let one = DimExpr::nat(1);
        Some(match op {
            BinOp::Eq => Constraint::Eq(l, r),
            BinOp::Le => Constraint::Le(l, r),
            BinOp::Lt => Constraint::Le(l.add(&one), r),
            BinOp::Ge => Constraint::Le(r, l),
            BinOp::Gt => Constraint::Le(r.add(&one), l),
            _ => {
                self.error(Diagnostic::error("E1007", "unsupported assertion").label(*span, format!("`{}` is not a comparison", op.text())));
                return None;
            }
        })
    }
}
