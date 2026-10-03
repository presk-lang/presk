//! Unification of types, shapes (with open prefixes) and dimensions, and
//! broadcasting.

use std::collections::BTreeMap;

use presk_syntax::{Diagnostic, Span};

use super::Checker;
use crate::dim::{DimExpr, DimVar};
use crate::presburger::{Constraint, Validity, display_assignment, entails};
use crate::ty::{Axis, ModRef, PrefixVar, Shape, Ty};

impl Checker<'_> {
    // ----- resolution --------------------------------------------------

    pub(crate) fn resolve_dim(&self, e: &DimExpr) -> DimExpr {
        let mut cur = e.clone();
        for _ in 0..64 {
            let next = cur.subst(&self.dim_subst);
            if next == cur {
                return cur;
            }
            cur = next;
        }
        cur
    }

    /// Expand bound prefixes and resolve dimensions.
    pub(crate) fn resolve_shape(&self, s: &Shape) -> Shape {
        let mut prefix = s.prefix;
        let mut axes: Vec<Axis> = vec![];
        let mut guard = 0;
        while let Some(p) = prefix {
            guard += 1;
            if guard > 64 {
                break;
            }
            match self.prefix_subst.get(&p) {
                Some(b) => {
                    let mut front = b.axes.clone();
                    front.append(&mut axes);
                    axes = front;
                    prefix = b.prefix;
                }
                None => break,
            }
        }
        axes.extend(s.axes.iter().cloned());
        Shape { prefix, axes: axes.iter().map(|a| Axis { name: a.name.clone(), len: self.resolve_dim(&a.len) }).collect() }
    }

    pub(crate) fn zonk(&self, t: &Ty) -> Ty {
        match t {
            Ty::Infer(v) => match self.ty_subst.get(v) {
                Some(b) => self.zonk(b),
                None => t.clone(),
            },
            Ty::Real(s) => Ty::Real(self.resolve_shape(s)),
            Ty::Idx { bound, shape } => Ty::Idx { bound: self.resolve_dim(bound), shape: self.resolve_shape(shape) },
            Ty::Truth(s) => Ty::Truth(self.resolve_shape(s)),
            Ty::Tuple(ts) => Ty::Tuple(ts.iter().map(|t| self.zonk(t)).collect()),
            Ty::Fn { params, ret } => Ty::Fn { params: params.iter().map(|t| self.zonk(t)).collect(), ret: Box::new(self.zonk(ret)) },
            Ty::Params(m) => Ty::Params(self.zonk_modref(m)),
            Ty::Module(m) => Ty::Module(self.zonk_modref(m)),
            t => t.clone(),
        }
    }

    pub(crate) fn zonk_modref(&self, m: &ModRef) -> ModRef {
        m.subst(&self.dim_subst)
    }

    // ----- instantiation -----------------------------------------------

    /// Replace one prefix variable by another throughout a type.
    pub(crate) fn map_prefix(t: &Ty, from: PrefixVar, to: PrefixVar) -> Ty {
        fn shape(s: &Shape, from: PrefixVar, to: PrefixVar) -> Shape {
            Shape { prefix: if s.prefix == Some(from) { Some(to) } else { s.prefix }, axes: s.axes.clone() }
        }
        match t {
            Ty::Real(s) => Ty::Real(shape(s, from, to)),
            Ty::Idx { bound, shape: s } => Ty::Idx { bound: bound.clone(), shape: shape(s, from, to) },
            Ty::Truth(s) => Ty::Truth(shape(s, from, to)),
            Ty::Tuple(ts) => Ty::Tuple(ts.iter().map(|t| Self::map_prefix(t, from, to)).collect()),
            Ty::Fn { params, ret } => Ty::Fn { params: params.iter().map(|t| Self::map_prefix(t, from, to)).collect(), ret: Box::new(Self::map_prefix(ret, from, to)) },
            t => t.clone(),
        }
    }

    /// Fresh inference dimensions for a set of rigid ones.
    pub(crate) fn freshen(&mut self, vars: impl IntoIterator<Item = DimVar>, map: &mut BTreeMap<DimVar, DimExpr>) {
        for v in vars {
            map.entry(v).or_insert_with(|| DimExpr::var(self.fresh_dim()));
        }
    }

    // ----- unification ---------------------------------------------------

    /// Unify with diagnostics; returns whether it succeeded.
    pub(crate) fn unify(&mut self, expected: &Ty, actual: &Ty, span: Span) -> bool {
        let (e, a) = (self.zonk(expected), self.zonk(actual));
        match (&e, &a) {
            (Ty::Infer(v), _) => {
                if a != Ty::Infer(*v) {
                    self.ty_subst.insert(*v, a.clone());
                }
                true
            }
            (_, Ty::Infer(v)) => {
                self.ty_subst.insert(*v, e.clone());
                true
            }
            (Ty::Real(s1), Ty::Real(s2)) => self.unify_shape(s1, s2, span),
            (Ty::Idx { bound: b1, shape: s1 }, Ty::Idx { bound: b2, shape: s2 }) => self.unify_dim(b1, b2, span) & self.unify_shape(s1, s2, span),
            (Ty::Truth(s1), Ty::Truth(s2)) => self.unify_shape(s1, s2, span),
            (Ty::Nat, Ty::Nat) | (Ty::Int, Ty::Int) | (Ty::Rat, Ty::Rat) | (Ty::F64, Ty::F64) | (Ty::Bool, Ty::Bool) | (Ty::Key, Ty::Key) => true,
            (Ty::Tuple(xs), Ty::Tuple(ys)) if xs.len() == ys.len() => xs.iter().zip(ys).fold(true, |ok, (x, y)| self.unify(x, y, span) && ok),
            (Ty::Params(m1), Ty::Params(m2)) | (Ty::Module(m1), Ty::Module(m2)) => self.unify_modref(m1, m2, span),
            (Ty::Fn { params: p1, ret: r1 }, Ty::Fn { params: p2, ret: r2 }) if p1.len() == p2.len() => {
                let ok = p1.iter().zip(p2).fold(true, |ok, (x, y)| self.unify(x, y, span) && ok);
                self.unify(r1, r2, span) && ok
            }
            _ => {
                let (es, as_) = (self.show(&e), self.show(&a));
                self.error(Diagnostic::error("E2002", format!("type mismatch: expected `{es}`, found `{as_}`")).label(span, format!("this has type `{as_}`")));
                false
            }
        }
    }

    pub(crate) fn unify_modref(&mut self, a: &ModRef, b: &ModRef, span: Span) -> bool {
        match (a, b) {
            (ModRef::Inst { def: d1, args: a1 }, ModRef::Inst { def: d2, args: a2 }) => {
                if d1 != d2 {
                    let n = self.prog.names();
                    let (n1, n2) = (a.display(&n), b.display(&n));
                    self.error(Diagnostic::error("E2002", format!("module mismatch: expected `{n1}`, found `{n2}`")).label(span, format!("this is `{n2}`")));
                    return false;
                }
                a1.iter().zip(a2).fold(true, |ok, (p, q)| self.unify_dim(p, q, span) && ok)
            }
            (ModRef::Stack { n: n1, inner: i1 }, ModRef::Stack { n: n2, inner: i2 }) => self.unify_dim(n1, n2, span) & self.unify_modref(i1, i2, span),
            (ModRef::Compose(xs), ModRef::Compose(ys)) if xs.len() == ys.len() => xs.iter().zip(ys).fold(true, |ok, (x, y)| self.unify_modref(x, y, span) && ok),
            _ => {
                let n = self.prog.names();
                let (s1, s2) = (a.display(&n), b.display(&n));
                self.error(Diagnostic::error("E2002", format!("module mismatch: expected `{s1}`, found `{s2}`")).label(span, format!("this is `{s2}`")));
                false
            }
        }
    }

    /// Two dimension expressions must be equal: solve for an inference
    /// variable if there is one, otherwise ask the decision procedure.
    pub(crate) fn unify_dim(&mut self, a: &DimExpr, b: &DimExpr, span: Span) -> bool {
        let (ra, rb) = (self.resolve_dim(a), self.resolve_dim(b));
        let d = ra.sub(&rb);
        if d.is_zero() {
            return true;
        }
        if let Some((v, c)) = d.coeffs.iter().find(|(v, _)| self.infer_dims.contains(v)).map(|(v, c)| (*v, c.clone())) {
            // v = -(d - c·v) / c
            let mut rest = d.clone();
            rest.coeffs.remove(&v);
            let sol = rest.scale(&(-c.recip()));
            self.dim_subst.insert(v, sol);
            return true;
        }
        let goal = Constraint::Eq(ra.clone(), rb.clone());
        match entails(&self.prog.facts, &goal) {
            Validity::Valid => true,
            other => {
                let (sa, sb) = (self.show_dim(&ra), self.show_dim(&rb));
                let why = match other {
                    Validity::Invalid(w) => format!("differ for {}", display_assignment(&w, &goal, &|v| self.prog.dim_name(v))),
                    _ => "cannot be proved equal".into(),
                };
                self.error(Diagnostic::error("E2006", format!("dimension mismatch: `{sa}` vs `{sb}`")).label(span, why));
                false
            }
        }
    }

    pub(crate) fn unify_axis(&mut self, x: &Axis, y: &Axis, span: Span) -> bool {
        if let (Some(n1), Some(n2)) = (&x.name, &y.name)
            && n1 != n2
        {
            self.error(
                Diagnostic::error("E2003", format!("axis name mismatch: expected `{n1}`, found `{n2}`"))
                    .label(span, format!("axis `{n2}` here, where `{n1}` is expected"))
                    .note("two axes of the same length but different names are different axes"),
            );
            return false;
        }
        self.unify_dim(&x.len, &y.len, span)
    }

    /// Unify two shapes, binding flexible prefixes.
    pub(crate) fn unify_shape(&mut self, expected: &Shape, actual: &Shape, span: Span) -> bool {
        let (e, a) = (self.resolve_shape(expected), self.resolve_shape(actual));
        match (e.prefix, a.prefix) {
            (None, None) => {
                if e.axes.len() != a.axes.len() {
                    return self.rank_error(&e, &a, span);
                }
                self.unify_axes(&e.axes, &a.axes, span)
            }
            (Some(p), None) => self.bind_prefix(p, &e.axes, None, &a.axes, span, &e, &a),
            (None, Some(q)) => self.bind_prefix(q, &a.axes, None, &e.axes, span, &e, &a),
            (Some(p), Some(q)) => {
                if p == q {
                    if e.axes.len() != a.axes.len() {
                        return self.rank_error(&e, &a, span);
                    }
                    return self.unify_axes(&e.axes, &a.axes, span);
                }
                if self.flexible.contains(&p) && a.axes.len() >= e.axes.len() {
                    self.bind_prefix(p, &e.axes, Some(q), &a.axes, span, &e, &a)
                } else if self.flexible.contains(&q) && e.axes.len() >= a.axes.len() {
                    self.bind_prefix(q, &a.axes, Some(p), &e.axes, span, &e, &a)
                } else {
                    self.rank_error(&e, &a, span)
                }
            }
        }
    }

    /// Bind prefix `p` (followed by `tail`) against `other_prefix ++ other`.
    #[allow(clippy::too_many_arguments)]
    fn bind_prefix(&mut self, p: PrefixVar, tail: &[Axis], other_prefix: Option<PrefixVar>, other: &[Axis], span: Span, e: &Shape, a: &Shape) -> bool {
        if !self.flexible.contains(&p) || other.len() < tail.len() {
            return self.rank_error(e, a, span);
        }
        let k = other.len() - tail.len();
        self.prefix_subst.insert(p, Shape { prefix: other_prefix, axes: other[..k].to_vec() });
        self.unify_axes(tail, &other[k..], span)
    }

    fn unify_axes(&mut self, xs: &[Axis], ys: &[Axis], span: Span) -> bool {
        xs.iter().zip(ys).fold(true, |ok, (x, y)| self.unify_axis(x, y, span) && ok)
    }

    fn rank_error(&mut self, e: &Shape, a: &Shape, span: Span) -> bool {
        let n = self.prog.names();
        let (se, sa) = (e.display(&n), a.display(&n));
        self.error(Diagnostic::error("E2003", format!("shape mismatch: expected `ℝ{se}`, found `ℝ{sa}`")).label(span, format!("this has shape `{sa}`; {} expected", if e.prefix.is_some() { "more axes".to_string() } else { format!("{} ax{}", e.axes.len(), if e.axes.len() == 1 { "is" } else { "es" }) })));
        false
    }

    // ----- trial unification --------------------------------------------

    /// Run `f` without diagnostics; on `false`, undo its bindings.
    pub(crate) fn trial(&mut self, f: impl FnOnce(&mut Self) -> bool) -> bool {
        let saved = (self.dim_subst.clone(), self.prefix_subst.clone(), self.ty_subst.clone());
        self.quiet += 1;
        let ok = f(self);
        self.quiet -= 1;
        if !ok {
            (self.dim_subst, self.prefix_subst, self.ty_subst) = saved;
        }
        ok
    }

    // ----- broadcasting --------------------------------------------------

    /// The shape of an elementwise operation between two real tensors:
    /// positional alignment from the trailing axis, else alignment by name.
    pub(crate) fn broadcast(&mut self, x: &Shape, y: &Shape, span: Span) -> Option<Shape> {
        let (a, b) = (self.resolve_shape(x), self.resolve_shape(y));
        if a.prefix.is_none() && a.axes.is_empty() {
            return Some(b);
        }
        if b.prefix.is_none() && b.axes.is_empty() {
            return Some(a);
        }
        // Which operand is "longer" (the result)?
        let (long, short) = match (a.prefix, b.prefix) {
            (Some(p), Some(q)) if p == q => {
                if a.axes.len() >= b.axes.len() { (a.clone(), b.clone()) } else { (b.clone(), a.clone()) }
            }
            (Some(_), Some(_)) => {
                return self.broadcast_error(&a, &b, span, "two different open prefixes");
            }
            (Some(_), None) => {
                if b.axes.len() > a.axes.len() {
                    return self.broadcast_error(&a, &b, span, "the closed shape has more axes than the open one's tail");
                }
                (a.clone(), b.clone())
            }
            (None, Some(_)) => {
                if a.axes.len() > b.axes.len() {
                    return self.broadcast_error(&a, &b, span, "the closed shape has more axes than the open one's tail");
                }
                (b.clone(), a.clone())
            }
            (None, None) => {
                if a.axes.len() >= b.axes.len() { (a.clone(), b.clone()) } else { (b.clone(), a.clone()) }
            }
        };
        // Trailing alignment.
        let k = short.axes.len();
        let off = long.axes.len() - k;
        let trailing = self.trial(|c| c.unify_axes(&long.axes[off..], &short.axes, span));
        if trailing {
            let mut axes = long.axes.clone();
            for (i, s) in short.axes.iter().enumerate() {
                if axes[off + i].name.is_none() {
                    axes[off + i].name = s.name.clone();
                }
            }
            return Some(Shape { prefix: long.prefix, axes });
        }
        // Alignment by name: every axis of the short operand named and present in the long one.
        if short.axes.iter().all(|s| s.name.is_some()) {
            let mut ok = true;
            for s in &short.axes {
                match long.axes.iter().find(|l| l.name == s.name).cloned() {
                    Some(l) => ok &= self.unify_dim(&l.len, &s.len, span),
                    None => ok = false,
                }
            }
            if ok {
                return Some(long);
            }
        }
        self.broadcast_error(&a, &b, span, "no alignment by position or by name")
    }

    fn broadcast_error(&mut self, a: &Shape, b: &Shape, span: Span, why: &str) -> Option<Shape> {
        let n = self.prog.names();
        let (sa, sb) = (a.display(&n), b.display(&n));
        self.error(
            Diagnostic::error("E2003", format!("cannot broadcast `ℝ{sa}` with `ℝ{sb}`"))
                .label(span, why)
                .note("axes align from the last one, or by name when every axis of the smaller operand is named"),
        );
        None
    }
}
