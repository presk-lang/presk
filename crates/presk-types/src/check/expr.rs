//! Typing of expressions.

use std::collections::BTreeMap;

use presk_syntax::ast::{Arg, BinOp, Expr, IndexSel, Lit, Pat, Path, UnOp};
use presk_syntax::{Diagnostic, Span};

use super::{Binding, Callee, Checker, EinInfo, Env};
use crate::dim::{DimExpr, DimVar};
use crate::elab::{FnId, FnSig};
use crate::ty::{Axis, ModRef, Shape, Ty};

/// A typed call argument.
#[derive(Clone, Debug)]
pub(crate) enum TArg {
    Pos(Ty, Span),
    Named(String, Ty, Span),
    Axis(String, Span),
    /// `name: dimexpr`
    NamedAxis(String, DimExpr, Span),
}

impl TArg {
    pub(crate) fn span(&self) -> Span {
        match self {
            TArg::Pos(_, s) | TArg::Named(_, _, s) | TArg::Axis(_, s) | TArg::NamedAxis(_, _, s) => *s,
        }
    }
}

impl Checker<'_> {
    pub(crate) fn expr(&mut self, e: &Expr, env: &mut Env) -> Option<Ty> {
        let t = self.expr_inner(e, env)?;
        self.record(e.span(), &t);
        Some(t)
    }

    fn expr_inner(&mut self, e: &Expr, env: &mut Env) -> Option<Ty> {
        match e {
            Expr::Lit { lit, span } => match lit {
                Lit::Nat(_) => Some(Ty::Nat),
                Lit::Float(_) => Some(Ty::F64),
                Lit::Str(_) | Lit::Quantity { .. } => {
                    self.error(Diagnostic::error("E2015", "strings and quantities are only allowed in configuration blocks").label(*span, "not a value of the core language"));
                    None
                }
            },
            Expr::Var { path } => self.var(path, env),
            Expr::Let { pat, init, body, .. } => {
                let t = self.expr(init, env)?;
                env.push();
                self.bind_pat(pat, &t, env);
                let r = self.expr(body, env);
                env.pop();
                r
            }
            Expr::Lambda { pat, body, .. } => {
                let names = pat_names(pat);
                let mut params = Vec::new();
                env.push();
                for n in names {
                    let t = self.fresh_ty();
                    env.bind(n, Binding::Val(t.clone()));
                    params.push(t);
                }
                let ret = self.expr(body, env);
                env.pop();
                Some(Ty::Fn { params, ret: Box::new(ret?) })
            }
            Expr::Call { callee, args, span } => self.call(callee, args, *span, env),
            Expr::Field { base, field, span } => {
                self.error(Diagnostic::error("E2005", format!("field access `.{}` on an expression", field.name)).label(*span, "only module members can be accessed with `.`"));
                let _ = base;
                None
            }
            Expr::Index { base, sels, span } => self.index(base, sels, *span, env),
            Expr::Inst { span, .. } => {
                self.error(Diagnostic::error("E2005", "a module instance is not a value").label(*span, "module instances appear in types and `use` declarations only"));
                None
            }
            Expr::Ein { axes, body, span } => self.ein(axes, body, *span, env),
            Expr::Binary { op, lhs, rhs, span } => self.binary(*op, lhs, rhs, *span, env),
            Expr::Unary { op: UnOp::Neg, expr, span } => {
                let t = self.expr(expr, env)?;
                match t {
                    Ty::Real(_) | Ty::Nat | Ty::Int | Ty::Rat | Ty::F64 => Some(if t == Ty::Nat { Ty::Int } else { t }),
                    _ => {
                        let s = self.show(&t);
                        self.error(Diagnostic::error("E2002", format!("cannot negate a value of type `{s}`")).label(*span, "expected a real tensor or a number"));
                        None
                    }
                }
            }
            Expr::Ascribe { expr, ty, span } => {
                let t = self.expr(expr, env)?;
                // Only `ℝ` (the scalar) and exact tensor ascriptions are needed in 0.1.0.
                match ty {
                    presk_syntax::ast::Ty::Real { axes, .. } if axes.is_empty() => match t {
                        Ty::Nat | Ty::Int | Ty::Rat | Ty::F64 => Some(Ty::scalar_real()),
                        Ty::Real(ref s) if self.resolve_shape(s).rank() == Some(0) => Some(t),
                        _ => {
                            let s = self.show(&t);
                            self.error(Diagnostic::error("E2002", format!("`{s}` cannot be injected into `ℝ`")).label(*span, "only hyperparameters (`nat`, `f64`, …) inject into the real scalar"));
                            None
                        }
                    },
                    _ => {
                        self.error(Diagnostic::error("E2012", "only ascription to `ℝ` is supported").label(ty.span(), "not implemented in 0.1.0"));
                        None
                    }
                }
            }
            Expr::Tuple { elems, .. } => {
                let mut ts = Vec::new();
                for e in elems {
                    ts.push(self.expr(e, env)?);
                }
                Some(Ty::Tuple(ts))
            }
        }
    }

    fn bind_pat(&mut self, pat: &Pat, t: &Ty, env: &mut Env) {
        match pat {
            Pat::Ident(id) => {
                if let Some(fid) = self.fn_id {
                    self.typed.lets.push((fid, id.name.clone(), t.clone()));
                }
                env.bind(id.name.clone(), Binding::Val(t.clone()));
            }
            Pat::Tuple { elems, span } => {
                let z = self.zonk(t);
                let parts: Vec<Ty> = match z {
                    Ty::Tuple(ts) if ts.len() == elems.len() => ts,
                    Ty::Infer(_) => {
                        let fresh: Vec<Ty> = elems.iter().map(|_| self.fresh_ty()).collect();
                        self.unify(&Ty::Tuple(fresh.clone()), t, *span);
                        fresh
                    }
                    other => {
                        let s = self.show(&other);
                        self.error(Diagnostic::error("E2002", format!("tuple pattern of {} names against a value of type `{s}`", elems.len())).label(*span, "not a tuple of that size"));
                        return;
                    }
                };
                for (p, t) in elems.iter().zip(parts) {
                    self.bind_pat(p, &t, env);
                }
            }
        }
    }

    // ----- names ---------------------------------------------------------

    fn var(&mut self, path: &Path, env: &mut Env) -> Option<Ty> {
        if path.segs.len() == 1 {
            let name = &path.segs[0].name;
            match env.lookup(name) {
                Some(Binding::Val(t)) => return Some(t.clone()),
                Some(Binding::Module { r, .. }) => return Some(Ty::Module(r.clone())),
                Some(Binding::Const(_)) => return Some(Ty::Nat),
                None => {}
            }
        }
        if let Some(n) = self.lookup_const(path) {
            let _ = n;
            return Some(Ty::Nat);
        }
        if path.segs.len() == 1 && self.prog.units[self.unit].fns.contains_key(&path.segs[0].name) {
            self.error(Diagnostic::error("E2012", format!("function `{}` used as a value", path.dotted())).label(path.span, "functions are called, or passed through `grad(λ …)`; first-class functions are not implemented"));
            return None;
        }
        self.error(Diagnostic::error("E2001", format!("unresolved name `{}`", path.dotted())).label(path.span, "not a variable, parameter, module member or constant in scope"));
        None
    }

    /// `V`, `m.V`
    pub(crate) fn lookup_const(&self, path: &Path) -> Option<u64> {
        let unit = if path.segs.len() == 2 { *self.prog.units[self.unit].imports.get(&path.segs[0].name)? } else if path.segs.len() == 1 { self.unit } else { return None };
        self.prog.units[unit].consts.get(&path.segs.last()?.name).map(|c| c.value)
    }

    /// A name in dimension position inside a body (`positions(seq: T)`).
    pub(crate) fn dim_of_expr(&mut self, e: &Expr, env: &Env) -> Option<DimExpr> {
        match e {
            Expr::Lit { lit: Lit::Nat(n), .. } => n.parse().ok().map(DimExpr::nat),
            Expr::Var { path } => {
                if let Some(n) = self.lookup_const(path) {
                    return Some(DimExpr::nat(n));
                }
                if path.segs.len() == 1
                    && let Some(Binding::Const(n)) = env.lookup(&path.segs[0].name)
                {
                    return Some(DimExpr::nat(*n));
                }
                self.lookup_dim_var(path).map(DimExpr::var)
            }
            Expr::Binary { op, lhs, rhs, .. } => {
                let (l, r) = (self.dim_of_expr(lhs, env)?, self.dim_of_expr(rhs, env)?);
                match op {
                    BinOp::Add => Some(l.add(&r)),
                    BinOp::Sub => Some(l.sub(&r)),
                    BinOp::Mul => l.mul(&r),
                    BinOp::Div => l.div(&r),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Dimensions visible in the current function: its type parameters,
    /// implicit dimensions, the module's type parameters and the file's `dim`s.
    fn lookup_dim_var(&self, path: &Path) -> Option<DimVar> {
        let f = self.fn_id.map(|id| self.prog.function(id))?;
        let name = &path.segs.last()?.name;
        let unit = if path.segs.len() == 2 { *self.prog.units[self.unit].imports.get(&path.segs[0].name)? } else { self.unit };
        if path.segs.len() == 1 {
            for v in f.typarams.iter().chain(&f.implicit_dims) {
                if &self.prog.dims[v.0 as usize].name == name {
                    return Some(*v);
                }
            }
            if let Some(m) = f.owner {
                for v in &self.prog.module(m).typarams {
                    if &self.prog.dims[v.0 as usize].name == name {
                        return Some(*v);
                    }
                }
            }
        }
        self.prog.units[unit].dims.get(name).copied()
    }

    // ----- calls ---------------------------------------------------------

    fn args(&mut self, args: &[Arg], env: &mut Env) -> Option<Vec<TArg>> {
        let mut out = Vec::new();
        for a in args {
            out.push(match a {
                Arg::Pos(e) => TArg::Pos(self.expr(e, env)?, e.span()),
                Arg::Named { name, value } => TArg::Named(name.name.clone(), self.expr(value, env)?, name.span.to(value.span())),
                Arg::Axis(id) => TArg::Axis(id.name.clone(), id.span),
                Arg::NamedAxis { name, value } => {
                    let Some(d) = self.dim_of_expr(value, env) else {
                        self.error(Diagnostic::error("E2001", "not a dimension").label(value.span(), "expected a constant or a dimension in scope"));
                        return None;
                    };
                    TArg::NamedAxis(name.name.clone(), d, name.span.to(value.span()))
                }
            });
        }
        Some(out)
    }

    fn call(&mut self, callee: &Expr, args: &[Arg], span: Span, env: &mut Env) -> Option<Ty> {
        // A path callee: user function, module member, builtin.
        if let Expr::Var { path } = callee {
            if let Some((r, explicit)) = self.module_of_path(path, env) {
                let fname = &path.segs.last().unwrap().name;
                let targs = self.args(args, env)?;
                return self.call_member(&r, explicit, fname, path.span, targs, span);
            }
            if path.segs.len() == 1 {
                let name = path.segs[0].name.as_str();
                if !matches!(env.lookup(name), Some(Binding::Val(_))) {
                    if let Some(&fid) = self.prog.units[self.unit].fns.get(name) {
                        let targs = self.args(args, env)?;
                        self.typed.callees.insert(span, Callee::Fn { id: fid, module: None, explicit_theta: false });
                        self.calls.extend(self.fn_id.map(|caller| (caller, fid, span)));
                        return self.call_fn(fid, None, targs, span);
                    }
                    let targs = self.args(args, env)?;
                    if let Some(r) = self.builtin(name, &targs, span, env) {
                        self.typed.callees.insert(span, Callee::Builtin(name.to_string()));
                        return r;
                    }
                    if !self.is_builtin(name) {
                        self.error(Diagnostic::error("E2001", format!("unresolved function `{name}`")).label(path.span, "not a function, a module member or a builtin"));
                        return None;
                    }
                }
            }
        }
        // A value of function type.
        let ft = self.expr(callee, env)?;
        let targs = self.args(args, env)?;
        let z = self.zonk(&ft);
        match z {
            Ty::Fn { params, ret } => {
                if params.len() != targs.len() {
                    self.error(Diagnostic::error("E2004", format!("this function takes {} argument{}, {} given", params.len(), if params.len() == 1 { "" } else { "s" }, targs.len())).label(span, "wrong number of arguments"));
                    return None;
                }
                for (p, a) in params.iter().zip(&targs) {
                    match a {
                        TArg::Pos(t, s) => {
                            self.unify(p, t, *s);
                        }
                        other => {
                            self.error(Diagnostic::error("E2004", "only positional arguments for a function value").label(other.span(), "named or axis argument"));
                        }
                    }
                }
                self.typed.callees.insert(span, Callee::Value);
                Some(*ret)
            }
            other => {
                let s = self.show(&other);
                self.error(Diagnostic::error("E2005", format!("`{s}` is not callable")).label(callee.span(), "expected a function"));
                None
            }
        }
    }

    /// `l1.apply`, `net.apply`, `m.MiniGPT.apply`, `MiniGPT.apply`: the module
    /// and whether θ is passed explicitly.
    fn module_of_path(&mut self, path: &Path, env: &Env) -> Option<(ModRef, bool)> {
        if path.segs.len() < 2 {
            return None;
        }
        let head = &path.segs[0].name;
        if path.segs.len() == 2
            && let Some(Binding::Module { r, explicit_theta }) = env.lookup(head)
        {
            return Some((r.clone(), *explicit_theta));
        }
        // `[alias.]Module.fn`
        let (unit, mname) = if path.segs.len() == 3 { (*self.prog.units[self.unit].imports.get(head)?, &path.segs[1].name) } else { (self.unit, head) };
        let mid = *self.prog.units[unit].modules.get(mname)?;
        let def = self.prog.module(mid);
        if !def.typarams.is_empty() {
            self.error(Diagnostic::error("E1003", format!("`{}` needs {} dimension arguments to be called", def.name, def.typarams.len())).label(path.span, "instantiate the module in a `use` or a parameter type first"));
            return None;
        }
        Some((ModRef::Inst { def: mid, args: vec![] }, true))
    }

    fn call_member(&mut self, r: &ModRef, explicit: bool, fname: &str, path_span: Span, mut targs: Vec<TArg>, span: Span) -> Option<Ty> {
        let Some((mid, margs)) = self.inst_of(r) else {
            self.error(Diagnostic::error("E2005", "cannot call a member of a composed model").label(path_span, "`model A ∘ B` composition is not callable in 0.1.0"));
            return None;
        };
        let def = self.prog.module(mid);
        let Some(&fid) = def.fns.iter().find(|f| self.prog.function(**f).name == fname) else {
            self.error(Diagnostic::error("E2005", format!("`{}` has no function `{fname}`", def.name)).label(path_span, "not a member of this module"));
            return None;
        };
        self.typed.callees.insert(span, Callee::Fn { id: fid, module: Some(r.clone()), explicit_theta: explicit });
        self.calls.extend(self.fn_id.map(|caller| (caller, fid, span)));
        if explicit {
            if targs.is_empty() {
                self.error(Diagnostic::error("E2004", format!("`{}.{fname}` called outside its module needs the parameter tree as first argument", def.name)).label(span, "missing `θ`"));
                return None;
            }
            let theta = targs.remove(0);
            let TArg::Pos(t, s) = theta else {
                self.error(Diagnostic::error("E2004", "the parameter tree must be a positional first argument").label(theta.span(), "expected `θ`"));
                return None;
            };
            self.unify(&Ty::Params(r.clone()), &t, s);
        }
        // Through `stack{L}(M)`, the function is applied L times: result must feed back.
        let ret = self.call_fn(fid, Some((mid, margs)), targs.clone(), span)?;
        if let ModRef::Stack { .. } = r
            && let Some(TArg::Pos(first, s)) = targs.first()
        {
            let first = first.clone();
            if !self.unify(&first, &ret, *s) {
                self.error(Diagnostic::error("E2002", format!("`{fname}` through `stack` must return the type of its first argument")).label(span, "so that the stacked applications chain"));
            }
        }
        Some(ret)
    }

    /// Instantiate a signature and check the arguments.
    fn call_fn(&mut self, fid: FnId, inst: Option<(crate::ty::ModuleId, Vec<DimExpr>)>, targs: Vec<TArg>, span: Span) -> Option<Ty> {
        let f: &FnSig = self.prog.function(fid);
        let f = f.clone();
        let mut map: BTreeMap<DimVar, DimExpr> = BTreeMap::new();
        if let Some((def, args)) = inst {
            map = self.prog.inst_subst(def, &args);
        }
        let unit_dims: Vec<DimVar> = self.prog.units[f.unit].dims.values().copied().collect();
        self.freshen(f.typarams.iter().copied().chain(f.implicit_dims.iter().copied()).chain(unit_dims), &mut map);
        self.typed.call_subst.insert(span, map.clone());
        let fresh_prefix = f.prefix.map(|_| self.fresh_prefix());
        let inst_ty = |_c: &Self, t: &Ty| -> Ty {
            let t = t.subst(&map);
            match (f.prefix, fresh_prefix) {
                (Some(p), Some(q)) => Self::map_prefix(&t, p, q),
                _ => t,
            }
        };
        let params: Vec<(String, Option<Ty>, bool, bool)> = f.params.iter().map(|p| (p.name.clone(), p.ty.as_ref().map(|t| inst_ty(self, t)), p.default.is_some(), p.axis)).collect();
        let ret = f.ret.as_ref().map(|t| inst_ty(self, t));
        // Match arguments to parameters.
        let mut bound: Vec<Option<TArg>> = vec![None; params.len()];
        let mut pos = 0;
        for a in targs {
            match &a {
                TArg::Named(n, _, s) | TArg::NamedAxis(n, _, s) => match params.iter().position(|p| &p.0 == n) {
                    Some(i) => bound[i] = Some(a.clone()),
                    None => {
                        self.error(Diagnostic::error("E2004", format!("`{}` has no parameter `{n}`", f.name)).label(*s, "unknown named argument"));
                        return None;
                    }
                },
                _ => {
                    while pos < params.len() && bound[pos].is_some() {
                        pos += 1;
                    }
                    if pos >= params.len() {
                        self.error(Diagnostic::error("E2004", format!("`{}` takes {} argument{}, more given", f.name, params.len(), if params.len() == 1 { "" } else { "s" })).label(a.span(), "extra argument"));
                        return None;
                    }
                    bound[pos] = Some(a.clone());
                    pos += 1;
                }
            }
        }
        for (i, (name, pty, has_default, _)) in params.iter().enumerate() {
            match (&bound[i], pty) {
                (Some(TArg::Pos(t, s)) | Some(TArg::Named(_, t, s)), Some(p)) => {
                    self.unify(p, t, *s);
                }
                (Some(TArg::Pos(_, _)) | Some(TArg::Named(..)), None) => {}
                (Some(other), _) => {
                    self.error(Diagnostic::error("E2004", format!("parameter `{name}` takes a value")).label(other.span(), "not an axis argument"));
                }
                (None, _) if *has_default => {}
                (None, _) => {
                    self.error(Diagnostic::error("E2004", format!("missing argument `{name}` in call to `{}`", f.name)).label(span, format!("`{name}` is not given and has no default")));
                    return None;
                }
            }
        }
        match ret {
            Some(r) => Some(r),
            None => {
                // Untyped return: the body's inferred type, if this function was already checked.
                match self.typed.body_ty.get(&fid).cloned() {
                    Some(t) => Some(inst_ty(self, &t)),
                    None => {
                        self.error(Diagnostic::error("E2012", format!("`{}` has no return type and is called before being checked", f.name)).label(span, "declare its return type"));
                        None
                    }
                }
            }
        }
    }

    // ----- indexing, ein, operators ---------------------------------------

    fn index(&mut self, base: &Expr, sels: &[IndexSel], span: Span, env: &mut Env) -> Option<Ty> {
        let t = self.expr(base, env)?;
        let z = self.zonk(&t);
        let (mut shape, rebuild): (Shape, Box<dyn Fn(Shape) -> Ty>) = match z {
            Ty::Real(s) => (s, Box::new(Ty::Real)),
            Ty::Idx { bound, shape } => (shape, Box::new(move |s| Ty::Idx { bound: bound.clone(), shape: s })),
            Ty::Truth(s) => (s, Box::new(Ty::Truth)),
            other => {
                let s = self.show(&other);
                self.error(Diagnostic::error("E2002", format!("cannot index a value of type `{s}`")).label(span, "expected a tensor"));
                return None;
            }
        };
        for sel in sels {
            match sel {
                IndexSel::Named { axis, value } => {
                    let vt = self.expr(value, env)?;
                    if !matches!(self.zonk(&vt), Ty::Nat | Ty::Int) {
                        self.error(Diagnostic::error("E2002", "an index must be an integer literal").label(value.span(), "only constant indices are supported in 0.1.0"));
                        return None;
                    }
                    let Some(pos) = shape.axes.iter().position(|a| a.name.as_deref() == Some(&axis.name)) else {
                        return self.no_axis(&axis.name, &shape, axis.span);
                    };
                    shape.axes.remove(pos);
                }
                IndexSel::Pos(value) => {
                    let vt = self.expr(value, env)?;
                    if !matches!(self.zonk(&vt), Ty::Nat | Ty::Int) {
                        self.error(Diagnostic::error("E2002", "an index must be an integer literal").label(value.span(), "only constant indices are supported in 0.1.0"));
                        return None;
                    }
                    if shape.prefix.is_some() || shape.axes.is_empty() {
                        self.error(Diagnostic::error("E2008", "positional index on a shape without a known first axis").label(value.span(), "use a named selector `[axis: i]`"));
                        return None;
                    }
                    shape.axes.remove(0);
                }
            }
        }
        Some(rebuild(shape))
    }

    pub(crate) fn no_axis(&mut self, name: &str, shape: &Shape, span: Span) -> Option<Ty> {
        let n = self.prog.names();
        let s = shape.display(&n);
        self.error(Diagnostic::error("E2008", format!("no axis named `{name}`")).label(span, format!("the tensor has shape `{s}`")).help("axes are selected by the names given in the type; an unnamed axis cannot be selected"));
        None
    }

    /// The tensor operands of an `ein` body: a product `a * b * …`.
    fn ein_operands<'e>(e: &'e Expr, out: &mut Vec<&'e Expr>) {
        match e {
            Expr::Binary { op: BinOp::Mul, lhs, rhs, .. } => {
                Self::ein_operands(lhs, out);
                Self::ein_operands(rhs, out);
            }
            e => out.push(e),
        }
    }

    fn ein(&mut self, axes: &[presk_syntax::ast::Ident], body: &Expr, span: Span, env: &mut Env) -> Option<Ty> {
        let mut ops = Vec::new();
        Self::ein_operands(body, &mut ops);
        let mut operands: Vec<(Span, Vec<Axis>)> = Vec::new();
        for o in &ops {
            let t = self.expr(o, env)?;
            let z = self.zonk(&t);
            let Ty::Real(s) = z else {
                let st = self.show(&t);
                self.error(Diagnostic::error("E2007", format!("`ein` operand of type `{st}`")).label(o.span(), "expected a real tensor"));
                return None;
            };
            if s.prefix.is_some() || s.axes.iter().any(|a| a.name.is_none()) {
                let n = self.prog.names();
                self.error(Diagnostic::error("E2007", "`ein` needs every axis of its operands to be named").label(o.span(), format!("this operand has shape `{}`", s.display(&n))).help("name the axes in the type of this expression"));
                return None;
            }
            operands.push((o.span(), s.axes));
        }
        let out_names: Vec<String> = axes.iter().map(|a| a.name.clone()).collect();
        // Relabeling rule (spec question 4): an axis name shared by two
        // operands, with exactly one dangling axis of the same length
        // (in the output but in no operand, or in one operand but not in the
        // output), is renamed in the right operand.
        if operands.len() == 2 {
            let shared: Vec<String> = operands[0].1.iter().filter_map(|a| a.name.clone()).filter(|n| operands[1].1.iter().any(|b| b.name.as_deref() == Some(n))).collect();
            for n in shared {
                let len = operands[0].1.iter().find(|a| a.name.as_deref() == Some(&n)).unwrap().len.clone();
                let in_ops = |name: &str, ops: &Vec<(Span, Vec<Axis>)>| ops.iter().filter(|(_, ax)| ax.iter().any(|a| a.name.as_deref() == Some(name))).count();
                // Candidates are the copies of `n`: names `n<digits>` (`seq` → `seq2`)
                // that are either in the output and in no operand, or in one
                // operand only and not in the output; a known length must match.
                let is_copy = |cand: &str| cand.len() > n.len() && cand.starts_with(n.as_str()) && cand[n.len()..].chars().all(|c| c.is_ascii_digit());
                let mut dangling: Vec<(String, Option<DimExpr>)> = Vec::new();
                for o in &out_names {
                    if in_ops(o, &operands) == 0 && is_copy(o) {
                        dangling.push((o.clone(), None));
                    }
                }
                for (_, ax) in &operands {
                    for a in ax {
                        let an = a.name.clone().unwrap();
                        if an != n && in_ops(&an, &operands) == 1 && !out_names.contains(&an) && is_copy(&an) {
                            dangling.push((an, Some(a.len.clone())));
                        }
                    }
                }
                let dangling: Vec<String> = dangling.into_iter().filter(|(_, l)| l.as_ref().is_none_or(|l| self.trial(|c| c.unify_dim(&len, l, span)))).map(|(n, _)| n).collect();
                if dangling.len() == 1 {
                    let new = dangling[0].clone();
                    if let Some(a) = operands[1].1.iter_mut().find(|a| a.name.as_deref() == Some(&n)) {
                        a.name = Some(new);
                    }
                } else if dangling.len() > 1 {
                    self.error(
                        Diagnostic::error("E2007", format!("ambiguous `ein`: axis `{n}` appears in both operands and could be renamed to {}", dangling.iter().map(|d| format!("`{d}`")).collect::<Vec<_>>().join(" or ")))
                            .label(span, "rename an axis explicitly in a type")
                            .note("rule: a name shared by both operands is renamed in the right operand when exactly one axis of the same length is left unmatched"),
                    );
                    return None;
                }
            }
        }
        // Check output names and shared lengths; compute the contraction.
        let mut output = Vec::new();
        for o in axes {
            let found: Vec<&Axis> = operands.iter().flat_map(|(_, ax)| ax.iter()).filter(|a| a.name.as_deref() == Some(&o.name)).collect();
            let Some(first) = found.first() else {
                self.error(Diagnostic::error("E2007", format!("output axis `{}` appears in no operand", o.name)).label(o.span, "not an axis of the operands"));
                return None;
            };
            output.push(Axis { name: Some(o.name.clone()), len: first.len.clone() });
        }
        let mut contracted = Vec::new();
        let all_names: Vec<String> = operands.iter().flat_map(|(_, ax)| ax.iter().filter_map(|a| a.name.clone())).collect();
        for name in &all_names {
            let lens: Vec<DimExpr> = operands.iter().flat_map(|(_, ax)| ax.iter()).filter(|a| a.name.as_deref() == Some(name)).map(|a| a.len.clone()).collect();
            for l in &lens[1..] {
                self.unify_dim(&lens[0], l, span);
            }
            if !out_names.contains(name) && !contracted.contains(name) {
                contracted.push(name.clone());
            }
        }
        self.typed.eins.insert(span, EinInfo { operands, output: output.clone(), contracted });
        Some(Ty::Real(Shape::closed(output)))
    }

    fn binary(&mut self, op: BinOp, lhs: &Expr, rhs: &Expr, span: Span, env: &mut Env) -> Option<Ty> {
        let (lt, rt) = (self.expr(lhs, env)?, self.expr(rhs, env)?);
        let (l, r) = (self.zonk(&lt), self.zonk(&rt));
        match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => match (&l, &r) {
                (Ty::Real(a), Ty::Real(b)) => Some(Ty::Real(self.broadcast(a, b, span)?)),
                (Ty::Nat, Ty::Nat) => Some(if op == BinOp::Sub || op == BinOp::Div { Ty::Int } else { Ty::Nat }),
                (Ty::F64, Ty::F64) | (Ty::Int, Ty::Int) | (Ty::Rat, Ty::Rat) => Some(l),
                (Ty::Idx { .. }, _) | (_, Ty::Idx { .. }) => {
                    self.error(
                        Diagnostic::error("E2009", format!("`{}` between an index tensor and a real", op.text()))
                            .label(span, "indices do not mix with reals")
                            .help("inject explicitly with `real(…)` if the numeric value of the index is meant"),
                    );
                    None
                }
                _ => {
                    let (sl, sr) = (self.show(&l), self.show(&r));
                    self.error(Diagnostic::error("E2002", format!("`{}` between `{sl}` and `{sr}`", op.text())).label(span, "operands must both be real tensors or both be numbers of the same type").help("hyperparameters enter the real world through `real(x)` or `(x : ℝ)`"));
                    None
                }
            },
            BinOp::Pow => {
                if !matches!(r, Ty::Nat) {
                    self.error(Diagnostic::error("E2002", "the exponent of `^` must be a natural literal").label(rhs.span(), "expected `nat`"));
                    return None;
                }
                match l {
                    Ty::Real(_) | Ty::F64 | Ty::Nat | Ty::Int | Ty::Rat => Some(l),
                    _ => {
                        let s = self.show(&l);
                        self.error(Diagnostic::error("E2002", format!("cannot raise `{s}` to a power")).label(lhs.span(), "expected a real tensor or a number"));
                        None
                    }
                }
            }
            BinOp::Dot => match (&l, &r) {
                (Ty::Real(a), Ty::Real(b)) => {
                    let (a, b) = (self.resolve_shape(a), self.resolve_shape(b));
                    if a.axes.is_empty() || b.prefix.is_some() || b.axes.is_empty() {
                        let n = self.prog.names();
                        self.error(Diagnostic::error("E2003", "`·` needs a last axis on the left and a known first axis on the right").label(span, format!("`ℝ{}` · `ℝ{}`", a.display(&n), b.display(&n))));
                        return None;
                    }
                    let k = a.axes.last().unwrap().clone();
                    self.unify_dim(&k.len, &b.axes[0].len, span);
                    let mut axes = a.axes[..a.axes.len() - 1].to_vec();
                    axes.extend(b.axes[1..].iter().cloned());
                    Some(Ty::Real(Shape { prefix: a.prefix, axes }))
                }
                _ => {
                    let (sl, sr) = (self.show(&l), self.show(&r));
                    self.error(Diagnostic::error("E2002", format!("`·` between `{sl}` and `{sr}`")).label(span, "both operands must be real tensors"));
                    None
                }
            },
            BinOp::Eq | BinOp::Le | BinOp::Ge | BinOp::Lt | BinOp::Gt => {
                if self.unify(&l, &r, span) { Some(Ty::Bool) } else { None }
            }
            BinOp::Pipe | BinOp::Alt | BinOp::Tensor | BinOp::Compose => {
                self.error(Diagnostic::error("E2015", format!("`{}` is only meaningful in configuration blocks", op.text())).label(span, "not an operator of the core language"));
                None
            }
            BinOp::Rem => {
                self.error(Diagnostic::error("E2015", "`%` is only allowed in `static assert`").label(span, "not an operator of the core language"));
                None
            }
        }
    }
}

pub(crate) fn pat_names(p: &Pat) -> Vec<String> {
    match p {
        Pat::Ident(i) => vec![i.name.clone()],
        Pat::Tuple { elems, .. } => elems.iter().flat_map(pat_names).collect(),
    }
}
