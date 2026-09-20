//! Expressions to ANF.

use presk_syntax::ast::{Arg, BinOp, Expr, IndexSel, Lit, Pat, Path, UnOp};
use presk_syntax::{Diagnostic, Span};
use presk_types::check::Callee;
use presk_types::dim::DimExpr;
use presk_types::elab::FnId;
use presk_types::ty::{ModRef, Ty};

use super::{Bound, Builder, Lowerer};
use crate::ir::{Block, Func, Lit as CLit, Origin, Param, PrimOp, Rhs, VarId};

impl Lowerer<'_> {
    fn ty_of(&self, e: &Expr) -> Ty {
        self.typed.ty_of.get(&e.span()).cloned().unwrap_or(Ty::Nat)
    }

    /// Lower an expression; the result is a variable of the block.
    pub(crate) fn expr(&mut self, e: &Expr, b: &mut Builder) -> Option<VarId> {
        let ty = self.ty_of(e);
        let span = e.span();
        match e {
            Expr::Lit { lit, .. } => match lit {
                Lit::Nat(n) => Some(b.emit(None, Ty::Nat, Rhs::Lit(CLit::Nat(n.parse().unwrap_or(u64::MAX))), span)),
                Lit::Float(f) => Some(b.emit(None, Ty::F64, Rhs::Lit(CLit::Float(f.clone())), span)),
                _ => {
                    self.error(Diagnostic::error("E5001", "not a value of the core language").label(span, "cannot be lowered"));
                    None
                }
            },
            Expr::Var { path } => self.var(path, &ty, b),
            Expr::Let { pat, init, body, .. } => {
                let v = self.expr(init, b)?;
                let mark = b.scope.len();
                self.bind_pat(pat, v, &self.ty_of(init), b);
                let r = self.expr(body, b);
                b.scope.truncate(mark);
                r
            }
            Expr::Lambda { pat, body, .. } => self.lambda(pat, body, &ty, span, b),
            Expr::Call { callee, args, .. } => self.call(callee, args, &ty, span, b),
            Expr::Index { base, sels, .. } => {
                let mut v = self.expr(base, b)?;
                let mut cur_ty = self.ty_of(base);
                for sel in sels {
                    let (axis, index) = match sel {
                        IndexSel::Named { axis, value } => (axis.name.clone(), const_index(value)?),
                        IndexSel::Pos(value) => {
                            let Some(first) = cur_ty.shape().and_then(|s| s.axes.first()).and_then(|a| a.name.clone()) else {
                                self.error(Diagnostic::error("E5001", "positional index on an unnamed axis").label(value.span(), "use `[axis: i]`"));
                                return None;
                            };
                            (first, const_index(value)?)
                        }
                    };
                    cur_ty = remove_axis(&cur_ty, &axis);
                    v = b.emit(None, cur_ty.clone(), Rhs::Prim { op: PrimOp::IndexConst { axis, index }, args: vec![v] }, span);
                }
                Some(v)
            }
            Expr::Ein { body, .. } => {
                let info = self.typed.eins.get(&span).cloned();
                let mut ops = Vec::new();
                collect_product(body, &mut ops);
                let mut args = Vec::new();
                for o in &ops {
                    args.push(self.expr(o, b)?);
                }
                let Some(info) = info else {
                    self.error(Diagnostic::error("E5001", "`ein` without typing information").label(span, "internal: the checker recorded no axes"));
                    return None;
                };
                let operands = info.operands.iter().map(|(_, ax)| ax.iter().map(|a| a.name.clone().unwrap_or_default()).collect()).collect();
                let output = info.output.iter().map(|a| a.name.clone().unwrap_or_default()).collect();
                Some(b.emit(None, ty, Rhs::Prim { op: PrimOp::Ein { operands, output }, args }, span))
            }
            Expr::Binary { op: BinOp::Pow, lhs, rhs, .. } => {
                let Expr::Lit { lit: Lit::Nat(n), .. } = &**rhs else {
                    self.error(Diagnostic::error("E5001", "`^` needs a literal exponent").label(rhs.span(), "not a literal"));
                    return None;
                };
                let l = self.expr(lhs, b)?;
                Some(b.emit(None, ty, Rhs::Prim { op: PrimOp::Pow(n.parse().unwrap_or(1)), args: vec![l] }, span))
            }
            Expr::Binary { op, lhs, rhs, .. } => {
                let (l, r) = (self.expr(lhs, b)?, self.expr(rhs, b)?);
                let prim = match op {
                    BinOp::Add => PrimOp::Add,
                    BinOp::Sub => PrimOp::Sub,
                    BinOp::Mul => PrimOp::Mul,
                    BinOp::Div => PrimOp::Div,
                    BinOp::Dot => PrimOp::Dot,
                    other => {
                        self.error(Diagnostic::error("E5001", format!("`{}` is not lowered", other.text())).label(span, "comparisons and configuration operators have no Core form"));
                        return None;
                    }
                };
                Some(b.emit(None, ty, Rhs::Prim { op: prim, args: vec![l, r] }, span))
            }
            Expr::Unary { op: UnOp::Neg, expr, .. } => {
                let v = self.expr(expr, b)?;
                Some(b.emit(None, ty, Rhs::Prim { op: PrimOp::Neg, args: vec![v] }, span))
            }
            Expr::Ascribe { expr, .. } => {
                let v = self.expr(expr, b)?;
                let inner = self.ty_of(expr);
                if inner.is_tensor() {
                    Some(v)
                } else {
                    Some(b.emit(None, ty, Rhs::Prim { op: PrimOp::Real, args: vec![v] }, span))
                }
            }
            Expr::Tuple { elems, .. } => {
                let mut vs = Vec::new();
                for e in elems {
                    vs.push(self.expr(e, b)?);
                }
                Some(b.emit(None, ty, Rhs::Tuple(vs), span))
            }
            Expr::Field { .. } | Expr::Inst { .. } => {
                self.error(Diagnostic::error("E5001", "not a value of the core language").label(span, "cannot be lowered"));
                None
            }
        }
    }

    fn bind_pat(&mut self, pat: &Pat, v: VarId, ty: &Ty, b: &mut Builder) {
        match pat {
            Pat::Ident(id) => {
                // Give the variable its source name (for readable dumps).
                if let Some(l) = b.lets.iter_mut().rev().find(|l| l.var == v)
                    && l.name.is_none()
                {
                    l.name = Some(id.name.clone());
                }
                b.scope.push((id.name.clone(), Bound::Var(v)));
            }
            Pat::Tuple { elems, span } => {
                let parts: Vec<Ty> = match ty {
                    Ty::Tuple(ts) => ts.clone(),
                    _ => vec![Ty::Nat; elems.len()],
                };
                for (i, p) in elems.iter().enumerate() {
                    let pt = parts.get(i).cloned().unwrap_or(Ty::Nat);
                    let name = match p {
                        Pat::Ident(id) => Some(id.name.clone()),
                        _ => None,
                    };
                    let pv = b.emit(name, pt.clone(), Rhs::Proj(v, i), *span);
                    self.bind_pat(p, pv, &pt, b);
                }
            }
        }
    }

    fn var(&mut self, path: &Path, ty: &Ty, b: &mut Builder) -> Option<VarId> {
        let span = path.span;
        if path.segs.len() == 1 {
            let name = &path.segs[0].name;
            match b.lookup(name).cloned() {
                Some(Bound::Var(v)) => {
                    // A module parameter is a field of θ.
                    if Some(v) == b.theta && b.owner.is_some_and(|m| self.elab.module(m).params.iter().any(|p| &p.name == name)) {
                        return Some(b.emit(Some(name.clone()), ty.clone(), Rhs::Field(v, name.clone()), span));
                    }
                    return Some(v);
                }
                Some(Bound::Module) => {
                    self.error(Diagnostic::error("E5001", format!("module `{name}` used as a value")).label(span, "modules are compile-time"));
                    return None;
                }
                None => {}
            }
        }
        if let Some(n) = self.lookup_const(path, b) {
            return Some(b.emit(None, Ty::Nat, Rhs::Lit(CLit::Nat(n)), span));
        }
        self.error(Diagnostic::error("E5001", format!("unresolved `{}` at lowering", path.dotted())).label(span, "internal: the checker accepted a name the lowering cannot place"));
        None
    }

    fn unit_of(&self, b: &Builder) -> usize {
        b.fn_id.map(|f| self.elab.function(f).unit).unwrap_or(self.elab.main)
    }

    fn lookup_const(&self, path: &Path, b: &Builder) -> Option<u64> {
        let unit = self.unit_of(b);
        let unit = if path.segs.len() == 2 { *self.elab.units[unit].imports.get(&path.segs[0].name)? } else { unit };
        self.elab.units[unit].consts.get(&path.segs.last()?.name).map(|c| c.value)
    }

    /// `grad(λ θ. body)`: the lambda becomes a function `parent.lambdaN(captured…, θ)`
    /// and the expression a closure over the captured variables.
    fn lambda(&mut self, pat: &Pat, body: &Expr, ty: &Ty, span: Span, b: &mut Builder) -> Option<VarId> {
        let Ty::Fn { params: ptys, .. } = ty else {
            self.error(Diagnostic::error("E5001", "lambda without a function type").label(span, "internal"));
            return None;
        };
        // Captured: every variable of the enclosing scope that the lambda's own
        // parameters do not shadow (the body may use any of them).
        let names = super::expr::pat_names(pat);
        let mut captured: Vec<(String, VarId)> = Vec::new();
        for (n, bd) in &b.scope {
            if let Bound::Var(v) = bd
                && Some(*v) != b.theta
                && !names.contains(n)
                && !captured.iter().any(|(m, _)| m == n)
            {
                captured.push((n.clone(), *v));
            }
        }
        let parent = b.self_id;
        let name = format!("{}.lambda{}", self.funcs[parent.0 as usize].name, b.lambdas);
        b.lambdas += 1;
        let lam_id = crate::ir::FuncId(self.funcs.len() as u32);
        self.funcs.push(Func { name: name.clone(), origin: Origin::Lambda { parent }, dims: vec![], params: vec![], ret: Ty::Nat, effects: vec![], body: Block { lets: vec![], result: VarId(0) }, span });
        let mut lb = Builder::new(lam_id);
        lb.owner = b.owner;
        lb.fn_id = b.fn_id;
        let mut params = Vec::new();
        for (n, v) in &captured {
            let ty = self.var_ty(b, *v);
            let nv = lb.fresh();
            params.push(Param { var: nv, name: n.clone(), ty });
            lb.scope.push((n.clone(), Bound::Var(nv)));
        }
        // Compile-time bindings (modules) are visible as well.
        for (n, bd) in &b.scope {
            if matches!(bd, Bound::Module) {
                lb.scope.push((n.clone(), bd.clone()));
            }
        }
        for (n, pty) in names.iter().zip(ptys) {
            let nv = lb.fresh();
            params.push(Param { var: nv, name: n.clone(), ty: pty.clone() });
            lb.scope.push((n.clone(), Bound::Var(nv)));
        }
        let result = self.expr(body, &mut lb)?;
        let ret = self.ty_of(body);
        let dims = b.fn_id.map(|f| self.fn_dims(f)).unwrap_or_default();
        let lam = &mut self.funcs[lam_id.0 as usize];
        lam.dims = dims;
        lam.params = params;
        lam.ret = ret;
        lam.body = Block { lets: lb.lets, result };
        Some(b.emit(None, ty.clone(), Rhs::Closure { f: lam_id, captured: captured.iter().map(|(_, v)| *v).collect() }, span))
    }

    fn var_ty(&self, b: &Builder, v: VarId) -> Ty {
        if let Some(l) = b.lets.iter().find(|l| l.var == v) {
            return l.ty.clone();
        }
        // A parameter of the function being built.
        let f = &self.funcs[b.self_id.0 as usize];
        if let Some(p) = f.params.iter().find(|p| p.var == v) {
            return p.ty.clone();
        }
        if let Some(fid) = b.fn_id {
            let sig = self.elab.function(fid);
            let mut idx = 0;
            if b.theta == Some(v)
                && let Some(mid) = sig.owner
            {
                let m = self.elab.module(mid);
                return Ty::Params(ModRef::Inst { def: mid, args: m.typarams.iter().map(|d| DimExpr::var(*d)).collect() });
            }
            for (i, p) in sig.params.iter().enumerate() {
                if matches!(p.ty, Some(Ty::Module(_))) {
                    continue;
                }
                idx += 1;
                let base = if sig.owner.is_some() { 1 } else { 0 };
                if VarId((base + idx - 1) as u32) == v {
                    return self.typed.param_ty.get(&(fid, i)).cloned().unwrap_or(Ty::Nat);
                }
            }
        }
        Ty::Nat
    }

    fn args(&mut self, args: &[Arg], b: &mut Builder) -> Option<Vec<LArg>> {
        let mut out = Vec::new();
        for a in args {
            out.push(match a {
                // Module-typed arguments (`mse(net, …)`) are compile-time: dropped.
                Arg::Pos(e) if matches!(self.ty_of(e), Ty::Module(_)) => continue,
                Arg::Pos(e) => LArg::Val(self.expr(e, b)?),
                Arg::Named { name, value } => LArg::Named(name.name.clone(), self.expr(value, b)?),
                Arg::Axis(id) => LArg::Axis(id.name.clone()),
                Arg::NamedAxis { name, value } => {
                    let d = self.dim_of(value, b);
                    LArg::NamedAxis(name.name.clone(), d)
                }
            });
        }
        Some(out)
    }

    fn dim_of(&self, e: &Expr, b: &Builder) -> DimExpr {
        // The checker validated it; re-derive the linear form from the syntax.
        match e {
            Expr::Lit { lit: Lit::Nat(n), .. } => DimExpr::nat(n.parse().unwrap_or(0)),
            Expr::Var { path } => {
                if let Some(n) = self.lookup_const(path, b) {
                    return DimExpr::nat(n);
                }
                let unit = self.unit_of(b);
                let name = &path.segs.last().unwrap().name;
                if let Some(fid) = b.fn_id {
                    let f = self.elab.function(fid);
                    for v in f.typarams.iter().chain(&f.implicit_dims) {
                        if &self.elab.dims[v.0 as usize].name == name {
                            return DimExpr::var(*v);
                        }
                    }
                    if let Some(m) = f.owner {
                        for v in &self.elab.module(m).typarams {
                            if &self.elab.dims[v.0 as usize].name == name {
                                return DimExpr::var(*v);
                            }
                        }
                    }
                }
                self.elab.units[unit].dims.get(name).map(|v| DimExpr::var(*v)).unwrap_or_else(|| DimExpr::nat(0))
            }
            Expr::Binary { op, lhs, rhs, .. } => {
                let (l, r) = (self.dim_of(lhs, b), self.dim_of(rhs, b));
                match op {
                    BinOp::Add => l.add(&r),
                    BinOp::Sub => l.sub(&r),
                    BinOp::Mul => l.mul(&r).unwrap_or_else(|| DimExpr::nat(0)),
                    BinOp::Div => l.div(&r).unwrap_or_else(|| DimExpr::nat(0)),
                    _ => DimExpr::nat(0),
                }
            }
            _ => DimExpr::nat(0),
        }
    }

    fn call(&mut self, callee: &Expr, args: &[Arg], ty: &Ty, span: Span, b: &mut Builder) -> Option<VarId> {
        let resolved = self.typed.callees.get(&span).cloned();
        match resolved {
            Some(Callee::Fn { id, module, explicit_theta }) => {
                let largs = self.args(args, b)?;
                self.call_fn(id, module, explicit_theta, callee, largs, ty, span, b)
            }
            Some(Callee::Builtin(name)) => {
                let largs = self.args(args, b)?;
                self.builtin(&name, largs, ty, span, b)
            }
            Some(Callee::Value) | None => {
                let fun = self.expr(callee, b)?;
                let mut vs = Vec::new();
                for a in args {
                    let Arg::Pos(e) = a else {
                        self.error(Diagnostic::error("E5001", "function values take positional arguments").label(span, "not lowered"));
                        return None;
                    };
                    vs.push(self.expr(e, b)?);
                }
                Some(b.emit(None, ty.clone(), Rhs::Apply { fun, args: vs }, span))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn call_fn(&mut self, id: FnId, module: Option<ModRef>, explicit_theta: bool, callee: &Expr, mut largs: Vec<LArg>, ty: &Ty, span: Span, b: &mut Builder) -> Option<VarId> {
        let sig = self.elab.function(id).clone();
        let core = self.fn_ids[&id];
        let mut dim_args = self.call_dims(id, span);
        // θ: explicit first argument, or a field of the enclosing θ for `use` members.
        let mut theta: Option<VarId> = None;
        if let Some(r) = &module {
            if explicit_theta {
                let Some(LArg::Val(v)) = (if largs.is_empty() { None } else { Some(largs.remove(0)) }) else {
                    self.error(Diagnostic::error("E5001", "missing parameter tree").label(span, "internal"));
                    return None;
                };
                theta = Some(v);
            } else if let Expr::Var { path } = callee
                && let Some(t) = b.theta
            {
                let member = path.segs[0].name.clone();
                theta = Some(b.emit(Some(member.clone()), Ty::Params(r.clone()), Rhs::Field(t, member), span));
            }
            // Module type parameters map to the instance arguments.
            if let Some((def, args)) = inst_of(r) {
                for (v, a) in self.elab.module(def).typarams.iter().zip(args) {
                    dim_args.insert(*v, a.clone());
                }
            }
        }
        // Match arguments to parameters (defaults fill the gaps).
        let mut bound: Vec<Option<VarId>> = vec![None; sig.params.len()];
        let mut pos = 0;
        for a in largs {
            match a {
                LArg::Named(n, v) => {
                    if let Some(i) = sig.params.iter().position(|p| p.name == n) {
                        bound[i] = Some(v);
                    }
                }
                LArg::Val(v) => {
                    while pos < sig.params.len() && (bound[pos].is_some() || matches!(sig.params[pos].ty, Some(Ty::Module(_)))) {
                        pos += 1;
                    }
                    if pos < sig.params.len() {
                        bound[pos] = Some(v);
                        pos += 1;
                    }
                }
                LArg::Axis(_) | LArg::NamedAxis(..) => {}
            }
        }
        let mut call_args = Vec::new();
        if let Some(t) = theta {
            call_args.push(t);
        }
        for (i, p) in sig.params.iter().enumerate() {
            if matches!(p.ty, Some(Ty::Module(_))) || p.axis {
                continue;
            }
            match bound[i] {
                Some(v) => call_args.push(v),
                None => {
                    let Some(d) = &p.default else {
                        self.error(Diagnostic::error("E5001", format!("missing argument `{}`", p.name)).label(span, "internal"));
                        return None;
                    };
                    let v = self.expr(d, b)?;
                    call_args.push(v);
                }
            }
        }
        // `stack{L}(M)`: a `Fold` over the stacked parameter tree (or `L` calls with `--unroll-stack`).
        if let Some(ModRef::Stack { n, .. }) = &module
            && !self.opts.unroll_stack
        {
            let stacked = theta?;
            let x = *call_args.get(1)?;
            let rest: Vec<VarId> = call_args[2..].to_vec();
            return Some(b.emit(None, ty.clone(), Rhs::Fold { body: core, dim_args, n: n.clone(), stacked: vec![stacked], init: x, extra: rest, index: false, reverse: false, scan: false }, span));
        }
        if let Some(ModRef::Stack { n, .. }) = &module {
            let Some(count) = n.as_nat() else {
                self.error(Diagnostic::error("E5002", "`stack` with a symbolic count is not supported in 0.1.0").label(span, format!("the count `{}` must be a constant", n.display(&|v| self.elab.dim_name(v)))));
                return None;
            };
            let stacked = theta?;
            let mut x = *call_args.get(1)?;
            let rest: Vec<VarId> = call_args[2..].to_vec();
            let inner_ty = ty.clone();
            for i in 0..count {
                let inner_ref = match &module {
                    Some(ModRef::Stack { inner, .. }) => (**inner).clone(),
                    _ => unreachable!(),
                };
                let slice = b.emit(None, Ty::Params(inner_ref), Rhs::Prim { op: PrimOp::SliceParams { index: i }, args: vec![stacked] }, span);
                let mut a = vec![slice, x];
                a.extend(rest.iter().copied());
                x = b.emit(None, inner_ty.clone(), Rhs::Call { f: core, dim_args: dim_args.clone(), args: a }, span);
            }
            return Some(x);
        }
        Some(b.emit(None, ty.clone(), Rhs::Call { f: core, dim_args, args: call_args }, span))
    }

    fn builtin(&mut self, name: &str, args: Vec<LArg>, ty: &Ty, span: Span, b: &mut Builder) -> Option<VarId> {
        let vals: Vec<VarId> = args.iter().filter_map(|a| match a {
            LArg::Val(v) | LArg::Named(_, v) => Some(*v),
            _ => None,
        }).collect();
        let axis = args.iter().find_map(|a| match a {
            LArg::Axis(n) => Some(n.clone()),
            _ => None,
        });
        let axes_sel: Vec<String> = args.iter().filter_map(|a| match a {
            LArg::Axis(n) => Some(n.clone()),
            _ => None,
        }).collect();
        let named_axis = args.iter().find_map(|a| match a {
            LArg::NamedAxis(n, d) => Some((n.clone(), d.clone())),
            _ => None,
        });
        let unit = self.unit_of(b);
        let consts = |n: &str| self.elab.units[unit].consts.get(n).map(|c| c.value);
        let op = match name {
            "gelu" => PrimOp::Gelu,
            "relu" => PrimOp::Relu,
            "exp" => PrimOp::Exp,
            "log" => PrimOp::Log,
            "tanh" => PrimOp::Tanh,
            "sqrt" => PrimOp::Sqrt,
            "abs" => PrimOp::Abs,
            "sigmoid" => PrimOp::Sigmoid,
            "sin" => PrimOp::Sin,
            "cos" => PrimOp::Cos,
            "real" => {
                // `real` of an index tensor is the identity on values (indices become reals).
                PrimOp::Real
            }
            "mean" => match axis {
                Some(a) => PrimOp::Mean { axis: a },
                None => PrimOp::MeanAll,
            },
            "sum" => match axis {
                Some(a) => PrimOp::Sum { axis: a },
                None => PrimOp::SumAll,
            },
            "softmax" => PrimOp::Softmax { axis: axis? },
            "embed" => PrimOp::Embed,
            "one_hot" => PrimOp::OneHot,
            "split_key" => PrimOp::SplitKey { n: 2 },
            "merge" => PrimOp::MergeAxes { a: axes_sel.first()?.clone(), b: axes_sel.get(1)?.clone() },
            "cross_entropy" => PrimOp::CrossEntropy,
            "argmax" => PrimOp::Argmax,
            "sample" => PrimOp::Sample,
            "matmul" => PrimOp::Matmul,
            "split_heads" => PrimOp::SplitHeads { h: consts("H")?, hd: consts("HD")? },
            "merge_heads" => PrimOp::MergeHeads { h: consts("H")?, hd: consts("HD")? },
            "positions" => {
                let (a, d) = named_axis?;
                PrimOp::Positions { axis: a, len: d }
            }
            "causal_mask" => {
                let (a, d) = named_axis?;
                PrimOp::CausalMask { axis: a, len: d }
            }
            "dropout" => {
                let (rate, key, x) = (*vals.first()?, *vals.get(1)?, *vals.get(2)?);
                let mask = b.emit(None, ty.clone(), Rhs::Prim { op: PrimOp::DropoutMask, args: vec![rate, key, x] }, span);
                return Some(b.emit(None, ty.clone(), Rhs::Prim { op: PrimOp::Mul, args: vec![x, mask] }, span));
            }
            "grad" => {
                return Some(b.emit(None, ty.clone(), Rhs::Grad(*vals.first()?), span));
            }
            other => {
                self.error(Diagnostic::error("E5001", format!("builtin `{other}` is not lowered")).label(span, "no Core primitive"));
                return None;
            }
        };
        Some(b.emit(None, ty.clone(), Rhs::Prim { op, args: vals }, span))
    }
}

#[derive(Clone, Debug)]
enum LArg {
    Val(VarId),
    Named(String, VarId),
    Axis(String),
    NamedAxis(String, DimExpr),
}

fn const_index(e: &Expr) -> Option<i64> {
    match e {
        Expr::Lit { lit: Lit::Nat(n), .. } => n.parse().ok(),
        Expr::Unary { op: UnOp::Neg, expr, .. } => const_index(expr).map(|i| -i),
        _ => None,
    }
}

fn remove_axis(t: &Ty, axis: &str) -> Ty {
    match t {
        Ty::Real(s) => {
            let mut s = s.clone();
            s.axes.retain(|a| a.name.as_deref() != Some(axis));
            Ty::Real(s)
        }
        Ty::Idx { bound, shape } => {
            let mut s = shape.clone();
            s.axes.retain(|a| a.name.as_deref() != Some(axis));
            Ty::Idx { bound: bound.clone(), shape: s }
        }
        t => t.clone(),
    }
}

fn collect_product<'e>(e: &'e Expr, out: &mut Vec<&'e Expr>) {
    match e {
        Expr::Binary { op: BinOp::Mul, lhs, rhs, .. } => {
            collect_product(lhs, out);
            collect_product(rhs, out);
        }
        e => out.push(e),
    }
}

fn inst_of(r: &ModRef) -> Option<(presk_types::ty::ModuleId, &Vec<DimExpr>)> {
    match r {
        ModRef::Inst { def, args } => Some((*def, args)),
        ModRef::Stack { inner, .. } => inst_of(inner),
        ModRef::Compose(_) => None,
    }
}

pub(crate) fn pat_names(p: &Pat) -> Vec<String> {
    match p {
        Pat::Ident(i) => vec![i.name.clone()],
        Pat::Tuple { elems, .. } => elems.iter().flat_map(pat_names).collect(),
    }
}
