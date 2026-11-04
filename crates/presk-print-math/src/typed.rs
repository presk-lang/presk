//! The first entry: the typed program, the source's own names (plan 0.13.0
//! step 4a).
//!
//! Each module function, and each function of a file, becomes one display:
//! the equations of its `let`s in their order, then its value, in the
//! reference's chapter 3 (section "Reading a program as equations"). An
//! operation is written by its name (`\code{softmax axis}\ a\ x`), never
//! lowered; `ein` is written as its sum, at an index named by the axes of
//! its result, because that is the form in which an index error shows; a
//! call through a stack is written as its recurrence. Nothing here reads a
//! realization, a plan or a machine.

use std::collections::{HashMap, HashSet};

use presk_syntax::ast::{self, Arg, AxisSpec, BinOp, Decl, Expr, IndexSel, Lit, Member, ModExpr, Pat, UnOp};
use presk_syntax::{SourceMap, Span};
use presk_types::check::{Callee, TypedProgram};
use presk_types::elab::{ElabProgram, FnId};
use presk_types::ty::{ModRef, ModuleId, Shape, Ty};

use crate::m::{B, F, M, R, T};

/// One equation of a display, `lhs = rhs`, with the range of its index
/// when it is a recurrence's step (`0 ≤ ℓ < L`).
#[derive(Clone, Debug)]
pub struct Equation {
    pub lhs: M,
    pub rhs: M,
    pub cond: Option<M>,
}

/// One function as equations.
#[derive(Clone, Debug)]
pub struct Display {
    /// The function's name as PSK-Core and the provenance table name it:
    /// `Linear.apply`, `loss`.
    pub name: String,
    /// Its parameters with their types, `x : ℝ[…, In]`.
    pub of: Vec<M>,
    /// `uses Rand`.
    pub effects: Vec<String>,
    pub eqs: Vec<Equation>,
    pub span: Span,
}

/// A module, or a function of a file.
#[derive(Clone, Debug)]
pub enum Item {
    Module { head: M, params: Vec<M>, fns: Vec<Display>, span: Span },
    Fn(Display),
}

/// One source file.
#[derive(Clone, Debug)]
pub struct Unit {
    /// The file's name.
    pub file: String,
    /// Its constants, dimensions, types and assertions.
    pub consts: Vec<M>,
    pub dims: Vec<M>,
    pub types: Vec<M>,
    pub asserts: Vec<M>,
    pub items: Vec<Item>,
}

/// The program as equations, and what the printer could not write.
#[derive(Clone, Debug)]
pub struct Listing {
    pub units: Vec<Unit>,
    /// Each form the printer does not write, at its span, with its source
    /// text: printed as that text, and reported.
    pub gaps: Vec<(Span, String)>,
    /// The formula of every expression a function's body holds, by span:
    /// what the page shows on the elaborated program's `expr` cards.
    pub formulas: HashMap<Span, M>,
}

impl Listing {
    /// A function's display, by the name PSK-Core gives it.
    pub fn display(&self, name: &str) -> Option<&Display> {
        self.units.iter().flat_map(|u| &u.items).find_map(|i| match i {
            Item::Fn(d) if d.name == name => Some(d),
            Item::Module { fns, .. } => fns.iter().find(|d| d.name == name),
            _ => None,
        })
    }
}

/// The typed program, every file of it in load order (imports first).
pub fn typed_program(sm: &SourceMap, elab: &ElabProgram, typed: &TypedProgram) -> Listing {
    let mut out = Listing { units: Vec::new(), gaps: Vec::new(), formulas: HashMap::new() };
    for u in &elab.units {
        let mut p = P { sm, elab, typed, module: None, module_ast: None, used: HashSet::new(), names: HashMap::new(), pending: Vec::new(), gaps: Vec::new(), formulas: HashMap::new() };
        let mut unit = Unit { file: u.path.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string(), consts: Vec::new(), dims: Vec::new(), types: Vec::new(), asserts: Vec::new(), items: Vec::new() };
        for d in &u.ast.decls {
            match d {
                Decl::Const { name, value, .. } => {
                    let mut c = M::rel(R::Eq, M::name(&name.name), p.dim(value));
                    if !matches!(value, Expr::Lit { .. })
                        && let Some(v) = u.consts.get(&name.name)
                    {
                        c = M::rel(R::Eq, c, M::num(v.value.to_string()));
                    }
                    unit.consts.push(c);
                }
                Decl::Dim { name, bound, .. } => unit.dims.push(match bound {
                    Some(b) => M::rel(R::Le, M::name(&name.name), p.dim(b)),
                    None => M::name(&name.name),
                }),
                Decl::TypeAlias { name, ty, .. } => unit.types.push(M::rel(R::Eq, M::code(&name.name), p.ty(ty))),
                Decl::StaticAssert { cond, .. } => unit.asserts.push(p.dim(cond)),
                Decl::Module(m) => {
                    let mid = u.modules[&m.name.name];
                    unit.items.push(p.module_item(mid, m));
                }
                Decl::Fn(f) => {
                    let fid = u.fns[&f.name.name];
                    unit.items.push(Item::Fn(p.function(fid, f)));
                }
                _ => {}
            }
        }
        out.gaps.append(&mut p.gaps);
        out.formulas.extend(p.formulas);
        out.units.push(unit);
    }
    out
}

struct P<'a> {
    sm: &'a SourceMap,
    elab: &'a ElabProgram,
    typed: &'a TypedProgram,
    /// The module whose function is being written.
    module: Option<ModuleId>,
    module_ast: Option<&'a ast::ModuleDecl>,
    /// Every name the function can see, which a fresh name avoids.
    used: HashSet<String>,
    /// The name each binding of the function is written with: its own, or
    /// primed when it binds a name bound before (`x = x + …` is no
    /// equation; `x' = x + …` is).
    names: HashMap<String, String>,
    /// The equations a `let` needs before its own: an `ein` an operation
    /// reads, a stack's recurrence.
    pending: Vec<Equation>,
    gaps: Vec<(Span, String)>,
    formulas: HashMap<Span, M>,
}

/// The operands of an `ein`'s product, in order.
fn product(e: &Expr) -> Vec<&Expr> {
    match e {
        Expr::Binary { op: BinOp::Mul, lhs, rhs, .. } => {
            let mut v = product(lhs);
            v.extend(product(rhs));
            v
        }
        e => vec![e],
    }
}

const ELEMENTWISE: &[&str] = &["gelu", "relu", "exp", "tanh", "sqrt", "abs", "log", "sigmoid", "sin", "cos"];

impl<'a> P<'a> {
    fn text(&self, span: Span) -> String {
        let t = self.sm.file(span.file).text.get(span.start as usize..span.end as usize).unwrap_or("");
        t.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    fn gap(&mut self, span: Span) -> M {
        let t = self.text(span);
        self.gaps.push((span, t.clone()));
        M::Gap(t)
    }

    fn ty_of(&self, e: &Expr) -> Option<&'a Ty> {
        self.typed.ty_of.get(&e.span())
    }

    fn shape_of(&self, e: &Expr) -> Option<&'a Shape> {
        match self.ty_of(e) {
            Some(Ty::Real(s) | Ty::Idx { shape: s, .. } | Ty::Truth(s)) => Some(s),
            _ => None,
        }
    }

    /// A name no binding of the function has, from `cands`, primed when
    /// they all are taken.
    fn fresh(&mut self, cands: &[&str]) -> String {
        let mut primes = String::new();
        loop {
            for c in cands {
                let n = format!("{c}{primes}");
                if !self.used.contains(&n) {
                    self.used.insert(n.clone());
                    return n;
                }
            }
            primes.push('\'');
        }
    }

    // ----- declarations -------------------------------------------------------

    fn module_item(&mut self, mid: ModuleId, m: &'a ast::ModuleDecl) -> Item {
        let def = self.elab.module(mid);
        let head = if m.typarams.is_empty() { M::code(&m.name.name) } else { M::Braced(Box::new(M::code(&m.name.name)), m.typarams.iter().map(|t| M::name(&t.name.name)).collect()) };
        let mut params = Vec::new();
        for member in &m.members {
            match member {
                Member::Param { names, ty, .. } => {
                    for n in names {
                        params.push(M::rel(R::Has, M::name(&n.name), self.ty(ty)));
                    }
                }
                Member::Use { names, module, .. } => {
                    for n in names {
                        params.push(M::rel(R::Has, M::name(&n.name), self.modexpr(module)));
                    }
                }
                Member::Fn(_) => {}
            }
        }
        self.module = Some(mid);
        self.module_ast = Some(m);
        let mut fns = Vec::new();
        for member in &m.members {
            if let Member::Fn(f) = member
                && let Some(fid) = def.fns.iter().copied().find(|id| self.elab.function(*id).name == f.name.name)
            {
                fns.push(self.function(fid, f));
            }
        }
        self.module = None;
        self.module_ast = None;
        Item::Module { head, params, fns, span: m.span }
    }

    fn function(&mut self, fid: FnId, f: &'a ast::FnDecl) -> Display {
        let sig = self.elab.function(fid);
        let name = presk_core::lower::fn_name(self.elab, fid);
        self.used.clear();
        self.names.clear();
        if let Some(m) = self.module {
            let def = self.elab.module(m);
            self.used.extend(def.params.iter().map(|p| p.name.clone()));
            self.used.extend(def.uses.iter().map(|u| u.name.clone()));
            self.used.insert("θ".into());
        }
        collect_lets(&f.body, &mut self.used);
        let mut of = Vec::new();
        let mut args = Vec::new();
        if self.module.is_some() {
            args.push(M::name("θ"));
        }
        for (i, p) in f.params.iter().enumerate() {
            self.used.insert(p.name.name.clone());
            let n = self.bind_name(&p.name.name);
            if p.axis {
                of.push(M::op("axis", vec![], vec![n]));
                continue;
            }
            let mut b = match &p.ty {
                Some(t) => M::rel(R::Has, n.clone(), self.ty(t)),
                None => n.clone(),
            };
            if let Some(d) = &p.default {
                let v = self.tensor(d);
                b = M::rel(R::Eq, b, v);
            }
            of.push(b);
            // A module is no value: a function's denotation does not take it
            // (definition `den-function`).
            let module = matches!(sig.params.get(i).and_then(|q| q.ty.as_ref()), Some(Ty::Module(_))) || matches!(self.typed.param_ty.get(&(fid, i)), Some(Ty::Module(_)));
            if !module {
                args.push(n);
            }
        }
        let mut eqs = Vec::new();
        let mut cur = &f.body;
        let value = M::apply(M::den(M::code(&name)), args);
        loop {
            match cur {
                Expr::Let { pat, init, body, .. } => {
                    self.equation(Err(pat), init, &mut eqs);
                    cur = body;
                }
                e => {
                    self.equation(Ok(value), e, &mut eqs);
                    break;
                }
            }
        }
        Display { name, of, effects: sig.effects.clone(), eqs, span: f.span }
    }

    /// The name a binding of `n` is written with: `n`, or `n` primed when
    /// `n` is bound already.
    fn bind_name(&mut self, n: &str) -> M {
        if n == "_" {
            return M::name("_");
        }
        let shown = if self.names.contains_key(n) {
            let mut c = format!("{n}'");
            while self.used.contains(&c) {
                c.push('\'');
            }
            self.used.insert(c.clone());
            c
        } else {
            n.to_string()
        };
        self.names.insert(n.to_string(), shown.clone());
        M::name(shown)
    }

    fn pat(&self, p: &Pat) -> M {
        match p {
            Pat::Ident(i) => M::name(&i.name),
            Pat::Tuple { elems, .. } => M::Tuple(elems.iter().map(|e| self.pat(e)).collect()),
        }
    }

    fn bind(&mut self, p: &Pat) -> M {
        match p {
            Pat::Ident(i) => self.bind_name(&i.name),
            Pat::Tuple { elems, .. } => M::Tuple(elems.iter().map(|e| self.bind(e)).collect()),
        }
    }

    /// The equation `lhs = e`, after the ones it needs; at an index when
    /// `e` holds an `ein` that only elementwise operations read. Its left
    /// side is the function's value, or a pattern bound once `e` is
    /// written.
    fn equation(&mut self, lhs: Result<M, &Pat>, e: &'a Expr, eqs: &mut Vec<Equation>) {
        let index = match self.shape_of(e) {
            Some(s) if ein_at_top(e) && !s.axes.is_empty() => Some(fresh_terms(s)),
            _ => None,
        };
        let rhs = match &index {
            Some(terms) => self.at(e, terms),
            None => self.tensor(e),
        };
        let lhs = match lhs {
            Ok(m) => m,
            Err(p) => self.bind(p),
        };
        let lhs = match index {
            Some(terms) => M::index(lhs, terms),
            None => lhs,
        };
        eqs.append(&mut self.pending);
        eqs.push(Equation { lhs, rhs, cond: None });
    }

    // ----- types and dimensions ---------------------------------------------------

    /// A dimension, a hyperparameter or a constant's expression.
    fn dim(&mut self, e: &Expr) -> M {
        match e {
            Expr::Lit { lit: Lit::Nat(n) | Lit::Float(n), .. } => M::num(n),
            Expr::Var { path } => M::name(path.dotted()),
            Expr::Binary { op, lhs, rhs, .. } => {
                let (l, r) = (self.dim(lhs), self.dim(rhs));
                match op {
                    BinOp::Add => M::bin(B::Add, l, r),
                    BinOp::Sub => M::bin(B::Sub, l, r),
                    BinOp::Mul => M::bin(B::DimMul, l, r),
                    BinOp::Div => M::bin(B::Div, l, r),
                    BinOp::Rem => M::bin(B::Rem, l, r),
                    BinOp::Pow => M::Pow(Box::new(l), Box::new(r)),
                    BinOp::Eq => M::rel(R::Eq, l, r),
                    BinOp::Le => M::rel(R::Le, l, r),
                    BinOp::Lt => M::rel(R::Lt, l, r),
                    BinOp::Ge => M::rel(R::Ge, l, r),
                    BinOp::Gt => M::rel(R::Gt, l, r),
                    _ => self.gap(e.span()),
                }
            }
            Expr::Unary { op: UnOp::Neg, expr, .. } => M::Neg(Box::new(self.dim(expr))),
            _ => self.gap(e.span()),
        }
    }

    fn axes(&mut self, axes: &[AxisSpec]) -> Vec<M> {
        axes.iter()
            .map(|a| match a {
                AxisSpec::Axis { name: Some(n), dim } => M::AxisLen(n.name.clone(), Box::new(self.dim(dim))),
                AxisSpec::Axis { name: None, dim } => self.dim(dim),
                AxisSpec::Ellipsis { .. } => M::Ellipsis,
            })
            .collect()
    }

    /// A type as the source writes it.
    fn ty(&mut self, t: &ast::Ty) -> M {
        match t {
            ast::Ty::Real { axes, .. } => M::Tensor(T::Real, None, self.axes(axes)),
            ast::Ty::Idx { bound, axes, .. } => {
                let b = self.dim(bound);
                M::Tensor(T::Idx, Some(Box::new(b)), self.axes(axes))
            }
            ast::Ty::Bool { axes, .. } => M::Tensor(T::Bool, None, self.axes(axes)),
            ast::Ty::Param { inner, .. } => {
                let i = self.ty(inner);
                M::op("param", vec![], vec![i])
            }
            ast::Ty::Params { module, .. } => M::op("params", vec![], vec![M::code(module.dotted())]),
            ast::Ty::Named { path, args, .. } => {
                if args.is_empty() {
                    M::code(path.dotted())
                } else {
                    M::Braced(Box::new(M::code(path.dotted())), args.iter().map(|a| self.dim(a)).collect())
                }
            }
            ast::Ty::Tuple { elems, .. } => M::Tuple(elems.iter().map(|e| self.ty(e)).collect()),
        }
    }

    fn modexpr(&mut self, m: &ModExpr) -> M {
        match m {
            ModExpr::App { path, dim_args, mod_args, .. } => {
                let head = M::code(path.dotted());
                let head = if dim_args.is_empty() { head } else { M::Braced(Box::new(head), dim_args.iter().map(|a| self.dim(a)).collect()) };
                if mod_args.is_empty() { head } else { M::apply(head, mod_args.iter().map(|a| self.modexpr(a)).collect()) }
            }
            ModExpr::Compose { parts, .. } => {
                let mut it = parts.iter();
                let first = it.next().map(|p| self.modexpr(p)).unwrap_or(M::Ellipsis);
                it.fold(first, |acc, p| {
                    let r = self.modexpr(p);
                    M::bin(B::Compose, acc, r)
                })
            }
        }
    }

    // ----- expressions --------------------------------------------------------------

    /// An expression as a tensor: the operations by their names.
    fn tensor(&mut self, e: &'a Expr) -> M {
        let m = self.tensor_inner(e);
        if !matches!(e, Expr::Var { .. } | Expr::Lit { .. }) {
            self.formulas.entry(e.span()).or_insert_with(|| m.clone());
        }
        m
    }

    fn tensor_inner(&mut self, e: &'a Expr) -> M {
        match e {
            Expr::Lit { lit, .. } => match lit {
                Lit::Nat(n) | Lit::Float(n) => M::num(n),
                Lit::Str(s) => M::code(format!("\"{s}\"")),
                Lit::Quantity { num, unit } => M::code(format!("{num}{unit}")),
            },
            Expr::Var { path } => {
                if path.segs.len() == 1 {
                    let n = &path.segs[0].name;
                    M::name(self.names.get(n).cloned().unwrap_or_else(|| n.clone()))
                } else {
                    M::code(path.dotted())
                }
            }
            Expr::Let { .. } => self.gap(e.span()),
            Expr::Lambda { pat, body, .. } => {
                // A binder may take a name bound outside it: `θ ↦ …` is no
                // equation, and is written as the source writes it.
                let saved = self.names.clone();
                let mut bound = HashSet::new();
                pat_names(pat, &mut bound);
                for n in bound {
                    self.names.insert(n.clone(), n);
                }
                let p = self.pat(pat);
                let b = self.tensor(body);
                self.names = saved;
                M::MapsTo(vec![p], Box::new(b))
            }
            Expr::Call { callee, args, span } => self.call(callee, args, *span, e),
            Expr::Field { base, field, .. } => {
                let b = self.tensor(base);
                M::Field(Box::new(b), field.name.clone())
            }
            Expr::Index { base, sels, .. } => {
                let mut x = self.tensor(base);
                for s in sels {
                    let attr = match s {
                        IndexSel::Named { axis, value } => M::AxisLen(axis.name.clone(), Box::new(self.dim(value))),
                        IndexSel::Pos(value) => self.dim(value),
                        IndexSel::Slice { axis, lo, hi } => {
                            let range = M::Cat(vec![self.dim(lo), M::code(".."), self.dim(hi)]);
                            let attr = match axis {
                                Some(a) => M::AxisLen(a.name.clone(), Box::new(range)),
                                None => range,
                            };
                            x = M::op("slice", vec![vec![attr]], vec![x]);
                            continue;
                        }
                    };
                    x = M::op("index", vec![vec![attr]], vec![x]);
                }
                x
            }
            Expr::Inst { base, args, .. } => {
                let b = self.tensor(base);
                M::Braced(Box::new(b), args.iter().map(|a| self.dim(a)).collect())
            }
            Expr::Ein { .. } => {
                // An `ein` an operation reads: its sum as an equation of
                // its own, for a fresh name.
                let Some(s) = self.shape_of(e) else { return self.gap(e.span()) };
                let terms = fresh_terms(s);
                let u = self.fresh(&["u", "v", "z"]);
                let rhs = self.at(e, &terms);
                self.pending.push(Equation { lhs: M::index(M::name(&u), terms), rhs, cond: None });
                M::name(u)
            }
            Expr::Binary { op, lhs, rhs, .. } => {
                let b = match op {
                    BinOp::Add => Some(B::Add),
                    BinOp::Sub => Some(B::Sub),
                    BinOp::Mul => Some(B::Mul),
                    BinOp::Div => Some(B::Div),
                    BinOp::Rem => Some(B::Rem),
                    BinOp::Compose => Some(B::Compose),
                    _ => None,
                };
                let r = match op {
                    BinOp::Eq => Some(R::Eq),
                    BinOp::Le => Some(R::Le),
                    BinOp::Lt => Some(R::Lt),
                    BinOp::Ge => Some(R::Ge),
                    BinOp::Gt => Some(R::Gt),
                    _ => None,
                };
                match (op, b, r) {
                    (BinOp::Pow, ..) => {
                        let (l, r) = (self.tensor(lhs), self.tensor(rhs));
                        M::Pow(Box::new(l), Box::new(r))
                    }
                    (BinOp::Dot, ..) => {
                        let (l, r) = (self.tensor(lhs), self.tensor(rhs));
                        M::op("dot", vec![], vec![l, r])
                    }
                    (_, Some(b), _) => {
                        let (l, r) = (self.tensor(lhs), self.tensor(rhs));
                        M::bin(b, l, r)
                    }
                    (_, _, Some(r)) => {
                        let (l, rr) = (self.tensor(lhs), self.tensor(rhs));
                        M::rel(r, l, rr)
                    }
                    _ => self.gap(e.span()),
                }
            }
            Expr::Unary { op: UnOp::Neg, expr, .. } => M::Neg(Box::new(self.tensor(expr))),
            // An ascription, or the injection of a hyperparameter into ℝ:
            // exact, so the number itself.
            Expr::Ascribe { expr, .. } => self.tensor(expr),
            Expr::Tuple { elems, .. } => M::Tuple(elems.iter().map(|x| self.tensor(x)).collect()),
        }
    }

    /// The positional arguments, the `axis a` names, the named ones.
    fn split_args(args: &'a [Arg]) -> (Vec<&'a Expr>, Vec<&'a str>, Vec<(&'a str, &'a Expr)>) {
        let mut pos = Vec::new();
        let mut axes = Vec::new();
        let mut named = Vec::new();
        for a in args {
            match a {
                Arg::Pos(e) => pos.push(e),
                Arg::Axis(i) => axes.push(i.name.as_str()),
                Arg::Named { name, value } | Arg::NamedAxis { name, value } => named.push((name.name.as_str(), value)),
            }
        }
        (pos, axes, named)
    }

    fn call(&mut self, callee: &'a Expr, args: &'a [Arg], span: Span, whole: &'a Expr) -> M {
        match self.typed.callees.get(&span) {
            Some(Callee::Builtin(name)) => self.builtin(name, args, whole),
            Some(Callee::Fn { id, module: Some(ModRef::Stack { n, .. }), .. }) => self.stack(*id, n, callee, args, whole),
            Some(Callee::Fn { .. }) => {
                let path = match callee {
                    Expr::Var { path } => path.dotted(),
                    c => self.text(c.span()),
                };
                let mut vals = Vec::new();
                for a in args {
                    match a {
                        Arg::Pos(x) if matches!(self.ty_of(x), Some(Ty::Module(_))) => {}
                        Arg::Pos(x) | Arg::Named { value: x, .. } => vals.push(self.tensor(x)),
                        _ => {}
                    }
                }
                M::apply(M::den(M::code(path)), vals)
            }
            Some(Callee::Value) => {
                let f = self.tensor(callee);
                let vals = args
                    .iter()
                    .filter_map(|a| match a {
                        Arg::Pos(x) => Some(x),
                        _ => None,
                    })
                    .map(|x| self.tensor(x))
                    .collect();
                M::apply(f, vals)
            }
            None => self.gap(whole.span()),
        }
    }

    /// A call through `stack{d}(m)`: its recurrence, chapter 3's
    /// `y₀ = x`, `y_{ℓ+1} = ⟦f⟧(θ[ℓ], y_ℓ, …)`, and `y_d` in its place.
    fn stack(&mut self, id: FnId, n: &presk_types::dim::DimExpr, callee: &'a Expr, args: &'a [Arg], whole: &'a Expr) -> M {
        let Expr::Var { path } = callee else { return self.gap(whole.span()) };
        let member = path.segs[0].name.clone();
        let (pos, _, _) = Self::split_args(args);
        let Some(first) = pos.first() else { return self.gap(whole.span()) };
        let init = self.tensor(first);
        let extras: Vec<M> = pos[1..].iter().map(|x| self.tensor(x)).collect();
        // The count as the source writes it, at the stack's `use`.
        let count = self
            .module_ast
            .and_then(|m| {
                m.members.iter().find_map(|mem| match mem {
                    Member::Use { names, module: ModExpr::App { path, dim_args, .. }, .. } if path.is_single("stack") && names.iter().any(|x| x.name == member) => dim_args.first(),
                    _ => None,
                })
            })
            .map(|d| self.dim(d))
            .unwrap_or_else(|| M::num(n.display(&|v| self.elab.dim_name(v))));
        let y = self.fresh(&["y", "z", "u"]);
        let l = M::name("ℓ");
        let f = presk_core::lower::fn_name(self.elab, id);
        let mut call_args = vec![M::index(M::Field(Box::new(M::name("θ")), member), vec![l.clone()]), M::subscript(M::name(&y), l.clone())];
        call_args.extend(extras);
        self.pending.push(Equation { lhs: M::subscript(M::name(&y), M::num("0")), rhs: init, cond: None });
        let range = M::rel(R::Lt, M::rel(R::Le, M::num("0"), l.clone()), count.clone());
        self.pending.push(Equation { lhs: M::subscript(M::name(&y), M::bin(B::Add, l, M::num("1"))), rhs: M::apply(M::den(M::code(f)), call_args), cond: Some(range) });
        M::subscript(M::name(y), count)
    }

    fn builtin(&mut self, name: &str, args: &'a [Arg], whole: &'a Expr) -> M {
        let (pos, axes, named) = Self::split_args(args);
        let x = |p: &mut Self, i: usize| -> M {
            match pos.get(i) {
                Some(e) => p.tensor(e),
                None => p.gap(whole.span()),
            }
        };
        let code_axes = |axes: &[&str]| axes.iter().map(|a| M::code(*a)).collect::<Vec<_>>();
        match name {
            "exp" => M::Fun(F::Exp, Box::new(x(self, 0))),
            "log" => M::Fun(F::Ln, Box::new(x(self, 0))),
            "tanh" => M::Fun(F::Tanh, Box::new(x(self, 0))),
            "sin" => M::Fun(F::Sin, Box::new(x(self, 0))),
            "cos" => M::Fun(F::Cos, Box::new(x(self, 0))),
            "sqrt" => M::Fun(F::Sqrt, Box::new(x(self, 0))),
            "abs" => M::Fun(F::Abs, Box::new(x(self, 0))),
            "gelu" | "relu" | "sigmoid" | "one_hot" | "argmax" => {
                let a = x(self, 0);
                M::op(name, vec![], vec![a])
            }
            "mean" | "sum" | "softmax" => {
                let mut ops = vec![x(self, 0)];
                if let Some((_, m)) = named.iter().find(|(n, _)| *n == "where") {
                    ops.push(self.tensor(m));
                }
                if axes.is_empty() { M::op(name, vec![], ops) } else { M::op(format!("{name} axis"), vec![code_axes(&axes)], ops) }
            }
            "embed" | "cross_entropy" | "sample" => {
                let (a, b) = (x(self, 0), x(self, 1));
                M::op(name, vec![], vec![a, b])
            }
            "matmul" => {
                let (a, b) = (x(self, 0), x(self, 1));
                M::op("dot", vec![], vec![a, b])
            }
            "max" | "min" => {
                let (a, b) = (x(self, 0), x(self, 1));
                M::op(name, vec![], vec![a, b])
            }
            // `clamp(x, lo, hi)`: `min (max x, lo), hi`, which it denotes.
            "clamp" => {
                let (a, lo, hi) = (x(self, 0), x(self, 1), x(self, 2));
                M::op("min", vec![], vec![M::op("max", vec![], vec![a, lo]), hi])
            }
            "concat" => {
                let (a, b) = (x(self, 0), x(self, 1));
                M::op("concat", vec![code_axes(&axes)], vec![a, b])
            }
            "dropout" => {
                let (r, k, v) = (x(self, 0), x(self, 1), x(self, 2));
                M::op("dropout", vec![], vec![r, k, v])
            }
            // The injection of a hyperparameter into ℝ, or of an index
            // tensor: exact, the number itself.
            "real" => x(self, 0),
            "merge" => {
                let a = x(self, 0);
                M::op("merge", vec![code_axes(&axes)], vec![a])
            }
            "split" => {
                let a = x(self, 0);
                let parts: Vec<M> = named.iter().map(|(n, v)| M::AxisLen(n.to_string(), Box::new(self.dim(v)))).collect();
                M::op("split", vec![code_axes(&axes), vec![M::code("into")], parts], vec![a])
            }
            "split_heads" | "merge_heads" => {
                // The constants `H` and `HD` of the file of the call.
                let a = x(self, 0);
                M::op(name, vec![vec![M::Cat(vec![M::name("H"), M::code("x"), M::name("HD")])]], vec![a])
            }
            "positions" | "causal_mask" => {
                let attrs: Vec<M> = named.iter().map(|(n, v)| M::AxisLen(n.to_string(), Box::new(self.dim(v)))).collect();
                M::op(name, vec![attrs], vec![])
            }
            "split_key" => {
                let k = x(self, 0);
                M::op("split_key", vec![vec![M::num("2")]], vec![k])
            }
            "grad" => M::Nabla(Box::new(x(self, 0))),
            _ => self.gap(whole.span()),
        }
    }

    /// An expression at an index: `terms` index the axes of its type (its
    /// `...` first). Elementwise operations and `ein` are written at the
    /// index; any other is written as a tensor and indexed.
    fn at(&mut self, e: &'a Expr, terms: &[M]) -> M {
        let Some(shape) = self.shape_of(e) else { return self.tensor(e) };
        let side = |p: &mut Self, x: &'a Expr| -> M {
            match p.shape_of(x) {
                Some(s) if !s.axes.is_empty() || s.prefix.is_some() => {
                    let t = align(terms, shape, s);
                    p.at(x, &t)
                }
                _ => p.tensor(x),
            }
        };
        let m = match e {
            Expr::Ein { span, body, .. } => {
                let Some(info) = self.typed.eins.get(span) else { return self.gap(e.span()) };
                let off = usize::from(shape.prefix.is_some());
                let outer: HashMap<&str, M> = info.output.iter().enumerate().filter_map(|(k, a)| Some((a.name.as_deref()?, terms.get(off + k)?.clone()))).collect();
                let ops = product(body);
                let mut factors = Vec::new();
                for (op, (_, axes)) in ops.iter().zip(&info.operands) {
                    let t: Vec<M> = axes.iter().map(|a| a.name.as_deref().map(|n| outer.get(n).cloned().unwrap_or_else(|| M::code(n))).unwrap_or(M::Ellipsis)).collect();
                    factors.push(self.at(op, &t));
                }
                let prod = if factors.len() == 1 { factors.pop().unwrap() } else { M::Prod(factors) };
                if info.contracted.is_empty() { prod } else { M::Sum(info.contracted.iter().map(M::code).collect(), Box::new(prod)) }
            }
            Expr::Binary { op: op @ (BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div), lhs, rhs, .. } => {
                let b = match op {
                    BinOp::Add => B::Add,
                    BinOp::Sub => B::Sub,
                    BinOp::Mul => B::Mul,
                    _ => B::Div,
                };
                let (l, r) = (side(self, lhs), side(self, rhs));
                M::bin(b, l, r)
            }
            Expr::Binary { op: BinOp::Pow, lhs, rhs, .. } => {
                let (l, r) = (side(self, lhs), self.tensor(rhs));
                M::Pow(Box::new(l), Box::new(r))
            }
            Expr::Unary { op: UnOp::Neg, expr, .. } => M::Neg(Box::new(side(self, expr))),
            Expr::Ascribe { expr, .. } => side(self, expr),
            Expr::Call { args, span, .. } if matches!(self.typed.callees.get(span), Some(Callee::Builtin(n)) if ELEMENTWISE.contains(&n.as_str()) || n == "real") => {
                let Some(Callee::Builtin(n)) = self.typed.callees.get(span) else { unreachable!("matched above") };
                let (pos, _, _) = Self::split_args(args);
                let Some(a) = pos.first() else { return self.gap(e.span()) };
                let a = side(self, a);
                match n.as_str() {
                    "exp" => M::Fun(F::Exp, Box::new(a)),
                    "log" => M::Fun(F::Ln, Box::new(a)),
                    "tanh" => M::Fun(F::Tanh, Box::new(a)),
                    "sin" => M::Fun(F::Sin, Box::new(a)),
                    "cos" => M::Fun(F::Cos, Box::new(a)),
                    "sqrt" => M::Fun(F::Sqrt, Box::new(a)),
                    "abs" => M::Fun(F::Abs, Box::new(a)),
                    "real" => a,
                    other => M::op(other, vec![], vec![a]),
                }
            }
            Expr::Lit { .. } => self.tensor(e),
            _ => {
                let t = self.tensor(e);
                if terms.is_empty() { t } else { M::index(t, terms.to_vec()) }
            }
        };
        self.formulas.entry(e.span()).or_insert_with(|| m.clone());
        m
    }
}

/// Whether `e` holds an `ein` that only elementwise operations read: then
/// its equation is written at an index.
fn ein_at_top(e: &Expr) -> bool {
    match e {
        Expr::Ein { .. } => true,
        Expr::Binary { op: BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Pow, lhs, rhs, .. } => ein_at_top(lhs) || ein_at_top(rhs),
        Expr::Unary { expr, .. } | Expr::Ascribe { expr, .. } => ein_at_top(expr),
        _ => false,
    }
}

/// The index of a tensor of shape `s` named by its axes: the name of a
/// named axis, `i_k` for the `k`th axis without one, `…` for the prefix.
fn fresh_terms(s: &Shape) -> Vec<M> {
    let mut t = Vec::new();
    if s.prefix.is_some() {
        t.push(M::Ellipsis);
    }
    for (k, a) in s.axes.iter().enumerate() {
        t.push(match &a.name {
            Some(n) => M::code(n),
            None => M::subscript(M::name("i"), M::num((k + 1).to_string())),
        });
    }
    t
}

/// The restriction of an index of `from` to an operand of shape `to`, by
/// the broadcast's alignment (definition `den-align`): an axis to the axis
/// of its name, else to the axis at the same distance from the end.
fn align(terms: &[M], from: &Shape, to: &Shape) -> Vec<M> {
    let off = usize::from(from.prefix.is_some());
    let fa = &from.axes;
    let mut out = Vec::new();
    if to.prefix.is_some() {
        out.push(M::Ellipsis);
    }
    for (j, a) in to.axes.iter().enumerate() {
        if let Some(n) = &a.name
            && let Some(p) = fa.iter().position(|b| b.name.as_deref() == Some(n.as_str()))
        {
            out.push(terms[off + p].clone());
            continue;
        }
        let k = to.axes.len() - 1 - j;
        out.push(if k < fa.len() { terms[off + fa.len() - 1 - k].clone() } else { M::Ellipsis });
    }
    out
}

fn pat_names(p: &Pat, out: &mut HashSet<String>) {
    match p {
        Pat::Ident(i) => {
            out.insert(i.name.clone());
        }
        Pat::Tuple { elems, .. } => elems.iter().for_each(|e| pat_names(e, out)),
    }
}

fn collect_lets(e: &Expr, out: &mut HashSet<String>) {
    match e {
        Expr::Let { pat, init, body, .. } => {
            pat_names(pat, out);
            collect_lets(init, out);
            collect_lets(body, out);
        }
        Expr::Lambda { pat, body, .. } => {
            pat_names(pat, out);
            collect_lets(body, out);
        }
        _ => {}
    }
}
