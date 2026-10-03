//! An entry refused at the dimensions where an operation of its body is
//! undefined (plan 0.10.0 step 7b, item 4; the reference's definition
//! "Undefined at dimensions" and requirement "Entries", its table of
//! conditions).
//!
//! Before anything is computed, the body of the entry is walked under the
//! ground dimensions it is given, and so are the bodies its calls name (under
//! each call's instantiation), the closures it applies, and every `fold` whose
//! count is at least one. No value is computed: each binding's type is
//! evaluated at the dimensions, an open prefix `...` bound, as the
//! type-checker bound it, to the leading axes of the argument whose parameter
//! has it. An operation whose condition on dimensions fails refuses the entry
//! (`E7003`, at the operation's span): `mean` over no element, `mean axis a`
//! over an axis of length 0, a selection outside its axis, and `embed`,
//! `cross_entropy`, `argmax` and `sample` with no index below `V`; a
//! derivative takes its operation's condition. So does an entry a type of
//! whose parameters has no value (indices below 0 in a tensor that has
//! elements). What depends on values, a division by zero or a masked softmax
//! whose mask empties a row, stays an error of the run.
//!
//! A length the walk cannot evaluate (a dimension a call leaves unbound, a
//! prefix it cannot bind) decides nothing: the check refuses only what it
//! proves undefined.

use std::collections::{BTreeMap, HashMap, HashSet};

use presk_core::ir::{Func, Rhs, unstack_ty};
use presk_core::{FuncId, PrimOp, Program, VarId};
use presk_types::dim::{DimExpr, DimVar, Rational, rat};
use presk_types::elab::ElabProgram;
use presk_types::ty::{PrefixVar, Shape, Ty};

use crate::eval::{Dims, Result, RunError};
use crate::value::Axis;

/// The code of the refusal.
pub const CODE: &str = "E7003";

/// One axis at the dimensions: its name, its length when known, and the
/// length's expression in the function's dimensions (`None` for an axis of
/// a bound prefix, which the caller's types named).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Ax {
    name: Option<String>,
    len: Option<u64>,
    expr: Option<DimExpr>,
}

/// A shape at the dimensions: its axes, all of them when `whole`, else the
/// trailing ones after a prefix that could not be bound.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CShape {
    whole: bool,
    axes: Vec<Ax>,
}

/// A value's type at the dimensions, as far as the conditions read it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum CTy {
    Real(CShape),
    Idx { bound: Option<u64>, bound_expr: DimExpr, shape: CShape },
    Tuple(Vec<CTy>),
    /// A function value: a closure of `f` over values of these types.
    Fn(FuncId, Vec<CTy>),
    Other,
}

/// Refuse `entry` at `dims` (and the axes `prefix` gives the open prefix of
/// its parameters, when known) where an operation of its body is undefined.
pub fn check(prog: &Program, elab: &ElabProgram, entry: FuncId, dims: &Dims, prefix: Option<&[Axis]>) -> Result<()> {
    let f = prog.func(entry);
    let mut w = Walk { prog, elab, entry: source_name(&f.name), seen: HashSet::new(), stack: Vec::new() };
    let mut frame = Frame { dims: dims.clone(), prefixes: HashMap::new() };
    if let Some(axes) = prefix {
        let bound: Vec<Ax> = axes.iter().map(|a| Ax { name: a.name.clone(), len: Some(a.len as u64), expr: None }).collect();
        for p in &f.params {
            for v in prefixes_of(&p.ty) {
                frame.prefixes.insert(v, bound.clone());
            }
        }
    }
    let args: Vec<CTy> = f.params.iter().map(|p| frame.cty(&p.ty)).collect();
    for (p, a) in f.params.iter().zip(&args) {
        if let Some(why) = no_value(a) {
            let name = |v: DimVar| elab.dim_name(v);
            return Err(RunError { code: Some(CODE), message: format!("`{}` is refused at the dimensions given: its parameter `{}` has no value, {}", source_name(&f.name), p.name, why.text(&name)), span: Some(f.span) });
        }
    }
    w.func(entry, frame, args)
}

/// Why a condition fails: what is undefined and the length that makes it so.
struct Why {
    what: String,
    /// The axis or bound at fault: its expression (if the types name one)
    /// and its length.
    expr: Option<DimExpr>,
    len: i128,
}

impl Why {
    fn text(&self, name: &dyn Fn(DimVar) -> String) -> String {
        match &self.expr {
            Some(e) if !e.is_const() => format!("{} (`{}` = {})", self.what, e.display(name), self.len),
            _ => self.what.clone(),
        }
    }
}

/// A parameter type with no value at the dimensions: indices below 0 in a
/// tensor that has an element.
fn no_value(t: &CTy) -> Option<Why> {
    match t {
        CTy::Idx { bound: Some(0), bound_expr, shape } if shape.whole && shape.axes.iter().all(|a| a.len.is_some_and(|n| n > 0)) => {
            Some(Why { what: format!("an index below {} in a tensor of {} element{}", 0, numel(shape), if numel(shape) == 1 { "" } else { "s" }), expr: Some(bound_expr.clone()), len: 0 })
        }
        CTy::Tuple(ts) => ts.iter().find_map(no_value),
        _ => None,
    }
}

fn numel(s: &CShape) -> u64 {
    s.axes.iter().map(|a| a.len.unwrap_or(1)).product()
}

/// The prefix variables a type mentions.
fn prefixes_of(t: &Ty) -> Vec<PrefixVar> {
    match t {
        Ty::Real(s) | Ty::Idx { shape: s, .. } => s.prefix.into_iter().collect(),
        Ty::Tuple(ts) => ts.iter().flat_map(prefixes_of).collect(),
        _ => vec![],
    }
}

#[derive(Clone)]
struct Frame {
    dims: Dims,
    prefixes: HashMap<PrefixVar, Vec<Ax>>,
}

impl Frame {
    fn eval(&self, e: &DimExpr) -> Option<u64> {
        let map: BTreeMap<DimVar, Rational> = e.vars().filter_map(|v| self.dims.get(&v).map(|n| (v, rat(*n as i64)))).collect();
        let r = e.eval(&map)?;
        if r.is_integer() && !num_traits::Signed::is_negative(&r) { num_traits::ToPrimitive::to_u64(&r.to_integer()) } else { None }
    }

    fn shape(&self, s: &Shape) -> CShape {
        let mut axes = Vec::new();
        let mut whole = true;
        if let Some(p) = s.prefix {
            match self.prefixes.get(&p) {
                Some(pre) => axes.extend(pre.iter().cloned()),
                None => whole = false,
            }
        }
        axes.extend(s.axes.iter().map(|a| Ax { name: a.name.clone(), len: self.eval(&a.len), expr: Some(a.len.clone()) }));
        CShape { whole, axes }
    }

    fn cty(&self, t: &Ty) -> CTy {
        match t {
            Ty::Real(s) | Ty::Truth(s) => CTy::Real(self.shape(s)),
            Ty::Idx { bound, shape } => CTy::Idx { bound: self.eval(bound), bound_expr: bound.clone(), shape: self.shape(shape) },
            Ty::Tuple(ts) => CTy::Tuple(ts.iter().map(|t| self.cty(t)).collect()),
            _ => CTy::Other,
        }
    }
}

/// Bind the open prefix of a parameter to the leading axes of its argument.
fn bind(p: &Ty, a: &CTy, out: &mut HashMap<PrefixVar, Vec<Ax>>) {
    match (p, a) {
        (Ty::Real(ps) | Ty::Idx { shape: ps, .. }, CTy::Real(xs) | CTy::Idx { shape: xs, .. }) => {
            if let Some(v) = ps.prefix
                && xs.whole
                && xs.axes.len() >= ps.axes.len()
            {
                out.entry(v).or_insert_with(|| xs.axes[..xs.axes.len() - ps.axes.len()].to_vec());
            }
        }
        (Ty::Tuple(ps), CTy::Tuple(xs)) => ps.iter().zip(xs).for_each(|(p, a)| bind(p, a, out)),
        _ => {}
    }
}

struct Walk<'a> {
    prog: &'a Program,
    elab: &'a ElabProgram,
    entry: &'a str,
    /// The frames walked already: a function at its dimensions, prefixes and
    /// function values.
    seen: HashSet<String>,
    /// The functions being walked, which a call back into is not walked again.
    stack: Vec<FuncId>,
}

impl Walk<'_> {
    fn func(&mut self, id: FuncId, mut frame: Frame, args: Vec<CTy>) -> Result<()> {
        if self.stack.contains(&id) {
            return Ok(());
        }
        let f: &Func = self.prog.func(id);
        for (p, a) in f.params.iter().zip(&args) {
            bind(&p.ty, a, &mut frame.prefixes);
        }
        let key = format!("{id:?} {:?} {:?} {:?}", frame.dims, { let mut p: Vec<_> = frame.prefixes.iter().collect(); p.sort_by_key(|(v, _)| **v); p }, args.iter().filter(|a| matches!(a, CTy::Fn(..))).collect::<Vec<_>>());
        if !self.seen.insert(key) {
            return Ok(());
        }
        self.stack.push(id);
        let mut env: HashMap<VarId, CTy> = HashMap::new();
        for (p, a) in f.params.iter().zip(args) {
            env.insert(p.var, a);
        }
        for l in &f.body.lets {
            let ty = |env: &HashMap<VarId, CTy>, v: &VarId| env.get(v).cloned().unwrap_or(CTy::Other);
            let c = match &l.rhs {
                Rhs::Prim { op, args } => {
                    let ops: Vec<CTy> = args.iter().map(|v| ty(&env, v)).collect();
                    if let Some(why) = undefined(op, &ops) {
                        let name = |v: DimVar| self.elab.dim_name(v);
                        let source = source_name(&f.name);
                        let at = if source == self.entry { String::new() } else { format!("in `{source}`, ") };
                        return Err(RunError { code: Some(CODE), message: format!("`{}` is refused at the dimensions given: {at}{}", self.entry, why.text(&name)), span: Some(l.span) });
                    }
                    frame.cty(&l.ty)
                }
                Rhs::Call { f: callee, dim_args, args } => {
                    let dims: Dims = dim_args.iter().filter_map(|(v, e)| frame.eval(e).map(|n| (*v, n))).collect();
                    let vals: Vec<CTy> = args.iter().map(|v| ty(&env, v)).collect();
                    self.func(*callee, Frame { dims, prefixes: HashMap::new() }, vals)?;
                    frame.cty(&l.ty)
                }
                Rhs::Fold { body, dim_args, n, stacked, init, extra, index, .. } => {
                    if frame.eval(n).is_some_and(|n| n >= 1) {
                        let dims: Dims = dim_args.iter().filter_map(|(v, e)| frame.eval(e).map(|n| (*v, n))).collect();
                        let mut vals = Vec::new();
                        if *index {
                            vals.push(CTy::Other);
                        }
                        for s in stacked {
                            vals.push(slice(&frame, f, *s));
                        }
                        vals.push(ty(&env, init));
                        vals.extend(extra.iter().map(|v| ty(&env, v)));
                        self.func(*body, Frame { dims, prefixes: HashMap::new() }, vals)?;
                    }
                    frame.cty(&l.ty)
                }
                Rhs::Closure { f: g, captured } => CTy::Fn(*g, captured.iter().map(|v| ty(&env, v)).collect()),
                Rhs::Apply { fun, args } => {
                    if let CTy::Fn(g, caps) = ty(&env, fun) {
                        let mut vals = caps;
                        vals.extend(args.iter().map(|v| ty(&env, v)));
                        // A lambda shares the dimensions of the frame that
                        // applies it, as the interpreter's `apply` does.
                        self.func(g, frame.clone(), vals)?;
                    }
                    frame.cty(&l.ty)
                }
                Rhs::Tuple(vs) => CTy::Tuple(vs.iter().map(|v| ty(&env, v)).collect()),
                Rhs::Proj(x, i) => match ty(&env, x) {
                    CTy::Tuple(ts) if *i < ts.len() => ts[*i].clone(),
                    _ => frame.cty(&l.ty),
                },
                _ => frame.cty(&l.ty),
            };
            env.insert(l.var, c);
        }
        self.stack.pop();
        Ok(())
    }
}

/// The source function a Core function was made from: a differentiated
/// function's parts (`f.vjp`, `f.fwd`, `f.bwd`, `f.grad`, `f.value_vjp`, a
/// backward fold's body `f.fold_bwd`), a generation function's greedy copy
/// (`f.greedy`) and a lambda lifted out of it (`f.lambda0`) are named after
/// it.
fn source_name(name: &str) -> &str {
    let mut n = name;
    loop {
        let lambda = n.rsplit_once(".lambda").filter(|(_, k)| !k.is_empty() && k.bytes().all(|b| b.is_ascii_digit())).map(|(base, _)| base);
        match [".vjp", ".fwd", ".bwd", ".grad", ".value_vjp", ".fold_bwd", ".greedy"].iter().find_map(|s| n.strip_suffix(s)).or(lambda) {
            Some(base) => n = base,
            None => return n,
        }
    }
}

/// One slice of a stacked value: its type without the leading axis, as
/// [`unstack_ty`] gives it (a shape with an open prefix is its own slice's,
/// as [`presk_core::ir::stack_ty`] makes it).
fn slice(frame: &Frame, f: &Func, v: VarId) -> CTy {
    let sym = f.body.lets.iter().find(|l| l.var == v).map(|l| &l.ty).or_else(|| f.params.iter().find(|p| p.var == v).map(|p| &p.ty));
    sym.map_or(CTy::Other, |t| frame.cty(&unstack_ty(t)))
}

/// The axis named `a` of a shape, from the end.
fn axis<'s>(s: &'s CShape, a: &str) -> Option<&'s Ax> {
    s.axes.iter().rev().find(|x| x.name.as_deref() == Some(a))
}

fn shape(t: &CTy) -> Option<&CShape> {
    match t {
        CTy::Real(s) | CTy::Idx { shape: s, .. } => Some(s),
        _ => None,
    }
}

/// Where `op` is undefined at the operands' types: what, and the length.
fn undefined(op: &PrimOp, ops: &[CTy]) -> Option<Why> {
    let empty_axis = |s: &CShape| s.axes.iter().find(|a| a.len == Some(0)).cloned();
    let at = |what: String, a: &Ax| Some(Why { what, expr: a.expr.clone(), len: a.len.unwrap_or(0) as i128 });
    let named = |a: &Ax| a.name.as_deref().map_or_else(|| "an unnamed axis".to_string(), |n| format!("the axis `{n}`"));
    let mean_all = |what: &str, t: Option<&CTy>| {
        let s = shape(t?)?;
        let a = empty_axis(s)?;
        at(format!("{what} averages over no element: {} of its operand has length 0", named(&a)), &a)
    };
    let mean_axis = |what: &str, t: Option<&CTy>, ax: &str| {
        let a = axis(shape(t?)?, ax)?;
        (a.len == Some(0)).then_some(())?;
        at(format!("{what} averages over the axis `{ax}`, of length 0"), a)
    };
    let index = |what: &str, t: Option<&CTy>, ax: &str, i: i64| {
        let a = axis(shape(t?)?, ax)?;
        let n = a.len? as i128;
        let i = i as i128;
        (i < -n || i >= n).then_some(())?;
        Some(Why { what: format!("{what} selects position {i} of the axis `{ax}`, of length {n}"), expr: a.expr.clone(), len: n })
    };
    // `V`: the last axis of the logits.
    let last = |what: &str, t: Option<&CTy>| {
        let s = shape(t?)?;
        let a = s.axes.last()?;
        (a.len == Some(0)).then_some(())?;
        at(format!("{what} has no index to choose: the last axis of its operand has length 0"), a)
    };
    // `V`: the bound of the indices.
    let bound = |what: &str, t: Option<&CTy>| match t? {
        CTy::Idx { bound: Some(0), bound_expr, .. } => Some(Why { what: format!("{what} has no row to read: its indices are below 0"), expr: Some(bound_expr.clone()), len: 0 }),
        _ => None,
    };
    match op {
        PrimOp::MeanAll => mean_all("`mean`", ops.first()),
        PrimOp::Mean { axis: a } => mean_axis(&format!("`mean axis {a}`"), ops.first(), a),
        PrimOp::IndexConst { axis: a, index: i } => index(&format!("`[{a}: {i}]`"), ops.first(), a, *i),
        PrimOp::Embed => bound("`embed`", ops.get(1)),
        PrimOp::CrossEntropy => last("`cross_entropy`", ops.first()),
        PrimOp::Argmax => last("`argmax`", ops.first()),
        // The logits are its last operand, the key (if any) before them.
        PrimOp::Sample => last("`sample`", ops.last()),
        // A derivative takes its operation's condition.
        PrimOp::MeanVjp { axis: None } => mean_all("the derivative of `mean`", ops.get(1)),
        PrimOp::MeanVjp { axis: Some(a) } => mean_axis(&format!("the derivative of `mean axis {a}`"), ops.get(1), a),
        PrimOp::ScatterIndex { axis: a, index: i } => index(&format!("the derivative of `[{a}: {i}]`"), ops.first(), a, *i),
        PrimOp::EmbedVjp => bound("the derivative of `embed`", ops.get(1)),
        PrimOp::CrossEntropyVjp => last("the derivative of `cross_entropy`", ops.first()),
        _ => None,
    }
}
