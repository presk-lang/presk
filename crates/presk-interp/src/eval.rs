//! The evaluator: frames with a dimension environment and variable
//! bindings, calls with dimension instantiation, closures, and re-tagging of
//! axis names from the types of the bindings.

use std::collections::{BTreeMap, HashMap};

use presk_core::ir::{Func, Lit, Rhs};
use presk_core::{FuncId, Program, VarId};
use presk_syntax::Span;
use presk_types::dim::{DimExpr, DimVar};
use presk_types::elab::ElabProgram;
use presk_types::ty::{Shape, Ty};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

use crate::key::Key;
use crate::real::Real;
use crate::value::{Axis, Tensor, Value};

/// A runtime error, with the span of the binding being evaluated.
#[derive(Debug, Clone)]
pub struct RunError {
    pub message: String,
    pub span: Option<Span>,
}

pub type Result<T> = std::result::Result<T, RunError>;

pub fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(RunError { message: msg.into(), span: None })
}

pub type Dims = BTreeMap<DimVar, u64>;

pub struct Interp<'p, R: Real> {
    pub prog: &'p Program,
    pub elab: &'p ElabProgram,
    pub prec: u32,
    /// Stream used by `sample` (the `Rand` effect without an explicit key in 0.1.0).
    pub sample_rng: ChaCha20Rng,
    /// Whether `dropout` drops: off, its mask is all ones, which is the
    /// expectation of the random function it denotes.
    pub training: bool,
    /// When set, the largest `|x|` seen at each site, `"<function>/%<let>"`
    /// — the magnitude envelope of plan 0.5.0 §3.2. It is measured here,
    /// on the exact interpreter, because the bound it feeds is a statement
    /// about the *denotation* and must not be read off a rounded run.
    pub magnitudes: Option<BTreeMap<String, f64>>,
    /// Alongside it, the smallest non-zero `|x|` at each site. A division or
    /// a square root cannot be bounded from a maximum: the error of `a / b`
    /// grows as `b` shrinks, so the analysis needs a floor as well as a
    /// ceiling (0.5.0 step 4).
    pub minima: Option<BTreeMap<String, f64>>,
    /// And `Σ |x|` at each site. The sensitivity analysis wants a **1-norm**
    /// of an adjoint, not a maximum: perturbing every element of `v` by `δ`
    /// moves the output by `‖∂L/∂v‖₁ · δ`, which is the quantity a budget is
    /// allocated against (0.5.0 step 5).
    pub sums: Option<BTreeMap<String, f64>>,
    /// And how many real elements each site holds. The solver's objective is
    /// traffic, and traffic is elements times bytes per element, so the count
    /// belongs with the magnitudes (0.5.0 step 6).
    pub elems: Option<BTreeMap<String, u64>>,
    _r: std::marker::PhantomData<R>,
}

impl<'p, R: Real> Interp<'p, R> {
    pub fn new(prog: &'p Program, elab: &'p ElabProgram, prec: u32, seed: u64) -> Self {
        Interp { prog, elab, prec, sample_rng: ChaCha20Rng::seed_from_u64(seed ^ 0x5a5a_5a5a), training: false, magnitudes: None, minima: None, sums: None, elems: None, _r: std::marker::PhantomData }
    }

    /// Evaluate a dimension expression under a dimension environment.
    pub fn dim(&self, e: &DimExpr, dims: &Dims) -> Result<u64> {
        let map: BTreeMap<DimVar, presk_types::dim::Rational> = dims.iter().map(|(v, n)| (*v, presk_types::dim::rat(*n as i64))).collect();
        match e.eval(&map) {
            Some(r) if r.is_integer() && !num_traits::Signed::is_negative(&r) => Ok(num_traits::ToPrimitive::to_u64(&r.to_integer()).unwrap_or(0)),
            Some(r) => err(format!("dimension `{}` evaluates to {r}, not a natural number", e.display(&|v| self.elab.dim_name(v)))),
            None => {
                let missing: Vec<String> = e.vars().filter(|v| !dims.contains_key(v)).map(|v| self.elab.dim_name(v)).collect();
                err(format!("unbound dimension{} {}", if missing.len() == 1 { "" } else { "s" }, missing.join(", ")))
            }
        }
    }

    /// The runtime axes of a shape type; an open prefix `...` is replaced by
    /// `prefix` (the axes given to `presk run` for the entry point).
    pub fn axes_of(&self, s: &Shape, dims: &Dims) -> Result<Vec<Axis>> {
        self.axes_of_with(s, dims, &[])
    }

    pub fn axes_of_with(&self, s: &Shape, dims: &Dims, prefix: &[Axis]) -> Result<Vec<Axis>> {
        let mut axes = Vec::new();
        if s.prefix.is_some() {
            if prefix.is_empty() {
                return err("cannot materialize a shape with an open prefix `...` (give its axes to `presk run` with extra `--dim` names)");
            }
            axes.extend(prefix.iter().cloned());
        }
        for a in &s.axes {
            axes.push(Axis { name: a.name.clone(), len: self.dim(&a.len, dims)? as usize });
        }
        Ok(axes)
    }

    /// Call a function with instantiated dimensions and argument values.
    pub fn call(&mut self, f: FuncId, dims: Dims, args: Vec<Value<R>>) -> Result<Value<R>> {
        let func: &Func = self.prog.func(f);
        if func.params.len() != args.len() {
            return err(format!("`{}` called with {} arguments, {} expected", func.name, args.len(), func.params.len()));
        }
        let mut env: HashMap<VarId, Value<R>> = HashMap::new();
        for (p, a) in func.params.iter().zip(args) {
            env.insert(p.var, a);
        }
        let mut closures: HashMap<VarId, (FuncId, Vec<Value<R>>)> = HashMap::new();
        for l in &func.body.lets {
            let v = self.rhs(&l.rhs, &l.ty, &dims, &env, &closures).map_err(|e| RunError {
                span: e.span.or(Some(l.span)),
                // Name the function once, from the innermost frame.
                message: if e.message.starts_with("in `") { e.message } else { format!("in `{}` ({}): {}", func.name, l.name.clone().unwrap_or_else(|| format!("%{}", l.var.0)), e.message) },
            })?;
            if let Rhs::Closure { f, captured } = &l.rhs {
                closures.insert(l.var, (*f, captured.iter().map(|c| env[c].clone()).collect()));
            }
            let v = self.retag(v, &l.ty, &dims);
            if self.magnitudes.is_some() || self.sums.is_some() || self.minima.is_some() || self.elems.is_some() {
                let site = format!("{}/%{}", func.name, l.var.0);
                let m = max_abs(&v);
                if self.magnitudes.is_some() && m.is_finite() {
                    let e = self.magnitudes.as_mut().expect("just checked").entry(site.clone()).or_insert(0.0);
                    if m > *e {
                        *e = m;
                    }
                }
                if let Some(el) = self.elems.as_mut() {
                    let n = count_reals(&v) as u64;
                    let e = el.entry(site.clone()).or_insert(0);
                    if n > *e {
                        *e = n;
                    }
                }
                if self.sums.is_some() {
                    let t = sum_abs(&v);
                    if t.is_finite() {
                        let e = self.sums.as_mut().expect("just checked").entry(site.clone()).or_insert(0.0);
                        if t > *e {
                            *e = t;
                        }
                    }
                }
                if let Some(mins) = self.minima.as_mut() {
                    let lo = min_abs_nonzero(&v);
                    if lo.is_finite() && lo > 0.0 {
                        let e = mins.entry(site).or_insert(f64::INFINITY);
                        if lo < *e {
                            *e = lo;
                        }
                    }
                }
            }
            env.insert(l.var, v);
        }
        env.remove(&func.body.result).ok_or_else(|| RunError { message: "result variable undefined".into(), span: Some(func.span) })
    }

    fn rhs(&mut self, rhs: &Rhs, ty: &Ty, dims: &Dims, env: &HashMap<VarId, Value<R>>, closures: &HashMap<VarId, (FuncId, Vec<Value<R>>)>) -> Result<Value<R>> {
        let get = |v: &VarId| -> Result<Value<R>> { env.get(v).cloned().ok_or_else(|| RunError { message: format!("variable %{} undefined", v.0), span: None }) };
        match rhs {
            Rhs::Lit(Lit::Nat(n)) => Ok(Value::Nat(*n)),
            Rhs::Lit(Lit::Float(s)) => Ok(Value::F64(s.parse().map_err(|_| RunError { message: format!("bad float literal `{s}`"), span: None })?)),
            Rhs::Prim { op, args } => {
                let vals: Vec<Value<R>> = args.iter().map(get).collect::<Result<_>>()?;
                self.prim(op, &vals, ty, dims)
            }
            Rhs::Call { f, dim_args, args } => {
                let vals: Vec<Value<R>> = args.iter().map(get).collect::<Result<_>>()?;
                let mut callee_dims = Dims::new();
                for (v, e) in dim_args {
                    // A dimension the callee never uses at run time may be unconstrained
                    // at the call site (a file-level `dim` absent from its types): leave
                    // it unbound; a use would then fail at the use site.
                    if let Ok(n) = self.dim(e, dims) {
                        callee_dims.insert(*v, n);
                    }
                }
                self.call(*f, callee_dims, vals)
            }
            Rhs::Tuple(vs) => Ok(Value::Tuple(vs.iter().map(get).collect::<Result<_>>()?)),
            Rhs::Proj(v, i) => match get(v)? {
                Value::Tuple(vs) => vs.into_iter().nth(*i).ok_or_else(|| RunError { message: format!("tuple has no element {i}"), span: None }),
                other => err(format!("projection on a {}", other.kind())),
            },
            Rhs::Record(fields) => Ok(Value::Record(fields.iter().map(|(n, v)| Ok((n.clone(), get(v)?))).collect::<Result<_>>()?)),
            Rhs::Field(v, name) => match get(v)? {
                Value::Record(fs) => fs.into_iter().find(|(n, _)| n == name).map(|(_, v)| v).ok_or_else(|| RunError { message: format!("record has no field `{name}`"), span: None }),
                other => err(format!("field `{name}` of a {}", other.kind())),
            },
            Rhs::Closure { f, captured } => Ok(Value::Closure { f: *f, captured: captured.iter().map(get).collect::<Result<_>>()? }),
            Rhs::Grad(_) => err("`grad` reached the interpreter: run automatic differentiation first"),
            Rhs::Fold { body, dim_args, n, stacked, init, extra, index, reverse, scan } => {
                let count = self.dim(n, dims)? as usize;
                let sts: Vec<Value<R>> = stacked.iter().map(get).collect::<Result<_>>()?;
                let ex: Vec<Value<R>> = extra.iter().map(get).collect::<Result<_>>()?;
                let mut acc = get(init)?;
                let mut callee_dims = Dims::new();
                for (v, e) in dim_args {
                    if let Ok(n) = self.dim(e, dims) {
                        callee_dims.insert(*v, n);
                    }
                }
                let order: Vec<usize> = if *reverse { (0..count).rev().collect() } else { (0..count).collect() };
                let mut pre: Vec<(usize, Value<R>)> = Vec::new();
                for i in order {
                    if *scan {
                        pre.push((i, acc.clone()));
                    }
                    let mut args = Vec::with_capacity(sts.len() + ex.len() + 2);
                    if *index {
                        args.push(Value::Nat(i as u64));
                    }
                    for s in &sts {
                        args.push(self.slice_tree(s, i));
                    }
                    args.push(acc);
                    args.extend(ex.iter().cloned());
                    acc = self.call(*body, callee_dims.clone(), args)?;
                }
                if *scan {
                    pre.sort_by_key(|(i, _)| *i);
                    let parts: Vec<Value<R>> = pre.into_iter().map(|(_, v)| v).collect();
                    let stacked = self.stack_trees(&parts)?;
                    Ok(Value::Tuple(vec![acc, stacked]))
                } else {
                    Ok(acc)
                }
            }
            Rhs::Apply { fun, args } => {
                let (f, captured) = match closures.get(fun) {
                    Some(c) => c.clone(),
                    None => match get(fun)? {
                        Value::Closure { f, captured } => (f, captured),
                        other => return err(format!("applying a {}", other.kind())),
                    },
                };
                let mut all = captured;
                for a in args {
                    all.push(get(a)?);
                }
                // A lambda shares the dimensions of the function it was lifted from.
                self.call(f, dims.clone(), all)
            }
        }
    }

    /// Give a tensor exactly the axis names of its type: for a closed type,
    /// every axis; for an open prefix, the trailing axes, the prefix keeping
    /// the names it has. Names are what the primitives select axes by, so a
    /// name must never outlive the type that gave it.
    fn retag(&self, v: Value<R>, ty: &Ty, dims: &Dims) -> Value<R> {
        fn apply<T: Clone>(t: &mut crate::value::Tensor<T>, s: &Shape) {
            let rank = t.rank();
            if s.prefix.is_none() && s.axes.len() != rank {
                return;
            }
            let k = s.axes.len().min(rank);
            for (a, ta) in t.axes[rank - k..].iter_mut().zip(&s.axes[s.axes.len() - k..]) {
                a.name = ta.name.clone();
            }
        }
        let _ = dims;
        match (v, ty) {
            (Value::Real(mut t), Ty::Real(s)) => {
                apply(&mut t, s);
                Value::Real(t)
            }
            (Value::Idx { bound, mut t }, Ty::Idx { shape, .. }) => {
                apply(&mut t, shape);
                Value::Idx { bound, t }
            }
            (v, _) => v,
        }
    }

    // ----- helpers used by the primitives -------------------------------

    pub(crate) fn real(&self, n: u64) -> R {
        R::from_u64(n, self.prec)
    }

    pub(crate) fn scalar(&self, x: R) -> Value<R> {
        Value::Real(Tensor::scalar(x))
    }

    pub(crate) fn zeros_like_tensor(&self, t: &Tensor<R>) -> Tensor<R> {
        Tensor { axes: t.axes.clone(), data: vec![R::from_u64(0, self.prec); t.numel()] }
    }

    pub(crate) fn key_of(&self, v: &Value<R>) -> Result<Key> {
        match v {
            Value::Key(k) => Ok(k.clone()),
            other => err(format!("expected a key, found a {}", other.kind())),
        }
    }
}


/// The largest `|x|` over every real leaf of a value, as an `f64`.
///
/// One number a site, even when the value is a whole parameter tree: the
/// site is the unit a realization assigns a type to (D34), so a bound that
/// is to cover it must cover its widest leaf. Indices and naturals are not
/// reals and contribute nothing; infinities are skipped by the caller, since
/// a `-inf` mask entry says nothing about the magnitude of anything.
pub fn max_abs<R: Real>(v: &Value<R>) -> f64 {
    match v {
        Value::Real(t) => t.data.iter().map(|x| x.abs().to_f64()).filter(|x| x.is_finite()).fold(0.0, f64::max),
        Value::Tuple(xs) => xs.iter().map(max_abs).fold(0.0, f64::max),
        Value::Record(xs) => xs.iter().map(|(_, x)| max_abs(x)).fold(0.0, f64::max),
        Value::F64(x) => x.abs(),
        _ => 0.0,
    }
}

/// The smallest non-zero `|x|` over every real leaf of a value, or infinity
/// when every element is zero.
///
/// Zeros are skipped rather than counted: a masked entry or a zero-
/// initialized head says nothing about how small a *divisor* gets, and a
/// floor of zero would make every quotient's bound infinite.
pub fn min_abs_nonzero<R: Real>(v: &Value<R>) -> f64 {
    match v {
        Value::Real(t) => t
            .data
            .iter()
            .map(|x| x.abs().to_f64())
            .filter(|x| *x > 0.0 && x.is_finite())
            .fold(f64::INFINITY, f64::min),
        Value::Tuple(xs) => xs.iter().map(min_abs_nonzero).fold(f64::INFINITY, f64::min),
        Value::Record(xs) => xs.iter().map(|(_, x)| min_abs_nonzero(x)).fold(f64::INFINITY, f64::min),
        Value::F64(x) if *x != 0.0 => x.abs(),
        _ => f64::INFINITY,
    }
}

/// `Σ |x|` over every real leaf of a value: the 1-norm.
pub fn sum_abs<R: Real>(v: &Value<R>) -> f64 {
    match v {
        Value::Real(t) => t.data.iter().map(|x| x.abs().to_f64()).filter(|x| x.is_finite()).sum(),
        Value::Tuple(xs) => xs.iter().map(sum_abs).sum(),
        Value::Record(xs) => xs.iter().map(|(_, x)| sum_abs(x)).sum(),
        Value::F64(x) => x.abs(),
        _ => 0.0,
    }
}

/// How many real elements a value holds, over every leaf.
pub fn count_reals<R: Real>(v: &Value<R>) -> usize {
    match v {
        Value::Real(t) => t.data.len(),
        Value::Tuple(xs) => xs.iter().map(count_reals).sum(),
        Value::Record(xs) => xs.iter().map(|(_, x)| count_reals(x)).sum(),
        _ => 0,
    }
}
