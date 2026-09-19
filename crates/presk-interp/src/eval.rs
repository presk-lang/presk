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
    _r: std::marker::PhantomData<R>,
}

impl<'p, R: Real> Interp<'p, R> {
    pub fn new(prog: &'p Program, elab: &'p ElabProgram, prec: u32, seed: u64) -> Self {
        Interp { prog, elab, prec, sample_rng: ChaCha20Rng::seed_from_u64(seed ^ 0x5a5a_5a5a), _r: std::marker::PhantomData }
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
