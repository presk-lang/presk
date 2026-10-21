//! The evaluator: frames with a dimension environment and variable
//! bindings, calls with dimension instantiation, closures, and re-tagging of
//! axis names from the types of the bindings.

use std::collections::{BTreeMap, HashMap};

use presk_core::ir::{Func, Lit, Rhs};
use presk_core::{FuncId, PrimOp, Program, VarId};
use presk_syntax::Span;
use presk_types::dim::{DimExpr, DimVar};
use presk_types::elab::ElabProgram;
use presk_types::ty::{Shape, Ty};

use crate::key::Key;
use crate::real::Real;
use crate::value::{Axis, Tensor, Value};

/// A runtime error, with the span of the binding being evaluated.
#[derive(Debug, Clone)]
pub struct RunError {
    /// The diagnostic's code when it is not the run's `E7001`: `E7003`, an
    /// entry refused at its dimensions ([`crate::defined`]).
    pub code: Option<&'static str>,
    pub message: String,
    pub span: Option<Span>,
}

pub type Result<T> = std::result::Result<T, RunError>;

pub fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(RunError { code: None, message: msg.into(), span: None })
}

pub type Dims = BTreeMap<DimVar, u64>;

pub struct Interp<'p, R: Real> {
    pub prog: &'p Program,
    pub elab: &'p ElabProgram,
    pub prec: u32,
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
    /// And the largest rounding error of the site's values in each storage
    /// type, `max |fl_d(x) − x|` for `d` = `f64`, `f32`, `bf16` in that order
    /// ([`max_rounding`]): what the estimate's term `u(d) · max|x|` stands
    /// for, measured on the same values (D155). It is 0 where every value is
    /// exact in the type, as 1 and 32 are in `bf16`.
    pub roundings: Option<BTreeMap<String, [f64; 3]>>,
    /// And `Σ |x|` at each site. The sensitivity analysis wants a **1-norm**
    /// of an adjoint, not a maximum: perturbing every element of `v` by `δ`
    /// moves the output by `‖∂L/∂v‖₁ · δ`, which is the quantity a budget is
    /// allocated against (0.5.0 step 5).
    pub sums: Option<BTreeMap<String, f64>>,
    /// And how many real elements each site holds. The solver's objective is
    /// traffic, and traffic is elements times bytes per element, so the count
    /// belongs with the magnitudes (0.5.0 step 6).
    pub elems: Option<BTreeMap<String, u64>>,
    /// And the primitive each site computes. The solver's objective becomes
    /// *time* in 0.6.0 step 4, and the roofline needs `W` as well as `Q`;
    /// the work of a site is its element count times the weight of its
    /// operation, and only the name of that operation is needed here.
    pub ops: Option<BTreeMap<String, String>>,
    /// And, at each sum and contraction, what its terms add up to *without*
    /// cancelling: the condition number's numerator `Σ|xᵢwᵢ|`, the
    /// root-sum-square `√Σ(xᵢwᵢ)²`, and per operand the sum the rule for
    /// that operand's error needs (0.6.0 step 6, D51). Each is a maximum over
    /// output elements. They cost a second and third evaluation of every
    /// contraction, which is why they are recorded only when asked for.
    pub terms: Option<BTreeMap<String, TermSums>>,
    /// When set, the reals bound in the environments of the calls in
    /// progress, now and at most: what the tests of liveness count (plan
    /// 0.10.1 step 1). A value bound twice counts twice.
    pub live: Option<Live>,
    /// Per function, the variables each `let` reads last ([`Deaths`]),
    /// computed at its first call.
    deaths: Vec<Option<std::sync::Arc<Deaths>>>,
    /// The zeros `zeros_like` gives, one buffer per length, shared by every
    /// value that holds them and freed with the last ([`Interp::shared_zeros`]).
    zeros: std::sync::Mutex<HashMap<usize, crate::value::WeakData<R>>>,
    _r: std::marker::PhantomData<R>,
}

/// The reals bound in the environments of the calls in progress (plan
/// 0.10.1 step 1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Live {
    pub now: usize,
    pub peak: usize,
}

/// When each variable of a function is read for the last time (plan 0.10.1
/// step 1, D105): the evaluator frees a value then, rather than when the
/// function returns. A `.vjp` keeps every partial cotangent otherwise, which
/// is what kept `presk solve` from fitting on one block of the Llama 2
/// decoder at the 7B width (D104).
#[derive(Debug, Default)]
struct Deaths {
    /// The parameters nothing reads.
    unread: Vec<VarId>,
    /// After each `let`, the variables it reads last, and its own when
    /// nothing reads it; never the function's result.
    after: Vec<Vec<VarId>>,
}

impl Deaths {
    fn of(func: &Func) -> Deaths {
        let mut last: HashMap<VarId, usize> = HashMap::new();
        for (i, l) in func.body.lets.iter().enumerate() {
            for v in l.rhs.vars() {
                last.insert(v, i);
            }
        }
        let result = func.body.result;
        let mut after: Vec<Vec<VarId>> = vec![Vec::new(); func.body.lets.len()];
        for (v, i) in &last {
            if *v != result {
                after[*i].push(*v);
            }
        }
        for (i, l) in func.body.lets.iter().enumerate() {
            if l.var != result && !last.contains_key(&l.var) {
                after[i].push(l.var);
            }
        }
        // In the order of the variables, so that a run frees them in the
        // same order every time.
        for a in &mut after {
            a.sort_by_key(|v| v.0);
        }
        let unread = func.params.iter().map(|p| p.var).filter(|v| *v != result && !last.contains_key(v)).collect();
        Deaths { unread, after }
    }
}

impl<'p, R: Real> Interp<'p, R> {
    pub fn new(prog: &'p Program, elab: &'p ElabProgram, prec: u32) -> Self {
        Interp { prog, elab, prec, training: false, magnitudes: None, minima: None, roundings: None, sums: None, elems: None, ops: None, terms: None, live: None, deaths: Vec::new(), zeros: Default::default(), _r: std::marker::PhantomData }
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
        let deaths = self.deaths_of(f);
        // A `.vjp`'s cotangent argument is the adjoint of the function's
        // result (`presk_core::ad::AdjointMap` maps the result to it): its
        // 1-norm is recorded as a site's are, so that a function's result
        // has a sensitivity too (plan 0.12.0 step 4, D119). The entry's own
        // is the sweep's seed.
        if self.sums.is_some()
            && func.name.ends_with(".vjp")
            && let (Some(p), Some(ct)) = (func.params.last(), args.last())
            && p.name == "ct"
        {
            let t = sum_abs(ct);
            let e = self.sums.as_mut().expect("just checked").entry(format!("{}/%{}", func.name, p.var.0)).or_insert(0.0);
            if t > *e {
                *e = t;
            }
        }
        let mut env: HashMap<VarId, Value<R>> = HashMap::new();
        for (p, a) in func.params.iter().zip(args) {
            self.bind(&mut env, p.var, a);
        }
        for v in &deaths.unread {
            self.unbind(&mut env, v);
        }
        let mut closures: HashMap<VarId, (FuncId, Vec<Value<R>>)> = HashMap::new();
        for (li, l) in func.body.lets.iter().enumerate() {
            let dying = &deaths.after[li];
            let v = self.rhs(&l.rhs, &l.ty, &dims, &mut env, &closures, dying).map_err(|e| RunError {
                code: e.code,
                span: e.span.or(Some(l.span)),
                // Name the function once, from the innermost frame.
                message: if e.message.starts_with("in `") { e.message } else { format!("in `{}` ({}): {}", func.name, l.name.clone().unwrap_or_else(|| format!("%{}", l.var.0)), e.message) },
            })?;
            if let Rhs::Closure { f, captured } = &l.rhs {
                closures.insert(l.var, (*f, captured.iter().map(|c| env[c].clone()).collect()));
            }
            let v = self.retag(v, &l.ty, &dims);
            if self.terms.is_some()
                && let Rhs::Prim { op, args } = &l.rhs
                && let Some(kind) = TermKind::of(op)
            {
                let vals: Vec<Value<R>> = args.iter().filter_map(|a| env.get(a).cloned()).collect();
                if vals.len() == args.len() {
                    let t = self.term_sums(op, kind, &vals, &l.ty, &dims)?;
                    let site = format!("{}/%{}", func.name, l.var.0);
                    self.terms.as_mut().expect("just checked").entry(site).or_default().join(&t);
                }
            }
            if self.magnitudes.is_some() || self.sums.is_some() || self.minima.is_some() || self.roundings.is_some() || self.elems.is_some() || self.ops.is_some() {
                let site = format!("{}/%{}", func.name, l.var.0);
                let m = max_abs(&v);
                if self.magnitudes.is_some() {
                    let e = self.magnitudes.as_mut().expect("just checked").entry(site.clone()).or_insert(0.0);
                    if m > *e {
                        *e = m;
                    }
                }
                if let Some(rs) = self.roundings.as_mut() {
                    let r = max_rounding(&v);
                    let e = rs.entry(site.clone()).or_insert([0.0; 3]);
                    for (a, b) in e.iter_mut().zip(r) {
                        *a = a.max(b);
                    }
                }
                if let Some(o) = self.ops.as_mut()
                    && let Rhs::Prim { op, .. } = &l.rhs
                {
                    o.insert(site.clone(), op.name().to_string());
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
                    let e = self.sums.as_mut().expect("just checked").entry(site.clone()).or_insert(0.0);
                    if t > *e {
                        *e = t;
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
            self.bind(&mut env, l.var, v);
            // Freed after the `let` that reads them last, the statistics
            // above having read this one's operands (plan 0.10.1 step 1).
            for d in dying {
                self.unbind(&mut env, d);
                closures.remove(d);
            }
        }
        let out = self.unbind(&mut env, &func.body.result).ok_or_else(|| RunError { code: None, message: "result variable undefined".into(), span: Some(func.span) });
        if let Some(live) = self.live.as_mut() {
            live.now -= env.values().map(count_reals).sum::<usize>();
        }
        out
    }

    /// The variables each `let` of `f` reads last, computed once.
    fn deaths_of(&mut self, f: FuncId) -> std::sync::Arc<Deaths> {
        let i = f.0 as usize;
        if self.deaths.len() <= i {
            self.deaths.resize(i + 1, None);
        }
        let prog = self.prog;
        self.deaths[i].get_or_insert_with(|| std::sync::Arc::new(Deaths::of(prog.func(f)))).clone()
    }

    fn bind(&mut self, env: &mut HashMap<VarId, Value<R>>, var: VarId, v: Value<R>) {
        if let Some(live) = self.live.as_mut() {
            live.now += count_reals(&v);
            live.peak = live.peak.max(live.now);
        }
        if let Some(old) = env.insert(var, v)
            && let Some(live) = self.live.as_mut()
        {
            live.now -= count_reals(&old);
        }
    }

    fn unbind(&mut self, env: &mut HashMap<VarId, Value<R>>, var: &VarId) -> Option<Value<R>> {
        let v = env.remove(var)?;
        if let Some(live) = self.live.as_mut() {
            live.now -= count_reals(&v);
        }
        Some(v)
    }

    /// The values of `args`, each moved out of the environment at its last
    /// read (`dying`, its last occurrence in `args`) and cloned otherwise:
    /// a callee then holds the only reference to what its caller no longer
    /// reads, and frees it when it is done with it.
    fn take(&mut self, env: &mut HashMap<VarId, Value<R>>, args: &[VarId], dying: &[VarId]) -> Result<Vec<Value<R>>> {
        let mut out = Vec::with_capacity(args.len());
        for (i, v) in args.iter().enumerate() {
            let last = dying.contains(v) && !args[i + 1..].contains(v);
            let x = if last { self.unbind(env, v) } else { env.get(v).cloned() };
            out.push(x.ok_or_else(|| RunError { code: None, message: format!("variable %{} undefined", v.0), span: None })?);
        }
        Ok(out)
    }

    /// One right-hand side. The operands of a call, an application, a fold,
    /// a tuple and a record are moved out of `env` at their last read
    /// (`dying`); a primitive's and a closure's are cloned, since the
    /// statistics and the closure table read them from `env` after.
    fn rhs(&mut self, rhs: &Rhs, ty: &Ty, dims: &Dims, env: &mut HashMap<VarId, Value<R>>, closures: &HashMap<VarId, (FuncId, Vec<Value<R>>)>, dying: &[VarId]) -> Result<Value<R>> {
        match rhs {
            Rhs::Call { f, dim_args, args } => {
                let vals = self.take(env, args, dying)?;
                let mut callee_dims = Dims::new();
                for (v, e) in dim_args {
                    // A dimension the callee never uses at run time may be unconstrained
                    // at the call site (a file-level `dim` absent from its types): leave
                    // it unbound; a use would then fail at the use site.
                    if let Ok(n) = self.dim(e, dims) {
                        callee_dims.insert(*v, n);
                    }
                }
                return self.call(*f, callee_dims, vals);
            }
            Rhs::Tuple(vs) => return Ok(Value::tuple(self.take(env, vs, dying)?)),
            // A stack of parameters, its parts moved into it (no statistic
            // reads a stack's operands).
            Rhs::Prim { op: PrimOp::StackParams { n }, args } if !args.is_empty() => {
                if args.len() != *n as usize {
                    return err("stack_params: wrong number of parts");
                }
                let vals = self.take(env, args, dying)?;
                return self.stack_trees(vals);
            }
            Rhs::Record(fields) => {
                let vars: Vec<VarId> = fields.iter().map(|(_, v)| *v).collect();
                let vals = self.take(env, &vars, dying)?;
                return Ok(Value::record(fields.iter().map(|(n, _)| n.clone()).zip(vals).collect()));
            }
            Rhs::Apply { fun, args } => {
                let (f, captured) = match closures.get(fun) {
                    Some(c) => c.clone(),
                    None => match env.get(fun).cloned() {
                        Some(Value::Closure { f, captured }) => (f, captured),
                        Some(other) => return err(format!("applying a {}", other.kind())),
                        None => return err(format!("variable %{} undefined", fun.0)),
                    },
                };
                let mut all = captured;
                all.extend(self.take(env, args, dying)?);
                // A lambda shares the dimensions of the function it was lifted from.
                return self.call(f, dims.clone(), all);
            }
            Rhs::Fold { body, dim_args, n, stacked, init, extra, index, reverse, scan } => {
                let count = self.dim(n, dims)? as usize;
                let mut ins: Vec<VarId> = stacked.clone();
                ins.push(*init);
                ins.extend(extra.iter().copied());
                let mut vals = self.take(env, &ins, dying)?;
                let ex: Vec<Value<R>> = vals.split_off(stacked.len() + 1);
                let mut acc = vals.pop().expect("the fold's initial value");
                let sts = vals;
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
                    // No iteration: the stack of none, from the fold's type.
                    let stacked = match (parts.is_empty(), ty) {
                        (true, Ty::Tuple(ts)) if ts.len() == 2 => self.empty_stack(&ts[1], dims)?,
                        _ => self.stack_trees(parts)?,
                    };
                    return Ok(Value::tuple(vec![acc, stacked]));
                }
                return Ok(acc);
            }
            _ => {}
        }
        let env: &HashMap<VarId, Value<R>> = env;
        let look = |v: &VarId| -> Result<&Value<R>> { env.get(v).ok_or_else(|| RunError { code: None, message: format!("variable %{} undefined", v.0), span: None }) };
        let get = |v: &VarId| -> Result<Value<R>> { look(v).cloned() };
        match rhs {
            Rhs::Lit(Lit::Nat(n)) => Ok(Value::Nat(*n)),
            Rhs::Lit(Lit::Float(s)) => Ok(Value::F64(s.parse().map_err(|_| RunError { code: None, message: format!("bad float literal `{s}`"), span: None })?)),
            Rhs::Prim { op, args } => {
                let vals: Vec<Value<R>> = args.iter().map(get).collect::<Result<_>>()?;
                self.prim(op, &vals, ty, dims)
            }
            // A projection or a field clones the element it selects and
            // nothing else (0.6.0 step 8).
            Rhs::Proj(v, i) => match look(v)? {
                Value::Tuple(vs) => vs.get(*i).cloned().ok_or_else(|| RunError { code: None, message: format!("tuple has no element {i}"), span: None }),
                other => err(format!("projection on a {}", other.kind())),
            },
            Rhs::Field(v, name) => match look(v)? {
                Value::Record(fs) => fs.iter().find(|(n, _)| n == name).map(|(_, v)| v.clone()).ok_or_else(|| RunError { code: None, message: format!("record has no field `{name}`"), span: None }),
                other => err(format!("field `{name}` of a {}", other.kind())),
            },
            Rhs::Closure { f, captured } => Ok(Value::Closure { f: *f, captured: captured.iter().map(get).collect::<Result<_>>()? }),
            Rhs::Grad(_) => err("`grad` reached the interpreter: run automatic differentiation first"),
            Rhs::Call { .. } | Rhs::Tuple(_) | Rhs::Record(_) | Rhs::Apply { .. } | Rhs::Fold { .. } => unreachable!("evaluated above"),
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
            (Value::Idx { bound, mut t }, Ty::Idx { shape, .. } | Ty::Truth(shape)) => {
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
        Tensor { axes: t.axes.clone(), data: vec![R::from_u64(0, self.prec); t.numel()].into() }
    }

    /// Zeros of `t`'s axes, the elements shared with every other value of
    /// zeros of that length while one is alive (plan 0.10.1 step 2). A
    /// gradient assembles a parameter tree's cotangent field by field, each
    /// on `zeros_like θ` (`set_field`, then `add`): with zeros of their own,
    /// every field of every one of those trees was a parameter-sized tensor
    /// of exact zeros, three or four copies of θ at once in `presk solve`'s
    /// sweep. A write to them copies them first, as to any shared buffer.
    pub(crate) fn shared_zeros(&self, t: &Tensor<R>) -> Tensor<R> {
        let n = t.numel();
        let mut zeros = self.zeros.lock().unwrap_or_else(|e| e.into_inner());
        let data = match zeros.get(&n).and_then(|w| w.upgrade()) {
            Some(z) => z,
            None => {
                let z = crate::value::Data::from(vec![R::from_u64(0, self.prec); n]);
                zeros.insert(n, z.downgrade());
                z
            }
        };
        Tensor { axes: t.axes.clone(), data }
    }

    /// Whether `t`'s elements are the shared zeros of their length.
    pub(crate) fn is_shared_zeros(&self, t: &Tensor<R>) -> bool {
        let zeros = self.zeros.lock().unwrap_or_else(|e| e.into_inner());
        zeros.get(&t.data.len()).and_then(|w| w.upgrade()).is_some_and(|z| z.same(&t.data))
    }

    /// The sums of terms of one sum or contraction, by evaluating it again on
    /// transformed operands. Every primitive this applies to is linear in
    /// each operand it sums over, so on `|x|` and `|w|` it computes
    /// `Σ|xᵢwᵢ|` per output element, on `x²` and `w²` it computes
    /// `Σ(xᵢwᵢ)²`, and with operand `j` replaced by ones and the others by
    /// their absolute values it computes `Σ Π_{i≠j}|·|` — the factor that
    /// multiplies operand `j`'s error in a normwise rule.
    fn term_sums(&mut self, op: &PrimOp, kind: TermKind, vals: &[Value<R>], ty: &Ty, dims: &Dims) -> Result<TermSums> {
        let summed = match kind {
            TermKind::Linear => 1,
            TermKind::Multilinear => vals.len(),
        };
        let with = |f: &dyn Fn(&R) -> R, which: &dyn Fn(usize) -> bool| -> Vec<Value<R>> {
            vals.iter()
                .enumerate()
                .map(|(i, v)| if which(i) { v.map_leaves(&mut |leaf| match leaf { Value::Real(t) => Value::Real(t.map(|x| f(x))), other => other.clone() }) } else { v.clone() })
                .collect()
        };
        let one = R::from_u64(1, self.prec);
        let s = self.prim(op, &with(&|x: &R| x.abs(), &|i| i < summed), ty, dims)?;
        let mut t = TermSums { abs: max_abs(&s), rss: 0.0, norms: Vec::new() };
        if matches!(op, PrimOp::SumSquares) {
            // Its terms are `xᵢ²`: the squares of the terms are `xᵢ⁴`, and
            // the factor on `x`'s error is `2Σ|xᵢ|`.
            let q = self.prim(op, &with(&|x: &R| x.mul(x), &|_| true), ty, dims)?;
            t.rss = max_abs(&q).sqrt();
            t.norms = vec![2.0 * vals.iter().map(sum_abs).sum::<f64>()];
            return Ok(t);
        }
        let q = self.prim(op, &with(&|x: &R| x.mul(x), &|i| i < summed), ty, dims)?;
        t.rss = max_abs(&q).sqrt();
        if kind == TermKind::Multilinear {
            for j in 0..summed {
                let args: Vec<Value<R>> = with(&|x: &R| x.abs(), &|i| i != j)
                    .into_iter()
                    .enumerate()
                    .map(|(i, v)| if i == j { v.map_leaves(&mut |leaf| match leaf { Value::Real(t) => Value::Real(t.map(|_| one.clone())), other => other.clone() }) } else { v })
                    .collect();
                t.norms.push(max_abs(&self.prim(op, &args, ty, dims)?));
            }
        }
        Ok(t)
    }

    pub(crate) fn key_of(&self, v: &Value<R>) -> Result<Key> {
        match v {
            Value::Key(k) => Ok(k.clone()),
            other => err(format!("expected a key, found a {}", other.kind())),
        }
    }
}


/// What a sum or contraction adds up to without cancelling (0.6.0 step 6).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TermSums {
    /// `max Σ|xᵢwᵢ|` over output elements: the condition number's numerator.
    pub abs: f64,
    /// `max √Σ(xᵢwᵢ)²`: what independent errors in the terms add up to.
    pub rss: f64,
    /// Per summed operand `j`, `max Σ Π_{i≠j}|·|`: the normwise factor on
    /// that operand's error. Empty for a plain sum, whose factor is `k`.
    pub norms: Vec<f64>,
}

impl TermSums {
    /// The pointwise maximum, since the envelope must cover every call.
    pub fn join(&mut self, o: &TermSums) {
        self.abs = self.abs.max(o.abs);
        self.rss = self.rss.max(o.rss);
        if self.norms.len() < o.norms.len() {
            self.norms.resize(o.norms.len(), 0.0);
        }
        for (a, b) in self.norms.iter_mut().zip(&o.norms) {
            *a = a.max(*b);
        }
    }
}

/// How a primitive sums: over its first operand only, or over the product of
/// all of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TermKind {
    Linear,
    Multilinear,
}

impl TermKind {
    fn of(op: &PrimOp) -> Option<TermKind> {
        match op {
            PrimOp::Dot | PrimOp::Matmul | PrimOp::Ein { .. } | PrimOp::DotVjpL | PrimOp::DotVjpR => Some(TermKind::Multilinear),
            PrimOp::SumAll | PrimOp::Sum { .. } | PrimOp::SumTo | PrimOp::MeanAll | PrimOp::Mean { .. } | PrimOp::SumSquares => Some(TermKind::Linear),
            _ => None,
        }
    }
}

/// The largest `|x|` over every real leaf of a value, as an `f64`.
///
/// One number a site, even when the value is a whole parameter tree: the
/// site is the unit a realization assigns a type to (D34), so a bound that
/// is to cover it must cover its widest leaf. Indices, masks and naturals are
/// not reals and contribute nothing. An infinity is a magnitude like any
/// other: it was skipped while a mask was a real tensor holding `-∞`, which
/// no tensor holds since plan 0.10.0 step 7b.
pub fn max_abs<R: Real>(v: &Value<R>) -> f64 {
    match v {
        Value::Real(t) => t.data.iter().map(|x| x.abs().to_f64()).fold(0.0, f64::max),
        Value::Tuple(xs) => xs.iter().map(max_abs).fold(0.0, f64::max),
        Value::Record(xs) => xs.iter().map(|(_, x)| max_abs(x)).fold(0.0, f64::max),
        Value::F64(x) => x.abs(),
        _ => 0.0,
    }
}

/// The largest rounding error over every real leaf of a value, in each
/// storage type: `max |fl_d(x) − x|` for `d` = `f64`, `f32`, `bf16`.
///
/// Each `x` is rounded as the runtime rounds it: to the nearest `f64`, that
/// to the nearest `f32`, and that to the nearest `bf16`, ties to even on the
/// `f32`'s bits (`presk_rt::f32_to_bf16`, which this crate cannot call: the
/// runtime depends on it). The difference is taken at the value's own
/// precision, so the `f64` term is the 128-bit value's distance to its double,
/// and a value exact in a type contributes 0 to that type's term. Never more
/// than `u(d) · max|x|`, the term the estimate used before D155. A value that
/// overflows a type has an infinite error there; a NaN is skipped.
pub fn max_rounding<R: Real>(v: &Value<R>) -> [f64; 3] {
    let join = |a: [f64; 3], b: [f64; 3]| [a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2])];
    match v {
        Value::Real(t) => t.data.iter().map(|x| rounding_of(x)).fold([0.0; 3], join),
        Value::Tuple(xs) => xs.iter().map(max_rounding).fold([0.0; 3], join),
        Value::Record(xs) => xs.iter().map(|(_, x)| max_rounding(x)).fold([0.0; 3], join),
        Value::F64(x) => rounding_of_f64(*x),
        _ => [0.0; 3],
    }
}

/// One value's rounding error in `f64`, `f32` and `bf16`.
fn rounding_of<R: Real>(x: &R) -> [f64; 3] {
    let x64 = x.to_f64();
    if x64.is_nan() {
        return [0.0; 3];
    }
    let p = x.prec();
    let off = |y: f64| if y.is_finite() { x.sub(&R::from_f64(y, p)).abs().to_f64() } else { f64::INFINITY };
    let x32 = x64 as f32;
    [off(x64), off(x32 as f64), off(bf16_of(x32) as f64)]
}

/// The same for a machine double, whose `f64` error is 0.
pub fn rounding_of_f64(x: f64) -> [f64; 3] {
    if x.is_nan() {
        return [0.0; 3];
    }
    let x32 = x as f32;
    let off = |y: f32| if y.is_finite() { (x - y as f64).abs() } else { f64::INFINITY };
    [0.0, off(x32), off(bf16_of(x32))]
}

/// The `bf16` nearest an `f32`, ties to even, as an `f32`: the runtime's
/// `f32_to_bf16` followed by `bf16_to_f32`.
pub fn bf16_of(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    f32::from_bits(((bits + 0x7fff + lsb) >> 16) << 16)
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
        Value::Real(t) => t.data.iter().map(|x| x.abs().to_f64()).sum(),
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
