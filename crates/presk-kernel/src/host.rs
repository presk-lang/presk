//! The host program of PSK-Kernel v0: PSK-Core bodies evaluated over runtime
//! values, every tensor primitive launched as a kernel specialized on its
//! operands' shapes (built on first use, cached), structural primitives,
//! initializers and sampling executed on the host with the interpreter's
//! ChaCha keys. Implements [`Backend`] as [`KernelBackend`].

use std::collections::{BTreeMap, HashMap};

use presk_core::ir::{Lit, Rhs};
use presk_core::{FuncId, PrimOp, Program, VarId};
use presk_interp::eval::Dims;
use presk_interp::key::{Key, normal, uniform01};
use presk_rt::{Axis, Backend, Buf, Prepared, Realize, RtError, Tensor, Value, err};
use presk_types::dim::DimExpr;
use presk_types::elab::ElabProgram;
use presk_types::ty::{Shape, Ty};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

use crate::exec::{output_buf, run_kernel};
use crate::ir::{BufKind, Dtype, Kernel};
use crate::lower::{self, Lowered, Operand};

type Result<T> = std::result::Result<T, RtError>;

/// How kernels are executed: the kernel-IR interpreter ([`InterpExec`]) or
/// compiled code (the Cranelift backend). `idx` identifies the kernel in the
/// backend's list, so that an executor can compile it once.
pub trait KernelExec {
    fn name(&self) -> &'static str;
    fn run(&mut self, idx: usize, k: &Kernel, inputs: &[&Buf], out: &mut Buf) -> Result<()>;
}

/// The kernel-IR interpreter as an executor.
pub struct InterpExec;

impl KernelExec for InterpExec {
    fn name(&self) -> &'static str {
        "kernel"
    }
    fn run(&mut self, _idx: usize, k: &Kernel, inputs: &[&Buf], out: &mut Buf) -> Result<()> {
        run_kernel(k, inputs, out);
        Ok(())
    }
}

pub struct KernelBackend<'p> {
    pub prog: &'p Program,
    pub elab: &'p ElabProgram,
    pub dtype: Dtype,
    /// Kernels in order of first use.
    pub kernels: Vec<Kernel>,
    outs: Vec<(Vec<Axis>, BufKind, u64)>,
    cache: HashMap<String, usize>,
    sample_rng: ChaCha20Rng,
    /// Kernel launches so far.
    pub launches: u64,
    /// Whether `dropout` drops (see the interpreter's field of the same name).
    pub training: bool,
    /// The fusion plan of each function, computed once (`presk-sched`).
    plans: HashMap<FuncId, presk_sched::Plan>,
    /// Kernel launches fusion removed: each group of `n` steps launches once
    /// where `n` kernels would have run, per leaf of a parameter tree.
    pub fused: u64,
    pub exec: Box<dyn KernelExec>,
}

impl Backend for KernelBackend<'_> {
    fn name(&self) -> &'static str {
        self.exec.name()
    }

    fn set_training(&mut self, on: bool) {
        self.training = on;
    }

    fn launch_stats(&self) -> (u64, u64) {
        (self.launches, self.fused)
    }

    fn prepare(&mut self, entry: FuncId, dims: &Dims, realize: Realize) -> Result<Prepared> {
        Ok(Prepared { func: entry, dims: dims.clone(), realize, handle: 0 })
    }

    fn call(&mut self, p: &Prepared, args: Vec<Value>) -> Result<Value> {
        let args = args.iter().map(|a| self.realize_value(a)).collect();
        self.call_fn(p.func, p.dims.clone(), args)
    }
}

impl<'p> KernelBackend<'p> {
    pub fn new(prog: &'p Program, elab: &'p ElabProgram, realize: Realize, seed: u64) -> Self {
        Self::with_exec(prog, elab, realize, seed, Box::new(InterpExec))
    }

    /// A kernel backend whose kernels are run by `exec`.
    pub fn with_exec(prog: &'p Program, elab: &'p ElabProgram, realize: Realize, seed: u64, exec: Box<dyn KernelExec>) -> Self {
        let dtype = match realize {
            Realize::F64 => Dtype::F64,
            Realize::F32 => Dtype::F32,
        };
        KernelBackend { prog, elab, dtype, kernels: vec![], outs: vec![], cache: HashMap::new(), sample_rng: ChaCha20Rng::seed_from_u64(seed ^ 0x5a5a_5a5a), launches: 0, training: false, plans: HashMap::new(), fused: 0, exec }
    }

    // ----- buffers and values ------------------------------------------------

    fn buf(&self, data: Vec<f64>) -> Buf {
        match self.dtype {
            Dtype::F64 => Buf::F64(data),
            Dtype::F32 => Buf::F32(data.into_iter().map(|x| x as f32).collect()),
        }
    }

    fn scalar(&self, x: f64) -> Value {
        Value::Real(Tensor { axes: vec![], data: self.buf(vec![x]) })
    }

    /// Every real leaf in the backend's dtype (exact inputs rounded once).
    fn realize_value(&self, v: &Value) -> Value {
        v.map_leaves(&mut |leaf| match leaf {
            Value::Real(t) => Value::Real(Tensor { axes: t.axes.clone(), data: self.buf(t.data.to_f64()) }),
            other => other.clone(),
        })
    }

    pub fn dim(&self, e: &DimExpr, dims: &Dims) -> Result<u64> {
        let map: BTreeMap<_, _> = dims.iter().map(|(v, n)| (*v, presk_types::dim::rat(*n as i64))).collect();
        match e.eval(&map) {
            Some(r) if r.is_integer() && !num_traits::Signed::is_negative(&r) => Ok(num_traits::ToPrimitive::to_u64(&r.to_integer()).unwrap_or(0)),
            Some(r) => err("E7001", format!("dimension `{}` evaluates to {r}, not a natural number", e.display(&|v| self.elab.dim_name(v)))),
            None => {
                let missing: Vec<String> = e.vars().filter(|v| !dims.contains_key(v)).map(|v| self.elab.dim_name(v)).collect();
                err("E7001", format!("unbound dimension{} {}", if missing.len() == 1 { "" } else { "s" }, missing.join(", ")))
            }
        }
    }

    fn axes_of(&self, s: &Shape, dims: &Dims) -> Result<Vec<Axis>> {
        if s.prefix.is_some() {
            return err("E7001", "cannot materialize a shape with an open prefix `...`");
        }
        s.axes.iter().map(|a| Ok(Axis { name: a.name.clone(), len: self.dim(&a.len, dims)? as usize })).collect()
    }

    /// Runs (or reuses) a kernel on `inputs`.
    fn launch(&mut self, key: String, lowered: impl FnOnce() -> std::result::Result<Lowered, String>, inputs: &[&Buf]) -> Result<Value> {
        let idx = match self.cache.get(&key) {
            Some(i) => *i,
            None => {
                let l = lowered().map_err(|m| RtError { code: "E7001", message: m, span: None })?;
                let i = self.kernels.len();
                self.kernels.push(l.kernel);
                self.outs.push((l.out_axes, l.out_kind, l.out_bound));
                self.cache.insert(key, i);
                i
            }
        };
        let k = &self.kernels[idx];
        let mut out = output_buf(k);
        self.exec.run(idx, k, inputs, &mut out)?;
        self.launches += 1;
        let (axes, kind, bound) = self.outs[idx].clone();
        Ok(match kind {
            BufKind::Real => Value::Real(Tensor { axes, data: out }),
            BufKind::Idx => Value::Idx { bound, t: Tensor { axes, data: out } },
        })
    }

    // ----- the host program ----------------------------------------------------

    /// Give a tensor exactly the axis names of its type (as the interpreter does).
    fn retag(&self, v: Value, ty: &Ty) -> Value {
        fn apply(t: &mut Tensor, s: &Shape) {
            let rank = t.rank();
            if s.prefix.is_none() && s.axes.len() != rank {
                return;
            }
            let k = s.axes.len().min(rank);
            for (a, ta) in t.axes[rank - k..].iter_mut().zip(&s.axes[s.axes.len() - k..]) {
                a.name = ta.name.clone();
            }
        }
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

    pub fn call_fn(&mut self, f: FuncId, dims: Dims, args: Vec<Value>) -> Result<Value> {
        let func = self.prog.func(f);
        if func.params.len() != args.len() {
            return err("E7001", format!("`{}` called with {} arguments, {} expected", func.name, args.len(), func.params.len()));
        }
        let mut env: HashMap<VarId, Value> = HashMap::new();
        for (p, a) in func.params.iter().zip(args) {
            let a = self.retag(a, &p.ty);
            env.insert(p.var, a);
        }
        let mut closures: HashMap<VarId, (FuncId, Vec<Value>)> = HashMap::new();
        // Fusion groups: computed once per function, structural, so the same
        // plan serves every shape and every realization (`presk-sched`).
        let plan = self.plans.entry(f).or_insert_with(|| presk_sched::plan(func)).clone();
        for l in &func.body.lets {
            // An interior `let` of a group never reaches memory: its value is
            // an expression inside the group's single loop nest.
            if plan.interior.contains(&l.var) {
                continue;
            }
            if let Some(g) = plan.groups.get(&l.var) {
                let v = self.fused_group(g, &env).map_err(|e| RtError {
                    span: e.span.or(Some(l.span)),
                    message: if e.message.starts_with("in `") { e.message } else { format!("in `{}` (fused group at %{}): {}", func.name, l.var.0, e.message) },
                    code: e.code,
                })?;
                let v = self.retag(v, &l.ty);
                env.insert(l.var, v);
                continue;
            }
            let v = match &l.rhs {
                Rhs::Closure { f, captured } => {
                    let c: Vec<Value> = captured.iter().map(|c| env.get(c).cloned().ok_or_else(|| RtError { code: "E7001", message: "captured variable undefined".into(), span: Some(l.span) })).collect::<Result<_>>()?;
                    closures.insert(l.var, (*f, c));
                    Value::Nat(0)
                }
                rhs => self.rhs(rhs, &l.ty, &dims, &env, &closures).map_err(|e| RtError {
                    span: e.span.or(Some(l.span)),
                    message: if e.message.starts_with("in `") { e.message } else { format!("in `{}` ({}): {}", func.name, l.name.clone().unwrap_or_else(|| format!("%{}", l.var.0)), e.message) },
                    code: e.code,
                })?,
            };
            let v = self.retag(v, &l.ty);
            env.insert(l.var, v);
        }
        env.remove(&func.body.result).ok_or_else(|| RtError { code: "E7001", message: "result variable undefined".into(), span: Some(func.span) })
    }

    /// One launch for a whole fusion group. The result's axes are computed by
    /// replaying the steps' broadcasts, not read off the type, because a type
    /// inside a module function may carry an open prefix.
    fn fused_group(&mut self, g: &presk_sched::Group, env: &HashMap<VarId, Value>) -> Result<Value> {
        let vals: Vec<Value> = g
            .inputs
            .iter()
            .map(|v| env.get(v).cloned().ok_or_else(|| RtError { code: "E7001", message: format!("variable %{} undefined", v.0), span: None }))
            .collect::<Result<_>>()?;
        self.fused_on(g, &vals)
    }

    /// The group applied to one set of values. A parameter tree maps the whole
    /// group over its leaves — the optimizer's update is one chain over a tree,
    /// and without this it would launch every step again for every leaf.
    fn fused_on(&mut self, g: &presk_sched::Group, vals: &[Value]) -> Result<Value> {
        if let Some(template) = vals.iter().find(|v| matches!(v, Value::Record(_) | Value::Tuple(_))) {
            let template = template.clone();
            let mut per_input: Vec<Vec<Value>> = Vec::with_capacity(vals.len());
            let mut n = 0;
            for v in vals {
                match v {
                    Value::Record(_) | Value::Tuple(_) => {
                        let mut out = Vec::new();
                        collect_leaves(v, &mut out);
                        n = n.max(out.len());
                        per_input.push(out);
                    }
                    // A scalar rides along with every leaf (`lr · g`).
                    leaf => per_input.push(vec![leaf.clone()]),
                }
            }
            let mut leaves = Vec::with_capacity(n);
            for i in 0..n {
                let args: Vec<Value> = per_input.iter().map(|xs| if xs.len() == 1 { xs[0].clone() } else { xs[i].clone() }).collect();
                leaves.push(self.fused_on(g, &args)?);
            }
            return Ok(rebuild(&template, &mut leaves.into_iter()));
        }
        let mut ins: Vec<Tensor> = Vec::with_capacity(vals.len());
        for value in vals {
            ins.push(self.as_tensor(value)?);
        }
        let in_axes: Vec<Vec<Axis>> = ins.iter().map(|t| t.axes.clone()).collect();
        let mut shapes: Vec<Vec<Axis>> = Vec::with_capacity(g.steps.len());
        for s in &g.steps {
            let of = |src: &presk_sched::Src| -> Vec<Axis> {
                match src {
                    presk_sched::Src::Input(k) => in_axes[*k].clone(),
                    presk_sched::Src::Step(k) => shapes[*k].clone(),
                }
            };
            let axes = match s.args.as_slice() {
                [x] => of(x),
                [x, y] => lower::result_axes(&of(x), &of(y)).ok_or_else(|| RtError {
                    code: "E7001",
                    message: format!("fused: `{}` cannot broadcast {} with {}", s.op.name(), lower::show(&of(x)), lower::show(&of(y))),
                    span: None,
                })?,
                other => return err("E7001", format!("fused: `{}` takes {} arguments", s.op.name(), other.len())),
            };
            shapes.push(axes);
        }
        let out = shapes.pop().expect("a group has at least one step");
        let operands: Vec<Operand> = in_axes.iter().map(|a| Operand { axes: a.clone(), kind: BufKind::Real }).collect();
        let dtype = self.dtype;
        let key = format!("fused|{}|{dtype}|{}|{}", g.signature(), operands.iter().map(|o| lower::show(&o.axes)).collect::<Vec<_>>().join(";"), lower::show(&out));
        let bufs: Vec<&Buf> = ins.iter().map(|t| &t.data).collect();
        let steps = g.steps.clone();
        self.fused += g.steps.len() as u64 - 1;
        self.launch(key, move || lower::fused(&steps, &operands, &out, dtype), &bufs)
    }

    fn rhs(&mut self, rhs: &Rhs, ty: &Ty, dims: &Dims, env: &HashMap<VarId, Value>, closures: &HashMap<VarId, (FuncId, Vec<Value>)>) -> Result<Value> {
        let get = |v: &VarId| -> Result<Value> { env.get(v).cloned().ok_or_else(|| RtError { code: "E7001", message: format!("variable %{} undefined", v.0), span: None }) };
        match rhs {
            Rhs::Lit(Lit::Nat(n)) => Ok(Value::Nat(*n)),
            Rhs::Lit(Lit::Float(s)) => Ok(Value::F64(s.parse().map_err(|_| RtError { code: "E7001", message: format!("bad float literal `{s}`"), span: None })?)),
            Rhs::Prim { op, args } => {
                let vals: Vec<Value> = args.iter().map(get).collect::<Result<_>>()?;
                self.prim(op, &vals, ty, dims)
            }
            Rhs::Call { f, dim_args, args } => {
                let vals: Vec<Value> = args.iter().map(get).collect::<Result<_>>()?;
                let callee_dims = self.callee_dims(dim_args, dims);
                self.call_fn(*f, callee_dims, vals)
            }
            Rhs::Tuple(vs) => Ok(Value::Tuple(vs.iter().map(get).collect::<Result<_>>()?)),
            Rhs::Proj(v, i) => match get(v)? {
                Value::Tuple(vs) => vs.into_iter().nth(*i).ok_or_else(|| RtError { code: "E7001", message: format!("tuple has no element {i}"), span: None }),
                other => err("E7001", format!("projection on a {}", other.kind())),
            },
            Rhs::Record(fields) => Ok(Value::Record(fields.iter().map(|(n, v)| Ok((n.clone(), get(v)?))).collect::<Result<_>>()?)),
            Rhs::Field(v, name) => match get(v)? {
                Value::Record(fs) => fs.into_iter().find(|(n, _)| n == name).map(|(_, v)| v).ok_or_else(|| RtError { code: "E7001", message: format!("record has no field `{name}`"), span: None }),
                other => err("E7001", format!("field `{name}` of a {}", other.kind())),
            },
            Rhs::Closure { .. } => Ok(Value::Nat(0)),
            Rhs::Grad(_) => err("E7001", "`grad` reached the kernel backend: run automatic differentiation first"),
            Rhs::Apply { fun, args } => {
                let Some((f, captured)) = closures.get(fun) else {
                    return err("E7001", "application of an unknown function value");
                };
                let mut all = captured.clone();
                for a in args {
                    all.push(get(a)?);
                }
                self.call_fn(*f, dims.clone(), all)
            }
            Rhs::Fold { body, dim_args, n, stacked, init, extra, index, reverse, scan } => {
                let count = self.dim(n, dims)? as usize;
                let sts: Vec<Value> = stacked.iter().map(get).collect::<Result<_>>()?;
                let ex: Vec<Value> = extra.iter().map(get).collect::<Result<_>>()?;
                let mut acc = get(init)?;
                let callee_dims = self.callee_dims(dim_args, dims);
                let order: Vec<usize> = if *reverse { (0..count).rev().collect() } else { (0..count).collect() };
                let mut pre: Vec<(usize, Value)> = Vec::new();
                for i in order {
                    if *scan {
                        pre.push((i, acc.clone()));
                    }
                    let mut args = Vec::with_capacity(sts.len() + ex.len() + 2);
                    if *index {
                        args.push(Value::Nat(i as u64));
                    }
                    for s in &sts {
                        args.push(slice_tree(s, i));
                    }
                    args.push(acc);
                    args.extend(ex.iter().cloned());
                    acc = self.call_fn(*body, callee_dims.clone(), args)?;
                }
                if *scan {
                    pre.sort_by_key(|(i, _)| *i);
                    let parts: Vec<Value> = pre.into_iter().map(|(_, v)| v).collect();
                    Ok(Value::Tuple(vec![acc, stack_trees(&parts)?]))
                } else {
                    Ok(acc)
                }
            }
        }
    }

    fn callee_dims(&self, dim_args: &BTreeMap<presk_types::dim::DimVar, DimExpr>, dims: &Dims) -> Dims {
        let mut callee_dims = Dims::new();
        for (v, e) in dim_args {
            if let Ok(n) = self.dim(e, dims) {
                callee_dims.insert(*v, n);
            }
        }
        callee_dims
    }

    // ----- primitives -----------------------------------------------------------

    /// Evaluates one primitive (public for the conformance suite).
    #[allow(clippy::only_used_in_recursion)]
    pub fn prim(&mut self, op: &PrimOp, args: &[Value], ty: &Ty, dims: &Dims) -> Result<Value> {
        let a = |i: usize| -> Result<&Value> { args.get(i).ok_or_else(|| RtError { code: "E7001", message: format!("`{}`: missing argument {i}", op.name()), span: None }) };
        match op {
            // ----- host: structure -----
            PrimOp::ZerosLike => Ok(a(0)?.map_leaves(&mut |leaf| match leaf {
                Value::Real(t) => Value::Real(Tensor { axes: t.axes.clone(), data: self.buf(vec![0.0; t.numel()]) }),
                other => other.clone(),
            })),
            PrimOp::MergeAxes { a: an, b: bn } => {
                let t = self.as_tensor(a(0)?)?;
                let Some(p) = t.axes.iter().position(|x| x.name.as_deref() == Some(an.as_str())) else {
                    return err("E7001", format!("merge: no axis `{an}` in {}", lower::show(&t.axes)));
                };
                if t.axes.get(p + 1).and_then(|x| x.name.as_deref()) != Some(bn.as_str()) {
                    return err("E7001", format!("merge: `{an}` is not immediately followed by `{bn}` in {}", lower::show(&t.axes)));
                }
                let mut axes = t.axes[..p].to_vec();
                axes.push(Axis { name: None, len: t.axes[p].len * t.axes[p + 1].len });
                axes.extend_from_slice(&t.axes[p + 2..]);
                Ok(Value::Real(Tensor { axes, data: t.data.clone() }))
            }
            PrimOp::ReshapeLike => {
                let (x, template) = (self.as_tensor(a(0)?)?, self.as_tensor(a(1)?)?);
                if x.numel() != template.numel() {
                    return err("E7001", format!("reshape_like: {} elements against {}", x.numel(), template.numel()));
                }
                Ok(Value::Real(Tensor { axes: template.axes.clone(), data: x.data.clone() }))
            }
            PrimOp::SetField { name } => match a(0)? {
                Value::Record(fs) => Ok(Value::Record(fs.iter().map(|(n, v)| (n.clone(), if n == name { a(1).unwrap().clone() } else { v.clone() })).collect())),
                other => err("E7001", format!("set_field on a {}", other.kind())),
            },
            PrimOp::SetElem { index } => match a(0)? {
                Value::Tuple(vs) => Ok(Value::Tuple(vs.iter().enumerate().map(|(i, v)| if i == *index { a(1).unwrap().clone() } else { v.clone() }).collect())),
                other => err("E7001", format!("set_elem on a {}", other.kind())),
            },
            PrimOp::SliceParams { index } => Ok(slice_tree(a(0)?, *index as usize)),
            PrimOp::StackParams { n } => {
                if args.len() != *n as usize {
                    return err("E7001", "stack_params: wrong number of parts");
                }
                stack_trees(args)
            }
            PrimOp::SetSlice => {
                let i = match a(1)? {
                    Value::Nat(i) => *i as usize,
                    other => return err("E7001", format!("set_slice: index is a {}", other.kind())),
                };
                Ok(set_slice(a(0)?, i, a(2)?))
            }
            PrimOp::ScatterParams { index } => Ok(scatter_params(a(0)?, *index as usize, a(1)?)),
            PrimOp::SumSquares => {
                let mut acc = 0.0f64;
                let mut leaves: Vec<Tensor> = Vec::new();
                a(0)?.map_leaves(&mut |leaf| {
                    if let Value::Real(t) = leaf {
                        leaves.push(t.clone());
                    }
                    leaf.clone()
                });
                for t in &leaves {
                    let v = self.tensor_prim(&PrimOp::SumSquares, &[t])?;
                    acc += v.scalar_f64();
                }
                Ok(self.scalar(acc))
            }
            // ----- host: scalars, constants, randomness -----
            PrimOp::Pi => Ok(self.scalar(std::f64::consts::PI)),
            PrimOp::Real => match a(0)? {
                Value::Nat(n) => Ok(self.scalar(*n as f64)),
                Value::Int(n) => Ok(self.scalar(*n as f64)),
                Value::F64(x) => Ok(self.scalar(*x)),
                Value::Real(t) => Ok(Value::Real(t.clone())),
                Value::Idx { t, .. } => self.tensor_prim(op, &[t]),
                other => err("E7001", format!("real: cannot inject a {}", other.kind())),
            },
            PrimOp::Zeros { ty: t } | PrimOp::Ones { ty: t } => {
                let axes = self.axes_of(shape_of(t)?, dims)?;
                let value = if matches!(op, PrimOp::Zeros { .. }) { 0.0 } else { 1.0 };
                let dtype = self.dtype;
                let key = format!("{}|{dtype}|{}", op.name(), lower::show(&axes));
                self.launch(key, move || Ok(lower::fill(&op.name(), &axes, value, dtype)), &[])
            }
            PrimOp::DropoutMask => {
                let rate = match a(0)? {
                    Value::F64(p) => *p,
                    Value::Nat(n) => *n as f64,
                    Value::Real(t) if t.rank() == 0 => t.data.to_f64()[0],
                    other => return err("E7001", format!("dropout: the rate is a {}", other.kind())),
                };
                if !(0.0..1.0).contains(&rate) {
                    return err("E7001", format!("dropout: the rate {rate} is not in [0, 1)"));
                }
                let key = key_of(a(1)?)?;
                let axes = self.as_tensor(a(2)?)?.axes.clone();
                let n: usize = axes.iter().map(|x| x.len).product();
                let data: Vec<f64> = if !self.training {
                    vec![1.0; n]
                } else {
                    let (keep, scale) = (1.0 - rate, 1.0 / (1.0 - rate));
                    let mut rng = key.rng();
                    (0..n).map(|_| if uniform01(&mut rng) < keep { scale } else { 0.0 }).collect()
                };
                Ok(Value::Real(Tensor { axes, data: self.buf(data) }))
            }
            PrimOp::Normal { ty: t, std } => {
                let axes = self.axes_of(shape_of(t)?, dims)?;
                let key = key_of(a(0)?)?;
                let std: f64 = std.parse().unwrap_or(1.0);
                let mut rng = key.rng();
                let n: usize = axes.iter().map(|x| x.len).product();
                let data: Vec<f64> = (0..n).map(|_| normal(&mut rng) * std).collect();
                Ok(Value::Real(Tensor { axes, data: self.buf(data) }))
            }
            PrimOp::XavierUniform { ty: t } => {
                let axes = self.axes_of(shape_of(t)?, dims)?;
                let key = key_of(a(0)?)?;
                let (fan_in, fan_out) = match axes.len() {
                    0 => (1, 1),
                    1 => (axes[0].len, axes[0].len),
                    _ => (axes[0].len, axes[axes.len() - 1].len),
                };
                let bound = (6.0 / (fan_in + fan_out) as f64).sqrt();
                let mut rng = key.rng();
                let n: usize = axes.iter().map(|x| x.len).product();
                let data: Vec<f64> = (0..n).map(|_| (2.0 * uniform01(&mut rng) - 1.0) * bound).collect();
                Ok(Value::Real(Tensor { axes, data: self.buf(data) }))
            }
            PrimOp::SplitKey { n } => Ok(Value::Tuple(key_of(a(0)?)?.split(*n).into_iter().map(Value::Key).collect())),
            PrimOp::Sample => {
                let Value::Real(t) = a(0)? else { return err("E7001", "sample: expected a real tensor") };
                if t.axes.is_empty() {
                    return err("E7001", "sample: scalar argument");
                }
                let v = t.axes[t.axes.len() - 1].len;
                let data = t.data.to_f64();
                let n = t.numel() / v.max(1);
                let mut out = Vec::with_capacity(n);
                for p in 0..n {
                    let row = &data[p * v..(p + 1) * v];
                    let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exps: Vec<f64> = row.iter().map(|x| (x - m).exp()).collect();
                    let z: f64 = exps.iter().sum();
                    let u = uniform01(&mut self.sample_rng);
                    let mut acc = 0.0;
                    let mut pick = v - 1;
                    for (i, e) in exps.iter().enumerate() {
                        acc += e / z;
                        if u < acc {
                            pick = i;
                            break;
                        }
                    }
                    out.push(pick as i64);
                }
                Ok(Value::Idx { bound: v as u64, t: Tensor { axes: t.axes[..t.axes.len() - 1].to_vec(), data: Buf::I64(out) } })
            }
            PrimOp::Positions { axis, len } => {
                let n = self.dim(len, dims)? as usize;
                let (axis, dtype) = (axis.clone(), self.dtype);
                self.launch(format!("positions {axis} {n} {dtype}"), move || Ok(lower::positions(&axis, n, dtype)), &[])
            }
            PrimOp::OneHot => {
                let Value::Idx { bound, t } = a(0)? else {
                    return err("E7001", format!("`one_hot`: expected an index tensor, found a {}", a(0)?.kind()));
                };
                let mut axes = t.axes.clone();
                axes.push(Axis { name: None, len: *bound as usize });
                let dtype = self.dtype;
                let (key, ax) = (format!("one_hot {} {dtype}", lower::show(&axes)), axes.clone());
                self.launch(key, move || Ok(lower::one_hot(&ax, dtype)), &[&t.data])
            }
            PrimOp::CausalMask { axis, len } => {
                let n = self.dim(len, dims)? as usize;
                let (axis, dtype) = (axis.clone(), self.dtype);
                self.launch(format!("causal_mask {axis} {n} {dtype}"), move || Ok(lower::causal_mask(&axis, n, dtype)), &[])
            }
            // ----- elementwise on trees and hyperparameters -----
            PrimOp::Add | PrimOp::Sub | PrimOp::Mul | PrimOp::Div if args.len() == 2 => {
                let (x, y) = (a(0)?, a(1)?);
                match (x, y) {
                    (Value::Nat(p), Value::Nat(q)) => {
                        let r = hyper(op, *p as f64, *q as f64);
                        Ok(if r >= 0.0 && r.fract() == 0.0 { Value::Nat(r as u64) } else { Value::Int(r as i64) })
                    }
                    (Value::F64(p), Value::F64(q)) => Ok(Value::F64(hyper(op, *p, *q))),
                    (Value::Record(_) | Value::Tuple(_), Value::Real(t)) if t.rank() == 0 => {
                        let mut out = Vec::new();
                        collect_leaves(x, &mut out);
                        let mut mapped = Vec::new();
                        for leaf in out {
                            mapped.push(self.prim(op, &[leaf, y.clone()], ty, dims)?);
                        }
                        Ok(rebuild(x, &mut mapped.into_iter()))
                    }
                    (Value::Real(t), Value::Record(_) | Value::Tuple(_)) if t.rank() == 0 => {
                        let mut out = Vec::new();
                        collect_leaves(y, &mut out);
                        let mut mapped = Vec::new();
                        for leaf in out {
                            mapped.push(self.prim(op, &[x.clone(), leaf], ty, dims)?);
                        }
                        Ok(rebuild(y, &mut mapped.into_iter()))
                    }
                    (Value::Record(_) | Value::Tuple(_), _) | (_, Value::Record(_) | Value::Tuple(_)) => {
                        let (mut lx, mut ly) = (Vec::new(), Vec::new());
                        collect_leaves(x, &mut lx);
                        collect_leaves(y, &mut ly);
                        if lx.len() != ly.len() {
                            return err("E7001", format!("{}: structure mismatch", op.name()));
                        }
                        let mut mapped = Vec::new();
                        for (p, q) in lx.into_iter().zip(ly) {
                            mapped.push(self.prim(op, &[p, q], ty, dims)?);
                        }
                        Ok(rebuild(x, &mut mapped.into_iter()))
                    }
                    _ => {
                        let (tx, ty_) = (self.as_tensor(x)?, self.as_tensor(y)?);
                        self.tensor_prim(op, &[&tx, &ty_])
                    }
                }
            }
            // ----- everything else: unary on trees / hyperparameters, or a kernel -----
            _ => {
                if args.len() == 1 {
                    match a(0)? {
                        Value::Record(_) | Value::Tuple(_) => {
                            let x = a(0)?;
                            let mut leaves = Vec::new();
                            collect_leaves(x, &mut leaves);
                            let mut mapped = Vec::new();
                            for leaf in leaves {
                                mapped.push(self.prim(op, &[leaf], ty, dims)?);
                            }
                            return Ok(rebuild(x, &mut mapped.into_iter()));
                        }
                        Value::Nat(n) if lower::lower(op, &[Operand { axes: vec![], kind: BufKind::Real }], self.dtype).is_some() => {
                            let t = self.tensor_prim(op, &[&Tensor { axes: vec![], data: self.buf(vec![*n as f64]) }])?;
                            return Ok(Value::F64(t.scalar_f64()));
                        }
                        Value::F64(x) if lower::lower(op, &[Operand { axes: vec![], kind: BufKind::Real }], self.dtype).is_some() => {
                            let t = self.tensor_prim(op, &[&Tensor { axes: vec![], data: self.buf(vec![*x]) }])?;
                            return Ok(Value::F64(t.scalar_f64()));
                        }
                        _ => {}
                    }
                }
                let mut tensors: Vec<Tensor> = Vec::with_capacity(args.len());
                let mut kinds: Vec<BufKind> = Vec::new();
                for v in args {
                    match v {
                        Value::Idx { t, .. } => {
                            tensors.push(t.clone());
                            kinds.push(BufKind::Idx);
                        }
                        other => {
                            tensors.push(self.as_tensor(other)?);
                            kinds.push(BufKind::Real);
                        }
                    }
                }
                let refs: Vec<&Tensor> = tensors.iter().collect();
                self.tensor_prim_kinds(op, &refs, &kinds)
            }
        }
    }

    fn as_tensor(&self, v: &Value) -> Result<Tensor> {
        match v {
            Value::Real(t) => Ok(t.clone()),
            Value::Nat(n) => Ok(Tensor { axes: vec![], data: self.buf(vec![*n as f64]) }),
            Value::Int(n) => Ok(Tensor { axes: vec![], data: self.buf(vec![*n as f64]) }),
            Value::F64(x) => Ok(Tensor { axes: vec![], data: self.buf(vec![*x]) }),
            other => err("E7001", format!("expected a real tensor, found a {}", other.kind())),
        }
    }

    fn tensor_prim(&mut self, op: &PrimOp, ins: &[&Tensor]) -> Result<Value> {
        let kinds: Vec<BufKind> = ins.iter().map(|t| if matches!(t.data, Buf::I64(_)) { BufKind::Idx } else { BufKind::Real }).collect();
        self.tensor_prim_kinds(op, ins, &kinds)
    }

    fn tensor_prim_kinds(&mut self, op: &PrimOp, ins: &[&Tensor], kinds: &[BufKind]) -> Result<Value> {
        let operands: Vec<Operand> = ins.iter().zip(kinds).map(|(t, k)| Operand { axes: t.axes.clone(), kind: *k }).collect();
        let dtype = self.dtype;
        let key = format!("{op:?}|{dtype}|{}", operands.iter().map(|o| format!("{}{:?}", lower::show(&o.axes), o.kind)).collect::<Vec<_>>().join(";"));
        let bufs: Vec<&Buf> = ins.iter().map(|t| &t.data).collect();
        let op = op.clone();
        self.launch(key, move || lower::lower(&op, &operands, dtype).unwrap_or_else(|| Err(format!("`{}`: no kernel for these operands", op.name()))), &bufs)
    }
}

fn hyper(op: &PrimOp, x: f64, y: f64) -> f64 {
    match op {
        PrimOp::Add => x + y,
        PrimOp::Sub => x - y,
        PrimOp::Mul => x * y,
        _ => x / y,
    }
}

fn shape_of(ty: &Ty) -> Result<&Shape> {
    match ty {
        Ty::Real(s) | Ty::Idx { shape: s, .. } => Ok(s),
        _ => err("E7001", "initializer without a tensor type"),
    }
}

fn key_of(v: &Value) -> Result<Key> {
    match v {
        Value::Key(k) => Ok(k.clone()),
        other => err("E7001", format!("expected a key, found a {}", other.kind())),
    }
}

fn collect_leaves(v: &Value, out: &mut Vec<Value>) {
    match v {
        Value::Tuple(vs) => vs.iter().for_each(|x| collect_leaves(x, out)),
        Value::Record(fs) => fs.iter().for_each(|(_, x)| collect_leaves(x, out)),
        leaf => out.push(leaf.clone()),
    }
}

fn rebuild(template: &Value, leaves: &mut impl Iterator<Item = Value>) -> Value {
    match template {
        Value::Tuple(vs) => Value::Tuple(vs.iter().map(|x| rebuild(x, leaves)).collect()),
        Value::Record(fs) => Value::Record(fs.iter().map(|(n, x)| (n.clone(), rebuild(x, leaves))).collect()),
        _ => leaves.next().unwrap_or_else(|| template.clone()),
    }
}

fn with_data(t: &Tensor, axes: Vec<Axis>, data: Vec<f64>) -> Tensor {
    let buf = match &t.data {
        Buf::F32(_) => Buf::F32(data.into_iter().map(|x| x as f32).collect()),
        Buf::I64(_) => Buf::I64(data.into_iter().map(|x| x as i64).collect()),
        _ => Buf::F64(data),
    };
    Tensor { axes, data: buf }
}

/// Slice `i` along the leading axis of every tensor leaf.
pub fn slice_tree(v: &Value, i: usize) -> Value {
    v.map_leaves(&mut |leaf| match leaf {
        Value::Real(t) if t.rank() >= 1 => {
            let chunk = t.numel() / t.axes[0].len.max(1);
            let data = t.data.to_f64()[i * chunk..(i + 1) * chunk].to_vec();
            Value::Real(with_data(t, t.axes[1..].to_vec(), data))
        }
        other => other.clone(),
    })
}

/// Stack trees leaf by leaf along a new leading axis.
pub fn stack_trees(parts: &[Value]) -> Result<Value> {
    match &parts[0] {
        Value::Real(first) => {
            let mut axes = vec![Axis { name: None, len: parts.len() }];
            axes.extend(first.axes.iter().cloned());
            let mut data = Vec::with_capacity(first.numel() * parts.len());
            for p in parts {
                match p {
                    Value::Real(t) => data.extend(t.data.to_f64()),
                    other => return err("E7001", format!("stack_params: expected a real tensor, found a {}", other.kind())),
                }
            }
            Ok(Value::Real(with_data(first, axes, data)))
        }
        Value::Record(fs) => {
            let mut out = Vec::new();
            for (i, (n, _)) in fs.iter().enumerate() {
                let subs: Vec<Value> = parts.iter().map(|p| match p {
                    Value::Record(g) => g[i].1.clone(),
                    other => other.clone(),
                }).collect();
                out.push((n.clone(), stack_trees(&subs)?));
            }
            Ok(Value::Record(out))
        }
        Value::Tuple(vs) => {
            let mut out = Vec::new();
            for i in 0..vs.len() {
                let subs: Vec<Value> = parts.iter().map(|p| match p {
                    Value::Tuple(g) => g[i].clone(),
                    other => other.clone(),
                }).collect();
                out.push(stack_trees(&subs)?);
            }
            Ok(Value::Tuple(out))
        }
        other => Ok(other.clone()),
    }
}

fn set_slice(stacked: &Value, i: usize, value: &Value) -> Value {
    let mut leaves = Vec::new();
    collect_leaves(value, &mut leaves);
    let mut k = 0;
    stacked.map_leaves(&mut |leaf| {
        let v = leaves.get(k).cloned();
        k += 1;
        match (leaf, v) {
            (Value::Real(t), Some(Value::Real(v))) if t.rank() >= 1 && i < t.axes[0].len => {
                let chunk = t.numel() / t.axes[0].len;
                let mut data = t.data.to_f64();
                let vd = v.data.to_f64();
                data[i * chunk..(i + 1) * chunk].copy_from_slice(&vd[..chunk.min(vd.len())]);
                Value::Real(with_data(t, t.axes.clone(), data))
            }
            (leaf, _) => leaf.clone(),
        }
    })
}

fn scatter_params(tmpl: &Value, index: usize, ct: &Value) -> Value {
    let mut leaves = Vec::new();
    collect_leaves(ct, &mut leaves);
    let mut k = 0;
    tmpl.map_leaves(&mut |leaf| {
        let c = leaves.get(k).cloned();
        k += 1;
        match (leaf, c) {
            (Value::Real(t), Some(Value::Real(c))) => {
                let mut data = vec![0.0; t.numel()];
                let cd = c.data.to_f64();
                let start = index * cd.len();
                data[start..start + cd.len()].copy_from_slice(&cd);
                Value::Real(with_data(t, t.axes.clone(), data))
            }
            (other, _) => other.clone(),
        }
    })
}
