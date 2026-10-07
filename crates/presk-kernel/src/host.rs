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
use presk_rt::{Axis, Backend, Buf, Elems, Layout, NestKind, Plan, Prepared, Realization, RtError, Tensor, Value, err};
use presk_types::dim::DimExpr;
use presk_types::elab::ElabProgram;
use presk_types::ty::{ParamsTree, Shape, Ty};

use crate::exec::{launch_buf, run_kernel};
use crate::ir::{BufKind, Dtype, Kernel, Threads};
use crate::lower::{self, Lowered, Operand};
use crate::profile::{Overhead, Phase, SharedProfile, copied_bytes, freed_bytes, since, value_bytes};

type Result<T> = std::result::Result<T, RtError>;

/// How kernels are executed: the kernel-IR interpreter ([`InterpExec`]) or
/// compiled code (the Cranelift backend). `idx` identifies the kernel in the
/// backend's list, so that an executor can compile it once.
pub trait KernelExec {
    fn name(&self) -> &'static str;
    fn run(&mut self, idx: usize, k: &Kernel, inputs: &[&Buf], out: &mut Buf) -> Result<()>;
    /// The workers a split may use, `p_M` of the language reference, for an
    /// executor that runs kernels on several; `None` for one that runs every
    /// kernel on one worker whatever its split (the kernel interpreter), where
    /// a plan's worker count is not checked (condition 5).
    fn workers(&self) -> Option<usize> {
        None
    }
    /// Profiling the host (`presk bench --host-profile`, plan 0.10.0 step
    /// 5): when on, the executor times what of each launch is not the
    /// kernel's code, which [`KernelExec::overhead`] gives after the launch.
    /// It reads the clock and nothing else.
    fn set_profiling(&mut self, _on: bool) {}
    /// What the last launch spent compiling, dispatching, and starting and
    /// joining threads; nothing for an executor that does not time them.
    fn overhead(&self) -> Overhead {
        Overhead::default()
    }
    /// `f(0)`, …, `f(n − 1)`, on the executor's workers where it has them,
    /// returning when all are done: the host's own work divided among them
    /// (a training dropout mask, plan 0.10.0 step 6). In order, on the
    /// calling thread, for an executor without workers.
    fn parallel(&mut self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        (0..n).for_each(f);
    }
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

/// One launch of a nest as the planning tools replay it (0.7.0 step 5): what
/// it computes, at which operands and types, and its input buffers as the
/// kernel received them, so that it can be lowered again under any layout and
/// timed alone (`presk autotune`).
#[derive(Clone, Debug)]
pub struct Launch {
    /// The nest, `"<function>/%<let>"`.
    pub nest: String,
    pub what: LaunchOp,
    pub operands: Vec<Operand>,
    pub dtype: Dtype,
    pub order: presk_rt::ReductionOrder,
    /// The accumulation type the realization states for the nest.
    pub accum: Dtype,
    pub inputs: Vec<Buf>,
    /// How many times the entry launches the nest at these operands, the
    /// first included: what `presk plan` weighs a shape's prediction by when
    /// one layout must serve a nest launched at several.
    pub count: usize,
    /// For a contraction nest whose value one `add` alone reads, in the
    /// product's type, at an operand that fits its output: the `add`, which
    /// the nest can take as its epilogue (plan 0.10.0 step 7). Noted at
    /// every launch of the nest at these operands, or at none.
    pub epilogue: Option<EpilogueAt>,
}

/// The `add` a captured contraction can take as its epilogue: its other
/// operand as the kernel receives it, and how it adds.
#[derive(Clone, Debug)]
pub struct EpilogueAt {
    pub operand: Operand,
    pub input: Buf,
    pub epilogue: lower::Epilogue,
}

/// What a captured launch computes.
#[derive(Clone, Debug)]
pub enum LaunchOp {
    Prim(PrimOp),
    /// A fusion group's steps, and its result's axes.
    Fused { steps: Vec<presk_realize::Step>, out: Vec<Axis> },
}

/// An input as a kernel of type `dtype` receives it: converted at the
/// kernel's edge unless it is an index, exact, or integer weights.
fn kernel_input(b: &Buf, dtype: Dtype, keep_quant: bool) -> Buf {
    match b {
        Buf::I64(_) | Buf::Exact(_) => b.clone(),
        Buf::Quant(_) if keep_quant => b.clone(),
        b if b.dtype() == dtype.name() => b.clone(),
        b => Buf::of(dtype, &b.to_f64()),
    }
}

pub struct KernelBackend<'p> {
    pub prog: &'p Program,
    pub elab: &'p ElabProgram,
    /// What the backend is allowed to do (D34). `dtype` below is this
    /// realization resolved at the site currently being executed.
    pub realize: Realization,
    /// The storage type of the site being executed. Constant for a uniform
    /// realization, which is every realization earlier versions could write.
    pub dtype: Dtype,
    /// `"<function>/%<let>"` — the fusion group a dtype is assigned to.
    site: String,
    /// The realization plan (0.7.0 step 1): its layout is read per nest at
    /// every launch ([`KernelBackend::layout_at`]). Set by
    /// [`KernelBackend::set_plan`]; a backend without one runs the default
    /// layout everywhere, which is what `--realize` means.
    pub plan: Plan,
    /// The layout of every nest the plan does not decide, over the
    /// lowering's own defaults. Its one member in use is the lanes, which
    /// were a field of their own until 0.7.0 (0.6.0 step 1): unstated, each
    /// kernel takes **its own natural width** where its form has lanes, since
    /// a mixed realization has kernels of several types; `Some(1)` is scalar,
    /// every kernel before 0.6.0. Lanes are independent output elements, so a
    /// vector kernel is bit-identical to its scalar form.
    pub default_layout: Layout,
    /// Kernels in order of first use.
    pub kernels: Vec<Kernel>,
    outs: Vec<(Vec<Axis>, BufKind, u64)>,
    cache: HashMap<String, usize>,
    /// Kernel launches so far.
    pub launches: u64,
    /// Whether `dropout` drops (see the interpreter's field of the same name).
    pub training: bool,
    /// The fusion groups of each function, computed once (`presk-realize`).
    fusion: HashMap<FuncId, presk_realize::Plan>,
    /// Kernel launches fusion removed: each group of `n` steps launches once
    /// where `n` kernels would have run, per leaf of a parameter tree.
    pub fused: u64,
    pub exec: Box<dyn KernelExec>,
    /// When set, the first launch of every nest at every shape is recorded
    /// here (0.7.0 step 5): the planning tools' view of an entry.
    pub capture: Option<Vec<Launch>>,
    /// When set, the host program of the next call is recorded, for the C
    /// export (plan 0.9.0 step 3, D79). Its hooks only observe: the backend
    /// computes and launches what it does without it.
    pub recorder: Option<Box<crate::record::Recorder>>,
    /// When set, the host's time is added to it by phase (plan 0.10.0 step
    /// 5, [`crate::profile`]); set with [`KernelBackend::set_host_profile`].
    profile: Option<SharedProfile>,
    /// Buffers of one value, by type, length and value's bits, which the host
    /// fills itself: one for every `zeros_like` and every dropout mask
    /// outside training of that length (plan 0.10.0 step 6). Shared, so
    /// never written.
    filled: HashMap<(presk_rt::Dtype, usize, u64), Buf>,
    /// The products of each function that can take their `add` as an
    /// epilogue (`presk_core::fusion::epilogues`), computed once.
    epilogues: HashMap<FuncId, std::rc::Rc<HashMap<VarId, presk_core::fusion::Epilogue>>>,
    /// Whether the plan states an epilogue anywhere: else no `let` asks.
    epilogue_stated: bool,
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

    fn prepare(&mut self, entry: FuncId, dims: &Dims, realize: &Realization) -> Result<Prepared> {
        presk_interp::defined::check(self.prog, self.elab, entry, dims, None)?;
        crate::plan::check_accum(realize)?;
        self.realize = realize.clone();
        self.dtype = realize.default;
        Ok(Prepared { func: entry, dims: dims.clone(), realize: realize.clone(), handle: 0 })
    }

    fn call(&mut self, p: &Prepared, args: Vec<Value>) -> Result<Value> {
        let t = self.tick();
        let realized: Vec<Value> = args.iter().map(|a| self.realize_value(a)).collect();
        // The bytes converted: a leaf already in its type is shared.
        self.tock(Phase::Entry, t, || args.iter().map(|a| self.converted_bytes(a)).sum());
        if let Some(r) = self.recorder.as_deref_mut() {
            r.entry(&realized);
        }
        let result = self.call_fn(p.func, p.dims.clone(), realized);
        // The caller's arguments are freed where they always were, after the
        // call.
        let t = self.tick();
        drop(args);
        self.tock_time(Phase::Entry, t);
        result
    }
}

impl<'p> KernelBackend<'p> {
    pub fn new(prog: &'p Program, elab: &'p ElabProgram, realize: &Realization) -> Self {
        Self::with_exec(prog, elab, realize, Box::new(InterpExec))
    }

    /// A kernel backend whose kernels are run by `exec`.
    pub fn with_exec(prog: &'p Program, elab: &'p ElabProgram, realize: &Realization, exec: Box<dyn KernelExec>) -> Self {
        let dtype = realize.default;
        let plan = Plan::from_realization(realize.clone());
        let default_layout = Layout { lanes: Some(1), ..Layout::default() };
        KernelBackend { prog, elab, realize: realize.clone(), dtype, site: String::new(), plan, default_layout, kernels: vec![], outs: vec![], cache: HashMap::new(), launches: 0, training: false, fusion: HashMap::new(), fused: 0, exec, capture: None, recorder: None, profile: None, filled: HashMap::new(), epilogues: HashMap::new(), epilogue_stated: false }
    }

    /// Runs under `plan`: its realization, and its layout per nest. The plan
    /// is checked against the nests when they are launched; check it against
    /// the program first with [`crate::plan::check_plan`].
    pub fn set_plan(&mut self, plan: Plan) {
        self.realize = plan.realization.clone();
        self.dtype = self.realize.default;
        self.epilogue_stated = plan.layout.values().any(|l| l.epilogue == Some(true));
        self.plan = plan;
    }

    /// Times the host by phase into `profile` from the next call on, the
    /// executor's overhead with it; `None` stops. Only the clock is read.
    pub fn set_host_profile(&mut self, profile: Option<SharedProfile>) {
        self.exec.set_profiling(profile.is_some());
        self.profile = profile;
    }

    /// The clock, when the host profiles.
    fn tick(&self) -> Option<std::time::Instant> {
        self.profile.as_ref().map(|_| std::time::Instant::now())
    }

    /// A region of `phase` since `t0`, of `bytes` (computed after the clock
    /// is read).
    fn tock(&self, phase: Phase, t0: Option<std::time::Instant>, bytes: impl FnOnce() -> u64) {
        if let (Some(p), Some(t0)) = (&self.profile, t0) {
            let s = since(t0);
            p.borrow_mut().add(phase, s, bytes());
        }
    }

    /// Time added to `phase` since `t0`, whose region was counted already.
    fn tock_time(&self, phase: Phase, t0: Option<std::time::Instant>) {
        if let (Some(p), Some(t0)) = (&self.profile, t0) {
            let s = since(t0);
            p.borrow_mut().add_time(phase, s);
        }
    }

    /// The layout the plan states for a nest (empty: every decision at its
    /// default).
    pub fn layout_at(&self, nest: &str) -> Layout {
        self.plan.layout_at(nest)
    }

    /// Lanes for the nests the plan does not decide: one is scalar; more asks
    /// each kernel for its own natural width where its form has lanes.
    pub fn set_default_lanes(&mut self, lanes: usize) {
        self.default_layout.lanes = if lanes > 1 { None } else { Some(1) };
    }

    /// The layout the nest being executed is lowered with, as `kind`: the
    /// plan's, checked (`E7002`), over the backend's default.
    fn lowered_layout(&self, kind: &NestKind) -> Result<Layout> {
        crate::plan::lowered(&self.layout_at(&self.site), &self.default_layout, kind).map_err(|m| RtError { code: "E7002", message: format!("the plan's layout of `{}`: {m}", self.site), span: None })
    }

    // ----- buffers and values ------------------------------------------------

    fn buf(&self, data: Vec<f64>) -> Buf {
        Buf::of(self.dtype, &data)
    }

    fn scalar(&self, x: f64) -> Value {
        Value::Real(Tensor { axes: vec![], data: self.buf(vec![x]) })
    }

    /// A stack of no part from its type, as the interpreter's `empty_stack`
    /// builds it (D99's addendum, 0.10.0 step 8): the parameters of
    /// `stack{0}(M)` and the scan of a fold of count 0.
    fn empty_stack(&self, ty: &Ty, dims: &Dims) -> Result<Value> {
        Ok(match ty {
            Ty::Params(m) => self.tree_value(&self.elab.params_tree(m), &mut Vec::new(), dims)?,
            Ty::Real(s) => {
                let axes = self.axes_of(s, dims)?;
                let n: usize = axes.iter().map(|x| x.len).product();
                Value::Real(Tensor { axes, data: self.buf(vec![0.0; n]) })
            }
            Ty::Idx { shape, .. } | Ty::Truth(shape) => {
                let bound = match ty {
                    Ty::Idx { bound, .. } => self.dim(bound, dims)?,
                    _ => 2,
                };
                let axes = self.axes_of(shape, dims)?;
                let n: usize = axes.iter().map(|x| x.len).product();
                Value::Idx { bound, t: Tensor { axes, data: Buf::I64(vec![0i64; n].into()) } }
            }
            Ty::Tuple(ts) => Value::Tuple(ts.iter().map(|t| self.empty_stack(t, dims)).collect::<Result<_>>()?),
            _ => Value::Stacked(Vec::new()),
        })
    }

    fn tree_value(&self, t: &ParamsTree, lead: &mut Vec<Axis>, dims: &Dims) -> Result<Value> {
        match t {
            ParamsTree::Leaf(Ty::Real(s)) => {
                let mut axes = lead.clone();
                axes.extend(self.axes_of(s, dims)?);
                let n: usize = axes.iter().map(|x| x.len).product();
                Ok(Value::Real(Tensor { axes, data: self.buf(vec![0.0; n]) }))
            }
            ParamsTree::Leaf(other) => err("E7001", format!("stack_params: a parameter of type `{other:?}`")),
            ParamsTree::Node(fields) => Ok(Value::Record(fields.iter().map(|(n, s)| Ok((n.clone(), self.tree_value(s, lead, dims)?))).collect::<Result<Vec<_>>>()?)),
            ParamsTree::Stacked { n, inner } => {
                lead.push(Axis { name: None, len: self.dim(n, dims)? as usize });
                let v = self.tree_value(inner, lead, dims);
                lead.pop();
                v
            }
        }
    }

    /// `n` elements of `x` in the site's type, the one buffer of them the
    /// host keeps for that length (plan 0.10.0 step 6): no fresh pages, and
    /// nothing written but once.
    fn filled(&mut self, x: f64, n: usize) -> Buf {
        let d = self.dtype;
        self.filled.entry((d, n, x.to_bits())).or_insert_with(|| Buf::of(d, &vec![x; n])).clone()
    }

    /// Every real leaf in the backend's dtype (exact inputs rounded once). A
    /// leaf already in it is shared as it is (plan 0.10.0 step 6): its
    /// conversion to `f64` and back gave its own bits.
    fn realize_value(&self, v: &Value) -> Value {
        v.map_leaves(&mut |leaf| match leaf {
            // Integer weights stay integers: the kernels load them (D44).
            Value::Real(t) if matches!(t.data, Buf::Quant(_)) => leaf.clone(),
            Value::Real(t) if t.data.dtype() == self.dtype.name() => leaf.clone(),
            Value::Real(t) => Value::Real(Tensor { axes: t.axes.clone(), data: self.buf(t.data.to_f64()) }),
            other => other.clone(),
        })
    }

    /// The bytes [`KernelBackend::realize_value`] converts of a value.
    fn converted_bytes(&self, v: &Value) -> u64 {
        match v {
            Value::Real(t) if matches!(t.data, Buf::Quant(_)) || t.data.dtype() == self.dtype.name() => 0,
            Value::Real(t) => (t.numel() * self.dtype.bytes()) as u64,
            Value::Tuple(vs) => vs.iter().map(|x| self.converted_bytes(x)).sum(),
            Value::Record(fs) => fs.iter().map(|(_, x)| self.converted_bytes(x)).sum(),
            _ => 0,
        }
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

    /// Records the first launch of the nest being executed at these operand
    /// shapes, when capturing (0.7.0 step 5), and counts the others.
    fn record(&mut self, what: LaunchOp, operands: &[Operand], dtype: Dtype, accum: Dtype, inputs: &[Buf]) {
        let site = self.site.clone();
        let order = self.realize.order;
        let Some(c) = self.capture.as_mut() else { return };
        let same = |l: &Launch| l.nest == site && l.dtype == dtype && l.operands.len() == operands.len() && l.operands.iter().zip(operands).all(|(a, b)| a.axes == b.axes);
        match c.iter_mut().find(|l| same(l)) {
            Some(l) => l.count += 1,
            None => c.push(Launch { nest: site, what, operands: operands.to_vec(), dtype, order, accum, inputs: inputs.to_vec(), count: 1, epilogue: None }),
        }
    }

    /// Notes on the captured launch of the nest `nest` at `operands` the
    /// epilogue it can take.
    fn note_epilogue(&mut self, nest: &str, dtype: Dtype, operands: &[Operand], at: EpilogueAt) {
        let Some(c) = self.capture.as_mut() else { return };
        if let Some(l) = c.iter_mut().find(|l| l.nest == nest && l.dtype == dtype && l.operands.len() == operands.len() && l.operands.iter().zip(operands).all(|(a, b)| a.axes == b.axes)) {
            l.epilogue = Some(at);
        }
    }

    /// The thread split a nest's kernel carries (0.7.0 step 3). The plan's,
    /// checked against the kernel — its loop at the stated depth must be
    /// parallel with two iterations or more (condition 8) — and against the
    /// executor's workers (condition 5); else the backend default's, where the
    /// kernel admits it; else the executor's own default.
    fn threads_for(&self, k: &Kernel) -> std::result::Result<Threads, String> {
        let stated = self.layout_at(&self.site).split;
        let (split, lenient) = match stated {
            Some(s) => (Some(s), false),
            None => (self.default_layout.split, true),
        };
        crate::plan::threads_of(split, k, self.exec.workers(), lenient).map_err(|m| format!("the plan's layout of `{}`: {m}", self.site))
    }

    /// Runs (or reuses) a kernel on `inputs`. `split`: the kernel is its
    /// nest's, and carries the nest's thread split; the copy of a pack does
    /// not, and takes the executor's default.
    fn launch(&mut self, key: String, lowered: impl FnOnce() -> std::result::Result<Lowered, String>, inputs: &[&Buf], split: bool) -> Result<Value> {
        let t = self.tick();
        let pack = key.starts_with("pack|");
        let hit = self.cache.get(&key).copied();
        // The key was counted where it was formatted.
        self.tock_time(Phase::Key, t);
        let t = self.tick();
        let idx = match hit {
            Some(i) => i,
            None => {
                let mut l = lowered().map_err(|m| RtError { code: "E7001", message: m, span: None })?;
                if split {
                    l.kernel.threads = self.threads_for(&l.kernel).map_err(|m| RtError { code: "E7002", message: m, span: None })?;
                }
                let i = self.kernels.len();
                self.kernels.push(l.kernel);
                self.outs.push((l.out_axes, l.out_kind, l.out_bound));
                self.cache.insert(key, i);
                self.tock(Phase::Lower, t, || 0);
                i
            }
        };
        let t = self.tick();
        let k = &self.kernels[idx];
        // A mixed realization can hand this kernel a buffer another site
        // wrote in a different type. Converting here, at the kernel's edge,
        // is where the cost of a mixed assignment actually is, and it keeps
        // the loop nest monomorphic.
        // A buffer already in the kernel's type is lent, not copied: a copy
        // per launch of every weight is a cost of the host, not of the
        // realization, and it would be charged to every float row of a
        // benchmark and to none of the integer ones.
        let converted: Vec<std::borrow::Cow<Buf>> = inputs
            .iter()
            .zip(&k.inputs)
            .map(|(b, spec)| match b {
                Buf::I64(_) | Buf::Exact(_) => std::borrow::Cow::Borrowed(*b),
                // Integer weights reach a kernel whose input the host marked;
                // any other kernel reads their values, `f32(q)·s`.
                Buf::Quant(_) if matches!(spec.kind, BufKind::Quant { .. }) => std::borrow::Cow::Borrowed(*b),
                b if b.dtype() == k.dtype.name() => std::borrow::Cow::Borrowed(*b),
                b => std::borrow::Cow::Owned(Buf::of(k.dtype, &b.to_f64())),
            })
            .collect();
        let refs: Vec<&Buf> = converted.iter().map(|c| c.as_ref()).collect();
        let owned = || converted.iter().filter(|c| matches!(c, std::borrow::Cow::Owned(_))).map(|c| crate::profile::buf_bytes(c)).sum();
        self.tock(Phase::Convert, t, owned);
        let k = &self.kernels[idx];
        let t = self.tick();
        let mut out = launch_buf(k);
        self.tock(Phase::Alloc, t, || crate::profile::buf_bytes(&out));
        let t = self.tick();
        self.exec.run(idx, &self.kernels[idx], &refs, &mut out)?;
        if let (Some(p), Some(t)) = (&self.profile, t) {
            let s = since(t);
            p.borrow_mut().launch(s, self.exec.overhead());
        }
        self.launches += 1;
        let (axes, kind, bound) = self.outs[idx].clone();
        if let Some(r) = self.recorder.as_deref_mut() {
            let shape: Vec<usize> = axes.iter().map(|a| a.len).collect();
            r.launch(idx, &self.kernels[idx], pack, &shape, (kind == BufKind::Idx).then_some(bound));
        }
        if self.profile.is_some() && converted.iter().any(|c| matches!(c, std::borrow::Cow::Owned(_))) {
            let t = self.tick();
            drop(refs);
            drop(converted);
            self.tock(Phase::Free, t, || 0);
        }
        Ok(match kind {
            BufKind::Real | BufKind::Quant { .. } => Value::Real(Tensor { axes, data: out }),
            BufKind::Idx => Value::Idx { bound, t: Tensor { axes, data: out } },
        })
    }

    // ----- the host program ----------------------------------------------------

    /// Give a tensor exactly the axis names of its type (as the interpreter does).
    fn retag(&self, v: Value, ty: &Ty) -> Value {
        match (v, ty) {
            (Value::Real(mut t), Ty::Real(s)) => {
                retag_axes(&mut t.axes, s);
                Value::Real(t)
            }
            (Value::Idx { bound, mut t }, Ty::Idx { shape, .. }) => {
                retag_axes(&mut t.axes, shape);
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
        let elab = self.elab;
        if let Some(r) = self.recorder.as_deref_mut() {
            let named: Vec<(String, u64)> = func.dims.iter().filter_map(|v| dims.get(v).map(|n| (elab.dim_name(*v), *n))).collect();
            r.enter(&func.name, named, &func.params.iter().map(|p| p.var).collect::<Vec<_>>(), func.params.iter().map(|p| p.name.clone()).collect());
        }
        let mut env: HashMap<VarId, Value> = HashMap::new();
        for (p, a) in func.params.iter().zip(args) {
            let a = self.retag(a, &p.ty);
            env.insert(p.var, a);
        }
        let mut closures: HashMap<VarId, (FuncId, Vec<Value>)> = HashMap::new();
        // Fusion groups: computed once per function, structural, so the same
        // plan serves every shape and every realization (`presk-realize`).
        let t = self.tick();
        let plan = self.fusion.entry(f).or_insert_with(|| presk_realize::plan(func)).clone();
        // The products that can take their `add` as an epilogue (plan 0.10.0
        // step 7), wanted where the plan states one or the launches are
        // captured; and the `add`s whose product launched nothing.
        let epis = if self.epilogue_stated || self.capture.is_some() { Some(self.epilogues.entry(f).or_insert_with(|| std::rc::Rc::new(presk_core::fusion::epilogues(func, &plan))).clone()) } else { None };
        let mut absorbed: HashMap<VarId, presk_core::fusion::Epilogue> = HashMap::new();
        self.tock(Phase::Groups, t, || 0);
        let outer_site = std::mem::take(&mut self.site);
        let outer_dtype = self.dtype;
        for l in &func.body.lets {
            // An interior `let` of a group never reaches memory: its value is
            // an expression inside the group's single loop nest.
            if plan.interior.contains(&l.var) {
                continue;
            }
            // The site a realization assigns a type to: a group's root `let`
            // in a named function (D34). A site the realization does not
            // mention takes its default, which is every site of a uniform one.
            let t = self.tick();
            self.site = format!("{}/%{}", func.name, l.var.0);
            self.dtype = self.realize.dtype_at(&self.site);
            self.tock(Phase::Site, t, || 0);
            // A product whose plan states its epilogue launches nothing: the
            // `add` that alone reads it launches the product's kernel.
            if self.epilogue_stated && self.layout_at(&self.site).epilogue == Some(true) {
                match epis.as_ref().and_then(|e| e.get(&l.var)) {
                    Some(e) => {
                        absorbed.insert(e.add, *e);
                        continue;
                    }
                    None => {
                        let why = presk_core::fusion::epilogue(func, &plan, l.var).err().unwrap_or_default();
                        return Err(RtError { code: "E7002", message: format!("the plan's layout of `{}`: an epilogue, and {why}", self.site), span: Some(l.span) });
                    }
                }
            }
            if let Some(e) = absorbed.remove(&l.var) {
                let v = self.epilogue_launch(func, &e, &env).map_err(|e| RtError { span: e.span.or(Some(l.span)), message: if e.message.starts_with("in `") || e.code == "E7002" { e.message } else { format!("in `{}` ({}): {}", func.name, l.name.clone().unwrap_or_else(|| format!("%{}", l.var.0)), e.message) }, code: e.code })?;
                let v = self.retag(v, &l.ty);
                if let Some(r) = self.recorder.as_deref_mut()
                    && let Rhs::Prim { args, .. } = &func.body.lets.iter().find(|p| p.var == e.product).expect("the product's let").rhs
                {
                    r.epilogue(l.var, args, e.other, self.dtype, &format!("{}/%{}", func.name, e.product.0), &self.site, l.name.as_deref());
                }
                env.insert(l.var, v);
                continue;
            }
            if let Some(g) = plan.groups.get(&l.var) {
                let v = self.fused_group(g, &env).map_err(|e| RtError {
                    span: e.span.or(Some(l.span)),
                    message: if e.message.starts_with("in `") { e.message } else { format!("in `{}` (fused group at %{}): {}", func.name, l.var.0, e.message) },
                    code: e.code,
                })?;
                let v = self.retag(v, &l.ty);
                if let Some(r) = self.recorder.as_deref_mut() {
                    r.fused(l.var, &g.inputs, self.dtype, &self.site, l.name.as_deref());
                }
                env.insert(l.var, v);
                continue;
            }
            if let Some(r) = self.recorder.as_deref_mut() {
                r.let_begin(l.var, &self.site, l.name.as_deref());
            }
            let v = match &l.rhs {
                Rhs::Closure { f, captured } => {
                    let t = self.tick();
                    let c: Vec<Value> = captured.iter().map(|c| env.get(c).cloned().ok_or_else(|| RtError { code: "E7001", message: "captured variable undefined".into(), span: Some(l.span) })).collect::<Result<_>>()?;
                    self.tock(Phase::Args, t, || c.iter().map(copied_bytes).sum());
                    closures.insert(l.var, (*f, c));
                    if let Some(r) = self.recorder.as_deref_mut() {
                        r.closure(l.var, *f, captured);
                    }
                    Value::Nat(0)
                }
                rhs => self.rhs(rhs, &l.ty, &dims, &env, &closures).map_err(|e| RtError {
                    span: e.span.or(Some(l.span)),
                    message: if e.message.starts_with("in `") { e.message } else { format!("in `{}` ({}): {}", func.name, l.name.clone().unwrap_or_else(|| format!("%{}", l.var.0)), e.message) },
                    code: e.code,
                })?,
            };
            let v = self.retag(v, &l.ty);
            if let Some(r) = self.recorder.as_deref_mut()
                && !matches!(l.rhs, Rhs::Closure { .. })
            {
                r.let_end(l.var, &l.rhs, &v, self.dtype, &self.site, l.name.as_deref());
            }
            if self.capture.is_some()
                && let Some(e) = epis.as_ref().and_then(|m| m.values().find(|e| e.add == l.var))
            {
                self.note_add(func, e, &env);
            }
            env.insert(l.var, v);
        }
        if let Some(r) = self.recorder.as_deref_mut() {
            r.leave(func.body.result);
        }
        self.site = outer_site;
        self.dtype = outer_dtype;
        let result = env.remove(&func.body.result).ok_or_else(|| RtError { code: "E7001", message: "result variable undefined".into(), span: Some(func.span) });
        if self.profile.is_some() {
            // Freed in the order they would be at the end of the function.
            let t = self.tick();
            drop(plan);
            self.tock_time(Phase::Groups, t);
            let bytes = freed_bytes(env.values().chain(closures.values().flat_map(|(_, c)| c)));
            let t = self.tick();
            drop(closures);
            drop(env);
            self.tock(Phase::Free, t, || bytes);
        }
        result
    }

    /// One launch for a whole fusion group. The result's axes are computed by
    /// replaying the steps' broadcasts, not read off the type, because a type
    /// inside a module function may carry an open prefix.
    fn fused_group(&mut self, g: &presk_realize::Group, env: &HashMap<VarId, Value>) -> Result<Value> {
        let t = self.tick();
        let vals: Vec<Value> = g
            .inputs
            .iter()
            .map(|v| env.get(v).cloned().ok_or_else(|| RtError { code: "E7001", message: format!("variable %{} undefined", v.0), span: None }))
            .collect::<Result<_>>()?;
        self.tock(Phase::Operands, t, || vals.iter().map(copied_bytes).sum());
        let r = self.fused_on(g, &vals);
        let t = self.tick();
        drop(vals);
        self.tock_time(Phase::Operands, t);
        r
    }

    /// The group applied to one set of values. A parameter tree maps the whole
    /// group over its leaves — the optimizer's update is one chain over a tree,
    /// and without this it would launch every step again for every leaf.
    fn fused_on(&mut self, g: &presk_realize::Group, vals: &[Value]) -> Result<Value> {
        if let Some(template) = vals.iter().find(|v| matches!(v, Value::Record(_) | Value::Tuple(_))) {
            let t = self.tick();
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
            self.tock(Phase::Tuples, t, || copied_bytes(&template) + per_input.iter().flatten().map(copied_bytes).sum::<u64>());
            let mut leaves = Vec::with_capacity(n);
            for i in 0..n {
                let t = self.tick();
                let args: Vec<Value> = per_input.iter().map(|xs| if xs.len() == 1 { xs[0].clone() } else { xs[i].clone() }).collect();
                self.tock(Phase::Tuples, t, || args.iter().map(copied_bytes).sum());
                let leaf = self.fused_on(g, &args);
                let t = self.tick();
                drop(args);
                self.tock_time(Phase::Tuples, t);
                leaves.push(leaf?);
            }
            let t = self.tick();
            let tree = rebuild(&template, &mut leaves.into_iter());
            drop(per_input);
            drop(template);
            self.tock(Phase::Tuples, t, || 0);
            return Ok(tree);
        }
        let t = self.tick();
        let mut ins: Vec<Tensor> = Vec::with_capacity(vals.len());
        for value in vals {
            ins.push(self.as_tensor(value)?);
        }
        self.tock(Phase::AsTensor, t, || 0);
        let in_axes: Vec<Vec<Axis>> = ins.iter().map(|t| t.axes.clone()).collect();
        let mut shapes: Vec<Vec<Axis>> = Vec::with_capacity(g.steps.len());
        for s in &g.steps {
            let of = |src: &presk_realize::Src| -> Vec<Axis> {
                match src {
                    presk_realize::Src::Input(k) => in_axes[*k].clone(),
                    presk_realize::Src::Step(k) => shapes[*k].clone(),
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
        let t = self.tick();
        let layout = self.lowered_layout(&NestKind::Other)?;
        self.tock(Phase::Layout, t, || 0);
        if self.capture.is_some() {
            let accum = self.realize.accum_at(&self.site);
            self.record(LaunchOp::Fused { steps: g.steps.clone(), out: out.clone() }, &operands, dtype, accum, &ins.iter().map(|t| kernel_input(&t.data, dtype, false)).collect::<Vec<_>>());
        }
        let t = self.tick();
        let key = format!("fused|{}|{dtype}|{}|{}|{}", g.signature(), layout.key(), operands.iter().map(|o| lower::show(&o.axes)).collect::<Vec<_>>().join(";"), lower::show(&out));
        let steps = g.steps.clone();
        self.tock(Phase::Key, t, || 0);
        let bufs: Vec<&Buf> = ins.iter().map(|t| &t.data).collect();
        self.fused += g.steps.len() as u64 - 1;
        let r = self.launch(key, move || lower::fused(&steps, &operands, &out, dtype), &bufs, true);
        drop(bufs);
        let t = self.tick();
        drop(ins);
        self.tock_time(Phase::AsTensor, t);
        r
    }

    /// The product's operands, and the `add`'s other, of an epilogue.
    fn epilogue_operands<'v>(&self, func: &'p presk_core::Func, e: &presk_core::fusion::Epilogue, env: &'v HashMap<VarId, Value>) -> std::result::Result<(&'p PrimOp, &'p Ty, [&'v Value; 3]), String> {
        let prod = func.body.lets.iter().find(|l| l.var == e.product).ok_or("the product's binding is missing")?;
        let Rhs::Prim { op, args } = &prod.rhs else { return Err("the product is not a primitive".into()) };
        let get = |v: &VarId| env.get(v).ok_or_else(|| format!("variable %{} undefined", v.0));
        Ok((op, &prod.ty, [get(&args[0])?, get(&args[1])?, get(&e.other)?]))
    }

    /// The `add` `e` evaluated by its product's kernel, as its epilogue (plan
    /// 0.10.0 step 7): one launch (or a pack's and the contraction's) of the
    /// product's nest, under its layout, at its site, in its type, which is
    /// the `add`'s or the plan is refused (`E7002`).
    fn epilogue_launch(&mut self, func: &'p presk_core::Func, e: &presk_core::fusion::Epilogue, env: &HashMap<VarId, Value>) -> Result<Value> {
        let prod_site = format!("{}/%{}", func.name, e.product.0);
        let fail = |m: String| RtError { code: "E7002", message: format!("the plan's layout of `{prod_site}`: {m}"), span: None };
        let dp = self.realize.dtype_at(&prod_site);
        if dp != self.dtype {
            return Err(fail(format!("an epilogue adds in the product's kernel, in its type, {dp}, and the `add` that reads it, `{}`, is typed {}", self.site, self.dtype)));
        }
        let (op, prod_ty, [x, w, o]) = self.epilogue_operands(func, e, env).map_err(|m| RtError { code: "E7001", message: m, span: None })?;
        let tensor = |v: &'_ Value, what: &str| -> Result<Tensor> {
            match v {
                Value::Real(t) => Ok(t.clone()),
                other => Err(fail(format!("the {what} is a {}, and an epilogue takes real tensors", other.kind()))),
            }
        };
        let (tx, tw, to) = (tensor(x, "product's first operand")?, tensor(w, "product's second operand")?, tensor(o, "`add`'s other operand")?);
        let add_site = std::mem::replace(&mut self.site, prod_site);
        let r = self.epilogue_prim(op, prod_ty, [&tx, &tw, &to], e.first);
        self.site = add_site;
        r
    }

    /// The `add`'s output, when the product's kernel can compute it: the
    /// contraction nest at these operands, and the epilogue.
    fn epilogue_of(op: &PrimOp, prod_ty: &Ty, ins: [&Tensor; 3], first: bool, order: presk_rt::ReductionOrder) -> std::result::Result<(lower::Contraction, lower::Epilogue), String> {
        let operands = [Operand { axes: ins[0].axes.clone(), kind: BufKind::Real }, Operand { axes: ins[1].axes.clone(), kind: BufKind::Real }];
        let Some(c) = lower::Contraction::of(op, &operands, order) else {
            return Err(format!("an epilogue is a decision of a contraction nest, and `{}` of {} and {} is not one", op.name(), lower::show(&ins[0].axes), lower::show(&ins[1].axes)));
        };
        let mut prod = c.out.clone();
        if let Ty::Real(s) = prod_ty {
            retag_axes(&mut prod, s);
        }
        let out = if first { lower::result_axes(&prod, &ins[2].axes) } else { lower::result_axes(&ins[2].axes, &prod) };
        match out {
            Some(out) if out.len() == prod.len() && out.iter().zip(&prod).all(|(a, b)| a.len == b.len) => Ok((c, lower::Epilogue { first, out })),
            _ => Err(format!("the `add`'s operand {} does not broadcast into the product's output {}", lower::show(&ins[2].axes), lower::show(&prod))),
        }
    }

    /// The product's kernel with its epilogue, at the product's site.
    fn epilogue_prim(&mut self, op: &PrimOp, prod_ty: &Ty, ins: [&Tensor; 3], first: bool) -> Result<Value> {
        let t = self.tick();
        let fail = |site: &str, m: String| RtError { code: "E7002", message: format!("the plan's layout of `{site}`: {m}"), span: None };
        let (dtype, order) = (self.dtype, self.realize.order);
        let (c, epi) = Self::epilogue_of(op, prod_ty, ins, first, order).map_err(|m| fail(&self.site, m))?;
        let operands: Vec<Operand> = ins.iter().map(|t| Operand { axes: t.axes.clone(), kind: BufKind::Real }).collect();
        let kind = NestKind::Contraction(c.kind(dtype, self.realize.accum_at(&self.site)));
        let layout = self.lowered_layout(&kind)?;
        // Lanes read the operand's columns: contiguous, or one value.
        if layout.lanes.unwrap_or(1) > 1 && layout.form == Some(presk_rt::layout::Form::Tiled) {
            let map = lower::align(&ins[2].axes, &epi.out).unwrap_or_default();
            let j = epi.out.len() - 1;
            if map.iter().position(|p| *p == j).is_some_and(|q| q + 1 != map.len()) {
                return Err(fail(&self.site, format!("{} lanes, and the `add`'s operand {} runs along J, but not innermost: an epilogue in lanes needs it contiguous along J, or constant", layout.lanes.unwrap_or(1), lower::show(&ins[2].axes))));
            }
        }
        self.tock(Phase::Layout, t, || 0);
        if self.capture.is_some() {
            let accum = self.realize.accum_at(&self.site);
            self.record(LaunchOp::Prim(op.clone()), &operands[..2], dtype, accum, &[kernel_input(&ins[0].data, dtype, true), kernel_input(&ins[1].data, dtype, true)]);
            let at = EpilogueAt { operand: operands[2].clone(), input: kernel_input(&ins[2].data, dtype, true), epilogue: epi.clone() };
            let site = self.site.clone();
            self.note_epilogue(&site, dtype, &operands[..2], at);
        }
        let t = self.tick();
        let quant: Vec<Option<BufKind>> = ins.iter().map(|t| match &t.data {
            Buf::Quant(q) => BufKind::of_quant(q, &t.axes.iter().map(|a| a.len).collect::<Vec<_>>()),
            _ => None,
        }).collect();
        let key = format!("{op:?}+add{}|{dtype}|{}|{}|{}|{}|{:?}", if first { "" } else { " (second)" }, order.name(), layout.key(), operands.iter().map(|o| lower::show(&o.axes)).collect::<Vec<_>>().join(";"), lower::show(&epi.out), quant);
        self.tock(Phase::Key, t, || 0);
        let bufs: Vec<&Buf> = ins.iter().map(|t| &t.data).collect();
        // One launch where two were: the `add`'s.
        self.fused += 1;
        if layout.pack == Some(true) {
            let w = operands[1].clone();
            let (cc, q1) = (c.clone(), quant[1]);
            let panel = self.launch(format!("pack|{key}"), move || {
                let mut l = lower::pack_panel(&cc, dtype, &w)?;
                if let Some(kind) = q1 {
                    l.kernel.inputs[0].kind = kind;
                }
                Ok(l)
            }, &[bufs[1]], false)?;
            let Value::Real(panel) = panel else { return err("E7001", "a pack gives a real tensor") };
            let ops = vec![operands[0].clone(), Operand { axes: panel.axes.clone(), kind: BufKind::Real }, operands[2].clone()];
            let (packed, q0, q2) = (c.packed(), quant[0], quant[2]);
            return self.launch(format!("packed|{key}"), move || {
                let mut l = lower::emit_contraction_with(&packed, dtype, &ops, &layout, Some(&epi))?;
                if let Some(kind) = q0 {
                    l.kernel.inputs[0].kind = kind;
                }
                if let Some(kind) = q2 {
                    l.kernel.inputs[2].kind = kind;
                }
                Ok(l)
            }, &[bufs[0], &panel.data, bufs[2]], true);
        }
        self.launch(key, move || {
            let mut l = lower::emit_contraction_with(&c, dtype, &operands, &layout, Some(&epi))?;
            for (spec, q) in l.kernel.inputs.iter_mut().zip(&quant) {
                if let Some(kind) = q {
                    spec.kind = *kind;
                }
            }
            Ok(l)
        }, &bufs, true)
    }

    /// When capturing: the `add` `e` was launched apart, its product before
    /// it; notes on the product's launch the epilogue it could take, where
    /// the two sites have one type and the operand fits the output.
    fn note_add(&mut self, func: &'p presk_core::Func, e: &presk_core::fusion::Epilogue, env: &HashMap<VarId, Value>) {
        let prod_site = format!("{}/%{}", func.name, e.product.0);
        if self.realize.dtype_at(&prod_site) != self.dtype {
            return;
        }
        let Ok((op, prod_ty, [Value::Real(x), Value::Real(w), Value::Real(o)])) = self.epilogue_operands(func, e, env) else { return };
        let Ok((_, epi)) = Self::epilogue_of(op, prod_ty, [x, w, o], e.first, self.realize.order) else { return };
        let dtype = self.dtype;
        let operands = [Operand { axes: x.axes.clone(), kind: BufKind::Real }, Operand { axes: w.axes.clone(), kind: BufKind::Real }];
        let at = EpilogueAt { operand: Operand { axes: o.axes.clone(), kind: BufKind::Real }, input: kernel_input(&o.data, dtype, true), epilogue: epi };
        self.note_epilogue(&prod_site, dtype, &operands, at);
    }

    fn rhs(&mut self, rhs: &Rhs, ty: &Ty, dims: &Dims, env: &HashMap<VarId, Value>, closures: &HashMap<VarId, (FuncId, Vec<Value>)>) -> Result<Value> {
        let get = |v: &VarId| -> Result<Value> { env.get(v).cloned().ok_or_else(|| RtError { code: "E7001", message: format!("variable %{} undefined", v.0), span: None }) };
        match rhs {
            Rhs::Lit(Lit::Nat(n)) => Ok(Value::Nat(*n)),
            Rhs::Lit(Lit::Float(s)) => Ok(Value::F64(s.parse().map_err(|_| RtError { code: "E7001", message: format!("bad float literal `{s}`"), span: None })?)),
            Rhs::Prim { op, args } => {
                let t = self.tick();
                let vals: Vec<Value> = args.iter().map(get).collect::<Result<_>>()?;
                self.tock(Phase::Operands, t, || vals.iter().map(copied_bytes).sum());
                let r = self.prim(op, &vals, ty, dims);
                let t = self.tick();
                drop(vals);
                self.tock_time(Phase::Operands, t);
                r
            }
            Rhs::Call { f, dim_args, args } => {
                let t = self.tick();
                let vals: Vec<Value> = args.iter().map(get).collect::<Result<_>>()?;
                self.tock(Phase::Args, t, || vals.iter().map(copied_bytes).sum());
                let callee_dims = self.callee_dims(dim_args, dims);
                if let Some(r) = self.recorder.as_deref_mut() {
                    r.call(args);
                }
                self.call_fn(*f, callee_dims, vals)
            }
            Rhs::Tuple(vs) => {
                let t = self.tick();
                let v = Value::Tuple(vs.iter().map(get).collect::<Result<_>>()?);
                self.tock(Phase::Tuples, t, || copied_bytes(&v));
                Ok(v)
            }
            // A projection or a field clones the element it selects and
            // nothing else: a gradient that keeps its residuals projects them
            // out of tuples that hold every layer's (0.10.0 step 4), as the
            // interpreter does since 0.6.0 step 8.
            Rhs::Proj(v, i) => {
                let t = self.tick();
                let r = match env.get(v) {
                    Some(Value::Tuple(vs)) => vs.get(*i).cloned().ok_or_else(|| RtError { code: "E7001", message: format!("tuple has no element {i}"), span: None }),
                    Some(other) => err("E7001", format!("projection on a {}", other.kind())),
                    None => get(v),
                };
                self.tock(Phase::Tuples, t, || r.as_ref().map_or(0, copied_bytes));
                r
            }
            Rhs::Record(fields) => {
                let t = self.tick();
                let v = Value::Record(fields.iter().map(|(n, v)| Ok((n.clone(), get(v)?))).collect::<Result<_>>()?);
                self.tock(Phase::Tuples, t, || copied_bytes(&v));
                Ok(v)
            }
            Rhs::Field(v, name) => {
                let t = self.tick();
                let r = match env.get(v) {
                    Some(Value::Record(fs)) => fs.iter().find(|(n, _)| n == name).map(|(_, v)| v.clone()).ok_or_else(|| RtError { code: "E7001", message: format!("record has no field `{name}`"), span: None }),
                    Some(other) => err("E7001", format!("field `{name}` of a {}", other.kind())),
                    None => get(v),
                };
                self.tock(Phase::Tuples, t, || r.as_ref().map_or(0, copied_bytes));
                r
            }
            Rhs::Closure { .. } => Ok(Value::Nat(0)),
            Rhs::Grad(_) => err("E7001", "`grad` reached the kernel backend: run automatic differentiation first"),
            Rhs::Apply { fun, args } => {
                let Some((f, captured)) = closures.get(fun) else {
                    return err("E7001", "application of an unknown function value");
                };
                let t = self.tick();
                let mut all = captured.clone();
                for a in args {
                    all.push(get(a)?);
                }
                self.tock(Phase::Args, t, || all.iter().map(copied_bytes).sum());
                if let Some(r) = self.recorder.as_deref_mut() {
                    r.apply(*fun, args);
                }
                self.call_fn(*f, dims.clone(), all)
            }
            Rhs::Fold { body, dim_args, n, stacked, init, extra, index, reverse, scan } => {
                let count = self.dim(n, dims)? as usize;
                let t = self.tick();
                let sts: Vec<Value> = stacked.iter().map(get).collect::<Result<_>>()?;
                let ex: Vec<Value> = extra.iter().map(get).collect::<Result<_>>()?;
                let mut acc = get(init)?;
                self.tock(Phase::Fold, t, || sts.iter().chain(&ex).map(copied_bytes).sum::<u64>() + copied_bytes(&acc));
                let callee_dims = self.callee_dims(dim_args, dims);
                let order: Vec<usize> = if *reverse { (0..count).rev().collect() } else { (0..count).collect() };
                let mut pre: Vec<(usize, Value)> = Vec::new();
                if let Some(r) = self.recorder.as_deref_mut() {
                    r.fold_begin(stacked, extra, *init, *index, *reverse, *scan, count);
                }
                for i in order {
                    let t = self.tick();
                    if *scan {
                        pre.push((i, acc.clone()));
                    }
                    self.tock(Phase::Fold, t, || if *scan { copied_bytes(&acc) } else { 0 });
                    let mut args = Vec::with_capacity(sts.len() + ex.len() + 2);
                    if *index {
                        args.push(Value::Nat(i as u64));
                    }
                    let t = self.tick();
                    for s in &sts {
                        args.push(slice_tree(s, i));
                    }
                    self.tock(Phase::Slice, t, || args.iter().map(copied_bytes).sum());
                    args.push(acc);
                    let t = self.tick();
                    args.extend(ex.iter().cloned());
                    self.tock(Phase::Fold, t, || ex.iter().map(copied_bytes).sum());
                    if let Some(r) = self.recorder.as_deref_mut() {
                        r.fold_iter(i);
                    }
                    acc = self.call_fn(*body, callee_dims.clone(), args)?;
                    if let Some(r) = self.recorder.as_deref_mut() {
                        r.fold_iter_end();
                    }
                }
                // No iteration of a scan: the stack of none, from the fold's
                // type, which the recorder takes as it is.
                let empty = match ty {
                    Ty::Tuple(ts) if *scan && pre.is_empty() && ts.len() == 2 => Some(self.empty_stack(&ts[1], dims)?),
                    _ => None,
                };
                if let Some(r) = self.recorder.as_deref_mut() {
                    r.fold_end(empty.as_ref());
                }
                let r = if *scan {
                    pre.sort_by_key(|(i, _)| *i);
                    let t = self.tick();
                    let parts: Vec<Value> = pre.into_iter().map(|(_, v)| v).collect();
                    let stack = match empty {
                        Some(v) => Ok(v),
                        None => stack_trees(&parts),
                    };
                    drop(parts);
                    self.tock(Phase::Slice, t, || stack.as_ref().map_or(0, copied_bytes));
                    Ok(Value::Tuple(vec![acc, stack?]))
                } else {
                    Ok(acc)
                };
                let t = self.tick();
                drop(ex);
                drop(sts);
                self.tock_time(Phase::Fold, t);
                r
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
    pub fn prim(&mut self, op: &PrimOp, args: &[Value], ty: &Ty, dims: &Dims) -> Result<Value> {
        // The primitives the host evaluates without a kernel are timed whole,
        // by what they do; the others time their regions themselves.
        let phase = match op {
            PrimOp::DropoutMask => Some(Phase::Dropout),
            PrimOp::ZerosLike | PrimOp::Normal { .. } | PrimOp::XavierUniform { .. } | PrimOp::SplitKey { .. } | PrimOp::Sample => Some(Phase::HostPrim),
            PrimOp::MergeAxes { .. } | PrimOp::SplitAxes { .. } | PrimOp::ReshapeLike => Some(Phase::View),
            PrimOp::SetField { .. } | PrimOp::SetElem { .. } => Some(Phase::Tuples),
            PrimOp::SliceParams { .. } | PrimOp::StackParams { .. } | PrimOp::SetSlice | PrimOp::ScatterParams { .. } => Some(Phase::Slice),
            _ => None,
        };
        match phase {
            Some(phase) if self.profile.is_some() => {
                let t = self.tick();
                let r = self.prim_on(op, args, ty, dims);
                // A field or an element set shares the rest of the tree; a
                // slice, a stack, a slice set or a scatter are views and parts
                // of their operands.
                let size = if matches!(phase, Phase::Tuples | Phase::Slice) { copied_bytes } else { value_bytes };
                self.tock(phase, t, || r.as_ref().map_or(0, size));
                r
            }
            _ => self.prim_on(op, args, ty, dims),
        }
    }

    #[allow(clippy::only_used_in_recursion)]
    fn prim_on(&mut self, op: &PrimOp, args: &[Value], ty: &Ty, dims: &Dims) -> Result<Value> {
        let a = |i: usize| -> Result<&Value> { args.get(i).ok_or_else(|| RtError { code: "E7001", message: format!("`{}`: missing argument {i}", op.name()), span: None }) };
        match op {
            // ----- host: structure -----
            PrimOp::ZerosLike => Ok(a(0)?.map_leaves(&mut |leaf| match leaf {
                Value::Real(t) => Value::Real(Tensor { axes: t.axes.clone(), data: self.filled(0.0, t.numel()) }),
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
            PrimOp::SplitAxes { a: an, n1, l1, n2, l2 } => {
                let t = self.as_tensor(a(0)?)?;
                let Some(p) = t.axes.iter().position(|x| x.name.as_deref() == Some(an.as_str())) else {
                    return err("E7001", format!("split: no axis `{an}` in {}", lower::show(&t.axes)));
                };
                let (u, v) = (self.dim(l1, dims)? as usize, self.dim(l2, dims)? as usize);
                if u * v != t.axes[p].len {
                    return err("E7001", format!("split: `{an}` has length {} but {u} · {v} = {}", t.axes[p].len, u * v));
                }
                let mut axes = t.axes[..p].to_vec();
                axes.push(Axis { name: Some(n1.clone()), len: u });
                axes.push(Axis { name: Some(n2.clone()), len: v });
                axes.extend_from_slice(&t.axes[p + 1..]);
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
                if args.is_empty() {
                    return self.empty_stack(ty, dims);
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
                let t = self.tick();
                a(0)?.map_leaves(&mut |leaf| {
                    if let Value::Real(t) = leaf {
                        leaves.push(t.clone());
                    }
                    leaf.clone()
                });
                self.tock(Phase::Tuples, t, || 0);
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
                let t = self.tick();
                let layout = self.lowered_layout(&NestKind::Other)?;
                self.tock(Phase::Layout, t, || 0);
                let t = self.tick();
                let key = format!("{}|{dtype}|{}|{}", op.name(), layout.key(), lower::show(&axes));
                self.tock(Phase::Key, t, || 0);
                self.launch(key, move || Ok(lower::fill(&op.name(), &axes, value, dtype)), &[], true)
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
                // Outside training every element is kept, `x · 1 = x`: the
                // mask is the host's buffer of ones, drawn nowhere (plan
                // 0.10.0 step 6).
                let data = if self.training { self.dropout_mask(&key, n, rate) } else { self.filled(1.0, n) };
                Ok(Value::Real(Tensor { axes, data }))
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
                // From its key: one uniform per row, the row's rank its
                // draw's (plan 0.10.0 step 7b, the reference's "Draws").
                let mut rng = key_of(a(0)?)?.rng();
                let Value::Real(t) = a(1)? else { return err("E7001", "sample: expected a real tensor") };
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
                    // The `libm` crate's `exp`, musl's, as the native
                    // kernels call, not the platform's (`f64::exp`): the
                    // same draws on every machine, and the C export's
                    // `presk_exp` (plan 0.9.0 step 6, D83).
                    let exps: Vec<f64> = row.iter().map(|x| libm::exp(x - m)).collect();
                    let z: f64 = exps.iter().sum();
                    let u = uniform01(&mut rng);
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
                Ok(Value::Idx { bound: v as u64, t: Tensor { axes: t.axes[..t.axes.len() - 1].to_vec(), data: Buf::I64(out.into()) } })
            }
            PrimOp::Positions { axis, len } => {
                let n = self.dim(len, dims)? as usize;
                let (axis, dtype) = (axis.clone(), self.dtype);
                let t = self.tick();
                let layout = self.lowered_layout(&NestKind::Other)?;
                self.tock(Phase::Layout, t, || 0);
                let t = self.tick();
                let key = format!("positions {axis} {n} {dtype} {}", layout.key());
                self.tock(Phase::Key, t, || 0);
                self.launch(key, move || Ok(lower::positions(&axis, n, dtype)), &[], true)
            }
            PrimOp::OneHot => {
                let Value::Idx { bound, t } = a(0)? else {
                    return err("E7001", format!("`one_hot`: expected an index tensor, found a {}", a(0)?.kind()));
                };
                let mut axes = t.axes.clone();
                axes.push(Axis { name: None, len: *bound as usize });
                let dtype = self.dtype;
                let t0 = self.tick();
                let layout = self.lowered_layout(&NestKind::Other)?;
                self.tock(Phase::Layout, t0, || 0);
                let t0 = self.tick();
                let (key, ax) = (format!("one_hot {} {dtype} {}", lower::show(&axes), layout.key()), axes.clone());
                self.tock(Phase::Key, t0, || 0);
                self.launch(key, move || Ok(lower::one_hot(&ax, dtype)), &[&t.data], true)
            }
            PrimOp::CausalMask { axis, len } => {
                let n = self.dim(len, dims)? as usize;
                let (axis, dtype) = (axis.clone(), self.dtype);
                let t = self.tick();
                let layout = self.lowered_layout(&NestKind::Other)?;
                self.tock(Phase::Layout, t, || 0);
                let t = self.tick();
                let key = format!("causal_mask {axis} {n} {dtype} {}", layout.key());
                self.tock(Phase::Key, t, || 0);
                self.launch(key, move || Ok(lower::causal_mask(&axis, n, dtype)), &[], true)
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
                        let t = self.tick();
                        let mut out = Vec::new();
                        collect_leaves(x, &mut out);
                        self.tock(Phase::Tuples, t, || copied_bytes(x));
                        let mut mapped = Vec::new();
                        for leaf in out {
                            mapped.push(self.prim(op, &[leaf, y.clone()], ty, dims)?);
                        }
                        let t = self.tick();
                        let tree = rebuild(x, &mut mapped.into_iter());
                        self.tock(Phase::Tuples, t, || 0);
                        Ok(tree)
                    }
                    (Value::Real(t), Value::Record(_) | Value::Tuple(_)) if t.rank() == 0 => {
                        let t = self.tick();
                        let mut out = Vec::new();
                        collect_leaves(y, &mut out);
                        self.tock(Phase::Tuples, t, || copied_bytes(y));
                        let mut mapped = Vec::new();
                        for leaf in out {
                            mapped.push(self.prim(op, &[x.clone(), leaf], ty, dims)?);
                        }
                        let t = self.tick();
                        let tree = rebuild(y, &mut mapped.into_iter());
                        self.tock(Phase::Tuples, t, || 0);
                        Ok(tree)
                    }
                    (Value::Record(_) | Value::Tuple(_), _) | (_, Value::Record(_) | Value::Tuple(_)) => {
                        let t = self.tick();
                        let (mut lx, mut ly) = (Vec::new(), Vec::new());
                        collect_leaves(x, &mut lx);
                        collect_leaves(y, &mut ly);
                        self.tock(Phase::Tuples, t, || copied_bytes(x) + copied_bytes(y));
                        if lx.len() != ly.len() {
                            return err("E7001", format!("{}: structure mismatch", op.name()));
                        }
                        let mut mapped = Vec::new();
                        for (p, q) in lx.into_iter().zip(ly) {
                            mapped.push(self.prim(op, &[p, q], ty, dims)?);
                        }
                        let t = self.tick();
                        let tree = rebuild(x, &mut mapped.into_iter());
                        self.tock(Phase::Tuples, t, || 0);
                        Ok(tree)
                    }
                    _ => {
                        let t = self.tick();
                        let (tx, ty_) = (self.as_tensor(x)?, self.as_tensor(y)?);
                        self.tock(Phase::AsTensor, t, || 0);
                        let r = self.tensor_prim(op, &[&tx, &ty_]);
                        let t = self.tick();
                        drop(tx);
                        drop(ty_);
                        self.tock_time(Phase::AsTensor, t);
                        r
                    }
                }
            }
            // ----- everything else: unary on trees / hyperparameters, or a kernel -----
            _ => {
                if args.len() == 1 {
                    match a(0)? {
                        Value::Record(_) | Value::Tuple(_) => {
                            let x = a(0)?;
                            let t = self.tick();
                            let mut leaves = Vec::new();
                            collect_leaves(x, &mut leaves);
                            self.tock(Phase::Tuples, t, || copied_bytes(x));
                            let mut mapped = Vec::new();
                            for leaf in leaves {
                                mapped.push(self.prim(op, &[leaf], ty, dims)?);
                            }
                            let t = self.tick();
                            let tree = rebuild(x, &mut mapped.into_iter());
                            self.tock(Phase::Tuples, t, || 0);
                            return Ok(tree);
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
                let t = self.tick();
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
                self.tock(Phase::AsTensor, t, || 0);
                let refs: Vec<&Tensor> = tensors.iter().collect();
                let r = self.tensor_prim_kinds(op, &refs, &kinds);
                drop(refs);
                let t = self.tick();
                drop(tensors);
                self.tock_time(Phase::AsTensor, t);
                r
            }
        }
    }

    /// A training dropout mask of `n` elements in the site's type: element
    /// `e` is `1/(1 − rate)` when the key's `e`-th uniform number is below
    /// `1 − rate`, and zero otherwise. That number is the stream's 64-bit
    /// word `e`, its 32-bit words `2e` and `2e + 1`, so a piece of the mask
    /// starting at `e` is drawn from the stream sought to word `2e`: the
    /// pieces are drawn apart, on the executor's workers, and give the bits
    /// one stream drawn in order gives (plan 0.10.0 step 6).
    fn dropout_mask(&mut self, key: &Key, n: usize, rate: f64) -> Buf {
        let (keep, scale) = (1.0 - rate, 1.0 / (1.0 - rate));
        let x = move |u: f64| if u < keep { scale } else { 0.0 };
        // Each element in the site's type as `Buf::of` rounds it.
        match self.dtype {
            Dtype::F64 => Buf::F64(self.draw(key, n, x)),
            Dtype::F32 => Buf::F32(self.draw(key, n, move |u| x(u) as f32)),
            Dtype::Bf16 => Buf::Bf16(self.draw(key, n, move |u| presk_rt::f32_to_bf16(x(u) as f32))),
        }
    }

    /// `f(u_0)`, …, `f(u_{n−1})`, `u_e` the key's `e`-th uniform number, in
    /// pieces of at least `DRAWN_APART` elements on the executor's workers.
    fn draw<T: presk_rt::Pooled>(&mut self, key: &Key, n: usize, f: impl Fn(f64) -> T + Sync) -> presk_rt::Elems<T> {
        const DRAWN_APART: usize = 1 << 16;
        let mut out = presk_rt::Elems::fresh(n);
        let workers = self.exec.workers().unwrap_or(1).min(n / DRAWN_APART).max(1);
        let chunk = n.div_ceil(workers).max(1);
        let pieces: Vec<std::sync::Mutex<&mut [T]>> = out.chunks_mut(chunk).map(std::sync::Mutex::new).collect();
        self.exec.parallel(pieces.len(), &|p| {
            let mut piece = pieces[p].lock().expect("a piece of a mask");
            let mut rng = key.rng();
            rng.set_word_pos(2 * (p * chunk) as u128);
            for y in piece.iter_mut() {
                *y = f(uniform01(&mut rng));
            }
        });
        drop(pieces);
        out
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
        let t = self.tick();
        let operands: Vec<Operand> = ins.iter().zip(kinds).map(|(t, k)| Operand { axes: t.axes.clone(), kind: *k }).collect();
        let dtype = self.dtype;
        // The declared reduction order selects the kernel, so it belongs in
        // the cache key: two realizations that differ only in their order are
        // two different programs and must not share a compiled loop nest.
        let order = self.realize.order;
        // The nest's layout (0.7.0 step 1): the plan's, checked against this
        // launch's kind and shapes, over the backend's default. It is part of
        // the key, so two nests of one shape with two layouts never share a
        // compiled kernel.
        let kind = crate::plan::nest_kind(op, &operands, order, dtype, self.realize.accum_at(&self.site));
        let layout = self.lowered_layout(&kind)?;
        self.tock(Phase::Layout, t, || 0);
        if self.capture.is_some() {
            let accum = self.realize.accum_at(&self.site);
            self.record(LaunchOp::Prim(op.clone()), &operands, dtype, accum, &ins.iter().map(|t| kernel_input(&t.data, dtype, true)).collect::<Vec<_>>());
        }
        let t = self.tick();
        // Integer weights: the lowering sees a real operand and emits real
        // loads, and the kernel's input is marked afterwards, so the loads
        // dequantise (D44). The marking is part of the kernel, so it is part
        // of the key.
        let quant: Vec<Option<BufKind>> = ins.iter().map(|t| match &t.data {
            Buf::Quant(q) => BufKind::of_quant(q, &t.axes.iter().map(|a| a.len).collect::<Vec<_>>()),
            _ => None,
        }).collect();
        let key = format!("{op:?}|{dtype}|{}|{}|{}|{:?}", order.name(), layout.key(), operands.iter().map(|o| format!("{}{:?}", lower::show(&o.axes), o.kind)).collect::<Vec<_>>().join(";"), quant);
        self.tock(Phase::Key, t, || 0);
        let bufs: Vec<&Buf> = ins.iter().map(|t| &t.data).collect();
        // A pack (0.7.0 step 3): two launches for one nest, the copy of
        // operand 1 to its panel, then the contraction reading the panel,
        // which is NN (spec question 66).
        if layout.pack == Some(true)
            && let Some(c) = lower::Contraction::of(op, &operands, order)
        {
            let w = operands[1].clone();
            let (cc, q1) = (c.clone(), quant[1]);
            let panel = self.launch(format!("pack|{key}"), move || {
                let mut l = lower::pack_panel(&cc, dtype, &w)?;
                if let Some(kind) = q1 {
                    l.kernel.inputs[0].kind = kind;
                }
                Ok(l)
            }, &[bufs[1]], false)?;
            let Value::Real(panel) = panel else { return err("E7001", "a pack gives a real tensor") };
            let ops = vec![operands[0].clone(), Operand { axes: panel.axes.clone(), kind: BufKind::Real }];
            let (packed, q0) = (c.packed(), quant[0]);
            return self.launch(format!("packed|{key}"), move || {
                let mut l = lower::emit_contraction(&packed, dtype, &ops, &layout)?;
                if let Some(kind) = q0 {
                    l.kernel.inputs[0].kind = kind;
                }
                Ok(l)
            }, &[bufs[0], &panel.data], true);
        }
        let op = op.clone();
        self.launch(key, move || {
            let mut l = lower::lower_planned(&op, &operands, dtype, order, &layout).unwrap_or_else(|| Err(format!("`{}`: no kernel for these operands", op.name())))?;
            for (spec, q) in l.kernel.inputs.iter_mut().zip(&quant) {
                if let Some(kind) = q {
                    spec.kind = *kind;
                }
            }
            Ok(l)
        }, &bufs, true)
    }
}

/// Gives axes exactly the names of a shape (as the interpreter does): every
/// axis when the shape has no prefix and as many, else the trailing ones.
fn retag_axes(axes: &mut [Axis], s: &Shape) {
    let rank = axes.len();
    if s.prefix.is_none() && s.axes.len() != rank {
        return;
    }
    let k = s.axes.len().min(rank);
    for (a, ta) in axes[rank - k..].iter_mut().zip(&s.axes[s.axes.len() - k..]) {
        a.name = ta.name.clone();
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
        _ => Buf::F64(data.into()),
    };
    Tensor { axes, data: buf }
}

/// Slice `i` along the leading axis of every tensor leaf; a stacked leaf of
/// another kind gives its value of layer `i`.
pub fn slice_tree(v: &Value, i: usize) -> Value {
    v.map_leaves(&mut |leaf| match leaf {
        // A stacked matrix keeps its integers: each layer's slice is still
        // quantised, which is what puts a `stack{L}` fold's blocks on them.
        Value::Real(Tensor { axes, data: Buf::Quant(q) }) if let Some(s) = q.slice_leading(i, axes[0].len) => Value::Real(Tensor { axes: axes[1..].to_vec(), data: Buf::Quant(std::sync::Arc::new(s)) }),
        Value::Real(t) if t.rank() >= 1 => Value::Real(slice_tensor(t, i)),
        Value::Idx { bound, t } if t.rank() >= 1 => Value::Idx { bound: *bound, t: slice_tensor(t, i) },
        Value::Stacked(vs) => vs[i].clone(),
        other => other.clone(),
    })
}

/// Layer `i` of a stacked tensor: a view of the stack's elements, in their
/// type, as the C program passes `w + l · chunk` (plan 0.10.0 step 6); a
/// layer of exact reals, or of integer weights `slice_leading` gives up on,
/// a copy of its values in `f64`. A `bf16` layer stays `bf16`, as the
/// recorder's view of it always was; it was widened to `f64`, exactly,
/// before step 6.
fn slice_tensor(t: &Tensor, i: usize) -> Tensor {
    let chunk = t.numel() / t.axes[0].len.max(1);
    let (lo, axes) = (i * chunk, t.axes[1..].to_vec());
    let data = match &t.data {
        Buf::F64(e) => Buf::F64(e.view(lo, chunk)),
        Buf::F32(e) => Buf::F32(e.view(lo, chunk)),
        Buf::Bf16(e) => Buf::Bf16(e.view(lo, chunk)),
        Buf::I64(e) => Buf::I64(e.view(lo, chunk)),
        other => return with_data(t, axes, other.range_f64(lo, lo + chunk)),
    };
    Tensor { axes, data }
}

/// Stack trees leaf by leaf along a new leading axis. A leaf with no axis to
/// stack along (a key, a scalar) is stacked as the sequence of its values,
/// one per part, as the interpreter's `stack_trees` does (defect 3f, 0.10.0
/// step 4: the first was kept for every layer).
pub fn stack_trees(parts: &[Value]) -> Result<Value> {
    match &parts[0] {
        Value::Real(first) => {
            let mut axes = vec![Axis { name: None, len: parts.len() }];
            axes.extend(first.axes.iter().cloned());
            let mut ts = Vec::with_capacity(parts.len());
            for p in parts {
                match p {
                    Value::Real(t) => ts.push(t),
                    other => return err("E7001", format!("stack_params: expected a real tensor, found a {}", other.kind())),
                }
            }
            Ok(Value::Real(stack_tensors(first, axes, &ts)))
        }
        Value::Idx { bound, t: first } => {
            let mut axes = vec![Axis { name: None, len: parts.len() }];
            axes.extend(first.axes.iter().cloned());
            let mut ts = Vec::with_capacity(parts.len());
            for p in parts {
                match p {
                    Value::Idx { t, .. } => ts.push(t),
                    other => return err("E7001", format!("stack: expected an index tensor, found a {}", other.kind())),
                }
            }
            Ok(Value::Idx { bound: *bound, t: stack_tensors(first, axes, &ts) })
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
        _ => Ok(Value::Stacked(parts.to_vec())),
    }
}

/// Tensors stacked along a new leading axis, `axes`. Of the stack's type,
/// `with_data`'s from the first (`f32` stays `f32`, indices stay indices,
/// anything else becomes `f64`), the parts laid end to end without a copy
/// when every one is of that type (plan 0.10.0 step 6): a scan's results,
/// which a fold then slices layer by layer; otherwise their values copied
/// into it, as before.
fn stack_tensors(first: &Tensor, axes: Vec<Axis>, parts: &[&Tensor]) -> Tensor {
    fn all<T: presk_rt::Pooled>(parts: &[&Tensor], of: impl Fn(&Buf) -> Option<&Elems<T>>) -> Option<Elems<T>> {
        parts.iter().map(|t| of(&t.data).cloned()).collect::<Option<Vec<_>>>().map(Elems::stack)
    }
    let data = match &first.data {
        Buf::F32(_) => all(parts, |b| if let Buf::F32(e) = b { Some(e) } else { None }).map(Buf::F32),
        Buf::I64(_) => all(parts, |b| if let Buf::I64(e) = b { Some(e) } else { None }).map(Buf::I64),
        Buf::F64(_) => all(parts, |b| if let Buf::F64(e) = b { Some(e) } else { None }).map(Buf::F64),
        _ => None,
    };
    match data {
        Some(data) => Tensor { axes, data },
        None => {
            let mut data = Vec::with_capacity(first.numel() * parts.len());
            for t in parts {
                data.extend(t.data.to_f64());
            }
            with_data(first, axes, data)
        }
    }
}

/// `stacked` with its slice `i` replaced by `value`, its parts in their
/// places, when the two are of one type and the slice of its length (plan
/// 0.10.0 step 6): the other slices are views of `stacked`, nothing is
/// copied.
fn replaced<T: presk_rt::Pooled>(stacked: &Elems<T>, n: usize, i: usize, value: &Elems<T>) -> Option<Elems<T>> {
    let chunk = stacked.len() / n.max(1);
    (value.len() == chunk && chunk * n == stacked.len() && i < n).then(|| Elems::stack((0..n).map(|k| if k == i { value.clone() } else { stacked.view(k * chunk, chunk) }).collect()))
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
                let n = t.axes[0].len;
                let data = match (&t.data, &v.data) {
                    (Buf::F32(s), Buf::F32(x)) => replaced(s, n, i, x).map(Buf::F32),
                    (Buf::F64(s), Buf::F64(x)) => replaced(s, n, i, x).map(Buf::F64),
                    (Buf::I64(s), Buf::I64(x)) => replaced(s, n, i, x).map(Buf::I64),
                    _ => None,
                };
                if let Some(data) = data {
                    return Value::Real(Tensor { axes: t.axes.clone(), data });
                }
                let chunk = t.numel() / n;
                let mut data = t.data.to_f64();
                let vd = v.data.to_f64();
                data[i * chunk..(i + 1) * chunk].copy_from_slice(&vd[..chunk.min(vd.len())]);
                Value::Real(with_data(t, t.axes.clone(), data))
            }
            (leaf, _) => leaf.clone(),
        }
    })
}

/// Zeros of `n` slices of `ct`'s length, `ct` in slice `index`: one slice
/// of zeros shared by the others (plan 0.10.0 step 6).
fn scattered<T: presk_rt::Pooled>(total: usize, index: usize, ct: &Elems<T>, zero: T) -> Option<Elems<T>> {
    let chunk = ct.len();
    let n = total.checked_div(chunk)?;
    (n * chunk == total && index < n).then(|| {
        let zeros = Elems::from(vec![zero; chunk]);
        Elems::stack((0..n).map(|k| if k == index { ct.clone() } else { zeros.clone() }).collect())
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
                let n = t.numel();
                let data = match (&t.data, &c.data) {
                    (Buf::F32(_), Buf::F32(x)) => scattered(n, index, x, 0.0).map(Buf::F32),
                    (Buf::F64(_), Buf::F64(x)) => scattered(n, index, x, 0.0).map(Buf::F64),
                    (Buf::I64(_), Buf::I64(x)) => scattered(n, index, x, 0).map(Buf::I64),
                    _ => None,
                };
                if let Some(data) = data {
                    return Value::Real(Tensor { axes: t.axes.clone(), data });
                }
                let mut data = vec![0.0; n];
                let cd = c.data.to_f64();
                let start = index * cd.len();
                data[start..start + cd.len()].copy_from_slice(&cd);
                Value::Real(with_data(t, t.axes.clone(), data))
            }
            (other, _) => other.clone(),
        }
    })
}
