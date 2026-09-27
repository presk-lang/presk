//! The recorder (plan 0.9.0 step 3, D79): the host program of one call of an
//! entry, as the kernel backend runs it, written down as a straight line of
//! operations on buffers, which the C export prints.
//!
//! **Why a mirror.** Values in the kernel backend carry no identity: a `let`
//! reads its operands with `env.get(v).cloned()`, a slice of a stacked tree
//! is a copy, a view (`merge_axes`, `split_axes`, `reshape_like`) clones its
//! data. So a recorder that only watched launches could not tell which
//! buffer a launch reads, least of all on zeros. The recorder therefore keeps
//! a symbolic copy of the host's environment: every value the host holds has
//! a [`Sym`] beside it, whose tensor leaves are [`View`]s of the recording's
//! buffers (an input, a launch's output, a fill, a copy), at an offset. The
//! backend tells it, through hooks that do nothing when it is off, where
//! calls start and end, where each `let` ends, how a fold iterates and what
//! each launch is; the recorder follows tuples, records, projections,
//! fields, closures and their application, the fold's slices and its scan's
//! stack, and each primitive's decomposition into launches (a parameter tree
//! by a scalar, a fusion group over a tree, a pack before its contraction),
//! in the order the host makes them.
//!
//! **What it decides: nothing.** Kernels, layouts, types and shapes are the
//! backend's: each launch is consumed from the backend's own event, and a
//! launch the mirror did not predict, or one it predicted that did not
//! happen, is an internal error, never a guess. Control flow is static (no
//! conditional in PSK-Core, a fold's count is a dimension), so one run
//! records the host for every input of these shapes.
//!
//! **The structure** (plan 0.9.0 step 4, D80). Beside the flat list of
//! operations, the recording keeps where each was made: every call of a
//! Presk function ([`Call`]: its function, dimensions, parameter names,
//! arguments and result) and every fold ([`Fold`]: its kind, its iterations
//! as calls, its scan's stack), as a tree of [`Node`]s whose leaves are the
//! operations, each in exactly one body. Walked depth first, the tree gives
//! the flat list back. Buffers carry the source name of the `let` that made
//! them. The C export's structured host is built from the tree, and its
//! straight-line host, the oracle, from the list.
//!
//! **Randomness** (plan 0.9.0 step 6, D83). `sample` takes no key: it draws
//! from the backend's own stream, seeded by the run's seed, and is a host
//! operation ([`Op::Sample`]), which the C program runs on its own stream,
//! seeded alike and advancing across calls. An `f64` argument of the entry
//! (a temperature, an `eps`) is a host scalar computed at run time
//! ([`Scalar::Var`]), not a constant: the C program takes it as an argument,
//! and the value it was recorded with is kept ([`Recording::scalar_args`]).
//!
//! **Integer weights** (plan 0.9.0 step 7, D84). A quantized leaf of the
//! entry's arguments is an input buffer of integers and scales
//! ([`Store::Quant`]), whose geometry the recording keeps ([`RecBuf::quant`]).
//! A kernel the host marks reads it as it is; any other reads its values, a
//! dequantizing [`Op::Copy`] at the kernel's edge, as the host converts it
//! (a fusion group's γ and β). A slice of it (`slice_tree`) is a view that
//! stays integers, as `QBuf::slice_leading` keeps it, or, where that gives
//! up (the scales constant along the sliced axis, an `int4` slice of odd
//! length), a copy of its values in `f64`, made while a fold feeds its
//! iteration ([`Fold::slices`]).
//!
//! **What it refuses**, with the export's codes: an initializer (`E8004`,
//! a host primitive the C runtime does not provide), a key anywhere but
//! `split_key`, a dropout mask and the structure that carries them
//! (`E8006`), integer weights reshaped by a view, drawn from, or in the
//! result (`E8007`, which refused every integer weight until step 7), and a
//! result the C program cannot write, a scalar (`E8008`). `sample` was
//! refused (`E8005`) until step 6.

use std::collections::{HashMap, VecDeque};

use presk_core::ir::{Lit, Rhs};
use presk_core::{FuncId, PrimOp, VarId};
use presk_rt::{Buf, Value};

use crate::ir::{BufKind, Dtype, Kernel};
use crate::lower::{self, Operand};

/// The machine type of a buffer of the recording: the reals of a site,
/// indices, or integer weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Store {
    F64,
    F32,
    Bf16,
    I64,
    /// Integer weights of `bits` (8 or 4) and their scales (step 7): the
    /// scales constant along an axis of length `k`, the product of the axes
    /// after it `inner`, so that element `e` has scale
    /// `(e / (k·inner))·inner + e % inner`.
    Quant { bits: u8, k: usize, inner: usize },
}

impl Store {
    pub fn of(d: Dtype) -> Store {
        match d {
            Dtype::F64 => Store::F64,
            Dtype::F32 => Store::F32,
            Dtype::Bf16 => Store::Bf16,
        }
    }

    /// Bytes of one element: for integer weights, of the byte an `int8`
    /// takes, which two `int4`s share, their scales apart.
    pub fn bytes(self) -> usize {
        match self {
            Store::F64 | Store::I64 => 8,
            Store::F32 => 4,
            Store::Bf16 => 2,
            Store::Quant { .. } => 1,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Store::F64 => "f64",
            Store::F32 => "f32",
            Store::Bf16 => "bf16",
            Store::I64 => "i64",
            Store::Quant { bits: 8, .. } => "int8",
            Store::Quant { .. } => "int4",
        }
    }

    /// The type of a buffer the host builds from another's values with
    /// `with_data` (a stack, a slice written back, a scattered slice): `f32`
    /// stays `f32`, indices stay indices, anything else becomes `f64`, as
    /// `presk-kernel`'s host does. Exact for every value the source holds.
    fn rebuilt(self) -> Store {
        match self {
            Store::F32 => Store::F32,
            Store::I64 => Store::I64,
            Store::F64 | Store::Bf16 | Store::Quant { .. } => Store::F64,
        }
    }

    /// The store of a kernel's buffer: its reals, indices, or the integer
    /// weights the host marked it to load.
    pub fn of_spec(spec: &crate::ir::BufSpec, dtype: Dtype) -> Store {
        match spec.kind {
            BufKind::Idx => Store::I64,
            BufKind::Real => Store::of(dtype),
            BufKind::Quant { bits, over } => Store::Quant { bits, k: spec.dims[over], inner: spec.dims[over + 1..].iter().product() },
        }
    }
}

/// Where the scales of an input of integer weights are constant (step 7):
/// along axis `over` of the `rank` axes the input has. A view that has lost
/// leading axes by slicing has them along `over − (rank − its rank)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantAxis {
    pub over: usize,
    pub rank: usize,
}

/// A buffer of the recording.
#[derive(Clone, Debug)]
pub struct RecBuf {
    pub store: Store,
    /// Elements.
    pub len: usize,
    pub made: Made,
    /// Where it was made: the `let`'s site, `"<function>/%<let>"`, with its
    /// source name when it has one; or the input's path.
    pub site: String,
    /// The `let` that made it (its variable's number) and its source name,
    /// when it was made by one.
    pub var: Option<u32>,
    pub name: Option<String>,
    /// For integer weights, the axis their scales are constant along.
    pub quant: Option<QuantAxis>,
}

/// What made a buffer.
#[derive(Clone, Debug, PartialEq)]
pub enum Made {
    /// Leaf `leaf` of the entry's argument `arg`, in `collect_leaves` order.
    Input { arg: usize, leaf: usize },
    /// A launch's output.
    Launch,
    /// Every element one value ([`Op::Fill`]).
    Fill,
    /// One element, a host scalar ([`Op::Scalar`]).
    Scalar,
    /// Copied from other buffers ([`Op::Copy`]): a conversion at a kernel's
    /// edge, a stack, a slice written back, a scattered slice.
    Copy,
    /// Tokens drawn by `sample` ([`Op::Sample`]).
    Sample,
}

/// A tensor as the host holds it: the elements of `shape`, row-major, of
/// buffer `buf` from element `off`.
#[derive(Clone, Debug, PartialEq)]
pub struct View {
    pub buf: usize,
    pub off: usize,
    pub shape: Vec<usize>,
    /// `Some(n)` for indices below `n`.
    pub bound: Option<u64>,
}

impl View {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A host `f64`: known when the entry is exported, or computed while it runs
/// (read back from a kernel's output), the `n`-th of the recording's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalar {
    Const(f64),
    Var(usize),
}

/// A value of the host, symbolically: [`Value`] with its tensors as views.
#[derive(Clone, Debug, PartialEq)]
pub enum Sym {
    Tensor(View),
    Nat(u64),
    Int(i64),
    F64(Scalar),
    Bool(bool),
    /// A key: read by nothing the C program runs (§3.3), dropped.
    Key,
    Tuple(Vec<Sym>),
    Record(Vec<(String, Sym)>),
}

/// One operation of the host, in the order the backend performs it.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// `kernel` (an index into the backend's kernels) on `inputs`, each
    /// already of the type the kernel reads, into the fresh buffer `out`.
    Launch { kernel: usize, inputs: Vec<View>, out: usize, site: String },
    /// Every element of `buf` is `value`, in `buf`'s type
    /// (`Buf::of(dtype, [value; n])`): `zeros_like`, a dropout mask outside
    /// training, the zeros under a scattered slice.
    Fill { buf: usize, value: f64 },
    /// `buf`, of one element, holds the host scalar `x` rounded to its type
    /// (`Buf::of(dtype, [x])`).
    Scalar { buf: usize, x: Scalar },
    /// `n` elements of `src` into `dst` from element `at`, converted to
    /// `dst`'s type through `f64`, as `Buf::of(to, b.to_f64())`.
    Copy { dst: usize, at: usize, src: View, n: usize },
    /// `out = a op b` in `f64`, the host's arithmetic on hyperparameters.
    Hyper { out: usize, op: HyperOp, a: Scalar, b: Scalar },
    /// `out` = the first element of `src`, as `f64` (`Value::scalar_f64`).
    Read { out: usize, src: View },
    /// `sample`: `rows` rows of `vocab` logits at `src`, a token each into
    /// the index buffer `out`, one draw of the backend's stream per row
    /// (plan 0.9.0 step 6).
    Sample { out: usize, src: View, rows: usize, vocab: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HyperOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl HyperOp {
    fn of(op: &PrimOp) -> HyperOp {
        match op {
            PrimOp::Add => HyperOp::Add,
            PrimOp::Sub => HyperOp::Sub,
            PrimOp::Mul => HyperOp::Mul,
            _ => HyperOp::Div,
        }
    }

    /// The host's `hyper`.
    pub fn apply(self, x: f64, y: f64) -> f64 {
        match self {
            HyperOp::Add => x + y,
            HyperOp::Sub => x - y,
            HyperOp::Mul => x * y,
            HyperOp::Div => x / y,
        }
    }

    pub fn c(self) -> &'static str {
        match self {
            HyperOp::Add => "+",
            HyperOp::Sub => "-",
            HyperOp::Mul => "*",
            HyperOp::Div => "/",
        }
    }
}

/// A place in the structure: an operation (an index into
/// [`Recording::ops`]), a call ([`Recording::calls`]) or a fold
/// ([`Recording::folds`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Node {
    Op(usize),
    Call(usize),
    Fold(usize),
}

/// One call of a Presk function (plan 0.9.0 step 4): a `Call`, the
/// application of a closure, or an iteration of a fold.
#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    /// The PSK-Core function, `"Linear.apply"`.
    pub func: String,
    /// Its dimensions at this call, in the order it declares them.
    pub dims: Vec<(String, u64)>,
    /// Its parameters' source names.
    pub params: Vec<String>,
    /// The arguments it received (a closure's captured values first).
    pub args: Vec<Sym>,
    pub result: Option<Sym>,
    /// What it did, in order.
    pub body: Vec<Node>,
    /// The `let` of the caller that made the call (a fold's, for an
    /// iteration): its variable's number and source name. `None` for the
    /// entry.
    pub var: Option<u32>,
    pub name: Option<String>,
    /// Entered through the application of a closure.
    pub apply: bool,
}

/// One fold (plan 0.9.0 step 4): `n` calls of its body over the slices of
/// `stacked`, the accumulator threaded from `init`.
#[derive(Clone, Debug, PartialEq)]
pub struct Fold {
    /// The `let` that folds: its site, variable's number and source name.
    pub site: String,
    pub var: Option<u32>,
    pub name: Option<String>,
    pub n: usize,
    /// The body takes the iteration's index first.
    pub index: bool,
    /// From `n − 1` down to 0.
    pub reverse: bool,
    /// The value is the pair of the accumulator and the stack of the
    /// accumulators before each iteration.
    pub scan: bool,
    pub stacked: Vec<Sym>,
    pub extra: Vec<Sym>,
    pub init: Sym,
    /// The iterations in the order they ran: the index, and the call.
    pub iterations: Vec<(usize, usize)>,
    /// The fold's value.
    pub result: Option<Sym>,
    /// With `scan`, the operations that stack the accumulators, made once
    /// the last iteration has returned.
    pub stack: Vec<usize>,
    /// The copies that slicing makes as it feeds an iteration (step 7): a
    /// slice of integer weights that does not stay integers is a copy of its
    /// values in `f64`. `(iteration index, operation)`.
    pub slices: Vec<(usize, usize)>,
}

/// The host of one call of the entry.
#[derive(Clone, Debug, Default)]
pub struct Recording {
    pub bufs: Vec<RecBuf>,
    pub ops: Vec<Op>,
    /// The entry's arguments, as the call received them (realized).
    pub args: Vec<Sym>,
    /// Its result.
    pub result: Option<Sym>,
    /// Host scalars computed at run time ([`Scalar::Var`]).
    pub vars: usize,
    /// The entry's `f64` arguments, in the order of their leaves: each is a
    /// host scalar computed at run time, and the value it was recorded with
    /// (step 6).
    pub scalar_args: Vec<(usize, f64)>,
    /// The structure (step 4): `calls[0]` is the entry's call, and the
    /// tree below it holds every operation of `ops` once.
    pub calls: Vec<Call>,
    pub folds: Vec<Fold>,
}

/// Why an entry cannot be exported: a code of the export and a message in
/// the source's terms.
#[derive(Clone, Debug, PartialEq)]
pub struct Refusal {
    pub code: &'static str,
    pub message: String,
}

/// A launch as the backend performs it.
#[derive(Clone, Debug)]
struct Event {
    kernel: usize,
    /// The copy of a pack's operand to its panel (0.7.0 step 3).
    pack: bool,
    /// The type of each input the kernel reads.
    inputs: Vec<Store>,
    out: Store,
    out_shape: Vec<usize>,
    out_bound: Option<u64>,
}

#[derive(Default)]
struct Frame {
    env: HashMap<VarId, Sym>,
    closures: HashMap<VarId, (FuncId, Vec<Sym>)>,
}

struct OpenFold {
    stacked: Vec<Sym>,
    extra: Vec<Sym>,
    acc: Sym,
    index: bool,
    scan: bool,
    pre: Vec<(usize, Sym)>,
    /// Its place in [`Recording::folds`].
    id: usize,
}

/// A call or a fold being recorded.
#[derive(Clone, Copy)]
enum Open {
    Call(usize),
    Fold(usize),
}

/// Records the host (see the module's documentation). Set it on
/// [`crate::KernelBackend::recorder`] just before the entry's call, so that
/// nothing else the backend runs (an initializer) is recorded.
#[derive(Default)]
pub struct Recorder {
    rec: Recording,
    frames: Vec<Frame>,
    /// The arguments of the next function entered.
    pending: Option<Vec<Sym>>,
    /// Results of calls and folds, for the `let` that made them.
    results: Vec<Sym>,
    folds: Vec<OpenFold>,
    /// Launches since the last `let` ended.
    events: VecDeque<Event>,
    /// The site of the `let` being evaluated, for the buffers it makes, and
    /// its variable's number and source name.
    site: String,
    var: Option<u32>,
    name: Option<String>,
    /// The calls and folds being recorded, innermost last.
    open: Vec<Open>,
    /// The next call entered is an iteration of the innermost fold, of this
    /// index; or the application of a closure.
    next_iter: Option<usize>,
    next_apply: bool,
    /// The `let` whose right-hand side is being evaluated, as `let_begin`
    /// gave it: `(site, variable, name)`.
    let_info: (String, Option<u32>, Option<String>),
    /// The fold whose scan's stack is being made.
    stacking: Option<usize>,
    /// The fold, and its iteration, whose stacked values are being sliced.
    slicing: Option<(usize, usize)>,
    /// The entry's parameters' names, for the refusals.
    names: Vec<String>,
    refused: Option<Refusal>,
}

type R<T> = std::result::Result<T, Refusal>;

fn refuse<T>(code: &'static str, message: impl Into<String>) -> R<T> {
    Err(Refusal { code, message: message.into() })
}

/// The code of a fault of the recorder, not of the program: its mirror and
/// the backend disagree.
pub const INTERNAL: &str = "internal";

fn internal<T>(message: impl Into<String>) -> R<T> {
    refuse(INTERNAL, message)
}

fn collect(s: &Sym, out: &mut Vec<Sym>) {
    match s {
        Sym::Tuple(xs) => xs.iter().for_each(|x| collect(x, out)),
        Sym::Record(fs) => fs.iter().for_each(|(_, x)| collect(x, out)),
        leaf => out.push(leaf.clone()),
    }
}

/// The leaves of a value, in the host's `collect_leaves` order.
pub fn leaves(s: &Sym) -> Vec<Sym> {
    let mut out = Vec::new();
    collect(s, &mut out);
    out
}

fn rebuild(template: &Sym, leaves: &mut impl Iterator<Item = Sym>) -> Sym {
    match template {
        Sym::Tuple(xs) => Sym::Tuple(xs.iter().map(|x| rebuild(x, leaves)).collect()),
        Sym::Record(fs) => Sym::Record(fs.iter().map(|(n, x)| (n.clone(), rebuild(x, leaves))).collect()),
        _ => leaves.next().unwrap_or_else(|| template.clone()),
    }
}

fn map_leaves(s: &Sym, f: &mut dyn FnMut(&Sym) -> R<Sym>) -> R<Sym> {
    Ok(match s {
        Sym::Tuple(xs) => Sym::Tuple(xs.iter().map(|x| map_leaves(x, f)).collect::<R<_>>()?),
        Sym::Record(fs) => Sym::Record(fs.iter().map(|(n, x)| Ok((n.clone(), map_leaves(x, f)?))).collect::<R<_>>()?),
        leaf => f(leaf)?,
    })
}

fn is_tree(s: &Sym) -> bool {
    matches!(s, Sym::Tuple(_) | Sym::Record(_))
}

/// A real tensor of rank zero (`Value::Real(t) if t.rank() == 0`).
fn is_real_scalar(s: &Sym) -> bool {
    matches!(s, Sym::Tensor(v) if v.bound.is_none() && v.shape.is_empty())
}

fn kind(s: &Sym) -> &'static str {
    match s {
        Sym::Tensor(v) if v.bound.is_some() => "index tensor",
        Sym::Tensor(_) => "tensor",
        Sym::Nat(_) => "nat",
        Sym::Int(_) => "int",
        Sym::F64(_) => "f64",
        Sym::Bool(_) => "bool",
        Sym::Key => "key",
        Sym::Tuple(_) => "tuple",
        Sym::Record(_) => "record",
    }
}

/// `E8006`: a key reaching what the C program would have to compute with it.
fn key_refusal<T>(what: &str) -> R<T> {
    refuse("E8006", format!("a key reaches {what}: the C program holds no key, so a key may only be split (`split_key`) or given to a dropout mask outside training, which reads none (`sample` draws from the run's stream, not from a key)"))
}

impl Recorder {
    pub fn new() -> Self {
        Self::default()
    }

    /// A recorder whose refusals name the entry's arguments.
    pub fn named(names: Vec<String>) -> Self {
        Recorder { names, ..Self::default() }
    }

    fn arg_name(&self, arg: usize) -> String {
        match self.names.get(arg) {
            Some(n) => format!("`{n}`"),
            None => format!("argument {}", arg + 1),
        }
    }

    /// The recording, or why the entry cannot be exported.
    pub fn finish(self) -> std::result::Result<Recording, Refusal> {
        if let Some(r) = self.refused {
            return Err(r);
        }
        let Some(result) = &self.rec.result else { return internal("the entry's call did not return") };
        for leaf in leaves(result) {
            match leaf {
                Sym::Tensor(v) if matches!(self.rec.bufs[v.buf].store, Store::Quant { .. }) => {
                    return refuse("E8007", "the entry's result holds integer weights, its argument's as they were quantized: the C program writes the reals it computes, and the weights are read, not computed");
                }
                Sym::Tensor(_) => {}
                Sym::Key => return key_refusal("the entry's result"),
                other => return refuse("E8008", format!("the entry's result holds a {}, which a `.psk.d` file cannot hold: the C program writes tensors only", kind(&other))),
            }
        }
        Ok(self.rec)
    }

    /// Runs `f` unless the recording was refused already; a refusal stops it.
    fn guard(&mut self, f: impl FnOnce(&mut Self) -> R<()>) {
        if self.refused.is_none()
            && let Err(e) = f(self)
        {
            self.refused = Some(e);
        }
    }

    fn buf(&mut self, store: Store, len: usize, made: Made) -> usize {
        self.rec.bufs.push(RecBuf { store, len, made, site: self.site.clone(), var: self.var, name: self.name.clone(), quant: None });
        self.rec.bufs.len() - 1
    }

    /// Appends an operation, to the flat list and to the body of the call
    /// being recorded (or to the stack of the fold being ended, or to the
    /// slices of the one feeding an iteration).
    fn push(&mut self, op: Op) {
        let i = self.rec.ops.len();
        self.rec.ops.push(op);
        if let Some(f) = self.stacking {
            self.rec.folds[f].stack.push(i);
        } else if let Some((f, it)) = self.slicing {
            self.rec.folds[f].slices.push((it, i));
        } else if let Some(Open::Call(c)) = self.open.last() {
            self.rec.calls[*c].body.push(Node::Op(i));
        } else {
            // Checked by the export: every operation is in the tree.
        }
    }

    fn var(&mut self) -> usize {
        self.rec.vars += 1;
        self.rec.vars - 1
    }

    fn frame(&mut self) -> R<&mut Frame> {
        match self.frames.last_mut() {
            Some(f) => Ok(f),
            None => internal("no function is being evaluated"),
        }
    }

    fn get(&self, v: &VarId) -> R<Sym> {
        match self.frames.last().and_then(|f| f.env.get(v)) {
            Some(s) => Ok(s.clone()),
            None => internal(format!("variable %{} has no value", v.0)),
        }
    }

    fn gets(&self, vs: &[VarId]) -> R<Vec<Sym>> {
        vs.iter().map(|v| self.get(v)).collect()
    }

    // ----- hooks ------------------------------------------------------------

    /// The entry's arguments, realized (`Backend::call`): each tensor leaf an
    /// input buffer, in the order of the arguments' leaves.
    pub fn entry(&mut self, args: &[Value]) {
        self.guard(|r| {
            let mut syms = Vec::with_capacity(args.len());
            for (i, a) in args.iter().enumerate() {
                let mut leaf = 0;
                syms.push(r.input(i, a, &mut leaf)?);
            }
            r.rec.args = syms.clone();
            r.pending = Some(syms);
            Ok(())
        });
    }

    fn input(&mut self, arg: usize, v: &Value, leaf: &mut usize) -> R<Sym> {
        Ok(match v {
            Value::Real(t) => {
                let store = match &t.data {
                    Buf::F64(_) => Store::F64,
                    Buf::F32(_) => Store::F32,
                    Buf::Bf16(_) => Store::Bf16,
                    Buf::Quant(q) => Store::Quant { bits: q.bits, k: q.k, inner: q.inner },
                    other => return internal(format!("an input of type {}", other.dtype())),
                };
                let b = self.input_buf(store, t.numel(), arg, leaf);
                if let Buf::Quant(q) = &t.data {
                    self.rec.bufs[b].quant = Some(QuantAxis { over: q.over, rank: t.rank() });
                }
                Sym::Tensor(View { buf: b, off: 0, shape: t.axes.iter().map(|a| a.len).collect(), bound: None })
            }
            Value::Idx { bound, t } => {
                let b = self.input_buf(Store::I64, t.numel(), arg, leaf);
                Sym::Tensor(View { buf: b, off: 0, shape: t.axes.iter().map(|a| a.len).collect(), bound: Some(*bound) })
            }
            Value::Nat(n) => Sym::Nat(*n),
            Value::Int(n) => Sym::Int(*n),
            // An argument of the C function, not a constant of it (step 6).
            Value::F64(x) => {
                let v = self.var();
                self.rec.scalar_args.push((v, *x));
                Sym::F64(Scalar::Var(v))
            }
            Value::Bool(b) => Sym::Bool(*b),
            Value::Key(_) => Sym::Key,
            Value::Tuple(vs) => Sym::Tuple(vs.iter().map(|x| self.input(arg, x, leaf)).collect::<R<_>>()?),
            Value::Record(fs) => Sym::Record(fs.iter().map(|(n, x)| Ok((n.clone(), self.input(arg, x, leaf)?))).collect::<R<_>>()?),
        })
    }

    fn input_buf(&mut self, store: Store, len: usize, arg: usize, leaf: &mut usize) -> usize {
        self.site = format!("argument {}", self.arg_name(arg));
        (self.var, self.name) = (None, None);
        let b = self.buf(store, len, Made::Input { arg, leaf: *leaf });
        *leaf += 1;
        b
    }

    /// The function `func` is entered at `dims` with `params`, whose source
    /// names are `names`: they take the pending arguments. The call joins the
    /// structure: the entry's, a callee in the caller's body, or an iteration
    /// of the innermost fold.
    pub fn enter(&mut self, func: &str, dims: Vec<(String, u64)>, params: &[VarId], names: Vec<String>) {
        self.guard(|r| {
            let Some(args) = r.pending.take() else { return internal("a function entered without arguments") };
            if args.len() != params.len() {
                return internal(format!("{} arguments for {} parameters", args.len(), params.len()));
            }
            let id = r.rec.calls.len();
            // An iteration is made by its fold's `let`, a callee by the `let`
            // being evaluated.
            let (var, name) = match r.open.last() {
                None => (None, None),
                Some(Open::Fold(f)) => (r.rec.folds[*f].var, r.rec.folds[*f].name.clone()),
                Some(Open::Call(_)) => (r.let_info.1, r.let_info.2.clone()),
            };
            let apply = std::mem::take(&mut r.next_apply);
            r.rec.calls.push(Call { func: func.to_string(), dims, params: names, args: args.clone(), result: None, body: Vec::new(), var, name, apply });
            match (r.open.last().copied(), r.next_iter.take()) {
                (None, None) => {}
                (Some(Open::Call(c)), None) => r.rec.calls[c].body.push(Node::Call(id)),
                (Some(Open::Fold(f)), Some(i)) => r.rec.folds[f].iterations.push((i, id)),
                _ => return internal("a call where the structure expects none"),
            }
            r.open.push(Open::Call(id));
            r.frames.push(Frame { env: params.iter().copied().zip(args).collect(), closures: HashMap::new() });
            Ok(())
        });
    }

    /// The function returns the value of `result`.
    pub fn leave(&mut self, result: VarId) {
        self.guard(|r| {
            let v = r.get(&result)?;
            r.frames.pop();
            match r.open.pop() {
                Some(Open::Call(c)) => r.rec.calls[c].result = Some(v.clone()),
                _ => return internal("a function left outside its call"),
            }
            if r.frames.is_empty() {
                r.rec.result = Some(v);
            } else {
                r.results.push(v);
            }
            Ok(())
        });
    }

    /// The `let` of `var` at `site`, source name `name`, starts: a call or
    /// a fold it makes is recorded under it.
    pub fn let_begin(&mut self, var: VarId, site: &str, name: Option<&str>) {
        self.let_info = (site.to_string(), Some(var.0), name.map(String::from));
    }

    /// A `Call`'s arguments.
    pub fn call(&mut self, args: &[VarId]) {
        self.guard(|r| {
            r.pending = Some(r.gets(args)?);
            Ok(())
        });
    }

    /// An `Apply`'s arguments: the closure's captured values, then `args`.
    pub fn apply(&mut self, fun: VarId, args: &[VarId]) {
        self.guard(|r| {
            let Some((_, captured)) = r.frames.last().and_then(|f| f.closures.get(&fun)).cloned() else { return internal("an application of an unknown closure") };
            let mut all = captured;
            all.extend(r.gets(args)?);
            r.pending = Some(all);
            r.next_apply = true;
            Ok(())
        });
    }

    /// `let var = closure f(captured…)`: bound to `Nat(0)`, as the host binds it.
    pub fn closure(&mut self, var: VarId, f: FuncId, captured: &[VarId]) {
        self.guard(|r| {
            let c = r.gets(captured)?;
            let frame = r.frame()?;
            frame.closures.insert(var, (f, c));
            frame.env.insert(var, Sym::Nat(0));
            Ok(())
        });
    }

    /// A fold of `n` iterations starts: its stacked values, extras and
    /// initial accumulator. It joins the structure in the caller's body.
    #[allow(clippy::too_many_arguments)]
    pub fn fold_begin(&mut self, stacked: &[VarId], extra: &[VarId], init: VarId, index: bool, reverse: bool, scan: bool, n: usize) {
        self.guard(|r| {
            let Some(Open::Call(c)) = r.open.last().copied() else { return internal("a fold outside a call") };
            let id = r.rec.folds.len();
            let (site, var, name) = r.let_info.clone();
            let (stacked, extra, acc) = (r.gets(stacked)?, r.gets(extra)?, r.get(&init)?);
            r.rec.folds.push(Fold { site, var, name, n, index, reverse, scan, stacked: stacked.clone(), extra: extra.clone(), init: acc.clone(), iterations: Vec::new(), result: None, stack: Vec::new(), slices: Vec::new() });
            r.rec.calls[c].body.push(Node::Fold(id));
            r.open.push(Open::Fold(id));
            r.folds.push(OpenFold { stacked, extra, acc, index, scan, pre: Vec::new(), id });
            Ok(())
        });
    }

    /// Iteration `i`: the body's arguments, `[i,] slice_i(stacked…), acc, extra…`.
    pub fn fold_iter(&mut self, i: usize) {
        self.guard(|r| {
            r.next_iter = Some(i);
            let Some(f) = r.folds.last_mut() else { return internal("an iteration outside a fold") };
            if f.scan {
                f.pre.push((i, f.acc.clone()));
            }
            let mut args = Vec::new();
            if f.index {
                args.push(Sym::Nat(i as u64));
            }
            let (stacked, acc, extra, id) = (f.stacked.clone(), f.acc.clone(), f.extra.clone(), f.id);
            // A slice copied is made by the fold's `let`, for the iteration.
            let (site, var, name) = { let x = &r.rec.folds[id]; (x.site.clone(), x.var, x.name.clone()) };
            r.site = match &name {
                Some(n) => format!("{site} ({n})"),
                None => site,
            };
            (r.var, r.name) = (var, name);
            r.slicing = Some((id, i));
            let sliced: R<Vec<Sym>> = stacked.iter().map(|s| r.slice(s, i)).collect();
            r.slicing = None;
            args.extend(sliced?);
            args.push(acc);
            args.extend(extra);
            r.pending = Some(args);
            Ok(())
        });
    }

    /// Iteration done: its result is the accumulator.
    pub fn fold_iter_end(&mut self) {
        self.guard(|r| {
            let Some(acc) = r.results.pop() else { return internal("an iteration returned nothing") };
            match r.folds.last_mut() {
                Some(f) => f.acc = acc,
                None => return internal("an iteration outside a fold"),
            }
            Ok(())
        });
    }

    /// The fold's value: the accumulator, or with `scan` the pair of it and
    /// the stack of the accumulators before each iteration.
    pub fn fold_end(&mut self) {
        self.guard(|r| {
            let Some(mut f) = r.folds.pop() else { return internal("the end of no fold") };
            if !matches!(r.open.pop(), Some(Open::Fold(id)) if id == f.id) {
                return internal("a fold ended outside itself");
            }
            let v = if f.scan {
                // The stack is the caller's, made under the fold's `let`.
                let (site, var, name) = { let x = &r.rec.folds[f.id]; (x.site.clone(), x.var, x.name.clone()) };
                r.site = match &name {
                    Some(n) => format!("{site} ({n})"),
                    None => site,
                };
                (r.var, r.name) = (var, name);
                f.pre.sort_by_key(|(i, _)| *i);
                let parts: Vec<Sym> = f.pre.into_iter().map(|(_, s)| s).collect();
                r.stacking = Some(f.id);
                let stacked = r.stack(&parts);
                r.stacking = None;
                Sym::Tuple(vec![f.acc, stacked?])
            } else {
                f.acc
            };
            r.rec.folds[f.id].result = Some(v.clone());
            r.results.push(v);
            Ok(())
        });
    }

    /// A kernel launch, as the backend performs it (`launch`).
    pub fn launch(&mut self, kernel: usize, k: &Kernel, pack: bool, out_shape: &[usize], out_bound: Option<u64>) {
        self.guard(|r| {
            let inputs = k.inputs.iter().map(|s| Store::of_spec(s, k.dtype)).collect();
            let out = Store::of_spec(&k.output, k.dtype);
            r.events.push_back(Event { kernel, pack, inputs, out, out_shape: out_shape.to_vec(), out_bound });
            Ok(())
        });
    }

    /// The `let` of `var` at `site` evaluated to `value` (`dtype`: its site's
    /// type): its symbolic value, from the launches it made.
    #[allow(clippy::too_many_arguments)]
    pub fn let_end(&mut self, var: VarId, rhs: &Rhs, value: &Value, dtype: Dtype, site: &str, name: Option<&str>) {
        self.guard(|r| {
            r.site = match name {
                Some(n) => format!("{site} ({n})"),
                None => site.to_string(),
            };
            (r.var, r.name) = (Some(var.0), name.map(String::from));
            let s = match rhs {
                Rhs::Lit(Lit::Nat(n)) => Sym::Nat(*n),
                Rhs::Lit(Lit::Float(_)) => match value {
                    Value::F64(x) => Sym::F64(Scalar::Const(*x)),
                    other => return internal(format!("a float literal evaluated to a {}", other.kind())),
                },
                Rhs::Prim { op, args } => {
                    let args = r.gets(args)?;
                    r.prim(op, &args, Some(value), dtype)?
                }
                Rhs::Call { .. } | Rhs::Apply { .. } | Rhs::Fold { .. } => match r.results.pop() {
                    Some(s) => s,
                    None => return internal("a call returned nothing"),
                },
                Rhs::Tuple(vs) => Sym::Tuple(r.gets(vs)?),
                Rhs::Proj(v, i) => match r.get(v)? {
                    Sym::Tuple(xs) => match xs.into_iter().nth(*i) {
                        Some(x) => x,
                        None => return internal(format!("a tuple has no element {i}")),
                    },
                    other => return internal(format!("a projection of a {}", kind(&other))),
                },
                Rhs::Record(fs) => Sym::Record(fs.iter().map(|(n, v)| Ok((n.clone(), r.get(v)?))).collect::<R<_>>()?),
                Rhs::Field(v, name) => match r.get(v)? {
                    Sym::Record(fs) => match fs.into_iter().find(|(n, _)| n == name) {
                        Some((_, x)) => x,
                        None => return internal(format!("a record has no field `{name}`")),
                    },
                    other => return internal(format!("field `{name}` of a {}", kind(&other))),
                },
                Rhs::Closure { .. } => Sym::Nat(0),
                Rhs::Grad(_) => return internal("`grad` reached the backend"),
            };
            r.bind(var, s)
        });
    }

    /// A fusion group's root `let`, evaluated by one launch per leaf.
    pub fn fused(&mut self, var: VarId, inputs: &[VarId], dtype: Dtype, site: &str, name: Option<&str>) {
        self.guard(|r| {
            r.site = format!("{site} (fused)");
            (r.var, r.name) = (Some(var.0), name.map(String::from));
            let vals = r.gets(inputs)?;
            let s = r.fused_on(&vals, dtype)?;
            r.bind(var, s)
        });
    }

    fn bind(&mut self, var: VarId, s: Sym) -> R<()> {
        if let Some(e) = self.events.front() {
            return internal(format!("{} launch(es) of this `let` not accounted for, the first of kernel {}", self.events.len(), e.kernel));
        }
        self.frame()?.env.insert(var, s);
        Ok(())
    }

    // ----- the host's primitives, symbolically ----------------------------------

    /// The next launch, whose inputs are `inputs`, and its output.
    fn consume(&mut self, inputs: &[View], pack: bool) -> R<View> {
        let Some(e) = self.events.pop_front() else { return internal("a launch the backend did not make") };
        if e.pack != pack || e.inputs.len() != inputs.len() {
            return internal(format!("kernel {} takes {} input(s) (pack: {}), the mirror has {} (pack: {pack})", e.kernel, e.inputs.len(), e.pack, inputs.len()));
        }
        let mut ins = Vec::with_capacity(inputs.len());
        for (v, want) in inputs.iter().zip(&e.inputs) {
            let have = self.rec.bufs[v.buf].store;
            if (have == Store::I64) != (*want == Store::I64) || (matches!(want, Store::Quant { .. }) && have != *want) {
                return internal(format!("kernel {} reads {} where the mirror has {}", e.kernel, want.name(), have.name()));
            }
            if have == *want {
                // Integer weights start on a channel, and `int4` ones on a
                // byte: the pointers a kernel gets.
                if let Store::Quant { bits, k, inner } = have
                    && (v.off % (k * inner).max(1) != 0 || (bits == 4 && v.off % 2 != 0))
                {
                    return internal(format!("kernel {} reads integer weights from element {}, inside a channel or a byte", e.kernel, v.off));
                }
                ins.push(v.clone());
            } else {
                // Converted at the kernel's edge, as the backend converts it.
                let n = v.numel();
                let b = self.buf(*want, n, Made::Copy);
                self.push(Op::Copy { dst: b, at: 0, src: v.clone(), n });
                ins.push(View { buf: b, off: 0, shape: v.shape.clone(), bound: v.bound });
            }
        }
        let n: usize = e.out_shape.iter().product();
        let out = self.buf(e.out, n, Made::Launch);
        self.push(Op::Launch { kernel: e.kernel, inputs: ins, out, site: self.site.clone() });
        Ok(View { buf: out, off: 0, shape: e.out_shape, bound: e.out_bound })
    }

    /// `Buf::of(dtype, [x])`: a host scalar as a buffer of one element.
    fn scalar_buf(&mut self, x: Scalar, dtype: Dtype) -> View {
        let b = self.buf(Store::of(dtype), 1, Made::Scalar);
        self.push(Op::Scalar { buf: b, x });
        View { buf: b, off: 0, shape: vec![], bound: None }
    }

    /// The host's `as_tensor`.
    fn as_tensor(&mut self, s: &Sym, dtype: Dtype) -> R<View> {
        match s {
            Sym::Tensor(v) if v.bound.is_none() => Ok(v.clone()),
            Sym::Nat(n) => Ok(self.scalar_buf(Scalar::Const(*n as f64), dtype)),
            Sym::Int(n) => Ok(self.scalar_buf(Scalar::Const(*n as f64), dtype)),
            Sym::F64(x) => Ok(self.scalar_buf(*x, dtype)),
            Sym::Key => key_refusal("a kernel"),
            other => internal(format!("a {} where a real tensor is expected", kind(other))),
        }
    }

    /// The host's `tensor_prim_kinds`: one launch, or a pack and its
    /// contraction.
    fn tensor_prim(&mut self, ins: &[View]) -> R<Sym> {
        let pack = self.events.front().is_some_and(|e| e.pack);
        let out = if pack {
            if ins.len() != 2 {
                return internal("a pack of an operation of other than two operands");
            }
            let panel = self.consume(&ins[1..], true)?;
            self.consume(&[ins[0].clone(), panel], false)?
        } else {
            self.consume(ins, false)?
        };
        Ok(Sym::Tensor(out))
    }

    fn read(&mut self, v: View) -> Scalar {
        let out = self.var();
        self.push(Op::Read { out, src: v });
        Scalar::Var(out)
    }

    fn hyper(&mut self, op: HyperOp, a: Scalar, b: Scalar) -> Scalar {
        match (a, b) {
            (Scalar::Const(x), Scalar::Const(y)) => Scalar::Const(op.apply(x, y)),
            _ => {
                let out = self.var();
                self.push(Op::Hyper { out, op, a, b });
                Scalar::Var(out)
            }
        }
    }

    /// A buffer of `n` elements of `store`, filled with `value`.
    fn fill(&mut self, store: Store, n: usize, value: f64) -> usize {
        let b = self.buf(store, n, Made::Fill);
        self.push(Op::Fill { buf: b, value });
        b
    }

    /// The host's `prim`, its decomposition into launches included. `top`:
    /// the `let`'s value, for the shapes of views, when this is the `let`'s
    /// own primitive and not one leaf of it.
    fn prim(&mut self, op: &PrimOp, args: &[Sym], top: Option<&Value>, dtype: Dtype) -> R<Sym> {
        let a = |i: usize| -> R<&Sym> {
            match args.get(i) {
                Some(s) => Ok(s),
                None => internal(format!("`{}`: missing argument {i}", op.name())),
            }
        };
        let top_shape = || -> R<Vec<usize>> {
            match top {
                Some(Value::Real(t)) => Ok(t.axes.iter().map(|x| x.len).collect()),
                _ => internal(format!("`{}` evaluated to no tensor", op.name())),
            }
        };
        match op {
            PrimOp::ZerosLike => map_leaves(a(0)?, &mut |leaf| match leaf {
                Sym::Tensor(v) if v.bound.is_none() => {
                    let b = self.fill(Store::of(dtype), v.numel(), 0.0);
                    Ok(Sym::Tensor(View { buf: b, off: 0, shape: v.shape.clone(), bound: None }))
                }
                other => Ok(other.clone()),
            }),
            // A view: the same elements under the `let`'s shape. The template
            // of `reshape_like` gives a shape and no data.
            PrimOp::MergeAxes { .. } | PrimOp::SplitAxes { .. } | PrimOp::ReshapeLike => {
                let v = self.as_tensor(a(0)?, dtype)?;
                if matches!(self.rec.bufs[v.buf].store, Store::Quant { .. }) {
                    // The host keeps the integers and their scales' axis
                    // under the new shape, which need not be the axis the
                    // scales are constant along any more.
                    return refuse("E8007", format!("`{}` reshapes integer weights, whose scales are constant along an axis of their shape: the C export follows a slice of integer weights, not a reshape", op.name()));
                }
                Ok(Sym::Tensor(View { shape: top_shape()?, ..v }))
            }
            PrimOp::SetField { name } => match a(0)? {
                Sym::Record(fs) => Ok(Sym::Record(fs.iter().map(|(n, v)| Ok((n.clone(), if n == name { a(1)?.clone() } else { v.clone() }))).collect::<R<_>>()?)),
                other => internal(format!("set_field on a {}", kind(other))),
            },
            PrimOp::SetElem { index } => match a(0)? {
                Sym::Tuple(xs) => Ok(Sym::Tuple(xs.iter().enumerate().map(|(i, v)| Ok(if i == *index { a(1)?.clone() } else { v.clone() })).collect::<R<_>>()?)),
                other => internal(format!("set_elem on a {}", kind(other))),
            },
            PrimOp::SliceParams { index } => self.slice(a(0)?, *index as usize),
            PrimOp::StackParams { .. } => self.stack(args),
            PrimOp::SetSlice => {
                let Sym::Nat(i) = a(1)? else { return internal("set_slice at a non-static index") };
                self.set_slice(a(0)?, *i as usize, a(2)?)
            }
            PrimOp::ScatterParams { index } => self.scatter(a(0)?, *index as usize, a(1)?),
            PrimOp::SumSquares => {
                let mut acc = Scalar::Const(0.0);
                for leaf in leaves(a(0)?) {
                    if let Sym::Tensor(v) = leaf
                        && v.bound.is_none()
                    {
                        let Sym::Tensor(s) = self.tensor_prim(&[v])? else { unreachable!() };
                        let x = self.read(s);
                        acc = self.hyper(HyperOp::Add, acc, x);
                    }
                }
                Ok(Sym::Tensor(self.scalar_buf(acc, dtype)))
            }
            PrimOp::Pi => Ok(Sym::Tensor(self.scalar_buf(Scalar::Const(std::f64::consts::PI), dtype))),
            PrimOp::Real => match a(0)? {
                Sym::Nat(n) => Ok(Sym::Tensor(self.scalar_buf(Scalar::Const(*n as f64), dtype))),
                Sym::Int(n) => Ok(Sym::Tensor(self.scalar_buf(Scalar::Const(*n as f64), dtype))),
                Sym::F64(x) => Ok(Sym::Tensor(self.scalar_buf(*x, dtype))),
                Sym::Tensor(v) if v.bound.is_none() => Ok(Sym::Tensor(v.clone())),
                Sym::Tensor(v) => self.tensor_prim(std::slice::from_ref(v)),
                Sym::Key => key_refusal("`real`"),
                other => internal(format!("real of a {}", kind(other))),
            },
            PrimOp::Zeros { .. } | PrimOp::Ones { .. } | PrimOp::Positions { .. } | PrimOp::CausalMask { .. } => self.tensor_prim(&[]),
            PrimOp::DropoutMask => {
                if !matches!(a(1)?, Sym::Key) {
                    return internal("a dropout mask without a key");
                }
                let v = self.as_tensor(a(2)?, dtype)?;
                // Outside training a mask is ones, whatever its key: the C
                // program is never a training run (§3.3).
                let b = self.fill(Store::of(dtype), v.numel(), 1.0);
                Ok(Sym::Tensor(View { buf: b, off: 0, shape: v.shape, bound: None }))
            }
            PrimOp::Normal { .. } | PrimOp::XavierUniform { .. } => refuse("E8004", format!("the entry calls the initializer `{}`, which the C runtime does not provide: an exported model reads its parameters", op.name())),
            PrimOp::SplitKey { n } => match a(0)? {
                Sym::Key => Ok(Sym::Tuple(vec![Sym::Key; *n])),
                other => internal(format!("split_key of a {}", kind(other))),
            },
            // A host operation on the backend's stream (step 6): the rows of
            // the last axis, a token each.
            PrimOp::Sample => match a(0)? {
                Sym::Tensor(v) if matches!(self.rec.bufs[v.buf].store, Store::Quant { .. }) => refuse("E8007", "`sample` draws from integer weights: the C program samples from the logits it computes"),
                Sym::Tensor(v) if v.bound.is_none() && !v.shape.is_empty() => {
                    let vocab = v.shape[v.shape.len() - 1];
                    let rows = v.numel() / vocab.max(1);
                    let out = self.buf(Store::I64, rows, Made::Sample);
                    self.push(Op::Sample { out, src: v.clone(), rows, vocab });
                    Ok(Sym::Tensor(View { buf: out, off: 0, shape: v.shape[..v.shape.len() - 1].to_vec(), bound: Some(vocab as u64) }))
                }
                other => internal(format!("sample of a {}", kind(other))),
            },
            PrimOp::OneHot => match a(0)? {
                Sym::Tensor(v) if v.bound.is_some() => self.tensor_prim(std::slice::from_ref(v)),
                other => internal(format!("one_hot of a {}", kind(other))),
            },
            PrimOp::Add | PrimOp::Sub | PrimOp::Mul | PrimOp::Div if args.len() == 2 => {
                let (x, y) = (a(0)?, a(1)?);
                match (x, y) {
                    (Sym::Nat(p), Sym::Nat(q)) => {
                        let r = HyperOp::of(op).apply(*p as f64, *q as f64);
                        Ok(if r >= 0.0 && r.fract() == 0.0 { Sym::Nat(r as u64) } else { Sym::Int(r as i64) })
                    }
                    (Sym::F64(p), Sym::F64(q)) => Ok(Sym::F64(self.hyper(HyperOp::of(op), *p, *q))),
                    (Sym::Tuple(_) | Sym::Record(_), y) if is_real_scalar(y) => {
                        let mut mapped = Vec::new();
                        for leaf in leaves(x) {
                            mapped.push(self.prim(op, &[leaf, y.clone()], None, dtype)?);
                        }
                        Ok(rebuild(x, &mut mapped.into_iter()))
                    }
                    (x, Sym::Tuple(_) | Sym::Record(_)) if is_real_scalar(x) => {
                        let mut mapped = Vec::new();
                        for leaf in leaves(y) {
                            mapped.push(self.prim(op, &[x.clone(), leaf], None, dtype)?);
                        }
                        Ok(rebuild(y, &mut mapped.into_iter()))
                    }
                    (x, y) if is_tree(x) || is_tree(y) => {
                        let (lx, ly) = (leaves(x), leaves(y));
                        if lx.len() != ly.len() {
                            return internal(format!("{}: structure mismatch", op.name()));
                        }
                        let mut mapped = Vec::new();
                        for (p, q) in lx.into_iter().zip(ly) {
                            mapped.push(self.prim(op, &[p, q], None, dtype)?);
                        }
                        Ok(rebuild(x, &mut mapped.into_iter()))
                    }
                    (x, y) => {
                        let (tx, ty) = (self.as_tensor(x, dtype)?, self.as_tensor(y, dtype)?);
                        self.tensor_prim(&[tx, ty])
                    }
                }
            }
            _ => {
                if args.len() == 1 {
                    let scalar = || lower::lower(op, &[Operand { axes: vec![], kind: BufKind::Real }], dtype).is_some();
                    match &args[0] {
                        x if is_tree(x) => {
                            let mut mapped = Vec::new();
                            for leaf in leaves(x) {
                                mapped.push(self.prim(op, &[leaf], None, dtype)?);
                            }
                            return Ok(rebuild(x, &mut mapped.into_iter()));
                        }
                        Sym::Nat(n) if scalar() => {
                            let v = self.scalar_buf(Scalar::Const(*n as f64), dtype);
                            let Sym::Tensor(t) = self.tensor_prim(&[v])? else { unreachable!() };
                            return Ok(Sym::F64(self.read(t)));
                        }
                        Sym::F64(x) if scalar() => {
                            let v = self.scalar_buf(*x, dtype);
                            let Sym::Tensor(t) = self.tensor_prim(&[v])? else { unreachable!() };
                            return Ok(Sym::F64(self.read(t)));
                        }
                        _ => {}
                    }
                }
                let mut ins = Vec::with_capacity(args.len());
                for s in args {
                    match s {
                        Sym::Tensor(v) if v.bound.is_some() => ins.push(v.clone()),
                        other => ins.push(self.as_tensor(other, dtype)?),
                    }
                }
                self.tensor_prim(&ins)
            }
        }
    }

    /// The host's `fused_on`: a parameter tree maps the group over its
    /// leaves, a scalar riding along with every leaf.
    fn fused_on(&mut self, vals: &[Sym], dtype: Dtype) -> R<Sym> {
        if let Some(template) = vals.iter().find(|v| is_tree(v)) {
            let template = template.clone();
            let per_input: Vec<Vec<Sym>> = vals.iter().map(|v| if is_tree(v) { leaves(v) } else { vec![v.clone()] }).collect();
            let n = vals.iter().filter(|v| is_tree(v)).map(|v| leaves(v).len()).max().unwrap_or(0);
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let args: Vec<Sym> = per_input.iter().map(|xs| if xs.len() == 1 { xs[0].clone() } else { xs[i].clone() }).collect();
                out.push(self.fused_on(&args, dtype)?);
            }
            return Ok(rebuild(&template, &mut out.into_iter()));
        }
        let mut ins = Vec::with_capacity(vals.len());
        for v in vals {
            ins.push(self.as_tensor(v, dtype)?);
        }
        Ok(Sym::Tensor(self.consume(&ins, false)?))
    }

    /// The host's `stack_trees`: leaf by leaf along a new leading axis.
    fn stack(&mut self, parts: &[Sym]) -> R<Sym> {
        let Some(first) = parts.first() else { return internal("a stack of nothing") };
        match first {
            Sym::Tensor(f) if f.bound.is_none() => {
                let store = self.rec.bufs[f.buf].store.rebuilt();
                let mut views = Vec::with_capacity(parts.len());
                for p in parts {
                    match p {
                        Sym::Tensor(v) if v.bound.is_none() => views.push(v.clone()),
                        other => return internal(format!("a stack of a tensor and a {}", kind(other))),
                    }
                }
                let n: usize = views.iter().map(|v| v.numel()).sum();
                let b = self.buf(store, n, Made::Copy);
                let mut at = 0;
                for v in views {
                    let k = v.numel();
                    self.push(Op::Copy { dst: b, at, src: v, n: k });
                    at += k;
                }
                let mut shape = vec![parts.len()];
                shape.extend(f.shape.iter().copied());
                Ok(Sym::Tensor(View { buf: b, off: 0, shape, bound: None }))
            }
            Sym::Record(fs) => {
                let mut out = Vec::with_capacity(fs.len());
                for (i, (n, _)) in fs.iter().enumerate() {
                    let subs: Vec<Sym> = parts.iter().map(|p| match p {
                        Sym::Record(g) => g[i].1.clone(),
                        other => other.clone(),
                    }).collect();
                    out.push((n.clone(), self.stack(&subs)?));
                }
                Ok(Sym::Record(out))
            }
            Sym::Tuple(xs) => {
                let mut out = Vec::with_capacity(xs.len());
                for i in 0..xs.len() {
                    let subs: Vec<Sym> = parts.iter().map(|p| match p {
                        Sym::Tuple(g) => g[i].clone(),
                        other => other.clone(),
                    }).collect();
                    out.push(self.stack(&subs)?);
                }
                Ok(Sym::Tuple(out))
            }
            other => Ok(other.clone()),
        }
    }

    /// The host's `set_slice`: each leaf of `stacked` with its slice `i`
    /// replaced by the matching leaf of `value`, in a new buffer.
    fn set_slice(&mut self, stacked: &Sym, i: usize, value: &Sym) -> R<Sym> {
        let vs = leaves(value);
        let mut k = 0;
        map_leaves(stacked, &mut |leaf| {
            let v = vs.get(k).cloned();
            k += 1;
            match (leaf, v) {
                (Sym::Tensor(t), Some(Sym::Tensor(v))) if t.bound.is_none() && v.bound.is_none() && !t.shape.is_empty() && i < t.shape[0] => {
                    let n = t.numel();
                    let chunk = n / t.shape[0];
                    let b = self.buf(self.rec.bufs[t.buf].store.rebuilt(), n, Made::Copy);
                    self.push(Op::Copy { dst: b, at: 0, src: t.clone(), n });
                    self.push(Op::Copy { dst: b, at: i * chunk, src: v.clone(), n: chunk.min(v.numel()) });
                    Ok(Sym::Tensor(View { buf: b, off: 0, shape: t.shape.clone(), bound: None }))
                }
                (leaf, _) => Ok(leaf.clone()),
            }
        })
    }

    /// The host's `scatter_params`: zeros of each leaf of `tmpl`, the leaf of
    /// `ct` at slice `index`.
    fn scatter(&mut self, tmpl: &Sym, index: usize, ct: &Sym) -> R<Sym> {
        let cs = leaves(ct);
        let mut k = 0;
        map_leaves(tmpl, &mut |leaf| {
            let c = cs.get(k).cloned();
            k += 1;
            match (leaf, c) {
                (Sym::Tensor(t), Some(Sym::Tensor(c))) if t.bound.is_none() && c.bound.is_none() => {
                    let b = self.fill(self.rec.bufs[t.buf].store.rebuilt(), t.numel(), 0.0);
                    let n = c.numel();
                    self.push(Op::Copy { dst: b, at: index * n, src: c.clone(), n });
                    Ok(Sym::Tensor(View { buf: b, off: 0, shape: t.shape.clone(), bound: None }))
                }
                (other, _) => Ok(other.clone()),
            }
        })
    }
}

impl Recorder {
    /// The host's `slice_tree`: slice `i` along the leading axis of every
    /// real tensor leaf, a view at an offset. A leaf of integer weights stays
    /// integers where `QBuf::slice_leading` keeps them; where it gives up (the
    /// scales constant along the leading axis, or an `int4` slice of odd
    /// length, which would start inside a byte) the slice is a copy of its
    /// values in `f64`, as `with_data` makes it (step 7).
    fn slice(&mut self, s: &Sym, i: usize) -> R<Sym> {
        map_leaves(s, &mut |leaf| match leaf {
            Sym::Tensor(v) if v.bound.is_none() && !v.shape.is_empty() => {
                let chunk = v.numel() / v.shape[0].max(1);
                let at = View { buf: v.buf, off: v.off + i * chunk, shape: v.shape[1..].to_vec(), bound: None };
                let rb = &self.rec.bufs[v.buf];
                if let (Store::Quant { bits, .. }, Some(q)) = (rb.store, rb.quant) {
                    let over = (q.over + v.shape.len()).checked_sub(q.rank);
                    if over == Some(0) || v.shape[0] == 0 || (bits == 4 && chunk % 2 == 1) {
                        let b = self.buf(Store::F64, chunk, Made::Copy);
                        self.push(Op::Copy { dst: b, at: 0, src: at, n: chunk });
                        return Ok(Sym::Tensor(View { buf: b, off: 0, shape: v.shape[1..].to_vec(), bound: None }));
                    }
                    if over.is_none() {
                        return internal("a slice of integer weights of more axes than they have");
                    }
                }
                Ok(Sym::Tensor(at))
            }
            other => Ok(other.clone()),
        })
    }
}
