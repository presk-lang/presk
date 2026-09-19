//! The runtime seam of the compiler (plan 0.3.0, step 1).
//!
//! A [`Backend`] executes PSK-Core entries: it `prepare`s an entry at bound
//! dimensions and a realization, then `call`s it on runtime [`Value`]s. The
//! reference interpreter is the first backend ([`InterpBackend`]); the
//! kernel-IR interpreter and the Cranelift JIT follow (steps 2 and 3).
//!
//! A runtime value is a tree (tuples, records) of tensors whose buffers are
//! `f64`, `f32`, `i64` — or *exact* MPFR reals ([`Buf::Exact`]), so that the
//! interpreter's values cross the seam without rounding: the training loop
//! written over the trait reproduces the reference digit for digit, and the
//! reference stays the oracle of the other backends.

use std::collections::BTreeMap;
use std::path::Path;

use presk_core::{FuncId, Program};
use presk_data::{Layout, Leaf, Reader};
use presk_interp::eval::Dims;
use presk_interp::key::Key;
pub use presk_interp::value::Axis;
use presk_interp::{Interp, RunOptions};
use presk_types::dim::DimVar;
use presk_types::elab::ElabProgram;
use presk_types::ty::Shape;
use rug::Float;

type IValue = presk_interp::Value<Float>;

/// The machine type of every real tensor of a native realization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Realize {
    F64,
    F32,
}

impl Realize {
    pub fn parse(s: &str) -> Option<Realize> {
        match s {
            "f64" => Some(Realize::F64),
            "f32" => Some(Realize::F32),
            _ => None,
        }
    }
}

/// The data of a tensor.
#[derive(Clone, Debug, PartialEq)]
pub enum Buf {
    /// Exact reals of the reference interpreter (MPFR at its precision).
    Exact(Vec<Float>),
    F64(Vec<f64>),
    F32(Vec<f32>),
    I64(Vec<i64>),
}

impl Buf {
    pub fn len(&self) -> usize {
        match self {
            Buf::Exact(v) => v.len(),
            Buf::F64(v) => v.len(),
            Buf::F32(v) => v.len(),
            Buf::I64(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Every element as `f64` (exact reals rounded once).
    pub fn to_f64(&self) -> Vec<f64> {
        match self {
            Buf::Exact(v) => v.iter().map(|x| x.to_f64()).collect(),
            Buf::F64(v) => v.clone(),
            Buf::F32(v) => v.iter().map(|x| *x as f64).collect(),
            Buf::I64(v) => v.iter().map(|x| *x as f64).collect(),
        }
    }

    pub fn dtype(&self) -> &'static str {
        match self {
            Buf::Exact(_) => "exact",
            Buf::F64(_) => "f64",
            Buf::F32(_) => "f32",
            Buf::I64(_) => "i64",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub axes: Vec<Axis>,
    pub data: Buf,
}

impl Tensor {
    pub fn rank(&self) -> usize {
        self.axes.len()
    }

    pub fn numel(&self) -> usize {
        self.axes.iter().map(|a| a.len).product()
    }

    pub fn scalar_f64(x: f64) -> Tensor {
        Tensor { axes: vec![], data: Buf::F64(vec![x]) }
    }
}

/// A runtime value: the interpreter's value shapes, with machine or exact buffers.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Real(Tensor),
    /// Indices below `bound`, `i64` data.
    Idx { bound: u64, t: Tensor },
    Nat(u64),
    Int(i64),
    F64(f64),
    Bool(bool),
    Key(Key),
    Tuple(Vec<Value>),
    Record(Vec<(String, Value)>),
}

#[derive(Clone, Debug)]
pub struct RtError {
    pub code: &'static str,
    pub message: String,
    pub span: Option<presk_syntax::Span>,
}

pub type Result<T> = std::result::Result<T, RtError>;

pub fn err<T>(code: &'static str, message: impl Into<String>) -> Result<T> {
    Err(RtError { code, message: message.into(), span: None })
}

impl From<presk_interp::RunError> for RtError {
    fn from(e: presk_interp::RunError) -> Self {
        RtError { code: "E7001", message: e.message, span: e.span }
    }
}

impl From<presk_data::DataError> for RtError {
    fn from(e: presk_data::DataError) -> Self {
        RtError { code: e.code, message: e.message, span: None }
    }
}

impl Value {
    /// The interpreter's value, exactly (no rounding).
    pub fn from_interp(v: &IValue) -> Result<Value> {
        Ok(match v {
            IValue::Real(t) => Value::Real(Tensor { axes: t.axes.clone(), data: Buf::Exact(t.data.clone()) }),
            IValue::Idx { bound, t } => Value::Idx { bound: *bound, t: Tensor { axes: t.axes.clone(), data: Buf::I64(t.data.clone()) } },
            IValue::Nat(n) => Value::Nat(*n),
            IValue::Int(n) => Value::Int(*n),
            IValue::F64(x) => Value::F64(*x),
            IValue::Bool(b) => Value::Bool(*b),
            IValue::Key(k) => Value::Key(k.clone()),
            IValue::Tuple(vs) => Value::Tuple(vs.iter().map(Value::from_interp).collect::<Result<_>>()?),
            IValue::Record(fs) => Value::Record(fs.iter().map(|(n, v)| Ok((n.clone(), Value::from_interp(v)?))).collect::<Result<_>>()?),
            IValue::Closure { .. } => return err("E7001", "a function value cannot cross the backend seam"),
        })
    }

    /// The interpreter's value at `prec` bits: exact buffers as they are,
    /// machine buffers injected exactly.
    pub fn to_interp(&self, prec: u32) -> IValue {
        match self {
            Value::Real(t) => IValue::Real(presk_interp::Tensor {
                axes: t.axes.clone(),
                data: match &t.data {
                    Buf::Exact(v) => v.iter().map(|x| Float::with_val(prec, x)).collect(),
                    Buf::F64(v) => v.iter().map(|x| Float::with_val(prec, *x)).collect(),
                    Buf::F32(v) => v.iter().map(|x| Float::with_val(prec, *x)).collect(),
                    Buf::I64(v) => v.iter().map(|x| Float::with_val(prec, *x)).collect(),
                },
            }),
            Value::Idx { bound, t } => IValue::Idx {
                bound: *bound,
                t: presk_interp::Tensor {
                    axes: t.axes.clone(),
                    data: match &t.data {
                        Buf::I64(v) => v.clone(),
                        other => other.to_f64().into_iter().map(|x| x as i64).collect(),
                    },
                },
            },
            Value::Nat(n) => IValue::Nat(*n),
            Value::Int(n) => IValue::Int(*n),
            Value::F64(x) => IValue::F64(*x),
            Value::Bool(b) => IValue::Bool(*b),
            Value::Key(k) => IValue::Key(k.clone()),
            Value::Tuple(vs) => IValue::Tuple(vs.iter().map(|v| v.to_interp(prec)).collect()),
            Value::Record(fs) => IValue::Record(fs.iter().map(|(n, v)| (n.clone(), v.to_interp(prec))).collect()),
        }
    }

    /// Every real buffer in the machine type of `r` (exact reals rounded once).
    pub fn realize(&self, r: Realize) -> Value {
        self.map_leaves(&mut |leaf| match leaf {
            Value::Real(t) => Value::Real(Tensor {
                axes: t.axes.clone(),
                data: match r {
                    Realize::F64 => Buf::F64(t.data.to_f64()),
                    Realize::F32 => Buf::F32(t.data.to_f64().into_iter().map(|x| x as f32).collect()),
                },
            }),
            other => other.clone(),
        })
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Value::Real(_) => "real tensor",
            Value::Idx { .. } => "index tensor",
            Value::Nat(_) => "nat",
            Value::Int(_) => "int",
            Value::F64(_) => "f64",
            Value::Bool(_) => "bool",
            Value::Key(_) => "key",
            Value::Tuple(_) => "tuple",
            Value::Record(_) => "record",
        }
    }

    pub fn map_leaves(&self, f: &mut dyn FnMut(&Value) -> Value) -> Value {
        match self {
            Value::Tuple(vs) => Value::Tuple(vs.iter().map(|v| v.map_leaves(f)).collect()),
            Value::Record(fs) => Value::Record(fs.iter().map(|(n, v)| (n.clone(), v.map_leaves(f))).collect()),
            leaf => f(leaf),
        }
    }

    pub fn field(&self, name: &str) -> Option<&Value> {
        match self {
            Value::Record(fs) => fs.iter().find(|(n, _)| n == name).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The scalar as `f64`, `NaN` when the value is not a real scalar.
    pub fn scalar_f64(&self) -> f64 {
        match self {
            Value::Real(t) if t.rank() == 0 => t.data.to_f64()[0],
            Value::F64(x) => *x,
            Value::Nat(n) => *n as f64,
            _ => f64::NAN,
        }
    }

    /// The sum of squares of every real leaf, in `f64` (for logs).
    pub fn sum_squares_f64(&self) -> f64 {
        let mut acc = 0.0;
        self.map_leaves(&mut |leaf| {
            if let Value::Real(t) = leaf {
                for x in t.data.to_f64() {
                    acc += x * x;
                }
            }
            leaf.clone()
        });
        acc
    }
}

/// The runtime axes of a closed shape at bound dimensions.
pub fn axes_of(s: &Shape, dims: &Dims) -> Result<Vec<Axis>> {
    if s.prefix.is_some() {
        return err("E7001", "cannot materialize a shape with an open prefix `...`");
    }
    let map: BTreeMap<DimVar, presk_types::dim::Rational> = dims.iter().map(|(v, n)| (*v, presk_types::dim::rat(*n as i64))).collect();
    s.axes
        .iter()
        .map(|a| match a.len.eval(&map) {
            Some(r) if r.is_integer() => Ok(Axis { name: a.name.clone(), len: r.to_integer().to_string().parse().unwrap_or(0) }),
            _ => err("E7001", "unbound or non-integer dimension in a shape"),
        })
        .collect()
}

// ----- the seam ------------------------------------------------------------

/// An entry point made ready by a backend.
#[derive(Clone, Debug)]
pub struct Prepared {
    pub func: FuncId,
    pub dims: Dims,
    pub realize: Realize,
    /// Backend-specific handle (a compiled kernel program's index, …).
    pub handle: usize,
}

/// An executor of PSK-Core entries.
pub trait Backend {
    fn name(&self) -> &'static str;
    /// Bind an entry to concrete dimensions and a realization.
    fn prepare(&mut self, entry: FuncId, dims: &Dims, realize: Realize) -> Result<Prepared>;
    /// Run a prepared entry.
    fn call(&mut self, p: &Prepared, args: Vec<Value>) -> Result<Value>;
}

/// The reference interpreter behind the seam: exact values pass through
/// untouched; machine buffers are injected exactly at its precision.
pub struct InterpBackend<'p> {
    pub interp: Interp<'p, Float>,
    pub prec: u32,
}

impl<'p> InterpBackend<'p> {
    pub fn new(prog: &'p Program, elab: &'p ElabProgram, bits: u32, seed: u64) -> Self {
        InterpBackend { interp: Interp::new(prog, elab, bits, seed), prec: bits }
    }
}

impl Backend for InterpBackend<'_> {
    fn name(&self) -> &'static str {
        "interp"
    }

    fn prepare(&mut self, entry: FuncId, dims: &Dims, realize: Realize) -> Result<Prepared> {
        Ok(Prepared { func: entry, dims: dims.clone(), realize, handle: 0 })
    }

    fn call(&mut self, p: &Prepared, args: Vec<Value>) -> Result<Value> {
        let args: Vec<IValue> = args.iter().map(|a| a.to_interp(self.prec)).collect();
        let out = self.interp.call(p.func, p.dims.clone(), args)?;
        Value::from_interp(&out)
    }
}

/// `presk run` over a backend: the interpreter synthesizes the inputs, the
/// backend runs the entry; the output is the interpreter's `RunOutput` (the
/// result at `prec` bits, exact for the interpreter, rounded once otherwise).
pub fn run_with(backend: &mut dyn Backend, prog: &Program, elab: &ElabProgram, opts: &RunOptions, realize: Realize) -> Result<presk_interp::RunOutput> {
    let (func, dims, inputs) = presk_interp::run::prepare_run(prog, elab, opts)?;
    let p = backend.prepare(func, &dims, realize)?;
    let args: Vec<Value> = inputs.iter().map(|(_, v)| Value::from_interp(v)).collect::<Result<_>>()?;
    let result = backend.call(&p, args)?;
    let dims_named: Vec<(String, u64)> = dims.iter().map(|(v, n)| (elab.dim_name(*v), *n)).collect();
    Ok(presk_interp::RunOutput { entry: opts.entry.clone(), func, dims: dims_named, inputs, result: result.to_interp(opts.bits) })
}

// ----- parameter trees on disk -----------------------------------------------

/// A parameter tree as a `.psk.d` sample: `f64` leaves, fields in order.
pub fn params_to_data(v: &Value) -> Result<(Layout, presk_data::Value)> {
    match v {
        Value::Record(fs) => {
            let mut ls = Vec::new();
            let mut vs = Vec::new();
            for (n, sub) in fs {
                let (l, d) = params_to_data(sub)?;
                ls.push((n.clone(), l));
                vs.push((n.clone(), d));
            }
            Ok((Layout::Record(ls), presk_data::Value::Record(vs)))
        }
        Value::Tuple(ts) => {
            let (ls, vs): (Vec<_>, Vec<_>) = ts.iter().map(params_to_data).collect::<Result<Vec<_>>>()?.into_iter().unzip();
            Ok((Layout::Tuple(ls), presk_data::Value::Tuple(vs)))
        }
        Value::Real(t) => Ok((Layout::Leaf(Leaf::Real { shape: t.axes.iter().map(|a| a.len).collect() }), presk_data::Value::Real(t.data.to_f64()))),
        other => err("E9005", format!("a parameter tree holds no {}", other.kind())),
    }
}

/// The parameter tree read from a `.psk.d` sample, with the axes of `template`
/// and its buffers in `f64` (the file's precision).
pub fn params_from_data(template: &Value, d: &presk_data::Value) -> Result<Value> {
    match (template, d) {
        (Value::Record(fs), presk_data::Value::Record(ds)) if fs.len() == ds.len() => Ok(Value::Record(fs.iter().zip(ds).map(|((n, t), (_, x))| Ok((n.clone(), params_from_data(t, x)?))).collect::<Result<_>>()?)),
        (Value::Tuple(ts), presk_data::Value::Tuple(ds)) if ts.len() == ds.len() => Ok(Value::Tuple(ts.iter().zip(ds).map(|(t, x)| params_from_data(t, x)).collect::<Result<_>>()?)),
        (Value::Real(t), presk_data::Value::Real(xs)) if t.numel() == xs.len() => Ok(Value::Real(Tensor { axes: t.axes.clone(), data: Buf::F64(xs.clone()) })),
        _ => err("E9005", "the checkpoint does not have the structure of the model's parameters"),
    }
}

/// Writes `θ` as a one-sample `.psk.d` of type `params <model>`.
pub fn save_params(path: &Path, model: &str, theta: &Value) -> Result<()> {
    let (layout, data) = params_to_data(theta)?;
    presk_data::write(path, &format!("params {model}"), &layout, std::slice::from_ref(&data))?;
    Ok(())
}

/// Reads a checkpoint written by [`save_params`], shaped like `template`.
pub fn load_params(path: &Path, template: &Value) -> Result<Value> {
    let (layout, _) = params_to_data(template)?;
    let mut r = Reader::open(path)?.with_layout(layout)?;
    if r.is_empty() {
        return err("E9003", format!("{}: no sample", path.display()));
    }
    let d = r.get(0)?;
    params_from_data(template, &d)
}
