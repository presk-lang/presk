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
use presk_data::{Layout as DataLayout, Leaf, Reader};
use presk_interp::eval::Dims;
use presk_interp::key::Key;
pub use presk_interp::value::Axis;
use presk_interp::{Interp, RunOptions};
use presk_types::dim::DimVar;
use presk_types::elab::ElabProgram;
use presk_types::ty::Shape;
use rug::Float;

pub mod elems;
pub mod layout;
pub mod quant;
pub use elems::{Elems, FreeList, Pooled};
pub use layout::{Layout, NestKind, Plan};
pub use quant::{AxisRef, QBuf, QTensor, Quant, QuantScheme};

type IValue = presk_interp::Value<Float>;

/// A machine type a real tensor can be stored in.
///
/// `Bf16` is **storage only**: buffers hold sixteen bits, arithmetic is done
/// in `f32`, and conversion happens on load and on store with
/// round-to-nearest-even. That is the standard mixed-precision idiom, it is
/// exactly implementable, and it is what gives a solver a third rung with a
/// real traffic win on the memory-bound sweeps. There is no `f16`: its
/// dynamic range is a separate problem (plan 0.5.0 §3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    F64,
    F32,
    Bf16,
}

impl Dtype {
    pub fn parse(s: &str) -> Option<Dtype> {
        match s {
            "f64" => Some(Dtype::F64),
            "f32" => Some(Dtype::F32),
            "bf16" => Some(Dtype::Bf16),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Dtype::F64 => "f64",
            Dtype::F32 => "f32",
            Dtype::Bf16 => "bf16",
        }
    }

    /// Bytes a stored element occupies. This is the number the roofline's `Q`
    /// is counted in, so it is what a narrower storage type actually buys.
    pub fn bytes(self) -> usize {
        match self {
            Dtype::F64 => 8,
            Dtype::F32 => 4,
            Dtype::Bf16 => 2,
        }
    }

    /// Lanes a 128-bit vector register holds of this type, which is the only
    /// width the backends emit. `bf16` is one: a narrow lane needs a
    /// conversion per element, which is a different kernel.
    ///
    /// This belongs to the *type*, not to a realization's default, because a
    /// mixed assignment has kernels of several types and each takes its own
    /// width. Deriving it from the default is how a mixed realization came to
    /// ask for a two-lane `bf16` kernel, which is refused.
    pub fn natural_lanes(self) -> usize {
        match self {
            Dtype::F64 => 2,
            Dtype::F32 => 4,
            Dtype::Bf16 => 1,
        }
    }

    /// The type arithmetic is carried out in. `bf16` computes in `f32`.
    pub fn arith(self) -> Dtype {
        match self {
            Dtype::Bf16 => Dtype::F32,
            other => other,
        }
    }

    /// The unit roundoff `u` of this type: half an ulp of 1, and the constant
    /// every first-order error rule is written in terms of (step 4). A type
    /// of `p` bits of significand has `u = 2^-p`: 2^-53, 2^-24, and 2^-8 for
    /// `bf16`'s 8 (7 stored and the implicit one). It was 2^-9 until 0.10.0
    /// step 8, every `bf16` rounding counted at half its size (the language
    /// reference's chapter 4).
    pub fn unit_roundoff(self) -> f64 {
        match self {
            Dtype::F64 => f64::EPSILON / 2.0,
            Dtype::F32 => f32::EPSILON as f64 / 2.0,
            Dtype::Bf16 => 2f64.powi(-8),
        }
    }
}

impl std::fmt::Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Round an `f32` to the nearest `bf16`, ties to even, and keep it as the
/// sixteen bits that are stored. NaN is preserved as a quiet NaN.
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let lsb = (bits >> 16) & 1;
    let rounded = bits + 0x7fff + lsb;
    (rounded >> 16) as u16
}

/// The `f32` a stored `bf16` denotes. Exact: `bf16` is a prefix of `f32`.
pub fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

/// How a sum is allowed to be accumulated.
///
/// This is the field decisions D31 and D32 of 0.4.0 both needed and could not
/// write down. It is not a performance hint: it is an argument of the error
/// bound (step 4), because the constant in front of `u` in a contraction's
/// rule depends on it, and it is what a conformance report names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum ReductionOrder {
    /// Every addition in index order. The semantics of 0.1.0 to 0.4.0.
    Sequential,
    /// Register blocking: several *results* accumulated at once, each still
    /// summing its own terms in index order. Bit-identical to `Sequential`
    /// (D32), and declared separately only because the kernels differ.
    Blocked { width: u32 },
    /// `n` partial accumulators summed at the end — a vector lane, a packed
    /// micro-kernel, a thread split. This one **reassociates**, so it is a
    /// different realization and owes a measured gap (D31).
    Interleaved { partials: u32 },
}

impl ReductionOrder {
    /// Whether this order preserves the exact order of additions, and so is
    /// bit-identical to `Sequential` at the same dtype.
    pub fn is_exact(self) -> bool {
        !matches!(self, ReductionOrder::Interleaved { .. })
    }

    pub fn name(self) -> String {
        match self {
            ReductionOrder::Sequential => "sequential".into(),
            ReductionOrder::Blocked { width } => format!("blocked({width})"),
            ReductionOrder::Interleaved { partials } => format!("interleaved({partials})"),
        }
    }
}

/// What the backends are allowed to do with a program's arithmetic.
///
/// Versions 0.1.0 to 0.4.0 had a two-valued flag here. It became an object in
/// 0.5.0 (D34) because two separate needs met: the precision solver has to
/// *produce* a per-site assignment, and the fast kernels have to *declare* a
/// reduction order before they may ship. One record answers both.
///
/// A site is `"<function>/<let>"` — a fusion group's root, which is the
/// smallest unit at which a type is a decision rather than a conversion. A
/// site the map does not mention takes its function's type from
/// `functions`, and failing that `default`.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Realization {
    pub default: Dtype,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub storage: BTreeMap<String, Dtype>,
    /// A type for every site of a function, by the function's name
    /// (`Linear.apply.vjp`, `adamw.step`): the unit at which the solver
    /// chooses over a training step, where the reverse sweep and the
    /// optimizer have no per-site sensitivity to rank by (0.6.0 step 7b,
    /// D52). A site named in `storage` overrides it.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub functions: BTreeMap<String, Dtype>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub accum: BTreeMap<String, Dtype>,
    pub order: ReductionOrder,
    /// The magnitude envelope any bound about this realization assumed
    /// (step 3). A bound without one is a different and unsupported claim
    /// (D35), so the reference travels with the realization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub envelope: Option<String>,
    /// Parameters stored as integers with per-channel scales, keyed by their
    /// path in the parameter tree (`blocks.attn.wq.w`), not by site: a
    /// quantised weight is a property of one tensor feeding a contraction,
    /// not of a loop nest (D44). The scales are not here — they are derived
    /// from this and the weights whenever the weights are loaded.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub quant: BTreeMap<String, Quant>,
}

impl Realization {
    /// One dtype everywhere, additions in index order: what `--realize f64`
    /// and `--realize f32` meant in every version up to 0.4.0, and still do.
    pub fn uniform(d: Dtype) -> Realization {
        Realization { default: d, storage: BTreeMap::new(), functions: BTreeMap::new(), accum: BTreeMap::new(), order: ReductionOrder::Sequential, envelope: None, quant: BTreeMap::new() }
    }

    /// `f64`, `f32`, `bf16`, or `@PATH` for a realization file.
    pub fn parse(spec: &str) -> std::result::Result<Realization, String> {
        if let Some(path) = spec.strip_prefix('@') {
            let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
            return serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"));
        }
        match Dtype::parse(spec) {
            Some(d) => Ok(Realization::uniform(d)),
            None => Err(format!("unknown realization `{spec}`; expected f64, f32, bf16 or @FILE")),
        }
    }

    /// The storage type of a site: its own, else its function's, else the
    /// default.
    pub fn dtype_at(&self, site: &str) -> Dtype {
        if let Some(d) = self.storage.get(site) {
            return *d;
        }
        if let Some(d) = site.rsplit_once('/').and_then(|(f, _)| self.functions.get(f)) {
            return *d;
        }
        match fwd_base(site) {
            Some(base) => self.dtype_at(&base),
            None => self.default,
        }
    }

    /// The accumulator type of a site's contraction. It defaults to the
    /// *arithmetic* type of the storage, so `bf16` storage accumulates in
    /// `f32` unless something says otherwise — the idiom, not a surprise.
    pub fn accum_at(&self, site: &str) -> Dtype {
        if let Some(d) = self.accum.get(site) {
            return *d;
        }
        match fwd_base(site) {
            Some(base) if !self.storage.contains_key(site) && !site.rsplit_once('/').is_some_and(|(f, _)| self.functions.contains_key(f)) => self.accum_at(&base),
            _ => self.dtype_at(site).arith(),
        }
    }

    /// Whether a fusion group can compute a value otherwise than its site
    /// would alone under this realization (plan 0.10.0 step 4b, D94). A
    /// group computes every member in its root site's arithmetic type and
    /// stores only the root, and never spans two functions; so it computes
    /// otherwise under a `bf16` site, whose arithmetic type is `f32`, and
    /// under a site typed apart from the rest of its function, a `storage`
    /// entry other than the type the site would take without it. Otherwise
    /// every site of a function computes and stores in one type, and how its
    /// `let`s are grouped changes no bit: differentiation need not keep the
    /// groups (`presk_core::ad::Options::groups`). Integer weights, the
    /// accumulators and the order are the contractions', which no group
    /// holds.
    pub fn groups_visible(&self) -> bool {
        let bf16 = self.default == Dtype::Bf16 || self.storage.values().chain(self.functions.values()).any(|d| *d == Dtype::Bf16);
        bf16 || self.storage.iter().any(|(site, d)| {
            let mut without = self.clone();
            without.storage.remove(site);
            without.dtype_at(site) != *d
        })
    }

    /// Whether every site of this realization has one type and additions keep
    /// their order. Uniform realizations are the ones earlier versions could
    /// express, and the ones whose goldens must not move.
    pub fn is_uniform(&self) -> bool {
        self.storage.is_empty() && self.functions.is_empty() && self.accum.is_empty() && self.order == ReductionOrder::Sequential && self.quant.is_empty()
    }

    /// A short name for reports and cache keys.
    pub fn label(&self) -> String {
        if self.is_uniform() {
            return self.default.name().into();
        }
        let mut s = format!("{}+{}", self.default, self.order.name());
        if !self.storage.is_empty() {
            s.push_str(&format!("+{}sites", self.storage.len()));
        }
        if !self.functions.is_empty() {
            s.push_str(&format!("+{}fns", self.functions.len()));
        }
        if !self.quant.is_empty() {
            s.push_str(&format!("+{}quant", self.quant.len()));
        }
        s
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a realization serializes")
    }

    /// The parameters this realization runs on: `theta` with every quantised
    /// leaf replaced by its integers and scales ([`Buf::Quant`]), which the
    /// kernels load as `f32(q)·s` and the reference interpreter reads as the
    /// same values. The identity when nothing is quantised.
    pub fn quantized_params(&self, theta: &Value) -> Result<Value> {
        if self.quant.is_empty() {
            return Ok(theta.clone());
        }
        Ok(quant::quantized_leaves(theta, &quant::quantize_params(theta, &self.quant)?))
    }

    /// Quantise an entry's parameters, its first argument, before a run. A
    /// realization that quantises must never be run on the true weights
    /// unnoticed, so an entry without a parameter tree first is refused.
    pub fn quantize_args(&self, args: &mut [Value]) -> Result<()> {
        if self.quant.is_empty() {
            return Ok(());
        }
        match args.first_mut() {
            Some(theta @ Value::Record(_)) => {
                *theta = self.quantized_params(theta)?;
                Ok(())
            }
            _ => err("E7001", "the realization quantises parameters, and this entry's first argument is not a parameter tree"),
        }
    }
}

/// The data of a tensor. A machine buffer's elements are shared by its
/// clones ([`Elems`], plan 0.10.0 step 6): a value read, passed or put in a
/// tuple is not copied.
#[derive(Clone, Debug, PartialEq)]
pub enum Buf {
    /// Exact reals of the reference interpreter (MPFR at its precision).
    Exact(Vec<Float>),
    F64(Elems<f64>),
    F32(Elems<f32>),
    /// Storage-only sixteen-bit floats, held as their raw bits. Arithmetic on
    /// them happens in `f32`; only the traffic is halved.
    Bf16(Elems<u16>),
    I64(Elems<i64>),
    /// Integers with a scale per channel (0.6.0 step 5b, D44): a parameter a
    /// realization quantises. Read anywhere as `f32(q)·s`.
    Quant(std::sync::Arc<QBuf>),
}

impl Buf {
    pub fn len(&self) -> usize {
        match self {
            Buf::Exact(v) => v.len(),
            Buf::F64(v) => v.len(),
            Buf::F32(v) => v.len(),
            Buf::Bf16(v) => v.len(),
            Buf::I64(v) => v.len(),
            Buf::Quant(q) => q.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Every element as `f64` (exact reals rounded once).
    /// Elements `lo..hi` as `f64`: `to_f64()[lo..hi]`, without converting
    /// the rest (a stacked fold's slice, 0.10.0 step 4, where the stack holds
    /// every layer's residuals).
    pub fn range_f64(&self, lo: usize, hi: usize) -> Vec<f64> {
        match self {
            Buf::Exact(v) => v[lo..hi].iter().map(|x| x.to_f64()).collect(),
            Buf::F64(v) => v[lo..hi].to_vec(),
            Buf::F32(v) => v[lo..hi].iter().map(|x| *x as f64).collect(),
            Buf::Bf16(v) => v[lo..hi].iter().map(|x| bf16_to_f32(*x) as f64).collect(),
            Buf::I64(v) => v[lo..hi].iter().map(|x| *x as f64).collect(),
            Buf::Quant(q) => (lo..hi).map(|e| q.value(e) as f64).collect(),
        }
    }

    pub fn to_f64(&self) -> Vec<f64> {
        match self {
            Buf::Exact(v) => v.iter().map(|x| x.to_f64()).collect(),
            Buf::F64(v) => v.to_vec(),
            Buf::F32(v) => v.iter().map(|x| *x as f64).collect(),
            Buf::Bf16(v) => v.iter().map(|x| bf16_to_f32(*x) as f64).collect(),
            Buf::I64(v) => v.iter().map(|x| *x as f64).collect(),
            Buf::Quant(q) => (0..q.len()).map(|e| q.value(e) as f64).collect(),
        }
    }

    /// A buffer of `n` reals in the machine type `d`, rounded once from `f64`.
    pub fn of(d: Dtype, v: &[f64]) -> Buf {
        match d {
            Dtype::F64 => Buf::F64(v.to_vec().into()),
            Dtype::F32 => Buf::F32(v.iter().map(|x| *x as f32).collect()),
            Dtype::Bf16 => Buf::Bf16(v.iter().map(|x| f32_to_bf16(*x as f32)).collect()),
        }
    }

    /// Zeros of the machine type `d`.
    pub fn zeros(d: Dtype, n: usize) -> Buf {
        match d {
            Dtype::F64 => Buf::F64(vec![0.0; n].into()),
            Dtype::F32 => Buf::F32(vec![0.0; n].into()),
            Dtype::Bf16 => Buf::Bf16(vec![0u16; n].into()),
        }
    }

    pub fn dtype(&self) -> &'static str {
        match self {
            Buf::Exact(_) => "exact",
            Buf::F64(_) => "f64",
            Buf::F32(_) => "f32",
            Buf::Bf16(_) => "bf16",
            Buf::I64(_) => "i64",
            Buf::Quant(q) => q.dtype(),
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
        Tensor { axes: vec![], data: Buf::F64(vec![x].into()) }
    }
}

/// A site or nest of `g.fwd`, the forward pass a gradient keeps the
/// residuals of (plan 0.10.0 step 4), is the site of `g` of the same number:
/// `g.fwd`'s bindings are `g`'s. So a realization or a plan that states it
/// of `g` states it of `g.fwd`, unless it names `g.fwd`'s own.
pub fn fwd_base(site: &str) -> Option<String> {
    let (f, rest) = site.rsplit_once('/')?;
    f.strip_suffix(".fwd").map(|g| format!("{g}/{rest}"))
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
    /// A leaf with no axis to stack along (a key, a scalar), stacked by a
    /// scanned `fold`: one value per layer (the interpreter's
    /// `Value::Stacked`).
    Stacked(Vec<Value>),
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
        RtError { code: e.code.unwrap_or("E7001"), message: e.message, span: e.span }
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
            IValue::Idx { bound, t } => Value::Idx { bound: *bound, t: Tensor { axes: t.axes.clone(), data: Buf::I64(t.data.clone().into()) } },
            IValue::Nat(n) => Value::Nat(*n),
            IValue::Int(n) => Value::Int(*n),
            IValue::F64(x) => Value::F64(*x),
            IValue::Bool(b) => Value::Bool(*b),
            IValue::Key(k) => Value::Key(k.clone()),
            IValue::Tuple(vs) => Value::Tuple(vs.iter().map(Value::from_interp).collect::<Result<_>>()?),
            IValue::Record(fs) => Value::Record(fs.iter().map(|(n, v)| Ok((n.clone(), Value::from_interp(v)?))).collect::<Result<_>>()?),
            IValue::Stacked(vs) => Value::Stacked(vs.iter().map(Value::from_interp).collect::<Result<_>>()?),
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
                    Buf::Bf16(v) => v.iter().map(|x| Float::with_val(prec, bf16_to_f32(*x))).collect(),
                    Buf::I64(v) => v.iter().map(|x| Float::with_val(prec, *x)).collect(),
                    Buf::Quant(q) => (0..q.len()).map(|e| Float::with_val(prec, q.value(e))).collect(),
                },
            }),
            Value::Idx { bound, t } => IValue::Idx {
                bound: *bound,
                t: presk_interp::Tensor {
                    axes: t.axes.clone(),
                    data: match &t.data {
                        Buf::I64(v) => v.to_vec(),
                        other => other.to_f64().into_iter().map(|x| x as i64).collect(),
                    },
                },
            },
            Value::Nat(n) => IValue::Nat(*n),
            Value::Int(n) => IValue::Int(*n),
            Value::F64(x) => IValue::F64(*x),
            Value::Bool(b) => IValue::Bool(*b),
            Value::Key(k) => IValue::Key(k.clone()),
            Value::Tuple(vs) => IValue::tuple(vs.iter().map(|v| v.to_interp(prec)).collect()),
            Value::Record(fs) => IValue::record(fs.iter().map(|(n, v)| (n.clone(), v.to_interp(prec))).collect()),
            Value::Stacked(vs) => IValue::Stacked(std::sync::Arc::new(vs.iter().map(|v| v.to_interp(prec)).collect())),
        }
    }

    /// The same, consuming the value: an exact buffer is moved rather than
    /// copied, each real rounded in place to `prec` bits — the value
    /// `with_val` gives, and no change at the precision it was made at. A
    /// parameter tree then crosses the seam into the interpreter without a
    /// second copy (0.6.0 step 8: BERT-large's 335M exact parameters).
    pub fn into_interp(self, prec: u32) -> IValue {
        match self {
            Value::Real(Tensor { axes, data: Buf::Exact(mut v) }) => {
                for x in &mut v {
                    x.set_prec(prec);
                }
                IValue::Real(presk_interp::Tensor { axes, data: v })
            }
            Value::Tuple(vs) => IValue::tuple(vs.into_iter().map(|v| v.into_interp(prec)).collect()),
            Value::Record(fs) => IValue::record(fs.into_iter().map(|(n, v)| (n, v.into_interp(prec))).collect()),
            other => other.to_interp(prec),
        }
    }

    /// Every real buffer in the machine type `d` (exact reals rounded once).
    pub fn realize(&self, d: Dtype) -> Value {
        self.map_leaves(&mut |leaf| match leaf {
            Value::Real(t) => Value::Real(Tensor { axes: t.axes.clone(), data: Buf::of(d, &t.data.to_f64()) }),
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
            Value::Stacked(_) => "stacked value",
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

    /// `‖a − b‖₂²` over the real leaves of two values of the same type.
    /// Where either side is exact the difference is taken exactly and
    /// rounded once, so a distance between an update and its exact
    /// reference is not swamped by rounding `θ` itself: the update is a
    /// thousand times smaller than the parameters (0.6.0 step 7b).
    pub fn dist2_f64(&self, other: &Value) -> f64 {
        fn exact(b: &Buf, prec: u32) -> Vec<Float> {
            match b {
                Buf::Exact(v) => v.clone(),
                other => other.to_f64().into_iter().map(|x| Float::with_val(prec, x)).collect(),
            }
        }
        fn go(a: &Value, b: &Value, acc: &mut f64) {
            match (a, b) {
                (Value::Real(x), Value::Real(y)) => {
                    let prec = [&x.data, &y.data].iter().find_map(|d| if let Buf::Exact(v) = d { v.first().map(|f| f.prec()) } else { None });
                    match prec {
                        Some(p) => {
                            for (u, v) in exact(&x.data, p).iter().zip(exact(&y.data, p)) {
                                let d = Float::with_val(p, u - &v).to_f64();
                                *acc += d * d;
                            }
                        }
                        None => {
                            for (u, v) in x.data.to_f64().into_iter().zip(y.data.to_f64()) {
                                *acc += (u - v) * (u - v);
                            }
                        }
                    }
                }
                (Value::Tuple(xs), Value::Tuple(ys)) => xs.iter().zip(ys).for_each(|(x, y)| go(x, y, acc)),
                (Value::Record(xs), Value::Record(ys)) => xs.iter().zip(ys).for_each(|((_, x), (_, y))| go(x, y, acc)),
                _ => {}
            }
        }
        let mut acc = 0.0;
        go(self, other, &mut acc);
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
    pub realize: Realization,
    /// Backend-specific handle (a compiled kernel program's index, …).
    pub handle: usize,
}

/// An executor of PSK-Core entries.
pub trait Backend {
    fn name(&self) -> &'static str;
    /// Bind an entry to concrete dimensions and a realization. An entry
    /// undefined at the dimensions is refused here, before anything is
    /// computed ([`presk_interp::defined::check`], `E7003`).
    fn prepare(&mut self, entry: FuncId, dims: &Dims, realize: &Realization) -> Result<Prepared>;
    /// Run a prepared entry.
    fn call(&mut self, p: &Prepared, args: Vec<Value>) -> Result<Value>;
    /// Whether this run trains: `dropout` drops only when it does. Off by
    /// default, so `run`, `serve` and the export evaluate the expectation.
    fn set_training(&mut self, _on: bool) {}
    /// Kernels launched so far, and launches fusion removed. `(0, 0)` for a
    /// backend that launches no kernels, such as the reference interpreter.
    fn launch_stats(&self) -> (u64, u64) {
        (0, 0)
    }
}

/// The reference interpreter behind the seam: exact values pass through
/// untouched; machine buffers are injected exactly at its precision.
pub struct InterpBackend<'p> {
    pub interp: Interp<'p, Float>,
    pub prec: u32,
}

impl<'p> InterpBackend<'p> {
    pub fn new(prog: &'p Program, elab: &'p ElabProgram, bits: u32) -> Self {
        InterpBackend { interp: Interp::new(prog, elab, bits), prec: bits }
    }
}

impl Backend for InterpBackend<'_> {
    fn name(&self) -> &'static str {
        "interp"
    }

    fn set_training(&mut self, on: bool) {
        self.interp.training = on;
    }

    fn prepare(&mut self, entry: FuncId, dims: &Dims, realize: &Realization) -> Result<Prepared> {
        presk_interp::defined::check(self.interp.prog, self.interp.elab, entry, dims, None)?;
        Ok(Prepared { func: entry, dims: dims.clone(), realize: realize.clone(), handle: 0 })
    }

    fn call(&mut self, p: &Prepared, args: Vec<Value>) -> Result<Value> {
        let args: Vec<IValue> = args.into_iter().map(|a| a.into_interp(self.prec)).collect();
        let out = self.interp.call(p.func, p.dims.clone(), args)?;
        Value::from_interp(&out)
    }
}

/// `presk run` over a backend: the interpreter synthesizes the inputs, the
/// backend runs the entry; the output is the interpreter's `RunOutput` (the
/// result at `prec` bits, exact for the interpreter, rounded once otherwise).
pub fn run_with(backend: &mut dyn Backend, prog: &Program, elab: &ElabProgram, opts: &RunOptions, realize: &Realization) -> Result<presk_interp::RunOutput> {
    let (func, dims, inputs) = presk_interp::run::prepare_run(prog, elab, opts)?;
    let p = backend.prepare(func, &dims, realize)?;
    let mut args: Vec<Value> = inputs.iter().map(|(_, v)| Value::from_interp(v)).collect::<Result<_>>()?;
    realize.quantize_args(&mut args)?;
    let result = backend.call(&p, args)?;
    let dims_named: Vec<(String, u64)> = dims.iter().map(|(v, n)| (elab.dim_name(*v), *n)).collect();
    Ok(presk_interp::RunOutput { entry: opts.entry.clone(), func, dims: dims_named, inputs, result: result.to_interp(opts.bits) })
}

// ----- parameter trees on disk -----------------------------------------------

/// A parameter tree as a `.psk.d` sample: `f64` leaves, fields in order.
pub fn params_to_data(v: &Value) -> Result<(DataLayout, presk_data::Value)> {
    match v {
        Value::Record(fs) => {
            let mut ls = Vec::new();
            let mut vs = Vec::new();
            for (n, sub) in fs {
                let (l, d) = params_to_data(sub)?;
                ls.push((n.clone(), l));
                vs.push((n.clone(), d));
            }
            Ok((DataLayout::Record(ls), presk_data::Value::Record(vs)))
        }
        Value::Tuple(ts) => {
            let (ls, vs): (Vec<_>, Vec<_>) = ts.iter().map(params_to_data).collect::<Result<Vec<_>>>()?.into_iter().unzip();
            Ok((DataLayout::Tuple(ls), presk_data::Value::Tuple(vs)))
        }
        Value::Real(t) => Ok((DataLayout::Leaf(Leaf::Real { shape: t.axes.iter().map(|a| a.len).collect() }), presk_data::Value::Real(t.data.to_f64()))),
        other => err("E9005", format!("a parameter tree holds no {}", other.kind())),
    }
}

/// The parameter tree read from a `.psk.d` sample, with the axes of `template`
/// and its buffers in `f64` (the file's precision).
pub fn params_from_data(template: &Value, d: &presk_data::Value) -> Result<Value> {
    match (template, d) {
        (Value::Record(fs), presk_data::Value::Record(ds)) if fs.len() == ds.len() => Ok(Value::Record(fs.iter().zip(ds).map(|((n, t), (_, x))| Ok((n.clone(), params_from_data(t, x)?))).collect::<Result<_>>()?)),
        (Value::Tuple(ts), presk_data::Value::Tuple(ds)) if ts.len() == ds.len() => Ok(Value::Tuple(ts.iter().zip(ds).map(|(t, x)| params_from_data(t, x)).collect::<Result<_>>()?)),
        (Value::Real(t), presk_data::Value::Real(xs)) if t.numel() == xs.len() => Ok(Value::Real(Tensor { axes: t.axes.clone(), data: Buf::F64(xs.clone().into()) })),
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
