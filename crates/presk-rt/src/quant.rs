//! Integer quantisation of parameters (plan 0.6.0 step 5, D44).
//!
//! A quantised parameter is a buffer of `int8` or `int4` and one scale per
//! channel, and it means `f32(q·s)`: each weight is dequantised to `f32` when
//! it is loaded, and every contraction then runs in `f32` exactly as it does
//! for any other realization. That makes it a *storage* type, like `bf16`
//! (D38) with a scale, and it needs no new arithmetic and no new reduction
//! order.
//!
//! Only parameters are quantised, and their scales are computed from the
//! parameters themselves — symmetric absmax, round to nearest — so a
//! quantised realization is not conditional on a magnitude envelope (D35).
//! The *scheme* belongs to the realization and the *scales* are data derived
//! from it and a checkpoint (question 57): they are recomputed whenever the
//! weights are loaded rather than written into the realization file.
//!
//! A scale is constant along the axis a parameter is *contracted* over, and
//! there is one per index of every other axis. That is what lets the scale
//! factor out of the sum, and it is also what "per output channel" means once
//! a parameter is stacked (`stack{L}` adds a leading axis, and each layer
//! must get its own scales) or split into heads.

use std::collections::BTreeMap;

use crate::{Axis, Buf, Result, Tensor, Value, err};

/// How the scale of a channel is chosen from its largest magnitude `m`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantScheme {
    /// `s = m / qmax`, rounded to `f32`: the practice, and the tightest.
    #[default]
    Absmax,
    /// `s` the smallest power of two with `m / s ≤ qmax`. Up to twice as
    /// coarse, and in exchange `q·s` is exact in `f32`, so applying the scale
    /// on load or once after a sum gives the same bits.
    Pow2,
}

/// An axis of a parameter, by name or by position. Module parameters are
/// often written without axis names (`param w : ℝ[In, Out]`), so a name
/// cannot be required; a position counts from the end when negative, which
/// survives the leading axis `stack{L}` adds.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum AxisRef {
    Pos(i64),
    Name(String),
}

impl AxisRef {
    /// The index of this axis among `axes`.
    pub fn resolve(&self, axes: &[Axis]) -> Option<usize> {
        match self {
            AxisRef::Pos(p) => {
                let n = axes.len() as i64;
                let i = if *p < 0 { n + p } else { *p };
                (0..n).contains(&i).then_some(i as usize)
            }
            AxisRef::Name(s) => {
                let mut hits = axes.iter().enumerate().filter(|(_, a)| a.name.as_deref() == Some(s.as_str()));
                match (hits.next(), hits.next()) {
                    (Some((i, _)), None) => Some(i),
                    _ => None,
                }
            }
        }
    }
}

impl std::fmt::Display for AxisRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AxisRef::Pos(p) => write!(f, "{p}"),
            AxisRef::Name(s) => f.write_str(s),
        }
    }
}

/// How one parameter is quantised: what a realization records about it.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Quant {
    /// 8 or 4.
    pub bits: u8,
    /// The axis the scale is constant along: the one the parameter is
    /// contracted over.
    pub over: AxisRef,
    #[serde(default)]
    pub scheme: QuantScheme,
}

impl Quant {
    /// The largest magnitude a quantised integer takes, `2^(b−1) − 1`. The
    /// range is symmetric, so `-2^(b−1)` is never used and negation is exact.
    pub fn qmax(&self) -> i32 {
        (1 << (self.bits - 1)) - 1
    }

    pub fn name(&self) -> String {
        match self.scheme {
            QuantScheme::Absmax => format!("int{}", self.bits),
            QuantScheme::Pow2 => format!("int{}/pow2", self.bits),
        }
    }
}

/// A quantised tensor: integers, packed, and a scale per channel.
#[derive(Clone, Debug, PartialEq)]
pub struct QTensor {
    pub axes: Vec<Axis>,
    /// The index of the axis the scales are constant along.
    pub over: usize,
    pub quant: Quant,
    /// One per index of the axes other than `over`, row-major over them.
    pub scales: Vec<f32>,
    /// `int8`: one byte per element. `int4`: two per byte, the even element
    /// in the low nibble, two's complement.
    pub data: Vec<u8>,
}

impl QTensor {
    /// Quantise a real tensor. Its buffer is read in `f64`, which is the
    /// checkpoint's precision.
    pub fn quantize(t: &Tensor, q: &Quant) -> Result<QTensor> {
        if q.bits != 8 && q.bits != 4 {
            return err("E7001", format!("quantisation to {} bits: only 8 and 4 are implemented", q.bits));
        }
        let Some(over) = q.over.resolve(&t.axes) else {
            return err("E7001", format!("no axis `{}` to quantise over in a tensor of rank {}", q.over, t.rank()));
        };
        let w = t.data.to_f64();
        let (outer, k, inner) = split(&t.axes, over);
        let qmax = q.qmax() as f64;
        let mut scales = vec![0f32; outer * inner];
        for o in 0..outer {
            for i in 0..inner {
                let m = (0..k).map(|j| w[(o * k + j) * inner + i].abs()).fold(0.0, f64::max);
                scales[o * inner + i] = scale_of(m, qmax, q.scheme);
            }
        }
        let mut ints = vec![0i8; w.len()];
        for (e, x) in w.iter().enumerate() {
            let s = scales[channel(e, k, inner)] as f64;
            // `|x/s| ≤ qmax` up to the rounding of `s` to `f32`, which moves
            // it by 2⁻²⁴ relative and so never past `qmax + ½`: the clamp is
            // a guard, not a saturation.
            let v = if s == 0.0 { 0.0 } else { (x / s).round_ties_even().clamp(-qmax, qmax) };
            ints[e] = v as i8;
        }
        Ok(QTensor { axes: t.axes.clone(), over, quant: q.clone(), scales, data: pack(&ints, q.bits) })
    }

    pub fn numel(&self) -> usize {
        self.axes.iter().map(|a| a.len).product()
    }

    /// The integer at flat index `e`.
    pub fn int(&self, e: usize) -> i8 {
        match self.quant.bits {
            8 => self.data[e] as i8,
            _ => nibble(self.data[e / 2], e % 2 == 1),
        }
    }

    /// The scale that applies to flat index `e`.
    pub fn scale(&self, e: usize) -> f32 {
        let (_, k, inner) = split(&self.axes, self.over);
        self.scales[channel(e, k, inner)]
    }

    /// The value the realization denotes: `f32(q·s)`, one rounding, none
    /// under `pow2`.
    pub fn dequantize(&self) -> Tensor {
        let (_, k, inner) = split(&self.axes, self.over);
        let v = (0..self.numel()).map(|e| self.int(e) as f32 * self.scales[channel(e, k, inner)]).collect();
        Tensor { axes: self.axes.clone(), data: Buf::F32(v) }
    }

    /// The error rule of plan 0.6.0 §3.3: `|w − q·s| ≤ s/2`, absolute, and
    /// this is its largest instance over the channels. Dequantising to `f32`
    /// adds one rounding of `q·s` on top, which `pow2` does not have.
    pub fn half_step(&self) -> f64 {
        self.scales.iter().map(|s| *s as f64 / 2.0).fold(0.0, f64::max)
    }

    /// Bytes stored: the integers and the `f32` scales.
    pub fn bytes(&self) -> usize {
        self.data.len() + 4 * self.scales.len()
    }
}

impl QTensor {
    /// The runtime tensor: integers and scales in a [`Buf::Quant`], which a
    /// kernel loads as `f32(q)·s` and everything else reads as its
    /// dequantised values.
    pub fn to_tensor(&self) -> Tensor {
        let (_, k, inner) = split(&self.axes, self.over);
        let q = QBuf { bits: self.quant.bits, over: self.over, k, inner, data: padded(self.data.clone(), self.quant.bits), scales: self.scales.clone() };
        Tensor { axes: self.axes.clone(), data: Buf::Quant(std::sync::Arc::new(q)) }
    }
}

/// A quantised buffer as it crosses the runtime seam (0.6.0 step 5b).
///
/// It carries its own geometry — which axis the scales are constant along,
/// its length `k` and the product `inner` of the axes after it — so that
/// anything holding it can read element `e` as `f32(q_e)·s_c(e)` without the
/// tensor's axes. The integers and scales are shared, not copied, when the
/// buffer is.
#[derive(Clone, Debug, PartialEq)]
pub struct QBuf {
    pub bits: u8,
    pub over: usize,
    pub k: usize,
    pub inner: usize,
    pub data: Vec<u8>,
    pub scales: Vec<f32>,
}

impl QBuf {
    pub fn len(&self) -> usize {
        self.scales.len() * self.k
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn int(&self, e: usize) -> i8 {
        match self.bits {
            8 => self.data[e] as i8,
            _ => nibble(self.data[e / 2], e % 2 == 1),
        }
    }

    /// The index of the scale element `e` uses.
    pub fn channel(&self, e: usize) -> usize {
        channel(e, self.k, self.inner)
    }

    /// Element `e` as the realization denotes it: `f32(q)·s`, one `f32`
    /// multiplication. Every executor must compute exactly this.
    pub fn value(&self, e: usize) -> f32 {
        self.int(e) as f32 * self.scales[self.channel(e)]
    }

    pub fn dtype(&self) -> &'static str {
        if self.bits == 8 { "int8" } else { "int4" }
    }

    /// Slice `i` of `n` along the leading axis, still quantised when that axis
    /// is not the one the scales are constant along — which is the case for
    /// every stacked matrix (`[L, In, Out]` over `In`), and what keeps a
    /// `stack{L}` fold's blocks on integer weights. `None` otherwise.
    pub fn slice_leading(&self, i: usize, n: usize) -> Option<QBuf> {
        if self.over == 0 || n == 0 {
            return None;
        }
        let (len, sc) = (self.len() / n, self.scales.len() / n);
        let data = match self.bits {
            8 => self.data[i * len..(i + 1) * len].to_vec(),
            // Two to a byte: a slice is bytes only when it starts on one.
            _ if len % 2 == 0 => self.data[i * len / 2..(i + 1) * len / 2].to_vec(),
            _ => return None,
        };
        Some(QBuf { bits: self.bits, over: self.over - 1, k: self.k, inner: self.inner, data: padded(data, self.bits), scales: self.scales[i * sc..(i + 1) * sc].to_vec() })
    }
}

/// An `int4` buffer carries four zero bytes past its last element, because
/// the JIT reads four lanes as one 32-bit word, which from an odd element
/// reaches two bytes further than the lanes themselves.
fn padded(mut data: Vec<u8>, bits: u8) -> Vec<u8> {
    if bits == 4 {
        data.extend_from_slice(&[0; 4]);
    }
    data
}

/// `(outer, k, inner)` of a row-major tensor split at axis `over`.
fn split(axes: &[Axis], over: usize) -> (usize, usize, usize) {
    let outer = axes[..over].iter().map(|a| a.len).product();
    let inner = axes[over + 1..].iter().map(|a| a.len).product();
    (outer, axes[over].len, inner)
}

/// The channel of flat index `e`: its index with the `over` axis removed.
fn channel(e: usize, k: usize, inner: usize) -> usize {
    (e / (k * inner)) * inner + e % inner
}

fn scale_of(m: f64, qmax: f64, scheme: QuantScheme) -> f32 {
    if m == 0.0 {
        // An all-zero channel — a zero-initialized head — stores zeros and
        // never divides by its scale.
        return 0.0;
    }
    match scheme {
        QuantScheme::Absmax => (m / qmax) as f32,
        QuantScheme::Pow2 => {
            let mut s = 2f64.powi((m / qmax).log2().ceil() as i32);
            // `log2` may land a hair on either side of an exact power.
            while m / (s / 2.0) <= qmax {
                s /= 2.0;
            }
            while m / s > qmax {
                s *= 2.0;
            }
            s as f32
        }
    }
}

fn pack(ints: &[i8], bits: u8) -> Vec<u8> {
    match bits {
        8 => ints.iter().map(|x| *x as u8).collect(),
        _ => ints.chunks(2).map(|c| (c[0] as u8 & 0x0f) | (c.get(1).map_or(0, |x| *x as u8 & 0x0f) << 4)).collect(),
    }
}

fn nibble(b: u8, high: bool) -> i8 {
    let n = if high { b >> 4 } else { b & 0x0f };
    // Sign-extend four bits.
    ((n << 4) as i8) >> 4
}

// ----- parameter trees -------------------------------------------------------

/// The path of every real leaf of a parameter tree, in order: record fields
/// joined by `.`, tuple components by their index. `blocks.attn.wq.w` is the
/// query projection's matrix, stacked over the blocks.
pub fn param_paths(theta: &Value) -> Vec<String> {
    let mut out = Vec::new();
    walk(theta, "", &mut |p, v| {
        if matches!(v, Value::Real(_)) {
            out.push(p.to_string());
        }
    });
    out
}

fn walk(v: &Value, path: &str, f: &mut dyn FnMut(&str, &Value)) {
    let join = |n: &str| if path.is_empty() { n.to_string() } else { format!("{path}.{n}") };
    match v {
        Value::Record(fs) => fs.iter().for_each(|(n, x)| walk(x, &join(n), f)),
        Value::Tuple(xs) => xs.iter().enumerate().for_each(|(i, x)| walk(x, &join(&i.to_string()), f)),
        leaf => f(path, leaf),
    }
}

/// Quantise the parameters a realization names. Every name must be a real
/// leaf of `theta`: a misspelt path quietly quantising nothing would be the
/// worst failure this could have.
pub fn quantize_params(theta: &Value, quant: &BTreeMap<String, Quant>) -> Result<BTreeMap<String, QTensor>> {
    let mut leaves = BTreeMap::new();
    walk(theta, "", &mut |p, v| {
        if let Value::Real(t) = v {
            leaves.insert(p.to_string(), t.clone());
        }
    });
    quant
        .iter()
        .map(|(path, q)| match leaves.get(path) {
            Some(t) => QTensor::quantize(t, q).map(|qt| (path.clone(), qt)).map_err(|mut e| {
                e.message = format!("parameter `{path}`: {}", e.message);
                e
            }),
            None => err("E7001", format!("the realization quantises `{path}`, which is not a parameter of the model (it has {})", param_paths(theta).join(", "))),
        })
        .collect()
}

/// `theta` with every quantised leaf replaced by its integers and scales,
/// which is what an entry runs on (step 5b).
pub fn quantized_leaves(theta: &Value, q: &BTreeMap<String, QTensor>) -> Value {
    fn go(v: &Value, path: &str, q: &BTreeMap<String, QTensor>) -> Value {
        let join = |n: &str| if path.is_empty() { n.to_string() } else { format!("{path}.{n}") };
        match v {
            Value::Record(fs) => Value::Record(fs.iter().map(|(n, x)| (n.clone(), go(x, &join(n), q))).collect()),
            Value::Tuple(xs) => Value::Tuple(xs.iter().enumerate().map(|(i, x)| go(x, &join(&i.to_string()), q)).collect()),
            leaf => match q.get(path) {
                Some(qt) => Value::Real(qt.to_tensor()),
                None => leaf.clone(),
            },
        }
    }
    go(theta, "", q)
}

/// `theta` with every quantised leaf replaced by the value it denotes,
/// `f32(q·s)`. This is what the reference interpreter runs on to measure the
/// representation part of a quantised realization's gap, and what a backend
/// without integer kernels runs on — the same numbers either way.
pub fn dequantize_params(theta: &Value, q: &BTreeMap<String, QTensor>) -> Value {
    fn go(v: &Value, path: &str, q: &BTreeMap<String, QTensor>) -> Value {
        let join = |n: &str| if path.is_empty() { n.to_string() } else { format!("{path}.{n}") };
        match v {
            Value::Record(fs) => Value::Record(fs.iter().map(|(n, x)| (n.clone(), go(x, &join(n), q))).collect()),
            Value::Tuple(xs) => Value::Tuple(xs.iter().enumerate().map(|(i, x)| go(x, &join(&i.to_string()), q)).collect()),
            leaf => match q.get(path) {
                Some(qt) => Value::Real(qt.dequantize()),
                None => leaf.clone(),
            },
        }
    }
    go(theta, "", q)
}
