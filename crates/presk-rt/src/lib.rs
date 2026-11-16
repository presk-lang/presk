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

/// A machine type a real tensor can be stored in: a float format.
///
/// Since plan 0.18.0 step 1 (D204) a type is a format described by its
/// parameters ([`presk_format::Float`]), and `f64`, `f32` and `bf16` are
/// three of them; every property of a type is its format's. Since step 3
/// (D206) a realization may store a value in any float format: the native,
/// LLVM and C backends compute in the three *native* types alone
/// ([`Dtype::is_native`]) and refuse a realization that names another
/// (`E7005`, [`Realization::check_native`]), and the kernel interpreter
/// emulates it. The solver's menu offers the native types until step 5
/// ([`Dtype::native`], `E7004`).
///
/// `Bf16` is **storage only**: buffers hold sixteen bits, arithmetic is done
/// in `f32`, and conversion happens on load and on store with
/// round-to-nearest-even. That is the standard mixed-precision idiom, it is
/// exactly implementable, and it is what gives a solver a third rung with a
/// real traffic win on the memory-bound sweeps.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Dtype(presk_format::Float);

impl Dtype {
    pub const F64: Dtype = Dtype(presk_format::F64);
    pub const F32: Dtype = Dtype(presk_format::F32);
    pub const BF16: Dtype = Dtype(presk_format::BF16);
    /// The types a backend computes in, the native types, the most precise
    /// first.
    pub const ALL: [Dtype; 3] = [Dtype::F64, Dtype::F32, Dtype::BF16];

    /// The type of a float format.
    pub fn new(f: presk_format::Float) -> Dtype {
        Dtype(f)
    }

    /// The type of a format, if it is a native type: what the solver's menu
    /// offers until step 5 of plan 0.18.0 (`E7004` otherwise).
    pub fn native(f: presk_format::Float) -> std::result::Result<Dtype, String> {
        match Dtype::ALL.into_iter().find(|d| d.0 == f) {
            Some(d) => Ok(d),
            None => Err(format!("[E7004] `{f}` is a format the solver does not offer yet: its floats are f64, f32 and bf16")),
        }
    }

    /// Whether the backends compute in this type: `f64`, `f32` or `bf16`.
    pub fn is_native(self) -> bool {
        Dtype::ALL.contains(&self)
    }

    /// The type a float format's name denotes; `None` for a name that is
    /// no float format's.
    pub fn parse(s: &str) -> Option<Dtype> {
        Dtype::of_name(s).ok()
    }

    /// The type a float format's name denotes, or why there is none.
    pub fn of_name(s: &str) -> std::result::Result<Dtype, String> {
        presk_format::Float::parse(s).map(Dtype)
    }

    /// The format this type is.
    pub fn format(self) -> presk_format::Float {
        self.0
    }

    /// The format's name: its alias (`f16`), else its parameters, each
    /// distinct spelling kept once for the run.
    pub fn name(self) -> &'static str {
        if let Some(a) = self.0.alias() {
            return a;
        }
        static NAMES: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<presk_format::Float, &'static str>>> = std::sync::OnceLock::new();
        let mut names = NAMES.get_or_init(Default::default).lock().expect("the names' table");
        names.entry(self.0).or_insert_with(|| Box::leak(self.0.spelling().into_boxed_str()))
    }

    /// Bytes a stored element occupies. This is the number the roofline's `Q`
    /// is counted in, so it is what a narrower storage type actually buys.
    pub fn bytes(self) -> usize {
        self.0.bits().div_ceil(8) as usize
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
        if self == Dtype::F64 || self == Dtype::F32 { 128 / self.0.bits() as usize } else { 1 }
    }

    /// The type arithmetic is carried out in, α(τ): the type itself for
    /// `f64` and `f32`, and `f32` for a narrower format `f32` holds, so
    /// `bf16` computes in `f32`.
    pub fn arith(self) -> Dtype {
        if self != Dtype::F64 && presk_format::F32.contains(self.0) { Dtype::F32 } else { self }
    }

    /// The unit roundoff `u` of this type: half an ulp of 1, and the constant
    /// every first-order error rule is written in terms of (step 4). A type
    /// of `p` bits of significand has `u = 2^-p`: 2^-53, 2^-24, and 2^-8 for
    /// `bf16`'s 8 (7 stored and the implicit one). It was 2^-9 until 0.10.0
    /// step 8, every `bf16` rounding counted at half its size (the language
    /// reference's chapter 4).
    pub fn unit_roundoff(self) -> f64 {
        self.0.unit_roundoff()
    }

    /// What a native backend's match over `f64`, `f32` and `bf16` does with
    /// any other type: never reached, since a native backend refuses a
    /// realization that names one before anything runs
    /// ([`Realization::check_native`]).
    pub fn unreachable(self) -> ! {
        unreachable!("{self}: a native backend computes in f64, f32 and bf16 only (checked by `Realization::check_native`)")
    }
}

/// The format a value of the type is: what a realization's `arith`,
/// `product` and `accum` name.
impl From<Dtype> for presk_format::Format {
    fn from(d: Dtype) -> presk_format::Format {
        presk_format::Format::Float(d.format())
    }
}

/// The variants' names of the enum this type was until 0.18.0, which keys
/// built from `{:?}` still read.
impl std::fmt::Debug for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Dtype::F64 => f.write_str("F64"),
            Dtype::F32 => f.write_str("F32"),
            Dtype::BF16 => f.write_str("Bf16"),
            Dtype(other) => write!(f, "Dtype({other})"),
        }
    }
}

/// The most precise first: `f64` < `f32` < `bf16`, the order of the enum.
impl PartialOrd for Dtype {
    fn partial_cmp(&self, other: &Dtype) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Dtype {
    fn cmp(&self, other: &Dtype) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl std::fmt::Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl serde::Serialize for Dtype {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.serialize_str(self.name())
    }
}

/// By name, as [`Dtype::of_name`] reads it: any float format.
impl<'de> serde::Deserialize<'de> for Dtype {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Dtype, D::Error> {
        <presk_format::Float as serde::Deserialize>::deserialize(d).map(Dtype)
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
    /// Sums of `size` consecutive terms each, from zero and in index order,
    /// the chunks' sums then added in order from zero, the last chunk
    /// shorter when `size` does not divide the terms (plan 0.18.0 step 3,
    /// D206): a systolic array's tiles along the summed axis. It
    /// reassociates, as `Interleaved` does.
    Chunked { size: u32 },
}

impl ReductionOrder {
    /// Whether this order preserves the exact order of additions, and so is
    /// bit-identical to `Sequential` at the same dtype.
    pub fn is_exact(self) -> bool {
        !matches!(self, ReductionOrder::Interleaved { .. } | ReductionOrder::Chunked { .. })
    }

    pub fn name(self) -> String {
        match self {
            ReductionOrder::Sequential => "sequential".into(),
            ReductionOrder::Blocked { width } => format!("blocked({width})"),
            ReductionOrder::Interleaved { partials } => format!("interleaved({partials})"),
            ReductionOrder::Chunked { size } => format!("chunked({size})"),
        }
    }
}

/// A site's value stored in an affine format (plan 0.18.0 step 4, D207): a
/// quantized activation, its integer codes and its static scale and zero
/// point. The realization records the format itself, its numbers, which is
/// what every executor reads; or the scheme they are to be computed by from
/// an envelope's magnitudes at the site, which `presk scales` turns into the
/// format and every executor refuses until then (`E7006`).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum Scale {
    Static(presk_format::Affine),
    Scheme(ScaleScheme),
}

/// How a site's scale and zero point are computed from the envelope.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScaleScheme {
    /// The codes' integer format, which saturates: `i8/sat`, `u8/sat`.
    pub int: String,
    pub scheme: ScaleRule,
    /// The rescaling's parameters, as an affine format's spelling writes
    /// them (`away`, `x/s+zero`, `f32`); its defaults where absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tie: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rounds: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quotient: Option<String>,
}

/// The rule a static scale is computed by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScaleRule {
    /// Symmetric, `s = max|x| / qmax` ([`presk_format::Affine::absmax`]).
    Absmax,
    /// Symmetric, a power of two ([`presk_format::Affine::pow2`]).
    Pow2,
    /// Asymmetric, from the smallest and the largest value, with a zero
    /// point ([`presk_format::Affine::minmax`]): an envelope measured with
    /// `presk calibrate --range`.
    Minmax,
}

impl ScaleRule {
    pub fn name(self) -> &'static str {
        match self {
            ScaleRule::Absmax => "absmax",
            ScaleRule::Pow2 => "pow2",
            ScaleRule::Minmax => "minmax",
        }
    }
}

impl ScaleScheme {
    /// The format the scheme computes from a site's largest `|x|` and, for
    /// `minmax`, its smallest and largest value.
    pub fn resolve(&self, max_abs: f64, range: Option<(f64, f64)>) -> std::result::Result<presk_format::Affine, String> {
        let mut spelling = format!("affine({},scale=1", self.int);
        for (key, value) in [("tie", &self.tie), ("rounds", &self.rounds), ("quotient", &self.quotient)] {
            if let Some(v) = value {
                spelling.push_str(&format!(",{key}={v}"));
            }
        }
        spelling.push(')');
        let t = presk_format::Affine::parse(&spelling)?;
        let a = match (self.scheme, range) {
            (ScaleRule::Absmax, _) => presk_format::Affine::absmax(t.int, max_abs),
            (ScaleRule::Pow2, _) => presk_format::Affine::pow2(t.int, max_abs),
            (ScaleRule::Minmax, Some((lo, hi))) => presk_format::Affine::minmax(t.int, lo, hi),
            (ScaleRule::Minmax, None) => return Err("`minmax` needs the site's smallest and largest value, which an envelope measured with `presk calibrate --range` records".into()),
        }?;
        Ok(presk_format::Affine { tie: t.tie, zero_rounded: t.zero_rounded, quotient: t.quotient, ..a })
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
    /// The format a site's operations are rounded to, its *arithmetic
    /// format*, by site (plan 0.18.0 step 3, D206): every elementary
    /// operation of the nest, and every operand it loads, converted to it.
    /// A site the map does not mention computes in its storage type's
    /// arithmetic type, `Dtype::arith` ([`Realization::arith_at`]).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub arith: BTreeMap<String, presk_format::Format>,
    /// The format each product of a sum's terms is rounded to, by site
    /// (D206); the site's arithmetic format where it is not mentioned.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub product: BTreeMap<String, presk_format::Format>,
    /// The format each addition of a sum is rounded to, its *accumulation
    /// format*, by site; the site's arithmetic format where it is not
    /// mentioned. A format since D206, an integer and a fixed-point one
    /// included.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub accum: BTreeMap<String, presk_format::Format>,
    /// Values stored in an affine format, by site (plan 0.18.0 step 4,
    /// D207): a quantized activation. Its nest rounds its result into the
    /// format by its rescaling, and every nest reads it as it is,
    /// `(q − z)·s`, never converted to its own storage type; the nest
    /// computes in its storage type's arithmetic type unless `arith` says
    /// otherwise. Where a nest's product format is an integer or a
    /// fixed-point one, it reads such a value, and an integer weight, as its
    /// integer counted in its scale (the kernel interpreter's counts,
    /// `presk_kernel::exec::Mixed`).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub scaled: BTreeMap<String, Scale>,
    pub order: ReductionOrder,
    /// The magnitude envelope any bound about this realization assumed
    /// (step 3). A bound without one is a different and unsupported claim
    /// (D35), so the reference travels with the realization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub envelope: Option<String>,
    /// Parameters stored at rest in a floating-point type of their own,
    /// `f32` or `bf16`, keyed by their path like `quant` (plan 0.12.0, D119;
    /// the reference's chapter 4, W18): the leaf is rounded once to it when
    /// the entry is given it, and every nest reads it from that type. A
    /// path here is not in `quant` ([`Realization::check_params`]).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub params: BTreeMap<String, Dtype>,
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
        Realization { default: d, storage: BTreeMap::new(), functions: BTreeMap::new(), arith: BTreeMap::new(), product: BTreeMap::new(), accum: BTreeMap::new(), scaled: BTreeMap::new(), order: ReductionOrder::Sequential, envelope: None, params: BTreeMap::new(), quant: BTreeMap::new() }
    }

    /// `f64`, `f32`, `bf16`, another float format's name (`f16`, every
    /// value stored in it and computed in its arithmetic type), or `@PATH`
    /// for a realization file.
    pub fn parse(spec: &str) -> std::result::Result<Realization, String> {
        if let Some(path) = spec.strip_prefix('@') {
            let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
            return serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"));
        }
        match Dtype::of_name(spec) {
            Ok(d) => Ok(Realization::uniform(d)),
            Err(_) => Err(format!("unknown realization `{spec}`; expected f64, f32, bf16, another float format's name or @FILE")),
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

    /// A site's entry in one of the maps by site, or its base's in the
    /// forward pass a gradient keeps (`fwd_base`) when the site has no type
    /// of its own.
    fn by_site<V: Clone>(&self, map: &BTreeMap<String, V>, site: &str) -> Option<V> {
        if let Some(f) = map.get(site) {
            return Some(f.clone());
        }
        match fwd_base(site) {
            Some(base) if !self.storage.contains_key(site) && !site.rsplit_once('/').is_some_and(|(f, _)| self.functions.contains_key(f)) => self.by_site(map, &base),
            _ => None,
        }
    }

    /// The arithmetic format of a site: its own (`arith`), else the
    /// *arithmetic* type of its storage, α(σ), so that `bf16` storage
    /// computes in `f32` (plan 0.18.0 step 3, D206).
    pub fn arith_at(&self, site: &str) -> presk_format::Format {
        self.by_site(&self.arith, site).unwrap_or(presk_format::Format::Float(self.dtype_at(site).arith().format()))
    }

    /// The format a site's products of terms are rounded to: its own
    /// (`product`), else its arithmetic format.
    pub fn product_at(&self, site: &str) -> presk_format::Format {
        self.by_site(&self.product, site).unwrap_or_else(|| self.arith_at(site))
    }

    /// The accumulation format of a site's sums: its own (`accum`), else
    /// its arithmetic format, so that `bf16` storage accumulates in `f32`
    /// unless something says otherwise — the idiom, not a surprise.
    pub fn accum_at(&self, site: &str) -> presk_format::Format {
        self.by_site(&self.accum, site).unwrap_or_else(|| self.arith_at(site))
    }

    /// The affine format a site's value is stored in (plan 0.18.0 step 4,
    /// D207), its own or its base's in the forward pass a gradient keeps;
    /// `None` where it is stored in its storage type. A scheme whose scale
    /// is not computed yet is refused (`E7006`).
    pub fn scaled_at(&self, site: &str) -> Result<Option<presk_format::Affine>> {
        match self.by_site(&self.scaled, site) {
            None => Ok(None),
            Some(Scale::Static(a)) => Ok(Some(a)),
            Some(Scale::Scheme(s)) => Err(unresolved(site, &s)),
        }
    }

    /// The formats a site is stored and computed in, as `presk explain` and
    /// the network diagram label it (plan 0.18.0 step 5, D208): its storage
    /// type's name, or an affine format's by its integer; then, where they
    /// are not its storage type's arithmetic type, the format its
    /// operations are rounded to and its sums' products and additions:
    /// `f16 in f16`, `affine(i8/sat) (i32×, i32+)`. A site that states none
    /// is its type's name, as until then.
    pub fn label_at(&self, site: &str) -> String {
        let d = self.dtype_at(site);
        let mut out = match self.by_site(&self.scaled, site) {
            Some(Scale::Static(a)) => format!("affine({})", a.int),
            Some(Scale::Scheme(s)) => format!("affine({})", s.int),
            None => d.name().to_string(),
        };
        let (alpha, native) = (self.arith_at(site), presk_format::Format::Float(d.arith().format()));
        if alpha != native {
            out.push_str(&format!(" in {alpha}"));
        }
        let (mu, kappa) = (self.product_at(site), self.accum_at(site));
        if mu != alpha || kappa != alpha {
            out.push_str(&format!(" ({mu}×, {kappa}+)"));
        }
        out
    }

    /// Every scale stated as a scheme refused (`E7006`): an executor reads
    /// a format's numbers, which `presk scales` computes from an envelope.
    pub fn check_scales(&self) -> Result<()> {
        match self.scaled.iter().find_map(|(site, s)| if let Scale::Scheme(s) = s { Some((site, s)) } else { None }) {
            Some((site, s)) => Err(unresolved(site, s)),
            None => Ok(()),
        }
    }

    /// Whether every site computes as a native backend does, in the native
    /// types alone and each nest in its storage type's arithmetic type
    /// (plan 0.18.0 step 3, D206): the condition under which the native and
    /// LLVM backends and the C export run a realization. `E7005` otherwise,
    /// naming the first site or member that fails it; the kernel
    /// interpreter emulates any format.
    pub fn check_native(&self, executor: &str) -> Result<()> {
        let fail = |what: String| err("E7005", format!("the realization {what}: {executor} computes in f64, f32 and bf16, each nest in its storage type's arithmetic type; the kernel interpreter emulates any format (`--backend kernel`)"));
        let stored = std::iter::once((String::from("every value"), self.default)).chain(self.storage.iter().map(|(s, d)| (format!("`{s}`"), *d))).chain(self.functions.iter().map(|(f, d)| (format!("the function `{f}`"), *d))).chain(self.params.iter().map(|(p, d)| (format!("the parameter `{p}`"), *d)));
        for (what, d) in stored {
            if !d.is_native() {
                return fail(format!("stores {what} in {d}"));
            }
        }
        for (map, what) in [(&self.arith, "computes"), (&self.product, "rounds the products of"), (&self.accum, "accumulates")] {
            for (site, f) in map {
                let native = presk_format::Format::Float(self.dtype_at(site).arith().format());
                if *f != native {
                    return fail(format!("{what} `{site}` in {f}, where its storage type's arithmetic type is {native}"));
                }
            }
        }
        if let Some((site, s)) = self.scaled.iter().next() {
            return fail(match s {
                Scale::Static(a) => format!("stores `{site}` in {a}"),
                Scale::Scheme(s) => format!("stores `{site}` in an affine format of {} ({})", s.int, s.scheme.name()),
            });
        }
        Ok(())
    }

    /// Whether some site computes otherwise than a native backend does
    /// ([`Realization::check_native`] fails): what the kernel interpreter
    /// emulates, and counts.
    pub fn emulates(&self) -> bool {
        self.check_native("").is_err()
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
        // A type stored apart from its arithmetic, `bf16` or a format of
        // plan 0.18.0 step 3, or a site computing apart from its storage.
        let narrow = std::iter::once(&self.default).chain(self.storage.values()).chain(self.functions.values()).any(|d| d.arith() != *d);
        let apart = self.arith.iter().any(|(site, f)| *f != presk_format::Format::Float(self.dtype_at(site).format()));
        narrow || apart || !self.scaled.is_empty() || self.storage.iter().any(|(site, d)| {
            let mut without = self.clone();
            without.storage.remove(site);
            without.dtype_at(site) != *d
        })
    }

    /// Whether every site of this realization has one type and additions keep
    /// their order. Uniform realizations are the ones earlier versions could
    /// express, and the ones whose goldens must not move.
    pub fn is_uniform(&self) -> bool {
        self.storage.is_empty() && self.functions.is_empty() && self.arith.is_empty() && self.product.is_empty() && self.accum.is_empty() && self.scaled.is_empty() && self.order == ReductionOrder::Sequential && self.params.is_empty() && self.quant.is_empty()
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
        for (map, what) in [(&self.arith, "arith"), (&self.product, "product"), (&self.accum, "accum")] {
            if !map.is_empty() {
                s.push_str(&format!("+{}{what}", map.len()));
            }
        }
        if !self.scaled.is_empty() {
            s.push_str(&format!("+{}scaled", self.scaled.len()));
        }
        if !self.params.is_empty() {
            s.push_str(&format!("+{}params", self.params.len()));
        }
        if !self.quant.is_empty() {
            s.push_str(&format!("+{}quant", self.quant.len()));
        }
        s
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a realization serializes")
    }

    /// The parameters as this realization stores them (the reference's
    /// chapter 4, realized value, item 1): `theta` with every leaf `params`
    /// types converted once to its type, from the value given (an exact
    /// value through `f64`, as every real enters a realization), and every
    /// leaf `quant` names replaced by its integers and scales
    /// ([`Buf::Quant`]), which the kernels load as `f32(q)·s` and the
    /// reference interpreter reads as the same values. Every other leaf as
    /// given; the identity when the record types and quantises nothing.
    pub fn stored_params(&self, theta: &Value) -> Result<Value> {
        self.check_params(Some(theta))?;
        let typed = if self.params.is_empty() { theta.clone() } else { quant::typed_leaves(theta, theta, &self.params, self.emulates()) };
        if self.quant.is_empty() {
            return Ok(typed);
        }
        Ok(quant::quantized_leaves(&typed, &quant::quantize_params(theta, &self.quant)?))
    }

    /// The conditions on `params` (the reference's chapter 4, realized
    /// types, item 3): each type a float format narrower than `f64` (`f32`
    /// or `bf16` until plan 0.18.0 step 5, D208; the native backends refuse
    /// any other, `E7005`), no path also in `quant`, and, given the entry's
    /// parameter tree, each path a real leaf of it. A misspelt path quietly
    /// typing nothing would be the worst failure this could have, as for
    /// `quant`.
    pub fn check_params(&self, theta: Option<&Value>) -> Result<()> {
        for (path, d) in &self.params {
            if *d == Dtype::F64 {
                return err("E7001", format!("the realization stores the parameter `{path}` in `{d}`: a parameter's own type is a float format narrower than `f64`, and every other parameter is stored in the default type"));
            }
            if self.quant.contains_key(path) {
                return err("E7001", format!("the realization both stores the parameter `{path}` in `{d}` and quantises it: a parameter is typed or quantised, not both"));
            }
        }
        if let Some(theta) = theta {
            let paths = quant::param_paths(theta);
            if let Some(path) = self.params.keys().find(|p| !paths.contains(p)) {
                return err("E7001", format!("the realization stores `{path}` in its own type, which is not a parameter of the model (it has {})", paths.join(", ")));
            }
        }
        Ok(())
    }

    /// Store an entry's parameters, its first argument, as this realization
    /// says before a run ([`Realization::stored_params`]). A realization
    /// that types or quantises parameters must never be run on the true
    /// weights unnoticed, so an entry without a parameter tree first is
    /// refused.
    pub fn store_params(&self, args: &mut [Value]) -> Result<()> {
        if self.quant.is_empty() && self.params.is_empty() {
            return Ok(());
        }
        match args.first_mut() {
            Some(theta @ Value::Record(_)) => {
                *theta = self.stored_params(theta)?;
                Ok(())
            }
            _ if self.quant.is_empty() => err("E7001", "the realization stores parameters in their own types, and this entry's first argument is not a parameter tree"),
            _ => err("E7001", "the realization quantises parameters, and this entry's first argument is not a parameter tree"),
        }
    }

    /// An entry's arguments as this realization realizes them (item 1):
    /// every real leaf converted to the default type, but the parameters
    /// `params` types converted to theirs from the values given, not from
    /// their default-typed copies (under a `bf16` default an `f32` parameter
    /// keeps its bits), and those `quant` names quantised as
    /// [`Realization::store_params`] quantises them.
    pub fn realize_args(&self, args: &[Value]) -> Result<Vec<Value>> {
        let mut out: Vec<Value> = args.iter().map(|v| v.realize(self.default)).collect();
        if !self.params.is_empty()
            && let (Some(theta @ Value::Record(_)), Some(slot)) = (args.first(), out.first_mut())
        {
            self.check_params(Some(theta))?;
            *slot = quant::typed_leaves(slot, theta, &self.params, self.emulates());
        }
        self.store_params(&mut out)?;
        Ok(out)
    }

    /// The type an entry's real leaf is stored in: its own for a parameter
    /// `params` types (in argument `arg` 0, the parameter tree, at `path`),
    /// else the default.
    pub fn leaf_type(&self, arg: usize, path: &str) -> Dtype {
        match self.params.get(path) {
            Some(d) if arg == 0 => *d,
            _ => self.default,
        }
    }
}

/// The data of a tensor. A machine buffer's elements are shared by its
/// clones ([`Elems`], plan 0.10.0 step 6), and so are exact ones, the
/// interpreter's own buffer ([`presk_interp::Data`], plan 0.10.1 step 2): a
/// value read, passed or put in a tuple is not copied, nor one crossing the
/// seam.
#[derive(Clone, Debug, PartialEq)]
pub enum Buf {
    /// Exact reals of the reference interpreter (MPFR at its precision).
    Exact(presk_interp::Data<Float>),
    F64(Elems<f64>),
    F32(Elems<f32>),
    /// Storage-only sixteen-bit floats, held as their raw bits. Arithmetic on
    /// them happens in `f32`; only the traffic is halved.
    Bf16(Elems<u16>),
    I64(Elems<i64>),
    /// Integers with a scale per channel (0.6.0 step 5b, D44): a parameter a
    /// realization quantises. Read anywhere as `f32(q)·s`.
    Quant(std::sync::Arc<QBuf>),
    /// Values of a format no backend computes in natively, each held exactly
    /// as an `f64` ([`presk_format::Format::exact_in_f64`]): what the kernel
    /// interpreter's emulation reads and writes (plan 0.18.0 step 2, D205).
    /// No realization stores a value in such a format yet, so no other
    /// backend meets one ([`Buf::unemulated`]).
    Format(presk_format::Format, Elems<f64>),
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
            Buf::Format(_, v) => v.len(),
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
            Buf::Format(_, v) => v[lo..hi].to_vec(),
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
            Buf::Format(_, v) => v.to_vec(),
        }
    }

    /// A buffer of values of the format `f` ([`Buf::Format`]): each of `v`
    /// as it is when it is a number or a special value of a float format,
    /// −0 included where a conversion never gives it (the code a file or a
    /// machine's memory holds), else rounded once to it, ties to even.
    pub fn in_format(f: presk_format::Format, v: &[f64]) -> Buf {
        assert!(f.exact_in_f64(), "a buffer of {f} holds values no f64 holds");
        let held = |x: f64| matches!(f, presk_format::Format::Float(g) if g.holds(x));
        Buf::Format(f, v.iter().map(|x| if held(*x) { *x } else { f.round_noted(*x, presk_format::Tie::Even).0 }).collect())
    }

    /// What a backend that computes only in `f64`, `f32` and `bf16` does
    /// with a [`Buf::Format`]: nothing, since such a backend refuses a
    /// realization that stores a value in another format before anything
    /// runs ([`Realization::check_native`], plan 0.18.0 step 3).
    pub fn unemulated(&self) -> ! {
        unreachable!("a buffer of {} reached a backend that does not compute in it: only the kernel interpreter emulates it", self.dtype())
    }

    /// A buffer of `n` reals in the machine type `d`, rounded once from
    /// `f64`; `bf16` through `f32`, twice (D38). A type other than the
    /// native ones is a [`Buf::Format`], each value rounded once, ties to
    /// even (plan 0.18.0 step 3, D206): [`presk_format::Float::store`].
    pub fn of(d: Dtype, v: &[f64]) -> Buf {
        match d {
            Dtype::F64 => Buf::F64(v.to_vec().into()),
            Dtype::F32 => Buf::F32(v.iter().map(|x| *x as f32).collect()),
            Dtype::BF16 => Buf::Bf16(v.iter().map(|x| f32_to_bf16(*x as f32)).collect()),
            d => Buf::Format(presk_format::Format::Float(d.format()), v.iter().map(|x| d.format().store(*x)).collect()),
        }
    }

    /// [`Buf::of`], counting in `c` what the type's overflow rule produced
    /// (plan 0.18.0 step 6, D209): a conversion the kernel interpreter makes
    /// outside a kernel when it emulates, an argument stored in its type or
    /// an operand at a nest's edge, counted as a kernel counts its own.
    pub fn of_counted(d: Dtype, v: &[f64], c: &mut presk_format::Counts) -> Buf {
        let b = Buf::of(d, v);
        let f = presk_format::Format::Float(d.format());
        for (x, y) in v.iter().zip(b.to_f64()) {
            let (_, o) = f.round_noted(*x, presk_format::Tie::Even);
            c.note(f, y, o, x.is_infinite(), x.is_nan());
        }
        b
    }

    /// Zeros of the machine type `d`.
    pub fn zeros(d: Dtype, n: usize) -> Buf {
        match d {
            Dtype::F64 => Buf::F64(vec![0.0; n].into()),
            Dtype::F32 => Buf::F32(vec![0.0; n].into()),
            Dtype::BF16 => Buf::Bf16(vec![0u16; n].into()),
            d => Buf::Format(presk_format::Format::Float(d.format()), vec![0.0; n].into()),
        }
    }

    pub fn dtype(&self) -> std::borrow::Cow<'static, str> {
        match self {
            Buf::Exact(_) => "exact".into(),
            Buf::F64(_) => "f64".into(),
            Buf::F32(_) => "f32".into(),
            Buf::Bf16(_) => "bf16".into(),
            Buf::I64(_) => "i64".into(),
            Buf::Quant(q) => q.dtype().into(),
            Buf::Format(f, _) => f.name(),
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

/// The refusal of a scale stated as a scheme (`E7006`).
fn unresolved(site: &str, s: &ScaleScheme) -> RtError {
    RtError { code: "E7006", message: format!("the realization stores `{site}` in {} with a scale computed by `{}`, and states no scale: `presk scales --realize @FILE --precision ENVELOPE.json` computes it from an envelope", s.int, s.scheme.name()), span: None }
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
    /// The interpreter's value, exactly (no rounding): its exact buffers
    /// shared, not copied (plan 0.10.1 step 2).
    pub fn from_interp(v: &IValue) -> Result<Value> {
        Ok(match v {
            IValue::Real(t) => Value::Real(Tensor { axes: t.axes.clone(), data: Buf::Exact(t.data.clone()) }),
            IValue::Idx { bound, t } => Value::Idx { bound: *bound, t: Tensor { axes: t.axes.clone(), data: Buf::I64(t.data.to_vec().into()) } },
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

    /// The interpreter's value at `prec` bits: exact buffers as they are
    /// (shared when every element is already at `prec` bits, plan 0.10.1
    /// step 2), machine buffers injected exactly.
    pub fn to_interp(&self, prec: u32) -> IValue {
        match self {
            Value::Real(t) => IValue::Real(presk_interp::Tensor {
                axes: t.axes.clone(),
                data: match &t.data {
                    Buf::Exact(v) if v.iter().all(|x| x.prec() == prec) => v.clone(),
                    Buf::Exact(v) => v.iter().map(|x| Float::with_val(prec, x)).collect(),
                    Buf::F64(v) => v.iter().map(|x| Float::with_val(prec, *x)).collect(),
                    Buf::F32(v) => v.iter().map(|x| Float::with_val(prec, *x)).collect(),
                    Buf::Bf16(v) => v.iter().map(|x| Float::with_val(prec, bf16_to_f32(*x))).collect(),
                    Buf::I64(v) => v.iter().map(|x| Float::with_val(prec, *x)).collect(),
                    Buf::Quant(q) => (0..q.len()).map(|e| Float::with_val(prec, q.value(e))).collect(),
                    Buf::Format(_, v) => v.iter().map(|x| Float::with_val(prec, *x)).collect(),
                },
            }),
            Value::Idx { bound, t } => IValue::Idx {
                bound: *bound,
                t: presk_interp::Tensor {
                    axes: t.axes.clone(),
                    data: match &t.data {
                        Buf::I64(v) => v.to_vec().into(),
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

    /// The interpreter's value in binary64 ([`F64InterpBackend`]): every
    /// real rounded once to `f64`.
    pub fn to_interp_f64(&self) -> presk_interp::Value<f64> {
        use presk_interp::Value as V;
        match self {
            Value::Real(t) => V::Real(presk_interp::Tensor {
                axes: t.axes.clone(),
                data: match &t.data {
                    Buf::F64(v) => v.iter().copied().collect(),
                    other => other.to_f64().into_iter().collect(),
                },
            }),
            Value::Idx { bound, t } => V::Idx {
                bound: *bound,
                t: presk_interp::Tensor {
                    axes: t.axes.clone(),
                    data: match &t.data {
                        Buf::I64(v) => v.to_vec().into(),
                        other => other.to_f64().into_iter().map(|x| x as i64).collect(),
                    },
                },
            },
            Value::Nat(n) => V::Nat(*n),
            Value::Int(n) => V::Int(*n),
            Value::F64(x) => V::F64(*x),
            Value::Bool(b) => V::Bool(*b),
            Value::Key(k) => V::Key(k.clone()),
            Value::Tuple(vs) => V::tuple(vs.iter().map(Value::to_interp_f64).collect()),
            Value::Record(fs) => V::record(fs.iter().map(|(n, v)| (n.clone(), v.to_interp_f64())).collect()),
            Value::Stacked(vs) => V::Stacked(std::sync::Arc::new(vs.iter().map(Value::to_interp_f64).collect())),
        }
    }

    /// A value of [`F64InterpBackend`], its reals as `f64` buffers.
    pub fn from_interp_f64(v: &presk_interp::Value<f64>) -> Result<Value> {
        use presk_interp::Value as V;
        Ok(match v {
            V::Real(t) => Value::Real(Tensor { axes: t.axes.clone(), data: Buf::F64(t.data.iter().copied().collect()) }),
            V::Idx { bound, t } => Value::Idx { bound: *bound, t: Tensor { axes: t.axes.clone(), data: Buf::I64(t.data.to_vec().into()) } },
            V::Nat(n) => Value::Nat(*n),
            V::Int(n) => Value::Int(*n),
            V::F64(x) => Value::F64(*x),
            V::Bool(b) => Value::Bool(*b),
            V::Key(k) => Value::Key(k.clone()),
            V::Tuple(vs) => Value::Tuple(vs.iter().map(Value::from_interp_f64).collect::<Result<_>>()?),
            V::Record(fs) => Value::Record(fs.iter().map(|(n, v)| Ok((n.clone(), Value::from_interp_f64(v)?))).collect::<Result<_>>()?),
            V::Stacked(vs) => Value::Stacked(vs.iter().map(Value::from_interp_f64).collect::<Result<_>>()?),
            V::Closure { .. } => return err("E7001", "a function value cannot cross the backend seam"),
        })
    }

    /// The same, consuming the value: an exact buffer is moved rather than
    /// copied, each real rounded in place to `prec` bits — the value
    /// `with_val` gives, and no change at the precision it was made at. A
    /// parameter tree then crosses the seam into the interpreter without a
    /// second copy (0.6.0 step 8: BERT-large's 335M exact parameters). A
    /// buffer already at `prec` bits is not written, so that one another
    /// value shares stays shared (plan 0.10.1 step 2).
    pub fn into_interp(self, prec: u32) -> IValue {
        match self {
            Value::Real(Tensor { axes, data: Buf::Exact(mut v) }) => {
                if !v.iter().all(|x| x.prec() == prec) {
                    for x in v.iter_mut() {
                        x.set_prec(prec);
                    }
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

    /// [`Value::map_leaves`] with each leaf's path in the tree, as
    /// [`quant::param_paths`] names it: record fields joined by `.`, tuple
    /// components by their index.
    pub fn map_leaves_at(&self, f: &mut dyn FnMut(&str, &Value) -> Value) -> Value {
        fn go(v: &Value, path: &str, f: &mut dyn FnMut(&str, &Value) -> Value) -> Value {
            let join = |n: &str| if path.is_empty() { n.to_string() } else { format!("{path}.{n}") };
            match v {
                Value::Tuple(vs) => Value::Tuple(vs.iter().enumerate().map(|(i, x)| go(x, &join(&i.to_string()), f)).collect()),
                Value::Record(fs) => Value::Record(fs.iter().map(|(n, x)| (n.clone(), go(x, &join(n), f))).collect()),
                leaf => f(path, leaf),
            }
        }
        go(self, "", f)
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
                Buf::Exact(v) => v.to_vec(),
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
    /// Kernels compiled so far, and the seconds their lowering and their
    /// compiling took (plan 0.17.2 step 4, D200): what `presk serve` counts
    /// as compiling. `(0, 0.0)` for a backend that compiles nothing.
    fn compile_stats(&self) -> (u64, f64) {
        (0, 0.0)
    }
    /// What the formats a realization emulates produced so far besides
    /// rounding (plan 0.18.0 step 3, D206): the kernel interpreter's counts
    /// under a realization the native backends would refuse; `None` for a
    /// backend or a realization that emulates nothing.
    fn counts(&self) -> Option<presk_format::Counts> {
        None
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

/// The reference interpreter's evaluator in binary64 behind the seam (plan
/// 0.17.1 §4, D185): `presk solve --sweep f64` ranks the sites with it where
/// MPFR does not fit in memory, a real taking 8 bytes and not about 80. Each
/// real crossing the seam is rounded once to `f64`; a machine buffer's
/// numbers cross as they are.
pub struct F64InterpBackend<'p> {
    pub interp: Interp<'p, f64>,
}

impl<'p> F64InterpBackend<'p> {
    pub fn new(prog: &'p Program, elab: &'p ElabProgram) -> Self {
        F64InterpBackend { interp: Interp::new(prog, elab, 53) }
    }
}

impl Backend for F64InterpBackend<'_> {
    fn name(&self) -> &'static str {
        "interp-f64"
    }

    fn set_training(&mut self, on: bool) {
        self.interp.training = on;
    }

    fn prepare(&mut self, entry: FuncId, dims: &Dims, realize: &Realization) -> Result<Prepared> {
        presk_interp::defined::check(self.interp.prog, self.interp.elab, entry, dims, None)?;
        Ok(Prepared { func: entry, dims: dims.clone(), realize: realize.clone(), handle: 0 })
    }

    fn call(&mut self, p: &Prepared, args: Vec<Value>) -> Result<Value> {
        let args: Vec<presk_interp::Value<f64>> = args.iter().map(Value::to_interp_f64).collect();
        let out = self.interp.call(p.func, p.dims.clone(), args)?;
        Value::from_interp_f64(&out)
    }
}

/// `presk run` over a backend: the interpreter synthesizes the inputs, the
/// backend runs the entry; the output is the interpreter's `RunOutput` (the
/// result at `prec` bits, exact for the interpreter, rounded once otherwise).
pub fn run_with(backend: &mut dyn Backend, prog: &Program, elab: &ElabProgram, opts: &RunOptions, realize: &Realization) -> Result<presk_interp::RunOutput> {
    let (func, dims, inputs) = presk_interp::run::prepare_run(prog, elab, opts)?;
    let p = backend.prepare(func, &dims, realize)?;
    let mut args: Vec<Value> = inputs.iter().map(|(_, v)| Value::from_interp(v)).collect::<Result<_>>()?;
    realize.store_params(&mut args)?;
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
/// Only the template's shape is read, and the file's numbers are moved into
/// the tree, not copied (0.17.0 step 7, D180: Llama 3.2 1B's parameters are
/// 9.9 GB in `f64`, and each copy was one more).
pub fn load_params(path: &Path, template: &Value) -> Result<Value> {
    let mut r = Reader::open(path)?.with_layout(params_layout(template)?)?;
    if r.is_empty() {
        return err("E9003", format!("{}: no sample", path.display()));
    }
    let d = r.get(0)?;
    params_from_owned(template, d)
}

/// The layout of a parameter tree, [`params_to_data`]'s without the values.
fn params_layout(v: &Value) -> Result<DataLayout> {
    match v {
        Value::Record(fs) => Ok(DataLayout::Record(fs.iter().map(|(n, sub)| Ok((n.clone(), params_layout(sub)?))).collect::<Result<_>>()?)),
        Value::Tuple(ts) => Ok(DataLayout::Tuple(ts.iter().map(params_layout).collect::<Result<_>>()?)),
        Value::Real(t) => Ok(DataLayout::Leaf(Leaf::Real { shape: t.axes.iter().map(|a| a.len).collect() })),
        other => err("E9005", format!("a parameter tree holds no {}", other.kind())),
    }
}

/// [`params_from_data`] taking the sample, whose numbers become the tree's.
fn params_from_owned(template: &Value, d: presk_data::Value) -> Result<Value> {
    match (template, d) {
        (Value::Record(fs), presk_data::Value::Record(ds)) if fs.len() == ds.len() => Ok(Value::Record(fs.iter().zip(ds).map(|((n, t), (_, x))| Ok((n.clone(), params_from_owned(t, x)?))).collect::<Result<_>>()?)),
        (Value::Tuple(ts), presk_data::Value::Tuple(ds)) if ts.len() == ds.len() => Ok(Value::Tuple(ts.iter().zip(ds).map(|(t, x)| params_from_owned(t, x)).collect::<Result<_>>()?)),
        (Value::Real(t), presk_data::Value::Real(xs)) if t.numel() == xs.len() => Ok(Value::Real(Tensor { axes: t.axes.clone(), data: Buf::F64(xs.into()) })),
        _ => err("E9005", "the checkpoint does not have the structure of the model's parameters"),
    }
}
