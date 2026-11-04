//! The `.psk.d` data container and the training data pipeline.
//!
//! A `.psk.d` file is a sequence of *samples* of one Presk type:
//!
//! ```text
//! magic     "PSKD"                       4 bytes
//! version   u16 = 1                      little-endian
//! type_len  u32, then the type, UTF-8    the sample type in Presk syntax
//! count     u64                          number of samples
//! data      the samples, back to back: the leaves of the type in order,
//!           idx[N] as the smallest of u8/u16/u32/u64 holding N−1,
//!           ℝ as f64 little-endian
//! ```
//!
//! Version 2 (plan 0.17.0 step 7, D180) adds, after the count, one byte per
//! leaf in storage order, the leaf's *encoding*:
//!
//! ```text
//! leaves    u32                          number of leaves of the type
//! encodings u8 per leaf                  0 f64 (and every idx), 1 f32, 2 bf16
//! ```
//!
//! and stores each real leaf in its encoding, little-endian (`bf16` as the
//! upper sixteen bits of an `f32`). Every number of `f32` and of `bf16` is
//! a number of `f64`, so a leaf is read exactly whatever its encoding. A
//! writer writes version 1 unless a leaf is narrower, so that a file of
//! `f64` leaves is the same bytes as before.
//!
//! Real values are `f64` and are injected exactly into ℝ when read (the
//! hyperparameter rule of the language). The type is parsed with the Presk
//! parser; a *closed* type (every dimension a literal) gives the leaf layout
//! directly. A `params M` type needs the program to know its layout: the
//! reader then returns the header only, and the caller supplies the
//! [`Layout`] with [`Reader::with_layout`].
//!
//! Errors carry a code: `E9001` (type not closed or not a data type),
//! `E9002` (not a `.psk.d` file), `E9003` (truncated), `E9004` (a value out
//! of its bound or unparsable), `E9005` (type mismatch).

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use presk_syntax::ast::{self, AxisSpec, Expr, Lit};
use presk_syntax::{SourceMap, parse_type};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

pub const MAGIC: &[u8; 4] = b"PSKD";
/// The version a file of `f64` leaves is written in.
pub const VERSION: u16 = 1;
/// The version of a file with a leaf narrower than `f64`.
pub const VERSION_ENCODED: u16 = 2;

/// How a real leaf's elements are stored (version 2). Each is exactly a
/// number of `f64`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Enc {
    F64,
    F32,
    Bf16,
}

impl Enc {
    pub fn code(self) -> u8 {
        match self {
            Enc::F64 => 0,
            Enc::F32 => 1,
            Enc::Bf16 => 2,
        }
    }

    pub fn of_code(c: u8) -> Option<Enc> {
        Some(match c {
            0 => Enc::F64,
            1 => Enc::F32,
            2 => Enc::Bf16,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Enc::F64 => "f64",
            Enc::F32 => "f32",
            Enc::Bf16 => "bf16",
        }
    }

    /// Bytes of one real element.
    pub fn width(self) -> usize {
        match self {
            Enc::F64 => 8,
            Enc::F32 => 4,
            Enc::Bf16 => 2,
        }
    }

    /// The bytes of `x`, which must be a number of the encoding exactly.
    fn encode(self, x: f64, out: &mut Vec<u8>) -> Result<()> {
        match self {
            Enc::F64 => out.extend_from_slice(&x.to_le_bytes()),
            Enc::F32 => {
                let y = x as f32;
                if f64::from(y).to_bits() != x.to_bits() && !(x.is_nan() && y.is_nan()) {
                    return err("E9004", format!("{x} is not a number of f32"));
                }
                out.extend_from_slice(&y.to_le_bytes());
            }
            Enc::Bf16 => {
                let y = x as f32;
                let bits = y.to_bits();
                if f64::from(y).to_bits() != x.to_bits() && !(x.is_nan() && y.is_nan()) || bits & 0xffff != 0 {
                    return err("E9004", format!("{x} is not a number of bf16"));
                }
                out.extend_from_slice(&((bits >> 16) as u16).to_le_bytes());
            }
        }
        Ok(())
    }

    /// The `f64` that the bytes at `b` hold.
    fn decode(self, b: &[u8]) -> f64 {
        match self {
            Enc::F64 => f64::from_le_bytes(b[..8].try_into().unwrap()),
            Enc::F32 => f64::from(f32::from_le_bytes(b[..4].try_into().unwrap())),
            Enc::Bf16 => f64::from(f32::from_bits(u32::from(u16::from_le_bytes(b[..2].try_into().unwrap())) << 16)),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DataError {
    pub code: &'static str,
    pub message: String,
}

impl std::fmt::Display for DataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

impl std::error::Error for DataError {}

pub type Result<T> = std::result::Result<T, DataError>;

fn err<T>(code: &'static str, message: impl Into<String>) -> Result<T> {
    Err(DataError { code, message: message.into() })
}

fn io(e: std::io::Error) -> DataError {
    DataError { code: "E9003", message: e.to_string() }
}

// ----- layout ----------------------------------------------------------

/// One tensor leaf of a sample type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Leaf {
    /// `idx[bound][shape]`
    Idx { bound: u64, shape: Vec<usize> },
    /// `ℝ[shape]`
    Real { shape: Vec<usize> },
}

impl Leaf {
    pub fn numel(&self) -> usize {
        match self {
            Leaf::Idx { shape, .. } | Leaf::Real { shape } => shape.iter().product(),
        }
    }

    /// Bytes of one element on disk.
    pub fn width(&self) -> usize {
        match self {
            Leaf::Idx { bound, .. } => idx_width(*bound),
            Leaf::Real { .. } => 8,
        }
    }

    pub fn bytes(&self) -> usize {
        self.numel() * self.width()
    }
}

/// The width of an `idx[N]` element: the smallest unsigned integer holding `N−1`.
pub fn idx_width(bound: u64) -> usize {
    match bound.saturating_sub(1) {
        n if n <= u8::MAX as u64 => 1,
        n if n <= u16::MAX as u64 => 2,
        n if n <= u32::MAX as u64 => 4,
        _ => 8,
    }
}

/// The leaf structure of a sample type, in the order the leaves are stored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Layout {
    Leaf(Leaf),
    Tuple(Vec<Layout>),
    /// A parameter tree (`params M`): named fields in declaration order.
    Record(Vec<(String, Layout)>),
}

impl Layout {
    /// The layout of a closed data type written in Presk syntax.
    pub fn from_type_text(text: &str) -> Result<Layout> {
        let mut sm = SourceMap::new();
        let id = sm.add("<type>", text);
        let (ty, diags) = parse_type(&sm, id);
        if let Some(d) = diags.iter().find(|d| d.is_error()) {
            return err("E9001", format!("`{text}` is not a type: {}", d.message));
        }
        let Some(ty) = ty else { return err("E9001", format!("`{text}` is not a type")) };
        Self::from_ast(&ty, text)
    }

    fn from_ast(ty: &ast::Ty, text: &str) -> Result<Layout> {
        match ty {
            ast::Ty::Real { axes, .. } => Ok(Layout::Leaf(Leaf::Real { shape: Self::shape(axes, text)? })),
            ast::Ty::Idx { bound, axes, .. } => {
                let bound = Self::nat(bound, text)?;
                if bound == 0 {
                    return err("E9001", format!("`{text}`: an `idx[0]` has no values"));
                }
                Ok(Layout::Leaf(Leaf::Idx { bound, shape: Self::shape(axes, text)? }))
            }
            // A mask is stored as indices below 2, false 0 and true 1 (plan
            // 0.10.0 step 7b, D99).
            ast::Ty::Bool { axes, .. } => Ok(Layout::Leaf(Leaf::Idx { bound: 2, shape: Self::shape(axes, text)? })),
            ast::Ty::Param { inner, .. } => Self::from_ast(inner, text),
            ast::Ty::Tuple { elems, .. } => Ok(Layout::Tuple(elems.iter().map(|t| Self::from_ast(t, text)).collect::<Result<_>>()?)),
            ast::Ty::Params { module, .. } => {
                err("E9001", format!("`{text}`: the layout of `params {}` needs the program (open the file from `presk train`)", module.dotted()))
            }
            ast::Ty::Named { path, .. } => err("E9001", format!("`{text}`: `{}` is not a data type (only `ℝ[…]`, `idx[N][…]`, `bool[…]` and tuples of them, with literal dimensions)", path.dotted())),
        }
    }

    fn shape(axes: &[AxisSpec], text: &str) -> Result<Vec<usize>> {
        axes.iter()
            .map(|a| match a {
                AxisSpec::Axis { dim, .. } => Self::nat(dim, text).map(|n| n as usize),
                AxisSpec::Ellipsis { .. } => err("E9001", format!("`{text}`: a sample type cannot have an open prefix `...`")),
            })
            .collect()
    }

    fn nat(e: &Expr, text: &str) -> Result<u64> {
        match e {
            Expr::Lit { lit: Lit::Nat(n), .. } => n.parse().map_err(|_| DataError { code: "E9001", message: format!("`{text}`: `{n}` is too large") }),
            _ => err("E9001", format!("`{text}`: every dimension of a sample type must be a literal (a `.psk.d` holds concrete tensors)")),
        }
    }

    /// The leaves, in storage order.
    pub fn leaves(&self) -> Vec<&Leaf> {
        let mut out = Vec::new();
        self.collect(&mut out);
        out
    }

    fn collect<'a>(&'a self, out: &mut Vec<&'a Leaf>) {
        match self {
            Layout::Leaf(l) => out.push(l),
            Layout::Tuple(ts) => ts.iter().for_each(|t| t.collect(out)),
            Layout::Record(fs) => fs.iter().for_each(|(_, t)| t.collect(out)),
        }
    }

    /// Bytes of one sample on disk, every real leaf in `f64`.
    pub fn sample_bytes(&self) -> usize {
        self.leaves().iter().map(|l| l.bytes()).sum()
    }

    /// Bytes of one sample on disk, each leaf in its encoding (`encs` in
    /// storage order; empty: every real leaf in `f64`, as version 1 has it).
    pub fn sample_bytes_enc(&self, encs: &[Enc]) -> usize {
        self.leaves().iter().enumerate().map(|(i, l)| leaf_bytes(l, encs.get(i).copied().unwrap_or(Enc::F64))).sum()
    }
}

fn leaf_bytes(l: &Leaf, e: Enc) -> usize {
    match l {
        Leaf::Real { .. } => l.numel() * e.width(),
        Leaf::Idx { .. } => l.bytes(),
    }
}

// ----- values ------------------------------------------------------------

/// One sample: the same tree as its [`Layout`], with flat row-major data per leaf.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Idx(Vec<u64>),
    Real(Vec<f64>),
    Tuple(Vec<Value>),
    Record(Vec<(String, Value)>),
}

impl Value {
    fn leaves(&self) -> Vec<&Value> {
        let mut out = Vec::new();
        self.collect(&mut out);
        out
    }

    fn collect<'a>(&'a self, out: &mut Vec<&'a Value>) {
        match self {
            Value::Idx(_) | Value::Real(_) => out.push(self),
            Value::Tuple(ts) => ts.iter().for_each(|t| t.collect(out)),
            Value::Record(fs) => fs.iter().for_each(|(_, t)| t.collect(out)),
        }
    }

    /// Checks that the value has the layout's structure, counts and bounds.
    pub fn check(&self, layout: &Layout) -> Result<()> {
        let ls = layout.leaves();
        let vs = self.leaves();
        if ls.len() != vs.len() {
            return err("E9005", format!("sample has {} leaves, the type has {}", vs.len(), ls.len()));
        }
        for (i, (l, v)) in ls.iter().zip(&vs).enumerate() {
            match (l, v) {
                (Leaf::Idx { bound, .. }, Value::Idx(d)) => {
                    if d.len() != l.numel() {
                        return err("E9005", format!("leaf {i}: {} values, {} expected", d.len(), l.numel()));
                    }
                    if let Some(x) = d.iter().find(|x| **x >= *bound) {
                        return err("E9004", format!("leaf {i}: index {x} is not below the bound {bound}"));
                    }
                }
                (Leaf::Real { .. }, Value::Real(d)) => {
                    if d.len() != l.numel() {
                        return err("E9005", format!("leaf {i}: {} values, {} expected", d.len(), l.numel()));
                    }
                }
                (Leaf::Idx { .. }, _) => return err("E9005", format!("leaf {i}: indices expected, reals found")),
                (Leaf::Real { .. }, _) => return err("E9005", format!("leaf {i}: reals expected, indices found")),
            }
        }
        Ok(())
    }

    /// Parses one sample from whitespace-separated values, leaves in order.
    pub fn parse_text(layout: &Layout, line: &str) -> Result<Value> {
        let mut toks = line.split_whitespace();
        let v = Self::parse_tokens(layout, &mut toks)?;
        if let Some(extra) = toks.next() {
            return err("E9004", format!("extra value `{extra}` after a complete sample"));
        }
        Ok(v)
    }

    fn parse_tokens<'a>(layout: &Layout, toks: &mut impl Iterator<Item = &'a str>) -> Result<Value> {
        match layout {
            Layout::Leaf(Leaf::Idx { bound, shape }) => {
                let n: usize = shape.iter().product();
                let mut d = Vec::with_capacity(n);
                for _ in 0..n {
                    let t = toks.next().ok_or_else(|| DataError { code: "E9004", message: "sample ends before all values are given".into() })?;
                    let x: u64 = t.parse().map_err(|_| DataError { code: "E9004", message: format!("`{t}` is not an index") })?;
                    if x >= *bound {
                        return err("E9004", format!("index {x} is not below the bound {bound}"));
                    }
                    d.push(x);
                }
                Ok(Value::Idx(d))
            }
            Layout::Leaf(Leaf::Real { shape }) => {
                let n: usize = shape.iter().product();
                let mut d = Vec::with_capacity(n);
                for _ in 0..n {
                    let t = toks.next().ok_or_else(|| DataError { code: "E9004", message: "sample ends before all values are given".into() })?;
                    let x: f64 = t.parse().map_err(|_| DataError { code: "E9004", message: format!("`{t}` is not a real") })?;
                    d.push(x);
                }
                Ok(Value::Real(d))
            }
            Layout::Tuple(ts) => Ok(Value::Tuple(ts.iter().map(|t| Self::parse_tokens(t, toks)).collect::<Result<_>>()?)),
            Layout::Record(fs) => Ok(Value::Record(fs.iter().map(|(n, t)| Self::parse_tokens(t, toks).map(|v| (n.clone(), v))).collect::<Result<_>>()?)),
        }
    }

    /// Nested JSON: tensors as nested lists, tuples as lists, records as objects.
    pub fn to_json(&self, layout: &Layout) -> serde_json::Value {
        match (self, layout) {
            (Value::Idx(d), Layout::Leaf(Leaf::Idx { shape, .. })) => nest(shape, &d.iter().map(|x| serde_json::json!(x)).collect::<Vec<_>>()),
            (Value::Real(d), Layout::Leaf(Leaf::Real { shape })) => nest(shape, &d.iter().map(|x| serde_json::json!(x)).collect::<Vec<_>>()),
            (Value::Tuple(vs), Layout::Tuple(ts)) => serde_json::Value::Array(vs.iter().zip(ts).map(|(v, t)| v.to_json(t)).collect()),
            (Value::Record(fs), Layout::Record(ls)) => serde_json::Value::Object(fs.iter().zip(ls).map(|((n, v), (_, t))| (n.clone(), v.to_json(t))).collect()),
            _ => serde_json::Value::Null,
        }
    }

    /// One-line rendering for `presk data info`.
    pub fn render(&self, layout: &Layout) -> String {
        match (self, layout) {
            (Value::Idx(d), Layout::Leaf(Leaf::Idx { shape, .. })) => nest_text(shape, &d.iter().map(|x| x.to_string()).collect::<Vec<_>>()),
            (Value::Real(d), Layout::Leaf(Leaf::Real { shape })) => nest_text(shape, &d.iter().map(|x| format!("{x}")).collect::<Vec<_>>()),
            (Value::Tuple(vs), Layout::Tuple(ts)) => format!("({})", vs.iter().zip(ts).map(|(v, t)| v.render(t)).collect::<Vec<_>>().join(", ")),
            (Value::Record(fs), Layout::Record(ls)) => format!("{{{}}}", fs.iter().zip(ls).map(|((n, v), (_, t))| format!("{n} = {}", v.render(t))).collect::<Vec<_>>().join(", ")),
            _ => "?".into(),
        }
    }
}

fn nest(shape: &[usize], flat: &[serde_json::Value]) -> serde_json::Value {
    match shape {
        [] => flat.first().cloned().unwrap_or(serde_json::Value::Null),
        [_] => serde_json::Value::Array(flat.to_vec()),
        [n, rest @ ..] => {
            let chunk = flat.len() / n.max(&1);
            serde_json::Value::Array((0..*n).map(|i| nest(rest, &flat[i * chunk..(i + 1) * chunk])).collect())
        }
    }
}

fn nest_text(shape: &[usize], flat: &[String]) -> String {
    match shape {
        [] => flat.first().cloned().unwrap_or_default(),
        [_] => format!("[{}]", flat.join(", ")),
        [n, rest @ ..] => {
            let chunk = flat.len() / n.max(&1);
            format!("[{}]", (0..*n).map(|i| nest_text(rest, &flat[i * chunk..(i + 1) * chunk])).collect::<Vec<_>>().join(", "))
        }
    }
}

// ----- files -------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    pub version: u16,
    /// The sample type, in Presk syntax.
    pub sample_ty: String,
    pub count: u64,
    /// Version 2: each leaf's encoding, in storage order. Empty in a
    /// version 1 file, whose real leaves are all `f64`.
    pub encodings: Vec<Enc>,
}

impl Header {
    /// Bytes of the header on disk.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> u64 {
        let v2 = if self.version == VERSION_ENCODED { 4 + self.encodings.len() } else { 0 };
        (4 + 2 + 4 + self.sample_ty.len() + 8 + v2) as u64
    }

    /// The encoding of leaf `i`.
    pub fn encoding(&self, i: usize) -> Enc {
        self.encodings.get(i).copied().unwrap_or(Enc::F64)
    }
}

/// Writes a `.psk.d` file: `sample_ty` is the header text, `layout` its
/// leaf structure (needed for `params M`, whose layout the text does not give).
pub fn write<'a>(path: &Path, sample_ty: &str, layout: &Layout, samples: impl IntoIterator<Item = &'a Value>) -> Result<u64> {
    write_encoded(path, sample_ty, layout, &[], samples)
}

/// [`write`], each real leaf in its encoding (`encs` in storage order,
/// empty for every leaf in `f64`). The file is version 2 when a real leaf
/// is narrower than `f64`, and version 1 otherwise, byte for byte what
/// [`write`] writes. A value that is not a number of its leaf's encoding is
/// refused (`E9004`): the file holds the numbers it is given, exactly.
pub fn write_encoded<'a>(path: &Path, sample_ty: &str, layout: &Layout, encs: &[Enc], samples: impl IntoIterator<Item = &'a Value>) -> Result<u64> {
    let leaves = layout.leaves();
    if !encs.is_empty() && encs.len() != leaves.len() {
        return err("E9005", format!("{} encodings for {} leaves", encs.len(), leaves.len()));
    }
    let enc = |i: usize| encs.get(i).copied().unwrap_or(Enc::F64);
    let narrow = leaves.iter().enumerate().any(|(i, l)| matches!(l, Leaf::Real { .. }) && enc(i) != Enc::F64);
    if let Some((i, _)) = leaves.iter().enumerate().find(|(i, l)| matches!(l, Leaf::Idx { .. }) && enc(*i) != Enc::F64) {
        return err("E9005", format!("leaf {i} holds indices, which have no real encoding"));
    }
    let f = File::create(path).map_err(io)?;
    let mut w = BufWriter::new(f);
    w.write_all(MAGIC).map_err(io)?;
    w.write_all(&(if narrow { VERSION_ENCODED } else { VERSION }).to_le_bytes()).map_err(io)?;
    w.write_all(&(sample_ty.len() as u32).to_le_bytes()).map_err(io)?;
    w.write_all(sample_ty.as_bytes()).map_err(io)?;
    let count_pos = 4 + 2 + 4 + sample_ty.len() as u64;
    w.write_all(&0u64.to_le_bytes()).map_err(io)?;
    if narrow {
        w.write_all(&(leaves.len() as u32).to_le_bytes()).map_err(io)?;
        let codes: Vec<u8> = (0..leaves.len()).map(|i| enc(i).code()).collect();
        w.write_all(&codes).map_err(io)?;
    }
    let mut count = 0u64;
    for s in samples {
        s.check(layout)?;
        for (li, (leaf, v)) in leaves.iter().zip(s.leaves()).enumerate() {
            match (leaf, v) {
                (Leaf::Idx { .. }, Value::Idx(d)) => {
                    for x in d {
                        let b = x.to_le_bytes();
                        w.write_all(&b[..leaf.width()]).map_err(io)?;
                    }
                }
                (Leaf::Real { .. }, Value::Real(d)) => {
                    let e = enc(li);
                    let mut bytes = Vec::with_capacity(d.len() * e.width());
                    for x in d {
                        e.encode(*x, &mut bytes).map_err(|m| DataError { code: m.code, message: format!("leaf {li}: {}", m.message) })?;
                    }
                    w.write_all(&bytes).map_err(io)?;
                }
                _ => unreachable!("checked"),
            }
        }
        count += 1;
    }
    w.flush().map_err(io)?;
    let mut f = w.into_inner().map_err(|e| io(e.into_error()))?;
    f.seek(SeekFrom::Start(count_pos)).map_err(io)?;
    f.write_all(&count.to_le_bytes()).map_err(io)?;
    Ok(count)
}

/// A reader with random access to samples (fixed sample size).
#[derive(Debug)]
pub struct Reader {
    file: BufReader<File>,
    header: Header,
    layout: Option<Layout>,
    data_start: u64,
    file_len: u64,
}

impl Reader {
    /// Opens a file and reads its header; the layout is derived from the
    /// type when it is closed (`E9001` is deferred to [`Reader::layout`]).
    pub fn open(path: &Path) -> Result<Reader> {
        let f = File::open(path).map_err(|e| DataError { code: "E9002", message: format!("{}: {e}", path.display()) })?;
        let file_len = f.metadata().map_err(io)?.len();
        let mut r = BufReader::new(f);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic).map_err(|_| DataError { code: "E9002", message: format!("{}: not a .psk.d file (too short)", path.display()) })?;
        if &magic != MAGIC {
            return err("E9002", format!("{}: not a .psk.d file (bad magic)", path.display()));
        }
        let mut b2 = [0u8; 2];
        r.read_exact(&mut b2).map_err(io)?;
        let version = u16::from_le_bytes(b2);
        if version != VERSION && version != VERSION_ENCODED {
            return err("E9002", format!("{}: .psk.d version {version}, this tool reads versions {VERSION} and {VERSION_ENCODED}", path.display()));
        }
        let mut b4 = [0u8; 4];
        r.read_exact(&mut b4).map_err(io)?;
        let tlen = u32::from_le_bytes(b4) as usize;
        let mut tbuf = vec![0u8; tlen];
        r.read_exact(&mut tbuf).map_err(|_| DataError { code: "E9003", message: format!("{}: truncated header", path.display()) })?;
        let sample_ty = String::from_utf8(tbuf).map_err(|_| DataError { code: "E9002", message: format!("{}: the sample type is not UTF-8", path.display()) })?;
        let mut b8 = [0u8; 8];
        r.read_exact(&mut b8).map_err(|_| DataError { code: "E9003", message: format!("{}: truncated header", path.display()) })?;
        let count = u64::from_le_bytes(b8);
        let mut encodings = Vec::new();
        if version == VERSION_ENCODED {
            r.read_exact(&mut b4).map_err(|_| DataError { code: "E9003", message: format!("{}: truncated header", path.display()) })?;
            let n = u32::from_le_bytes(b4) as usize;
            let mut codes = vec![0u8; n];
            r.read_exact(&mut codes).map_err(|_| DataError { code: "E9003", message: format!("{}: truncated header", path.display()) })?;
            for (i, c) in codes.iter().enumerate() {
                encodings.push(Enc::of_code(*c).ok_or_else(|| DataError { code: "E9002", message: format!("{}: leaf {i}: unknown encoding {c}", path.display()) })?);
            }
        }
        let header = Header { version, sample_ty, count, encodings };
        let data_start = header.len();
        let layout = Layout::from_type_text(&header.sample_ty).ok();
        if let Some(l) = &layout {
            check_encodings(l, &header, path)?;
        }
        let mut reader = Reader { file: r, header, layout, data_start, file_len };
        reader.check_len(path)?;
        Ok(reader)
    }

    fn check_len(&mut self, path: &Path) -> Result<()> {
        if let Some(l) = &self.layout {
            let size = l.sample_bytes_enc(&self.header.encodings);
            let need = self.data_start + self.header.count * size as u64;
            if self.file_len < need {
                return err("E9003", format!("{}: truncated: {} samples of {} bytes announced, {} bytes of data present", path.display(), self.header.count, size, self.file_len.saturating_sub(self.data_start)));
            }
        }
        Ok(())
    }

    /// Supplies the layout of a `params M` file (from the program).
    ///
    /// A `params` file records its model's name and not its leaves' shapes,
    /// so the layout comes from the program at the dimensions it is opened
    /// with, and the file must then be *exactly* the size that layout
    /// implies. A longer file is not a truncation check passing: it is a
    /// model of other dimensions, and decoding it by this layout would read
    /// every leaf after the first mismatch from the wrong offsets (0.6.0 step
    /// 5d, D48 — `pos : ℝ[T, D]` trained at `T = 256` and opened at `T = 32`).
    pub fn with_layout(mut self, layout: Layout) -> Result<Reader> {
        check_encodings(&layout, &self.header, Path::new("the file"))?;
        let size = layout.sample_bytes_enc(&self.header.encodings);
        let need = self.data_start + self.header.count * size as u64;
        if self.file_len < need {
            return err("E9003", format!("truncated: {} samples of {} bytes announced, {} bytes of data present", self.header.count, size, self.file_len.saturating_sub(self.data_start)));
        }
        if self.file_len > need {
            return err("E9005", format!("{} bytes of data where {} samples of {} bytes were expected: the file holds a model of other dimensions (a parameter shaped by a dimension, such as `pos : ℝ[T, D]`, at another `T`?)", self.file_len.saturating_sub(self.data_start), self.header.count, size));
        }
        self.layout = Some(layout);
        Ok(self)
    }

    /// Every value of a `params M` file, in file order, without its layout.
    /// A parameter tree holds only reals, stored as `f64`, so two
    /// checkpoints of one model can be compared value for value without
    /// the program that gives the tree its shape (`presk data diff`).
    pub fn params_reals(&mut self) -> Result<Vec<f64>> {
        if !self.header.sample_ty.starts_with("params ") {
            return err("E9005", format!("`{}` is not a parameter tree", self.header.sample_ty));
        }
        // Without the program the leaves' sizes are unknown: one encoding
        // for every leaf reads them all.
        let e = self.header.encodings.first().copied().unwrap_or(Enc::F64);
        if self.header.encodings.iter().any(|x| *x != e) {
            return err("E9005", "the leaves of the file have several encodings: their sizes need the program");
        }
        let n = self.file_len.saturating_sub(self.data_start);
        if n % e.width() as u64 != 0 {
            return err("E9003", format!("{n} bytes of parameters, not a whole number of {}", e.name()));
        }
        self.file.seek(SeekFrom::Start(self.data_start)).map_err(io)?;
        let mut buf = vec![0u8; n as usize];
        self.file.read_exact(&mut buf).map_err(io)?;
        Ok(buf.chunks_exact(e.width()).map(|c| e.decode(c)).collect())
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    /// The layout, or `E9001` when the type needs the program.
    pub fn layout(&self) -> Result<&Layout> {
        match &self.layout {
            Some(l) => Ok(l),
            None => Layout::from_type_text(&self.header.sample_ty).map(|_| unreachable!()),
        }
    }

    pub fn len(&self) -> usize {
        self.header.count as usize
    }

    pub fn is_empty(&self) -> bool {
        self.header.count == 0
    }

    /// The `i`-th sample.
    pub fn get(&mut self, i: usize) -> Result<Value> {
        let layout = self.layout()?.clone();
        if i >= self.len() {
            return err("E9003", format!("sample {i} requested, {} in the file", self.len()));
        }
        let size = layout.sample_bytes_enc(&self.header.encodings);
        self.file.seek(SeekFrom::Start(self.data_start + (i * size) as u64)).map_err(io)?;
        let mut buf = vec![0u8; size];
        self.file.read_exact(&mut buf).map_err(|_| DataError { code: "E9003", message: format!("truncated at sample {i}") })?;
        let (mut pos, mut leaf) = (0, 0);
        Ok(decode(&layout, &self.header.encodings, &buf, &mut pos, &mut leaf))
    }

    /// All samples, in file order.
    pub fn read_all(&mut self) -> Result<Vec<Value>> {
        (0..self.len()).map(|i| self.get(i)).collect()
    }
}

/// A version 2 header has one encoding per leaf of the type, and none of an
/// index leaf.
fn check_encodings(layout: &Layout, header: &Header, path: &Path) -> Result<()> {
    if header.version != VERSION_ENCODED {
        return Ok(());
    }
    let leaves = layout.leaves();
    if header.encodings.len() != leaves.len() {
        return err("E9005", format!("{}: {} encodings for the {} leaves of `{}`", path.display(), header.encodings.len(), leaves.len(), header.sample_ty));
    }
    if let Some((i, _)) = leaves.iter().enumerate().find(|(i, l)| matches!(l, Leaf::Idx { .. }) && header.encodings[*i] != Enc::F64) {
        return err("E9005", format!("{}: leaf {i} holds indices and has a real encoding", path.display()));
    }
    Ok(())
}

fn decode(layout: &Layout, encs: &[Enc], buf: &[u8], pos: &mut usize, leaf_no: &mut usize) -> Value {
    match layout {
        Layout::Leaf(leaf @ Leaf::Idx { .. }) => {
            *leaf_no += 1;
            let w = leaf.width();
            let d = (0..leaf.numel())
                .map(|_| {
                    let mut b = [0u8; 8];
                    b[..w].copy_from_slice(&buf[*pos..*pos + w]);
                    *pos += w;
                    u64::from_le_bytes(b)
                })
                .collect();
            Value::Idx(d)
        }
        Layout::Leaf(leaf @ Leaf::Real { .. }) => {
            let e = encs.get(*leaf_no).copied().unwrap_or(Enc::F64);
            *leaf_no += 1;
            let w = e.width();
            let d = (0..leaf.numel())
                .map(|_| {
                    let x = e.decode(&buf[*pos..*pos + w]);
                    *pos += w;
                    x
                })
                .collect();
            Value::Real(d)
        }
        Layout::Tuple(ts) => Value::Tuple(ts.iter().map(|t| decode(t, encs, buf, pos, leaf_no)).collect()),
        Layout::Record(fs) => Value::Record(fs.iter().map(|(n, t)| (n.clone(), decode(t, encs, buf, pos, leaf_no))).collect()),
    }
}

// ----- pipeline ----------------------------------------------------------

/// `stream … |> shuffle(seed) |> batch(B) |> take(n)` over sample indices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pipeline {
    /// `shuffle(seed = s)`: a permutation per epoch derived from `s`.
    pub shuffle: Option<u64>,
    /// `batch(B = n)`: groups of `n` consecutive samples; an incomplete last group is dropped.
    pub batch: usize,
    /// `take(n)`: only the first `n` batches of each epoch.
    pub take: Option<usize>,
}

impl Pipeline {
    /// The batches of one epoch (1-based), as groups of sample indices.
    pub fn batches(&self, count: usize, epoch: u64) -> Vec<Vec<usize>> {
        let order: Vec<usize> = match self.shuffle {
            Some(seed) => shuffle_indices(count, seed, epoch),
            None => (0..count).collect(),
        };
        let b = self.batch.max(1);
        let mut out: Vec<Vec<usize>> = order.chunks_exact(b).map(|c| c.to_vec()).collect();
        if let Some(n) = self.take {
            out.truncate(n);
        }
        out
    }

    /// Batches per epoch (what the schedules count).
    pub fn batches_per_epoch(&self, count: usize) -> usize {
        let n = count / self.batch.max(1);
        self.take.map_or(n, |t| n.min(t))
    }
}

/// A permutation of `0..n` from `seed` and the epoch (1-based): the same on
/// every machine, different for every epoch.
pub fn shuffle_indices(n: usize, seed: u64, epoch: u64) -> Vec<usize> {
    let mut master = ChaCha20Rng::seed_from_u64(seed);
    let mut key = [0u8; 32];
    for _ in 0..epoch.max(1) {
        master.fill_bytes(&mut key);
    }
    let mut rng = ChaCha20Rng::from_seed(key);
    let mut v: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

// ----- text -------------------------------------------------------------

/// The character vocabulary of a text: every distinct character, sorted, one
/// token each. Tokens are `String`s so that the same format can later hold
/// sub-word tokens; index `i` denotes `vocab[i]`.
pub fn char_vocab(text: &str) -> Vec<String> {
    let mut chars: Vec<char> = text.chars().collect();
    chars.sort_unstable();
    chars.dedup();
    chars.into_iter().map(String::from).collect()
}

/// Encode a text against a vocabulary. Only single-character vocabularies are
/// supported: sub-word tokens need longest-match encoding, which comes with
/// `presk data bpe`.
pub fn encode(text: &str, vocab: &[String]) -> Result<Vec<u64>> {
    let mut map = std::collections::HashMap::with_capacity(vocab.len());
    for (i, t) in vocab.iter().enumerate() {
        let mut cs = t.chars();
        let (Some(c), None) = (cs.next(), cs.next()) else {
            return err("E9004", format!("the vocabulary holds the multi-character token `{t}`; only character vocabularies can be encoded in this version"));
        };
        map.insert(c, i as u64);
    }
    text.chars()
        .map(|c| map.get(&c).copied().ok_or_else(|| DataError { code: "E9004", message: format!("the character `{}` (U+{:04X}) is not in the vocabulary", c.escape_debug(), c as u32) }))
        .collect()
}

/// Decode a token stream back to text (the inverse of [`encode`]).
pub fn decode_text(tokens: &[u64], vocab: &[String]) -> Result<String> {
    tokens
        .iter()
        .map(|t| vocab.get(*t as usize).map(String::as_str).ok_or_else(|| DataError { code: "E9004", message: format!("token {t} is beyond the vocabulary of {}", vocab.len()) }))
        .collect()
}

// ----- data preparation --------------------------------------------------

/// Language-modelling windows of a token stream: sample `k` is
/// `(tokens[k·stride .. +seq], tokens[k·stride + 1 .. +seq])`, of type
/// `(idx[vocab][seq: T], idx[vocab][seq: T])`.
pub fn lm_windows(tokens: &[u64], vocab: u64, seq: usize, stride: usize) -> Result<(String, Layout, Vec<Value>)> {
    if seq == 0 || stride == 0 {
        return err("E9004", "`--seq` and `--stride` must be at least 1");
    }
    if let Some(x) = tokens.iter().find(|x| **x >= vocab) {
        return err("E9004", format!("token {x} is not below the vocabulary size {vocab}"));
    }
    let ty = format!("(idx[{vocab}][seq: {seq}], idx[{vocab}][seq: {seq}])");
    let layout = Layout::from_type_text(&ty)?;
    let mut out = Vec::new();
    let mut k = 0;
    while k + seq < tokens.len() {
        out.push(Value::Tuple(vec![Value::Idx(tokens[k..k + seq].to_vec()), Value::Idx(tokens[k + 1..k + 1 + seq].to_vec())]));
        k += stride;
    }
    Ok((ty, layout, out))
}

/// `presk data info`: the header and the first `n` samples.
pub fn info(reader: &mut Reader, n: usize) -> Result<String> {
    let h = reader.header().clone();
    let mut s = format!("type    {}\ncount   {}\nversion {}\n", h.sample_ty, h.count, h.version);
    if !h.encodings.is_empty() {
        // Version 2: the leaves' encodings, runs of one encoding together.
        let mut runs: Vec<(Enc, usize)> = Vec::new();
        for e in &h.encodings {
            match runs.last_mut() {
                Some((f, n)) if f == e => *n += 1,
                _ => runs.push((*e, 1)),
            }
        }
        s.push_str(&format!("leaves  {}\n", runs.iter().map(|(e, n)| format!("{n} × {}", e.name())).collect::<Vec<_>>().join(", ")));
    }
    match reader.layout() {
        Ok(l) => {
            let l = l.clone();
            s.push_str(&format!("sample  {} bytes, {} leaves\n", l.sample_bytes_enc(&h.encodings), l.leaves().len()));
            for i in 0..n.min(reader.len()) {
                let v = reader.get(i)?;
                s.push_str(&format!("[{i}] {}\n", v.render(&l)));
            }
        }
        Err(e) => s.push_str(&format!("sample  layout not derivable from the type ({}): open it from the program\n", e.message)),
    }
    Ok(s)
}
