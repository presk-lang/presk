//! The host (plan 0.9.0 §3.3, steps 3 and 4; D79, D80): a [`Recording`] of
//! the kernel backend's run of an entry becomes a C program. [`export`]
//! writes the structured host of step 4 ([`crate::structure`]): a C function
//! per distinct recording of a Presk function, a fold as a loop, the arena
//! released per call. [`export_flat`] writes step 3's host, kept as its
//! oracle: one function, `presk_<entry>`, performs every operation of the
//! recording in order, every buffer bump-allocated from one arena. In both,
//! `main` reads the inputs, calls the entry and writes the outputs.
//!
//! What both print alike:
//!
//! - A launch is a call of its kernel's function (`kernels.c`, deduplicated
//!   by text) over its range, divided among the workers the native backend
//!   divides it among (step 5, D81): one call when it runs on one, else an
//!   OpenMP region whose threads each call it on their piece
//!   ([`crate::launch`]). The split belongs to the launch, not to the
//!   kernel's text: a kernel under the executor's default and the same kernel
//!   stated on one worker print alike. Every kernel is always inlined into
//!   the host, and on GCC `model.c` is compiled without its scheduling
//!   before register allocation (`MODEL_HEAD`, D97's addendum).
//! - A view is a pointer at an offset; a slice of a stacked tree too.
//! - A buffer converted at a kernel's edge, a stack, a slice written back are
//!   `presk_convert`s; a fill (`zeros_like`, a dropout mask outside training)
//!   and a host scalar made a buffer are `presk_fill`s, both rounding as
//!   `Buf::of` rounds.
//! - The host's arithmetic on hyperparameters is `double` arithmetic, and a
//!   scalar read back from a kernel's output is an element widened to
//!   `double`, as `Value::scalar_f64` reads it.
//! - The straight-line host releases nothing: its arena holds the inputs and
//!   every buffer, its size the sum of them, stated in `model.h`.
//! - `sample` (step 6, D83) is `presk_sample` drawing from a key argument of
//!   the entry (plan 0.10.0 step 7b, D99), a `presk_key` of its `_in` struct
//!   which `main` derives from `-s SEED` as `presk run --seed` does; a key
//!   nothing draws from is no field. An `f64` argument of the entry is a
//!   `double` of its `_in` struct, which `main` sets to the value it was
//!   exported with.
//! - With `--serve` ([`export_serve`]), the program holds a generation
//!   function and its greedy copy, two entries on the same structs, and
//!   `generate.c`, `presk serve`'s loop over them.
//! - Integer weights (step 7, D84): a leaf the realization quantizes is a
//!   `presk_q8` or `presk_q4` of the `_in` struct, its integers and scales.
//!   `main` and `generate` read its reals from the file and quantize them
//!   (`presk_quantize`), as `presk run` quantizes a checkpoint it loads: the
//!   scheme is the realization's (`Meta::quant`), the scales are data. A
//!   kernel the native backend hands the integers takes them, the scales
//!   after its output; any other reads a `presk_dequantize` copy.

use std::collections::HashMap;
use std::fmt::Write as _;

use presk_kernel::Kernel;
use presk_kernel::record::{Made, Op, Recording, Scalar, Store, Sym, View, leaves};

use crate::structure::{self, EntryIo, Structured};
use crate::{CFLAGS, OPENMP, RUNTIME, WARNINGS, kernel_fn, launch, ranged, workers};

type Result<T> = std::result::Result<T, String>;

/// How a kernel of `kernels.c` is declared (plan 0.10.0 step 7's follow-up,
/// D97's addendum): a static function always inlined where the host
/// launches it. Left to choose, GCC inlined some of a program's products and
/// not others, and its choice moved with the program's other kernels.
const KERNEL_DECL: &str = "/* Every kernel is inlined where the host launches it (presk D97). */\n#if defined(__GNUC__)\n#define PRESK_KERNEL static inline __attribute__((always_inline))\n#else\n#define PRESK_KERNEL static inline\n#endif\n\n";

/// The head of `model.c`, before `model.h`: on GCC, every function of the
/// file compiled without the scheduling that runs before register
/// allocation. A product's inner loop under a 4×16 tile holds sixteen
/// accumulators, four rows and four scalars, at the edge of the 32 vector
/// registers; that scheduling hoisted the sixteen products ahead of their
/// sums, and wherever the allocator then ran short an accumulator lived on
/// the stack, read and written at every iteration, the product about 30 %
/// slower (D97's addendum). Out of line it ran short in every case measured;
/// inlined, in some. No rounding moves: scheduling reorders instructions
/// that do not depend on one another, and the program's bits are the
/// native backend's either way (C1).
const MODEL_HEAD: &str = "#if defined(__GNUC__) && !defined(__clang__)\n/* GCC: no scheduling before register allocation, which spilled the\n * accumulators of a product's inner loop (presk D97). */\n#pragma GCC optimize (\"no-schedule-insns\")\n#endif\n\n";

/// How `model.c` includes the kernels.
const INCLUDE_KERNELS: &str = "/* The kernels are static and always inlined into the host's functions:\n * included here rather than compiled apart. */\n#include \"kernels.c\"\n";

/// What the program is exported from, for the README and the comments.
#[derive(Clone, Debug, Default)]
pub struct Meta {
    /// The Presk file, as given.
    pub file: String,
    /// The entry as `presk export --entry` names it.
    pub entry: String,
    /// `NAME = N`, the entry's dimensions.
    pub dims: Vec<(String, u64)>,
    /// `--const` as given.
    pub consts: Vec<String>,
    /// The realization, `f64`, `f32` or `bf16`.
    pub realize: String,
    pub seed: u64,
    /// The checkpoint given with `--weights`, when one was.
    pub weights: Option<String>,
    /// The compiler that exported it, and the commit of the program's
    /// repository when it is in one.
    pub version: String,
    pub commit: Option<String>,
    /// Whether `conformance/` holds the exact reference's outputs.
    pub exact: bool,
    /// The realization plan the entry was exported under, as given
    /// (`--plan @FILE`), when one was.
    pub plan: Option<String>,
    /// The native executor's workers (`--threads`, or the cores), by which a
    /// kernel under the executor's default is divided ([`crate::workers`]).
    pub threads: usize,
    /// The realization's `quant` map (step 7): each parameter it quantizes,
    /// by its path in the entry's first argument (`blocks.mlp.l1.w`), and
    /// its scheme, `absmax` or `pow2`.
    pub quant: std::collections::BTreeMap<String, String>,
}

/// An argument of the entry: its source name, and whether it is the
/// parameter tree read from a checkpoint (`params M`, the first argument, as
/// `presk run --weights` replaces it).
#[derive(Clone, Debug)]
pub struct Arg {
    pub name: String,
    pub weights: bool,
}

/// A `.psk.d` file the program reads or writes: its sample type as the
/// header writes it, and its leaves in order, `(bound, elements)`, `bound` 0
/// for reals.
#[derive(Clone, Debug, PartialEq)]
pub struct DataFile {
    pub ty: String,
    pub leaves: Vec<(u64, usize)>,
}

/// The exported program.
#[derive(Clone, Debug)]
pub struct Exported {
    /// Every file of the directory but the conformance data, `(path, text)`.
    pub files: Vec<(String, String)>,
    /// The inputs the program reads besides the weights, `None` when it
    /// reads none.
    pub inputs: Option<DataFile>,
    /// The weights, when the entry takes a parameter tree from a checkpoint;
    /// its type is the one given.
    pub weights: Option<DataFile>,
    pub outputs: DataFile,
    /// Bytes of the arena: the inputs and the host's peak (the structured
    /// host), or every buffer it makes (the straight-line one).
    pub arena: usize,
    /// Bytes the straight-line host's arena takes, for comparison.
    pub arena_flat: usize,
    /// C functions of the structured host besides the entry's (0 for the
    /// straight-line one), loops, and how each fold was emitted.
    pub functions: usize,
    pub loops: usize,
    pub folds: Vec<String>,
    pub launches: usize,
    /// Launches whose output nothing the result depends on reads: the
    /// native backend makes them too (the forward pass a gradient runs
    /// before its reverse pass, and discards).
    pub dead: usize,
    /// Distinct kernel functions.
    pub kernels: usize,
    /// Launches divided among threads, and the most workers one takes.
    pub split: usize,
    pub max_workers: usize,
    /// Leaves of integer weights, quantized when they are read (step 7).
    pub quantized: usize,
    /// The C function `k<n>` of `kernels.c` each launched kernel of the
    /// listing is, `(its place in the listing, n)`: the kernels are
    /// deduplicated by their C text (plan 0.13.0 step 1, D134).
    pub kernel_of: Vec<(usize, usize)>,
    /// The lines of `kernels.c` each `k<n>` takes, its PSK-Kernel text's
    /// comment included, first and last, from 1.
    pub kernel_lines: Vec<(usize, usize)>,
}

/// The indices in `rec.ops` of the launches the result depends on; the
/// others compute values nothing reads (a gradient's forward pass, which it
/// runs before its reverse pass and discards), as the native backend does.
pub fn live_launches(rec: &Recording) -> Vec<usize> {
    let mut bufs = vec![false; rec.bufs.len()];
    let mut vars = vec![false; rec.vars];
    if let Some(r) = &rec.result {
        for l in leaves(r) {
            if let Sym::Tensor(v) = l {
                bufs[v.buf] = true;
            }
        }
    }
    let var = |x: &Scalar, vars: &mut Vec<bool>| {
        if let Scalar::Var(i) = x {
            vars[*i] = true;
        }
    };
    let mut live = Vec::new();
    for (i, op) in rec.ops.iter().enumerate().rev() {
        match op {
            Op::Launch { inputs, out, .. } => {
                if bufs[*out] {
                    inputs.iter().for_each(|v| bufs[v.buf] = true);
                    live.push(i);
                }
            }
            Op::Copy { dst, src, .. } => {
                if bufs[*dst] {
                    bufs[src.buf] = true;
                }
            }
            Op::Scalar { buf, x } => {
                if bufs[*buf] {
                    var(x, &mut vars);
                }
            }
            Op::Hyper { out, a, b, .. } => {
                if vars[*out] {
                    var(a, &mut vars);
                    var(b, &mut vars);
                }
            }
            Op::Read { out, src } => {
                if vars[*out] {
                    bufs[src.buf] = true;
                }
            }
            // A draw is a function of its key and its logits (plan 0.10.0
            // step 7b): read when its tokens are.
            Op::Sample { out, src, .. } => {
                if bufs[*out] {
                    bufs[src.buf] = true;
                }
            }
            Op::Fill { .. } => {}
        }
    }
    live.reverse();
    live
}

const ALIGN: usize = 64;

fn aligned(bytes: usize) -> usize {
    bytes.div_ceil(ALIGN) * ALIGN
}

/// The C type of an element; for integer weights, of the leaf, the struct
/// of its integers and scales (step 7).
pub(crate) fn ctype(s: Store) -> &'static str {
    match s {
        Store::F64 => "double",
        Store::F32 => "float",
        Store::Bf16 => "uint16_t",
        Store::I64 => "int64_t",
        Store::Quant { .. } => crate::structure::quant_type(s),
    }
}

/// The runtime's name of a type; integer weights are read from a file as
/// the reals they are quantized from (step 7), and converted by
/// `presk_dequantize`, never `presk_convert`.
pub(crate) fn cdtype(s: Store) -> &'static str {
    match s {
        Store::F64 | Store::Quant { .. } => "PRESK_F64",
        Store::F32 => "PRESK_F32",
        Store::Bf16 => "PRESK_BF16",
        Store::I64 => "PRESK_I64",
    }
}

/// A leaf's type in Presk's syntax, with literal dimensions: `ℝ[2, 4]`,
/// `idx[64][2, 8]`.
fn leaf_type(v: &View) -> String {
    let dims = v.shape.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(", ");
    match v.bound {
        Some(b) => format!("idx[{b}][{dims}]"),
        None => format!("ℝ[{dims}]"),
    }
}

/// The closed type of a file of these leaves: the leaf's type, or the tuple
/// of them. `presk data` reads it without the program.
pub fn closed_type(views: &[View]) -> String {
    match views {
        [one] => leaf_type(one),
        many => format!("({})", many.iter().map(leaf_type).collect::<Vec<_>>().join(", ")),
    }
}

/// A C string literal of `s`: every byte outside printable ASCII as an octal
/// escape, so that the type text is compared byte for byte whatever the
/// compiler's source character set.
pub fn c_string(s: &str) -> String {
    let mut out = String::from("\"");
    for b in s.bytes() {
        match b {
            b'"' | b'\\' => {
                out.push('\\');
                out.push(b as char);
            }
            0x20..=0x7e => out.push(b as char),
            _ => {
                let _ = write!(out, "\\{b:03o}");
            }
        }
    }
    out.push('"');
    out
}

const KEYWORDS: &[&str] = &[
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else", "enum", "extern", "float", "for", "goto", "if", "inline", "int", "long", "register", "restrict", "return", "short", "signed", "sizeof", "static", "struct", "switch", "typedef", "union", "unsigned", "void", "volatile", "while",
];

fn translit(c: char) -> Option<&'static str> {
    Some(match c {
        'α' => "alpha",
        'β' => "beta",
        'γ' => "gamma",
        'δ' => "delta",
        'ε' => "epsilon",
        'ζ' => "zeta",
        'η' => "eta",
        'θ' => "theta",
        'ι' => "iota",
        'κ' => "kappa",
        'λ' => "lambda",
        'μ' => "mu",
        'ν' => "nu",
        'ξ' => "xi",
        'π' => "pi",
        'ρ' => "rho",
        'σ' => "sigma",
        'τ' => "tau",
        'υ' => "upsilon",
        'φ' => "phi",
        'χ' => "chi",
        'ψ' => "psi",
        'ω' => "omega",
        'Γ' => "Gamma",
        'Δ' => "Delta",
        'Θ' => "Theta",
        'Λ' => "Lambda",
        'Ξ' => "Xi",
        'Π' => "Pi",
        'Σ' => "Sigma",
        'Φ' => "Phi",
        'Ψ' => "Psi",
        'Ω' => "Omega",
        'ℝ' => "R",
        // A hat, precomposed or combining: `ŷ` is `y_hat`, `x̂` is `x_hat`.
        'ŷ' => "y_hat",
        '\u{302}' => "_hat",
        '\u{304}' => "_bar",
        '′' | '\'' => "_prime",
        '₀' | '⁰' => "0",
        '₁' | '¹' => "1",
        '₂' | '²' => "2",
        '₃' | '³' => "3",
        _ => return None,
    })
}

/// A Presk identifier or path as a C identifier: ASCII letters, digits and
/// `_` kept, Greek letters and a few marks spelled out (`θ` is `theta`),
/// anything else `_`, runs of `_` collapsed; `LayerNorm{8}.apply` is
/// `LayerNorm_8_apply`.
pub fn c_ident(s: &str) -> String {
    let mut raw = String::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            raw.push(c);
        } else if let Some(t) = translit(c) {
            raw.push_str(t);
        } else {
            raw.push('_');
        }
    }
    let mut out = String::new();
    for c in raw.chars() {
        if c == '_' && (out.is_empty() || out.ends_with('_')) {
            continue;
        }
        out.push(c);
    }
    while out.ends_with('_') {
        out.pop();
    }
    if out.is_empty() || out.starts_with(|c: char| c.is_ascii_digit()) {
        out.insert(0, '_');
    }
    if KEYWORDS.contains(&out.as_str()) {
        out.push('_');
    }
    out
}

/// Names made unique within one scope.
#[derive(Default)]
struct Names(HashMap<String, usize>);

impl Names {
    fn fresh(&mut self, base: String) -> String {
        let n = self.0.entry(base.clone()).or_insert(0);
        *n += 1;
        if *n == 1 { base } else { format!("{base}_{n}") }
    }
}

/// The leaves of a value with their paths: `.l1.w`, `.0.tok`.
pub(crate) fn paths(s: &Sym, prefix: &str, out: &mut Vec<(String, Sym)>) {
    match s {
        Sym::Tuple(xs) => xs.iter().enumerate().for_each(|(i, x)| paths(x, &format!("{prefix}.{i}"), out)),
        Sym::Record(fs) => fs.iter().for_each(|(n, x)| paths(x, &format!("{prefix}.{n}"), out)),
        leaf => out.push((prefix.to_string(), leaf.clone())),
    }
}

/// A real as C reads it back exactly: Rust's shortest representation, which
/// C compilers round correctly (D77).
pub(crate) fn c_real(x: f64) -> String {
    if x.is_nan() {
        "NAN".into()
    } else if x.is_infinite() {
        if x > 0.0 { "HUGE_VAL".into() } else { "(-HUGE_VAL)".into() }
    } else {
        let s = format!("{x:?}");
        if s.starts_with('-') { format!("({s})") } else { s }
    }
}

fn c_scalar(x: Scalar) -> String {
    match x {
        Scalar::Const(c) => c_real(c),
        Scalar::Var(i) => format!("s{i}"),
    }
}

fn view(v: &View, names: &[String]) -> String {
    if v.off == 0 { names[v.buf].clone() } else { format!("{} + {}", names[v.buf], v.off) }
}

/// The kernels a set of recordings launches, printed once each: their C
/// texts, and for each backend kernel its function's number (`k<n>`), its
/// range and the workers its launches are divided among, with `threads` the
/// native executor's. The recordings are of one backend, whose kernels they
/// share.
struct KernelTable {
    text: Vec<String>,
    name_of: HashMap<usize, usize>,
    hi: HashMap<usize, usize>,
    workers: HashMap<usize, usize>,
}

fn kernel_table(kernels: &[Kernel], recs: &[&Recording], threads: usize) -> Result<KernelTable> {
    let mut t = KernelTable { text: Vec::new(), name_of: HashMap::new(), hi: HashMap::new(), workers: HashMap::new() };
    let mut by_text: HashMap<String, usize> = HashMap::new();
    for op in recs.iter().flat_map(|r| &r.ops) {
        if let Op::Launch { kernel, .. } = op {
            if t.workers.contains_key(kernel) {
                continue;
            }
            let k = &kernels[*kernel];
            let text = kernel_fn(k, "KERNEL")?;
            let n = match by_text.get(&text) {
                Some(n) => *n,
                None => {
                    let n = t.text.len();
                    by_text.insert(text.clone(), n);
                    t.text.push(text);
                    n
                }
            };
            t.name_of.insert(*kernel, n);
            t.hi.insert(*kernel, ranged(k).map(|(_, n)| n).unwrap_or(0));
            t.workers.insert(*kernel, workers(k, threads));
        }
    }
    Ok(t)
}

impl KernelTable {
    /// A recording's launches, and those divided among threads.
    fn counts(&self, rec: &Recording) -> (usize, usize) {
        let mut n = (0, 0);
        for op in &rec.ops {
            if let Op::Launch { kernel, .. } = op {
                n.0 += 1;
                n.1 += usize::from(self.workers[kernel] > 1);
            }
        }
        n
    }
}

/// Whether a recording draws (`sample`, step 6) from a key of its
/// arguments.
pub fn samples(rec: &Recording) -> bool {
    rec.ops.iter().any(|o| matches!(o, Op::Sample { .. }))
}

/// The structured host of `rec` (step 4), its kernels under the default
/// executor divided among `threads` workers, before it is printed: what a
/// test mutates.
pub fn structure(kernels: &[Kernel], rec: &Recording, threads: usize) -> Result<Structured> {
    structure_all(kernels, &[rec], threads)
}

/// The structured host of several entries recorded on one backend (a
/// generation function and its greedy copy, step 6): one set of functions,
/// shared where their recordings' forms are equal, and a root per entry.
pub fn structure_all(kernels: &[Kernel], recs: &[&Recording], threads: usize) -> Result<Structured> {
    let t = kernel_table(kernels, recs, threads)?;
    structure::build(recs, &t.name_of, &t.hi, &t.workers, t.text.len())
}

/// The C program of `rec`, an entry recorded on the kernel backend whose
/// kernels are `kernels`, with the structured host. `args` names the entry's
/// arguments; `weights_type` is the checkpoint's sample type, when the first
/// argument is read from one.
pub fn export(meta: &Meta, kernels: &[Kernel], rec: &Recording, args: &[Arg], weights_type: Option<&str>) -> Result<Exported> {
    let s = structure(kernels, rec, meta.threads)?;
    export_with(meta, kernels, &[(&meta.entry, rec)], args, weights_type, Some(&s), None)
}

/// The same program with `s` as its host (a structure a test changed).
pub fn export_structured(meta: &Meta, kernels: &[Kernel], rec: &Recording, args: &[Arg], weights_type: Option<&str>, s: &Structured) -> Result<Exported> {
    export_with(meta, kernels, &[(&meta.entry, rec)], args, weights_type, Some(s), None)
}

/// The same program with step 3's straight-line host, the structured one's
/// oracle.
pub fn export_flat(meta: &Meta, kernels: &[Kernel], rec: &Recording, args: &[Arg], weights_type: Option<&str>) -> Result<Exported> {
    export_with(meta, kernels, &[(&meta.entry, rec)], args, weights_type, None, None)
}

/// The generation loop of `presk serve` as a C program (`presk export
/// --target c --serve`, plan 0.9.0 step 6, D83): what the driver,
/// `generate.c`, needs to know of the generation function beside its two
/// recordings.
#[derive(Clone, Debug)]
pub struct Serve {
    /// The model's module, as `presk serve` names it.
    pub model: String,
    /// The context's vocabulary, and the window the parameters fix: its
    /// dimension's name (`T`) and length.
    pub vocab: u64,
    pub seq_dim: String,
    pub window: u64,
    /// The entry's arguments that are the key (plan 0.10.0 step 7b), the
    /// context and the temperature, by position.
    pub key: Option<usize>,
    pub ctx: usize,
    pub temperature: Option<usize>,
    /// The temperature when none is given: the function's default, or 1,
    /// as `presk serve` takes it.
    pub default_temperature: f64,
    /// The greedy copy (every `sample` an `argmax`), as `presk serve` makes
    /// it: its Presk name.
    pub greedy: String,
}

/// The generation program: the entry `rec`, the generation function as
/// written, which `conformance/` checks; its greedy copy `greedy`, recorded
/// on the same backend; and the driver. Structured, or straight-line when
/// `straight_line`.
#[allow(clippy::too_many_arguments)]
pub fn export_serve(meta: &Meta, kernels: &[Kernel], rec: &Recording, greedy: &Recording, serve: &Serve, args: &[Arg], weights_type: Option<&str>, straight_line: bool) -> Result<Exported> {
    let entries = [(meta.entry.as_str(), rec), (serve.greedy.as_str(), greedy)];
    if straight_line {
        export_with(meta, kernels, &entries, args, weights_type, None, Some(serve))
    } else {
        let s = structure_all(kernels, &[rec, greedy], meta.threads)?;
        export_with(meta, kernels, &entries, args, weights_type, Some(&s), Some(serve))
    }
}

/// A tensor leaf of the entry's arguments: a pointer of the `_in` struct,
/// or integer weights' struct.
struct In {
    field: String,
    buf: usize,
    path: String,
    ty: String,
    weights: bool,
    bound: u64,
    /// The argument it is a leaf of.
    arg: usize,
    quant: Option<QuantIn>,
}

/// How a leaf of integer weights is made from the reals read (step 7): its
/// geometry, `outer × k × inner` split at the axis `over` its scales are
/// constant along, and the realization's scheme.
struct QuantIn {
    bits: u8,
    outer: usize,
    k: usize,
    inner: usize,
    over: usize,
    scheme: String,
}

impl QuantIn {
    /// Bytes of its integers, and its scales.
    fn bytes(&self) -> (usize, usize) {
        let n = self.outer * self.k * self.inner;
        (if self.bits == 4 { n.div_ceil(2) } else { n }, self.outer * self.inner)
    }

    fn describe(&self) -> String {
        format!("int{} over axis {}, {}", self.bits, self.over, self.scheme)
    }
}

/// Bytes an input leaf takes in the arena: its elements, or integer
/// weights' integers and scales.
fn input_bytes(rec: &Recording, i: &In) -> usize {
    match &i.quant {
        Some(q) => {
            let (ints, scales) = q.bytes();
            aligned(ints) + aligned(4 * scales)
        }
        None => aligned(rec.bufs[i.buf].len * rec.bufs[i.buf].store.bytes()),
    }
}

/// An `f64` argument (step 6): a `double` of the `_in` struct, the host
/// scalar `var` of the recording, which `main` sets to the value it was
/// exported with.
struct Dbl {
    field: String,
    path: String,
    var: usize,
    value: f64,
    arg: usize,
}

/// A key argument `sample` draws from (plan 0.10.0 step 7b): a `presk_key`
/// of the `_in` struct, the recording's key argument `n`, which `main`
/// derives from the seed as `presk run` does.
struct KeyIn {
    field: String,
    path: String,
    n: usize,
    arg: usize,
}

/// The entry's arguments as the program takes them.
struct Fields {
    ins: Vec<In>,
    dbls: Vec<Dbl>,
    keys: Vec<KeyIn>,
    /// Constants fixed at export: `(path, what)`.
    fixed: Vec<(String, String)>,
    /// The `_in` struct's members, in the order of the leaves.
    decl: Vec<String>,
}

/// `sampled`, by the recording's key arguments: whether an entry of the
/// program draws from it, which makes it a field.
fn fields_of(rec: &Recording, args: &[Arg], quant: &std::collections::BTreeMap<String, String>, sampled: &[bool]) -> Result<Fields> {
    let mut names = Names::default();
    let mut f = Fields { ins: Vec::new(), dbls: Vec::new(), keys: Vec::new(), fixed: Vec::new(), decl: Vec::new() };
    for (i, (arg, sym)) in args.iter().zip(&rec.args).enumerate() {
        let mut ps = Vec::new();
        paths(sym, &arg.name, &mut ps);
        for (path, leaf) in ps {
            match leaf {
                Sym::Tensor(v) => {
                    if !matches!(rec.bufs[v.buf].made, Made::Input { arg: a, .. } if a == i) {
                        return Err(format!("argument `{path}` is not an input buffer"));
                    }
                    let field = names.fresh(c_ident(&path));
                    let ty = leaf_type(&v);
                    let rb = &rec.bufs[v.buf];
                    let q = match (rb.store, rb.quant) {
                        (Store::Quant { bits, k, inner }, Some(axis)) => {
                            // The realization names it by its path in the
                            // first argument, the parameters.
                            let rel = path.strip_prefix(&format!("{}.", arg.name)).filter(|_| i == 0);
                            let Some(scheme) = rel.and_then(|r| quant.get(r)) else { return Err(format!("argument `{path}` holds integer weights the realization does not quantize")) };
                            let outer = rb.len.checked_div(k * inner).unwrap_or(0);
                            Some(QuantIn { bits, outer, k, inner, over: axis.over, scheme: scheme.clone() })
                        }
                        (Store::Quant { .. }, None) => return Err(format!("argument `{path}` holds integer weights of no known axis")),
                        _ => None,
                    };
                    match &q {
                        Some(q) => f.decl.push(format!("    {} {field}; /* {} : {ty}, {} */", ctype(rb.store), path.replace("*/", "* /"), q.describe())),
                        None => f.decl.push(format!("    const {} *{field}; /* {} : {ty} */", ctype(rb.store), path.replace("*/", "* /"))),
                    }
                    f.ins.push(In { field, buf: v.buf, ty, path, weights: arg.weights, bound: v.bound.unwrap_or(0), arg: i, quant: q });
                }
                Sym::F64(Scalar::Var(var)) => {
                    let Some(&(_, value)) = rec.scalar_args.iter().find(|(x, _)| *x == var) else { return Err(format!("argument `{path}` is a scalar the recording does not state")) };
                    let field = names.fresh(c_ident(&path));
                    f.decl.push(format!("    double {field}; /* {} : f64 */", path.replace("*/", "* /")));
                    f.dbls.push(Dbl { field, path, var, value, arg: i });
                }
                Sym::Nat(n) => f.fixed.push((path, format!("{n} (nat)"))),
                Sym::Int(n) => f.fixed.push((path, format!("{n} (int)"))),
                Sym::F64(Scalar::Const(x)) => f.fixed.push((path, format!("{x:?} (f64)"))),
                Sym::Bool(b) => f.fixed.push((path, format!("{b} (bool)"))),
                Sym::Key(Some(n)) if sampled.get(n).copied().unwrap_or(false) => {
                    let field = names.fresh(c_ident(&path));
                    f.decl.push(format!("    presk_key {field}; /* {} : key */", path.replace("*/", "* /")));
                    f.keys.push(KeyIn { field, path, n, arg: i });
                }
                Sym::Key(_) => f.fixed.push((path, "a key, dropped: nothing the program runs draws from it".into())),
                other => return Err(format!("argument `{path}` is {other:?}")),
            }
        }
    }
    Ok(f)
}

/// The entry's result, leaf by leaf: the `_out` struct's members and the
/// views they point to.
fn outs_of(rec: &Recording) -> Result<Vec<(String, View)>> {
    let Some(result) = &rec.result else { return Err("the recording has no result".into()) };
    let mut ps = Vec::new();
    paths(result, "", &mut ps);
    let single = ps.len() == 1;
    let mut names = Names::default();
    let mut out = Vec::new();
    for (path, leaf) in ps {
        let Sym::Tensor(v) = leaf else { return Err(format!("the result's leaf `{path}` is not a tensor")) };
        let base = if single { "result".to_string() } else { c_ident(&format!("result{path}")) };
        out.push((names.fresh(base), v));
    }
    Ok(out)
}

fn out_decl(rec: &Recording, outs: &[(String, View)]) -> Vec<String> {
    outs.iter().map(|(f, v)| format!("    const {} *{f}; /* {} */", ctype(rec.bufs[v.buf].store), leaf_type(v))).collect()
}

/// The declaration of an entry's C function: the arena, the entry's `_in`
/// and `_out` structs, `types` their prefix.
fn signature(func: &str, types: &str) -> String {
    format!("void {func}(presk_arena *arena, const {types}_in *in, {types}_out *out)")
}

/// Step 3's straight-line function of an entry: every operation of `rec`
/// in order, every buffer bump-allocated.
fn flat_fn(sig: &str, rec: &Recording, f: &Fields, outs: &[(String, View)], t: &KernelTable) -> String {
    let bufnames: Vec<String> = (0..rec.bufs.len()).map(|b| format!("b{b}")).collect();
    // What is read: an input nothing reads is not declared, and a scalar
    // nothing reads is marked so, for -Wall.
    let mut read_bufs = vec![false; rec.bufs.len()];
    let mut read_vars = vec![false; rec.vars];
    let mut mark = |x: &Scalar| {
        if let Scalar::Var(i) = x {
            read_vars[*i] = true;
        }
    };
    for op in &rec.ops {
        match op {
            Op::Launch { inputs, .. } => inputs.iter().for_each(|v| read_bufs[v.buf] = true),
            Op::Copy { src, .. } | Op::Read { src, .. } | Op::Sample { src, .. } => read_bufs[src.buf] = true,
            Op::Scalar { x, .. } => mark(x),
            Op::Hyper { a, b, .. } => {
                mark(a);
                mark(b);
            }
            Op::Fill { .. } => {}
        }
    }
    for (_, v) in outs {
        read_bufs[v.buf] = true;
    }
    let mut body = String::new();
    for i in &f.ins {
        if read_bufs[i.buf] {
            match i.quant {
                Some(_) => {
                    let _ = writeln!(body, "    const {} {} = in->{};", ctype(rec.bufs[i.buf].store), bufnames[i.buf], i.field);
                }
                None => {
                    let _ = writeln!(body, "    const {} *{} = in->{};", ctype(rec.bufs[i.buf].store), bufnames[i.buf], i.field);
                }
            }
        }
    }
    for d in &f.dbls {
        let _ = writeln!(body, "    double s{} = in->{};", d.var, d.field);
        if !read_vars[d.var] {
            let _ = writeln!(body, "    (void)s{};", d.var);
        }
    }
    let mut made = vec![false; rec.bufs.len()];
    for i in &f.ins {
        made[i.buf] = true;
    }
    let mut last_site = String::new();
    let alloc = |b: usize, made: &mut Vec<bool>, body: &mut String| {
        if !made[b] {
            made[b] = true;
            let rb = &rec.bufs[b];
            let _ = writeln!(body, "    {} *{} = presk_alloc(arena, {} * sizeof({}));", ctype(rb.store), bufnames[b], rb.len, ctype(rb.store));
        }
    };
    for op in &rec.ops {
        let site = match op {
            Op::Launch { site, .. } => site.clone(),
            Op::Fill { buf, .. } | Op::Scalar { buf, .. } | Op::Copy { dst: buf, .. } | Op::Sample { out: buf, .. } => rec.bufs[*buf].site.clone(),
            Op::Hyper { .. } | Op::Read { .. } => last_site.clone(),
        };
        if site != last_site {
            let _ = writeln!(body, "    /* {} */", site.replace("*/", "* /"));
            last_site = site;
        }
        match op {
            Op::Launch { kernel, inputs, out, .. } => {
                alloc(*out, &mut made, &mut body);
                // Integer weights: the integers where the input is, the
                // scales after the output.
                let (mut a, mut scales) = (Vec::new(), Vec::new());
                for v in inputs {
                    match rec.bufs[v.buf].store {
                        s @ Store::Quant { .. } => {
                            let (q, sc) = crate::structure::quant_ptrs(&bufnames[v.buf], s, crate::structure::Aff { a: v.off as i64, b: 0 }, "i");
                            a.push(q);
                            scales.push(sc);
                        }
                        _ => a.push(view(v, &bufnames)),
                    }
                }
                a.push(bufnames[*out].clone());
                a.extend(scales);
                body.push_str(&launch("    ", &format!("k{}", t.name_of[kernel]), &a, t.hi[kernel], t.workers[kernel]));
            }
            Op::Fill { buf, value } => {
                alloc(*buf, &mut made, &mut body);
                let rb = &rec.bufs[*buf];
                let _ = writeln!(body, "    presk_fill({}, {}, {}, {});", bufnames[*buf], cdtype(rb.store), c_real(*value), rb.len);
            }
            Op::Scalar { buf, x } => {
                alloc(*buf, &mut made, &mut body);
                let _ = writeln!(body, "    presk_fill({}, {}, {}, 1);", bufnames[*buf], cdtype(rec.bufs[*buf].store), c_scalar(*x));
            }
            Op::Copy { dst, at, src, n } => {
                alloc(*dst, &mut made, &mut body);
                let d = if *at == 0 { bufnames[*dst].clone() } else { format!("{} + {at}", bufnames[*dst]) };
                match rec.bufs[src.buf].store {
                    Store::Quant { bits, k, inner } => {
                        let b = &bufnames[src.buf];
                        let _ = writeln!(body, "    presk_dequantize({d}, {}, {b}.q, {b}.s, {bits}, {k}, {inner}, {}, {n});", cdtype(rec.bufs[*dst].store), src.off);
                    }
                    _ => {
                        let _ = writeln!(body, "    presk_convert({d}, {}, {}, {}, {n});", cdtype(rec.bufs[*dst].store), view(src, &bufnames), cdtype(rec.bufs[src.buf].store));
                    }
                }
            }
            Op::Hyper { out, op, a, b } => {
                let _ = writeln!(body, "    double s{out} = {} {} {};", c_scalar(*a), op.c(), c_scalar(*b));
                if !read_vars[*out] {
                    let _ = writeln!(body, "    (void)s{out};");
                }
            }
            Op::Read { out, src } => {
                let e = format!("{}[{}]", bufnames[src.buf], src.off);
                let e = match rec.bufs[src.buf].store {
                    Store::F64 => e,
                    Store::F32 | Store::I64 => format!("(double){e}"),
                    Store::Bf16 => format!("(double)presk_bf16_to_f32({e})"),
                    Store::Quant { .. } => unreachable!("a host scalar is read from a kernel's output, never from integer weights"),
                };
                let _ = writeln!(body, "    double s{out} = {e};");
                if !read_vars[*out] {
                    let _ = writeln!(body, "    (void)s{out};");
                }
            }
            Op::Sample { out, src, rows, vocab, key } => {
                alloc(*out, &mut made, &mut body);
                let field = f.keys.iter().find(|k| k.n == *key).map_or("", |k| k.field.as_str());
                let _ = writeln!(body, "    presk_sample({}, {}, {}, {rows}, {vocab}, &in->{field});", bufnames[*out], view(src, &bufnames), cdtype(rec.bufs[src.buf].store));
            }
        }
    }
    for (fld, v) in outs {
        let _ = writeln!(body, "    out->{fld} = {};", view(v, &bufnames));
    }
    let unread = !f.ins.iter().any(|i| read_bufs[i.buf]) && f.dbls.is_empty() && !samples(rec);
    format!("{sig}\n{{\n{}{body}}}\n", if unread { "    (void)in;\n" } else { "" })
}

fn export_with(meta: &Meta, kernels: &[Kernel], entries: &[(&str, &Recording)], args: &[Arg], weights_type: Option<&str>, host: Option<&Structured>, serve: Option<&Serve>) -> Result<Exported> {
    let rec = entries[0].1;
    if args.len() != rec.args.len() {
        return Err(format!("{} argument names for {} arguments", args.len(), rec.args.len()));
    }
    let recs: Vec<&Recording> = entries.iter().map(|(_, r)| *r).collect();
    let funcs: Vec<String> = entries.iter().map(|(e, _)| format!("presk_{}", c_ident(e))).collect();
    let func = funcs[0].clone();
    let dead = rec.ops.iter().filter(|o| matches!(o, Op::Launch { .. })).count() - live_launches(rec).len();
    let sampling: Vec<bool> = recs.iter().map(|r| samples(r)).collect();
    // A key argument is a field when an entry draws from it.
    let mut sampled = vec![false; rec.key_args.len()];
    for op in recs.iter().flat_map(|r| &r.ops) {
        if let Op::Sample { key, .. } = op
            && let Some(x) = sampled.get_mut(*key)
        {
            *x = true;
        }
    }

    // ----- kernels, deduplicated by text ---------------------------------
    let t = kernel_table(kernels, &recs, meta.threads)?;
    let (launches, split) = t.counts(rec);
    let split_any: usize = recs.iter().map(|r| t.counts(r).1).sum();
    let max_workers = t.workers.values().copied().max().unwrap_or(1);
    let names_list = entries.iter().map(|(e, _)| format!("`{e}`")).collect::<Vec<_>>().join(" and ");
    let mut kernels_c = format!(
        "/* kernels.c: the kernels of {} ({}), one C function each, deduplicated,\n * each under its PSK-Kernel text (presk dump --stage kernel). Included by\n * model.c. Printed by presk-export-c (plan 0.9.0 §3.2, presk D77). */\n\n{KERNEL_DECL}",
        names_list, meta.file
    );
    let mut kernel_lines = Vec::with_capacity(t.text.len());
    for (n, text) in t.text.iter().enumerate() {
        let first = kernels_c.matches('\n').count() + 1;
        let text = text.replace("static void KERNEL(", &format!("PRESK_KERNEL void k{n}("));
        kernel_lines.push((first, first + text.lines().count().max(1) - 1));
        kernels_c.push_str(&text);
        kernels_c.push('\n');
    }
    let mut kernel_of: Vec<(usize, usize)> = t.name_of.iter().map(|(k, n)| (*k, *n)).collect();
    kernel_of.sort();

    // ----- the arguments and the result, the same for every entry -------
    let fields: Vec<Fields> = recs.iter().map(|r| fields_of(r, args, &meta.quant, &sampled)).collect::<Result<_>>()?;
    let outs: Vec<Vec<(String, View)>> = recs.iter().map(|r| outs_of(r)).collect::<Result<_>>()?;
    for k in 1..entries.len() {
        if fields[k].decl != fields[0].decl || out_decl(recs[k], &outs[k]) != out_decl(rec, &outs[0]) {
            return Err(format!("`{}` takes or returns other types than `{}`", entries[k].0, entries[0].0));
        }
    }
    let Fields { ins, dbls, keys, fixed, decl } = &fields[0];
    let outs_named = &outs[0];

    // ----- model.h ------------------------------------------------------
    let arena_in: usize = ins.iter().map(|i| input_bytes(rec, i)).sum();
    let flat_host: usize = recs.iter().map(|r| r.bufs.iter().filter(|b| !matches!(b.made, Made::Input { .. })).map(|b| aligned(b.len * b.store.bytes())).sum::<usize>()).max().unwrap_or(0);
    let arena_host = match host {
        Some(s) => s.arena(),
        None => flat_host,
    };
    let arena = arena_in + arena_host;
    let (functions, loops, folds) = match host {
        Some(s) => {
            let (loops, notes) = s.folds();
            (s.funcs.len(), loops, notes)
        }
        None => (0, 0, Vec::new()),
    };
    let upper = c_ident(&meta.entry).to_uppercase();
    let several = if entries.len() > 1 { format!(" and {}", entries[1..].iter().map(|(e, _)| format!("`{e}`")).collect::<Vec<_>>().join(", ")) } else { String::new() };
    let mut h = String::new();
    let _ = writeln!(h, "/* model.h: `{}`{several} of {} as C functions (presk export --target c).", meta.entry, meta.file);
    let _ = writeln!(h, " * {}. Realized in {}. Generated by presk {}; do not edit. */", dims_line(meta), meta.realize, meta.version);
    let _ = writeln!(h, "#ifndef PRESK_{upper}_H\n#define PRESK_{upper}_H\n\n#include \"presk_rt.h\"\n");
    let peak = if entries.len() > 1 { "the peak of the entry that reaches the highest" } else { "the entry's peak" };
    match host {
        Some(_) => {
            let _ = writeln!(h, "/* The arena's size: the inputs, {arena_in} bytes, and {peak},\n * {arena_host} bytes: its outputs and the deepest chain of frames, each\n * allocation rounded to {ALIGN}. Every function releases what it allocated\n * when it returns (plan 0.9.0 step 4); released nothing, the same call\n * would take {flat_host} bytes. */");
        }
        None => {
            let _ = writeln!(h, "/* The arena's size: the inputs, {arena_in} bytes, and every buffer the host\n * makes, {arena_host} bytes, each rounded to {ALIGN}. Nothing is released\n * during a call (plan 0.9.0 step 3). */");
        }
    }
    let _ = writeln!(h, "#define PRESK_{upper}_ARENA ((size_t){arena})\n");
    let scalars = if dbls.is_empty() { "" } else { " An f64 argument is a\n * double, which the caller sets (plan 0.9.0 step 6)." };
    let keyed = if keys.is_empty() { "" } else { " A key `sample` draws\n * from is a presk_key, which the caller sets (plan 0.10.0 step 7b)." };
    let integers = if ins.iter().any(|i| i.quant.is_some()) { " Integer weights are a presk_q8 or a\n * presk_q4, their integers and their scales (plan 0.9.0 step 7)." } else { "" };
    let _ = writeln!(h, "/* The entry's arguments, leaf by leaf, each tensor in the type the\n * realization stores it in.{integers}{scalars}{keyed} */\ntypedef struct {{");
    for d in decl {
        let _ = writeln!(h, "{d}");
    }
    if decl.is_empty() {
        let _ = writeln!(h, "    char none; /* the entry takes no tensor */");
    }
    let _ = writeln!(h, "}} {func}_in;\n\n/* Its result, leaf by leaf: pointers into the arena, or into the inputs. */\ntypedef struct {{");
    for d in out_decl(rec, outs_named) {
        let _ = writeln!(h, "{d}");
    }
    let _ = writeln!(h, "}} {func}_out;\n");
    if sampling[0] {
        let _ = writeln!(h, "/* One call of `{}`: every launch of the native backend's host, in order,\n * each buffer bump-allocated from `arena`. Its `sample` draws from a key of\n * `in`, one uniform per row: given the key the native backend is given, it\n * draws its tokens (plan 0.10.0 step 7b). */", meta.entry);
    } else {
        let _ = writeln!(h, "/* One call of `{}`: every launch of the native backend's host, in order,\n * each buffer bump-allocated from `arena`. */", meta.entry);
    }
    let _ = writeln!(h, "{};", signature(&func, &func));
    for k in 1..entries.len() {
        let what = match serve {
            Some(s) if entries[k].0 == s.greedy => format!("`{}` with every `sample` an `argmax`, the function `presk serve`\n * calls at temperature 0 (with temperature 1)", entries[0].0),
            _ => "another entry".to_string(),
        };
        let _ = writeln!(h, "\n/* `{}`: {what}. The same arguments and result. */\n{};", entries[k].0, signature(&funcs[k], &func));
    }
    let _ = writeln!(h, "\n#endif");

    // ----- model.c ------------------------------------------------------
    let mut model_c = String::new();
    match host {
        None => {
            let _ = writeln!(model_c, "/* model.c: `{}`{several} of {}, straight-line (plan 0.9.0 step 3): every", meta.entry, meta.file);
            let _ = writeln!(model_c, " * operation of the native backend's host in order, {launches} launches of {} kernels.", t.text.len());
            let _ = writeln!(model_c, " * A comment names the site of the operations below it, \"<function>/%<let>\".");
            let _ = writeln!(model_c, " * Generated by presk {}; do not edit. */\n", meta.version);
            let _ = writeln!(model_c, "{MODEL_HEAD}#include \"model.h\"\n\n{INCLUDE_KERNELS}");
            for (k, r) in recs.iter().enumerate() {
                if k > 0 {
                    model_c.push('\n');
                }
                model_c.push_str(&flat_fn(&signature(&funcs[k], &func), r, &fields[k], &outs[k], &t));
            }
        }
        Some(s) => {
            let _ = writeln!(model_c, "/* model.c: `{}`{several} of {}, structured (plan 0.9.0 step 4): a C function", meta.entry, meta.file);
            let _ = writeln!(model_c, " * per distinct recording of a Presk function, callees first, then the");
            let _ = writeln!(model_c, " * entry's; a fold is a loop. {launches} launches of {} kernels, in the order", t.text.len());
            let _ = writeln!(model_c, " * the native backend makes them. A comment names the site of the operations");
            let _ = writeln!(model_c, " * below it, \"<function>/%<let>\". Every function releases what it allocated");
            let _ = writeln!(model_c, " * when it returns; what it returns is written into buffers its caller");
            let _ = writeln!(model_c, " * allocated. Generated by presk {}; do not edit. */\n", meta.version);
            let _ = writeln!(model_c, "{MODEL_HEAD}#include \"model.h\"\n\n{INCLUDE_KERNELS}");
            let in_fields: Vec<String> = ins.iter().map(|i| i.field.clone()).collect();
            let in_doubles: Vec<String> = dbls.iter().map(|d| d.field.clone()).collect();
            let out_fields: Vec<String> = outs_named.iter().map(|(f, _)| f.clone()).collect();
            let in_keys: Vec<Option<String>> = (0..rec.key_args.len()).map(|n| keys.iter().find(|k| k.n == n).map(|k| k.field.clone())).collect();
            let ios: Vec<EntryIo> = funcs.iter().map(|f| EntryIo { func: f, types: &func, in_fields: &in_fields, in_doubles: &in_doubles, out_fields: &out_fields, in_keys: &in_keys }).collect();
            model_c.push_str(&s.print(&ios));
        }
    }

    // ----- main.c -------------------------------------------------------
    let weights: Vec<&In> = ins.iter().filter(|i| i.weights).collect();
    let others: Vec<&In> = ins.iter().filter(|i| !i.weights).collect();
    let weights_file = if weights.is_empty() {
        None
    } else {
        let Some(ty) = weights_type else { return Err("the weights' type is not given".into()) };
        Some(DataFile { ty: ty.to_string(), leaves: weights.iter().map(|i| (i.bound, rec.bufs[i.buf].len)).collect() })
    };
    let input_views: Vec<View> = others.iter().map(|i| View { buf: i.buf, off: 0, shape: shape_of(&rec.args, i.buf), bound: (i.bound > 0).then_some(i.bound) }).collect();
    let inputs_file = (!others.is_empty()).then(|| DataFile { ty: closed_type(&input_views), leaves: others.iter().map(|i| (i.bound, rec.bufs[i.buf].len)).collect() });
    let out_views: Vec<View> = outs_named.iter().map(|(_, v)| v.clone()).collect();
    let outputs = DataFile { ty: closed_type(&out_views), leaves: out_views.iter().map(|v| (v.bound.unwrap_or(0), v.numel())).collect() };

    let mut files_args: Vec<(&str, &str)> = Vec::new();
    if weights_file.is_some() {
        files_args.push(("WEIGHTS", "weights"));
    }
    if inputs_file.is_some() {
        files_args.push(("INPUTS", "inputs"));
    }
    let usage = files_args.iter().map(|(u, _)| format!("{u}.psk.d")).chain(std::iter::once("OUT.psk.d".to_string())).collect::<Vec<_>>().join(" ");
    let opts = if keys.is_empty() { "[-r R]" } else { "[-r R] [-s SEED]" };
    let mut m = String::new();
    let seeded = if keys.is_empty() {
        String::new()
    } else {
        format!(" Its draws come from the\n * keys of its arguments, which it derives from the seed given with -s, {} by\n * default, the one it was exported with, as presk run --seed does (plan\n * 0.10.0 step 7b): every call computes the same outputs.", meta.seed)
    };
    let _ = writeln!(m, "/* main.c: reads the inputs of `{}`, runs it and writes its outputs, as\n * .psk.d files (presk-data's format). With -r R, runs it R times, the arena\n * released between calls, and prints the median and the best wall time on\n * stderr (plan 0.9.0 step 5); the outputs written are the last call's.{seeded}\n * Generated by presk {}; do not edit.\n *\n *   usage: model {opts} {usage}\n */\n", meta.entry, meta.version);
    let _ = writeln!(m, "#include \"model.h\"\n\n#include <stdlib.h>\n#include <time.h>\n");
    let _ = writeln!(m, "/* Milliseconds of wall time, by C11's clock. */\nstatic double now_ms(void)\n{{\n    struct timespec ts;\n    timespec_get(&ts, TIME_UTC);\n    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;\n}}\n");
    let _ = writeln!(m, "static int ascending(const void *a, const void *b)\n{{\n    double x = *(const double *)a, y = *(const double *)b;\n    return (x > y) - (x < y);\n}}\n");
    let tables = |m: &mut String, with_outputs: bool| {
        let table = |m: &mut String, name: &str, f: &DataFile, stores: &[Store]| {
            let _ = writeln!(m, "/* {} */\nstatic const char {name}_TYPE[] = {};", f.ty.replace("*/", "* /"), c_string(&f.ty));
            let _ = writeln!(m, "static const presk_leaf {name}[] = {{{}}};", f.leaves.iter().map(|(b, n)| format!("{{{b}, {n}}}")).collect::<Vec<_>>().join(", "));
            let _ = writeln!(m, "static const presk_dtype {name}_DTYPES[] = {{{}}};\n", stores.iter().map(|s| cdtype(*s)).collect::<Vec<_>>().join(", "));
        };
        if let Some(f) = &weights_file {
            table(m, "WEIGHTS", f, &weights.iter().map(|i| rec.bufs[i.buf].store).collect::<Vec<_>>());
        }
        if with_outputs {
            if let Some(f) = &inputs_file {
                table(m, "INPUTS", f, &others.iter().map(|i| rec.bufs[i.buf].store).collect::<Vec<_>>());
            }
            table(m, "OUTPUTS", &outputs, &out_views.iter().map(|v| rec.bufs[v.buf].store).collect::<Vec<_>>());
        }
    };
    tables(&mut m, true);
    let _ = writeln!(m, "int main(int argc, char **argv)\n{{");
    let _ = writeln!(m, "    long repeat = 1;\n    int timed = 0;");
    if !keys.is_empty() {
        let _ = writeln!(m, "    uint64_t seed = {};", meta.seed);
    }
    let _ = writeln!(m, "    int a = 1;\n    for (; a + 1 < argc; a += 2) {{\n        if (strcmp(argv[a], \"-r\") == 0) {{\n            repeat = strtol(argv[a + 1], NULL, 10);\n            timed = 1;");
    if !keys.is_empty() {
        let _ = writeln!(m, "        }} else if (strcmp(argv[a], \"-s\") == 0) {{\n            seed = strtoull(argv[a + 1], NULL, 10);");
    }
    let _ = writeln!(m, "        }} else\n            break;\n    }}");
    let _ = writeln!(m, "    if (argc - a != {} || repeat < 1) {{\n        fprintf(stderr, \"usage: %s {opts} {usage}\\n\", argv[0]);\n        return 2;\n    }}", files_args.len() + 1);
    let _ = writeln!(m, "    presk_arena arena;\n    if (presk_arena_init(&arena, PRESK_{upper}_ARENA)) {{\n        fprintf(stderr, \"%s\\n\", presk_error());\n        return 1;\n    }}");
    let _ = writeln!(m, "    {func}_in in;");
    if decl.is_empty() {
        let _ = writeln!(m, "    in.none = 0;");
    }
    for d in dbls {
        let _ = writeln!(m, "    in.{} = {}; /* {}, as exported */", d.field, c_real(d.value), d.path.replace("*/", "* /"));
    }
    for k in keys {
        m.push_str(&key_c(k, &rec.key_args[k.n], args.len()));
    }
    for (k, (u, _)) in files_args.iter().enumerate() {
        let group: &[&In] = if *u == "WEIGHTS" { &weights } else { &others };
        m.push_str(&load_c(group, rec, &u.to_lowercase(), u, &format!("argv[a + {k}]")));
    }
    let _ = writeln!(m, "    double *ms = malloc((size_t)repeat * sizeof *ms);\n    if (!ms) {{\n        fprintf(stderr, \"-r %ld: not that much memory\\n\", repeat);\n        return 1;\n    }}");
    let _ = writeln!(m, "    {func}_out out;\n    size_t mark = presk_mark(&arena);\n    for (long r = 0; r < repeat; r++) {{\n        presk_release(&arena, mark);");
    let _ = writeln!(m, "        double t0 = now_ms();\n        {func}(&arena, &in, &out);");
    let _ = writeln!(m, "        ms[r] = now_ms() - t0;\n    }}");
    let _ = writeln!(m, "    if (timed) {{\n        qsort(ms, (size_t)repeat, sizeof *ms, ascending);\n        fprintf(stderr, \"{func}: %ld call%s, median %.3f ms, best %.3f ms\\n\", repeat, repeat == 1 ? \"\" : \"s\", ms[repeat / 2], ms[0]);\n    }}\n    free(ms);");
    let _ = writeln!(m, "    const void *outputs[] = {{{}}};", outs_named.iter().map(|(f, _)| format!("out.{f}")).collect::<Vec<_>>().join(", "));
    let n = files_args.len();
    let _ = writeln!(m, "    if (presk_write(argv[a + {n}], OUTPUTS_TYPE, OUTPUTS, {}, 1, outputs, OUTPUTS_DTYPES)) {{\n        fprintf(stderr, \"%s: %s\\n\", argv[a + {n}], presk_error());\n        return 1;\n    }}", outs_named.len());
    let _ = writeln!(m, "    presk_arena_free(&arena);\n    return 0;\n}}");

    // ----- generate.c ---------------------------------------------------
    let generate = match serve {
        Some(s) => {
            let mut g = String::new();
            tables(&mut g, false);
            Some(generate_c(meta, s, &funcs, &func, &upper, ins, dbls, keys, &weights, rec, &g)?)
        }
        None => None,
    };

    // ----- Makefile, README, conformance script -------------------------
    let strict: Vec<&str> = CFLAGS.iter().copied().filter(|f| *f != "-O2").collect();
    let openmp = if split_any > 0 {
        format!("# OPENMP: {split} of the {launches} launches are divided among threads, each an\n# OpenMP region (plan 0.9.0 §3.6). `make OPENMP=` builds without OpenMP:\n# one thread, the same bits.\nOPENMP = {OPENMP}")
    } else {
        format!("# OPENMP: no launch is divided among threads, so the program needs no\n# OpenMP; `make OPENMP={OPENMP}` builds with it, to the same program.\nOPENMP =")
    };
    let sources = "model.c model.h kernels.c presk_rt.c presk_rt.h presk_libm.c";
    let (all, gen_rule, gen_clean) = if generate.is_some() {
        ("all: model generate\n\n", format!("\ngenerate: generate.c {sources}\n\t$(CC) $(STRICT) $(CFLAGS) $(OPENMP) -o $@ generate.c model.c presk_rt.c presk_libm.c -lm\n"), " generate")
    } else {
        ("", String::new(), "")
    };
    let makefile = format!(
        "# Builds `{entry}` (presk export --target c). Generated by presk {version}.\n#\n# STRICT holds the flags bit identity with the native backend rests on\n# (plan 0.9.0 §3.5): ISO C11 and no contraction of a product and a sum\n# into one rounding. Never add -ffast-math or any of its parts to CFLAGS.\n\nSTRICT = {strict}\nCFLAGS = -O2 {warn}\n{openmp}\n\n{all}model: main.c {sources}\n\t$(CC) $(STRICT) $(CFLAGS) $(OPENMP) -o $@ main.c model.c presk_rt.c presk_libm.c -lm\n{gen_rule}\ncheck: model\n\tsh conformance/check.sh\n\nclean:\n\trm -f model{gen_clean} conformance/out.psk.d\n\n.PHONY: {}check clean\n",
        if generate.is_some() { "all " } else { "" },
        entry = meta.entry,
        version = meta.version,
        strict = strict.join(" "),
        warn = WARNINGS.join(" "),
    );
    let run_args: Vec<String> = files_args
        .iter()
        .map(|(u, f)| match (*u, &meta.weights) {
            ("WEIGHTS", Some(w)) => shell_quote(w),
            _ => format!("conformance/{f}.psk.d"),
        })
        .collect();
    let check = format!(
        "#!/bin/sh\n# Builds the program, runs it on the conformance inputs and compares its\n# outputs with the native backend's, bit for bit: statement C1 of plan\n# 0.9.0. With presk on the PATH (or in $PRESK), also the distance to the\n# exact reference (C2 is reported by presk's tests).\nset -e\ncd \"$(dirname \"$0\")/..\"\nmake -s model\n./model {} conformance/out.psk.d\nif cmp -s conformance/out.psk.d conformance/native.psk.d; then\n    echo \"C1 holds: the outputs are the native backend's, bit for bit\"\nelse\n    echo \"C1 fails: the outputs differ from the native backend's\" >&2\n    cmp conformance/out.psk.d conformance/native.psk.d >&2 || true\n    exit 1\nfi\nPRESK=${{PRESK:-presk}}\nif [ -f conformance/exact.psk.d ] && command -v \"$PRESK\" >/dev/null 2>&1; then\n    echo \"against the exact reference:\"\n    \"$PRESK\" data diff conformance/exact.psk.d conformance/out.psk.d\nfi\n",
        run_args.join(" ")
    );
    let shape = host.map(|_| HostShape { functions, loops, folds: &folds, flat: arena_in + flat_host });
    let others_doc: Vec<(String, String, usize)> = (1..entries.len()).map(|k| (entries[k].0.to_string(), funcs[k].clone(), t.counts(recs[k]).0)).collect();
    let doc = Doc {
        func: &func,
        ins: ins.iter().map(|i| (i.path.clone(), i.ty.clone(), i.weights, i.quant.as_ref().map(|q| q.describe()))).collect(),
        dbls: dbls.iter().map(|d| (d.path.clone(), d.field.clone(), d.value)).collect(),
        fixed,
        outs: outs_named,
        counts: (arena, launches, dead, t.text.len()),
        split: (split_any, max_workers),
        shape,
        usage: &format!("{opts} {usage}"),
        run_args: &run_args,
        weights: weights_file.as_ref(),
        inputs: inputs_file.as_ref(),
        outputs: &outputs,
        samples: sampling[0],
        others: others_doc,
        serve,
    };
    let readme = readme(meta, &doc);

    let mut files = vec![
        ("model.h".to_string(), h),
        ("model.c".to_string(), model_c),
        ("kernels.c".to_string(), kernels_c),
        ("main.c".to_string(), m),
        ("Makefile".to_string(), makefile),
        ("README.md".to_string(), readme),
        ("conformance/check.sh".to_string(), check),
    ];
    if let Some(g) = generate {
        files.push(("generate.c".to_string(), g));
    }
    for (name, text) in RUNTIME {
        files.push((name.to_string(), text.to_string()));
    }
    let quantized = ins.iter().filter(|i| i.quant.is_some()).count();
    Ok(Exported { files, inputs: inputs_file, weights: weights_file, outputs, arena, arena_flat: arena_in + flat_host, functions, loops, folds, launches, dead, kernels: t.text.len(), split, max_workers, quantized, kernel_of, kernel_lines })
}

/// A string as a JSON string literal, as `serde_json` writes one: `"` and
/// `\` escaped, the control characters as `\n`, `\t`, … or `\u00XX`,
/// everything else as it is.
pub(crate) fn json_string(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The C of `main` and `generate` that reads a group of leaves from the file
/// `path` (a C expression) into the arena, through the array of pointers
/// `bufs` and the tables `<u>`, `<u>_TYPE` and `<u>_DTYPES`. A leaf of
/// integer weights is read as its reals into memory of its own, then
/// quantized into the arena, as `presk run` quantizes a checkpoint it
/// loads (step 7).
fn load_c(group: &[&In], rec: &Recording, bufs: &str, u: &str, path: &str) -> String {
    let mut m = String::new();
    let _ = writeln!(m, "    void *{bufs}[{}];", group.len());
    for (j, i) in group.iter().enumerate() {
        let rb = &rec.bufs[i.buf];
        match &i.quant {
            Some(_) => {
                let _ = writeln!(m, "    {bufs}[{j}] = malloc({} * sizeof(double)); /* {}: its reals, quantized below */\n    if (!{bufs}[{j}]) {{\n        fprintf(stderr, \"%s: not that much memory\\n\", {});\n        return 1;\n    }}", rb.len, i.path.replace("*/", "* /"), c_string(&i.path));
            }
            None => {
                let _ = writeln!(m, "    {bufs}[{j}] = presk_alloc(&arena, {} * sizeof({}));\n    in.{} = {bufs}[{j}];", rb.len, ctype(rb.store), i.field);
            }
        }
    }
    let _ = writeln!(m, "    if (presk_load({path}, {u}_TYPE, {u}, {}, {bufs}, {u}_DTYPES)) {{\n        fprintf(stderr, \"%s: %s\\n\", {path}, presk_error());\n        return 1;\n    }}", group.len());
    for (j, i) in group.iter().enumerate() {
        let Some(q) = &i.quant else { continue };
        let (ints, scales) = q.bytes();
        let (int_t, scheme) = (if q.bits == 8 { "int8_t" } else { "uint8_t" }, if q.scheme == "pow2" { "PRESK_POW2" } else { "PRESK_ABSMAX" });
        let _ = writeln!(m, "    /* {}: {}, quantized from its reals as presk-rt quantizes a\n     * checkpoint it loads, its scales computed from them (presk D44, D84). */", i.path.replace("*/", "* /"), q.describe());
        let _ = writeln!(m, "    {{\n        {int_t} *q = presk_alloc(&arena, {ints});\n        float *s = presk_alloc(&arena, {scales} * sizeof(float));\n        presk_quantize(q, s, {bufs}[{j}], {}, {}, {}, {}, {scheme});\n        in.{} = ({}){{q, s}};\n        free({bufs}[{j}]);\n    }}", q.outer, q.k, q.inner, q.bits, i.field, ctype(rec.bufs[i.buf].store));
    }
    m
}

/// The C of `main` that sets the key field `k` as `presk run --seed`
/// synthesizes it (plan 0.10.0 step 7b): the key the seed names, split
/// among the entry's `nargs` arguments, then down the argument's tuples.
fn key_c(k: &KeyIn, a: &presk_kernel::record::KeyArg, nargs: usize) -> String {
    let mut m = String::new();
    let _ = writeln!(m, "    /* {}: the key presk run --seed synthesizes, argument {} of {nargs}. */
    {{
        presk_key k;
        presk_key_from_seed(&k, seed);
        presk_key_split(&k, {}, &k);", k.path.replace("*/", "* /"), a.arg + 1, a.arg);
    for (_, pos) in &a.path {
        let _ = writeln!(m, "        presk_key_split(&k, {pos}, &k);");
    }
    let _ = writeln!(m, "        in.{} = k;
    }}", k.field);
    m
}

/// `generate.c` (step 6): `presk serve`'s loop over the exported entry and
/// its greedy copy, its output `presk serve`'s, `--json` included.
#[allow(clippy::too_many_arguments)]
fn generate_c(meta: &Meta, s: &Serve, funcs: &[String], types: &str, upper: &str, ins: &[In], dbls: &[Dbl], keys: &[KeyIn], weights: &[&In], rec: &Recording, tables: &str) -> Result<String> {
    let Some(ctx) = ins.iter().find(|i| i.arg == s.ctx && !i.weights) else { return Err("the context is not an input of the entry".into()) };
    if rec.bufs[ctx.buf].len as u64 != s.window || ctx.bound != s.vocab {
        return Err(format!("the context `{}` is not {} tokens below {}", ctx.path, s.window, s.vocab));
    }
    if ins.iter().any(|i| !i.weights && i.arg != s.ctx) {
        return Err("the entry reads an input besides its parameters and its context".into());
    }
    let temp = match s.temperature {
        Some(a) => Some(dbls.iter().find(|d| d.arg == a).ok_or("the temperature is not an f64 argument of the entry")?),
        None => None,
    };
    if dbls.iter().any(|d| Some(d.arg) != s.temperature) {
        return Err("the entry takes an f64 besides its temperature".into());
    }
    // The key a token is given, when the function draws from one.
    let key = match s.key {
        Some(a) => keys.iter().find(|k| k.arg == a),
        None => None,
    };
    if keys.iter().any(|k| Some(k.arg) != s.key) {
        return Err("the entry draws from a key besides the one presk serve gives it".into());
    }
    let (sampled, greedy) = (&funcs[0], &funcs[1]);
    let mut g = String::new();
    let _ = writeln!(
        g,
        "/* generate.c: the generation loop of `presk serve` over `{entry}`\n * ({file}; plan 0.9.0 step 6, presk D83). The context is the last {w}\n * tokens, the window {model}'s parameters fix; each step computes one token\n * from it, appends it, and moves the window on. At temperature 0 the step\n * is {greedy}, the argmax copy, at temperature 1, as presk serve's\n * greedy path; above it, {sampled}, which draws from the key it is given:\n * token i (from 1) is given the child i of the key the seed names, as\n * presk serve gives it (plan 0.10.0 step 7b, presk D99). Its output is\n * presk serve's: the tokens on one line, or with --json the same JSON\n * document. Generated by presk {version}; do not edit.\n *\n *   usage: generate --weights W.psk.d --prompt \"t1 t2 ...\" [--tokens N]\n *                   [--temperature X] [--seed S] [--json]\n */\n",
        entry = meta.entry,
        file = meta.file,
        w = s.window,
        model = s.model,
        version = meta.version,
    );
    let _ = writeln!(g, "#include \"model.h\"\n\n#include <errno.h>\n#include <stdlib.h>\n");
    g.push_str(tables);
    let _ = writeln!(g, "#define WINDOW {}\n#define VOCAB {}\n", s.window, s.vocab);
    let _ = writeln!(g, "static int usage(const char *argv0)\n{{\n    fprintf(stderr, \"usage: %s --weights W.psk.d --prompt \\\"t1 t2 ...\\\" [--tokens N] [--temperature X] [--seed S] [--json]\\n\", argv0);\n    return 2;\n}}\n");
    let _ = writeln!(g, "/* A decimal natural number, the whole of s. */\nstatic int natural(const char *s, uint64_t *x)\n{{\n    char *end;\n    if (*s < '0' || *s > '9')\n        return 1;\n    errno = 0;\n    unsigned long long v = strtoull(s, &end, 10);\n    if (errno || *end)\n        return 1;\n    *x = (uint64_t)v;\n    return 0;\n}}\n");
    let _ = writeln!(g, "/* A list of tokens as serde_json prints one in presk serve's --json. */\nstatic void json_list(const char *key, const uint64_t *xs, size_t n, int last)\n{{\n    printf(\"  \\\"%s\\\": [\", key);\n    for (size_t i = 0; i < n; i++)\n        printf(\"%s\\n    %llu\", i ? \",\" : \"\", (unsigned long long)xs[i]);\n    printf(\"%s]%s\\n\", n ? \"\\n  \" : \"\", last ? \"\" : \",\");\n}}\n");
    let _ = writeln!(g, "/* A string as a JSON string, as serde_json escapes one. */\nstatic void json_string(const char *s)\n{{\n    putchar('\"');\n    for (; *s; s++) {{\n        unsigned char c = (unsigned char)*s;\n        if (c == '\"' || c == '\\\\')\n            printf(\"\\\\%c\", c);\n        else if (c == '\\n')\n            fputs(\"\\\\n\", stdout);\n        else if (c == '\\r')\n            fputs(\"\\\\r\", stdout);\n        else if (c == '\\t')\n            fputs(\"\\\\t\", stdout);\n        else if (c == '\\b')\n            fputs(\"\\\\b\", stdout);\n        else if (c == '\\f')\n            fputs(\"\\\\f\", stdout);\n        else if (c < 0x20)\n            printf(\"\\\\u%04x\", c);\n        else\n            putchar(c);\n    }}\n    putchar('\"');\n}}\n");
    let _ = writeln!(g, "int main(int argc, char **argv)\n{{");
    let _ = writeln!(g, "    const char *weights = NULL, *prompt = NULL;\n    uint64_t tokens = 16, seed = {};\n    double temperature = {};\n    int json = 0;", meta.seed, c_real(s.default_temperature));
    let _ = writeln!(g, "    for (int a = 1; a < argc; a++) {{\n        const char *o = argv[a];\n        if (strcmp(o, \"--json\") == 0) {{\n            json = 1;\n            continue;\n        }}\n        if (a + 1 == argc)\n            return usage(argv[0]);\n        const char *v = argv[++a];\n        if (strcmp(o, \"--weights\") == 0)\n            weights = v;\n        else if (strcmp(o, \"--prompt\") == 0)\n            prompt = v;\n        else if (strcmp(o, \"--tokens\") == 0) {{\n            if (natural(v, &tokens))\n                return usage(argv[0]);\n        }} else if (strcmp(o, \"--seed\") == 0) {{\n            if (natural(v, &seed))\n                return usage(argv[0]);\n        }} else if (strcmp(o, \"--temperature\") == 0) {{\n            char *end;\n            temperature = strtod(v, &end);\n            if (end == v || *end)\n                return usage(argv[0]);\n        }} else\n            return usage(argv[0]);\n    }}\n    if (!weights || !prompt)\n        return usage(argv[0]);");
    let _ = writeln!(g, "    if (temperature != temperature || temperature < 0) {{\n        fprintf(stderr, \"temperature %g: must be >= 0 (0 is greedy)\\n\", temperature);\n        return 1;\n    }}\n    int greedy = temperature == 0;");
    let _ = writeln!(g, "    /* The prompt, then the tokens generated. */\n    size_t len = 0, cap = strlen(prompt) / 2 + 1 + (size_t)tokens;\n    uint64_t *toks = malloc(cap * sizeof *toks);\n    if (!toks) {{\n        fprintf(stderr, \"--tokens %llu: not that much memory\\n\", (unsigned long long)tokens);\n        return 1;\n    }}\n    for (const char *p = prompt; *p;) {{\n        while (*p == ' ' || *p == '\\t' || *p == '\\n')\n            p++;\n        if (!*p)\n            break;\n        size_t n = strcspn(p, \" \\t\\n\");\n        char word[32] = {{0}};\n        memcpy(word, p, n < sizeof word - 1 ? n : sizeof word - 1);\n        if (n >= sizeof word || natural(word, &toks[len])) {{\n            fprintf(stderr, \"--prompt takes token indices separated by spaces, not `%.*s`\\n\", (int)n, p);\n            return 2;\n        }}\n        len++;\n        p += n;\n    }}\n    size_t prompt_len = len;");
    let _ = writeln!(g, "    if (prompt_len == 0) {{\n        fprintf(stderr, \"empty prompt: give at least one token\\n\");\n        return 1;\n    }}\n    if (prompt_len < WINDOW) {{\n        fprintf(stderr, \"the prompt has %zu token%s but the model's parameters fix the context length {seq} = %d: give at least %d tokens\\n\", prompt_len, prompt_len == 1 ? \"\" : \"s\", WINDOW, WINDOW);\n        return 1;\n    }}\n    for (size_t i = 0; i < prompt_len; i++)\n        if (toks[i] >= VOCAB) {{\n            fprintf(stderr, \"token %llu is not below the vocabulary %d\\n\", (unsigned long long)toks[i], VOCAB);\n            return 1;\n        }}", seq = s.seq_dim);
    let _ = writeln!(g, "    presk_arena arena;\n    if (presk_arena_init(&arena, PRESK_{upper}_ARENA)) {{\n        fprintf(stderr, \"%s\\n\", presk_error());\n        return 1;\n    }}\n    {types}_in in;");
    if weights.is_empty() {
        let _ = writeln!(g, "    (void)weights;");
    } else {
        g.push_str(&load_c(weights, rec, "weights_bufs", "WEIGHTS", "weights"));
    }
    let _ = writeln!(g, "    int64_t *ctx = presk_alloc(&arena, WINDOW * sizeof(int64_t));\n    in.{} = ctx;", ctx.field);
    if key.is_some() {
        let _ = writeln!(g, "    /* The serve's key: the one the seed names. */\n    presk_key serve_key;\n    presk_key_from_seed(&serve_key, seed);");
    } else {
        let _ = writeln!(g, "    (void)seed;");
    }
    let _ = writeln!(g, "    {types}_out out;\n    size_t mark = presk_mark(&arena);\n    for (uint64_t step = 0; step < tokens; step++) {{\n        for (size_t i = 0; i < WINDOW; i++)\n            ctx[i] = (int64_t)toks[len - WINDOW + i];\n        presk_release(&arena, mark);");
    if let Some(k) = key {
        let _ = writeln!(g, "        presk_key_child(&serve_key, step + 1, &in.{});", k.field);
    }
    let _ = writeln!(g, "        if (greedy) {{");
    if let Some(d) = temp {
        let _ = writeln!(g, "            in.{} = 1.0;", d.field);
    }
    let _ = writeln!(g, "            {greedy}(&arena, &in, &out);\n        }} else {{");
    if let Some(d) = temp {
        let _ = writeln!(g, "            in.{} = temperature;", d.field);
    }
    let _ = writeln!(g, "            {sampled}(&arena, &in, &out);\n        }}\n        toks[len++] = (uint64_t)out.{}[0];\n    }}", outs_first(rec)?);
    let _ = writeln!(
        g,
        "    if (json) {{\n        printf(\"{{\\n\");\n        json_list(\"generated\", toks + prompt_len, len - prompt_len, 0);\n        printf(\"  \\\"greedy\\\": %s,\\n\", greedy ? \"true\" : \"false\");\n        printf(\"  \\\"model\\\": %s,\\n\", {model});\n        json_list(\"prompt\", toks, prompt_len, 0);\n        printf(\"  \\\"vocab\\\": %d,\\n  \\\"weights\\\": \", VOCAB);\n        json_string(weights);\n        printf(\",\\n  \\\"window\\\": %d\\n}}\\n\", WINDOW);\n    }} else {{\n        fprintf(stderr, \"generate: {m} from %s, prompt of %zu tokens, %zu generated, %s, context window %d\\n\", weights, prompt_len, len - prompt_len, greedy ? \"greedy\" : \"sampled\", WINDOW);\n        for (size_t i = prompt_len; i < len; i++)\n            printf(\"%s%llu\", i > prompt_len ? \" \" : \"\", (unsigned long long)toks[i]);\n        printf(\"\\n\");\n    }}\n    free(toks);\n    presk_arena_free(&arena);\n    return 0;\n}}",
        model = c_string(&json_string(&s.model)),
        m = s.model.replace('%', "%%").replace('"', "\\\""),
    );
    Ok(g)
}

/// The field of the entry's result, when it is one tensor.
fn outs_first(rec: &Recording) -> Result<String> {
    let outs = outs_of(rec)?;
    match outs.as_slice() {
        [(f, v)] if v.bound.is_some() && v.numel() == 1 => Ok(f.clone()),
        _ => Err("the generation function does not return one token".into()),
    }
}

/// The shape of input buffer `buf`, from the arguments.
fn shape_of(args: &[Sym], buf: usize) -> Vec<usize> {
    args.iter().flat_map(leaves).find_map(|s| match s {
        Sym::Tensor(v) if v.buf == buf => Some(v.shape),
        _ => None,
    }).unwrap_or_default()
}

fn shell_quote(s: &str) -> String {
    if s.chars().all(|c| c.is_ascii_alphanumeric() || "/._-".contains(c)) { s.to_string() } else { format!("'{}'", s.replace('\'', "'\\''")) }
}

fn dims_line(meta: &Meta) -> String {
    if meta.dims.is_empty() {
        "No dimension".into()
    } else {
        format!("At {}", meta.dims.iter().map(|(n, v)| format!("{n} = {v}")).collect::<Vec<_>>().join(", "))
    }
}

/// The structured host's shape, for the README.
struct HostShape<'a> {
    functions: usize,
    loops: usize,
    folds: &'a [String],
    /// The straight-line host's arena.
    flat: usize,
}

/// What the README says of a program.
struct Doc<'a> {
    func: &'a str,
    /// Tensor arguments: `(path, type, from the weights, how quantized)`.
    ins: Vec<(String, String, bool, Option<String>)>,
    /// `f64` arguments: `(path, field, value exported with)`.
    dbls: Vec<(String, String, f64)>,
    fixed: &'a [(String, String)],
    outs: &'a [(String, View)],
    /// `(arena, launches, dead, kernels)`.
    counts: (usize, usize, usize, usize),
    /// `(divided launches, most workers)`.
    split: (usize, usize),
    shape: Option<HostShape<'a>>,
    usage: &'a str,
    run_args: &'a [String],
    weights: Option<&'a DataFile>,
    inputs: Option<&'a DataFile>,
    outputs: &'a DataFile,
    /// The entry draws from a key of its arguments.
    samples: bool,
    /// The other entries: `(Presk name, C function, launches)`.
    others: Vec<(String, String, usize)>,
    serve: Option<&'a Serve>,
}

fn readme(meta: &Meta, d: &Doc) -> String {
    let Doc { func, ins, dbls, fixed, outs, counts: (arena, launches, dead, kernels), split: (split, max_workers), shape, usage, run_args, weights, inputs, outputs, samples, others, serve } = d;
    let (arena, launches, dead, kernels, split, max_workers) = (*arena, *launches, *dead, *kernels, *split, *max_workers);
    let mut r = String::new();
    let _ = writeln!(r, "# `{}` as a C program\n", meta.entry);
    let _ = writeln!(r, "Exported by `presk export --target c` (presk {}) from `{}`{}. It computes the bits `presk run --backend native` computes with the same realization, dimensions, seed and weights (statement C1 of plan 0.9.0).\n", meta.version, meta.file, meta.commit.as_ref().map(|c| format!(", at commit `{c}` of its repository")).unwrap_or_default());
    let _ = writeln!(r, "## What was exported\n");
    let _ = writeln!(r, "- **Entry:** `{}`, the C function `{func}` (`model.h`).", meta.entry);
    for (e, f, n) in others {
        let what = match serve {
            Some(s) if *e == s.greedy => format!(": `{}` with every `sample` an `argmax`, the copy `presk serve` calls at temperature 0", meta.entry),
            _ => String::new(),
        };
        let _ = writeln!(r, "- **Also:** `{e}`, the C function `{f}`{what}, {n} launches. It takes the same arguments and returns the same result; `conformance/` checks `{}`.", meta.entry);
    }
    let _ = writeln!(r, "- **Dimensions:** {}.", if meta.dims.is_empty() { "none".to_string() } else { meta.dims.iter().map(|(n, v)| format!("`{n} = {v}`")).collect::<Vec<_>>().join(", ") });
    let _ = writeln!(r, "- **Constants:** {}.", if meta.consts.is_empty() { "the file's".to_string() } else { meta.consts.iter().map(|c| format!("`{c}`")).collect::<Vec<_>>().join(", ") + ", the others the file's" });
    let layout = match &meta.plan {
        Some(p) => format!("each nest in the layout the plan `{p}` states, the default layout elsewhere"),
        None => "the default layout of every nest".to_string(),
    };
    let _ = writeln!(r, "- **Realization:** `{}`, {layout}. Every buffer is stored in that type, indices as `int64_t`, `bf16` as `uint16_t` computed in `float`.", meta.realize);
    let _ = writeln!(r, "- **Seed:** {} (the inputs `presk run --seed {}` synthesizes, in `conformance/`{}).", meta.seed, meta.seed, if *samples { ", and the keys `sample` draws from" } else { "" });
    if !fixed.is_empty() {
        let _ = writeln!(r, "- **Fixed at export**, as `presk run` synthesizes them: {}. A `.psk.d` holds no scalar, so they are constants of the program, like the dimensions.", fixed.iter().map(|(p, v)| format!("`{p}` = {v}")).collect::<Vec<_>>().join("; "));
    }
    if !dbls.is_empty() {
        let list = dbls.iter().map(|(p, f, v)| format!("`{p}` (`in.{f}`, {v:?})")).collect::<Vec<_>>().join(", ");
        let what = if dbls.len() == 1 { format!("a `double` of `{func}_in`, which the caller sets; `main` sets it to the value it was exported with") } else { format!("`double`s of `{func}_in`, which the caller sets; `main` sets them to the values they were exported with") };
        let _ = writeln!(r, "- **Set at run time:** {list}: {what}, the ones `presk run` synthesizes (the defaults).");
    }
    let dead = if dead == 0 { String::new() } else { format!(" {dead} of them compute values the result does not depend on, as the native backend computes them (the forward pass a gradient runs before its reverse pass, and discards).") };
    match shape {
        None => {
            let _ = writeln!(r, "- **Host:** straight-line, {launches} launches of {kernels} kernels, in the order the native backend launches them.{dead} One arena of {arena} bytes holds the inputs and every buffer; nothing is released during a call.\n");
        }
        Some(s) => {
            let _ = writeln!(r, "- **Host:** structured, {launches} launches of {kernels} kernels, in the order the native backend launches them.{dead} A C function per distinct recording of a Presk function ({} besides the entry's), a fold as a loop ({} loop{}). Each function writes its result into buffers its caller allocated, and releases what it allocated when it returns: one arena of {arena} bytes holds the inputs and the peak of a call, where releasing nothing would take {} bytes.", s.functions, s.loops, if s.loops == 1 { "" } else { "s" }, s.flat);
            if !s.folds.is_empty() {
                let _ = writeln!(r, "- **Folds:**");
                for f in s.folds {
                    let _ = writeln!(r, "  - {f}.");
                }
                let _ = writeln!(r, "\n  An iteration whose recording differs from the next one's runs apart, as its own function; a fold whose iterations all differ is unrolled.");
            }
            r.push('\n');
        }
    }
    let _ = writeln!(r, "## Files\n");
    let _ = writeln!(r, "| file | holds |\n|---|---|");
    let _ = writeln!(r, "| `model.h`, `model.c` | the entry's function, `{func}`{}{}, and the arena's size |", others.iter().map(|(_, f, _)| format!(" and `{f}`")).collect::<String>(), if shape.is_some() { ", the functions they call" } else { "" });
    let _ = writeln!(r, "| `kernels.c` | its kernels, one C function each, under its PSK-Kernel text |");
    let _ = writeln!(r, "| `main.c` | reads the inputs, runs the entry, writes the outputs |");
    if serve.is_some() {
        let _ = writeln!(r, "| `generate.c` | the generation loop of `presk serve` over the entry and its greedy copy |");
    }
    let _ = writeln!(r, "| `presk_rt.h`, `presk_rt.c` | the runtime: conversions between machine types, the `.psk.d` reader and writer, the arena{} |", if *samples { ", the keys `sample` draws from and `sample` itself" } else { "" });
    let _ = writeln!(r, "| `presk_libm.c` | the transcendental functions, musl 1.1.22's, those of the native backend |");
    let _ = writeln!(r, "| `conformance/` | the inputs as `.psk.d`, the native backend's outputs (`native.psk.d`), {}and `check.sh` |\n", if meta.exact { "the exact reference's (`exact.psk.d`, rounded to `f64`), " } else { "" });
    let _ = writeln!(r, "## Build and run\n\n```sh\nmake\n./model {usage}\nmake check    # C1: ./model {} conformance/out.psk.d, compared with conformance/native.psk.d\n```\n", run_args.join(" "));
    let _ = writeln!(r, "**The flags that must not change** (`STRICT` in the `Makefile`): `-std=c11 -ffp-contract=off`. The kernels round every product before they add it, as the native backend does; a compiler allowed to contract `a + b*c` into one fused multiply-add changes the bits, and GCC does by default in its GNU modes. Never build with `-ffast-math` or any of its parts. On x86, the program needs SSE arithmetic (any x86-64), not the x87.\n");
    let _ = writeln!(r, "**The kernels are inlined** into the host's functions, always (`PRESK_KERNEL` in `kernels.c`), and on GCC `model.c` is compiled without the scheduling that runs before register allocation (the `#pragma GCC optimize` at its head). A product's inner loop holds about as many values as the machine has vector registers; kept out of line, or scheduled so, it can put one of its sums on the stack, and the product then takes about 30 % longer (presk D97). Neither changes a bit.\n");
    let _ = writeln!(r, "**Timing:** `./model -r R {}` runs the entry `R` times, the arena released between calls, prints the median and the best wall time on standard error, and writes the last call's outputs.\n", run_args.iter().map(|s| s.as_str()).chain(std::iter::once("OUT.psk.d")).collect::<Vec<_>>().join(" "));
    if *samples {
        let _ = writeln!(r, "## Randomness\n");
        let _ = writeln!(r, "`sample(key, logits)` draws a token per row of its logits from its key, which is an argument of the entry: a `presk_key` of `{func}_in` (`presk_rt.h`), 32 bytes, set by the caller. The row of rank `p` draws the uniform `p` of the key: the 64-bit word `p` of its keystream, ChaCha20 as `rand_chacha`'s `ChaCha20Rng` seeded with the key, its top 53 bits over 2^53 (presk D99). Given the key the native backend is given, the program draws its tokens; a function of its arguments, it computes the same outputs at every call.\n");
        let _ = writeln!(r, "- **The draw** is the native backend's, in `f64`: the row's largest logit `m`, `e_i = exp(x_i − m)`, their sum `z` in order, and the first token whose running sum of `e_i / z` exceeds the draw. `exp` is `presk_exp`, musl's, the function the native kernels call, so the tokens do not depend on the C library.");
        let _ = writeln!(r, "- **`main`** sets each key as `presk run --seed` synthesizes it from `-s SEED`, {} by default, the seed the program was exported with and `conformance/native.psk.d` was computed with: the key the seed names (`presk_key_from_seed`), split among the entry's arguments (`presk_key_split`). A key nothing draws from is no field of `{func}_in`.\n", meta.seed);
    }
    if let Some(s) = serve {
        let _ = writeln!(r, "## Generation\n");
        let _ = writeln!(r, "`generate` (`generate.c`) is the loop of `presk serve` over this program: the context is the last {} tokens, the window `{}`'s parameters fix; each step computes one token from it, appends it and moves the window on. At temperature 0 a step calls the greedy copy with temperature 1, as `presk serve` does; above 0 it calls `{func}`{}. Its output is `presk serve`'s: the tokens on one line, or with `--json` the same document, byte for byte (statement C5 of plan 0.9.0).\n", s.window, s.model, if s.key.is_some() { ", token `i` (from 1) given the child `i` of the key `--seed` names (`presk_key_child`), as `presk serve` gives it" } else { "" });
        let _ = writeln!(r, "```sh\nmake generate\n./generate --weights {} --prompt \"{}\" --tokens 8 --temperature 0 --json\n./generate --weights {} --prompt \"{}\" --tokens 8 --seed 1\n```\n", run_args.first().map(|s| s.as_str()).unwrap_or("W.psk.d"), (1..=s.window).map(|t| (t % s.vocab).to_string()).collect::<Vec<_>>().join(" "), run_args.first().map(|s| s.as_str()).unwrap_or("W.psk.d"), (1..=s.window).map(|t| (t % s.vocab).to_string()).collect::<Vec<_>>().join(" "));
        let _ = writeln!(r, "The defaults are `presk serve`'s: 16 tokens, the temperature {:?} (the function's default), and the seed {} (the export's). The prompt has at least {} tokens, each below {}.\n", s.default_temperature, meta.seed, s.window, s.vocab);
    }
    let _ = writeln!(r, "## Threads\n");
    let threads = if meta.threads > 1 { format!("{} workers", meta.threads) } else { "one worker".to_string() };
    if split == 0 {
        let _ = writeln!(r, "No launch is divided among threads: {}, and the native backend divides a kernel it is left to decide for (the executor's default) only above its threshold of work, which none reaches at these dimensions, with {threads}. The program needs no OpenMP (`OPENMP` is empty in the `Makefile`).\n", if meta.plan.is_some() { "the plan states no split" } else { "no plan states a split" });
    } else {
        let _ = writeln!(r, "{split} of the {launches} launches are divided among up to {max_workers} threads, as the native backend divides them: where the plan states a split, among its workers, and elsewhere by the native backend's rule, resolved when the program was exported with {threads} (`presk export --threads`): the outermost parallel loop of a kernel above its threshold of work, among every worker. Each such launch is one OpenMP region in `model.c`, whose threads each call the kernel on their piece of the divided loop, the pieces the native backend computes.\n");
        let _ = writeln!(r, "- **The team** of a region is its workers, at most the threads OpenMP would use: `OMP_NUM_THREADS` lowers it and never raises it (`presk_team` in `presk_rt.h`), and `OMP_NUM_THREADS=1 ./model …` runs on one thread. A `num_threads` clause alone would override `OMP_NUM_THREADS`.");
        let _ = writeln!(r, "- **The bits do not depend on the team.** Every element of a divided loop is computed by one thread, by the same loops in the same order as on one thread, so any team, one thread included, computes the same outputs.");
        let _ = writeln!(r, "- **Without OpenMP** (`make OPENMP=`, or a compiler without it), each region is one call over the whole loop: one thread, the same bits.");
        let _ = writeln!(r, "- **How the threads wait** between regions is the OpenMP runtime's setting, not the program's. An idle worker spins a while, then sleeps, and a sleeping one takes a few hundred microseconds to wake. GCC's libgomp spins 300 000 times by default, about 0.1 ms on the Arm cores it was measured on, less than the host often runs undivided kernels between two regions; the native backend's workers spin about 2 ms. To wait as they do, run with `GOMP_SPINCOUNT=8M` (libgomp: about 2 ms there), or with `OMP_WAIT_POLICY=active`, which every OpenMP runtime reads and which keeps the workers spinning on their cores: the course GPT's forward pass and gradient take about 15 % less time either way (presk D97). The bits are the same.\n");
    }
    let _ = writeln!(r, "## Inputs and outputs\n");
    if let Some(w) = weights {
        let _ = writeln!(r, "The weights, `WEIGHTS.psk.d`, are a checkpoint of type `{}`, as `presk train` writes one: {} leaves, reals stored as `f64` and rounded to `{}` when read, as the native backend rounds them (`bf16` through `f32`).\n", w.ty, w.leaves.len(), meta.realize);
    }
    let quantized: Vec<&(String, String, bool, Option<String>)> = ins.iter().filter(|i| i.3.is_some()).collect();
    if !quantized.is_empty() {
        let _ = writeln!(r, "**Integer weights.** The realization quantizes {} of the leaves (plan 0.9.0 step 7). No file holds their integers: `main` reads their reals, as `presk run` reads a checkpoint, and quantizes them as it reads them (`presk_quantize` in `presk_rt.c`), as the native backend quantizes the weights it loads. The scheme is the realization's; the scales, one per channel, are computed from the reals, per channel of the axis the scale is constant along (presk D44, question 57). Each is a `presk_q8` or `presk_q4` of `{func}_in`, its integers and its scales, and a kernel loads an element as `f32(q)·s`, one `float` multiplication. A program that embeds `{func}` quantizes its weights with `presk_quantize` too.\n", quantized.len());
    }
    if let Some(i) = inputs {
        let _ = writeln!(r, "The other inputs, `INPUTS.psk.d`, are one sample of type `{}`.\n", i.ty);
    }
    let _ = writeln!(r, "| leaf | type | read from |\n|---|---|---|");
    for (p, t, w, q) in ins {
        let _ = writeln!(r, "| `{p}` | `{t}`{} | {} |", q.as_ref().map(|q| format!(", {q}")).unwrap_or_default(), if *w { "WEIGHTS" } else { "INPUTS" });
    }
    for (p, f, v) in dbls {
        let _ = writeln!(r, "| `{p}` | `f64` | `in.{f}`, {v:?} in `main` |");
    }
    let _ = writeln!(r, "\nThe outputs, `OUT.psk.d`, are one sample of type `{}`, reals written as `f64` (exact, whatever the realization). `presk data diff` compares two such files.\n", outputs.ty);
    let _ = writeln!(r, "| output | type |\n|---|---|");
    for (f, v) in outs.iter() {
        let _ = writeln!(r, "| `{f}` | `{}` |", leaf_type(v));
    }
    r
}
