//! The SCoP mode (plan 0.9.0 §3.7, step 8; D85): each nest of an entry at
//! point zero, its default layout on one lane and one worker (0.6.0's
//! kernel, the kernel every point of RA4's layout space is compared with),
//! as a standalone C function between `#pragma scop` and `#pragma endscop`,
//! the input polyhedral tools read; and a driver that times each one and
//! checks its bits against the native backend's.
//!
//! The nests are the ones `presk plan` ranks: the launches of the entry's
//! fusion groups and tensor primitives, captured from their shapes (D65),
//! one per nest, shape and type. The kernels the host launches itself
//! (`positions`, `causal_mask`, `one_hot`, fills) are not nests: no plan
//! states their layout.
//!
//! What the form changes from [`crate::kernel_fn`]'s, the bits kept:
//!
//! - **Arrays of literal dimensions**, one index per axis, as PolyBench/C
//!   writes its kernels: `const float in0[restrict 2][8][16]`, accessed
//!   `in0[i0][i1][i2]`. The kernel keeps one index per axis; the linear
//!   offset was a choice of the program's pointers.
//! - **No range and no OpenMP**: every loop from 0 to its literal bound.
//! - **A branch on a computed value that only assigns scalars is a
//!   selection** of each value (`m = x > m ? x : m`), its condition computed
//!   once when it selects more than one: the running maximum of a softmax,
//!   an argmax. The control then depends on no data.
//! - **The transcendental functions are declared `const`** in the file: they
//!   read and write no memory (musl's functions set no `errno`), so a tool
//!   may take a call for an operation of its statement.
//! - **`#pragma scop` and `#pragma endscop`** are unknown to C compilers,
//!   which warn about them under `-Wall`: each file ignores
//!   `-Wunknown-pragmas` around its function (`#pragma GCC diagnostic`,
//!   which GCC and Clang read), and warns about any other.
//!
//! **Static control** ([`static_control`]): every loop runs from 0 to a
//! literal bound, which every kernel does; no loop variable is assigned; and
//! every access indexes each axis by an affine function of the loop
//! variables. A gather (`embed`, a cross-entropy's `logits[…, y]`) or a
//! scatter (`embed_vjp`) indexes an axis by a value loaded from an index
//! buffer, which no affine function describes.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use presk_kernel::ir::{BufKind, BufSpec, Dtype, Expr, Kernel, Stmt};

use crate::host::{c_string, json_string};
use crate::{CFLAGS, PRESK_LIBM_C, PRESK_RT_H, Printer, WARNINGS, affine, calls};

type Result<T> = std::result::Result<T, String>;

/// How Polly reads a nest (D85): Clang with the Polly LLVM's releases link
/// in (`-mllvm -polly`; LLVM 23.1.2's in the tests, which `make llvm`
/// installs), each file compiled on its own, and the SCoPs it finds printed
/// as isl ASTs of their original schedules.
///
/// - `-polly-position=early`: Polly right after its own canonicalization,
///   before LLVM's loop passes. At its default position, before the
///   vectorizer, `-O2` has already rewritten the outer loops of most nests
///   (their bounds are no longer affine functions Polly derives) and fully
///   unrolled the short ones.
/// - `-polly-detect-full-functions`: a SCoP may be the whole function, which
///   is what a nest is; it also takes a nest of no loop, one statement.
/// - `-polly-process-unprofitable`: a SCoP of one loop, or with little
///   work, is still a SCoP.
/// - `-polly-optimizer=none`: the AST is the nest's own schedule, its loops
///   the kernel's, not a tiled one.
/// - `-fno-math-errno`: `sqrt` is then the operation, not a call that may
///   write `errno`. It is correctly rounded either way, so no bit moves.
/// - `-polly-code-generation=ast`: the AST and no code. Polly 23.1.2 crashes
///   generating the code of a SCoP that is a whole function of no loop
///   (`polly::simplifyRegion`), and only its analysis is wanted.
pub const POLLY_FLAGS: &[&str] = &[
    "-std=c11",
    "-O2",
    "-ffp-contract=off",
    "-fno-math-errno",
    "-mllvm",
    "-polly",
    "-mllvm",
    "-polly-position=early",
    "-mllvm",
    "-polly-detect-full-functions",
    "-mllvm",
    "-polly-process-unprofitable",
    "-mllvm",
    "-polly-optimizer=none",
    "-mllvm",
    "-polly-code-generation=ast",
    "-mllvm",
    "-polly-print-ast",
];

/// The SCoPs Polly printed in `function` (its standard output under
/// [`POLLY_FLAGS`], whose blocks start `:: isl ast :: FUNCTION :: REGION`):
/// the loops of each, the `for` statements of its AST up to the `else` of
/// the original code. A file's other functions, `presk_rt.h`'s conversions,
/// may have SCoPs of their own.
pub fn polly_scops(output: &str, function: &str) -> Vec<usize> {
    output
        .split(":: isl ast ::")
        .skip(1)
        .filter(|block| block.split("::").next().is_some_and(|f| f.trim() == function))
        .map(|block| block.lines().skip(1).take_while(|l| l.trim() != "else").filter(|l| l.trim_start().starts_with("for (")).count())
        .collect()
}

/// A nest of the entry at one shape and type, lowered at point zero.
#[derive(Clone, Debug)]
pub struct Nest {
    /// Its C function, unique in the program.
    pub name: String,
    /// The Presk nest, `<function>/%<let>`.
    pub nest: String,
    /// Its operands' shapes, `[2, 8, 16] [16, 64]`.
    pub shape: String,
    /// How many times the entry launches it at this shape.
    pub launches: usize,
    /// Its kernel at point zero.
    pub kernel: Kernel,
}

/// What the program is exported from, for the README and the comments.
#[derive(Clone, Debug, Default)]
pub struct Meta {
    pub file: String,
    pub entry: String,
    pub dims: Vec<(String, u64)>,
    pub consts: Vec<String>,
    pub realize: String,
    /// The seed of the capture and of the driver's draws.
    pub seed: u64,
    pub version: String,
    pub commit: Option<String>,
}

/// The C element type of a buffer of `k`.
fn elem(k: &Kernel, s: &BufSpec) -> Result<&'static str> {
    Ok(match (s.kind, k.dtype) {
        (BufKind::Idx, _) => "int64_t",
        (BufKind::Real, Dtype::F64) => "double",
        (BufKind::Real, Dtype::F32) => "float",
        (BufKind::Real, Dtype::Bf16) => "uint16_t",
        (BufKind::Quant { .. }, _) => return Err(format!("kernel `{}` reads integer weights, which the SCoP mode does not print", k.name)),
    })
}

/// `const float in0[restrict 2][8]`: an array of the buffer's literal
/// dimensions, `[1]` for rank 0.
fn param(k: &Kernel, s: &BufSpec, name: &str, input: bool) -> Result<String> {
    let dims: Vec<usize> = if s.dims.is_empty() { vec![1] } else { s.dims.clone() };
    let mut p = format!("{}{} {name}[restrict {}]", if input { "const " } else { "" }, elem(k, s)?, dims[0]);
    for d in &dims[1..] {
        let _ = write!(p, "[{d}]");
    }
    Ok(p)
}

/// The nest's C signature, `void name(const float in0[restrict 2][8], …,
/// float out[restrict 2][8])`.
pub fn signature(k: &Kernel, name: &str) -> Result<String> {
    let mut params = Vec::new();
    for (i, s) in k.inputs.iter().enumerate() {
        params.push(param(k, s, &format!("in{i}"), true)?);
    }
    params.push(param(k, &k.output, "out", false)?);
    Ok(format!("void {name}({})", params.join(", ")))
}

/// The nest as a standalone C function: its locals, then its loops between
/// `#pragma scop` and `#pragma endscop`, the pragmas' warning ignored
/// around it.
pub fn scop_fn(k: &Kernel, name: &str) -> Result<String> {
    if k.lanes != 1 {
        return Err(format!("kernel `{}` has {} lanes, and the SCoP mode prints each nest on one lane", k.name, k.lanes));
    }
    let mut p = Printer::new(k, true);
    p.block(&k.body, 1, None)?;
    let mut text = String::new();
    let _ = writeln!(text, "#pragma GCC diagnostic push");
    let _ = writeln!(text, "#pragma GCC diagnostic ignored \"-Wunknown-pragmas\"");
    let _ = writeln!(text, "{} {{", signature(k, name)?);
    text.push_str(&p.declarations());
    let _ = writeln!(text, "#pragma scop");
    text.push_str(&p.body);
    let _ = writeln!(text, "#pragma endscop");
    let _ = writeln!(text, "}}");
    let _ = writeln!(text, "#pragma GCC diagnostic pop");
    Ok(text)
}

/// The loops of a kernel that iterate, two times or more. A loop of one
/// iteration is its body run once, and is no loop in a polyhedral tool's
/// AST (a nest at `B = 1`).
pub fn loops(k: &Kernel) -> usize {
    fn count(stmts: &[Stmt]) -> usize {
        stmts
            .iter()
            .map(|s| match s {
                Stmt::For { bound, body, .. } => usize::from(*bound >= 2) + count(body),
                Stmt::If { then, .. } => count(then),
                _ => 0,
            })
            .sum()
    }
    count(&k.body)
}

/// Every access of a kernel: `(buffer, index, is a store)`, the indices of
/// accesses inside indices included.
fn accesses(k: &Kernel) -> Vec<(usize, Vec<Expr>, bool)> {
    fn expr(e: &Expr, out: &mut Vec<(usize, Vec<Expr>, bool)>) {
        match e {
            Expr::Load(b, ix) | Expr::LoadI(b, ix) | Expr::VLoad(b, ix) => {
                out.push((*b, ix.clone(), false));
                ix.iter().for_each(|x| expr(x, out));
            }
            Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) | Expr::VBin(_, a, b) => {
                expr(a, out);
                expr(b, out);
            }
            Expr::Select(c, a, b) => {
                expr(c, out);
                expr(a, out);
                expr(b, out);
            }
            Expr::Un(_, a) | Expr::ToF(a) | Expr::ToI(a) | Expr::VSplat(a) => expr(a, out),
            Expr::F(_) | Expr::I(_) | Expr::Iv(_) | Expr::Sv(_) | Expr::Si(_) | Expr::Vv(_) => {}
        }
    }
    fn walk(stmts: &[Stmt], out: &mut Vec<(usize, Vec<Expr>, bool)>) {
        for s in stmts {
            match s {
                Stmt::For { body, .. } => walk(body, out),
                Stmt::If { cond, then } => {
                    expr(cond, out);
                    walk(then, out);
                }
                Stmt::Store { buf, index, value } | Stmt::StoreI { buf, index, value } | Stmt::VStore { buf, index, value } => {
                    out.push((*buf, index.clone(), true));
                    index.iter().for_each(|x| expr(x, out));
                    expr(value, out);
                }
                Stmt::Set { value, .. } | Stmt::SetI { value, .. } | Stmt::SetL { value, .. } | Stmt::VSet { value, .. } => expr(value, out),
            }
        }
    }
    let mut out = Vec::new();
    walk(&k.body, &mut out);
    out
}

/// The index buffer an index expression loads from, when it loads from one.
fn loaded_from(e: &Expr) -> Option<usize> {
    match e {
        Expr::LoadI(b, _) => Some(*b),
        Expr::Bin(_, a, b) => loaded_from(a).or_else(|| loaded_from(b)),
        Expr::ToI(a) | Expr::ToF(a) => loaded_from(a),
        Expr::Select(c, a, b) => loaded_from(c).or_else(|| loaded_from(a)).or_else(|| loaded_from(b)),
        _ => None,
    }
}

fn buf_name(k: &Kernel, b: usize) -> String {
    if b == k.out_buf() { "out".into() } else { format!("in{b}") }
}

/// Whether the kernel is a static-control part, and if not, why: every loop
/// runs from 0 to a literal bound (every kernel's does), no loop variable is
/// assigned, every access indexes each axis by an affine function of the
/// loop variables, and every branch on a computed value only assigns
/// scalars (the SCoP form selects them).
pub fn static_control(k: &Kernel) -> std::result::Result<(), String> {
    let mut loop_vars = BTreeSet::new();
    let mut reasons: Vec<String> = Vec::new();
    fn walk(stmts: &[Stmt], vars: &mut BTreeSet<usize>, reasons: &mut Vec<String>) {
        for s in stmts {
            match s {
                Stmt::For { var, body, .. } => {
                    vars.insert(*var);
                    walk(body, vars, reasons);
                }
                Stmt::SetL { var, .. } => reasons.push(format!("it assigns the loop variable `i{var}`")),
                Stmt::If { then, .. } => {
                    if !then.iter().all(|s| matches!(s, Stmt::Set { .. } | Stmt::SetI { .. })) {
                        reasons.push("a branch on a computed value guards more than scalars".into());
                    }
                    walk(then, vars, reasons);
                }
                _ => {}
            }
        }
    }
    walk(&k.body, &mut loop_vars, &mut reasons);
    // Stores first: a scatter that accumulates reads the element it writes,
    // and is one reason.
    let mut all = accesses(k);
    all.sort_by_key(|(_, _, store)| !*store);
    let mut scattered = BTreeSet::new();
    for (b, ix, store) in all {
        for e in &ix {
            let ok = affine(e).is_some_and(|(vars, _)| vars.iter().all(|(v, _)| v.strip_prefix('i').and_then(|n| n.parse::<usize>().ok()).is_some_and(|n| loop_vars.contains(&n))));
            if ok || (!store && scattered.contains(&b)) {
                continue;
            }
            let what = if store { format!("writes `{}`", buf_name(k, b)) } else { format!("reads `{}`", buf_name(k, b)) };
            let r = match loaded_from(e) {
                Some(j) => format!("it {what} at an index loaded from `{}`, a {}", buf_name(k, j), if store { "scatter" } else { "gather" }),
                None => format!("it {what} at an index computed from data"),
            };
            if store {
                scattered.insert(b);
            }
            if !reasons.contains(&r) {
                reasons.push(r);
            }
        }
    }
    if reasons.is_empty() { Ok(()) } else { Err(reasons.join("; ")) }
}

/// What the driver draws each input below: 0 for a real input; for an index
/// input, the least length of an axis it indexes or of a loop it is compared
/// with (a cross-entropy's gradient compares its labels with the classes),
/// or, compared with a literal `c`, `⌊c⌋ + 2`, so that both sides of the
/// comparison are drawn (a mask of a softmax, compared with `0.5`, is drawn
/// below 2: plan 0.10.0 step 7b); and 1, zeros, when it does none of these.
pub fn index_bounds(k: &Kernel) -> Vec<u64> {
    let mut bounds: Vec<Option<u64>> = vec![None; k.inputs.len()];
    let mut lower = |j: usize, n: usize| {
        if j < bounds.len() {
            bounds[j] = Some(bounds[j].map_or(n as u64, |b| b.min(n as u64)));
        }
    };
    let dims = |b: usize| if b == k.out_buf() { k.output.dims.clone() } else { k.inputs[b].dims.clone() };
    for (b, ix, _) in accesses(k) {
        for (a, e) in ix.iter().enumerate() {
            if let Expr::LoadI(j, _) = e {
                lower(*j, dims(b)[a]);
            }
        }
    }
    let mut loop_bounds = BTreeMap::new();
    fn fors(stmts: &[Stmt], out: &mut BTreeMap<usize, usize>) {
        for s in stmts {
            match s {
                Stmt::For { var, bound, body } => {
                    out.insert(*var, *bound);
                    fors(body, out);
                }
                Stmt::If { then, .. } => fors(then, out),
                _ => {}
            }
        }
    }
    fors(&k.body, &mut loop_bounds);
    let strip = |e: &Expr| -> Expr {
        match e {
            Expr::ToF(a) => (**a).clone(),
            other => other.clone(),
        }
    };
    let mut cmps: Vec<(Expr, Expr)> = Vec::new();
    fn find(e: &Expr, out: &mut Vec<(Expr, Expr)>) {
        match e {
            Expr::Cmp(_, a, b) => {
                out.push(((**a).clone(), (**b).clone()));
                find(a, out);
                find(b, out);
            }
            Expr::Bin(_, a, b) | Expr::VBin(_, a, b) => {
                find(a, out);
                find(b, out);
            }
            Expr::Select(c, a, b) => {
                find(c, out);
                find(a, out);
                find(b, out);
            }
            Expr::Un(_, a) | Expr::ToF(a) | Expr::ToI(a) | Expr::VSplat(a) => find(a, out),
            Expr::Load(_, ix) | Expr::LoadI(_, ix) | Expr::VLoad(_, ix) => ix.iter().for_each(|x| find(x, out)),
            _ => {}
        }
    }
    crate::walk_exprs(&k.body, &mut |e| find(e, &mut cmps));
    for (a, b) in cmps {
        for (x, y) in [(strip(&a), strip(&b)), (strip(&b), strip(&a))] {
            if let (Expr::LoadI(j, _), Expr::Iv(v)) = (&x, &y)
                && let Some(n) = loop_bounds.get(v)
            {
                lower(*j, *n);
            }
            if let (Expr::LoadI(j, _), Expr::F(c)) = (&x, &y)
                && *c >= 0.0
            {
                lower(*j, c.floor() as usize + 2);
            }
        }
    }
    k.inputs.iter().zip(bounds).map(|(s, b)| if s.kind == BufKind::Idx { b.unwrap_or(1).max(1) } else { 0 }).collect()
}

/// The driver's draws, `splitmix64`: the same numbers in Rust (the export's
/// native run) and in C (`driver.c`), seeded per nest by the export's seed
/// and the nest's place in the program.
pub struct Draw(u64);

impl Draw {
    pub fn new(seed: u64, nest: usize) -> Draw {
        Draw(seed ^ 0xd1b5_4a32_d192_ed03u64.wrapping_mul(nest as u64 + 1))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A real in [-1, 1), exact in `f64`: the top 53 bits over 2^53, twice,
    /// less one.
    pub fn real(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0) * 2.0 - 1.0
    }
}

/// An input the driver draws: reals in `f64`, which the driver rounds to the
/// kernel's type as `Buf::of` does (`bf16` through `f32`), or indices.
#[derive(Clone, Debug, PartialEq)]
pub enum Drawn {
    Real(Vec<f64>),
    Idx(Vec<i64>),
}

/// The inputs of nest number `nest` (from 0, in the program's order), drawn
/// as `driver.c` draws them: input by input, element by element, one number
/// each; a real in [-1, 1), an index below its bound ([`index_bounds`]).
pub fn draw_inputs(k: &Kernel, seed: u64, nest: usize) -> Vec<Drawn> {
    let mut g = Draw::new(seed, nest);
    let bounds = index_bounds(k);
    k.inputs
        .iter()
        .zip(bounds)
        .map(|(s, bound)| match s.kind {
            BufKind::Idx => Drawn::Idx((0..s.numel()).map(|_| (g.next_u64() % bound) as i64).collect()),
            _ => Drawn::Real((0..s.numel()).map(|_| g.real()).collect()),
        })
        .collect()
}

/// A nest's static control, for the README and `nests.json`.
fn verdict(k: &Kernel) -> (bool, String) {
    match static_control(k) {
        Ok(()) => (true, String::new()),
        Err(r) => (false, r),
    }
}

fn kind_code(k: &Kernel, s: &BufSpec) -> &'static str {
    match (s.kind, k.dtype) {
        (BufKind::Idx, _) => "PRESK_SCOP_IDX",
        (_, Dtype::F64) => "PRESK_SCOP_F64",
        (_, Dtype::F32) => "PRESK_SCOP_F32",
        (_, Dtype::Bf16) => "PRESK_SCOP_BF16",
    }
}

/// The program: `nests/NAME.c` per nest, `nests.h`, `driver.c`, the runtime
/// header and the transcendental functions, a `Makefile`, a `README.md` and
/// `nests.json`, as `(path, text)`. The native outputs `conformance/native.bin`
/// holds are the caller's to write.
pub fn export(meta: &Meta, nests: &[Nest]) -> Result<Vec<(String, String)>> {
    let mut files = Vec::new();
    let mut header = String::new();
    let _ = writeln!(header, "/* The nests of `{}` at point zero (presk export --target c --mode scop). */\n#ifndef PRESK_NESTS_H\n#define PRESK_NESTS_H\n\n#include <stdint.h>\n", meta.entry);
    for n in nests {
        files.push((format!("nests/{}.c", n.name), nest_file(meta, n)?));
        let _ = writeln!(header, "{};", signature(&n.kernel, &n.name)?);
    }
    let _ = writeln!(header, "\n#endif");
    files.push(("nests.h".to_string(), header));
    files.push(("driver.c".to_string(), driver(meta, nests)?));
    files.push(("Makefile".to_string(), makefile(meta, nests)));
    files.push(("README.md".to_string(), readme(meta, nests)));
    files.push(("nests.json".to_string(), json(nests)));
    files.push(("presk_rt.h".to_string(), PRESK_RT_H.to_string()));
    files.push(("presk_libm.c".to_string(), PRESK_LIBM_C.to_string()));
    Ok(files)
}

/// A nest's file, `nests/NAME.c`: what it is and its PSK-Kernel text in a
/// comment, the includes (`presk_rt.h` one directory up), the transcendental
/// functions it calls declared `const`, and the function ([`scop_fn`]).
pub fn nest_file(meta: &Meta, n: &Nest) -> Result<String> {
    let k = &n.kernel;
    let dims = if meta.dims.is_empty() { "no dimension".to_string() } else { meta.dims.iter().map(|(n, v)| format!("{n} = {v}")).collect::<Vec<_>>().join(", ") };
    let mut c = String::new();
    let _ = writeln!(c, "// `{}` of `{}` ({}), at {dims} in {}: operands {}, launched {} time{} by the entry.", n.nest, meta.entry, meta.file, meta.realize, n.shape, n.launches, if n.launches == 1 { "" } else { "s" });
    let _ = writeln!(c, "// Point zero: the nest's default layout, one lane, one worker (plan 0.9.0 §3.7).");
    match static_control(k) {
        Ok(()) => {
            let _ = writeln!(c, "// Static control: literal bounds, affine accesses.");
        }
        Err(why) => {
            let _ = writeln!(c, "// Not static control: {why}.");
        }
    }
    let _ = writeln!(c, "//");
    for line in k.to_string().lines() {
        let _ = writeln!(c, "// {line}");
    }
    let _ = writeln!(c, "#include <math.h>\n#include <stdint.h>\n#include \"../presk_rt.h\"\n");
    let fs = calls(k);
    if !fs.is_empty() {
        let real = if k.dtype == Dtype::F64 { "double" } else { "float" };
        let _ = writeln!(c, "// Read and write no memory (musl's functions set no errno): a call is an\n// operation of its statement.\n#if defined(__GNUC__)");
        for f in &fs {
            let _ = writeln!(c, "{real} {f}({real}) __attribute__((const));");
        }
        let _ = writeln!(c, "#endif\n");
    }
    c.push_str(&scop_fn(k, &n.name)?);
    Ok(c)
}

fn driver(meta: &Meta, nests: &[Nest]) -> Result<String> {
    let maxb = nests.iter().map(|n| n.kernel.inputs.len() + 1).max().unwrap_or(1);
    let mut d = String::new();
    let _ = writeln!(
        d,
        "/* Runs the nests of `{}` at point zero (presk export --target c --mode scop,\n * presk {}): each on inputs it draws, timed, or checked against the native\n * backend's outputs.\n *\n *   ./scop [-r R] [-s SEED] [NEST...]   R calls of each nest (default 5), the\n *                                        median and the best wall time\n *   ./scop -c conformance/native.bin    one call of each, its output compared\n *                                        with the native backend's, bit for bit\n */",
        meta.entry, meta.version
    );
    d.push_str(
        r#"#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "presk_rt.h"
#include "nests.h"

enum { PRESK_SCOP_F64, PRESK_SCOP_F32, PRESK_SCOP_BF16, PRESK_SCOP_IDX };

typedef struct {
    size_t n;
    int kind;
    uint64_t bound;
} buf_spec;

"#,
    );
    let _ = writeln!(d, "#define MAX_BUFS {maxb}\n");
    d.push_str(
        r#"typedef struct {
    const char *name, *nest, *shape, *type;
    int nbufs;
    buf_spec bufs[MAX_BUFS];
    void (*call)(void *const *b);
} nest_spec;

"#,
    );
    for n in nests {
        let args: Vec<String> = (0..=n.kernel.inputs.len()).map(|i| format!("b[{i}]")).collect();
        let _ = writeln!(d, "static void call_{}(void *const *b) {{\n    {}({});\n}}\n", n.name, n.name, args.join(", "));
    }
    let _ = writeln!(d, "static const nest_spec NESTS[] = {{");
    for n in nests {
        let k = &n.kernel;
        let bounds = index_bounds(k);
        let mut bufs: Vec<String> = k.inputs.iter().zip(&bounds).map(|(s, b)| format!("{{{}, {}, {b}}}", s.numel(), kind_code(k, s))).collect();
        bufs.push(format!("{{{}, {}, 0}}", k.output.numel(), kind_code(k, &k.output)));
        let _ = writeln!(d, "    {{{}, {}, {}, {}, {}, {{{}}}, call_{}}},", c_string(&n.name), c_string(&n.nest), c_string(&n.shape), c_string(k.dtype.name()), bufs.len(), bufs.join(", "), n.name);
    }
    let _ = writeln!(d, "}};\n#define NNESTS {}\n", nests.len());
    let _ = writeln!(d, "#define SEED UINT64_C({})\n", meta.seed);
    d.push_str(
        r#"/* splitmix64, seeded per nest as presk seeds it (`scop::Draw`). */
static uint64_t next(uint64_t *s) {
    uint64_t z = (*s += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static size_t elem_size(int kind) {
    return kind == PRESK_SCOP_F32 ? 4 : kind == PRESK_SCOP_BF16 ? 2 : 8;
}

/* The inputs of nest `i`: a real in [-1, 1), rounded to the kernel's type as
 * presk rounds it (bf16 through float); an index below its bound. The output
 * is zeros, as the native backend allocates it. */
static void **make(const nest_spec *s, size_t i, uint64_t seed) {
    void **b = calloc((size_t)s->nbufs, sizeof *b);
    uint64_t g = seed ^ (UINT64_C(0xd1b54a32d192ed03) * (uint64_t)(i + 1));
    if (!b) return NULL;
    for (int j = 0; j < s->nbufs; j++) {
        const buf_spec *f = &s->bufs[j];
        b[j] = calloc(f->n ? f->n : 1, elem_size(f->kind));
        if (!b[j]) return NULL;
        if (j == s->nbufs - 1) break;
        for (size_t e = 0; e < f->n; e++) {
            uint64_t u = next(&g);
            double x = (double)(u >> 11) * 0x1p-53 * 2.0 - 1.0;
            switch (f->kind) {
            case PRESK_SCOP_F64: ((double *)b[j])[e] = x; break;
            case PRESK_SCOP_F32: ((float *)b[j])[e] = (float)x; break;
            case PRESK_SCOP_BF16: ((uint16_t *)b[j])[e] = presk_f32_to_bf16((float)x); break;
            default: ((int64_t *)b[j])[e] = (int64_t)(u % f->bound); break;
            }
        }
    }
    return b;
}

static void release(const nest_spec *s, void **b) {
    for (int j = 0; j < s->nbufs; j++) free(b[j]);
    free(b);
}

static double now_us(void) {
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return (double)t.tv_sec * 1e6 + (double)t.tv_nsec * 1e-3;
}

static int ascending(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static size_t out_bytes(const nest_spec *s) {
    const buf_spec *o = &s->bufs[s->nbufs - 1];
    return o->n * elem_size(o->kind);
}

int main(int argc, char **argv) {
    long repeat = 5;
    uint64_t seed = SEED;
    const char *check = NULL;
    int a = 1;
    for (; a < argc && argv[a][0] == '-'; a++) {
        if (!strcmp(argv[a], "-r") && a + 1 < argc) repeat = strtol(argv[++a], NULL, 10);
        else if (!strcmp(argv[a], "-s") && a + 1 < argc) seed = strtoull(argv[++a], NULL, 10);
        else if (!strcmp(argv[a], "-c") && a + 1 < argc) check = argv[++a];
        else {
            fprintf(stderr, "usage: %s [-r R] [-s SEED] [-c native.bin] [NEST...]\n", argv[0]);
            return 2;
        }
    }
    if (repeat < 1) repeat = 1;
    unsigned char *expected = NULL;
    size_t total = 0;
    if (check) {
        for (size_t i = 0; i < NNESTS; i++) total += out_bytes(&NESTS[i]);
        FILE *f = fopen(check, "rb");
        expected = malloc(total ? total : 1);
        if (!f || !expected || fread(expected, 1, total, f) != total || fgetc(f) != EOF) {
            fprintf(stderr, "%s: not the %zu bytes of the nests' outputs\n", check, total);
            return 2;
        }
        fclose(f);
        if (seed != SEED) {
            fprintf(stderr, "-c: the native outputs were computed with the seed %llu\n", (unsigned long long)SEED);
            return 2;
        }
        repeat = 1;
    } else {
        printf("%-40s %-5s %12s %12s  %s\n", "nest", "type", "median us", "best us", "operands");
    }
    double *us = malloc((size_t)repeat * sizeof *us);
    size_t offset = 0, checked = 0, differ = 0;
    for (size_t i = 0; i < NNESTS; i++) {
        const nest_spec *s = &NESTS[i];
        size_t ob = out_bytes(s);
        int chosen = a == argc;
        for (int j = a; j < argc; j++) chosen |= !strcmp(argv[j], s->name) || !strcmp(argv[j], s->nest);
        if (!chosen) {
            offset += ob;
            continue;
        }
        void **b = make(s, i, seed);
        if (!b || !us) {
            fprintf(stderr, "out of memory\n");
            return 1;
        }
        for (long r = 0; r < repeat; r++) {
            memset(b[s->nbufs - 1], 0, ob);
            double t0 = now_us();
            s->call(b);
            us[r] = now_us() - t0;
        }
        if (check) {
            checked++;
            if (memcmp(b[s->nbufs - 1], expected + offset, ob)) {
                differ++;
                fprintf(stderr, "%s (%s, %s %s): the output differs from the native backend's\n", s->name, s->nest, s->type, s->shape);
            }
        } else {
            qsort(us, (size_t)repeat, sizeof *us, ascending);
            printf("%-40s %-5s %12.3f %12.3f  %s\n", s->name, s->type, us[repeat / 2], us[0], s->shape);
        }
        offset += ob;
        release(s, b);
    }
    free(us);
    free(expected);
    if (check) {
        if (differ) {
            fprintf(stderr, "%zu of the %zu nests differ from the native backend\n", differ, checked);
            return 1;
        }
        printf("the %zu nests compute the native backend's bits\n", checked);
    }
    return 0;
}
"#,
    );
    Ok(d)
}

fn makefile(meta: &Meta, nests: &[Nest]) -> String {
    let strict: Vec<&str> = CFLAGS.iter().copied().filter(|f| *f != "-O2").collect();
    let list = |ext: &str| nests.iter().map(|n| format!("nests/{}.{ext}", n.name)).collect::<Vec<_>>().join(" \\\n\t");
    format!(
        "# Builds the nests of `{entry}` at point zero and their driver (presk export\n# --target c --mode scop). Generated by presk {version}.\n#\n# STRICT holds the flags the native backend's bits rest on (plan 0.9.0 §3.5):\n# ISO C11 and no contraction of a product and a sum into one rounding.\n\nSTRICT = {strict}\nCFLAGS = -O2 {warn}\n\nOBJS = \\\n\t{objs}\n\nscop: driver.c nests.h presk_rt.h presk_libm.c $(OBJS)\n\t$(CC) $(STRICT) $(CFLAGS) -o $@ driver.c $(OBJS) presk_libm.c -lm\n\n$(OBJS): presk_rt.h\n\n# One call of each nest on the inputs it draws, its output compared with the\n# native backend's, bit for bit.\ncheck: scop\n\t./scop -c conformance/native.bin\n\n# Each nest called eleven times, the median and the best wall time.\ntime: scop\n\t./scop -r 11\n\n# Polly's view of each nest, `make -j polly POLLY=/path/to/clang` with a clang\n# that links Polly, as LLVM's releases do: the SCoPs it finds in nests/NAME.c,\n# each printed as the isl AST of its original schedule, in nests/NAME.polly.\n# A static-control nest is one SCoP, the whole function, with the nest's loops.\nPOLLY = clang\nPOLLYFLAGS = {polly} {warn}\nSCOPS = \\\n\t{scops}\n\npolly: $(SCOPS)\n\n$(SCOPS): presk_rt.h\n\n.SUFFIXES: .c .o .polly\n.c.o:\n\t$(CC) $(STRICT) $(CFLAGS) -c -o $@ $<\n.c.polly:\n\t$(POLLY) $(POLLYFLAGS) -c -o /dev/null $< > $@\n\nclean:\n\trm -f scop $(OBJS) $(SCOPS)\n\n.PHONY: check time polly clean\n",
        entry = meta.entry,
        version = meta.version,
        strict = strict.join(" "),
        warn = WARNINGS.join(" "),
        objs = list("o"),
        scops = list("polly"),
        polly = POLLY_FLAGS.join(" "),
    )
}

fn json(nests: &[Nest]) -> String {
    let mut j = String::from("[\n");
    for (i, n) in nests.iter().enumerate() {
        let k = &n.kernel;
        let (is_static, why) = verdict(k);
        let _ = write!(
            j,
            "  {{\"name\": {}, \"nest\": {}, \"kernel\": {}, \"operands\": {}, \"type\": {}, \"launches\": {}, \"loops\": {}, \"static_control\": {is_static}, \"reason\": {}}}{}",
            json_string(&n.name),
            json_string(&n.nest),
            json_string(&k.name),
            json_string(&n.shape),
            json_string(k.dtype.name()),
            n.launches,
            loops(k),
            if is_static { "null".to_string() } else { json_string(&why) },
            if i + 1 < nests.len() { ",\n" } else { "\n" }
        );
    }
    j.push_str("]\n");
    j
}

fn readme(meta: &Meta, nests: &[Nest]) -> String {
    let mut r = String::new();
    let statics = nests.iter().filter(|n| static_control(&n.kernel).is_ok()).count();
    let names: BTreeSet<&str> = nests.iter().map(|n| n.nest.as_str()).collect();
    let _ = writeln!(r, "# The nests of `{}` at point zero\n", meta.entry);
    let _ = writeln!(r, "Exported by `presk export --target c --mode scop` (presk {}) from `{}`{}: each nest of the entry, at the shape and in the type the entry launches it, as a standalone C function between `#pragma scop` and `#pragma endscop`, the input a polyhedral tool (pet, PPCG, PLUTO, Polly) reads (plan 0.9.0 §3.7).\n", meta.version, meta.file, meta.commit.as_ref().map(|c| format!(", at commit `{c}` of its repository")).unwrap_or_default());
    let _ = writeln!(r, "- **Entry:** `{}` at {}, in `{}`; constants {}.", meta.entry, if meta.dims.is_empty() { "no dimension".to_string() } else { meta.dims.iter().map(|(n, v)| format!("`{n} = {v}`")).collect::<Vec<_>>().join(", ") }, meta.realize, if meta.consts.is_empty() { "the file's".to_string() } else { meta.consts.iter().map(|c| format!("`{c}`")).collect::<Vec<_>>().join(", ") });
    let _ = writeln!(r, "- **Nests:** {} at {} shapes, a C function each, {} of them static-control parts. A nest is a fusion group or a tensor primitive of a function, `<function>/%<let>`, the unit a realization plan states a layout for and `presk plan` ranks; a nest launched at several shapes, or in several types, is one function per shape and type. The kernels the host launches itself (positions, causal masks, one-hot rows, fills) are not nests, no plan stating their layout, and are not here.", names.len(), nests.len(), statics);
    let _ = writeln!(r, "- **Point zero:** each nest's default layout on one lane and one worker, no pack: 0.6.0's kernel, the one every point of a nest's layout space computes the bits of, before any plan. RA4 (`presk plan`, `presk autotune`) ranks the other points of the same nests.\n");
    let _ = writeln!(r, "## The form\n");
    let _ = writeln!(r, "- **Arrays of literal dimensions**, one index per axis, as PolyBench/C writes its kernels: `const float in0[restrict 2][8][16]`, accessed `in0[i0][i1][i2]`; a buffer of rank 0 is `[1]`. Inputs are `in0`, `in1`, …, the output `out`, which the nest reads only where it accumulates, from zeros as the native backend allocates it. Reals are `double`, `float`, or `uint16_t` for `bf16`, computed in `float` (`presk_bf16_to_f32`, `presk_f32_to_bf16` in `presk_rt.h`); indices are `int64_t`.");
    let _ = writeln!(r, "- **Every loop** runs from 0 to a literal bound, its variable an `int64_t` declared with the scalars before `#pragma scop`.");
    let _ = writeln!(r, "- **A branch on a computed value that only assigns scalars is a selection** of each value, `m = x > m ? x : m`: a softmax's running maximum, an argmax. Its condition is computed once, into `c0`, when it selects more than one value. The bits are the branch's.");
    let _ = writeln!(r, "- **The transcendental functions** (`presk_expf`, …, musl 1.1.22's, in `presk_libm.c`) are declared `__attribute__((const))` in each file that calls them: they read and write no memory, so a tool may take a call for an operation of its statement.");
    let _ = writeln!(r, "- **The pragmas' warning.** C compilers do not know `#pragma scop` and warn about it under `-Wall`. Each file ignores `-Wunknown-pragmas` around its function (`#pragma GCC diagnostic push`, `ignored`, `pop`, which GCC and Clang read), so it builds without a warning and still warns about any other unknown pragma.");
    let _ = writeln!(r, "- **The kernel's PSK-Kernel text** (`presk dump --stage kernel`) is the comment above each function.\n");
    let _ = writeln!(r, "## Static control\n");
    let _ = writeln!(r, "A nest is a static-control part when every loop runs from 0 to a literal bound, which every nest's does; no loop variable is assigned; and every access indexes each axis of its array by an affine function of the loop variables. A gather (an embedding's `table[ids[b][t]][d]`, a cross-entropy's `logits[b][t][y[b][t]]`) or a scatter (an embedding's gradient, `out[ids[b][t]][d] += …`) indexes an axis by a value loaded from an index array, which no affine function describes: a tool that takes such an access over-approximates it by the whole axis. `nests.json` lists every nest with its loops (those of two iterations or more: a loop of one iteration is no loop in a tool's AST) and whether it is static control, and why not.\n");
    let _ = writeln!(r, "## Build, check, time\n\n```sh\nmake          # every nest compiled on its own (nests/*.o), and the driver, scop\nmake check    # ./scop -c conformance/native.bin: each nest's output against the native backend's, bit for bit\nmake time     # ./scop -r 11: each nest called eleven times, the median and the best wall time in µs\n./scop -r 5 NEST...   # some nests, by C name or Presk name\n```\n");
    let _ = writeln!(r, "**The driver's inputs** are drawn, `splitmix64` seeded per nest by the seed {} and the nest's place in the program, the numbers presk draws for `conformance/native.bin`: a real uniformly in [-1, 1), rounded to the nest's type as presk rounds it (`bf16` through `float`); an index below the least length of an axis it indexes or of a loop it is compared with, so that every gather and scatter stays inside its array, and 0 when it does neither. `-s SEED` draws others (not with `-c`).\n", meta.seed);
    let _ = writeln!(r, "**`conformance/native.bin`** holds the native backend's outputs of every nest on those inputs at point zero, one after the other in the program's order, little-endian. `make check` compares each nest's output with its own, byte for byte.\n");
    let _ = writeln!(r, "**The flags that must not change** (`STRICT` in the `Makefile`): `-std=c11 -ffp-contract=off`, for the native backend's bits; a polyhedral tool's output is compared the same way.\n");
    let _ = writeln!(r, "## Polly\n");
    let _ = writeln!(r, "`make -j polly POLLY=/path/to/clang` runs Polly, with a clang that links it as LLVM's releases do (presk's `make llvm` installs LLVM 23.1.2's), on each nest's file on its own, and writes the isl AST of every SCoP it finds, at the nest's own schedule, into `nests/NAME.polly`. A static-control nest is one SCoP, the whole function, whose AST has the nest's loops; a gather or a scatter is not, unless non-affine accesses are allowed (`-mllvm -polly-allow-nonaffine`), when Polly over-approximates them by the whole axis. `POLLYFLAGS`:\n");
    let _ = writeln!(r, "- `-mllvm -polly-position=early`: Polly after its own canonicalization, before LLVM's loop passes, which would otherwise have rewritten the outer loops and unrolled the short ones;");
    let _ = writeln!(r, "- `-mllvm -polly-detect-full-functions`, `-mllvm -polly-process-unprofitable`: the whole function may be a SCoP, and a small one is still one;");
    let _ = writeln!(r, "- `-mllvm -polly-optimizer=none`: the AST keeps the nest's schedule, not a tiled one; `-mllvm -polly-code-generation=ast`: no code, only the analysis;");
    let _ = writeln!(r, "- `-fno-math-errno`: `sqrt` is the operation, not a call that may write `errno` (it is correctly rounded either way).\n");
    let _ = writeln!(r, "pet, PPCG and PLUTO read the region between the pragmas instead.\n");
    let _ = writeln!(r, "## The nests\n");
    let _ = writeln!(r, "| C function | nest | kernel | operands | type | launches | loops | static control |\n|---|---|---|---|---|---|---|---|");
    for n in nests {
        let k = &n.kernel;
        let s = match static_control(k) {
            Ok(()) => "yes".to_string(),
            Err(why) => format!("no: {why}"),
        };
        let _ = writeln!(r, "| `{}` | `{}` | `{}` | `{}` | `{}` | {} | {} | {} |", n.name, n.nest, k.name.replace('|', "\\|"), n.shape, k.dtype.name(), n.launches, loops(k), s);
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use presk_kernel::ir::{BinOp, CmpOp, Threads, UnOp};

    fn buf(dims: &[usize], kind: BufKind) -> BufSpec {
        BufSpec { dims: dims.to_vec(), kind, names: vec![None; dims.len()] }
    }

    fn kernel(dtype: Dtype, inputs: Vec<BufSpec>, output: BufSpec, body: Vec<Stmt>) -> Kernel {
        Kernel { name: "t".into(), dtype, inputs, output, nloops: 4, nreal: 4, nint: 2, nvec: 0, lanes: 1, body, threads: Threads::One }
    }

    fn bx(e: Expr) -> Box<Expr> {
        Box::new(e)
    }

    fn at(v: &[usize]) -> Vec<Expr> {
        v.iter().map(|i| Expr::Iv(*i)).collect()
    }

    /// `out[i0, i2] += x[i0, i1] · w[i1, i2]` over `[2, 3] · [3, 4]`, as the
    /// scalar dot at point zero is.
    fn dot() -> Kernel {
        let acc = Expr::Bin(BinOp::Add, bx(Expr::Load(2, at(&[0, 2]))), bx(Expr::Bin(BinOp::Mul, bx(Expr::Load(0, at(&[0, 1]))), bx(Expr::Load(1, at(&[1, 2]))))));
        let body = vec![Stmt::For { var: 0, bound: 2, body: vec![Stmt::For { var: 1, bound: 3, body: vec![Stmt::For { var: 2, bound: 4, body: vec![Stmt::Store { buf: 2, index: at(&[0, 2]), value: acc }] }] }] }];
        kernel(Dtype::F32, vec![buf(&[2, 3], BufKind::Real), buf(&[3, 4], BufKind::Real)], buf(&[2, 4], BufKind::Real), body)
    }

    #[test]
    fn a_nest_is_arrays_between_the_pragmas() {
        let text = scop_fn(&dot(), "dot_3").unwrap();
        let lines: Vec<&str> = text.lines().map(str::trim).collect();
        assert!(lines.contains(&"void dot_3(const float in0[restrict 2][3], const float in1[restrict 3][4], float out[restrict 2][4]) {"), "{text}");
        assert!(lines.contains(&"out[i0][i2] = out[i0][i2] + in0[i0][i1] * in1[i1][i2];"), "{text}");
        assert!(lines.contains(&"for (i0 = 0; i0 < 2; i0++) {"), "{text}");
        // The pragmas enclose the loops, the warning about them ignored around
        // the function alone.
        let at = |l: &str| lines.iter().position(|x| *x == l).unwrap_or_else(|| panic!("{l}: {text}"));
        assert!(at("#pragma GCC diagnostic push") < at("#pragma scop"));
        assert!(at("int64_t i0 = 0, i1 = 0, i2 = 0;") < at("#pragma scop"));
        assert!(at("#pragma scop") < at("for (i0 = 0; i0 < 2; i0++) {"));
        assert!(at("#pragma endscop") < at("#pragma GCC diagnostic pop"));
        assert!(!text.contains("int64_t lo") && !text.contains("< hi") && !text.contains("omp"), "{text}");
        assert_eq!(loops(&dot()), 3);
        // A loop of one iteration is no loop to a tool's AST.
        let mut one = dot();
        let Stmt::For { bound, .. } = &mut one.body[0] else { unreachable!() };
        *bound = 1;
        assert_eq!(loops(&one), 2);
        assert_eq!(static_control(&dot()), Ok(()));
        // A buffer of rank 0 is an array of one.
        let k = kernel(Dtype::F64, vec![buf(&[], BufKind::Real)], buf(&[], BufKind::Real), vec![Stmt::Store { buf: 1, index: vec![], value: Expr::Un(UnOp::Sqrt, bx(Expr::Load(0, vec![]))) }]);
        let text = scop_fn(&k, "s").unwrap();
        assert!(text.contains("void s(const double in0[restrict 1], double out[restrict 1])") && text.contains("out[0] = sqrt(in0[0]);"), "{text}");
        assert_eq!(loops(&k), 0);
    }

    /// The running maximum and an argmax: a selection of each value, the
    /// condition computed once when it selects two.
    #[test]
    fn a_branch_that_assigns_scalars_is_a_selection() {
        let x = || Expr::Load(0, at(&[0]));
        let max = vec![Stmt::Set { var: 0, value: Expr::F(f64::NEG_INFINITY) }, Stmt::For { var: 0, bound: 5, body: vec![Stmt::If { cond: Expr::Cmp(CmpOp::Gt, bx(x()), bx(Expr::Sv(0))), then: vec![Stmt::Set { var: 0, value: x() }] }] }, Stmt::Store { buf: 1, index: vec![], value: Expr::Sv(0) }];
        let k = kernel(Dtype::F32, vec![buf(&[5], BufKind::Real)], buf(&[], BufKind::Real), max);
        let text = scop_fn(&k, "m").unwrap();
        assert!(text.contains("s0 = in0[i0] > s0 ? in0[i0] : s0;"), "{text}");
        assert!(!text.contains("if ("), "{text}");
        assert_eq!(static_control(&k), Ok(()));
        let argmax = vec![Stmt::For { var: 0, bound: 5, body: vec![Stmt::If { cond: Expr::Cmp(CmpOp::Ge, bx(x()), bx(Expr::Sv(0))), then: vec![Stmt::Set { var: 0, value: x() }, Stmt::SetI { var: 1, value: Expr::Iv(0) }] }] }, Stmt::StoreI { buf: 1, index: vec![], value: Expr::Si(1) }];
        let k = kernel(Dtype::F64, vec![buf(&[5], BufKind::Real)], buf(&[], BufKind::Idx), argmax);
        let text = scop_fn(&k, "a").unwrap();
        for l in ["int c0 = 0;", "c0 = in0[i0] >= s0;", "s0 = c0 ? in0[i0] : s0;", "n1 = c0 ? i0 : n1;"] {
            assert!(text.contains(l), "{l}: {text}");
        }
        // A branch that guards a store stays a branch, and is not static control.
        let guard = vec![Stmt::For { var: 0, bound: 5, body: vec![Stmt::If { cond: Expr::Cmp(CmpOp::Gt, bx(x()), bx(Expr::F(0.0))), then: vec![Stmt::Store { buf: 1, index: at(&[0]), value: x() }] }] }];
        let k = kernel(Dtype::F32, vec![buf(&[5], BufKind::Real)], buf(&[5], BufKind::Real), guard);
        assert!(scop_fn(&k, "g").unwrap().contains("if (in0[i0] > 0.0f) {"));
        assert_eq!(static_control(&k), Err("a branch on a computed value guards more than scalars".into()));
    }

    /// A gather, a scatter that accumulates, an index compared with a loop:
    /// the rule, and what the driver draws the indices below.
    #[test]
    fn gathers_and_scatters_are_not_static_control() {
        // out[i0, j] = table[ids[i0], j]: `embed`.
        let gather = vec![Stmt::For { var: 0, bound: 4, body: vec![Stmt::For { var: 1, bound: 3, body: vec![Stmt::Store { buf: 2, index: at(&[0, 1]), value: Expr::Load(0, vec![Expr::LoadI(1, at(&[0])), Expr::Iv(1)]) }] }] }];
        let k = kernel(Dtype::F32, vec![buf(&[7, 3], BufKind::Real), buf(&[4], BufKind::Idx)], buf(&[4, 3], BufKind::Real), gather);
        assert!(scop_fn(&k, "e").unwrap().contains("out[i0][i1] = in0[in1[i0]][i1];"));
        assert_eq!(static_control(&k), Err("it reads `in0` at an index loaded from `in1`, a gather".into()));
        assert_eq!(index_bounds(&k), vec![0, 7]);
        // out[ids[i0], j] += ct[i0, j]: `embed_vjp`, one reason.
        let ix = vec![Expr::LoadI(1, at(&[0])), Expr::Iv(1)];
        let scatter = vec![Stmt::For { var: 0, bound: 4, body: vec![Stmt::For { var: 1, bound: 3, body: vec![Stmt::Store { buf: 3, index: ix.clone(), value: Expr::Bin(BinOp::Add, bx(Expr::Load(3, ix)), bx(Expr::Load(2, at(&[0, 1])))) }] }] }];
        let k = kernel(Dtype::F64, vec![buf(&[7, 3], BufKind::Real), buf(&[4], BufKind::Idx), buf(&[4, 3], BufKind::Real)], buf(&[7, 3], BufKind::Real), scatter);
        assert_eq!(static_control(&k), Err("it writes `out` at an index loaded from `in1`, a scatter".into()));
        assert_eq!(index_bounds(&k), vec![0, 7, 0]);
        // real(k) == real(y[i0]) over k < 9: the labels below the classes;
        // an index read as a value and compared with nothing, zeros.
        let hit = Expr::Cmp(CmpOp::Eq, bx(Expr::ToF(bx(Expr::Iv(1)))), bx(Expr::ToF(bx(Expr::LoadI(0, at(&[0]))))));
        let onehot = vec![Stmt::For { var: 0, bound: 4, body: vec![Stmt::For { var: 1, bound: 9, body: vec![Stmt::Store { buf: 2, index: at(&[0, 1]), value: Expr::Select(bx(hit), bx(Expr::ToF(bx(Expr::LoadI(1, at(&[0]))))), bx(Expr::F(0.0))) }] }] }];
        let k = kernel(Dtype::F32, vec![buf(&[4], BufKind::Idx), buf(&[4], BufKind::Idx)], buf(&[4, 9], BufKind::Real), onehot);
        assert_eq!(static_control(&k), Ok(()));
        assert_eq!(index_bounds(&k), vec![9, 1]);
        // An assigned loop variable.
        let setl = vec![Stmt::For { var: 0, bound: 4, body: vec![Stmt::SetL { var: 1, value: Expr::Bin(BinOp::IAdd, bx(Expr::Iv(0)), bx(Expr::I(1))) }, Stmt::Store { buf: 1, index: at(&[0]), value: Expr::Load(0, at(&[1])) }] }];
        let k = kernel(Dtype::F32, vec![buf(&[5], BufKind::Real)], buf(&[4], BufKind::Real), setl);
        assert!(static_control(&k).unwrap_err().starts_with("it assigns the loop variable `i1`"));
    }

    #[test]
    fn integer_weights_and_lanes_are_refused() {
        let mut k = dot();
        k.inputs[1].kind = BufKind::Quant { bits: 8, over: 0 };
        assert!(scop_fn(&k, "q").unwrap_err().contains("integer weights"));
        let mut k = dot();
        k.lanes = 4;
        assert!(scop_fn(&k, "v").unwrap_err().contains("one lane"));
    }

    /// The draws are splitmix64's, each real exact in `f64`, each index
    /// below its bound.
    #[test]
    fn the_draws_are_splitmix64() {
        // splitmix64 from 0: its published first output.
        let mut g = Draw(0);
        assert_eq!(g.next_u64(), 0xe220_a839_7b1d_cdaf);
        let k = kernel(Dtype::F32, vec![buf(&[1000], BufKind::Real), buf(&[4], BufKind::Idx)], buf(&[4], BufKind::Real), vec![Stmt::For { var: 0, bound: 4, body: vec![Stmt::Store { buf: 2, index: at(&[0]), value: Expr::Load(0, vec![Expr::LoadI(1, at(&[0]))]) }] }]);
        let d = draw_inputs(&k, 3, 5);
        let (Drawn::Real(x), Drawn::Idx(i)) = (&d[0], &d[1]) else { panic!("{d:?}") };
        assert!(x.iter().all(|v| (-1.0..1.0).contains(v) && (v + 1.0) * 2f64.powi(52) == ((v + 1.0) * 2f64.powi(52)).floor()));
        assert!(i.iter().all(|v| (0..1000).contains(v)));
        assert_eq!(draw_inputs(&k, 3, 5), d);
        assert_ne!(draw_inputs(&k, 3, 6), d);
    }

    #[test]
    fn pollys_scops_are_read_per_function() {
        let out = "Printing analysis 'Polly - Generate an AST of the SCoP (isl)'%4 => %30' in function 'k_3':\n:: isl ast :: k_3 :: %4---%30\n\nif (1)\n\n    for (int c0 = 0; c0 <= 1; c0 += 1)\n      for (int c1 = 0; c1 <= 7; c1 += 1)\n        Stmt2(c0, c1);\n\nelse\n    {  /* original code */ }\n\nPrinting analysis 'Polly'%3 => %18' in function 'presk_f32_to_bf16':\n:: isl ast :: presk_f32_to_bf16 :: %3---%18\n\nif (1)\n\n    Stmt0();\n\nelse\n    {  /* original code */ }\n";
        assert_eq!(polly_scops(out, "k_3"), vec![2]);
        assert_eq!(polly_scops(out, "presk_f32_to_bf16"), vec![0]);
        assert_eq!(polly_scops(out, "k_4"), Vec::<usize>::new());
    }
}
