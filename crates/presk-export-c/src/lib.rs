//! The C export (plan 0.9.0): a Presk entry as a C program that computes the
//! bits of the native backend. Step 1 is the kernel printer: a PSK-Kernel
//! kernel becomes one C function ([`kernel_fn`]), its textual form (what
//! `presk dump --stage kernel` prints) the comment above it (§3.2). Step 2
//! is the C the program holds beside its own ([`RUNTIME`]): the runtime
//! (`presk_rt.h`, `presk_rt.c`: conversions between machine types, the
//! `.psk.d` reader and writer, the arena) and the transcendental functions
//! (`presk_libm.c`), D78. Steps 3 and 4 are the host ([`host`]): a recording
//! of the native backend's run of an entry becomes `model.c`, straight-line
//! (step 3, D79) or structured as C functions and loops ([`structure`], step
//! 4, D80), with `main.c`, a Makefile, a README and the conformance script.
//! Step 6 (D83) adds the stream `sample` draws from (`presk_rng`, ChaCha20
//! as `rand_chacha`'s, and `presk_sample` in `presk_rt.c`), the entry's `f64`
//! arguments as `double`s, and with `--serve` a generation function and its
//! greedy copy in one program with `generate.c`, `presk serve`'s loop
//! ([`host::export_serve`]). Step 8 (D85) is the SCoP mode ([`scop`]): each
//! nest of an entry at point zero as a standalone C function between
//! `#pragma scop` and `#pragma endscop`, in arrays of literal dimensions,
//! the input a polyhedral tool reads, with a driver that times it and checks
//! its bits.
//!
//! - `For`, `Set`, `SetI`, `SetL` and `If` become loops, locals and `if`s,
//!   every bound a literal. A load or a store is an access at its linear
//!   offset, the row-major strides written out as literals.
//! - The vector form becomes loops over a local array of the lane width.
//!   Lanes are distinct output elements, never partial sums of one (D42), so
//!   the loops keep the bits.
//! - `bf16` buffers are `uint16_t`; the kernel computes in `float`, a load
//!   widens and a store rounds to nearest, ties to even (`presk_rt.h`).
//! - The loop a split divides runs over the caller's `[lo, hi)`, the loops
//!   around it in full, as in the native backend ([`ranged`]). The host
//!   divides a launch among the workers the native backend would
//!   ([`workers`]), each thread of an OpenMP region calling the kernel on its
//!   piece ([`launch`], step 5, D81).
//! - `sqrt` and `abs` are C's; the other functions are `presk_exp`,
//!   `presk_expf`, … ([`calls`]), which `presk_rt.h` declares and
//!   `presk_libm.c` defines: musl 1.1.22's, the ones the `libm` crate of
//!   the native backend ports, bit for bit (D78).
//! - Integer weights (`BufKind::Quant`, step 7, D84): an `int8` input is
//!   `const int8_t *`, an `int4` one `const uint8_t *` (two to a byte), and
//!   their scales come after the output, `const float *sc<i>` for input `i`,
//!   as the native backend passes them. A load is `f32(q)·s`, one `float`
//!   multiplication (`(float)in0[e] * sc0[c]`, `presk_int4` for a nibble),
//!   widened in a `double` kernel; the scale's index is the element's with
//!   the scales' strides (`BufSpec::scale_strides`), zero along the axis
//!   they are constant along. A vector load is the same, lane by lane.
//!
//! The C is built with [`CFLAGS`], and bit identity rests on
//! `-ffp-contract=off` (D77, D78). The printed expressions keep the kernel's tree:
//! a parenthesis is dropped only where C's precedence and associativity give
//! the same tree back, so the compiler rounds each operation where the
//! kernel does.

use std::collections::BTreeSet;
use std::fmt::Write as _;

pub mod host;
pub mod scop;
pub mod structure;

use presk_kernel::ir::{BinOp, BufKind, BufSpec, CmpOp, Dtype, Expr, Kernel, Stmt, Threads, UnOp};

/// The header the kernels include: the `bf16` conversions, the saturating
/// conversion of a real to an integer, and the declarations of the
/// transcendental functions and of the runtime.
pub const PRESK_RT_H: &str = include_str!("../c/presk_rt.h");

/// The runtime: the `.psk.d` reader and writer, the arena (§3.3, §3.4), and
/// the stream `sample` draws from (step 6).
pub const PRESK_RT_C: &str = include_str!("../c/presk_rt.c");

/// The transcendental functions the kernels call, `presk_exp`, …: musl's,
/// the ones the native backend's `libm` crate ports (§3.5, rule 2; D78).
pub const PRESK_LIBM_C: &str = include_str!("../c/presk_libm.c");

/// The files an exported program holds beside its own, by name.
pub const RUNTIME: &[(&str, &str)] = &[("presk_rt.h", PRESK_RT_H), ("presk_rt.c", PRESK_RT_C), ("presk_libm.c", PRESK_LIBM_C)];

/// The flags an exported program is built with (§3.5): ISO C11, optimized,
/// and no contraction of a product and a sum into one rounding.
pub const CFLAGS: &[&str] = &["-std=c11", "-O2", "-ffp-contract=off"];

/// The warnings statement C3 asks the C to compile without.
pub const WARNINGS: &[&str] = &["-Wall", "-Wextra"];

/// The flag that builds the launches divided among threads as OpenMP
/// regions (§3.6); without it, each is one call, with the same bits.
pub const OPENMP: &str = "-fopenmp";

type Result<T> = std::result::Result<T, String>;

/// The loop a kernel's `[lo, hi)` bounds, `(depth, iterations)`: the one at
/// the depth its split states, or the outermost when it is parallel. The
/// native backend's rule (`ranged` in `presk-cranelift`), kept apart so that
/// the native backend's code does not change; `None` when no loop is, and the
/// range is unused.
pub fn ranged(k: &Kernel) -> Option<(usize, usize)> {
    match k.threads {
        Threads::At { depth, .. } => k.parallel_at(depth).map(|n| (depth, n)),
        Threads::Default | Threads::One => k.parallel_outer().map(|n| (0, n)),
    }
}

/// The workers a launch of `k` is divided among, `threads` the native
/// executor's (`PRESK_THREADS`, which `--threads` sets, or the cores): the
/// native backend's rule (`NativeExec::run` in `presk-cranelift`), kept apart
/// as [`ranged`] is. A split stated by the plan (`Threads::At`) takes its
/// workers, at most one per iteration, whatever the work; the executor's
/// default (`Threads::Default`) divides the outermost parallel loop among
/// every worker when there are two or more, the loop has two iterations or
/// more and the kernel's work reaches `PARALLEL_MIN_WORK`; a split stated as
/// none (`Threads::One`) is one worker. So the default is resolved when the
/// entry is exported (plan 0.9.0 §3.6).
pub fn workers(k: &Kernel, threads: usize) -> usize {
    let total = ranged(k).map_or(0, |(_, n)| n);
    match k.threads {
        Threads::Default => match ranged(k) {
            Some((_, n)) if threads > 1 && n >= 2 && presk_kernel::cost::kernel_cost(k).work >= presk_kernel::cost::PARALLEL_MIN_WORK => threads.min(n),
            _ => 1,
        },
        Threads::One => 1,
        Threads::At { workers, .. } => workers.min(total).max(1),
    }
}

/// A launch of the kernel function `name` on `args` (the inputs' and the
/// output's pointers) over its range `0..hi`, at indentation `ind`, divided
/// among `workers` (§3.6). One worker is one call over the whole range. More
/// is one OpenMP region: each thread of the team computes its piece of the
/// range, `hi·t/w` to `hi·(t+1)/w`, the native backend's pieces, and the
/// loops above the divided one in full. The team is the launch's workers, at
/// most the threads OpenMP would use (`presk_team`: `OMP_NUM_THREADS`, or
/// the cores), and the pieces are the team's, whatever size it gets. Built
/// without OpenMP, the launch is the one call, with the same bits (§3.5,
/// rule 4). The region's variables are `t` and `w`, which no name of the host
/// takes.
pub fn launch(ind: &str, name: &str, args: &[String], hi: usize, workers: usize) -> String {
    let args = args.join(", ");
    if workers <= 1 {
        return format!("{ind}{name}({args}, 0, {hi});\n");
    }
    format!(
        "#ifdef _OPENMP\n#pragma omp parallel num_threads(presk_team({workers}))\n{ind}{{\n{ind}    int64_t t = omp_get_thread_num(), w = omp_get_num_threads();\n{ind}    {name}({args}, {hi} * t / w, {hi} * (t + 1) / w);\n{ind}}}\n#else\n{ind}{name}({args}, 0, {hi});\n#endif\n"
    )
}

/// The transcendental functions a kernel calls, by their C names
/// (`presk_expf`, …).
pub fn calls(k: &Kernel) -> BTreeSet<String> {
    fn expr(e: &Expr, suffix: &str, out: &mut BTreeSet<String>) {
        match e {
            Expr::Un(op, a) => {
                if let Some(n) = transcendental(*op) {
                    out.insert(format!("presk_{n}{suffix}"));
                }
                expr(a, suffix, out);
            }
            Expr::Load(_, ix) | Expr::LoadI(_, ix) | Expr::VLoad(_, ix) => ix.iter().for_each(|x| expr(x, suffix, out)),
            Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) | Expr::VBin(_, a, b) => {
                expr(a, suffix, out);
                expr(b, suffix, out);
            }
            Expr::Select(c, a, b) => {
                expr(c, suffix, out);
                expr(a, suffix, out);
                expr(b, suffix, out);
            }
            Expr::ToF(a) | Expr::ToI(a) | Expr::VSplat(a) => expr(a, suffix, out),
            Expr::F(_) | Expr::I(_) | Expr::Iv(_) | Expr::Sv(_) | Expr::Si(_) | Expr::Vv(_) => {}
        }
    }
    let suffix = if k.dtype == Dtype::F64 { "" } else { "f" };
    let mut out = BTreeSet::new();
    walk_exprs(&k.body, &mut |e| expr(e, suffix, &mut out));
    out
}

fn transcendental(op: UnOp) -> Option<&'static str> {
    match op {
        UnOp::Exp => Some("exp"),
        UnOp::Log => Some("log"),
        UnOp::Tanh => Some("tanh"),
        UnOp::Sin => Some("sin"),
        UnOp::Cos => Some("cos"),
        UnOp::Erf => Some("erf"),
        UnOp::Neg | UnOp::Sqrt | UnOp::Abs => None,
    }
}

/// Every expression at the top of a statement: values, indices, conditions.
fn walk_exprs(stmts: &[Stmt], f: &mut impl FnMut(&Expr)) {
    for s in stmts {
        match s {
            Stmt::For { body, .. } => walk_exprs(body, f),
            Stmt::Set { value, .. } | Stmt::SetI { value, .. } | Stmt::SetL { value, .. } | Stmt::VSet { value, .. } => f(value),
            Stmt::Store { index, value, .. } | Stmt::StoreI { index, value, .. } | Stmt::VStore { index, value, .. } => {
                index.iter().for_each(&mut *f);
                f(value);
            }
            Stmt::If { cond, then } => {
                f(cond);
                walk_exprs(then, f);
            }
        }
    }
}

/// The C element type of a buffer of `k`.
fn elem(k: &Kernel, s: &BufSpec) -> Result<&'static str> {
    Ok(match s.kind {
        BufKind::Idx => "int64_t",
        BufKind::Real => match k.dtype {
            Dtype::F64 => "double",
            Dtype::F32 => "float",
            Dtype::Bf16 => "uint16_t",
        },
        BufKind::Quant { bits: 8, .. } => "int8_t",
        BufKind::Quant { bits: 4, .. } => "uint8_t",
        BufKind::Quant { bits, .. } => return Err(format!("kernel `{}` reads int{bits} weights: only int8 and int4 exist", k.name)),
    })
}

/// The inputs of `k` that hold integer weights, whose scales the kernel
/// takes after its output, in input order (`NativeExec::run`'s order).
pub fn quant_inputs(k: &Kernel) -> Vec<usize> {
    k.inputs.iter().enumerate().filter(|(_, s)| matches!(s.kind, BufKind::Quant { .. })).map(|(i, _)| i).collect()
}

/// A kernel as one C function, `static void name(const T *restrict in0, …,
/// T *restrict out, [const float *restrict sc<i>, …,] int64_t lo, int64_t
/// hi)`, with the kernel's textual form as the comment above it: the scales
/// of each input `i` of integer weights after the output (step 7).
pub fn kernel_fn(k: &Kernel, name: &str) -> Result<String> {
    let mut params = Vec::new();
    for (i, s) in k.inputs.iter().enumerate() {
        params.push(format!("const {} *restrict in{i}", elem(k, s)?));
    }
    params.push(format!("{} *restrict out", elem(k, &k.output)?));
    for i in quant_inputs(k) {
        params.push(format!("const float *restrict sc{i}"));
    }
    params.push("int64_t lo".into());
    params.push("int64_t hi".into());
    let mut p = Printer::new(k, false);
    let range = ranged(k);
    p.block(&k.body, 1, range.map(|(d, _)| d))?;

    let mut text = String::new();
    for line in k.to_string().lines() {
        let _ = writeln!(text, "// {line}");
    }
    let _ = writeln!(text, "// {}", range_note(k, range));
    let _ = writeln!(text, "static void {name}({}) {{", params.join(", "));
    text.push_str(&p.declarations());
    text.push_str(&p.body);
    text.push_str("}\n");
    Ok(text)
}

/// What the function's `[lo, hi)` is, for its comment.
fn range_note(k: &Kernel, range: Option<(usize, usize)>) -> String {
    let Some((depth, n)) = range else { return "lo, hi: unused, no loop of this kernel may be divided among threads".into() };
    let mut stmts = k.body.as_slice();
    for _ in 0..depth {
        if let [Stmt::For { body, .. }] = stmts {
            stmts = body;
        }
    }
    let var = match stmts {
        [Stmt::For { var, .. }] => format!("i{var}"),
        _ => "?".into(),
    };
    let what = if depth == 0 { "the outermost loop".to_string() } else { format!("the loop at depth {depth}") };
    format!("lo, hi: the range of {var} in 0..{n}, {what}, which a thread split divides")
}

/// The kernel `kernel` behind the native backend's calling convention,
/// `void name(void *const *bufs, int64_t lo, int64_t hi)`: the inputs' and
/// the output's pointers in order, then the scales of the integer inputs
/// (step 7), then the range. What a loader that knows no kernel's signature
/// calls, as the Cranelift backend calls its own.
pub fn entry_fn(k: &Kernel, kernel: &str, name: &str) -> Result<String> {
    let mut args = Vec::new();
    for (i, s) in k.inputs.iter().enumerate() {
        args.push(format!("(const {} *)bufs[{i}]", elem(k, s)?));
    }
    args.push(format!("({} *)bufs[{}]", elem(k, &k.output)?, k.inputs.len()));
    for (j, _) in quant_inputs(k).into_iter().enumerate() {
        args.push(format!("(const float *)bufs[{}]", k.inputs.len() + 1 + j));
    }
    Ok(format!("void {name}(void *const *bufs, int64_t lo, int64_t hi) {{\n    {kernel}({}, lo, hi);\n}}\n", args.join(", ")))
}

// ----- expressions -------------------------------------------------------------

// C's precedence levels, higher binding tighter.
const PRIMARY: u8 = 15;
const UNARY: u8 = 14;
const MUL: u8 = 13;
const ADD: u8 = 12;
const REL: u8 = 10;
const EQ: u8 = 9;
const COND: u8 = 3;

/// An expression's text and the precedence of its outermost operator.
type Text = (String, u8);

fn wrap((s, p): Text, need: u8) -> String {
    if p < need { format!("({s})") } else { s }
}

/// A left-associative binary operator: the left operand may share its
/// level, the right may not, so `a - (b - c)` keeps its parenthesis and
/// `(a - b) - c` loses it, and C builds the kernel's tree either way.
fn binary(op: &str, prec: u8, a: Text, b: Text) -> Text {
    let l = wrap(a, prec);
    let r = if b.0.starts_with('-') { format!("({})", b.0) } else { wrap(b, prec + 1) };
    (format!("{l} {op} {r}"), prec)
}

fn neg(a: Text) -> Text {
    let s = if a.0.starts_with('-') { format!("({})", a.0) } else { wrap(a, UNARY) };
    (format!("-{s}"), UNARY)
}

fn cast(ty: &str, a: Text) -> Text {
    (format!("({ty}){}", wrap(a, UNARY)), UNARY)
}

fn call(f: &str, a: Text) -> Text {
    (format!("{f}({})", a.0), PRIMARY)
}

fn ternary(c: Text, a: Text, b: Text) -> Text {
    (format!("{} ? {} : {}", wrap(c, COND + 1), wrap(a, COND + 1), wrap(b, COND)), COND)
}

fn int_lit(i: i64) -> Text {
    let s = if i == i64::MIN {
        "INT64_MIN".to_string()
    } else if i32::try_from(i).is_ok() {
        i.to_string()
    } else {
        format!("INT64_C({i})")
    };
    let p = if s.starts_with('-') { UNARY } else { PRIMARY };
    (s, p)
}

/// A kernel's C: the body of [`kernel_fn`], or with `scop` the SCoP mode's
/// form ([`scop`]): an access indexes a multidimensional array one axis at a
/// time instead of at its linear offset, no loop runs over a range, and a
/// branch on a computed value that only assigns scalars is printed as a
/// selection of each value.
pub(crate) struct Printer<'k> {
    k: &'k Kernel,
    scop: bool,
    /// Conditions a SCoP holds in a variable, `c0`, `c1`, …: one per branch
    /// printed as selections of more than one value.
    conds: usize,
    /// `double` or `float`: `bf16` computes in `float`.
    real: &'static str,
    strides: Vec<Vec<usize>>,
    body: String,
    /// Variables written and read, per class: loops, reals, integers,
    /// vectors; and the buffers read.
    written: [BTreeSet<usize>; 4],
    read: [BTreeSet<usize>; 4],
    bufs_read: BTreeSet<usize>,
}

const LOOP: usize = 0;
const REAL: usize = 1;
const INT: usize = 2;
const VEC: usize = 3;

impl<'k> Printer<'k> {
    pub(crate) fn new(k: &'k Kernel, scop: bool) -> Self {
        let real = if k.dtype == Dtype::F64 { "double" } else { "float" };
        let strides = k.inputs.iter().map(|s| s.strides()).chain(std::iter::once(k.output.strides())).collect();
        let mut p = Printer { k, scop, conds: 0, real, strides, body: String::new(), written: Default::default(), read: Default::default(), bufs_read: BTreeSet::new() };
        p.collect(&k.body);
        p
    }

    fn collect(&mut self, stmts: &[Stmt]) {
        fn expr(p: &mut Printer, e: &Expr) {
            match e {
                Expr::Iv(v) => {
                    p.read[LOOP].insert(*v);
                }
                Expr::Sv(v) => {
                    p.read[REAL].insert(*v);
                }
                Expr::Si(v) => {
                    p.read[INT].insert(*v);
                }
                Expr::Vv(v) => {
                    p.read[VEC].insert(*v);
                }
                Expr::Load(b, ix) | Expr::LoadI(b, ix) | Expr::VLoad(b, ix) => {
                    p.bufs_read.insert(*b);
                    ix.iter().for_each(|x| expr(p, x));
                }
                Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) | Expr::VBin(_, a, b) => {
                    expr(p, a);
                    expr(p, b);
                }
                Expr::Select(c, a, b) => {
                    expr(p, c);
                    expr(p, a);
                    expr(p, b);
                }
                Expr::Un(_, a) | Expr::ToF(a) | Expr::ToI(a) | Expr::VSplat(a) => expr(p, a),
                Expr::F(_) | Expr::I(_) => {}
            }
        }
        for s in stmts {
            match s {
                Stmt::For { var, body, .. } => {
                    self.written[LOOP].insert(*var);
                    self.read[LOOP].insert(*var);
                    self.collect(body);
                }
                Stmt::Set { var, value } => {
                    self.written[REAL].insert(*var);
                    expr(self, value);
                }
                Stmt::SetI { var, value } => {
                    self.written[INT].insert(*var);
                    expr(self, value);
                }
                Stmt::SetL { var, value } => {
                    self.written[LOOP].insert(*var);
                    expr(self, value);
                }
                Stmt::VSet { var, value } => {
                    self.written[VEC].insert(*var);
                    expr(self, value);
                }
                Stmt::Store { index, value, .. } | Stmt::StoreI { index, value, .. } | Stmt::VStore { index, value, .. } => {
                    index.iter().for_each(|x| expr(self, x));
                    expr(self, value);
                }
                Stmt::If { cond, then } => {
                    expr(self, cond);
                    self.collect(then);
                }
            }
        }
    }

    /// The locals, each at zero as the native backend starts them, then a
    /// `(void)` for what is never read: parameters and variables written
    /// only, which `-Wall -Wextra` would otherwise report.
    pub(crate) fn declarations(&self) -> String {
        let mut out = String::new();
        let zero = if self.real == "float" { "0.0f" } else { "0.0" };
        let names = |class: usize, prefix: &str, init: &str| -> Vec<String> {
            self.written[class].union(&self.read[class]).map(|v| format!("{prefix}{v}{init}")).collect()
        };
        let lanes = self.k.lanes;
        for (ty, vars) in [
            ("int64_t", names(LOOP, "i", " = 0")),
            (self.real, names(REAL, "s", &format!(" = {zero}"))),
            ("int64_t", names(INT, "n", " = 0")),
            (self.real, names(VEC, "v", &format!("[{lanes}] = {{0}}"))),
        ] {
            if !vars.is_empty() {
                let _ = writeln!(out, "    {ty} {};", vars.join(", "));
            }
        }
        if self.conds > 0 {
            let _ = writeln!(out, "    int {};", (0..self.conds).map(|c| format!("c{c} = 0")).collect::<Vec<_>>().join(", "));
        }
        let mut unused: Vec<String> = Vec::new();
        for b in 0..self.k.inputs.len() {
            if !self.bufs_read.contains(&b) {
                unused.push(format!("in{b}"));
                if matches!(self.k.inputs[b].kind, BufKind::Quant { .. }) {
                    unused.push(format!("sc{b}"));
                }
            }
        }
        if !self.scop && ranged(self.k).is_none() {
            unused.push("lo".into());
            unused.push("hi".into());
        }
        for (class, prefix) in [(LOOP, "i"), (REAL, "s"), (INT, "n"), (VEC, "v")] {
            for v in self.written[class].difference(&self.read[class]) {
                unused.push(format!("{prefix}{v}"));
            }
        }
        if !unused.is_empty() {
            let _ = writeln!(out, "    {}", unused.iter().map(|u| format!("(void){u};")).collect::<Vec<_>>().join(" "));
        }
        out
    }

    fn spec(&self, b: usize) -> &BufSpec {
        if b == self.k.out_buf() { &self.k.output } else { &self.k.inputs[b] }
    }

    fn buf_name(&self, b: usize) -> String {
        if b == self.k.out_buf() { "out".into() } else { format!("in{b}") }
    }

    /// Element `ix` of buffer `b`: at its linear offset, `in0[i0 * 8 + i1]`;
    /// in a SCoP, one index per axis, `in0[i0][i1]`, each multiplied out as
    /// an offset is, and `[0]` for a buffer of rank 0.
    fn access(&self, b: usize, ix: &[Expr]) -> Result<String> {
        if !self.scop {
            return Ok(format!("{}[{}]", self.buf_name(b), self.offset(b, ix)?));
        }
        let rank = self.spec(b).dims.len();
        if ix.len() != rank {
            return Err(format!("kernel `{}`: buffer %{b} indexed by {} expressions, it has {rank} axes", self.k.name, ix.len()));
        }
        let mut at = self.buf_name(b);
        if ix.is_empty() {
            at.push_str("[0]");
        }
        for e in ix {
            let _ = write!(at, "[{}]", self.offset_by(b, std::slice::from_ref(e), &[1])?);
        }
        Ok(at)
    }

    /// The SCoP mode prints point zero, one lane.
    fn no_lanes(&self) -> Result<()> {
        if self.scop { Err(format!("kernel `{}` has vector lanes, and the SCoP mode prints each nest on one lane", self.k.name)) } else { Ok(()) }
    }

    fn narrow(&self) -> bool {
        self.k.dtype == Dtype::Bf16
    }

    /// The linear offset of an index: each axis's index times its stride,
    /// multiplied out. An affine index (sums and literal multiples of loop
    /// and integer variables) becomes one coefficient per variable and one
    /// constant, `i2 * 48 + i1 * 8 + 3`; any other, such as a loaded index,
    /// is a term times its stride. Integer arithmetic, so no rounding moves.
    fn offset(&self, b: usize, ix: &[Expr]) -> Result<String> {
        self.offset_by(b, ix, &self.strides[b])
    }

    /// The same with other strides: those of the scales of integer weights,
    /// zero along the axis the scales are constant along, whose index then
    /// does not appear.
    fn offset_by(&self, b: usize, ix: &[Expr], strides: &[usize]) -> Result<String> {
        if ix.len() != strides.len() {
            return Err(format!("kernel `{}`: buffer %{b} indexed by {} expressions, it has {} axes", self.k.name, ix.len(), strides.len()));
        }
        let mut coef: Vec<(String, i64)> = Vec::new();
        let mut constant: i64 = 0;
        let mut others: Vec<String> = Vec::new();
        for (e, &s) in ix.iter().zip(strides) {
            let s = s as i64;
            if s == 0 {
                continue;
            }
            match affine(e) {
                Some((vars, c)) => {
                    constant += c * s;
                    for (v, a) in vars {
                        match coef.iter_mut().find(|(w, _)| *w == v) {
                            Some((_, x)) => *x += a * s,
                            None => coef.push((v, a * s)),
                        }
                    }
                }
                None => {
                    let t = self.int(e)?;
                    others.push(if s == 1 { wrap(t, ADD) } else { format!("{} * {}", wrap(t, MUL), int_lit(s).0) });
                }
            }
        }
        let mut text = String::new();
        let term = |text: &mut String, neg: bool, t: String| {
            if text.is_empty() {
                text.push_str(&if neg { format!("-{t}") } else { t });
            } else {
                let _ = write!(text, " {} {t}", if neg { '-' } else { '+' });
            }
        };
        for (v, a) in coef.into_iter().filter(|(_, a)| *a != 0) {
            let m = a.unsigned_abs() as i64;
            term(&mut text, a < 0, if m == 1 { v } else { format!("{v} * {}", int_lit(m).0) });
        }
        for t in others {
            term(&mut text, false, t);
        }
        if constant != 0 || text.is_empty() {
            term(&mut text, constant < 0, int_lit(constant.abs()).0);
        }
        Ok(text)
    }

    fn real_lit(&self, x: f64) -> Text {
        let wide = self.real == "double";
        let y = if wide { x } else { x as f32 as f64 };
        if y.is_nan() {
            return ("NAN".into(), PRIMARY);
        }
        if y.is_infinite() {
            let s = if wide { "HUGE_VAL" } else { "HUGE_VALF" };
            return if y < 0.0 { (format!("-{s}"), UNARY) } else { (s.into(), PRIMARY) };
        }
        // Rust's shortest representation reads back as the same number, and
        // the C compiler rounds a decimal constant correctly.
        let s = if wide { format!("{x:?}") } else { format!("{:?}f", x as f32) };
        let p = if s.starts_with('-') { UNARY } else { PRIMARY };
        (s, p)
    }

    /// Element `ix` of integer input `b` (step 7), lane `lane` on along the
    /// last axis when it is not empty: `f32(q)·s`, one `float`
    /// multiplication, as `presk_rt::QBuf::value` and the native backend's
    /// `load_quant` compute it; widened exactly in a `double` kernel. Its
    /// scale is at the element's index with the scales' strides, which move
    /// with the lanes along the last axis unless it is the one they are
    /// constant along.
    fn quant_load(&self, b: usize, bits: u8, ix: &[Expr], lane: &str) -> Result<Text> {
        let sstrides = self.spec(b).scale_strides();
        let at = |off: String, step: usize| -> String {
            match (lane, step, off.as_str()) {
                ("", _, _) | (_, 0, _) => off,
                (l, 1, "0") => l.to_string(),
                (l, 1, _) => format!("{off} + {l}"),
                (l, s, "0") => format!("{l} * {s}"),
                (l, s, _) => format!("{off} + {l} * {s}"),
            }
        };
        let e = at(self.offset(b, ix)?, 1);
        let c = at(self.offset_by(b, ix, &sstrides)?, sstrides.last().copied().unwrap_or(0));
        let q = if bits == 4 { format!("presk_int4(in{b}, {e})") } else { format!("in{b}[{e}]") };
        let x = binary("*", MUL, cast("float", (q, PRIMARY)), (format!("sc{b}[{c}]"), PRIMARY));
        Ok(if self.real == "double" { cast("double", x) } else { x })
    }

    /// A real of the kernel's arithmetic type: the native backend's `gen_f`.
    fn real(&self, e: &Expr) -> Result<Text> {
        Ok(match e {
            Expr::F(x) => self.real_lit(*x),
            Expr::Load(b, ix) => {
                if self.scop && matches!(self.spec(*b).kind, BufKind::Quant { .. }) {
                    return Err(format!("kernel `{}` reads integer weights, which the SCoP mode does not print", self.k.name));
                }
                let at = self.access(*b, ix)?;
                match self.spec(*b).kind {
                    BufKind::Real if self.narrow() => call("presk_bf16_to_f32", (at, PRIMARY)),
                    BufKind::Real => (at, PRIMARY),
                    BufKind::Idx => return Err(format!("kernel `{}`: a real load from index buffer %{b}", self.k.name)),
                    BufKind::Quant { bits, .. } => self.quant_load(*b, bits, ix, "")?,
                }
            }
            Expr::Sv(v) => (format!("s{v}"), PRIMARY),
            Expr::Bin(op @ (BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div), a, b) => {
                let (s, p) = match op {
                    BinOp::Add => ("+", ADD),
                    BinOp::Sub => ("-", ADD),
                    BinOp::Mul => ("*", MUL),
                    _ => ("/", MUL),
                };
                binary(s, p, self.real(a)?, self.real(b)?)
            }
            Expr::Un(op, a) => {
                let x = self.real(a)?;
                let f = if self.real == "float" { "f" } else { "" };
                match op {
                    UnOp::Neg => neg(x),
                    UnOp::Sqrt => call(&format!("sqrt{f}"), x),
                    UnOp::Abs => call(&format!("fabs{f}"), x),
                    op => call(&format!("presk_{}{f}", transcendental(*op).expect("the other operators are above")), x),
                }
            }
            Expr::Select(c, a, b) => ternary(self.cond(c)?, self.real(a)?, self.real(b)?),
            Expr::ToF(a) => cast(self.real, self.int(a)?),
            Expr::Cmp(..) => return Err(format!("kernel `{}`: a comparison used as a real", self.k.name)),
            Expr::Vv(_) | Expr::VLoad(..) | Expr::VSplat(_) | Expr::VBin(..) => return Err(format!("kernel `{}`: a vector in a scalar position", self.k.name)),
            // An integer where a real is expected is converted, as `gen_f`
            // converts it.
            other => cast(self.real, self.int(other)?),
        })
    }

    /// An integer, `int64_t`: the native backend's `gen_i`.
    fn int(&self, e: &Expr) -> Result<Text> {
        Ok(match e {
            Expr::I(i) => int_lit(*i),
            Expr::Iv(v) => (format!("i{v}"), PRIMARY),
            Expr::Si(v) => (format!("n{v}"), PRIMARY),
            Expr::LoadI(b, ix) => match self.spec(*b).kind {
                BufKind::Idx => (self.access(*b, ix)?, PRIMARY),
                _ => return Err(format!("kernel `{}`: an index load from real buffer %{b}", self.k.name)),
            },
            Expr::Bin(BinOp::IAdd, a, b) => binary("+", ADD, self.int(a)?, self.int(b)?),
            Expr::Bin(BinOp::IMul, a, b) => binary("*", MUL, self.int(a)?, self.int(b)?),
            Expr::Select(c, a, b) => ternary(self.cond(c)?, self.int(a)?, self.int(b)?),
            Expr::Cmp(..) => return Err(format!("kernel `{}`: a comparison used as an integer", self.k.name)),
            Expr::Vv(_) | Expr::VLoad(..) | Expr::VSplat(_) | Expr::VBin(..) => return Err(format!("kernel `{}`: a vector in a scalar position", self.k.name)),
            // `ToI` and a real where an integer is expected: rounded toward
            // zero and saturated, as `fcvt_to_sint_sat`.
            Expr::ToI(a) => self.to_int(self.real(a)?),
            other => self.to_int(self.real(other)?),
        })
    }

    fn to_int(&self, x: Text) -> Text {
        call(if self.real == "float" { "presk_f32_to_i64" } else { "presk_f64_to_i64" }, x)
    }

    /// A condition: a comparison of reals (ordered, as Cranelift's), or an
    /// integer against zero.
    fn cond(&self, e: &Expr) -> Result<Text> {
        Ok(match e {
            Expr::Cmp(op, a, b) => {
                let (s, p) = match op {
                    CmpOp::Lt => ("<", REL),
                    CmpOp::Le => ("<=", REL),
                    CmpOp::Gt => (">", REL),
                    CmpOp::Ge => (">=", REL),
                    CmpOp::Eq => ("==", EQ),
                };
                binary(s, p, self.real(a)?, self.real(b)?)
            }
            other => binary("!=", EQ, self.int(other)?, ("0".into(), PRIMARY)),
        })
    }

    /// Lane `l` of a vector expression: `gen_v`, one lane at a time. A
    /// scalar in a vector position is the same in every lane.
    fn lane(&self, e: &Expr) -> Result<Text> {
        self.no_lanes()?;
        Ok(match e {
            Expr::Vv(v) => (format!("v{v}[l]"), PRIMARY),
            Expr::VLoad(b, ix) => {
                let off = self.offset(*b, ix)?;
                let at = format!("{}[{}]", self.buf_name(*b), if off == "0" { "l".to_string() } else { format!("{off} + l") });
                match self.spec(*b).kind {
                    BufKind::Real if self.narrow() => call("presk_bf16_to_f32", (at, PRIMARY)),
                    BufKind::Real => (at, PRIMARY),
                    BufKind::Idx => return Err(format!("kernel `{}`: a vector load from index buffer %{b}", self.k.name)),
                    BufKind::Quant { bits, .. } => self.quant_load(*b, bits, ix, "l")?,
                }
            }
            Expr::VSplat(x) => self.real(x)?,
            Expr::VBin(op, a, b) => {
                let (s, p) = match op {
                    BinOp::Add | BinOp::IAdd => ("+", ADD),
                    BinOp::Sub => ("-", ADD),
                    BinOp::Mul | BinOp::IMul => ("*", MUL),
                    BinOp::Div => ("/", MUL),
                };
                binary(s, p, self.lane(a)?, self.lane(b)?)
            }
            other => self.real(other)?,
        })
    }

    // ----- statements ------------------------------------------------------------

    fn line(&mut self, ind: usize, s: &str) {
        let _ = writeln!(self.body, "{}{s}", "    ".repeat(ind));
    }

    /// Statements; `range`: the depth, from here, of the loop that runs over
    /// `[lo, hi)`. The loops above it form a perfect nest and run in full, as
    /// the native backend's `range_at` has them.
    pub(crate) fn block(&mut self, stmts: &[Stmt], ind: usize, range: Option<usize>) -> Result<()> {
        if let (Some(depth), [Stmt::For { var, bound, body }]) = (range, stmts) {
            if depth == 0 {
                self.line(ind, &format!("for (i{var} = lo; i{var} < hi; i{var}++) {{"));
                self.block(body, ind + 1, None)?;
            } else {
                self.line(ind, &format!("for (i{var} = 0; i{var} < {}; i{var}++) {{", int_lit(*bound as i64).0));
                self.block(body, ind + 1, Some(depth - 1))?;
            }
            self.line(ind, "}");
            return Ok(());
        }
        for s in stmts {
            self.stmt(s, ind)?;
        }
        Ok(())
    }

    fn store_target(&self, buf: usize, want: BufKind) -> Result<()> {
        if buf != self.k.out_buf() {
            return Err(format!("kernel `{}` stores to its input %{buf}", self.k.name));
        }
        let kind = self.k.output.kind;
        let ok = match want {
            BufKind::Idx => kind == BufKind::Idx,
            _ => kind == BufKind::Real,
        };
        if ok { Ok(()) } else { Err(format!("kernel `{}`: a {} store to a {} output", self.k.name, if want == BufKind::Idx { "index" } else { "real" }, if kind == BufKind::Idx { "index" } else { "real" })) }
    }

    /// The value stored, rounded to `bf16` when the buffers are.
    fn stored(&self, x: Text) -> String {
        if self.narrow() { call("presk_f32_to_bf16", x).0 } else { x.0 }
    }

    fn stmt(&mut self, s: &Stmt, ind: usize) -> Result<()> {
        match s {
            Stmt::For { var, bound, body } => {
                self.line(ind, &format!("for (i{var} = 0; i{var} < {}; i{var}++) {{", int_lit(*bound as i64).0));
                self.block(body, ind + 1, None)?;
                self.line(ind, "}");
            }
            Stmt::Set { var, value } => {
                let x = self.real(value)?.0;
                self.line(ind, &format!("s{var} = {x};"));
            }
            Stmt::SetI { var, value } => {
                let x = self.int(value)?.0;
                self.line(ind, &format!("n{var} = {x};"));
            }
            Stmt::SetL { var, value } => {
                let x = self.int(value)?.0;
                self.line(ind, &format!("i{var} = {x};"));
            }
            Stmt::VSet { var, value } => {
                self.no_lanes()?;
                let x = self.lane(value)?.0;
                let lanes = self.k.lanes;
                self.line(ind, &format!("for (int l = 0; l < {lanes}; l++) v{var}[l] = {x};"));
            }
            Stmt::VStore { buf, index, value } => {
                self.no_lanes()?;
                self.store_target(*buf, BufKind::Real)?;
                let off = self.offset(*buf, index)?;
                let at = if off == "0" { "out[l]".to_string() } else { format!("out[{off} + l]") };
                let x = self.lane(value)?;
                let lanes = self.k.lanes;
                // The native backend computes every lane before it stores
                // any. A value that reads the output could otherwise see a
                // lane this store has already written, so it goes through a
                // temporary.
                if reads(value, *buf) {
                    self.line(ind, "{");
                    self.line(ind + 1, &format!("{} t[{lanes}];", self.real));
                    self.line(ind + 1, &format!("for (int l = 0; l < {lanes}; l++) t[l] = {};", x.0));
                    let t = self.stored(("t[l]".into(), PRIMARY));
                    self.line(ind + 1, &format!("for (int l = 0; l < {lanes}; l++) {at} = {t};"));
                    self.line(ind, "}");
                } else {
                    let v = self.stored(x);
                    self.line(ind, &format!("for (int l = 0; l < {lanes}; l++) {at} = {v};"));
                }
            }
            Stmt::Store { buf, index, value } => {
                self.store_target(*buf, BufKind::Real)?;
                let at = self.access(*buf, index)?;
                let v = self.stored(self.real(value)?);
                self.line(ind, &format!("{at} = {v};"));
            }
            Stmt::StoreI { buf, index, value } => {
                self.store_target(*buf, BufKind::Idx)?;
                let at = self.access(*buf, index)?;
                let v = self.int(value)?.0;
                self.line(ind, &format!("{at} = {v};"));
            }
            // In a SCoP, a branch whose body only assigns scalars (the
            // running maximum of a softmax, an argmax) is a selection of
            // each value: `m = x > m ? x : m`. The same bits, and control
            // that no longer depends on data. With more than one value, the
            // condition is computed once, before any of them changes.
            Stmt::If { cond, then } if self.scop && then.iter().all(|s| matches!(s, Stmt::Set { .. } | Stmt::SetI { .. })) => {
                let c = self.cond(cond)?;
                let c = if then.len() == 1 {
                    c
                } else {
                    let name = format!("c{}", self.conds);
                    self.conds += 1;
                    self.line(ind, &format!("{name} = {};", c.0));
                    (name, PRIMARY)
                };
                for s in then {
                    let (v, x) = match s {
                        Stmt::Set { var, value } => (format!("s{var}"), self.real(value)?),
                        Stmt::SetI { var, value } => (format!("n{var}"), self.int(value)?),
                        _ => unreachable!("only assignments"),
                    };
                    let sel = ternary(c.clone(), x, (v.clone(), PRIMARY)).0;
                    self.line(ind, &format!("{v} = {sel};"));
                }
            }
            Stmt::If { cond, then } => {
                let c = self.cond(cond)?.0;
                self.line(ind, &format!("if ({c}) {{"));
                self.block(then, ind + 1, None)?;
                self.line(ind, "}");
            }
        }
        Ok(())
    }
}

/// An integer expression as `Σ a·v + c` over loop and integer variables,
/// when it is one: sums, and products by an expression that is a constant.
fn affine(e: &Expr) -> Option<(Vec<(String, i64)>, i64)> {
    match e {
        Expr::I(c) => Some((vec![], *c)),
        Expr::Iv(v) => Some((vec![(format!("i{v}"), 1)], 0)),
        Expr::Si(v) => Some((vec![(format!("n{v}"), 1)], 0)),
        Expr::Bin(BinOp::IAdd, a, b) => {
            let (mut va, ca) = affine(a)?;
            let (vb, cb) = affine(b)?;
            for (v, x) in vb {
                match va.iter_mut().find(|(w, _)| *w == v) {
                    Some((_, y)) => *y += x,
                    None => va.push((v, x)),
                }
            }
            Some((va, ca + cb))
        }
        Expr::Bin(BinOp::IMul, a, b) => {
            let ((va, ca), (vb, cb)) = (affine(a)?, affine(b)?);
            match (va.is_empty(), vb.is_empty()) {
                (_, true) => Some((va.into_iter().map(|(v, x)| (v, x * cb)).collect(), ca * cb)),
                (true, false) => Some((vb.into_iter().map(|(v, x)| (v, x * ca)).collect(), ca * cb)),
                (false, false) => None,
            }
        }
        _ => None,
    }
}

/// Whether an expression loads from buffer `b`.
fn reads(e: &Expr, b: usize) -> bool {
    match e {
        Expr::Load(x, ix) | Expr::LoadI(x, ix) | Expr::VLoad(x, ix) => *x == b || ix.iter().any(|i| reads(i, b)),
        Expr::Bin(_, p, q) | Expr::Cmp(_, p, q) | Expr::VBin(_, p, q) => reads(p, b) || reads(q, b),
        Expr::Select(c, p, q) => reads(c, b) || reads(p, b) || reads(q, b),
        Expr::Un(_, a) | Expr::ToF(a) | Expr::ToI(a) | Expr::VSplat(a) => reads(a, b),
        Expr::F(_) | Expr::I(_) | Expr::Iv(_) | Expr::Sv(_) | Expr::Si(_) | Expr::Vv(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buf(dims: &[usize], kind: BufKind) -> BufSpec {
        BufSpec { dims: dims.to_vec(), kind, names: vec![None; dims.len()] }
    }

    fn kernel(dtype: Dtype, inputs: Vec<BufSpec>, output: BufSpec, body: Vec<Stmt>) -> Kernel {
        Kernel { name: "t".into(), dtype, inputs, output, nloops: 4, nreal: 4, nint: 2, nvec: 0, lanes: 1, body, threads: Threads::Default }
    }

    fn bx(e: Expr) -> Box<Expr> {
        Box::new(e)
    }

    fn load(b: usize, ix: Vec<Expr>) -> Expr {
        Expr::Load(b, ix)
    }

    /// The statement `out[i0] = value` in a loop over four elements, printed.
    fn store(dtype: Dtype, value: Expr) -> String {
        let k = kernel(dtype, vec![buf(&[4], BufKind::Real), buf(&[4], BufKind::Real)], buf(&[4], BufKind::Real), vec![Stmt::For { var: 0, bound: 4, body: vec![Stmt::Store { buf: 2, index: vec![Expr::Iv(0)], value }] }]);
        let text = kernel_fn(&k, "k").unwrap();
        text.lines().find(|l| l.trim_start().starts_with("out[")).unwrap().trim().to_string()
    }

    #[test]
    fn parentheses_keep_the_kernels_tree() {
        let (a, b) = (load(0, vec![Expr::Iv(0)]), load(1, vec![Expr::Iv(0)]));
        let sub = |x: Expr, y: Expr| Expr::Bin(BinOp::Sub, bx(x), bx(y));
        let mul = |x: Expr, y: Expr| Expr::Bin(BinOp::Mul, bx(x), bx(y));
        // A left operand at its operator's level needs none, a right one does.
        assert_eq!(store(Dtype::F64, sub(sub(a.clone(), b.clone()), Expr::F(1.0))), "out[i0] = in0[i0] - in1[i0] - 1.0;");
        assert_eq!(store(Dtype::F64, sub(a.clone(), sub(b.clone(), Expr::F(1.0)))), "out[i0] = in0[i0] - (in1[i0] - 1.0);");
        assert_eq!(store(Dtype::F64, mul(sub(a.clone(), b.clone()), a.clone())), "out[i0] = (in0[i0] - in1[i0]) * in0[i0];");
        // No `--`, and a negative constant on the right is parenthesized.
        assert_eq!(store(Dtype::F64, Expr::Un(UnOp::Neg, bx(Expr::Un(UnOp::Neg, bx(a.clone()))))), "out[i0] = -(-in0[i0]);");
        assert_eq!(store(Dtype::F64, mul(a.clone(), Expr::F(-0.5))), "out[i0] = in0[i0] * (-0.5);");
        // `f32` constants are `float`s, as `f32const(x as f32)` is.
        assert_eq!(store(Dtype::F32, mul(a.clone(), Expr::F(0.1))), "out[i0] = in0[i0] * 0.1f;");
        assert_eq!(store(Dtype::F32, Expr::F(f64::NEG_INFINITY)), "out[i0] = -HUGE_VALF;");
        let cmp = Expr::Cmp(CmpOp::Le, bx(a.clone()), bx(b.clone()));
        assert_eq!(store(Dtype::F64, Expr::Select(bx(cmp), bx(Expr::F(0.0)), bx(a.clone()))), "out[i0] = in0[i0] <= in1[i0] ? 0.0 : in0[i0];");
        assert_eq!(store(Dtype::Bf16, Expr::Un(UnOp::Exp, bx(a))), "out[i0] = presk_f32_to_bf16(presk_expf(presk_bf16_to_f32(in0[i0])));");
    }

    #[test]
    fn offsets_are_multiplied_out() {
        // `%0[i0, (i1 * 4) + 3]` over `[2, 8]`, and a gather `%1[%0[i0], i1]`.
        let ix = vec![Expr::Iv(0), Expr::Bin(BinOp::IAdd, bx(Expr::Bin(BinOp::IMul, bx(Expr::Iv(1)), bx(Expr::I(4)))), bx(Expr::I(3)))];
        let k = kernel(
            Dtype::F32,
            vec![buf(&[2, 8], BufKind::Real), buf(&[2], BufKind::Idx), buf(&[5, 8], BufKind::Real)],
            buf(&[2, 8], BufKind::Real),
            vec![Stmt::For {
                var: 0,
                bound: 2,
                body: vec![Stmt::For {
                    var: 1,
                    bound: 2,
                    body: vec![Stmt::Store { buf: 3, index: ix.clone(), value: Expr::Bin(BinOp::Add, bx(load(0, ix)), bx(load(2, vec![Expr::LoadI(1, vec![Expr::Iv(0)]), Expr::Iv(1)]))) }],
                }],
            }],
        );
        let text = kernel_fn(&k, "k").unwrap();
        assert!(text.contains("out[i0 * 8 + i1 * 4 + 3] = in0[i0 * 8 + i1 * 4 + 3] + in2[i1 + in1[i0] * 8];"), "{text}");
    }

    /// Integer weights (step 7): the scales after the output, a load
    /// `f32(q)·s` whose scale index skips the axis the scales are constant
    /// along, widened in a `double` kernel, a nibble for `int4`, and lanes.
    #[test]
    fn integer_weights_load_as_their_values() {
        let at = vec![Expr::Iv(0), Expr::Iv(1)];
        let body = |value: Expr| vec![Stmt::For { var: 0, bound: 3, body: vec![Stmt::For { var: 1, bound: 4, body: vec![Stmt::Store { buf: 2, index: at.clone(), value }] }] }];
        let add = Expr::Bin(BinOp::Add, bx(load(0, at.clone())), bx(load(1, at.clone())));
        let k = kernel(Dtype::F32, vec![buf(&[3, 4], BufKind::Real), buf(&[3, 4], BufKind::Quant { bits: 8, over: 0 })], buf(&[3, 4], BufKind::Real), body(add.clone()));
        let text = kernel_fn(&k, "k").unwrap();
        assert!(text.contains("static void k(const float *restrict in0, const int8_t *restrict in1, float *restrict out, const float *restrict sc1, int64_t lo, int64_t hi)"), "{text}");
        assert!(text.contains("out[i0 * 4 + i1] = in0[i0 * 4 + i1] + (float)in1[i0 * 4 + i1] * sc1[i1];"), "{text}");
        let k = kernel(Dtype::F64, vec![buf(&[3, 4], BufKind::Real), buf(&[3, 4], BufKind::Quant { bits: 4, over: 1 })], buf(&[3, 4], BufKind::Real), body(add));
        let text = kernel_fn(&k, "k").unwrap();
        assert!(text.contains("const uint8_t *restrict in1"), "{text}");
        assert!(text.contains("out[i0 * 4 + i1] = in0[i0 * 4 + i1] + (double)((float)presk_int4(in1, i0 * 4 + i1) * sc1[i0]);"), "{text}");
        let e = kernel_fn(&kernel(Dtype::F32, vec![buf(&[4], BufKind::Quant { bits: 2, over: 0 })], buf(&[4], BufKind::Real), vec![]), "k").unwrap_err();
        assert!(e.contains("int2"), "{e}");
    }
}
