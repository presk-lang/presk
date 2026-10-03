//! The structured host (plan 0.9.0 §3.3, step 4; D80): a [`Recording`]'s
//! tree of calls and folds becomes C functions and loops.
//!
//! - **One C function per distinct recording.** Each call of a Presk
//!   function is put in a canonical form ([`Form`]): its arguments by
//!   position, the buffers it makes by order of first definition, its
//!   kernels by their C text, its callees by their own forms. Calls whose
//!   forms are equal share one function, named after the Presk function,
//!   with its dimensions when it has several (`presk_Linear_apply_In16_Out64`).
//!   A constant argument (`eps`, a `nat`) is folded into the operations that
//!   read it, so two calls that read different constants have different
//!   forms.
//! - **Results by destination.** A function's result is written into
//!   buffers its caller allocates and passes (`out`), so nothing is moved
//!   when it returns: it releases everything it allocated (`presk_mark`,
//!   `presk_release`). A leaf of the result that is a view of an argument is
//!   not an output: the caller has it already.
//! - **A fold is a loop.** Its iterations are cut into runs whose forms are
//!   equal up to offsets affine in the iteration's index (a slice written
//!   back at `i`); a run of two or more is a loop over `l` calling one
//!   function, the stacked values sliced as `w + l * chunk`, the
//!   accumulator's buffers alternating between two (`buf[l % 2]`). A run of
//!   one is a call, so an iteration that differs from the others (the first,
//!   under a mixed realization) runs apart, and a fold whose iterations all
//!   differ is unrolled. A scan's stack is written slot by slot inside the
//!   loop.
//! - **The arena** is sized at its peak: the inputs, the entry's outputs,
//!   and the deepest chain of frames, each function's allocations plus its
//!   callee's frame at each call ([`Structured::arena`]).
//! - **The keys** (step 6, D83; plan 0.10.0 step 7b, D99): a function that
//!   draws (`sample`), or calls one that does, takes the entry's keys after
//!   the arena, `const presk_key *const *keys`, which the entry points at
//!   the keys of its `_in` struct; `sample` reads the key argument it draws
//!   from. The entry's `f64` arguments are `double`s of its `_in` struct.
//! - **Several entries** (step 6): a generation function and its greedy
//!   copy are recorded on one backend and share their functions; each has
//!   its root ([`Structured::root`], [`Structured::others`]).
//! - **Integer weights** (step 7, D84): a leaf of integer weights is one C
//!   value, `presk_q8` or `presk_q4`, its integers and its scales; a view of
//!   it at an element offset is the two pointers moved (`w.q + e`,
//!   `w.s + e / k`, the offset a whole number of channels), a kernel takes
//!   the integers where its input is and the scales after its output, and a
//!   kernel that reads reals gets a `presk_dequantize` copy. A fold's slice
//!   that the host copies to `f64` (`Feed::Dequant`) is dequantized into
//!   one buffer before each call.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use presk_kernel::record::{HyperOp, Made, Node, Op, Recording, Scalar, Store, Sym, View, leaves};

use crate::host::{c_ident, c_real, cdtype, ctype, paths};

type Result<T> = std::result::Result<T, String>;

const ALIGN: usize = 64;

fn aligned(bytes: usize) -> usize {
    bytes.div_ceil(ALIGN) * ALIGN
}

/// An element offset `a + b·i`, `i` the fold index a function takes (`b`
/// is 0 except in the body of a fold whose operations move with its index).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Aff {
    pub a: i64,
    pub b: i64,
}

impl Aff {
    fn at(a: i64) -> Aff {
        Aff { a, b: 0 }
    }

    /// The offset at `i = x`.
    fn eval(self, x: i64) -> i64 {
        self.a + self.b * x
    }
}

/// Where a function finds a pointer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Loc {
    /// Its parameter `k`, a tensor leaf of its arguments.
    Param(usize),
    /// Its output `k`, a buffer its caller allocated for its result.
    Out(usize),
    /// Its buffer `k`, which it allocates.
    Local(usize),
    /// The variable `v` of the accumulator of its fold `f` (the `f`-th fold
    /// statement of its body, in order).
    Acc(usize, usize),
}

/// A view: a pointer at an offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Ref {
    pub loc: Loc,
    pub off: Aff,
}

/// A `double` of the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SRef {
    /// A constant, its bits.
    Const(u64),
    Param(usize),
    /// The function's variable `s<v>`.
    Var(usize),
    Acc(usize, usize),
}

/// A parameter of a function, one per leaf of its arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PKind {
    /// A tensor of this type and this many elements.
    Ptr(Store, usize),
    Double,
    /// The index of the fold iteration the function runs.
    Index,
    /// A constant or a key: not passed, folded into the operations.
    Fixed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Arg {
    Ptr(Ref),
    Double(SRef),
    None,
}

/// A leaf of a function's result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Res {
    Ptr(Ref),
    Double(SRef),
    Fixed,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Stmt {
    /// Local `k` is allocated from the arena.
    Alloc(usize),
    /// Kernel `k<kernel>` over `0..hi`, divided among `workers` (step 5):
    /// the split is part of the form, since two launches of one kernel's
    /// text may be divided differently.
    Launch { kernel: usize, hi: usize, workers: usize, ins: Vec<Ref>, out: Ref, site: String },
    Fill { dst: Ref, store: Store, value: u64, n: usize, site: String },
    Scalar { dst: Ref, store: Store, x: SRef, site: String },
    Copy { dst: Ref, to: Store, src: Ref, from: Store, n: usize, site: String },
    Hyper { out: usize, op: HyperOp, a: SRef, b: SRef },
    Read { out: usize, src: Ref, store: Store },
    /// `double s<out> = x`.
    SetVar { out: usize, x: SRef },
    /// A call of function `f` (an index into [`Structured::funcs`]): an
    /// argument per parameter of `f`, a buffer per output, a variable per
    /// scalar output.
    Call { f: usize, args: Vec<Arg>, outs: Vec<Ref>, souts: Vec<usize>, site: String },
    Fold(Box<FoldStmt>),
    /// A result that a fold left in its own buffers, copied to output `out`.
    CopyOut { out: usize, src: Ref, store: Store, n: usize },
    /// `sample` (step 6): `rows` rows of `vocab` logits of type `from` at
    /// `src`, a token each into `out`, from the entry's key argument `key`
    /// (plan 0.10.0 step 7b).
    Sample { out: Ref, src: Ref, from: Store, rows: usize, vocab: usize, key: usize, site: String },
    /// Everything allocated since the function's start released: only a
    /// mutation of the tests puts one anywhere but at the end.
    Release,
}

/// How a parameter of a fold's body is fed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Feed {
    /// The iteration's index.
    Index,
    /// The slice at the index of a stacked tensor: `base + l · chunk`.
    Slice { base: Ref, chunk: usize },
    /// The slice at the index of stacked integer weights, dequantized into
    /// the local `tmp` of `chunk` reals in `f64` before the call (step 7):
    /// where the host's slice is a copy of their values.
    Dequant { base: Ref, chunk: usize, tmp: usize },
    /// The same value at every iteration: an extra, or a stacked leaf that
    /// is not sliced.
    Whole(Arg),
    /// Leaf `k` of the accumulator.
    Acc(usize),
    Fixed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AccKind {
    Ptr(Store),
    Double,
}

/// A variable of a fold's accumulator: leaf `leaf`'s value over some
/// iterations (a leaf whose type changes takes a new variable).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AccVar {
    pub leaf: usize,
    pub kind: AccKind,
    pub name: String,
    /// The value before the first iteration, for a leaf's first variable.
    pub init: Option<Arg>,
}

/// How a run sets a leaf of the accumulator after each iteration, from the
/// body's result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Upd {
    /// The body's output `n`, at an offset.
    Out(usize, Aff),
    /// What its parameter `m` was fed, at an offset.
    Param(usize, Aff),
    /// Its scalar output `q`.
    Sout(usize),
    SParam(usize),
}

/// Iterations `first`, `first ± 1`, …, `count` of them, of one function.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Run {
    pub f: usize,
    pub first: usize,
    pub count: usize,
    /// A local per output of `f`: two buffers, alternating, when the run
    /// loops (`count ≥ 2`), else one.
    pub sets: Vec<usize>,
    /// A variable per scalar output of `f`.
    pub svars: Vec<usize>,
    /// Per leaf of the accumulator: the variable the run reads.
    pub ins: Vec<Option<usize>>,
    /// Per leaf: the variable it sets after each iteration, and how.
    pub outs: Vec<Option<(usize, Upd)>>,
    /// Per leaf with a scan's stack: the type the stack is copied from.
    pub from: Vec<Option<Store>>,
}

/// A fold, as runs of calls.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FoldStmt {
    pub site: String,
    pub n: usize,
    pub reverse: bool,
    pub feed: Vec<Feed>,
    pub vars: Vec<AccVar>,
    /// Per leaf of the accumulator, with a scan: its stack (a local or an
    /// output of the function, from element 0), a slot's elements, its type.
    pub stacks: Vec<Option<(Ref, usize, Store)>>,
    pub runs: Vec<Run>,
    /// How the fold was emitted, for the README and a comment.
    pub note: String,
    /// The locals its dequantized slices go to, allocated before it (step 7).
    pub temps: Vec<usize>,
}

/// The pointers of a view of integer weights at element offset `off` from
/// their base `base` (a `presk_q8` or `presk_q4`): the integers' (an `int4`
/// is half a byte) and the scales', a whole number of channels on. `var`
/// is the index an affine offset is written in.
pub(crate) fn quant_ptrs(base: &str, store: Store, off: Aff, var: &str) -> (String, String) {
    let Store::Quant { bits, k, .. } = store else { return (plus(base, off, var), String::new()) };
    let per = |d: i64| Aff { a: off.a / d, b: off.b / d };
    let bytes = if bits == 4 { per(2) } else { off };
    (plus(&format!("{base}.q"), bytes, var), plus(&format!("{base}.s"), per(k.max(1) as i64), var))
}

/// A view of integer weights as one C value: the base itself, or a
/// `presk_q8` of its pointers moved.
pub(crate) fn quant_view(base: &str, store: Store, off: Aff, var: &str) -> String {
    if off == Aff::default() {
        return base.to_string();
    }
    let (q, sc) = quant_ptrs(base, store, off, var);
    format!("({}){{{q}, {sc}}}", quant_type(store))
}

/// The C type of a leaf of integer weights.
pub(crate) fn quant_type(store: Store) -> &'static str {
    match store {
        Store::Quant { bits: 8, .. } => "presk_q8",
        _ => "presk_q4",
    }
}

/// Whether a view of integer weights at `off` starts on a channel, and an
/// `int4` one on a byte: what its pointers need (step 7).
fn aligned_quant(store: Store, off: Aff) -> bool {
    let Store::Quant { bits, k, inner } = store else { return true };
    let unit = ((k * inner).max(1) as i64) * if bits == 4 && (k * inner) % 2 == 1 { 2 } else { 1 };
    off.a % unit == 0 && off.b % unit == 0
}

/// A distinct recording of a Presk function: what its C function does.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Form {
    pub func: String,
    pub dims: Vec<(String, u64)>,
    pub params: Vec<PKind>,
    pub param_names: Vec<String>,
    pub outs: Vec<(Store, usize)>,
    pub out_names: Vec<String>,
    pub locals: Vec<(Store, usize)>,
    pub local_names: Vec<String>,
    /// Variables `s0`, `s1`, …
    pub vars: usize,
    /// The variables returned, as `double *` outputs.
    pub souts: Vec<usize>,
    pub sout_names: Vec<String>,
    pub body: Vec<Stmt>,
    pub result: Vec<Res>,
}

/// A C function of the structured host.
#[derive(Clone, Debug)]
pub struct Func {
    pub form: Form,
    /// Its C name.
    pub name: String,
    /// The calls of the recording it runs.
    pub calls: usize,
}

/// The structured host: the functions, callees first, and the entry's own
/// form, which `presk_<entry>` runs; the other entries' forms, when several
/// were recorded on one backend (step 6).
#[derive(Clone, Debug)]
pub struct Structured {
    pub funcs: Vec<Func>,
    pub root: Form,
    pub others: Vec<Form>,
    /// The kernels' C functions, `k0` to `k<kernels − 1>`.
    pub kernels: usize,
}

// ----- building ---------------------------------------------------------------

/// Names made unique within a function, never a kernel's (`k12`), a
/// variable's (`s3`), the runtime's (`presk_…`) or the function's own words.
struct Scope {
    used: HashSet<String>,
}

const WORDS: &[&str] = &["arena", "mark", "l", "i", "memcpy", "t", "w", "keys"];

fn numbered(s: &str, p: char) -> bool {
    s.len() > 1 && s.starts_with(p) && s[1..].bytes().all(|b| b.is_ascii_digit())
}

impl Scope {
    /// The entry's scope also keeps the names of its structs, `in` and `out`.
    fn new(entry: bool) -> Scope {
        let used = if entry { ["in", "out"].iter().map(|s| s.to_string()).collect() } else { HashSet::new() };
        Scope { used }
    }

    fn bad(&self, s: &str) -> bool {
        self.used.contains(s) || WORDS.contains(&s) || s.starts_with("presk_") || numbered(s, 'k') || numbered(s, 's')
    }

    fn fresh(&mut self, base: String) -> String {
        let name = if !self.bad(&base) {
            base
        } else {
            (2..).map(|n| format!("{base}_{n}")).find(|c| !self.bad(c)).unwrap()
        };
        self.used.insert(name.clone());
        name
    }
}

/// A function being built from a call.
struct Frame {
    form: Form,
    /// How the function reaches each buffer of the recording it reads:
    /// `(loc, element of the buffer where loc points, elements it covers)`.
    bufs: HashMap<usize, Vec<(Loc, i64, usize)>>,
    vars: HashMap<usize, SRef>,
    /// Buffers made inside the call that its result reads, in the order
    /// the result reads them first: its outputs. And those made so far.
    kept_set: HashSet<usize>,
    kept: Vec<usize>,
    made: HashSet<usize>,
    skept_set: HashSet<usize>,
    skept: Vec<usize>,
    /// The name of each output, by buffer.
    out_paths: HashMap<usize, String>,
    sout_paths: HashMap<usize, String>,
    folds: usize,
    scope: Scope,
}

impl Frame {
    fn r(&self, v: &View) -> Result<Ref> {
        let (off, n) = (v.off as i64, v.numel() as i64);
        let found = self.bufs.get(&v.buf).and_then(|cs| cs.iter().find(|(_, base, cover)| *base <= off && off + n <= base + *cover as i64));
        match found {
            Some((loc, base, _)) => Ok(Ref { loc: *loc, off: Aff::at(off - base) }),
            None => Err(format!("`{}` reads buffer {} where it was neither made nor given", self.form.func, v.buf)),
        }
    }

    fn sr(&self, x: &Scalar) -> Result<SRef> {
        match x {
            Scalar::Const(c) => Ok(SRef::Const(c.to_bits())),
            Scalar::Var(v) => self.vars.get(v).copied().ok_or_else(|| format!("`{}` reads the host scalar {v} where it was neither made nor given", self.form.func)),
        }
    }

    fn arg(&self, s: &Sym, kind: PKind) -> Result<Arg> {
        Ok(match (kind, s) {
            (PKind::Ptr(store, _), Sym::Tensor(v)) => {
                let r = self.r(v)?;
                if !aligned_quant(store, r.off) {
                    return Err(format!("`{}` passes integer weights from element {}, inside a channel or a byte", self.form.func, r.off.a));
                }
                Arg::Ptr(r)
            }
            (PKind::Double, Sym::F64(x)) => Arg::Double(self.sr(x)?),
            (PKind::Ptr(..) | PKind::Double, other) => return Err(format!("a parameter fed a {other:?}")),
            _ => Arg::None,
        })
    }

    fn own(&self, b: usize) -> Option<Loc> {
        self.bufs.get(&b).and_then(|cs| cs.iter().find(|(l, base, _)| matches!(l, Loc::Local(_) | Loc::Out(_)) && *base == 0)).map(|(l, _, _)| *l)
    }

    fn var(&mut self) -> usize {
        self.form.vars += 1;
        self.form.vars - 1
    }

    /// The recording's scalar `x` is the function's variable `v`: returned,
    /// when the result holds it.
    fn defvar(&mut self, x: usize, v: usize) {
        self.vars.insert(x, SRef::Var(v));
        if self.skept_set.contains(&x) && !self.skept.contains(&x) {
            self.skept.push(x);
            self.form.souts.push(v);
            let base = self.sout_paths.get(&x).cloned().unwrap_or_else(|| "out".into());
            let name = self.scope.fresh(base);
            self.form.sout_names.push(name);
        }
    }

    /// A buffer the function makes: an output when its result reads it,
    /// else a local, allocated here.
    fn define(&mut self, rec: &Recording, b: usize, hint: Option<String>) -> Result<Loc> {
        if self.bufs.contains_key(&b) {
            return Err(format!("`{}` makes buffer {b} twice", self.form.func));
        }
        let rb = &rec.bufs[b];
        let loc = if let Some(n) = self.kept.iter().position(|k| *k == b) {
            self.made.insert(b);
            Loc::Out(n)
        } else {
            let n = self.local(rb.store, rb.len, hint.unwrap_or_else(|| local_base(rec, b)));
            self.form.body.push(Stmt::Alloc(n));
            Loc::Local(n)
        };
        self.bufs.insert(b, vec![(loc, 0, rb.len)]);
        Ok(loc)
    }

    /// A local of the function, not allocated yet.
    fn local(&mut self, store: Store, len: usize, base: String) -> usize {
        let n = self.form.locals.len();
        self.form.locals.push((store, len));
        let name = self.scope.fresh(base);
        self.form.local_names.push(name);
        n
    }

    fn name_of(&self, loc: Loc, vars: &[AccVar]) -> String {
        match loc {
            Loc::Param(k) => self.form.param_names[k].clone(),
            Loc::Out(k) => self.form.out_names[k].clone(),
            Loc::Local(k) => self.form.local_names[k].clone(),
            Loc::Acc(_, v) => vars.get(v).map(|x| x.name.clone()).unwrap_or_else(|| "acc".into()),
        }
    }
}

/// The name of a buffer the function makes: its `let`'s source name, or
/// `t<let>`.
fn local_base(rec: &Recording, b: usize) -> String {
    let rb = &rec.bufs[b];
    match (&rb.name, rb.var) {
        (Some(n), _) => c_ident(n),
        (None, Some(v)) => format!("t{v}"),
        (None, None) => "t".into(),
    }
}

/// The name of what a caller's `let` binds: its source name, or `t<let>`.
fn let_base(var: Option<u32>, name: &Option<String>) -> String {
    match (name, var) {
        (Some(n), _) => c_ident(n),
        (None, Some(v)) => format!("t{v}"),
        (None, None) => "t".into(),
    }
}

/// A form built from one call, and what its caller needs of that call: the
/// recording's buffers and scalars its outputs hold, in order.
struct Built {
    form: Form,
    kept: Vec<usize>,
    skept: Vec<usize>,
}

struct Builder<'a> {
    rec: &'a Recording,
    /// A backend kernel's C function (`k<n>`), its range and its workers.
    kname: &'a HashMap<usize, usize>,
    his: &'a HashMap<usize, usize>,
    workers: &'a HashMap<usize, usize>,
    forms: Vec<Form>,
    ids: HashMap<Form, usize>,
    calls: Vec<usize>,
}

impl Builder<'_> {
    /// The launches and copies a form makes, its callees' included.
    fn weight(&self, f: &Form) -> (usize, usize) {
        let (mut l, mut c) = (0, 0);
        for st in &f.body {
            match st {
                Stmt::Launch { .. } => l += 1,
                Stmt::Copy { .. } => c += 1,
                Stmt::Call { f: g, .. } => {
                    let (a, b) = self.weight(&self.forms[*g]);
                    l += a;
                    c += b;
                }
                Stmt::Fold(fs) => {
                    for r in &fs.runs {
                        let (a, b) = self.weight(&self.forms[r.f]);
                        l += a * r.count;
                        c += b * r.count;
                    }
                }
                _ => {}
            }
        }
        (l, c)
    }

    fn register(&mut self, form: Form, calls: usize) -> usize {
        if let Some(&i) = self.ids.get(&form) {
            self.calls[i] += calls;
            return i;
        }
        let i = self.forms.len();
        self.ids.insert(form.clone(), i);
        self.forms.push(form);
        self.calls.push(calls);
        i
    }

    /// The form of call `c`. `index`: it is an iteration of a fold that
    /// takes the index, its first argument.
    fn build(&mut self, c: usize, index: bool) -> Result<Built> {
        let rec = self.rec;
        let call = &rec.calls[c];
        let mut fr = Frame {
            form: Form {
                func: call.func.clone(),
                dims: call.dims.clone(),
                params: vec![],
                param_names: vec![],
                outs: vec![],
                out_names: vec![],
                locals: vec![],
                local_names: vec![],
                vars: 0,
                souts: vec![],
                sout_names: vec![],
                body: vec![],
                result: vec![],
            },
            bufs: HashMap::new(),
            vars: HashMap::new(),
            kept_set: HashSet::new(),
            kept: vec![],
            made: HashSet::new(),
            skept_set: HashSet::new(),
            skept: vec![],
            out_paths: HashMap::new(),
            sout_paths: HashMap::new(),
            folds: 0,
            scope: Scope::new(c == 0),
        };
        // Parameters: every leaf of the arguments.
        let (mut arg_bufs, mut arg_vars) = (HashSet::new(), HashSet::new());
        for (j, a) in call.args.iter().enumerate() {
            let pname = call.params.get(j).cloned().unwrap_or_else(|| format!("arg{j}"));
            let mut ps = Vec::new();
            paths(a, &pname, &mut ps);
            for (path, leaf) in ps {
                let k = fr.form.params.len();
                let kind = match &leaf {
                    Sym::Tensor(v) => {
                        fr.bufs.entry(v.buf).or_default().push((Loc::Param(k), v.off as i64, v.numel()));
                        arg_bufs.insert(v.buf);
                        PKind::Ptr(rec.bufs[v.buf].store, v.numel())
                    }
                    Sym::F64(Scalar::Var(x)) => {
                        fr.vars.entry(*x).or_insert(SRef::Param(k));
                        arg_vars.insert(*x);
                        PKind::Double
                    }
                    Sym::Nat(_) if index && k == 0 => PKind::Index,
                    _ => PKind::Fixed,
                };
                let name = if kind == PKind::Index { "i".to_string() } else { fr.scope.fresh(c_ident(&path)) };
                fr.form.params.push(kind);
                fr.form.param_names.push(name);
            }
        }
        // What the result reads that the call made.
        let Some(result) = &call.result else { return Err(format!("the call of `{}` did not return", call.func)) };
        let mut ps = Vec::new();
        paths(result, if c == 0 { "result" } else { "out" }, &mut ps);
        for (path, leaf) in &ps {
            match leaf {
                Sym::Tensor(v) if !arg_bufs.contains(&v.buf) => {
                    if fr.kept_set.insert(v.buf) {
                        fr.kept.push(v.buf);
                    }
                    fr.out_paths.entry(v.buf).or_insert_with(|| c_ident(path));
                }
                Sym::F64(Scalar::Var(x)) if !arg_vars.contains(x) => {
                    fr.skept_set.insert(*x);
                    fr.sout_paths.entry(*x).or_insert_with(|| c_ident(path));
                }
                _ => {}
            }
        }
        if fr.kept_set.len() + fr.skept_set.len() == 1 {
            let one = if c == 0 { "result" } else { "out" };
            fr.out_paths.values_mut().for_each(|p| *p = one.into());
            fr.sout_paths.values_mut().for_each(|p| *p = one.into());
        }
        for b in fr.kept.clone() {
            let rb = &rec.bufs[b];
            fr.form.outs.push((rb.store, rb.len));
            let name = fr.scope.fresh(fr.out_paths[&b].clone());
            fr.form.out_names.push(name);
        }
        for node in &call.body {
            match *node {
                Node::Op(i) => self.op(&mut fr, i)?,
                Node::Call(d) => self.call(&mut fr, d)?,
                Node::Fold(f) => self.fold(&mut fr, f)?,
            }
        }
        for (_, leaf) in &ps {
            let r = match leaf {
                Sym::Tensor(v) => Res::Ptr(fr.r(v)?),
                Sym::F64(Scalar::Var(x)) => Res::Double(fr.sr(&Scalar::Var(*x))?),
                _ => Res::Fixed,
            };
            fr.form.result.push(r);
        }
        if fr.made.len() != fr.kept.len() || fr.skept.len() != fr.skept_set.len() {
            return Err(format!("the result of `{}` reads what it did not make", call.func));
        }
        Ok(Built { form: fr.form, kept: fr.kept, skept: fr.skept })
    }

    fn op(&mut self, fr: &mut Frame, i: usize) -> Result<()> {
        let rec = self.rec;
        match &rec.ops[i] {
            Op::Launch { kernel, inputs, out, site } => {
                let ins = inputs.iter().map(|v| fr.r(v)).collect::<Result<Vec<_>>>()?;
                for (v, x) in inputs.iter().zip(&ins) {
                    if !aligned_quant(rec.bufs[v.buf].store, x.off) {
                        return Err(format!("kernel {kernel} reads integer weights from element {} of a view, inside a channel or a byte", x.off.a));
                    }
                }
                let loc = fr.define(rec, *out, None)?;
                let (Some(k), Some(hi), Some(workers)) = (self.kname.get(kernel), self.his.get(kernel), self.workers.get(kernel)) else { return Err(format!("kernel {kernel} is not printed")) };
                fr.form.body.push(Stmt::Launch { kernel: *k, hi: *hi, workers: *workers, ins, out: Ref { loc, off: Aff::default() }, site: site.clone() });
            }
            Op::Fill { buf, value } => {
                let loc = fr.define(rec, *buf, None)?;
                let rb = &rec.bufs[*buf];
                fr.form.body.push(Stmt::Fill { dst: Ref { loc, off: Aff::default() }, store: rb.store, value: value.to_bits(), n: rb.len, site: rb.site.clone() });
            }
            Op::Scalar { buf, x } => {
                let x = fr.sr(x)?;
                let loc = fr.define(rec, *buf, None)?;
                let rb = &rec.bufs[*buf];
                fr.form.body.push(Stmt::Scalar { dst: Ref { loc, off: Aff::default() }, store: rb.store, x, site: rb.site.clone() });
            }
            Op::Copy { dst, at, src, n } => {
                let s = fr.r(src)?;
                let (to, from) = (rec.bufs[*dst].store, rec.bufs[src.buf].store);
                let loc = match fr.own(*dst) {
                    Some(l) => l,
                    None => {
                        // A conversion at a kernel's edge is named after
                        // what it converts.
                        let hint = (*at == 0 && *n == rec.bufs[*dst].len && to != from && matches!(rec.bufs[*dst].made, Made::Copy)).then(|| format!("{}_{}", fr.name_of(s.loc, &[]), to.name()));
                        fr.define(rec, *dst, hint)?
                    }
                };
                fr.form.body.push(Stmt::Copy { dst: Ref { loc, off: Aff::at(*at as i64) }, to, src: s, from, n: *n, site: rec.bufs[*dst].site.clone() });
            }
            Op::Hyper { out, op, a, b } => {
                let (a, b) = (fr.sr(a)?, fr.sr(b)?);
                let v = fr.var();
                fr.form.body.push(Stmt::Hyper { out: v, op: *op, a, b });
                fr.defvar(*out, v);
            }
            Op::Read { out, src } => {
                let s = fr.r(src)?;
                let v = fr.var();
                fr.form.body.push(Stmt::Read { out: v, src: s, store: rec.bufs[src.buf].store });
                fr.defvar(*out, v);
            }
            Op::Sample { out, src, rows, vocab, key } => {
                let s = fr.r(src)?;
                let loc = fr.define(rec, *out, None)?;
                fr.form.body.push(Stmt::Sample { out: Ref { loc, off: Aff::default() }, src: s, from: rec.bufs[src.buf].store, rows: *rows, vocab: *vocab, key: *key, site: rec.bufs[*out].site.clone() });
            }
        }
        Ok(())
    }

    fn call(&mut self, fr: &mut Frame, d: usize) -> Result<()> {
        let rec = self.rec;
        let callee = &rec.calls[d];
        let built = self.build(d, false)?;
        let mut args = Vec::new();
        let mut k = 0;
        for a in &callee.args {
            for leaf in leaves(a) {
                args.push(fr.arg(&leaf, built.form.params[k])?);
                k += 1;
            }
        }
        let base = let_base(callee.var, &callee.name);
        let single = built.kept.len() == 1;
        let mut outs = Vec::new();
        for (n, b) in built.kept.iter().enumerate() {
            let hint = if single { base.clone() } else { format!("{base}_{}", built.form.out_names[n].trim_start_matches("out_")) };
            let loc = fr.define(rec, *b, Some(hint))?;
            outs.push(Ref { loc, off: Aff::default() });
        }
        let souts: Vec<usize> = built.skept.iter().map(|_| fr.var()).collect();
        let site = format!("{}/%{}{}", fr.form.func, callee.var.unwrap_or(0), callee.name.as_ref().map(|n| format!(" ({n})")).unwrap_or_default());
        let f = self.register(built.form, 1);
        fr.form.body.push(Stmt::Call { f, args, outs, souts: souts.clone(), site });
        for (x, v) in built.skept.iter().zip(souts) {
            fr.defvar(*x, v);
        }
        Ok(())
    }

    fn fold(&mut self, fr: &mut Frame, fi: usize) -> Result<()> {
        let rec = self.rec;
        let f = &rec.folds[fi];
        let fs = fr.folds;
        fr.folds += 1;
        let site = match &f.name {
            Some(n) => format!("{} ({n})", f.site),
            None => f.site.clone(),
        };
        // How each parameter of the body is fed.
        let whole = |fr: &Frame, leaf: &Sym| -> Result<Feed> {
            Ok(match leaf {
                Sym::Tensor(v) => Feed::Whole(Arg::Ptr(fr.r(v)?)),
                Sym::F64(x @ Scalar::Var(_)) => Feed::Whole(Arg::Double(fr.sr(x)?)),
                _ => Feed::Fixed,
            })
        };
        let mut feed = Vec::new();
        if f.index {
            feed.push(Feed::Index);
        }
        // The arguments each iteration received, to tell a slice the host
        // copied (integer weights it could not keep, step 7) from a view.
        let iter_args: Vec<Vec<Sym>> = f.iterations.iter().map(|(_, c)| rec.calls[*c].args.iter().flat_map(leaves).collect()).collect();
        let mut dequant = 0;
        for s in &f.stacked {
            for leaf in leaves(s) {
                let m = feed.len();
                feed.push(match &leaf {
                    Sym::Tensor(v) if v.bound.is_none() && !v.shape.is_empty() => {
                        let chunk = v.numel() / v.shape[0].max(1);
                        let base = fr.r(v)?;
                        let copied = |k: usize| matches!(iter_args[k].get(m), Some(Sym::Tensor(a)) if a.buf != v.buf);
                        if (0..iter_args.len()).any(copied) {
                            // Every iteration's slice a copy, made while the
                            // fold fed it: `off + i·chunk` of the stacked
                            // integers, `chunk` reals in `f64`.
                            for (k, (i, _)) in f.iterations.iter().enumerate() {
                                let Some(Sym::Tensor(a)) = iter_args[k].get(m) else { return Err(format!("iteration {i} of `{site}` has no slice {m}")) };
                                let made = f.slices.iter().filter(|(j, _)| j == i).any(|(_, op)| matches!(&rec.ops[*op], Op::Copy { dst, at: 0, src, n } if *dst == a.buf && src.buf == v.buf && src.off == v.off + i * chunk && *n == chunk));
                                if !made || rec.bufs[a.buf].store != Store::F64 || !matches!(rec.bufs[v.buf].store, Store::Quant { .. }) {
                                    return Err(format!("the slice {m} of iteration {i} of `{site}` is neither a view nor the copy of integer weights"));
                                }
                                dequant += 1;
                            }
                            let tmp = fr.local(Store::F64, chunk, format!("{}_f64", fr.name_of(base.loc, &[])));
                            Feed::Dequant { base, chunk, tmp }
                        } else {
                            if !aligned_quant(rec.bufs[v.buf].store, Aff { a: base.off.a, b: chunk as i64 }) {
                                return Err(format!("`{site}` slices integer weights inside a channel or a byte"));
                            }
                            Feed::Slice { base, chunk }
                        }
                    }
                    other => whole(fr, other)?,
                });
            }
        }
        if dequant != f.slices.len() {
            return Err(format!("`{site}` copies {} slices, {dequant} of them of integer weights", f.slices.len()));
        }
        let acc0 = feed.len();
        let init = leaves(&f.init);
        feed.extend((0..init.len()).map(Feed::Acc));
        for e in &f.extra {
            for leaf in leaves(e) {
                feed.push(whole(fr, &leaf)?);
            }
        }
        // The accumulator's variables, named after the body's parameter.
        let accname = f.iterations.first().and_then(|(_, c)| rec.calls[*c].params.get(usize::from(f.index) + f.stacked.len())).cloned().unwrap_or_else(|| "acc".into());
        let mut ps = Vec::new();
        paths(&f.init, &accname, &mut ps);
        let leaf_names: Vec<String> = ps.iter().map(|(p, _)| c_ident(p)).collect();
        let mut vars: Vec<AccVar> = Vec::new();
        let mut cur: Vec<Option<usize>> = vec![None; init.len()];
        for (k, leaf) in init.iter().enumerate() {
            let (kind, a) = match leaf {
                Sym::Tensor(v) if matches!(rec.bufs[v.buf].store, Store::Quant { .. }) => return Err(format!("the accumulator of `{site}` holds integer weights")),
                Sym::Tensor(v) => (AccKind::Ptr(rec.bufs[v.buf].store), Arg::Ptr(fr.r(v)?)),
                Sym::F64(x @ Scalar::Var(_)) => (AccKind::Double, Arg::Double(fr.sr(x)?)),
                _ => continue,
            };
            cur[k] = Some(vars.len());
            let name = fr.scope.fresh(leaf_names[k].clone());
            vars.push(AccVar { leaf: k, kind, name, init: Some(a) });
        }
        // The scan's stack: allocated before the fold, a slot written at the
        // start of each iteration. A scan of no iteration has its stack of
        // none made before it, in the caller's body (D99's addendum).
        let mut stacks: Vec<Option<(Ref, usize, Store)>> = vec![None; init.len()];
        let mut stack_bufs: HashMap<usize, usize> = HashMap::new();
        if f.scan && !f.iterations.is_empty() {
            let Some(Sym::Tuple(parts)) = &f.result else { return Err("a scan without its stack".into()) };
            for (k, leaf) in leaves(&parts[1]).iter().enumerate() {
                if let Sym::Tensor(v) = leaf
                    && v.bound.is_none()
                    && matches!(init.get(k), Some(Sym::Tensor(x)) if x.bound.is_none())
                {
                    let loc = fr.define(rec, v.buf, Some(format!("{}_stack", leaf_names[k])))?;
                    stacks[k] = Some((Ref { loc, off: Aff::default() }, rec.bufs[v.buf].len / f.n.max(1), rec.bufs[v.buf].store));
                    stack_bufs.insert(v.buf, k);
                }
            }
        }
        // Each iteration's form, and the type its stack slot is copied from.
        let mut forms = Vec::new();
        let mut from: Vec<Vec<Option<Store>>> = Vec::new();
        for (_, c) in &f.iterations {
            let b = self.build(*c, f.index)?;
            let args = rec.calls[*c].args.iter().flat_map(leaves).collect::<Vec<_>>();
            from.push(
                (0..init.len())
                    .map(|k| match (&stacks[k], args.get(acc0 + k)) {
                        (Some(_), Some(Sym::Tensor(v))) => Some(rec.bufs[v.buf].store),
                        _ => None,
                    })
                    .collect(),
            );
            forms.push(b.form);
        }
        // The stack's operations are what the loop does: a copy per leaf and
        // per iteration, of the accumulator before it, at its slot.
        let mut seen = 0;
        for &op in &f.stack {
            let Op::Copy { dst, at, src, n } = &rec.ops[op] else { return Err("a scan's stack made otherwise than by copies".into()) };
            let Some(&k) = stack_bufs.get(dst) else { return Err("a scan's copy into another buffer".into()) };
            let (_, chunk, _) = stacks[k].unwrap();
            let slot = at.checked_div(chunk).unwrap_or(0);
            let pre = f.iterations.iter().find(|(i, _)| *i == slot).map(|(_, c)| rec.calls[*c].args.iter().flat_map(leaves).nth(acc0 + k));
            if *n != chunk || at % chunk.max(1) != 0 || pre != Some(Some(Sym::Tensor(src.clone()))) {
                return Err(format!("the stack of `{site}` does not copy the accumulator of iteration {slot}"));
            }
            seen += 1;
        }
        if seen != f.stack.len() || seen != stacks.iter().flatten().count() * f.iterations.len() {
            return Err(format!("the stack of `{site}` is not one copy per slot"));
        }
        // Runs of iterations equal up to offsets affine in the index.
        let idx: Vec<i64> = f.iterations.iter().map(|(i, _)| *i as i64).collect();
        let stripped: Vec<Form> = forms.iter().map(|x| strip(x.clone())).collect();
        let mut runs = Vec::new();
        let mut notes = Vec::new();
        let mut p = 0;
        while p < forms.len() {
            let mut q = p + 1;
            let mut law: Vec<i64> = Vec::new();
            if loopable(&forms[p], &feed, &cur, &vars) {
                while q < forms.len() && stripped[q] == stripped[p] && from[q] == from[p] {
                    let (ap, aq) = (affs(&forms[p]), affs(&forms[q]));
                    let di = idx[q] - idx[p];
                    if q == p + 1 {
                        law = ap.iter().zip(&aq).map(|(x, y)| (y - x) / di).collect();
                        if ap.iter().zip(&aq).any(|(x, y)| (y - x) % di != 0) || (!f.index && law.iter().any(|b| *b != 0)) {
                            break;
                        }
                    }
                    if !ap.iter().zip(&aq).zip(&law).all(|((x, y), b)| *y == x + b * di) {
                        break;
                    }
                    q += 1;
                }
            }
            if q == p + 1 {
                law.clear();
            }
            // The run's function: the first iteration's form, its offsets
            // written in the index.
            let mut t = forms[p].clone();
            let mut j = 0;
            visit_affs(&mut t, &mut |x| {
                let b = law.get(j).copied().unwrap_or(0);
                x.a -= b * idx[p];
                x.b = b;
                j += 1;
            });
            let count = q - p;
            let fid = self.register(t.clone(), count);
            let looped = count >= 2;
            let set_base = |n: usize| -> String {
                let k = t.result.iter().position(|r| matches!(r, Res::Ptr(Ref { loc: Loc::Out(m), .. }) if *m == n));
                let leaf = k.map(|k| leaf_names[k].clone()).unwrap_or_else(|| "acc".into());
                if looped { format!("{leaf}_buf") } else { format!("{leaf}_at{}", idx[p]) }
            };
            let sets: Vec<usize> = t.outs.iter().enumerate().map(|(n, (s, len))| fr.local(*s, *len, set_base(n))).collect();
            let svars: Vec<usize> = t.souts.iter().map(|_| fr.var()).collect();
            let ins = cur.clone();
            let mut outs = vec![None; init.len()];
            for k in 0..init.len() {
                let (kind, upd) = match t.result.get(k) {
                    Some(Res::Ptr(Ref { loc: Loc::Out(n), off })) => (AccKind::Ptr(t.outs[*n].0), Upd::Out(*n, *off)),
                    Some(Res::Ptr(Ref { loc: Loc::Param(m), off })) => {
                        if feed[*m] == Feed::Acc(k) && *off == Aff::default() {
                            continue;
                        }
                        if matches!(feed[*m], Feed::Dequant { .. }) {
                            return Err(format!("an iteration of `{site}` returns the slice it was fed, which the next one overwrites"));
                        }
                        let PKind::Ptr(s, _) = t.params[*m] else { return Err("a tensor leaf of a fold's result from a scalar".into()) };
                        (AccKind::Ptr(s), Upd::Param(*m, *off))
                    }
                    Some(Res::Double(SRef::Var(x))) => match t.souts.iter().position(|v| v == x) {
                        Some(q) => (AccKind::Double, Upd::Sout(q)),
                        None => return Err("a fold's scalar result not returned".into()),
                    },
                    Some(Res::Double(SRef::Param(m))) => {
                        if feed[*m] == Feed::Acc(k) {
                            continue;
                        }
                        (AccKind::Double, Upd::SParam(*m))
                    }
                    Some(Res::Fixed) | None => {
                        cur[k] = None;
                        continue;
                    }
                    Some(other) => return Err(format!("a fold's result leaf {other:?}")),
                };
                // A loop keeps its variables; a single call takes new ones,
                // so that no update reads a variable another has set.
                let v = match cur[k] {
                    Some(v) if looped && vars[v].kind == kind => v,
                    _ => {
                        let name = fr.scope.fresh(leaf_names[k].clone());
                        vars.push(AccVar { leaf: k, kind, name, init: None });
                        vars.len() - 1
                    }
                };
                outs[k] = Some((v, upd));
                cur[k] = Some(v);
            }
            if looped {
                notes.push(format!("a loop over iterations {}..{} of `{}`", idx[p].min(idx[q - 1]), idx[p].max(idx[q - 1]), self.forms[fid].func));
            } else if forms.len() > 1 {
                let other = if p + 1 < forms.len() { p + 1 } else { p.saturating_sub(1) };
                let ((l0, c0), (l1, c1)) = (self.weight(&forms[p]), self.weight(&forms[other]));
                notes.push(format!("iteration {} apart (its recording differs from iteration {}'s: {l0} launches and {c0} copies against {l1} and {c1})", idx[p], idx[other]));
            }
            runs.push(Run { f: fid, first: f.iterations[p].0, count, sets, svars, ins, outs, from: from[p].clone() });
            p = q;
        }
        let note = match (runs.len(), runs.first()) {
            (0, _) => "no iteration".into(),
            (1, Some(r)) if r.count >= 2 => format!("one loop over its {} iterations", r.count),
            (1, _) => "one iteration, a call".into(),
            (n, _) if n == forms.len() => format!("unrolled, since its {n} iterations' recordings all differ"),
            _ => notes.join("; then "),
        };
        // The fold's value: what each leaf's last variable points to.
        let fin = match (&f.result, f.scan) {
            (Some(Sym::Tuple(p)), true) => p[0].clone(),
            (Some(s), false) => s.clone(),
            _ => return Err("a fold without its value".into()),
        };
        let fin_leaves = leaves(&fin);
        let mut inside = Vec::new();
        for (k, leaf) in fin_leaves.iter().enumerate() {
            match leaf {
                Sym::Tensor(v) if !fr.bufs.contains_key(&v.buf) => {
                    let Some(var) = cur[k] else { return Err(format!("the fold at `{site}` returns a buffer it has no variable for")) };
                    fr.bufs.entry(v.buf).or_default().push((Loc::Acc(fs, var), v.off as i64, v.numel()));
                    inside.push((k, v.clone(), var));
                }
                Sym::F64(Scalar::Var(x)) if !fr.vars.contains_key(x) => {
                    let Some(var) = cur[k] else { return Err(format!("the fold at `{site}` returns a scalar it has no variable for")) };
                    fr.vars.insert(*x, SRef::Acc(fs, var));
                }
                _ => {}
            }
        }
        let n = f.n;
        let temps = feed.iter().filter_map(|x| if let Feed::Dequant { tmp, .. } = x { Some(*tmp) } else { None }).collect();
        fr.form.body.push(Stmt::Fold(Box::new(FoldStmt { site: site.clone(), n, reverse: f.reverse, feed, vars, stacks, runs, note, temps })));
        // A buffer of the fold's value that the function returns is copied
        // out of the fold's alternating buffers into its output.
        let mut copied = HashSet::new();
        for (_, v, var) in inside {
            if let Some(n) = fr.kept.iter().position(|k| *k == v.buf)
                && copied.insert(v.buf)
            {
                let rb = &rec.bufs[v.buf];
                fr.made.insert(v.buf);
                fr.form.body.push(Stmt::CopyOut { out: n, src: Ref { loc: Loc::Acc(fs, var), off: Aff::at(-(v.off as i64)) }, store: rb.store, n: rb.len });
                fr.bufs.get_mut(&v.buf).unwrap().insert(0, (Loc::Out(n), 0, rb.len));
            }
        }
        for leaf in &fin_leaves {
            if let Sym::F64(Scalar::Var(x)) = leaf
                && let Some(SRef::Acc(..)) = fr.vars.get(x).copied()
                && fr.skept_set.contains(x)
                && !fr.skept.contains(x)
            {
                let v = fr.var();
                let a = fr.vars[x];
                fr.form.body.push(Stmt::SetVar { out: v, x: a });
                fr.defvar(*x, v);
            }
        }
        Ok(())
    }
}

/// Whether a run starting at a form may loop: each leaf of the accumulator
/// keeps its type, and none is set from another leaf, which alternating
/// buffers could overwrite before it is read.
fn loopable(t: &Form, feed: &[Feed], cur: &[Option<usize>], vars: &[AccVar]) -> bool {
    let kind_in = |k: usize| cur.get(k).copied().flatten().map(|v| vars[v].kind);
    (0..cur.len()).all(|k| match t.result.get(k) {
        Some(Res::Ptr(Ref { loc: Loc::Out(n), .. })) => kind_in(k) == Some(AccKind::Ptr(t.outs[*n].0)),
        Some(Res::Ptr(Ref { loc: Loc::Param(m), .. })) => match feed.get(*m) {
            Some(Feed::Acc(j)) => *j == k,
            _ => matches!(t.params.get(*m), Some(PKind::Ptr(s, _)) if kind_in(k) == Some(AccKind::Ptr(*s))),
        },
        Some(Res::Double(SRef::Param(m))) => matches!(feed.get(*m), Some(Feed::Acc(j)) if *j == k) || (!matches!(feed.get(*m), Some(Feed::Acc(_))) && kind_in(k) == Some(AccKind::Double)),
        Some(Res::Double(_)) => kind_in(k) == Some(AccKind::Double),
        Some(Res::Fixed) | None => kind_in(k).is_none(),
        Some(Res::Ptr(_)) => false,
    })
}


/// Every offset of a form, in one order.
fn visit_affs(f: &mut Form, g: &mut impl FnMut(&mut Aff)) {
    fn r(x: &mut Ref, g: &mut impl FnMut(&mut Aff)) {
        g(&mut x.off);
    }
    fn arg(a: &mut Arg, g: &mut impl FnMut(&mut Aff)) {
        if let Arg::Ptr(x) = a {
            r(x, g);
        }
    }
    for s in &mut f.body {
        match s {
            Stmt::Launch { ins, out, .. } => {
                ins.iter_mut().for_each(|x| r(x, g));
                r(out, g);
            }
            Stmt::Fill { dst, .. } | Stmt::Scalar { dst, .. } => r(dst, g),
            Stmt::Copy { dst, src, .. } => {
                r(dst, g);
                r(src, g);
            }
            Stmt::Read { src, .. } | Stmt::CopyOut { src, .. } => r(src, g),
            Stmt::Sample { out, src, .. } => {
                r(out, g);
                r(src, g);
            }
            Stmt::Call { args, outs, .. } => {
                args.iter_mut().for_each(|a| arg(a, g));
                outs.iter_mut().for_each(|x| r(x, g));
            }
            Stmt::Fold(fs) => {
                for fd in &mut fs.feed {
                    match fd {
                        Feed::Slice { base, .. } | Feed::Dequant { base, .. } => r(base, g),
                        Feed::Whole(a) => arg(a, g),
                        _ => {}
                    }
                }
                for v in &mut fs.vars {
                    if let Some(a) = &mut v.init {
                        arg(a, g);
                    }
                }
                for (x, _, _) in fs.stacks.iter_mut().flatten() {
                    r(x, g);
                }
                for run in &mut fs.runs {
                    for (_, u) in run.outs.iter_mut().flatten() {
                        if let Upd::Out(_, o) | Upd::Param(_, o) = u {
                            g(o);
                        }
                    }
                }
            }
            Stmt::Alloc(_) | Stmt::Hyper { .. } | Stmt::SetVar { .. } | Stmt::Release => {}
        }
    }
    for x in &mut f.result {
        if let Res::Ptr(x) = x {
            r(x, g);
        }
    }
}

fn affs(f: &Form) -> Vec<i64> {
    let mut f = f.clone();
    let mut out = Vec::new();
    visit_affs(&mut f, &mut |x| out.push(x.a));
    out
}

fn strip(mut f: Form) -> Form {
    visit_affs(&mut f, &mut |x| *x = Aff::default());
    f
}

/// The structured host of `recs`, the entries recorded on one backend, the
/// first the one `presk_<entry>` runs. `kname` maps each backend kernel the
/// recordings launch to its C function's number, `his` to its range,
/// `workers` to the workers its launches are divided among. The entries
/// share the functions whose forms are equal.
pub fn build(recs: &[&Recording], kname: &HashMap<usize, usize>, his: &HashMap<usize, usize>, workers: &HashMap<usize, usize>, kernels: usize) -> Result<Structured> {
    for rec in recs {
        if rec.calls.is_empty() {
            return Err("the recording has no structure".into());
        }
        // Every operation once in the tree.
        let mut seen = vec![0u32; rec.ops.len()];
        for c in &rec.calls {
            for n in &c.body {
                if let Node::Op(i) = n {
                    seen[*i] += 1;
                }
            }
        }
        for f in &rec.folds {
            for i in f.stack.iter().chain(f.slices.iter().map(|(_, op)| op)) {
                seen[*i] += 1;
            }
        }
        if let Some(i) = seen.iter().position(|n| *n != 1) {
            return Err(format!("operation {i} is {} times in the structure", seen[i]));
        }
    }
    let mut b = Builder { rec: recs[0], kname, his, workers, forms: vec![], ids: HashMap::new(), calls: vec![] };
    let root = b.build(0, false)?;
    let mut others = Vec::new();
    for rec in &recs[1..] {
        b.rec = rec;
        others.push(b.build(0, false)?.form);
    }
    // C names: the Presk function's, with its dimensions when it has
    // several forms at different ones, numbered when that is not enough.
    let base: Vec<String> = b.forms.iter().map(|f| format!("presk_{}", c_ident(&f.func))).collect();
    let stem: Vec<String> = b
        .forms
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let several_dims = b.forms.iter().enumerate().any(|(j, g)| base[j] == base[i] && g.dims != f.dims);
            if several_dims && !f.dims.is_empty() { format!("{}_{}", base[i], c_ident(&f.dims.iter().map(|(n, v)| format!("{n}{v}")).collect::<Vec<_>>().join("_"))) } else { base[i].clone() }
        })
        .collect();
    // Of several recordings at the same dimensions, the one called most
    // takes the plain name, the others a number.
    let mut order: Vec<usize> = (0..b.forms.len()).collect();
    order.sort_by_key(|i| std::cmp::Reverse(b.calls[*i]));
    let mut names = vec![String::new(); b.forms.len()];
    let mut taken: HashSet<String> = HashSet::new();
    for i in order {
        let mut name = stem[i].clone();
        let mut n = 2;
        while taken.contains(&name) {
            name = format!("{}_{n}", stem[i]);
            n += 1;
        }
        taken.insert(name.clone());
        names[i] = name;
    }
    let funcs = b.forms.into_iter().zip(names).zip(b.calls).map(|((form, name), calls)| Func { form, name, calls }).collect();
    Ok(Structured { funcs, root: root.form, others, kernels })
}

// ----- what a function uses, and its frame ------------------------------------------

/// What a form's body reads: its parameters, its index, the arena.
struct Uses {
    params: Vec<bool>,
    index: bool,
    arena: bool,
    /// It draws from a key, or calls what does.
    keys: bool,
    /// Variables read.
    vars: HashSet<usize>,
    /// Accumulator variables read, `(fold, var)`.
    accs: HashSet<(usize, usize)>,
}

/// Of every function, callees first: the parameters it takes (the index
/// counted as one), and whether it takes the arena.
struct Analysis {
    params: Vec<Vec<bool>>,
    arena: Vec<bool>,
    keys: Vec<bool>,
}

fn uses(f: &Form, an: &Analysis) -> Uses {
    let mut u = Uses { params: vec![false; f.params.len()], index: false, arena: false, keys: false, vars: HashSet::new(), accs: HashSet::new() };
    fn r(u: &mut Uses, x: &Ref) {
        match x.loc {
            Loc::Param(k) => u.params[k] = true,
            Loc::Acc(f, v) => {
                u.accs.insert((f, v));
            }
            _ => {}
        }
        u.index |= x.off.b != 0;
    }
    fn s(u: &mut Uses, x: &SRef) {
        match x {
            SRef::Param(k) => u.params[*k] = true,
            SRef::Var(v) => {
                u.vars.insert(*v);
            }
            SRef::Acc(f, v) => {
                u.accs.insert((*f, *v));
            }
            SRef::Const(_) => {}
        }
    }
    fn arg(u: &mut Uses, a: &Arg) {
        match a {
            Arg::Ptr(x) => r(u, x),
            Arg::Double(x) => s(u, x),
            Arg::None => {}
        }
    }
    let mut fold = 0;
    for st in &f.body {
        match st {
            Stmt::Alloc(_) | Stmt::Release => u.arena = true,
            Stmt::Launch { ins, out, .. } => {
                ins.iter().for_each(|x| r(&mut u, x));
                r(&mut u, out);
            }
            Stmt::Fill { dst, .. } => r(&mut u, dst),
            Stmt::Scalar { dst, x, .. } => {
                r(&mut u, dst);
                s(&mut u, x);
            }
            Stmt::Copy { dst, src, .. } => {
                r(&mut u, dst);
                r(&mut u, src);
            }
            Stmt::Hyper { a, b, .. } => {
                s(&mut u, a);
                s(&mut u, b);
            }
            Stmt::Read { src, .. } | Stmt::CopyOut { src, .. } => r(&mut u, src),
            Stmt::Sample { out, src, .. } => {
                r(&mut u, out);
                r(&mut u, src);
                u.keys = true;
            }
            Stmt::SetVar { x, .. } => s(&mut u, x),
            Stmt::Call { f: g, args, outs, .. } => {
                u.arena |= an.arena[*g];
                u.keys |= an.keys[*g];
                for (k, a) in args.iter().enumerate() {
                    if an.params[*g][k] {
                        arg(&mut u, a);
                    }
                }
                outs.iter().for_each(|x| r(&mut u, x));
            }
            Stmt::Fold(fs) => {
                u.arena = true;
                let feed = |u: &mut Uses, m: usize, ins: &[Option<usize>]| match &fs.feed[m] {
                    Feed::Slice { base, .. } | Feed::Dequant { base, .. } => r(u, base),
                    Feed::Whole(a) => arg(u, a),
                    Feed::Acc(k) => {
                        if let Some(v) = ins[*k] {
                            u.accs.insert((fold, v));
                        }
                    }
                    Feed::Index | Feed::Fixed => {}
                };
                for run in &fs.runs {
                    u.keys |= an.keys[run.f];
                    for m in 0..fs.feed.len() {
                        if an.params[run.f].get(m).copied().unwrap_or(false) {
                            feed(&mut u, m, &run.ins);
                        }
                    }
                    for (k, st) in fs.stacks.iter().enumerate() {
                        if let (Some((x, ..)), Some(v)) = (st, run.ins[k]) {
                            r(&mut u, x);
                            u.accs.insert((fold, v));
                        }
                    }
                    for (_, upd) in run.outs.iter().flatten() {
                        if let Upd::Param(m, _) | Upd::SParam(m) = upd {
                            feed(&mut u, *m, &run.ins);
                        }
                    }
                }
                for v in &fs.vars {
                    if let Some(a) = &v.init {
                        arg(&mut u, a);
                    }
                }
                fold += 1;
            }
        }
    }
    for v in &f.souts {
        u.vars.insert(*v);
    }
    u
}

impl Structured {
    fn analysis(&self) -> Analysis {
        let mut an = Analysis { params: Vec::new(), arena: Vec::new(), keys: Vec::new() };
        for func in &self.funcs {
            let u = uses(&func.form, &an);
            an.params.push(u.params.iter().zip(&func.form.params).map(|(used, k)| *used || (*k == PKind::Index && u.index)).collect());
            an.arena.push(u.arena);
            an.keys.push(u.keys);
        }
        an
    }

    /// The bytes a function's frame reaches: what it allocates, and at each
    /// call its callee's frame on top.
    fn frame(&self, f: &Form, memo: &mut HashMap<usize, usize>) -> usize {
        let (mut top, mut peak) = (0usize, 0usize);
        for st in &f.body {
            match st {
                Stmt::Alloc(k) => {
                    let (s, n) = f.locals[*k];
                    top += aligned(n * s.bytes());
                }
                Stmt::Release => top = 0,
                Stmt::Call { f: g, .. } => peak = peak.max(top + self.frame_of(*g, memo)),
                Stmt::Fold(fs) => {
                    for k in &fs.temps {
                        let (s, n) = f.locals[*k];
                        top += aligned(n * s.bytes());
                    }
                    for run in &fs.runs {
                        let copies = if run.count >= 2 { 2 } else { 1 };
                        for k in &run.sets {
                            let (s, n) = f.locals[*k];
                            top += copies * aligned(n * s.bytes());
                        }
                        peak = peak.max(top + self.frame_of(run.f, memo));
                    }
                }
                _ => {}
            }
            peak = peak.max(top);
        }
        peak
    }

    fn frame_of(&self, g: usize, memo: &mut HashMap<usize, usize>) -> usize {
        if let Some(b) = memo.get(&g) {
            return *b;
        }
        let b = self.frame(&self.funcs[g].form, memo);
        memo.insert(g, b);
        b
    }

    /// The arena the entry needs beside its inputs: its outputs, and its
    /// frame's peak; with several entries, the most any needs.
    pub fn arena(&self) -> usize {
        let mut memo = HashMap::new();
        std::iter::once(&self.root)
            .chain(&self.others)
            .map(|root| {
                let outs: usize = root.outs.iter().map(|(s, n)| aligned(n * s.bytes())).sum();
                outs + self.frame(root, &mut memo)
            })
            .max()
            .unwrap_or(0)
    }

    /// Folds as loops, peeled or unrolled: `(loops, notes)`.
    pub fn folds(&self) -> (usize, Vec<String>) {
        let mut loops = 0;
        let mut notes = Vec::new();
        for form in std::iter::once(&self.root).chain(&self.others).chain(self.funcs.iter().map(|f| &f.form)) {
            for st in &form.body {
                if let Stmt::Fold(fs) = st {
                    loops += fs.runs.iter().filter(|r| r.count >= 2).count();
                    notes.push(format!("`{}`: {}", fs.site, fs.note));
                }
            }
        }
        (loops, notes)
    }
}

// ----- printing ----------------------------------------------------------------------

/// How an entry's function reaches its inputs and outputs: the field of
/// `in` of each tensor parameter of the root, in order, of each `double`
/// parameter (an `f64` argument, step 6), and of `out` of each tensor leaf
/// of its result. `types` names its structs (`<types>_in`), which several
/// entries share; `in_keys`, by the recording's key arguments, the field of
/// `in` of each that `sample` draws from (plan 0.10.0 step 7b).
pub struct EntryIo<'a> {
    pub func: &'a str,
    pub types: &'a str,
    pub in_fields: &'a [String],
    pub in_doubles: &'a [String],
    pub out_fields: &'a [String],
    pub in_keys: &'a [Option<String>],
}

/// An offset in C: `a`, `i * b`, `a + i * b`.
fn aff_c(x: Aff, var: &str) -> String {
    match (x.a, x.b) {
        (a, 0) => a.to_string(),
        (0, b) => format!("{var} * {b}"),
        (a, b) if a < 0 => format!("{var} * {b} - {}", -a),
        (a, b) => format!("{a} + {var} * {b}"),
    }
}

fn plus(base: &str, x: Aff, var: &str) -> String {
    match (x.a, x.b) {
        (0, 0) => base.to_string(),
        (a, 0) if a < 0 => format!("{base} - {}", -a),
        (a, 0) => format!("{base} + {a}"),
        (0, _) => format!("{base} + {}", aff_c(x, var)),
        (a, b) if a < 0 => format!("{base} + {var} * {b} - {}", -a),
        _ => format!("{base} + {}", aff_c(x, var)),
    }
}

struct Printer<'a> {
    s: &'a Structured,
    form: &'a Form,
    folds: Vec<&'a FoldStmt>,
    /// The C expression of each parameter (its name; `in->…` for the entry).
    params: Vec<String>,
    /// Of each output (its name; the entry's allocated outputs too).
    outs: Vec<String>,
    an: &'a Analysis,
    uses: Uses,
    out: String,
    site: String,
}

impl Printer<'_> {
    fn loc(&self, loc: Loc) -> String {
        match loc {
            Loc::Param(k) => self.params[k].clone(),
            Loc::Out(k) => self.outs[k].clone(),
            Loc::Local(k) => self.form.local_names[k].clone(),
            Loc::Acc(f, v) => self.folds[f].vars[v].name.clone(),
        }
    }

    /// The type of what `loc` points to, when it is a tensor.
    fn store_of(&self, loc: Loc) -> Option<Store> {
        match loc {
            Loc::Param(k) => match self.form.params[k] {
                PKind::Ptr(s, _) => Some(s),
                _ => None,
            },
            Loc::Out(k) => Some(self.form.outs[k].0),
            Loc::Local(k) => Some(self.form.locals[k].0),
            Loc::Acc(f, v) => match self.folds[f].vars[v].kind {
                AccKind::Ptr(s) => Some(s),
                AccKind::Double => None,
            },
        }
    }

    /// Whether `loc` holds integer weights, and of which geometry.
    fn quant(&self, loc: Loc) -> Option<Store> {
        self.store_of(loc).filter(|s| matches!(s, Store::Quant { .. }))
    }

    /// A view as one C value: a pointer at an offset, or integer weights'
    /// struct of the two pointers moved.
    fn r(&self, x: &Ref) -> String {
        match self.quant(x.loc) {
            Some(s) => quant_view(&self.loc(x.loc), s, x.off, "i"),
            None => plus(&self.loc(x.loc), x.off, "i"),
        }
    }

    /// The element a view starts at, `t5[0]`.
    fn elem(&self, x: &Ref) -> String {
        format!("{}[{}]", self.loc(x.loc), aff_c(x.off, "i"))
    }

    fn sr(&self, x: &SRef) -> String {
        match x {
            SRef::Const(c) => c_real(f64::from_bits(*c)),
            SRef::Param(k) => self.params[*k].clone(),
            SRef::Var(v) => format!("s{v}"),
            SRef::Acc(f, v) => self.folds[*f].vars[*v].name.clone(),
        }
    }

    fn arg(&self, a: &Arg) -> String {
        match a {
            Arg::Ptr(x) => self.r(x),
            Arg::Double(x) => self.sr(x),
            Arg::None => String::new(),
        }
    }

    fn comment(&mut self, site: &str, ind: &str) {
        if site != self.site {
            let _ = writeln!(self.out, "{ind}/* {} */", site.replace("*/", "* /"));
            self.site = site.to_string();
        }
    }

    fn body(&mut self) {
        let mut fold = 0;
        for (j, st) in self.form.body.iter().enumerate() {
            // An allocation comes under the comment of what it is for.
            if matches!(st, Stmt::Alloc(_))
                && let Some(site) = self.form.body[j..].iter().find_map(site_of)
            {
                self.comment(site, "    ");
            }
            self.stmt(st, &mut fold);
        }
    }

    fn stmt(&mut self, st: &Stmt, fold: &mut usize) {
        let ind = "    ";
        match st {
            Stmt::Alloc(k) => {
                let (s, n) = self.form.locals[*k];
                let _ = writeln!(self.out, "{ind}{} *{} = presk_alloc(arena, {n} * sizeof({}));", ctype(s), self.form.local_names[*k], ctype(s));
            }
            Stmt::Launch { kernel, hi, workers, ins, out, site } => {
                self.comment(site, ind);
                // Integer weights: the integers where the input is, the
                // scales after the output.
                let (mut a, mut scales) = (Vec::new(), Vec::new());
                for x in ins {
                    match self.quant(x.loc) {
                        Some(s) => {
                            let (q, sc) = quant_ptrs(&self.loc(x.loc), s, x.off, "i");
                            a.push(q);
                            scales.push(sc);
                        }
                        None => a.push(self.r(x)),
                    }
                }
                a.push(self.r(out));
                a.extend(scales);
                self.out.push_str(&crate::launch(ind, &format!("k{kernel}"), &a, *hi, *workers));
            }
            Stmt::Fill { dst, store, value, n, site } => {
                self.comment(site, ind);
                let _ = writeln!(self.out, "{ind}presk_fill({}, {}, {}, {n});", self.r(dst), cdtype(*store), c_real(f64::from_bits(*value)));
            }
            Stmt::Scalar { dst, store, x, site } => {
                self.comment(site, ind);
                let _ = writeln!(self.out, "{ind}presk_fill({}, {}, {}, 1);", self.r(dst), cdtype(*store), self.sr(x));
            }
            Stmt::Copy { dst, to, src, from, n, site } => {
                self.comment(site, ind);
                match from {
                    Store::Quant { bits, k, inner } => {
                        let base = self.loc(src.loc);
                        let _ = writeln!(self.out, "{ind}presk_dequantize({}, {}, {base}.q, {base}.s, {bits}, {k}, {inner}, {}, {n});", self.r(dst), cdtype(*to), aff_c(src.off, "i"));
                    }
                    _ => {
                        let _ = writeln!(self.out, "{ind}presk_convert({}, {}, {}, {}, {n});", self.r(dst), cdtype(*to), self.r(src), cdtype(*from));
                    }
                }
            }
            Stmt::Hyper { out, op, a, b } => {
                let _ = writeln!(self.out, "{ind}double s{out} = {} {} {};", self.sr(a), op.c(), self.sr(b));
                self.unread(*out, ind);
            }
            Stmt::Read { out, src, store } => {
                let e = self.elem(src);
                let e = match store {
                    Store::F64 => e,
                    Store::F32 | Store::I64 => format!("(double){e}"),
                    Store::Bf16 => format!("(double)presk_bf16_to_f32({e})"),
                    Store::Quant { .. } => unreachable!("a host scalar is read from a kernel's output, never from integer weights"),
                };
                let _ = writeln!(self.out, "{ind}double s{out} = {e};");
                self.unread(*out, ind);
            }
            Stmt::SetVar { out, x } => {
                let _ = writeln!(self.out, "{ind}double s{out} = {};", self.sr(x));
                self.unread(*out, ind);
            }
            Stmt::Call { f, args, outs, souts, site } => {
                self.comment(site, ind);
                if !souts.is_empty() {
                    let _ = writeln!(self.out, "{ind}double {};", souts.iter().map(|v| format!("s{v}")).collect::<Vec<_>>().join(", "));
                }
                let used = &self.an.params[*f];
                let mut a = Vec::new();
                if self.an.arena[*f] {
                    a.push("arena".to_string());
                }
                if self.an.keys[*f] {
                    a.push("keys".to_string());
                }
                for (k, x) in args.iter().enumerate() {
                    if used[k] {
                        a.push(self.arg(x));
                    }
                }
                a.extend(outs.iter().map(|x| self.r(x)));
                a.extend(souts.iter().map(|v| format!("&s{v}")));
                let _ = writeln!(self.out, "{ind}{}({});", self.s.funcs[*f].name, a.join(", "));
            }
            Stmt::Fold(fs) => {
                self.fold(fs, *fold);
                *fold += 1;
            }
            Stmt::CopyOut { out, src, store, n } => {
                let _ = writeln!(self.out, "{ind}memcpy({}, {}, {n} * sizeof({}));", self.outs[*out], self.r(src), ctype(*store));
            }
            Stmt::Sample { out, src, from, rows, vocab, key, site } => {
                self.comment(site, ind);
                let _ = writeln!(self.out, "{ind}presk_sample({}, {}, {}, {rows}, {vocab}, keys[{key}]);", self.r(out), self.r(src), cdtype(*from));
            }
            Stmt::Release => {
                let _ = writeln!(self.out, "{ind}presk_release(arena, mark);");
            }
        }
    }

    fn unread(&mut self, v: usize, ind: &str) {
        if !self.uses.vars.contains(&v) {
            let _ = writeln!(self.out, "{ind}(void)s{v};");
        }
    }

    fn acc_type(&self, v: &AccVar) -> String {
        match v.kind {
            AccKind::Ptr(s) => format!("const {} *", ctype(s)),
            AccKind::Double => "double ".into(),
        }
    }

    /// A fold: its variables, then each run, a loop or a call.
    fn fold(&mut self, fs: &FoldStmt, fi: usize) {
        let ind = "    ";
        self.comment(&fs.site, ind);
        let mut o = String::new();
        let _ = writeln!(o, "{ind}/* a fold of {} iterations{}: {} */", fs.n, if fs.reverse { ", from the last" } else { "" }, fs.note);
        for k in &fs.temps {
            let (s, len) = self.form.locals[*k];
            let _ = writeln!(o, "{ind}{} *{} = presk_alloc(arena, {len} * sizeof({})); /* a slice of integer weights, dequantized for each call */", ctype(s), self.form.local_names[*k], ctype(s));
        }
        for v in &fs.vars {
            if let Some(a) = &v.init {
                let _ = writeln!(o, "{ind}{}{} = {};", self.acc_type(v), v.name, self.arg(a));
            }
        }
        for run in &fs.runs {
            let callee = &self.s.funcs[run.f];
            let used = &self.an.params[run.f];
            let looped = run.count >= 2;
            let (first, count) = (run.first as i64, run.count as i64);
            let (lo, hi) = if fs.reverse { (first + 1 - count, first) } else { (first, first + count - 1) };
            // The buffers the run's calls write.
            for k in &run.sets {
                let (s, len) = self.form.locals[*k];
                let name = &self.form.local_names[*k];
                let alloc = format!("presk_alloc(arena, {len} * sizeof({}))", ctype(s));
                if looped {
                    let _ = writeln!(o, "{ind}{} *{name}[2] = {{{alloc}, {alloc}}};", ctype(s));
                } else {
                    let _ = writeln!(o, "{ind}{} *{name} = {alloc};", ctype(s));
                }
            }
            let (ind2, l) = if looped {
                if fs.reverse {
                    let _ = writeln!(o, "{ind}for (int64_t l = {hi}; l >= {lo}; l--) {{");
                } else {
                    let _ = writeln!(o, "{ind}for (int64_t l = {lo}; l < {}; l++) {{", hi + 1);
                }
                ("        ", None)
            } else {
                let _ = writeln!(o, "{ind}/* iteration {} */", run.first);
                (ind, Some(run.first as i64))
            };
            // The expression of `l · chunk` and of the index: symbolic in a
            // loop, a literal for a single call.
            let set = |k: usize| -> String {
                let name = self.form.local_names[run.sets[k]].clone();
                if looped { format!("{name}[l % 2]") } else { name }
            };
            let feed = |m: usize| -> String {
                match &fs.feed[m] {
                    Feed::Index => match l {
                        Some(x) => x.to_string(),
                        None => "l".into(),
                    },
                    Feed::Slice { base, chunk } => match (l, self.quant(base.loc)) {
                        (Some(x), Some(s)) => quant_view(&self.loc(base.loc), s, Aff { a: base.off.a + *chunk as i64 * x, b: base.off.b }, "i"),
                        (Some(x), None) => plus(&self.loc(base.loc), Aff { a: base.off.a + *chunk as i64 * x, b: base.off.b }, "i"),
                        (None, Some(s @ Store::Quant { bits, k, .. })) => {
                            let (q, sc) = quant_ptrs(&self.loc(base.loc), s, base.off, "i");
                            format!("({}){{{q} + l * {}, {sc} + l * {}}}", quant_type(s), if bits == 4 { chunk / 2 } else { *chunk }, chunk / k.max(1))
                        }
                        (None, _) => format!("{} + l * {chunk}", self.r(base)),
                    },
                    Feed::Dequant { tmp, .. } => self.form.local_names[*tmp].clone(),
                    Feed::Whole(a) => self.arg(a),
                    Feed::Acc(k) => run.ins[*k].map(|v| fs.vars[v].name.clone()).unwrap_or_default(),
                    Feed::Fixed => String::new(),
                }
            };
            // The scan's slot: the accumulator before the iteration.
            for (k, st) in fs.stacks.iter().enumerate() {
                if let (Some((x, chunk, to)), Some(v), Some(from)) = (st, run.ins[k], run.from[k]) {
                    let dst = match l {
                        Some(i) => plus(&self.loc(x.loc), Aff { a: x.off.a + *chunk as i64 * i, b: x.off.b }, "i"),
                        None => format!("{} + l * {chunk}", self.r(x)),
                    };
                    let _ = writeln!(o, "{ind2}presk_convert({dst}, {}, {}, {}, {chunk});", cdtype(*to), fs.vars[v].name, cdtype(from));
                }
            }
            // A slice of integer weights the host copies: dequantized into
            // its buffer for this call.
            for (m, fd) in fs.feed.iter().enumerate() {
                let Feed::Dequant { base, chunk, tmp } = fd else { continue };
                if let (true, Some(Store::Quant { bits, k, inner })) = (used.get(m).copied().unwrap_or(false), self.quant(base.loc)) {
                    let at = match l {
                        Some(x) => aff_c(Aff { a: base.off.a + *chunk as i64 * x, b: base.off.b }, "i"),
                        None if base.off == Aff::default() => format!("l * {chunk}"),
                        None => format!("{} + l * {chunk}", aff_c(base.off, "i")),
                    };
                    let b = self.loc(base.loc);
                    let _ = writeln!(o, "{ind2}presk_dequantize({}, PRESK_F64, {b}.q, {b}.s, {bits}, {k}, {inner}, {at}, {chunk});", self.form.local_names[*tmp]);
                }
            }
            if !run.svars.is_empty() {
                let _ = writeln!(o, "{ind2}double {};", run.svars.iter().map(|v| format!("s{v}")).collect::<Vec<_>>().join(", "));
            }
            let mut a = Vec::new();
            if self.an.arena[run.f] {
                a.push("arena".to_string());
            }
            if self.an.keys[run.f] {
                a.push("keys".to_string());
            }
            for m in 0..fs.feed.len() {
                if used.get(m).copied().unwrap_or(false) {
                    a.push(feed(m));
                }
            }
            a.extend((0..run.sets.len()).map(set));
            a.extend(run.svars.iter().map(|v| format!("&s{v}")));
            let _ = writeln!(o, "{ind2}{}({});", callee.name, a.join(", "));
            // The accumulator after the iteration.
            for (v, upd) in run.outs.iter().flatten() {
                let var = &fs.vars[*v];
                let e = match upd {
                    Upd::Out(n, off) => match l {
                        Some(x) => plus(&set(*n), Aff::at(off.eval(x)), "i"),
                        None => plus(&set(*n), *off, "l"),
                    },
                    Upd::Param(m, off) => match l {
                        Some(x) => plus(&feed(*m), Aff::at(off.eval(x)), "i"),
                        None => plus(&feed(*m), *off, "l"),
                    },
                    Upd::Sout(q) => format!("s{}", run.svars[*q]),
                    Upd::SParam(m) => feed(*m),
                };
                let declared = run.ins.get(var.leaf).copied().flatten() == Some(*v);
                if declared {
                    let _ = writeln!(o, "{ind2}{} = {e};", var.name);
                } else {
                    let _ = writeln!(o, "{ind2}{}{} = {e};", self.acc_type(var), var.name);
                }
            }
            if looped {
                let _ = writeln!(o, "{ind}}}");
            }
        }
        // A variable nothing reads.
        for (v, var) in fs.vars.iter().enumerate() {
            if !self.uses.accs.contains(&(fi, v)) {
                let _ = writeln!(o, "{ind}(void){};", var.name);
            }
        }
        self.out.push_str(&o);
    }
}

impl Structured {
    fn signature(&self, g: usize, an: &Analysis) -> String {
        let f = &self.funcs[g].form;
        let used = &an.params[g];
        let mut ps = Vec::new();
        if an.arena[g] {
            ps.push("presk_arena *arena".to_string());
        }
        if an.keys[g] {
            ps.push("const presk_key *const *keys".to_string());
        }
        for (k, kind) in f.params.iter().enumerate() {
            if !used[k] {
                continue;
            }
            ps.push(match kind {
                PKind::Ptr(s @ Store::Quant { .. }, _) => format!("{} {}", quant_type(*s), f.param_names[k]),
                PKind::Ptr(s, _) => format!("const {} *{}", ctype(*s), f.param_names[k]),
                PKind::Double => format!("double {}", f.param_names[k]),
                PKind::Index => "int64_t i".into(),
                PKind::Fixed => continue,
            });
        }
        for (k, (s, _)) in f.outs.iter().enumerate() {
            ps.push(format!("{} *{}", ctype(*s), f.out_names[k]));
        }
        for n in &f.sout_names {
            ps.push(format!("double *{n}"));
        }
        if ps.is_empty() {
            ps.push("void".into());
        }
        format!("static void {}({})", self.funcs[g].name, ps.join(", "))
    }

    /// The functions of `model.c`, then each entry's: `ios[0].func` runs
    /// the root, `ios[k]` the `k`-th of the others.
    pub fn print(&self, ios: &[EntryIo]) -> String {
        let an = self.analysis();
        let mut out = String::new();
        for (g, func) in self.funcs.iter().enumerate() {
            let f = &func.form;
            let dims = if f.dims.is_empty() { String::new() } else { format!(" at {}", f.dims.iter().map(|(n, v)| format!("{n} = {v}")).collect::<Vec<_>>().join(", ")) };
            let twins: Vec<&str> = self.funcs.iter().filter(|g| g.form.func == f.func && g.form.dims == f.dims && g.name != func.name).map(|g| g.name.as_str()).collect();
            let twin = match twins.as_slice() {
                [] => String::new(),
                [one] => format!("\n * Its other recording at these dimensions, {one}, differs in an argument's\n * type or in an operation."),
                many => format!("\n * Its other recordings at these dimensions, {}, differ in an argument's\n * type or in an operation.", many.join(", ")),
            };
            let _ = writeln!(out, "/* {}{dims}: {} call{} of the recording.{twin} */", f.func, func.calls, if func.calls == 1 { "" } else { "s" });
            let _ = writeln!(out, "{}\n{{", self.signature(g, &an));
            let uses = uses(f, &an);
            let mut p = Printer { s: self, form: f, folds: folds_of(f), params: f.param_names.clone(), outs: f.out_names.clone(), an: &an, uses, out: String::new(), site: String::new() };
            let allocates = f.body.iter().any(|s| matches!(s, Stmt::Alloc(_) | Stmt::Fold(_) | Stmt::Release));
            if allocates {
                let _ = writeln!(p.out, "    const size_t mark = presk_mark(arena);");
            }
            p.body();
            for (v, name) in f.souts.iter().zip(&f.sout_names) {
                let _ = writeln!(p.out, "    *{name} = s{v};");
            }
            if allocates {
                let _ = writeln!(p.out, "    presk_release(arena, mark);");
            }
            out.push_str(&p.out);
            out.push_str("}\n\n");
        }
        for (root, io) in std::iter::once(&self.root).chain(&self.others).zip(ios) {
            out.push_str(&self.print_entry(root, io, &an));
            out.push('\n');
        }
        out.pop();
        out
    }

    /// An entry's function: its root's form, its parameters the fields of
    /// `in` (a `double` read where it is used), its outputs allocated before
    /// its mark, so that they outlive it.
    fn print_entry(&self, f: &Form, io: &EntryIo, an: &Analysis) -> String {
        let mut out = String::new();
        let uses = uses(f, an);
        let _ = writeln!(out, "void {}(presk_arena *arena, const {}_in *in, {}_out *out)\n{{", io.func, io.types, io.types);
        let mut body = String::new();
        if uses.keys {
            // The keys `sample` draws from, by the recording's key arguments;
            // one nothing draws from is no field of `in`.
            let ks: Vec<String> = io.in_keys.iter().map(|k| k.as_ref().map_or("NULL".to_string(), |f| format!("&in->{f}"))).collect();
            let _ = writeln!(body, "    const presk_key *const keys[] = {{{}}};", ks.join(", "));
        }
        let mut params = f.param_names.clone();
        let (mut field, mut double) = (io.in_fields.iter(), io.in_doubles.iter());
        for (k, kind) in f.params.iter().enumerate() {
            match kind {
                PKind::Ptr(s, _) => {
                    let fld = field.next().cloned().unwrap_or_default();
                    if !uses.params[k] {
                        params[k] = format!("in->{fld}");
                    } else if matches!(s, Store::Quant { .. }) {
                        let _ = writeln!(body, "    const {} {} = in->{fld};", quant_type(*s), f.param_names[k]);
                    } else {
                        let _ = writeln!(body, "    const {} *{} = in->{fld};", ctype(*s), f.param_names[k]);
                    }
                }
                PKind::Double => params[k] = format!("in->{}", double.next().cloned().unwrap_or_default()),
                PKind::Index | PKind::Fixed => {}
            }
        }
        if !uses.params.iter().any(|u| *u) {
            let _ = writeln!(body, "    (void)in;");
        }
        for (k, (s, n)) in f.outs.iter().enumerate() {
            let _ = writeln!(body, "    {} *{} = presk_alloc(arena, {n} * sizeof({}));", ctype(*s), f.out_names[k], ctype(*s));
        }
        let _ = writeln!(body, "    const size_t mark = presk_mark(arena);");
        let mut p = Printer { s: self, form: f, folds: folds_of(f), params, outs: f.out_names.clone(), an, uses, out: body, site: String::new() };
        p.body();
        let mut fields = io.out_fields.iter();
        for r in &f.result {
            if let Res::Ptr(x) = r {
                let fld = fields.next().cloned().unwrap_or_default();
                let e = p.r(x);
                let _ = writeln!(p.out, "    out->{fld} = {e};");
            }
        }
        let _ = writeln!(p.out, "    presk_release(arena, mark);");
        out.push_str(&p.out);
        out.push_str("}\n");
        out
    }
}

/// The site a statement is commented with.
fn site_of(st: &Stmt) -> Option<&str> {
    match st {
        Stmt::Launch { site, .. } | Stmt::Fill { site, .. } | Stmt::Scalar { site, .. } | Stmt::Copy { site, .. } | Stmt::Call { site, .. } | Stmt::Sample { site, .. } => Some(site),
        Stmt::Fold(fs) => Some(&fs.site),
        _ => None,
    }
}

fn folds_of(f: &Form) -> Vec<&FoldStmt> {
    f.body
        .iter()
        .filter_map(|s| match s {
            Stmt::Fold(fs) => Some(fs.as_ref()),
            _ => None,
        })
        .collect()
}

