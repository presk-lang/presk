//! The host's check, by mutation (plan 0.9.0 steps 3 and 4, D79, D80). An
//! entry is recorded on the native backend, exported, built by its own
//! `Makefile` and run as a subprocess, so that a crash stays out of the
//! test; its outputs are compared with the native backend's bit for bit.
//! Unmutated, they agree, through the straight-line host and through the
//! structured one, whose arena is exactly the peak the program reaches.
//! Each mutation must make them differ. Of the straight-line host (step 3),
//! acting on the recording's operations:
//!
//! - **a launch dropped**, each live launch in turn: its buffer keeps what
//!   the arena held, which the test fills with NaNs (all ones), so that a
//!   launch whose output is zero is caught as well;
//! - **two buffers swapped**: two live launches of the same type and size
//!   write each other's buffer, so every later reader of one reads the
//!   other;
//! - **a `bf16` input rounded once**: the reader rounds an `f64` straight to
//!   `bf16`, not through `f32` as the native backend does (`Buf::of`), on an
//!   input where the two roundings differ.
//!
//! Of the structured host (step 4), acting on its functions, on MiniGPT's
//! forward pass (a fold as a loop) and on its gradient (a forward fold, a
//! scan and a backward fold, each a loop):
//!
//! - **a slice stride off by a layer**: a loop hands layer `l + 1`'s slice of
//!   a stacked parameter to iteration `l` (the last reads past its tensor);
//! - **a loop bound off by one**: a loop runs one iteration less, or one
//!   more;
//! - **a release before a buffer's last use**: a function releases its
//!   frame right after a buffer is written, and the next allocation takes
//!   the buffer's place while it is still to be read.
//!
//! And a host scalar computed at run time, which no source program makes
//! cross a call, crosses two in a recording written by hand.
//!
//! `presk-cli`'s tests check C1 through `presk export` itself on the whole
//! test set.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, Once};

use presk_export_c::host::{self, Arg, Meta};
use presk_export_c::structure::{Feed, FoldStmt, Form, Loc, Stmt, Structured};
use presk_kernel::record::{Node, Scalar, Sym, View, leaves as sym_leaves};
use std::collections::HashSet;
use presk_interp::RunOptions;
use presk_kernel::Kernel;
use presk_kernel::record::{Op, Recorder, Recording};
use presk_rt::{Backend, Buf, Dtype, Realization, Value};
use presk_syntax::diag::render_all;
use presk_syntax::{SourceMap, parse_file};
use presk_types::check;
use presk_types::elab::{self, ElabProgram, Options};
use rug::Float;

fn chdir_root() {
    static ONCE: Once = Once::new();
    let root: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap();
    ONCE.call_once(|| std::env::set_current_dir(root).unwrap());
}

fn prepare(rel: &str, consts: &[(&str, u64)]) -> (ElabProgram, presk_core::Program) {
    chdir_root();
    let mut sm = SourceMap::new();
    let id = sm.load(Path::new(rel)).unwrap();
    let (program, diags) = parse_file(&sm, id);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let opts = Options { const_overrides: consts.iter().map(|(n, v)| (n.to_string(), *v)).collect() };
    let (prog, diags) = elab::elaborate(&mut sm, id, program, &opts);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (typed, diags) = check::check(&sm, &prog);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (mut core, diags) = presk_core::lower(&prog, &typed);
    assert!(diags.is_empty(), "{}", render_all(&diags, &sm, false));
    let diags = presk_core::ad::transform(&mut core);
    assert!(diags.is_empty(), "{}", render_all(&diags, &sm, false));
    presk_core::verify(&core).unwrap();
    (prog, core)
}

/// An entry recorded on the native backend, with what its program needs.
struct Recorded {
    rec: Recording,
    kernels: Vec<Kernel>,
    /// The native backend's outputs, as the `f64` bits a `.psk.d` holds.
    native: Vec<u64>,
    /// The arguments before realization, as `presk run` makes them.
    args: Vec<Value>,
    names: Vec<String>,
    /// The first argument's module, when it is a module's parameters.
    module: Option<String>,
    realize: Dtype,
    entry: String,
}

/// Records `entry` as `presk export` does, its arguments changed by `tweak`.
fn record(rel: &str, consts: &[(&str, u64)], dims: &[(&str, u64)], entry: &str, realize: Dtype, tweak: impl FnOnce(&mut Vec<Value>)) -> Recorded {
    let (prog, core) = prepare(rel, consts);
    let opts = RunOptions { entry: entry.into(), dims: dims.iter().map(|(n, v)| (n.to_string(), *v)).collect(), seed: 0, bits: 256, digits: 12 };
    let (func, fdims, inputs) = presk_interp::run::prepare_run(&core, &prog, &opts).unwrap();
    let names: Vec<String> = inputs.iter().map(|(n, _)| n.clone()).collect();
    let mut args: Vec<Value> = inputs.iter().map(|(_, v)| Value::from_interp(v).unwrap()).collect();
    tweak(&mut args);
    let module = core.func(func).params.first().and_then(|p| match &p.ty {
        presk_types::ty::Ty::Params(presk_types::ty::ModRef::Inst { def, .. }) => Some(prog.module(*def).name.clone()),
        _ => None,
    });
    let r = Realization::uniform(realize);
    let mut b = presk_cranelift::native_backend(&core, &prog, &r).unwrap();
    let p = b.prepare(func, &fdims, &r).unwrap();
    b.recorder = Some(Box::new(Recorder::named(names.clone())));
    let out = b.call(&p, args.clone()).unwrap();
    let rec = b.recorder.take().unwrap().finish().unwrap();
    Recorded { rec, kernels: b.kernels.clone(), native: bits(&out), args, names, module, realize, entry: entry.into() }
}

fn leaves(v: &Value, out: &mut Vec<Value>) {
    match v {
        Value::Tuple(vs) => vs.iter().for_each(|x| leaves(x, out)),
        Value::Record(fs) => fs.iter().for_each(|(_, x)| leaves(x, out)),
        leaf => out.push(leaf.clone()),
    }
}

fn bits(v: &Value) -> Vec<u64> {
    let mut ls = Vec::new();
    leaves(v, &mut ls);
    ls.iter()
        .flat_map(|l| match l {
            Value::Real(t) => t.data.to_f64().iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            Value::Idx { t, .. } => t.data.to_f64().iter().map(|x| *x as u64).collect(),
            _ => vec![],
        })
        .collect()
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("presk-host-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(d.join("conformance")).unwrap();
    d
}

/// A change to the files of a program, `(name, text)`.
type Edit<'a> = &'a dyn Fn(&mut Vec<(String, String)>);

/// Which host a program is exported with.
#[derive(Clone, Copy)]
enum Host<'a> {
    /// The straight-line host of this recording.
    Flat(&'a Recording),
    /// This structured host of the subject's recording.
    Structured(&'a Structured),
}

/// A program's run: its outputs' bits, the arena's peak, and the arena it
/// was given.
struct Ran {
    bits: Vec<u64>,
    peak: usize,
    arena: usize,
}

/// The program of `r` with `host`, its files changed by `edit`, built and
/// run in `dir`: its outputs' bits, or why it failed.
fn run(r: &Recorded, rec: &Recording, dir: &Path, edit: Edit) -> Result<Vec<u64>, String> {
    run_with(r, Host::Flat(rec), dir, edit).map(|x| x.bits)
}

fn meta_args(r: &Recorded) -> (Meta, Vec<Arg>, Option<String>) {
    let meta = Meta { file: "test".into(), entry: r.entry.clone(), realize: r.realize.name().into(), version: "test".into(), exact: false, ..Meta::default() };
    let args: Vec<Arg> = r.names.iter().enumerate().map(|(i, n)| Arg { name: n.clone(), weights: i == 0 && r.module.is_some() }).collect();
    (meta, args, r.module.as_ref().map(|m| format!("params {m}")))
}

fn run_with(r: &Recorded, host: Host, dir: &Path, edit: Edit) -> Result<Ran, String> {
    let (x, run_args) = write_program(r, host, dir, edit)?;
    build(dir, None)?;
    execute(dir, &run_args, None, x.arena)
}

/// The program of `r` with `host`, its files changed by `edit`, written in
/// `dir` with its inputs: the export, and the program's arguments.
fn write_program(r: &Recorded, host: Host, dir: &Path, edit: Edit) -> Result<(host::Exported, Vec<PathBuf>), String> {
    let (meta, args, wty) = meta_args(r);
    let x = match host {
        Host::Flat(rec) => host::export_flat(&meta, &r.kernels, rec, &args, wty.as_deref()),
        Host::Structured(st) => host::export_structured(&meta, &r.kernels, &r.rec, &args, wty.as_deref(), st),
    }
    .map_err(|e| format!("export: {e}"))?;
    let mut files = x.files.clone();
    // An arena of NaNs: a buffer no launch wrote is caught even where its
    // value would have been zero. And the peak the program reached.
    for (name, text) in files.iter_mut() {
        if name == "main.c" {
            let at = text.find("    presk_arena arena;").unwrap();
            let end = at + text[at..].find("    }\n").unwrap() + "    }\n".len();
            text.insert_str(end, "    memset(arena.base, 0xff, arena.size);\n");
            let at = text.find("&in, &out);\n").unwrap() + "&in, &out);\n".len();
            text.insert_str(at, "    fprintf(stderr, \"presk-peak %zu\\n\", arena.peak);\n");
        }
    }
    edit(&mut files);
    for (name, text) in &files {
        std::fs::write(dir.join(name), text).unwrap();
    }
    let mut run_args = Vec::new();
    if let Some(m) = &r.module {
        let w = dir.join("conformance/weights.psk.d");
        presk_rt::save_params(&w, m, &r.args[0]).unwrap();
        run_args.push(w);
    }
    if let Some(f) = &x.inputs {
        let mut ls = Vec::new();
        for a in r.args.iter().skip(usize::from(r.module.is_some())) {
            leaves(a, &mut ls);
        }
        let data: Vec<presk_data::Value> = ls
            .iter()
            .filter_map(|l| match l {
                Value::Real(t) => Some(presk_data::Value::Real(t.data.to_f64())),
                Value::Idx { t, .. } => Some(presk_data::Value::Idx(t.data.to_f64().iter().map(|x| *x as u64).collect())),
                _ => None,
            })
            .collect();
        let v = if data.len() == 1 && !f.ty.starts_with('(') { data.into_iter().next().unwrap() } else { presk_data::Value::Tuple(data) };
        let layout = presk_data::Layout::from_type_text(&f.ty).unwrap();
        let path = dir.join("conformance/inputs.psk.d");
        presk_data::write(&path, &f.ty, &layout, std::slice::from_ref(&v)).unwrap();
        run_args.push(path);
    }
    Ok((x, run_args))
}

/// Builds the program in `dir` by its `Makefile`: with OpenMP or without it
/// when `openmp` says, else as the `Makefile` has it. A warning is an error
/// (statement C3).
fn build(dir: &Path, openmp: Option<bool>) -> Result<(), String> {
    let mut make = Command::new("make");
    make.arg("-s").arg("-B").arg("-C").arg(dir);
    if let Some(on) = openmp {
        make.arg(if on { format!("OPENMP={}", presk_export_c::OPENMP) } else { "OPENMP=".into() });
    }
    let make = make.arg("model").output().unwrap();
    let err = String::from_utf8_lossy(&make.stderr);
    if !make.status.success() || err.contains("warning") {
        return Err(format!("make: {err}"));
    }
    Ok(())
}

/// Runs the program built in `dir`, on `threads` threads when given
/// (`OMP_NUM_THREADS`): its outputs' bits and the arena's peak.
fn execute(dir: &Path, run_args: &[PathBuf], threads: Option<usize>, arena: usize) -> Result<Ran, String> {
    let out = dir.join("conformance/out.psk.d");
    let _ = std::fs::remove_file(&out);
    let mut cmd = Command::new(dir.join("model"));
    cmd.args(run_args).arg(&out);
    if let Some(n) = threads {
        cmd.env("OMP_NUM_THREADS", n.to_string());
    }
    let p = cmd.output().unwrap();
    if !p.status.success() {
        return Err(format!("the program: {:?} {}", p.status, String::from_utf8_lossy(&p.stderr)));
    }
    let stderr = String::from_utf8_lossy(&p.stderr);
    let peak = stderr.lines().find_map(|l| l.strip_prefix("presk-peak ")).and_then(|n| n.trim().parse().ok()).ok_or("no peak printed")?;
    let mut reader = presk_data::Reader::open(&out).map_err(|e| e.to_string())?;
    let mut got = Vec::new();
    fn flat(v: &presk_data::Value, out: &mut Vec<u64>) {
        match v {
            presk_data::Value::Real(d) => out.extend(d.iter().map(|x| x.to_bits())),
            presk_data::Value::Idx(d) => out.extend(d),
            presk_data::Value::Tuple(vs) => vs.iter().for_each(|x| flat(x, out)),
            presk_data::Value::Record(fs) => fs.iter().for_each(|(_, x)| flat(x, out)),
        }
    }
    for v in reader.read_all().map_err(|e| e.to_string())? {
        flat(&v, &mut got);
    }
    Ok(Ran { bits: got, peak, arena })
}

/// Whether a run gives the native backend's bits.
fn same(r: &Recorded, got: &Result<Vec<u64>, String>) -> bool {
    matches!(got, Ok(b) if *b == r.native)
}

/// Runs `jobs` on every core.
fn parallel<T: Send>(jobs: Vec<Box<dyn FnOnce() -> T + Send + '_>>) -> Vec<T> {
    let n = jobs.len();
    let queue = Mutex::new(jobs.into_iter().enumerate().collect::<Vec<_>>());
    let results = Mutex::new((0..n).map(|_| None).collect::<Vec<Option<T>>>());
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(n.max(1));
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                loop {
                    let Some((i, job)) = queue.lock().unwrap().pop() else { break };
                    let r = job();
                    results.lock().unwrap()[i] = Some(r);
                }
            });
        }
    });
    results.into_inner().unwrap().into_iter().map(|r| r.unwrap()).collect()
}

/// Every parameter an initializer sets to zero (a bias, a shift) drawn
/// instead, from a fixed sequence: adding a zero bias changes nothing, so
/// on the synthesized parameters a launch that adds one could be skipped
/// unseen.
fn draw_zeros(args: &mut [Value]) {
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut draw = |v: &Value| match v {
        Value::Real(t) => {
            let mut t = t.clone();
            if let Buf::Exact(d) = &mut t.data {
                for x in d.iter_mut().filter(|x| x.is_zero()) {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    *x = Float::with_val(256, ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.2);
                }
            }
            Value::Real(t)
        }
        other => other.clone(),
    };
    args[0] = args[0].map_leaves(&mut draw);
}

fn subjects() -> Vec<Recorded> {
    vec![
        record("tests/examples/layernorm/layernorm.psk", &[], &[("batch", 3)], "LayerNorm{8}.apply", Dtype::F32, |a| draw_zeros(a)),
        record("tests/examples/mlp/mlp.psk", &[], &[("D", 4), ("B", 2)], "mse_grad", Dtype::F64, |a| draw_zeros(a)),
        record("corpus/nlp/05_gpt.psk", &[("V", 92), ("D", 16), ("H", 2), ("L", 1)], &[("B", 2), ("T", 8)], "SelfAttention.apply", Dtype::Bf16, |a| draw_zeros(a)),
    ]
}

#[test]
fn dropping_any_live_launch_changes_the_outputs() {
    let subjects = subjects();
    let mut jobs: Vec<Box<dyn FnOnce() -> (String, bool) + Send + '_>> = Vec::new();
    for (s, r) in subjects.iter().enumerate() {
        jobs.push(Box::new(move || {
            let got = run(r, &r.rec, &scratch(&format!("drop{s}")), &|_| {});
            (format!("{} unmutated: {:?}", r.entry, got.as_ref().err()), same(r, &got))
        }));
        for (n, i) in host::live_launches(&r.rec).into_iter().enumerate() {
            jobs.push(Box::new(move || {
                let mut rec = r.rec.clone();
                let op = rec.ops.remove(i);
                let got = run(r, &rec, &scratch(&format!("drop{s}-{n}")), &|_| {});
                (format!("{} without launch {n} ({op:?})", r.entry), !same(r, &got))
            }));
        }
    }
    let results = parallel(jobs);
    let failed: Vec<&String> = results.iter().filter(|(_, ok)| !ok).map(|(w, _)| w).collect();
    assert!(failed.is_empty(), "{} of {} checks fail:\n{}", failed.len(), results.len(), failed.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n"));
    eprintln!("{} programs: every live launch dropped changes the outputs", results.len());
}

#[test]
fn swapping_the_buffers_of_two_launches_changes_the_outputs() {
    let subjects = subjects();
    let mut jobs: Vec<Box<dyn FnOnce() -> (String, bool) + Send + '_>> = Vec::new();
    for (s, r) in subjects.iter().enumerate() {
        let live = host::live_launches(&r.rec);
        let out = |i: usize| match &r.rec.ops[i] {
            Op::Launch { out, .. } => *out,
            _ => unreachable!(),
        };
        // The launches that read each buffer.
        let mut readers: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for (k, op) in r.rec.ops.iter().enumerate() {
            if let Op::Launch { inputs, .. } = op {
                for v in inputs {
                    readers.entry(v.buf).or_default().push(k);
                }
            }
        }
        // Pairs of live launches whose outputs have one type and size, but
        // for two that one launch alone reads: it may read them as operands
        // of a product, where exchanging them changes no bit (`mse.bwd`'s
        // `(ŷ - y)` and `mean_vjp`, since the gradient keeps the first as a
        // residual, 0.10.0 step 4).
        let mut pairs = Vec::new();
        for (a, &i) in live.iter().enumerate() {
            for &j in &live[a + 1..] {
                let (bi, bj) = (&r.rec.bufs[out(i)], &r.rec.bufs[out(j)]);
                let (ri, rj) = (readers.get(&out(i)), readers.get(&out(j)));
                if bi.store == bj.store && bi.len == bj.len && !(ri == rj && ri.is_some_and(|v| v.len() == 1)) {
                    pairs.push((i, j));
                }
            }
        }
        assert!(!pairs.is_empty(), "{}: no two launches of one type and size", r.entry);
        for (n, (i, j)) in pairs.into_iter().take(24).enumerate() {
            jobs.push(Box::new(move || {
                let mut rec = r.rec.clone();
                let (oi, oj) = (out(i), out(j));
                if let Op::Launch { out, .. } = &mut rec.ops[i] {
                    *out = oj;
                }
                if let Op::Launch { out, .. } = &mut rec.ops[j] {
                    *out = oi;
                }
                let got = run(r, &rec, &scratch(&format!("swap{s}-{n}")), &|_| {});
                (format!("{}: launches {i} and {j} swapped", r.entry), !same(r, &got))
            }));
        }
    }
    let results = parallel(jobs);
    let failed: Vec<&String> = results.iter().filter(|(_, ok)| !ok).map(|(w, _)| w).collect();
    assert!(failed.is_empty(), "{} of {} swaps go unseen:\n{}", failed.len(), results.len(), failed.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n"));
    eprintln!("{} swaps, each changes the outputs", results.len());
}

/// `1 + 2⁻⁸ + 2⁻³⁰`: rounded to `f32` it is `1 + 2⁻⁸`, halfway between
/// the `bf16` neighbours 1 and `1 + 2⁻⁷`, which ties to even give 1; rounded
/// straight to `bf16` it is above the halfway point, `1 + 2⁻⁷`.
const TIE: f64 = 1.0 + 1.0 / 256.0 + 1.0 / 1_073_741_824.0;

#[test]
fn a_bf16_input_rounded_once_changes_the_outputs() {
    let f = TIE as f32;
    assert_eq!(f.to_bits() & 0xffff, 0x8000, "the f32 rounding lands on a bf16 tie");
    assert_eq!(presk_rt::f32_to_bf16(f), 0x3f80, "which ties to even");
    let r = record("tests/examples/layernorm/layernorm.psk", &[], &[("batch", 3)], "LayerNorm{8}.apply", Dtype::Bf16, |args| {
        let Value::Real(t) = &mut args[1] else { panic!("x is a tensor") };
        let Buf::Exact(d) = &mut t.data else { panic!("an exact input") };
        d[0] = Float::with_val(256, TIE);
    });
    let got = run(&r, &r.rec, &scratch("tie"), &|_| {});
    assert!(same(&r, &got), "unmutated: {:?}", got.err());
    // The reader rounds an f64 straight to bf16: to nearest, ties to even,
    // at bit 45 of the f64 (a normal value in f32's range).
    let once = |files: &mut Vec<(String, String)>| {
        for (name, text) in files.iter_mut() {
            if name == "presk_rt.h" {
                let old = "    return presk_f32_to_bf16((float)x);\n}";
                assert!(text.contains(old));
                *text = text.replacen(old, "    uint64_t b;\n    memcpy(&b, &x, sizeof b);\n    b = (b + (((uint64_t)1 << 44) - 1) + ((b >> 45) & 1)) & ~(((uint64_t)1 << 45) - 1);\n    memcpy(&x, &b, sizeof x);\n    return presk_f32_to_bf16((float)x);\n}", 1);
            }
        }
    };
    let got = run(&r, &r.rec, &scratch("tie-once"), &once);
    assert!(got.is_ok(), "{:?}", got.err());
    assert!(!same(&r, &got), "a bf16 input rounded once gives the native backend's bits");
}

// ----- the structured host (step 4) ---------------------------------------------------

/// MiniGPT's forward pass, its three blocks a loop, and its gradient at two
/// blocks: a forward fold, a scan and a backward fold, each a loop. The
/// zeros drawn, so that a block's outputs reach the logits.
fn folded() -> Vec<Recorded> {
    let c3 = [("V", 64), ("D", 16), ("H", 2), ("L", 3)];
    let c2 = [("V", 64), ("D", 16), ("H", 2), ("L", 2)];
    let d = [("B", 2), ("T", 8)];
    vec![
        record("tests/examples/minigpt/minigpt.psk", &c3, &d, "MiniGPT.apply", Dtype::F32, |a| draw_zeros(a)),
        record("tests/plan/minigpt_grad.psk", &c2, &d, "loss_grad", Dtype::F64, |a| draw_zeros(a)),
    ]
}

fn forms_mut(s: &mut Structured) -> Vec<&mut Form> {
    let mut out: Vec<&mut Form> = s.funcs.iter_mut().map(|f| &mut f.form).collect();
    out.push(&mut s.root);
    out
}

/// The loops of a structured host: `(form, statement, run)`, the form
/// counted as in [`forms_mut`].
fn loops(s: &Structured) -> Vec<(usize, usize, usize)> {
    let forms: Vec<&Form> = s.funcs.iter().map(|f| &f.form).chain(std::iter::once(&s.root)).collect();
    let mut out = Vec::new();
    for (i, f) in forms.iter().enumerate() {
        for (j, st) in f.body.iter().enumerate() {
            if let Stmt::Fold(fs) = st {
                for (k, run) in fs.runs.iter().enumerate() {
                    if run.count >= 2 {
                        out.push((i, j, k));
                    }
                }
            }
        }
    }
    out
}

fn fold_at(s: &mut Structured, (i, j): (usize, usize)) -> &mut FoldStmt {
    match &mut forms_mut(s).into_iter().nth(i).unwrap().body[j] {
        Stmt::Fold(fs) => fs,
        _ => unreachable!(),
    }
}

type Mutation = (String, Structured);

/// Runs each mutated host of each subject; each must change the outputs.
fn mutations_change_the_outputs(subjects: &[Recorded], what: &str, mutate: impl Fn(&Recorded, &Structured) -> Vec<Mutation>) {
    let mut jobs: Vec<Box<dyn FnOnce() -> (String, bool) + Send + '_>> = Vec::new();
    for (si, r) in subjects.iter().enumerate() {
        let st = host::structure(&r.kernels, &r.rec, 1).unwrap();
        let ms = mutate(r, &st);
        assert!(!ms.is_empty(), "{}: no {what} to mutate", r.entry);
        for (n, (label, m)) in ms.into_iter().enumerate() {
            jobs.push(Box::new(move || {
                let got = run_with(r, Host::Structured(&m), &scratch(&format!("{what}{si}-{n}")), &|_| {}).map(|x| x.bits);
                (format!("{}: {label}", r.entry), !same(r, &got))
            }));
        }
    }
    let results = parallel(jobs);
    let failed: Vec<&String> = results.iter().filter(|(_, ok)| !ok).map(|(w, _)| w).collect();
    assert!(failed.is_empty(), "{} of {} mutations go unseen:\n{}", failed.len(), results.len(), failed.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n"));
    eprintln!("{} programs: every {what} changes the outputs", results.len());
}

#[test]
fn the_structured_host_computes_the_native_bits_in_an_arena_at_its_peak() {
    let mut subjects = subjects();
    subjects.extend(folded());
    type Both = (String, Result<(Ran, Ran), String>);
    let jobs: Vec<Box<dyn FnOnce() -> Both + Send + '_>> = subjects
        .iter()
        .enumerate()
        .map(|(i, r)| -> Box<dyn FnOnce() -> Both + Send + '_> {
            Box::new(move || {
                let st = host::structure(&r.kernels, &r.rec, 1).unwrap();
                let s = run_with(r, Host::Structured(&st), &scratch(&format!("peak{i}")), &|_| {});
                let f = run_with(r, Host::Flat(&r.rec), &scratch(&format!("peak{i}-flat")), &|_| {});
                (r.entry.clone(), s.and_then(|s| f.map(|f| (s, f))))
            })
        })
        .collect();
    for ((entry, got), r) in parallel(jobs).into_iter().zip(&subjects) {
        let (s, f) = got.unwrap_or_else(|e| panic!("{entry}: {e}"));
        assert!(s.bits == r.native, "{entry}: the structured host's outputs are not the native backend's");
        assert!(f.bits == r.native, "{entry}: the straight-line host's outputs are not the native backend's");
        assert_eq!(s.peak, s.arena, "{entry}: the arena is not the peak the structured host reaches");
        assert_eq!(f.peak, f.arena, "{entry}: the straight-line host fills its arena");
        eprintln!("{entry}: an arena of {} bytes, {} straight-line", s.arena, f.arena);
    }
}

/// Which inputs a kernel loads from: an operand of `sum_to` is read for its
/// shape, never for its elements.
fn loads(k: &Kernel) -> Vec<bool> {
    use presk_kernel::ir::{Expr, Stmt as K};
    fn e(x: &Expr, out: &mut Vec<bool>) {
        match x {
            Expr::Load(b, ix) | Expr::LoadI(b, ix) | Expr::VLoad(b, ix) => {
                if let Some(o) = out.get_mut(*b) {
                    *o = true;
                }
                ix.iter().for_each(|y| e(y, out));
            }
            Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) | Expr::VBin(_, a, b) => {
                e(a, out);
                e(b, out);
            }
            Expr::Un(_, a) | Expr::ToF(a) | Expr::ToI(a) | Expr::VSplat(a) => e(a, out),
            Expr::Select(c, a, b) => {
                e(c, out);
                e(a, out);
                e(b, out);
            }
            _ => {}
        }
    }
    fn st(body: &[K], out: &mut Vec<bool>) {
        for x in body {
            match x {
                K::For { body, .. } => st(body, out),
                K::Set { value, .. } | K::SetI { value, .. } | K::SetL { value, .. } | K::VSet { value, .. } => e(value, out),
                K::Store { index, value, .. } | K::StoreI { index, value, .. } | K::VStore { index, value, .. } => {
                    index.iter().for_each(|y| e(y, out));
                    e(value, out);
                }
                K::If { cond, then } => {
                    e(cond, out);
                    st(then, out);
                }
            }
        }
    }
    let mut out = vec![false; k.inputs.len()];
    st(&k.body, &mut out);
    out
}

/// The backend kernel each printed kernel (`k<n>`) is, deduplicated by
/// text in the order of first launch, as the export numbers them.
fn printed(r: &Recorded) -> Vec<usize> {
    let mut texts: Vec<String> = Vec::new();
    let mut out = Vec::new();
    for op in &r.rec.ops {
        if let Op::Launch { kernel, .. } = op {
            let t = presk_export_c::kernel_fn(&r.kernels[*kernel], "K").unwrap();
            if !texts.contains(&t) {
                texts.push(t);
                out.push(*kernel);
            }
        }
    }
    out
}

/// What each operation of a recording reads that the result depends on:
/// nothing for an operation whose output nothing live reads, and of a
/// launch only the inputs its kernel loads.
fn live_reads(rec: &Recording, kernels: &[Kernel]) -> Vec<Vec<View>> {
    let mut bufs = vec![false; rec.bufs.len()];
    let mut vars = vec![false; rec.vars];
    for l in sym_leaves(rec.result.as_ref().unwrap()) {
        if let Sym::Tensor(v) = l {
            bufs[v.buf] = true;
        }
    }
    let mut out = vec![Vec::new(); rec.ops.len()];
    for (i, op) in rec.ops.iter().enumerate().rev() {
        match op {
            Op::Launch { kernel, inputs, out: o, .. } if bufs[*o] => {
                let ld = loads(&kernels[*kernel]);
                let read: Vec<View> = inputs.iter().zip(&ld).filter(|(_, l)| **l).map(|(v, _)| v.clone()).collect();
                read.iter().for_each(|v| bufs[v.buf] = true);
                out[i] = read;
            }
            Op::Copy { dst, src, .. } if bufs[*dst] => {
                bufs[src.buf] = true;
                out[i] = vec![src.clone()];
            }
            Op::Read { out: o, src } if vars[*o] => {
                bufs[src.buf] = true;
                out[i] = vec![src.clone()];
            }
            Op::Scalar { buf, x: Scalar::Var(v) } if bufs[*buf] => vars[*v] = true,
            Op::Hyper { out: o, a, b, .. } if vars[*o] => {
                for x in [a, b] {
                    if let Scalar::Var(v) = x {
                        vars[*v] = true;
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// The operations under a call of the recording.
fn ops_under(rec: &Recording, c: usize, out: &mut Vec<usize>) {
    for n in &rec.calls[c].body {
        match *n {
            Node::Op(i) => out.push(i),
            Node::Call(d) => ops_under(rec, d, out),
            Node::Fold(f) => {
                for (_, d) in &rec.folds[f].iterations {
                    ops_under(rec, *d, out);
                }
                out.extend(&rec.folds[f].stack);
            }
        }
    }
}

/// The folds' sliced parameters no live operation of any iteration reads,
/// by the fold's site and the parameter's place among its body's: shifting
/// such a slice cannot change the outputs (the forward pass a gradient
/// recomputes inside a `vjp` reads a bias, and discards what it computes).
fn dead_slices(rec: &Recording, kernels: &[Kernel]) -> HashSet<(String, usize)> {
    let reads = live_reads(rec, kernels);
    let mut live: HashSet<(String, usize)> = HashSet::new();
    let mut all: HashSet<(String, usize)> = HashSet::new();
    for f in &rec.folds {
        let site = match &f.name {
            Some(n) => format!("{} ({n})", f.site),
            None => f.site.clone(),
        };
        for (_, c) in &f.iterations {
            let mut ops = Vec::new();
            ops_under(rec, *c, &mut ops);
            let args: Vec<Sym> = rec.calls[*c].args.iter().flat_map(sym_leaves).collect();
            for (m, a) in args.iter().enumerate() {
                let Sym::Tensor(v) = a else { continue };
                all.insert((site.clone(), m));
                let hit = ops.iter().flat_map(|i| &reads[*i]).any(|r| r.buf == v.buf && r.off < v.off + v.numel() && v.off < r.off + r.numel().max(1));
                if hit {
                    live.insert((site.clone(), m));
                }
            }
        }
    }
    all.difference(&live).cloned().collect()
}

#[test]
fn a_folds_value_returned_as_it_is_is_copied_into_the_callers_buffer() {
    // `tests/export_c/fold_value.psk`: the entry returns a fold's value, and
    // `scaled` calls a function that does. No entry of the test set does.
    let d = [("B", 2)];
    let subjects = [
        record("tests/export_c/fold_value.psk", &[], &d, "Norms.apply", Dtype::F32, |a| draw_zeros(a)),
        record("tests/export_c/fold_value.psk", &[], &d, "scaled", Dtype::Bf16, |a| draw_zeros(a)),
    ];
    for (i, r) in subjects.iter().enumerate() {
        let st = host::structure(&r.kernels, &r.rec, 1).unwrap();
        let copies = st.funcs.iter().map(|f| &f.form).chain(std::iter::once(&st.root)).flat_map(|f| &f.body).filter(|s| matches!(s, Stmt::CopyOut { .. })).count();
        assert_eq!(copies, 1, "{}: one copy out of a fold", r.entry);
        let s = run_with(r, Host::Structured(&st), &scratch(&format!("value{i}")), &|_| {}).unwrap();
        assert_eq!(s.bits, r.native, "{}: the structured host's outputs", r.entry);
        assert_eq!(s.peak, s.arena, "{}: the arena is the peak", r.entry);
    }
}

#[test]
fn a_slice_stride_off_by_a_layer_changes_the_outputs() {
    // Each stacked leaf a loop slices and a live operation reads, in turn:
    // iteration l reads layer l + 1.
    mutations_change_the_outputs(&folded(), "slice", |r, st| {
        let dead = dead_slices(&r.rec, &r.kernels);
        let mut out = Vec::new();
        for (i, j, _) in loops(st) {
            let fs = fold_at(&mut st.clone(), (i, j)).clone();
            for (m, fd) in fs.feed.iter().enumerate() {
                if let Feed::Slice { chunk, .. } = fd {
                    if dead.contains(&(fs.site.clone(), m)) {
                        eprintln!("{}: `{}` slices its body's parameter {m}, which no live operation reads: not mutated", r.entry, fs.site);
                        continue;
                    }
                    let mut x = st.clone();
                    if let Feed::Slice { base, .. } = &mut fold_at(&mut x, (i, j)).feed[m] {
                        base.off.a += *chunk as i64;
                    }
                    out.push((format!("fold {i}/{j}, the slice of parameter {m} off by a layer"), x));
                }
            }
        }
        out
    });
}

#[test]
fn a_loop_bound_off_by_one_changes_the_outputs() {
    mutations_change_the_outputs(&folded(), "bound", |_, st| {
        let mut out = Vec::new();
        for (i, j, k) in loops(st) {
            for (d, label) in [(-1i64, "one iteration less"), (1, "one iteration more")] {
                let mut x = st.clone();
                let run = &mut fold_at(&mut x, (i, j)).runs[k];
                run.count = (run.count as i64 + d) as usize;
                out.push((format!("fold {i}/{j}, run {k}: {label}"), x));
            }
        }
        out
    });
}

/// Whether a statement reads local `k`: of a launch, an input its kernel
/// loads (`loaded[n]` for kernel `k<n>`).
fn reads(st: &Stmt, k: usize, loaded: &[Vec<bool>]) -> bool {
    let local = |l: &Loc| *l == Loc::Local(k);
    match st {
        Stmt::Launch { kernel, ins, .. } => ins.iter().zip(&loaded[*kernel]).any(|(r, l)| *l && local(&r.loc)),
        Stmt::Copy { src, .. } | Stmt::Read { src, .. } | Stmt::CopyOut { src, .. } => local(&src.loc),
        Stmt::Call { args, .. } => args.iter().any(|a| matches!(a, presk_export_c::structure::Arg::Ptr(r) if local(&r.loc))),
        Stmt::Fold(fs) => fs.feed.iter().any(|f| matches!(f, Feed::Slice { base, .. } if local(&base.loc)) || matches!(f, Feed::Whole(presk_export_c::structure::Arg::Ptr(r)) if local(&r.loc))) || fs.vars.iter().any(|v| matches!(v.init, Some(presk_export_c::structure::Arg::Ptr(r)) if local(&r.loc))),
        _ => false,
    }
}

/// Whether a statement writes local `k`.
fn writes(st: &Stmt, k: usize) -> bool {
    let local = |l: &Loc| *l == Loc::Local(k);
    match st {
        Stmt::Launch { out, .. } => local(&out.loc),
        Stmt::Fill { dst, .. } | Stmt::Scalar { dst, .. } | Stmt::Copy { dst, .. } => local(&dst.loc),
        Stmt::Call { outs, .. } => outs.iter().any(|r| local(&r.loc)),
        _ => false,
    }
}

#[test]
fn a_release_before_a_buffers_last_use_changes_the_outputs() {
    // In each function that allocates, its frame released right after its
    // first buffer is written: the next buffer allocated takes its place,
    // and is written before the first is read for the last time. (Written
    // by the statement that reads the first, an element-wise map would
    // compute in place, and rightly so.)
    mutations_change_the_outputs(&folded(), "release", |r, st| {
        let loaded: Vec<Vec<bool>> = printed(r).iter().map(|k| loads(&r.kernels[*k])).collect();
        let mut out = Vec::new();
        let n = st.funcs.len() + 1;
        for i in 0..n {
            let mut x = st.clone();
            let form = forms_mut(&mut x).into_iter().nth(i).unwrap();
            let Some(a) = form.body.iter().position(|s| matches!(s, Stmt::Alloc(_))) else { continue };
            let Stmt::Alloc(k) = form.body[a] else { unreachable!() };
            let Some(w) = form.body.iter().position(|s| writes(s, k)) else { continue };
            let Some(last) = form.body.iter().rposition(|s| reads(s, k, &loaded)) else { continue };
            let Some(a2) = form.body[w + 1..].iter().position(|s| matches!(s, Stmt::Alloc(_))).map(|p| w + 1 + p) else { continue };
            let Stmt::Alloc(k2) = form.body[a2] else { unreachable!() };
            let Some(w2) = form.body.iter().position(|s| writes(s, k2)) else { continue };
            if w2 >= last {
                continue;
            }
            let name = form.local_names[k].clone();
            form.body.insert(w + 1, Stmt::Release);
            out.push((format!("`{}` released after `{name}` is written", form.func), x));
        }
        out
    });
}

#[test]
fn a_scalar_computed_at_run_time_crosses_calls_as_a_double() {
    // Written by hand, since no source program makes one cross a call: the
    // entry `f(x)` reads x[0] (a host scalar), hands it to `g`, which
    // doubles it and returns it, and fills its result with it.
    use presk_kernel::record::{Call, HyperOp, Made, RecBuf, Store};
    let buf = |store, len, made, site: &str| RecBuf { store, len, made, site: site.into(), var: None, name: None, quant: None };
    let x = View { buf: 0, off: 0, shape: vec![4], bound: None };
    let y = View { buf: 1, off: 0, shape: vec![3], bound: None };
    let rec = Recording {
        bufs: vec![buf(Store::F64, 4, Made::Input { arg: 0, leaf: 0 }, "argument `x`"), buf(Store::F32, 3, Made::Fill, "f/%3 (y)")],
        ops: vec![
            Op::Read { out: 0, src: x.clone() },
            Op::Hyper { out: 1, op: HyperOp::Mul, a: Scalar::Var(0), b: Scalar::Const(2.5) },
            Op::Scalar { buf: 1, x: Scalar::Var(1) },
        ],
        args: vec![Sym::Tensor(x.clone())],
        result: Some(Sym::Tensor(y.clone())),
        vars: 2,
        scalar_args: vec![],
        key_args: vec![],
        calls: vec![
            Call { func: "f".into(), dims: vec![], params: vec!["x".into()], args: vec![Sym::Tensor(x.clone())], result: Some(Sym::Tensor(y.clone())), body: vec![Node::Op(0), Node::Call(1), Node::Op(2)], var: None, name: None, apply: false },
            Call { func: "g".into(), dims: vec![], params: vec!["a".into()], args: vec![Sym::F64(Scalar::Var(0))], result: Some(Sym::F64(Scalar::Var(1))), body: vec![Node::Op(1)], var: Some(2), name: None, apply: false },
        ],
        folds: vec![],
    };
    // A fill of one element, a fill of three: the host fills a buffer of
    // one, so the recording makes its result one element long.
    let mut rec = rec;
    rec.bufs[1].len = 1;
    let y1 = View { buf: 1, off: 0, shape: vec![], bound: None };
    rec.result = Some(Sym::Tensor(y1.clone()));
    rec.calls[0].result = Some(Sym::Tensor(y1));
    let xs = [1.25f64, -3.0, 0.5, 7.0];
    let r = Recorded { rec, kernels: vec![], native: vec![((xs[0] * 2.5) as f32 as f64).to_bits()], args: vec![Value::Real(presk_rt::Tensor { axes: vec![presk_rt::Axis { name: None, len: 4 }], data: Buf::F64(xs.to_vec().into()) })], names: vec!["x".into()], module: None, realize: Dtype::F32, entry: "f".into() };
    let st = host::structure(&r.kernels, &r.rec, 1).unwrap();
    let g = st.funcs.iter().find(|f| f.form.func == "g").expect("`g` is a function");
    assert_eq!(g.form.souts.len(), 1, "`g` returns its scalar through a pointer");
    let s = run_with(&r, Host::Structured(&st), &scratch("scalar"), &|_| {}).unwrap();
    let f = run_with(&r, Host::Flat(&r.rec), &scratch("scalar-flat"), &|_| {}).unwrap();
    assert_eq!(s.bits, r.native, "the structured host");
    assert_eq!(f.bits, r.native, "the straight-line host");
    let text = host::export_structured(&meta_args(&r).0, &r.kernels, &r.rec, &meta_args(&r).1, None, &st).unwrap();
    let model = &text.files.iter().find(|(n, _)| n == "model.c").unwrap().1;
    assert!(model.contains("static void presk_g(double a, double *out)"), "{model}");
}

// ----- threads (step 5) ----------------------------------------------------------------

/// The depths at which a kernel's loop may be divided: parallel, with two
/// iterations or more (condition 8).
fn legal_depths(k: &Kernel) -> Vec<usize> {
    (0..=3).filter(|d| matches!(k.parallel_at(*d), Some(n) if n >= 2)).collect()
}

/// `r` with each kernel it launches divided at its `d`-th legal depth
/// (cyclically) among `p` workers, as a plan stating that split would lower
/// it: the kernel's text ranges over that loop, and the launch is divided.
/// The native bits do not depend on the split (§3.5, rule 4), so the
/// recording's outputs stay the program's. `None` when no kernel changes.
fn split_at(r: &Recorded, d: usize, p: usize) -> Option<Recorded> {
    let mut kernels = r.kernels.clone();
    let mut changed = false;
    for k in kernels.iter_mut() {
        let depths = legal_depths(k);
        if depths.is_empty() {
            continue;
        }
        let depth = depths[d % depths.len()];
        let t = presk_kernel::Threads::At { depth, workers: p };
        changed |= k.threads != t;
        k.threads = t;
    }
    changed.then(|| Recorded { rec: r.rec.clone(), kernels, native: r.native.clone(), args: r.args.clone(), names: r.names.clone(), module: r.module.clone(), realize: r.realize, entry: r.entry.clone() })
}

/// The ways a divided program is run: built with OpenMP on one thread, on
/// three, on every thread the launch's workers allow; and without OpenMP.
const RUNS: &[(Option<bool>, Option<usize>, &str)] = &[(Some(true), Some(1), "one thread"), (Some(true), Some(3), "three threads"), (Some(true), None, "every thread"), (Some(false), None, "without OpenMP")];

/// Exports `r` with `host` into `dir` and runs it every way of [`RUNS`]:
/// what differs from the native backend's bits, if anything.
fn run_every_way(r: &Recorded, host: Host, dir: &Path, edit: Edit) -> Vec<String> {
    let (x, run_args) = match write_program(r, host, dir, edit) {
        Ok(p) => p,
        Err(e) => return vec![e],
    };
    let mut failed = Vec::new();
    let mut built = None;
    for (openmp, threads, label) in RUNS {
        if built != Some(*openmp) {
            if let Err(e) = build(dir, *openmp) {
                failed.push(format!("{label}: {e}"));
                continue;
            }
            built = Some(*openmp);
        }
        let got = execute(dir, &run_args, *threads, x.arena);
        match got {
            Ok(g) if g.bits == r.native => {}
            Ok(_) => failed.push(format!("{label}: other bits")),
            Err(e) => failed.push(format!("{label}: {e}")),
        }
    }
    failed
}

#[test]
fn splits_at_every_legal_depth_compute_the_native_bits_on_any_team() {
    // Each subject with every kernel divided at its first, second, third
    // and fourth legal depth in turn (cyclically), among 3, 4, 2 and 5
    // workers: the pieces uneven, fewer than the iterations or more.
    let mut subjects = subjects();
    subjects.extend(folded());
    let mut variants: Vec<(String, Recorded)> = Vec::new();
    for r in &subjects {
        for (d, p) in [(0usize, 3usize), (1, 4), (2, 2), (3, 5)] {
            if let Some(v) = split_at(r, d, p) {
                variants.push((format!("{} at depth choice {d}, {p} workers", r.entry), v));
            }
        }
    }
    type Job<'a> = Box<dyn FnOnce() -> (String, Vec<String>, usize) + Send + 'a>;
    let mut jobs: Vec<Job> = Vec::new();
    for (i, (label, v)) in variants.iter().enumerate() {
        for flat in [false, true] {
            jobs.push(Box::new(move || {
                let st;
                let host = if flat {
                    Host::Flat(&v.rec)
                } else {
                    st = host::structure(&v.kernels, &v.rec, 1).unwrap();
                    Host::Structured(&st)
                };
                let dir = scratch(&format!("split{i}{}", if flat { "-flat" } else { "" }));
                let failed = run_every_way(v, host, &dir, &|_| {});
                let model = std::fs::read_to_string(dir.join("model.c")).unwrap_or_default();
                (format!("{label}{}", if flat { ", straight-line" } else { "" }), failed, model.matches("#pragma omp parallel").count())
            }));
        }
    }
    let results = parallel(jobs);
    let failed: Vec<String> = results.iter().filter(|(_, f, _)| !f.is_empty()).map(|(w, f, _)| format!("{w}: {}", f.join("; "))).collect();
    assert!(failed.is_empty(), "{} of {} programs differ:\n{}", failed.len(), results.len(), failed.join("\n"));
    assert!(results.iter().all(|(_, _, regions)| *regions > 0), "a program without a divided launch");
    let regions: usize = results.iter().map(|(_, _, n)| n).sum();
    eprintln!("{} programs, {regions} OpenMP regions: the native bits on one thread, three, every one and without OpenMP", results.len());
}

/// A check of the pieces, added to a program's `model.c`: each thread of a
/// region records the range it calls its kernel on, and after the region
/// the pieces must cover the loop's range exactly once, or the program
/// stops. An overlap computes some elements twice, each time with the same
/// bits (§3.5, rule 4), which a comparison of the outputs sees only when
/// two threads race on an output they accumulate in.
const PIECE_CHECK: &str = r#"
#ifdef _OPENMP
#include <stdlib.h>
static int64_t check_lo[1024], check_hi[1024];
static int check_n;
static void check_piece(int64_t lo, int64_t hi)
{
#ifdef _OPENMP
#pragma omp critical
#endif
    {
        check_lo[check_n] = lo;
        check_hi[check_n] = hi;
        check_n++;
    }
}
static void check_pieces(int64_t n)
{
    int64_t covered = 0;
    for (int64_t at = 0; at < n;) {
        int found = 0;
        for (int i = 0; i < check_n; i++) {
            if (check_lo[i] == at && check_hi[i] > at) {
                covered += check_hi[i] - check_lo[i];
                at = check_hi[i];
                found = 1;
                break;
            }
        }
        if (!found)
            break;
    }
    int64_t total = 0;
    for (int i = 0; i < check_n; i++)
        total += check_hi[i] > check_lo[i] ? check_hi[i] - check_lo[i] : 0;
    if (covered != n || total != n) {
        fprintf(stderr, "presk-piece-check: %d pieces cover %lld of 0..%lld, %lld elements in all\n", check_n, (long long)covered, (long long)n, (long long)total);
        exit(3);
    }
    check_n = 0;
}
#endif
"#;

/// `model.c` with [`PIECE_CHECK`]: in each region, the kernel's range
/// recorded before the call, and the pieces checked after the region
/// against the range of the call without OpenMP.
fn with_piece_check(model: &str) -> String {
    let mut out = String::new();
    let lines: Vec<&str> = model.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let l = lines[i];
        out.push_str(l);
        out.push('\n');
        if l.starts_with("#include \"model.h\"") {
            out.push_str(PIECE_CHECK);
        }
        if l.trim_start().starts_with("int64_t t = omp_get_thread_num()") {
            // The call: `kN(args…, LO, HI);`, its last two arguments at the
            // top level of its parentheses.
            let call = lines[i + 1];
            let inner = &call[call.find('(').unwrap() + 1..call.rfind(')').unwrap()];
            let mut depth = 0;
            let mut cuts = Vec::new();
            for (j, c) in inner.char_indices() {
                match c {
                    '(' => depth += 1,
                    ')' => depth -= 1,
                    ',' if depth == 0 => cuts.push(j),
                    _ => {}
                }
            }
            let (a, b) = (cuts[cuts.len() - 2], cuts[cuts.len() - 1]);
            out.push_str(&format!("        check_piece({}, {});\n", inner[a + 1..b].trim(), inner[b + 1..].trim()));
        }
        if l == "#else" && i >= 1 && lines[i - 1].trim() == "}" {
            // The call without OpenMP, `kN(args…, 0, n);`: after the
            // region, the pieces of n.
            let call = lines[i + 1];
            let n = call[call.rfind(',').unwrap() + 1..call.rfind(')').unwrap()].trim();
            // Moved before `#else`: out ends with "#else\n".
            out.truncate(out.len() - "#else\n".len());
            out.push_str(&format!("    check_pieces({n});\n#else\n"));
        }
        i += 1;
    }
    out
}

#[test]
fn a_split_that_leaves_a_gap_or_overlaps_is_caught() {
    // MiniGPT's forward pass and the attention in bf16, every kernel divided
    // at its first legal depth among four workers, run on three threads.
    // A gap leaves elements of the arena's NaNs, which reach the outputs.
    // An overlap computes elements twice, each time with the same bits: the
    // outputs are the native backend's unless two threads race on an output
    // they accumulate in, which is not certain, so they are reported, and
    // the check of the pieces catches it.
    let subjects = [&folded()[0], &subjects()[2]].map(|r| split_at(r, 0, 4).unwrap());
    type Mutate = fn(&str) -> String;
    let mutations: [(&str, Mutate, bool); 4] = [
        ("each piece one element short", |m| m.replace(" * (t + 1) / w);", " * (t + 1) / w - 1);"), true),
        ("the last thread's piece dropped", |m| {
            let re = |s: &str| {
                // `, N * t / w, N * (t + 1) / w);` → the last thread's empty.
                let at = s.find(" * t / w, ").unwrap();
                let start = s[..at].rfind(", ").unwrap() + 2;
                let n = &s[start..at];
                s.replacen(&format!("{n} * t / w, {n} * (t + 1) / w);"), &format!("{n} * t / w, t + 1 < w ? {n} * (t + 1) / w : {n} * t / w);"), 1)
            };
            m.lines().map(|l| if l.contains(" * t / w, ") { re(l) } else { l.to_string() }).collect::<Vec<_>>().join("\n") + "\n"
        }, true),
        ("each piece one element into the one before", |m| {
            m.lines()
                .map(|l| match l.find(" * t / w, ") {
                    Some(at) => {
                        let start = l[..at].rfind(", ").unwrap() + 2;
                        let n = &l[start..at];
                        l.replacen(&format!("{n} * t / w, "), &format!("t > 0 ? {n} * t / w - 1 : 0, "), 1)
                    }
                    None => l.to_string(),
                })
                .collect::<Vec<_>>()
                .join("\n")
                + "\n"
        }, false),
        ("every thread the whole range", |m| {
            m.lines()
                .map(|l| match l.find(" * t / w, ") {
                    Some(at) => {
                        let start = l[..at].rfind(", ").unwrap() + 2;
                        let n = l[start..at].to_string();
                        // 0 to n, still reading t and w (-Wunused-variable).
                        l.replacen(&format!("{n} * t / w, {n} * (t + 1) / w"), &format!("{n} * t / w * 0, {n} * w / w"), 1)
                    }
                    None => l.to_string(),
                })
                .collect::<Vec<_>>()
                .join("\n")
                + "\n"
        }, false),
    ];
    let mut jobs: Vec<Box<dyn FnOnce() -> Result<String, String> + Send + '_>> = Vec::new();
    for (si, r) in subjects.iter().enumerate() {
        // Unmutated: the pieces checked on one thread, three and every one.
        jobs.push(Box::new(move || {
            let st = host::structure(&r.kernels, &r.rec, 1).unwrap();
            let dir = scratch(&format!("pieces{si}"));
            let edit = |files: &mut Vec<(String, String)>| files.iter_mut().filter(|(n, _)| n == "model.c").for_each(|(_, t)| *t = with_piece_check(t));
            let failed = run_every_way(r, Host::Structured(&st), &dir, &edit);
            if failed.is_empty() { Ok(format!("{} unmutated: the pieces cover each range once", r.entry)) } else { Err(format!("{} unmutated: {}", r.entry, failed.join("; "))) }
        }));
        for (mi, (label, mutate, gap)) in mutations.iter().enumerate() {
            jobs.push(Box::new(move || {
                let st = host::structure(&r.kernels, &r.rec, 1).unwrap();
                let run = |check: bool, tag: &str| -> Result<Ran, String> {
                    let dir = scratch(&format!("pieces{si}-{mi}{tag}"));
                    let edit = |files: &mut Vec<(String, String)>| {
                        for (_, t) in files.iter_mut().filter(|(n, _)| n == "model.c") {
                            let m = mutate(t);
                            assert_ne!(&m, t, "the mutation changes nothing");
                            *t = if check { with_piece_check(&m) } else { m };
                        }
                    };
                    let (x, run_args) = write_program(r, Host::Structured(&st), &dir, &edit)?;
                    build(&dir, Some(true))?;
                    execute(&dir, &run_args, Some(3), x.arena)
                };
                // The outputs, the arena of NaNs: a gap is caught.
                let outputs = run(false, "");
                let caught_by_outputs = !matches!(&outputs, Ok(g) if g.bits == r.native);
                if *gap && !caught_by_outputs {
                    return Err(format!("{}: {label}: the outputs are the native backend's", r.entry));
                }
                // The check of the pieces catches both.
                let outputs = if caught_by_outputs { "the outputs differ" } else { "the outputs the native backend's" };
                match run(true, "-check") {
                    Err(e) if e.contains("presk-piece-check") => Ok(format!("{}: {label}: caught by the check of the pieces; {outputs}", r.entry)),
                    other => Err(format!("{}: {label}: the check of the pieces passes: {:?}", r.entry, other.map(|g| g.bits == r.native))),
                }
            }));
        }
    }
    let results = parallel(jobs);
    let failed: Vec<&String> = results.iter().filter_map(|r| r.as_ref().err()).collect();
    for r in results.iter().flatten() {
        eprintln!("{r}");
    }
    assert!(failed.is_empty(), "{}", failed.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n"));
}

#[test]
fn a_launch_divided_otherwise_is_another_function() {
    // MiniGPT's three blocks are one loop over one function. One kernel of
    // the block, copied, and launched by iterations 1 and 2 divided among
    // three workers where iteration 0 runs it on one: the kernel prints
    // alike (one `k<n>`), and iteration 0 becomes a function of its own.
    let r = &folded()[0];
    let fold = &r.rec.folds[0];
    let in_iter = |it: usize| {
        let mut ops = Vec::new();
        ops_under(&r.rec, fold.iterations[it].1, &mut ops);
        ops
    };
    let first = in_iter(0);
    let k = first
        .iter()
        .find_map(|i| match &r.rec.ops[*i] {
            Op::Launch { kernel, .. } if matches!(r.kernels[*kernel].parallel_at(0), Some(n) if n >= 2) && r.kernels[*kernel].threads != presk_kernel::Threads::At { depth: 0, workers: 3 } => Some(*kernel),
            _ => None,
        })
        .expect("a block's kernel with a parallel outermost loop");
    let mut v = Recorded { rec: r.rec.clone(), kernels: r.kernels.clone(), native: r.native.clone(), args: r.args.clone(), names: r.names.clone(), module: r.module.clone(), realize: r.realize, entry: r.entry.clone() };
    v.kernels[k].threads = presk_kernel::Threads::At { depth: 0, workers: 1 };
    let mut copy = v.kernels[k].clone();
    copy.threads = presk_kernel::Threads::At { depth: 0, workers: 3 };
    assert_eq!(presk_export_c::kernel_fn(&v.kernels[k], "K").unwrap(), presk_export_c::kernel_fn(&copy, "K").unwrap(), "the copy prints alike");
    v.kernels.push(copy);
    let k2 = v.kernels.len() - 1;
    for it in 1..fold.iterations.len() {
        for i in in_iter(it) {
            if let Op::Launch { kernel, .. } = &mut v.rec.ops[i]
                && *kernel == k
            {
                *kernel = k2;
            }
        }
    }
    let blocks = |st: &Structured| st.funcs.iter().filter(|f| f.form.func == "Block.apply").count();
    let before = host::structure(&v.kernels, &r.rec, 1).unwrap();
    let after = host::structure(&v.kernels, &v.rec, 1).unwrap();
    assert_eq!(blocks(&before), 1, "undivided, one function for the three blocks");
    assert_eq!(blocks(&after), 2, "the first block apart, the other two a loop");
    assert_eq!(before.kernels, after.kernels, "the copy is the same C function");
    let got = run_every_way(&v, Host::Structured(&after), &scratch("divided-otherwise"), &|_| {});
    assert!(got.is_empty(), "{}", got.join("; "));
}
