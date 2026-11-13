//! NB0 for the LLVM backend (plan 0.17.0 step 3, D176): the native
//! backend's bits, bit for bit, on the C export's test set
//! (`examples/tests/plan/entries.txt`) in `f64`, `f32` and `bf16`, MiniGPT's entries
//! also under the mixed realization, each on the parameters `presk run`
//! synthesizes and with their zeros drawn, on one worker and on four; the
//! test set under `examples/tests/plan/default.json`; and the attention under
//! `examples/tests/plan/attention.json`, which states a tile in lanes, a k-block, a
//! split over ten workers and packing.

use std::path::{Path, PathBuf};
use std::sync::Once;

use presk_interp::RunOptions;
use presk_rt::{Backend, Buf, Plan, Realization, Value};
use presk_syntax::diag::render_all;
use presk_syntax::{SourceMap, parse_file};
use presk_types::check;
use presk_types::elab::{self, ElabProgram, Options};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn chdir_root() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| std::env::set_current_dir(root()).unwrap());
}

fn prepare(rel: &str, consts: &[(String, u64)]) -> (ElabProgram, presk_core::Program) {
    chdir_root();
    let mut sm = SourceMap::new();
    let id = sm.load(Path::new(rel)).unwrap();
    let (program, diags) = parse_file(&sm, id);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let opts = Options { const_overrides: consts.iter().cloned().collect() };
    let (prog, diags) = elab::elaborate(&mut sm, id, program, &opts);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (typed, diags) = check::check(&sm, &prog);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (mut core, diags) = presk_core::lower(&prog, &typed);
    assert!(diags.is_empty(), "{}", render_all(&diags, &sm, false));
    let diags = presk_core::ad::transform(&mut core);
    assert!(diags.is_empty(), "{}", render_all(&diags, &sm, false));
    (prog, core)
}

/// An entry of the test set: the file, the entry, its constants and its
/// dimensions.
struct Case {
    file: String,
    entry: String,
    consts: Vec<(String, u64)>,
    dims: Vec<(String, u64)>,
}

fn pairs(args: &[&str], flag: &str) -> Vec<(String, u64)> {
    args.windows(2)
        .filter(|w| w[0] == flag)
        .map(|w| {
            let (n, v) = w[1].split_once('=').unwrap();
            (n.to_string(), v.parse().unwrap())
        })
        .collect()
}

fn test_set() -> Vec<Case> {
    let text = std::fs::read_to_string(root().join("examples/tests/plan/entries.txt")).unwrap();
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut p = l.splitn(3, '|');
            let (file, entry, args) = (p.next().unwrap(), p.next().unwrap(), p.next().unwrap_or(""));
            let args: Vec<&str> = args.split_whitespace().collect();
            Case { file: file.into(), entry: entry.into(), consts: pairs(&args, "--const"), dims: pairs(&args, "--dim") }
        })
        .collect()
}

fn leaf_bits(v: &Value, out: &mut Vec<u64>) {
    match v {
        Value::Real(t) | Value::Idx { t, .. } => match &t.data {
            Buf::F64(x) => out.extend(x.iter().map(|y| y.to_bits())),
            Buf::F32(x) => out.extend(x.iter().map(|y| y.to_bits() as u64)),
            Buf::Bf16(x) => out.extend(x.iter().map(|y| *y as u64)),
            Buf::I64(x) => out.extend(x.iter().map(|y| *y as u64)),
            other => panic!("an output of kind {other:?}"),
        },
        Value::Tuple(xs) | Value::Stacked(xs) => xs.iter().for_each(|x| leaf_bits(x, out)),
        Value::Record(xs) => xs.iter().for_each(|(_, x)| leaf_bits(x, out)),
        Value::Nat(n) => out.push(*n),
        Value::Int(i) => out.push(*i as u64),
        Value::F64(x) => out.push(x.to_bits()),
        Value::Bool(b) => out.push(*b as u64),
        Value::Key(_) => {}
    }
}

/// `v` with every real that is exactly zero replaced by a small value from
/// a fixed sequence, as the C export's test draws a checkpoint's zeros.
fn draw_zeros(v: &Value, state: &mut u64) -> Value {
    let mut draw = || {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        ((*state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.2
    };
    match v {
        Value::Real(t) => match &t.data {
            Buf::F64(x) => {
                let data: Vec<f64> = x.iter().map(|&y| if y == 0.0 { draw() } else { y }).collect();
                Value::Real(presk_rt::Tensor { axes: t.axes.clone(), data: Buf::F64(data.into()) })
            }
            _ => v.clone(),
        },
        Value::Tuple(xs) => Value::Tuple(xs.iter().map(|x| draw_zeros(x, state)).collect()),
        Value::Stacked(xs) => Value::Stacked(xs.iter().map(|x| draw_zeros(x, state)).collect()),
        Value::Record(xs) => Value::Record(xs.iter().map(|(n, x)| (n.clone(), draw_zeros(x, state))).collect()),
        other => other.clone(),
    }
}

/// The outputs' bits of `case` on the native backend and on the LLVM
/// backend, under `plan`, on `threads` workers, the first argument's zeros
/// drawn or not.
fn both(case: &Case, prog: &ElabProgram, core: &presk_core::Program, plan: &Plan, threads: usize, drawn: bool) -> (Vec<u64>, Vec<u64>) {
    let opts = RunOptions { entry: case.entry.clone(), dims: case.dims.iter().cloned().collect(), seed: 0, bits: 64, digits: 17 };
    let (func, fdims, inputs) = presk_interp::run::prepare_run(core, prog, &opts).unwrap();
    let mut args: Vec<Value> = inputs.iter().map(|(_, v)| Value::from_interp(v).unwrap()).collect();
    if drawn && !args.is_empty() {
        args[0] = draw_zeros(&args[0], &mut 0x9e37_79b9_7f4a_7c15);
    }
    let r = &plan.realization;
    let args = r.realize_args(&args).unwrap();
    let run = |mut b: presk_kernel::KernelBackend| -> Vec<u64> {
        b.set_plan(plan.clone());
        let p = Backend::prepare(&mut b, func, &fdims, r).unwrap();
        let out = Backend::call(&mut b, &p, args.clone()).unwrap();
        let mut bits = Vec::new();
        leaf_bits(&out, &mut bits);
        bits
    };
    let native = run(presk_cranelift::native_backend_with_threads(core, prog, r, threads).unwrap());
    let llvm = run(presk_llvm::llvm_backend_with_threads(core, prog, r, threads).unwrap());
    (native, llvm)
}

#[test]
fn the_test_set_computes_the_native_backends_bits() {
    if let Err(m) = presk_llvm::available() {
        eprintln!("skipped: {m}");
        return;
    }
    let mut programs = 0;
    for case in test_set() {
        let (prog, core) = prepare(&case.file, &case.consts);
        let minigpt = case.file.contains("minigpt");
        for realize in ["f64", "f32", "bf16", "@examples/tests/plan/mixed-realization.json"] {
            if realize.starts_with('@') && !minigpt {
                continue;
            }
            let plan = Plan::from_realization(Realization::parse(realize).unwrap());
            for threads in [1, 4] {
                for drawn in [false, true] {
                    let (native, llvm) = both(&case, &prog, &core, &plan, threads, drawn);
                    assert!(!native.is_empty(), "{} {realize}: no output", case.entry);
                    assert!(native == llvm, "{} {realize}, {threads} workers, zeros drawn {drawn}: the LLVM backend's bits are not the native backend's", case.entry);
                    programs += 1;
                }
            }
        }
    }
    eprintln!("{programs} runs, the same bits on both backends");
}

#[test]
fn the_plans_of_the_test_directory_compute_the_native_backends_bits() {
    if let Err(m) = presk_llvm::available() {
        eprintln!("skipped: {m}");
        return;
    }
    // The default plan on every entry of the test set, and the attention's
    // plan (a tiled contraction in lanes, k-blocked, split and packed, a
    // two-pass softmax) at a T its tile divides, on its ten workers.
    let mut runs: Vec<(Case, &str, usize)> = test_set().into_iter().map(|c| (c, "@examples/tests/plan/default.json", 4)).collect();
    runs.push((
        Case {
            file: "examples/corpus/nlp/05_gpt.psk".into(),
            entry: "SelfAttention.apply".into(),
            consts: vec![("V".into(), 92), ("D".into(), 64), ("H".into(), 2), ("L".into(), 1)],
            dims: vec![("B".into(), 2), ("T".into(), 16)],
        },
        "@examples/tests/plan/attention.json",
        10,
    ));
    for (case, spec, threads) in runs {
        let (prog, core) = prepare(&case.file, &case.consts);
        let plan = Plan::parse(spec).unwrap();
        let (native, llvm) = both(&case, &prog, &core, &plan, threads, true);
        assert!(native == llvm, "{} under {spec}, {threads} workers: the LLVM backend's bits are not the native backend's", case.entry);
    }
}
