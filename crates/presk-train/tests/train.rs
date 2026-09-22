//! `presk train` on the interpreter (0.2.0 step 5): a linear regression that
//! converges, the tiny MiniGPT for 20 steps at 64 bits (its loss log is a
//! golden), and the checkpoint round trip.

use std::path::{Path, PathBuf};
use std::sync::Once;

use presk_syntax::diag::render_all;
use presk_syntax::{SourceMap, parse_file};
use presk_rt::{Buf, InterpBackend, Dtype, Value, Realization};
use presk_train::{TrainOptions, load_params, prepare_training, save_params, train};
use presk_types::check::{self, TypedProgram};
use presk_types::elab::{self, ElabProgram, Options};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn chdir_root() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| std::env::set_current_dir(repo_root()).unwrap());
}

fn prepare(rel: &str, consts: &[(&str, u64)]) -> (ElabProgram, TypedProgram, presk_core::Program) {
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
    (prog, typed, core)
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("presk-train-tests-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// y = 2·x₀ − x₁ + 0.5 is learnt by a linear layer under AdamW in 200 steps.
#[test]
fn regression_converges_and_checkpoints_round_trip() {
    let (prog, typed, mut core) = prepare("tests/train/mlp_regression.psk", &[]);
    let ckpt = tmp("reg");
    let opts = TrainOptions { data: None, steps: Some(200), epochs: None, seed: 0, bits: 256, realize: Realization::uniform(Dtype::F64), ckpt: Some(ckpt.clone()), checkpoints: true };
    let fns = prepare_training(&mut core, &prog, &typed).unwrap();
    let mut backend = InterpBackend::new(&core, &prog, 256, 0);
    let out = train(&mut backend, &core, &prog, &typed, &fns, &opts, |_| {}).unwrap();
    assert_eq!(out.logs.len(), 200);
    assert_eq!(out.total, 200);
    let last = out.logs.last().unwrap();
    assert!(last.loss < 1e-3, "final loss {}", last.loss);
    assert!(out.logs[0].loss > last.loss);
    assert!((out.logs[0].lr - 0.05).abs() < 1e-12, "constant schedule");
    // The weights: w ≈ (2, −1), b ≈ 0.5.
    let l = out.theta.field("l").unwrap();
    let Value::Real(w) = l.field("w").unwrap() else { panic!() };
    let Value::Real(b) = l.field("b").unwrap() else { panic!() };
    assert!(matches!(w.data, Buf::Exact(_)), "the interpreter's values cross the seam exactly");
    let (w, b) = (w.data.to_f64(), b.data.to_f64());
    assert!((w[0] - 2.0).abs() < 0.05 && (w[1] + 1.0).abs() < 0.05, "w = {w:?}");
    assert!((b[0] - 0.5).abs() < 0.05);
    // Checkpoints: final.psk.d reloads to the same f64 values.
    let final_path = out.final_path.clone().unwrap();
    assert_eq!(final_path, ckpt.join("final.psk.d"));
    let back = load_params(&final_path, &out.theta).unwrap();
    let Value::Real(w2) = back.field("l").unwrap().field("w").unwrap() else { panic!() };
    assert_eq!(w2.data.to_f64(), w);
    let again = tmp("again.psk.d");
    save_params(&again, &out.model, &back).unwrap();
    assert_eq!(std::fs::read(&again).unwrap(), std::fs::read(&final_path).unwrap(), "saving a reloaded tree is byte-identical");
    let r = presk_data::Reader::open(&final_path).unwrap();
    assert_eq!(r.header().sample_ty, "params Reg");
    assert_eq!(r.len(), 1);
    assert_eq!(r.layout().unwrap_err().code, "E9001", "a params file needs the program");
}

/// The tiny MiniGPT of the plan's acceptance: 20 steps at 64 bits, the loss
/// decreases, and the log is a golden output.
#[test]
fn minigpt_twenty_steps() {
    let (prog, typed, mut core) = prepare("tests/examples/minigpt/train_minigpt.psk", &[("V", 64), ("D", 16), ("H", 2), ("L", 2)]);
    // The example's `checkpoint every 1000 steps to "ckpt/"` would write next to the
    // example file: redirect to a temporary directory.
    let ckpt = tmp("minigpt");
    let opts = TrainOptions { data: None, steps: Some(20), epochs: None, seed: 0, bits: 64, realize: Realization::uniform(Dtype::F64), ckpt: Some(ckpt.clone()), checkpoints: true };
    let fns = prepare_training(&mut core, &prog, &typed).unwrap();
    let mut backend = InterpBackend::new(&core, &prog, 64, 0);
    let out = train(&mut backend, &core, &prog, &typed, &fns, &opts, |_| {}).unwrap();
    assert_eq!(out.logs.len(), 20);
    assert!(out.logs[19].loss < out.logs[0].loss);
    assert_eq!(out.logs[15].epoch, 2, "15 batches of 32 in 510 windows: the second epoch starts at step 16");
    assert_eq!(out.final_path, Some(ckpt.join("final.psk.d")));
    assert!(out.checkpoints.is_empty(), "every 1000 steps: none within 20");
    let text: String = out.logs.iter().map(|l| format!("step {:>2} epoch {} loss {:.9} lr {:.3e} |g| {:.6}\n", l.step, l.epoch, l.loss, l.lr, l.grad_norm)).collect();
    insta::assert_snapshot!("minigpt_train_20", text);
}

/// A training run whose objective takes a per-step key (0.4.0 step 3):
/// `dropout` drops while training, the run is reproducible from its seed, and
/// the same model evaluated outside training is its own expectation.
#[test]
fn dropout_trains_from_a_per_step_key() {
    let (prog, typed, mut core) = prepare("tests/train/dropout_reg.psk", &[]);
    let fns = prepare_training(&mut core, &prog, &typed).unwrap();
    assert!(fns.key, "the objective takes a key between θ and the batch");
    let opts = TrainOptions { data: None, steps: Some(120), epochs: None, seed: 0, bits: 256, realize: Realization::uniform(Dtype::F64), ckpt: None, checkpoints: false };
    let run = |seed: u64| {
        let o = TrainOptions { seed, ..opts.clone() };
        let mut backend = InterpBackend::new(&core, &prog, 256, seed);
        train(&mut backend, &core, &prog, &typed, &fns, &o, |_| {}).unwrap()
    };
    let a = run(0);
    assert_eq!(a.logs.len(), 120);
    assert!(a.logs.last().unwrap().loss < a.logs[0].loss, "{} -> {}", a.logs[0].loss, a.logs.last().unwrap().loss);
    // Same seed, same masks, same trajectory; another seed, another one.
    let b = run(0);
    assert_eq!(a.logs.last().unwrap().loss, b.logs.last().unwrap().loss, "reproducible from the seed");
    let c = run(1);
    assert!(c.logs.last().unwrap().loss != a.logs.last().unwrap().loss, "another seed drops differently");
    // That dropping changes the objective at all, and that its gradient is
    // right while it does, are checked in `presk-interp` against finite
    // differences; here the point is that the loop threads a fresh key.
}

/// A quantised realization (plan 0.6.0 step 5a, D44) runs on the dequantised
/// parameters wherever an entry is run, and training refuses it: it applies
/// to trained weights, and a quantised forward pass inside a training step
/// would be quantisation-aware training.
#[test]
fn a_quantised_realization_runs_on_dequantised_weights() {
    use presk_rt::{AxisRef, Quant, QuantScheme};
    let (prog, typed, mut core) = prepare("tests/train/dropout_reg.psk", &[]);
    let fns = prepare_training(&mut core, &prog, &typed).unwrap();
    let mut r = Realization::uniform(Dtype::F64);
    for (path, bits) in [("l1.w", 8u8), ("l2.w", 4)] {
        r.quant.insert(path.into(), Quant { bits, over: AxisRef::Pos(-2), scheme: QuantScheme::Absmax });
    }
    let opts = presk_interp::RunOptions { entry: "loss".into(), dims: vec![("B".into(), 4)], seed: 3, bits: 128, digits: 17 };
    let run = |realize: &Realization| {
        let mut b = InterpBackend::new(&core, &prog, 128, 3);
        Value::from_interp(&presk_rt::run_with(&mut b, &core, &prog, &opts, realize).unwrap().result).unwrap()
    };
    let exact = run(&Realization::uniform(Dtype::F64));
    let quantised = run(&r);
    assert_ne!(exact, quantised, "the quant map took effect");

    // The same numbers as quantising the arguments by hand: the reference
    // path is `f32(q·s)` substituted for the parameters, and nothing else.
    let (func, fdims, inputs) = presk_interp::run::prepare_run(&core, &prog, &opts).unwrap();
    let mut args: Vec<Value> = inputs.iter().map(|(_, v)| Value::from_interp(v).unwrap()).collect();
    args[0] = r.quantized_params(&args[0]).unwrap();
    let mut b = InterpBackend::new(&core, &prog, 128, 3);
    let p = presk_rt::Backend::prepare(&mut b, func, &fdims, &r).unwrap();
    let by_hand = presk_rt::Backend::call(&mut b, &p, args).unwrap();
    assert_eq!(by_hand, quantised);

    // A misspelt parameter is refused rather than ignored.
    let mut bad = r.clone();
    bad.quant.insert("l3.w".into(), Quant { bits: 8, over: AxisRef::Pos(-2), scheme: QuantScheme::Absmax });
    let mut b = InterpBackend::new(&core, &prog, 128, 3);
    let Err(e) = presk_rt::run_with(&mut b, &core, &prog, &opts, &bad) else { panic!("a misspelt parameter ran") };
    assert_eq!(e.code, "E7001");

    let topts = TrainOptions { data: None, steps: Some(1), epochs: None, seed: 0, bits: 64, realize: r, ckpt: None, checkpoints: false };
    let mut b = InterpBackend::new(&core, &prog, 64, 0);
    let e = train(&mut b, &core, &prog, &typed, &fns, &topts, |_| {}).unwrap_err();
    assert!(e.message.contains("trained weights"), "{}", e.message);
}
