//! The interpreter is the oracle: every golden of `tests/run` and the
//! training golden of `presk-train` must be reproduced by the rewritten
//! program. The snapshots are the existing files, so a difference is a
//! rewrite that changed a value (or its axis names).

mod common;

use common::{optimize, prepare, repo_root};
use presk_interp::RunOptions;
use presk_rt::{InterpBackend, Dtype, Realization};
use presk_train::{TrainOptions, prepare_training, train};

fn run_opt(rel: &str, consts: &[(&str, u64)], entry: &str, dims: &[(&str, u64)], bits: u32) -> String {
    run_with(rel, consts, entry, dims, bits, None)
}

/// A plausible machine profile (the fitted constants of a laptop), for the
/// roofline extraction without measuring anything.
fn synthetic_profile() -> presk_opt::Profile {
    presk_opt::Profile { version: 1, dtype: "f64".into(), machine: "test".into(), pi: 2.0e9, beta: 1.0e10, lambda: 2.0e-7, spearman: Default::default(), kernels: vec![], others: Default::default(), machine_desc: None, layout: None }
}

fn run_with(rel: &str, consts: &[(&str, u64)], entry: &str, dims: &[(&str, u64)], bits: u32, profile: Option<presk_opt::Profile>) -> String {
    let (prog, _, mut core) = prepare(rel, consts);
    let opts = presk_opt::Options { profile, dims: dims.iter().map(|(n, v)| (n.to_string(), *v)).collect(), ..Default::default() };
    let report = presk_opt::optimize(&mut core, &prog, &opts);
    presk_core::verify(&core).unwrap();
    assert!(report.lets_after() <= report.lets_before(), "{}", report.summary());
    let opts = RunOptions { entry: entry.into(), dims: dims.iter().map(|(n, v)| (n.to_string(), *v)).collect(), seed: 0, bits, digits: 12 };
    let out = presk_interp::run(&core, &prog, &opts).unwrap_or_else(|e| panic!("{rel} --entry {entry}: {}", e.message));
    format!("{} = {}\n", presk_interp::run::describe(&out.result), presk_interp::run::render(&out.result, 12, 0))
}

fn golden(dir: &str, name: &str, text: String) {
    let mut settings = insta::Settings::clone_current();
    settings.set_snapshot_path(repo_root().join(dir));
    settings.set_prepend_module_to_snapshot(false);
    settings.bind(|| insta::assert_snapshot!(name, text));
}

const TINY: &[(&str, u64)] = &[("V", 64), ("D", 16), ("H", 2), ("L", 2)];

#[test]
fn run_goldens_unchanged() {
    golden("tests/run", "mlp_mse", run_opt("examples/spec/mlp/mlp.psk", &[], "mse", &[("D", 4), ("B", 2)], 256));
    golden("tests/run", "mlp_mse_grad", run_opt("examples/spec/mlp/mlp.psk", &[], "mse_grad", &[("D", 4), ("B", 2)], 256));
    golden("tests/run", "layernorm_apply", run_opt("examples/spec/layernorm/layernorm.psk", &[], "LayerNorm{8}.apply", &[("batch", 3)], 256));
    golden("tests/run", "rmsnorm_apply", run_opt("examples/spec/layernorm/layernorm.psk", &[], "RMSNorm{8}.apply", &[("batch", 3)], 256));
    golden("tests/run", "minigpt_forward", run_opt("examples/spec/minigpt/minigpt.psk", TINY, "MiniGPT.apply", &[("B", 2), ("T", 8)], 256));
    golden("tests/run", "minigpt_loss", run_opt("examples/spec/minigpt/train_minigpt.psk", TINY, "loss", &[("B", 2), ("T", 8)], 256));
    golden("tests/run", "minigpt_next_token", run_opt("examples/spec/minigpt/serve_minigpt.psk", TINY, "next_token", &[("B", 2), ("T", 8)], 256));
    golden("tests/run", "stack_grad", run_opt("examples/tests/data/stack_grad.psk", &[], "loss_grad", &[("B", 2)], 512));
}

/// The same goldens when the extraction is decided by a roofline profile.
#[test]
fn run_goldens_unchanged_with_profile() {
    let p = || Some(synthetic_profile());
    golden("tests/run", "mlp_mse_grad", run_with("examples/spec/mlp/mlp.psk", &[], "mse_grad", &[("D", 4), ("B", 2)], 256, p()));
    golden("tests/run", "layernorm_apply", run_with("examples/spec/layernorm/layernorm.psk", &[], "LayerNorm{8}.apply", &[("batch", 3)], 256, p()));
    golden("tests/run", "minigpt_loss", run_with("examples/spec/minigpt/train_minigpt.psk", TINY, "loss", &[("B", 2), ("T", 8)], 256, p()));
    golden("tests/run", "minigpt_next_token", run_with("examples/spec/minigpt/serve_minigpt.psk", TINY, "next_token", &[("B", 2), ("T", 8)], 256, p()));
}

/// The regression of `presk-train` still converges with the rewritten
/// objective, adjoint and optimizer.
#[test]
fn regression_converges_after_rewriting() {
    let (prog, typed, mut core) = prepare("examples/tests/train/mlp_regression.psk", &[]);
    let fns = prepare_training(&mut core, &prog, &typed).unwrap();
    let report = optimize(&mut core, &prog);
    assert!(report.lets_after() < report.lets_before(), "{}", report.summary());
    let opts = TrainOptions { data: None, steps: Some(200), epochs: None, seed: 0, bits: 256, realize: Realization::uniform(Dtype::F64), ckpt: None, checkpoints: false };
    let mut backend = InterpBackend::new(&core, &prog, 256);
    let out = train(&mut backend, &core, &prog, &typed, &fns, &opts, |_| {}).unwrap();
    assert!(out.logs.last().unwrap().loss < 1e-3, "final loss {}", out.logs.last().unwrap().loss);
}

/// The twenty-step MiniGPT log of `presk-train` is reproduced to the printed
/// digits (the same snapshot file).
#[test]
fn minigpt_train_golden_unchanged() {
    let (prog, typed, mut core) = prepare("examples/spec/minigpt/train_minigpt.psk", TINY);
    let fns = prepare_training(&mut core, &prog, &typed).unwrap();
    optimize(&mut core, &prog);
    let ckpt = std::env::temp_dir().join(format!("presk-opt-tests-{}", std::process::id()));
    let opts = TrainOptions { data: None, steps: Some(20), epochs: None, seed: 0, bits: 64, realize: Realization::uniform(Dtype::F64), ckpt: Some(ckpt), checkpoints: false };
    let mut backend = InterpBackend::new(&core, &prog, 64);
    let out = train(&mut backend, &core, &prog, &typed, &fns, &opts, |_| {}).unwrap();
    let text: String = out.logs.iter().map(|l| format!("step {:>2} epoch {} loss {:.9} lr {:.3e} |g| {:.6}\n", l.step, l.epoch, l.loss, l.lr, l.grad_norm)).collect();
    golden("crates/presk-train/tests/snapshots", "train__minigpt_train_20", text);
}
