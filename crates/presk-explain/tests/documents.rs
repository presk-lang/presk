//! The documents of `presk explain` on the examples of `tests/examples/`
//! and the tutorial's `07_train.psk`, as snapshots (plan 0.11.0 step 3): a
//! change to any stage shows in them as a diff. The profile is the frozen
//! X925 model's (`ra4-model-0.8.0`, D71) without its calibration samples,
//! which only a refit reads (`tests/data/README.md`).

use std::path::PathBuf;

use presk_driver::Collect;
use presk_explain::{Options, Target, explain};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The document of `file` at `entry`, its paths as the test names them.
fn document(name: &str, file: &str, entry: &str, dims: &[&str], consts: &[&str], how: (Option<f64>, Option<&str>), targets: &[Target]) -> String {
    let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("explain").join(name);
    let o = Options {
        file: PathBuf::from("../..").join(file),
        entry: entry.into(),
        dims: dims.iter().map(|d| d.to_string()).collect(),
        consts: consts.iter().map(|c| c.to_string()).collect(),
        budget: how.0,
        realize: how.1.map(String::from),
        plan: None,
        profile: Some(PathBuf::from("tests/data/x925-frozen.json")),
        threads: Some(10),
        remat: presk_core::ad::Remat::Store,
        opt: false,
        weights: None,
        seed: 0,
        bits: 128,
        envelope: None,
        no_exact: false,
        targets: targets.to_vec(),
        measure: false,
        sites: Vec::new(),
        out: out.clone(),
    };
    assert!(root().join(file).exists());
    let mut report = Collect::default();
    let e = explain(&o, &mut report).unwrap_or_else(|s| panic!("{name}: stopped with {s:?}: {:?}", report.items));
    std::fs::read_to_string(e.document).unwrap()
}

#[test]
fn the_mlp_under_a_budget() {
    insta::assert_snapshot!("mlp", document("mlp", "tests/examples/mlp/mlp.psk", "mse", &["D=4", "B=2"], &[], (Some(1e-3), None), &[Target::Native]));
}

#[test]
fn the_layer_norm_of_a_module_entry() {
    insta::assert_snapshot!("layernorm", document("layernorm", "tests/examples/layernorm/layernorm.psk", "LayerNorm{8}.apply", &["batch=3"], &[], (None, Some("f32")), &[Target::Native]));
}

#[test]
fn the_minigpt_in_f32() {
    insta::assert_snapshot!("minigpt", document("minigpt", "tests/examples/minigpt/minigpt.psk", "MiniGPT.apply", &["B=2", "T=8"], &["V=64", "D=16", "H=2", "L=2"], (None, Some("f32")), &[Target::Native]));
}

#[test]
fn the_tutorial_loss_under_a_budget_to_every_target() {
    insta::assert_snapshot!("tutorial", document("tutorial", "tutorial/07_train.psk", "loss", &["B=4"], &[], (Some(1e-3), None), &[Target::Native, Target::C, Target::Pytorch]));
}
