//! `presk plan --mode training` (plan 0.16.0 step 5, D164): the plan of a
//! training run is planned on the program with its masks. Since D160 every
//! command but a training run lowers the inference reading, which has no
//! `dropout_mask` and so numbers its nests otherwise; a plan of that program
//! named a `let` that `presk train`'s program fuses into another nest, and
//! the run refused it (`E7002`).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn presk(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_presk")).current_dir(root()).args(args).env("NO_COLOR", "1").output().expect("running presk")
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn frozen_profile() -> PathBuf {
    let p = std::env::var("RA4_FROZEN").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap()).join("presk-data/ra4-model-0.8.0/profile.json"));
    assert!(p.exists(), "{}: the frozen profile is absent (RA4_FROZEN names another)", p.display());
    p
}

fn plan(mode: &str, out: &Path) -> serde_json::Map<String, serde_json::Value> {
    let profile = frozen_profile();
    let o = presk(&["plan", "tests/train/dropout_reg_grad.psk", "--entry", "loss_grad", "--dim", "B=16", "--realize", "f32", "--mode", mode, "--profile", profile.to_str().unwrap(), "--out", out.to_str().unwrap()]);
    assert!(o.status.success(), "presk plan --mode {mode}: {}", stderr(&o));
    let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(out).unwrap()).unwrap();
    json["layout"].as_object().cloned().unwrap_or_default()
}

#[test]
fn a_training_run_takes_the_plan_of_its_own_program() {
    let d = std::env::temp_dir().join(format!("presk-plan-mode-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let training = plan("training", &d.join("training.json"));
    let inference = plan("inference", &d.join("inference.json"));
    assert!(!training.is_empty(), "the training plan states no layout");
    let (t, i): (Vec<_>, Vec<_>) = (training.keys().collect(), inference.keys().collect());
    assert_ne!(t, i, "the two modes' programs name the same nests: the masks moved nothing");
    let run = |p: &Path| {
        let plan = format!("@{}", p.display());
        let ckpt = d.join("ckpt");
        let _ = std::fs::remove_dir_all(&ckpt);
        presk(&["train", "tests/train/dropout_reg.psk", "--steps", "2", "--backend", "native", "--plan", &plan, "--ckpt", ckpt.to_str().unwrap()])
    };
    let o = run(&d.join("training.json"));
    assert!(o.status.success(), "presk train under the training plan: {}", stderr(&o));
    let _ = std::fs::remove_dir_all(&d);
}
