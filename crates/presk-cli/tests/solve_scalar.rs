//! `presk solve` takes a scalar entry (D101): it ranks the sites by the
//! sensitivity `∂L/∂v` of a scalar `L`, as `presk dump --stage sensitivity`
//! does, and refuses any other entry before computing anything. Until then
//! the refusal came from the reverse sweep's first `.vjp`, after its forward
//! half, which on BERT-large's `squad` at `L = 2` was 39 minutes.

use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn presk(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_presk")).current_dir(root()).args(args).env("NO_COLOR", "1").output().expect("running presk")
}

#[test]
fn a_tensor_valued_entry_is_calibrated_and_not_solved() {
    const FILE: &str = "tests/data/logits.psk";
    let dir = std::env::temp_dir().join(format!("presk-solve-scalar-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let env = dir.join("env.json");
    let cal = presk(&["calibrate", "--precision", FILE, "--entry", "logits", "--out", env.to_str().unwrap()]);
    assert!(cal.status.success(), "{}", String::from_utf8_lossy(&cal.stderr));
    let solve = presk(&["solve", FILE, "--precision", env.to_str().unwrap(), "--budget", "1e-3"]);
    let err = String::from_utf8_lossy(&solve.stderr);
    assert_eq!(solve.status.code(), Some(2), "{err}");
    assert!(err.contains("`logits` must return a scalar `ℝ[]`"), "{err}");
    std::fs::remove_dir_all(&dir).ok();
}
