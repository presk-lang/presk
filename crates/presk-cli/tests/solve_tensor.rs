//! `presk solve` on an entry whose result is a tensor (plan 0.12.0 step 4,
//! its question 5): the budget bounds the output's largest elementwise gap,
//! relative to its largest magnitude (`--budget`) or absolute
//! (`--budget-abs`); the sites are ranked by one reverse sweep seeded with
//! ones, and the measurement decides. Until 0.12.0 it refused such an entry
//! before computing anything (D101).

use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn presk(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_presk")).current_dir(root()).args(args).env("NO_COLOR", "1").output().expect("running presk")
}

#[test]
fn a_tensor_valued_entry_is_solved_under_a_budget_on_its_largest_gap() {
    const FILE: &str = "examples/tests/data/logits.psk";
    let dir = std::env::temp_dir().join(format!("presk-solve-tensor-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let env = dir.join("env.json");
    let cal = presk(&["calibrate", "--precision", FILE, "--entry", "logits", "--out", env.to_str().unwrap()]);
    assert!(cal.status.success(), "{}", String::from_utf8_lossy(&cal.stderr));
    for (how, says) in [(["--budget", "1e-3"], "relative on the largest elementwise gap of its 8 outputs"), (["--budget-abs", "1e-4"], "1.000e-4 absolute on the largest elementwise gap of its 8 outputs")] {
        let out = dir.join(format!("solved{}.json", how[0]));
        let solve = presk(&["solve", FILE, "--precision", env.to_str().unwrap(), how[0], how[1], "--out", out.to_str().unwrap()]);
        let text = String::from_utf8_lossy(&solve.stdout);
        assert!(solve.status.success(), "{}", String::from_utf8_lossy(&solve.stderr));
        assert!(text.contains(says), "{text}");
        assert!(text.contains("inside the budget"), "{text}");
        assert!(out.exists());
    }
    // One budget or the other.
    let both = presk(&["solve", FILE, "--precision", env.to_str().unwrap(), "--budget", "1e-3", "--budget-abs", "1e-4"]);
    assert_eq!(both.status.code(), Some(2));
    std::fs::remove_dir_all(&dir).ok();
}
