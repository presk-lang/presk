//! The gap of a realization that overflows where the reference does not
//! (plan 0.10.0 step 8; the language reference's chapter 4, finding b): an
//! element whose difference is not finite makes the gap +∞, as the
//! reference's definition of the gap does, for `presk check --precision
//! --measure` and `presk solve`, which therefore refuses `f32` for the
//! overflowing group here. Until
//! step 8 such an element was left out: the gap read 0, and `presk solve`
//! accepted the realization that overflows.

use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn presk(args: &[&str]) -> String {
    let o = Command::new(env!("CARGO_BIN_EXE_presk")).current_dir(root()).args(args).env("NO_COLOR", "1").output().expect("running presk");
    assert!(o.status.success(), "{args:?}: {}", String::from_utf8_lossy(&o.stderr));
    format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
}

#[test]
fn a_non_finite_difference_makes_the_gap_infinite() {
    const FILE: &str = "examples/tests/data/overflow.psk";
    let dir = std::env::temp_dir().join(format!("presk-gap-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (env, solved) = (dir.join("env.json"), dir.join("solved.json"));
    presk(&["calibrate", "--precision", FILE, "--entry", "f", "--out", env.to_str().unwrap()]);
    // The exact mean is finite, the `f32` one +∞.
    assert!(presk(&["run", FILE, "--entry", "f", "--backend", "native", "--realize", "f32"]).contains("= inf"));
    let check = presk(&["check", "--precision", env.to_str().unwrap(), FILE, "--realize", "f32", "--measure"]);
    assert!(check.lines().any(|l| l.trim_start().starts_with("measured") && l.contains(" inf ")), "{check}");
    let f64_check = presk(&["check", "--precision", env.to_str().unwrap(), FILE, "--realize", "f64", "--measure"]);
    assert!(!f64_check.contains(" inf "), "{f64_check}");
    // The solver measures its proposal, which overflows, and widens it: the
    // group of `exp` (root `f/%4`, `mul` its member) goes back to `f64`.
    // Since 0.11.0 step 2b (D112) the round widens the decision whose share,
    // its member's with its own, is largest, which is that group, and the
    // constant `real 120`, exact in `f32`, may stay narrowed; until then the
    // round widened every site, the member's type among them.
    let solve = presk(&["solve", FILE, "--precision", env.to_str().unwrap(), "--budget", "1e-3", "--out", solved.to_str().unwrap(), "--sites", "0"]);
    assert!(solve.contains("inside the budget") && solve.contains("widening round"), "{solve}");
    assert!(solve.lines().any(|l| l.trim_start().starts_with("measured") && !l.contains("inf")), "{solve}");
    let r = std::fs::read_to_string(&solved).unwrap();
    assert!(!r.contains("\"f/%4\"") && !r.contains("\"f/%3\""), "{r}");
    std::fs::remove_dir_all(&dir).ok();
}
