//! `W0004` (plan 0.10.0 step 4b, D94), through the binary: under `--remat
//! store` a realization that types a `.vjp`, by function or by site, types
//! nothing of a stored gradient, whose `.vjp` only calls `.fwd` and `.bwd`.
//! A warning and not an error: the run goes on, and under `--remat
//! recompute`, where the `.vjp`s run their sweeps, nothing is said, nor of
//! a `.vjp` whose `.fwd` or `.bwd` the realization names too.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("presk-remat-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn presk(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_presk")).current_dir(root()).args(args).env("NO_COLOR", "1").output().expect("running presk")
}

#[test]
fn a_realization_typing_a_stored_vjp_is_warned_of() {
    let d = scratch("w0004");
    let by_function = d.join("functions.json");
    std::fs::write(&by_function, r#"{"default": "f64", "order": {"kind": "sequential"}, "functions": {"loss.vjp": "bf16", "Reg.apply.vjp": "bf16", "Linear.apply": "bf16"}}"#).unwrap();
    let by_site = d.join("sites.json");
    std::fs::write(&by_site, r#"{"default": "f64", "order": {"kind": "sequential"}, "storage": {"loss.vjp/%5": "bf16", "loss/%5": "bf16"}}"#).unwrap();
    let train = |r: &Path, remat: &str| {
        let spec = format!("@{}", r.display());
        let ckpt = d.join(format!("ckpt-{remat}"));
        let o = presk(&["train", "examples/tests/train/mlp_regression.psk", "--steps", "2", "--backend", "native", "--realize", &spec, "--remat", remat, "--ckpt", ckpt.to_str().unwrap()]);
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        String::from_utf8_lossy(&o.stderr).into_owned()
    };
    let e = train(&by_function, "store");
    assert!(e.contains("W0004") && e.contains("`loss.vjp`") && e.contains("`Reg.apply.vjp`"), "{e}");
    assert!(!e.contains("`Linear.apply`,") && e.matches("W0004").count() == 2, "a forward function is typed, not warned of: {e}");
    assert!(e.contains("--remat recompute"), "{e}");
    let e = train(&by_site, "store");
    assert!(e.contains("W0004") && e.contains("`loss.vjp`") && e.matches("W0004").count() == 1, "a site of a stored `.vjp`: {e}");
    for r in [&by_function, &by_site] {
        let e = train(r, "recompute");
        assert!(!e.contains("W0004"), "recomputing, the `.vjp`s run their sweeps: {e}");
    }
    // A realization that names the `.vjp` beside its `.bwd`, as `presk
    // solve --step` writes one under `store`, was written for this program.
    let both = d.join("both.json");
    std::fs::write(&both, r#"{"default": "f64", "order": {"kind": "sequential"}, "functions": {"loss.vjp": "bf16", "loss.bwd": "bf16", "Reg.apply.vjp": "bf16"}}"#).unwrap();
    let e = train(&both, "store");
    assert!(e.matches("W0004").count() == 1 && e.contains("`Reg.apply.vjp`") && !e.contains("`loss.vjp`"), "{e}");
    let _ = std::fs::remove_dir_all(&d);
}
