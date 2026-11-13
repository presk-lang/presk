//! A parameter's own storage type, `params` (plan 0.12.0, D119, D120): the
//! record's conditions are checked where the parameters are loaded, on every
//! executor, and `presk train` refuses a realization that types parameters.

use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn presk(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_presk")).current_dir(root()).args(args).env("NO_COLOR", "1").output().expect("running presk")
}

const FILE: &str = "examples/tutorial/07_train.psk";

/// A realization file with `params` as given, in a directory of its own.
fn record(dir: &Path, name: &str, params: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, format!(r#"{{"default":"f64","order":{{"kind":"sequential"}},"params":{params}}}"#)).unwrap();
    format!("@{}", p.display())
}

#[test]
fn the_records_conditions_are_checked_on_every_executor() {
    let dir = std::env::temp_dir().join(format!("presk-params-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cases = [
        (record(&dir, "typo.json", r#"{"norm.gamma":"bf16"}"#), "`norm.gamma` in its own type, which is not a parameter of the model (it has blocks.norm.γ"),
        (record(&dir, "wide.json", r#"{"w":"f64"}"#), "a parameter's own type is `f32` or `bf16`"),
    ];
    for (realize, says) in &cases {
        for backend in ["interp", "kernel", "native"] {
            let out = presk(&["run", FILE, "--entry", "loss", "--dim", "B=4", "--backend", backend, "--realize", realize]);
            let err = String::from_utf8_lossy(&out.stderr);
            assert_eq!(out.status.code(), Some(1), "{backend}: {err}");
            assert!(err.contains("E7001") && err.contains(says), "{backend}: {err}");
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}
