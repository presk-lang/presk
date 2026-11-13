//! Two of the defects D110 found in the CLI, fixed in plan 0.11.0's step 7
//! (D117): `presk dump --stage elab` and `--stage core` ignored `--const`,
//! and a missing `--weights` file was named twice in `presk solve`'s
//! message.

use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn presk(args: &[&str]) -> (bool, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_presk")).current_dir(root()).args(args).env("NO_COLOR", "1").output().expect("running presk");
    (o.status.success(), format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr)))
}

#[test]
fn the_dumps_bind_the_constants() {
    let (ok, elab) = presk(&["dump", "examples/tutorial/06_tower.psk", "--stage", "elab", "--const", "L=1"]);
    assert!(ok && elab.contains("const L = 1  (--const)") && elab.contains("stack{1}(Block)"), "{elab}");
    let (ok, core) = presk(&["dump", "examples/tutorial/06_tower.psk", "--stage", "core", "--const", "L=1"]);
    assert!(ok && core.contains("fold 1 Block.apply"), "{core}");
}

#[test]
fn a_missing_weights_file_is_named_once() {
    let dir = std::env::temp_dir().join(format!("presk-weights-once-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (env, missing) = (dir.join("env.json"), dir.join("missing.psk.d"));
    let (ok, _) = presk(&["calibrate", "--precision", "examples/tutorial/07_train.psk", "--entry", "loss", "--dim", "B=4", "--out", env.to_str().unwrap()]);
    assert!(ok);
    let (ok, solve) = presk(&["solve", "examples/tutorial/07_train.psk", "--precision", env.to_str().unwrap(), "--budget", "1e-3", "--weights", missing.to_str().unwrap()]);
    assert!(!ok && solve.matches(missing.to_str().unwrap()).count() == 1, "{solve}");
    std::fs::remove_dir_all(&dir).ok();
}
