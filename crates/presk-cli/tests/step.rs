//! `presk calibrate --step` and `presk solve --step` under `--remat`, through
//! the binary (D93): the step's functions are the `.fwd`s and `.bwd`s
//! storing and the `.vjp`s recomputing, so an envelope records the policy it
//! was calibrated under, and a solve under the other one is refused.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn presk(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_presk")).current_dir(root()).args(args).env("NO_COLOR", "1").output().expect("running presk")
}

#[test]
fn a_step_is_solved_under_the_policy_it_was_calibrated_under() {
    let d = std::env::temp_dir().join(format!("presk-step-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    // A checkpoint of the tutorial's regression: the step is measured from trained weights.
    let ckpt = d.join("ckpt");
    let o = presk(&["train", "tutorial/07_train.psk", "--steps", "20", "--backend", "native", "--ckpt", ckpt.to_str().unwrap()]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let weights = ckpt.join("final.psk.d");
    for calibrated in ["store", "recompute"] {
        let env = d.join(format!("{calibrated}.json"));
        let o = presk(&["calibrate", "--step", "tutorial/07_train.psk", "--weights", weights.to_str().unwrap(), "--warmup", "2", "--samples", "4", "--remat", calibrated, "--out", env.to_str().unwrap()]);
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        let text = std::fs::read_to_string(&env).unwrap();
        assert!(text.contains(&format!("\"remat\": \"{calibrated}\"")), "the envelope records its policy");
        assert_eq!(text.contains(".bwd/%"), calibrated == "store", "the sites are the policy's functions");
        for solved in ["store", "recompute"] {
            let out = d.join(format!("{calibrated}-{solved}.json"));
            let o = presk(&["solve", "--step", "tutorial/07_train.psk", "--precision", env.to_str().unwrap(), "--budget", "1e-3", "--remat", solved, "--out", out.to_str().unwrap()]);
            let e = String::from_utf8_lossy(&o.stderr);
            if solved == calibrated {
                assert!(o.status.success(), "{e}");
            } else {
                assert_eq!(o.status.code(), Some(2), "{e}");
                assert!(e.contains(&format!("calibrated under --remat {calibrated}")), "{e}");
            }
        }
    }
    let _ = std::fs::remove_dir_all(&d);
}
