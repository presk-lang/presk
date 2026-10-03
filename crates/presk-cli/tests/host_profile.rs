//! `presk bench --host-profile` and `presk train --host-profile` through the
//! binary (plan 0.10.0 step 5, D95): the table of phases, the timers'
//! cost, the outputs compared, and a training run's weights the same bytes
//! with the flag and without.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn presk(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_presk")).current_dir(root()).args(args).env("NO_COLOR", "1").output().expect("running presk")
}

#[test]
fn bench_prints_the_host_by_phase_and_the_same_bits() {
    let o = presk(&["bench", "tests/plan/minigpt_grad.psk", "--entry", "loss_grad", "--const", "V=64", "--const", "D=16", "--const", "H=2", "--const", "L=2", "--dim", "B=2", "--dim", "T=8", "--realize", "f32", "--threads", "2", "--repeat", "2", "--host-profile"]);
    let out = String::from_utf8_lossy(&o.stdout);
    assert!(o.status.success(), "{out}{}", String::from_utf8_lossy(&o.stderr));
    assert!(out.contains("2 calls timed by phase and 2 not, alternated, after 1 warm-up"), "{out}");
    assert!(out.contains("the timers' cost"), "{out}");
    assert!(out.contains("-- the outputs of a call with the timers and of one without: the same bits"), "{out}");
    for key in ["kernels ", "find.key", "buffers.alloc", "views.slice", "copies.operands", "copies.as-tensor", "threads.start-join", "left.interpreter", "total "] {
        assert!(out.lines().any(|l| l.trim_start().starts_with(key)), "no row `{key}`:\n{out}");
    }
    // Refused with what it cannot mean.
    let o = presk(&["bench", "tests/plan/minigpt_grad.psk", "--entry", "loss_grad", "--host-profile", "--calibrate", "x.json"]);
    assert_eq!(o.status.code(), Some(2));
}

#[test]
fn train_with_the_timers_trains_the_same_weights() {
    let d = std::env::temp_dir().join(format!("presk-host-profile-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let mut weights = Vec::new();
    for flag in [false, true] {
        let ckpt = d.join(if flag { "with" } else { "without" });
        let mut args = vec!["train", "tutorial/07_train.psk", "--steps", "4", "--backend", "native", "--ckpt", ckpt.to_str().unwrap()];
        if flag {
            args.push("--host-profile");
        }
        let o = presk(&args);
        let e = String::from_utf8_lossy(&o.stderr);
        assert!(o.status.success(), "{e}");
        assert_eq!(e.contains("presk train --host-profile: the host by phase, the mean of steps 2 to 4"), flag, "{e}");
        weights.push(std::fs::read(ckpt.join("final.psk.d")).unwrap());
    }
    assert_eq!(weights[0], weights[1], "the timers moved the weights");
    let o = presk(&["train", "tutorial/07_train.psk", "--steps", "1", "--host-profile"]);
    assert_eq!(o.status.code(), Some(2), "the reference interpreter has no host to profile");
    let _ = std::fs::remove_dir_all(&d);
}
