//! `presk bench --host-profile` and `presk train --host-profile` through the
//! binary (plan 0.10.0 step 5, D95): the table of phases, the timers'
//! cost, the outputs compared, and a training run's weights the same bytes
//! with the flag and without; `--launches`, each launch of a call listed
//! (plan 0.17.2 step 4, D200).

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
    let o = presk(&["bench", "examples/tests/plan/minigpt_grad.psk", "--entry", "loss_grad", "--const", "V=64", "--const", "D=16", "--const", "H=2", "--const", "L=2", "--dim", "B=2", "--dim", "T=8", "--realize", "f32", "--threads", "2", "--repeat", "2", "--host-profile"]);
    let out = String::from_utf8_lossy(&o.stdout);
    assert!(o.status.success(), "{out}{}", String::from_utf8_lossy(&o.stderr));
    assert!(out.contains("2 calls timed by phase and 2 not, alternated, after 1 warm-up"), "{out}");
    assert!(out.contains("the timers' cost"), "{out}");
    assert!(out.contains("-- the outputs of a call with the timers and of one without: the same bits"), "{out}");
    for key in ["kernels ", "find.key", "buffers.alloc", "views.slice", "copies.operands", "copies.as-tensor", "threads.start-join", "left.interpreter", "total "] {
        assert!(out.lines().any(|l| l.trim_start().starts_with(key)), "no row `{key}`:\n{out}");
    }
    // Without `--launches`, nothing per launch: the table ends where it did.
    assert!(out.lines().last().is_some_and(|l| l.contains("launches a call")), "{out}");
    assert!(!out.contains("--launches") && !out.contains("huge pages"), "{out}");
    // Refused with what it cannot mean.
    let o = presk(&["bench", "examples/tests/plan/minigpt_grad.psk", "--entry", "loss_grad", "--host-profile", "--calibrate", "x.json"]);
    assert_eq!(o.status.code(), Some(2));
    let o = presk(&["bench", "examples/tests/plan/minigpt_grad.psk", "--entry", "loss_grad", "--launches"]);
    assert_eq!(o.status.code(), Some(2), "--launches without --host-profile");
}

/// `--launches`: a row per launch of a call, in its order, as many as the
/// table counts a call; a buffer of 64 KiB or more listed under its launch
/// with its address and the address's residues; the same bits.
#[test]
fn launches_lists_each_launch_of_a_call() {
    let o = presk(&["bench", "examples/tests/plan/minigpt_grad.psk", "--entry", "loss_grad", "--const", "V=64", "--const", "D=16", "--const", "H=2", "--const", "L=2", "--dim", "B=2", "--dim", "T=8", "--realize", "f32", "--threads", "2", "--repeat", "2", "--host-profile", "--launches"]);
    let out = String::from_utf8_lossy(&o.stdout);
    assert!(o.status.success(), "{out}{}", String::from_utf8_lossy(&o.stderr));
    assert!(out.contains("-- the outputs of a call with the timers and of one without: the same bits"), "{out}");
    assert!(out.contains("-- transparent huge pages: "), "{out}");
    let per_call: usize = out.lines().find_map(|l| l.strip_prefix("-- ")?.split_once(" launches a call")?.0.parse().ok()).expect("the launches a call");
    let rows: Vec<&str> = out.lines().skip_while(|l| !l.starts_with("launch ")).skip(2).filter(|l| !l.starts_with("      ")).collect();
    assert_eq!(rows.len(), per_call, "{out}");
    for (i, r) in rows.iter().enumerate() {
        let f: Vec<&str> = r.split_whitespace().collect();
        assert_eq!(f[0], (i + 1).to_string(), "{r}");
        assert!(f[1].starts_with('k') && r.trim_end().ends_with('`'), "{r}");
    }
    // A buffer of 4 MiB: in0, in1 and out, each with its residues.
    let o = presk(&["bench", "examples/experiments/gemm_1024.psk", "--entry", "gemm", "--realize", "f32", "--threads", "1", "--repeat", "1", "--host-profile", "--launches"]);
    let out = String::from_utf8_lossy(&o.stdout);
    assert!(o.status.success(), "{out}{}", String::from_utf8_lossy(&o.stderr));
    let bufs: Vec<Vec<&str>> = out.lines().filter(|l| l.starts_with("        ")).map(|l| l.split_whitespace().collect::<Vec<_>>()).filter(|b| b[0] != "buffer").collect();
    assert_eq!(bufs.iter().map(|b| b[0]).collect::<Vec<_>>(), ["in0", "in1", "out"], "{out}");
    for b in &bufs {
        let addr = u64::from_str_radix(b[2].trim_start_matches("0x"), 16).unwrap();
        assert_eq!((b[1], b[3], b[4]), ("4194304", (addr % 4096).to_string().as_str(), (addr % 65536).to_string().as_str()), "{out}");
    }
}

#[test]
fn train_with_the_timers_trains_the_same_weights() {
    let d = std::env::temp_dir().join(format!("presk-host-profile-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let mut weights = Vec::new();
    for flag in [false, true] {
        let ckpt = d.join(if flag { "with" } else { "without" });
        let mut args = vec!["train", "examples/tutorial/07_train.psk", "--steps", "4", "--backend", "native", "--ckpt", ckpt.to_str().unwrap()];
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
    let o = presk(&["train", "examples/tutorial/07_train.psk", "--steps", "1", "--host-profile"]);
    assert_eq!(o.status.code(), Some(2), "the reference interpreter has no host to profile");
    let _ = std::fs::remove_dir_all(&d);
}
