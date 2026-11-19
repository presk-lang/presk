//! `presk bench --calibrate --plan-only` through the binary (plan 0.17.0
//! step 1, D174): the plan's kernels timed alone, the profile without the
//! other realizations' fits, the kernels the same as without the flag. And
//! binary16 and `e8m15` priced (plan 0.18.2 step 4, D218): a profile fits
//! both among its other realizations, a layout calibration in them fits
//! their groups at one lane, and `presk plan` plans their kernels there.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn presk(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_presk")).current_dir(root()).args(args).env("NO_COLOR", "1").output().expect("running presk")
}

fn calibrate(profile: &Path, plan_only: bool) -> serde_json::Value {
    let p = profile.to_str().unwrap();
    let mut args = vec!["bench", "examples/tests/plan/minigpt_grad.psk", "--entry", "loss_grad", "--const", "V=64", "--const", "D=16", "--const", "H=2", "--const", "L=2", "--dim", "B=2", "--dim", "T=8", "--plan", "@examples/tests/plan/default.json", "--threads", "2", "--repeat", "2"];
    if plan_only {
        args.push("--plan-only");
    }
    args.extend(["--calibrate", p]);
    let o = presk(&args);
    assert!(o.status.success(), "{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
    serde_json::from_str(&std::fs::read_to_string(profile).unwrap()).unwrap()
}

fn signatures(p: &serde_json::Value) -> Vec<String> {
    let mut s: Vec<String> = p["kernels"].as_array().unwrap().iter().map(|k| k["signature"].as_str().unwrap().to_string()).collect();
    s.sort();
    s
}

#[test]
fn plan_only_calibrates_the_plans_kernels_alone() {
    let d = std::env::temp_dir().join(format!("presk-calibrate-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let all = calibrate(&d.join("all.json"), false);
    let only = calibrate(&d.join("only.json"), true);
    // Without the flag the profile fits the other uniform realizations too,
    // binary16 and `e8m15` among them (D218); with it, none.
    let others = all["others"].as_object().expect("the other realizations' fits");
    for r in ["bf16", "f16", "e8m15"] {
        assert!(others.contains_key(r), "{r}: {all}");
    }
    assert!(only.get("others").is_none_or(|o| o.as_object().is_some_and(|o| o.is_empty())), "{only}");
    // The same kernels, timed under the same plan.
    assert_eq!(signatures(&all), signatures(&only));
    assert_eq!(all["dtype"], only["dtype"]);
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn binary16_and_e8m15_are_planned_at_one_lane() {
    let d = std::env::temp_dir().join(format!("presk-calibrate-codes-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let profile = d.join("layout.json");
    let p = profile.to_str().unwrap();
    for r in ["f16", "e8m15"] {
        let o = presk(&["bench", "examples/experiments/gemm.psk", "--entry", "gemm", "--dim", "N=32", "--realize", r, "--calibrate", p, "--space", "calib", "--repeat", "2"]);
        assert!(o.status.success(), "{r}: {}", String::from_utf8_lossy(&o.stderr));
    }
    let fit: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&profile).unwrap()).unwrap();
    let groups: Vec<&String> = fit["layout"]["pi"].as_object().unwrap().keys().collect();
    for r in ["f16", "e8m15"] {
        assert!(groups.iter().any(|g| g.starts_with(&format!("{r}+lanes1+"))), "{r}: {groups:?}");
        assert!(!groups.iter().any(|g| g.starts_with(&format!("{r}+lanes4"))), "{r}: {groups:?}");
        let o = presk(&["plan", "examples/experiments/gemm.psk", "--entry", "gemm", "--dim", "N=32", "--realize", r, "--profile", p]);
        let out = String::from_utf8_lossy(&o.stdout);
        assert!(o.status.success() && out.contains("lanes 1") && !out.contains("lanes 4"), "{r}: {out}{}", String::from_utf8_lossy(&o.stderr));
    }
    std::fs::remove_dir_all(&d).ok();
}
