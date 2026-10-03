//! A stack of no layer, `stack{0}(M)` (D99's addendum, plan 0.10.0 step 8),
//! through the binary. Its parameters are built from their type, every leaf
//! of `M` with a leading axis of length 0, on the reference interpreter and
//! the kernel backend's host alike (both took a stack's shapes from its first
//! part, and panicked on a stack of none); the fold of count 0 over them
//! gives its initial value, on every executor and in the C export. And the
//! refusal at dimensions (`E7003`), which skips the body of a fold of count
//! 0, defines `Tower.apply` at `T = 0` with no layer and refuses it with
//! two.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn presk(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_presk")).current_dir(root()).args(args).env("NO_COLOR", "1").output().expect("running presk")
}

fn ok(o: &Output, what: &str) -> String {
    assert!(o.status.success(), "{what}: exit {:?}\n{}", o.status.code(), String::from_utf8_lossy(&o.stderr));
    String::from_utf8_lossy(&o.stdout).to_string()
}

const FILE: &str = "tests/data/stack_none.psk";

fn json(o: &Output, what: &str) -> serde_json::Value {
    serde_json::from_str(&ok(o, what)).unwrap()
}

/// The C export of `entry`, checked against the native backend (C1).
fn c1(entry: &str, extra: &[&str]) {
    let out = std::env::temp_dir().join(format!("presk-stack-none-{}-{entry}-{}", std::process::id(), extra.join("")));
    let _ = std::fs::remove_dir_all(&out);
    let mut args = vec!["export", FILE, "--target", "c", "--entry", entry, "--dim", "T=2", "-o", out.to_str().unwrap()];
    args.extend(extra);
    ok(&presk(&args), &format!("presk export {entry} {extra:?}"));
    let check = Command::new("sh").arg(out.join("conformance/check.sh")).env("PRESK", "/nonexistent").output().unwrap();
    assert!(check.status.success() && String::from_utf8_lossy(&check.stdout).contains("C1 holds"), "{entry} {extra:?}, C1:\n{}{}", String::from_utf8_lossy(&check.stdout), String::from_utf8_lossy(&check.stderr));
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn a_stack_of_no_layer_runs_on_every_executor_and_in_c() {
    // The parameters: each leaf of `Layer` with a leading axis of length 0.
    for backend in ["interp", "kernel", "native"] {
        let o = presk(&["run", FILE, "--entry", "Tower.init", "--dim", "T=2", "--backend", backend, "--json"]);
        let j = json(&o, &format!("Tower.init on {backend}"));
        for leaf in ["w", "g"] {
            let v = &j["result"]["layers"][leaf];
            assert_eq!(v["axes"], serde_json::json!([{"name": null, "len": 0}, {"name": "model", "len": 4}]), "{backend}: {v}");
            assert_eq!(v["data"], serde_json::json!([]), "{backend}");
        }
    }
    // `Tower.apply`, a fold of count 0, gives its input times `out`, and
    // the gradient through it, under both policies: the kernel interpreter's
    // bits natively in `f64`, and the exact reference's to twelve digits
    // (the parameters are the reference interpreter's initializer's).
    for (entry, remat) in [("Tower.apply", "store"), ("loss_grad", "store"), ("loss_grad", "recompute")] {
        let run = |backend: &str, digits: &str| {
            let o = presk(&["run", FILE, "--entry", entry, "--dim", "T=2", "--backend", backend, "--remat", remat, "--json", "--digits", digits]);
            json(&o, &format!("{entry} on {backend} ({remat})"))["result"].clone()
        };
        assert_eq!(run("kernel", "17"), run("native", "17"), "{entry} ({remat})");
        assert_eq!(run("interp", "12"), run("native", "12"), "{entry} ({remat})");
    }
    // The C export: the forward pass, and the gradient by both policies.
    c1("Tower.apply", &[]);
    c1("loss_grad", &["--remat", "store"]);
    c1("loss_grad", &["--remat", "recompute"]);
}

#[test]
fn a_fold_of_count_zero_is_not_walked_by_the_refusal_at_dimensions() {
    // `T = 0` empties `seq`, which `Layer.apply` averages over: with no
    // layer its body is never applied, and the entry is defined.
    const DEFINED: &str = "tests/data/defined.psk";
    for backend in ["interp", "kernel", "native"] {
        let o = presk(&["run", DEFINED, "--const", "L=0", "--entry", "Tower.apply", "--dim", "T=0", "--backend", backend, "--json"]);
        let j = json(&o, &format!("Tower.apply at T = 0 on {backend}"));
        assert_eq!(j["result"]["axes"], serde_json::json!([{"name": "seq", "len": 0}, {"name": "model", "len": 4}]));
        let o = presk(&["run", DEFINED, "--const", "L=2", "--entry", "Tower.apply", "--dim", "T=0", "--backend", backend]);
        assert_eq!(o.status.code(), Some(1), "{backend}");
        assert!(String::from_utf8_lossy(&o.stderr).contains("[E7003]"), "{backend}: {}", String::from_utf8_lossy(&o.stderr));
    }
}
