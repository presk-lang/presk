//! A solve in a machine's types (plan 0.18.0 step 5, D208): `presk solve
//! --target` offers exactly a target description's menu, puts each
//! contraction in one of its modes, measures on the kernel interpreter with
//! its counts, and writes no realization when none it measured is inside the
//! budget; `presk solve --offers` writes what the range rules offer every
//! stored value; `presk calibrate --formats` records a value's rounding in
//! more formats.

use std::path::{Path, PathBuf};
use std::process::Command;

const LAYER: &str = "examples/tests/formats/w8a8.psk";
const TARGET: &str = "examples/tests/targets/accelerator.json";
const LAYERNORM: &str = "examples/spec/layernorm/layernorm.psk";
const E4M3: &str = "float(p=4,emin=-6,emax=8,specials=none,underflow=+0)";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

/// The exit code, standard output and standard error of `presk args`.
fn presk(args: &[&str]) -> (Option<i32>, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_presk")).current_dir(root()).args(args).env("NO_COLOR", "1").output().expect("running presk");
    (o.status.code(), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

fn json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn a_layer_is_solved_in_a_machine_s_types() {
    let dir = std::env::temp_dir().join(format!("presk-target-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (env, solved) = (dir.join("env.json"), dir.join("solved.json"));
    let (env_s, solved_s) = (env.to_str().unwrap(), solved.to_str().unwrap());
    let formats = format!("f16,{E4M3}");
    let (code, _, err) = presk(&["calibrate", "--precision", LAYER, "--entry", "layer", "--bits", "128", "--range", "--formats", &formats, "--out", env_s]);
    assert_eq!(code, Some(0), "{err}");
    // The rounding in the formats asked, beside the three.
    let e = json(&env);
    let r = &e["rounding"]["Dense.apply/%2"];
    assert!(r["f16"].as_f64().is_some() && r[E4M3].as_f64().is_some() && r["bf16"].as_f64().is_some(), "{r}");

    // At a tenth: the integer mode, inside the budget, every value stored
    // in the target's formats and computed in its arithmetic.
    let (code, out, err) = presk(&["solve", LAYER, "--precision", env_s, "--target", TARGET, "--budget", "1e-1", "--out", solved_s]);
    assert_eq!(code, Some(0), "{out}{err}");
    assert!(out.contains("  target      `test-accelerator`: stores in f16, ") && out.contains("contraction modes (fp8, int8), order chunked(32)"), "{out}");
    assert!(out.contains("; contractions 1 int8") && out.contains("inside the budget"), "{out}");
    assert!(out.contains("  emulated    the realization's run: 0 saturations, 0 infinities, 0 NaNs, 0 wraps; the 1 emulated measurements:"), "{out}");
    let r = json(&solved);
    assert_eq!(r["default"], "f16");
    assert_eq!(r["order"], serde_json::json!({"kind": "chunked", "size": 32}));
    assert_eq!((r["product"]["Dense.apply/%4"].as_str(), r["accum"]["Dense.apply/%4"].as_str()), (Some("i32"), Some("i32")));
    assert!(r["scaled"]["Dense.apply/%2"].as_str().is_some_and(|a| a.starts_with("affine(i8/sat,scale=")), "a computed scale, not a scheme: {r}");
    assert_eq!(r["quant"]["w"]["bits"], 8);
    for site in ["Dense.apply/%2", "Dense.apply/%4"] {
        assert_eq!(r["arith"][site], "f16", "{site}: {r}");
    }
    // It runs as it was measured, and the native backend refuses it.
    let (code, _, err) = presk(&["run", LAYER, "--entry", "layer", "--backend", "kernel", "--realize", &format!("@{solved_s}")]);
    assert_eq!(code, Some(0), "{err}");
    let (code, _, err) = presk(&["run", LAYER, "--entry", "layer", "--backend", "native", "--realize", &format!("@{solved_s}")]);
    assert!(code != Some(0) && err.contains("[E7005]"), "{err}");

    // At 10⁻³ no realization in the target's types is inside: none is
    // written, and the smallest gap is said.
    std::fs::remove_file(&solved).unwrap();
    let (code, out, _) = presk(&["solve", LAYER, "--precision", env_s, "--target", TARGET, "--budget", "1e-3", "--out", solved_s]);
    assert_eq!(code, Some(3), "{out}");
    assert!(out.contains("is inside the budget: the smallest gap found") && out.contains("no realization is written"), "{out}");
    assert!(!solved.exists());
    // Asked apart, the one of smallest gap is written as the refusal's
    // evidence (D209), and it runs; --out still gets nothing.
    let refused = dir.join("refused.json");
    let (code, out, _) = presk(&["solve", LAYER, "--precision", env_s, "--target", TARGET, "--budget", "1e-3", "--out", solved_s, "--refused", refused.to_str().unwrap()]);
    assert_eq!(code, Some(3), "{out}");
    assert!(out.contains("the realization of smallest gap, outside the budget: the refusal's evidence, not a solution"), "{out}");
    assert!(!solved.exists() && json(&refused)["order"] == serde_json::json!({"kind": "chunked", "size": 32}));
    let (code, _, err) = presk(&["run", LAYER, "--entry", "layer", "--backend", "kernel", "--realize", &format!("@{}", refused.display())]);
    assert_eq!(code, Some(0), "{err}");
    let (code, _, err) = presk(&["solve", LAYER, "--precision", env_s, "--budget", "1e-3", "--refused", refused.to_str().unwrap()]);
    assert!(code == Some(2) && err.contains("--target"), "{err}");

    // A description that is not one, and one with --menu.
    let bad = dir.join("bad.json");
    std::fs::write(&bad, r#"{"name": "t", "storage": [], "arith": {"*": "f16"}}"#).unwrap();
    let (code, _, err) = presk(&["solve", LAYER, "--precision", env_s, "--target", bad.to_str().unwrap()]);
    assert!(code == Some(2) && err.contains("presk solve: --target: ") && err.contains("`storage` is empty"), "{err}");
    let (code, _, err) = presk(&["solve", LAYER, "--precision", env_s, "--target", TARGET, "--menu", "f32"]);
    assert!(code == Some(2) && err.contains("--menu"), "{err}");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The range rules on the specification's layer norm: a format its values
/// pass with the margin, the divisor's floor, a constant rounded to zero.
#[test]
fn the_range_rules_are_written_by_offers() {
    let dir = std::env::temp_dir().join(format!("presk-offers-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (env, offers) = (dir.join("env.json"), dir.join("offers.json"));
    let (env_s, offers_s) = (env.to_str().unwrap(), offers.to_str().unwrap());
    let (code, _, err) = presk(&["calibrate", "--precision", LAYERNORM, "--entry", "LayerNorm{8}.apply", "--dim", "batch=3", "--bits", "128", "--out", env_s]);
    assert_eq!(code, Some(0), "{err}");
    let (code, out, err) = presk(&["solve", LAYERNORM, "--precision", env_s, "--menu", "f32,f16,e4m3fn,e2m1", "--offers", offers_s]);
    assert_eq!(code, Some(0), "{err}");
    assert!(out.contains("6 stored values of `LayerNorm{8}.apply`, 4 formats refused by the range rules (1 range, 1 divisor, 2 constant), margin 2"), "{out}");
    let o = json(&offers);
    let value = |site: &str| o["values"].as_array().unwrap().iter().find(|v| v["site"] == site).unwrap().clone();
    let refused = |site: &str| value(site)["refused"].as_array().unwrap().iter().map(|r| (r[0].as_str().unwrap().to_string(), r[1].as_str().unwrap().to_string())).collect::<Vec<_>>();
    // The normalized value's square root, which divides: E2M1's smallest
    // normal number is 1.
    assert_eq!(value("LayerNorm.apply/%10")["divisor"], true);
    assert_eq!(refused("LayerNorm.apply/%10"), [("e2m1".to_string(), "divisor".to_string())]);
    // The squares reach 4.94: with a margin of 2, past E2M1's 6.
    assert_eq!(refused("LayerNorm.apply/%5"), [("e2m1".to_string(), "range".to_string())]);
    // ε = 10⁻⁵ rounds to zero in both eight-bit and four-bit formats, and
    // not in binary16, which holds it as a subnormal.
    assert_eq!(value("LayerNorm.apply/%8")["constant"], true);
    assert_eq!(refused("LayerNorm.apply/%8"), [("e4m3fn".to_string(), "constant".to_string()), ("e2m1".to_string(), "constant".to_string())]);
    // A wider margin refuses more.
    let (_, out, _) = presk(&["solve", LAYERNORM, "--precision", env_s, "--menu", "f32,f16,e4m3fn,e2m1", "--range-margin", "4", "--offers", offers_s]);
    assert!(out.contains("(2 range, 1 divisor, 2 constant), margin 4"), "{out}");
    std::fs::remove_dir_all(&dir).unwrap();
}
