//! Quantized activations, run by the commands (plan 0.18.0 step 4, D207):
//! `presk calibrate --range` records each site's smallest and largest
//! value, `presk scales` computes a realization's static scales from them,
//! and the kernel interpreter runs a dense layer on integers, its result the
//! integer sum written out from the inputs `--json` prints; a scale not yet
//! computed is refused (`E7006`), and the native backend refuses the
//! formats (`E7005`).

use std::path::{Path, PathBuf};
use std::process::Command;

use presk_format::{Affine, Format, Tie};

const LAYER: &str = "examples/tests/formats/w8a8.psk";
const W8A8: &str = "examples/tests/formats/w8a8.json";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

/// The exit status, standard output and standard error of `presk args`.
fn presk(args: &[&str]) -> (bool, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_presk")).current_dir(root()).args(args).env("NO_COLOR", "1").output().expect("running presk");
    (o.status.success(), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

fn matrix(v: &serde_json::Value) -> Vec<Vec<f64>> {
    v["data"].as_array().unwrap().iter().map(|r| r.as_array().unwrap().iter().map(|x| x.as_str().unwrap().parse::<f64>().unwrap()).collect()).collect()
}

#[test]
fn a_layer_runs_on_integers() {
    let dir = std::env::temp_dir().join(format!("presk-scaled-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (env, solved) = (dir.join("env.json"), dir.join("w8a8.json"));
    let (env, solved) = (env.to_str().unwrap(), solved.to_str().unwrap());
    // Refused until its scales are numbers, before anything runs.
    let (ok, _, err) = presk(&["run", LAYER, "--entry", "layer", "--backend", "kernel", "--realize", &format!("@{W8A8}")]);
    assert!(!ok && err.contains("[E7006]") && err.contains("`Dense.apply/%2` in u8/sat with a scale computed by `minmax`") && !err.contains("emulated"), "{err}");
    // Without the envelope's range, `minmax` has nothing to read.
    let (ok, _, err) = presk(&["calibrate", "--precision", LAYER, "--entry", "layer", "--bits", "128", "--out", env]);
    assert!(ok, "{err}");
    let (ok, _, err) = presk(&["scales", "--realize", &format!("@{W8A8}"), "--precision", env, "--out", solved]);
    assert!(!ok && err.contains("presk calibrate --range"), "{err}");
    let (ok, _, err) = presk(&["calibrate", "--precision", LAYER, "--entry", "layer", "--bits", "128", "--out", env, "--range"]);
    assert!(ok, "{err}");
    let e: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(env).unwrap()).unwrap();
    let [lo, hi] = [0, 1].map(|i| e["range"]["Dense.apply/%2"][i].as_f64().unwrap());
    assert!(lo == 0.0 && hi > 0.0 && hi == e["sites"]["Dense.apply/%2"].as_f64().unwrap(), "relu's range [{lo}, {hi}]");
    let (ok, out, err) = presk(&["scales", "--realize", &format!("@{W8A8}"), "--precision", env, "--out", solved]);
    assert!(ok, "{err}");
    let r: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(solved).unwrap()).unwrap();
    let affine = |site: &str| Affine::parse(r["scaled"][site].as_str().unwrap()).unwrap();
    let (a, y) = (affine("Dense.apply/%2"), affine("Dense.apply/%4"));
    assert_eq!(a, Affine::minmax(a.int, lo, hi).unwrap());
    assert_eq!(y, Affine::absmax(y.int, e["sites"]["Dense.apply/%4"].as_f64().unwrap()).unwrap());
    assert!(out.contains(&format!("presk scales: Dense.apply/%2: minmax of [0e0, {hi:e}]: {a}")), "{out}");
    assert_eq!(r["envelope"].as_str(), Some(env));
    // The layer on the kernel interpreter, against the integers written out.
    let (ok, out, err) = presk(&["run", LAYER, "--entry", "layer", "--backend", "kernel", "--realize", &format!("@{solved}"), "--json", "--digits", "40"]);
    assert!(ok && err.contains("emulated: 0 saturations, 0 infinities, 0 NaNs, 0 wraps"), "{err}");
    let j: serde_json::Value = serde_json::from_str(&out).unwrap();
    let (x, w, got) = (matrix(&j["inputs"]["x"]), matrix(&j["inputs"]["θ"]["w"]), matrix(&j["result"]));
    // The weights' integers, a scale per column (D44's absmax).
    let sw: Vec<f32> = (0..8).map(|c| ((0..16).map(|k| w[k][c].abs()).fold(0.0, f64::max) / 127.0) as f32).collect();
    let qw = |k: usize, c: usize| (w[k][c] / sw[c] as f64).round_ties_even().clamp(-127.0, 127.0) as i128;
    // The activation's codes, `relu` of the entry's `f32` operand rescaled.
    let qa = |i: usize, k: usize| a.round_noted((x[i][k] as f32).max(0.0) as f64).0 - i128::from(a.zero);
    for (i, row) in got.iter().enumerate() {
        for (c, g) in row.iter().enumerate() {
            let n: i128 = (0..16).map(|k| qa(i, k) * qw(k, c)).sum();
            // `n·s_a·s_w`, exactly: the scales' product is an `f64`, and
            // the integer sum times its significand is rescaled once.
            let unit = a.scale() * sw[c] as f64;
            let bits = unit.to_bits();
            let (m, e) = ((bits & ((1u64 << 52) - 1)) | (1u64 << 52), (bits >> 52) as i32 - 1075);
            let want = Format::Affine(y).round_scaled(n * m as i128, e, Tie::Even).0;
            assert_eq!(g.to_bits(), want.to_bits(), "[{i}, {c}]: n = {n}");
        }
    }
    let (ok, _, err) = presk(&["run", LAYER, "--entry", "layer", "--backend", "native", "--realize", &format!("@{solved}")]);
    assert!(!ok && err.contains("[E7005]"), "{err}");
    std::fs::remove_dir_all(&dir).unwrap();
}
