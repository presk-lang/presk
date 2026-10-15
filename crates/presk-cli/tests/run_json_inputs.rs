//! `presk run --json` prints the arguments the entry ran on (D118's
//! follow-up). With `--data` it printed a synthesized batch, and with
//! `--weights` the synthesized initialization, while it ran on the data
//! file's samples and the checkpoint's parameters: the result was right and
//! the record of its inputs was not (found by `make margins`, D118).

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn presk(args: &[&str]) -> String {
    let o = Command::new(env!("CARGO_BIN_EXE_presk")).current_dir(root()).args(args).env("NO_COLOR", "1").output().expect("running presk");
    assert!(o.status.success(), "presk {}: {}", args.join(" "), String::from_utf8_lossy(&o.stderr));
    String::from_utf8_lossy(&o.stdout).into_owned()
}

/// A JSON tensor's elements as numbers, whether printed as decimal strings
/// (`presk run --json`) or as numbers (`presk data export`).
fn numbers(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(xs) => xs.iter().flat_map(numbers).collect(),
        Value::String(s) => vec![s.parse().unwrap()],
        Value::Number(n) => vec![n.as_f64().unwrap()],
        other => panic!("not a tensor's data: {other}"),
    }
}

#[test]
fn the_inputs_printed_are_the_data_and_the_checkpoint() {
    let d = std::env::temp_dir().join(format!("presk-run-json-inputs-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let (data, from, samples) = ("tutorial/data/pairs.psk.d", 2, 3);
    let export = d.join("pairs.json");
    presk(&["data", "export", data, "--out", export.to_str().unwrap()]);
    let exported: Value = serde_json::from_str(&std::fs::read_to_string(&export).unwrap()).unwrap();
    // The samples run, leaf by leaf: (x, y) of samples 2, 3 and 4.
    let leaf = |i: usize| -> Vec<f64> { (from..from + samples).flat_map(|s| numbers(&exported["samples"][s][i])).collect() };
    let ckpt = d.join("ckpt");
    presk(&["train", "tutorial/07_train.psk", "--steps", "20", "--backend", "native", "--ckpt", ckpt.to_str().unwrap()]);
    let weights = ckpt.join("final.psk.d");
    let (from_s, samples_s) = (from.to_string(), samples.to_string());
    let run = |extra: &[&str]| -> Value {
        let mut args = vec!["run", "tutorial/07_train.psk", "--entry", "loss", "--data", data, "--from", &from_s, "--samples", &samples_s, "--json", "--digits", "17"];
        args.extend_from_slice(extra);
        serde_json::from_str(&presk(&args)).unwrap()
    };
    let synthesized = run(&[]);
    for (backend, out) in [("interp", run(&["--weights", weights.to_str().unwrap()])), ("native", run(&["--weights", weights.to_str().unwrap(), "--backend", "native"]))] {
        // The batch is the data file's, on both paths, with and without weights.
        for o in [&synthesized, &out] {
            for i in 0..2 {
                assert_eq!(numbers(&o["inputs"]["batch"][i]["data"]), leaf(i), "{backend}: leaf {i} of the batch");
            }
        }
        // The parameters are the checkpoint's: trained, so not the
        // initialization a run without `--weights` synthesizes, and the same
        // tree whichever executor ran.
        assert_ne!(out["inputs"]["θ"], synthesized["inputs"]["θ"], "{backend}: the initialization printed for the checkpoint");
        assert_ne!(out["result"], synthesized["result"], "{backend}");
    }
    let (interp, native) = (run(&["--weights", weights.to_str().unwrap()]), run(&["--weights", weights.to_str().unwrap(), "--backend", "native"]));
    assert_eq!(interp["inputs"]["θ"], native["inputs"]["θ"]);
    std::fs::remove_dir_all(&d).ok();
}
