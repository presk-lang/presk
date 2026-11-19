//! Statement FC6, the overflow guard (plan 0.18.2 step 4, D218; question
//! 87), through the binary. `examples/tests/guard/guard.psk` stores `x · x`
//! in the realization's type. Its envelope, calibrated on inputs of
//! magnitude at most 100, offers binary16 at the site, and the solver takes
//! it. On those inputs every executor runs the realization, with the same
//! bits. On an input of 300, off the envelope, the store rounds 90 000 to
//! an infinity: the kernel interpreter, the native backend, the LLVM
//! backend (where `make llvm` installed its compiler) and the exported C
//! each fail the run with the guard's error, `E7007`, in the same words,
//! naming the site. The C program built with `-DPRESK_NO_GUARD` tests
//! nothing and carries the infinity, as before the guard; and the kernel
//! interpreter's emulation of a realization the native types do not run
//! (the site computed in binary16) counts the infinity and fails nothing,
//! as since D205. `edge`, the product in `f32` and the sum in binary16,
//! overflows where the host converts the product at the sum's edge: the
//! same refusal, in the host's words, on every executor and in both of the
//! C export's hosts.

use std::path::{Path, PathBuf};
use std::process::Command;

const GUARD: &str = "examples/tests/guard/guard.psk";

/// The guard's message at the program's one site, as every executor and
/// the exported C print it.
const MESSAGE: &str = "the overflow guard: at `square/%1`, the kernel `mul` rounded a finite value past f16's largest finite number, 65504, to an infinity; a realization holds on its envelope, and this run's values leave it (question 87)";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

/// The exit status, standard output and standard error of `presk args`.
fn presk(args: &[&str]) -> (bool, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_presk")).current_dir(root()).args(args).env("NO_COLOR", "1").output().expect("running presk");
    (o.status.success(), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

fn scratch() -> PathBuf {
    let d = std::env::temp_dir().join(format!("presk-guard-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// One sample of type `ty`, the values `v`, written as `f64`.
fn data(path: &Path, ty: &str, v: &[f64]) -> String {
    let layout = presk_data::Layout::from_type_text(ty).unwrap();
    presk_data::write_encoded(path, ty, &layout, &[presk_data::Enc::F64], [&presk_data::Value::Real(v.to_vec())]).unwrap();
    path.to_str().unwrap().to_string()
}

const INSIDE: [f64; 8] = [1.0, -2.0, 3.5, -100.0, 50.0, 0.25, 7.0, -9.0];
const OFF: [f64; 8] = [1.0, -2.0, 3.5, -100.0, 300.0, 0.25, 7.0, -9.0];

#[test]
fn a_store_past_the_format_fails_the_run_on_every_executor() {
    let dir = scratch();
    let (inside, off) = (data(&dir.join("in.psk.d"), "ℝ[i: 8]", &INSIDE), data(&dir.join("off.psk.d"), "ℝ[i: 8]", &OFF));
    let env = dir.join("env.json");
    let (ok, _, err) = presk(&["calibrate", "--precision", GUARD, "--entry", "square", "--dim", "N=8", "--data", &inside, "--out", env.to_str().unwrap()]);
    assert!(ok, "{err}");
    // On the envelope the range rules offer binary16, and the solver,
    // measuring natively, stores the site in it.
    let solved = dir.join("solved.json");
    let (ok, _, err) = presk(&["solve", GUARD, "--precision", env.to_str().unwrap(), "--budget", "1e-2", "--menu", "f64,f32,f16", "--measure", "native", "--out", solved.to_str().unwrap()]);
    assert!(ok, "{err}");
    let r: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&solved).unwrap()).unwrap();
    assert_eq!(r["storage"]["square/%1"], "f16", "{r}");
    let realize = format!("@{}", solved.display());
    let llvm = root().join(".llvm/bin/clang").exists();
    let backends: Vec<&str> = ["kernel", "native", "llvm"].into_iter().filter(|b| *b != "llvm" || llvm).collect();
    let run = |backend: &str, input: &str| presk(&["run", GUARD, "--entry", "square", "--dim", "N=8", "--backend", backend, "--realize", &realize, "--data", input, "--codes"]);
    let (ok, want, err) = run("kernel", &inside);
    assert!(ok, "{err}");
    for b in &backends {
        let (ok, out, err) = run(b, &inside);
        assert!(ok && out == want, "{b} inside the envelope: {err}");
        let (ok, out, err) = run(b, &off);
        assert!(!ok && out.is_empty(), "{b} off the envelope: {out}");
        assert!(err.contains(&format!("[E7007] Error: in `square` (%1): {MESSAGE}")), "{b}: {err}");
    }

    // The exported C: the same program at one sample.
    let c = dir.join("c");
    let (ok, _, err) = presk(&["export", "--target", "c", GUARD, "--entry", "square", "--dim", "N=8", "--dim", "B=1", "--realize", &realize, "--out", c.to_str().unwrap()]);
    assert!(ok, "{err}");
    let model_h = std::fs::read_to_string(c.join("model.h")).unwrap();
    assert!(model_h.contains("int presk_square(presk_arena *arena"), "{model_h}");
    let make = |flags: Option<&str>| {
        let mut m = Command::new("make");
        m.arg("-s").arg("-B").arg("-C").arg(&c).arg("model");
        if let Some(f) = flags {
            m.arg(format!("CFLAGS=-O2 {f}"));
        }
        let o = m.output().unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    };
    let program = |input: &str, out: &Path| {
        let o = Command::new(c.join("model")).arg(input).arg(out).output().unwrap();
        (o.status.code(), String::from_utf8_lossy(&o.stderr).into_owned())
    };
    let (cin, coff) = (data(&dir.join("in-c.psk.d"), "ℝ[1, 8]", &INSIDE), data(&dir.join("off-c.psk.d"), "ℝ[1, 8]", &OFF));
    make(None);
    let o = Command::new("sh").arg(c.join("conformance/check.sh")).output().unwrap();
    assert!(o.status.success(), "C1: {}", String::from_utf8_lossy(&o.stderr));
    let (code, err) = program(&cin, &dir.join("c-in.psk.d"));
    assert_eq!(code, Some(0), "{err}");
    let (code, err) = program(&coff, &dir.join("c-off.psk.d"));
    assert_eq!(code, Some(1), "{err}");
    assert_eq!(err, format!("[E7007] Error: {MESSAGE}\n"));
    assert!(!dir.join("c-off.psk.d").exists(), "a program the guard stopped writes no outputs");
    // Without the guard, the infinity is the output, as before it.
    make(Some("-DPRESK_NO_GUARD"));
    let (code, err) = program(&coff, &dir.join("c-off.psk.d"));
    assert_eq!(code, Some(0), "{err}");
    let (_, info, _) = presk(&["data", "info", dir.join("c-off.psk.d").to_str().unwrap()]);
    assert!(info.contains("[[1, 4, 12.25, 10000, inf, 0.0625, 49, 81]]"), "{info}");

    // The host's narrowing at a nest's edge.
    let edge = dir.join("edge.json");
    std::fs::write(&edge, r#"{"default": "f32", "storage": {"edge/%4": "f16"}, "order": {"kind": "sequential"}}"#).unwrap();
    let at_edge = "the overflow guard: at `edge/%4`, an operand converted to f16 at the nest's edge rounded past f16's largest finite number, 65504, to an infinity; a realization holds on its envelope, and this run's values leave it (question 87)";
    let edge_realize = format!("@{}", edge.display());
    for b in &backends {
        let (ok, _, err) = presk(&["run", GUARD, "--entry", "edge", "--dim", "N=8", "--backend", b, "--realize", &edge_realize, "--data", &inside]);
        assert!(ok, "{b}: {err}");
        let (ok, _, err) = presk(&["run", GUARD, "--entry", "edge", "--dim", "N=8", "--backend", b, "--realize", &edge_realize, "--data", &off]);
        assert!(!ok && err.contains(&format!("[E7007] Error: in `edge` (%4): {at_edge}")), "{b}: {err}");
    }
    for (host, flags) in [("structured", vec![]), ("straight-line", vec!["--straight-line"])] {
        let ce = dir.join(format!("edge-{host}"));
        let mut args = vec!["export", "--target", "c", GUARD, "--entry", "edge", "--dim", "N=8", "--dim", "B=1", "--realize", &edge_realize, "--out", ce.to_str().unwrap()];
        args.extend(flags);
        let (ok, _, err) = presk(&args);
        assert!(ok, "{err}");
        let o = Command::new("sh").arg(ce.join("conformance/check.sh")).output().unwrap();
        assert!(o.status.success(), "{host}: C1: {}", String::from_utf8_lossy(&o.stderr));
        let o = Command::new(ce.join("model")).arg(&coff).arg(dir.join("edge-off.psk.d")).output().unwrap();
        assert_eq!(o.status.code(), Some(1), "{host}");
        assert_eq!(String::from_utf8_lossy(&o.stderr), format!("[E7007] Error: {at_edge}\n"), "{host}");
    }

    // A realization the native types do not run, the site computed in
    // binary16: emulated, its infinity counted, nothing failed (D205).
    let emulated = dir.join("emulated.json");
    std::fs::write(&emulated, r#"{"default": "f64", "storage": {"square/%1": "f16"}, "arith": {"square/%1": "f16"}, "order": {"kind": "sequential"}}"#).unwrap();
    let (ok, _, err) = presk(&["run", GUARD, "--entry", "square", "--dim", "N=8", "--backend", "kernel", "--realize", &format!("@{}", emulated.display()), "--data", &off]);
    assert!(ok && err.contains("emulated: 0 saturations, 1 infinity, 0 NaNs, 0 wraps"), "{err}");
    std::fs::remove_dir_all(&dir).unwrap();
}
