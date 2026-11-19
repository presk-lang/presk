//! A constant that a nest's storage type does not hold (plan 0.18.2 step 6,
//! D220), through the binary. `examples/tests/formats/constant.psk`'s
//! `update` divides by `√v + ε` and by ε, with ε = 10⁻⁸ a `real` of its own
//! site; `literal` divides by a float literal 10⁻⁸. The realization stores
//! the nests that read ε in binary16 and ε where the range rules leave it,
//! `f32`: binary16 rounds 10⁻⁸ to zero (D208). Each nest reads ε in its
//! arithmetic type, `f32`, and not converted to binary16 at its edge, so
//! that where `v` is zero the update is `y / ε` and not an infinity, as
//! AdamW's was on FC4's realization (D219). The kernel interpreter, the
//! native backend and the LLVM backend (where `make llvm` installed its
//! compiler) give the same codes, all finite; the exported C, under both
//! of its hosts, gives the native backend's bits (C1) and finite outputs.

use std::path::{Path, PathBuf};
use std::process::Command;

const PROGRAM: &str = "examples/tests/formats/constant.psk";

/// The nests that read 10⁻⁸ in binary16, everything else in `f32`.
const REALIZATION: &str = r#"{"default": "f32", "storage": {"update/%12": "f16", "update/%13": "f16", "literal/%6": "f16"}, "order": {"kind": "sequential"}}"#;

const X: [f64; 8] = [1.0, -2.0, 3.0, -1.0, 5.0, -3.0, 2.0, -5.0];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

/// The exit status, standard output and standard error of `presk args`.
fn presk(args: &[&str]) -> (bool, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_presk")).current_dir(root()).args(args).env("NO_COLOR", "1").output().expect("running presk");
    (o.status.success(), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

fn scratch() -> PathBuf {
    let d = std::env::temp_dir().join(format!("presk-constant-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// One sample of type `ty`, the values `v`, written as `f64`.
fn data(path: &Path, ty: &str, v: &[f64]) -> String {
    let layout = presk_data::Layout::from_type_text(ty).unwrap();
    presk_data::write_encoded(path, ty, &layout, &[presk_data::Enc::F64], [&presk_data::Value::Real(v.to_vec())]).unwrap();
    path.to_str().unwrap().to_string()
}

/// The numbers of a printed result's tensors, in order, each tensor's own:
/// `[batch: 1, i: 8] [[1, -2000, …]]`.
fn tensors(printed: &str) -> Vec<Vec<f64>> {
    printed
        .split("[[")
        .skip(1)
        .map(|t| t.split("]]").next().unwrap().split(',').map(|x| x.trim().parse::<f64>().unwrap_or_else(|_| panic!("`{x}` in {printed}"))).collect())
        .collect()
}

#[test]
fn a_nest_reads_a_constant_its_storage_does_not_hold_in_its_arithmetic_type() {
    let dir = scratch();
    let input = data(&dir.join("x.psk.d"), "ℝ[i: 8]", &X);
    let r = dir.join("r.json");
    std::fs::write(&r, REALIZATION).unwrap();
    let realize = format!("@{}", r.display());
    let llvm = root().join(".llvm/bin/clang").exists();
    let backends: Vec<&str> = ["kernel", "native", "llvm"].into_iter().filter(|b| *b != "llvm" || llvm).collect();
    for entry in ["update", "literal"] {
        let run = |backend: &str, codes: bool| {
            let mut args = vec!["run", PROGRAM, "--entry", entry, "--dim", "N=8", "--backend", backend, "--realize", &realize, "--data", &input];
            if codes {
                args.push("--codes");
            }
            presk(&args)
        };
        let (ok, want, err) = run("kernel", true);
        assert!(ok, "{entry}: {err}");
        for b in &backends {
            let (ok, codes, err) = run(b, true);
            assert!(ok && codes == want, "{entry} on {b}: {err}");
            let (ok, out, err) = run(b, false);
            assert!(ok, "{entry} on {b}: {err}");
            let ts = tensors(&out);
            assert!(ts.iter().flatten().all(|x| x.is_finite()), "{entry} on {b}: {out}");
            // `y / ε` (update's second result, literal's only one) is a
            // thousand times the input, binary16's 10⁻⁵ read for `y`.
            let quotient = ts.last().unwrap();
            for (q, x) in quotient.iter().zip(X) {
                assert!((q / (1000.0 * x) - 1.0).abs() < 2e-3, "{entry} on {b}: {q} for {x}: {out}");
            }
            // Where `y` is negative, `v` is zero and the update is `y / ε`
            // too; elsewhere it is `y / (y + ε)`, one in binary16.
            if entry == "update" {
                for (u, x) in ts[0].iter().zip(X) {
                    let want = if x < 0.0 { 1000.0 * x } else { 1.0 };
                    assert!((u / want - 1.0).abs() < 2e-3, "{entry} on {b}: {u} for {x}: {out}");
                }
            }
        }
        // The exported C, under both hosts: the native backend's bits on the
        // export's own inputs (C1), and finite outputs on these.
        let cin = data(&dir.join(format!("{entry}-c.psk.d")), "ℝ[1, 8]", &X);
        for (host, flags) in [("structured", vec![]), ("straight-line", vec!["--straight-line"])] {
            let c = dir.join(format!("{entry}-{host}"));
            let mut args = vec!["export", "--target", "c", PROGRAM, "--entry", entry, "--dim", "N=8", "--dim", "B=1", "--realize", &realize, "--out", c.to_str().unwrap()];
            args.extend(flags);
            let (ok, _, err) = presk(&args);
            assert!(ok, "{entry} ({host}): {err}");
            let o = Command::new("sh").arg(c.join("conformance/check.sh")).output().unwrap();
            assert!(o.status.success(), "{entry} ({host}): C1: {}", String::from_utf8_lossy(&o.stderr));
            let out = dir.join(format!("{entry}-{host}-out.psk.d"));
            let o = Command::new(c.join("model")).arg(&cin).arg(&out).output().unwrap();
            assert_eq!(o.status.code(), Some(0), "{entry} ({host}): {}", String::from_utf8_lossy(&o.stderr));
            let (ok, info, err) = presk(&["data", "info", out.to_str().unwrap()]);
            assert!(ok, "{err}");
            assert!(!info.contains("inf") && !info.contains("NaN"), "{entry} ({host}): {info}");
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
