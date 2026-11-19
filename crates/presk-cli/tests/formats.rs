//! What a nest computes in, run by the commands (plan 0.18.0 step 3, D206):
//! the kernel interpreter emulates a realization in any formats and `presk
//! run` reports what they produced; the native backend (since plan 0.18.2
//! step 2, D216), the LLVM backend and the C export (step 3, D217) compute
//! in `f64`, `f32` and `bf16` and store values in binary16 and `e8m15` too,
//! with the emulation's bits, and refuse the rest before anything runs
//! (`E7005`); a tile of E4M3 operands summed in binary16 in chunks of 32 is
//! the sum written out.

use std::path::{Path, PathBuf};
use std::process::Command;

use presk_format::{Tie, E4M3_SATURATING, F16};

const TILE: &str = "examples/tests/formats/tile.psk";
const CHUNKED: &str = "@examples/tests/formats/e4m3-f16-chunked.json";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

/// The exit status, standard output and standard error of `presk args`.
fn presk(args: &[&str]) -> (bool, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_presk")).current_dir(root()).args(args).env("NO_COLOR", "1").output().expect("running presk");
    (o.status.success(), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

#[test]
fn the_kernel_interpreter_emulates_and_counts() {
    // A realization a native backend computes prints what it always did.
    let (ok, _, err) = presk(&["run", TILE, "--entry", "tile", "--backend", "kernel", "--realize", "f32"]);
    assert!(ok && !err.contains("emulated"), "{err}");
    for realize in ["f16", "e4m3fn", CHUNKED] {
        let (ok, out, err) = presk(&["run", TILE, "--entry", "tile", "--backend", "kernel", "--realize", realize]);
        assert!(ok && out.starts_with("ℝ[4, 3] = "), "{realize}: {err}");
        assert!(err.contains("presk run: the realization's formats, emulated: 0 saturations, 0 infinities, 0 NaNs, 0 wraps"), "{realize}: {err}");
    }
}

#[test]
fn the_native_executors_refuse_what_they_cannot_compute() {
    // binary16 and `e8m15` natively (D216) and on the LLVM backend when
    // `make llvm` installed its compiler (D217), the emulation's codes.
    let llvm = root().join(".llvm/bin/clang").exists();
    for realize in ["f16", "e8m15"] {
        let (ok, emulated, err) = presk(&["run", TILE, "--entry", "tile", "--backend", "kernel", "--realize", realize, "--codes"]);
        assert!(ok && err.contains("emulated"), "{realize}: {err}");
        for backend in ["native", "llvm"].into_iter().filter(|b| *b == "native" || llvm) {
            let (ok, out, err) = presk(&["run", TILE, "--entry", "tile", "--backend", backend, "--realize", realize, "--codes"]);
            assert!(ok && !err.contains("emulated"), "{realize} on {backend}: {err}");
            assert_eq!(out, emulated, "{realize} on {backend}: the kernel interpreter's codes");
        }
    }
    for backend in ["native", "llvm"].into_iter().filter(|b| *b == "native" || llvm) {
        let (ok, _, err) = presk(&["run", TILE, "--entry", "tile", "--backend", backend, "--realize", "e4m3fn"]);
        assert!(!ok && err.contains("E7005") && err.contains(&format!("the realization stores every value in e4m3fn: the {backend} backend computes in f64, f32, bf16, f16 and e8m15")), "{err}");
        let (ok, _, err) = presk(&["run", TILE, "--entry", "tile", "--backend", backend, "--realize", CHUNKED]);
        assert!(!ok && err.contains("E7005"), "{err}");
    }
    // The C export: binary16 and `e8m15` exported, and the program's check
    // passes (C1); any other format refused, nothing written.
    let base = std::env::temp_dir().join(format!("presk-formats-{}", std::process::id()));
    for realize in ["f16", "e8m15"] {
        let dir = base.join(realize);
        let (ok, _, err) = presk(&["export", "--target", "c", TILE, "--entry", "tile", "--realize", realize, "-o", dir.to_str().unwrap()]);
        assert!(ok, "{realize}: {err}");
        let check = Command::new("sh").arg("conformance/check.sh").current_dir(&dir).output().unwrap();
        assert!(check.status.success() && String::from_utf8_lossy(&check.stdout).contains("C1 holds"), "{realize}: {}{}", String::from_utf8_lossy(&check.stdout), String::from_utf8_lossy(&check.stderr));
    }
    let dir = base.join("e4m3fn");
    let (ok, _, err) = presk(&["export", "--target", "c", TILE, "--entry", "tile", "--realize", "e4m3fn", "-o", dir.to_str().unwrap()]);
    assert!(!ok && err.contains("E7005") && err.contains("the C export computes in f64, f32, bf16, f16 and e8m15"), "{err}");
    assert!(!dir.exists(), "nothing written");
    let _ = std::fs::remove_dir_all(&base);
}

/// The tile under `e4m3-f16-chunked.json`, against the sum written out:
/// the entry's operands rounded once to the saturating E4M3, each product
/// of two (exact in binary16) rounded to binary16, each chunk of 32 terms
/// summed from zero in binary16, the two chunks' sums then added from zero.
#[test]
fn a_tile_computes_in_its_formats() {
    let (ok, out, err) = presk(&["run", TILE, "--entry", "tile", "--backend", "kernel", "--realize", CHUNKED, "--json", "--digits", "40"]);
    assert!(ok, "{err}");
    let j: serde_json::Value = serde_json::from_str(&out).unwrap();
    let matrix = |v: &serde_json::Value| -> Vec<Vec<f64>> { v["data"].as_array().unwrap().iter().map(|r| r.as_array().unwrap().iter().map(|x| x.as_str().unwrap().parse::<f64>().unwrap()).collect()).collect() };
    let e4m3 = |m: Vec<Vec<f64>>| -> Vec<Vec<f64>> { m.into_iter().map(|r| r.into_iter().map(|x| E4M3_SATURATING.round(x, Tie::Even)).collect()).collect() };
    let (x, w, got) = (e4m3(matrix(&j["inputs"]["x"])), e4m3(matrix(&j["inputs"]["w"])), matrix(&j["result"]));
    let add = |s: f64, t: f64| F16.round(s + t, Tie::Even);
    for i in 0..4 {
        for c in 0..3 {
            let terms: Vec<f64> = (0..64).map(|k| F16.round(x[i][k] * w[k][c], Tie::Even)).collect();
            let want = terms.chunks(32).map(|ch| ch.iter().fold(0.0, |s, t| add(s, *t))).fold(0.0, add);
            assert_eq!(got[i][c].to_bits(), want.to_bits(), "[{i}, {c}]: {} and {want}", got[i][c]);
        }
    }
}

const SYSTOLIC: &str = "examples/tests/formats/systolic.psk";
const E4M3: &str = "float(p=4,emin=-6,emax=8,specials=none,underflow=+0)";

/// A scratch directory of this test's own.
fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("presk-formats-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// One sample of `ℝ[i: 8]`, written in `enc`.
fn data(dir: &Path, name: &str, enc: presk_data::Enc, v: &[f64]) -> PathBuf {
    let p = dir.join(name);
    let ty = format!("ℝ[i: {}]", v.len());
    let layout = presk_data::Layout::from_type_text(&ty).unwrap();
    presk_data::write_encoded(&p, &ty, &layout, &[enc], [&presk_data::Value::Real(v.to_vec())]).unwrap();
    p
}

/// `presk run --codes` of `convert` on `data` under the realization `r`:
/// the counts line and the codes.
fn codes(dir: &Path, r: &str, data: &Path, n: usize) -> (String, Vec<String>) {
    let real = dir.join("r.json");
    std::fs::write(&real, r).unwrap();
    let (ok, out, err) = presk(&["run", SYSTOLIC, "--entry", "convert", "--dim", &format!("N={n}"), "--backend", "kernel", "--realize", &format!("@{}", real.display()), "--data", data.to_str().unwrap(), "--codes"]);
    assert!(ok, "{err}");
    let j: serde_json::Value = serde_json::from_str(&out).unwrap();
    (err, j["result"]["codes"].as_array().unwrap().iter().map(|c| c.as_str().unwrap().to_string()).collect())
}

/// What the host converts outside a kernel counts as a kernel's own
/// conversions do (D209): an argument stored in its type (70 000 in
/// binary16, an infinity), an operand converted at a nest's edge to its
/// storage type (1000, −600 and that infinity into E4M3, saturated). And
/// `--codes` prints the codes, a NaN's sign and −0's included.
#[test]
fn the_host_s_conversions_count_and_the_codes_print() {
    let dir = scratch("codes");
    let v = [1000.0, -600.0, 70000.0, f64::NAN, -f64::NAN, -0.0, 1.5, -2.25];
    let d = data(&dir, "x.psk.d", presk_data::Enc::F64, &v);
    let r = format!(r#"{{"default": "f16", "storage": {{"convert/%3": "{E4M3}"}}, "order": {{"kind": "sequential"}}}}"#);
    let (err, c) = codes(&dir, &r, &d, v.len());
    assert!(err.contains("emulated: 5 saturations, 1 infinity, 0 NaNs, 0 wraps"), "{err}");
    assert_eq!(c, ["7f", "ff", "7f", "7f", "ff", "00", "3c", "c1"]);
    let (err, c) = codes(&dir, r#"{"default": "f16", "order": {"kind": "sequential"}}"#, &d, v.len());
    assert!(err.contains("emulated: 0 saturations, 1 infinity, 0 NaNs, 0 wraps"), "{err}");
    assert_eq!(c, ["63d0", "e0b0", "7c00", "7e00", "fe00", "8000", "3e00", "c080"]);
    let (ok, _, err) = presk(&["run", SYSTOLIC, "--entry", "convert", "--dim", "N=8", "--data", d.to_str().unwrap(), "--codes"]);
    assert!(!ok && err.contains("--codes prints the codes"), "{err}");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A data file's leaf in a format given by its parameters holds that
/// format's values, −0 included, which no conversion gives under an
/// underflow to +0 (D209): E4M3's −0 read from its file is −0 in binary16,
/// and the same −0 read from `f64` is converted to +0.
#[test]
fn a_file_of_a_format_s_codes_is_read_as_they_are() {
    let dir = scratch("v3");
    let v = [-0.0, 0.0, 480.0, -0.001953125, 1.5, -2.25, 0.25, -480.0];
    let r = format!(r#"{{"default": "{E4M3}", "storage": {{"convert/%3": "f16"}}, "arith": {{"convert/%3": "f16"}}, "order": {{"kind": "sequential"}}}}"#);
    let (_, c) = codes(&dir, &r, &data(&dir, "e4m3.psk.d", presk_data::Enc::Format(presk_format::E4M3_SATURATING), &v), v.len());
    assert_eq!(c, ["8000", "0000", "5f80", "9800", "3e00", "c080", "3400", "df80"]);
    let (_, c) = codes(&dir, &r, &data(&dir, "f64.psk.d", presk_data::Enc::F64, &v), v.len());
    assert_eq!(c[0], "0000");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A format whose NaN is its default one (D209): x·1 of a negative NaN is
/// the positive quiet NaN in binary16 computing with `nans=default`, and
/// the operand's in binary16.
#[test]
fn an_operation_s_nan_is_the_format_s_rule() {
    let dir = scratch("nans");
    let d = data(&dir, "x.psk.d", presk_data::Enc::F64, &[-f64::NAN, f64::NAN, -1.5, 0.0]);
    for (arith, want) in [("float(p=11,emin=-14,emax=15,nans=default)", "7e00"), ("f16", "fe00")] {
        let r = format!(r#"{{"default": "f16", "arith": {{"convert/%3": "{arith}"}}, "order": {{"kind": "sequential"}}}}"#);
        let (_, c) = codes(&dir, &r, &d, 4);
        assert_eq!(c, [want, "7e00", "be00", "0000"], "{arith}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
