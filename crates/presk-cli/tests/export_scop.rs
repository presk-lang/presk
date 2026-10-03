//! `presk export --target c --mode scop` (plan 0.9.0 §3.7, step 8, D85),
//! through the binary.
//!
//! - The options: `--entry` required; `--plan`, `--threads`,
//!   `--straight-line`, `--serve`, `--weights` and `--no-exact` refused, as
//!   is a realization with integer weights; `--mode scop` refused with the
//!   PyTorch target and the PyTorch modes with this one; nothing written.
//! - Every nest of every entry of `tests/plan/entries.txt`, in `f64`, `f32`
//!   and `bf16`: emitted, each file compiled on its own by its `Makefile`
//!   with GCC and `-Werror`, and run by the driver, once against the native
//!   backend's outputs (`make check`, bit for bit) and once timed. The nests
//!   the rule says are not static control are the gathers and the scatters.
//! - The check checked, by two mutations that change bits: the nests built
//!   in GCC's GNU mode, which contracts `a + b*c`, and a softmax's running
//!   maximum made a minimum.
//! - In `#[ignore]`d tests (`make c-conformance`), with Polly (`POLLY`, or
//!   LLVM 23.1.2's clang in `.llvm`, which `make llvm` installs): every nest
//!   the rule says is static control is one SCoP for Polly, the whole
//!   function, with the kernel's loops, and every other nest is not; allowed
//!   non-affine accesses, Polly takes those too. The course GPT's nests at
//!   its constants, at `B = 1` in the three types and at `B = 16` in `f32`,
//!   are checked the same way, bits and Polly.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use presk_export_c::scop;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("presk-export-scop-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn presk(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_presk")).current_dir(root()).args(args).env("NO_COLOR", "1").output().expect("running presk")
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn make(dir: &Path, args: &[&str]) -> Output {
    Command::new("make").arg("-s").arg("-C").arg(dir).arg(format!("-j{}", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4))).args(args).output().expect("running make")
}

const MLP: &str = "tests/examples/mlp/mlp.psk";

#[test]
fn the_options_of_the_scop_mode_are_checked_before_anything_is_written() {
    let out = scratch("options");
    let o = out.to_str().unwrap();
    let quant = scratch("options-quant");
    std::fs::create_dir_all(&quant).unwrap();
    let r = quant.join("int8.json");
    std::fs::write(&r, r#"{ "default": "f32", "order": { "kind": "sequential" }, "quant": { "l1.w": { "bits": 8, "over": 0, "scheme": "absmax" } } }"#).unwrap();
    let r = format!("@{}", r.display());
    let scop = |extra: &[&'static str]| -> Vec<String> {
        let mut a: Vec<String> = ["export", MLP, "--target", "c", "--mode", "scop", "--entry", "mse", "--dim", "D=4", "--dim", "B=2", "-o", o].iter().map(|s| s.to_string()).collect();
        a.extend(extra.iter().map(|s| s.to_string()));
        a
    };
    let cases: Vec<(Vec<String>, &str)> = vec![
        (["export", MLP, "--target", "c", "--mode", "scop", "-o", o].iter().map(|s| s.to_string()).collect(), "--entry is required"),
        (scop(&["--plan", "@plan.json"]), "`--plan` does not apply"),
        (scop(&["--threads", "4"]), "`--threads` does not apply"),
        (scop(&["--straight-line"]), "`--straight-line` does not apply"),
        (scop(&["--serve"]), "`--serve` does not apply"),
        (scop(&["--weights", "w.psk.d"]), "`--weights` does not apply"),
        (scop(&["--no-exact"]), "`--no-exact` does not apply"),
        (scop(&["--stack", "loop"]), "`--stack` is an option of the PyTorch export"),
        ({
            let mut a = scop(&[]);
            a.push("--realize".into());
            a.push(r.clone());
            a
        }, "quantizes weights"),
        (["export", MLP, "--target", "pytorch", "--mode", "scop", "-o", o].iter().map(|s| s.to_string()).collect(), "`--mode scop` is an option of --target c"),
        (["export", MLP, "--target", "c", "--mode", "graph", "--entry", "mse", "-o", o].iter().map(|s| s.to_string()).collect(), "`--mode graph` is a mode of the PyTorch export; --target c has `--mode scop`"),
    ];
    for (args, expected) in cases {
        let a: Vec<&str> = args.iter().map(String::as_str).collect();
        let r = presk(&a);
        let err = stderr(&r);
        assert_eq!(r.status.code(), Some(2), "{args:?}: {err}");
        assert!(err.contains(expected), "{args:?}: {err}");
        assert!(!out.exists(), "{args:?}: {} was written", out.display());
    }
}

/// One entry of the test set in one type, exported in the SCoP mode.
struct Exported {
    what: String,
    dir: PathBuf,
    nests: Vec<serde_json::Value>,
}

fn export(file: &str, entry: &str, args: &[&str], realize: &str, out: &Path) -> Exported {
    let what = format!("{entry} ({realize})");
    let mut a = vec!["export", file, "--target", "c", "--mode", "scop", "--entry", entry, "--realize", realize, "-o", out.to_str().unwrap()];
    a.extend_from_slice(args);
    let r = presk(&a);
    assert!(r.status.success(), "{what}: {}", stderr(&r));
    let nests: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(out.join("nests.json")).unwrap()).unwrap();
    let files = std::fs::read_dir(out.join("nests")).unwrap().filter(|e| e.as_ref().unwrap().path().extension().is_some_and(|x| x == "c")).count();
    assert_eq!(files, nests.len(), "{what}: a file per nest");
    let summary = stderr(&r);
    assert!(summary.contains(&format!("at {} shapes at point zero", nests.len())), "{what}: {summary}");
    Exported { what, dir: out.to_path_buf(), nests }
}

/// Builds every nest file on its own with GCC and `-Werror`, runs each once
/// against the native backend's outputs, then each once timed.
fn build_and_check(x: &Exported) {
    let built = make(&x.dir, &["CFLAGS=-O2 -Wall -Wextra -Werror", "scop"]);
    assert!(built.status.success() && built.stderr.is_empty(), "{}: {}", x.what, stderr(&built));
    let checked = Command::new(x.dir.join("scop")).arg("-c").arg(x.dir.join("conformance/native.bin")).output().unwrap();
    let said = String::from_utf8_lossy(&checked.stdout);
    assert!(checked.status.success(), "{}: {}{said}", x.what, stderr(&checked));
    assert_eq!(said.trim(), format!("the {} nests compute the native backend's bits", x.nests.len()), "{}", x.what);
    let timed = Command::new(x.dir.join("scop")).args(["-r", "1"]).output().unwrap();
    assert!(timed.status.success(), "{}: {}", x.what, stderr(&timed));
    assert_eq!(String::from_utf8_lossy(&timed.stdout).lines().count(), x.nests.len() + 1, "{}: a line per nest", x.what);
}

fn test_set() -> Vec<(String, String, Vec<String>)> {
    std::fs::read_to_string(root().join("tests/plan/entries.txt"))
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('|').collect();
            (f[0].to_string(), f[1].to_string(), f[2].split_whitespace().map(String::from).collect())
        })
        .collect()
}

fn static_control(n: &serde_json::Value) -> bool {
    n["static_control"].as_bool().unwrap()
}

/// Every entry of the test set in the three types.
fn every_entry(tag: &str) -> Vec<Exported> {
    let base = scratch(tag);
    let mut out = Vec::new();
    for (i, (file, entry, args)) in test_set().iter().enumerate() {
        for realize in ["f64", "f32", "bf16"] {
            let a: Vec<&str> = args.iter().map(String::as_str).collect();
            out.push(export(file, entry, &a, realize, &base.join(format!("{i}-{realize}"))));
        }
    }
    out
}

#[test]
fn every_nest_of_the_test_set_computes_the_native_backends_bits() {
    let started = std::time::Instant::now();
    let all = every_entry("test-set");
    std::thread::scope(|s| {
        for x in &all {
            s.spawn(|| build_and_check(x));
        }
    });
    let mut per_type = std::collections::BTreeMap::new();
    for x in &all {
        for n in &x.nests {
            let t = per_type.entry(n["type"].as_str().unwrap().to_string()).or_insert((0usize, 0usize));
            t.0 += 1;
            if static_control(n) {
                t.1 += 1;
                assert!(n["reason"].is_null(), "{}: {n}", x.what);
            } else {
                // The rule's only misses on the test set: an index loaded
                // from an index array.
                let why = n["reason"].as_str().unwrap();
                assert!(why.contains("a gather") || why.contains("a scatter"), "{}: {n}", x.what);
                assert!(["embed", "embed_vjp", "cross_entropy"].contains(&n["kernel"].as_str().unwrap()), "{}: {n}", x.what);
            }
        }
    }
    // The same nests in `f64` and `f32`; `bf16`'s gradients keep the
    // launches that hold its fusion groups (0.10.0 step 4b, D94).
    let (f64n, f32n, bf16n) = (per_type["f64"], per_type["f32"], per_type["bf16"]);
    assert!(f64n == f32n && bf16n.0 >= f32n.0, "the same nests in f64 and f32, and bf16's held ones besides: {per_type:?}");
    eprintln!("the test set's nests at point zero: {per_type:?} (nests, static control), every one built with -Werror and bit-identical to the native backend; {:.1} s", started.elapsed().as_secs_f64());
}

/// The check checked: two changes of the nests' bits, each of which `make
/// check` reports.
#[test]
fn a_nest_that_departs_from_the_native_backend_fails_the_check() {
    let x = export("tests/examples/minigpt/train_minigpt.psk", "loss", &["--const", "V=64", "--const", "D=16", "--const", "H=2", "--const", "L=2", "--dim", "B=2", "--dim", "T=8"], "f32", &scratch("mutations"));
    build_and_check(&x);
    let check = || Command::new(x.dir.join("scop")).arg("-c").arg(x.dir.join("conformance/native.bin")).output().unwrap();
    // GCC's GNU mode contracts `a + b*c` into one rounding (§3.5, rule 1):
    // on aarch64, which has the instruction, the dots move.
    let _ = make(&x.dir, &["clean"]);
    let built = make(&x.dir, &["STRICT=-std=gnu11", "scop"]);
    assert!(built.status.success(), "{}", stderr(&built));
    let r = check();
    if cfg!(target_arch = "aarch64") {
        assert_eq!(r.status.code(), Some(1), "built with contraction, every nest kept its bits");
        let err = stderr(&r);
        let dots: Vec<&str> = x.nests.iter().filter(|n| n["kernel"].as_str().unwrap().starts_with("dot")).map(|n| n["name"].as_str().unwrap()).collect();
        assert!(!dots.is_empty() && dots.iter().all(|d| err.contains(&format!("{d} ("))), "every dot moves: {err}");
        eprintln!("contracted: {}", err.lines().last().unwrap_or(""));
    }
    // A softmax's running maximum made a minimum: the same softmax in exact
    // arithmetic, other bits.
    let _ = make(&x.dir, &["clean"]);
    let softmax = x.nests.iter().find(|n| n["kernel"].as_str().unwrap().starts_with("softmax")).expect("the loss has a softmax");
    let name = softmax["name"].as_str().unwrap();
    let path = x.dir.join(format!("nests/{name}.c"));
    let text = std::fs::read_to_string(&path).unwrap();
    let mutated = text.replacen("] > s0 ? ", "] < s0 ? ", 1);
    assert_ne!(mutated, text, "{name} selects no maximum");
    std::fs::write(&path, mutated).unwrap();
    let built = make(&x.dir, &["scop"]);
    assert!(built.status.success(), "{}", stderr(&built));
    let r = check();
    assert_eq!(r.status.code(), Some(1), "the minimum kept every bit");
    let err = stderr(&r);
    assert!(err.contains(&format!("{name} (")) && err.contains("1 of the"), "{err}");
}

// ----- Polly (make c-conformance) ---------------------------------------------

/// A clang that links Polly: `POLLY`, or LLVM 23.1.2's in `.llvm` (`make llvm`).
fn polly() -> PathBuf {
    let p = std::env::var_os("POLLY").map(PathBuf::from).unwrap_or_else(|| root().join(".llvm/bin/clang"));
    assert!(p.exists(), "{}: no clang with Polly; `make llvm` installs LLVM 23.1.2's (or set POLLY)", p.display());
    p
}

/// Polly on every nest of an export (`make polly`, Clang with `-Werror`):
/// the rule and Polly agree on every nest, `(static control, not)` counted.
fn agree_with_polly(x: &Exported, clang: &Path) -> (usize, usize) {
    let mut flags: Vec<&str> = scop::POLLY_FLAGS.to_vec();
    flags.extend_from_slice(presk_export_c::WARNINGS);
    flags.push("-Werror");
    let r = make(&x.dir, &[&format!("POLLY={}", clang.display()), &format!("POLLYFLAGS={}", flags.join(" ")), "polly"]);
    assert!(r.status.success() && r.stderr.is_empty(), "{}: {}", x.what, stderr(&r));
    let mut counts = (0, 0);
    for n in &x.nests {
        let name = n["name"].as_str().unwrap();
        let scops = scop::polly_scops(&std::fs::read_to_string(x.dir.join(format!("nests/{name}.polly"))).unwrap(), name);
        let loops = n["loops"].as_u64().unwrap() as usize;
        let one = scops == [loops];
        assert_eq!(one, static_control(n), "{}: {name} ({}, {}): {} loops, Polly's SCoPs {scops:?}", x.what, n["kernel"], n["operands"], loops);
        if static_control(n) {
            counts.0 += 1;
        } else {
            counts.1 += 1;
            // Allowed a non-affine access, which it over-approximates by the
            // whole axis, Polly takes the nest whole.
            let out = Command::new(clang).args(&flags).args(["-mllvm", "-polly-allow-nonaffine", "-c", "-o", "/dev/null"]).arg(x.dir.join(format!("nests/{name}.c"))).output().unwrap();
            assert!(out.status.success(), "{}", stderr(&out));
            assert_eq!(scop::polly_scops(&String::from_utf8_lossy(&out.stdout), name), vec![loops], "{}: {name} with non-affine accesses", x.what);
        }
    }
    (counts.0, counts.1)
}

#[test]
#[ignore = "needs a clang with Polly: make llvm (make c-conformance)"]
fn every_static_control_nest_is_one_scop_for_polly() {
    let clang = polly();
    let all = every_entry("polly");
    let counts: Vec<(usize, usize)> = std::thread::scope(|s| {
        let hs: Vec<_> = all.iter().map(|x| s.spawn(|| agree_with_polly(x, &clang))).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let (st, not): (usize, usize) = counts.iter().fold((0, 0), |a, c| (a.0 + c.0, a.1 + c.1));
    eprintln!("Polly on the test set's nests in three types: {st} static-control nests each one SCoP of the kernel's loops, {not} gathers and scatters none (one SCoP with non-affine accesses allowed)");
}

#[test]
#[ignore = "the course GPT at its constants, and Polly: make c-conformance"]
fn the_course_gpts_nests_compute_the_native_backends_bits() {
    let clang = polly();
    let consts = ["--const", "V=92", "--const", "D=384", "--const", "H=6", "--const", "L=6"];
    let mut runs: Vec<(Vec<&str>, &str)> = ["f64", "f32", "bf16"].iter().map(|t| (vec!["--dim", "B=1", "--dim", "T=256"], *t)).collect();
    runs.push((vec!["--dim", "B=16", "--dim", "T=256"], "f32"));
    let base = scratch("course");
    for (i, (dims, realize)) in runs.iter().enumerate() {
        let mut a: Vec<&str> = consts.to_vec();
        a.extend_from_slice(dims);
        let x = export("corpus/nlp/05_gpt.psk", "GPT.apply", &a, realize, &base.join(i.to_string()));
        build_and_check(&x);
        let (st, not) = agree_with_polly(&x, &clang);
        eprintln!("the course GPT at {} in {realize}: {} nests bit-identical, {st} static control and one SCoP each for Polly, {not} gathers", dims.join(" "), x.nests.len());
    }
}
