//! `examples/tests/diag/run_*.psk`: an entry refused at the dimensions where an
//! operation of its body is undefined (`E7003`, plan 0.10.0 step 7b), through
//! the binary. Each file's header names its commands: `--   presk …`, each
//! refused, and `-- setup: presk …`, run first, which must succeed; `$OUT`
//! is a scratch directory. Every command runs as written, on the reference
//! interpreter, and again on the kernel backend and natively; a `presk run`
//! also as `presk export --target c`, which writes nothing, and as the
//! PyTorch export's conformance run. Every one refuses with the same report,
//! which the snapshot holds once per command.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn presk(args: &[String]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_presk")).current_dir(root()).args(args).env("NO_COLOR", "1").output().expect("running presk")
}

/// The report of a refusal: from its code to the end of what was printed.
fn report(o: &Output, what: &str) -> String {
    let err = String::from_utf8_lossy(&o.stderr).to_string();
    assert_eq!(o.status.code(), Some(1), "{what}: exit {:?}\n{err}", o.status.code());
    let at = err.find("[E7003]").unwrap_or_else(|| panic!("{what}: no E7003\n{err}"));
    err[at..].to_string()
}

fn words(line: &str, out: &Path) -> Vec<String> {
    line.split_whitespace().map(|w| w.replace("$OUT", out.to_str().unwrap())).collect()
}

/// The same command with `flag value` added.
fn with(args: &[String], extra: &[&str]) -> Vec<String> {
    args.iter().cloned().chain(extra.iter().map(|s| s.to_string())).collect()
}

#[test]
fn refusals_at_dimensions() {
    insta::glob!(root().join("examples/tests/diag"), "run_*.psk", |path| {
        let text = std::fs::read_to_string(path).unwrap();
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let out = std::env::temp_dir().join(format!("presk-run-diag-{}-{stem}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        std::fs::create_dir_all(&out).unwrap();
        for line in text.lines().filter_map(|l| l.strip_prefix("-- setup: presk ")) {
            let o = presk(&words(line, &out));
            assert!(o.status.success(), "setup `{line}`: {}", String::from_utf8_lossy(&o.stderr));
        }
        let mut snap = String::new();
        let lines: Vec<&str> = text.lines().filter_map(|l| l.strip_prefix("--   presk ")).collect();
        assert!(!lines.is_empty(), "{stem}: no command in its header");
        for line in lines {
            let args = words(line, &out);
            let base = report(&presk(&args), line);
            let mut variants: Vec<(String, Vec<String>)> = ["kernel", "native"].iter().map(|b| (format!("{line} --backend {b}"), with(&args, &["--backend", b]))).collect();
            if args[0] == "run" {
                let c = out.join("c");
                let mut export = vec!["export".to_string(), args[1].clone(), "--target".into(), "c".into(), "-o".into(), c.display().to_string()];
                export.extend(args[2..].iter().cloned());
                variants.push(("the C export".into(), export));
                let mut py = vec!["export".to_string(), args[1].clone(), "--target".into(), "pytorch".into(), "-o".into(), out.join("py").display().to_string()];
                py.extend(args[2..].iter().map(|a| if a == "--entry" { "--conformance-entry".to_string() } else { a.clone() }));
                variants.push(("the PyTorch export's conformance run".into(), py));
            }
            for (what, v) in &variants {
                let r = report(&presk(v), what);
                assert_eq!(r, base, "{stem}: `{what}` reports otherwise than `{line}`");
            }
            assert!(!out.join("c").exists(), "{stem}: the C export wrote {}", out.join("c").display());
            snap.push_str(&format!("$ presk {line}\n{base}\n"));
        }
        insta::assert_snapshot!(snap);
        let _ = std::fs::remove_dir_all(&out);
    });
}

/// What is defined at a zero length runs on every executor: a softmax over
/// an axis of length 0 is empty, and a sum over it is 0. A call and a closure
/// applied are walked for what they compute (`examples/tests/data/defined.psk`).
#[test]
fn what_is_defined_at_a_zero_length_runs() {
    let file = "examples/tests/data/defined.psk".to_string();
    let run = |entry: &str, b: &str| presk(&["run".into(), file.clone(), "--entry".into(), entry.into(), "--dim".into(), "T=0".into(), "--backend".into(), b.into()]);
    for b in ["interp", "kernel", "native"] {
        let o = run("fine", b);
        assert!(o.status.success(), "{b}: {}", String::from_utf8_lossy(&o.stderr));
        assert_eq!(String::from_utf8_lossy(&o.stdout).trim(), "ℝ[model: 4] = [model: 4] [0, 0, 0, 0]", "{b}");
        let e = report(&run("Tower.apply", b), b);
        assert!(e.contains("in `Layer.apply`, `mean axis seq` averages over the axis `seq`, of length 0 (`T` = 0)"), "{b}: {e}");
        let e = report(&run("through", b), b);
        assert!(e.contains("in `centered`, `mean axis seq`"), "{b}: {e}");
    }
}
