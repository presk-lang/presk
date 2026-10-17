//! X3 (plan 0.11.0): the figures of a `presk explain` document are the
//! commands' own. Every line `presk calibrate --precision`, `presk dump
//! --stage sensitivity`, `presk solve`, `presk check --precision`, `presk
//! plan` and `presk dump --stage kernel` print is found verbatim in the
//! document, but for what holds a time or a path (the pricing's and the
//! capture's seconds, the wall time, the file written, the envelope's path,
//! which the document names `$DOC/…`); and two runs write the same bytes.

use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn presk(args: &[&str], threads: bool) -> String {
    let mut c = Command::new(env!("CARGO_BIN_EXE_presk"));
    c.current_dir(root()).args(args).env("NO_COLOR", "1");
    if threads {
        c.env("PRESK_THREADS", "10");
    }
    let o = c.output().expect("running presk");
    assert!(o.status.success(), "{args:?}: {}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8_lossy(&o.stdout).to_string()
}

/// Every line of `out` but those `skip` names is a line of `doc`.
fn quoted(doc: &str, out: &str, what: &str, skip: impl Fn(&str) -> bool) {
    let lines: std::collections::BTreeSet<&str> = doc.lines().collect();
    let mut n = 0;
    for l in out.lines().filter(|l| !l.is_empty() && !skip(l)) {
        assert!(lines.contains(l), "{what}: the document does not have the line\n{l}");
        n += 1;
    }
    assert!(n > 0, "{what}: no line compared");
}

#[test]
fn the_document_quotes_the_commands() {
    const FILE: &str = "tutorial/07_train.psk";
    const PROFILE: &str = "crates/presk-explain/tests/data/x925-frozen.json";
    let dir = std::env::temp_dir().join(format!("presk-explain-x3-{}", std::process::id()));
    let (a, b) = (dir.join("a"), dir.join("b"));
    let explain = |out: &Path| presk(&["explain", FILE, "--entry", "loss", "--dim", "B=4", "--budget", "1e-3", "--profile", PROFILE, "--threads", "10", "--out", out.to_str().unwrap()], false);
    explain(&a);
    explain(&b);
    let doc = std::fs::read_to_string(a.join("explain.md")).unwrap();
    assert_eq!(doc, std::fs::read_to_string(b.join("explain.md")).unwrap(), "two runs, two documents");
    let art = |n: &str| a.join("artifacts").join(n).to_str().unwrap().to_string();
    let (env, realization, plan) = (art("envelope.json"), format!("@{}", art("realization.json")), format!("@{}", art("plan.json")));

    let calibrate = presk(&["calibrate", "--precision", FILE, "--entry", "loss", "--dim", "B=4", "--bits", "128"], false);
    quoted(&doc, &calibrate, "presk calibrate", |_| false);
    let sens = presk(&["dump", FILE, "--stage", "sensitivity", "--entry", "loss", "--dim", "B=4", "--sites", "30"], false);
    quoted(&doc, &sens, "presk dump --stage sensitivity", |_| false);
    let solve = presk(&["solve", FILE, "--precision", &env, "--budget", "1e-3", "--profile", PROFILE, "--sites", "0"], false);
    quoted(&doc, &solve, "presk solve", |_| false);
    // Under a profile the costs are times: until 0.11.0 the exhaustive
    // search's line called them bytes, and its ratio read 0.000x. Since
    // 0.12.0 the tutorial's space, with its four parameters, is past the
    // search's size; the MLP's is not.
    let mlp_env = dir.join("mlp_env.json");
    presk(&["calibrate", "--precision", "tests/examples/mlp/mlp.psk", "--entry", "mse", "--dim", "D=4", "--dim", "B=2", "--bits", "128", "--out", mlp_env.to_str().unwrap()], false);
    let mlp = presk(&["solve", "tests/examples/mlp/mlp.psk", "--precision", mlp_env.to_str().unwrap(), "--budget", "1e-3", "--profile", PROFILE, "--sites", "0"], false);
    assert!(mlp.contains(" s by exhaustive search") && !mlp.contains("0.000x"), "{mlp}");
    let check = presk(&["check", "--precision", &env, FILE, "--realize", &realization, "--model", "all", "--sites", "8", "--measure"], false);
    quoted(&doc, &check, "presk check --precision", |l| l.starts_with("  envelope "));
    let planned = presk(&["plan", FILE, "--entry", "loss", "--dim", "B=4", "--realize", &realization, "--profile", PROFILE], true);
    quoted(&doc, &planned, "presk plan", |l| l.starts_with("-- wall time") || l.contains(" points costed in "));
    // The totals' line without its seconds.
    let totals = planned.lines().find(|l| l.contains(" points costed in ")).unwrap();
    let (head, tail) = totals.split_once(" costed in ").unwrap();
    let u = tail.rsplit_once("; ").unwrap().1;
    assert!(doc.lines().any(|l| l == format!("{head} costed; {u}")), "presk plan: {totals}");
    let kernels = presk(&["dump", FILE, "--stage", "kernel", "--entry", "loss", "--dim", "B=4", "--plan", &plan], false);
    quoted(&doc, &kernels, "presk dump --stage kernel", |_| false);
    std::fs::remove_dir_all(&dir).ok();
}

/// `--measure` (plan 0.11.0 step 5): the realization's gap natively against
/// the exact reference is the solver's, which measured it on the kernel
/// interpreter (U5: the two compute the same bits); the section's times are
/// not compared, they change from run to run. A site named by `--site` is
/// traced through every stage.
#[test]
fn the_measured_gap_is_the_solver_s() {
    const FILE: &str = "tutorial/07_train.psk";
    const PROFILE: &str = "crates/presk-explain/tests/data/x925-frozen.json";
    let dir = std::env::temp_dir().join(format!("presk-explain-measure-{}", std::process::id()));
    presk(&["explain", FILE, "--entry", "loss", "--dim", "B=4", "--budget", "1e-3", "--profile", PROFILE, "--threads", "2", "--measure", "--site", "Linear.apply/%3", "--out", dir.to_str().unwrap()], false);
    let doc = std::fs::read_to_string(dir.join("explain.md")).unwrap();
    let solver = doc.lines().find(|l| l.starts_with("  measured    ")).and_then(|l| l.split_whitespace().nth(1)).unwrap().to_string();
    let row = doc.lines().find(|l| l.starts_with("| Gap, native against the exact reference |")).unwrap_or_else(|| panic!("no gap row:\n{doc}"));
    assert!(row.contains(&format!("| {solver}, the solver's measured gap")), "{row} (the solver: {solver})");
    assert!(doc.contains("| A call, native, 2 threads |"), "{doc}");
    assert!(doc.contains("**`Linear.apply/%3`**") && doc.contains("| Kernel (§9) | `k5`:"), "{doc}");
    std::fs::remove_dir_all(&dir).ok();
}

/// The manual's chapter `explain_loss` is the document `presk explain`
/// writes for the tutorial's `loss` (`make explain`), its metadata block a
/// heading: this test holds it current, and `make explain-manual`
/// (`PRESK_UPDATE_MANUAL=1`) writes it again (plan 0.11.0 step 6).
#[test]
fn the_manual_holds_the_tutorial_s_document() {
    let dir = std::env::temp_dir().join(format!("presk-explain-manual-{}", std::process::id()));
    presk(&["explain", "tutorial/07_train.psk", "--entry", "loss", "--dim", "B=4", "--budget", "1e-3", "--profile", "crates/presk-explain/tests/data/x925-frozen.json", "--threads", "10", "--target", "native,c,pytorch", "--out", dir.to_str().unwrap()], false);
    let doc = std::fs::read_to_string(dir.join("explain.md")).unwrap();
    let body = doc.strip_prefix("---\n").and_then(|d| d.split_once("\n---\n")).map(|(meta, body)| (meta.to_string(), body.to_string())).expect("a metadata block");
    let field = |name: &str| body.0.lines().find_map(|l| l.strip_prefix(&format!("{name}: \""))).and_then(|v| v.strip_suffix('"')).unwrap_or("").replace("\\\"", "\"");
    let chapter = format!(
        "# {}\n\n*{}*\n\n> **Generated.** This chapter is the document `presk explain` writes for the tutorial's `loss` (`make explain`, `make explain-pdf` for its PDF), as it is: a test holds it current, and `make explain-manual` writes it again. Its listings and artifacts are written beside the document, not into the manual. See the `explain` chapter.\n{}",
        field("title"),
        field("subtitle"),
        body.1
    );
    let path = root().join("crates/presk-manual/manual/explain_loss.md");
    if std::env::var_os("PRESK_UPDATE_MANUAL").is_some() {
        std::fs::write(&path, &chapter).unwrap();
    } else {
        let have = std::fs::read_to_string(&path).unwrap_or_default();
        assert!(have == chapter, "crates/presk-manual/manual/explain_loss.md is not the document presk explain writes now: `make explain-manual` writes it again");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// `explain.json` is the same from one process to the next under the native
/// target (plan 0.14.0, D145): each kernel's machine code is kept before it
/// is linked, where a kernel that calls `libm` held the function's address,
/// which the loader places differently in each process.
#[test]
fn the_json_is_the_same_from_one_run_to_the_next() {
    let base = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("explain-twice");
    let mut texts = Vec::new();
    for k in ["a", "b"] {
        let out = base.join(k);
        let o = out.to_str().unwrap();
        presk(&["explain", "tutorial/07_train.psk", "--entry", "loss", "--dim", "B=4", "--realize", "f32", "--no-exact", "--target", "native", "--format", "json", "--out", o], false);
        texts.push(std::fs::read_to_string(out.join("explain.json")).unwrap());
    }
    assert!(texts[0].contains("\"relocs\": [\n") || texts[0].contains("\"relocs\""), "no relocations recorded");
    assert!(texts[0] == texts[1], "explain.json differs between two runs");
}
