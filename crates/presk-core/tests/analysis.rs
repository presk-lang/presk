//! Regularity and effects: summaries of the examples are snapshotted, the
//! `reg_*`/`eff_*` corpus yields its diagnostics, and a C⁰⁺ path is accepted
//! by `grad`.

use std::path::{Path, PathBuf};
use std::sync::Once;

use presk_syntax::diag::render_all;
use presk_syntax::{SourceMap, parse_file};
use presk_types::check;
use presk_types::elab::{self, Options};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn chdir_root() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| std::env::set_current_dir(repo_root()).unwrap());
}

/// Front-end up to the analyses; returns the analysis summary and all diagnostics.
fn analyze(rel: &str) -> (SourceMap, String, Vec<presk_syntax::Diagnostic>) {
    chdir_root();
    let mut sm = SourceMap::new();
    let id = sm.load(Path::new(rel)).unwrap();
    let (program, mut diags) = parse_file(&sm, id);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (prog, more) = elab::elaborate(&mut sm, id, program, &Options::default());
    diags.extend(more);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (typed, more) = check::check(&sm, &prog);
    diags.extend(more);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (core, more) = presk_core::lower(&prog, &typed);
    diags.extend(more);
    presk_core::verify(&core).unwrap();
    let (analysis, more) = presk_core::analyze(&core, &prog);
    diags.extend(more);
    (sm, presk_core::analysis::summary(&core, &analysis), diags)
}

#[test]
fn examples_analysis() {
    for rel in ["mlp/mlp.psk", "layernorm/layernorm.psk", "minigpt/minigpt.psk", "minigpt/train_minigpt.psk", "minigpt/serve_minigpt.psk"] {
        let (sm, summary, diags) = analyze(&format!("tests/examples/{rel}"));
        assert!(diags.iter().all(|d| d.code == "W0001"), "{rel}:\n{}", render_all(&diags, &sm, false));
        let name = format!("analysis_{}", Path::new(rel).file_stem().unwrap().to_str().unwrap());
        insta::assert_snapshot!(name, summary);
    }
}

#[test]
fn diagnostics_corpus() {
    chdir_root();
    insta::glob!(repo_root().join("tests/diag"), "{reg,eff}_*.psk", |path| {
        let rel = path.strip_prefix(repo_root()).unwrap().display().to_string();
        let (sm, _summary, diags) = analyze(&rel);
        assert!(diags.iter().any(|d| d.is_error()), "{rel} produced no error");
        insta::assert_snapshot!(render_all(&diags, &sm, false));
    });
}

#[test]
fn piecewise_smooth_is_differentiable() {
    let (sm, summary, diags) = analyze("tests/analysis_ok_relu.psk");
    assert!(diags.is_empty(), "{}", render_all(&diags, &sm, false));
    assert!(summary.contains("f : [x C⁰⁺]"), "{summary}");
}
