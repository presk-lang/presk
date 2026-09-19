//! Shape typing: the five examples pass with only `W0001`, their
//! `--summary` is snapshotted, and every `shape_*.psk` of the corpus yields
//! its diagnostic.

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

fn run(rel: &str) -> (SourceMap, String, Vec<presk_syntax::Diagnostic>) {
    chdir_root();
    let mut sm = SourceMap::new();
    let id = sm.load(Path::new(rel)).unwrap();
    let (program, mut diags) = parse_file(&sm, id);
    assert!(diags.iter().all(|d| !d.is_error()), "{rel} does not parse:\n{}", render_all(&diags, &sm, false));
    let (prog, more) = elab::elaborate(&mut sm, id, program, &Options::default());
    diags.extend(more);
    assert!(diags.iter().all(|d| !d.is_error()), "{rel} does not elaborate:\n{}", render_all(&diags, &sm, false));
    let (typed, more) = check::check(&sm, &prog);
    diags.extend(more);
    let summary = check::summary(&prog, &typed);
    (sm, summary, diags)
}

#[test]
fn examples_type_check() {
    for rel in ["mlp/mlp.psk", "layernorm/layernorm.psk", "minigpt/minigpt.psk", "minigpt/train_minigpt.psk", "minigpt/serve_minigpt.psk"] {
        let (sm, summary, diags) = run(&format!("tests/examples/{rel}"));
        assert!(diags.iter().all(|d| d.code == "W0001"), "{rel}:\n{}", render_all(&diags, &sm, false));
        let name = format!("summary_{}", Path::new(rel).file_stem().unwrap().to_str().unwrap());
        insta::assert_snapshot!(name, summary);
    }
}

#[test]
fn diagnostics_corpus() {
    chdir_root();
    insta::glob!(repo_root().join("tests/diag"), "shape_*.psk", |path| {
        let rel = path.strip_prefix(repo_root()).unwrap().display().to_string();
        let (sm, _summary, diags) = run(&rel);
        let errors: Vec<_> = diags.iter().filter(|d| d.is_error()).collect();
        assert!(!errors.is_empty(), "{rel} produced no error");
        insta::assert_snapshot!(render_all(&diags, &sm, false));
    });
}
