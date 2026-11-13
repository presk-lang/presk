//! Elaboration snapshots of the five examples and the `elab_*` diagnostic
//! corpus. Paths are made relative to the repository root so snapshots do
//! not depend on the checkout location.

use std::path::{Path, PathBuf};
use std::sync::Once;

use presk_syntax::diag::render_all;
use presk_syntax::{SourceMap, parse_file};
use presk_types::elab::{self, Options};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

/// Imports are resolved relative to the file's path as given, so the tests
/// run from the repository root with relative paths.
fn chdir_root() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| std::env::set_current_dir(repo_root()).unwrap());
}

fn run(rel: &str, overrides: Vec<(String, u64)>) -> (SourceMap, elab::ElabProgram, Vec<presk_syntax::Diagnostic>) {
    chdir_root();
    let mut sm = SourceMap::new();
    let id = sm.load(Path::new(rel)).unwrap();
    let (program, mut diags) = parse_file(&sm, id);
    assert!(diags.iter().all(|d| !d.is_error()), "{rel} does not parse:\n{}", render_all(&diags, &sm, false));
    let (prog, more) = elab::elaborate(&mut sm, id, program, &Options { const_overrides: overrides });
    diags.extend(more);
    (sm, prog, diags)
}

#[test]
fn elab_examples() {
    for rel in ["mlp/mlp.psk", "layernorm/layernorm.psk", "minigpt/minigpt.psk", "minigpt/train_minigpt.psk", "minigpt/serve_minigpt.psk"] {
        let (sm, prog, diags) = run(&format!("examples/spec/{rel}"), vec![]);
        assert!(diags.iter().all(|d| !d.is_error()), "{rel}:\n{}", render_all(&diags, &sm, false));
        assert!(diags.iter().all(|d| d.code == "W0001"), "{rel}: unexpected warning:\n{}", render_all(&diags, &sm, false));
        let name = format!("elab_{}", Path::new(rel).file_stem().unwrap().to_str().unwrap());
        insta::assert_snapshot!(name, format!("{}\n-- diagnostics --\n{}", elab::dump(&prog), render_all(&diags, &sm, false)));
    }
}

#[test]
fn const_override_fails_assert() {
    let (sm, _prog, diags) = run("examples/spec/minigpt/minigpt.psk", vec![("H".into(), 7)]);
    let rendered = render_all(&diags, &sm, false);
    assert!(diags.iter().any(|d| d.code == "E1004"), "{rendered}");
    assert!(diags.iter().any(|d| d.code == "E1005"), "{rendered}");
    insta::assert_snapshot!("const_override_h7", rendered);
}

#[test]
fn diagnostics_corpus() {
    chdir_root();
    insta::glob!(repo_root().join("examples/tests/diag"), "elab_*.psk", |path| {
        let rel = path.strip_prefix(repo_root()).unwrap().display().to_string();
        let (sm, _prog, diags) = run(&rel, vec![]);
        assert!(!diags.is_empty(), "{rel} produced no diagnostic");
        insta::assert_snapshot!(render_all(&diags, &sm, false));
    });
}
