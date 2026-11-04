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
    for rel in ["mlp/mlp.psk", "layernorm/layernorm.psk", "minigpt/minigpt.psk", "minigpt/train_minigpt.psk", "minigpt/serve_minigpt.psk", "concat/concat.psk", "rope/rope.psk"] {
        let (sm, summary, diags) = run(&format!("tests/examples/{rel}"));
        assert!(diags.iter().all(|d| d.code == "W0001" || d.code == "W0002"), "{rel}:\n{}", render_all(&diags, &sm, false));
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

/// `train_*.psk`: the `training run` block against its objective (0.2.0 step 2).
#[test]
fn training_corpus() {
    insta::glob!(repo_root().join("tests/diag"), "train_*.psk", |path| {
        let rel = path.strip_prefix(repo_root()).unwrap().display().to_string();
        let (sm, _summary, diags) = run(&rel);
        let errors: Vec<_> = diags.iter().filter(|d| d.is_error()).collect();
        assert!(!errors.is_empty(), "{rel} produced no error");
        insta::assert_snapshot!(render_all(&diags, &sm, false));
    });
}

/// `serve_*.psk`: the `inference serve` block (0.3.0 step 5): W0003 when the
/// file has no generation function.
#[test]
fn serving_corpus() {
    insta::glob!(repo_root().join("tests/diag"), "serve_*.psk", |path| {
        let rel = path.strip_prefix(repo_root()).unwrap().display().to_string();
        let (sm, _summary, diags) = run(&rel);
        assert!(!diags.is_empty(), "{rel} produced no diagnostic");
        insta::assert_snapshot!(render_all(&diags, &sm, false));
    });
}

/// A valid `training run` with its data file present: the record binds `B`
/// from the pipeline and `T` from the file, and no diagnostic is emitted.
#[test]
fn training_record() {
    chdir_root();
    let mut sm = SourceMap::new();
    let id = sm.load(Path::new("tests/data/train_ok.psk")).unwrap();
    let (program, diags) = parse_file(&sm, id);
    assert!(diags.is_empty());
    let (prog, diags) = elab::elaborate(&mut sm, id, program, &Options::default());
    assert!(diags.is_empty(), "{}", render_all(&diags, &sm, false));
    let (typed, diags) = check::check(&sm, &prog);
    assert!(diags.is_empty(), "{}", render_all(&diags, &sm, false));
    assert_eq!(typed.training.len(), 1);
    let run = &typed.training[0];
    let d = run.data.as_ref().unwrap();
    assert_eq!((d.shuffle, d.batch, d.take), (Some(1), 4, Some(2)));
    assert_eq!(d.file_ty.as_deref(), Some("(idx[64][seq: 8], idx[64][seq: 8])"));
    assert_eq!(run.epochs, Some(3));
    assert_eq!(run.checkpoint, Some((10, "ckpt/".to_string())));
    let dims: Vec<(String, u64)> = run.dims.iter().map(|(v, k)| (prog.dim_name(*v), *k)).collect();
    assert_eq!(dims, vec![("B".to_string(), 4), ("T".to_string(), 8)]);
    insta::assert_snapshot!(check::training::summary(&prog, &typed.training));
}
