//! The generated Python of the five examples is snapshotted and, when a
//! `python3` is available, byte-compiled (PyTorch itself is not required).

use std::path::{Path, PathBuf};
use std::sync::Once;

use presk_export_pytorch::{Options, StackMode, export};
use presk_syntax::diag::render_all;
use presk_syntax::{SourceMap, parse_file};
use presk_types::check;
use presk_types::elab::{self, Options as ElabOptions};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn chdir_root() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| std::env::set_current_dir(repo_root()).unwrap());
}

fn generated(rel: &str, opts: &Options) -> Vec<(String, String)> {
    chdir_root();
    let mut sm = SourceMap::new();
    let id = sm.load(Path::new(rel)).unwrap();
    let (program, diags) = parse_file(&sm, id);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (prog, diags) = elab::elaborate(&mut sm, id, program, &ElabOptions::default());
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (typed, diags) = check::check(&sm, &prog);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (files, diags) = export(&sm, &prog, &typed, opts);
    assert!(diags.is_empty(), "{rel}: export diagnostics:\n{}", render_all(&diags, &sm, false));
    files.into_iter().map(|g| (g.name, g.content)).collect()
}

fn py_compile(dir: &Path) {
    let Ok(out) = std::process::Command::new("python3").arg("-m").arg("py_compile").args(std::fs::read_dir(dir).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "py"))).output() else {
        eprintln!("python3 not found: skipping byte-compilation");
        return;
    };
    assert!(out.status.success(), "py_compile failed:\n{}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn export_examples() {
    let tmp = std::env::temp_dir().join(format!("presk-export-{}", std::process::id()));
    for (rel, opts) in [
        ("mlp/mlp.psk", Options::default()),
        ("layernorm/layernorm.psk", Options::default()),
        ("minigpt/minigpt.psk", Options { shapes_check: true, stack: StackMode::Loop }),
        ("minigpt/train_minigpt.psk", Options::default()),
        ("minigpt/serve_minigpt.psk", Options { shapes_check: false, stack: StackMode::Unroll }),
    ] {
        let files = generated(&format!("tests/examples/{rel}"), &opts);
        let stem = Path::new(rel).file_stem().unwrap().to_str().unwrap();
        let dir = tmp.join(stem);
        std::fs::create_dir_all(&dir).unwrap();
        for (name, content) in &files {
            std::fs::write(dir.join(name), content).unwrap();
        }
        py_compile(&dir);
        // Snapshot the main file only (imports are snapshotted by their own example).
        let main = files.iter().find(|(n, _)| *n == format!("{stem}.py")).unwrap();
        insta::assert_snapshot!(format!("export_{stem}"), main.1);
    }
    let _ = std::fs::remove_dir_all(&tmp);
}
