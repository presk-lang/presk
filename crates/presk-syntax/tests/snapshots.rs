//! Snapshot tests: tokens of layernorm.psk, ASTs of the five example
//! programs, and the rendered diagnostic of every `examples/tests/diag/*.psk` file
//! (one diagnostic corpus entry per file, ASCII rendering, no color).

use std::path::{Path, PathBuf};

use presk_syntax::diag::render_all;
use presk_syntax::{SourceMap, lexer, parse_file};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn example(rel: &str) -> PathBuf {
    repo_root().join("examples/spec").join(rel)
}

/// Load a file under a repo-relative name so snapshots do not depend on the
/// absolute path of the checkout.
fn load(sm: &mut SourceMap, path: &Path) -> presk_syntax::FileId {
    let text = std::fs::read_to_string(path).unwrap();
    let name = path.strip_prefix(repo_root()).unwrap().display().to_string();
    sm.add(name, text)
}

#[test]
fn tokens_layernorm() {
    let mut sm = SourceMap::new();
    let id = load(&mut sm, &example("layernorm/layernorm.psk"));
    let lexed = lexer::lex(id, sm.text(id));
    assert!(lexed.diags.is_empty(), "{}", render_all(&lexed.diags, &sm, false));
    insta::assert_snapshot!("tokens_layernorm", lexer::dump_tokens(&lexed.tokens));
}

#[test]
fn ast_examples() {
    for rel in ["mlp/mlp.psk", "layernorm/layernorm.psk", "minigpt/minigpt.psk", "minigpt/train_minigpt.psk", "minigpt/serve_minigpt.psk"] {
        let mut sm = SourceMap::new();
        let id = load(&mut sm, &example(rel));
        let (program, diags) = parse_file(&sm, id);
        assert!(diags.is_empty(), "{rel}:\n{}", render_all(&diags, &sm, false));
        let name = format!("ast_{}", Path::new(rel).file_stem().unwrap().to_str().unwrap());
        insta::assert_snapshot!(name, format!("{program:#?}"));
    }
}

#[test]
fn diagnostics_corpus() {
    insta::glob!(repo_root().join("examples/tests/diag"), "parse_*.psk", |path| {
        let mut sm = SourceMap::new();
        let id = load(&mut sm, path);
        let (_program, diags) = parse_file(&sm, id);
        assert!(!diags.is_empty(), "{} produced no diagnostic", path.display());
        insta::assert_snapshot!(render_all(&diags, &sm, false));
    });
}

/// The ASCII spellings (`Real`, `\`, `@`, `><`, `<<`) parse to the same AST as
/// the Unicode ones; only spans differ.
#[test]
fn ascii_spellings() {
    fn ast_without_spans(path: &Path) -> String {
        let mut sm = SourceMap::new();
        let id = load(&mut sm, path);
        let (program, diags) = parse_file(&sm, id);
        assert!(diags.is_empty(), "{}:\n{}", path.display(), render_all(&diags, &sm, false));
        let text = format!("{program:#?}");
        // Spans print as `start..end`; blank them out.
        let mut out = String::new();
        for line in text.lines() {
            if line.contains("..") && line.trim_start().chars().next().is_some_and(|c| c.is_ascii_digit() || line.contains("span")) {
                continue;
            }
            out.push_str(line);
            out.push('\n');
        }
        out
    }
    let dir = repo_root().join("examples/tests/ascii");
    assert_eq!(ast_without_spans(&example("mlp/mlp.psk")), ast_without_spans(&dir.join("mlp_ascii.psk")));
    assert_eq!(ast_without_spans(&dir.join("symbols_unicode.psk")), ast_without_spans(&dir.join("symbols.psk")));
}
