//! D199 (plan 0.17.2 step 2): every listing of Presk in the manual is a
//! program of `examples/` or a part of one, or says why it is not. A block
//! is preceded by a line `<!-- psk: … -->` that names its source:
//!
//! - `<!-- psk: examples/PATH -->`: the block is the file;
//! - `<!-- psk: examples/PATH#NAME -->`: the block is the file's part
//!   `NAME`, the lines between `-- quote: NAME` and the next `-- end quote`
//!   (the report's convention, plan 0.17.2 step 3), or, when the file has
//!   no such part, its declaration `NAME` as the parser's span holds it
//!   (`training run` and `inference serve` for the unnamed blocks);
//! - either, followed by `, same tokens`: the block has the source's tokens
//!   and may differ from it in comments and layout;
//! - `<!-- psk: check examples/PATH -->`: the block is what `presk check
//!   examples/PATH` prints;
//! - `<!-- psk: none (why) -->`: the block is not a program.
//!
//! A block that parses as Presk, with at least one declaration, and has no
//! marker fails. `PRESK_UPDATE_MANUAL=1` (`make manual-quotes`) writes the
//! blocks again from their sources, but for those of the same tokens.

use std::path::{Path, PathBuf};
use std::process::Command;

use presk_syntax::ast::Decl;
use presk_syntax::token::TokKind;
use presk_syntax::{lexer, parse_file, FileId, SourceMap};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

/// A fenced block of a chapter: the line of its opening fence, its
/// indentation, the marker before it, and its lines without the indentation.
struct Block {
    fence: usize,
    indent: String,
    tag: String,
    marker: Option<String>,
    body: Vec<String>,
}

fn blocks(lines: &[&str]) -> Vec<Block> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let t = lines[i].trim_start();
        let Some(tag) = t.strip_prefix("```") else {
            i += 1;
            continue;
        };
        let indent = lines[i][..lines[i].len() - t.len()].to_string();
        let marker = (i > 0)
            .then(|| lines[i - 1].trim())
            .and_then(|l| l.strip_prefix("<!-- psk:"))
            .and_then(|l| l.strip_suffix("-->"))
            .map(|m| m.trim().to_string());
        let mut j = i + 1;
        while j < lines.len() && lines[j].trim() != "```" {
            j += 1;
        }
        let body = lines[i + 1..j].iter().map(|l| l.strip_prefix(indent.as_str()).unwrap_or(l.trim_start()).to_string()).collect();
        out.push(Block { fence: i, indent, tag: tag.trim().to_string(), marker, body });
        i = j + 1;
    }
    out
}

/// The program's declarations, when `text` parses as Presk without an error.
fn declarations(text: &str) -> Option<(SourceMap, Vec<Decl>)> {
    let mut sm = SourceMap::new();
    let id = sm.add("block.psk", text);
    let (program, diags) = parse_file(&sm, id);
    diags.iter().all(|d| !d.is_error()).then_some((sm, program.decls))
}

/// The text of `file`'s part or declaration `name`, without its last newline.
fn part(file: &str, text: &str, name: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if let Some(first) = lines.iter().position(|l| *l == format!("-- quote: {name}")) {
        let last = lines[first + 1..].iter().position(|l| *l == "-- end quote").unwrap_or_else(|| panic!("{file}: the part {name} has no `-- end quote`"));
        return lines[first + 1..first + 1 + last].join("\n");
    }
    let (sm, decls) = declarations(text).unwrap_or_else(|| panic!("{file} does not parse"));
    let found: Vec<&Decl> = decls
        .iter()
        .filter(|d| match d {
            Decl::Training { .. } => name == "training run",
            Decl::Inference { .. } => name == "inference serve",
            _ => d.name().is_some_and(|n| n.name == name),
        })
        .collect();
    assert!(found.len() == 1, "{file}: {} declarations named {name}", found.len());
    sm.slice(found[0].span()).trim_end().to_string()
}

/// The kinds of `text`'s tokens: what a comment or the layout cannot change.
fn tokens(text: &str) -> Vec<TokKind> {
    let mut kinds: Vec<TokKind> = lexer::lex(FileId(0), text).tokens.into_iter().map(|t| t.kind).collect();
    while matches!(kinds.first(), Some(TokKind::Newline)) {
        kinds.remove(0);
    }
    kinds
}

/// What `presk check FILE` prints, each line without its trailing blanks.
fn check(file: &str) -> String {
    let o = Command::new(env!("CARGO_BIN_EXE_presk")).current_dir(root()).args(["check", file]).env("NO_COLOR", "1").output().expect("running presk");
    let text = format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
    text.lines().map(str::trim_end).collect::<Vec<_>>().join("\n").trim_end().to_string()
}

#[test]
fn the_manual_quotes_the_examples() {
    let update = std::env::var_os("PRESK_UPDATE_MANUAL").is_some();
    let dir = root().join("crates/presk-manual/manual");
    let mut chapters: Vec<PathBuf> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "md")).collect();
    chapters.sort();
    let mut failures = Vec::new();
    let mut quoted = 0;
    for path in &chapters {
        let chapter = path.file_name().unwrap().to_string_lossy().to_string();
        let text = std::fs::read_to_string(path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        let mut replaced: Vec<(usize, usize, Vec<String>)> = Vec::new();
        for b in blocks(&lines) {
            let at = format!("{chapter}:{}", b.fence + 1);
            let body = b.body.join("\n");
            let Some(marker) = &b.marker else {
                if (b.tag.is_empty() || b.tag == "text") && declarations(&format!("{body}\n")).is_some_and(|(_, d)| !d.is_empty()) {
                    failures.push(format!("{at}: a block of Presk without `<!-- psk: … -->`"));
                }
                continue;
            };
            let (want, same_tokens) = if let Some(why) = marker.strip_prefix("none") {
                if !(why.trim().starts_with('(') && why.trim().ends_with(')') && why.trim().len() > 2) {
                    failures.push(format!("{at}: `none` without its reason in parentheses"));
                }
                continue;
            } else if let Some(file) = marker.strip_prefix("check ") {
                (check(file.trim()), false)
            } else {
                let (source, same_tokens) = match marker.strip_suffix(", same tokens") {
                    Some(s) => (s, true),
                    None => (marker.as_str(), false),
                };
                let (file, name) = match source.split_once('#') {
                    Some((f, n)) => (f, Some(n)),
                    None => (source, None),
                };
                if !file.starts_with("examples/") {
                    failures.push(format!("{at}: {file} is not under examples/"));
                    continue;
                }
                let Ok(src) = std::fs::read_to_string(root().join(file)) else {
                    failures.push(format!("{at}: no file {file}"));
                    continue;
                };
                let want = match name {
                    Some(n) => part(file, &src, n),
                    None => src.trim_end_matches('\n').to_string(),
                };
                (want, same_tokens)
            };
            quoted += 1;
            if same_tokens {
                if tokens(&body) != tokens(&want) {
                    failures.push(format!("{at}: not the tokens of {marker}"));
                }
            } else if body != want {
                if update {
                    replaced.push((b.fence + 1, b.fence + 1 + b.body.len(), want.lines().map(|l| if l.is_empty() { String::new() } else { format!("{}{l}", b.indent) }).collect()));
                } else {
                    failures.push(format!("{at}: not the text of {marker} (`make manual-quotes` writes it again)"));
                }
            }
        }
        if !replaced.is_empty() {
            let mut out: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
            for (from, to, with) in replaced.into_iter().rev() {
                out.splice(from..to, with);
            }
            std::fs::write(path, out.join("\n") + "\n").unwrap();
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(quoted >= 20, "only {quoted} blocks quote a source");
}
