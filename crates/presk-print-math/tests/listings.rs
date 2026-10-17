//! Statement P5 of plan 0.13.0 (D137): the printer of the denotation, as
//! snapshots. The typed program of the tutorial, of the examples and of the
//! course GPT, in LaTeX and in Unicode; PSK-Core after differentiation of
//! the tutorial's gradient and of the MLP's; and every program of the test
//! set that types written with no form left as its source text.

use std::path::{Path, PathBuf};

use presk_driver::Collect;
use presk_print_math::{Format, Select};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn typed(file: &str) -> presk_print_math::Listing {
    let mut r = Collect::default();
    let (sm, prog, typed) = presk_driver::front::typed_front(&root().join(file), &[], &mut r).unwrap_or_else(|_| panic!("{file} types"));
    presk_print_math::typed_program(&sm, &prog, &typed)
}

fn typed_listing(file: &str, f: Format) -> String {
    let l = typed(file);
    assert!(l.gaps.is_empty(), "{file}: {:?}", l.gaps);
    presk_print_math::listing(&l, f, &Select::default(), Path::new(file).file_name().unwrap().to_str().unwrap(), false)
}

fn core_listing(file: &str, f: Format) -> String {
    let mut r = Collect::default();
    let (_, prog, mut core) = presk_driver::front::core(&root().join(file), &[], false, &mut r).unwrap_or_else(|_| panic!("{file} lowers"));
    let (diags, links) = presk_driver::ad::differentiate_traced(&mut core, presk_core::ad::Remat::Store, false);
    assert!(!presk_syntax::diag::has_errors(&diags), "{file} differentiates");
    let ds = presk_print_math::core::core_program(&core, &prog, &links);
    presk_print_math::core::listing(&ds, f, &Select::default(), Path::new(file).file_name().unwrap().to_str().unwrap(), false)
}

#[test]
fn the_tutorial_as_equations() {
    insta::assert_snapshot!("tutorial-train-latex", typed_listing("tutorial/07_train.psk", Format::Latex));
    insta::assert_snapshot!("tutorial-train-unicode", typed_listing("tutorial/07_train.psk", Format::Unicode));
    insta::assert_snapshot!("tutorial-attention-latex", typed_listing("tutorial/03_attention.psk", Format::Latex));
    insta::assert_snapshot!("tutorial-attention-unicode", typed_listing("tutorial/03_attention.psk", Format::Unicode));
}

#[test]
fn the_course_gpt_as_equations() {
    insta::assert_snapshot!("gpt-latex", typed_listing("corpus/nlp/05_gpt.psk", Format::Latex));
    insta::assert_snapshot!("gpt-unicode", typed_listing("corpus/nlp/05_gpt.psk", Format::Unicode));
}

#[test]
fn the_examples_as_equations() {
    for (name, file) in [("mlp", "tests/examples/mlp/mlp.psk"), ("layernorm", "tests/examples/layernorm/layernorm.psk"), ("minigpt", "tests/examples/minigpt/minigpt.psk")] {
        insta::assert_snapshot!(format!("{name}-latex"), typed_listing(file, Format::Latex));
    }
}

#[test]
fn the_adjoints_as_equations() {
    insta::assert_snapshot!("grad-core-latex", core_listing("tutorial/04_grad.psk", Format::Latex));
    insta::assert_snapshot!("grad-core-unicode", core_listing("tutorial/04_grad.psk", Format::Unicode));
    insta::assert_snapshot!("mlp-core-unicode", core_listing("tests/examples/mlp/mlp.psk", Format::Unicode));
}

/// Every program of the repository that types, at both entries: no form
/// left as its source text.
#[test]
fn every_program_is_written_whole() {
    let mut files = Vec::new();
    fn walk(d: &Path, out: &mut Vec<PathBuf>) {
        for e in std::fs::read_dir(d).unwrap().flatten() {
            let p = e.path();
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            if p.is_dir() {
                if !["target", "out", ".git", "diag", ".llvm", ".gappa"].contains(&name.as_str()) {
                    walk(&p, out);
                }
            } else if name.ends_with(".psk") {
                out.push(p);
            }
        }
    }
    walk(&root(), &mut files);
    files.sort();
    let mut written = 0;
    for f in &files {
        let mut r = Collect::default();
        let Ok((sm, prog, typed)) = presk_driver::front::typed_front(f, &[], &mut r) else { continue };
        let l = presk_print_math::typed_program(&sm, &prog, &typed);
        assert!(l.gaps.is_empty(), "{}: {:?}", f.display(), l.gaps);
        for u in &l.units {
            for i in &u.items {
                let ds: Vec<&presk_print_math::Display> = match i {
                    presk_print_math::typed::Item::Fn(d) => vec![d],
                    presk_print_math::typed::Item::Module { fns, .. } => fns.iter().collect(),
                };
                for d in ds {
                    assert!(d.eqs.iter().all(|e| !e.lhs.has_gap() && !e.rhs.has_gap()), "{}: {}", f.display(), d.name);
                }
            }
        }
        let mut r = Collect::default();
        if let Ok((_, prog, mut core)) = presk_driver::front::core(f, &[], false, &mut r) {
            let (diags, links) = presk_driver::ad::differentiate_traced(&mut core, presk_core::ad::Remat::Store, false);
            if !presk_syntax::diag::has_errors(&diags) {
                let ds = presk_print_math::core::core_program(&core, &prog, &links);
                assert_eq!(ds.len(), core.funcs.len());
            }
        }
        written += 1;
    }
    assert!(written >= 50, "only {written} programs typed");
}
