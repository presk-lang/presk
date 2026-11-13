//! Automatic differentiation: after `ad::transform`, no `grad` node remains,
//! every program verifies, and the differentiated `mlp.psk` is snapshotted.

use std::path::{Path, PathBuf};
use std::sync::Once;

use presk_core::Rhs;
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

fn after_ad(rel: &str) -> (presk_core::Program, String) {
    chdir_root();
    let mut sm = SourceMap::new();
    let id = sm.load(Path::new(rel)).unwrap();
    let (program, diags) = parse_file(&sm, id);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (prog, diags) = elab::elaborate(&mut sm, id, program, &Options::default());
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (typed, diags) = check::check(&sm, &prog);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (mut core, diags) = presk_core::lower(&prog, &typed);
    assert!(diags.is_empty(), "{}", render_all(&diags, &sm, false));
    let diags = presk_core::ad::transform(&mut core);
    assert!(diags.is_empty(), "{rel}: AD diagnostics:\n{}", render_all(&diags, &sm, false));
    presk_core::verify(&core).unwrap_or_else(|errs| panic!("{rel}: PSK-Core after AD does not verify:\n{}", errs.join("\n")));
    let text = presk_core::pretty(&core, &prog);
    (core, text)
}

#[test]
fn no_grad_remains_and_programs_verify() {
    for rel in ["mlp/mlp.psk", "layernorm/layernorm.psk", "minigpt/minigpt.psk", "minigpt/train_minigpt.psk", "minigpt/serve_minigpt.psk"] {
        let (core, _) = after_ad(&format!("examples/spec/{rel}"));
        for f in &core.funcs {
            for l in &f.body.lets {
                assert!(!matches!(l.rhs, Rhs::Grad(_)), "{rel}: `grad` left in {}", f.name);
            }
        }
    }
}

#[test]
fn mlp_after_ad() {
    let (core, text) = after_ad("examples/spec/mlp/mlp.psk");
    // The generated functions: a `.grad` for the lambda and a `.vjp` for every
    // function on the differentiated path.
    for name in ["mse_grad.lambda0.grad", "mse_grad.lambda0.vjp", "mse.vjp", "MLP.apply.vjp", "Linear.apply.vjp"] {
        assert!(core.find(name).is_some(), "missing {name}\n{text}");
    }
    // Only the functions of the differentiated path have a VJP.
    assert!(core.find("Linear.init.vjp").is_none());
    insta::assert_snapshot!("ad_mlp", text);
}
