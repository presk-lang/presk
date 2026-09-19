//! Lowering of the five examples: PSK-Core verifies and its rendering is
//! snapshotted.

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

fn core_of(rel: &str) -> String {
    chdir_root();
    let mut sm = SourceMap::new();
    let id = sm.load(Path::new(rel)).unwrap();
    let (program, diags) = parse_file(&sm, id);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (prog, diags) = elab::elaborate(&mut sm, id, program, &Options::default());
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (typed, diags) = check::check(&sm, &prog);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (core, diags) = presk_core::lower(&prog, &typed);
    assert!(diags.is_empty(), "{rel}: lowering diagnostics:\n{}", render_all(&diags, &sm, false));
    presk_core::verify(&core).unwrap_or_else(|errs| panic!("{rel}: PSK-Core does not verify:\n{}", errs.join("\n")));
    presk_core::pretty(&core, &prog)
}

#[test]
fn core_examples() {
    for rel in ["mlp/mlp.psk", "layernorm/layernorm.psk", "minigpt/minigpt.psk", "minigpt/train_minigpt.psk", "minigpt/serve_minigpt.psk"] {
        let text = core_of(&format!("tests/examples/{rel}"));
        let name = format!("core_{}", Path::new(rel).file_stem().unwrap().to_str().unwrap());
        insta::assert_snapshot!(name, text);
    }
}

/// `grad(λ θ. mse(net, θ, x, y))(θ)`: the lambda is lifted with its free
/// variables `x, y` (not the shadowed `θ`) and applied through a closure.
#[test]
fn closure_conversion() {
    let text = core_of("tests/examples/mlp/mlp.psk");
    assert!(text.contains("closure mse_grad.lambda0[x, y]"), "{text}");
    assert!(text.contains("fn mse_grad.lambda0{D B}(x : ℝ[batch: B, model: D], y : ℝ[batch: B, model: D], θ : params MLP{D, 4*D}) -> ℝ[]"), "{text}");
    assert!(text.contains("grad %"), "{text}");
}
