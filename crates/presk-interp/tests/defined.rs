//! The refusal at dimensions walks what the reference's definition
//! "Undefined at dimensions" reads (plan 0.10.0 step 7b): calls, closures
//! applied, and folds whose count is at least 1, not those of count 0.
//! Through the binary, `presk-cli/tests/run_diag.rs`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use presk_syntax::diag::render_all;
use presk_syntax::{SourceMap, parse_file};
use presk_types::check;
use presk_types::elab::{self, ElabProgram, Options};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn prepare(rel: &str, consts: &[(&str, u64)]) -> (ElabProgram, presk_core::Program) {
    let mut sm = SourceMap::new();
    let id = sm.load(&repo_root().join(rel)).unwrap();
    let (program, diags) = parse_file(&sm, id);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let opts = Options { const_overrides: consts.iter().map(|(n, v)| (n.to_string(), *v)).collect() };
    let (prog, diags) = elab::elaborate(&mut sm, id, program, &opts);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (typed, diags) = check::check(&sm, &prog);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (mut core, _) = presk_core::lower(&prog, &typed);
    presk_core::ad::transform(&mut core);
    (prog, core)
}

/// `Tower.apply` at `T = 0`, with `layers` stacked `L` times.
fn tower(l: u64) -> Result<(), String> {
    let (prog, core) = prepare("tests/data/defined.psk", &[("L", l)]);
    let f = core.find("Tower.apply").unwrap();
    let dims: BTreeMap<_, _> = core.func(f).dims.iter().map(|d| (*d, 0)).collect();
    presk_interp::defined::check(&core, &prog, f, &dims, None).map_err(|e| e.message)
}

#[test]
fn a_fold_is_walked_when_its_count_is_at_least_one() {
    let e = tower(2).unwrap_err();
    assert!(e.contains("in `Layer.apply`, `mean axis seq` averages over the axis `seq`, of length 0"), "{e}");
    assert!(tower(1).is_err());
    // No layer: the fold's body is never applied, and the entry is defined.
    assert_eq!(tower(0), Ok(()));
}
