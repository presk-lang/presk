//! The pipeline up to a differentiated PSK-Core program, shared by the tests.

use std::path::{Path, PathBuf};
use std::sync::Once;

use presk_syntax::diag::render_all;
use presk_syntax::{SourceMap, parse_file};
use presk_types::check::{self, TypedProgram};
use presk_types::elab::{self, ElabProgram, Options};

pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

pub fn chdir_root() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| std::env::set_current_dir(repo_root()).unwrap());
}

pub fn prepare(rel: &str, consts: &[(&str, u64)]) -> (ElabProgram, TypedProgram, presk_core::Program) {
    chdir_root();
    let mut sm = SourceMap::new();
    let id = sm.load(Path::new(rel)).unwrap();
    let (program, diags) = parse_file(&sm, id);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let opts = Options { const_overrides: consts.iter().map(|(n, v)| (n.to_string(), *v)).collect() };
    let (prog, diags) = elab::elaborate(&mut sm, id, program, &opts);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (typed, diags) = check::check(&sm, &prog);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (mut core, diags) = presk_core::lower(&prog, &typed);
    assert!(diags.is_empty(), "{}", render_all(&diags, &sm, false));
    let diags = presk_core::ad::transform(&mut core);
    assert!(diags.is_empty(), "{}", render_all(&diags, &sm, false));
    presk_core::verify(&core).unwrap();
    (prog, typed, core)
}

/// Optimize with the default limits; the result verifies.
#[allow(dead_code)]
pub fn optimize(core: &mut presk_core::Program, prog: &ElabProgram) -> presk_opt::Report {
    let report = presk_opt::optimize(core, prog, &presk_opt::Options::default());
    presk_core::verify(core).unwrap_or_else(|e| panic!("optimized program does not verify: {}", e.join("\n")));
    for f in &report.funcs {
        if let presk_opt::Outcome::Skipped(why) = &f.outcome {
            panic!("{}: skipped: {why}", f.name);
        }
    }
    report
}
