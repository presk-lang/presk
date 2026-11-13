//! A training run under a realization solved outside training (plan 0.16.0
//! step 3, D160): `masks_typed` gives each mask, and the product by it, its
//! operand's type, and leaves alone a realization that types the mask, its
//! product or their function, and a uniform one.

use std::path::{Path, PathBuf};
use std::sync::Once;

use presk_rt::{Dtype, Plan, Realization};
use presk_syntax::diag::render_all;
use presk_syntax::{SourceMap, parse_file};
use presk_types::check;
use presk_types::elab::{self, Options};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn written(rel: &str) -> presk_core::Program {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| std::env::set_current_dir(repo_root()).unwrap());
    let mut sm = SourceMap::new();
    let id = sm.load(Path::new(rel)).unwrap();
    let (program, diags) = parse_file(&sm, id);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (prog, diags) = elab::elaborate(&mut sm, id, program, &Options::default());
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (typed, diags) = check::check(&sm, &prog);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    presk_core::lower(&prog, &typed).0
}

#[test]
fn a_mask_takes_its_operand_s_type() {
    // `dropout(0.3, key, tanh(x · w1))` in `M.apply`.
    let core = written("examples/tests/data/dropout_grad.psk");
    let m = presk_core::inference::masks(&core);
    assert_eq!(m.len(), 1);
    let m = &m[0];
    let site = |v: presk_core::VarId| format!("{}/%{}", m.func, v.0);
    let presk_core::inference::Operand::Let(x) = m.operand else { panic!("the operand is a let") };
    let mut r = Realization::uniform(Dtype::F64);
    r.storage.insert(site(x), Dtype::Bf16);
    let typed = presk_train::masks_typed(&Plan::from_realization(r.clone()), &core).realization;
    assert_eq!(typed.storage.get(&site(m.mask)), Some(&Dtype::Bf16));
    assert_eq!(typed.storage.get(&site(m.product)), Some(&Dtype::Bf16));
    // A mask the realization types keeps its type.
    let mut stated = r.clone();
    stated.storage.insert(site(m.mask), Dtype::F32);
    assert_eq!(presk_train::masks_typed(&Plan::from_realization(stated.clone()), &core).realization, stated);
    // So does one in a function typed as a whole (a step's realization).
    let mut whole = r.clone();
    whole.functions.insert(m.func.clone(), Dtype::F32);
    assert_eq!(presk_train::masks_typed(&Plan::from_realization(whole.clone()), &core).realization, whole);
    // And a uniform realization is unchanged.
    let uniform = Realization::uniform(Dtype::F32);
    assert_eq!(presk_train::masks_typed(&Plan::from_realization(uniform.clone()), &core).realization, uniform);
}
