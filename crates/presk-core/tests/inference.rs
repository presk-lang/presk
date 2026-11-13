//! The inference reading (plan 0.16.0 step 3, D160): lowered in the
//! inference mode, a program that calls `dropout` has no `dropout_mask`,
//! before differentiation or after; it verifies; and every `let` it keeps
//! is the training mode's, at the same number, reading `x` where it read a
//! product by a mask.

use std::path::{Path, PathBuf};
use std::sync::Once;

use presk_core::inference::{Mode, masks};
use presk_core::lower::{Options, lower_with};
use presk_core::{PrimOp, Rhs};
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

fn lowered(rel: &str, mode: Mode) -> presk_core::Program {
    chdir_root();
    let mut sm = SourceMap::new();
    let id = sm.load(Path::new(rel)).unwrap();
    let (program, diags) = parse_file(&sm, id);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (prog, diags) = elab::elaborate(&mut sm, id, program, &ElabOptions::default());
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (typed, diags) = check::check(&sm, &prog);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (core, diags) = lower_with(&prog, &typed, &Options { unroll_stack: false, mode });
    assert!(diags.is_empty(), "{}", render_all(&diags, &sm, false));
    presk_core::verify(&core).unwrap_or_else(|errs| panic!("{rel} ({mode:?}) does not verify:\n{}", errs.join("\n")));
    core
}

fn mask_count(core: &presk_core::Program) -> usize {
    core.funcs.iter().flat_map(|f| &f.body.lets).filter(|l| matches!(&l.rhs, Rhs::Prim { op: PrimOp::DropoutMask, .. })).count()
}

const DROPS: [(&str, usize); 6] = [
    ("examples/corpus/nlp/05_gpt.psk", 4),
    ("examples/experiments/margins.psk", 4),
    ("examples/tests/train/dropout_reg.psk", 2),
    ("examples/tests/data/dropout_grad.psk", 1),
    ("examples/tests/data/stack_keys.psk", 2),
    ("examples/tests/data/stack_dropout_grad.psk", 1),
];

#[test]
fn an_inference_program_has_no_mask() {
    for (rel, n) in DROPS {
        let training = lowered(rel, Mode::Training);
        let mut inference = lowered(rel, Mode::Inference);
        assert_eq!(mask_count(&training), n, "{rel}: the masks as written");
        assert_eq!(masks(&training).len(), n, "{rel}: each mask with its product");
        assert_eq!(mask_count(&inference), 0, "{rel}: a mask in the inference program");
        let diags = presk_core::ad::transform(&mut inference);
        assert!(diags.is_empty(), "{rel}: differentiation");
        assert_eq!(mask_count(&inference), 0, "{rel}: a mask after differentiation");
    }
}

#[test]
fn the_lets_kept_are_the_training_program_s() {
    for (rel, _) in DROPS {
        let training = lowered(rel, Mode::Training);
        let inference = lowered(rel, Mode::Inference);
        assert_eq!(training.funcs.len(), inference.funcs.len(), "{rel}: a function gained or lost");
        let products: std::collections::HashMap<(String, u32), presk_core::VarId> = masks(&training)
            .into_iter()
            .map(|m| ((m.func.clone(), m.product.0), match m.operand {
                presk_core::inference::Operand::Let(x) | presk_core::inference::Operand::Param(x) => x,
            }))
            .collect();
        for (t, i) in training.funcs.iter().zip(&inference.funcs) {
            assert_eq!(t.name, i.name);
            let mut lost = 0;
            for l in &i.body.lets {
                let w = t.body.lets.iter().find(|w| w.var == l.var).unwrap_or_else(|| panic!("{rel}: {}/%{} is not the training program's", i.name, l.var.0));
                // The same right-hand side, a product's read replaced by
                // what it multiplied.
                let read = |v: presk_core::VarId| {
                    let mut v = v;
                    while let Some(x) = products.get(&(t.name.clone(), v.0)) {
                        v = *x;
                    }
                    v
                };
                assert_eq!(format!("{:?}", w.rhs.map_vars(read)), format!("{:?}", l.rhs), "{rel}: {}/%{}", i.name, l.var.0);
            }
            lost += t.body.lets.len() - i.body.lets.len();
            let dropped = masks(&training).iter().filter(|m| m.func == t.name).count();
            assert!(lost >= 2 * dropped, "{rel}: {}: {lost} lets removed for {dropped} masks", t.name);
        }
    }
}
