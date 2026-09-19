//! What the rewriting does to the examples: the attention block (little to
//! do on a forward pass), the adjoint of `mse` and the adjoints of the MLP
//! (the structural cleanup), as snapshots and as counts.

mod common;

use common::{optimize, prepare};
use presk_core::{PrimOp, Rhs};

fn func_text(core: &presk_core::Program, prog: &presk_types::elab::ElabProgram, name: &str) -> String {
    let f = core.func(core.find(name).unwrap_or_else(|| panic!("no function {name}")));
    presk_core::pretty::pretty_func(f, core, prog)
}

const TINY: &[(&str, u64)] = &[("V", 64), ("D", 16), ("H", 2), ("L", 2)];

#[test]
fn self_attention_apply() {
    let (prog, _, mut core) = prepare("tests/examples/minigpt/minigpt.psk", TINY);
    let before = core.func(core.find("SelfAttention.apply").unwrap()).body.lets.len();
    optimize(&mut core, &prog);
    let text = func_text(&core, &prog, "SelfAttention.apply");
    assert!(core.func(core.find("SelfAttention.apply").unwrap()).body.lets.len() <= before);
    insta::assert_snapshot!("opt_self_attention_apply", text);
}

#[test]
fn mse_vjp_and_mlp_adjoints() {
    let (prog, _, mut core) = prepare("tests/examples/mlp/mlp.psk", &[]);
    let lets = |core: &presk_core::Program, name: &str| core.func(core.find(name).unwrap()).body.lets.len();
    let before: Vec<usize> = ["mse.vjp", "MLP.apply.vjp", "Linear.apply.vjp"].iter().map(|n| lets(&core, n)).collect();
    let report = optimize(&mut core, &prog);
    let after: Vec<usize> = ["mse.vjp", "MLP.apply.vjp", "Linear.apply.vjp"].iter().map(|n| lets(&core, n)).collect();
    for (i, name) in ["mse.vjp", "MLP.apply.vjp", "Linear.apply.vjp"].iter().enumerate() {
        assert!(after[i] < before[i], "{name}: {} → {} lets\n{}", before[i], after[i], report.summary());
    }
    // The adjoint of `Linear.apply` is one record, with no zeros to add.
    let lin = core.func(core.find("Linear.apply.vjp").unwrap());
    assert!(lin.body.lets.iter().any(|l| matches!(l.rhs, Rhs::Record(_))));
    assert!(!lin.body.lets.iter().any(|l| matches!(l.rhs, Rhs::Prim { op: PrimOp::ZerosLike | PrimOp::SetField { .. } | PrimOp::Add, .. })), "{}", func_text(&core, &prog, "Linear.apply.vjp"));
    // The dead forward recomputation (`dot`, `add`) of the adjoint is gone.
    assert!(!lin.body.lets.iter().any(|l| matches!(l.rhs, Rhs::Prim { op: PrimOp::Dot, .. })));
    let text = [func_text(&core, &prog, "mse.vjp"), func_text(&core, &prog, "MLP.apply.vjp"), func_text(&core, &prog, "Linear.apply.vjp")].join("\n");
    insta::assert_snapshot!("opt_mlp_adjoints", text);
}

/// The report's totals: fewer lets, lower cost, every function verified.
#[test]
fn report_totals() {
    let (prog, typed, mut core) = prepare("tests/examples/minigpt/train_minigpt.psk", TINY);
    presk_train::prepare_training(&mut core, &prog, &typed).unwrap();
    let report = optimize(&mut core, &prog);
    assert!(report.lets_after() < report.lets_before(), "{}", report.summary());
    assert!(report.cost_after() < report.cost_before(), "{}", report.summary());
    assert!(report.summary().starts_with("presk-opt: "));
}
