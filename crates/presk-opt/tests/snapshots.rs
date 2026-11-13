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
    let (prog, _, mut core) = prepare("examples/spec/minigpt/minigpt.psk", TINY);
    let before = core.func(core.find("SelfAttention.apply").unwrap()).body.lets.len();
    optimize(&mut core, &prog);
    let text = func_text(&core, &prog, "SelfAttention.apply");
    assert!(core.func(core.find("SelfAttention.apply").unwrap()).body.lets.len() <= before);
    insta::assert_snapshot!("opt_self_attention_apply", text);
}

/// The reverse sweeps, since 0.10.0 step 4 in the `.bwd`s: a `.vjp` is its
/// `.bwd` after its `.fwd`.
#[test]
fn mse_vjp_and_mlp_adjoints() {
    let (prog, _, mut core) = prepare("examples/spec/mlp/mlp.psk", &[]);
    let lets = |core: &presk_core::Program, name: &str| core.func(core.find(name).unwrap()).body.lets.len();
    let sweeps = ["mse.bwd", "MLP.apply.bwd", "Linear.apply.bwd"];
    let before: Vec<usize> = sweeps.iter().map(|n| lets(&core, n)).collect();
    let report = optimize(&mut core, &prog);
    let after: Vec<usize> = sweeps.iter().map(|n| lets(&core, n)).collect();
    for (i, name) in sweeps.iter().enumerate() {
        assert!(after[i] < before[i], "{name}: {} → {} lets\n{}", before[i], after[i], report.summary());
    }
    // The adjoint of `Linear.apply` is one record, with no zeros to add.
    let lin = core.func(core.find("Linear.apply.bwd").unwrap());
    assert!(lin.body.lets.iter().any(|l| matches!(l.rhs, Rhs::Record(_))));
    assert!(!lin.body.lets.iter().any(|l| matches!(l.rhs, Rhs::Prim { op: PrimOp::ZerosLike | PrimOp::SetField { .. } | PrimOp::Add, .. })), "{}", func_text(&core, &prog, "Linear.apply.bwd"));
    // No forward product: the sweep reads the one its `.fwd` made, or here
    // its operands, `Linear.apply`'s parameters.
    assert!(!lin.body.lets.iter().any(|l| matches!(l.rhs, Rhs::Prim { op: PrimOp::Dot, .. })));
    let text = sweeps.iter().map(|n| func_text(&core, &prog, n)).collect::<Vec<_>>().join("\n");
    insta::assert_snapshot!("opt_mlp_adjoints", text);
}

/// The report's totals: fewer lets, lower cost, every function verified.
#[test]
fn report_totals() {
    let (prog, typed, mut core) = prepare("examples/spec/minigpt/train_minigpt.psk", TINY);
    presk_train::prepare_training(&mut core, &prog, &typed).unwrap();
    let report = optimize(&mut core, &prog);
    assert!(report.lets_after() < report.lets_before(), "{}", report.summary());
    assert!(report.cost_after() < report.cost_before(), "{}", report.summary());
    assert!(report.summary().starts_with("presk-opt: "));
}
