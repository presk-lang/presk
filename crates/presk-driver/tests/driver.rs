//! The driver's values against what the commands computed before 0.11.0
//! (plan 0.11.0, steps 1 and 2).

use std::path::{Path, PathBuf};

use presk_core::ad::Remat;
use presk_driver::Collect;
use presk_rt::Plan;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

/// The file differentiated as the commands differentiate it.
fn program(file: &Path, consts: &[String]) -> (presk_syntax::SourceMap, presk_types::elab::ElabProgram, presk_core::Program) {
    let mut report = Collect::default();
    let (sm, prog, mut core) = presk_driver::front::core(file, consts, false, &mut report).unwrap_or_else(|s| panic!("{}: exit {} {:?}", file.display(), s.0, report.items));
    let diags = presk_driver::ad::differentiate(&mut core, Remat::Store, true);
    assert!(!presk_syntax::diag::has_errors(&diags));
    (sm, prog, core)
}

/// X4: the kernels from the entry's shapes are the ones the kernel
/// interpreter's run launched, in order, with the same counts.
#[test]
fn kernels_from_shapes_are_the_run_s() {
    let cases: [(&str, &str, &[&str], &[&str]); 4] = [
        ("tests/examples/mlp/mlp.psk", "mse_grad", &["D=4", "B=2"], &[]),
        ("tests/examples/layernorm/layernorm.psk", "LayerNorm{8}.apply", &["batch=3"], &[]),
        ("tests/examples/minigpt/minigpt.psk", "MiniGPT.apply", &["B=2", "T=8"], &["V=64", "D=16", "H=2", "L=2"]),
        ("tutorial/07_train.psk", "loss", &["B=4"], &[]),
    ];
    for (file, entry, dims, consts) in cases {
        let consts: Vec<String> = consts.iter().map(|c| c.to_string()).collect();
        let (_, prog, core) = program(&root().join(file), &consts);
        let dims: Vec<(String, u64)> = dims.iter().map(|d| d.split_once('=').map(|(n, v)| (n.to_string(), v.parse().unwrap())).unwrap()).collect();
        for realize in ["f64", "f32", "bf16"] {
            let plan = Plan::from_realization(presk_rt::Realization::parse(realize).unwrap());
            let listing = presk_driver::layout::kernels(&core, &prog, entry, dims.clone(), &plan, 0).unwrap();
            let mut b = presk_kernel::KernelBackend::new(&core, &prog, &plan.realization);
            b.set_plan(plan.clone());
            let opts = presk_interp::RunOptions { entry: entry.into(), dims: dims.clone(), seed: 0, bits: 64, digits: 12 };
            presk_rt::run_with(&mut b, &core, &prog, &opts, &plan.realization).unwrap();
            let text = |ks: &[presk_kernel::Kernel]| ks.iter().map(|k| k.to_string()).collect::<String>();
            assert_eq!(text(&listing.kernels), text(&b.kernels), "{entry} in {realize}");
            assert_eq!((listing.launches, listing.fused), (b.launches, b.fused), "{entry} in {realize}");
        }
    }
}

/// The solver's breakdown: its terms are the estimate, site by site.
#[test]
fn breakdown_sums_to_the_estimate() {
    let file = root().join("tutorial/07_train.psk");
    let (_, prog, mut core) = program(&file, &[]);
    let c = presk_driver::precision::Calibrate { file: &file, entry: "loss", dims: vec![("B".into(), 4)], consts: &[], data: None, samples: None, from: 0, seed: 0, bits: 128, weights: None };
    let env = presk_driver::precision::calibrate(&core, &prog, &c).unwrap();
    let a = presk_driver::precision::SolveArgs { budget: presk_precision::solve::Budget::Relative(1e-3), weights: None, verify: true, bits: 128, seed: 0, profile: None, quant: None, options: Default::default() };
    let Ok(solved) = presk_driver::precision::solve(&mut core, &prog, &env, &a) else { panic!("the tutorial's loss solves") };
    use presk_precision::solve::Role;
    let sum: f64 = solved.breakdown.iter().map(|s| s.err).sum();
    assert!((sum - solved.solution.estimated_error).abs() <= 1e-15 * solved.solution.estimated_error, "{sum} against {}", solved.solution.estimated_error);
    let decisions: Vec<_> = solved.breakdown.iter().filter(|s| s.role == Role::Decision).collect();
    assert_eq!(decisions.len(), solved.solution.considered);
    let narrowed = decisions.iter().filter(|s| s.dtype != presk_rt::Dtype::F64).count();
    assert_eq!(narrowed, solved.solution.realization.storage.len() + solved.solution.realization.params.len());
    // The parameters are decisions, keyed by path (plan 0.12.0 step 4).
    assert!(solved.solution.realization.params.keys().all(|k| decisions.iter().any(|d| d.param && &d.name == k)));
    // Only a decision has a type in the realization (step 2b).
    assert!(solved.solution.realization.storage.keys().all(|k| decisions.iter().any(|d| &d.name == k)));
}

/// D155: the envelope's measured rounding is at most the unit term at every
/// site, `u(d)·max|x|`, and 0 in `bf16` at a site of exact constants; the
/// solver under it and under the narrowing step returns a realization that
/// holds; the default options are the measured term's since 0.16.0 (D157).
#[test]
fn the_measured_rounding_is_at_most_the_unit_term() {
    use presk_precision::solve::{Narrow, Options, RoundingTerm};
    use presk_rt::Dtype;
    let file = root().join("tutorial/07_train.psk");
    let (_, prog, mut core) = program(&file, &[]);
    let c = presk_driver::precision::Calibrate { file: &file, entry: "loss", dims: vec![("B".into(), 4)], consts: &[], data: None, samples: None, from: 0, seed: 0, bits: 128, weights: None };
    let env = presk_driver::precision::calibrate(&core, &prog, &c).unwrap();
    assert!(!env.rounding.is_empty(), "the calibration records it");
    for (site, r) in &env.rounding {
        let m = env.get(site).unwrap_or(0.0);
        for d in [Dtype::F64, Dtype::F32, Dtype::Bf16] {
            assert!(r.at(d) <= d.unit_roundoff() * m * (1.0 + 1e-12), "{site} in {d}: {} against u·|x| {}", r.at(d), d.unit_roundoff() * m);
        }
    }
    let solve = |core: &mut presk_core::Program, options: Options| {
        let a = presk_driver::precision::SolveArgs { budget: presk_precision::solve::Budget::Relative(1e-3), weights: None, verify: true, bits: 128, seed: 0, profile: None, quant: None, options };
        presk_driver::precision::solve(core, &prog, &env, &a).ok().expect("the tutorial's loss solves").solution
    };
    let unit = solve(&mut core, Options { rounding: RoundingTerm::Unit, narrow: Narrow::Off, ..Options::default() });
    let measured = solve(&mut core, Options { rounding: RoundingTerm::Measured, narrow: Narrow::Off, ..Options::default() });
    assert_eq!(solve(&mut core, Options::default()).realization, measured.realization, "the default is the measured term");
    let both = solve(&mut core, Options { rounding: RoundingTerm::Measured, narrow: Narrow::Measured, margin: 1.0, ..Options::default() });
    for s in [&unit, &measured, &both] {
        assert!(s.measured.is_some_and(|g| g <= s.budget_abs), "{:?} {} {}", s.options, s.measured.unwrap_or(f64::NAN), s.budget_abs);
    }
    assert!(measured.estimated_error <= unit.estimated_error * (1.0 + 1e-12) || measured.bytes <= unit.bytes, "the measured term is never the larger");
    assert!(both.bytes <= measured.bytes, "narrowing only narrows");
    assert_eq!(both.measurements, both.rounds + 1 + both.moves.iter().filter(|m| matches!(m, presk_precision::solve::Move::Narrowed { .. })).count());
}

/// D128: an entry whose first argument is a tensor, not a parameter tree,
/// has no parameter to offer. Until D128 the solver offered the tensor as
/// the parameter of path ``, which no executor runs in a type of its own:
/// every measurement narrowing it failed, read as an infinite gap, and the
/// solver widened every other site back before it came to that one. On the
/// attention row of experiments/widening.psk, by the softmax primitive, the
/// proposal is accepted as it stands.
#[test]
fn a_tensor_first_argument_is_no_parameter() {
    let file = root().join("experiments/widening.psk");
    let (_, prog, mut core) = program(&file, &[]);
    let c = presk_driver::precision::Calibrate { file: &file, entry: "attn_softmax", dims: vec![("B".into(), 8)], consts: &[], data: None, samples: None, from: 0, seed: 0, bits: 128, weights: None };
    let env = presk_driver::precision::calibrate(&core, &prog, &c).unwrap();
    let a = presk_driver::precision::SolveArgs { budget: presk_precision::solve::Budget::Relative(1e-3), weights: None, verify: true, bits: 128, seed: 0, profile: None, quant: None, options: Default::default() };
    let Ok(solved) = presk_driver::precision::solve(&mut core, &prog, &env, &a) else { panic!("the attention row solves") };
    assert!(solved.breakdown.iter().all(|s| !s.param), "a parameter offered");
    assert!(solved.solution.realization.params.is_empty());
    assert_eq!(solved.solution.rounds, 0);
    let gap = solved.solution.measured.unwrap();
    assert!(gap.is_finite() && gap <= solved.solution.budget_abs, "{gap} against {}", solved.solution.budget_abs);
    assert!(!solved.solution.realization.storage.is_empty(), "nothing narrowed");
}

/// Step 2b: the roles of the tutorial's sites, as the reference's chapter 4
/// gives a type a meaning — where a value is stored.
#[test]
fn roles_of_the_tutorial() {
    use presk_precision::solve::{Role, Roles};
    let (_, _, core) = program(&root().join("tutorial/07_train.psk"), &[]);
    let roles = Roles::of(&core);
    let follows = |s: &str| Role::Follows(vec![s.to_string()]);
    for (site, role) in [
        ("Linear.apply/%3", Role::Decision),                  // dot, a nest
        ("Linear.apply/%2", follows("Linear.apply/%3")),      // θ.w, rounded by the dot that reads it
        ("RMSNorm.apply/%6", follows("RMSNorm.apply/%7")),    // add, fused into sqrt's group
        ("RMSNorm.apply/%7", Role::Decision),                 // the group's root
        ("Block.apply/%2", Role::Carried),                    // θ.norm, a subtree passed to a call
        ("Block.apply/%4", Role::Carried),                    // a call's result
        ("Tower.apply/%3", Role::Carried),                    // a fold's result
    ] {
        assert_eq!(roles.role(site), role, "{site}");
    }
}

/// Plan 0.12.0 step 4: the parameter paths each field site reads, through
/// the fold over the tower's blocks and the calls to `RMSNorm.apply` from
/// the blocks and for the final norm.
#[test]
fn parameter_reads_of_the_tutorial() {
    use std::collections::BTreeSet;
    let (_, prog, core) = program(&root().join("tutorial/07_train.psk"), &[]);
    let (entry, _, _) = presk_interp::run::resolve_entry(&core, &prog, "loss").unwrap();
    let reads = presk_precision::solve::param_reads(&core, entry);
    let set = |ps: &[&str]| ps.iter().map(|p| p.to_string()).collect::<BTreeSet<String>>();
    assert_eq!(reads["Linear.apply/%2"], set(&["blocks.proj.w"]));
    assert_eq!(reads["Block.apply/%2"], set(&["blocks.norm"]));
    let gamma: Vec<&BTreeSet<String>> = reads.iter().filter(|(site, _)| site.starts_with("RMSNorm.apply/")).map(|(_, ps)| ps).collect();
    assert_eq!(gamma, vec![&set(&["blocks.norm.γ", "final.γ"])]);
    let leaves: BTreeSet<String> = reads.values().flatten().filter(|p| ["blocks.norm.γ", "blocks.proj.w", "blocks.proj.b", "final.γ"].contains(&p.as_str())).cloned().collect();
    assert_eq!(leaves.len(), 4);
}

/// D108's finding 1: a module entry's own arguments are dimensions of its
/// bound, which is then the one behind a wrapper that binds them itself.
#[test]
fn module_entry_bound_is_the_wrapper_s() {
    let bound = |file: &str, entry: &str, consts: &[String]| {
        let path = root().join(file);
        let (_, prog, core) = program(&path, consts);
        let c = presk_driver::precision::Calibrate { file: &path, entry, dims: vec![("batch".into(), 8)], consts, data: None, samples: None, from: 0, seed: 0, bits: 128, weights: None };
        let env = presk_driver::precision::calibrate(&core, &prog, &c).unwrap();
        let realize = presk_rt::Realization::parse("f32").unwrap();
        let Ok(b) = presk_driver::precision::bound(&core, &prog, &env, &realize, &[presk_precision::propagate::Model::Worst, presk_precision::propagate::Model::Conditioned], false) else { panic!("{entry} is bounded") };
        (b.runs[0].output.err, b.runs[1].output.err)
    };
    let direct = bound("tests/examples/layernorm/layernorm.psk", "LayerNorm{8}.apply", &[]);
    let wrapped = bound("experiments/gappa_layernorm.psk", "Norm.apply", &["D=8".to_string()]);
    assert_eq!(direct, wrapped);
}

/// The sensitivities' sweep is a program of its own, its `.vjp` first.
#[test]
fn sensitivity_program_is_printable() {
    let (_, prog, mut core) = program(&root().join("tutorial/07_train.psk"), &[]);
    let (entry, _, _) = presk_interp::run::resolve_entry(&core, &prog, "loss").unwrap();
    let (added, diags) = presk_driver::precision::sensitivity_program(&mut core, entry);
    assert!(!presk_syntax::diag::has_errors(&diags));
    assert!(core.func(added[0]).name.ends_with(".vjp"), "{}", core.func(added[0]).name);
}
