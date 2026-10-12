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
    let a = presk_driver::precision::SolveArgs { budget: 1e-3, weights: None, verify: true, bits: 128, seed: 0, profile: None, quant: None };
    let Ok(solved) = presk_driver::precision::solve(&mut core, &prog, &env, &a) else { panic!("the tutorial's loss solves") };
    use presk_precision::solve::Role;
    let sum: f64 = solved.breakdown.iter().map(|s| s.err).sum();
    assert!((sum - solved.solution.estimated_error).abs() <= 1e-15 * solved.solution.estimated_error, "{sum} against {}", solved.solution.estimated_error);
    let decisions: Vec<_> = solved.breakdown.iter().filter(|s| s.role == Role::Decision).collect();
    assert_eq!(decisions.len(), solved.solution.considered);
    let narrowed = decisions.iter().filter(|s| s.dtype != presk_rt::Dtype::F64).count();
    assert_eq!(narrowed, solved.solution.realization.storage.len());
    // Only a decision has a type in the realization (step 2b).
    assert!(solved.solution.realization.storage.keys().all(|k| decisions.iter().any(|d| &d.name == k)));
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
