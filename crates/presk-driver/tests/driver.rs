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
        ("examples/spec/mlp/mlp.psk", "mse_grad", &["D=4", "B=2"], &[]),
        ("examples/spec/layernorm/layernorm.psk", "LayerNorm{8}.apply", &["batch=3"], &[]),
        ("examples/spec/minigpt/minigpt.psk", "MiniGPT.apply", &["B=2", "T=8"], &["V=64", "D=16", "H=2", "L=2"]),
        ("examples/tutorial/07_train.psk", "loss", &["B=4"], &[]),
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
    let file = root().join("examples/tutorial/07_train.psk");
    let (_, prog, mut core) = program(&file, &[]);
    let c = presk_driver::precision::Calibrate { file: &file, entry: "loss", dims: vec![("B".into(), 4)], consts: &[], data: None, samples: None, from: 0, seed: 0, bits: 128, weights: None };
    let env = presk_driver::precision::calibrate(&core, &prog, &c).unwrap();
    let a = presk_driver::precision::SolveArgs { budget: presk_precision::solve::Budget::Relative(1e-3), weights: None, verify: true, bits: 128, seed: 0, profile: None, quant: None, menu: Default::default(), options: Default::default(), exec: Default::default() };
    let Ok(solved) = presk_driver::precision::solve(&mut core, &prog, &env, &a) else { panic!("the tutorial's loss solves") };
    use presk_precision::solve::Role;
    let sum: f64 = solved.breakdown.iter().map(|s| s.err).sum();
    assert!((sum - solved.solution.estimated_error).abs() <= 1e-15 * solved.solution.estimated_error, "{sum} against {}", solved.solution.estimated_error);
    let decisions: Vec<_> = solved.breakdown.iter().filter(|s| s.role == Role::Decision).collect();
    assert_eq!(decisions.len(), solved.solution.considered);
    let narrowed = decisions.iter().filter(|s| s.rung != presk_rt::Dtype::F64.into()).count();
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
    let file = root().join("examples/tutorial/07_train.psk");
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
        let a = presk_driver::precision::SolveArgs { budget: presk_precision::solve::Budget::Relative(1e-3), weights: None, verify: true, bits: 128, seed: 0, profile: None, quant: None, menu: Default::default(), options, exec: Default::default() };
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
/// attention row of examples/experiments/widening.psk, by the softmax primitive, the
/// proposal is accepted as it stands.
#[test]
fn a_tensor_first_argument_is_no_parameter() {
    let file = root().join("examples/experiments/widening.psk");
    let (_, prog, mut core) = program(&file, &[]);
    let c = presk_driver::precision::Calibrate { file: &file, entry: "attn_softmax", dims: vec![("B".into(), 8)], consts: &[], data: None, samples: None, from: 0, seed: 0, bits: 128, weights: None };
    let env = presk_driver::precision::calibrate(&core, &prog, &c).unwrap();
    let a = presk_driver::precision::SolveArgs { budget: presk_precision::solve::Budget::Relative(1e-3), weights: None, verify: true, bits: 128, seed: 0, profile: None, quant: None, menu: Default::default(), options: Default::default(), exec: Default::default() };
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
    let (_, _, core) = program(&root().join("examples/tutorial/07_train.psk"), &[]);
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
    let (_, prog, core) = program(&root().join("examples/tutorial/07_train.psk"), &[]);
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
    let direct = bound("examples/spec/layernorm/layernorm.psk", "LayerNorm{8}.apply", &[]);
    let wrapped = bound("examples/experiments/gappa_layernorm.psk", "Norm.apply", &["D=8".to_string()]);
    assert_eq!(direct, wrapped);
}

/// The sensitivities' sweep is a program of its own, its `.vjp` first.
#[test]
fn sensitivity_program_is_printable() {
    let (_, prog, mut core) = program(&root().join("examples/tutorial/07_train.psk"), &[]);
    let (entry, _, _) = presk_interp::run::resolve_entry(&core, &prog, "loss").unwrap();
    let (added, diags) = presk_driver::precision::sensitivity_program(&mut core, entry);
    assert!(!presk_syntax::diag::has_errors(&diags));
    assert!(core.func(added[0]).name.ends_with(".vjp"), "{}", core.func(added[0]).name);
}

/// One menu (plan 0.17.0 step 4, D177) on the tutorial's tower: its stacked
/// matrix is offered its four integer rungs, scaled along −2, the axis `·`
/// contracts through the fold that slices it; its vectors none. The solve's
/// measurement, on the kernel interpreter with the weights quantized, is
/// inside the budget, the terms still sum to the estimate, and the
/// realization's `quant` map reads back as written.
#[test]
fn one_menu_offers_the_matrices_their_integers() {
    use presk_precision::solve::{Menu, Rung};
    let file = root().join("examples/tutorial/07_train.psk");
    let (_, prog, mut core) = program(&file, &[]);
    let c = presk_driver::precision::Calibrate { file: &file, entry: "loss", dims: vec![("B".into(), 4)], consts: &[], data: None, samples: None, from: 0, seed: 0, bits: 128, weights: None };
    let env = presk_driver::precision::calibrate(&core, &prog, &c).unwrap();
    let menu = Menu::parse("f64,f32,bf16,int8,int4").unwrap();
    let a = presk_driver::precision::SolveArgs { budget: presk_precision::solve::Budget::Relative(0.5), weights: None, verify: true, bits: 128, seed: 0, profile: None, quant: None, menu, options: Default::default(), exec: Default::default() };
    let Ok(solved) = presk_driver::precision::solve(&mut core, &prog, &env, &a) else { panic!("the tutorial's loss solves") };
    let paths: Vec<(&str, usize)> = solved.params.iter().map(|p| (p.path.as_str(), p.ints.len())).collect();
    assert!(paths.iter().any(|(p, n)| *p == "blocks.proj.w" && *n == 4), "{paths:?}");
    assert!(paths.iter().filter(|(p, _)| *p != "blocks.proj.w").all(|(_, n)| *n == 0), "only the matrix: {paths:?}");
    let w = solved.params.iter().find(|p| p.path == "blocks.proj.w").unwrap();
    // Three layers of 4 × 4: int8 is 48 bytes and twelve f32 scales, as
    // many bytes as bf16; int4 is 24 and the same scales.
    assert!(w.ints.iter().all(|q| q.axis == -2 && q.contracted && q.round > 0.0), "{:?}", w.ints);
    assert_eq!(w.ints.iter().map(|q| q.bytes).collect::<Vec<_>>(), [96.0, 96.0, 72.0, 72.0], "{:?}", w.ints);
    let s = &solved.solution;
    assert!(s.measured.is_some_and(|g| g <= s.budget_abs), "{:?} against {}", s.measured, s.budget_abs);
    let q = s.realization.quant.get("blocks.proj.w").expect("a loose budget quantizes the matrix");
    assert_eq!(q.over, presk_rt::AxisRef::Pos(-2));
    assert!(!s.realization.params.contains_key("blocks.proj.w"));
    let c = solved.breakdown.iter().find(|c| c.name == "blocks.proj.w").unwrap();
    assert_eq!(c.rung, Rung::Int { bits: q.bits, scheme: q.scheme });
    let sum: f64 = solved.breakdown.iter().map(|s| s.err).sum();
    assert!((sum - s.estimated_error).abs() <= 1e-15 * s.estimated_error, "{sum} against {}", s.estimated_error);
    let path = std::env::temp_dir().join(format!("presk-one-menu-{}.json", std::process::id()));
    std::fs::write(&path, s.realization.to_json()).unwrap();
    let back = presk_rt::Realization::parse(&format!("@{}", path.display()));
    let _ = std::fs::remove_file(&path);
    assert_eq!(back.unwrap(), s.realization);
}

/// D180: a checkpoint's parameters are read against a tree that is never
/// drawn, the initializer on the kernel backend with no executor. That tree
/// is the drawn parameters' own (the same records and axes), the other
/// inputs are drawn as before, and a checkpoint reads to the same values
/// either way, bit for bit.
#[test]
fn given_params_have_the_drawn_tree() {
    let cases: [(&str, &str, &[&str], &[&str]); 5] = [
        ("examples/spec/minigpt/minigpt.psk", "MiniGPT.apply", &["B=2", "T=8"], &["V=64", "D=16", "H=2", "L=2"]),
        ("examples/spec/layernorm/layernorm.psk", "LayerNorm{8}.apply", &["batch=3"], &[]),
        ("examples/coverage/llama3.psk", "Llama3.apply", &["B=1", "T=8"], &["V=50", "D=32", "H=4", "G=2", "FF=64", "L=2"]),
        ("examples/coverage/llama.psk", "Llama.apply", &["B=2", "T=16"], &["V=100", "D=32", "H=2", "G=2", "FF=64", "L=2"]),
        ("examples/coverage/bert.psk", "BertSquad.apply", &["B=2", "T=16"], &["V=100", "D=32", "H=2", "L=2"]),
    ];
    for (file, entry, dims, consts) in cases {
        let consts: Vec<String> = consts.iter().map(|c| c.to_string()).collect();
        let (_, prog, core) = program(&root().join(file), &consts);
        let dims: Vec<(String, u64)> = dims.iter().map(|d| d.split_once('=').map(|(n, v)| (n.to_string(), v.parse().unwrap())).unwrap()).collect();
        let opts = presk_interp::RunOptions { entry: entry.into(), dims, seed: 7, bits: 64, digits: 12 };
        let rt = |inputs: &presk_interp::run::Inputs| inputs.iter().map(|(_, v)| presk_rt::Value::from_interp(v).unwrap()).collect::<Vec<_>>();
        let (_, _, drawn) = presk_interp::run::prepare_run(&core, &prog, &opts).unwrap();
        let drawn = rt(&drawn);
        let (_, _, given, described) = presk_interp::run::prepare_run_given_params(&core, &prog, &opts).unwrap();
        let given = rt(&given);
        assert_eq!(given[1..], drawn[1..], "{entry}: the other inputs");
        let Some(s @ presk_interp::run::Shaped::Params { init, dims, key }) = &described else { panic!("{entry}: the first argument is a module's parameters") };
        let zeros = presk_driver::layout::params_zeros(&core, &prog, *init, dims, key).unwrap();
        // The same tree: records, names, axes; every number zero.
        fn shape(v: &presk_rt::Value) -> String {
            match v {
                presk_rt::Value::Record(fs) => format!("{{{}}}", fs.iter().map(|(n, x)| format!("{n}: {}", shape(x))).collect::<Vec<_>>().join(", ")),
                presk_rt::Value::Tuple(ts) => format!("({})", ts.iter().map(shape).collect::<Vec<_>>().join(", ")),
                presk_rt::Value::Real(t) => format!("{:?}", t.axes),
                other => format!("{other:?}"),
            }
        }
        assert_eq!(shape(&zeros), shape(&drawn[0]), "{entry}: the tree");
        let path = std::env::temp_dir().join(format!("presk-given-params-{}.psk.d", std::process::id()));
        presk_rt::save_params(&path, "M", &drawn[0]).unwrap();
        let read = presk_driver::entry::read_given_params(&core, &prog, s, &path).unwrap();
        let before = presk_rt::load_params(&path, &drawn[0]).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(read, before, "{entry}: the checkpoint read either way");
        // The drawn tree holds the interpreter's exact numbers; the file, f64.
        assert_eq!(read, drawn[0].realize(presk_rt::Dtype::F64), "{entry}: the checkpoint is the drawn parameters");
    }
}

/// What a solve's moves decided: a proposal's order and choices, and every
/// measured move whole. A proposal's `added` and `spent` are the estimate's
/// sums, which a sweep in binary64 moves in their last bits.
fn decided(moves: &[presk_precision::solve::Move]) -> Vec<String> {
    use presk_precision::solve::Move;
    moves
        .iter()
        .map(|m| match m {
            Move::Tried { site, to, saved, taken, rank, .. } => format!("tried {site} {to:?} {saved} {taken} {rank}"),
            other => format!("{other:?}"),
        })
        .collect()
}

/// D185 (plan 0.17.1 §4): the sweep in binary64 and the measurements on the
/// native backend change no decision. On the tutorial's loss and on a small
/// `examples/coverage/llama3.psk` under the one menu, narrowed by measurement, the
/// solve under either option and under both returns the default's
/// realization, its decisions and every measured gap bit for bit; the
/// sweep's sensitivities are the exact sweep's to binary64's rounding.
#[test]
fn the_executors_change_no_decision() {
    use presk_driver::precision::{Measure, SolveExec, Sweep};
    use presk_precision::solve::{Narrow, Options};
    let cases: [(&str, &str, &[(&str, u64)], &[&str]); 2] = [
        ("examples/tutorial/07_train.psk", "loss", &[("B", 4)], &[]),
        ("examples/coverage/llama3.psk", "Llama3.apply", &[("B", 2), ("T", 8)], &["V=64", "D=32", "H=4", "G=2", "FF=64", "L=2"]),
    ];
    for (file, entry, dims, consts) in cases {
        let file = root().join(file);
        let consts: Vec<String> = consts.iter().map(|c| c.to_string()).collect();
        let (_, prog, mut core) = program(&file, &consts);
        let dims: Vec<(String, u64)> = dims.iter().map(|(n, v)| (n.to_string(), *v)).collect();
        let c = presk_driver::precision::Calibrate { file: &file, entry, dims, consts: &consts, data: None, samples: None, from: 0, seed: 0, bits: 128, weights: None };
        let env = presk_driver::precision::calibrate(&core, &prog, &c).unwrap();
        let menu = presk_precision::solve::Menu::parse("f64,f32,bf16,int8,int4").unwrap();
        let mut solve = |exec: SolveExec| {
            let a = presk_driver::precision::SolveArgs { budget: presk_precision::solve::Budget::Relative(1e-2), weights: None, verify: true, bits: 128, seed: 0, profile: None, quant: None, menu: menu.clone(), options: Options { narrow: Narrow::Measured, ..Options::default() }, exec };
            presk_driver::precision::solve(&mut core, &prog, &env, &a).ok().unwrap_or_else(|| panic!("{entry} solves"))
        };
        let base = solve(SolveExec::default());
        assert!(base.solution.measurements > 1, "{entry}: the narrowing measured");
        for exec in [SolveExec { sweep: Sweep::F64, measure: Measure::Kernel }, SolveExec { sweep: Sweep::Exact, measure: Measure::Native }, SolveExec { sweep: Sweep::F64, measure: Measure::Native }] {
            let s = solve(exec);
            assert_eq!(s.solution.realization, base.solution.realization, "{entry} under {exec:?}");
            assert_eq!(decided(&s.solution.moves), decided(&base.solution.moves), "{entry} under {exec:?}");
            assert_eq!(s.solution.measured.map(f64::to_bits), base.solution.measured.map(f64::to_bits), "{entry} under {exec:?}");
            assert_eq!(s.solution.measurements, base.solution.measurements, "{entry} under {exec:?}");
            assert_eq!(s.sens.sites.keys().collect::<Vec<_>>(), base.sens.sites.keys().collect::<Vec<_>>());
            for (site, x) in &base.sens.sites {
                let y = s.sens.sites[site];
                assert!((x - y).abs() <= 1e-12 * x.abs().max(1e-300), "{entry} {site}: {x} against {y} under {exec:?}");
            }
        }
    }
}

/// Plan 0.17.1 (D185): `nll` of `examples/coverage/llama3.psk` is the next-token
/// cross-entropy of `Llama3.apply`'s logits on the window's first T tokens,
/// at each position: on the reference at 128 bits, at a small width, its
/// value is log Σ exp z − z[target] computed in binary64 from the logits
/// of the same window's context, to 10⁻¹².
#[test]
fn llama3_nll_is_the_cross_entropy_of_its_logits() {
    use presk_interp::Value as I;
    let file = root().join("examples/coverage/llama3.psk");
    let consts: Vec<String> = ["V=64", "D=32", "H=4", "G=2", "FF=64", "L=2"].iter().map(|c| c.to_string()).collect();
    let (_, prog, core) = program(&file, &consts);
    let (b, t, v) = (2usize, 8usize, 64usize);
    let dims = vec![("B".to_string(), b as u64), ("T".to_string(), t as u64)];
    let opts = presk_interp::RunOptions { entry: "nll".into(), dims: dims.clone(), seed: 3, bits: 128, digits: 17 };
    let nll = presk_interp::run(&core, &prog, &opts).unwrap();
    let (theta, window) = (&nll.inputs[0].1, &nll.inputs[1].1);
    let I::Idx { bound, t: w } = window else { panic!("a window of ids") };
    assert_eq!(w.data.len(), b * (t + 1));
    // The context, the window's first T ids, as `Llama3.apply`'s argument.
    let ctx: Vec<i64> = (0..b).flat_map(|i| w.data[i * (t + 1)..i * (t + 1) + t].to_vec()).collect();
    let mut axes = w.axes.clone();
    axes[1].len = t;
    let ctx = I::Idx { bound: *bound, t: presk_interp::Tensor { axes, data: ctx.into_iter().collect() } };
    let at = presk_interp::RunOptions { entry: "Llama3.apply".into(), dims, seed: 3, bits: 128, digits: 17 };
    let (apply, fdims, _) = presk_interp::run::bind_run(&core, &prog, &at).unwrap();
    let mut it = presk_rt::InterpBackend::new(&core, &prog, 128);
    let z = it.interp.call(apply, fdims, vec![theta.clone(), ctx]).unwrap();
    let (I::Real(z), I::Real(got)) = (z, &nll.result) else { panic!("real results") };
    assert_eq!(got.data.len(), b * t);
    for i in 0..b {
        for p in 0..t {
            let row: Vec<f64> = (0..v).map(|k| z.data[(i * t + p) * v + k].to_f64()).collect();
            let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lse = m + row.iter().map(|x| (x - m).exp()).sum::<f64>().ln();
            let target = w.data[i * (t + 1) + p + 1] as usize;
            let want = lse - row[target];
            let have = got.data[i * t + p].to_f64();
            assert!((have - want).abs() <= 1e-12 * want.abs().max(1.0), "window {i}, position {p}: {have} against {want}");
        }
    }
}
