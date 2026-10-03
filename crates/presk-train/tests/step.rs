//! One training step as the thing a precision budget binds (plan 0.6.0
//! §3.7, step 7b): the state, the exact step, the gaps, and the solver over
//! the step's functions, on the tiny MiniGPT.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use presk_kernel::KernelBackend;
use presk_precision::envelope::Envelope;
use presk_precision::solve::{Menu, Traffic, solve_step, step_groups};
use presk_rt::{Dtype, InterpBackend, Realization};
use presk_syntax::diag::render_all;
use presk_syntax::{SourceMap, parse_file};
use presk_train::step::{StepOptions, StepSetup, setup};
use presk_train::{TrainOptions, TrainingFns, prepare_training, train};
use presk_types::check::{self, TypedProgram};
use presk_types::elab::{self, ElabProgram, Options};

struct Tiny {
    prog: ElabProgram,
    typed: TypedProgram,
    core: presk_core::Program,
    fns: TrainingFns,
    ckpt: PathBuf,
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

/// The tiny MiniGPT, and a checkpoint of five native `f64` steps (a trained
/// state rather than the initialisation, so that the head is not zero).
fn tiny() -> &'static Tiny {
    static T: OnceLock<Tiny> = OnceLock::new();
    T.get_or_init(|| {
        std::env::set_current_dir(root()).unwrap();
        let mut sm = SourceMap::new();
        let id = sm.load(Path::new("tests/examples/minigpt/train_minigpt.psk")).unwrap();
        let (program, diags) = parse_file(&sm, id);
        assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
        let opts = Options { const_overrides: [("V", 64), ("D", 16), ("H", 2), ("L", 2)].iter().map(|(n, v)| (n.to_string(), *v)).collect() };
        let (prog, _) = elab::elaborate(&mut sm, id, program, &opts);
        let (typed, _) = check::check(&sm, &prog);
        let (mut core, _) = presk_core::lower(&prog, &typed);
        let _ = presk_core::ad::transform(&mut core);
        let fns = prepare_training(&mut core, &prog, &typed).unwrap();
        let dir = std::env::temp_dir().join(format!("presk-step-{}", std::process::id()));
        let topts = TrainOptions { data: None, steps: Some(5), epochs: None, seed: 0, bits: 64, realize: Realization::uniform(Dtype::F64), ckpt: Some(dir.clone()), checkpoints: true };
        let mut b = KernelBackend::new(&core, &prog, &Realization::uniform(Dtype::F64));
        train(&mut b, &core, &prog, &typed, &fns, &topts, |_| {}).unwrap();
        Tiny { prog, typed, core, fns, ckpt: dir.join("final.psk.d") }
    })
}

fn state(t: &Tiny, warmup: u64) -> StepSetup {
    let opts = StepOptions { weights: t.ckpt.clone(), warmup, data: None, from: 0, samples: 2, steps: None, seed: 0 };
    let mut warm = KernelBackend::new(&t.core, &t.prog, &Realization::uniform(Dtype::F64));
    setup(&mut warm, &t.core, &t.prog, &t.typed, &t.fns, &opts).unwrap()
}

/// All-`f64` is the exact step up to one rounding of `θ′`, which is the
/// floor no arithmetic can go below; all-`f32` is measurably further; and
/// the exact step reaches the reverse sweep's and the optimizer's sites.
#[test]
fn f64_is_the_exact_step_and_f32_is_not() {
    let t = tiny();
    let s = state(t, 30);
    assert_eq!(s.t, 31);
    let mut exact = InterpBackend::new(&t.core, &t.prog, 128);
    exact.interp.magnitudes = Some(Default::default());
    let e64 = s.step(&mut exact, &Realization::uniform(Dtype::F64)).unwrap();
    let sites = exact.interp.magnitudes.take().unwrap();
    assert!(sites.keys().any(|k| k.starts_with("adamw.step/")), "the optimizer's sites are measured");
    assert!(sites.keys().any(|k| k.contains(".vjp/")), "the reverse sweep's sites are measured");

    let one = |d: Dtype| {
        let r = Realization::uniform(d);
        let mut b = KernelBackend::new(&t.core, &t.prog, &r);
        (r.clone(), s.step(&mut b, &r).unwrap())
    };
    let (r64, n64) = one(Dtype::F64);
    let g64 = s.gap(&r64, &e64, &n64);
    assert!(g64.grad < 1e-12, "f64 gradient {:e}", g64.grad);
    assert!(g64.update < 1e-11 && g64.update < 1.1 * g64.floor + 1e-13, "f64 update {:e}, floor {:e}", g64.update, g64.floor);

    let (r32, n32) = one(Dtype::F32);
    let mut exact32 = InterpBackend::new(&t.core, &t.prog, 128);
    let e32 = s.step(&mut exact32, &r32).unwrap();
    let g32 = s.gap(&r32, &e32, &n32);
    assert!(g32.grad > 1e3 * g64.grad && g32.update > 1e3 * g64.update, "f32 {g32:?} against f64 {g64:?}");
    assert!(g32.update >= 0.9 * g32.floor, "the storage floor is part of the gap: {g32:?}");

    // Two batches' gradients are far apart: that is the scale.
    let mut b = KernelBackend::new(&t.core, &t.prog, &r64);
    let other = s.step_other_batch(&mut b, &r64).unwrap();
    assert!(presk_train::step::sampling_noise(&n64, &other) > 1e-2);
}

/// `presk train --realize` hands the one realization to every prepared
/// function: narrowing only the optimizer changes the run, and naming a
/// function the program does not have changes nothing.
#[test]
fn a_realization_by_function_reaches_the_optimizer() {
    let t = tiny();
    let run = |r: Realization| {
        let o = TrainOptions { data: None, steps: Some(2), epochs: None, seed: 0, bits: 64, realize: r.clone(), ckpt: None, checkpoints: false };
        let mut b = KernelBackend::new(&t.core, &t.prog, &r);
        train(&mut b, &t.core, &t.prog, &t.typed, &t.fns, &o, |_| {}).unwrap().theta
    };
    let base = run(Realization::uniform(Dtype::F64));
    let mut opt = Realization::uniform(Dtype::F64);
    opt.functions.insert("adamw.step".into(), Dtype::Bf16);
    assert!(run(opt).dist2_f64(&base) > 0.0, "adamw.step in bf16 moved θ");
    let mut none = Realization::uniform(Dtype::F64);
    none.functions.insert("no.such.function".into(), Dtype::Bf16);
    assert_eq!(run(none).dist2_f64(&base), 0.0);
}

/// The solver over the step: inside its budget by measurement, and a budget
/// too tight for `f32` anywhere gives all-`f64`.
#[test]
fn the_step_solver_is_inside_its_budget() {
    let t = tiny();
    let s = state(t, 30);
    let mut exact = InterpBackend::new(&t.core, &t.prog, 128);
    exact.interp.magnitudes = Some(Default::default());
    exact.interp.elems = Some(Default::default());
    let e64 = s.step(&mut exact, &Realization::uniform(Dtype::F64)).unwrap();
    let mut env = Envelope::new("tiny", "step", 128, "today");
    env.observe(&exact.interp.magnitudes.take().unwrap());
    env.observe_elems(&exact.interp.elems.take().unwrap());
    let menu = Menu::default();
    let groups = step_groups(&env, &menu, &Traffic);
    assert!(groups.iter().any(|g| g.name == "adamw.step") && groups.iter().any(|g| g.name.ends_with(".vjp")));
    let mut measure = |r: &Realization| {
        let mut b = KernelBackend::new(&t.core, &t.prog, r);
        s.gap(r, &e64, &s.step(&mut b, r).unwrap()).update
    };
    let sol = solve_step(groups.clone(), 1e-2, &menu, Dtype::F64, &mut measure, true);
    assert!(sol.measured.unwrap() <= 1e-2, "{:?}", sol.measured);
    assert!(sol.chosen.values().any(|d| *d != Dtype::F64), "a 1e-2 budget narrows something");
    assert!(sol.speedup() > 1.0);

    let tight = solve_step(groups, 1e-14, &menu, Dtype::F64, &mut measure, true);
    assert!(tight.realization.functions.is_empty(), "{:?}", tight.chosen);
    assert_eq!(tight.chosen, BTreeMap::from_iter(tight.chosen.keys().map(|k| (k.clone(), Dtype::F64))));
}
