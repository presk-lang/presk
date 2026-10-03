//! Statement T1 for the dead code a gradient loses after automatic
//! differentiation (plan 0.10.0 step 3, `presk_core::ad::prune`): every
//! gradient of the test set, the course GPT's included, gives the same bits
//! natively in `f64`, `f32` and `bf16` with the pass as without it, and
//! launches fewer kernels. `bf16` is the type the pass can get wrong: its
//! kernels keep a fusion group's intermediates in `f32`, so a value that
//! moved into a group would change.

use std::path::{Path, PathBuf};
use std::sync::Once;

use presk_cranelift::native_backend;
use presk_interp::RunOptions;
use presk_rt::{Backend, Dtype, Realization, Value};
use presk_syntax::diag::render_all;
use presk_syntax::{SourceMap, parse_file};
use presk_types::check;
use presk_types::elab::{self, ElabProgram, Options};

fn chdir_root() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let root: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap();
        std::env::set_current_dir(root).unwrap()
    });
}

/// The program after AD, with the pass or without it.
fn prepare(rel: &str, consts: &[(&str, u64)], prune: bool) -> (ElabProgram, presk_core::Program) {
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
    let diags = presk_core::ad::transform_with(&mut core, prune);
    assert!(diags.is_empty(), "{}", render_all(&diags, &sm, false));
    presk_core::verify(&core).unwrap_or_else(|e| panic!("{rel}: {}", e.join("; ")));
    (prog, core)
}

/// Every leaf's axes and bits.
fn leaves(v: &Value, out: &mut Vec<(Vec<presk_rt::Axis>, Vec<u64>)>) {
    match v {
        Value::Real(t) => out.push((t.axes.clone(), t.data.to_f64().into_iter().map(f64::to_bits).collect())),
        Value::Tuple(xs) => xs.iter().for_each(|x| leaves(x, out)),
        Value::Record(xs) => xs.iter().for_each(|(_, x)| leaves(x, out)),
        _ => {}
    }
}

/// The entry's output natively and its launches, `training` for `dropout`
/// to drop.
fn run(rel: &str, consts: &[(&str, u64)], entry: &str, dims: &[(&str, u64)], d: Dtype, training: bool, prune: bool) -> (Vec<(Vec<presk_rt::Axis>, Vec<u64>)>, u64) {
    let (prog, core) = prepare(rel, consts, prune);
    let realize = Realization::uniform(d);
    let opts = RunOptions { entry: entry.into(), dims: dims.iter().map(|(n, v)| (n.to_string(), *v)).collect(), seed: 0, bits: 256, digits: 17 };
    let (func, fdims, inputs) = presk_interp::run::prepare_run(&core, &prog, &opts).unwrap_or_else(|e| panic!("{entry}: {}", e.message));
    let args: Vec<Value> = inputs.iter().map(|(_, v)| Value::from_interp(v).unwrap()).collect();
    let mut b = native_backend(&core, &prog, &realize).unwrap();
    b.set_training(training);
    let p = b.prepare(func, &fdims, &realize).unwrap();
    let out = b.call(&p, args).unwrap_or_else(|e| panic!("{rel} {entry} ({d:?}): {}", e.message));
    let mut l = vec![];
    leaves(&out, &mut l);
    (l, b.launch_stats().0)
}

const TINY: &[(&str, u64)] = &[("V", 64), ("D", 16), ("H", 2), ("L", 2)];

#[test]
fn gradients_are_bit_identical_without_their_dead_code() {
    type Case = (&'static str, &'static [(&'static str, u64)], &'static str, &'static [(&'static str, u64)], bool);
    let cases: &[Case] = &[
        ("tests/examples/mlp/mlp.psk", &[], "mse_grad", &[("D", 4), ("B", 2)], false),
        ("tests/plan/minigpt_grad.psk", TINY, "loss_grad", &[("B", 2), ("T", 8)], false),
        ("experiments/gpt_grad.psk", &[], "loss_grad", &[("B", 2), ("T", 8)], false),
        ("tests/data/stack_grad.psk", &[], "loss_grad", &[("B", 2)], false),
        ("tests/data/stack_grad.psk", &[], "scale_grad", &[("B", 2)], false),
        ("tests/data/stack_prefix_grad.psk", &[], "loss_grad", &[("B", 3)], false),
        ("tests/data/dropout_grad.psk", &[], "loss_grad", &[("B", 2)], true),
        ("tests/data/stack_keys.psk", &[], "loss_grad", &[("B", 2)], true),
        ("tests/data/merge_grad.psk", &[], "loss_grad", &[("B", 2)], false),
        ("tests/data/split_grad.psk", &[], "loss_grad", &[("B", 2)], false),
        ("tests/data/closure_grad.psk", &[], "d_two", &[], false),
        ("tests/data/closure_grad.psk", &[], "d_captured", &[], false),
    ];
    let mut report = String::new();
    for (rel, consts, entry, dims, training) in cases {
        let mut launches = (0, 0);
        for d in [Dtype::F64, Dtype::F32, Dtype::Bf16] {
            let (with, n) = run(rel, consts, entry, dims, d, *training, true);
            let (without, m) = run(rel, consts, entry, dims, d, *training, false);
            assert_eq!(with, without, "{rel} {entry} ({d:?}): the pass changed a bit");
            assert!(n <= m, "{rel} {entry} ({d:?}): {n} launches with the pass, {m} without");
            launches = (m, n);
        }
        report.push_str(&format!("{rel} {entry}: {} → {} launches\n", launches.0, launches.1));
    }
    eprintln!("{report}");
}
