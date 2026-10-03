//! Statement T1 for residuals (plan 0.10.0 step 4, `presk_core::ad::store`):
//! every gradient of the test set, the course GPT's included, gives the same
//! bits natively in `f64`, `f32` and `bf16` when it keeps its forward pass's
//! residuals (`--remat store`) as when it recomputes them (`recompute`), with
//! `dropout` dropping or not, and launches fewer kernels. `bf16` is the type
//! a split can get wrong: its kernels keep a fusion group's intermediates in
//! `f32`, so a residual computed in other groups than the recomputing
//! `.vjp`'s forward copy computed it would round elsewhere. Training on is
//! where it can: `presk run` evaluates the expectation, every mask ones, and
//! a residual that differs by a mask's rounding does not differ there.
//!
//! Without the fusion groups kept (step 4b, D94: nothing held, every
//! residual shared), the same bits in `f64` and `f32`, where a group computes
//! what its members would alone; and under a realization that types a site
//! apart from its function, a group computes its members in its root's
//! type, so there the groups are kept and a stored gradient's value is its
//! function's own.

use std::path::{Path, PathBuf};
use std::sync::Once;

use presk_core::ad::{Options as Ad, Remat};
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

/// The program after AD under a policy.
fn prepare(rel: &str, consts: &[(&str, u64)], ad: Ad) -> (ElabProgram, presk_core::Program) {
    let (prog, mut core) = lowered(rel, consts);
    let diags = presk_core::ad::transform_as(&mut core, ad);
    assert!(diags.is_empty(), "{rel}: {diags:?}");
    presk_core::verify(&core).unwrap_or_else(|e| panic!("{rel}: {}", e.join("; ")));
    (prog, core)
}

/// The program before AD.
fn lowered(rel: &str, consts: &[(&str, u64)]) -> (ElabProgram, presk_core::Program) {
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
    let (core, diags) = presk_core::lower(&prog, &typed);
    assert!(diags.is_empty(), "{}", render_all(&diags, &sm, false));
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

/// Values of magnitudes 2⁻⁴ to 1, both signs, drawn from a fixed sequence.
fn drawn(n: usize, seed: &mut u64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            let m = 0.5 + (*seed >> 11) as f64 / (1u64 << 54) as f64;
            let e = ((*seed >> 3) % 5) as i32 - 4;
            (if *seed & 1 == 0 { m } else { -m }) * 2f64.powi(e)
        })
        .collect()
}

/// Every real leaf of a parameter tree drawn: the course GPT's head is zeros
/// at its initialization, and so is every gradient that reaches its blocks.
fn draw(v: &Value, seed: &mut u64) -> Value {
    v.map_leaves(&mut |leaf| match leaf {
        Value::Real(t) => Value::Real(presk_rt::Tensor { axes: t.axes.clone(), data: presk_rt::Buf::F64(drawn(t.data.to_f64().len(), seed).into()) }),
        other => other.clone(),
    })
}

/// The entry's output natively and its launches, its parameters drawn.
fn run(rel: &str, consts: &[(&str, u64)], entry: &str, dims: &[(&str, u64)], d: Dtype, training: bool, ad: Ad) -> (Vec<(Vec<presk_rt::Axis>, Vec<u64>)>, u64) {
    let (prog, core) = prepare(rel, consts, ad);
    let realize = Realization::uniform(d);
    let opts = RunOptions { entry: entry.into(), dims: dims.iter().map(|(n, v)| (n.to_string(), *v)).collect(), seed: 0, bits: 256, digits: 17 };
    let (func, fdims, inputs) = presk_interp::run::prepare_run(&core, &prog, &opts).unwrap_or_else(|e| panic!("{entry}: {}", e.message));
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let args: Vec<Value> = inputs.iter().map(|(_, v)| Value::from_interp(v).unwrap()).enumerate().map(|(i, v)| if i == 0 && matches!(v, Value::Record(_)) { draw(&v, &mut seed) } else { v }).collect();
    let mut b = native_backend(&core, &prog, &realize).unwrap();
    b.set_training(training);
    let p = b.prepare(func, &fdims, &realize).unwrap();
    let out = b.call(&p, args).unwrap_or_else(|e| panic!("{rel} {entry} ({d:?}, {}): {}", ad.remat.name(), e.message));
    let mut l = vec![];
    leaves(&out, &mut l);
    (l, b.launch_stats().0)
}

const TINY: &[(&str, u64)] = &[("V", 64), ("D", 16), ("H", 2), ("L", 2)];
const GPT: &[(&str, u64)] = &[("V", 92), ("D", 16), ("H", 2), ("L", 2)];

#[test]
fn stored_residuals_give_the_recomputed_bits() {
    type Case = (&'static str, &'static [(&'static str, u64)], &'static str, &'static [(&'static str, u64)], &'static [bool]);
    let cases: &[Case] = &[
        ("tests/examples/mlp/mlp.psk", &[], "mse_grad", &[("D", 4), ("B", 2)], &[false]),
        ("tests/plan/minigpt_grad.psk", TINY, "loss_grad", &[("B", 2), ("T", 8)], &[false]),
        ("experiments/gpt_grad.psk", GPT, "loss_grad", &[("B", 2), ("T", 8)], &[false, true]),
        ("tests/data/stack_grad.psk", &[], "loss_grad", &[("B", 2)], &[false]),
        ("tests/data/stack_grad.psk", &[], "scale_grad", &[("B", 2)], &[false]),
        ("tests/data/stack_prefix_grad.psk", &[], "loss_grad", &[("B", 3)], &[false]),
        ("tests/data/dropout_grad.psk", &[], "loss_grad", &[("B", 2)], &[false, true]),
        ("tests/data/stack_keys.psk", &[], "loss_grad", &[("B", 2)], &[false, true]),
        ("tests/data/stack_dropout_grad.psk", &[], "loss_grad", &[("B", 8)], &[false, true]),
        ("tests/data/merge_grad.psk", &[], "loss_grad", &[("B", 2)], &[false]),
        ("tests/data/split_grad.psk", &[], "loss_grad", &[("B", 2)], &[false]),
        ("tests/data/closure_grad.psk", &[], "d_two", &[], &[false]),
        ("tests/data/closure_grad.psk", &[], "d_captured", &[], &[false]),
    ];
    let mut report = String::new();
    let free = |remat| Ad { remat, groups: false };
    for (rel, consts, entry, dims, trainings) in cases {
        let mut launches = (0, 0, 0, 0);
        for &training in *trainings {
            for d in [Dtype::F64, Dtype::F32, Dtype::Bf16] {
                let (stored, n) = run(rel, consts, entry, dims, d, training, Remat::Store.into());
                let (recomputed, m) = run(rel, consts, entry, dims, d, training, Remat::Recompute.into());
                let zero = f64::to_bits(0.0);
                let nonzero = stored.iter().flat_map(|(_, b)| b).filter(|b| **b != zero && **b != f64::to_bits(-0.0)).count();
                let all = stored.iter().map(|(_, b)| b.len()).sum::<usize>();
                // The transformers' blocks are reached, not zeros.
                assert!(nonzero * 2 > all || !rel.contains("gpt"), "{rel} {entry} ({d:?}): {nonzero} of {all} gradient elements are not zero");
                assert_eq!(stored, recomputed, "{rel} {entry} ({d:?}, training {training}): storing changed a bit");
                assert!(n <= m, "{rel} {entry} ({d:?}): {n} launches storing, {m} recomputing");
                launches = (m, n, launches.2, launches.3);
                if d == Dtype::Bf16 {
                    continue;
                }
                // Nothing held, every residual shared: the same bits.
                let (stored_free, nf) = run(rel, consts, entry, dims, d, training, free(Remat::Store));
                let (recomputed_free, mf) = run(rel, consts, entry, dims, d, training, free(Remat::Recompute));
                assert_eq!(stored_free, recomputed, "{rel} {entry} ({d:?}, training {training}): storing without the groups changed a bit");
                assert_eq!(recomputed_free, recomputed, "{rel} {entry} ({d:?}, training {training}): removing the holds changed a bit");
                assert!(nf <= n && mf <= m, "{rel} {entry} ({d:?}): without the groups {nf} and {mf} launches, with them {n} and {m}");
                launches.2 = mf;
                launches.3 = nf;
            }
        }
        report.push_str(&format!("{rel} {entry}: {} → {} launches ({} → {} without the groups)\n", launches.0, launches.1, launches.2, launches.3));
    }
    eprintln!("{report}");
}

/// A group computes every member in its root site's type (D94): under a
/// realization that types the roots of `LayerNorm.apply`'s groups apart
/// from the rest, `f32` and `f64` mixed by site, sharing a residual those
/// groups keep inside would compute it in its own site's type. So such a
/// realization keeps the groups, and the value a stored gradient computes
/// for its objective, `loss.fwd`'s, is `loss`'s own bit for bit.
#[test]
fn a_site_typed_apart_keeps_the_groups() {
    let rel = "tests/plan/minigpt_grad.psk";
    let (prog, base) = lowered(rel, TINY);
    let mut r = Realization::uniform(Dtype::F32);
    let ln = base.funcs.iter().find(|f| f.name == "LayerNorm.apply").expect("LayerNorm.apply");
    for root in presk_core::fusion::plan(ln).groups.keys() {
        r.storage.insert(format!("LayerNorm.apply/%{}", root.0), Dtype::F64);
    }
    assert!(!r.storage.is_empty() && r.groups_visible());
    let opts = RunOptions { entry: "loss".into(), dims: vec![("B".into(), 2), ("T".into(), 8)], seed: 0, bits: 256, digits: 17 };
    let value = |groups: bool| -> (Vec<u64>, Vec<u64>) {
        let mut core = base.clone();
        assert!(presk_core::ad::transform_as(&mut core, Ad { remat: Remat::Store, groups }).is_empty());
        let loss = presk_core::FuncId(core.funcs.iter().position(|f| f.name == "loss").unwrap() as u32);
        let vv = presk_core::ad::value_vjp_function_as(&mut core, loss, Ad { remat: Remat::Store, groups });
        assert!(vv.diags.is_empty() && vv.again.is_empty());
        let (func, fdims, inputs) = presk_interp::run::prepare_run(&core, &prog, &opts).unwrap();
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let args: Vec<Value> = inputs.iter().map(|(_, v)| Value::from_interp(v).unwrap()).enumerate().map(|(i, v)| if i == 0 { draw(&v, &mut seed) } else { v }).collect();
        let mut b = native_backend(&core, &prog, &r).unwrap();
        let p = b.prepare(func, &fdims, &r).unwrap();
        let mut own = vec![];
        leaves(&b.call(&p, args.clone()).unwrap(), &mut own);
        let p = b.prepare(vv.value_vjp, &fdims, &r).unwrap();
        let mut with_ct = args;
        with_ct.push(Value::Real(presk_rt::Tensor { axes: vec![], data: presk_rt::Buf::F64(vec![1.0].into()) }));
        let Value::Tuple(out) = b.call(&p, with_ct).unwrap() else { panic!("value_vjp returns a tuple") };
        let mut stored = vec![];
        leaves(&out[0], &mut stored);
        (own[0].1.clone(), stored[0].1.clone())
    };
    let (own, kept) = value(true);
    assert_eq!(kept, own, "with the groups, loss.fwd's value is loss's");
    let (_, free) = value(false);
    eprintln!("loss {:e}; loss.fwd without the groups {:e} ({})", f64::from_bits(own[0]), f64::from_bits(free[0]), if free == own { "the same bits" } else { "other bits" });
}
