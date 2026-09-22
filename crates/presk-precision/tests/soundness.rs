//! The property a bound has to have: it must be **above** the error actually
//! committed. Tested where ground truth exists — the conformance entries,
//! whose exact values the reference interpreter supplies — and for every
//! realization the tool can run.
//!
//! The ratio between the two is recorded as well, because a bound that holds
//! by a factor of ten billion is sound and useless. That ratio is the raw
//! material of statement F of plan 0.5.0 §3.7, and this test prints it rather
//! than asserting a value: the number is the result, and step 9 reports it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Once;

use presk_interp::{Interp, RunOptions};
use presk_kernel::KernelBackend;
use presk_precision::envelope::Envelope;
use presk_precision::propagate::{Model, propagate_with};
use presk_rt::{Dtype, Realization, Value};
use presk_syntax::diag::render_all;
use presk_syntax::{SourceMap, parse_file};
use presk_types::check;
use presk_types::elab::{self, ElabProgram, Options};
use rug::Float;

type IValue = presk_interp::Value<Float>;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn chdir_root() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| std::env::set_current_dir(repo_root()).unwrap());
}

fn prepare(rel: &str, consts: &[(&str, u64)]) -> (ElabProgram, presk_core::Program) {
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
    let diags = presk_core::ad::transform(&mut core);
    assert!(diags.is_empty(), "{}", render_all(&diags, &sm, false));
    (prog, core)
}

/// The largest absolute difference over the leaves of two values.
fn gap(reference: &IValue, got: &Value) -> f64 {
    match (reference, got) {
        (IValue::Real(t), Value::Real(g)) => {
            let gd = g.data.to_f64();
            t.data
                .iter()
                .zip(&gd)
                .map(|(x, y)| {
                    let xf = x.to_f64();
                    if xf.is_infinite() && y.is_infinite() { 0.0 } else { (xf - y).abs() }
                })
                .filter(|d| d.is_finite())
                .fold(0.0, f64::max)
        }
        (IValue::Tuple(xs), Value::Tuple(ys)) => xs.iter().zip(ys).map(|(x, y)| gap(x, y)).fold(0.0, f64::max),
        (IValue::Record(xs), Value::Record(ys)) => xs.iter().zip(ys).map(|((_, x), (_, y))| gap(x, y)).fold(0.0, f64::max),
        _ => 0.0,
    }
}

/// `(measured gap, propagated bound)` for one entry under one realization,
/// the bound by the default rule.
fn measure(rel: &str, consts: &[(&str, u64)], entry: &str, dims: &[(&str, u64)], r: &Realization) -> (f64, f64) {
    let (g, b) = measure_all(rel, consts, entry, dims, r);
    (g, b[Model::ALL.iter().position(|m| *m == Model::default()).unwrap()])
}

/// The measured gap, and what each of the four rules propagates, in the
/// order of [`Model::ALL`].
fn measure_all(rel: &str, consts: &[(&str, u64)], entry: &str, dims: &[(&str, u64)], r: &Realization) -> (f64, [f64; 4]) {
    let (prog, core) = prepare(rel, consts);
    let opts = RunOptions { entry: entry.into(), dims: dims.iter().map(|(n, v)| (n.to_string(), *v)).collect(), seed: 0, bits: 256, digits: 17 };

    // The exact run, recording the magnitudes the bound will be conditional on.
    let (func, fdims, inputs) = presk_interp::run::prepare_run(&core, &prog, &opts).unwrap();
    let args: Vec<IValue> = inputs.iter().map(|(_, v)| v.clone()).collect();
    let mut interp: Interp<Float> = Interp::new(&core, &prog, 256, 0);
    interp.magnitudes = Some(Default::default());
    interp.minima = Some(Default::default());
    interp.terms = Some(Default::default());
    let reference = interp.call(func, fdims.clone(), args.clone()).unwrap();
    let mut env = Envelope::new(rel, entry, 256, "2026-09-20");
    env.observe(&interp.magnitudes.take().unwrap());
    env.observe_minima(&interp.minima.take().unwrap());
    env.observe_terms(&interp.terms.take().unwrap());

    // The same entry on the kernel backend, in this realization.
    let mut backend = KernelBackend::new(&core, &prog, r, 0);
    let rargs: Vec<Value> = args.iter().map(|v| Value::from_interp(v).unwrap().realize(r.default)).collect();
    let p = presk_rt::Backend::prepare(&mut backend, func, &fdims, r).unwrap();
    let got = presk_rt::Backend::call(&mut backend, &p, rargs).unwrap();

    let dmap: BTreeMap<_, _> = fdims.iter().map(|(v, n)| (*v, *n)).collect();
    let bounds = Model::ALL.map(|m| {
        let a = propagate_with(&core, &prog, func, &dmap, &env, r, m);
        assert!(m == Model::Worst || a.fallback.is_empty(), "{rel} {entry}: no terms recorded at {:?}", a.fallback);
        a.output.err
    });
    (gap(&reference, &got), bounds)
}

type Case = (&'static str, &'static [(&'static str, u64)], &'static str, &'static [(&'static str, u64)]);
const TINY: &[(&str, u64)] = &[("V", 64), ("D", 16), ("H", 2), ("L", 2)];

const CASES: &[Case] = &[
    ("tests/examples/mlp/mlp.psk", &[], "mse", &[("D", 4), ("B", 2)]),
    ("tests/examples/mlp/mlp.psk", &[], "mse_grad", &[("D", 4), ("B", 2)]),
    ("tests/examples/layernorm/layernorm.psk", &[], "LayerNorm{8}.apply", &[("batch", 3)]),
    ("tests/examples/layernorm/layernorm.psk", &[], "RMSNorm{8}.apply", &[("batch", 3)]),
    ("tests/examples/minigpt/minigpt.psk", TINY, "MiniGPT.apply", &[("B", 2), ("T", 8)]),
    ("tests/examples/minigpt/train_minigpt.psk", TINY, "loss", &[("B", 2), ("T", 8)]),
    ("tests/data/split_grad.psk", &[], "loss_grad", &[("B", 2)]),
    ("tests/train/mlp_regression.psk", &[], "loss", &[("B", 4)]),
];

/// A propagated bound must be above the error actually committed, for every
/// entry and every realization. This is the whole claim of the analysis, and
/// it is checkable exactly here because the reference interpreter gives the
/// exact value to compare against.
///
/// Four rules for sums and contractions are tabulated side by side (0.6.0
/// step 6, D51): the `k` rule of 0.5.0, the sound normwise one, the
/// condition number, and the root-sum-square estimate. The three bounds must
/// all be above the measurement; the estimate is not a bound, and how often
/// it falls below is reported rather than asserted.
#[test]
fn the_bound_is_above_the_measured_gap() {
    let mut report = String::from("\nentry                   realization   measured         k  normwise      cond       rss   k/gap  norm/gap cond/gap  rss/gap\n");
    let mut ratios: [Vec<f64>; 4] = Default::default();
    for (rel, consts, entry, dims) in CASES {
        for d in [Dtype::F64, Dtype::F32, Dtype::Bf16] {
            let r = Realization::uniform(d);
            let (measured, bounds) = measure_all(rel, consts, entry, dims, &r);
            for (m, bound) in Model::ALL.iter().zip(bounds) {
                assert!(
                    !m.is_bound() || bound >= measured || measured == 0.0,
                    "{rel} {entry} ({d}): measured {measured:.3e} exceeds the {} bound {bound:.3e} — the rule is unsound",
                    m.name()
                );
            }
            let r = bounds.map(|b| if measured > 0.0 { b / measured } else { f64::NAN });
            for (v, x) in ratios.iter_mut().zip(r) {
                if x.is_finite() {
                    v.push(x);
                }
            }
            report.push_str(&format!(
                "{entry:<22}  {d:<10} {measured:>10.2e} {:>9.2e} {:>9.2e} {:>9.2e} {:>9.2e} {:>7.1e} {:>9.1e} {:>8.1e} {:>8.1e}\n",
                bounds[0], bounds[1], bounds[2], bounds[3], r[0], r[1], r[2], r[3]
            ));
        }
    }
    for (m, v) in Model::ALL.iter().zip(ratios.iter_mut()) {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        report.push_str(&format!(
            "\n{:<9} {} ratios: median {:.1e}, worst {:.1e}, {} within 1e3, {} below 1",
            m.name(),
            v.len(),
            v[v.len() / 2],
            v.last().copied().unwrap_or(f64::NAN),
            v.iter().filter(|r| **r <= 1e3).count(),
            v.iter().filter(|r| **r < 1.0).count(),
        ));
    }
    report.push('\n');
    eprintln!("{report}");
    // A smoke test: the medians are the result and D51 reports them.
    assert!(ratios[0].iter().all(|r| *r < 1e25), "a bound is loose enough to be a missing rule:\n{report}");
}

/// The declared reduction order is an argument of the analysis, not
/// bookkeeping: interleaved partials reassociate, so the same program in the
/// same dtype must carry a *different* bound (D31, D34).
#[test]
fn the_reduction_order_changes_the_bound() {
    let seq = Realization::uniform(Dtype::F32);
    let mut blocked = seq.clone();
    blocked.order = presk_rt::ReductionOrder::Blocked { width: 8 };
    let mut inter = seq.clone();
    inter.order = presk_rt::ReductionOrder::Interleaved { partials: 8 };
    // A program whose output depends on a long contraction: MiniGPT at
    // initialization has a zero head, so under the conditioned rule its
    // whole bound is exactly zero — a product with an exact zero commits no
    // error — and no order can shorten it.
    let (rel, consts, entry, dims) = ("tests/examples/mlp/mlp.psk", &[][..], "mse", &[("D", 32), ("B", 4)][..]);
    let (_, b_seq) = measure(rel, consts, entry, dims, &seq);
    let (_, b_blocked) = measure(rel, consts, entry, dims, &blocked);
    let (_, b_inter) = measure(rel, consts, entry, dims, &inter);
    assert_eq!(b_seq, b_blocked, "blocking keeps every addition in order, so it cannot change the bound (D32)");
    assert!(b_inter < b_seq, "eight partials shorten the chain: {b_inter:.3e} against {b_seq:.3e}");
    assert!(b_inter > 0.0);
}

/// A sensitivity is read off the adjoints a reverse sweep already computes,
/// so the correspondence between a forward value and its adjoint has to
/// survive the differentiation pass. This checks that it does, and that the
/// ranking is not the degenerate all-zero one a zero-initialized head gives.
#[test]
fn sensitivities_come_off_one_reverse_sweep() {
    use presk_precision::sensitivity;
    let (prog, mut core) = prepare("tests/train/mlp_regression.psk", &[]);
    let (entry, _, _) = presk_interp::run::resolve_entry(&core, &prog, "loss").unwrap();
    let (vjp, adjoints, diags) = presk_core::ad::vjp_with_adjoints(&mut core, entry);
    assert!(diags.is_empty(), "the objective differentiates");
    assert!(!adjoints.is_empty(), "the adjoint table is not empty");

    let opts = RunOptions { entry: "loss".into(), dims: vec![("B".into(), 4)], seed: 0, bits: 128, digits: 17 };
    let (_, fdims, inputs) = presk_interp::run::prepare_run(&core, &prog, &opts).unwrap();
    let mut args: Vec<IValue> = inputs.iter().map(|(_, v)| v.clone()).collect();
    args.push(IValue::Real(presk_interp::Tensor { axes: vec![], data: vec![Float::with_val(128, 1)] }));
    let mut interp: Interp<Float> = Interp::new(&core, &prog, 128, 0);
    interp.sums = Some(Default::default());
    interp.call(vjp, fdims, args).unwrap();
    let s = sensitivity::from_run(&core, entry, &adjoints, &interp.sums.take().unwrap());

    let ranked = s.ranked();
    assert!(!ranked.is_empty(), "every site of the forward body gets an adjoint");
    assert!(ranked[0].1 > 0.0, "the ranking is not all zero: {:?}", &ranked[..ranked.len().min(3)]);
    // The cotangent seeded at 1 is the output's own sensitivity, so nothing
    // may exceed it by more than the chain that follows.
    assert!(ranked.iter().all(|(_, v)| v.is_finite()), "every sensitivity is finite");
    // Modules aggregate: the per-module view must account for every site.
    let total: f64 = s.by_module().iter().map(|(_, v)| v).sum();
    let sites: f64 = s.sites.values().sum();
    assert!((total - sites).abs() < 1e-9 * sites.max(1.0), "the per-module view loses nothing");
}

/// The solver's contract: what it returns must be inside the budget when
/// *measured* against the exact reference, not merely when estimated. And on
/// a program small enough to search exhaustively, the greedy proposal should
/// be at or near the optimum of the same estimate — which is the only fair
/// comparison, since the search never runs a measurement.
#[test]
fn the_solver_stays_inside_the_budget_and_near_the_optimum() {
    use presk_precision::{sensitivity, solve};
    let rel = "tests/train/mlp_regression.psk";
    let (prog, mut core) = prepare(rel, &[]);
    let (entry, _, _) = presk_interp::run::resolve_entry(&core, &prog, "loss").unwrap();
    let opts = RunOptions { entry: "loss".into(), dims: vec![("B".into(), 4)], seed: 0, bits: 128, digits: 17 };
    let (_, fdims, inputs) = presk_interp::run::prepare_run(&core, &prog, &opts).unwrap();
    let args: Vec<IValue> = inputs.iter().map(|(_, v)| v.clone()).collect();

    // Magnitudes, element counts and the exact value.
    let mut interp: Interp<Float> = Interp::new(&core, &prog, 128, 0);
    interp.magnitudes = Some(Default::default());
    interp.minima = Some(Default::default());
    interp.elems = Some(Default::default());
    let reference = interp.call(entry, fdims.clone(), args.clone()).unwrap();
    let mut env = Envelope::new(rel, "loss", 128, "2026-09-20");
    env.observe(&interp.magnitudes.take().unwrap());
    env.observe_minima(&interp.minima.take().unwrap());
    env.observe_elems(&interp.elems.take().unwrap());

    // Sensitivities, from one reverse sweep.
    let (vjp, adjoints, _) = presk_core::ad::vjp_with_adjoints(&mut core, entry);
    let mut sb: Interp<Float> = Interp::new(&core, &prog, 128, 0);
    sb.sums = Some(Default::default());
    let mut vargs = args.clone();
    vargs.push(IValue::Real(presk_interp::Tensor { axes: vec![], data: vec![Float::with_val(128, 1)] }));
    sb.call(vjp, fdims.clone(), vargs).unwrap();
    let sens = sensitivity::from_run(&core, entry, &adjoints, &sb.sums.take().unwrap());

    let output_site = format!("{}/%{}", core.func(entry).name, core.func(entry).body.result.0);
    let menu = solve::Menu::default();
    let rargs: Vec<Value> = args.iter().map(|v| Value::from_interp(v).unwrap()).collect();
    for budget in [1e-3, 1e-6] {
        let mut measure = |r: &Realization| -> f64 {
            let mut b = KernelBackend::new(&core, &prog, r, 0);
            let a: Vec<Value> = rargs.iter().map(|v| v.realize(r.default)).collect();
            let p = presk_rt::Backend::prepare(&mut b, entry, &fdims, r).unwrap();
            gap(&reference, &presk_rt::Backend::call(&mut b, &p, a).unwrap())
        };
        let sol = solve::solve(&env, &sens, &output_site, budget, &menu, Dtype::F64, &solve::Traffic, Some(&mut measure));
        let measured = sol.measured.expect("a verifier was supplied");
        assert!(
            measured <= sol.budget_abs,
            "budget {budget:.0e}: the solver returned an assignment whose measured gap {measured:.3e} exceeds {:.3e}",
            sol.budget_abs
        );
        assert!(sol.bytes <= sol.baseline_bytes, "narrowing cannot increase traffic");
        assert!(sol.bytes >= sol.relaxation_bytes, "and cannot beat the relaxation floor");

        // The proposal against the optimum of the same estimate.
        if let Some((_, best)) = solve::exhaustive(&env, &sens, &output_site, budget, &menu, Dtype::F64, &solve::Traffic, 200_000) {
            let ratio = sol.proposed_bytes / best.max(1.0);
            assert!(ratio >= 0.999, "the greedy proposal cannot beat an exhaustive search of the same estimate");
            assert!(ratio <= 1.25, "budget {budget:.0e}: greedy is {ratio:.3}x the optimum, which is worse than the heuristic should be");
        }
    }
}

/// The declared reduction order now reaches the *kernel*, not only the
/// bound: an interleaved order emits a different loop nest, which
/// reassociates, so it must give a different number from the sequential one
/// and a larger gap to the exact reference — while both executors of that
/// same order must agree with each other to the last bit.
#[test]
fn an_interleaved_order_reassociates_and_says_so() {
    use presk_rt::ReductionOrder;
    // The program has to *depend* on a long contraction for a reassociation
    // to be visible at all. Neither MiniGPT entry does at initialization: a
    // zero-initialized head makes the logits identically zero and the loss
    // exactly `ln V`, whatever happens upstream. The MLP does.
    let rel = "tests/examples/mlp/mlp.psk";
    let consts: &[(&str, u64)] = &[];
    let (prog, core) = prepare(rel, consts);
    let opts = RunOptions { entry: "mse".into(), dims: vec![("D".into(), 32), ("B".into(), 4)], seed: 0, bits: 256, digits: 17 };
    let (func, fdims, inputs) = presk_interp::run::prepare_run(&core, &prog, &opts).unwrap();
    let args: Vec<IValue> = inputs.iter().map(|(_, v)| v.clone()).collect();
    let mut interp: Interp<Float> = Interp::new(&core, &prog, 256, 0);
    let reference = interp.call(func, fdims.clone(), args.clone()).unwrap();

    let run = |r: &Realization| -> (f64, Vec<f64>) {
        let mut b = KernelBackend::new(&core, &prog, r, 0);
        let a: Vec<Value> = args.iter().map(|v| Value::from_interp(v).unwrap().realize(r.default)).collect();
        let p = presk_rt::Backend::prepare(&mut b, func, &fdims, r).unwrap();
        let out = presk_rt::Backend::call(&mut b, &p, a).unwrap();
        let Value::Real(t) = &out else { panic!("a scalar loss") };
        (gap(&reference, &out), t.data.to_f64())
    };

    let seq = Realization::uniform(Dtype::F32);
    let mut inter = seq.clone();
    inter.order = ReductionOrder::Interleaved { partials: 8 };
    let mut blocked = seq.clone();
    blocked.order = ReductionOrder::Blocked { width: 8 };

    let (g_seq, v_seq) = run(&seq);
    let (g_int, v_int) = run(&inter);
    let (_, v_blk) = run(&blocked);

    // Blocking changes no addition's order (D32), so it cannot change a bit.
    assert_eq!(v_seq.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), v_blk.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), "a blocked order is bit-identical to a sequential one");
    // Interleaving reassociates, so it must *not* be bit-identical: a
    // declared order that the kernel ignores would be the worst outcome of
    // this whole design, and this is the test that catches it.
    assert_ne!(
        v_seq.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        v_int.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "an interleaved order must actually reassociate"
    );
    assert!(g_int.is_finite() && g_seq.is_finite());
    eprintln!("gap to the exact reference: sequential {g_seq:.3e}, interleaved(8) {g_int:.3e}");
}

/// What a contraction records for the conditioned and statistical rules
/// (0.6.0 step 6), on one dot product checked by hand: `x = [1, −2]`,
/// `w = [3, 1]`, so `Σxᵢwᵢ = 1` while `Σ|xᵢwᵢ| = 5` — a condition number of
/// five — `√Σ(xᵢwᵢ)² = √13`, and the normwise factors are `Σ|w| = 4` for
/// `x`'s error and `Σ|x| = 3` for `w`'s.
#[test]
fn a_contraction_records_its_sums_of_terms() {
    let (prog, core) = prepare("tests/data/terms.psk", &[]);
    let (f, _, _) = presk_interp::run::resolve_entry(&core, &prog, "f").unwrap();
    let t = |xs: &[f64]| IValue::Real(presk_interp::Tensor { axes: vec![presk_interp::value::Axis { name: None, len: xs.len() }], data: xs.iter().map(|x| Float::with_val(128, *x)).collect() });
    let mut interp: Interp<Float> = Interp::new(&core, &prog, 128, 0);
    interp.magnitudes = Some(Default::default());
    interp.terms = Some(Default::default());
    interp.call(f, Default::default(), vec![t(&[1.0, -2.0]), t(&[3.0, 1.0])]).unwrap();
    let mut env = Envelope::new("tests/data/terms.psk", "f", 128, "2026-09-21");
    env.observe(&interp.magnitudes.take().unwrap());
    env.observe_terms(&interp.terms.take().unwrap());
    let site = "f/%2";
    let terms = env.terms(site).expect("the dot recorded its terms").clone();
    assert_eq!(terms.abs, 5.0);
    assert!((terms.rss - 13f64.sqrt()).abs() < 1e-15);
    assert_eq!(terms.norms, vec![4.0, 3.0]);
    assert_eq!(env.condition(site), Some(5.0));
    // The file form carries them, and an envelope without them still loads.
    let back: Envelope = serde_json::from_str(&env.to_json()).unwrap();
    assert_eq!(back, env);
    let mut old = env.clone();
    old.terms.clear();
    let old: Envelope = serde_json::from_str(&old.to_json()).unwrap();
    assert!(old.terms.is_empty());
    // Without them the rule falls back to `k`, and says where.
    let dims = BTreeMap::new();
    let r = Realization::uniform(Dtype::F32);
    let a = propagate_with(&core, &prog, f, &dims, &old, &r, Model::Conditioned);
    assert_eq!(a.fallback, vec![site.to_string()]);
    let a = propagate_with(&core, &prog, f, &dims, &env, &r, Model::Conditioned);
    assert!(a.fallback.is_empty());
}
