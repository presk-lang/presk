//! The conformance suite of the Cranelift backend (plan 0.3.0 §3.4): every example
//! entry and every tensor primitive against the reference interpreter at 256
//! bits, in f64 and in f32, within a stated tolerance per realization
//! `|x − x̂| ≤ ε·(|x| + a)`.

use std::path::{Path, PathBuf};
use std::sync::Once;

use presk_interp::{Interp, RunOptions};
use presk_cranelift::native_backend;
use presk_rt::{Realize, Value};
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
    presk_core::verify(&core).unwrap();
    (prog, core)
}

/// The tolerance of a realization: `(ε, a)`.
fn tol(r: Realize) -> (f64, f64) {
    match r {
        Realize::F64 => (2f64.powi(-40), 2f64.powi(-50)),
        Realize::F32 => (2f64.powi(-15), 2f64.powi(-20)),
    }
}

/// The largest `|x − x̂| / (|x| + a)` over the leaves, and the leaf path of the worst.
fn worst(reference: &IValue, got: &Value, a: f64, path: &str, out: &mut (f64, String)) {
    match (reference, got) {
        (IValue::Real(t), Value::Real(g)) => {
            assert_eq!(t.axes.iter().map(|x| x.len).collect::<Vec<_>>(), g.axes.iter().map(|x| x.len).collect::<Vec<_>>(), "{path}: shapes");
            let gd = g.data.to_f64();
            for (i, (x, y)) in t.data.iter().zip(&gd).enumerate() {
                let xf = x.to_f64();
                if xf.is_infinite() && y.is_infinite() && (xf > 0.0) == (*y > 0.0) {
                    continue;
                }
                let r = (xf - y).abs() / (xf.abs() + a);
                if r > out.0 || r.is_nan() {
                    *out = (if r.is_nan() { f64::INFINITY } else { r }, format!("{path}[{i}]: {xf} vs {y}"));
                }
            }
        }
        (IValue::Idx { t, .. }, Value::Idx { t: g, .. }) => {
            let gd = g.data.to_f64();
            for (i, (x, y)) in t.data.iter().zip(&gd).enumerate() {
                if *x as f64 != *y {
                    *out = (f64::INFINITY, format!("{path}[{i}]: index {x} vs {y}"));
                }
            }
        }
        (IValue::Tuple(xs), Value::Tuple(ys)) => xs.iter().zip(ys).enumerate().for_each(|(i, (x, y))| worst(x, y, a, &format!("{path}.{i}"), out)),
        (IValue::Record(xs), Value::Record(ys)) => xs.iter().zip(ys).for_each(|((n, x), (_, y))| worst(x, y, a, &format!("{path}.{n}"), out)),
        (IValue::Nat(x), Value::Nat(y)) => assert_eq!(x, y),
        (IValue::F64(x), Value::F64(y)) => assert_eq!(x, y),
        (x, y) => panic!("{path}: {} vs {}", x.kind(), y.kind()),
    }
}

fn check_entry(rel: &str, consts: &[(&str, u64)], entry: &str, dims: &[(&str, u64)], realize: Realize) -> f64 {
    let (prog, core) = prepare(rel, consts);
    let opts = RunOptions { entry: entry.into(), dims: dims.iter().map(|(n, v)| (n.to_string(), *v)).collect(), seed: 0, bits: 256, digits: 17 };
    let reference = presk_interp::run(&core, &prog, &opts).unwrap_or_else(|e| panic!("{entry}: {}", e.message));
    let mut backend = native_backend(&core, &prog, realize, 0).unwrap();
    let out = presk_rt::run_with(&mut backend, &core, &prog, &opts, realize).unwrap_or_else(|e| panic!("{entry} on the kernel backend: {}", e.message));
    let got = Value::from_interp(&out.result).unwrap();
    let (eps, a) = tol(realize);
    let mut w = (0.0, String::new());
    worst(&reference.result, &got, a, entry, &mut w);
    assert!(w.0 <= eps, "{rel} {entry} ({realize:?}): gap {:.3e} > {eps:.3e} at {}", w.0, w.1);
    w.0
}

const TINY: &[(&str, u64)] = &[("V", 64), ("D", 16), ("H", 2), ("L", 2)];

#[test]
fn examples_conform() {
    type Case = (&'static str, &'static [(&'static str, u64)], &'static str, &'static [(&'static str, u64)]);
    let cases: &[Case] = &[
        ("tests/examples/mlp/mlp.psk", &[], "mse", &[("D", 4), ("B", 2)]),
        ("tests/examples/mlp/mlp.psk", &[], "mse_grad", &[("D", 4), ("B", 2)]),
        ("tests/examples/layernorm/layernorm.psk", &[], "LayerNorm{8}.apply", &[("batch", 3)]),
        ("tests/examples/layernorm/layernorm.psk", &[], "RMSNorm{8}.apply", &[("batch", 3)]),
        ("tests/examples/minigpt/minigpt.psk", TINY, "MiniGPT.apply", &[("B", 2), ("T", 8)]),
        ("tests/examples/minigpt/train_minigpt.psk", TINY, "loss", &[("B", 2), ("T", 8)]),
        ("tests/examples/minigpt/train_minigpt.psk", TINY, "adamw.step", &[("B", 2), ("T", 8)]),
        ("tests/examples/minigpt/serve_minigpt.psk", TINY, "next_token", &[("B", 2), ("T", 8)]),
        ("tests/data/stack_grad.psk", &[], "loss_grad", &[("B", 2)]),
        ("tests/data/stack_prefix_grad.psk", &[], "loss_grad", &[("B", 3)]),
        ("tests/train/mlp_regression.psk", &[], "loss", &[("B", 4)]),
    ];
    let mut report = String::new();
    for (rel, consts, entry, dims) in cases {
        let g64 = check_entry(rel, consts, entry, dims, Realize::F64);
        let g32 = check_entry(rel, consts, entry, dims, Realize::F32);
        report.push_str(&format!("{entry:22} f64 {g64:.1e}  f32 {g32:.1e}\n"));
    }
    eprintln!("{report}");
}

/// Every tensor primitive on random operands, in both realizations.
#[test]
fn primitives_conform() {
    use presk_core::PrimOp;
    let (prog, core) = prepare("tests/examples/mlp/mlp.psk", &[]);
    let mut interp: Interp<Float> = Interp::new(&core, &prog, 256, 0);
    let ax = |name: Option<&str>, len: usize| presk_interp::value::Axis { name: name.map(String::from), len };
    let real = |axes: Vec<presk_interp::value::Axis>, seed: u64| -> IValue {
        let n: usize = axes.iter().map(|a| a.len).product();
        let mut s = seed;
        let data = (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                Float::with_val(256, ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0)
            })
            .collect();
        IValue::Real(presk_interp::Tensor { axes, data })
    };
    let idx = |axes: Vec<presk_interp::value::Axis>, bound: u64, seed: u64| -> IValue {
        let n: usize = axes.iter().map(|a| a.len).product();
        IValue::Idx { bound, t: presk_interp::Tensor { axes, data: (0..n as u64).map(|i| ((i * 7 + seed) % bound) as i64).collect() } }
    };
    let x = real(vec![ax(Some("batch"), 2), ax(Some("model"), 3)], 1);
    let y = real(vec![ax(Some("batch"), 2), ax(Some("model"), 3)], 2);
    let row = real(vec![ax(Some("model"), 3)], 3);
    let scalar = real(vec![], 4);
    let w = real(vec![ax(None, 3), ax(None, 4)], 5);
    let q = real(vec![ax(Some("batch"), 2), ax(Some("heads"), 2), ax(Some("seq"), 3), ax(Some("hd"), 2)], 6);
    let k = real(vec![ax(Some("batch"), 2), ax(Some("heads"), 2), ax(Some("seq"), 3), ax(Some("hd"), 2)], 7);
    let table = real(vec![ax(None, 5), ax(None, 3)], 8);
    let ids = idx(vec![ax(Some("batch"), 2), ax(Some("seq"), 4)], 5, 1);
    let logits = real(vec![ax(Some("batch"), 2), ax(Some("vocab"), 5)], 9);
    let targets = idx(vec![ax(Some("batch"), 2)], 5, 2);
    let ct2 = real(vec![ax(Some("batch"), 2)], 10);
    let ct3 = real(vec![ax(Some("batch"), 2), ax(Some("seq"), 4), ax(None, 3)], 11);
    let sm = real(vec![ax(Some("batch"), 2), ax(Some("seq"), 3), ax(Some("model"), 4)], 12);
    let cases: Vec<(PrimOp, Vec<IValue>)> = vec![
        (PrimOp::Add, vec![x.clone(), y.clone()]),
        (PrimOp::Sub, vec![x.clone(), row.clone()]),
        (PrimOp::Mul, vec![x.clone(), scalar.clone()]),
        (PrimOp::Div, vec![x.clone(), ct2.clone()]),
        (PrimOp::Neg, vec![x.clone()]),
        (PrimOp::Pow(3), vec![x.clone()]),
        (PrimOp::Exp, vec![x.clone()]),
        (PrimOp::Tanh, vec![x.clone()]),
        (PrimOp::Sin, vec![x.clone()]),
        (PrimOp::Cos, vec![x.clone()]),
        (PrimOp::Abs, vec![x.clone()]),
        (PrimOp::Gelu, vec![x.clone()]),
        (PrimOp::GeluDeriv, vec![x.clone()]),
        (PrimOp::Relu, vec![x.clone()]),
        (PrimOp::Heaviside, vec![x.clone()]),
        (PrimOp::Sign, vec![x.clone()]),
        (PrimOp::Sigmoid, vec![x.clone()]),
        (PrimOp::Dot, vec![x.clone(), w.clone()]),
        (PrimOp::DotVjpL, vec![real(vec![ax(Some("batch"), 2), ax(None, 4)], 13), w.clone()]),
        (PrimOp::DotVjpR, vec![x.clone(), real(vec![ax(Some("batch"), 2), ax(None, 4)], 14)]),
        (PrimOp::Ein { operands: vec![vec!["batch".into(), "heads".into(), "seq".into(), "hd".into()], vec!["batch".into(), "heads".into(), "seq2".into(), "hd".into()]], output: vec!["batch".into(), "heads".into(), "seq".into(), "seq2".into()] }, vec![q.clone(), k.clone()]),
        (PrimOp::SumAll, vec![x.clone()]),
        (PrimOp::MeanAll, vec![x.clone()]),
        (PrimOp::Sum { axis: "model".into() }, vec![x.clone()]),
        (PrimOp::Mean { axis: "model".into() }, vec![x.clone()]),
        (PrimOp::MeanVjp { axis: Some("model".into()) }, vec![ct2.clone(), x.clone()]),
        (PrimOp::MeanVjp { axis: None }, vec![scalar.clone(), x.clone()]),
        (PrimOp::Softmax { axis: "model".into() }, vec![x.clone()]),
        (PrimOp::Embed, vec![table.clone(), ids.clone()]),
        (PrimOp::EmbedVjp, vec![table.clone(), ids.clone(), ct3.clone()]),
        (PrimOp::CrossEntropy, vec![logits.clone(), targets.clone()]),
        (PrimOp::CrossEntropyVjp, vec![logits.clone(), targets.clone(), ct2.clone()]),
        (PrimOp::Argmax, vec![logits.clone()]),
        (PrimOp::IndexConst { axis: "seq".into(), index: -1 }, vec![sm.clone()]),
        (PrimOp::ScatterIndex { axis: "seq".into(), index: 1 }, vec![sm.clone(), real(vec![ax(Some("batch"), 2), ax(Some("model"), 4)], 15)]),
        (PrimOp::SplitHeads { h: 2, hd: 2 }, vec![sm.clone()]),
        (PrimOp::MergeHeads { h: 2, hd: 2 }, vec![q.clone()]),
        (PrimOp::SumTo, vec![x.clone(), row.clone()]),
        (PrimOp::BroadcastLike, vec![row.clone(), x.clone()]),
        (PrimOp::SumSquares, vec![x.clone()]),
        (PrimOp::Real, vec![ids.clone()]),
    ];
    for realize in [Realize::F64, Realize::F32] {
        let (eps, a) = tol(realize);
        let mut backend = native_backend(&core, &prog, realize, 0).unwrap();
        for (op, args) in &cases {
            let reference = interp.prim_public(op, args).unwrap_or_else(|e| panic!("{}: {}", op.name(), e.message));
            let rargs: Vec<Value> = args.iter().map(|v| Value::from_interp(v).unwrap().realize(realize)).collect();
            let got = backend.prim(op, &rargs, &presk_types::ty::Ty::Nat, &Default::default()).unwrap_or_else(|e| panic!("{} on the kernel backend: {}", op.name(), e.message));
            let mut w = (0.0, String::new());
            worst(&reference, &got, a, &op.name(), &mut w);
            assert!(w.0 <= eps, "{} ({realize:?}): gap {:.3e} > {eps:.3e} at {}", op.name(), w.0, w.1);
        }
    }
}

/// The textual form of the kernels of a small entry.
#[test]
fn kernel_dump_snapshot() {
    let (prog, core) = prepare("tutorial/01_linear.psk", &[]);
    let opts = RunOptions { entry: "Linear{3, 2}.apply".into(), dims: vec![("batch".into(), 4)], seed: 0, bits: 64, digits: 12 };
    let mut backend = native_backend(&core, &prog, Realize::F32, 0).unwrap();
    presk_rt::run_with(&mut backend, &core, &prog, &opts, Realize::F32).unwrap();
    let text: String = backend.kernels.iter().map(|k| k.to_string()).collect();
    assert!(text.contains("kernel dot"));
}

/// Training natively agrees with the reference: the regression's 200 steps
/// on the interpreter at 256 bits against the native backend in f64 and f32
/// (same seed, hence the same θ₀ and batches).
#[test]
fn native_training_matches_reference() {
    use presk_train::{TrainOptions, prepare_training, train};
    chdir_root();
    let mut sm = SourceMap::new();
    let id = sm.load(Path::new("tests/train/mlp_regression.psk")).unwrap();
    let (program, _) = parse_file(&sm, id);
    let (prog, _) = elab::elaborate(&mut sm, id, program, &Options::default());
    let (typed, _) = check::check(&sm, &prog);
    let (mut core, _) = presk_core::lower(&prog, &typed);
    presk_core::ad::transform(&mut core);
    let fns = prepare_training(&mut core, &prog, &typed).unwrap();
    let opts = TrainOptions { data: None, steps: Some(200), epochs: None, seed: 0, bits: 256, realize: Realize::F64, ckpt: None, checkpoints: false };
    let reference = train(&mut presk_rt::InterpBackend::new(&core, &prog, 256, 0), &core, &prog, &typed, &fns, &opts, |_| {}).unwrap();
    for (realize, bound) in [(Realize::F64, 1e-9), (Realize::F32, 1e-3)] {
        let o = TrainOptions { realize, ..opts.clone() };
        let native = train(&mut native_backend(&core, &prog, realize, 0).unwrap(), &core, &prog, &typed, &fns, &o, |_| {}).unwrap();
        assert_eq!(native.batches, reference.batches, "same batches from the same seed");
        let (mut w, _) = (0.0f64, ());
        let mut leaves_ref = Vec::new();
        let mut leaves_nat = Vec::new();
        reference.theta.map_leaves(&mut |l| {
            leaves_ref.push(l.clone());
            l.clone()
        });
        native.theta.map_leaves(&mut |l| {
            leaves_nat.push(l.clone());
            l.clone()
        });
        for (r, n) in leaves_ref.iter().zip(&leaves_nat) {
            if let (Value::Real(rt), Value::Real(nt)) = (r, n) {
                for (a, b) in rt.data.to_f64().iter().zip(nt.data.to_f64()) {
                    w = w.max((a - b).abs() / (a.abs() + 1e-12));
                }
            }
        }
        assert!(w < bound, "{realize:?}: relative gap {w:.3e} after 200 steps");
        assert!((native.logs.last().unwrap().loss - reference.logs.last().unwrap().loss).abs() < bound);
        eprintln!("native {realize:?} vs reference after 200 steps: worst relative gap {w:.3e}");
    }
}

/// Splitting a kernel's outermost axis across threads (0.4.0 step 5) changes
/// nothing but the time: each output element is computed by the same inner
/// loops in the same order, so one worker and twelve must agree bit for bit.
#[test]
fn threads_do_not_change_a_single_bit() {
    let (prog, core) = prepare("corpus/nlp/05_gpt.psk", &[("V", 92), ("D", 64), ("H", 2), ("L", 2)]);
    for realize in [Realize::F64, Realize::F32] {
        let opts = presk_interp::RunOptions { entry: "GPT.apply".into(), dims: vec![("B".into(), 4), ("T".into(), 24)], seed: 0, bits: 64, digits: 12 };
        let (func, dims, inputs) = presk_interp::run::prepare_run(&core, &prog, &opts).unwrap();
        let args: Vec<Value> = inputs.iter().map(|(_, v)| Value::from_interp(v).unwrap()).collect();
        let run = |threads: usize| -> Vec<f64> {
            let mut b = presk_cranelift::native_backend_with_threads(&core, &prog, realize, 0, threads).unwrap();
            let p = presk_rt::Backend::prepare(&mut b, func, &dims, realize).unwrap();
            let out = presk_rt::Backend::call(&mut b, &p, args.clone()).unwrap();
            let Value::Real(t) = out else { panic!("the model returns logits") };
            t.data.to_f64()
        };
        let one = run(1);
        assert!(one.len() > 1000, "a real tensor, not a scalar: {} elements", one.len());
        for threads in [2, 5, 12] {
            let many = run(threads);
            assert_eq!(one.len(), many.len());
            for (i, (a, b)) in one.iter().zip(&many).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "{realize:?}, {threads} threads, element {i}: {a} against {b}");
            }
        }
    }
}

/// The blocked contraction (0.4.0 step 6) holds a tile of the result in
/// registers across the whole sum. Each accumulator still adds `k` in
/// increasing order, so the result is bit-identical to the form used when the
/// width is not a multiple of the tile — which is what this compares.
#[test]
fn the_blocked_contraction_is_exact() {
    chdir_root();
    for realize in [Realize::F64, Realize::F32] {
        // 16 is a multiple of the tile and 12 is not, so the two widths take
        // the two paths; the shared prefix of the results must agree exactly.
        let wide = gemm_row(16, realize);
        let narrow = gemm_row(12, realize);
        assert_eq!(wide.len(), 16);
        assert_eq!(narrow.len(), 12);
        for (i, (a, b)) in wide.iter().zip(&narrow).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "{realize:?} column {i}: blocked {a} against row-wise {b}");
        }
    }
}

/// One row of `a · b` where `b` has `n` columns, `a` and the shared part of
/// `b` being the same values whatever `n`.
fn gemm_row(n: usize, realize: Realize) -> Vec<f64> {
    use presk_rt::{Axis, Buf, Tensor};
    let k = 37;
    let a: Vec<f64> = (0..k).map(|i| ((i * 7 + 1) as f64).sin()).collect();
    let b: Vec<f64> = (0..k).flat_map(|i| (0..n).map(move |j| ((i * 31 + j * 3) as f64).cos())).collect();
    let (prog, core) = prepare("tests/data/gemm_exact.psk", &[("K", k as u64), ("N", n as u64)]);
    let f = core.find("gemm").unwrap();
    let dims: presk_interp::eval::Dims = core.func(f).dims.iter().map(|d| (*d, 1)).collect();
    let mut back = presk_cranelift::native_backend(&core, &prog, realize, 0).unwrap();
    let mk = |axes: Vec<Axis>, data: Vec<f64>| Value::Real(Tensor { axes, data: if realize == Realize::F32 { Buf::F32(data.iter().map(|x| *x as f32).collect()) } else { Buf::F64(data) } });
    let args = vec![
        mk(vec![Axis { name: Some("row".into()), len: 1 }, Axis { name: None, len: k }], a),
        mk(vec![Axis { name: None, len: k }, Axis { name: None, len: n }], b),
    ];
    let p = presk_rt::Backend::prepare(&mut back, f, &dims, realize).unwrap();
    let out = presk_rt::Backend::call(&mut back, &p, args).unwrap();
    let Value::Real(t) = out else { panic!("a real tensor") };
    t.data.to_f64()
}
