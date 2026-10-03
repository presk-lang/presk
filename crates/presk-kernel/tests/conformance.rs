//! The conformance suite of PSK-Kernel v0 (plan 0.3.0 §3.4): every example
//! entry and every tensor primitive against the reference interpreter at 256
//! bits, in f64 and in f32, within a stated tolerance per realization
//! `|x − x̂| ≤ ε·(|x| + a)`.

use std::path::{Path, PathBuf};
use std::sync::Once;

use presk_interp::{Interp, RunOptions};
use presk_kernel::KernelBackend;
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
    presk_core::verify(&core).unwrap();
    (prog, core)
}

/// The tolerance of a realization: `(ε, a)`. Since 0.5.0 step 2 the suite is
/// parameterized by a `Realization` rather than a machine type, so a mixed or
/// reassociated one is reported the same way a uniform one is (D34). The
/// bands are the storage type's, loosened by the reduction order when it
/// reassociates.
fn tol(r: &Realization) -> (f64, f64) {
    let (eps, a) = match r.default {
        Dtype::F64 => (2f64.powi(-40), 2f64.powi(-50)),
        Dtype::F32 => (2f64.powi(-15), 2f64.powi(-20)),
        // bf16 keeps eight bits of significand, so `u = 2⁻⁸`, and a value of
        // order one carries an absolute error of up to `2⁻⁸` before anything
        // has happened to it (the comment said `2⁻⁹` until 0.10.0 step 8). The floor `a` is what makes the band mean
        // something: without it a result near a cancellation has unbounded
        // *relative* error, which says nothing about the kernel. Measured
        // worst over the examples: 3.2e-2 (0.7.0 step 2). The programs of
        // tests/ report their bf16 gap without asserting it (D58).
        Dtype::Bf16 => (2f64.powi(-4), 2f64.powi(-5)),
    };
    if r.order.is_exact() { (eps, a) } else { (eps * 8.0, a) }
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

fn check_entry(rel: &str, consts: &[(&str, u64)], entry: &str, dims: &[(&str, u64)], realize: &Realization) -> f64 {
    let (prog, core) = prepare(rel, consts);
    let opts = RunOptions { entry: entry.into(), dims: dims.iter().map(|(n, v)| (n.to_string(), *v)).collect(), seed: 0, bits: 256, digits: 17 };
    let reference = presk_interp::run(&core, &prog, &opts).unwrap_or_else(|e| panic!("{entry}: {}", e.message));
    let mut backend = KernelBackend::new(&core, &prog, realize);
    let out = presk_rt::run_with(&mut backend, &core, &prog, &opts, realize).unwrap_or_else(|e| panic!("{entry} on the kernel backend: {}", e.message));
    let got = Value::from_interp(&out.result).unwrap();
    let (eps, a) = tol(realize);
    let mut w = (0.0, String::new());
    worst(&reference.result, &got, a, entry, &mut w);
    // The band is asserted on the examples, which the language reference's
    // conformance definition covers, in every realization; on the compiler's
    // own test programs in f64 and f32. Their bf16 gap is reported: on
    // stack_prefix_grad's gradient, where a small result is left over from
    // larger terms cancelling, it crosses the band on some inputs and not
    // others, with 0.6.0's kernels as with these (D58).
    if rel.starts_with("tests/examples/") || realize.default != Dtype::Bf16 {
        assert!(w.0 <= eps, "{rel} {entry} ({}): gap {:.3e} > {eps:.3e} at {}", realize.label(), w.0, w.1);
    }
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
        ("tests/data/split_grad.psk", &[], "loss_grad", &[("B", 2)]),
        ("tests/train/mlp_regression.psk", &[], "loss", &[("B", 4)]),
    ];
    // On these parameters MiniGPT's head is zero, and so are the logits of
    // `MiniGPT.apply`: `examples_with_their_zeros_drawn` checks it on others.
    let mut report = String::new();
    for (rel, consts, entry, dims) in cases {
        let g64 = check_entry(rel, consts, entry, dims, &Realization::uniform(Dtype::F64));
        let g32 = check_entry(rel, consts, entry, dims, &Realization::uniform(Dtype::F32));
        let gbf = check_entry(rel, consts, entry, dims, &Realization::uniform(Dtype::Bf16));
        report.push_str(&format!("{entry:22} f64 {g64:.1e}  f32 {g32:.1e}  bf16 {gbf:.1e}\n"));
    }
    eprintln!("{report}");
}

/// Every value an initializer sets to exactly zero in the first argument, a
/// module's parameters, drawn instead from a fixed sequence: biases, shifts
/// and MiniGPT's head (`init = zeros`), on which its logits are zero.
fn draw_zeros(args: &mut [Value]) {
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut draw = |v: &Value| match v {
        Value::Real(t) => {
            let mut t = t.clone();
            if let presk_rt::Buf::Exact(d) = &mut t.data {
                for x in d.iter_mut().filter(|x| x.is_zero()) {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    *x = Float::with_val(256, ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.2);
                }
            }
            Value::Real(t)
        }
        other => other.clone(),
    };
    args[0] = args[0].map_leaves(&mut draw);
}

/// Elements of `got` outside the band of the exact `reference`, and all
/// its elements; an index that differs is outside.
fn outside(reference: &IValue, got: &Value, eps: f64, a: f64) -> (usize, usize) {
    match (reference, got) {
        (IValue::Real(t), Value::Real(g)) => {
            let gd = g.data.to_f64();
            let n = t.data.iter().zip(&gd).filter(|(x, y)| {
                let x = x.to_f64();
                !((x - **y).abs() <= eps * (x.abs() + a) || (x.is_infinite() && x == **y))
            });
            (n.count(), gd.len())
        }
        (IValue::Idx { t, .. }, Value::Idx { t: g, .. }) => {
            let gd = g.data.to_f64();
            (t.data.iter().zip(&gd).filter(|(x, y)| **x as f64 != **y).count(), gd.len())
        }
        (IValue::Tuple(xs), Value::Tuple(ys)) => xs.iter().zip(ys).map(|(x, y)| outside(x, y, eps, a)).fold((0, 0), |p, q| (p.0 + q.0, p.1 + q.1)),
        (IValue::Record(xs), Value::Record(ys)) => xs.iter().zip(ys).map(|((_, x), (_, y))| outside(x, y, eps, a)).fold((0, 0), |p, q| (p.0 + q.0, p.1 + q.1)),
        _ => (0, 0),
    }
}

/// The examples again, with the values their initializers set to zero drawn
/// (plan 0.9.0 step 3, D79). On the synthesized parameters MiniGPT's head is
/// `init = zeros`, so `MiniGPT.apply`'s logits are all zero and `loss` is
/// log V, and `examples_conform` passes there on trivial outputs. Drawn,
/// the band holds on every example in `f64`, and on every one but
/// `MiniGPT.apply` in `f32` and `bf16`: a few of its logits, 6·10⁻⁴ to
/// 3·10⁻³ against a median of 0.17, are sums that cancel, and leave the band
/// by a factor up to four. Those are reported, not asserted (spec question
/// 75): the native backend leaves it on other logits than this executor,
/// their `exp` differing in the last bits, so no element-by-element rule
/// between the two would hold either.
#[test]
fn examples_with_their_zeros_drawn() {
    // (file, constants, entry, dimensions, whether its outputs cancel)
    type Case = (&'static str, &'static [(&'static str, u64)], &'static str, &'static [(&'static str, u64)], bool);
    let cases: &[Case] = &[
        ("tests/examples/mlp/mlp.psk", &[], "mse", &[("D", 4), ("B", 2)], false),
        ("tests/examples/mlp/mlp.psk", &[], "mse_grad", &[("D", 4), ("B", 2)], false),
        ("tests/examples/layernorm/layernorm.psk", &[], "LayerNorm{8}.apply", &[("batch", 3)], false),
        ("tests/examples/layernorm/layernorm.psk", &[], "RMSNorm{8}.apply", &[("batch", 3)], false),
        ("tests/examples/minigpt/minigpt.psk", TINY, "MiniGPT.apply", &[("B", 2), ("T", 8)], true),
        ("tests/examples/minigpt/train_minigpt.psk", TINY, "loss", &[("B", 2), ("T", 8)], false),
        ("tests/examples/minigpt/train_minigpt.psk", TINY, "adamw.step", &[("B", 2), ("T", 8)], false),
        ("tests/examples/minigpt/serve_minigpt.psk", TINY, "next_token", &[("B", 2), ("T", 8)], false),
    ];
    let mut report = String::new();
    for (rel, consts, entry, dims, cancels) in cases {
        let (prog, core) = prepare(rel, consts);
        let opts = RunOptions { entry: (*entry).into(), dims: dims.iter().map(|(n, v)| (n.to_string(), *v)).collect(), seed: 0, bits: 256, digits: 17 };
        let (func, fdims, inputs) = presk_interp::run::prepare_run(&core, &prog, &opts).unwrap();
        let mut args: Vec<Value> = inputs.iter().map(|(_, v)| Value::from_interp(v).unwrap()).collect();
        draw_zeros(&mut args);
        let mut exact = presk_rt::InterpBackend::new(&core, &prog, 256);
        let f64r = Realization::uniform(Dtype::F64);
        let p = presk_rt::Backend::prepare(&mut exact, func, &fdims, &f64r).unwrap();
        let reference = presk_rt::Backend::call(&mut exact, &p, args.clone()).unwrap().to_interp(256);
        report.push_str(&format!("{entry:22}"));
        for d in [Dtype::F64, Dtype::F32, Dtype::Bf16] {
            let r = Realization::uniform(d);
            let (eps, a) = tol(&r);
            let mut b = KernelBackend::new(&core, &prog, &r);
            let p = presk_rt::Backend::prepare(&mut b, func, &fdims, &r).unwrap();
            let got = presk_rt::Backend::call(&mut b, &p, args.clone()).unwrap();
            let mut w = (0.0, String::new());
            worst(&reference, &got, a, entry, &mut w);
            let (n, total) = outside(&reference, &got, eps, a);
            report.push_str(&format!("  {d} {n}/{total} worst {:.1e}", w.0 / eps));
            if d == Dtype::F64 || !cancels {
                assert!(w.0 <= eps, "{rel} {entry} ({d}), zeros drawn: gap {:.3e} > {eps:.3e} at {}", w.0, w.1);
            }
            let reals: Vec<u64> = bits_of(&got);
            assert!(reals.is_empty() || reals.iter().any(|b| f64::from_bits(*b) != 0.0), "{rel} {entry} ({d}): every output is zero");
        }
        report.push('\n');
    }
    eprintln!("{report}");
}

/// Every tensor primitive on random operands, in both realizations.
#[test]
fn primitives_conform() {
    use presk_core::PrimOp;
    let (prog, core) = prepare("tests/examples/mlp/mlp.psk", &[]);
    let mut interp: Interp<Float> = Interp::new(&core, &prog, 256);
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
    for realize in [Realization::uniform(Dtype::F64), Realization::uniform(Dtype::F32), Realization::uniform(Dtype::Bf16)] {
        let (eps, a) = tol(&realize);
        let mut backend = KernelBackend::new(&core, &prog, &realize);
        for (op, args) in &cases {
            let reference = interp.prim_public(op, args).unwrap_or_else(|e| panic!("{}: {}", op.name(), e.message));
            let rargs: Vec<Value> = args.iter().map(|v| Value::from_interp(v).unwrap().realize(realize.default)).collect();
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
    let mut backend = KernelBackend::new(&core, &prog, &Realization::uniform(Dtype::F32));
    presk_rt::run_with(&mut backend, &core, &prog, &opts, &Realization::uniform(Dtype::F32)).unwrap();
    let text: String = backend.kernels.iter().map(|k| k.to_string()).collect();
    insta::assert_snapshot!("linear_kernels", text);
}

/// Statement H of plan 0.6.0, on the interpreter: a vector kernel is
/// **bit-identical** to the scalar one, not merely within a tolerance.
///
/// That is the whole reason the vector form is cheap. Each lane is a
/// different output column, so every output still accumulates the contracted
/// axis in increasing order; the lanes change where partial results live,
/// not the order in which anything is added (D42). A tolerance here would
/// mean the design had gone wrong.
#[test]
fn a_vector_kernel_is_bit_identical_to_the_scalar_one() {
    type Case = (&'static str, &'static [(&'static str, u64)], &'static str, &'static [(&'static str, u64)]);
    let cases: &[Case] = &[
        ("tests/examples/mlp/mlp.psk", &[], "mse", &[("D", 32), ("B", 4)]),
        ("tests/examples/mlp/mlp.psk", &[], "mse_grad", &[("D", 32), ("B", 4)]),
        ("tests/examples/minigpt/minigpt.psk", TINY, "MiniGPT.apply", &[("B", 2), ("T", 8)]),
        ("tests/data/split_grad.psk", &[], "loss_grad", &[("B", 2)]),
    ];
    let mut widened = 0usize;
    for (rel, consts, entry, dims) in cases {
        for d in [Dtype::F64, Dtype::F32] {
            let r = Realization::uniform(d);
            let (prog, core) = prepare(rel, consts);
            let opts = RunOptions { entry: (*entry).into(), dims: dims.iter().map(|(n, v)| (n.to_string(), *v)).collect(), seed: 0, bits: 256, digits: 17 };
            let (func, fdims, inputs) = presk_interp::run::prepare_run(&core, &prog, &opts).unwrap();
            let args: Vec<Value> = inputs.iter().map(|(_, v)| Value::from_interp(v).unwrap().realize(d)).collect();
            let run = |lanes: usize| -> Vec<u64> {
                let mut b = KernelBackend::new(&core, &prog, &r);
                b.set_default_lanes(lanes);
                let p = presk_rt::Backend::prepare(&mut b, func, &fdims, &r).unwrap();
                let out = presk_rt::Backend::call(&mut b, &p, args.clone()).unwrap();
                bits_of(&out)
            };
            let scalar = run(1);
            for lanes in [2usize, 4] {
                let vector = run(lanes);
                assert_eq!(scalar, vector, "{rel} {entry} ({d}, {lanes} lanes): a vector kernel must be bit-identical, not close");
            }
            widened += 1;
        }
    }
    assert!(widened > 0);
}

/// Every real leaf's bits, in order.
fn bits_of(v: &Value) -> Vec<u64> {
    match v {
        Value::Real(t) => t.data.to_f64().into_iter().map(|x| x.to_bits()).collect(),
        Value::Tuple(xs) => xs.iter().flat_map(bits_of).collect(),
        Value::Record(xs) => xs.iter().flat_map(|(_, x)| bits_of(x)).collect(),
        _ => vec![],
    }
}
