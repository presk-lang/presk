//! Golden outputs of `presk run` (snapshots under `tests/run/` at the repo
//! root) and the differential test of automatic differentiation against
//! central finite differences at 512 bits.

use std::path::{Path, PathBuf};
use std::sync::Once;

use presk_interp::{Interp, RunOptions, Value};
use presk_syntax::diag::render_all;
use presk_syntax::{SourceMap, parse_file};
use presk_types::check;
use presk_types::elab::{self, ElabProgram, Options};
use rug::Float;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn chdir_root() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| std::env::set_current_dir(repo_root()).unwrap());
}

/// The whole pipeline up to a runnable program.
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

fn run(rel: &str, consts: &[(&str, u64)], entry: &str, dims: &[(&str, u64)], bits: u32) -> presk_interp::RunOutput {
    let (prog, core) = prepare(rel, consts);
    let opts = RunOptions { entry: entry.into(), dims: dims.iter().map(|(n, v)| (n.to_string(), *v)).collect(), seed: 0, bits, digits: 12 };
    presk_interp::run(&core, &prog, &opts).unwrap_or_else(|e| panic!("{rel} --entry {entry}: {}", e.message))
}

fn snap(name: &str, out: &presk_interp::RunOutput) {
    let text = format!("{} = {}\n", presk_interp::run::describe(&out.result), presk_interp::run::render(&out.result, 12, 0));
    let mut settings = insta::Settings::clone_current();
    settings.set_snapshot_path(repo_root().join("tests/run"));
    settings.set_prepend_module_to_snapshot(false);
    settings.bind(|| insta::assert_snapshot!(name, text));
}

#[test]
fn golden_mlp() {
    snap("mlp_mse", &run("tests/examples/mlp/mlp.psk", &[], "mse", &[("D", 4), ("B", 2)], 256));
    snap("mlp_mse_grad", &run("tests/examples/mlp/mlp.psk", &[], "mse_grad", &[("D", 4), ("B", 2)], 256));
}

#[test]
fn golden_layernorm() {
    snap("layernorm_apply", &run("tests/examples/layernorm/layernorm.psk", &[], "LayerNorm{8}.apply", &[("batch", 3)], 256));
    snap("rmsnorm_apply", &run("tests/examples/layernorm/layernorm.psk", &[], "RMSNorm{8}.apply", &[("batch", 3)], 256));
}

const TINY: &[(&str, u64)] = &[("V", 64), ("D", 16), ("H", 2), ("L", 2)];

#[test]
fn golden_minigpt() {
    snap("minigpt_forward", &run("tests/examples/minigpt/minigpt.psk", TINY, "MiniGPT.apply", &[("B", 2), ("T", 8)], 256));
    snap("minigpt_loss", &run("tests/examples/minigpt/train_minigpt.psk", TINY, "loss", &[("B", 2), ("T", 8)], 256));
    snap("minigpt_next_token", &run("tests/examples/minigpt/serve_minigpt.psk", TINY, "next_token", &[("B", 2), ("T", 8)], 256));
}

#[test]
fn softmax_rows_sum_to_one() {
    // A tiny model with a non-zero head so that logits are non-trivial: build the
    // attention weights through `loss`, which exercises softmax; check the
    // invariant on a direct softmax instead: Σ_axis softmax(x) = 1 at 2^-200.
    let (prog, core) = prepare("tests/examples/minigpt/minigpt.psk", TINY);
    let mut interp: Interp<Float> = Interp::new(&core, &prog, 256);
    let x = Value::Real(presk_interp::Tensor { axes: vec![presk_interp::value::Axis { name: Some("seq2".into()), len: 5 }], data: (0..5).map(|i| Float::with_val(256, i as f64 * 0.7 - 1.0)).collect() });
    let sm = interp.prim_public(&presk_core::PrimOp::Softmax { axis: "seq2".into() }, &[x]).unwrap();
    let Value::Real(t) = sm else { panic!() };
    let mut s = Float::with_val(256, 0);
    for v in &t.data {
        s += v;
    }
    let diff = (s - 1u32).abs();
    assert!(diff < Float::with_val(256, 2f64.powi(-200)), "{diff}");
}

/// The gradient of `mse` computed by AD agrees with central finite
/// differences at 512 bits, h = 2^-120, to 2^-100 relative.
#[test]
fn ad_matches_finite_differences() {
    let bits = 512;
    let rel = "tests/examples/mlp/mlp.psk";
    let out = run(rel, &[], "mse_grad", &[("D", 2), ("B", 2)], bits);
    let (prog, core) = prepare(rel, &[]);
    let mse = core.find("mse").unwrap();
    let dims: std::collections::BTreeMap<_, _> = core.func(mse).dims.iter().map(|d| (*d, match prog.dim_name(*d).as_str() { "D" => 2, "B" => 2, _ => 1 })).collect();
    let theta = out.inputs[0].1.clone();
    let x = out.inputs[1].1.clone();
    let y = out.inputs[2].1.clone();
    let mut interp: Interp<Float> = Interp::new(&core, &prog, bits);
    let h = Float::with_val(bits, 2f64.powi(-120));
    let tol = Float::with_val(bits, 2f64.powi(-100));
    let paths = leaf_paths(&theta);
    let mut checked = 0;
    for path in &paths {
        let Value::Real(t) = get_leaf(&theta, path) else { continue };
        let Value::Real(g) = get_leaf(&out.result, path) else { panic!("gradient leaf missing") };
        for i in 0..t.data.len() {
            let plus = set_leaf_elem(&theta, path, i, Float::with_val(bits, &t.data[i] + &h));
            let minus = set_leaf_elem(&theta, path, i, Float::with_val(bits, &t.data[i] - &h));
            let fp = scalar(&interp.call(mse, dims.clone(), vec![plus, x.clone(), y.clone()]).unwrap());
            let fm = scalar(&interp.call(mse, dims.clone(), vec![minus, x.clone(), y.clone()]).unwrap());
            let fd = (fp - fm) / (Float::with_val(bits, 2) * &h);
            let err = (Float::with_val(bits, &fd - &g.data[i])).abs();
            let scale = Float::with_val(bits, g.data[i].clone().abs()).max(&Float::with_val(bits, 1));
            let bound = Float::with_val(bits, &tol * &scale);
            assert!(err <= bound, "leaf {path:?}[{i}]: ad = {}, fd = {}, err = {err}", g.data[i], fd);
            checked += 1;
        }
    }
    assert!(checked >= 20, "checked only {checked} components");
}

fn scalar(v: &Value<Float>) -> Float {
    match v {
        Value::Real(t) if t.rank() == 0 => t.data[0].clone(),
        other => panic!("not a scalar: {}", other.kind()),
    }
}

fn leaf_paths(v: &Value<Float>) -> Vec<Vec<String>> {
    match v {
        Value::Record(fs) => fs.iter().flat_map(|(n, sub)| leaf_paths(sub).into_iter().map(move |mut p| {
            p.insert(0, n.clone());
            p
        })).collect(),
        _ => vec![vec![]],
    }
}

fn get_leaf<'a>(v: &'a Value<Float>, path: &[String]) -> &'a Value<Float> {
    match path.split_first() {
        None => v,
        Some((h, rest)) => get_leaf(v.field(h).unwrap(), rest),
    }
}

fn set_leaf_elem(v: &Value<Float>, path: &[String], i: usize, x: Float) -> Value<Float> {
    match (v, path.split_first()) {
        (Value::Record(fs), Some((h, rest))) => Value::record(fs.iter().map(|(n, sub)| (n.clone(), if n == h { set_leaf_elem(sub, rest, i, x.clone()) } else { sub.clone() })).collect()),
        (Value::Real(t), None) => {
            let mut t = t.clone();
            t.data[i] = x;
            Value::Real(t)
        }
        _ => v.clone(),
    }
}

/// A closure applied inside a differentiated function (0.10.0 step 3, defect
/// 3a of the reference's chapter 5). The `.vjp` stored the body's closures
/// under the body's variables and looked them up under its own, so one
/// closure in scope was refused (`E6001`) and, of two, the other one was
/// differentiated, without a diagnostic. Each gradient agrees with central
/// finite differences of its primal at 512 bits, h = 2^-120, to 2^-100
/// relative, and the closure that is never applied changes no bit.
#[test]
fn ad_through_applied_closures_matches_finite_differences() {
    let bits = 512;
    let rel = "tests/data/closure_grad.psk";
    let (prog, core) = prepare(rel, &[]);
    let h = Float::with_val(bits, 2f64.powi(-120));
    let tol = Float::with_val(bits, 2f64.powi(-100));
    let mut interp: Interp<Float> = Interp::new(&core, &prog, bits);
    let mut grads = vec![];
    for (entry, primal) in [("d_two", "two"), ("d_one", "one"), ("d_captured", "captured")] {
        let out = run(rel, &[], entry, &[], bits);
        let f = core.find(primal).unwrap();
        let (x, y) = (out.inputs[0].1.clone(), out.inputs[1].1.clone());
        let (Value::Real(xt), Value::Real(g)) = (&x, &out.result) else { panic!("{entry}: tensors") };
        for i in 0..xt.data.len() {
            let at = |d: Float| {
                let mut t = xt.clone();
                t.data[i] = Float::with_val(bits, &t.data[i] + d);
                Value::Real(t)
            };
            let fp = scalar(&interp.call(f, Default::default(), vec![at(h.clone()), y.clone()]).unwrap());
            let fm = scalar(&interp.call(f, Default::default(), vec![at(-h.clone()), y.clone()]).unwrap());
            let fd = (fp - fm) / (Float::with_val(bits, 2) * &h);
            let err = (Float::with_val(bits, &fd - &g.data[i])).abs();
            let scale = Float::with_val(bits, g.data[i].clone().abs()).max(&Float::with_val(bits, 1));
            assert!(err <= Float::with_val(bits, &tol * &scale), "{entry}[{i}]: ad = {}, fd = {}, err = {err}", g.data[i], fd);
        }
        grads.push(g.data.clone());
    }
    assert_eq!(grads[0], grads[1], "a closure never applied changes nothing");
}

/// The optimizer functions lowered from `optimizer adamw { … }` (0.2.0 step 3):
/// the cosine schedule of `train_minigpt.psk` at chosen steps, and one AdamW
/// step from zero moments on `tests/data/train_ok.psk`, against the closed forms.
#[test]
fn optimizer_schedule_and_step() {
    use presk_interp::value::Axis;
    let (prog, core) = prepare("tests/examples/minigpt/train_minigpt.psk", &[("V", 64), ("D", 16), ("H", 2), ("L", 2)]);
    let mut interp: Interp<Float> = Interp::new(&core, &prog, 128);
    let lr = core.find("adamw.lr").unwrap();
    let mut f = |t: u64, total: u64| -> f64 {
        match interp.call(lr, Default::default(), vec![Value::Nat(t), Value::Nat(total)]).unwrap() {
            Value::Real(x) => x.data[0].to_f64(),
            other => panic!("{}", other.kind()),
        }
    };
    // cosine(warmup = 2000), lr = 3e-4: warm-up t/2000, then ½(1 + cos(π p)).
    assert!((f(1, 4000) - 1.5e-7).abs() < 1e-20);
    assert!((f(2000, 4000) - 3e-4).abs() < 1e-18);
    assert!((f(3000, 4000) - 1.5e-4).abs() < 1e-18);
    assert!(f(4000, 4000).abs() < 1e-18);
    assert!((f(5000, 4000)).abs() < 1e-18, "past the horizon the rate stays at 0");

    let (prog, core) = prepare("tests/data/train_ok.psk", &[]);
    let mut interp: Interp<Float> = Interp::new(&core, &prog, 128);
    let step = core.find("adamw.step").unwrap();
    let tensor = |x: f64| Value::Real(presk_interp::Tensor { axes: vec![Axis { name: None, len: 64 }, Axis { name: None, len: 4 }], data: vec![Float::with_val(128, x); 256].into() });
    let tree = |x: f64| Value::record(vec![("w".to_string(), tensor(x))]);
    let out = interp.call(step, Default::default(), vec![tree(1.0), Value::tuple(vec![tree(0.0), tree(0.0)]), tree(0.5), Value::Nat(1), Value::Nat(10)]).unwrap();
    let leaf = |v: &Value<Float>| -> f64 {
        match v {
            Value::Record(fs) => match &fs[0].1 {
                Value::Real(t) => t.data[7].to_f64(),
                other => panic!("{}", other.kind()),
            },
            other => panic!("{}", other.kind()),
        }
    };
    let Value::Tuple(parts) = &out else { panic!() };
    let Value::Tuple(state) = &parts[1] else { panic!() };
    // lr 1e-3, betas (0.9, 0.999), eps 1e-8, no decay: m' = 0.05, v' = 0.00025,
    // m̂ = 0.5, v̂ = 0.25, θ' = 1 − 1e-3 · 0.5 / (0.5 + 1e-8).
    assert!((leaf(&state[0]) - 0.05).abs() < 1e-15);
    assert!((leaf(&state[1]) - 0.00025).abs() < 1e-15);
    let expected = 1.0 - 1e-3 * 0.5 / (0.5 + 1e-8);
    assert!((leaf(&parts[0]) - expected).abs() < 1e-15, "{} vs {expected}", leaf(&parts[0]));
}

/// The gradient through a `Fold` (a stacked module, 0.2.0 step 4) agrees
/// with central finite differences: with respect to the stacked parameter
/// tree and to an extra argument of the loop.
#[test]
fn ad_through_fold_matches_finite_differences() {
    let bits = 512;
    let rel = "tests/data/stack_grad.psk";
    let out = run(rel, &[], "loss_grad", &[("B", 2)], bits);
    snap("stack_grad", &out);
    let (prog, core) = prepare(rel, &[]);
    let loss = core.find("loss").unwrap();
    let dims: std::collections::BTreeMap<_, _> = core.func(loss).dims.iter().map(|d| (*d, match prog.dim_name(*d).as_str() { "B" => 2, _ => 1 })).collect();
    let theta = out.inputs[0].1.clone();
    let x = out.inputs[1].1.clone();
    let scale = out.inputs[2].1.clone();
    let mut interp: Interp<Float> = Interp::new(&core, &prog, bits);
    let h = Float::with_val(bits, 2f64.powi(-120));
    let tol = Float::with_val(bits, 2f64.powi(-100));
    let mut checked = 0;
    for path in &leaf_paths(&theta) {
        let Value::Real(t) = get_leaf(&theta, path) else { continue };
        let Value::Real(g) = get_leaf(&out.result, path) else { panic!("gradient leaf missing") };
        assert_eq!(t.axes.len(), g.axes.len(), "stacked leaf {path:?} keeps its rank");
        for i in 0..t.data.len() {
            let plus = set_leaf_elem(&theta, path, i, Float::with_val(bits, &t.data[i] + &h));
            let minus = set_leaf_elem(&theta, path, i, Float::with_val(bits, &t.data[i] - &h));
            let fp = scalar(&interp.call(loss, dims.clone(), vec![plus, x.clone(), scale.clone()]).unwrap());
            let fm = scalar(&interp.call(loss, dims.clone(), vec![minus, x.clone(), scale.clone()]).unwrap());
            let fd = (fp - fm) / (Float::with_val(bits, 2) * &h);
            let err = (Float::with_val(bits, &fd - &g.data[i])).abs();
            let scale_ = Float::with_val(bits, g.data[i].clone().abs()).max(&Float::with_val(bits, 1));
            assert!(err <= Float::with_val(bits, &tol * &scale_), "leaf {path:?}[{i}]: ad = {}, fd = {}, err = {err}", g.data[i], fd);
            checked += 1;
        }
    }
    assert_eq!(checked, 3 * 9 + 3 * 3, "every element of the stacked tree");
    // The extra argument: d loss / d scale.
    let gs = run(rel, &[], "scale_grad", &[("B", 2)], bits);
    let s = scalar(&scale);
    let sp = Value::Real(presk_interp::Tensor { axes: vec![], data: vec![Float::with_val(bits, &s + &h)].into() });
    let sm = Value::Real(presk_interp::Tensor { axes: vec![], data: vec![Float::with_val(bits, &s - &h)].into() });
    let fp = scalar(&interp.call(loss, dims.clone(), vec![theta.clone(), x.clone(), sp]).unwrap());
    let fm = scalar(&interp.call(loss, dims.clone(), vec![theta.clone(), x.clone(), sm]).unwrap());
    let fd = (fp - fm) / (Float::with_val(bits, 2) * &h);
    let g = scalar(&gs.result);
    let err = (Float::with_val(bits, &fd - &g)).abs();
    assert!(err <= Float::with_val(bits, &tol * Float::with_val(bits, g.clone().abs()).max(&Float::with_val(bits, 1))), "scale: ad = {g}, fd = {fd}");
}

/// The gradient through `dropout` (0.4.0 step 3) agrees with central finite
/// differences *while dropping*. The mask is a function of the key alone, so
/// the backward pass, which recomputes the forward, recomputes the same mask;
/// a mask drawn from a stream instead would make this test fail, which is why
/// the key is explicit in the signature.
#[test]
fn ad_through_dropout_matches_finite_differences() {
    let bits = 512;
    let rel = "tests/data/dropout_grad.psk";
    let (prog, core) = prepare(rel, &[]);
    let opts = presk_interp::RunOptions { entry: "loss_grad".into(), dims: vec![("B".into(), 2)], seed: 0, bits, digits: 12 };
    let (gradf, dims, inputs) = presk_interp::run::prepare_run(&core, &prog, &opts).unwrap();
    let args: Vec<Value<Float>> = inputs.iter().map(|(_, v)| v.clone()).collect();
    let (theta, x, key) = (args[0].clone(), args[1].clone(), args[2].clone());
    // Training on: without it the mask is all ones and nothing is dropped.
    let mut interp: Interp<Float> = Interp::new(&core, &prog, bits);
    interp.training = true;
    let grad = interp.call(gradf, dims.clone(), args.clone()).unwrap();

    let loss = core.find("loss").unwrap();
    let ldims: std::collections::BTreeMap<_, _> = core.func(loss).dims.iter().map(|d| (*d, match prog.dim_name(*d).as_str() { "B" => 2, _ => 1 })).collect();
    let h = Float::with_val(bits, 2f64.powi(-120));
    let tol = Float::with_val(bits, 2f64.powi(-100));
    let mut checked = 0;
    for path in &leaf_paths(&theta) {
        let Value::Real(t) = get_leaf(&theta, path) else { continue };
        let Value::Real(g) = get_leaf(&grad, path) else { panic!("gradient leaf missing") };
        for i in 0..t.data.len() {
            let plus = set_leaf_elem(&theta, path, i, Float::with_val(bits, &t.data[i] + &h));
            let minus = set_leaf_elem(&theta, path, i, Float::with_val(bits, &t.data[i] - &h));
            let fp = scalar(&interp.call(loss, ldims.clone(), vec![plus, x.clone(), key.clone()]).unwrap());
            let fm = scalar(&interp.call(loss, ldims.clone(), vec![minus, x.clone(), key.clone()]).unwrap());
            let fd = (fp - fm) / (Float::with_val(bits, 2) * &h);
            let err = (Float::with_val(bits, &fd - &g.data[i])).abs();
            let scale = Float::with_val(bits, g.data[i].clone().abs()).max(&Float::with_val(bits, 1));
            assert!(err <= Float::with_val(bits, &tol * &scale), "leaf {path:?}[{i}]: ad = {}, fd = {}, err = {err}", g.data[i], fd);
            checked += 1;
        }
    }
    assert_eq!(checked, 4 * 6 + 6 * 4, "every element of both layers");

    // The gradient is a function of the key: the same one twice gives the same
    // answer, another one a different mask and so a different answer.
    let sum = |v: &Value<Float>| -> f64 {
        leaf_paths(v).iter().filter_map(|p| match get_leaf(v, p) {
            Value::Real(t) => Some(t.data.iter().map(|x| x.to_f64().abs()).sum::<f64>()),
            _ => None,
        }).sum()
    };
    let again = interp.call(gradf, dims.clone(), args.clone()).unwrap();
    assert_eq!(sum(&again), sum(&grad), "the same key gives the same mask");
    let Value::Key(k) = &key else { panic!("the third input is a key") };
    let other = vec![theta.clone(), x.clone(), Value::Key(k.child(1))];
    let elsewhere = interp.call(gradf, dims.clone(), other).unwrap();
    assert!(sum(&elsewhere) != sum(&grad), "another key gives another mask");

    // Training off, the mask is all ones: the model is its own expectation.
    let mut plain: Interp<Float> = Interp::new(&core, &prog, bits);
    let off = scalar(&plain.call(loss, ldims.clone(), vec![theta.clone(), x.clone(), key.clone()]).unwrap());
    let on = scalar(&interp.call(loss, ldims, vec![theta, x, key]).unwrap());
    assert!(off != on, "dropping must change the loss: off = {off}, on = {on}");
}

/// The adjoint of `merge` (0.4.0 step 1): flattening two adjacent axes moves
/// no data, so its adjoint is the retag back, and the gradient of a model that
/// goes through it must agree with central finite differences.
#[test]
fn ad_through_merge_matches_finite_differences() {
    let bits = 512;
    let rel = "tests/data/merge_grad.psk";
    let out = run(rel, &[], "loss_grad", &[("B", 2)], bits);
    let (prog, core) = prepare(rel, &[]);
    let loss = core.find("loss").unwrap();
    let dims: std::collections::BTreeMap<_, _> = core.func(loss).dims.iter().map(|d| (*d, match prog.dim_name(*d).as_str() { "B" => 2, _ => 1 })).collect();
    let theta = out.inputs[0].1.clone();
    let ids = out.inputs[1].1.clone();
    let mut interp: Interp<Float> = Interp::new(&core, &prog, bits);
    let h = Float::with_val(bits, 2f64.powi(-120));
    let tol = Float::with_val(bits, 2f64.powi(-100));
    let mut checked = 0;
    for path in &leaf_paths(&theta) {
        let Value::Real(t) = get_leaf(&theta, path) else { continue };
        let Value::Real(g) = get_leaf(&out.result, path) else { panic!("gradient leaf missing") };
        assert_eq!(t.axes, g.axes, "leaf {path:?} keeps its axes through the merge");
        for i in 0..t.data.len() {
            let plus = set_leaf_elem(&theta, path, i, Float::with_val(bits, &t.data[i] + &h));
            let minus = set_leaf_elem(&theta, path, i, Float::with_val(bits, &t.data[i] - &h));
            let fp = scalar(&interp.call(loss, dims.clone(), vec![plus, ids.clone()]).unwrap());
            let fm = scalar(&interp.call(loss, dims.clone(), vec![minus, ids.clone()]).unwrap());
            let fd = (fp - fm) / (Float::with_val(bits, 2) * &h);
            let err = (Float::with_val(bits, &fd - &g.data[i])).abs();
            let scale = Float::with_val(bits, g.data[i].clone().abs()).max(&Float::with_val(bits, 1));
            assert!(err <= Float::with_val(bits, &tol * &scale), "leaf {path:?}[{i}]: ad = {}, fd = {}, err = {err}", g.data[i], fd);
            checked += 1;
        }
    }
    assert_eq!(checked, 5 * 3 + 6 * 5, "every element of the table and the head");
}

/// Per-layer randomness inside a `stack{L}` fold (0.5.0 step 1, question 51).
/// A fold hands every iteration the same extras, so a key passed as an extra
/// is one key for all L blocks. Carried in the *accumulator* it is threaded,
/// and each layer draws its own mask — no new construct, and differentiable,
/// which an index-carrying fold would not have been.
#[test]
fn stacked_layers_draw_independent_keys() {
    let bits = 128;
    let rel = "tests/data/stack_keys.psk";
    let (prog, core) = prepare(rel, &[]);
    let zeros = |entry: &str| -> (usize, usize) {
        let opts = presk_interp::RunOptions { entry: entry.into(), dims: vec![("B".into(), 8)], seed: 0, bits, digits: 12 };
        let (f, dims, inputs) = presk_interp::run::prepare_run(&core, &prog, &opts).unwrap();
        let args: Vec<Value<Float>> = inputs.iter().map(|(_, v)| v.clone()).collect();
        let mut interp: Interp<Float> = Interp::new(&core, &prog, bits);
        interp.training = true;
        let Value::Real(t) = interp.call(f, dims, args).unwrap() else { panic!("a tensor") };
        (t.data.iter().filter(|x| x.is_zero()).count(), t.data.len())
    };
    let (chained, n) = zeros("chained");
    let (shared, m) = zeros("shared");
    assert_eq!((n, m), (8 * 16, 8 * 16));
    // Four independent masks at rate 1/2 leave about 1/16 of the elements;
    // one mask applied four times leaves about 1/2. The bands are wide enough
    // to be stable and far enough apart that no key scheme sits in both.
    assert!(chained as f64 / n as f64 > 0.85, "four independent masks should zero most of it, got {chained}/{n}");
    assert!((0.35..0.65).contains(&(shared as f64 / m as f64)), "one reused mask should zero about half, got {shared}/{m}");

    // And the chain is differentiable: a fold carrying its index would not
    // have been, since AD refuses to differentiate an indexed fold.
    let opts = presk_interp::RunOptions { entry: "loss_grad".into(), dims: vec![("B".into(), 2)], seed: 0, bits, digits: 12 };
    let (gradf, dims, inputs) = presk_interp::run::prepare_run(&core, &prog, &opts).unwrap();
    let args: Vec<Value<Float>> = inputs.iter().map(|(_, v)| v.clone()).collect();
    let mut interp: Interp<Float> = Interp::new(&core, &prog, bits);
    interp.training = true;
    let g = interp.call(gradf, dims, args).unwrap();
    let per_layer: Vec<f64> = leaf_paths(&g)
        .iter()
        .filter_map(|p| match get_leaf(&g, p) {
            Value::Real(t) => Some(t.data.iter().map(|x| x.to_f64().abs()).sum::<f64>()),
            _ => None,
        })
        .collect();
    assert!(!per_layer.is_empty(), "the gradient has leaves");
    assert!(per_layer.iter().all(|x| x.is_finite()), "every leaf is finite");
}

/// The gradient of a stacked module that drops, training on, agrees with
/// central finite differences (0.10.0 step 4, defect 3f of the reference's
/// chapter 5). The key rides the fold's accumulator, so the backward fold
/// reads each layer's key from what the forward fold stacked: a stack that
/// kept the first key gave every layer's reverse sweep layer 0's mask, off by
/// up to 77 % here. The blocks are residual, `x + dropout(tanh(x · γ))`, so
/// that a dropped position is no exact zero of what flows on, which hid the
/// defect in `tanh(dropout(x · γ))`.
#[test]
fn ad_through_stacked_dropout_matches_finite_differences() {
    let bits = 512;
    let rel = "tests/data/stack_dropout_grad.psk";
    let (prog, core) = prepare(rel, &[]);
    let opts = presk_interp::RunOptions { entry: "loss_grad".into(), dims: vec![("B".into(), 8)], seed: 0, bits, digits: 12 };
    let (gradf, dims, inputs) = presk_interp::run::prepare_run(&core, &prog, &opts).unwrap();
    let args: Vec<Value<Float>> = inputs.iter().map(|(_, v)| v.clone()).collect();
    let (theta, x, key) = (args[0].clone(), args[1].clone(), args[2].clone());
    let mut interp: Interp<Float> = Interp::new(&core, &prog, bits);
    interp.training = true;
    let grad = interp.call(gradf, dims.clone(), args.clone()).unwrap();

    let loss = core.find("loss").unwrap();
    let h = Float::with_val(bits, 2f64.powi(-120));
    let tol = Float::with_val(bits, 2f64.powi(-100));
    let mut checked = 0;
    for path in &leaf_paths(&theta) {
        let Value::Real(t) = get_leaf(&theta, path) else { continue };
        let Value::Real(g) = get_leaf(&grad, path) else { panic!("gradient leaf missing") };
        for i in 0..t.data.len() {
            let plus = set_leaf_elem(&theta, path, i, Float::with_val(bits, &t.data[i] + &h));
            let minus = set_leaf_elem(&theta, path, i, Float::with_val(bits, &t.data[i] - &h));
            let fp = scalar(&interp.call(loss, dims.clone(), vec![plus, x.clone(), key.clone()]).unwrap());
            let fm = scalar(&interp.call(loss, dims.clone(), vec![minus, x.clone(), key.clone()]).unwrap());
            let fd = (fp - fm) / (Float::with_val(bits, 2) * &h);
            let err = (Float::with_val(bits, &fd - &g.data[i])).abs();
            let scale = Float::with_val(bits, g.data[i].clone().abs()).max(&Float::with_val(bits, 1));
            assert!(err <= Float::with_val(bits, &tol * &scale), "leaf {path:?}[{i}]: ad = {}, fd = {}, err = {err}", g.data[i], fd);
            checked += 1;
        }
    }
    assert_eq!(checked, 3 * 4, "every element of the three layers");

    // Dropping must matter: training off, the loss differs.
    let mut plain: Interp<Float> = Interp::new(&core, &prog, bits);
    let off = scalar(&plain.call(loss, dims.clone(), vec![theta.clone(), x.clone(), key.clone()]).unwrap());
    let on = scalar(&interp.call(loss, dims, vec![theta, x, key]).unwrap());
    assert!(off != on, "dropping must change the loss: off = {off}, on = {on}");
}

/// The adjoint of `split` (0.5.0 step 1): reading one axis as two moves no
/// data, so its adjoint is the retag back — but only if the cotangent really
/// is put back under the original axis. Here the two halves are treated
/// differently (a per-pair scale broadcasts across the inner axis), which is
/// what rotary positions and grouped-query attention do, so a transposed or
/// column-major reading would survive the shapes and fail the numbers.
#[test]
fn ad_through_split_matches_finite_differences() {
    let bits = 512;
    let rel = "tests/data/split_grad.psk";
    let out = run(rel, &[], "loss_grad", &[("B", 2)], bits);
    let (prog, core) = prepare(rel, &[]);
    let loss = core.find("loss").unwrap();
    let dims: std::collections::BTreeMap<_, _> = core.func(loss).dims.iter().map(|d| (*d, match prog.dim_name(*d).as_str() { "B" => 2, _ => 1 })).collect();
    let theta = out.inputs[0].1.clone();
    let ids = out.inputs[1].1.clone();
    let mut interp: Interp<Float> = Interp::new(&core, &prog, bits);
    let h = Float::with_val(bits, 2f64.powi(-120));
    let tol = Float::with_val(bits, 2f64.powi(-100));
    let mut checked = 0;
    for path in &leaf_paths(&theta) {
        let Value::Real(t) = get_leaf(&theta, path) else { continue };
        let Value::Real(g) = get_leaf(&out.result, path) else { panic!("gradient leaf missing") };
        assert_eq!(t.axes, g.axes, "leaf {path:?} keeps its axes through the split");
        for i in 0..t.data.len() {
            let plus = set_leaf_elem(&theta, path, i, Float::with_val(bits, &t.data[i] + &h));
            let minus = set_leaf_elem(&theta, path, i, Float::with_val(bits, &t.data[i] - &h));
            let fp = scalar(&interp.call(loss, dims.clone(), vec![plus, ids.clone()]).unwrap());
            let fm = scalar(&interp.call(loss, dims.clone(), vec![minus, ids.clone()]).unwrap());
            let fd = (fp - fm) / (Float::with_val(bits, 2) * &h);
            let err = (Float::with_val(bits, &fd - &g.data[i])).abs();
            let scale = Float::with_val(bits, g.data[i].clone().abs()).max(&Float::with_val(bits, 1));
            assert!(err <= Float::with_val(bits, &tol * &scale), "leaf {path:?}[{i}]: ad = {}, fd = {}, err = {err}", g.data[i], fd);
            checked += 1;
        }
    }
    assert_eq!(checked, 3 * 4 + 2 + 4 * 3, "every element of the table, the per-pair scale and the head");
}

/// The derivatives of `concat`, of slices, and of `max`, `min` and `clamp`
/// (plan 0.17.0 step 6): a `concat`'s adjoint is two slices, a slice's the
/// cotangent placed into zeros, and `max` and `min` give it to the operand
/// they select. Away from their kinks, which drawn parameters never meet,
/// AD agrees with central finite differences at 512 bits; the clamp holds
/// some elements at its bounds, where the derivative is zero.
#[test]
fn ad_through_concat_slice_max_min_matches_finite_differences() {
    let bits = 512;
    let rel = "tests/data/concat_grad.psk";
    let out = run(rel, &[], "loss_grad", &[("B", 2)], bits);
    let (prog, core) = prepare(rel, &[]);
    let loss = core.find("loss").unwrap();
    let dims: std::collections::BTreeMap<_, _> = core.func(loss).dims.iter().map(|d| (*d, match prog.dim_name(*d).as_str() { "B" => 2, _ => 1 })).collect();
    let theta = out.inputs[0].1.clone();
    let x = out.inputs[1].1.clone();
    let mut interp: Interp<Float> = Interp::new(&core, &prog, bits);
    let h = Float::with_val(bits, 2f64.powi(-120));
    let tol = Float::with_val(bits, 2f64.powi(-100));
    let (mut checked, mut zeros) = (0, 0);
    for path in &leaf_paths(&theta) {
        let Value::Real(t) = get_leaf(&theta, path) else { continue };
        let Value::Real(g) = get_leaf(&out.result, path) else { panic!("gradient leaf missing") };
        assert_eq!(t.axes, g.axes, "leaf {path:?} keeps its axes");
        for i in 0..t.data.len() {
            let plus = set_leaf_elem(&theta, path, i, Float::with_val(bits, &t.data[i] + &h));
            let minus = set_leaf_elem(&theta, path, i, Float::with_val(bits, &t.data[i] - &h));
            let fp = scalar(&interp.call(loss, dims.clone(), vec![plus, x.clone()]).unwrap());
            let fm = scalar(&interp.call(loss, dims.clone(), vec![minus, x.clone()]).unwrap());
            let fd = (fp - fm) / (Float::with_val(bits, 2) * &h);
            let err = (Float::with_val(bits, &fd - &g.data[i])).abs();
            let scale = Float::with_val(bits, g.data[i].clone().abs()).max(&Float::with_val(bits, 1));
            assert!(err <= Float::with_val(bits, &tol * &scale), "leaf {path:?}[{i}]: ad = {}, fd = {}, err = {err}", g.data[i], fd);
            checked += 1;
            zeros += usize::from(g.data[i].is_zero());
        }
    }
    assert_eq!(checked, 3 * 2 + 2 * 2 + 2, "every element of a, b and w");
    assert!(zeros > 0 && zeros < checked, "some elements unselected or clamped, others not: {zeros} of {checked}");
}

/// The specification's examples of plan 0.17.0 step 6 (presk-specs
/// `examples/rope/`, `examples/concat/`), at the reference's precision.
#[test]
fn golden_rope_and_concat() {
    snap("rope_attention2", &run("tests/examples/rope/rope.psk", &[], "attention2", &[("B", 2), ("T", 8)], 256));
    snap("rope_attention3", &run("tests/examples/rope/rope.psk", &[], "attention3", &[("B", 2), ("T", 8)], 256));
    snap("concat_loss", &run("tests/examples/concat/concat.psk", &[], "loss", &[("B", 2), ("T", 8)], 256));
    snap("concat_loss_grad", &run("tests/examples/concat/concat.psk", &[], "loss_grad", &[("B", 2), ("T", 8)], 256));
}

/// A fold whose accumulator type has an open prefix (`ℝ[..., model: 4]`):
/// the scanned accumulators keep their axis names (a regression of the
/// tutorial's training run), and the gradient agrees with finite differences.
#[test]
fn ad_through_fold_with_open_prefix() {
    let bits = 512;
    let rel = "tests/data/stack_prefix_grad.psk";
    let out = run(rel, &[], "loss_grad", &[("B", 3)], bits);
    let (prog, core) = prepare(rel, &[]);
    let loss = core.find("loss").unwrap();
    let dims: std::collections::BTreeMap<_, _> = core.func(loss).dims.iter().map(|d| (*d, 3u64)).collect();
    let (theta, x, y) = (out.inputs[0].1.clone(), out.inputs[1].1.clone(), out.inputs[2].1.clone());
    let mut interp: Interp<Float> = Interp::new(&core, &prog, bits);
    let h = Float::with_val(bits, 2f64.powi(-120));
    let tol = Float::with_val(bits, 2f64.powi(-100));
    let mut checked = 0;
    for path in &leaf_paths(&theta) {
        let Value::Real(t) = get_leaf(&theta, path) else { continue };
        let Value::Real(g) = get_leaf(&out.result, path) else { panic!() };
        for i in 0..t.data.len() {
            let plus = set_leaf_elem(&theta, path, i, Float::with_val(bits, &t.data[i] + &h));
            let minus = set_leaf_elem(&theta, path, i, Float::with_val(bits, &t.data[i] - &h));
            let fp = scalar(&interp.call(loss, dims.clone(), vec![plus, x.clone(), y.clone()]).unwrap());
            let fm = scalar(&interp.call(loss, dims.clone(), vec![minus, x.clone(), y.clone()]).unwrap());
            let fd = (fp - fm) / (Float::with_val(bits, 2) * &h);
            let err = (Float::with_val(bits, &fd - &g.data[i])).abs();
            let scale = Float::with_val(bits, g.data[i].clone().abs()).max(&Float::with_val(bits, 1));
            assert!(err <= Float::with_val(bits, &tol * &scale), "leaf {path:?}[{i}]: ad = {}, fd = {}", g.data[i], fd);
            checked += 1;
        }
    }
    assert_eq!(checked, 2 * (16 + 4));
}

/// A chain of `n` links: `chain` adds a value to itself `n` times, `prod`
/// multiplies by `x` `n` times and sums; their gradients through `grad`.
fn chain_source(n: usize) -> String {
    let mut s = String::from("dim N\n\nfn chain(x : ℝ[n: N]) -> ℝ[n: N] =\n  let a0 = x + x\n");
    for i in 1..n {
        s.push_str(&format!("  let a{i} = a{} + a{}\n", i - 1, i - 1));
    }
    s.push_str(&format!("  a{}\n\nfn prod(x : ℝ[n: N]) -> ℝ[] =\n  let a0 = x * x\n", n - 1));
    for i in 1..n {
        s.push_str(&format!("  let a{i} = a{} * x\n", i - 1));
    }
    s.push_str(&format!("  sum(a{})\n\n", n - 1));
    s.push_str("fn chain_grad(x : ℝ[n: N]) -> ℝ[n: N] =\n  grad(λ y. sum(chain(y)))(x)\n\n");
    s.push_str("fn prod_grad(x : ℝ[n: N]) -> ℝ[n: N] =\n  grad(λ y. prod(y))(x)\n");
    s
}

/// The most reals the environments held at once while `entry` ran on `N`
/// elements, with the gradient stored or recomputed.
fn peak_live(n: usize, entry: &str, remat: presk_core::ad::Remat) -> usize {
    let mut sm = SourceMap::new();
    let id = sm.add(format!("chain{n}.psk"), chain_source(n));
    let (program, diags) = parse_file(&sm, id);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (prog, diags) = elab::elaborate(&mut sm, id, program, &Options::default());
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (typed, diags) = check::check(&sm, &prog);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (mut core, diags) = presk_core::lower(&prog, &typed);
    assert!(diags.is_empty(), "{}", render_all(&diags, &sm, false));
    let diags = presk_core::ad::transform_as(&mut core, presk_core::ad::Options { remat, groups: true });
    assert!(diags.is_empty(), "{}", render_all(&diags, &sm, false));
    let opts = RunOptions { entry: entry.into(), dims: vec![("N".into(), 100)], seed: 0, bits: 64, digits: 12 };
    let (func, dims, inputs) = presk_interp::run::prepare_run(&core, &prog, &opts).unwrap();
    let mut interp: Interp<Float> = Interp::new(&core, &prog, 64);
    interp.live = Some(Default::default());
    interp.call(func, dims, inputs.into_iter().map(|(_, v)| v).collect()).unwrap();
    let live = interp.live.unwrap();
    assert_eq!(live.now, 0, "{entry}: every binding is freed when the entry returns");
    live.peak
}

/// The evaluator frees a value after the `let` that reads it last (plan
/// 0.10.1 step 1): a chain of additions, and its gradient, hold a bounded
/// number of tensors however long the chain (two and four; at 0.10.0, one
/// more a link, and three more a link for the gradient); a product chain's
/// gradient, whose reverse sweep reads every forward value, at most two
/// more a link, the residuals of its product (at 0.10.0, ten storing and
/// five recomputing, every partial cotangent bound until the sweep
/// returned).
#[test]
fn liveness_bounds_what_a_chain_holds() {
    for remat in [presk_core::ad::Remat::Store, presk_core::ad::Remat::Recompute] {
        for entry in ["chain", "chain_grad"] {
            let (short, long) = (peak_live(8, entry, remat), peak_live(32, entry, remat));
            assert_eq!(short, long, "{entry} ({remat:?}): {short} reals at 8 links, {long} at 32");
            assert!(long <= 4 * 100, "{entry} ({remat:?}): {long} reals at once");
        }
        let (short, long) = (peak_live(8, "prod_grad", remat), peak_live(32, "prod_grad", remat));
        assert!(long - short <= 2 * 24 * 100, "prod_grad ({remat:?}): {short} reals at 8 links, {long} at 32");
    }
}

/// The shared zeros (plan 0.10.1 step 2): the evaluator gives `x + 0` as `x`
/// itself only where MPFR gives `x`, so never for `−0`, whose sum with `+0`
/// is `+0`, nor for a NaN.
#[test]
fn adding_zero_is_the_value_itself_but_for_minus_zero_and_nan() {
    use presk_interp::Real;
    let zero = Float::with_val(128, 0);
    for x in [Float::with_val(128, 1.5), Float::with_val(128, -2), Float::with_val(128, 0), Float::with_val(128, f64::INFINITY), Float::with_val(128, f64::NEG_INFINITY), Float::with_val(128, -0.0), Float::with_val(128, f64::NAN)] {
        let sum = Real::add(&x, &zero);
        let same = sum.prec() == x.prec() && (sum.is_nan() && x.is_nan() || sum == x && sum.is_sign_negative() == x.is_sign_negative());
        assert_eq!(x.plus_zero_is_self(), same && !x.is_nan(), "{x:?}");
    }
}
