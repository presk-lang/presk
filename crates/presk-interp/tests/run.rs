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
    let mut interp: Interp<Float> = Interp::new(&core, &prog, 256, 0);
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
    let mut interp: Interp<Float> = Interp::new(&core, &prog, bits, 0);
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
        (Value::Record(fs), Some((h, rest))) => Value::Record(fs.iter().map(|(n, sub)| (n.clone(), if n == h { set_leaf_elem(sub, rest, i, x.clone()) } else { sub.clone() })).collect()),
        (Value::Real(t), None) => {
            let mut t = t.clone();
            t.data[i] = x;
            Value::Real(t)
        }
        _ => v.clone(),
    }
}
