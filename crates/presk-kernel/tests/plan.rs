//! Realization plans on the kernel backend (plan 0.7.0, step 1): a plan with
//! the default layout lowers every example entry to the kernels it had; two
//! layouts of one nest never share a compiled kernel; and a layout this
//! version cannot lower, or one the reference calls ill-formed, is refused
//! with `E7002` rather than ignored.

use std::path::{Path, PathBuf};
use std::sync::Once;

use presk_interp::RunOptions;
use presk_kernel::KernelBackend;
use presk_kernel::plan::check_plan;
use presk_rt::{Plan, Realization, Value};
use presk_syntax::diag::render_all;
use presk_syntax::{SourceMap, parse_file};
use presk_types::check;
use presk_types::elab::{self, ElabProgram, Options};

fn chdir_root() {
    static ONCE: Once = Once::new();
    let root: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap();
    ONCE.call_once(|| std::env::set_current_dir(root).unwrap());
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

fn plan(json: &str) -> Plan {
    Plan::from_json(&serde_json::from_str(json).unwrap()).unwrap()
}

/// The entry run on the kernel interpreter under `plan`: its output, and
/// every kernel it compiled, as `presk dump --stage kernel` prints them.
fn run(core: &presk_core::Program, prog: &ElabProgram, entry: &str, dims: &[(&str, u64)], plan: Option<&Plan>, r: &Realization) -> presk_rt::Result<(Value, Vec<presk_kernel::Kernel>)> {
    let opts = RunOptions { entry: entry.into(), dims: dims.iter().map(|(n, v)| (n.to_string(), *v)).collect(), seed: 0, bits: 128, digits: 17 };
    let mut b = KernelBackend::new(core, prog, r, 0);
    if let Some(p) = plan {
        b.set_plan(p.clone());
    }
    let out = presk_rt::run_with(&mut b, core, prog, &opts, r)?;
    Ok((Value::from_interp(&out.result)?, b.kernels.clone()))
}

fn bits_of(v: &Value) -> Vec<u64> {
    match v {
        Value::Real(t) => t.data.to_f64().into_iter().map(|x| x.to_bits()).collect(),
        Value::Tuple(xs) => xs.iter().flat_map(bits_of).collect(),
        Value::Record(xs) => xs.iter().flat_map(|(_, x)| bits_of(x)).collect(),
        _ => vec![],
    }
}

const TINY: &[(&str, u64)] = &[("V", 64), ("D", 16), ("H", 2), ("L", 2)];

/// Step 1's check: `--realize` is a plan with the default layout, and a plan
/// that states no decision, or states each at the value the kernel already
/// has, lowers every example entry to the same kernels, text for text.
#[test]
fn a_default_plan_lowers_to_the_kernels_of_0_6_0() {
    type Case = (&'static str, &'static [(&'static str, u64)], &'static str, &'static [(&'static str, u64)]);
    let cases: &[Case] = &[
        ("tests/examples/mlp/mlp.psk", &[], "mse", &[("D", 4), ("B", 2)]),
        ("tests/examples/mlp/mlp.psk", &[], "mse_grad", &[("D", 4), ("B", 2)]),
        ("tests/examples/layernorm/layernorm.psk", &[], "LayerNorm{8}.apply", &[("batch", 3)]),
        ("tests/examples/minigpt/minigpt.psk", TINY, "MiniGPT.apply", &[("B", 2), ("T", 8)]),
        ("tests/examples/minigpt/train_minigpt.psk", TINY, "loss", &[("B", 2), ("T", 8)]),
        ("tests/examples/minigpt/train_minigpt.psk", TINY, "adamw.step", &[("B", 2), ("T", 8)]),
        ("tests/examples/minigpt/serve_minigpt.psk", TINY, "next_token", &[("B", 2), ("T", 8)]),
    ];
    for (rel, consts, entry, dims) in cases {
        let (prog, core) = prepare(rel, consts);
        for r in [Realization::uniform(presk_rt::Dtype::F64), Realization::uniform(presk_rt::Dtype::F32)] {
            let (_, bare) = run(&core, &prog, entry, dims, None, &r).unwrap();
            let (_, planned) = run(&core, &prog, entry, dims, Some(&Plan::from_realization(r.clone())), &r).unwrap();
            let text = |ks: &[presk_kernel::Kernel]| ks.iter().map(|k| k.to_string()).collect::<String>();
            assert_eq!(text(&bare), text(&planned), "{rel} {entry} ({}): a default plan changed a kernel", r.label());
        }
    }
    // The attention of the tiny course GPT, every decision stated at its value.
    let (prog, core) = prepare("corpus/nlp/05_gpt.psk", &[("V", 92), ("D", 16), ("H", 2), ("L", 1)]);
    let stated = plan(
        r#"{ "realization": "f32", "layout": {
             "SelfAttention.apply/%13": { "form": "tiled", "tile": [1, 1], "kblock": "none", "order": "IJ", "lanes": 1, "pack": false },
             "SelfAttention.apply/%20": { "passes": 3 },
             "SelfAttention.apply/%23": { "form": "tiled", "tile": [1, 1], "kblock": "none", "order": "IJ", "lanes": 1, "pack": false },
             "Linear.apply/%3": { "form": "tiled", "tile": [1, 8], "lanes": 1 } } }"#,
    );
    check_plan(&stated, &core).unwrap();
    let r = stated.realization.clone();
    let dims: &[(&str, u64)] = &[("B", 2), ("T", 8)];
    let (a, bare) = run(&core, &prog, "SelfAttention.apply", dims, None, &r).unwrap();
    let (b, planned) = run(&core, &prog, "SelfAttention.apply", dims, Some(&stated), &r).unwrap();
    assert_eq!(bare.iter().map(|k| k.to_string()).collect::<Vec<_>>(), planned.iter().map(|k| k.to_string()).collect::<Vec<_>>());
    assert_eq!(bits_of(&a), bits_of(&b));
}

/// Two nests of one shape with two layouts get two kernels, and a backend
/// given a second plan does not reuse the first plan's kernel for a nest
/// whose layout changed. The bits do not change: lanes are exact.
#[test]
fn two_layouts_of_one_shape_do_not_share_a_kernel() {
    let (prog, core) = prepare("tests/plan/two_dots.psk", &[]);
    let r = Realization::uniform(presk_rt::Dtype::F32);
    let dots = |ks: &[presk_kernel::Kernel]| ks.iter().filter(|k| k.name == "dot").map(|k| k.lanes).collect::<Vec<_>>();
    let (bare, ks) = run(&core, &prog, "twice", &[], None, &r).unwrap();
    assert_eq!(dots(&ks), [1], "without a plan the two nests share one scalar kernel");
    let two = plan(r#"{ "realization": "f32", "layout": { "twice/%2": { "lanes": 4 }, "twice/%3": { "lanes": 1 } } }"#);
    check_plan(&two, &core).unwrap();
    let (planned, ks) = run(&core, &prog, "twice", &[], Some(&two), &r).unwrap();
    assert_eq!(dots(&ks), [4, 1], "two layouts, two kernels");
    assert_eq!(bits_of(&bare), bits_of(&planned), "a layout moves no bit");

    // One backend, two plans that differ in one nest's layout.
    let opts = RunOptions { entry: "twice".into(), dims: vec![], seed: 0, bits: 128, digits: 17 };
    let mut b = KernelBackend::new(&core, &prog, &r, 0);
    b.set_plan(plan(r#"{ "realization": "f32", "layout": { "twice/%2": { "lanes": 1 } } }"#));
    presk_rt::run_with(&mut b, &core, &prog, &opts, &r).unwrap();
    assert_eq!(dots(&b.kernels), [1]);
    b.set_plan(plan(r#"{ "realization": "f32", "layout": { "twice/%2": { "lanes": 4 } } }"#));
    presk_rt::run_with(&mut b, &core, &prog, &opts, &r).unwrap();
    assert_eq!(dots(&b.kernels), [1, 4], "the second plan compiled its own kernel for %2 and reused the first for %3");
}

/// `E7002`: a plan that names no nest, or states a decision the nest cannot
/// have, is refused before anything runs; one that is ill-formed at the
/// shapes or against the kernel (a split at a loop that is not parallel),
/// when the nest is launched.
#[test]
fn a_plan_this_version_cannot_apply_is_refused() {
    let (prog, core) = prepare("tests/plan/two_dots.psk", &[]);
    let static_errors = [
        (r#"{ "twice/%9": { "lanes": 1 } }"#, "has no binding %9"),
        (r#"{ "thrice/%2": { "lanes": 1 } }"#, "no function `thrice`"),
        (r#"{ "twice/%2": { "passes": 2 } }"#, "`passes` is a decision of a softmax nest"),
        (r#"{ "attend/%3": { "tile": [1, 8] } }"#, "`tile` is a decision of a contraction nest"),
    ];
    for (layout, expect) in static_errors {
        let p = plan(&format!(r#"{{ "realization": "f32", "layout": {layout} }}"#));
        let e = check_plan(&p, &core).unwrap_err();
        assert_eq!(e.code, "E7002");
        assert!(e.message.contains(expect), "{layout}: {}", e.message);
    }
    check_plan(&plan(r#"{ "realization": "f32", "layout": { "attend/%3": { "split": "none" } } }"#), &core).unwrap();

    let at_launch = [
        ("attend", r#"{ "attend/%3": { "split": { "depth": 2, "workers": 2 } } }"#, "condition 8"),
        ("twice", r#"{ "twice/%2": { "split": { "depth": 2, "workers": 2 } } }"#, "condition 8"),
        ("twice", r#"{ "twice/%2": { "lanes": 2 } }"#, "condition 4"),
        ("twice", r#"{ "twice/%2": { "pack": true } }"#, "condition 9"),
        ("twice", r#"{ "twice/%2": { "tile": [1, 8], "kblock": 32, "order": "IJK" } }"#, "condition 3"),
        ("attend", r#"{ "attend/%2": { "lanes": 4 } }"#, "condition 4"),
    ];
    let r = Realization::uniform(presk_rt::Dtype::F32);
    for (entry, layout, expect) in at_launch {
        let p = plan(&format!(r#"{{ "realization": "f32", "layout": {layout} }}"#));
        check_plan(&p, &core).unwrap();
        let e = run(&core, &prog, entry, &[], Some(&p), &r).unwrap_err();
        assert_eq!(e.code, "E7002", "{layout}: {}", e.message);
        assert!(e.message.contains(expect), "{layout}: {}", e.message);
    }
    // In bf16 two passes would round twice: ill-formed before unimplemented.
    let p = plan(r#"{ "realization": "bf16", "layout": { "attend/%3": { "passes": 2 } } }"#);
    let e = run(&core, &prog, "attend", &[], Some(&p), &p.realization).unwrap_err();
    assert!(e.message.contains("condition 7"), "{}", e.message);
}
