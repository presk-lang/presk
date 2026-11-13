//! The generation loop on the tiny MiniGPT: the greedy path gives the same
//! tokens on the reference interpreter and natively in both realizations
//! (the weights are a fresh initialization saved as a checkpoint); the
//! sampling path produces tokens of the vocabulary; the window rule holds;
//! the loop's time is split into compiling and computing (plan 0.17.2 step
//! 4, D200).

use std::collections::BTreeMap;
use std::path::Path;

use presk_cranelift::native_backend;
use presk_interp::key::Key;
use presk_rt::{Backend, InterpBackend, Dtype, Value, save_params, Realization};
use presk_serve::{ServeOptions, prepare_serve, serve};
use presk_syntax::{SourceMap, parse_file};
use presk_types::check;
use presk_types::elab::{self, Options};

fn root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

#[test]
fn greedy_generation_agrees_across_backends() {
    std::env::set_current_dir(root()).unwrap();
    let mut sm = SourceMap::new();
    let id = sm.load(Path::new("examples/spec/minigpt/serve_minigpt.psk")).unwrap();
    let (program, _) = parse_file(&sm, id);
    let opts = Options { const_overrides: vec![("V".into(), 64), ("D".into(), 16), ("H".into(), 2), ("L".into(), 2)] };
    let (prog, diags) = elab::elaborate(&mut sm, id, program, &opts);
    assert!(diags.iter().all(|d| !d.is_error()));
    let (typed, diags) = check::check(&sm, &prog);
    assert!(diags.iter().all(|d| !d.is_error()));
    assert_eq!(typed.inference.len(), 1);
    assert!(typed.inference[0].generate.is_some(), "next_token is the generation function");
    let (mut core, _) = presk_core::lower(&prog, &typed);
    presk_core::ad::transform(&mut core);
    let fns = prepare_serve(&mut core, &prog, &typed, None).unwrap();
    assert!(fns.temperature);
    assert_eq!(fns.default_temperature, Some(0.8));
    assert_eq!((fns.batch_dim.as_str(), fns.seq_dim.as_str()), ("B", "T"));
    // Weights: a fresh initialization at T = 8, saved as `presk train` would.
    let dir = std::env::temp_dir().join(format!("presk-serve-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let weights = dir.join("final.psk.d");
    {
        let mut b = InterpBackend::new(&core, &prog, 64);
        // The initializer carries the file's dimensions (`B`, `T`); only `T` shapes a parameter.
        let dims: BTreeMap<_, _> = core.func(fns.init).dims.iter().map(|d| (*d, if prog.dim_name(*d) == "T" { 8 } else { 1 })).collect();
        let init = b.prepare(fns.init, &dims, &Realization::uniform(Dtype::F64)).unwrap();
        let theta = b.call(&init, vec![Value::Key(Key::from_seed(7))]).unwrap();
        save_params(&weights, &fns.model, &theta).unwrap();
    }
    let prompt: Vec<u64> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let sopts = ServeOptions { weights: Some(weights.clone()), prompt: prompt.clone(), tokens: 4, seed: 0, bits: 64, realize: Realization::uniform(Dtype::F64), temperature: Some(0.0), dims: BTreeMap::new() };
    let reference = serve(&mut InterpBackend::new(&core, &prog, 64), &core, &prog, &fns, &sopts, |_, _| {}).unwrap();
    assert_eq!(reference.window, Some(8), "pos : ℝ[T, D] ties the model to T");
    assert!(reference.greedy);
    assert_eq!(reference.vocab, 64);
    assert_eq!(reference.generated.len(), 4);
    // One length, the window's; the interpreter compiles nothing.
    assert_eq!((reference.compilations, reference.kernels), (1, 0));
    assert!(reference.compute_s > 0.0);
    for realize in [Realization::uniform(Dtype::F64), Realization::uniform(Dtype::F32)] {
        let o = ServeOptions { realize: realize.clone(), ..sopts.clone() };
        let native = serve(&mut native_backend(&core, &prog, &realize).unwrap(), &core, &prog, &fns, &o, |_, _| {}).unwrap();
        assert_eq!(native.generated, reference.generated, "{realize:?}");
        assert_eq!(native.compilations, 1, "{realize:?}");
        assert!(native.kernels > 0 && native.compile_s > 0.0 && native.compute_s > 0.0, "{realize:?}: {native:?}");
    }
    // Sampling: tokens of the vocabulary, deterministic for a seed.
    let s1 = ServeOptions { temperature: None, ..sopts.clone() };
    let a = serve(&mut native_backend(&core, &prog, &Realization::uniform(Dtype::F64)).unwrap(), &core, &prog, &fns, &s1, |_, _| {}).unwrap();
    let b = serve(&mut native_backend(&core, &prog, &Realization::uniform(Dtype::F64)).unwrap(), &core, &prog, &fns, &s1, |_, _| {}).unwrap();
    assert!(!a.greedy && a.generated.iter().all(|t| *t < 64));
    assert_eq!(a.generated, b.generated);
    // Every token draws from its own key, the child of the serve's (plan
    // 0.10.0 step 7b): the reference interpreter draws the native backend's
    // tokens, where each drew from a stream of its own; and another seed
    // names other keys.
    assert!(fns.key, "next_token takes a key");
    for seed in [0, 1, 7] {
        let o = ServeOptions { seed, ..s1.clone() };
        let interp = serve(&mut InterpBackend::new(&core, &prog, 256), &core, &prog, &fns, &o, |_, _| {}).unwrap();
        let native = serve(&mut native_backend(&core, &prog, &Realization::uniform(Dtype::F64)).unwrap(), &core, &prog, &fns, &o, |_, _| {}).unwrap();
        assert_eq!(interp.generated, native.generated, "seed {seed}");
        if seed > 0 {
            assert_ne!(native.generated, a.generated, "seed {seed} draws seed 0's tokens");
        }
    }
    // The window: a longer prompt slides; a shorter one is refused.
    let long = ServeOptions { prompt: (0..12).collect(), dims: BTreeMap::from([("T".to_string(), 8)]), ..sopts.clone() };
    assert_eq!(serve(&mut native_backend(&core, &prog, &Realization::uniform(Dtype::F64)).unwrap(), &core, &prog, &fns, &long, |_, _| {}).unwrap().window, Some(8));
    let short = ServeOptions { prompt: vec![1, 2], dims: BTreeMap::from([("T".to_string(), 8)]), ..sopts.clone() };
    let e = serve(&mut native_backend(&core, &prog, &Realization::uniform(Dtype::F64)).unwrap(), &core, &prog, &fns, &short, |_, _| {}).unwrap_err();
    assert!(e.message.contains("fix the context length T = 8"), "{}", e.message);
    // `--entry` names a function that is not a generation function.
    let e = prepare_serve(&mut core, &prog, &typed, Some("nope")).unwrap_err();
    assert!(e.message.contains("no top-level function"));
    std::fs::remove_dir_all(&dir).ok();
}

/// The field `generate` names the generation function (the reference's
/// decision W4, plan 0.10.0 step 8): `last`, the second of two functions
/// whose signatures fit, which `presk serve` then runs; `--entry` still
/// overrides it.
#[test]
fn the_generate_field_names_the_generation_function() {
    std::env::set_current_dir(root()).unwrap();
    let mut sm = SourceMap::new();
    let id = sm.load(Path::new("examples/tests/data/serve_generate.psk")).unwrap();
    let (program, _) = parse_file(&sm, id);
    let (prog, diags) = elab::elaborate(&mut sm, id, program, &Options::default());
    assert!(diags.iter().all(|d| !d.is_error()));
    let (typed, diags) = check::check(&sm, &prog);
    assert!(diags.is_empty(), "{:?}", diags.iter().map(|d| &d.code).collect::<Vec<_>>());
    let named = typed.inference[0].generate.map(|f| prog.function(f).name.clone());
    assert_eq!(named.as_deref(), Some("last"));
    let (core, _) = presk_core::lower(&prog, &typed);
    let mut served = core.clone();
    let fns = prepare_serve(&mut served, &prog, &typed, None).unwrap();
    assert_eq!(served.func(fns.generate).name, "last");
    let dir = std::env::temp_dir().join(format!("presk-serve-generate-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let weights = dir.join("final.psk.d");
    {
        let mut b = InterpBackend::new(&core, &prog, 64);
        let dims: BTreeMap<_, _> = core.func(fns.init).dims.iter().map(|d| (*d, 1)).collect();
        let init = b.prepare(fns.init, &dims, &Realization::uniform(Dtype::F64)).unwrap();
        let theta = b.call(&init, vec![Value::Key(Key::from_seed(3))]).unwrap();
        save_params(&weights, &fns.model, &theta).unwrap();
    }
    let sopts = ServeOptions { weights: Some(weights), prompt: vec![1, 2, 3], tokens: 4, seed: 0, bits: 64, realize: Realization::uniform(Dtype::F64), temperature: None, dims: BTreeMap::new() };
    let tokens = |entry: Option<&str>| {
        let mut core = core.clone();
        let fns = prepare_serve(&mut core, &prog, &typed, entry).unwrap();
        serve(&mut InterpBackend::new(&core, &prog, 64), &core, &prog, &fns, &sopts, |_, _| {}).unwrap().generated
    };
    let (field, last, first) = (tokens(None), tokens(Some("last")), tokens(Some("first")));
    assert_eq!(field, last);
    assert_ne!(field, first, "the least likely tokens are the most likely");
    // The whole context grows: one length a token, 3 to 6, each new length
    // compiling its kernels inside the call, where the backend counts them
    // (plan 0.17.2 step 4, D200).
    let native = |tokens: u64| {
        let mut core = core.clone();
        let fns = prepare_serve(&mut core, &prog, &typed, None).unwrap();
        let o = ServeOptions { tokens, ..sopts.clone() };
        serve(&mut native_backend(&core, &prog, &Realization::uniform(Dtype::F64)).unwrap(), &core, &prog, &fns, &o, |_, _| {}).unwrap()
    };
    let (one, four) = (native(1), native(4));
    assert_eq!(four.generated, field);
    assert_eq!((one.compilations, four.compilations), (1, 4));
    assert!(one.kernels > 0 && four.kernels > one.kernels, "{} kernels at one length, {} at four", one.kernels, four.kernels);
    std::fs::remove_dir_all(&dir).ok();
}
