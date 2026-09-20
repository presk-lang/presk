//! The generation loop on the tiny MiniGPT: the greedy path gives the same
//! tokens on the reference interpreter and natively in both realizations
//! (the weights are a fresh initialization saved as a checkpoint); the
//! sampling path produces tokens of the vocabulary; the window rule holds.

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
    let id = sm.load(Path::new("tests/examples/minigpt/serve_minigpt.psk")).unwrap();
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
        let mut b = InterpBackend::new(&core, &prog, 64, 7);
        // The initializer carries the file's dimensions (`B`, `T`); only `T` shapes a parameter.
        let dims: BTreeMap<_, _> = core.func(fns.init).dims.iter().map(|d| (*d, if prog.dim_name(*d) == "T" { 8 } else { 1 })).collect();
        let init = b.prepare(fns.init, &dims, &Realization::uniform(Dtype::F64)).unwrap();
        let theta = b.call(&init, vec![Value::Key(Key::from_seed(7))]).unwrap();
        save_params(&weights, &fns.model, &theta).unwrap();
    }
    let prompt: Vec<u64> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let sopts = ServeOptions { weights: Some(weights.clone()), prompt: prompt.clone(), tokens: 4, seed: 0, bits: 64, realize: Realization::uniform(Dtype::F64), temperature: Some(0.0), dims: BTreeMap::new() };
    let reference = serve(&mut InterpBackend::new(&core, &prog, 64, 0), &core, &prog, &fns, &sopts, |_, _| {}).unwrap();
    assert_eq!(reference.window, Some(8), "pos : ℝ[T, D] ties the model to T");
    assert!(reference.greedy);
    assert_eq!(reference.vocab, 64);
    assert_eq!(reference.generated.len(), 4);
    for realize in [Realization::uniform(Dtype::F64), Realization::uniform(Dtype::F32)] {
        let o = ServeOptions { realize: realize.clone(), ..sopts.clone() };
        let native = serve(&mut native_backend(&core, &prog, &realize, 0).unwrap(), &core, &prog, &fns, &o, |_, _| {}).unwrap();
        assert_eq!(native.generated, reference.generated, "{realize:?}");
    }
    // Sampling: tokens of the vocabulary, deterministic for a seed.
    let s1 = ServeOptions { temperature: None, ..sopts.clone() };
    let a = serve(&mut native_backend(&core, &prog, &Realization::uniform(Dtype::F64), 3).unwrap(), &core, &prog, &fns, &s1, |_, _| {}).unwrap();
    let b = serve(&mut native_backend(&core, &prog, &Realization::uniform(Dtype::F64), 3).unwrap(), &core, &prog, &fns, &s1, |_, _| {}).unwrap();
    assert!(!a.greedy && a.generated.iter().all(|t| *t < 64));
    assert_eq!(a.generated, b.generated);
    // The window: a longer prompt slides; a shorter one is refused.
    let long = ServeOptions { prompt: (0..12).collect(), dims: BTreeMap::from([("T".to_string(), 8)]), ..sopts.clone() };
    assert_eq!(serve(&mut native_backend(&core, &prog, &Realization::uniform(Dtype::F64), 0).unwrap(), &core, &prog, &fns, &long, |_, _| {}).unwrap().window, Some(8));
    let short = ServeOptions { prompt: vec![1, 2], dims: BTreeMap::from([("T".to_string(), 8)]), ..sopts.clone() };
    let e = serve(&mut native_backend(&core, &prog, &Realization::uniform(Dtype::F64), 0).unwrap(), &core, &prog, &fns, &short, |_, _| {}).unwrap_err();
    assert!(e.message.contains("fix the context length T = 8"), "{}", e.message);
    // `--entry` names a function that is not a generation function.
    let e = prepare_serve(&mut core, &prog, &typed, Some("nope")).unwrap_err();
    assert!(e.message.contains("no top-level function"));
    std::fs::remove_dir_all(&dir).ok();
}
