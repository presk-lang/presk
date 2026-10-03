//! The generated Python of the five examples is snapshotted and, when a
//! `python3` is available, byte-compiled (PyTorch itself is not required).

use std::path::{Path, PathBuf};
use std::sync::Once;

use presk_export_pytorch::{Options, StackMode, export};
use presk_syntax::diag::render_all;
use presk_syntax::{SourceMap, parse_file};
use presk_types::check;
use presk_types::elab::{self, Options as ElabOptions};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn chdir_root() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| std::env::set_current_dir(repo_root()).unwrap());
}

fn generated(rel: &str, opts: &Options) -> Vec<(String, String)> {
    chdir_root();
    let mut sm = SourceMap::new();
    let id = sm.load(Path::new(rel)).unwrap();
    let (program, diags) = parse_file(&sm, id);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (prog, diags) = elab::elaborate(&mut sm, id, program, &ElabOptions::default());
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (typed, diags) = check::check(&sm, &prog);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (files, diags) = export(&sm, &prog, &typed, opts);
    assert!(diags.is_empty(), "{rel}: export diagnostics:\n{}", render_all(&diags, &sm, false));
    files.into_iter().map(|g| (g.name, g.content)).collect()
}

fn py_compile(dir: &Path) {
    let Ok(out) = std::process::Command::new("python3").arg("-m").arg("py_compile").args(std::fs::read_dir(dir).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "py"))).output() else {
        eprintln!("python3 not found: skipping byte-compilation");
        return;
    };
    assert!(out.status.success(), "py_compile failed:\n{}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn export_examples() {
    let tmp = std::env::temp_dir().join(format!("presk-export-{}", std::process::id()));
    for (rel, opts) in [
        ("mlp/mlp.psk", Options::default()),
        ("layernorm/layernorm.psk", Options::default()),
        ("minigpt/minigpt.psk", Options { shapes_check: true, stack: StackMode::Loop }),
        ("minigpt/train_minigpt.psk", Options::default()),
        ("minigpt/serve_minigpt.psk", Options { shapes_check: false, stack: StackMode::Unroll }),
    ] {
        let files = generated(&format!("tests/examples/{rel}"), &opts);
        let stem = Path::new(rel).file_stem().unwrap().to_str().unwrap();
        let dir = tmp.join(stem);
        std::fs::create_dir_all(&dir).unwrap();
        for (name, content) in &files {
            std::fs::write(dir.join(name), content).unwrap();
        }
        py_compile(&dir);
        // Snapshot the main file only (imports are snapshotted by their own example).
        let main = files.iter().find(|(n, _)| *n == format!("{stem}.py")).unwrap();
        insta::assert_snapshot!(format!("export_{stem}"), main.1);
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

/// 0.2.0 step 6: the generated training loop, the `.psk.d` reader and the
/// graph-mode driver are snapshotted; the schedule expressions are checked.
#[test]
fn training_and_graph_files() {
    use presk_export_pytorch::{ConformanceData, Schedule, TrainSpec, graph_driver, schedule_py, train_files};
    assert_eq!(schedule_py(&Schedule::Constant), "1.0");
    assert!(schedule_py(&Schedule::Linear { warmup: 0 }).starts_with("1.0 * (1.0 - "));
    assert!(schedule_py(&Schedule::Cosine { warmup: 2000 }).starts_with("min(1.0, t / 2000) * 0.5 * (1.0 + math.cos("));
    let spec = TrainSpec {
        source: "tests/examples/minigpt/train_minigpt.psk".into(),
        model_stem: "minigpt".into(),
        ctor: "MiniGPT(8)".into(),
        objective_stem: "train_minigpt".into(),
        objective: "loss".into(),
        lr: "3e-4".into(),
        betas: ("0.9".into(), "0.95".into()),
        eps: "1e-8".into(),
        weight_decay: "0.1".into(),
        grad_clip: Some("1.0".into()),
        schedule: Schedule::Cosine { warmup: 2000 },
        batch: 32,
        shuffle: Some(1234),
        epochs: 10,
        checkpoint_every: Some(1000),
        checkpoint_dir: Some("ckpt/".into()),
        key: false,
        rand: false,
        dims: vec![("B".into(), 32), ("T".into(), 8)],
        comments: vec!["The `parallel` and `precision` clauses are reproduced as comments in train_minigpt.py.".into()],
    };
    let files = train_files(&spec);
    assert_eq!(files.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(), ["pskd.py", "train.py"]);
    insta::assert_snapshot!("pskd_py", files[0].content);
    // `--serve`: serve.py over the same module, with the parameter tree's layout.
    use presk_export_pytorch::{ServeSpec, layout_py, serve_files};
    use presk_types::ty::{ParamsTree, Ty};
    let leaf = || ParamsTree::Leaf(Ty::Nat);
    let tree = ParamsTree::Node(vec![("head".into(), leaf()), ("blocks".into(), ParamsTree::Stacked { n: presk_types::dim::DimExpr::constant(presk_types::dim::rat(2)), inner: Box::new(ParamsTree::Node(vec![("w".into(), leaf())])) })]);
    assert_eq!(layout_py(&tree), r#"{"head": None, "blocks": ["stack", {"w": None}]}"#);
    let sspec = ServeSpec {
        source: "tests/examples/minigpt/serve_minigpt.psk".into(),
        model_stem: "minigpt".into(),
        class: "MiniGPT".into(),
        ctor_dims: vec!["T".into()],
        main_stem: "serve_minigpt".into(),
        generate: Some("next_token".into()),
        key: true,
        temperature: Some("0.8".into()),
        seq_dim: Some("T".into()),
        weights: Some("ckpt/final".into()),
        layout: layout_py(&tree),
        ignored: vec!["precision".into(), "target".into(), "batching".into(), "cache".into()],
    };
    let sfiles = serve_files(&sspec);
    assert_eq!(sfiles.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(), ["pskd.py", "serve.py"]);
    insta::assert_snapshot!("serve_py", sfiles[1].content);
    insta::assert_snapshot!("train_py", files[1].content);
    let data = ConformanceData {
        entry: "MiniGPT.apply".into(),
        module_file: "minigpt".into(),
        class: Some(("MiniGPT".into(), vec![8])),
        function: None,
        theta_modules: vec![],
        optimizer: None,
        train: None,
        graph: true,
        inputs: vec!["θ".into(), "ids".into()],
        seed: 0,
        bits: 256,
        rtol: 1e-9,
        atol: 1e-12,
    };
    insta::assert_snapshot!("export_py", graph_driver(&data).unwrap().content);
    assert!(presk_export_pytorch::test_file(&data).contains("def test_onnx"));
}
