//! Statement P2 of plan 0.13.0 (D134): the provenance table is complete.
//! On the entries `make plan-check` runs, the coverage models at their tiny
//! constants and the course GPT's `loss` at D = 64, every stage run as
//! `presk explain` runs it, with and without the optimizer: every element of
//! every stage but the source has a link that made it, every element a link
//! names is in the table, and no link points backwards. On every program of
//! the test set, the same through the stages that need no entry, from each
//! of its functions.

use std::path::PathBuf;

use presk_driver::Collect;
use presk_explain::{Options, Target};
use presk_provenance::{Stage, Table};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn args(s: &str) -> (Vec<String>, Vec<String>) {
    let w: Vec<&str> = s.split_whitespace().collect();
    let (mut dims, mut consts) = (Vec::new(), Vec::new());
    for p in w.chunks(2) {
        match p[0] {
            "--dim" => dims.push(p[1].to_string()),
            "--const" => consts.push(p[1].to_string()),
            other => panic!("{other}"),
        }
    }
    (dims, consts)
}

/// The table of `file` at `entry` under the frozen profile's plan, or under
/// `--opt` without one.
fn table(name: &str, file: &str, entry: &str, flags: &str, opt: bool) -> Table {
    table_as(name, file, entry, flags, opt, true, (None, Some("f32")))
}

/// [`table`], with the exact reference's envelope unless `no_exact`, and a
/// budget or a realization.
fn table_as(name: &str, file: &str, entry: &str, flags: &str, opt: bool, no_exact: bool, (budget, realize): (Option<f64>, Option<&str>)) -> Table {
    let (dims, consts) = args(flags);
    let o = Options {
        file: root().join(file),
        entry: entry.into(),
        dims,
        consts,
        budget,
        menu: Default::default(),
        realize: realize.map(String::from),
        plan: None,
        profile: (!opt).then(|| PathBuf::from("tests/data/x925-frozen.json")),
        threads: Some(10),
        remat: presk_core::ad::Remat::Store,
        opt,
        weights: None,
        seed: 0,
        bits: 128,
        envelope: None,
        no_exact,
        targets: vec![Target::Native, Target::C],
        measure: false,
        sites: Vec::new(),
        json: false,
        html: false,
        out: PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("provenance").join(format!("{name}{}", if opt { "-opt" } else { "" })),
    };
    let mut report = Collect::default();
    presk_explain::provenance(&o, &mut report).unwrap_or_else(|s| panic!("{name}: stopped with {s:?}: {:?}", report.items))
}

/// What P2 asks of a table; the counts per stage, for the failure message.
fn complete(name: &str, t: &Table) -> Vec<String> {
    let mut bad = Vec::new();
    for id in t.orphans() {
        bad.push(format!("{name}: orphan {id} ({})", t.label(id).unwrap_or("")));
    }
    for id in t.dangling() {
        bad.push(format!("{name}: a link names {id}, which the table does not have"));
    }
    for l in t.backwards() {
        bad.push(format!("{name}: {} made from a later stage: {:?}", l.child, l.parents));
    }
    // Forwards, where the kernels were listed: every `let` of the program
    // they run is in a nest or ends with a reason, every nest launches a
    // kernel, and every kernel is a C function when the C was exported.
    if !t.elements(Stage::Kernel).is_empty() {
        let x = if t.elements(Stage::Opt).is_empty() { Stage::Ad } else { Stage::Opt };
        for id in t.elements(x).iter().filter(|id| id.key.contains("/%")) {
            if t.descendants(id, Stage::Fusion).is_empty() && t.ended(id).is_none() {
                bad.push(format!("{name}: {id} reaches no nest, and no end says why"));
            }
        }
        for id in t.elements(Stage::Fusion) {
            if t.descendants(id, Stage::Kernel).is_empty() {
                bad.push(format!("{name}: the nest {id} launches no kernel"));
            }
        }
        if !t.elements(Stage::C).is_empty() {
            for id in t.elements(Stage::Kernel) {
                if t.descendants(id, Stage::C).is_empty() {
                    bad.push(format!("{name}: the kernel {id} is no C function"));
                }
            }
        }
    }
    bad
}

fn counts(t: &Table) -> String {
    Stage::ALL.iter().map(|s| (s, t.elements(*s).len())).filter(|(_, n)| *n > 0).map(|(s, n)| format!("{} {n}", s.name())).collect::<Vec<_>>().join(", ")
}

/// `make plan-check`'s entries, the coverage models at `make coverage`'s
/// tiny constants, and the course GPT at D = 64.
fn entries() -> Vec<(String, String, String)> {
    let mut out: Vec<(String, String, String)> = std::fs::read_to_string(root().join("tests/plan/entries.txt"))
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let p: Vec<&str> = l.splitn(3, '|').collect();
            (p[0].to_string(), p[1].to_string(), p[2].to_string())
        })
        .collect();
    out.push(("coverage/bert.psk".into(), "BertSquad.apply".into(), "--const V=100 --const D=32 --const H=2 --const L=2 --dim B=2 --dim T=16".into()));
    out.push(("coverage/llama.psk".into(), "Llama.apply".into(), "--const V=100 --const D=32 --const H=2 --const G=2 --const FF=64 --const L=2 --dim B=2 --dim T=16".into()));
    out.push(("corpus/nlp/05_gpt.psk".into(), "loss".into(), "--const D=64 --const H=2 --const L=2 --dim B=2 --dim T=16".into()));
    out
}

#[test]
fn every_element_of_every_stage_has_a_parent() {
    let mut bad = Vec::new();
    for (file, entry, flags) in entries() {
        for opt in [false, true] {
            let name = format!("{file}:{entry}");
            let t = table(&name.replace(['/', ':', '{', '}', ','], "_"), &file, &entry, &flags, opt);
            eprintln!("{name}{}: {}", if opt { " --opt" } else { "" }, counts(&t));
            assert!(!t.elements(Stage::Kernel).is_empty(), "{name}: no kernel");
            bad.extend(complete(&name, &t));
        }
    }
    assert!(bad.is_empty(), "{} defects:\n{}", bad.len(), bad.iter().take(40).cloned().collect::<Vec<_>>().join("\n"));
}

/// The stages that need no entry, from every function of every program of
/// the test set (`make check-examples`' programs).
#[test]
fn every_program_of_the_test_set_through_its_front_stages() {
    let mut bad = Vec::new();
    for file in ["tests/examples/mlp/mlp.psk", "tests/examples/layernorm/layernorm.psk", "tests/examples/minigpt/minigpt.psk", "tests/examples/minigpt/train_minigpt.psk", "tests/examples/minigpt/serve_minigpt.psk"] {
        let mut report = Collect::default();
        let (sm, prog, typed, core, lowered) = presk_driver::front::typed_core_traced(&root().join(file), &[], false, &mut report).unwrap_or_else(|s| panic!("{file}: {s:?}"));
        let mut ad = core.clone();
        let (_, ad_links) = presk_driver::ad::differentiate_traced(&mut ad, presk_core::ad::Remat::Store, true);
        let mut opt = ad.clone();
        let (_, opt_links, opt_ends) = presk_opt::optimize_traced(&mut opt, &prog, &presk_opt::Options::default());
        for f in &core.funcs {
            if !matches!(f.origin, presk_core::ir::Origin::Source(_)) {
                continue;
            }
            for with_opt in [false, true] {
                let t = presk_driver::provenance::table(&presk_driver::provenance::Stages {
                    sm: &sm,
                    prog: &prog,
                    typed: &typed,
                    core: &core,
                    lowered: &lowered,
                    entry: &f.name,
                    ad: &ad,
                    ad_links: &ad_links,
                    precision: None,
                    opt: with_opt.then_some((&opt, &opt_links[..], &opt_ends[..])),
                    layouts: None,
                    kernels: None,
                    c: None,
                    native: None,
                });
                bad.extend(complete(&format!("{file}:{}", f.name), &t));
            }
        }
    }
    assert!(bad.is_empty(), "{} defects:\n{}", bad.len(), bad.iter().take(40).cloned().collect::<Vec<_>>().join("\n"));
}

/// The precision stages, which read a program differentiated with every
/// fusion group kept, while the kernels' program keeps them only for a
/// realization that can see them: a gradient under a uniform realization,
/// and under a budget; the tutorial's `loss` under a budget.
#[test]
fn the_precision_stages_link_to_the_differentiated_program() {
    let mut bad = Vec::new();
    for (file, entry, flags, how, opt) in [
        ("tests/examples/mlp/mlp.psk", "mse_grad", "--dim D=4 --dim B=2", (None, Some("f32")), false),
        ("tests/examples/mlp/mlp.psk", "mse_grad", "--dim D=4 --dim B=2", (None, Some("f32")), true),
        ("tests/examples/mlp/mlp.psk", "mse", "--dim D=4 --dim B=2", (Some(1e-3), None), false),
        ("tutorial/07_train.psk", "loss", "--dim B=4", (Some(1e-3), None), false),
    ] {
        let name = format!("{file}:{entry}");
        let t = table_as(&format!("{}-exact", name.replace(['/', ':'], "_")), file, entry, flags, opt, false, how);
        eprintln!("{name}{}: {}", if opt { " --opt" } else { "" }, counts(&t));
        assert!(!t.elements(Stage::Precision).is_empty(), "{name}: no precision element");
        bad.extend(complete(&name, &t));
    }
    assert!(bad.is_empty(), "{} defects:\n{}", bad.len(), bad.iter().take(40).cloned().collect::<Vec<_>>().join("\n"));
}
