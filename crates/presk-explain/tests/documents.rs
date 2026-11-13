//! The documents of `presk explain` on the examples of `examples/spec/`
//! and the tutorial's `07_train.psk`, as snapshots (plan 0.11.0 step 3): a
//! change to any stage shows in them as a diff. The profile is the frozen
//! X925 model's (`ra4-model-0.8.0`, D71) without its calibration samples,
//! which only a refit reads (`tests/data/README.md`).

use std::path::PathBuf;

use presk_driver::Collect;
use presk_explain::{Options, Target, explain};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The document of `file` at `entry`, its paths as the test names them.
fn document(name: &str, file: &str, entry: &str, dims: &[&str], consts: &[&str], how: (Option<f64>, Option<&str>), targets: &[Target]) -> String {
    document_as(name, file, entry, dims, consts, how, targets, false)
}

/// [`document`], under `--opt` when `opt`, which takes no profile.
#[allow(clippy::too_many_arguments)]
fn document_as(name: &str, file: &str, entry: &str, dims: &[&str], consts: &[&str], how: (Option<f64>, Option<&str>), targets: &[Target], opt: bool) -> String {
    written(name, file, entry, dims, consts, how, targets, opt, false).0
}

/// The document and, with `json`, `explain.json`, written under `name`.
#[allow(clippy::too_many_arguments)]
fn written(name: &str, file: &str, entry: &str, dims: &[&str], consts: &[&str], how: (Option<f64>, Option<&str>), targets: &[Target], opt: bool, json: bool) -> (String, Option<String>) {
    let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("explain").join(name);
    let o = Options {
        file: PathBuf::from("../..").join(file),
        entry: entry.into(),
        dims: dims.iter().map(|d| d.to_string()).collect(),
        consts: consts.iter().map(|c| c.to_string()).collect(),
        budget: how.0,
        menu: Default::default(),
        realize: how.1.map(String::from),
        plan: None,
        profile: (!opt).then(|| PathBuf::from("tests/data/x925-frozen.json")),
        threads: Some(10),
        remat: presk_core::ad::Remat::Store,
        opt,
        weights: None,
        seed: 0,
        bits: 128,
        envelope: None,
        no_exact: false,
        targets: targets.to_vec(),
        measure: false,
        sites: Vec::new(),
        json,
        html: false,
        out: out.clone(),
    };
    assert!(root().join(file).exists());
    let mut report = Collect::default();
    let e = explain(&o, &mut report).unwrap_or_else(|s| panic!("{name}: stopped with {s:?}: {:?}", report.items));
    (std::fs::read_to_string(e.document).unwrap(), std::fs::read_to_string(out.join("explain.json")).ok())
}

#[test]
fn the_mlp_under_a_budget() {
    insta::assert_snapshot!("mlp", document("mlp", "examples/spec/mlp/mlp.psk", "mse", &["D=4", "B=2"], &[], (Some(1e-3), None), &[Target::Native]));
}

#[test]
fn the_layer_norm_of_a_module_entry() {
    insta::assert_snapshot!("layernorm", document("layernorm", "examples/spec/layernorm/layernorm.psk", "LayerNorm{8}.apply", &["batch=3"], &[], (None, Some("f32")), &[Target::Native]));
}

#[test]
fn the_minigpt_in_f32() {
    insta::assert_snapshot!("minigpt", document("minigpt", "examples/spec/minigpt/minigpt.psk", "MiniGPT.apply", &["B=2", "T=8"], &["V=64", "D=16", "H=2", "L=2"], (None, Some("f32")), &[Target::Native]));
}

#[test]
fn the_tutorial_loss_under_a_budget_to_every_target() {
    insta::assert_snapshot!("tutorial", document("tutorial", "examples/tutorial/07_train.psk", "loss", &["B=4"], &[], (Some(1e-3), None), &[Target::Native, Target::C, Target::Pytorch]));
}

/// A realization given, one nest typed apart and two parameters stored in
/// their own types (plan 0.12.0, D121): the realization's parameters in §6.3
/// and the summary, the `bf16` weight a kernel widens as it loads it, and
/// its own type in the C program's arena.
#[test]
fn the_tutorial_loss_under_typed_parameters() {
    insta::assert_snapshot!("tutorial-typed", document("tutorial-typed", "examples/tutorial/07_train.psk", "loss", &["B=4"], &[], (None, Some("@../../examples/tests/plan/tutorial-typed.json")), &[Target::Native, Target::C]));
}

/// Under the optimizer (plan 0.13.0 step 1, D134), which writes each
/// function it optimizes back with new numbers: §12 follows each site the
/// earlier sections name to the `let` the optimizer made of it, by the
/// provenance table. 0.12.0 looked for the same number in the rewritten
/// program, and named no fusion group and no kernel (D133's finding 2).
#[test]
fn the_tutorial_loss_under_the_optimizer() {
    insta::assert_snapshot!("tutorial-opt", document_as("tutorial-opt", "examples/tutorial/07_train.psk", "loss", &["B=4"], &[], (None, Some("f32")), &[Target::Native, Target::C], true));
}

/// The document again from `explain.json`'s values.
fn from_json(v: &serde_json::Value) -> presk_explain::doc::Document {
    use presk_explain::doc::{Block, Document, Section};
    let strs = |v: &serde_json::Value| -> Vec<String> { v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect() };
    let s = |v: &serde_json::Value| v.as_str().unwrap().to_string();
    fn block(b: &serde_json::Value) -> Block {
        let s = |v: &serde_json::Value| v.as_str().unwrap().to_string();
        if let Some(t) = b.get("prose") {
            Block::Prose(s(t))
        } else if let Some(t) = b.get("table") {
            Block::Table { head: t["head"].as_array().unwrap().iter().map(s).collect(), rows: t["rows"].as_array().unwrap().iter().map(|r| r.as_array().unwrap().iter().map(s).collect()).collect() }
        } else if let Some(c) = b.get("code") {
            Block::Code { lang: s(&c["lang"]), text: s(&c["text"]), name: c["name"].as_str().map(String::from) }
        } else if let Some(n) = b.get("note") {
            Block::Note { title: s(&n["title"]), text: s(&n["text"]) }
        } else {
            let a = &b["attach"];
            Block::Attach { name: s(&a["name"]), text: s(&a["text"]) }
        }
    }
    fn section(v: &serde_json::Value) -> Section {
        let s = |v: &serde_json::Value| v.as_str().unwrap().to_string();
        Section {
            number: s(&v["number"]),
            title: s(&v["title"]),
            purpose: v["purpose"].as_array().unwrap().iter().map(s).collect(),
            blocks: v["blocks"].as_array().unwrap().iter().map(block).collect(),
            commands: v["commands"].as_array().unwrap().iter().map(s).collect(),
            sections: v["sections"].as_array().unwrap().iter().map(section).collect(),
        }
    }
    let d = &v["document"];
    let _ = strs;
    Document { title: s(&d["title"]), subtitle: s(&d["subtitle"]), header: d["header"].as_array().unwrap().iter().map(block).collect(), sections: d["sections"].as_array().unwrap().iter().map(section).collect() }
}

/// `--format json` (plan 0.13.0 step 2, D135): the JSON holds the whole
/// document, whose Markdown it renders again byte for byte, with the
/// provenance table; it is the same from one run to the next.
#[test]
fn the_json_holds_the_document_and_the_table() {
    for (name, how, opt) in [("tutorial-json", (Some(1e-3), None), false), ("tutorial-opt-json", (None, Some("f32")), true)] {
        let (md, json) = written(name, "examples/tutorial/07_train.psk", "loss", &["B=4"], &[], how, &[Target::Native, Target::C], opt, true);
        let json = json.expect("explain.json written");
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["format"], 4);
        // The network diagram, since format 2 (plan 0.14.0, D141), its
        // frames since format 3 (plan 0.14.1, D146).
        assert!(v["network"]["nodes"].as_array().is_some_and(|n| !n.is_empty()), "{name}: no network");
        assert!(v["network"]["frames"].as_array().is_some_and(|f| !f.is_empty()), "{name}: no frames");
        assert_eq!(v["network"]["views"].as_array().map(|v| v.len()), Some(2), "{name}: the default and the exploded openings");
        // The memory, since format 4 (plan 0.15.0 step 1, D148): the live
        // bytes at every operation, the peak among them, and the boxes'
        // shares, each call that the diagram draws naming its box.
        let m = &v["memory"];
        let live: Vec<u64> = m["live"].as_array().expect("memory.live").iter().map(|x| x.as_u64().unwrap()).collect();
        assert_eq!(live.iter().max().copied(), m["peak"].as_u64(), "{name}: the peak is the largest live set");
        assert_eq!(live[m["at"].as_u64().unwrap() as usize], m["peak"].as_u64().unwrap(), "{name}: the peak's operation");
        let calls = m["calls"].as_array().unwrap();
        let top: u64 = calls.iter().filter(|c| c["path"].as_array().unwrap().len() == 1).map(|c| c["share"].as_u64().unwrap()).sum();
        assert_eq!(top + m["own"].as_u64().unwrap(), m["peak"].as_u64().unwrap(), "{name}: the shares sum to the peak");
        assert!(calls.iter().any(|c| !c["nodes"].as_array().unwrap().is_empty()), "{name}: no call names a box of the diagram");
        let again = presk_explain::doc::markdown(&from_json(&v)).0;
        assert!(again == md, "{name}: the Markdown rendered from the JSON differs from explain.md");
        let stages: Vec<&str> = v["stages"].as_array().unwrap().iter().map(|s| s["stage"].as_str().unwrap()).collect();
        for want in ["source", "elab", "core", "ad", "fusion", "kernel", "launch", "c"] {
            assert!(stages.contains(&want), "{name}: no stage {want} in {stages:?}");
        }
        assert_eq!(stages.contains(&"opt"), opt, "{name}: an optimized column only under --opt");
        // Every link names elements the JSON has.
        let mut have = std::collections::BTreeSet::new();
        for s in v["stages"].as_array().unwrap() {
            for e in s["elements"].as_array().unwrap() {
                have.insert(format!("{}:{}", s["stage"].as_str().unwrap(), e["key"].as_str().unwrap()));
            }
        }
        for l in v["links"].as_array().unwrap() {
            for x in std::iter::once(&l["child"]).chain(l["parents"].as_array().unwrap()) {
                assert!(have.contains(x.as_str().unwrap()), "{name}: a link names {x}, not an element");
            }
        }
        // The page (plan 0.13.0 step 3, D136) carries the same JSON, whole.
        let page = presk_explain::page(&v);
        assert!(!page.contains("/*PRESK_DATA*/"), "{name}: the page's data is not filled");
        let open = "<script type=\"application/json\" id=\"presk-data\">";
        let start = page.find(open).expect("the data element") + open.len();
        let end = start + page[start..].find("</script>").expect("its end");
        // The text, not the values: serde_json's float parser does not
        // round-trip every last bit without its `float_roundtrip` feature.
        assert!(page[start..end].replace("<\\/", "</") == serde_json::to_string(&v).unwrap(), "{name}: the page's JSON is not explain.json's");
        let _: serde_json::Value = serde_json::from_str(&page[start..end]).unwrap();
        let (_, again) = written(&format!("{name}-again"), "examples/tutorial/07_train.psk", "loss", &["B=4"], &[], how, &[Target::Native, Target::C], opt, true);
        assert!(again.as_deref() == Some(json.as_str()), "{name}: two runs write different JSON");
    }
}

/// The page's mathematics (plan 0.13.0 step 4c, D137): every function of
/// the elaborated program and every `let` of PSK-Core carries its formula,
/// a site the solver offered its term of the estimate, a derivative's link
/// its rule, and the page inlines KaTeX and fetches nothing.
#[test]
fn the_page_writes_the_mathematics() {
    for (name, file, entry, dims, how, ad) in [
        ("tutorial-math", "examples/tutorial/07_train.psk", "loss", &["B=4"][..], (Some(1e-3), None), false),
        ("mlp-grad-math", "examples/spec/mlp/mlp.psk", "mse_grad", &["D=4", "B=2"][..], (None, Some("f32")), true),
    ] {
        let (_, json) = written(name, file, entry, dims, &[], how, &[Target::Native], false, true);
        let v: serde_json::Value = serde_json::from_str(&json.expect("explain.json written")).unwrap();
        let stage = |s: &str| v["stages"].as_array().unwrap().iter().find(|x| x["stage"] == s).unwrap_or_else(|| panic!("{name}: no stage {s}")).clone();
        let tex = |e: &serde_json::Value| e["facts"]["tex"].as_str().map(String::from);
        for e in stage("elab")["elements"].as_array().unwrap() {
            let key = e["key"].as_str().unwrap();
            if key.starts_with("fn ") {
                assert!(tex(e).is_some_and(|t| t.starts_with("\\begin{aligned}")), "{name}: {key} has no display");
            }
        }
        for s in ["core", "ad"] {
            for e in stage(s)["elements"].as_array().unwrap() {
                let key = e["key"].as_str().unwrap();
                if key.contains("/%") {
                    assert!(tex(e).is_some_and(|t| t.contains(" = ") && !t.contains("\\texttt{")), "{name}: {s}:{key} has no equation");
                }
            }
        }
        if how.0.is_some() {
            let terms = stage("precision")["elements"].as_array().unwrap().iter().filter(|e| e["facts"]["term"].as_f64().is_some_and(|t| t > 0.0)).count();
            let written = stage("precision")["elements"].as_array().unwrap().iter().filter(|e| tex(e).is_some_and(|t| t.starts_with("e_v ="))).count();
            assert!(terms > 0 && written == terms, "{name}: {written} of {terms} terms written");
            assert!(v["precision"]["estimate"].as_f64().is_some() && v["precision"]["bound"].as_f64().is_some(), "{name}: no estimate or bound");
        }
        if ad {
            let rules = v["rules"]["adjoint"].as_object().unwrap();
            for l in v["links"].as_array().unwrap().iter().filter(|l| l["how"] == "adjoint") {
                let r = l["rule"].as_str().unwrap();
                assert!(rules.contains_key(r) == presk_print_math::core::adjoint_rule(r).is_some(), "{name}: the rule of {r}");
            }
            assert!(rules.contains_key("dot") && rules.contains_key("mean"), "{name}: {:?}", rules.keys().collect::<Vec<_>>());
        }
        let page = presk_explain::page(&v);
        assert!(!page.contains("/*KATEX_") && page.contains("katex.renderToString") && !page.contains("url(fonts/"), "{name}: KaTeX not inlined whole");
    }
}

/// The native column (plan 0.13.0 step 5, D138): under `--format json` with
/// the native target, one element per kernel of the listing, linked to its
/// kernel, whose ranges hold every byte of its machine code once, each with
/// a line of the kernel's text; without the native target, none.
#[test]
fn the_json_holds_the_machine_code() {
    for (name, targets) in [("tutorial-native", &[Target::Native, Target::C][..]), ("tutorial-no-native", &[Target::C][..])] {
        let (_, json) = written(name, "examples/tutorial/07_train.psk", "loss", &["B=4"], &[], (None, Some("f32")), targets, false, true);
        let v: serde_json::Value = serde_json::from_str(&json.expect("explain.json written")).unwrap();
        let stage = |s: &str| v["stages"].as_array().unwrap().iter().find(|x| x["stage"] == s).cloned();
        let kernels = stage("kernel").expect("kernels")["elements"].as_array().unwrap().clone();
        let Some(native) = stage("native") else {
            assert!(!targets.contains(&Target::Native), "{name}: no native column");
            continue;
        };
        assert!(targets.contains(&Target::Native), "{name}: a native column without the native target");
        let native = native["elements"].as_array().unwrap().clone();
        assert_eq!(native.len(), kernels.len(), "{name}: one native element per kernel");
        for (e, k) in native.iter().zip(&kernels) {
            let lines = k["facts"]["text"].as_str().unwrap().lines().count() as u64;
            let bytes = e["facts"]["bytes"].as_u64().unwrap();
            assert_eq!(e["facts"]["hex"].as_str().unwrap().len() as u64, 2 * bytes);
            let mut at = 0;
            for r in e["facts"]["ranges"].as_array().unwrap() {
                let (a, b, l) = (r[0].as_u64().unwrap(), r[1].as_u64().unwrap(), r[2].as_u64().unwrap());
                assert!(a == at && b > a && (1..=lines).contains(&l), "{name}: {} at {at}", e["key"]);
                at = b;
            }
            assert_eq!(at, bytes, "{name}: {}", e["key"]);
            let key = format!("native:{}", e["key"].as_str().unwrap());
            assert!(v["links"].as_array().unwrap().iter().any(|l| l["child"] == key.as_str() && l["how"] == "code" && l["parents"][0] == format!("kernel:{}", k["key"].as_str().unwrap()).as_str()), "{name}: {key} not linked to its kernel");
        }
    }
}
