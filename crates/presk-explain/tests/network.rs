//! Statements N1 and N2 of plan 0.14.0 (D141): the network diagram is a view
//! of the provenance table, and its numbers are exact.
//!
//! On the entries `make plan-check` runs, the coverage models at their tiny
//! constants, the six snapshot documents' and the course GPT's `loss` at
//! D = 64, with and without the optimizer:
//! - N1: every node and every arrow names an element of the table;
//! - N2: the parameters over the entry's `θ` are `Hosted.params`' total; at
//!   every site an envelope holds, the largest count of elements over the
//!   arrows that carry it is the envelope's; and in every view, the work and
//!   the bytes of the nodes drawn and not opened sum to the entry's launches'.
//!
//! Every view of every diagram is written as an SVG under the test's
//! directory, `network/<name>/view-<k>.svg`, to be looked at.

use std::collections::BTreeMap;
use std::path::PathBuf;

use presk_diagram::{Kind, Network};
use presk_driver::Collect;
use presk_explain::{Options, Target};
use presk_provenance::{Id, Stage, Table};

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

fn out(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("network").join(name)
}

/// The diagram of `file` at `entry`, as `presk explain` draws it.
fn diagram(name: &str, file: &str, entry: &str, flags: &str, opt: bool, (budget, realize): (Option<f64>, Option<&str>), no_exact: bool) -> presk_explain::Network {
    let (dims, consts) = args(flags);
    let o = Options {
        file: root().join(file),
        entry: entry.into(),
        dims,
        consts,
        budget,
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
        targets: vec![Target::Native],
        measure: false,
        sites: Vec::new(),
        json: true,
        html: false,
        out: out(name).join("explain"),
    };
    let mut report = Collect::default();
    presk_explain::network(&o, &mut report).unwrap_or_else(|s| panic!("{name}: stopped with {s:?}: {:?}", report.items))
}

/// N1: every element the diagram names is in the table.
fn in_table(name: &str, n: &Network, t: &Table) -> Vec<String> {
    let mut bad = Vec::new();
    let mut want = |what: String, id: &Id| {
        if !t.has(id) {
            bad.push(format!("{name}: {what} names {id}, which the table does not have"));
        }
    };
    for x in &n.nodes {
        want(format!("node {} ({})", x.id, x.label), &x.element);
        if let Some(f) = &x.function {
            want(format!("node {}'s function", x.id), f);
        }
        for m in &x.members {
            want(format!("node {}'s member", x.id), m);
        }
    }
    for e in &n.edges {
        want(format!("edge {}", e.id), &e.site);
    }
    bad
}

/// N2: the numbers.
fn exact(name: &str, d: &presk_explain::Network, n: &Network) -> Vec<String> {
    let mut bad = Vec::new();
    // The parameters.
    if let Some(ps) = &d.params
        && !ps.is_empty()
    {
        let total: u64 = ps.iter().map(|(_, e)| *e as u64).sum();
        match &n.params {
            Some(p) if p.count == total => {}
            p => bad.push(format!("{name}: the diagram's parameters {:?}, Hosted.params' {total}", p.as_ref().map(|p| p.count))),
        }
    }
    // Every arrow's shape is known at the entry's dimensions.
    for e in &n.edges {
        if matches!(e.dtype.as_str(), "f64" | "f32" | "bf16" | "idx" | "bool") && e.bytes == 0 {
            bad.push(format!("{name}: edge {} ({}) has no shape at the entry's dimensions: {}", e.id, e.site, e.ty));
        }
    }
    // The elements against the envelope's.
    if let Some(env) = &d.elems {
        let mut largest: BTreeMap<String, u64> = BTreeMap::new();
        for e in &n.edges {
            if matches!(e.dtype.as_str(), "f64" | "f32" | "bf16") {
                let m = largest.entry(e.site.key.clone()).or_insert(0);
                *m = (*m).max(e.elems);
            }
        }
        for (site, m) in &largest {
            if let Some(want) = env.get(site)
                && want != m
            {
                bad.push(format!("{name}: {site} carries {m} elements at most, the envelope {want}"));
            }
        }
    }
    // W and Q over every view.
    if let (Some(w), Some(q)) = (n.work, n.bytes) {
        for (k, v) in n.views.iter().enumerate() {
            let drawn = v.rects.keys().filter(|i| **i != 0 && !v.open.contains(i)).map(|i| &n.nodes[*i]).filter(|x| !matches!(x.kind, Kind::Input | Kind::Output | Kind::Param));
            let (sw, sq) = drawn.fold((0.0, 0.0), |(a, b), x| (a + x.work.unwrap_or(0.0), b + x.bytes.unwrap_or(0.0)));
            if sw != w || sq != q {
                bad.push(format!("{name}: view {k} (open {:?}) sums W {sw}, Q {sq}; the launches {w}, {q}", v.open));
            }
        }
    }
    bad
}

fn views(name: &str, n: &Network) {
    let dir = out(name);
    std::fs::create_dir_all(&dir).unwrap();
    for (k, v) in n.views.iter().enumerate() {
        std::fs::write(dir.join(format!("view-{k}.svg")), presk_diagram::svg::svg(n, v, false)).unwrap();
    }
}

fn check(name: &str, d: &presk_explain::Network) -> Vec<String> {
    let n = d.network.as_ref().unwrap_or_else(|| panic!("{name}: no diagram"));
    views(name, n);
    eprintln!("{name}: {} nodes, {} edges, {} views", n.nodes.len(), n.edges.len(), n.views.len());
    let mut bad = in_table(name, n, &d.table);
    bad.extend(exact(name, d, n));
    bad
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

fn slug(s: &str) -> String {
    s.replace(['/', ':', '{', '}', ',', '.'], "_")
}

/// The course GPT's `loss` at D = 64, in `f64` without a budget: its
/// outline, every box's graph and the default opening's positions.
#[test]
fn the_course_gpt() {
    let d = diagram("gpt", "corpus/nlp/05_gpt.psk", "loss", "--const D=64 --const H=2 --const L=2 --dim B=2 --dim T=16", false, (None, Some("f64")), true);
    let bad = check("gpt", &d);
    assert!(bad.is_empty(), "{}", bad.join("\n"));
    insta::assert_snapshot!("network-gpt", presk_diagram::text::outline(d.network.as_ref().unwrap()));
}

#[test]
fn the_diagram_is_a_view_of_the_table_and_its_numbers_are_exact() {
    let mut bad = Vec::new();
    for (file, entry, flags) in entries() {
        for opt in [false, true] {
            let name = format!("{}{}", slug(&format!("{file}:{entry}")), if opt { "-opt" } else { "" });
            let d = diagram(&name, &file, &entry, &flags, opt, (None, Some("f32")), true);
            assert!(!d.table.elements(Stage::Kernel).is_empty(), "{name}: no kernel");
            bad.extend(check(&name, &d));
        }
    }
    assert!(bad.is_empty(), "{} defects:\n{}", bad.len(), bad.iter().take(40).cloned().collect::<Vec<_>>().join("\n"));
}

/// The six snapshot documents' compilations, and the course GPT under a
/// budget, with the exact reference's envelope: N2's elements.
#[test]
fn the_snapshot_documents_and_the_envelopes() {
    let mut bad = Vec::new();
    let mut compared = 0;
    for (name, file, entry, flags, how, opt) in [
        ("mlp", "tests/examples/mlp/mlp.psk", "mse", "--dim D=4 --dim B=2", (Some(1e-3), None), false),
        ("layernorm", "tests/examples/layernorm/layernorm.psk", "LayerNorm{8}.apply", "--dim batch=3", (None, Some("f32")), false),
        ("minigpt", "tests/examples/minigpt/minigpt.psk", "MiniGPT.apply", "--dim B=2 --dim T=8 --const V=64 --const D=16 --const H=2 --const L=2", (None, Some("f32")), false),
        ("tutorial", "tutorial/07_train.psk", "loss", "--dim B=4", (Some(1e-3), None), false),
        ("tutorial-typed", "tutorial/07_train.psk", "loss", "--dim B=4", (None, Some("@../../tests/plan/tutorial-typed.json")), false),
        ("tutorial-opt", "tutorial/07_train.psk", "loss", "--dim B=4", (None, Some("f32")), true),
        ("mlp-grad", "tests/examples/mlp/mlp.psk", "mse_grad", "--dim D=4 --dim B=2", (None, Some("f32")), false),
    ] {
        let d = diagram(name, file, entry, flags, opt, how, false);
        if how.0.is_some() {
            assert!(d.elems.is_some(), "{name}: no envelope of the kernels' program");
        }
        if let (Some(env), Some(n)) = (&d.elems, &d.network) {
            compared += n.edges.iter().filter(|e| env.contains_key(&e.site.key)).count();
        }
        bad.extend(check(name, &d));
        insta::assert_snapshot!(format!("network-{name}"), presk_diagram::text::outline(d.network.as_ref().unwrap()));
    }
    eprintln!("{compared} arrows compared with an envelope");
    assert!(compared > 50, "only {compared} arrows compared with an envelope");
    assert!(bad.is_empty(), "{} defects:\n{}", bad.len(), bad.iter().take(40).cloned().collect::<Vec<_>>().join("\n"));
}

/// The course GPT as `presk dump --stage network` draws it (no profile, the
/// native target, no exact reference, `f64`).
fn as_dumped() -> presk_diagram::Network {
    let (dims, consts) = args("--const D=64 --const H=2 --const L=2 --dim B=2 --dim T=16");
    let o = Options {
        file: root().join("corpus/nlp/05_gpt.psk"),
        entry: "loss".into(),
        dims,
        consts,
        budget: None,
        realize: Some("f64".into()),
        plan: None,
        profile: None,
        threads: None,
        remat: presk_core::ad::Remat::Store,
        opt: false,
        weights: None,
        seed: 0,
        bits: 128,
        envelope: None,
        no_exact: true,
        targets: vec![Target::Native],
        measure: false,
        sites: Vec::new(),
        json: true,
        html: false,
        out: out("dumped").join("explain"),
    };
    let mut report = Collect::default();
    presk_explain::network(&o, &mut report).unwrap_or_else(|s| panic!("stopped with {s:?}: {:?}", report.items)).network.expect("a diagram")
}

/// Statement N4 (plan 0.14.0 step 3, D143): the SVG and the TikZ draw the
/// positions of the JSON's views, every rectangle and every arrow's ends;
/// and the report's two figures of the course GPT, level 1 and `Block`
/// alone (the user's decision of 2026-10-17), each fit an A4 page with
/// margins of 2 cm at two thirds of their size or more, their 9 pt labels
/// printing at 6 pt or more.
#[test]
fn the_svg_and_the_tikz_draw_the_json_positions() {
    let n = as_dumped();
    let mut bad = Vec::new();
    let block = presk_diagram::layout::alone(&n, presk_diagram::layout::find_box(&n, "blocks").expect("the stack"));
    for (k, v) in [(n.default, &n.views[n.default]), (presk_diagram::layout::level(&n, 2), &n.views[presk_diagram::layout::level(&n, 2)]), (usize::MAX, &block)] {
        let svg = presk_diagram::svg::svg(&n, v, true);
        let tikz = presk_diagram::tikz::tikz(&n, v, false, "the test");
        for (id, r) in &v.rects {
            let x = &n.nodes[*id];
            if *id == 0 || (k == usize::MAX && v.open.first() == Some(id)) {
                continue;
            }
            let s = format!("x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\"", r.x, r.y, r.w, r.h);
            if !svg.contains(&s) {
                bad.push(format!("view {k}: node {id} ({}) not at {s} in the SVG", x.label));
            }
            let t = format!("({:.1},{:.1}) rectangle ({:.1},{:.1})", r.x, r.y, r.x + r.w, r.y + r.h);
            if !x.key && !tikz.contains(&t) {
                bad.push(format!("view {k}: node {id} ({}) not at {t} in the TikZ", x.label));
            }
        }
        for (id, pts) in &v.routes {
            let (a, z) = (pts[0], pts[pts.len() - 1]);
            if !svg.contains(&format!("M{:.1},{:.1}", a.0, a.1)) || !svg.contains(&format!("{:.1},{:.1}\" fill", z.0, z.1)) {
                bad.push(format!("view {k}: edge {id}'s route not in the SVG"));
            }
            let e = &n.edges[*id];
            if !e.key && !n.nodes[e.from].key && !n.nodes[e.to].key && !(tikz.contains(&format!("({:.1},{:.1}) ..", a.0, a.1)) && tikz.contains(&format!(".. ({:.1},{:.1});", z.0, z.1))) {
                bad.push(format!("view {k}: edge {id}'s route not in the TikZ"));
            }
        }
    }
    assert!(bad.is_empty(), "{} defects:\n{}", bad.len(), bad.iter().take(20).cloned().collect::<Vec<_>>().join("\n"));
    // A4, 595 × 842 pt, less 2 cm (57 pt) a side.
    for (what, v) in [("level 1", &n.views[presk_diagram::layout::level(&n, 1)]), ("Block alone", &block)] {
        let scale = ((595.0 - 114.0) / v.width).min((842.0 - 114.0) / v.height);
        eprintln!("the course GPT, {what}: {:.0} × {:.0} pt, on an A4 page at {scale:.2}", v.width, v.height);
        assert!(scale >= 0.66, "{what} is {:.0} × {:.0} pt: an A4 page holds it at {scale:.2} only", v.width, v.height);
    }
}

/// The manual's chapter `explain_network`: the course GPT at level 2 as
/// `make network-figure` writes it, held current (`PRESK_UPDATE_MANUAL=1`
/// writes it again).
#[test]
fn the_manual_holds_the_course_gpt_s_network() {
    let n = as_dumped();
    let v = &n.views[presk_diagram::layout::level(&n, 2)];
    let path = root().join("crates/presk-manual/manual/explain_network.md");
    let head = "# The course GPT's network\n\nThe network diagram of `corpus/nlp/05_gpt.psk`'s `loss` at `D = 64`, `H = 2`, `L = 2`, `B = 2`, `T = 16`, in `f64`, at level 2: `GPT.apply` opened and its stack of blocks opened to one layer, as the page's Network tab draws it (plan 0.14.0 step 3). Generated by `make network-figure`, that is\n\n```text\npresk dump corpus/nlp/05_gpt.psk --stage network --entry loss --const D=64 --const H=2 --const L=2 --dim B=2 --dim T=16 --level 2 --format svg\n```\n\nand held current by a test of `presk-explain`. Each box carries its function's formula in the printer's Unicode form, its parameters and the cost model's W and Q, a layer's inside the stack; each arrow is coloured by its realized type, its width by its bytes; the random keys are left out. The report's figure (presk-specs `report/figures/network-gpt.tex`) is the same layout in TikZ.\n\n";
    let want = format!("{head}{}", presk_diagram::svg::svg(&n, v, false));
    if std::env::var("PRESK_UPDATE_MANUAL").is_ok() {
        std::fs::write(&path, &want).unwrap();
    }
    let got = std::fs::read_to_string(&path).unwrap_or_default();
    assert!(got == want, "the manual's explain_network chapter is stale: make network-figure");
}
