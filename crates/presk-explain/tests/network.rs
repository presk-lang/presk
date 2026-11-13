//! Statements N1 and N2 of plan 0.14.0 (D141): the network diagram is a view
//! of the provenance table, and its numbers are exact; and E1 of plan 0.14.1
//! (D146): its openings compose.
//!
//! On the entries `make plan-check` runs, the coverage models at their tiny
//! constants, the six snapshot documents' and the course GPT's `loss` at
//! D = 64, with and without the optimizer:
//! - N1: every node and every arrow names an element of the table;
//! - N2: the parameters over the entry's `θ` are `Hosted.params`' total; at
//!   every site an envelope holds, the largest count of elements over the
//!   arrows that carry it is the envelope's; and in every opening, the work
//!   and the bytes of the nodes drawn and not opened sum to the entry's
//!   launches';
//! - E1: in every opening, a box with no child open is drawn as its frame,
//!   translated; an open box's children keep their frame's layers and their
//!   order in each layer; no two siblings overlap; every child lies inside
//!   its box; every arrow into or out of an open box meets it at its port;
//!   no arrow passes through a node of its box.
//!
//! The openings: each box opened with the boxes it is inside (0.14.0's
//! views), the levels, the exploded one, and 20 sets of boxes drawn at
//! random from a fixed seed. Each chain's opening and the exploded one are
//! written as SVGs under the test's directory, `network/<name>/`, to be
//! looked at.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use presk_diagram::layout::{View, alone, everything, level, view};
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
    // W and Q over every opening.
    if let (Some(w), Some(q)) = (n.work, n.bytes) {
        for (k, v) in openings(n).iter().enumerate() {
            let drawn = v.rects.keys().filter(|i| **i != 0 && !v.open.contains(i)).map(|i| &n.nodes[*i]).filter(|x| !matches!(x.kind, Kind::Input | Kind::Output | Kind::Param));
            let (sw, sq) = drawn.fold((0.0, 0.0), |(a, b), x| (a + x.work.unwrap_or(0.0), b + x.bytes.unwrap_or(0.0)));
            if sw != w || sq != q {
                bad.push(format!("{name}: opening {k} (open {:?}) sums W {sw}, Q {sq}; the launches {w}, {q}", v.open));
            }
        }
    }
    bad
}

/// The openings checked: each box with the boxes it is inside, the levels,
/// the exploded one, and 20 sets drawn at random (a fixed seed).
fn openings(n: &Network) -> Vec<View> {
    let boxes = everything(n);
    let mut sets: Vec<Vec<usize>> = vec![Vec::new(), n.default.clone()];
    for b in &boxes {
        let mut chain = vec![*b];
        let mut p = n.nodes[*b].parent;
        while let Some(x) = p.filter(|x| *x != 0) {
            chain.push(x);
            p = n.nodes[x].parent;
        }
        chain.reverse();
        sets.push(chain);
    }
    sets.extend((0..6).map(|k| level(n, k)));
    sets.push(boxes.clone());
    let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
    for _ in 0..20 {
        let mut set = Vec::new();
        for b in &boxes {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            if seed % 2 == 0 {
                set.push(*b);
            }
        }
        sets.push(set);
    }
    sets.iter().map(|s| view(n, s)).collect()
}

/// E1: the openings compose.
fn composes(name: &str, n: &Network) -> Vec<String> {
    let mut bad = Vec::new();
    let near = |a: f64, b: f64| (a - b).abs() <= 0.051;
    for (k, v) in openings(n).iter().enumerate() {
        let shown: BTreeSet<usize> = std::iter::once(0).chain(v.open.iter().copied()).collect();
        for b in &shown {
            let r = v.rects[b];
            let kids: Vec<usize> = n.nodes[*b].children.iter().copied().filter(|c| v.rects.contains_key(c)).collect();
            let f = &n.frames[b];
            // Every child inside its box; no two siblings overlapping.
            for c in &kids {
                let q = v.rects[c];
                if q.x < r.x - 0.05 || q.y < r.y - 0.05 || q.x + q.w > r.x + r.w + 0.05 || q.y + q.h > r.y + r.h + 0.05 {
                    bad.push(format!("{name}: opening {k}: node {c} ({}) is not inside box {b}", n.nodes[*c].label));
                }
                for d in &kids {
                    let p = v.rects[d];
                    if c < d && q.x < p.x + p.w - 0.05 && p.x < q.x + q.w - 0.05 && q.y < p.y + p.h - 0.05 && p.y < q.y + q.h - 0.05 {
                        bad.push(format!("{name}: opening {k}: nodes {c} and {d} overlap in box {b}"));
                    }
                }
            }
            // The frame's layers, and its order in each layer.
            for c in &kids {
                for d in &kids {
                    let ((lc, fc), (ld, fd)) = (f.rects[c], f.rects[d]);
                    let (qc, qd) = (v.rects[c], v.rects[d]);
                    if lc < ld && qc.y >= qd.y {
                        bad.push(format!("{name}: opening {k}: node {c} is drawn below node {d}, a later layer"));
                    }
                    if lc == ld && fc.x < fd.x && qc.x >= qd.x {
                        bad.push(format!("{name}: opening {k}: nodes {c} and {d} are drawn out of their layer's order"));
                    }
                }
            }
            // A box with no child open: its frame, translated.
            if !kids.iter().any(|c| shown.contains(c)) {
                let a = alone(n, *b);
                for c in &kids {
                    let (q, p) = (v.rects[c], a.rects[c]);
                    if !(near(q.x - r.x, p.x) && near(q.y - r.y, p.y) && near(q.w, p.w) && near(q.h, p.h)) {
                        bad.push(format!("{name}: opening {k}: node {c} is not where box {b}'s frame has it"));
                    }
                }
                for (e, pts) in &a.routes {
                    let got = &v.routes[e];
                    if got.len() != pts.len() || got.iter().zip(pts).any(|(g, p)| !(near(g.0 - r.x, p.0) && near(g.1 - r.y, p.1))) {
                        bad.push(format!("{name}: opening {k}: arrow {e} is not where box {b}'s frame has it"));
                    }
                }
            }
        }
        // Every arrow into or out of an open box meets it at its port, and
        // none passes through a node of its box.
        for (id, pts) in &v.routes {
            let e = &n.edges[*id];
            // Its points, and its straight stretches down through a layer.
            for w in pts.windows(2) {
                let (p, r) = (w[0], w[1]);
                let (top, low) = if (p.0 - r.0).abs() < 0.05 { (p.1.min(r.1), p.1.max(r.1)) } else { (r.1, r.1) };
                for c in n.nodes[e.parent].children.iter().filter(|c| **c != e.from && **c != e.to) {
                    if let Some(q) = v.rects.get(c)
                        && r.0 > q.x + 0.05
                        && r.0 < q.x + q.w - 0.05
                        && low > q.y + 0.05
                        && top < q.y + q.h - 0.05
                    {
                        bad.push(format!("{name}: opening {k}: arrow {id} passes through node {c} ({})", n.nodes[*c].label));
                    }
                }
            }
            if v.open.contains(&e.to)
                && let Some(i) = n.nodes[e.to].children.iter().find(|c| n.nodes[**c].kind == Kind::Input && n.nodes[**c].index == e.arg)
            {
                let q = v.rects[i];
                let z = pts[pts.len() - 1];
                if !(near(z.0, q.x + q.w / 2.0) && near(z.1, q.y)) {
                    bad.push(format!("{name}: opening {k}: arrow {id} does not end at box {}'s port", e.to));
                }
            }
            if v.open.contains(&e.from)
                && let Some(o) = n.nodes[e.from].children.iter().find(|c| n.nodes[**c].kind == Kind::Output)
            {
                let q = v.rects[o];
                if !(near(pts[0].0, q.x + q.w / 2.0) && near(pts[0].1, q.y + q.h)) {
                    bad.push(format!("{name}: opening {k}: arrow {id} does not start at box {}'s result", e.from));
                }
            }
        }
    }
    bad
}

fn views(name: &str, n: &Network) {
    let dir = out(name);
    std::fs::create_dir_all(&dir).unwrap();
    let boxes = everything(n);
    for b in &boxes {
        let mut chain = vec![*b];
        let mut p = n.nodes[*b].parent;
        while let Some(x) = p.filter(|x| *x != 0) {
            chain.push(x);
            p = n.nodes[x].parent;
        }
        chain.reverse();
        std::fs::write(dir.join(format!("open-{b}.svg")), presk_diagram::svg::svg(n, &view(n, &chain), false)).unwrap();
    }
    std::fs::write(dir.join("exploded.svg"), presk_diagram::svg::svg(n, &view(n, &boxes), false)).unwrap();
}

fn check(name: &str, d: &presk_explain::Network) -> Vec<String> {
    let n = d.network.as_ref().unwrap_or_else(|| panic!("{name}: no diagram"));
    views(name, n);
    eprintln!("{name}: {} nodes, {} edges, {} boxes", n.nodes.len(), n.edges.len(), n.frames.len() - 1);
    let mut bad = in_table(name, n, &d.table);
    bad.extend(exact(name, d, n));
    bad.extend(composes(name, n));
    bad
}

/// `make plan-check`'s entries, the coverage models at `make coverage`'s
/// tiny constants, and the course GPT at D = 64.
fn entries() -> Vec<(String, String, String)> {
    let mut out: Vec<(String, String, String)> = std::fs::read_to_string(root().join("examples/tests/plan/entries.txt"))
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let p: Vec<&str> = l.splitn(3, '|').collect();
            (p[0].to_string(), p[1].to_string(), p[2].to_string())
        })
        .collect();
    out.push(("examples/coverage/bert.psk".into(), "BertSquad.apply".into(), "--const V=100 --const D=32 --const H=2 --const L=2 --dim B=2 --dim T=16".into()));
    out.push(("examples/coverage/llama.psk".into(), "Llama.apply".into(), "--const V=100 --const D=32 --const H=2 --const G=2 --const FF=64 --const L=2 --dim B=2 --dim T=16".into()));
    out.push(("examples/corpus/nlp/05_gpt.psk".into(), "loss".into(), "--const D=64 --const H=2 --const L=2 --dim B=2 --dim T=16".into()));
    out
}

fn slug(s: &str) -> String {
    s.replace(['/', ':', '{', '}', ',', '.'], "_")
}

/// The course GPT's `loss` at D = 64, in `f64` without a budget: its
/// outline, every box's graph and the default opening's positions.
#[test]
fn the_course_gpt() {
    let d = diagram("gpt", "examples/corpus/nlp/05_gpt.psk", "loss", "--const D=64 --const H=2 --const L=2 --dim B=2 --dim T=16", false, (None, Some("f64")), true);
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
        ("mlp", "examples/spec/mlp/mlp.psk", "mse", "--dim D=4 --dim B=2", (Some(1e-3), None), false),
        ("layernorm", "examples/spec/layernorm/layernorm.psk", "LayerNorm{8}.apply", "--dim batch=3", (None, Some("f32")), false),
        ("minigpt", "examples/spec/minigpt/minigpt.psk", "MiniGPT.apply", "--dim B=2 --dim T=8 --const V=64 --const D=16 --const H=2 --const L=2", (None, Some("f32")), false),
        ("tutorial", "examples/tutorial/07_train.psk", "loss", "--dim B=4", (Some(1e-3), None), false),
        ("tutorial-typed", "examples/tutorial/07_train.psk", "loss", "--dim B=4", (None, Some("@../../examples/tests/plan/tutorial-typed.json")), false),
        ("tutorial-opt", "examples/tutorial/07_train.psk", "loss", "--dim B=4", (None, Some("f32")), true),
        ("mlp-grad", "examples/spec/mlp/mlp.psk", "mse_grad", "--dim D=4 --dim B=2", (None, Some("f32")), false),
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
        file: root().join("examples/corpus/nlp/05_gpt.psk"),
        entry: "loss".into(),
        dims,
        consts,
        budget: None,
        menu: Default::default(),
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

/// Statement N4 (plan 0.14.0 step 3, D143) and E3's second half (plan
/// 0.14.1, D146): the SVG and the TikZ draw the compiler's composition of
/// the default opening, level 2, the exploded one and `Block` alone, every
/// rectangle and every arrow's ends;
/// and the report's two figures of the course GPT, level 1 and `Block`
/// alone (the user's decision of 2026-10-17), each fit an A4 page with
/// margins of 2 cm at two thirds of their size or more, their 9 pt labels
/// printing at 6 pt or more.
#[test]
fn the_svg_and_the_tikz_draw_the_json_positions() {
    let n = as_dumped();
    let mut bad = Vec::new();
    let block = alone(&n, presk_diagram::layout::find_box(&n, "blocks").expect("the stack"));
    let (first, second, all) = (view(&n, &n.default), view(&n, &level(&n, 2)), view(&n, &everything(&n)));
    for (k, v) in [(1, &first), (2, &second), (3, &all), (usize::MAX, &block)] {
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
    for (what, v) in [("level 1", &view(&n, &level(&n, 1))), ("Block alone", &block)] {
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
    let v = &view(&n, &level(&n, 2));
    let path = root().join("crates/presk-manual/manual/explain_network.md");
    let head = "# The course GPT's network\n\nThe network diagram of `examples/corpus/nlp/05_gpt.psk`'s `loss` at `D = 64`, `H = 2`, `L = 2`, `B = 2`, `T = 16`, in `f64`, at level 2: `GPT.apply` opened and its stack of blocks opened to one layer, as the page's Network tab draws it, composed from each box's frame (plan 0.14.0 step 3, plan 0.14.1). Generated by `make network-figure`, that is\n\n```text\npresk dump examples/corpus/nlp/05_gpt.psk --stage network --entry loss --const D=64 --const H=2 --const L=2 --dim B=2 --dim T=16 --level 2 --format svg\n```\n\nand held current by a test of `presk-explain`. Each box carries its function's formula in the printer's Unicode form, its parameters and the cost model's W and Q, a layer's inside the stack; each arrow is coloured by its realized type, its width by its bytes; the random keys are left out. The report's two figures (presk-specs `report/figures/network-gpt.tex` and `network-gpt-block.tex`) are level 1 and `Block` alone, by the same layout in TikZ; the page opens any set of boxes, and `--level all` every one.\n\n";
    let want = format!("{head}{}", presk_diagram::svg::svg(&n, v, false));
    if std::env::var("PRESK_UPDATE_MANUAL").is_ok() {
        std::fs::write(&path, &want).unwrap();
    }
    let got = std::fs::read_to_string(&path).unwrap_or_default();
    assert!(got == want, "the manual's explain_network chapter is stale: make network-figure");
}

/// The memory overlay's shares (plan 0.15.0 step 4, D151): every box's in
/// [0, 1], an open box's at least the sum of its boxes', the box holding the
/// peak's operation at most 1; `--memory` draws them into the SVG and the
/// TikZ, each box's on its header, and leaves the plain drawing as it was.
#[test]
fn the_memory_overlay_colours_the_boxes_by_their_share() {
    let mut d = diagram("gpt-memory", "examples/corpus/nlp/05_gpt.psk", "loss", "--const D=64 --const H=2 --const L=2 --dim B=2 --dim T=16", false, (None, Some("f64")), true);
    let n = d.network.as_mut().unwrap();
    assert!(!d.memory.is_empty(), "no shares");
    for (id, f) in &d.memory {
        assert!((0.0..=1.0).contains(f), "box {id}: a share of {f}");
        let kids: f64 = n.nodes[*id].children.iter().filter_map(|c| d.memory.get(c)).sum();
        assert!(kids <= f + 1e-9, "box {id}: its boxes' shares {kids} above its own {f}");
    }
    // The feed-forward of the course GPT's blocks holds the peak at T = 16.
    let ff = n.nodes.iter().find(|x| x.kind.is_box() && x.label == "ff").map(|x| x.id).expect("a box ff");
    assert!(d.memory[&ff] > 0.8, "ff's share {}", d.memory[&ff]);
    let v = presk_diagram::layout::view(n, &presk_diagram::layout::everything(n));
    let plain = presk_diagram::svg::svg(n, &v, false);
    assert!(!plain.contains(" %</text>"), "the plain drawing has shares");
    n.memory = d.memory.clone();
    let svg = presk_diagram::svg::svg(n, &v, false);
    let shown = svg.matches(" %</text>").count();
    assert_eq!(shown, d.memory.keys().filter(|i| v.rects.contains_key(i) && n.nodes[**i].kind != presk_diagram::Kind::Root).count(), "a share on every drawn box's header");
    let tikz = presk_diagram::tikz::tikz(n, &v, false, "test");
    assert!(tikz.contains("\\definecolor{pkmem}") && tikz.contains("pkmem!"), "the TikZ has the memory's colour");
}
