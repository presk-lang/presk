//! The layered layout, top to bottom: a box's graph in layers by longest
//! path, its nodes ordered in each layer by the barycenters of their
//! neighbours over a few sweeps, then placed; a box opened is its graph
//! laid out first, inside it.
//!
//! Opening a box makes it larger and moves what is around it, so the layout
//! is computed once per opening a reader can reach: the entry alone, then
//! each box opened with the boxes it is inside, the others folded (a box
//! opened folds its siblings). The page, the SVG and the TikZ redraw these
//! positions; none lays anything out.
//!
//! No text is measured: a node's size comes from its strings' lengths in
//! characters (the Unicode printer's, for a formula). The page scales a
//! formula into its box.

use std::collections::{BTreeMap, HashMap};

use crate::{Edge, Kind, Network, Node};

/// A rectangle, in points: its top left corner, its width and its height.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

/// One opening, laid out.
#[derive(Clone, Debug)]
pub struct View {
    /// The boxes opened, outermost first; the entry's frame always is.
    pub open: Vec<usize>,
    pub width: f64,
    pub height: f64,
    /// Every node drawn, and its rectangle.
    pub rects: BTreeMap<usize, Rect>,
    /// Every edge drawn, and its route from its source to its target.
    pub routes: BTreeMap<usize, Vec<(f64, f64)>>,
}

/// The width of a character, the height of a line of text, in points.
pub const CHAR: f64 = 6.6;
pub const LINE: f64 = 15.0;
/// The gaps: between layers, between the nodes of a layer, inside a box.
const LAYER_GAP: f64 = 24.0;
const NODE_GAP: f64 = 12.0;
const PAD: f64 = 12.0;
/// A box's header when it is open: its name and its numbers.
const HEADER: f64 = 2.0 * LINE + 8.0;
/// The widest a folded box is drawn: a longer formula is scaled into it.
const MAX_W: f64 = 560.0;

fn chars(s: &str) -> f64 {
    s.chars().count() as f64
}

/// The lines a box's header writes: its name and instance, its numbers.
pub fn title(n: &Node) -> String {
    let mut t = n.label.clone();
    if let Some(l) = n.stack {
        t.push_str(&format!(" ×{l}"));
    }
    if let Some(i) = &n.instance
        && *i != n.label
    {
        t.push_str(&format!(" · {i}"));
    }
    t
}

/// The numbers a box or a node carries, in a line: its parameters, its
/// work and its bytes, per run.
pub fn numbers(n: &Node) -> String {
    let mut parts = Vec::new();
    if let Some(p) = &n.params {
        parts.push(format!("θ {} {}", count(p.count as f64), p.types.join("/")));
    }
    let per = n.runs.max(1) as f64;
    if let Some(w) = n.work
        && w > 0.0
    {
        parts.push(format!("W {}", count(w / per)));
    }
    if let Some(q) = n.bytes
        && q > 0.0
    {
        parts.push(format!("Q {}B", count(q / per)));
    }
    parts.join(" · ")
}

/// A count in three digits: `4.10 k`, `1.23 M`.
pub fn count(x: f64) -> String {
    let (v, u) = if x >= 1e9 {
        (x / 1e9, " G")
    } else if x >= 1e6 {
        (x / 1e6, " M")
    } else if x >= 1e3 {
        (x / 1e3, " k")
    } else {
        return format!("{}", x.round() as u64);
    };
    // Three significant digits, rounded before the precision is chosen.
    let digits = |v: f64| {
        if v >= 100.0 {
            0
        } else if v >= 10.0 {
            1
        } else {
            2
        }
    };
    let d = digits(v);
    let r = (v * 10f64.powi(d)).round() / 10f64.powi(d);
    let s = format!("{r:.*}", digits(r) as usize);
    format!("{s}{u}")
}

/// A node's size when it is not opened.
fn folded(n: &Node) -> (f64, f64) {
    match n.kind {
        Kind::Root | Kind::Call | Kind::Stack => {
            let formula: Vec<&str> = n.text.as_deref().map(|t| t.lines().collect()).unwrap_or_default();
            let wide = formula.iter().map(|l| chars(l)).fold(chars(&title(n)).max(chars(&numbers(n))), f64::max);
            let w = (wide * CHAR + 2.0 * PAD).clamp(110.0, MAX_W);
            let lines = 2.0 + formula.len() as f64;
            (w, lines * LINE + PAD + if formula.is_empty() { 0.0 } else { 6.0 })
        }
        Kind::Group | Kind::Op | Kind::Host => ((chars(&n.label) * CHAR + 2.0 * PAD).clamp(44.0, 260.0), if n.kind == Kind::Host { 22.0 } else { 26.0 }),
        Kind::Param => {
            let line = numbers(n);
            ((chars(&n.label).max(chars(&line)) * CHAR + 2.0 * PAD).clamp(40.0, 260.0), if line.is_empty() { 22.0 } else { 2.0 * LINE + 6.0 })
        }
        Kind::Input | Kind::Output => ((chars(&n.label) * CHAR + 2.0 * PAD).clamp(40.0, 200.0), 22.0),
    }
}

/// Lays out every opening of `net`, and picks the default one.
pub fn lay_out(net: &mut Network) {
    let boxes: Vec<usize> = net.nodes.iter().filter(|n| matches!(n.kind, Kind::Call | Kind::Stack)).map(|n| n.id).collect();
    let mut opens: Vec<Vec<usize>> = vec![Vec::new()];
    for b in &boxes {
        let mut chain = vec![*b];
        let mut p = net.nodes[*b].parent;
        while let Some(x) = p {
            if net.nodes[x].kind == Kind::Root {
                break;
            }
            chain.push(x);
            p = net.nodes[x].parent;
        }
        chain.reverse();
        opens.push(chain);
    }
    let by_parent: HashMap<usize, Vec<&Edge>> = net.edges.iter().fold(HashMap::new(), |mut m, e| {
        m.entry(e.parent).or_default().push(e);
        m
    });
    net.views = opens.into_iter().map(|open| view(net, &by_parent, 0, open)).collect();
    // The entry and its first box opened, every stack folded.
    let first = net.nodes[0].children.iter().copied().find(|c| net.nodes[*c].kind == Kind::Call).or_else(|| net.nodes[0].children.iter().copied().find(|c| net.nodes[*c].kind == Kind::Stack));
    net.default = first.and_then(|b| net.views.iter().position(|v| v.open == [b])).unwrap_or(0);
}

/// The opening of level `k`: the entry alone at 0, the default opening at
/// 1, then, in the box opened last, its first stack, else its first box, at
/// each level, as far as boxes go (`presk dump --stage network --level`).
pub fn level(net: &Network, k: usize) -> usize {
    let find = |open: &[usize]| net.views.iter().position(|v| v.open == open);
    if k == 0 {
        return find(&[]).unwrap_or(net.default);
    }
    let mut path = net.views[net.default].open.clone();
    while path.len() < k {
        let Some(last) = path.last().copied() else { break };
        let kids = &net.nodes[last].children;
        let next = kids.iter().copied().find(|c| net.nodes[*c].kind == Kind::Stack).or_else(|| kids.iter().copied().find(|c| net.nodes[*c].kind == Kind::Call));
        match next {
            Some(c) => path.push(c),
            None => break,
        }
    }
    find(&path).unwrap_or(net.default)
}

/// A graph laid out inside its box: each child's rectangle relative to the
/// box's corner, each edge's route, and the box's size.
struct Laid {
    rects: BTreeMap<usize, Rect>,
    routes: BTreeMap<usize, Vec<(f64, f64)>>,
    w: f64,
    h: f64,
}

/// The box `b` opened by itself, its graph laid out without what is around
/// it: a figure of one layer of a stack (`presk dump --stage network
/// --box NAME`). Not one of the openings the page draws.
pub fn alone(net: &Network, b: usize) -> View {
    let by_parent: HashMap<usize, Vec<&Edge>> = net.edges.iter().fold(HashMap::new(), |mut m, e| {
        m.entry(e.parent).or_default().push(e);
        m
    });
    view(net, &by_parent, b, vec![b])
}

/// The first box, outermost first, called `name` (its field: `blocks`,
/// `attn`).
pub fn find_box(net: &Network, name: &str) -> Option<usize> {
    let mut boxes: Vec<&Node> = net.nodes.iter().filter(|n| matches!(n.kind, Kind::Call | Kind::Stack) && n.label == name).collect();
    boxes.sort_by_key(|n| (n.depth, n.id));
    boxes.first().map(|n| n.id)
}

/// The opening `open` laid out from the box `root`, the entry's frame or a
/// box drawn alone.
fn view(net: &Network, by_parent: &HashMap<usize, Vec<&Edge>>, root: usize, open: Vec<usize>) -> View {
    let mut v = View { open: open.clone(), width: 0.0, height: 0.0, rects: BTreeMap::new(), routes: BTreeMap::new() };
    let laid = graph(net, by_parent, root, &open);
    let frame = Rect { x: 0.0, y: 0.0, w: laid.w, h: laid.h };
    v.width = frame.w;
    v.height = frame.h;
    v.rects.insert(root, frame);
    place(net, by_parent, root, &open, laid, (0.0, 0.0), &mut v);
    // An arrow into an opened box ends at the box's own input, one out of
    // it starts at its result.
    for (id, pts) in v.routes.iter_mut() {
        let e = &net.edges[*id];
        if v.open.contains(&e.to)
            && let Some(k) = e.arg
            && let Some(i) = net.nodes[e.to].children.iter().find(|c| net.nodes[**c].kind == Kind::Input && net.nodes[**c].index == Some(k))
            && let (Some(r), Some(last)) = (v.rects.get(i), pts.last_mut())
        {
            *last = (r.x + r.w / 2.0, r.y);
        }
        if v.open.contains(&e.from)
            && let Some(o) = net.nodes[e.from].children.iter().find(|c| net.nodes[**c].kind == Kind::Output)
            && let (Some(r), Some(first)) = (v.rects.get(o), pts.first_mut())
        {
            *first = (r.x + r.w / 2.0, r.y + r.h);
        }
    }
    v
}

/// Puts the graph of `b`, laid out as `laid`, at `origin`, and the graphs of
/// the boxes opened inside it.
fn place(net: &Network, by_parent: &HashMap<usize, Vec<&Edge>>, b: usize, open: &[usize], laid: Laid, origin: (f64, f64), v: &mut View) {
    for (id, r) in &laid.rects {
        let a = Rect { x: r.x + origin.0, y: r.y + origin.1, ..*r };
        v.rects.insert(*id, a);
    }
    for (id, pts) in laid.routes {
        v.routes.insert(id, pts.into_iter().map(|(x, y)| (x + origin.0, y + origin.1)).collect());
    }
    for c in &net.nodes[b].children {
        if open.contains(c) {
            let r = v.rects[c];
            let inner = graph(net, by_parent, *c, open);
            place(net, by_parent, *c, open, inner, (r.x, r.y), v);
        }
    }
}

/// An item of a layer: a node, or an edge passing through.
#[derive(Clone, Copy, PartialEq)]
enum Item {
    Node(usize),
    Pass(usize),
}

fn graph(net: &Network, by_parent: &HashMap<usize, Vec<&Edge>>, b: usize, open: &[usize]) -> Laid {
    let kids: Vec<usize> = net.nodes[b].children.clone();
    let none = Vec::new();
    let edges: Vec<&Edge> = by_parent.get(&b).unwrap_or(&none).iter().copied().filter(|e| kids.contains(&e.from) && kids.contains(&e.to) && e.from != e.to).collect();
    // Sizes: an opened child is its graph.
    let mut size: HashMap<usize, (f64, f64)> = HashMap::new();
    for k in &kids {
        let s = if open.contains(k) {
            let g = graph(net, by_parent, *k, open);
            (g.w, g.h)
        } else {
            folded(&net.nodes[*k])
        };
        size.insert(*k, s);
    }

    // Layers, by longest path; a source but an input sits just before its
    // first reader, the result after everything.
    let mut preds: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut succs: HashMap<usize, Vec<usize>> = HashMap::new();
    for e in &edges {
        preds.entry(e.to).or_default().push(e.from);
        succs.entry(e.from).or_default().push(e.to);
    }
    let mut layer: HashMap<usize, usize> = HashMap::new();
    fn depth(n: usize, preds: &HashMap<usize, Vec<usize>>, memo: &mut HashMap<usize, usize>, guard: usize) -> usize {
        if let Some(d) = memo.get(&n) {
            return *d;
        }
        let d = if guard > 10_000 { 0 } else { preds.get(&n).map_or(0, |ps| ps.iter().map(|p| depth(*p, preds, memo, guard + 1) + 1).max().unwrap_or(0)) };
        memo.insert(n, d);
        d
    }
    for k in &kids {
        let d = depth(*k, &preds, &mut layer, 0);
        layer.insert(*k, d);
    }
    for k in &kids {
        let n = &net.nodes[*k];
        if n.kind != Kind::Input
            && !preds.contains_key(k)
            && let Some(m) = succs.get(k).and_then(|s| s.iter().map(|x| layer[x]).min())
        {
            layer.insert(*k, m.saturating_sub(1));
        }
    }
    let last = kids.iter().filter(|k| net.nodes[**k].kind != Kind::Output).map(|k| layer[k]).max().unwrap_or(0);
    for k in &kids {
        if net.nodes[*k].kind == Kind::Output {
            layer.insert(*k, last + 1);
        }
    }
    let nl = kids.iter().map(|k| layer[k]).max().unwrap_or(0) + 1;

    // The layers' items, edges spanning several layers passing through.
    let mut layers: Vec<Vec<Item>> = vec![Vec::new(); nl];
    let mut order: Vec<usize> = kids.clone();
    order.sort_by_key(|k| (net.nodes[*k].kind != Kind::Input, net.nodes[*k].index.unwrap_or(0), *k));
    for k in &order {
        layers[layer[k]].push(Item::Node(*k));
    }
    let mut chain: HashMap<usize, Vec<Item>> = HashMap::new();
    for e in &edges {
        let (a, z) = (layer[&e.from], layer[&e.to]);
        let mut c = vec![Item::Node(e.from)];
        for items in layers.iter_mut().take(z).skip(a + 1) {
            items.push(Item::Pass(e.id));
            c.push(Item::Pass(e.id));
        }
        c.push(Item::Node(e.to));
        chain.insert(e.id, c);
    }
    // The neighbours of each item in the layer before and after.
    let mut up: HashMap<(usize, bool), Vec<Item>> = HashMap::new();
    let mut down: HashMap<(usize, bool), Vec<Item>> = HashMap::new();
    let key = |i: Item| match i {
        Item::Node(n) => (n, true),
        Item::Pass(e) => (e, false),
    };
    for e in &edges {
        for w in chain[&e.id].windows(2) {
            down.entry(key(w[0])).or_default().push(w[1]);
            up.entry(key(w[1])).or_default().push(w[0]);
        }
    }
    let mut pos: HashMap<(usize, bool), f64> = HashMap::new();
    let index = |layers: &Vec<Vec<Item>>, pos: &mut HashMap<(usize, bool), f64>| {
        for l in layers {
            for (k, i) in l.iter().enumerate() {
                pos.insert(key(*i), k as f64);
            }
        }
    };
    index(&layers, &mut pos);
    for sweep in 0..8 {
        let downward = sweep % 2 == 0;
        let range: Vec<usize> = if downward { (1..nl).collect() } else { (0..nl.saturating_sub(1)).rev().collect() };
        for l in range {
            let nbrs = if downward { &up } else { &down };
            let mut keyed: Vec<(f64, usize, Item)> = layers[l]
                .iter()
                .enumerate()
                .map(|(k, i)| {
                    let ns = nbrs.get(&key(*i)).map(|v| v.as_slice()).unwrap_or(&[]);
                    let bc = if ns.is_empty() { k as f64 } else { ns.iter().map(|n| pos[&key(*n)]).sum::<f64>() / ns.len() as f64 };
                    (bc, k, *i)
                })
                .collect();
            // Inputs keep the order of the arguments.
            if keyed.iter().all(|(_, _, i)| matches!(i, Item::Node(n) if net.nodes[*n].kind == Kind::Input)) {
                continue;
            }
            keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
            layers[l] = keyed.into_iter().map(|(_, _, i)| i).collect();
            for (k, i) in layers[l].iter().enumerate() {
                pos.insert(key(*i), k as f64);
            }
        }
    }

    // Coordinates, top to bottom: a layer's y after the tallest of the
    // layer before; in a layer, each item near the centre of its
    // neighbours, in order, left to right.
    let dims = |i: Item| -> (f64, f64) {
        match i {
            Item::Node(n) => size[&n],
            Item::Pass(_) => (2.0, 0.0),
        }
    };
    let header = HEADER;
    let mut ys = vec![0.0; nl];
    let mut y = header;
    for (l, items) in layers.iter().enumerate() {
        ys[l] = y;
        let h = items.iter().map(|i| dims(*i).1).fold(0.0, f64::max);
        y += h + LAYER_GAP;
    }
    let mut xs: HashMap<(usize, bool), f64> = HashMap::new();
    for items in &layers {
        let mut x = 0.0;
        for i in items {
            xs.insert(key(*i), x);
            x += dims(*i).0 + NODE_GAP;
        }
    }
    let centre = |i: Item, xs: &HashMap<(usize, bool), f64>| xs[&key(i)] + dims(i).0 / 2.0;
    for pass in 0..6 {
        let downward = pass % 2 == 0;
        let range: Vec<usize> = if downward { (1..nl).collect() } else { (0..nl.saturating_sub(1)).rev().collect() };
        for l in range {
            let nbrs = if downward { &up } else { &down };
            let want: Vec<f64> = layers[l]
                .iter()
                .map(|i| match nbrs.get(&key(*i)) {
                    Some(ns) if !ns.is_empty() => ns.iter().map(|n| centre(*n, &xs)).sum::<f64>() / ns.len() as f64 - dims(*i).0 / 2.0,
                    _ => xs[&key(*i)],
                })
                .collect();
            // In order, no two closer than the gap, then the layer moved by
            // the mean of what it missed, never to the left.
            let mut got = Vec::with_capacity(want.len());
            let mut right = f64::NEG_INFINITY;
            for (i, w) in layers[l].iter().zip(&want) {
                let gap = if matches!(i, Item::Pass(_)) { 6.0 } else { NODE_GAP };
                let x = w.max(right + gap);
                got.push(x);
                right = x + dims(*i).0;
            }
            let shift = (want.iter().zip(&got).map(|(w, g)| w - g).sum::<f64>() / want.len().max(1) as f64).min(0.0);
            for (i, g) in layers[l].iter().zip(got) {
                xs.insert(key(*i), g + shift);
            }
        }
    }
    let left = layers.iter().flatten().map(|i| xs[&key(*i)]).fold(f64::INFINITY, f64::min);
    let left = if left.is_finite() { left } else { 0.0 };
    let mut rects = BTreeMap::new();
    let mut bottom: f64 = header;
    let mut right: f64 = 0.0;
    for (l, items) in layers.iter().enumerate() {
        for i in items {
            if let Item::Node(n) = i {
                let (w, h) = size[n];
                let r = Rect { x: xs[&key(*i)] - left + PAD, y: ys[l], w, h };
                bottom = bottom.max(r.y + r.h);
                right = right.max(r.x + r.w);
                rects.insert(*n, r);
            }
        }
    }
    // Routes: from a source's bottom to a target's top, through the layers
    // between; the arrows into a node spread along its top.
    let mut routes = BTreeMap::new();
    let mut ins: HashMap<usize, Vec<&Edge>> = HashMap::new();
    for e in &edges {
        ins.entry(e.to).or_default().push(e);
    }
    for list in ins.values_mut() {
        list.sort_by(|a, b| {
            let xa = rects[&a.from].x + rects[&a.from].w / 2.0;
            let xb = rects[&b.from].x + rects[&b.from].w / 2.0;
            xa.partial_cmp(&xb).unwrap_or(std::cmp::Ordering::Equal).then(a.id.cmp(&b.id))
        });
    }
    for e in &edges {
        let s = rects[&e.from];
        let t = rects[&e.to];
        let list = &ins[&e.to];
        let k = list.iter().position(|x| x.id == e.id).unwrap_or(0);
        let tx = t.x + t.w * (k as f64 + 1.0) / (list.len() as f64 + 1.0);
        let mut pts = vec![(s.x + s.w / 2.0, s.y + s.h)];
        for i in &chain[&e.id][1..chain[&e.id].len() - 1] {
            let l = layer[&e.from] + pts.len();
            pts.push((xs[&key(*i)] - left + PAD + 1.0, ys[l.min(nl - 1)]));
        }
        pts.push((tx, t.y));
        routes.insert(e.id, pts);
    }
    let w = (right + PAD).max(folded(&net.nodes[b]).0.min(MAX_W));
    Laid { rects, routes, w, h: bottom + PAD }
}
