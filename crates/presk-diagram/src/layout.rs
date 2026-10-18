//! The layered layout, top to bottom, and its composition (plan 0.14.1,
//! D146).
//!
//! **A frame per box.** Each box's graph is laid out once, with every
//! child folded: in layers by longest path, its nodes ordered in each layer
//! by the barycenters of their neighbours over a few sweeps, then placed.
//! The frame keeps each child's layer and rectangle, relative to the box's
//! corner, and each arrow's route. Its numbers are snapped to a tenth of a
//! point, so that the page, which reads them from the JSON, composes from
//! the same numbers.
//!
//! **An opening composes the frames** ([`view`]), for any set of boxes
//! open:
//! - an open box takes its composed size;
//! - a layer grows by its tallest open box's growth, and the layers below
//!   it move down by that much;
//! - in a layer, each item moves right by the growth of the open boxes
//!   before it;
//! - nothing is reordered.
//!
//! An arrow leaves its source straight down to the bottom of the source's
//! layer, crosses each layer it passes through straight down, and bends
//! only between layers, so that it runs through no node. An arrow into or
//! out of an open box meets it at its port: the input it is, the result.
//!
//! A box with no child open is drawn exactly as its frame, translated. The
//! page, the SVG and the TikZ draw this composition; the page computes it
//! by the same arithmetic, in the same order, and checks itself against
//! the compiler's under `#bench`.
//!
//! No text is measured: a node's size comes from its strings' lengths in
//! characters (the Unicode printer's, for a formula). The page scales a
//! formula into its box.

use std::collections::{BTreeMap, BTreeSet, HashMap};

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
    /// The boxes drawn open, outermost first (by depth, then id); the
    /// entry's frame always is, and is not listed. A box drawn alone is.
    pub open: Vec<usize>,
    pub width: f64,
    pub height: f64,
    /// Every node drawn, and its rectangle.
    pub rects: BTreeMap<usize, Rect>,
    /// Every edge drawn, and its route from its source to its target.
    pub routes: BTreeMap<usize, Vec<(f64, f64)>>,
}

/// A box's graph laid out with every child folded, in the box's own
/// coordinates: what an opening composes.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub w: f64,
    pub h: f64,
    /// Each layer's top and its height, the tallest of its items.
    pub layers: Vec<(f64, f64)>,
    /// Each child's layer and rectangle.
    pub rects: BTreeMap<usize, (usize, Rect)>,
    /// Each arrow's places in the layers it passes through, after the
    /// source's, one each at the layer's top; and where it lands along its
    /// target's top, a fraction of its width.
    pub routes: BTreeMap<usize, (Vec<(f64, f64)>, f64)>,
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

/// Lays out the frame of every box of `net`, and picks the default
/// opening: the entry and its first box opened, every stack folded.
pub fn lay_out(net: &mut Network) {
    let by_parent: HashMap<usize, Vec<&Edge>> = net.edges.iter().fold(HashMap::new(), |mut m, e| {
        m.entry(e.parent).or_default().push(e);
        m
    });
    let boxes: Vec<usize> = net.nodes.iter().filter(|n| n.kind.is_box()).map(|n| n.id).collect();
    let frames = boxes.into_iter().map(|b| (b, frame(net, &by_parent, b))).collect();
    net.frames = frames;
    let first = net.nodes[0].children.iter().copied().find(|c| net.nodes[*c].kind == Kind::Call).or_else(|| net.nodes[0].children.iter().copied().find(|c| net.nodes[*c].kind == Kind::Stack));
    net.default = first.map(|b| vec![b]).unwrap_or_default();
}

/// The opening of level `k`: the entry alone at 0, the default opening at
/// 1, then, in the box opened last, its first stack, else its first box, at
/// each level, as far as boxes go (`presk dump --stage network --level`).
pub fn level(net: &Network, k: usize) -> Vec<usize> {
    if k == 0 {
        return Vec::new();
    }
    let mut path = net.default.clone();
    while path.len() < k {
        let Some(last) = path.last().copied() else { break };
        let kids = &net.nodes[last].children;
        let next = kids.iter().copied().find(|c| net.nodes[*c].kind == Kind::Stack).or_else(|| kids.iter().copied().find(|c| net.nodes[*c].kind == Kind::Call));
        match next {
            Some(c) => path.push(c),
            None => break,
        }
    }
    path
}

/// The exploded opening: every box (`--level all`).
pub fn everything(net: &Network) -> Vec<usize> {
    net.nodes.iter().filter(|n| matches!(n.kind, Kind::Call | Kind::Stack)).map(|n| n.id).collect()
}

/// The box `b` drawn by itself, its frame: a figure of one layer of a
/// stack (`presk dump --stage network --box NAME`).
pub fn alone(net: &Network, b: usize) -> View {
    let f = &net.frames[&b];
    let mut v = View { open: vec![b], width: f.w, height: f.h, rects: BTreeMap::new(), routes: BTreeMap::new() };
    v.rects.insert(b, Rect { x: 0.0, y: 0.0, w: f.w, h: f.h });
    place(net, b, &BTreeSet::new(), &mut HashMap::new(), (0.0, 0.0), &mut v);
    v
}

/// The first box, outermost first, called `name` (its field: `blocks`,
/// `attn`).
pub fn find_box(net: &Network, name: &str) -> Option<usize> {
    let mut boxes: Vec<&Node> = net.nodes.iter().filter(|n| matches!(n.kind, Kind::Call | Kind::Stack) && n.label == name).collect();
    boxes.sort_by_key(|n| (n.depth, n.id));
    boxes.first().map(|n| n.id)
}

/// The opening where the boxes of `open` are open, those inside a folded
/// box aside, composed from the frames.
pub fn view(net: &Network, open: &[usize]) -> View {
    let open: BTreeSet<usize> = open.iter().copied().filter(|b| matches!(net.nodes[*b].kind, Kind::Call | Kind::Stack)).collect();
    let mut sizes = HashMap::new();
    let (w, h) = size(net, 0, &open, &mut sizes);
    let mut v = View { open: Vec::new(), width: w, height: h, rects: BTreeMap::new(), routes: BTreeMap::new() };
    v.rects.insert(0, Rect { x: 0.0, y: 0.0, w, h });
    place(net, 0, &open, &mut sizes, (0.0, 0.0), &mut v);
    v.open.sort_by_key(|b| (net.nodes[*b].depth, *b));
    v
}

/// How a box's frame grows when the boxes of `open` inside it are: each
/// layer's growth and how far it moves down, and each open child's size.
struct Growth {
    dh: Vec<f64>,
    down: Vec<f64>,
    open: Vec<(usize, (f64, f64))>,
}

fn growth(net: &Network, b: usize, open: &BTreeSet<usize>, sizes: &mut HashMap<usize, (f64, f64)>) -> Growth {
    let f = &net.frames[&b];
    let mut kids: Vec<(usize, (f64, f64))> = Vec::new();
    for c in f.rects.keys() {
        if open.contains(c) {
            kids.push((*c, size(net, *c, open, sizes)));
        }
    }
    // A layer as tall as its tallest item, an open box at its size.
    let mut dh = Vec::with_capacity(f.layers.len());
    for (l, (_, hl)) in f.layers.iter().enumerate() {
        let mut nh = *hl;
        for (c, (_, h)) in &kids {
            if f.rects[c].0 == l {
                nh = nh.max(*h);
            }
        }
        dh.push(nh - hl);
    }
    let mut down = Vec::with_capacity(dh.len() + 1);
    let mut d = 0.0;
    for g in &dh {
        down.push(d);
        d += g;
    }
    down.push(d);
    Growth { dh, down, open: kids }
}

/// How far an item at `x` in layer `l` moves right: the growth in width of
/// the open boxes before it in the layer, from left to right.
fn shift(f: &Frame, g: &Growth, l: usize, x: f64) -> f64 {
    let mut before: Vec<(f64, f64)> = g.open.iter().filter(|(c, _)| f.rects[c].0 == l && f.rects[c].1.x < x).map(|(c, (w, _))| (f.rects[c].1.x, w - f.rects[c].1.w)).collect();
    before.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut s = 0.0;
    for (_, dw) in before {
        s += dw;
    }
    s
}

/// The size of box `b` composed with the boxes of `open` inside it.
fn size(net: &Network, b: usize, open: &BTreeSet<usize>, sizes: &mut HashMap<usize, (f64, f64)>) -> (f64, f64) {
    if let Some(s) = sizes.get(&b) {
        return *s;
    }
    let g = growth(net, b, open, sizes);
    let f = &net.frames[&b];
    let mut right = f64::NEG_INFINITY;
    for (c, (l, r)) in &f.rects {
        let w = g.open.iter().find(|(k, _)| k == c).map_or(r.w, |(_, (w, _))| *w);
        right = right.max(r.x + shift(f, &g, *l, r.x) + w);
    }
    let w = if right.is_finite() { f.w.max(right + PAD) } else { f.w };
    let s = (w, f.h + g.down[g.dh.len()]);
    sizes.insert(b, s);
    s
}

/// Puts the frame of `b`, composed, at `origin`, and the boxes open inside
/// it; then its arrows, which meet an open box at its port.
fn place(net: &Network, b: usize, open: &BTreeSet<usize>, sizes: &mut HashMap<usize, (f64, f64)>, origin: (f64, f64), v: &mut View) {
    let g = growth(net, b, open, sizes);
    let f = &net.frames[&b];
    for (c, (l, r)) in &f.rects {
        let (w, h) = g.open.iter().find(|(k, _)| k == c).map_or((r.w, r.h), |(_, s)| *s);
        let a = Rect { x: origin.0 + (r.x + shift(f, &g, *l, r.x)), y: origin.1 + (r.y + g.down[*l]), w, h };
        v.rects.insert(*c, a);
    }
    for (c, _) in &g.open {
        v.open.push(*c);
        let r = v.rects[c];
        place(net, *c, open, sizes, (r.x, r.y), v);
    }
    // A layer's bottom, composed.
    let bottom = |l: usize| origin.1 + (f.layers[l].0 + g.down[l]) + (f.layers[l].1 + g.dh[l]);
    let is_open = |c: usize| g.open.iter().any(|(k, _)| *k == c);
    for (id, (pts, at)) in &f.routes {
        let e = &net.edges[*id];
        let from = f.rects[&e.from].0;
        let output = || net.nodes[e.from].children.iter().find(|c| net.nodes[**c].kind == Kind::Output).and_then(|o| v.rects.get(o));
        let input = || e.arg.and_then(|k| net.nodes[e.to].children.iter().find(|c| net.nodes[**c].kind == Kind::Input && net.nodes[**c].index == Some(k))).and_then(|i| v.rects.get(i));
        let s = match (is_open(e.from), output()) {
            (true, Some(o)) => (o.x + o.w / 2.0, o.y + o.h),
            _ => {
                let s = v.rects[&e.from];
                (s.x + s.w / 2.0, s.y + s.h)
            }
        };
        let mut q = vec![s];
        if s.1 < bottom(from) {
            q.push((s.0, bottom(from)));
        }
        for (k, (x, y)) in pts.iter().enumerate() {
            let l = (from + k + 1).min(f.layers.len() - 1);
            let (x, y) = (origin.0 + (x + shift(f, &g, l, *x)), origin.1 + (y + g.down[l]));
            q.push((x, y));
            if bottom(l) > y {
                q.push((x, bottom(l)));
            }
        }
        let t = match (is_open(e.to), input()) {
            (true, Some(i)) => (i.x + i.w / 2.0, i.y),
            _ => {
                let t = v.rects[&e.to];
                (t.x + t.w * at, t.y)
            }
        };
        q.push(t);
        v.routes.insert(*id, q);
    }
}

/// A number snapped to a tenth of a point.
fn tenth(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// An item of a layer: a node, or an edge passing through the layer (its
/// id and the layer's: an edge passes through several, at a place in each).
#[derive(Clone, Copy, PartialEq)]
enum Item {
    Node(usize),
    Pass(usize, usize),
}

/// The frame of box `b`: its graph laid out with every child folded.
fn frame(net: &Network, by_parent: &HashMap<usize, Vec<&Edge>>, b: usize) -> Frame {
    let kids: Vec<usize> = net.nodes[b].children.clone();
    let none = Vec::new();
    let edges: Vec<&Edge> = by_parent.get(&b).unwrap_or(&none).iter().copied().filter(|e| kids.contains(&e.from) && kids.contains(&e.to) && e.from != e.to).collect();
    let size: HashMap<usize, (f64, f64)> = kids.iter().map(|k| (*k, folded(&net.nodes[*k]))).collect();

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
        for (l, items) in layers.iter_mut().enumerate().take(z).skip(a + 1) {
            items.push(Item::Pass(e.id, l));
            c.push(Item::Pass(e.id, l));
        }
        c.push(Item::Node(e.to));
        chain.insert(e.id, c);
    }
    // The neighbours of each item in the layer before and after.
    let mut up: HashMap<(usize, usize), Vec<Item>> = HashMap::new();
    let mut down: HashMap<(usize, usize), Vec<Item>> = HashMap::new();
    let key = |i: Item| match i {
        Item::Node(n) => (n, usize::MAX),
        Item::Pass(e, l) => (e, l),
    };
    for e in &edges {
        for w in chain[&e.id].windows(2) {
            down.entry(key(w[0])).or_default().push(w[1]);
            up.entry(key(w[1])).or_default().push(w[0]);
        }
    }
    let mut pos: HashMap<(usize, usize), f64> = HashMap::new();
    let index = |layers: &Vec<Vec<Item>>, pos: &mut HashMap<(usize, usize), f64>| {
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
            Item::Pass(..) => (2.0, 0.0),
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
    let mut xs: HashMap<(usize, usize), f64> = HashMap::new();
    for items in &layers {
        let mut x = 0.0;
        for i in items {
            xs.insert(key(*i), x);
            x += dims(*i).0 + NODE_GAP;
        }
    }
    let centre = |i: Item, xs: &HashMap<(usize, usize), f64>| xs[&key(i)] + dims(i).0 / 2.0;
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
                let gap = if matches!(i, Item::Pass(..)) { 6.0 } else { NODE_GAP };
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
    // Every number snapped to a tenth: the frame is what the JSON carries.
    let tops: Vec<(f64, f64)> = layers.iter().enumerate().map(|(l, items)| (tenth(ys[l]), tenth(items.iter().map(|i| dims(*i).1).fold(0.0, f64::max)))).collect();
    let mut rects = BTreeMap::new();
    let mut bottom: f64 = header;
    let mut right: f64 = 0.0;
    for (l, items) in layers.iter().enumerate() {
        for i in items {
            if let Item::Node(n) = i {
                let (w, h) = size[n];
                let r = Rect { x: tenth(xs[&key(*i)] - left + PAD), y: tops[l].0, w: tenth(w), h: tenth(h) };
                bottom = bottom.max(r.y + r.h);
                right = right.max(r.x + r.w);
                rects.insert(*n, r);
            }
        }
    }
    // Routes: each arrow's place in the layers between its source and its
    // target; the arrows into a node spread along its top, in the order of
    // their sources.
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
        let list = &ins[&e.to];
        let k = list.iter().position(|x| x.id == e.id).unwrap_or(0);
        let at = (k as f64 + 1.0) / (list.len() as f64 + 1.0);
        let mut pts = Vec::new();
        for i in &chain[&e.id][1..chain[&e.id].len() - 1] {
            let l = layer[&e.from] + 1 + pts.len();
            pts.push((tenth(xs[&key(*i)] - left + PAD + 1.0), tops[l.min(nl - 1)].0));
        }
        routes.insert(e.id, (pts, at));
    }
    let w = tenth((right + PAD).max(folded(&net.nodes[b]).0.min(MAX_W)));
    let rects = rects.into_iter().map(|(n, r)| (n, (layer[&n], r))).collect();
    Frame { w, h: tenth(bottom + PAD), layers: tops, rects, routes }
}
