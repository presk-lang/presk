//! The layout of a nest, and the realization plan that carries it (0.7.0
//! step 1). The language reference defines both: the layout decisions and
//! their domains in its §6.3, the plan object in §6.6, well-formedness in
//! §6.9.
//!
//! A layout never changes a result bit (invariant `realize-separation`): two
//! plans with the same realization compute the same values and differ only in
//! time. That is why a layout is a member of a plan beside the realization,
//! and not a field of the realization file: a checkpoint, a measured gap or a
//! served model names the realization, and a change of tile must not change
//! what they name (question 63).
//!
//! Every decision of a [`Layout`] is optional. An unstated decision takes its
//! default, which depends on the nest's kind and shapes
//! ([`Layout::default_for`]); a stated one is checked against them
//! ([`Layout::check`]) when the nest is launched, since a nest is generic in
//! its function's dimensions and one layout serves every shape it runs at.

use std::collections::BTreeMap;
use std::fmt;

use serde_json::{Map, Value as Json};

use crate::Realization;

/// How a contraction nest's kernel is arranged (reference §6.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Form {
    /// The outer loops and `I`, then `k` enclosing `J`: each running sum
    /// lives in the output buffer from one `k` to the next, any `J`.
    RowWise,
    /// Outer loops, block loops, and a micro-kernel holding a register tile.
    Tiled,
}

/// The width of the block loop over `K`, or none.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KBlock {
    None,
    Width(usize),
}

/// A block loop of a tiled contraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlockLoop {
    I,
    J,
    K,
}

/// A thread split: none, or the loop at `depth` divided among `workers`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Split {
    None,
    At { depth: usize, workers: usize },
}

/// The register tiles of the domain besides `(1, 1)`: `{1, 2, 4} × {4, 8, 16}`.
pub const TILE_MI: [usize; 3] = [1, 2, 4];
pub const TILE_NJ: [usize; 3] = [4, 8, 16];
/// The k-block widths of the domain.
pub const KBLOCKS: [usize; 2] = [32, 128];
/// The deepest loop a thread split may name.
pub const MAX_SPLIT_DEPTH: usize = 2;

/// What a nest is, as far as its layout decisions go (reference, definition
/// "Contraction nest; softmax nest"). Built by the kernel backend at a launch,
/// where the operands' shapes and the nest's types are known.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NestKind {
    Contraction(Contraction),
    /// A `softmax`. `narrow` when its storage type is not its arithmetic type
    /// (`bf16`), where two passes would round `exp(x − m)` twice.
    Softmax { narrow: bool },
    /// Every other nest: an elementwise group, a reduction, a contraction of
    /// another pattern. Its only layout decision is the thread split.
    Other,
}

/// A contraction nest at the shapes it is launched at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Contraction {
    /// A `dot` (else an `ein`).
    pub dot: bool,
    pub i: usize,
    pub j: usize,
    pub k: usize,
    /// Operand 1 has `K` as its innermost axis.
    pub nt: bool,
    /// The accumulation type is the storage type, so a running sum may be
    /// stored to the output and reloaded without a rounding (condition 7).
    pub accum_is_storage: bool,
    /// The vector register's width in the nest's arithmetic type on the
    /// machine: two for `f64`, four for `f32`, one for `bf16`.
    pub width: usize,
}

/// The layout decisions a plan states for one nest.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Layout {
    pub form: Option<Form>,
    /// `(m_i, n_j)`.
    pub tile: Option<(usize, usize)>,
    pub kblock: Option<KBlock>,
    /// The block loops, outermost first.
    pub order: Option<Vec<BlockLoop>>,
    pub lanes: Option<usize>,
    pub split: Option<Split>,
    pub pack: Option<bool>,
    pub passes: Option<usize>,
}

/// The members of a layout record, in the order they are printed.
pub const DECISIONS: [&str; 8] = ["form", "tile", "kblock", "order", "lanes", "split", "pack", "passes"];

impl Layout {
    /// Whether no decision is stated: every one takes its default.
    pub fn is_empty(&self) -> bool {
        *self == Layout::default()
    }

    /// The decisions this layout states, by name.
    pub fn stated(&self) -> Vec<&'static str> {
        let has = [self.form.is_some(), self.tile.is_some(), self.kblock.is_some(), self.order.is_some(), self.lanes.is_some(), self.split.is_some(), self.pack.is_some(), self.passes.is_some()];
        DECISIONS.iter().zip(has).filter(|(_, h)| *h).map(|(n, _)| *n).collect()
    }

    /// Every decision this layout states, and every other one from `base`.
    pub fn over(&self, base: &Layout) -> Layout {
        Layout {
            form: self.form.or(base.form),
            tile: self.tile.or(base.tile),
            kblock: self.kblock.or(base.kblock),
            order: self.order.clone().or_else(|| base.order.clone()),
            lanes: self.lanes.or(base.lanes),
            split: self.split.or(base.split),
            pack: self.pack.or(base.pack),
            passes: self.passes.or(base.passes),
        }
    }

    /// The reference's default layout of a nest of this kind: 0.6.0's kernel
    /// wherever that kernel realizes its plan. A contraction is tiled, `(1,
    /// 8)` on a `dot` whose `J` is a multiple of eight and `(1, 1)` otherwise,
    /// except that such a `dot` is row-wise where condition 7 allows it; no
    /// k-block, the order `I J`, the register's width in lanes where it
    /// divides `n_j`, no pack; a softmax takes three passes. The thread split
    /// is left unstated: its default, the outermost loop over every worker
    /// above a work threshold, is the executor's.
    pub fn default_for(kind: &NestKind) -> Layout {
        Layout::default().resolve(kind)
    }

    /// This layout with every decision it leaves unstated at its default for
    /// `kind`. The defaults depend on what is stated: without a k-block the
    /// order is `I J`, with one `I J K`; the lanes are the register's width
    /// where it divides the tile's columns and operand 1's `J` is contiguous
    /// (NN, or packed), one otherwise. The split stays as stated.
    pub fn resolve(&self, kind: &NestKind) -> Layout {
        match kind {
            NestKind::Contraction(c) => {
                let row_wise = c.dot && c.j % 8 != 0 && c.accum_is_storage;
                let form = self.form.unwrap_or(if row_wise { Form::RowWise } else { Form::Tiled });
                if form == Form::RowWise {
                    return Layout { form: Some(form), lanes: Some(self.lanes.unwrap_or(1)), split: self.split, ..Layout::default() };
                }
                let tile = self.tile.unwrap_or(if c.dot && c.j % 8 == 0 { (1, 8) } else { (1, 1) });
                let kblock = self.kblock.unwrap_or(KBlock::None);
                let order = self.order.clone().unwrap_or_else(|| match kblock {
                    KBlock::None => vec![BlockLoop::I, BlockLoop::J],
                    KBlock::Width(_) => vec![BlockLoop::I, BlockLoop::J, BlockLoop::K],
                });
                let pack = self.pack.unwrap_or(false);
                let lanes = self.lanes.unwrap_or(if tile.1 % c.width == 0 && (!c.nt || pack) { c.width } else { 1 });
                Layout { form: Some(form), tile: Some(tile), kblock: Some(kblock), order: Some(order), lanes: Some(lanes), split: self.split, pack: Some(pack), passes: None }
            }
            NestKind::Softmax { .. } => Layout { passes: Some(self.passes.unwrap_or(3)), lanes: Some(self.lanes.unwrap_or(1)), split: self.split, ..Layout::default() },
            NestKind::Other => Layout { lanes: Some(self.lanes.unwrap_or(1)), split: self.split, ..Layout::default() },
        }
    }

    /// The layout space of a nest (reference §6.3): every well-formed layout,
    /// each stating every decision the kind has. The thread split is left
    /// out: its domain is the executor's (the worker counts of the machine,
    /// the parallel loops of the kernel). The first point is the default.
    pub fn space(kind: &NestKind) -> Vec<Layout> {
        let mut points = vec![Layout::default_for(kind)];
        let mut push = |l: Layout| {
            if l.check(kind).is_ok() && !points.contains(&l) {
                points.push(l);
            }
        };
        match kind {
            NestKind::Contraction(c) => {
                push(Layout { form: Some(Form::RowWise), lanes: Some(1), ..Layout::default() });
                let mut tiles = vec![(1, 1)];
                tiles.extend(TILE_MI.iter().flat_map(|mi| TILE_NJ.iter().map(move |nj| (*mi, *nj))));
                let orders2 = [[BlockLoop::I, BlockLoop::J], [BlockLoop::J, BlockLoop::I]];
                let orders3 = [
                    [BlockLoop::I, BlockLoop::J, BlockLoop::K],
                    [BlockLoop::I, BlockLoop::K, BlockLoop::J],
                    [BlockLoop::J, BlockLoop::I, BlockLoop::K],
                    [BlockLoop::J, BlockLoop::K, BlockLoop::I],
                    [BlockLoop::K, BlockLoop::I, BlockLoop::J],
                    [BlockLoop::K, BlockLoop::J, BlockLoop::I],
                ];
                let mut blocks: Vec<(KBlock, Vec<BlockLoop>)> = orders2.iter().map(|o| (KBlock::None, o.to_vec())).collect();
                for b in KBLOCKS {
                    blocks.extend(orders3.iter().map(|o| (KBlock::Width(b), o.to_vec())));
                }
                let mut widths = vec![1];
                if c.width > 1 {
                    widths.push(c.width);
                }
                for tile in &tiles {
                    for (kblock, order) in &blocks {
                        for lanes in &widths {
                            for pack in [false, true] {
                                push(Layout { form: Some(Form::Tiled), tile: Some(*tile), kblock: Some(*kblock), order: Some(order.clone()), lanes: Some(*lanes), pack: Some(pack), ..Layout::default() });
                            }
                        }
                    }
                }
            }
            NestKind::Softmax { .. } => push(Layout { passes: Some(2), lanes: Some(1), ..Layout::default() }),
            NestKind::Other => {}
        }
        points
    }

    /// Checks this layout, stated over the default of `kind`, against the
    /// nest: which decisions the kind has, and well-formedness conditions 3,
    /// 4, 7 and 9 of the reference at these shapes. Conditions 5 and 8 (the
    /// worker count against the machine, the split's depth against the
    /// kernel) are the executor's. The message names the decision.
    pub fn check(&self, kind: &NestKind) -> Result<(), String> {
        let contraction_only = [("form", self.form.is_some()), ("tile", self.tile.is_some()), ("kblock", self.kblock.is_some()), ("order", self.order.is_some()), ("pack", self.pack.is_some())];
        match kind {
            NestKind::Contraction(_) => {
                if self.passes.is_some() {
                    return Err("`passes` is a decision of a softmax nest, and this is a contraction".into());
                }
            }
            NestKind::Softmax { .. } | NestKind::Other => {
                if let Some((name, _)) = contraction_only.iter().find(|(_, s)| *s) {
                    return Err(format!("`{name}` is a decision of a contraction nest, and this is {}", if matches!(kind, NestKind::Other) { "neither a contraction nor a softmax" } else { "a softmax" }));
                }
                if self.passes.is_some() && matches!(kind, NestKind::Other) {
                    return Err("`passes` is a decision of a softmax nest, and this is not one".into());
                }
            }
        }
        let eff = self.resolve(kind);
        match kind {
            NestKind::Contraction(c) => {
                let tiled = eff.form == Some(Form::Tiled);
                if !tiled && [self.tile.is_some(), self.kblock.is_some(), self.order.is_some(), self.pack.is_some()].iter().any(|s| *s) {
                    return Err("a register tile, a k-block, a loop order and a pack are decisions of a tiled contraction, and this one is row-wise (condition 3)".into());
                }
                if tiled {
                    let (mi, nj) = eff.tile.unwrap_or((1, 1));
                    if c.i % mi != 0 || c.j % nj != 0 {
                        return Err(format!("the tile [{mi}, {nj}] does not divide I = {} and J = {} (condition 3)", c.i, c.j));
                    }
                    let kb = eff.kblock.unwrap_or(KBlock::None);
                    if let KBlock::Width(b) = kb
                        && c.k % b != 0
                    {
                        return Err(format!("the k-block {b} does not divide K = {} (condition 3)", c.k));
                    }
                    // An order stated without the k-block, or a k-block without
                    // the order, takes the other's default: `I J` or `I J K`.
                    let order = eff.order.clone().unwrap_or_default();
                    let has_k = order.contains(&BlockLoop::K);
                    if has_k != matches!(kb, KBlock::Width(_)) {
                        return Err(format!("the order {} {} a k-block, and the k-block is {}", order_name(&order), if has_k { "needs" } else { "has no loop over the blocks of K, which" }, kblock_name(kb)));
                    }
                    let lanes = eff.lanes.unwrap_or(1);
                    if lanes != 1 && lanes != c.width {
                        return Err(format!("{lanes} lanes: the width is one or the register's, {} in this nest's type (condition 4)", c.width));
                    }
                    if nj % lanes != 0 {
                        return Err(format!("{lanes} lanes do not divide the tile's {nj} columns (condition 4)"));
                    }
                    if lanes != 1 && c.nt && eff.pack != Some(true) {
                        return Err(format!("{lanes} lanes on an NT contraction: operand 1 has K innermost, so its J is not contiguous, and lanes need the pack (condition 4)"));
                    }
                    // The running sum leaves its accumulator when the loop over
                    // the blocks of K encloses the one over I or over J.
                    let spills = has_k && order.last() != Some(&BlockLoop::K);
                    if spills && !c.accum_is_storage {
                        return Err(format!("the order {} keeps running sums in the output buffer between blocks of K, which rounds them unless the accumulation type is the storage type (condition 7)", order_name(&order)));
                    }
                    if eff.pack == Some(true) && !c.nt {
                        return Err("a pack is a decision of an NT contraction, and this one is NN (condition 9)".into());
                    }
                } else {
                    if !c.accum_is_storage {
                        return Err("the row-wise form keeps running sums in the output buffer, which rounds them unless the accumulation type is the storage type (condition 7)".into());
                    }
                    if eff.lanes.unwrap_or(1) != 1 {
                        return Err("a row-wise contraction has one lane (condition 4)".into());
                    }
                }
            }
            NestKind::Softmax { narrow } => {
                if eff.passes == Some(2) && *narrow {
                    return Err("two passes store exp(x − m) in the output, which rounds it a second time unless the storage type is the arithmetic type (condition 7)".into());
                }
                if eff.lanes.unwrap_or(1) != 1 {
                    return Err("only a tiled contraction has lanes (condition 4)".into());
                }
            }
            NestKind::Other => {
                if eff.lanes.unwrap_or(1) != 1 {
                    return Err("only a tiled contraction has lanes (condition 4)".into());
                }
            }
        }
        Ok(())
    }

    /// Reads a layout record, checking every value against its domain.
    pub fn from_json(v: &Json) -> Result<Layout, String> {
        let Json::Object(m) = v else { return Err(format!("a layout is a record of decisions, not {v}")) };
        let mut l = Layout::default();
        for (name, value) in m {
            let bad = |what: &str| format!("`{name}`: {what}, not {value}");
            match name.as_str() {
                "form" => {
                    l.form = Some(match value.as_str() {
                        Some("tiled") => Form::Tiled,
                        Some("row-wise") => Form::RowWise,
                        _ => return Err(bad("\"tiled\" or \"row-wise\"")),
                    })
                }
                "tile" => {
                    let pair: Option<Vec<usize>> = value.as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect());
                    let (mi, nj) = match pair.as_deref() {
                        Some([mi, nj]) if value.as_array().map(|a| a.len()) == Some(2) => (*mi, *nj),
                        _ => return Err(bad("[m_i, n_j]")),
                    };
                    if (mi, nj) != (1, 1) && !(TILE_MI.contains(&mi) && TILE_NJ.contains(&nj)) {
                        return Err(format!("`tile`: [{mi}, {nj}] is outside the domain, [1, 1] or {{1, 2, 4}} × {{4, 8, 16}}"));
                    }
                    l.tile = Some((mi, nj));
                }
                "kblock" => {
                    l.kblock = Some(match value {
                        Json::String(s) if s == "none" => KBlock::None,
                        Json::Number(n) => match n.as_u64() {
                            Some(b) if KBLOCKS.contains(&(b as usize)) => KBlock::Width(b as usize),
                            _ => return Err(format!("`kblock`: {n} is outside the domain, \"none\", 32 or 128")),
                        },
                        _ => return Err(bad("\"none\", 32 or 128")),
                    })
                }
                "order" => {
                    let Some(s) = value.as_str() else { return Err(bad("a string of block loops such as \"JIK\"")) };
                    let order: Vec<BlockLoop> = s
                        .chars()
                        .map(|c| match c {
                            'I' => Ok(BlockLoop::I),
                            'J' => Ok(BlockLoop::J),
                            'K' => Ok(BlockLoop::K),
                            _ => Err(format!("`order`: `{c}` in \"{s}\" is not a block loop (I, J or K)")),
                        })
                        .collect::<Result<_, _>>()?;
                    let mut sorted = order.clone();
                    sorted.sort_by_key(|b| *b as u8);
                    if sorted != [BlockLoop::I, BlockLoop::J] && sorted != [BlockLoop::I, BlockLoop::J, BlockLoop::K] {
                        return Err(format!("`order`: \"{s}\" is not a permutation of IJ or of IJK"));
                    }
                    l.order = Some(order);
                }
                "lanes" => match value.as_u64() {
                    Some(n) if n >= 1 => l.lanes = Some(n as usize),
                    _ => return Err(bad("a positive number of lanes")),
                },
                "split" => {
                    l.split = Some(match value {
                        Json::String(s) if s == "none" => Split::None,
                        Json::Object(o) => {
                            let field = |k: &str| o.get(k).and_then(|x| x.as_u64()).map(|n| n as usize);
                            if o.keys().any(|k| k != "depth" && k != "workers") {
                                return Err(bad("\"none\" or { \"depth\": d, \"workers\": p }"));
                            }
                            match (field("depth"), field("workers")) {
                                (Some(depth), Some(workers)) if depth <= MAX_SPLIT_DEPTH && workers >= 1 => Split::At { depth, workers },
                                (Some(depth), Some(_)) if depth > MAX_SPLIT_DEPTH => return Err(format!("`split`: depth {depth} is outside the domain, 0 to {MAX_SPLIT_DEPTH}")),
                                _ => return Err(bad("\"none\" or { \"depth\": d, \"workers\": p }, p at least one")),
                            }
                        }
                        _ => return Err(bad("\"none\" or { \"depth\": d, \"workers\": p }")),
                    })
                }
                "pack" => match value.as_bool() {
                    Some(b) => l.pack = Some(b),
                    None => return Err(bad("true or false")),
                },
                "passes" => match value.as_u64() {
                    Some(n @ (2 | 3)) => l.passes = Some(n as usize),
                    _ => return Err(bad("3 or 2")),
                },
                other => return Err(format!("unknown layout decision `{other}`; the decisions are {}", DECISIONS.join(", "))),
            }
        }
        if l.form == Some(Form::RowWise) && (l.tile.is_some() || l.kblock.is_some() || l.order.is_some() || l.pack.is_some()) {
            return Err("a register tile, a k-block, a loop order and a pack are decisions of a tiled contraction, and this layout says row-wise (condition 3)".into());
        }
        if let (Some(o), Some(kb)) = (&l.order, l.kblock) {
            let has_k = o.contains(&BlockLoop::K);
            if has_k != matches!(kb, KBlock::Width(_)) {
                return Err(format!("the order {} does not match the k-block {}: a permutation of IJ without one, of IJK with one", order_name(o), kblock_name(kb)));
            }
        }
        Ok(l)
    }

    /// The layout record, members in the order of [`DECISIONS`].
    pub fn to_json(&self) -> Json {
        let mut m = Map::new();
        if let Some(f) = self.form {
            m.insert("form".into(), Json::String(form_name(f).into()));
        }
        if let Some((mi, nj)) = self.tile {
            m.insert("tile".into(), Json::Array(vec![mi.into(), nj.into()]));
        }
        if let Some(kb) = self.kblock {
            m.insert("kblock".into(), match kb {
                KBlock::None => Json::String("none".into()),
                KBlock::Width(b) => b.into(),
            });
        }
        if let Some(o) = &self.order {
            m.insert("order".into(), Json::String(order_name(o)));
        }
        if let Some(n) = self.lanes {
            m.insert("lanes".into(), n.into());
        }
        if let Some(s) = self.split {
            m.insert("split".into(), match s {
                Split::None => Json::String("none".into()),
                Split::At { depth, workers } => {
                    let mut o = Map::new();
                    o.insert("depth".into(), depth.into());
                    o.insert("workers".into(), workers.into());
                    Json::Object(o)
                }
            });
        }
        if let Some(p) = self.pack {
            m.insert("pack".into(), p.into());
        }
        if let Some(n) = self.passes {
            m.insert("passes".into(), n.into());
        }
        Json::Object(m)
    }

    /// A short canonical form of the stated decisions, for cache keys: two
    /// layouts with the same key lower to the same kernel. Empty when nothing
    /// is stated.
    pub fn key(&self) -> String {
        let mut parts = Vec::new();
        if let Some(f) = self.form {
            parts.push(format!("form={}", form_name(f)));
        }
        if let Some((mi, nj)) = self.tile {
            parts.push(format!("tile={mi}x{nj}"));
        }
        if let Some(kb) = self.kblock {
            parts.push(format!("kblock={}", kblock_name(kb)));
        }
        if let Some(o) = &self.order {
            parts.push(format!("order={}", order_name(o)));
        }
        if let Some(n) = self.lanes {
            parts.push(format!("lanes={n}"));
        }
        if let Some(s) = self.split {
            parts.push(format!("split={}", split_name(s)));
        }
        if let Some(p) = self.pack {
            parts.push(format!("pack={}", if p { "yes" } else { "no" }));
        }
        if let Some(n) = self.passes {
            parts.push(format!("passes={n}"));
        }
        parts.join(",")
    }

    /// Reads a layout back from its [`key`](Layout::key), as a calibration
    /// sample records it (0.8.0 step 4, `presk refit --recost`), each value
    /// checked against its domain as [`from_json`](Layout::from_json) checks
    /// it.
    pub fn from_key(key: &str) -> Result<Layout, String> {
        // A split's value holds a comma, `(0, 5)`: a part without `=`
        // continues the one before.
        let mut parts: Vec<String> = Vec::new();
        for p in key.split(',').filter(|p| !p.is_empty()) {
            match parts.last_mut() {
                Some(last) if !p.contains('=') => {
                    last.push(',');
                    last.push_str(p);
                }
                _ => parts.push(p.to_string()),
            }
        }
        let mut m = Map::new();
        for p in &parts {
            let bad = || format!("`{p}` in the layout key `{key}`");
            let Some((name, value)) = p.split_once('=') else { return Err(bad()) };
            let n = |s: &str| s.trim().parse::<u64>().map_err(|_| bad());
            let v = match (name, value) {
                ("form" | "order", _) | ("kblock" | "split", "none") => Json::String(value.into()),
                ("tile", _) => {
                    let (mi, nj) = value.split_once('x').ok_or_else(bad)?;
                    Json::from(vec![n(mi)?, n(nj)?])
                }
                ("kblock" | "lanes" | "passes", _) => Json::from(n(value)?),
                ("split", _) => {
                    let (d, w) = value.strip_prefix('(').and_then(|v| v.strip_suffix(')')).and_then(|v| v.split_once(',')).ok_or_else(bad)?;
                    serde_json::json!({ "depth": n(d)?, "workers": n(w)? })
                }
                ("pack", "yes" | "no") => Json::Bool(value == "yes"),
                _ => return Err(bad()),
            };
            m.insert(name.to_string(), v);
        }
        Layout::from_json(&Json::Object(m))
    }
}

/// `tiled, tile [4, 16], kblock 32, order JIK, lanes 4, split (1, 10), pack`,
/// or `default` when nothing is stated.
impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if let Some(x) = self.form {
            parts.push(form_name(x).to_string());
        }
        if let Some((mi, nj)) = self.tile {
            parts.push(format!("tile [{mi}, {nj}]"));
        }
        if let Some(kb) = self.kblock {
            parts.push(format!("kblock {}", kblock_name(kb)));
        }
        if let Some(o) = &self.order {
            parts.push(format!("order {}", order_name(o)));
        }
        if let Some(n) = self.lanes {
            parts.push(format!("lanes {n}"));
        }
        if let Some(s) = self.split {
            parts.push(format!("split {}", split_name(s)));
        }
        if let Some(p) = self.pack {
            parts.push(if p { "pack".into() } else { "no pack".into() });
        }
        if let Some(n) = self.passes {
            parts.push(format!("passes {n}"));
        }
        if parts.is_empty() { f.write_str("default") } else { f.write_str(&parts.join(", ")) }
    }
}

fn form_name(f: Form) -> &'static str {
    match f {
        Form::RowWise => "row-wise",
        Form::Tiled => "tiled",
    }
}

fn kblock_name(kb: KBlock) -> String {
    match kb {
        KBlock::None => "none".into(),
        KBlock::Width(b) => b.to_string(),
    }
}

fn order_name(o: &[BlockLoop]) -> String {
    o.iter()
        .map(|b| match b {
            BlockLoop::I => 'I',
            BlockLoop::J => 'J',
            BlockLoop::K => 'K',
        })
        .collect()
}

fn split_name(s: Split) -> String {
    match s {
        Split::None => "none".into(),
        Split::At { depth, workers } => format!("({depth}, {workers})"),
    }
}

/// A realization plan (reference §6.6): the realization, the machine profile
/// it was chosen against, and the layout of the nests it states one for.
///
/// Not to be confused with `presk_realize::Plan`, the fusion groups of one
/// function, which is structural and part of no file.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub realization: Realization,
    /// How the file spelled its realization, when it named one (`"f32"`,
    /// `"@out/solved.json"`) rather than writing the record in place; kept so
    /// that a plan written back names the same file.
    pub realization_ref: Option<String>,
    /// The profile the plan was chosen against. Advisory elsewhere
    /// (question 64); not read by any command yet.
    pub profile: Option<String>,
    /// Layouts by nest, `"<function>/%<let>"`, as the realization's sites.
    pub layout: BTreeMap<String, Layout>,
}

impl Plan {
    /// The plan `--realize` means: this realization, no profile, the default
    /// layout everywhere. So no command line of 0.1.0–0.6.0 changes meaning.
    pub fn from_realization(r: Realization) -> Plan {
        Plan { realization: r, realization_ref: None, profile: None, layout: BTreeMap::new() }
    }

    /// `@FILE`: a plan file.
    pub fn parse(spec: &str) -> Result<Plan, String> {
        let Some(path) = spec.strip_prefix('@') else {
            return Err(format!("--plan takes @FILE, a plan file, not `{spec}`"));
        };
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let v: Json = serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
        Plan::from_json(&v).map_err(|m| format!("{path}: {m}"))
    }

    /// Reads a plan record: `realization` (a realization record, the name of
    /// a uniform realization, or `@FILE`), `profile`, `layout`.
    pub fn from_json(v: &Json) -> Result<Plan, String> {
        let Json::Object(m) = v else { return Err("a plan is a record with the members realization, profile and layout".into()) };
        if let Some(k) = m.keys().find(|k| !["realization", "profile", "layout"].contains(&k.as_str())) {
            return Err(format!("unknown plan member `{k}`; the members are realization, profile and layout"));
        }
        let (realization, realization_ref) = match m.get("realization") {
            None => return Err("a plan names its realization: the member `realization` is required".into()),
            Some(Json::String(s)) => (Realization::parse(s).map_err(|e| format!("realization: {e}"))?, Some(s.clone())),
            Some(r @ Json::Object(_)) => (serde_json::from_value(r.clone()).map_err(|e| format!("realization: {e}"))?, None),
            Some(other) => return Err(format!("realization: a record, a name or @FILE, not {other}")),
        };
        let profile = match m.get("profile") {
            None => None,
            Some(Json::String(s)) => Some(s.clone()),
            Some(other) => return Err(format!("profile: a file name, not {other}")),
        };
        let mut layout = BTreeMap::new();
        match m.get("layout") {
            None => {}
            Some(Json::Object(nests)) => {
                for (nest, record) in nests {
                    if !nest.contains("/%") {
                        return Err(format!("layout: `{nest}` is not a nest name, `<function>/%<let>`"));
                    }
                    layout.insert(nest.clone(), Layout::from_json(record).map_err(|e| format!("layout of `{nest}`: {e}"))?);
                }
            }
            Some(other) => return Err(format!("layout: a record of nests, not {other}")),
        }
        Ok(Plan { realization, realization_ref, profile, layout })
    }

    pub fn to_json(&self) -> String {
        let mut m = Map::new();
        m.insert("realization".into(), match &self.realization_ref {
            Some(s) => Json::String(s.clone()),
            None => serde_json::to_value(&self.realization).expect("a realization serializes"),
        });
        if let Some(p) = &self.profile {
            m.insert("profile".into(), Json::String(p.clone()));
        }
        if !self.layout.is_empty() {
            m.insert("layout".into(), Json::Object(self.layout.iter().map(|(n, l)| (n.clone(), l.to_json())).collect()));
        }
        serde_json::to_string_pretty(&Json::Object(m)).expect("a plan serializes")
    }

    /// The layout the plan states for a nest; the empty layout, every
    /// decision at its default, for a nest it does not mention.
    pub fn layout_at(&self, nest: &str) -> Layout {
        self.layout.get(nest).cloned().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(j: usize, accum_is_storage: bool, width: usize) -> NestKind {
        NestKind::Contraction(Contraction { dot: true, i: 256, j, k: 384, nt: false, accum_is_storage, width })
    }

    fn scores() -> NestKind {
        NestKind::Contraction(Contraction { dot: false, i: 256, j: 256, k: 64, nt: true, accum_is_storage: true, width: 4 })
    }

    fn layout(text: &str) -> Result<Layout, String> {
        Layout::from_json(&serde_json::from_str(text).unwrap())
    }

    #[test]
    fn a_layout_round_trips_and_keys_canonically() {
        let text = r#"{ "passes": 2, "tile": [4, 16], "form": "tiled", "kblock": 32, "order": "JIK", "lanes": 4,
                        "split": { "workers": 10, "depth": 1 }, "pack": true }"#;
        let l = layout(text).unwrap();
        assert_eq!(Layout::from_json(&l.to_json()).unwrap(), l);
        assert_eq!(l.key(), "form=tiled,tile=4x16,kblock=32,order=JIK,lanes=4,split=(1, 10),pack=yes,passes=2");
        assert_eq!(l.to_string(), "tiled, tile [4, 16], kblock 32, order JIK, lanes 4, split (1, 10), pack, passes 2");
        assert_eq!(Layout::default().key(), "");
        assert_eq!(Layout::default().to_string(), "default");
        // A key reads back as the layout it is of.
        assert_eq!(Layout::from_key(&l.key()).unwrap(), l);
        assert_eq!(Layout::from_key("").unwrap(), Layout::default());
        let calib = "form=tiled,tile=1x4,kblock=none,order=IJ,lanes=1,split=none";
        assert_eq!(Layout::from_key(calib).unwrap().key(), calib);
        assert!(Layout::from_key("tile=3x4").is_err() && Layout::from_key("split=(1 10)").is_err() && Layout::from_key("pack=maybe").is_err());
    }

    #[test]
    fn values_outside_the_domains_are_refused() {
        for bad in [
            r#"{ "tile": [3, 8] }"#,
            r#"{ "tile": [1, 32] }"#,
            r#"{ "tile": [1] }"#,
            r#"{ "kblock": 64 }"#,
            r#"{ "order": "IJJ" }"#,
            r#"{ "order": "IK" }"#,
            r#"{ "lanes": 0 }"#,
            r#"{ "split": { "depth": 3, "workers": 10 } }"#,
            r#"{ "split": { "depth": 0, "workers": 0 } }"#,
            r#"{ "passes": 1 }"#,
            r#"{ "tiles": [1, 8] }"#,
            r#"{ "form": "row-wise", "tile": [1, 8] }"#,
            r#"{ "kblock": 32, "order": "IJ" }"#,
            r#"{ "kblock": "none", "order": "IKJ" }"#,
        ] {
            assert!(layout(bad).is_err(), "{bad} was accepted");
        }
        assert_eq!(layout(r#"{ "tile": [1, 1] }"#).unwrap().tile, Some((1, 1)));
    }

    #[test]
    fn the_defaults_are_the_kernels_0_6_0_lowers_where_they_realize_their_plan() {
        // A `dot` whose J is a multiple of eight: blocked, eight columns, lanes.
        let d = Layout::default_for(&dot(384, true, 4));
        assert_eq!((d.form, d.tile, d.kblock, d.lanes, d.pack), (Some(Form::Tiled), Some((1, 8)), Some(KBlock::None), Some(4), Some(false)));
        // Any other `dot`, row-wise and scalar ...
        assert_eq!(Layout::default_for(&dot(92, true, 4)).form, Some(Form::RowWise));
        // ... unless it would keep bf16 partial sums (D56): then one accumulator per element.
        let b = Layout::default_for(&dot(92, false, 1));
        assert_eq!((b.form, b.tile, b.lanes), (Some(Form::Tiled), Some((1, 1)), Some(1)));
        // An `ein` is the naive nest: the tile (1, 1), one lane.
        let e = Layout::default_for(&scores());
        assert_eq!((e.form, e.tile, e.lanes), (Some(Form::Tiled), Some((1, 1)), Some(1)));
        assert_eq!(Layout::default_for(&NestKind::Softmax { narrow: false }).passes, Some(3));
        for kind in [dot(384, true, 4), dot(92, true, 4), dot(92, false, 1), scores(), NestKind::Softmax { narrow: true }, NestKind::Other] {
            assert_eq!(Layout::default().check(&kind), Ok(()), "{kind:?}: the default must be well-formed");
        }
    }

    #[test]
    fn the_conditions_are_checked_at_the_shapes() {
        let ok = |k: &NestKind, t: &str| layout(t).unwrap().check(k);
        assert!(ok(&scores(), r#"{ "tile": [4, 16], "kblock": 32, "order": "JIK", "lanes": 4, "pack": true }"#).is_ok());
        // Condition 3: divisibility, and tile decisions on a row-wise nest.
        assert!(ok(&scores(), r#"{ "kblock": 128 }"#).unwrap_err().contains("condition 3"));
        assert!(ok(&dot(92, true, 4), r#"{ "tile": [1, 8] }"#).unwrap_err().contains("condition 3"));
        // Condition 4: the register's width, dividing n_j; one on an ein's default tile.
        assert!(ok(&dot(384, true, 4), r#"{ "lanes": 2 }"#).unwrap_err().contains("condition 4"));
        assert!(ok(&scores(), r#"{ "lanes": 4 }"#).unwrap_err().contains("condition 4"));
        assert!(ok(&dot(384, true, 4), r#"{ "lanes": 1 }"#).is_ok());
        // Condition 7: a spill, the row-wise form and two passes need storage = accumulation.
        assert!(ok(&scores(), r#"{ "kblock": 32, "order": "IKJ" }"#).is_ok());
        let narrow = NestKind::Contraction(Contraction { dot: false, i: 256, j: 256, k: 64, nt: true, accum_is_storage: false, width: 1 });
        assert!(ok(&narrow, r#"{ "kblock": 32, "order": "IKJ" }"#).unwrap_err().contains("condition 7"));
        assert!(ok(&narrow, r#"{ "kblock": 32, "order": "IJK" }"#).is_ok());
        assert!(ok(&dot(92, false, 1), r#"{ "form": "row-wise" }"#).unwrap_err().contains("condition 7"));
        assert!(ok(&NestKind::Softmax { narrow: true }, r#"{ "passes": 2 }"#).unwrap_err().contains("condition 7"));
        // Condition 9: a pack on an NN nest.
        assert!(ok(&dot(384, true, 4), r#"{ "pack": true }"#).unwrap_err().contains("condition 9"));
        // Decisions of another kind.
        assert!(ok(&NestKind::Other, r#"{ "tile": [1, 8] }"#).is_err());
        assert!(ok(&dot(384, true, 4), r#"{ "passes": 2 }"#).is_err());
        assert!(ok(&NestKind::Other, r#"{ "split": "none" }"#).is_ok());
    }

    #[test]
    fn the_space_holds_the_default_and_only_well_formed_points() {
        let values = NestKind::Contraction(Contraction { dot: false, i: 256, j: 64, k: 256, nt: false, accum_is_storage: true, width: 4 });
        let space = Layout::space(&values);
        assert_eq!(space[0], Layout::default_for(&values));
        assert!(space.iter().all(|l| l.check(&values).is_ok()));
        // Row-wise; nine tiles by fourteen (order, k-block) by two lane counts,
        // and the tile (1, 1) by fourteen in one lane; no pack on an NN nest.
        assert_eq!(space.len(), 1 + 9 * 14 * 2 + 14);
        // The NT scores: k-block 128 does not divide 64, and lanes need the pack.
        let scores = NestKind::Contraction(Contraction { dot: false, i: 256, j: 256, k: 64, nt: true, accum_is_storage: true, width: 4 });
        let s = Layout::space(&scores);
        assert_eq!(s.len(), 1 + 10 * 8 * 2 + 9 * 8);
        assert!(s.iter().all(|l| l.lanes == Some(1) || l.pack == Some(true)));
        // bf16: no spill, no row-wise.
        let narrow = NestKind::Contraction(Contraction { dot: false, i: 256, j: 64, k: 256, nt: false, accum_is_storage: false, width: 1 });
        let n = Layout::space(&narrow);
        assert_eq!(n.len(), 10 * (2 + 2 * 2));
        assert_eq!(Layout::space(&NestKind::Softmax { narrow: false }).len(), 2);
        assert_eq!(Layout::space(&NestKind::Softmax { narrow: true }).len(), 1);
    }

    #[test]
    fn a_plan_reads_its_realization_three_ways() {
        let named: Json = serde_json::from_str(r#"{ "realization": "f32", "layout": { "f/%3": { "lanes": 1 } } }"#).unwrap();
        let p = Plan::from_json(&named).unwrap();
        assert_eq!(p.realization, Realization::uniform(crate::Dtype::F32));
        assert_eq!(p.layout_at("f/%3").lanes, Some(1));
        assert!(p.layout_at("f/%4").is_empty());
        let inline: Json = serde_json::from_str(r#"{ "realization": { "default": "bf16", "order": { "kind": "sequential" } } }"#).unwrap();
        assert_eq!(Plan::from_json(&inline).unwrap().realization, Realization::uniform(crate::Dtype::Bf16));
        assert_eq!(Plan::from_json(&serde_json::from_str(&p.to_json()).unwrap()).unwrap(), p);
        for bad in [r#"{ "layout": {} }"#, r#"{ "realization": "f16" }"#, r#"{ "realization": "f32", "layout": { "f3": {} } }"#, r#"{ "realization": "f32", "extra": 1 }"#] {
            assert!(Plan::from_json(&serde_json::from_str(bad).unwrap()).is_err(), "{bad} was accepted");
        }
        assert!(Plan::parse("plan.json").unwrap_err().contains("@FILE"));
    }
}
