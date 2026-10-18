//! The diagram as text: every box's graph, its nodes then its arrows, each
//! with the element it names and its numbers, and the rectangles of the
//! default opening, composed. What `presk dump --stage network` prints, and what the
//! snapshot tests keep.

use std::fmt::Write;

use crate::layout::{View, numbers, title, view};
use crate::{Kind, Network};

/// The outline of `n`.
pub fn outline(n: &Network) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "network of `{}`, read off the stage `{}`: {} nodes, {} arrows, {} boxes", n.entry, n.stage.name(), n.nodes.len(), n.edges.len(), n.frames.len() - 1);
    if let Some(p) = &n.params {
        let _ = writeln!(s, "parameters: {} ({} bytes, {})", p.count, p.bytes, p.types.join(", "));
    }
    if let (Some(w), Some(q)) = (n.work, n.bytes) {
        let _ = writeln!(s, "work W = {w}, bytes Q = {q}, over every launch of the entry");
    }
    let d = view(n, &n.default);
    let _ = writeln!(s, "default opening: {:?}, {:.0} × {:.0} pt", d.open, d.width, d.height);
    box_(&mut s, n, &d, 0, 0);
    s
}

fn box_(s: &mut String, n: &Network, d: &View, b: usize, depth: usize) {
    let pad = "  ".repeat(depth);
    let x = &n.nodes[b];
    let at = d.rects.get(&b).map(|r| format!(" @ {:.0},{:.0} {:.0}×{:.0}", r.x, r.y, r.w, r.h)).unwrap_or_default();
    let _ = writeln!(s, "{pad}[{}] {} {}  {}{}{}", x.id, x.kind.name(), title(x), x.element, x.function.as_ref().filter(|f| **f != x.element).map(|f| format!(" fn {f}")).unwrap_or_default(), at);
    let line = numbers(x);
    if !line.is_empty() {
        let _ = writeln!(s, "{pad}    {line}{}", if x.runs > 1 { format!(" (per run; {} runs)", x.runs) } else { String::new() });
    }
    if let Some((kind, of)) = &x.generated {
        let _ = writeln!(s, "{pad}    generated: {kind} of {of}");
    }
    for c in &x.children {
        let k = &n.nodes[*c];
        if k.kind.is_box() {
            box_(s, n, d, *c, depth + 1);
            continue;
        }
        let at = d.rects.get(c).map(|r| format!(" @ {:.0},{:.0} {:.0}×{:.0}", r.x, r.y, r.w, r.h)).unwrap_or_default();
        let what = match k.kind {
            Kind::Input => format!("input {} #{}", k.label, k.index.unwrap_or(0)),
            Kind::Output => "result".into(),
            Kind::Param => format!("param θ.{}", k.label),
            _ => format!("{} {}", k.kind.name(), k.label),
        };
        let _ = writeln!(s, "{pad}  [{}] {what}  {}{}{}", k.id, k.element, if k.key { " (key)" } else { "" }, at);
        let line = numbers(k);
        if !line.is_empty() && k.kind != Kind::Host {
            let _ = writeln!(s, "{pad}      {line}");
        }
        if k.members.len() > 1 {
            let _ = writeln!(s, "{pad}      members {}", k.members.iter().map(|m| m.key.clone()).collect::<Vec<_>>().join(", "));
        }
    }
    for e in n.edges_in(b) {
        let _ = writeln!(s, "{pad}  {} → {}{}: {}  {} = {}, {}, {} elements, {} B{}", e.from, e.to, e.arg.map(|a| format!(" #{a}")).unwrap_or_default(), e.site, e.ty, e.at, e.dtype, e.elems, e.bytes, if e.key { " (key)" } else { "" });
    }
}
