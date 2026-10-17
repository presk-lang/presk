//! The diagram as JSON: the `network` object of `explain.json`'s format 2,
//! which `presk dump --stage network --format json` prints too. Written by
//! hand, the crate having no dependency; deterministic, every number of a
//! position rounded to a tenth of a point.
//!
//! ```text
//! { "entry", "stage", "default", "params": {count, bytes, types} | null,
//!   "work", "bytes",
//!   "nodes": [{ "id", "kind", "parent", "element", "function", "callee",
//!               "label", "title", "numbers", "instance", "stack", "runs",
//!               "path", "tex", "text", "sum", "order", "params", "work",
//!               "bytes", "generated": {kind, of} | null, "members", "index",
//!               "children", "depth", "key" }],
//!   "edges": [{ "id", "from", "to", "parent", "arg", "site", "ty", "at",
//!               "dtype", "elems", "bytes", "key" }],
//!   "views": [{ "open": [ids], "width", "height",
//!               "rects": [[id, x, y, w, h]], "routes": [[id, x0, y0, x1, y1, …]] }] }
//! ```

use std::fmt::Write;

use crate::layout::{numbers, title};
use crate::{Network, Params};

/// A JSON string.
pub fn string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn opt_str(s: &Option<String>) -> String {
    s.as_deref().map_or_else(|| "null".into(), string)
}

/// A number: an integer as one, else its shortest form; `null` when it is
/// not finite.
pub fn number(x: f64) -> String {
    if !x.is_finite() {
        "null".into()
    } else if x == x.trunc() && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}

fn opt_num(x: Option<f64>) -> String {
    x.map_or_else(|| "null".into(), number)
}

/// A coordinate, to a tenth.
fn coord(x: f64) -> String {
    number((x * 10.0).round() / 10.0)
}

fn params(p: &Option<Params>) -> String {
    match p {
        None => "null".into(),
        Some(p) => format!("{{\"count\":{},\"bytes\":{},\"types\":[{}]}}", p.count, p.bytes, p.types.iter().map(|t| string(t)).collect::<Vec<_>>().join(",")),
    }
}

/// The diagram as one JSON object.
pub fn network(n: &Network) -> String {
    let mut s = String::new();
    let _ = write!(s, "{{\"entry\":{},\"stage\":{},\"default\":{},\"params\":{},\"work\":{},\"bytes\":{},\"nodes\":[", string(&n.entry), string(n.stage.name()), n.default, params(&n.params), opt_num(n.work), opt_num(n.bytes));
    for (k, x) in n.nodes.iter().enumerate() {
        if k > 0 {
            s.push(',');
        }
        let generated = match &x.generated {
            Some((kind, of)) => format!("{{\"kind\":{},\"of\":{}}}", string(kind), string(&of.to_string())),
            None => "null".into(),
        };
        let _ = write!(
            s,
            "{{\"id\":{},\"kind\":{},\"parent\":{},\"element\":{},\"function\":{},\"callee\":{},\"label\":{},\"title\":{},\"numbers\":{},\"instance\":{},\"stack\":{},\"runs\":{},\"path\":{},\"tex\":{},\"text\":{},\"sum\":{},\"order\":{},\"params\":{},\"work\":{},\"bytes\":{},\"generated\":{},\"members\":[{}],\"index\":{},\"children\":[{}],\"depth\":{},\"key\":{}}}",
            x.id,
            string(x.kind.name()),
            x.parent.map_or_else(|| "null".into(), |p| p.to_string()),
            string(&x.element.to_string()),
            x.function.as_ref().map_or_else(|| "null".into(), |f| string(&f.to_string())),
            opt_str(&x.callee),
            string(&x.label),
            string(&title(x)),
            string(&numbers(x)),
            opt_str(&x.instance),
            x.stack.map_or_else(|| "null".into(), |l| l.to_string()),
            x.runs,
            opt_str(&x.path),
            opt_str(&x.tex),
            opt_str(&x.text),
            opt_str(&x.sum),
            opt_str(&x.order),
            params(&x.params),
            opt_num(x.work),
            opt_num(x.bytes),
            generated,
            x.members.iter().map(|m| string(&m.to_string())).collect::<Vec<_>>().join(","),
            x.index.map_or_else(|| "null".into(), |i| i.to_string()),
            x.children.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(","),
            x.depth,
            x.key,
        );
    }
    s.push_str("],\"edges\":[");
    for (k, e) in n.edges.iter().enumerate() {
        if k > 0 {
            s.push(',');
        }
        let _ = write!(
            s,
            "{{\"id\":{},\"from\":{},\"to\":{},\"parent\":{},\"arg\":{},\"site\":{},\"ty\":{},\"at\":{},\"dtype\":{},\"elems\":{},\"bytes\":{},\"key\":{}}}",
            e.id,
            e.from,
            e.to,
            e.parent,
            e.arg.map_or_else(|| "null".into(), |a| a.to_string()),
            string(&e.site.to_string()),
            string(&e.ty),
            string(&e.at),
            string(&e.dtype),
            e.elems,
            e.bytes,
            e.key,
        );
    }
    s.push_str("],\"views\":[");
    for (k, v) in n.views.iter().enumerate() {
        if k > 0 {
            s.push(',');
        }
        let rects: Vec<String> = v.rects.iter().map(|(id, r)| format!("[{id},{},{},{},{}]", coord(r.x), coord(r.y), coord(r.w), coord(r.h))).collect();
        let routes: Vec<String> = v.routes.iter().map(|(id, pts)| format!("[{id},{}]", pts.iter().map(|(x, y)| format!("{},{}", coord(*x), coord(*y))).collect::<Vec<_>>().join(","))).collect();
        let _ = write!(s, "{{\"open\":[{}],\"width\":{},\"height\":{},\"rects\":[{}],\"routes\":[{}]}}", v.open.iter().map(|o| o.to_string()).collect::<Vec<_>>().join(","), coord(v.width), coord(v.height), rects.join(","), routes.join(","));
    }
    s.push_str("]}");
    s
}
