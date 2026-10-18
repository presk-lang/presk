//! One view of the diagram as a self-contained SVG: the positions of the
//! layout, the formulas as the Unicode printer's strings (an SVG has no
//! KaTeX), the arrows coloured by their realized type.

use std::fmt::Write;

use crate::layout::{CHAR, LINE, Rect, View, numbers, title};
use crate::{Kind, Network};

/// The colour of a realized type, as the page's variables give it.
pub fn colour(dtype: &str) -> &'static str {
    match dtype {
        "f64" => "#3567b5",
        "f32" => "#2e8b57",
        "bf16" => "#c0392b",
        "idx" | "bool" => "#8e6c2a",
        "key" => "#a0a0a0",
        "params" => "#7b52ab",
        _ => "#606060",
    }
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// An arrow's width, by the log of its bytes.
pub fn width(bytes: u64) -> f64 {
    if bytes == 0 { 0.8 } else { (0.8 + (bytes as f64).log10() * 0.35).clamp(0.8, 4.0) }
}

/// A route as a path: each segment a curve leaving and arriving vertically.
fn path(pts: &[(f64, f64)]) -> String {
    let mut d = String::new();
    for (k, (x, y)) in pts.iter().enumerate() {
        if k == 0 {
            let _ = write!(d, "M{x:.1},{y:.1}");
        } else {
            let (px, py) = pts[k - 1];
            let my = (py + y) / 2.0;
            let _ = write!(d, " C{px:.1},{my:.1} {x:.1},{my:.1} {x:.1},{y:.1}");
        }
    }
    d
}

/// The view `v` of `net`, with the keys drawn when `keys`.
pub fn svg(net: &Network, v: &View, keys: bool) -> String {
    let mut s = String::new();
    let (w, h) = (v.width + 2.0, v.height + 2.0);
    let _ = writeln!(s, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w:.0}\" height=\"{h:.0}\" viewBox=\"-1 -1 {w:.0} {h:.0}\" font-family=\"DejaVu Sans Mono, Menlo, monospace\" font-size=\"11\">");
    let _ = writeln!(s, "<defs><marker id=\"a\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerUnits=\"userSpaceOnUse\" markerWidth=\"8\" markerHeight=\"8\" orient=\"auto-start-reverse\"><path d=\"M0,0 L10,5 L0,10 z\" fill=\"context-stroke\"/></marker></defs>");
    let _ = writeln!(s, "<rect x=\"0\" y=\"0\" width=\"{:.1}\" height=\"{:.1}\" fill=\"#ffffff\"/>", v.width, v.height);
    // The boxes first, outermost first, so that what is inside is on top.
    let mut ids: Vec<usize> = v.rects.keys().copied().collect();
    ids.sort_by_key(|i| (net.nodes[*i].depth, *i));
    for id in &ids {
        let n = &net.nodes[*id];
        if !keys && n.key {
            continue;
        }
        let r = v.rects[id];
        let open = *id == 0 || v.open.contains(id);
        node(&mut s, n, r, open, net.memory.get(id).copied());
    }
    for (id, pts) in &v.routes {
        let e = &net.edges[*id];
        if !keys && e.key {
            continue;
        }
        let dash = if e.key { " stroke-dasharray=\"3,3\"" } else { "" };
        let _ = writeln!(
            s,
            "<path d=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{:.2}\"{dash} marker-end=\"url(#a)\"><title>{}: {}, {}, {} B</title></path>",
            path(pts),
            colour(&e.dtype),
            width(e.bytes),
            esc(&e.site.key),
            esc(&e.at),
            e.dtype,
            e.bytes
        );
    }
    s.push_str("</svg>\n");
    s
}

/// `#rrggbb` mixed `t` of the way towards the memory colour.
fn mem_mix(base: (u8, u8, u8), t: f64) -> String {
    let (m0, m1, m2) = (0xc2u8, 0x18u8, 0x5bu8);
    let mix = |a: u8, b: u8| ((a as f64) + ((b as f64) - (a as f64)) * t).round() as u8;
    format!("#{:02x}{:02x}{:02x}", mix(base.0, m0), mix(base.1, m1), mix(base.2, m2))
}

fn node(s: &mut String, n: &crate::Node, r: Rect, open: bool, share: Option<f64>) {
    let text = |s: &mut String, x: f64, y: f64, t: &str, extra: &str| {
        let _ = writeln!(s, "<text x=\"{x:.1}\" y=\"{y:.1}\"{extra}>{}</text>", esc(t));
    };
    match n.kind {
        Kind::Root | Kind::Call | Kind::Stack => {
            let fill = match (n.kind == Kind::Root, open, share) {
                (true, _, _) => "#ffffff".to_string(),
                // The memory's scale, as the page draws it: a folded box
                // takes it, an open one a light tint (plan 0.15.0 step 4).
                (_, true, Some(f)) => mem_mix((0xf7, 0xf8, 0xfb), 0.03 + 0.09 * f),
                (_, false, Some(f)) => mem_mix((0xee, 0xf2, 0xf8), 0.06 + 0.64 * f),
                (_, true, None) => "#f7f8fb".to_string(),
                (_, false, None) => "#eef2f8".to_string(),
            };
            let stroke = n.params.as_ref().and_then(|p| p.types.first()).map_or("#555555", |t| colour(t));
            if n.kind == Kind::Stack && !open {
                let _ = writeln!(s, "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" rx=\"6\" fill=\"{fill}\" stroke=\"{stroke}\"/>", r.x + 4.0, r.y + 4.0, r.w, r.h);
            }
            if n.kind != Kind::Root {
                let _ = writeln!(s, "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" rx=\"6\" fill=\"{fill}\" stroke=\"{stroke}\" stroke-width=\"1.4\"><title>{}</title></rect>", r.x, r.y, r.w, r.h, esc(&n.element.to_string()));
            }
            let x = r.x + 8.0;
            text(s, x, r.y + LINE, &title(n), " font-weight=\"bold\"");
            if let (Some(f), false) = (share, n.kind == Kind::Root) {
                text(s, r.x + r.w - 8.0, r.y + LINE, &format!("{:.0} %", 100.0 * f), " font-weight=\"bold\" text-anchor=\"end\"");
            }
            text(s, x, r.y + 2.0 * LINE, &numbers(n), " fill=\"#555555\" font-size=\"10\"");
            if !open && let Some(t) = &n.text {
                let longest = t.lines().map(|l| l.chars().count()).max().unwrap_or(1) as f64;
                let room = r.w - 16.0;
                let scale = (room / (longest * CHAR)).min(1.0);
                for (k, l) in t.lines().enumerate() {
                    let y = r.y + 2.0 * LINE + 6.0 + (k as f64 + 1.0) * LINE;
                    let extra = if scale < 1.0 { format!(" textLength=\"{:.1}\" lengthAdjust=\"spacingAndGlyphs\"", l.chars().count() as f64 * CHAR * scale) } else { String::new() };
                    text(s, x, y, l, &extra);
                }
            }
        }
        Kind::Group | Kind::Op | Kind::Host => {
            let (fill, dash) = match n.kind {
                Kind::Group => ("#fff4e0", ""),
                Kind::Op => ("#fffaf0", ""),
                _ => ("#f4f4f4", " stroke-dasharray=\"2,2\""),
            };
            let _ = writeln!(s, "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" rx=\"{:.1}\" fill=\"{fill}\" stroke=\"#9a7b4f\"{dash}><title>{}</title></rect>", r.x, r.y, r.w, r.h, r.h / 2.0, esc(n.text.as_deref().unwrap_or("")));
            text(s, r.x + r.w / 2.0, r.y + r.h / 2.0 + 4.0, &n.label, " text-anchor=\"middle\"");
        }
        Kind::Param => {
            let _ = writeln!(s, "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" rx=\"3\" fill=\"#f3eefa\" stroke=\"#7b52ab\"/>", r.x, r.y, r.w, r.h);
            text(s, r.x + 6.0, r.y + 14.0, &format!("θ.{}", n.label), "");
            let line = numbers(n);
            if !line.is_empty() {
                text(s, r.x + 6.0, r.y + 14.0 + LINE, &line, " fill=\"#555555\" font-size=\"10\"");
            }
        }
        Kind::Input | Kind::Output => {
            let _ = writeln!(s, "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" rx=\"11\" fill=\"#ffffff\" stroke=\"#888888\"/>", r.x, r.y, r.w, r.h);
            text(s, r.x + r.w / 2.0, r.y + 15.0, &n.label, " text-anchor=\"middle\" fill=\"#444444\"");
        }
    }
}
