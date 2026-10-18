//! `presk explain --format json` (plan 0.13.0 step 2, D135): the document's
//! facts as values, the provenance table with them, written as
//! `explain.json` beside `explain.md`.
//!
//! The JSON holds three things:
//! - **the document**, its header and its sections as values (prose, tables,
//!   code, notes), the same strings the Markdown prints, so that the
//!   Markdown is rendered again from it byte for byte;
//! - **the provenance table**: every stage's elements, each with its label
//!   and the facts a reader is shown on it (the envelope's magnitude, the
//!   sensitivity, the type and its term, the layout chosen, the kernel's
//!   text, the C function's lines), and the links and ends;
//! - **the sources**, each file's text, which the spans of the source stage
//!   index.
//!
//! It is deterministic without `--measure`, as the Markdown is: no time, no
//! absolute path, the compiler's version in one field. `format` is the
//! version of this layout: a reader refuses a format it does not know.

use serde_json::{Map, Value, json};

use std::collections::{BTreeMap, HashMap};

use presk_print_math::Format;
use presk_provenance::{How, Id, Stage};

use super::State;
use crate::doc::{Block, Document, Section};

/// The layout's version, `format` in the file: 2 since plan 0.14.0 step 1
/// (D141), which added `network`; 3 since plan 0.14.1 (D146), its frames
/// in place of its views.
pub const FORMAT: u64 = 3;

/// A number in LaTeX, three significant digits: `3.12 \cdot 10^{-2}`.
fn tex_num(x: f64) -> String {
    if x == 0.0 || !x.is_finite() {
        return format!("{x}");
    }
    let e = x.abs().log10().floor() as i32;
    if (-2..4).contains(&e) {
        return format!("{}", (x * 10f64.powi(2 - e)).round() / 10f64.powi(2 - e));
    }
    let m = x / 10f64.powi(e);
    let m = (m * 100.0).round() / 100.0;
    if m == 1.0 { format!("10^{{{e}}}") } else { format!("{m} \\cdot 10^{{{e}}}") }
}

/// A type's unit roundoff, `u_τ`, as a power of two.
fn unit(ty: &str) -> &'static str {
    match ty {
        "f64" => "2^{-53}",
        "f32" => "2^{-24}",
        "bf16" => "2^{-8}",
        _ => "u",
    }
}

/// What the page writes in mathematics, computed once per document: the
/// typed program's displays and formulas, by name and by span.
pub(crate) struct Math {
    displays: HashMap<String, String>,
    formulas: HashMap<String, String>,
    /// The listing itself, which the diagram's boxes read.
    listing: presk_print_math::Listing,
}

impl Math {
    fn new(s: &State) -> Math {
        let l = presk_print_math::typed_program(&s.sm, &s.prog, &s.typed);
        let mut displays = HashMap::new();
        for u in &l.units {
            for i in &u.items {
                let ds: Vec<&presk_print_math::Display> = match i {
                    presk_print_math::typed::Item::Fn(d) => vec![d],
                    presk_print_math::typed::Item::Module { fns, .. } => fns.iter().collect(),
                };
                for d in ds {
                    displays.insert(d.name.clone(), presk_print_math::aligned(d));
                }
            }
        }
        let formulas = l.formulas.iter().map(|(sp, m)| (presk_driver::provenance::span_key(*sp), m.render(Format::Latex))).collect();
        Math { displays, formulas, listing: l }
    }
}

/// The network diagram (plan 0.14.0, D141): the graph of the program the
/// kernels run, a view of the table, with its numbers and its layout.
/// `None` when the entry does not resolve.
pub(crate) fn network(s: &State, listing: &presk_print_math::Listing) -> Option<presk_diagram::Network> {
    let (program, stage) = if s.before_opt.is_some() { (&s.xcore, Stage::Opt) } else { (&s.ad_program, Stage::Ad) };
    // The entry as `presk run` binds it: its dimensions, and the `--dim`s it
    // does not have as its open prefix.
    let opts = presk_interp::RunOptions { entry: s.o.entry.clone(), dims: s.dims.clone(), seed: 0, bits: 64, digits: 12 };
    let (f, dims, prefix) = presk_interp::run::bind_run(program, &s.prog, &opts).ok()?;
    let prefix: Vec<(Option<String>, u64)> = prefix.iter().map(|a| (a.name.clone(), a.len as u64)).collect();
    let kernels = s.hosted.as_ref().ok().map(|h| (h.listing.kernels.as_slice(), h.launched.as_slice()));
    let i = presk_diagram::Inputs { prog: &s.prog, listing: Some(listing), program, stage, table: &s.table, entry: &program.func(f).name, dims: &dims, prefix: &prefix, realization: &s.realization, kernels };
    presk_diagram::network(&i)
}

/// The bound by the conditioned rule, the default, which §6 reports.
fn conditioned<'s>(s: &'s State) -> Option<&'s presk_precision::propagate::Analysis> {
    s.bounded.as_ref()?.runs.iter().find(|r| r.model == presk_precision::propagate::Model::Conditioned)
}

/// A number, or `null` when it is not finite (JSON has no infinity).
fn num(x: f64) -> Value {
    if x.is_finite() { json!(x) } else { Value::Null }
}

fn block(b: &Block) -> Value {
    match b {
        Block::Prose(t) => json!({ "prose": t }),
        Block::Table { head, rows } => json!({ "table": { "head": head, "rows": rows } }),
        Block::Code { lang, text, name } => json!({ "code": { "lang": lang, "text": text, "name": name } }),
        Block::Note { title, text } => json!({ "note": { "title": title, "text": text } }),
        Block::Attach { name, text } => json!({ "attach": { "name": name, "text": text } }),
    }
}

fn section(s: &Section) -> Value {
    json!({
        "number": s.number,
        "title": s.title,
        "purpose": s.purpose,
        "blocks": s.blocks.iter().map(block).collect::<Vec<_>>(),
        "commands": s.commands,
        "sections": s.sections.iter().map(section).collect::<Vec<_>>(),
    })
}

/// The document as values.
pub(crate) fn document(doc: &Document) -> Value {
    json!({
        "title": doc.title,
        "subtitle": doc.subtitle,
        "header": doc.header.iter().map(block).collect::<Vec<_>>(),
        "sections": doc.sections.iter().map(section).collect::<Vec<_>>(),
    })
}

/// The section of the document a stage is shown in, and its title.
fn stage_section(stage: Stage) -> (&'static str, &'static str) {
    match stage {
        Stage::Source => ("1", "Source"),
        Stage::Elab => ("2", "Elaborated"),
        Stage::Core => ("3", "PSK-Core"),
        Stage::Ad => ("4", "After differentiation"),
        Stage::Precision => ("6", "Precision"),
        Stage::Opt => ("5", "Optimized"),
        Stage::Fusion => ("7", "Fusion"),
        Stage::Layout => ("8", "Nests and layout"),
        Stage::Kernel => ("9", "Kernels"),
        Stage::Launch => ("9", "Launches"),
        Stage::C => ("10", "C"),
        Stage::Native => ("9", "Native code"),
    }
}

/// A link's reason as its name and its fields.
fn how(h: &How) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("how".into(), json!(h.name()));
    match h {
        How::LoweredFn { kind } | How::Generated { kind } => {
            m.insert("kind".into(), json!(kind));
        }
        How::Lowered { rule } | How::Adjoint { rule } => {
            m.insert("rule".into(), json!(rule));
        }
        How::Primal { mode } => {
            m.insert("mode".into(), json!(mode));
        }
        How::Site { role } => {
            m.insert("role".into(), json!(role));
        }
        How::Rewritten { rules } => {
            m.insert("rules".into(), json!(rules));
        }
        How::Fused { members, stop } => {
            m.insert("members".into(), json!(members));
            m.insert("stop".into(), json!(stop));
        }
        How::Alone { stop } => {
            m.insert("stop".into(), json!(stop));
        }
        How::LaidOut { point, points } => {
            m.insert("point".into(), json!(point));
            m.insert("points".into(), json!(points));
        }
        How::Kernel { shared_by } => {
            m.insert("shared_by".into(), json!(shared_by));
        }
        How::Launched { count } => {
            m.insert("count".into(), json!(count));
        }
        How::Function { first, last } => {
            m.insert("lines".into(), json!([first, last]));
        }
        _ => {}
    }
    m
}

/// The function an element belongs to, which the page folds by: a site's or
/// a nest's function, a function's own name, a kernel's first nest's.
fn group(s: &State, id: &Id) -> Option<String> {
    let site_fn = |k: &str| k.rsplit_once("/%").map(|(f, _)| f.to_string());
    match id.stage {
        Stage::Core | Stage::Ad | Stage::Opt => Some(site_fn(&id.key).unwrap_or_else(|| id.key.clone())),
        Stage::Precision => site_fn(&id.key),
        Stage::Fusion | Stage::Layout => site_fn(&id.key),
        Stage::Launch => id.key.rsplit_once(' ').and_then(|(n, _)| site_fn(n)),
        Stage::Kernel | Stage::C | Stage::Native => s.table.made(id).first().and_then(|l| l.parents.iter().find_map(|p| match p.stage {
            Stage::Kernel => group(s, p),
            _ => site_fn(&p.key),
        })),
        Stage::Elab => id.key.strip_prefix("fn ").map(String::from),
        Stage::Source => None,
    }
}

/// The program a stage's `let`s are in.
fn program<'s>(s: &'s State, stage: Stage) -> Option<&'s presk_core::Program> {
    match stage {
        Stage::Core => Some(&s.core),
        Stage::Ad => Some(&s.ad_program),
        Stage::Opt => Some(&s.xcore),
        _ => None,
    }
}

/// The facts a reader is shown on an element.
fn facts(s: &State, id: &Id, math: &Math) -> Map<String, Value> {
    let mut m = Map::new();
    match id.stage {
        Stage::Elab => {
            let tex = match id.key.split_once(' ') {
                Some(("fn", name)) => math.displays.get(name),
                Some(("expr", span)) => math.formulas.get(span),
                _ => None,
            };
            if let Some(t) = tex {
                m.insert("tex".into(), json!(t));
            }
        }
        Stage::Core | Stage::Ad | Stage::Opt => {
            if let Some(p) = program(s, id.stage)
                && let Some((f, l)) = super::site_let(p, &id.key)
            {
                m.insert("span".into(), json!(presk_driver::provenance::span_key(l.span)));
                if let Some(n) = &l.name {
                    m.insert("name".into(), json!(n));
                }
                let e = presk_print_math::core::equation(f, l, p, &s.prog);
                m.insert("tex".into(), json!(format!("{} = {}", e.lhs.render(Format::Latex), e.rhs.render(Format::Latex))));
            }
        }
        Stage::Precision => {
            let site = id.key.strip_prefix("param ").unwrap_or(&id.key);
            if let Some((e, _)) = &s.env
                && let Some(mag) = e.get(site)
            {
                m.insert("magnitude".into(), num(mag));
                m.insert("elements".into(), json!(e.elems(site)));
                if let Some(c) = e.condition(site) {
                    m.insert("condition".into(), num(c));
                }
            }
            if let Some(x) = s.sens.as_ref().and_then(|x| x.get(site)) {
                m.insert("sensitivity".into(), num(x));
            }
            let ty = if id.key.starts_with("param ") { s.realization.params.get(site).copied().unwrap_or(s.realization.default) } else { s.realization.dtype_at(site) };
            m.insert("type".into(), json!(ty.name()));
            if let Some(sv) = &s.solved
                && let Some(c) = sv.breakdown.iter().find(|c| c.name == site)
            {
                m.insert("term".into(), num(c.err));
                m.insert("rounded_in".into(), json!(c.dtype.name()));
                if c.role == presk_precision::solve::Role::Decision {
                    m.insert("why".into(), json!(super::why(sv, site, if sv.time { "s" } else { "bytes" })));
                }
                // The site's term of the solver's estimate, chapter 4's
                // first-order change: its sensitivity, the unit roundoff of
                // the type it is rounded in, its magnitude.
                if c.err > 0.0 {
                    let t = c.dtype.name();
                    m.insert(
                        "tex".into(),
                        json!(format!("e_v = \\Bigl\\lVert \\frac{{\\partial L}}{{\\partial v}} \\Bigr\\rVert_1 \\, u_{{\\code{{{t}}}}} \\, \\max |v| = {} \\times {} \\times {} = {}", tex_num(c.sens), unit(t), tex_num(c.mag), tex_num(c.err))),
                    );
                }
            }
            // The bound's error at the site, by the conditioned rule, the
            // one §6 reports.
            if let Some(b) = conditioned(s).and_then(|r| r.sites.get(site)) {
                m.insert("bound".into(), num(b.err));
            }
        }
        Stage::Fusion => {
            m.insert("type".into(), json!(s.realization.dtype_at(&id.key).name()));
        }
        Stage::Launch => {
            if let Some((nest, _)) = id.key.rsplit_once(' ') {
                m.insert("type".into(), json!(s.realization.dtype_at(nest).name()));
            }
        }
        Stage::Layout => {
            m.insert("type".into(), json!(s.realization.dtype_at(&id.key).name()));
            // A contraction nest: its sum, and the order its terms are added
            // in under the realization.
            if let Some((f, l)) = super::site_let(&s.xcore, &id.key)
                && let Some(c) = presk_print_math::rules::contraction(f, l, &s.prog)
            {
                m.insert("tex".into(), json!(c.render(Format::Latex)));
                let partials = match s.realization.order {
                    presk_rt::ReductionOrder::Interleaved { partials } => Some(u64::from(partials)),
                    _ => None,
                };
                m.insert("order".into(), json!(s.realization.order.name()));
                m.insert("order_tex".into(), json!(presk_print_math::rules::order(partials, Format::Latex)));
            }
            if let (Some(c), Some(p)) = (&s.captured, &s.planned)
                && let Some(first) = p.costed.iter().find(|k| c.launches[k.launch].nest == id.key)
            {
                let l = &c.launches[first.launch];
                m.insert("shape".into(), json!(presk_driver::layout::shape(l)));
                m.insert("predicted_us".into(), num(first.scored[0].0 * 1e6));
                if let Some((t, lay)) = first.scored.get(1) {
                    m.insert("runner_up".into(), json!(lay.to_string()));
                    m.insert("runner_up_us".into(), num(*t * 1e6));
                }
            }
        }
        Stage::Kernel => {
            if let Some(k) = id.key.strip_prefix('k').and_then(|j| j.parse::<usize>().ok()).and_then(|j| s.hosted.as_ref().ok().and_then(|h| h.listing.kernels.get(j))) {
                m.insert("text".into(), json!(k.to_string()));
                m.insert("dtype".into(), json!(k.dtype.name()));
                // `--measure`'s median launch, matched by its text: the
                // bench's kernels are another run's (§11).
                if let Some(Ok(me)) = &s.measured
                    && let Ok(kt) = &me.kernels
                    && let Some(i) = kt.kernels.iter().position(|x| x.to_string() == k.to_string())
                    && let Some(t) = kt.medians[i]
                {
                    m.insert("measured_us".into(), num(t * 1e6));
                }
            }
        }
        Stage::Native => {
            // The kernel's machine code (plan 0.13.0 step 5, D138): its
            // bytes, before they are linked (D145), each line's ranges of
            // them, the ranges Cranelift gave no line, its text of the
            // instructions, and the fields the linker writes.
            if let Some(code) = id.key.strip_prefix("native k").and_then(|j| j.parse::<usize>().ok()).and_then(|j| s.native.as_ref()?.get(j)) {
                m.insert("bytes".into(), json!(code.bytes.len()));
                m.insert("hex".into(), json!(code.bytes.iter().map(|b| format!("{b:02x}")).collect::<String>()));
                m.insert("ranges".into(), json!(code.ranges));
                m.insert("given".into(), json!(code.given));
                m.insert("disasm".into(), json!(code.disasm));
                m.insert("relocs".into(), json!(code.relocs));
            }
        }
        Stage::C => {
            if let Some(Ok(c)) = &s.c
                && let Some(text) = &c.kernels_c
                && let Some(l) = s.table.made(id).first()
                && let How::Function { first, last } = l.how
            {
                let lines: Vec<&str> = text.lines().skip(first.saturating_sub(1)).take(last + 1 - first).collect();
                m.insert("text".into(), json!(lines.join("\n")));
            }
        }
        _ => {}
    }
    m
}

/// The JSON of a document and the compilation it was built from.
pub(crate) fn json(s: &State, doc: &Document, traced: &[String]) -> (Value, Option<presk_diagram::Network>) {
    let t = &s.table;
    let math = Math::new(s);
    let mut stages = Vec::new();
    for stage in Stage::ALL {
        let ids = t.elements(stage);
        if ids.is_empty() {
            continue;
        }
        let (section, title) = stage_section(stage);
        let elements: Vec<Value> = ids
            .iter()
            .map(|id| {
                let mut e = Map::new();
                e.insert("key".into(), json!(id.key));
                e.insert("label".into(), json!(t.label(id).unwrap_or("")));
                if let Some(g) = group(s, id) {
                    e.insert("group".into(), json!(g));
                }
                let f = facts(s, id, &math);
                if !f.is_empty() {
                    e.insert("facts".into(), Value::Object(f));
                }
                if let Some(why) = t.ended(id) {
                    e.insert("ends".into(), json!(why));
                }
                Value::Object(e)
            })
            .collect();
        stages.push(json!({ "stage": stage.name(), "title": title, "section": section, "elements": elements }));
    }
    let links: Vec<Value> = t
        .links()
        .iter()
        .map(|l| {
            let mut m = how(&l.how);
            m.insert("child".into(), json!(l.child.to_string()));
            m.insert("parents".into(), json!(l.parents.iter().map(|p| p.to_string()).collect::<Vec<_>>()));
            Value::Object(m)
        })
        .collect();
    // The rules the links name, as the reference's tables write them: a
    // derivative's contributions (`core-adjoint`), a rewrite's identity
    // (`core-algebra`, `core-structure`).
    let mut adjoint = BTreeMap::new();
    let mut rewrite = BTreeMap::new();
    for l in t.links() {
        match &l.how {
            How::Adjoint { rule } => {
                if let Some(m) = presk_print_math::core::adjoint_rule(rule) {
                    adjoint.entry(rule.clone()).or_insert_with(|| m.render(Format::Latex));
                }
            }
            How::Rewritten { rules } => {
                for r in rules {
                    if let Some(x) = presk_print_math::rules::rewrite(r) {
                        rewrite.entry(r.clone()).or_insert_with(|| x.to_string());
                    }
                }
            }
            _ => {}
        }
    }
    let precision = s.solved.as_ref().map(|sv| {
        json!({
            "estimate": num(sv.solution.estimated_error),
            "budget": num(sv.solution.budget_abs),
            "measured": sv.solution.measured.map(num),
            "bound": conditioned(s).map(|r| num(r.output.err)),
        })
    });
    let sources: Vec<Value> = s.sm.files().map(|(id, f)| json!({ "id": id.0, "path": crate::shown(std::path::Path::new(&f.name)), "text": f.text })).collect();
    let dims: Vec<Value> = s.dims.iter().map(|(n, v)| json!([n, v])).collect();
    let net = network(s, &math.listing);
    let net_json = net.as_ref().and_then(|n| serde_json::from_str(&presk_diagram::json::network(n)).ok()).unwrap_or(Value::Null);
    let v = json!({
        "format": FORMAT,
        "compiler": env!("CARGO_PKG_VERSION"),
        "program": {
            "file": s.file,
            "entry": s.o.entry,
            "dims": dims,
            "consts": s.o.consts,
            "realization": s.realization.label(),
            "opt": s.o.opt,
            "planned": !matches!(s.plan_src, super::PlanSrc::None),
            "differentiated": s.ad_program.funcs.len() > s.core.funcs.len(),
        },
        "traced": traced,
        "rules": { "adjoint": adjoint, "rewrite": rewrite },
        "precision": precision,
        "sources": sources,
        "stages": stages,
        "links": links,
        "network": net_json,
        "document": document(doc),
    });
    (v, net)
}
