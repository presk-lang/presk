//! Effects: the effects a function uses (primitives drawing randomness,
//! callees) must be declared with `uses`. Initializers are implicitly
//! `uses Rand`; lifted lambdas are checked against their parent's
//! declaration.

use std::collections::HashMap;

use presk_syntax::{Diagnostic, Span};
use presk_types::elab::ElabProgram;

use crate::ir::{FuncId, Origin, Program, Rhs};

const KNOWN: &[&str] = &["Rand"];

pub fn analyze(prog: &Program, elab: &ElabProgram, diags: &mut Vec<Diagnostic>) -> HashMap<FuncId, Vec<String>> {
    let mut inferred: HashMap<FuncId, Vec<String>> = HashMap::new();
    for i in 0..prog.funcs.len() {
        infer(prog, FuncId(i as u32), &mut inferred);
    }
    // Unknown effect names.
    for f in &prog.funcs {
        for e in &f.effects {
            if !KNOWN.contains(&e.as_str()) {
                diags.push(Diagnostic::error("E4002", format!("unknown effect `{e}`")).label(f.span, format!("declared by `{}`", f.name)).help("the only effect of 0.1.0 is `Rand`"));
            }
        }
    }
    // Inferred ⊆ declared.
    for (i, f) in prog.funcs.iter().enumerate() {
        let id = FuncId(i as u32);
        let declared: Vec<String> = match &f.origin {
            Origin::Source(_) => f.effects.clone(),
            Origin::Init(_) => vec!["Rand".into()],
            Origin::Lambda { parent } => prog.func(*parent).effects.clone(),
            Origin::Optimizer { .. } => vec![],
        };
        for (e, span) in uses_with_spans(prog, id, &inferred) {
            if !declared.contains(&e) {
                let owner = match &f.origin {
                    Origin::Lambda { parent } => prog.func(*parent).name.clone(),
                    _ => f.name.clone(),
                };
                diags.push(
                    Diagnostic::error("E4001", format!("`{owner}` uses the effect `{e}` without declaring it"))
                        .label(span, format!("this uses `{e}`"))
                        .label(f.span, "declared without `uses`")
                        .help(format!("add `uses {e}` to the signature of `{owner}`")),
                );
            }
        }
    }
    let _ = elab;
    inferred
}

fn infer(prog: &Program, id: FuncId, out: &mut HashMap<FuncId, Vec<String>>) -> Vec<String> {
    if let Some(e) = out.get(&id) {
        return e.clone();
    }
    let mut effects: Vec<String> = Vec::new();
    for l in &prog.func(id).body.lets {
        match &l.rhs {
            Rhs::Prim { op, .. } if op.uses_rand() => push(&mut effects, "Rand"),
            Rhs::Call { f, .. } | Rhs::Closure { f, .. } | Rhs::Fold { body: f, .. } => {
                for e in infer(prog, *f, out) {
                    push(&mut effects, &e);
                }
            }
            _ => {}
        }
    }
    out.insert(id, effects.clone());
    effects
}

fn push(v: &mut Vec<String>, e: &str) {
    if !v.iter().any(|x| x == e) {
        v.push(e.to_string());
    }
}

/// The effects of a function with the span of the first use of each.
fn uses_with_spans(prog: &Program, id: FuncId, inferred: &HashMap<FuncId, Vec<String>>) -> Vec<(String, Span)> {
    let mut out: Vec<(String, Span)> = Vec::new();
    for l in &prog.func(id).body.lets {
        let es: Vec<String> = match &l.rhs {
            Rhs::Prim { op, .. } if op.uses_rand() => vec!["Rand".into()],
            Rhs::Call { f, .. } | Rhs::Closure { f, .. } | Rhs::Fold { body: f, .. } => inferred.get(f).cloned().unwrap_or_default(),
            _ => vec![],
        };
        for e in es {
            if !out.iter().any(|(x, _)| *x == e) {
                out.push((e, l.span));
            }
        }
    }
    out
}
