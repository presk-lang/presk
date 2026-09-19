//! Analyses on PSK-Core: regularity classes (`C∞ < C⁰⁺ < Disc`, per input)
//! and effects. Both are summaries per function, computed by dataflow over
//! the ANF and memoized across calls (Core is non-recursive).
//!
//! Error codes `E3xxx` (regularity), `E4xxx` (effects).

pub mod effects;
pub mod regularity;

use std::collections::HashMap;

use presk_syntax::Diagnostic;
use presk_types::elab::ElabProgram;

use crate::ir::{FuncId, Program};
pub use regularity::{Class, ParamClass};

#[derive(Default, Debug)]
pub struct Analysis {
    /// For every function: the class of its result with respect to each parameter.
    pub classes: HashMap<FuncId, Vec<ParamClass>>,
    /// For every function: the effects it uses (inferred).
    pub effects: HashMap<FuncId, Vec<String>>,
}

pub fn analyze(prog: &Program, elab: &ElabProgram) -> (Analysis, Vec<Diagnostic>) {
    let mut diags = Vec::new();
    let classes = regularity::analyze(prog, elab, &mut diags);
    let effects = effects::analyze(prog, elab, &mut diags);
    (Analysis { classes, effects }, diags)
}

/// `presk check --summary`: one line per function with classes and effects.
pub fn summary(prog: &Program, a: &Analysis) -> String {
    let mut out = String::new();
    for (i, f) in prog.funcs.iter().enumerate() {
        let id = FuncId(i as u32);
        let classes = a
            .classes
            .get(&id)
            .map(|cs| f.params.iter().zip(cs).map(|(p, c)| format!("{} {}", p.name, c.class)).collect::<Vec<_>>().join(", "))
            .unwrap_or_default();
        let eff = a.effects.get(&id).map(|e| if e.is_empty() { "none".to_string() } else { e.join(", ") }).unwrap_or_default();
        out.push_str(&format!("{} : [{}] uses {}\n", f.name, classes, eff));
    }
    out
}
