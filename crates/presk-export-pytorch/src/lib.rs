//! Readable PyTorch export: a printer over the *typed* program (not over
//! PSK-Core), so that the generated code is isomorphic to the source — one
//! `module` is one `nn.Module`, `apply` is `forward`, names are kept.
//!
//! Step 7b of the 0.1.0 plan. Error code `E8xxx`.

mod conformance;
mod graph;
mod py;
mod train;

use std::collections::BTreeMap;

use presk_syntax::{Diagnostic, SourceMap};
use presk_types::check::TypedProgram;
use presk_types::elab::ElabProgram;

pub use conformance::{ConformanceData, OptimizerConformance, TrainConformance, test_file};
pub use graph::graph_driver;
pub use train::{Schedule, TrainSpec, schedule_py, train_files};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StackMode {
    Loop,
    Unroll,
}

#[derive(Clone, Debug)]
pub struct Options {
    /// `--shapes=check`: executable shape assertions.
    pub shapes_check: bool,
    pub stack: StackMode,
}

impl Default for Options {
    fn default() -> Self {
        Options { shapes_check: false, stack: StackMode::Loop }
    }
}

/// A generated file: name relative to the output directory, and content.
#[derive(Clone, Debug)]
pub struct Generated {
    pub name: String,
    pub content: String,
}

/// The Python module name of a unit: its file stem made a valid Python
/// identifier (`01_linear.psk` → `_01_linear`, `mini-gpt.psk` → `mini_gpt`),
/// since the generated files import each other with `from <stem> import *`.
pub fn stem(elab: &ElabProgram, unit: usize) -> String {
    let raw = elab.units[unit].path.file_stem().and_then(|s| s.to_str()).unwrap_or("program");
    let mut s: String = raw.chars().map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' }).collect();
    if s.chars().next().is_none_or(|c| c.is_ascii_digit()) {
        s.insert(0, '_');
    }
    s
}

/// Export every unit (the main file and its imports) to one Python file each.
pub fn export(sm: &SourceMap, elab: &ElabProgram, typed: &TypedProgram, opts: &Options) -> (Vec<Generated>, Vec<Diagnostic>) {
    let mut out = Vec::new();
    let mut diags = Vec::new();
    for unit in 0..elab.units.len() {
        let mut p = py::Printer::new(sm, elab, typed, opts, unit, &mut diags);
        let content = p.unit();
        out.push(Generated { name: format!("{}.py", stem(elab, unit)), content });
    }
    (out, diags)
}

/// Dimension values of a run, by name, for the conformance test.
pub type DimValues = BTreeMap<String, u64>;

/// The file dimensions a module's constructor takes after its type parameters
/// (see `Printer::extra_dims`).
pub fn extra_dims(elab: &ElabProgram, mid: presk_types::ty::ModuleId) -> Vec<presk_types::dim::DimVar> {
    let typed = TypedProgram::default();
    let opts = Options::default();
    let mut diags = Vec::new();
    let sm = SourceMap::new();
    let p = py::Printer::new(&sm, elab, &typed, &opts, elab.main, &mut diags);
    p.extra_dims(mid)
}
