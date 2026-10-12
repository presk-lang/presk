//! The realization and the plan a command runs under.

use presk_rt::{Plan, Realization};
use presk_syntax::{Diagnostic, SourceMap};

use crate::{Report, Result, Stop};

/// `--realize f64|f32|bf16|@FILE`. The three names are uniform realizations,
/// what the flag has always meant; `@FILE` reads the object a solver writes
/// (D34). Parsing is deferred to `presk_rt::Realization::parse` so that one
/// spelling serves the CLI, the solver and the conformance report. The
/// error is the message the command prints after `presk: `.
pub fn parse_realize(spec: &str) -> std::result::Result<Realization, String> {
    Realization::parse(spec)
}

/// `--plan @FILE`, or the plan `--realize` means: its realization and the
/// default layout.
pub fn parse_plan(plan: Option<&str>, realize: &str) -> std::result::Result<Plan, String> {
    match plan {
        None => parse_realize(realize).map(Plan::from_realization),
        Some(spec) => Plan::parse(spec),
    }
}

/// Checks a plan's nests against the program it runs on (`E7002`), after
/// every transformation of the program: a nest is named in the program as it
/// is executed.
pub fn check_plan(plan: &Plan, core: &presk_core::Program, sm: &SourceMap, report: &mut dyn Report) -> Result<()> {
    presk_kernel::plan::check_plan(plan, core).map_err(|e| {
        report.diagnostics(&[Diagnostic::error(e.code, e.message)], sm);
        Stop(1)
    })
}
