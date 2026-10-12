//! The e-graph optimizer, as `--opt [--profile FILE]` applies it.

use std::path::PathBuf;

use presk_types::elab::ElabProgram;

use crate::{Report, Result, Stop};

/// `--opt [--profile FILE]`.
pub struct Opt {
    pub on: bool,
    pub profile: Option<PathBuf>,
}

impl Opt {
    /// Rewrite `core` with the e-graph optimizer when `--opt` is on; one
    /// summary line reported (the whole report with `verbose`, as `--verbose`
    /// and `dump --stage core-opt` ask). `dims` are the run's dimensions, for
    /// the roofline. The summary is returned too, for a document.
    pub fn apply(&self, core: &mut presk_core::Program, prog: &ElabProgram, verbose: bool, dims: &[(String, u64)], report: &mut dyn Report) -> Result<Option<String>> {
        self.apply_report(core, prog, verbose, dims, report).map(|r| r.map(|r| r.summary()))
    }

    /// [`Opt::apply`], the optimizer's report returned: what each function
    /// became, its lets and its cost before and after (plan 0.11.0 step 4).
    pub fn apply_report(&self, core: &mut presk_core::Program, prog: &ElabProgram, verbose: bool, dims: &[(String, u64)], report: &mut dyn Report) -> Result<Option<presk_opt::Report>> {
        if !self.on {
            return Ok(None);
        }
        let profile = match &self.profile {
            Some(path) => match presk_opt::Profile::load(path) {
                Ok(p) => Some(p),
                Err(m) => {
                    report.line(&format!("presk: --profile: {m}"));
                    return Err(Stop(2));
                }
            },
            None => None,
        };
        let opts = presk_opt::Options { limits: presk_opt::Limits::default(), profile, dims: dims.iter().cloned().collect(), default_dim: None };
        let r = presk_opt::optimize(core, prog, &opts);
        let summary = r.summary();
        if verbose {
            // `eprint!` of the whole summary: its lines, the last without a
            // newline if it has none.
            report.text(&summary);
        } else {
            report.line(summary.lines().next().unwrap_or(""));
        }
        Ok(Some(r))
    }
}
