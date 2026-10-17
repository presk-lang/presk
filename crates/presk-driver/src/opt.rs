//! The e-graph optimizer, as `--opt [--profile FILE]` applies it.

use std::path::PathBuf;

use presk_types::elab::ElabProgram;

use crate::{Report, Result, Stop};

/// The optimizer's report, and the links and ends of what it made.
pub type Traced = (presk_opt::Report, Vec<presk_provenance::Link>, Vec<presk_provenance::End>);

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
        self.apply_traced(core, prog, verbose, dims, report).map(|r| r.map(|(r, _, _)| r))
    }

    /// [`Opt::apply_report`], and the links and ends of what the optimizer
    /// made (plan 0.13.0 step 1, D134).
    pub fn apply_traced(&self, core: &mut presk_core::Program, prog: &ElabProgram, verbose: bool, dims: &[(String, u64)], report: &mut dyn Report) -> Result<Option<Traced>> {
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
        let (r, links, ends) = presk_opt::optimize_traced(core, prog, &opts);
        let summary = r.summary();
        if verbose {
            // `eprint!` of the whole summary: its lines, the last without a
            // newline if it has none.
            report.text(&summary);
        } else {
            report.line(summary.lines().next().unwrap_or(""));
        }
        Ok(Some((r, links, ends)))
    }
}

impl Opt {
    /// [`Opt::apply`], refusing a realization or a plan that names a site of
    /// a function the optimizer rewrote (`E7002`, plan 0.13.0 step 1, D134).
    /// The optimizer numbers every `let` of such a function anew, so the
    /// sites the realization types or the plan lays out name the program
    /// before it, and would type nothing they name, or another `let` (D133's
    /// finding 1). The message names each site and what the optimizer made
    /// of it, from its links.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_for(&self, core: &mut presk_core::Program, prog: &ElabProgram, verbose: bool, dims: &[(String, u64)], plan: &presk_rt::Plan, sm: &presk_syntax::SourceMap, report: &mut dyn Report) -> Result<Option<String>> {
        let Some((r, links, _)) = self.apply_traced(core, prog, verbose, dims, report)? else { return Ok(None) };
        if let Err(m) = check_sites(plan, &links) {
            report.diagnostics(&[presk_syntax::Diagnostic::error("E7002", m)], sm);
            return Err(Stop(1));
        }
        Ok(Some(r.summary()))
    }
}

/// The sites `plan` names in functions the optimizer rewrote, by its links:
/// `Err` with the message of [`Opt::apply_for`] when there is one.
pub fn check_sites(plan: &presk_rt::Plan, links: &[presk_provenance::Link]) -> std::result::Result<(), String> {
    use presk_provenance::{How, Stage};
    let rewritten: std::collections::BTreeSet<&str> = links.iter().filter(|l| l.child.stage == Stage::Opt && !l.child.key.contains("/%") && matches!(l.how, How::Rewritten { .. })).map(|l| l.child.key.as_str()).collect();
    let r = &plan.realization;
    let named: Vec<(&str, &String)> = r.storage.keys().map(|s| ("types", s)).chain(r.accum.keys().map(|s| ("accumulates", s))).chain(plan.layout.keys().map(|s| ("lays out", s))).collect();
    let mut hits: Vec<String> = Vec::new();
    for (what, site) in &named {
        let Some((f, _)) = site.rsplit_once("/%") else { continue };
        if !rewritten.contains(f) {
            continue;
        }
        let made: Vec<String> = links.iter().filter(|l| l.child.stage == Stage::Opt && l.parents.iter().any(|p| p.stage == Stage::Ad && &p.key == *site)).map(|l| format!("`{}`", l.child.key)).collect();
        let now = if made.is_empty() { "which it removed".to_string() } else { format!("now {}", made.join(", ")) };
        hits.push(format!("{what} `{site}`, {now}"));
    }
    if hits.is_empty() {
        return Ok(());
    }
    let n = hits.len();
    let shown: Vec<String> = hits.into_iter().take(4).collect();
    let more = if n > 4 { format!("; and {} more", n - 4) } else { String::new() };
    let what = if plan.layout.is_empty() { "realization" } else { "plan" };
    Err(format!(
        "--opt writes each function it optimizes back with new numbers, even where no rule changed it, and the {what} names sites of the program before it: it {}{more}. Under --opt the {what} would apply to nothing it names, or to another `let` (D133); run it without --opt",
        shown.join("; it ")
    ))
}
