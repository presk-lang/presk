//! The sections of a document, from the driver's values (plan 0.11.0 step
//! 3): the compilation is run once, stage by stage, as the commands run it,
//! and each section prints its stage's value.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use presk_core::ir::Rhs;
use presk_core::{FuncId, Program};
use presk_driver::layout::{Capture, Captured, Hosted, Planned, Space};
use presk_driver::precision::{Bounded, Solved};
use presk_driver::report::Reported;
use presk_driver::{Collect, Report, Stop, text};
use presk_precision::envelope::Envelope;
use presk_precision::sensitivity::Sensitivity;
use presk_precision::solve::{Role, Roles};
use presk_provenance::{Id, Stage};
use presk_rt::{Dtype, Plan, Realization};
use presk_syntax::SourceMap;
use presk_types::check::TypedProgram;
use presk_types::elab::ElabProgram;

use crate::doc::{Block, Document, Kind, Section};

mod json;
use crate::{Artifact, Options, Provenance, Target, shown};

/// A report that forwards to the command's and keeps a copy: the
/// diagnostics are printed as the command prints them, and listed in §1.
struct Tee<'a> {
    inner: &'a mut dyn Report,
    kept: Collect,
    /// What was reported already: the plan's capture and the C export read
    /// the file again, and would report its warnings again.
    seen: BTreeSet<String>,
}

impl Report for Tee<'_> {
    fn diagnostics(&mut self, diags: &[presk_syntax::Diagnostic], sm: &SourceMap) {
        let fresh: Vec<presk_syntax::Diagnostic> = diags.iter().filter(|d| self.seen.insert(presk_syntax::diag::render_all(std::slice::from_ref(*d), sm, false))).cloned().collect();
        if fresh.is_empty() {
            return;
        }
        self.inner.diagnostics(&fresh, sm);
        self.kept.diagnostics(&fresh, sm);
    }

    fn line(&mut self, text: &str) {
        self.inner.line(text);
    }

    fn text(&mut self, text: &str) {
        self.inner.text(text);
    }
}

/// The document's directory, as commands and paths in its body name it.
const DOC: &str = "$DOC";

/// Where the realization comes from.
enum RealSrc {
    /// No `--budget`, `--realize` or `--plan`: every site in `f64`.
    Default,
    /// `--realize f32`.
    Named(String),
    /// `--realize @FILE` or the realization of `--plan @FILE`.
    Given(String),
    /// Solved under `--budget`.
    Solved,
}

/// Where the layout plan comes from.
enum PlanSrc {
    /// No profile: the default layouts.
    None,
    Given(String),
    Computed,
}

/// Every value the document prints.
struct State<'a> {
    o: &'a Options,
    file: String,
    sm: SourceMap,
    prog: ElabProgram,
    typed: TypedProgram,
    /// PSK-Core as lowered.
    core: Program,
    analysis: presk_core::Analysis,
    entry: FuncId,
    dims: Vec<(String, u64)>,
    /// The diagnostics of the front end, rendered without colour.
    diags: Vec<String>,
    /// The program the precision stages run: differentiated, the fusion
    /// groups kept, as `presk calibrate`, `presk dump --stage sensitivity`
    /// and `presk solve` differentiate it.
    pcore: Program,
    env: Option<(Envelope, String)>,
    env_src: Option<Provenance>,
    sens: Option<Sensitivity>,
    /// Why there are no sensitivities.
    sens_none: Option<String>,
    /// The sweep's program: the functions it added, printed.
    sweep: Vec<String>,
    solved: Option<Solved>,
    roles: Roles,
    realization: Realization,
    real_src: RealSrc,
    bounded: Option<Bounded>,
    /// The program the kernels run: differentiated under `--remat`, and
    /// rewritten under `--opt`.
    xcore: Program,
    opt_summary: Option<String>,
    opt_report: Option<presk_opt::Report>,
    /// The program before the optimizer rewrote it, under `--opt`.
    before_opt: Option<Program>,
    plan: Plan,
    plan_src: PlanSrc,
    profile: Option<(presk_opt::profile::LayoutFit, presk_kernel::cost::Machine, String)>,
    captured: Option<Captured>,
    planned: Option<Planned>,
    hosted: Result<Hosted, String>,
    host_c: Option<String>,
    c: Option<Result<CExport, String>>,
    pytorch: Option<Result<Vec<(String, String)>, String>>,
    artifacts: Vec<Artifact>,
    measured: Option<Result<Measured, String>>,
    /// The provenance table of the compilation (plan 0.13.0 step 1, D134):
    /// what §12 reads.
    table: presk_provenance::Table,
    /// The program after differentiation the table's stage `ad` names: the
    /// kernels' before the optimizer.
    ad_program: Program,
    /// Each listed kernel's machine code with its lines, under `--format
    /// json` or `html` with the native target (plan 0.13.0 step 5).
    pub(crate) native: Option<Vec<presk_cranelift::MachineCode>>,
}

/// What `--measure` measured.
struct Measured {
    /// The load average over a minute, before and after.
    load: (Option<f64>, Option<f64>),
    /// The exact reference's output and the native one's, and their gap.
    exact: Option<presk_rt::Value>,
    native: presk_rt::Value,
    gap: Option<f64>,
    /// A call natively: median and best, in ms, and the calls.
    call: Option<(f64, f64)>,
    warmup: usize,
    repeat: usize,
    workers: usize,
    kernels: Result<presk_driver::bench::KernelTimes, String>,
    host: Result<presk_driver::bench::HostProfiled, String>,
}

/// What the C export wrote.
struct CExport {
    line: String,
    files: Vec<(String, usize)>,
    kernels: Vec<presk_kernel::Kernel>,
    /// `kernels.c`, its text.
    kernels_c: Option<String>,
}

/// The functions `entry` reaches through calls, closures and folds, itself
/// included, in the program's order.
pub fn reach(core: &Program, entry: FuncId) -> Vec<FuncId> {
    let mut seen: BTreeSet<FuncId> = BTreeSet::new();
    let mut stack = vec![entry];
    while let Some(f) = stack.pop() {
        if !seen.insert(f) {
            continue;
        }
        for l in &core.func(f).body.lets {
            match &l.rhs {
                Rhs::Call { f: g, .. } | Rhs::Closure { f: g, .. } | Rhs::Fold { body: g, .. } => stack.push(*g),
                _ => {}
            }
        }
    }
    seen.into_iter().collect()
}

/// The `let` a site names, in the program that has it.
fn site_let<'p>(core: &'p Program, site: &str) -> Option<(&'p presk_core::Func, &'p presk_core::ir::Let)> {
    let (f, v) = site.rsplit_once("/%")?;
    let v: u32 = v.parse().ok()?;
    let f = core.func(core.find(f)?);
    f.body.lets.iter().find(|l| l.var.0 == v).map(|l| (f, l))
}

/// A number as the commands print it in a table: three significant digits.
fn sci(x: f64) -> String {
    format!("{x:.3e}")
}

fn given(path: &Path, report: &mut dyn Report) -> presk_driver::Result<Provenance> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(Provenance::Given { path: shown(path), sha256: crate::sha256(&bytes) }),
        Err(e) => {
            report.line(&format!("presk explain: cannot read {}: {e}", path.display()));
            Err(Stop(2))
        }
    }
}

pub(crate) fn build(o: &Options, report: &mut dyn Report) -> presk_driver::Result<(Document, Vec<Artifact>, Option<serde_json::Value>)> {
    let mut tee = Tee { inner: report, kept: Collect::default(), seen: BTreeSet::new() };
    let s = run(o, &mut tee)?;
    let diags: Vec<String> = tee.kept.items.iter().filter_map(|r| if let Reported::Diagnostics(t, _) = r { Some(t.clone()) } else { None }).collect();
    let s = State { diags, ..s };
    let doc = Document {
        title: title(&s),
        subtitle: "What the Presk compiler does to one program: `presk explain`".into(),
        header: header(&s),
        sections: vec![summary(&s), source(&s), elaboration(&s), core(&s), differentiation(&s), optimizer(&s), precision(&s), fusion(&s), machine(&s), kernels(&s), targets(&s), measured(&s), traces(&s), appendix_placeholder()],
    };
    let doc = with_appendix(doc);
    let json = (o.json || o.html).then(|| {
        let traced = if o.sites.is_empty() { default_sites(&s) } else { o.sites.clone() };
        json::json(&s, &doc, &traced)
    });
    Ok((doc, s.artifacts, json))
}

/// The provenance table of the compilation, alone.
pub(crate) fn table(o: &Options, report: &mut dyn Report) -> presk_driver::Result<presk_provenance::Table> {
    let mut tee = Tee { inner: report, kept: Collect::default(), seen: BTreeSet::new() };
    Ok(run(o, &mut tee)?.table)
}

/// Runs the compilation, stage by stage, and keeps every value.
fn run<'a>(o: &'a Options, report: &mut dyn Report) -> presk_driver::Result<State<'a>> {
    let file = shown(&o.file);
    let dims = presk_driver::front::parse_dims(&o.dims, report)?;
    let (sm, prog, typed, core, lowered) = presk_driver::front::typed_core_traced(&o.file, &o.consts, false, report)?;
    let (analysis, diags) = presk_core::analyze(&core, &prog);
    report.diagnostics(&diags, &sm);
    if presk_syntax::diag::has_errors(&diags) {
        return Err(Stop(1));
    }
    let Ok((entry, _, _)) = presk_interp::run::resolve_entry(&core, &prog, &o.entry) else {
        report.line(&format!("presk explain: `{}` is not an entry of {}", o.entry, o.file.display()));
        return Err(Stop(2));
    };
    let mut artifacts: Vec<Artifact> = Vec::new();
    for (what, p) in [("the machine profile", &o.profile), ("the weights", &o.weights)] {
        if let Some(p) = p {
            artifacts.push(Artifact { what: what.into(), path: shown(p), provenance: given(p, report)? });
        }
    }
    let art = |name: &str| o.out.join("artifacts").join(name);

    // ----- the realization given, if any ---------------------------------------
    let (mut realization, mut real_src, given_plan) = match (&o.plan, &o.realize) {
        (Some(spec), _) => match Plan::parse(spec) {
            Ok(p) => {
                let path = Path::new(spec.trim_start_matches('@'));
                artifacts.push(Artifact { what: "the layout plan, its realization with it".into(), path: shown(path), provenance: given(path, report)? });
                (p.realization.clone(), RealSrc::Given(shown(path)), Some((p, shown(path))))
            }
            Err(m) => {
                report.line(&format!("presk: {m}"));
                return Err(Stop(2));
            }
        },
        (None, Some(spec)) => match Realization::parse(spec) {
            Ok(r) => match spec.strip_prefix('@') {
                Some(path) => {
                    artifacts.push(Artifact { what: "the realization".into(), path: shown(Path::new(path)), provenance: given(Path::new(path), report)? });
                    (r, RealSrc::Given(shown(Path::new(path))), None)
                }
                None => (r, RealSrc::Named(spec.clone()), None),
            },
            Err(m) => {
                report.line(&format!("presk: {m}"));
                return Err(Stop(2));
            }
        },
        (None, None) => (Realization::uniform(Dtype::F64), RealSrc::Default, None),
    };

    // ----- precision -----------------------------------------------------------
    let mut pcore = core.clone();
    let (d, pcore_links) = presk_driver::ad::differentiate_traced(&mut pcore, o.remat, true);
    report.diagnostics(&d, &sm);
    if presk_syntax::diag::has_errors(&d) {
        return Err(Stop(1));
    }
    let (env, env_src) = match &o.envelope {
        Some(path) => match Envelope::load(path) {
            Ok(e) => {
                if e.entry != o.entry {
                    report.line(&format!("presk explain: {} was measured on `{}`, not `{}`", path.display(), e.entry, o.entry));
                    return Err(Stop(2));
                }
                let p = given(path, report)?;
                artifacts.push(Artifact { what: "the envelope".into(), path: shown(path), provenance: p.clone() });
                (Some((e, shown(path))), Some(p))
            }
            Err(m) => {
                report.line(&format!("presk explain: {m}"));
                return Err(Stop(2));
            }
        },
        None if o.no_exact => {
            artifacts.push(Artifact { what: "the envelope".into(), path: String::new(), provenance: Provenance::Omitted("`--no-exact`".into()) });
            (None, None)
        }
        None => {
            let c = presk_driver::precision::Calibrate { file: &o.file, entry: &o.entry, dims: dims.clone(), consts: &o.consts, data: None, samples: None, from: 0, seed: o.seed, bits: o.bits, weights: o.weights.as_deref() };
            match presk_driver::precision::calibrate(&pcore, &prog, &c) {
                Ok(e) => {
                    if let Err(m) = e.save(&art("envelope.json")) {
                        report.line(&format!("presk explain: {m}"));
                        return Err(Stop(2));
                    }
                    artifacts.push(Artifact { what: "the envelope".into(), path: "artifacts/envelope.json".into(), provenance: Provenance::Computed });
                    (Some((e, format!("{DOC}/artifacts/envelope.json"))), Some(Provenance::Computed))
                }
                Err(m) => {
                    report.line(&format!("presk calibrate: {m}"));
                    return Err(Stop(1));
                }
            }
        }
    };
    if o.budget.is_some() && env.is_none() {
        report.line("presk explain: --budget needs an envelope: measured on the exact reference, or given with --envelope");
        return Err(Stop(2));
    }
    let scalar = presk_driver::precision::returns_scalar(&pcore, entry);
    let (mut sens, mut sens_none, mut solved) = (None, None, None);
    if let (Some(budget), Some((e, _))) = (o.budget, &env) {
        let args = presk_driver::precision::SolveArgs { budget: presk_precision::solve::Budget::Relative(budget), weights: o.weights.as_deref(), verify: !o.no_exact, bits: o.bits, seed: o.seed, profile: o.profile.as_deref(), quant: None };
        match presk_driver::precision::solve(&mut pcore.clone(), &prog, e, &args) {
            Ok(s) => {
                if let Err(m) = std::fs::write(art("realization.json"), s.solution.realization.to_json()) {
                    report.line(&format!("presk explain: {m}"));
                    return Err(Stop(2));
                }
                artifacts.push(Artifact { what: "the realization".into(), path: "artifacts/realization.json".into(), provenance: Provenance::Computed });
                realization = s.solution.realization.clone();
                real_src = RealSrc::Solved;
                sens = Some(s.sens.clone());
                solved = Some(s);
            }
            Err(e) => {
                use presk_driver::precision::SolveError as E;
                let m = match e {
                    E::NotEntry => format!("presk solve: `{}` is not an entry of {}", o.entry, o.file.display()),
                    E::NotScalar => format!("presk solve: `{}` must return a real tensor, since the sites are ranked by its sensitivity `∂y/∂v`", o.entry),
                    E::Inputs(m) => format!("presk solve: cannot bind the inputs of `{}`: {m}", o.entry),
                    E::NoGradient(m) => format!("presk solve: the objective does not differentiate: {m}"),
                    E::Reference(m) => format!("presk solve: {m}"),
                    E::QuantWithoutGradient => "presk solve: --quant needs the parameters' gradient".into(),
                };
                report.line(&m);
                return Err(Stop(1));
            }
        }
    } else if !scalar {
        sens_none = Some(format!("`{}` returns no scalar `ℝ[]`, and a sensitivity is `∂L/∂v` of a scalar `L`.", o.entry));
    } else if o.no_exact {
        sens_none = Some("`--no-exact`: the sensitivities are read off a reverse sweep on the exact reference.".into());
    } else {
        match presk_driver::precision::sensitivity(&mut pcore.clone(), &prog, &sm, &o.entry, dims.clone(), o.seed, o.bits, o.weights.as_deref(), report) {
            Ok(s) => sens = Some(s),
            Err(_) => sens_none = Some("the reverse sweep failed; `presk dump --stage sensitivity` says why.".into()),
        }
    }
    let sweep = if sens.is_some() {
        let mut c = pcore.clone();
        let (added, _) = presk_driver::precision::sensitivity_program(&mut c, entry);
        added.iter().map(|f| presk_core::pretty::pretty_func(c.func(*f), &c, &prog)).collect()
    } else {
        Vec::new()
    };
    let roles = Roles::of(&pcore);

    // ----- the program the kernels run -------------------------------------------
    let mut xcore = core.clone();
    let (mut d, ad_links) = presk_driver::ad::differentiate_traced(&mut xcore, o.remat, realization.groups_visible());
    d.extend(presk_driver::ad::stored_vjps(&xcore, &realization));
    report.diagnostics(&d, &sm);
    if presk_syntax::diag::has_errors(&d) {
        return Err(Stop(1));
    }
    // The table's differentiated program: the kernels' before the optimizer.
    let ad_program = xcore.clone();
    let bounded = match &env {
        Some((e, _)) => match presk_driver::precision::bound(&xcore, &prog, e, &realization, &presk_precision::propagate::Model::ALL, !o.no_exact) {
            Ok(b) => Some(b),
            Err(presk_driver::precision::BoundError::NotEntry) => None,
            Err(presk_driver::precision::BoundError::Measure(m)) => {
                report.line(&format!("presk check: {m}"));
                return Err(Stop(1));
            }
        },
        None => None,
    };
    let before_opt = o.opt.then(|| xcore.clone());
    let optimizer = presk_driver::opt::Opt { on: o.opt, profile: o.profile.clone() };
    let (opt_report, opt_links, opt_ends) = match optimizer.apply_traced(&mut xcore, &prog, true, &dims, &mut Collect::default())? {
        Some((r, links, ends)) => (Some(r), links, ends),
        None => (None, Vec::new(), Vec::new()),
    };
    let opt_summary = opt_report.as_ref().map(|r| r.summary());
    if let Err(m) = presk_driver::opt::check_sites(&Plan::from_realization(realization.clone()), &opt_links) {
        report.diagnostics(&[presk_syntax::Diagnostic::error("E7002", m)], &sm);
        return Err(Stop(1));
    }

    // ----- the plan --------------------------------------------------------------
    let fit = presk_driver::layout::load_fit(o.profile.as_deref(), report)?;
    let (plan, plan_src, profile, captured, planned) = match (given_plan, fit) {
        (Some((p, path)), fit) => {
            let profile = fit.map(|(f, m)| (f, m.unwrap_or_else(presk_kernel::cost::Machine::detect), String::new()));
            (p, PlanSrc::Given(path), profile, None, None)
        }
        (None, Some((fit, stored))) => {
            let base = Plan::from_realization(realization.clone());
            let mut captured = presk_driver::layout::capture(&o.file, &o.entry, &o.dims, &o.consts, &base, o.seed, &[], Capture::Shapes, o.remat, report)?;
            // The splits are counted on `--threads`, which `presk plan`
            // reads from `PRESK_THREADS`: the library does not read the
            // environment.
            if let Some(t) = o.threads {
                captured.machine.workers = t.max(1);
            }
            let model_machine = stored.unwrap_or_else(|| captured.machine.clone());
            let planned = match presk_driver::layout::choose(&fit, &model_machine, &captured, Space::Full) {
                Ok(p) => p,
                Err(u) => {
                    report.line(&format!("presk plan: the profile's layout model prices no layout of `{}` in {}: it was calibrated in {} only", o.entry, u.dtypes.join(", "), u.types.join(" and ")));
                    return Err(Stop(2));
                }
            };
            let profile_shown = o.profile.as_deref().map(shown).unwrap_or_default();
            let p = Plan { realization: realization.clone(), realization_ref: None, profile: o.profile.as_ref().map(|p| p.display().to_string()), layout: planned.layout.clone() };
            if let Err(e) = std::fs::write(art("plan.json"), p.to_json()) {
                report.line(&format!("presk explain: {e}"));
                return Err(Stop(2));
            }
            artifacts.push(Artifact { what: "the layout plan".into(), path: "artifacts/plan.json".into(), provenance: Provenance::Computed });
            (p, PlanSrc::Computed, Some((fit, model_machine, profile_shown)), Some(captured), Some(planned))
        }
        (None, None) => (Plan::from_realization(realization.clone()), PlanSrc::None, None, None, None),
    };
    presk_driver::plan::check_plan(&plan, &xcore, &sm, report)?;

    // ----- the kernels and the host ------------------------------------------------
    let hosted = presk_driver::layout::kernels_with_host(&xcore, &prog, &o.entry, dims.clone(), &plan, o.seed).map_err(|e| e.message);
    let host_exported = match &hosted {
        Ok(h) => match &h.recording {
            Ok(rec) => host_text(o, &xcore, &prog, entry, &dims, &realization, h, rec),
            Err(_) => None,
        },
        Err(_) => None,
    };
    let (host_c, c_names) = match host_exported {
        Some((text, of, lines)) => (Some(text), Some((of, lines))),
        None => (None, None),
    };

    // ----- the targets ---------------------------------------------------------------
    let realize_spec = match (&real_src, &o.realize) {
        (RealSrc::Solved, _) => format!("@{}", art("realization.json").display()),
        (_, Some(r)) => r.clone(),
        _ => "f64".to_string(),
    };
    let plan_spec = match &plan_src {
        PlanSrc::Computed => Some(format!("@{}", art("plan.json").display())),
        PlanSrc::Given(_) => o.plan.clone(),
        PlanSrc::None => None,
    };
    let c = o.targets.contains(&Target::C).then(|| {
        let dir = art("c");
        let _ = std::fs::remove_dir_all(&dir);
        let mut kept = Collect::default();
        let opts = presk_driver::export_c::Options { file: &o.file, entry: Some(&o.entry), dims: &o.dims, consts: &o.consts, realize: &realize_spec, plan: plan_spec.as_deref(), weights: o.weights.clone(), seed: o.seed, no_exact: o.no_exact, straight_line: false, serve: false, out: &dir, remat: o.remat };
        let real = o.out.join("artifacts").display().to_string();
        let said = |kept: &Collect| -> String {
            kept.items
                .iter()
                .map(|r| match r {
                    Reported::Diagnostics(t, _) => t.clone(),
                    Reported::Line(t) | Reported::Text(t) => t.clone(),
                })
                .collect::<Vec<_>>()
                .join("\n")
                .replace(&real, &format!("{DOC}/artifacts"))
        };
        match presk_driver::export_c::export(&opts, &mut kept) {
            Ok(()) => {
                let mut files: Vec<(String, usize)> = std::fs::read_dir(&dir).into_iter().flatten().flatten().filter(|e| e.path().is_file()).map(|e| (e.file_name().to_string_lossy().to_string(), std::fs::read_to_string(e.path()).map(|t| t.lines().count()).unwrap_or(0))).collect();
                files.sort();
                let kernels = hosted.as_ref().map(|h| h.listing.kernels.clone()).unwrap_or_default();
                Ok(CExport { line: said(&kept), files, kernels, kernels_c: std::fs::read_to_string(dir.join("kernels.c")).ok() })
            }
            Err(_) => Err(said(&kept)),
        }
    });
    if let Some(Ok(_)) = &c {
        artifacts.push(Artifact { what: "the C program".into(), path: "artifacts/c/".into(), provenance: Provenance::Computed });
    }
    let pytorch = o.targets.contains(&Target::Pytorch).then(|| {
        let dir = art("pytorch");
        let _ = std::fs::remove_dir_all(&dir);
        let (files, diags) = presk_export_pytorch::export(&sm, &prog, &typed, &presk_export_pytorch::Options { shapes_check: false, stack: presk_export_pytorch::StackMode::Loop });
        if presk_syntax::diag::has_errors(&diags) {
            return Err(presk_syntax::diag::render_all(&diags, &sm, false));
        }
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let mut out = Vec::new();
        for f in files {
            std::fs::write(dir.join(&f.name), &f.content).map_err(|e| e.to_string())?;
            out.push((f.name, f.content));
        }
        Ok(out)
    });
    if let Some(Ok(_)) = &pytorch {
        artifacts.push(Artifact { what: "the PyTorch module".into(), path: "artifacts/pytorch/".into(), provenance: Provenance::Computed });
    }
    let measured = o.measure.then(|| measure(o, &xcore, &prog, &dims, &realization, &plan, !matches!(plan_src, PlanSrc::None), report));

    // ----- the provenance table ------------------------------------------------------
    let layouts = match (&captured, &planned) {
        (Some(c), Some(p)) => {
            let mut nests: Vec<(String, String, usize)> = Vec::new();
            for k in &p.costed {
                let nest = &c.launches[k.launch].nest;
                if !nests.iter().any(|(n, _, _)| n == nest) {
                    let chosen = p.layout.get(nest).map(|x| x.to_string()).unwrap_or_else(|| k.scored[0].1.to_string());
                    nests.push((nest.clone(), chosen, k.space));
                }
            }
            Some(presk_driver::provenance::Layouts { nests })
        }
        _ => None,
    };
    let entry_name = core.func(entry).name.clone();
    // The native column (plan 0.13.0 step 5, D138): each listed kernel
    // compiled as the native backend compiles it, every byte with its line,
    // for the JSON and the page; the document does not print it.
    let native = match &hosted {
        Ok(h) if (o.json || o.html) && o.targets.contains(&Target::Native) => presk_cranelift::machine_code(&h.listing.kernels).ok(),
        _ => None,
    };
    let table = presk_driver::provenance::table(&presk_driver::provenance::Stages {
        sm: &sm,
        prog: &prog,
        typed: &typed,
        core: &core,
        lowered: &lowered,
        entry: &entry_name,
        ad: &ad_program,
        ad_links: &ad_links,
        precision: Some(presk_driver::provenance::Precision { links: &pcore_links, env: env.as_ref().map(|(e, _)| e), roles: &roles, realization: &realization, solved: solved.as_ref() }),
        opt: o.opt.then_some((&xcore, &opt_links[..], &opt_ends[..])),
        layouts,
        kernels: hosted.as_ref().ok().map(|h| (&h.listing.kernels[..], &h.launched[..])),
        c: match (&c, &c_names) {
            (Some(Ok(_)), Some((of, lines))) => Some((&of[..], &lines[..])),
            _ => None,
        },
        native: native.as_deref(),
    });
    Ok(State { measured, o, file, sm, prog, typed, core, analysis, entry, dims, diags: Vec::new(), pcore, env, env_src, sens, sens_none, sweep, solved, roles, realization, real_src, bounded, xcore, opt_summary, opt_report, before_opt, plan, plan_src, profile, captured, planned, hosted, host_c, c, pytorch, artifacts, table, ad_program, native })
}

/// The C host's functions, and the C names and lines of the kernels.
type HostText = (String, Vec<(usize, usize)>, Vec<(usize, usize)>);

/// The host as the C export prints it, from the recording of the shapes'
/// run: its functions, without the file's preamble; and which `k<n>` of
/// `kernels.c` each kernel is, with its lines.
#[allow(clippy::too_many_arguments)]
fn host_text(o: &Options, core: &Program, prog: &ElabProgram, entry: FuncId, dims: &[(String, u64)], r: &Realization, h: &Hosted, rec: &presk_kernel::record::Recording) -> Option<HostText> {
    let params_module = core.func(entry).params.first().and_then(|p| match &p.ty {
        presk_types::ty::Ty::Params(presk_types::ty::ModRef::Inst { def, .. }) => Some(prog.module(*def).name.clone()),
        _ => None,
    });
    let meta = presk_export_c::host::Meta {
        file: shown(&o.file),
        entry: o.entry.clone(),
        dims: dims.to_vec(),
        consts: o.consts.clone(),
        realize: r.label(),
        seed: o.seed,
        weights: None,
        version: String::new(),
        commit: None,
        exact: false,
        plan: None,
        threads: o.threads.unwrap_or_else(|| presk_kernel::cost::Machine::detect().workers).max(1),
        quant: r.quant.iter().map(|(p, q)| (p.clone(), if q.scheme == presk_rt::QuantScheme::Pow2 { "pow2" } else { "absmax" }.to_string())).collect(),
    };
    let args: Vec<presk_export_c::host::Arg> = h.names.iter().enumerate().map(|(i, n)| presk_export_c::host::Arg { name: n.clone(), weights: i == 0 && params_module.is_some() }).collect();
    let weights_type = params_module.as_ref().map(|m| format!("params {m}"));
    let exported = presk_export_c::host::export(&meta, &h.listing.kernels, rec, &args, weights_type.as_deref()).ok()?;
    let (_, model) = exported.files.iter().find(|(n, _)| n == "model.c")?;
    // From the first function on: the includes and the kernels' prototypes
    // are the C export's business.
    let mut start = model.find("\nstatic void ").or_else(|| model.find("\nvoid ")).map(|i| i + 1).unwrap_or(0);
    // With the comment that heads it.
    if let Some(prev) = model[..start.saturating_sub(1)].rfind('\n')
        && model[prev + 1..start].starts_with("/*")
    {
        start = prev + 1;
    }
    Some((model[start..].to_string(), exported.kernel_of.clone(), exported.kernel_lines.clone()))
}

// ----- the header ------------------------------------------------------------------

fn title(s: &State) -> String {
    let o = s.o;
    let mut t = format!("`{}` of `{}`", o.entry, s.file);
    match (&s.real_src, o.budget) {
        (RealSrc::Solved, Some(b)) => t.push_str(&format!(", under a budget of {b:e}")),
        (RealSrc::Named(n), _) => t.push_str(&format!(", in {n}")),
        (RealSrc::Given(p), _) => t.push_str(&format!(", under the realization `{p}`")),
        _ => {}
    }
    if let Some((_, _, p)) = &s.profile
        && !p.is_empty()
    {
        let p = Path::new(p);
        let name = match (p.parent().and_then(|d| d.file_name()), p.file_name()) {
            (Some(d), Some(f)) => format!("{}/{}", d.to_string_lossy(), f.to_string_lossy()),
            (None, Some(f)) => f.to_string_lossy().to_string(),
            _ => String::new(),
        };
        t.push_str(&format!(", planned under the profile `{name}`"));
    }
    t
}

fn header(s: &State) -> Vec<Block> {
    let o = s.o;
    let mut rows: Vec<Vec<String>> = vec![
        vec!["Program".into(), format!("`{}`", s.file), "the command line".into()],
        vec!["Entry".into(), format!("`{}`", o.entry), format!("`--entry {}`", o.entry)],
    ];
    let fdims: Vec<String> = s.dims.iter().map(|(n, v)| format!("`{n} = {v}`")).collect();
    rows.push(vec!["Dimensions".into(), if fdims.is_empty() { "none".into() } else { fdims.join(", ") }, if o.dims.is_empty() { "none given".into() } else { o.dims.iter().map(|d| format!("`--dim {d}`")).collect::<Vec<_>>().join(" ") }]);
    let consts = s.prog.units.iter().flat_map(|u| u.consts.iter().map(|(n, c)| format!("`{n} = {}`", c.value))).collect::<Vec<_>>();
    rows.push(vec!["Constants".into(), if consts.is_empty() { "none".into() } else { consts.join(", ") }, if o.consts.is_empty() { "the files'".into() } else { o.consts.iter().map(|c| format!("`--const {c}`")).collect::<Vec<_>>().join(" ") }]);
    rows.push(vec!["Weights".into(), match &o.weights { Some(w) => format!("`{}`", shown(w)), None => format!("the initialization, seed {}", o.seed) }, match &o.weights { Some(w) => format!("`--weights {}`", shown(w)), None => "no `--weights`".into() }]);
    rows.push(vec![
        "Realization".into(),
        match &s.real_src {
            RealSrc::Default => "every site in `f64`".into(),
            RealSrc::Named(n) => format!("every site in `{n}`"),
            RealSrc::Given(p) => format!("`{p}`"),
            RealSrc::Solved => format!("solved under a budget of {:e} relative to the output", o.budget.unwrap_or(0.0)),
        },
        match &s.real_src {
            RealSrc::Default => "no `--budget`, `--realize` or `--plan`".into(),
            RealSrc::Named(n) => format!("`--realize {n}`"),
            RealSrc::Given(p) => if o.plan.is_some() { format!("`--plan @{p}`") } else { format!("`--realize @{p}`") },
            RealSrc::Solved => format!("`--budget {:e}`", o.budget.unwrap_or(0.0)),
        },
    ]);
    rows.push(vec!["Exact reference".into(), if o.no_exact { "none".into() } else { format!("MPFR at {} bits", o.bits) }, if o.no_exact { "`--no-exact`".into() } else { format!("`--bits {}`", o.bits) }]);
    rows.push(vec![
        "Machine".into(),
        match &o.profile {
            Some(p) => format!("the profile `{}`", shown(p)),
            None => "no profile: no layout plan, the solver prices bytes".into(),
        },
        match &o.profile {
            Some(p) => format!("`--profile {}`", shown(p)),
            None => "no `--profile`".into(),
        },
    ]);
    if let Some(t) = o.threads {
        rows.push(vec!["Threads".into(), t.to_string(), format!("`--threads {t}`")]);
    }
    rows.push(vec!["Differentiation".into(), format!("`--remat {}`", remat_name(o.remat)), "the command line".into()]);
    rows.push(vec!["Optimizer (e-graph)".into(), if o.opt { "on".into() } else { "off".into() }, if o.opt { "`--opt`".into() } else { "no `--opt`".into() }]);
    let targets: Vec<&str> = o.targets.iter().map(|t| match t {
        Target::Native => "native (Cranelift)",
        Target::C => "C",
        Target::Pytorch => "PyTorch",
    }).collect();
    rows.push(vec!["Targets".into(), targets.join(", "), format!("`--target {}`", o.targets.iter().map(|t| match t {
        Target::Native => "native",
        Target::C => "c",
        Target::Pytorch => "pytorch",
    }).collect::<Vec<_>>().join(","))]);
    let mut blocks = vec![
        Block::Prose("## The parameters".into()),
        Block::Table { head: vec!["Parameter".into(), "Value".into(), "Where it comes from".into()], rows },
        Block::Prose(format!("Compiler: presk {}. Each section ends with the commands that print what it shows; `$DOC` is this document's directory, which holds every file it names.", env!("CARGO_PKG_VERSION"))),
    ];
    let arts: Vec<Vec<String>> = s
        .artifacts
        .iter()
        .map(|a| match &a.provenance {
            Provenance::Computed => vec![a.what.clone(), format!("`{DOC}/{}`", a.path), "computed in this run".into(), String::new()],
            Provenance::Given { path, sha256 } => vec![a.what.clone(), format!("`{path}`"), "given".into(), format!("`{}…`", &sha256[..16])],
            Provenance::Omitted(why) => vec![a.what.clone(), String::new(), format!("not computed: {why}"), String::new()],
        })
        .collect();
    if !arts.is_empty() {
        blocks.push(Block::Prose("**The artifacts** it was made from or wrote:".into()));
        blocks.push(Block::Table { head: vec!["What".into(), "Path".into(), "Provenance".into(), "SHA-256".into()], rows: arts });
        blocks.push(Block::Prose(format!("`{DOC}/artifacts/SHA256SUMS` holds the checksum of every file written there.")));
    }
    blocks
}

fn remat_name(r: presk_core::ad::Remat) -> &'static str {
    match r {
        presk_core::ad::Remat::Store => "store",
        presk_core::ad::Remat::Recompute => "recompute",
    }
}

// ----- the commands ------------------------------------------------------------------

/// `--entry E --dim B=4 --const D=8`.
fn entry_args(s: &State) -> String {
    let mut a = format!("--entry {}", s.o.entry);
    for d in &s.o.dims {
        a.push_str(&format!(" --dim {d}"));
    }
    a.push_str(&consts_args(s));
    a
}

fn consts_args(s: &State) -> String {
    s.o.consts.iter().map(|c| format!(" --const {c}")).collect()
}

fn weights_args(s: &State) -> String {
    s.o.weights.as_ref().map(|w| format!(" --weights {}", shown(w))).unwrap_or_default()
}

fn remat_args(s: &State) -> String {
    if s.o.remat == presk_core::ad::Remat::Store { String::new() } else { format!(" --remat {}", remat_name(s.o.remat)) }
}

fn profile_args(s: &State) -> String {
    s.o.profile.as_ref().map(|p| format!(" --profile {}", shown(p))).unwrap_or_default()
}

/// `--realize …` or `--plan @…` as the kernels' commands take them.
fn plan_args(s: &State) -> String {
    match (&s.plan_src, &s.real_src) {
        (PlanSrc::Computed, _) => format!(" --plan @{DOC}/artifacts/plan.json"),
        (PlanSrc::Given(p), _) => format!(" --plan @{p}"),
        (PlanSrc::None, RealSrc::Solved) => format!(" --realize @{DOC}/artifacts/realization.json"),
        (PlanSrc::None, RealSrc::Given(p)) => format!(" --realize @{p}"),
        (PlanSrc::None, RealSrc::Named(n)) => format!(" --realize {n}"),
        (PlanSrc::None, RealSrc::Default) => String::new(),
    }
}

/// `--realize …` of the realization alone.
fn realize_args(s: &State) -> String {
    match &s.real_src {
        RealSrc::Solved => format!(" --realize @{DOC}/artifacts/realization.json"),
        RealSrc::Given(p) if s.o.plan.is_none() => format!(" --realize @{p}"),
        RealSrc::Given(_) => format!(" --realize @{DOC}/artifacts/realization.json"),
        RealSrc::Named(n) => format!(" --realize {n}"),
        RealSrc::Default => String::new(),
    }
}

// ----- the sections ------------------------------------------------------------------

/// The realization's sites by role and type: decisions narrowed, by type;
/// followers; carried values.
fn roles_count(s: &State) -> (BTreeMap<String, usize>, usize, usize, usize) {
    let mut by_type: BTreeMap<String, usize> = BTreeMap::new();
    let (mut decisions, mut follows, mut carried) = (0, 0, 0);
    for (site, d) in &s.realization.storage {
        match s.roles.0.get(site) {
            Some(Role::Follows(_)) => follows += 1,
            Some(Role::Carried) => carried += 1,
            _ => {
                decisions += 1;
                *by_type.entry(d.name().to_string()).or_default() += 1;
            }
        }
    }
    (by_type, decisions, follows, carried)
}

fn summary(s: &State) -> Section {
    let mut sec = Section::new("0", "Summary");
    let mut rows: Vec<Vec<String>> = Vec::new();
    let default = s.realization.default.name();
    let (by_type, decisions, follows, carried) = roles_count(s);
    let mut real = if s.realization.storage.is_empty() {
        format!("every site in `{default}`")
    } else {
        let types = by_type.iter().map(|(t, n)| format!("{n} in `{t}`")).collect::<Vec<_>>().join(", ");
        let mut r = format!("{decisions} stored value{} away from `{default}`: {types}; the rest in `{default}`", if decisions == 1 { "" } else { "s" });
        if follows + carried > 0 {
            r.push_str(&format!(". The realization also types {} site{} whose type changes no bit ({follows} follower{}, {carried} carried; §6.3)", follows + carried, if follows + carried == 1 { "" } else { "s" }, if follows == 1 { "" } else { "s" }));
        }
        r
    };
    if !s.realization.params.is_empty() {
        let n = s.realization.params.len();
        real.push_str(&format!("; {n} parameter{} stored in {} own type{} (§6.3)", if n == 1 { "" } else { "s" }, if n == 1 { "its" } else { "their" }, if n == 1 { "" } else { "s" }));
    }
    let kind = if matches!(s.real_src, RealSrc::Solved) { Kind::Decision } else { Kind::Parameter };
    rows.push(vec!["Realization".into(), real, kind.name().into(), "§6".into()]);
    if let Some(sv) = &s.solved {
        let sol = &sv.solution;
        rows.push(vec!["Budget".into(), format!("{:e} of the output's magnitude {}: {} absolute", s.o.budget.unwrap_or(0.0), sci(sv.reference_max), sci(sol.budget_abs)), Kind::Parameter.name().into(), "§6.3".into()]);
        if let Some(g) = sol.measured {
            rows.push(vec!["Measured gap".into(), format!("{} against the exact reference, {} the budget", sci(g), if g <= sol.budget_abs { "inside" } else { "outside" }), Kind::Measurement.name().into(), "§6.3".into()]);
        }
        rows.push(vec!["Solver's estimate".into(), format!("{}: it ranks, it does not guarantee", sci(sol.estimated_error)), Kind::Estimate.name().into(), "§6.3".into()]);
        let unit = if s.o.profile.is_some() { "s, predicted by the profile's roofline" } else { "bytes moved" };
        rows.push(vec!["Cost".into(), format!("{:.4e} against {:.4e} all-`f64` ({unit}): {:.2}×", sol.bytes, sol.baseline_bytes, sol.speedup()), Kind::Estimate.name().into(), "§6.3".into()]);
    }
    if let Some(b) = &s.bounded {
        if let Some(r) = b.runs.iter().find(|r| r.model == presk_precision::propagate::Model::Conditioned) {
            let samples = s.env.as_ref().map(|(e, _)| e.samples).unwrap_or(1);
            rows.push(vec!["Bound, rule `cond`".into(), format!("{} ({} relative)", sci(r.output.err), sci(r.output.relative())), Kind::Bound.name().into(), format!("§6.4, conditional on the envelope's {samples} sample{}", if samples == 1 { "" } else { "s" })]);
        }
        if s.solved.is_none()
            && let Some(g) = b.measured
        {
            rows.push(vec!["Measured gap".into(), format!("{} against the exact reference", sci(g)), Kind::Measurement.name().into(), "§6.4".into()]);
        }
    }
    match (&s.planned, &s.plan_src) {
        (Some(p), _) => {
            let split = p.layout.values().filter(|l| matches!(l.split, Some(presk_rt::layout::Split::At { .. }))).count();
            let tiled = p.layout.values().filter(|l| l.tile.is_some()).count();
            let total: f64 = p.costed.iter().map(|c| c.scored[0].0 * s.captured.as_ref().map(|k| k.launches[c.launch].count as f64).unwrap_or(1.0)).sum();
            rows.push(vec!["Layout plan".into(), format!("{} nests, {} points costed, {tiled} tiled, {split} split; predicted {} a call over the nests it prices", p.layout.len(), p.total_points, duration(total)), format!("{}, {}", Kind::Decision.name(), Kind::Estimate.name()), "§8".into()]);
        }
        (None, PlanSrc::Given(path)) => rows.push(vec!["Layout plan".into(), format!("given, `{path}`: {} nests with a layout", s.plan.layout.len()), Kind::Parameter.name().into(), "§8".into()]),
        _ => rows.push(vec!["Layout plan".into(), "none: every nest at its default layout".into(), Kind::Fact.name().into(), "§8".into()]),
    }
    if let Ok(h) = &s.hosted {
        let l = &h.listing;
        let before = l.launches + l.fused;
        rows.insert(rows.len() - 1, vec!["Fusion".into(), format!("{before} launches without it, {} with it: {} saved", l.launches, l.fused), Kind::Decision.name().into(), "§7".into()]);
        rows.push(vec!["Kernels".into(), format!("{} distinct, {} launches a call", l.kernels.len(), l.launches), Kind::Fact.name().into(), "§9".into()]);
    }
    if let Some(Ok(c)) = &s.c {
        let lines: usize = c.files.iter().map(|(_, n)| n).sum();
        rows.push(vec!["C program".into(), format!("{} files, {lines} lines with its runtime", c.files.len()), Kind::Fact.name().into(), "§10".into()]);
    }
    sec.table(&["What", "Outcome", "Kind", "Where"], rows);
    sec.prose("A *fact* is a property of the program or of the compilation; a *decision* is what the compiler chose; a *bound* holds for the inputs its envelope covers and names them; an *estimate* ranks and does not guarantee; a *measurement* was taken in this run, against the exact reference unless it is a time.");
    sec
}

fn source(s: &State) -> Section {
    let mut sec = Section::new("1", "The source").purpose("The entry's file and the files it imports, as the compiler reads them.");
    for (_, f) in s.sm.files() {
        let width = f.text.lines().count().to_string().len().max(3);
        let numbered: String = f.text.lines().enumerate().map(|(i, l)| format!("{:>width$}  {l}\n", i + 1)).collect();
        sec.prose(format!("**`{}`**", shown(Path::new(&f.name))));
        sec.code("text", numbered);
    }
    if s.diags.is_empty() {
        sec.prose("Diagnostics: none (`presk check` reports no error and no warning).");
    } else {
        sec.prose("Diagnostics, as `presk check` reports them:");
        sec.code("text", s.diags.join(""));
    }
    sec.command(format!("presk check {}{}", s.file, consts_args(s)));
    sec
}

fn elaboration(s: &State) -> Section {
    let mut sec = Section::new("2", "Elaboration").purpose("Elaboration resolves the imports, binds the constants, instantiates every module with its dimensions, and checks shapes, regularity classes and effects. Its output is the program with every instance named and every parameter's shape known.");
    sec.listing("text", "elab.txt", presk_types::elab::dump(&s.prog));
    sec.prose("**Types, regularity and effects**, as `presk check --summary` prints them:");
    let mut t = presk_types::check::summary(&s.prog, &s.typed);
    t.push_str(&format!("-- regularity and effects --\n{}", presk_core::analysis::summary(&s.core, &s.analysis)));
    if !s.typed.training.is_empty() {
        t.push_str(&format!("-- training --\n{}", presk_types::check::training::summary(&s.prog, &s.typed.training)));
    }
    if !s.typed.inference.is_empty() {
        t.push_str(&format!("-- inference --\n{}", presk_types::check::inference::summary(&s.prog, &s.typed.inference)));
    }
    sec.listing("text", "summary.txt", t);
    let asserts: Vec<Vec<String>> = s
        .prog
        .units
        .iter()
        .flat_map(|u| u.asserts.iter().map(move |(span, ok)| (u, span, ok)))
        .map(|(_, span, ok)| {
            let text = s.sm.slice(*span).trim_start_matches("static assert").trim().to_string();
            let file = s.sm.file(span.file);
            let (line, _) = file.line_col(span.start);
            vec![format!("`{text}`"), format!("`{}:{line}`", shown(Path::new(&file.name))), if *ok { "discharged: valid in Presburger arithmetic over the constants and the facts before it, then kept as a fact".into() } else { "**failed**".into() }]
        })
        .collect();
    if asserts.is_empty() {
        sec.prose("**Static assertions:** none.");
    } else {
        sec.prose("**Static assertions**, each decided at elaboration by the decision procedure for Presburger arithmetic (`presk_types::dim::entails`), the dimensions' integer linear fragment with constant moduli:");
        sec.table(&["Assertion", "Where", "Outcome"], asserts);
    }
    if let Ok(h) = &s.hosted
        && !h.params.is_empty()
    {
        let total: usize = h.params.iter().map(|(_, n)| n).sum();
        sec.prose(format!("**The parameters** of `{}`, its first argument: {} tensors, {total} reals at these dimensions.", s.o.entry, h.params.len()));
        sec.listing_table(&["Parameter", "Reals"], h.params.iter().map(|(p, n)| vec![format!("`{p}`"), n.to_string()]).collect());
    }
    sec.command(format!("presk dump {} --stage elab{}", s.file, consts_args(s)));
    sec.command(format!("presk check {}{} --summary", s.file, consts_args(s)));
    sec
}

fn core(s: &State) -> Section {
    let mut sec = Section::new("3", "PSK-Core").purpose("PSK-Core is the program in A-normal form: one typed operation per line, the parameter trees explicit, a `stack{L}` a `fold`.");
    sec.prose("**A site is named `<function>/%<n>`**, `n` the number of its variable, whether the `let` has a source name (`ms`) or not (`%5`). That name is the thread every later section follows: the envelope, the sensitivities and the realization are keyed by it, a nest is named by the site it computes, and the C program comments each launch with it.");
    let reached = reach(&s.core, s.entry);
    sec.prose(format!("The functions `{}` reaches ({} of the program's {}):", s.o.entry, reached.len(), s.core.funcs.len()));
    let text: String = reached.iter().map(|f| presk_core::pretty::pretty_func(s.core.func(*f), &s.core, &s.prog)).collect::<Vec<_>>().join("\n");
    sec.listing("text", "core.txt", text);
    sec.command(format!("presk dump {} --stage core{}", s.file, consts_args(s)));
    sec
}

fn differentiation(s: &State) -> Section {
    let mut sec = Section::new("4", "Differentiation").purpose("Differentiation turns each `grad` into a reverse sweep: a `.vjp` that runs the function forward, keeps what the sweep needs (`--remat store`) or recomputes it (`--remat recompute`), and propagates the cotangents back.");
    let before: BTreeSet<&str> = s.core.funcs.iter().map(|f| f.name.as_str()).collect();
    let entry = s.xcore.find(&s.core.func(s.entry).name).unwrap_or(s.entry);
    let added: Vec<FuncId> = reach(&s.xcore, entry).into_iter().filter(|f| !before.contains(s.xcore.func(*f).name.as_str())).collect();
    if added.is_empty() {
        sec.prose(format!("`{}` calls no `grad`, so its compiled program holds no derivative.", s.o.entry));
    } else {
        sec.prose(format!("The derivatives `{}` reaches, under `--remat {}`:", s.o.entry, remat_name(s.o.remat)));
        sec.listing("text", "core-after-ad.txt", added.iter().map(|f| presk_core::pretty::pretty_func(s.xcore.func(*f), &s.xcore, &s.prog)).collect::<Vec<_>>().join("\n"));
        sec.command(format!("presk dump {} --stage core-after-ad{}{}", s.file, consts_args(s), remat_args(s)));
    }
    if !s.sweep.is_empty() {
        sec.prose(format!("The precision analysis differentiates `{}` once more, for its own use: the sensitivities of §6.2 are read off one reverse sweep at {} bits, the `.vjp` below, which keeps the adjoint of every intermediate. It is not part of the compiled program.", s.o.entry, s.o.bits));
        sec.attach("sensitivity-sweep.txt", s.sweep.join("\n"), "the sweep's program");
    }
    sec
}

fn optimizer(s: &State) -> Section {
    let mut sec = Section::new("5", "The e-graph optimizer").purpose("The optimizer saturates an e-graph of each function with the rewrite rules and extracts the cheapest form under its cost model.");
    match &s.opt_summary {
        Some(t) => {
            sec.code("text", t.clone());
            if let (Some(before), Some(report)) = (&s.before_opt, &s.opt_report) {
                let entry = s.xcore.find(&s.core.func(s.entry).name).unwrap_or(s.entry);
                // The functions the report says changed: every rewritten
                // body has its variables numbered anew, so a text that
                // differs is not one that changed.
                let rewritten: BTreeSet<&str> = report.funcs.iter().filter(|f| matches!(f.outcome, presk_opt::Outcome::Rewritten { lets_before, lets_after, cost_before, cost_after, .. } if lets_before != lets_after || cost_before != cost_after)).map(|f| f.name.as_str()).collect();
                let mut changed = 0;
                for f in reach(&s.xcore, entry) {
                    let after = s.xcore.func(f);
                    let Some(b) = before.find(&after.name) else { continue };
                    let (tb, ta) = (presk_core::pretty::pretty_func(before.func(b), before, &s.prog), presk_core::pretty::pretty_func(after, &s.xcore, &s.prog));
                    if rewritten.contains(after.name.as_str()) {
                        changed += 1;
                        sec.prose(format!("**`{}`**, before and after:", after.name));
                        sec.listing("text", &format!("opt-{}-before.txt", after.name.replace(['/', ' '], "_")), tb);
                        sec.listing("text", &format!("opt-{}-after.txt", after.name.replace(['/', ' '], "_")), ta);
                    }
                }
                if changed == 0 {
                    sec.prose(format!("No function `{}` reaches was rewritten. A function the optimizer visits has its variables numbered anew even when its lets and its cost stay, so the sites of the later sections are the rewritten program's.", s.o.entry));
                } else {
                    sec.prose("Every function the optimizer visits has its variables numbered anew, so the sites of the later sections are the rewritten program's.");
                }
            }
            sec.command(format!("presk dump {} --stage core-opt{}{}", s.file, consts_args(s), profile_args(s)));
        }
        None => sec.prose("Not run: `--opt` is off."),
    }
    sec
}

fn precision(s: &State) -> Section {
    let mut sec = Section::new("6", "Precision: from a budget to a type per site").purpose("A budget bounds the distance between what the machine computes and the mathematics. Four steps turn it into types: **measure** how large the values get (the envelope), **rank** where error matters (the sensitivities), **solve** for the cheapest types the estimate allows, then **discharge** the budget by running the candidate against the exact reference.");
    // 6.1
    let mut e = Section::new("6.1", "The envelope").purpose("The exact interpreter runs the entry once and records the largest |x| at every site, and at each sum and contraction what its terms add up to without cancelling: divided by the sum, its condition number.");
    match (&s.env, &s.env_src) {
        (Some((env, path)), src) => {
            e.code("text", text::calibrate(env, &s.o.entry, env.bits).join("\n"));
            let on = env.weights.as_deref().map(|w| format!("the weights `{}`", shown(Path::new(w)))).unwrap_or_else(|| "the initial weights".into());
            e.prose(format!("The envelope holds {} sites from **{} sample{}** on {on}, at {} bits; every bound below is conditional on it, and says so.", env.sites.len(), env.samples, if env.samples == 1 { "" } else { "s" }, env.bits));
            match src {
                Some(Provenance::Given { path: p, sha256 }) => e.prose(format!("Given: `{p}`, SHA-256 `{}…`.", &sha256[..16])),
                _ => e.command(format!("presk calibrate --precision {} {} --bits {}{} --out {path}", s.file, entry_args(s), s.o.bits, weights_args(s))),
            }
        }
        (None, _) => e.prose("Not measured: `--no-exact`, and no `--envelope` given. Without an envelope there is no bound and no solve."),
    }
    sec.sections.push(e);
    // 6.2
    let mut g = Section::new("6.2", "The sensitivities").purpose("`‖∂L/∂v‖₁` at each site, from one reverse sweep: how much an error at the site moves the output.");
    match (&s.sens, &s.sens_none) {
        (Some(sens), _) => {
            g.listing("text", "sensitivity.txt", text::sensitivity(sens, &s.o.entry, s.o.bits, 30).join("\n"));
            g.command(format!("presk dump {} --stage sensitivity {}{} --sites 30", s.file, entry_args(s), weights_args(s)));
        }
        (None, Some(why)) => g.prose(format!("None: {why}")),
        (None, None) => g.prose("None."),
    }
    sec.sections.push(g);
    // 6.3
    let mut v = Section::new("6.3", "The realization");
    match &s.solved {
        Some(sv) => {
            v.purpose.push("The solver ranks the values the program stores by sensitivity and narrows the cheapest first while its first-order estimate Σ s · u(type) · |v| stays under the budget, then measures the candidate against the exact reference and widens until the measurement holds.".into());
            let (env, _) = s.env.as_ref().expect("a solve has an envelope");
            v.code("text", text::solve(env, presk_precision::solve::Budget::Relative(s.o.budget.unwrap_or(0.0)), sv, 0).join("\n"));
            breakdown(s, sv, &mut v);
            v.command(format!("presk solve {} --precision {DOC}/artifacts/envelope.json --budget {:e}{}{}{} --sites 30 --out {DOC}/artifacts/realization.json", s.file, s.o.budget.unwrap_or(0.0), weights_args(s), profile_args(s), if s.o.no_exact { " --no-verify" } else { "" }));
        }
        None => {
            v.purpose.push("A realization gives each site a machine type; a site it does not name takes the default.".into());
            match &s.real_src {
                RealSrc::Default => v.prose("No budget and no realization given: every site in `f64`."),
                RealSrc::Named(n) => v.prose(format!("Given: every site in `{n}`.")),
                _ => {
                    v.prose(format!("Given: default `{}`, {} sites typed apart, {} parameters stored otherwise.", s.realization.default.name(), s.realization.storage.len(), s.realization.params.len() + s.realization.quant.len()));
                    stored_values(s, &mut v);
                    roles_note(s, &mut v);
                    params_table(s, &mut v);
                }
            }
        }
    }
    sec.sections.push(v);
    // 6.4
    let mut b = Section::new("6.4", "The bound and the measurement").purpose("Under the realization, each rule's bound on the output's error, beside the gap measured against the exact reference.");
    match (&s.bounded, &s.env) {
        (Some(bd), Some((env, path))) => {
            b.code("text", text::bound(env, path, &s.realization.label(), bd, 8).join("\n"));
            if let (Some(r), Some(g)) = (bd.runs.iter().find(|r| r.model == presk_precision::propagate::Model::Conditioned), bd.measured) {
                b.prose(format!("The sound rule (`cond`) bounds the error by {}, conditional on the envelope's {} sample{}; the measurement is {}, {:.1e}× under it.", sci(r.output.err), env.samples, if env.samples == 1 { "" } else { "s" }, sci(g), r.output.err / g.max(f64::MIN_POSITIVE)));
            }
            b.command(format!("presk check --precision {path} {}{} --model all --sites 8{}", s.file, realize_args(s), if s.o.no_exact { "" } else { " --measure" }));
        }
        _ => b.prose("No bound: there is no envelope."),
    }
    sec.sections.push(b);
    sec
}

fn op_of(s: &State, site: &str) -> String {
    match site_let(&s.pcore, site) {
        Some((f, l)) => {
            let line = presk_core::pretty::pretty_let(f, l, &s.pcore, &s.prog);
            let rhs = line.split_once(" = ").map(|(_, r)| r).unwrap_or(&line);
            format!("`{rhs}`")
        }
        None => String::new(),
    }
}

fn role_name(r: Option<&Role>) -> String {
    match r {
        None | Some(Role::Decision) => "decision".into(),
        Some(Role::Follows(of)) => format!("follows {}", of.iter().map(|x| format!("`{x}`")).collect::<Vec<_>>().join(", ")),
        Some(Role::Carried) => "carried".into(),
    }
}

/// The solver's breakdown: every site it offered, its role and type, and
/// its term of the estimate.
fn breakdown(s: &State, sv: &Solved, v: &mut Section) {
    let sum: f64 = sv.breakdown.iter().map(|c| c.err).sum();
    v.prose(format!(
        "**Why each site has its type.** The estimate is Σ s · u(type) · |v| over the sites, s the sensitivity, |v| the envelope's magnitude and u the unit roundoff (2⁻⁵³ for `f64`, 2⁻²⁴ for `f32`, 2⁻⁸ for `bf16`). The terms below sum to {}, the solver's {}. A **decision** is a value the program stores, whose type is the solver's choice; a value that **follows** another is rounded where that one computes it (a fused member in its group's root, a parameter's field in the nests that read it), and its term is charged at that type; a **carried** value (a call's or a fold's result, a parameter tree) is another site's value, with no term and no cost. A **parameter** is a decision keyed by its path (since 0.12.0): its term is its rounding at the narrowest of its own type and the types of the nests that read it, s its cotangent's 1-norm and |v| its largest magnitude, and the field sites that read it carry none.",
        sci(sum),
        sci(sv.solution.estimated_error)
    ));
    let mut rows: Vec<&presk_precision::solve::SiteCost> = sv.breakdown.iter().collect();
    rows.sort_by(|a, b| b.err.partial_cmp(&a.err).unwrap_or(std::cmp::Ordering::Equal).then_with(|| site_order(&a.name).cmp(&site_order(&b.name))));
    let share = |c: &presk_precision::solve::SiteCost| if sum > 0.0 { format!("{:.3} %", 100.0 * c.err / sum) } else { "-".into() };
    let unit = if sv.time { "s" } else { "bytes" };
    let decided: Vec<&&presk_precision::solve::SiteCost> = rows.iter().filter(|c| c.role == Role::Decision).collect();
    let decisions: Vec<Vec<String>> = decided.iter().map(|c| vec![format!("`{}`", c.name), if c.param { "a parameter, by its path".into() } else { op_of(s, &c.name) }, format!("`{}`", c.dtype.name()), format!("{}", c.elems), sci(c.mag), sci(c.sens), sci(c.err), share(c)]).collect();
    v.prose("**The decisions**, each with its term of the estimate:");
    v.listing_table(&["Site", "Operation", "Type", "Elements", "|v|", "s", "Term", "Share"], decisions);
    v.prose(format!("**Why each decision has its type.** The solver tries every narrowing in order of what it saves per unit of error it adds, the best first, and takes one when the estimate stays under the budget ({} absolute); then it measures, and widens the largest share by one type while the measured gap exceeds the budget. Savings are in {unit}{}.", sci(sv.solution.budget_abs), if sv.time { ", the profile's roofline" } else { "" }));
    let reasons: Vec<String> = decided.iter().map(|c| format!("- `{}`: {}.", c.name, why(sv, &c.name, unit))).collect();
    if reasons.len() > crate::doc::LONG {
        v.prose(reasons[..crate::doc::EXCERPT].join("\n"));
        v.attach("reasons.txt", reasons.join("\n"), &format!("the first {} of {}; every decision's reason", crate::doc::EXCERPT, reasons.len()));
    } else {
        v.prose(reasons.join("\n"));
    }
    let others: Vec<Vec<String>> = rows.iter().filter(|c| c.role != Role::Decision).map(|c| vec![format!("`{}`", c.name), op_of(s, &c.name), role_name(Some(&c.role)), format!("`{}`", c.dtype.name()), sci(c.mag), sci(c.sens), sci(c.err), share(c)]).collect();
    if !others.is_empty() {
        v.prose("**The values rounded elsewhere, or carried.** A follower's type is the narrowest of the types it is rounded to where it is read; a carried value's is its own, never rounded there.");
        v.listing_table(&["Site", "Operation", "Role", "Type", "|v|", "s", "Term", "Share"], others);
    }
    roles_note(s, v);
}

/// The reason of a decision's type, from the solver's moves.
fn why(sv: &Solved, site: &str, unit: &str) -> String {
    use presk_precision::solve::Move;
    let moves: Vec<&Move> = sv.solution.moves.iter().filter(|m| match m {
        Move::Tried { site: x, .. } | Move::Widened { site: x, .. } => x == site,
    }).collect();
    let mut parts: Vec<String> = Vec::new();
    for m in &moves {
        match m {
            Move::Tried { to, saved, added, spent, taken: true, rank, .. } => parts.push(format!("narrowed to `{}` (move {rank}): saves {} {unit}, adds {} to the estimate, then {}", to.name(), sci(*saved), sci(*added), sci(spent + added))),
            Move::Widened { round, from, to, gap, .. } => parts.push(format!("widened from `{}` to `{}` at round {round}, the gap {} over the budget", from.name(), to.name(), sci(*gap))),
            _ => {}
        }
    }
    if parts.is_empty() {
        match moves.iter().find_map(|m| if let Move::Tried { to, saved, added, spent, taken: false, rank, .. } = m { Some((to, saved, added, spent, rank)) } else { None }) {
            Some((to, saved, added, spent, rank)) => parts.push(format!("kept: `{}` (move {rank}) would save {} {unit} but add {} to the estimate, which stood at {}: past the budget", to.name(), sci(*saved), sci(*added), sci(*spent))),
            None => parts.push("kept: no narrower type saves anything under the machine".into()),
        }
    }
    parts.join("; ")
}

/// Every value the entry's program stores, with the type a given
/// realization gives it: the record's, or its default where the record
/// names none (plan 0.12.0 step 4; the decoder's document at 0.11.0 left
/// out `RMSNorm.apply/%2`, which D117's record leaves at `f64`). Then the
/// sites the record names that store nothing.
fn stored_values(s: &State, v: &mut Section) {
    let r = &s.realization;
    let mut rows: Vec<(String, Vec<String>)> = Vec::new();
    let mut named: BTreeSet<String> = BTreeSet::new();
    for f in reach(&s.pcore, s.entry) {
        let func = s.pcore.func(f);
        for l in &func.body.lets {
            let site = format!("{}/%{}", func.name, l.var.0);
            if s.roles.role(&site) != Role::Decision || !matches!(l.ty, presk_types::ty::Ty::Real(_)) {
                continue;
            }
            let (d, from) = match r.storage.get(&site) {
                Some(d) => (*d, "the record"),
                None => (r.dtype_at(&site), if site.rsplit_once('/').is_some_and(|(f, _)| r.functions.contains_key(f)) { "its function's" } else { "the default" }),
            };
            named.insert(site.clone());
            rows.push((site.clone(), vec![format!("`{site}`"), op_of(s, &site), format!("`{}`", d.name()), from.into()]));
        }
    }
    rows.sort_by_key(|(site, _)| site_order(site));
    v.prose("**Every value the program stores**, the solver's decisions, with its type and where the type comes from:");
    v.listing_table(&["Site", "Operation", "Type", "From"], rows.into_iter().map(|(_, r)| r).collect());
    let mut others: Vec<(&String, &Dtype)> = r.storage.iter().filter(|(site, _)| !named.contains(*site)).collect();
    if !others.is_empty() {
        others.sort_by_key(|(site, _)| site_order(site));
        let rows: Vec<Vec<String>> = others.iter().map(|(site, d)| vec![format!("`{site}`"), op_of(s, site), role_name(s.roles.0.get(*site)), format!("`{}`", d.name())]).collect();
        v.prose("**The sites the record types that store nothing**, whose type changes no bit:");
        v.table(&["Site", "Operation", "Role", "Type"], rows);
    }
}

/// Every parameter of the entry and the type a given realization stores it
/// in (plan 0.12.0): its own (`params`, rounded once when the entry is
/// given it), integers (`quant`), or the default.
fn params_table(s: &State, v: &mut Section) {
    let r = &s.realization;
    let paths: Vec<String> = match s.pcore.func(s.entry).params.first().map(|p| &p.ty) {
        Some(presk_types::ty::Ty::Params(m)) => s.prog.params_tree(m).declared_ranks().into_iter().map(|(p, _)| p).collect(),
        _ => Vec::new(),
    };
    if paths.is_empty() {
        return;
    }
    let rows: Vec<Vec<String>> = paths
        .iter()
        .map(|p| match (r.params.get(p), r.quant.get(p)) {
            (Some(d), _) => vec![format!("`{p}`"), format!("`{}`", d.name()), "its own type, rounded once when the entry is given it (`params`)".into()],
            (_, Some(q)) => vec![format!("`{p}`"), format!("`{}`", q.name()), "integers with a scale per channel (`quant`)".into()],
            _ => vec![format!("`{p}`"), format!("`{}`", r.default.name()), "the default".into()],
        })
        .collect();
    v.prose("**Every parameter**, decided by its path, and the type it is stored in:");
    v.table(&["Parameter", "Stored as", "From"], rows);
}

fn roles_note(s: &State, v: &mut Section) {
    let (_, _, follows, carried) = roles_count(s);
    if follows + carried > 0 {
        v.note("The realization types sites that change no bit.", format!("{follows} follower{} and {carried} carried value{} are named in the realization. By the reference's chapter 4 (\"Realized value\"), only a stored value is rounded to its own type; these are rounded where another site computes them, or not at all, so their type in the file changes nothing the program computes.", if follows == 1 { "" } else { "s" }, if carried == 1 { "" } else { "s" }));
    }
}

fn fusion(s: &State) -> Section {
    let mut sec = Section::new("7", "Fusion").purpose("PSK-Realize groups the element-wise operations that can share one loop nest: an intermediate inside a group never reaches memory. `iN` is a group's input N, `sN` its step N.");
    let entry = s.xcore.find(&s.core.func(s.entry).name).unwrap_or(s.entry);
    let (mut text, mut groups, mut saved) = (String::new(), 0, 0);
    for f in reach(&s.xcore, entry) {
        let func = s.xcore.func(f);
        let plan = presk_realize::plan(func);
        groups += plan.groups.len();
        saved += plan.saved();
        if !plan.groups.is_empty() {
            text.push_str(&presk_realize::dump(func, &plan));
        }
    }
    if groups == 0 {
        sec.prose(format!("No group in the functions `{}` reaches: every operation is its own nest.", s.o.entry));
    } else {
        sec.prose(format!("In the functions `{}` reaches, {groups} group{}, {saved} `let`{} fused away:", s.o.entry, if groups == 1 { "" } else { "s" }, if saved == 1 { "" } else { "s" }));
        sec.listing("text", "realize.txt", text);
    }
    if let Ok(h) = &s.hosted {
        let l = &h.listing;
        sec.prose(format!("Over one call, fusion takes {} launches to {}.", l.launches + l.fused, l.launches));
    }
    // Why each group ends where it does, and why each element-wise value
    // left alone is: what reads it.
    let mut rows: Vec<Vec<String>> = Vec::new();
    for f in reach(&s.xcore, entry) {
        let func = s.xcore.func(f);
        let plan = presk_realize::plan(func);
        let line = |v: presk_core::VarId| func.body.lets.iter().find(|l| l.var == v).map(|l| format!("`{}`", presk_core::pretty::pretty_let(func, l, &s.xcore, &s.prog))).unwrap_or_default();
        let site = |v: presk_core::VarId| format!("`{}/%{}`", func.name, v.0);
        for (v, stop) in presk_core::fusion::stops(func, &plan) {
            use presk_core::fusion::Stop;
            let what = if plan.groups.contains_key(&v) { format!("ends a group of {}", plan.groups[&v].members.len()) } else { "alone".to_string() };
            let why = match &stop {
                Stop::Result => "the function's result".to_string(),
                Stop::Unread => "nothing reads it".to_string(),
                Stop::Reader(r) => format!("read by {}: {}", line(*r), reader_kind(func, *r)),
                Stop::Shape(r) => format!("read by {}, of another shape: a group is one loop nest over one shape", line(*r)),
                Stop::Shared(rs) => format!("read by {}, not all in one group", rs.iter().map(|r| site(*r)).collect::<Vec<_>>().join(", ")),
                Stop::Size(r) => format!("its reader {}'s expression would pass {} nodes", site(*r), presk_core::fusion::MAX_TREE),
            };
            rows.push(vec![site(v), line(v), what, why]);
        }
    }
    if !rows.is_empty() {
        sec.prose("**Where each group ends**, and why each element-wise value outside a group is alone: what reads it. A value joins its reader's group when every reader is in that group, is element-wise and of the same shape, and the group's expression stays under its size limit.");
        sec.listing_table(&["Value", "Line", "", "Reaches memory because"], rows);
    }
    sec.command(format!("presk dump {} --stage realize{}{}", s.file, consts_args(s), plan_args(s)));
    sec
}

/// What a reader that is not element-wise is.
fn reader_kind(func: &presk_core::Func, r: presk_core::VarId) -> &'static str {
    use presk_core::PrimOp as P;
    match func.body.lets.iter().find(|l| l.var == r).map(|l| &l.rhs) {
        Some(Rhs::Prim { op: P::Sum { .. } | P::Mean { .. } | P::SumAll | P::MeanAll | P::Softmax { .. }, .. }) => "a reduction",
        Some(Rhs::Prim { op: P::Dot | P::Matmul | P::Ein { .. } | P::DotVjpL | P::DotVjpR, .. }) => "a contraction",
        Some(Rhs::Call { .. }) => "a call",
        Some(Rhs::Fold { .. }) => "a fold",
        Some(Rhs::Tuple(_) | Rhs::Record(_)) => "a tuple",
        Some(Rhs::Prim { .. }) => "not element-wise",
        _ => "not an operation of a group",
    }
}

fn machine(s: &State) -> Section {
    let mut sec = Section::new("8", "The machine and the layout plan").purpose("Each nest has a space of layouts (vector lanes, a split over workers, for a contraction a register tile, a k-block, a loop order, packing). The plan takes the point the profile's cost model predicts fastest; \"within 1 + u\" counts the points the model cannot tell from the best.");
    match &s.profile {
        Some((fit, m, _)) => {
            sec.prose("The profile describes the machine as the cost model sees it, and holds the constants fitted on its calibration.");
            sec.table(
                &["Machine", ""],
                vec![
                    vec!["Workers".into(), format!("{} (read from `{}`)", m.workers, m.source)],
                    vec!["L1, private".into(), format!("{} KiB", m.l1 >> 10)],
                    vec!["L2, private".into(), format!("{} KiB", m.l2 >> 10)],
                    vec!["L3".into(), format!("{} KiB, shared by {} workers", m.l3 >> 10, m.l3_sharing)],
                    vec!["Vector registers".into(), format!("{} × {} bytes", m.vector_registers, m.vector_bytes)],
                    vec!["Cache line".into(), format!("{} bytes", m.line)],
                ],
            );
            let combine = match fit.combine {
                presk_opt::profile::Combine::Max => "the largest term".to_string(),
                presk_opt::profile::Combine::Sum => "the in-core term plus the memory term".to_string(),
                presk_opt::profile::Combine::Pnorm(q) => format!("a p-norm, p = {q:.3}"),
            };
            sec.table(
                &["Layout model", ""],
                vec![
                    vec!["Starting a split launch (`spawn`)".into(), format!("{:.1} µs", fit.spawn * 1e6)],
                    vec!["Bandwidth from L2, L3, memory".into(), format!("{:.0}, {:.0}, {:.0} GB/s", fit.beta_l2 / 1e9, fit.beta_l3 / 1e9, fit.beta_mem / 1e9)],
                    vec!["Terms combined by".into(), combine],
                    vec!["Its error on the calibration (u)".into(), format!("{:.3}", fit.u)],
                    vec!["Rank correlation on the calibration".into(), format!("{:.3}", fit.spearman)],
                ],
            );
        }
        None => sec.prose("No profile: the cost model has no constants, so nothing is planned; every nest runs its default layout, the backend's natural vector width and no split."),
    }
    match (&s.captured, &s.planned, &s.profile) {
        (Some(c), Some(p), Some((fit, _, _))) => {
            sec.prose(format!("The plan, as `presk plan` prints it: each launch's operands, the points of its space that lower, the best prediction, and the layout chosen. The entry was captured from its shapes alone, on {} workers.", c.machine.workers));
            sec.listing("text", "plan.txt", text::plan_rows(c, p, false).into_iter().chain(text::plan_totals(c, p, fit.u, false)).collect::<Vec<_>>().join("\n"));
            sec.attach("plan-cost.txt", text::plan_rows(c, p, true).join("\n"), "every point's prediction, as `presk plan --cost` prints them,");
            layout_reasons(s, c, p, &mut sec);
            sec.command(format!("{}presk plan {} {}{} --profile {} --out {DOC}/artifacts/plan.json", s.o.threads.map(|t| format!("PRESK_THREADS={t} ")).unwrap_or_default(), s.file, entry_args(s), if realize_args(s).is_empty() { " --realize f64".to_string() } else { realize_args(s) }, s.o.profile.as_deref().map(shown).unwrap_or_default()));
        }
        _ => {
            if let PlanSrc::Given(path) = &s.plan_src {
                sec.prose(format!("The plan is given, `{path}`: {} nests with a layout, every other at its default.", s.plan.layout.len()));
                let rows: Vec<Vec<String>> = s.plan.layout.iter().map(|(n, l)| vec![format!("`{n}`"), presk_kernel::plan::nest_label(&s.xcore, n), l.to_string()]).collect();
                sec.table(&["Nest", "What", "Layout"], rows);
            }
        }
    }
    sec
}

/// A duration in the unit that reads best: s, ms or µs.
fn duration(t: f64) -> String {
    if t >= 1.0 {
        format!("{t:.3} s")
    } else if t >= 1e-3 {
        format!("{:.3} ms", t * 1e3)
    } else {
        format!("{:.3} µs", t * 1e6)
    }
}

/// A site's order: its function, then its variable's number, so that
/// `f/%3` comes before `f/%10`.
fn site_order(site: &str) -> (String, u32) {
    match site.rsplit_once("/%") {
        Some((f, n)) => (f.to_string(), n.parse().unwrap_or(u32::MAX)),
        None => (site.to_string(), u32::MAX),
    }
}

/// A time in µs, to four significant digits or so, in at most nine
/// characters.
fn us(t: f64) -> String {
    let u = t * 1e6;
    if u >= 1e6 {
        format!("{u:.3e}")
    } else if u >= 100.0 {
        format!("{u:.1}")
    } else if u >= 1.0 {
        format!("{u:.3}")
    } else {
        format!("{u:.4}")
    }
}

/// Why each nest takes its layout: the chosen point, its two runners-up
/// and the best point that splits, each prediction in its terms.
fn layout_reasons(s: &State, c: &Captured, p: &Planned, sec: &mut Section) {
    let Some((fit, m, _)) = &s.profile else { return };
    let head = format!("{:<5}{:>10} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}  {}", "", "predicted", "λ", "spawn", "work", "loads", "L2", "L3", "mem", "bound by");
    let mut blocks: Vec<(f64, String)> = Vec::new();
    for cost in &p.costed {
        let l = &c.launches[cost.launch];
        let mut picks: Vec<(String, &presk_rt::Layout)> = cost.scored.iter().take(3).enumerate().map(|(i, (_, lay))| (if i == 0 { "1".to_string() } else { (i + 1).to_string() }, lay)).collect();
        let split = |lay: &presk_rt::Layout| matches!(lay.split, Some(presk_rt::layout::Split::At { .. }));
        if !split(&cost.scored[0].1)
            && let Some((i, (_, lay))) = cost.scored.iter().enumerate().find(|(_, (_, lay))| split(lay))
            && i >= 3
        {
            picks.push((format!("{}", i + 1), lay));
        }
        let mut text = format!("{} {} — {} points, {} within 1 + u, {} launch{}\n{head}\n", l.nest, presk_driver::layout::shape(l), cost.space, cost.kept, l.count, if l.count == 1 { "" } else { "es" });
        for (rank, lay) in picks {
            let Some(ks) = presk_driver::layout::terms_at(fit, m, l, lay, c.machine.workers) else { continue };
            let terms: Vec<presk_opt::profile::Terms> = ks.iter().filter_map(|k| k.terms).collect();
            if terms.len() != ks.len() {
                continue;
            }
            let sum = |f: &dyn Fn(&presk_opt::profile::Terms) -> f64| terms.iter().map(f).sum::<f64>();
            let total = sum(&|t| t.total());
            let spawn = sum(&|t| t.spawn);
            let loads = if terms.iter().all(|t| t.loads.is_some()) { us(sum(&|t| t.loads.unwrap_or(0.0))) } else { "-".into() };
            let bound = if spawn > total - spawn - sum(&|t| t.lambda) { "spawn" } else { terms.iter().max_by(|a, b| a.total().partial_cmp(&b.total()).unwrap_or(std::cmp::Ordering::Equal)).map(|t| t.bound()).unwrap_or("-") };
            text.push_str(&format!("{rank:<5}{:>10} {:>9} {:>9} {:>9} {loads:>9} {:>9} {:>9} {:>9}  {bound}\n     {lay}\n", us(total), us(sum(&|t| t.lambda)), us(spawn), us(sum(&|t| t.work)), us(sum(&|t| t.l2)), us(sum(&|t| t.l3)), us(sum(&|t| t.mem))));
        }
        blocks.push((cost.scored[0].0 * l.count as f64, text));
    }
    if blocks.is_empty() {
        return;
    }
    sec.prose(format!(
        "**Why each nest takes its layout.** The chosen point (1), its two runners-up and the best point that splits the nest over workers, each prediction in its terms, in µs, its layout under it: `λ` a launch, `spawn` starting a split launch, then work, register loads, and the traffic from L2, L3 and memory, which the model combines by {}. `bound by` names the largest term. A split pays `spawn`, {} µs, so it wins only when it divides more than that.",
        match fit.combine {
            presk_opt::profile::Combine::Max => "their maximum".to_string(),
            presk_opt::profile::Combine::Sum => "the in-core term plus the memory term".to_string(),
            presk_opt::profile::Combine::Pnorm(q) => format!("a p-norm, p = {q:.3}"),
        },
        us(fit.spawn)
    ));
    let all = blocks.len();
    if all > 12 {
        let mut by_time = blocks.clone();
        by_time.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        sec.prose(format!("The twelve nests of the most predicted time a call, of {all}; every nest in `listings/plan-reasons.txt`."));
        sec.code("text", by_time.iter().take(12).map(|(_, t)| t.as_str()).collect::<Vec<_>>().join("\n"));
        sec.blocks.push(Block::Attach { name: "plan-reasons.txt".into(), text: blocks.iter().map(|(_, t)| t.as_str()).collect::<Vec<_>>().join("\n") });
    } else {
        sec.code("text", blocks.iter().map(|(_, t)| t.as_str()).collect::<Vec<_>>().join("\n"));
    }
}

fn kernels(s: &State) -> Section {
    let mut sec = Section::new("9", "The kernels and the host").purpose("PSK-Kernel is the loop nests the backends run: one kernel per distinct nest, under its type and its layout. The host program launches them, converting a value where a kernel reads it in another type.");
    match &s.hosted {
        Ok(h) => {
            let l = &h.listing;
            sec.code("text", text::kernel_heading(l).join("\n"));
            let mut sites: BTreeMap<usize, (Vec<String>, usize)> = BTreeMap::new();
            if let Ok(rec) = &h.recording {
                for op in &rec.ops {
                    if let presk_kernel::record::Op::Launch { kernel, site, .. } = op {
                        let e = sites.entry(*kernel).or_default();
                        if !e.0.contains(site) {
                            e.0.push(site.clone());
                        }
                        e.1 += 1;
                    }
                }
            }
            let mut text = String::new();
            for (i, k) in l.kernels.iter().enumerate() {
                match sites.get(&i) {
                    Some((ss, n)) => text.push_str(&format!("// k{i}: {} launch{} at {}\n", n, if *n == 1 { "" } else { "es" }, ss.join(", "))),
                    None => text.push_str(&format!("// k{i}\n")),
                }
                text.push_str(&k.to_string());
            }
            sec.prose("Each kernel under the sites that launch it, as the host recorded them:");
            sec.listing("text", "kernels.txt", text);
            sec.command(format!("presk dump {} --stage kernel {}{}{}", s.file, entry_args(s), plan_args(s), remat_args(s)));
            match (&h.recording, &s.host_c) {
                (Ok(_), Some(c)) => {
                    sec.prose("**The host**, as the C export prints it (§10): a function per Presk function, a fold a loop, each launch under the site it computes and each conversion where a kernel reads a value in another type.");
                    sec.listing("c", "host.c", c.clone());
                }
                (Err(m), _) => sec.prose(format!("The host was not recorded: {m}")),
                _ => {}
            }
        }
        Err(m) => sec.prose(format!("No kernels: {m}")),
    }
    sec
}

fn targets(s: &State) -> Section {
    let mut sec = Section::new("10", "Targets");
    if s.o.targets.contains(&Target::Native) {
        let n = s.hosted.as_ref().map(|h| h.listing.kernels.len()).unwrap_or(0);
        sec.prose(format!("**Native.** The Cranelift backend compiles each of the {n} kernels to machine code at its first launch, in process, and the host program of §9 runs them."));
    }
    match &s.c {
        Some(Ok(c)) => {
            sec.prose("**C.** `presk export --target c` writes the entry as a C program that computes the native backend's bits (statement C1):");
            sec.code("text", c.line.clone());
            sec.table(&["File", "Lines"], c.files.iter().map(|(n, l)| vec![format!("`{n}`"), l.to_string()]).collect());
            if let Some((i, k)) = c.kernels.iter().enumerate().find(|(_, k)| k.name.starts_with("dot") || k.name.starts_with("matmul") || k.name.starts_with("ein")) {
                if let Ok(text) = presk_export_c::kernel_fn(k, &format!("k{i}")) {
                    sec.prose(format!("The first contraction, `k{i}`, as C:"));
                    sec.code("c", text);
                }
            }
            sec.prose(format!("`make -C {DOC}/artifacts/c check` builds it and compares its outputs with the native backend's, bit for bit."));
            sec.command(format!("presk export {} --target c {}{}{}{}{} -o {DOC}/artifacts/c", s.file, entry_args(s), plan_args(s), weights_args(s), s.o.threads.map(|t| format!(" --threads {t}")).unwrap_or_default(), if s.o.no_exact { " --no-exact" } else { "" }));
        }
        Some(Err(m)) => sec.prose(format!("**C.** Not exported: {m}")),
        None => {}
    }
    match &s.pytorch {
        Some(Ok(files)) => {
            sec.prose("**PyTorch.** `presk export --target pytorch` writes one module per file:");
            sec.table(&["File", "Lines"], files.iter().map(|(n, t)| vec![format!("`{n}`"), t.lines().count().to_string()]).collect());
            sec.command(format!("presk export {} --target pytorch{} -o {DOC}/artifacts/pytorch", s.file, consts_args(s)));
        }
        Some(Err(m)) => sec.prose(format!("**PyTorch.** Not exported: {m}")),
        None => {}
    }
    if s.o.targets.is_empty() {
        sec.prose("No target asked.");
    }
    sec
}

fn appendix_placeholder() -> Section {
    Section::new("", "Appendix: the commands")
}

/// The appendix: every section's commands, in order.
fn with_appendix(mut doc: Document) -> Document {
    fn collect(s: &Section, out: &mut Vec<String>) {
        out.extend(s.commands.iter().cloned());
        for sub in &s.sections {
            collect(sub, out);
        }
    }
    let mut all = Vec::new();
    for s in &doc.sections {
        collect(s, &mut all);
    }
    let mut seen = BTreeSet::new();
    all.retain(|c| seen.insert(c.clone()));
    if let Some(last) = doc.sections.last_mut() {
        last.purpose.push("The commands behind the sections, in order, run from the directory the program was named from, with `DOC` set to this document's directory.".into());
        last.code("sh", all.join("\n"));
    }
    doc
}

impl Section {
    /// A listing the body names without showing it.
    fn attach(&mut self, name: &str, text: String, what: &str) {
        self.prose(format!("*{} in `listings/{name}` ({} lines).*", capitalize(what), text.lines().count()));
        self.blocks.push(Block::Attach { name: name.into(), text });
    }

    /// A table that may be long: past [`crate::doc::LONG`] rows its first
    /// rows in the body, the whole as a listing.
    fn listing_table(&mut self, head: &[&str], rows: Vec<Vec<String>>) {
        if rows.len() <= crate::doc::LONG {
            self.table(head, rows);
            return;
        }
        let whole: String = std::iter::once(head.join("\t")).chain(rows.iter().map(|r| r.join("\t"))).collect::<Vec<_>>().join("\n");
        let n = rows.len();
        self.table(head, rows.into_iter().take(crate::doc::EXCERPT).collect());
        self.prose(format!("*The first {} of {n} rows; the whole in `listings/breakdown.tsv`.*", crate::doc::EXCERPT));
        self.blocks.push(Block::Attach { name: "breakdown.tsv".into(), text: whole });
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().chain(c).collect(),
        None => String::new(),
    }
}

/// The minute's load average, from `/proc/loadavg`.
fn load_average() -> Option<f64> {
    std::fs::read_to_string("/proc/loadavg").ok()?.split_whitespace().next()?.parse().ok()
}

/// `--measure`: the realization natively beside the exact reference, a call
/// timed, each kernel timed, the host by phase (plan 0.11.0 step 5, D115).
#[allow(clippy::too_many_arguments)]
fn measure(o: &Options, core: &Program, prog: &ElabProgram, dims: &[(String, u64)], realize: &Realization, plan: &Plan, planned: bool, report: &mut dyn Report) -> Result<Measured, String> {
    use presk_driver::backend::{BackendKind, with_backend};
    let (warmup, repeat) = (3, 21);
    let before = load_average();
    let opts = presk_interp::RunOptions { entry: o.entry.clone(), dims: dims.to_vec(), seed: o.seed, bits: o.bits, digits: 17 };
    let (func, fdims, args) = presk_driver::entry::entry_args(core, prog, &opts, None, o.weights.as_deref())?;
    let exact = if o.no_exact {
        None
    } else {
        let mut ib = presk_rt::InterpBackend::new(core, prog, o.bits);
        let one = Realization::uniform(Dtype::F64);
        Some(presk_rt::Backend::prepare(&mut ib, func, &fdims, &one).and_then(|p| presk_rt::Backend::call(&mut ib, &p, args.clone())).map_err(|e| e.message)?)
    };
    let workers = o.threads.unwrap_or_else(|| presk_kernel::cost::Machine::detect().workers).max(1);
    let native = with_backend(BackendKind::Native, core, prog, o.bits, plan, report, |b| -> Result<presk_rt::Value, String> {
        let a = realize.realize_args(&args).map_err(|e| e.message)?;
        b.prepare(func, &fdims, realize).and_then(|p| b.call(&p, a)).map_err(|e| e.message)
    })?;
    let gap = exact.as_ref().map(|e| presk_driver::gap::leaf_gap(e, &native));
    let call = with_backend(BackendKind::Native, core, prog, o.bits, plan, report, |b| presk_driver::bench::time_call(b, func, &fdims, realize, &args, warmup, repeat).ok());
    let kernels = presk_driver::bench::kernel_times(core, prog, func, &fdims, &args, plan, repeat);
    let host = presk_driver::bench::host_profile(core, prog, func, &fdims, &args, plan, planned, repeat, warmup);
    Ok(Measured { load: (before, load_average()), exact, native, gap, call, warmup, repeat, workers, kernels, host })
}

/// A scalar output's value, to 17 digits; `None` for any other.
fn scalar(v: &presk_rt::Value) -> Option<String> {
    match v {
        presk_rt::Value::Real(t) if t.data.len() == 1 => t.data.to_f64().first().map(|x| format!("{x:.17}")),
        _ => None,
    }
}

/// The sites each kernel of the shapes' listing is launched at, by its text:
/// the bare site, without the `(fused)` or the source name the host's
/// comment adds. §11 names the kernels `presk bench` timed by it: they are
/// another run's, whose places in its own listing may differ, and their text
/// is what the two share; every other section reads the provenance table.
fn kernel_sites(s: &State) -> BTreeMap<String, (usize, Vec<String>)> {
    let mut out: BTreeMap<String, (usize, Vec<String>)> = BTreeMap::new();
    if let Ok(h) = &s.hosted
        && let Ok(rec) = &h.recording
    {
        for op in &rec.ops {
            if let presk_kernel::record::Op::Launch { kernel, site, .. } = op
                && let Some(k) = h.listing.kernels.get(*kernel)
            {
                let site = site.split(' ').next().unwrap_or(site).to_string();
                let e = out.entry(k.to_string()).or_insert((*kernel, Vec::new()));
                if !e.1.contains(&site) {
                    e.1.push(site);
                }
            }
        }
    }
    out
}

fn measured(s: &State) -> Section {
    let mut sec = Section::new("11", "Measured").purpose("What one run measured on this machine: the realization natively against the exact reference, a call timed, each kernel's time beside the model's prediction, and the host's time by phase. The one section that changes from run to run.");
    let m = match &s.measured {
        None => {
            sec.prose("Not measured: `--measure` was not given. Every other section is deterministic; this one would hold times.");
            return sec;
        }
        Some(Err(e)) => {
            sec.prose(format!("The measurement failed: {e}"));
            return sec;
        }
        Some(Ok(m)) => m,
    };
    let load = |x: Option<f64>| x.map(|v| format!("{v:.2}")).unwrap_or_else(|| "unknown".into());
    sec.note("Not a record.", format!("`presk explain` does not reserve the machine: the load average over a minute was {} before the runs and {} after. A time of record runs alone on a reserved machine.", load(m.load.0), load(m.load.1)));
    let mut rows: Vec<Vec<String>> = Vec::new();
    if let Some(e) = &m.exact {
        rows.push(vec![format!("Exact reference ({} bits)", s.o.bits), scalar(e).unwrap_or_else(|| format!("magnitude {}", sci(presk_driver::gap::leaf_max(e))))]);
    }
    rows.push(vec![format!("Native, {}{}", s.realization.label(), if matches!(s.plan_src, PlanSrc::None) { ", default layouts" } else { ", under the plan" }), scalar(&m.native).unwrap_or_else(|| format!("magnitude {}", sci(presk_driver::gap::leaf_max(&m.native))))]);
    if let Some(g) = m.gap {
        let note = match s.solved.as_ref().and_then(|sv| sv.solution.measured) {
            Some(sg) if sg == g => ", the solver's measured gap: the kernel interpreter it measured on computes the native backend's bits".to_string(),
            Some(sg) => format!("; the solver measured {} on the kernel interpreter", sci(sg)),
            None => String::new(),
        };
        rows.push(vec!["Gap, native against the exact reference".into(), format!("{}{note}", sci(g))]);
    }
    if let Some((med, best)) = m.call {
        rows.push(vec![format!("A call, native, {} thread{}", m.workers, if m.workers == 1 { "" } else { "s" }), format!("{med:.3} ms median, {best:.3} ms best ({} calls after {} warm-up)", m.repeat, m.warmup)]);
    }
    sec.table(&["", "Value"], rows);
    match &m.kernels {
        Ok(k) => {
            let sites = kernel_sites(s);
            let fit = s.profile.as_ref().map(|(f, mm, _)| (f, mm));
            let mut rows: Vec<Vec<String>> = Vec::new();
            for (i, kern) in k.kernels.iter().enumerate() {
                let n = k.launches[i] / k.repeat.max(1);
                let measured = k.medians[i];
                let predicted = fit.and_then(|(f, mm)| presk_driver::layout::predict(f, mm, kern.dtype, std::slice::from_ref(kern)));
                let at = sites.get(&kern.to_string()).map(|(j, ss)| format!("`k{j}` {}", ss.iter().map(|x| format!("`{x}`")).collect::<Vec<_>>().join(", "))).unwrap_or_else(|| format!("`{}`", kern.name));
                let ratio = match (measured, predicted) {
                    (Some(a), Some(b)) if b > 0.0 => format!("{:.2}", a / b),
                    _ => "-".into(),
                };
                rows.push(vec![at, n.to_string(), predicted.map(us).unwrap_or_else(|| "-".into()), measured.map(us).unwrap_or_else(|| "-".into()), ratio]);
            }
            sec.prose(format!("**Each kernel**, its median launch natively over {} calls, beside the profile's prediction (µs a launch). The kernels take {:.3} ms of a call, {} launches. A launch is timed around the executor's call, its dispatch included, which the model prices apart (`λ`); on nests of a few hundredths of a µs the dispatch is most of what is measured.", k.repeat, k.per_call_seconds() * 1e3, k.per_call_launches()));
            sec.listing_table(&["Kernel and sites", "Launches a call", "Predicted", "Measured", "Measured / predicted"], rows);
        }
        Err(e) => sec.prose(format!("The kernels were not timed: {e}")),
    }
    match &m.host {
        Ok(h) => {
            sec.prose("**Where a call's time goes**, the host by phase:");
            sec.code("text", presk_driver::text::host_profile(&s.o.entry, &s.realization.label(), !matches!(s.plan_src, PlanSrc::None), h));
        }
        Err(e) => sec.prose(format!("The host was not profiled: {e}")),
    }
    let threads = s.o.threads.map(|t| format!(" --threads {t}")).unwrap_or_default();
    sec.command(format!("presk run {} {}{} --backend interp --bits {} --digits 17", s.file, entry_args(s), weights_args(s), s.o.bits));
    sec.command(format!("presk run {} {}{}{}{threads} --backend native --digits 17", s.file, entry_args(s), weights_args(s), plan_args(s)));
    sec.command(format!("presk bench {} {}{}{}{threads} --native-only --plan-only --warmup {} --repeat {}", s.file, entry_args(s), weights_args(s), plan_args(s), m.warmup, m.repeat));
    sec.command(format!("presk bench {} {}{}{}{threads} --native-only --plan-only --warmup {} --repeat {} --host-profile", s.file, entry_args(s), weights_args(s), plan_args(s), m.warmup, m.repeat));
    sec
}

/// The default sites of §12: the three largest shares of the estimate (the
/// three largest bounds without a solve), then the three nests of the most
/// predicted time a call.
fn default_sites(s: &State) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    if let Some(sv) = &s.solved {
        let mut d: Vec<&presk_precision::solve::SiteCost> = sv.breakdown.iter().filter(|c| c.role == Role::Decision).collect();
        d.sort_by(|a, b| b.err.partial_cmp(&a.err).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.name.cmp(&b.name)));
        out.extend(d.iter().take(3).map(|c| c.name.clone()));
    } else if let Some(b) = &s.bounded
        && let Some(r) = b.runs.iter().find(|r| r.model == presk_precision::propagate::Model::Conditioned)
    {
        let mut v: Vec<(&String, f64)> = r.sites.iter().map(|(k, b)| (k, b.err)).collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(b.0)));
        out.extend(v.iter().take(3).map(|(k, _)| (*k).clone()));
    }
    if let (Some(c), Some(p)) = (&s.captured, &s.planned) {
        let mut v: Vec<(String, f64)> = Vec::new();
        for cost in &p.costed {
            let l = &c.launches[cost.launch];
            let t = cost.scored[0].0 * l.count as f64;
            match v.iter_mut().find(|(n, _)| *n == l.nest) {
                Some(e) => e.1 += t,
                None => v.push((l.nest.clone(), t)),
            }
        }
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
        for (n, _) in v.into_iter().take(3) {
            if !out.contains(&n) {
                out.push(n);
            }
        }
    }
    out
}

/// One site through every stage.
fn trace(s: &State, site: &str) -> Option<Vec<Vec<String>>> {
    let (program, (func, l)) = [&s.core, &s.xcore, &s.pcore].into_iter().find_map(|p| site_let(p, site).map(|x| (p, x)))?;
    let mut rows: Vec<Vec<String>> = vec![vec!["PSK-Core (§3)".into(), format!("`{}` in `{}`", presk_core::pretty::pretty_let(func, l, program, &s.prog), func.name)]];
    if let Some((env, _)) = &s.env {
        rows.push(vec![
            "Envelope (§6.1)".into(),
            match env.get(site) {
                Some(m) => format!("|v| ≤ {}, {} element{}{}", sci(m), env.elems(site), if env.elems(site) == 1 { "" } else { "s" }, env.condition(site).map(|c| format!(", condition number {}", sci(c))).unwrap_or_default()),
                None => "not recorded: no real value".into(),
            },
        ]);
    }
    if let Some(sens) = &s.sens {
        rows.push(vec!["Sensitivity (§6.2)".into(), sens.get(site).map(sci).unwrap_or_else(|| "none".into())]);
    }
    let ty = s.realization.storage.get(site).copied().unwrap_or(s.realization.default);
    let mut typed = format!("`{}`, {}", ty.name(), role_name(s.roles.0.get(site)));
    if let Some(sv) = &s.solved
        && let Some(c) = sv.breakdown.iter().find(|c| c.name == site)
    {
        typed = format!("`{}`, {}; term {}", c.dtype.name(), role_name(Some(&c.role)), sci(c.err));
        if c.role == Role::Decision {
            typed.push_str(&format!("; {}", why(sv, site, if sv.time { "s" } else { "bytes" })));
        }
    }
    rows.push(vec!["Type (§6.3)".into(), typed]);
    // The site in the program the kernels run, and its nest, its kernel and
    // its C function: the provenance table's links from it (plan 0.13.0
    // step 1, D134). Under `--opt` the site the earlier stages name is the
    // program's before the optimizer, which renumbers every `let`.
    let t = &s.table;
    let kstage = if s.o.opt { Stage::Opt } else { Stage::Ad };
    let start = [Stage::Ad, Stage::Opt].into_iter().map(|st| Id::new(st, site)).find(|id| t.has(id));
    let at = start.as_ref().and_then(|id| t.descendants(id, kstage).into_iter().next());
    let mut nest = site.to_string();
    if s.o.opt
        && let Some(at) = &at
        && let Some((xf, xl)) = site_let(&s.xcore, &at.key)
    {
        let how = match t.made(at).first().map(|l| &l.how) {
            Some(presk_provenance::How::Unchanged) if at.key == site => "left as it was".to_string(),
            Some(presk_provenance::How::Unchanged) => format!("`{site}` under a new number, no rule changed it"),
            Some(presk_provenance::How::Rewritten { .. }) => format!("the cheapest term of the class of `{site}`{}", match t.made(at).first().map(|l| l.parents.len()) {
                Some(n) if n > 1 => format!(", and of {} other `let`s", n - 1),
                _ => String::new(),
            }),
            Some(presk_provenance::How::Introduced) => "a `let` a rewrite introduced".to_string(),
            _ => "not traced".to_string(),
        };
        rows.push(vec!["Optimizer (§5)".into(), format!("`{}` in `{}`: {how}", presk_core::pretty::pretty_let(xf, xl, &s.xcore, &s.prog), xf.name)]);
    }
    if let Some(at) = &at
        && let Some((xf, xl)) = site_let(&s.xcore, &at.key)
    {
        let plan = presk_realize::plan(xf);
        let nests = t.descendants(at, Stage::Fusion);
        if let Some(n) = nests.first() {
            nest = n.key.clone();
        }
        let fusion = if plan.interior.contains(&xl.var) {
            format!("a member of the group that ends at `{nest}`: computed there, never stored")
        } else {
            match presk_core::fusion::stops(xf, &plan).into_iter().find(|(v, _)| *v == xl.var) {
                Some((v, stop)) => format!("{}: {}", if plan.groups.contains_key(&v) { format!("ends a group of {}", plan.groups[&v].members.len()) } else { "alone".to_string() }, stop_text(xf, &stop, s)),
                None => "not an element-wise operation".into(),
            }
        };
        rows.push(vec!["Fusion (§7)".into(), fusion]);
    }
    // The nest's layout.
    if let (Some(c), Some(p)) = (&s.captured, &s.planned) {
        let launches: Vec<&presk_driver::layout::Costed> = p.costed.iter().filter(|k| c.launches[k.launch].nest == nest).collect();
        if let Some(first) = launches.first() {
            let l = &c.launches[first.launch];
            let chosen = p.layout.get(&nest).map(|x| x.to_string()).unwrap_or_else(|| first.scored[0].1.to_string());
            let runner = first.scored.get(1).map(|(t, lay)| format!("; runner-up {lay}, {} µs", us(*t))).unwrap_or_default();
            let bound = s.profile.as_ref().and_then(|(fit, m, _)| presk_driver::layout::terms_at(fit, m, l, &first.scored[0].1, c.machine.workers)).and_then(|ks| ks.iter().filter_map(|k| k.terms).max_by(|a, b| a.total().partial_cmp(&b.total()).unwrap_or(std::cmp::Ordering::Equal)).map(|t| t.bound())).map(|b| format!(", bound by {b}")).unwrap_or_default();
            rows.push(vec!["Layout (§8)".into(), format!("nest `{nest}`, {} point{} at `{}`; chosen {chosen}, {} µs{bound}{runner}", first.space, if first.space == 1 { "" } else { "s" }, presk_driver::layout::shape(l), us(first.scored[0].0))]);
        } else {
            rows.push(vec!["Layout (§8)".into(), "not a nest: no layout space".into()]);
        }
    }
    // The kernel, and its C. A nest launches one kernel, or two when a pack
    // copies an operand first: the first by its text, as 0.12.0 named it.
    let kernels: Vec<(usize, String)> = [Stage::Layout, Stage::Fusion]
        .into_iter()
        .map(|st| Id::new(st, nest.clone()))
        .find(|id| t.has(id))
        .map(|id| t.descendants(&id, Stage::Kernel))
        .unwrap_or_default()
        .iter()
        .filter_map(|k| k.key.strip_prefix('k').and_then(|j| j.parse::<usize>().ok()))
        .filter_map(|j| s.hosted.as_ref().ok().and_then(|h| h.listing.kernels.get(j)).map(|k| (j, k.to_string())))
        .collect();
    let kern = kernels.iter().min_by(|a, b| a.1.cmp(&b.1)).map(|(j, text)| (*j, text.lines().next().unwrap_or("").trim_end_matches(" {").to_string()));
    // A nest a contraction's kernel computes as its epilogue is launched by
    // the contraction's nest.
    let by = kern.as_ref().and_then(|(j, _)| {
        let suffix = format!(" k{j}");
        t.elements(Stage::Launch).iter().filter_map(|l| l.key.strip_suffix(&suffix)).find(|n| *n != nest).filter(|_| !t.elements(Stage::Launch).iter().any(|l| l.key == format!("{nest}{suffix}"))).map(String::from)
    });
    match (&kern, &by) {
        (Some((j, head)), Some(product)) => rows.push(vec!["Kernel (§9)".into(), format!("`k{j}`, as the epilogue of `{product}`'s kernel: `{head}`")]),
        (Some((j, head)), None) => rows.push(vec!["Kernel (§9)".into(), format!("`k{j}`: `{head}`")]),
        (None, _) => rows.push(vec![
            "Kernel (§9)".into(),
            match s.roles.0.get(site) {
                Some(Role::Follows(of)) => format!("none of its own: read, and converted, by the kernels of {}", of.iter().map(|x| format!("`{x}`")).collect::<Vec<_>>().join(", ")),
                Some(Role::Carried) => "none: the host carries it".into(),
                _ => "none: the host computes it".into(),
            },
        ]),
    }
    if matches!(s.c, Some(Ok(_)))
        && let Some((j, _)) = &kern
        && let Some(cf) = t.descendants(&Id::new(Stage::Kernel, format!("k{j}")), Stage::C).first()
    {
        let n = cf.key.strip_prefix("kernels.c ").unwrap_or(&cf.key);
        rows.push(vec!["C (§10)".into(), format!("`{n}` in `kernels.c`, launched by `{}` in `model.c`, the launch commented `{nest}`", presk_export_c::host::c_ident(&format!("presk_{}", func.name)))]);
    }
    if let Some(Ok(m)) = &s.measured
        && let (Ok(k), Some((_, head))) = (&m.kernels, &kern)
        && let Some(i) = k.kernels.iter().position(|x| x.to_string().lines().next().unwrap_or("").trim_end_matches(" {") == head)
        && let Some(t) = k.medians[i]
    {
        rows.push(vec!["Time (§11)".into(), format!("{} µs a launch, median", us(t))]);
    }
    Some(rows)
}

fn stop_text(func: &presk_core::Func, stop: &presk_core::fusion::Stop, s: &State) -> String {
    use presk_core::fusion::Stop;
    let line = |v: presk_core::VarId| func.body.lets.iter().find(|l| l.var == v).map(|l| format!("`{}`", presk_core::pretty::pretty_let(func, l, &s.xcore, &s.prog))).unwrap_or_default();
    let site = |v: presk_core::VarId| format!("`{}/%{}`", func.name, v.0);
    match stop {
        Stop::Result => "the function's result".to_string(),
        Stop::Unread => "nothing reads it".to_string(),
        Stop::Reader(r) => format!("read by {}: {}", line(*r), reader_kind(func, *r)),
        Stop::Shape(r) => format!("read by {}, of another shape: a group is one loop nest over one shape", line(*r)),
        Stop::Shared(rs) => format!("read by {}, not all in one group", rs.iter().map(|r| site(*r)).collect::<Vec<_>>().join(", ")),
        Stop::Size(r) => format!("its reader {}'s expression would pass {} nodes", site(*r), presk_core::fusion::MAX_TREE),
    }
}

fn traces(s: &State) -> Section {
    let mut sec = Section::new("12", "Sites through every stage").purpose("A site is named `<function>/%<n>` at every stage, which is what lets one value be followed from PSK-Core to the C.");
    let (sites, chosen) = if s.o.sites.is_empty() { (default_sites(s), false) } else { (s.o.sites.clone(), true) };
    if chosen {
        sec.prose("The sites `--site` names:");
    } else {
        sec.prose("By default, the three largest shares of the solver's estimate (the three largest bounds without a solve), then the three nests of the most predicted time a call; `--site` names others.");
    }
    for site in &sites {
        match trace(s, site) {
            Some(rows) => {
                sec.prose(format!("**`{site}`**"));
                sec.table(&["Stage", ""], rows);
            }
            None => sec.prose(format!("**`{site}`**: not a site of the program.")),
        }
    }
    // The index: every site of the functions the entry reaches, and the
    // sections it appears in.
    let entry = s.xcore.find(&s.core.func(s.entry).name).unwrap_or(s.entry);
    // The nests a kernel launches, by the table's launches.
    let launched: BTreeSet<String> = s.table.elements(Stage::Launch).iter().filter_map(|l| l.key.rsplit_once(' ').map(|(n, _)| n.to_string())).collect();
    let nests: BTreeSet<&String> = s.captured.as_ref().map(|c| c.launches.iter().map(|l| &l.nest).collect()).unwrap_or_default();
    let breakdown: BTreeSet<&str> = s.solved.as_ref().map(|sv| sv.breakdown.iter().map(|c| c.name.as_str()).collect()).unwrap_or_default();
    let mut index = String::from("site\tline\tsections\n");
    for f in reach(&s.xcore, entry) {
        let func = s.xcore.func(f);
        for l in &func.body.lets {
            let name = format!("{}/%{}", func.name, l.var.0);
            let mut at = vec!["3"];
            if s.env.as_ref().is_some_and(|(e, _)| e.get(&name).is_some()) {
                at.push("6.1");
            }
            if s.sens.as_ref().is_some_and(|x| x.get(&name).is_some()) {
                at.push("6.2");
            }
            if breakdown.contains(name.as_str()) || s.realization.storage.contains_key(&name) {
                at.push("6.3");
            }
            if nests.contains(&name) {
                at.push("8");
            }
            if launched.contains(&name) {
                at.push("9");
            }
            index.push_str(&format!("{name}\t{}\t§{}\n", presk_core::pretty::pretty_let(func, l, &s.xcore, &s.prog), at.join(", §")));
        }
    }
    sec.attach("site-index.tsv", index, "every site of the functions the entry reaches, its line and the sections that name it,");
    sec
}
