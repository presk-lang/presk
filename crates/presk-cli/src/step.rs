//! `presk calibrate --step` and `presk solve --step`: a precision budget on
//! one training step's update (plan 0.6.0 §3.7, step 7b, D52).
//!
//! The state is a checkpoint followed by a warm-up in `f64`
//! (`presk_train::step`); the exact reference is the step on the reference
//! interpreter from that state rounded to the realization's default; the
//! candidates run natively. Everything measured is a distance between two
//! steps, relative to the step itself.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use presk_rt::{Dtype, Realization};
use presk_train::step::{StepOptions, StepResult, StepSetup};

use super::{color, has_errors, render_all, typed_front};

/// What `--step` needs from the command line.
pub struct StepArgs<'a> {
    pub file: &'a Path,
    pub consts: &'a [String],
    pub weights: &'a Path,
    pub warmup: u64,
    pub data: Option<&'a Path>,
    pub from: usize,
    pub samples: usize,
    pub steps: Option<u64>,
    pub seed: u64,
    pub bits: u32,
}

struct Ctx {
    prog: presk_types::elab::ElabProgram,
    core: presk_core::Program,
    setup: StepSetup,
}

fn fail(cmd: &str, m: impl std::fmt::Display) -> ExitCode {
    eprintln!("presk {cmd}: {m}");
    ExitCode::from(1)
}

/// Front end, the training functions, and the warmed-up state; the fusion
/// groups kept when `groups` (a realization the command runs can see them,
/// D94).
fn context(cmd: &str, a: &StepArgs, groups: bool) -> Result<Ctx, ExitCode> {
    let (sm, prog, typed) = typed_front(a.file, a.consts)?;
    if typed.training.is_empty() {
        eprintln!("presk {cmd}: --step needs a `training run`, and {} has none", a.file.display());
        return Err(ExitCode::from(2));
    }
    let (mut core, diags) = presk_core::lower(&prog, &typed);
    eprint!("{}", render_all(&diags, &sm, color()));
    if has_errors(&diags) {
        return Err(ExitCode::from(1));
    }
    let diags = crate::differentiate(&mut core, groups);
    eprint!("{}", render_all(&diags, &sm, color()));
    if has_errors(&diags) {
        return Err(ExitCode::from(1));
    }
    let fns = presk_train::prepare_training_as(&mut core, &prog, &typed, crate::ad_options(groups)).map_err(|e| fail(cmd, e.message))?;
    let opts = StepOptions { weights: a.weights.to_path_buf(), warmup: a.warmup, data: a.data.map(Path::to_path_buf), from: a.from, samples: a.samples, steps: a.steps, seed: a.seed };
    let setup = {
        let one = Realization::uniform(Dtype::F64);
        let mut warm = native(&core, &prog, &one).map_err(|m| fail(cmd, m))?;
        presk_train::step::setup(&mut warm, &core, &prog, &typed, &fns, &opts).map_err(|e| fail(cmd, e.message))?
    };
    Ok(Ctx { prog, core, setup })
}

fn native<'p>(core: &'p presk_core::Program, prog: &'p presk_types::elab::ElabProgram, r: &Realization) -> Result<presk_kernel::KernelBackend<'p>, String> {
    presk_cranelift::native_backend(core, prog, r).map_err(|e| format!("the native backend is unavailable: {}", e.message))
}

impl Ctx {
    /// The exact step from the state rounded to `input`, optionally
    /// recording the envelope's measurements on the way.
    fn exact(&self, input: Dtype, bits: u32, record: bool) -> Result<(StepResult, Option<presk_rt::InterpBackend<'_>>), String> {
        let mut b = presk_rt::InterpBackend::new(&self.core, &self.prog, bits);
        if record {
            b.interp.magnitudes = Some(Default::default());
            b.interp.minima = Some(Default::default());
            b.interp.elems = Some(Default::default());
            b.interp.ops = Some(Default::default());
            b.interp.terms = Some(Default::default());
        }
        let r = self.setup.step(&mut b, &Realization::uniform(input)).map_err(|e| e.message)?;
        Ok((r, record.then_some(b)))
    }

    /// One native step under `r`.
    fn native_step(&self, r: &Realization, other: bool) -> Result<StepResult, String> {
        let mut b = native(&self.core, &self.prog, r)?;
        if other { self.setup.step_other_batch(&mut b, r) } else { self.setup.step(&mut b, r) }.map_err(|e| e.message)
    }
}

/// References, one per input type, computed when first needed: the exact
/// step, or with `native` a native `f64` step standing in for it.
struct References<'c> {
    ctx: &'c Ctx,
    bits: u32,
    native: bool,
    by_input: BTreeMap<Dtype, StepResult>,
}

impl References<'_> {
    fn get(&mut self, d: Dtype) -> Result<&StepResult, String> {
        if !self.by_input.contains_key(&d) {
            let r = if self.native {
                let one = Realization::uniform(Dtype::F64);
                let mut b = native(&self.ctx.core, &self.ctx.prog, &one)?;
                self.ctx.setup.step_from(&mut b, &one, d).map_err(|e| e.message)?
            } else {
                self.ctx.exact(d, self.bits, false)?.0
            };
            self.by_input.insert(d, r);
        }
        Ok(&self.by_input[&d])
    }

    /// The update and gradient gaps of `r`, natively, against its exact step.
    fn gap(&mut self, r: &Realization) -> Result<presk_train::step::StepGap, String> {
        let got = self.ctx.native_step(r, false)?;
        let exact = self.get(r.default)?.clone();
        Ok(self.ctx.setup.gap(r, &exact, &got))
    }
}

fn machine(profile: Option<&Path>) -> Box<dyn presk_precision::solve::Machine> {
    struct Roofline(presk_opt::Profile);
    impl presk_precision::solve::Machine for Roofline {
        fn seconds(&self, d: Dtype, work: f64, bytes: f64) -> f64 {
            self.0.predict_for(d.name(), work, bytes)
        }
    }
    match profile.map(presk_opt::Profile::load) {
        Some(Ok(p)) => Box::new(Roofline(p)),
        Some(Err(e)) => {
            eprintln!("presk: {e}; pricing traffic instead of time");
            Box::new(presk_precision::solve::Traffic)
        }
        None => Box::new(presk_precision::solve::Traffic),
    }
}

fn describe(s: &StepSetup, a: &StepArgs, cmd: &str) {
    println!(
        "presk {cmd}: step {} of {} from {} after {} warm-up step(s) in f64, on samples {}..{} of {} (noise: the next {})",
        s.t,
        s.total,
        a.weights.display(),
        a.warmup,
        a.from,
        a.from + a.samples,
        s.data.display(),
        a.samples
    );
}

/// `presk calibrate --step`: the envelope of one exact step, the sampling
/// noise, and the gaps of the realizations asked for.
pub fn calibrate(a: &StepArgs, realize: &[String], out: Option<&Path>, profile: Option<&Path>, native_reference: bool) -> ExitCode {
    // Its `bf16` rows, or an envelope `presk solve --step` reads, whose menu
    // is `bf16` too, keep the fusion groups (D94).
    let groups = out.is_some() || realize.iter().any(|s| s == "recipe" || Realization::parse(s).map_or(true, |r| r.groups_visible()));
    let ctx = match context("calibrate", a, groups) {
        Ok(c) => c,
        Err(c) => return c,
    };
    describe(&ctx.setup, a, "calibrate");
    if native_reference {
        return against_f64(&ctx, a, realize, out);
    }
    let started = std::time::Instant::now();
    let (exact, interp) = match ctx.exact(Dtype::F64, a.bits, true) {
        Ok(x) => x,
        Err(m) => return fail("calibrate", m),
    };
    let mut interp = interp.expect("recorded").interp;
    let seconds = started.elapsed().as_secs_f64();
    let today = presk_precision::envelope::today();
    let mut env = presk_precision::envelope::Envelope::new(&a.file.display().to_string(), "step", a.bits, &today);
    env.consts = a.consts.to_vec();
    env.weights = Some(a.weights.display().to_string());
    env.data = Some(presk_precision::envelope::DataSlice { path: ctx.setup.data.display().to_string(), from: a.from, samples: Some(a.samples) });
    env.step = Some(presk_precision::envelope::StepSpec { warmup: a.warmup, total: ctx.setup.total, seed: a.seed, remat: Some(crate::remat().name().to_string()) });
    env.observe(&interp.magnitudes.take().unwrap_or_default());
    env.observe_minima(&interp.minima.take().unwrap_or_default());
    env.observe_elems(&interp.elems.take().unwrap_or_default());
    env.observe_ops(&interp.ops.take().unwrap_or_default());
    env.observe_terms(&interp.terms.take().unwrap_or_default());
    env.samples = a.samples;
    let fns: std::collections::BTreeSet<&str> = env.sites.keys().filter_map(|s| s.rsplit_once('/').map(|(f, _)| f)).collect();
    println!("presk calibrate: {} sites in {} functions of one exact step at {} bits ({:.1} s)", env.sites.len(), fns.len(), a.bits, seconds);
    let step_norm = exact.theta.dist2_f64(&ctx.setup.theta).sqrt();
    let gnorm = exact.grad.sum_squares_f64().sqrt();
    println!("  |θ′ − θ|    {step_norm:.4e}   |g| {gnorm:.4e}   |θ| {:.4e}", ctx.setup.theta.sum_squares_f64().sqrt());
    let one = Realization::uniform(Dtype::F64);
    match (ctx.native_step(&one, false), ctx.native_step(&one, true)) {
        (Ok(x), Ok(y)) => println!("  noise       {:.4e}   ‖g_B − g_B′‖₂ / ‖g_B‖₂ between two batches of {} (f64): what a rounding error is to be compared with", presk_train::step::sampling_noise(&x, &y), a.samples),
        (Err(m), _) | (_, Err(m)) => return fail("calibrate", m),
    }
    let groups = presk_precision::solve::step_groups(&env, &presk_precision::solve::Menu::default(), machine(profile).as_ref());
    let mut refs = References { ctx: &ctx, bits: a.bits, native: false, by_input: BTreeMap::from([(Dtype::F64, exact)]) };
    if !realize.is_empty() {
        println!("{:<28} {:>12} {:>13} {:>12} {:>14}", "realization", "update gap", "storage floor", "grad gap", if profile.is_some() { "predicted s" } else { "predicted B" });
    }
    let names: Vec<String> = groups.iter().map(|g| g.name.clone()).collect();
    for spec in realize {
        let r = match parse(spec, &names, &ctx.setup.opt_name) {
            Ok(r) => r,
            Err(m) => return fail("calibrate", m),
        };
        eprint!("{}", render_all(&crate::stored_vjps(&ctx.core, &r), &Default::default(), color()));
        match refs.gap(&r) {
            Ok(g) => println!("{:<28} {:>12.3e} {:>13.3e} {:>12.3e} {:>14.4e}", spec, g.update, g.floor, g.grad, presk_precision::solve::step_cost(&groups, &r)),
            Err(m) => return fail("calibrate", m),
        }
    }
    if let Some(path) = out {
        if let Err(m) = env.save(path) {
            return fail("calibrate", m);
        }
        println!("presk calibrate: wrote {}", path.display());
    }
    ExitCode::SUCCESS
}

/// A realization by name: `f64`, `f32`, `bf16`, `@FILE`, or `recipe` —
/// mixed precision as practitioners use it: the forward and backward
/// functions in `bf16` (arithmetic and accumulation in `f32`, D34), the
/// optimizer — its step, its schedule — and the master weights in `f32`
/// (§3.7, the folklore row).
fn parse(spec: &str, functions: &[String], opt: &str) -> Result<Realization, String> {
    if spec != "recipe" {
        return Realization::parse(spec);
    }
    Ok(recipe(functions, opt))
}

pub fn recipe(functions: &[String], opt: &str) -> Realization {
    // Every function of the optimizer (`adamw.step`, `adamw.lr`) stays wide.
    let optimizer = opt.rsplit_once('.').map_or(opt, |(o, _)| o);
    let mut r = Realization::uniform(Dtype::F32);
    r.functions = functions.iter().filter(|f| f.rsplit_once('.').map(|(o, _)| o) != Some(optimizer)).map(|f| (f.clone(), Dtype::Bf16)).collect();
    r
}

/// `--reference f64`: every realization against a native `f64` step from
/// the same rounded state, where the exact one is not affordable. No
/// envelope: its magnitudes are the exact interpreter's to measure.
fn against_f64(ctx: &Ctx, a: &StepArgs, realize: &[String], out: Option<&Path>) -> ExitCode {
    if out.is_some() {
        eprintln!("presk calibrate: --reference f64 writes no envelope (the magnitudes are the exact interpreter's)");
        return ExitCode::from(2);
    }
    // Every function: naming one the step does not run changes nothing.
    let names: Vec<String> = ctx.core.funcs.iter().map(|f| f.name.clone()).collect();
    let mut refs = References { ctx, bits: a.bits, native: true, by_input: BTreeMap::new() };
    let one = Realization::uniform(Dtype::F64);
    match (ctx.native_step(&one, false), ctx.native_step(&one, true)) {
        (Ok(x), Ok(y)) => println!("  noise       {:.4e}   ‖g_B − g_B′‖₂ / ‖g_B‖₂ between two batches of {} (f64)", presk_train::step::sampling_noise(&x, &y), a.samples),
        (Err(m), _) | (_, Err(m)) => return fail("calibrate", m),
    }
    println!("{:<28} {:>12} {:>13} {:>12}   (against a native f64 step, not the exact one)", "realization", "update gap", "storage floor", "grad gap");
    for spec in realize {
        let r = match parse(spec, &names, &ctx.setup.opt_name) {
            Ok(r) => r,
            Err(m) => return fail("calibrate", m),
        };
        eprint!("{}", render_all(&crate::stored_vjps(&ctx.core, &r), &Default::default(), color()));
        match refs.gap(&r) {
            Ok(g) => println!("{:<28} {:>12.3e} {:>13.3e} {:>12.3e}", spec, g.update, g.floor, g.grad),
            Err(m) => return fail("calibrate", m),
        }
    }
    ExitCode::SUCCESS
}

/// `presk solve --step`: the step solver over an envelope written by
/// `presk calibrate --step`.
pub fn solve(file: &Path, env_path: &Path, budget: f64, bits: u32, verify: bool, out: Option<&Path>, profile: Option<&Path>) -> ExitCode {
    let env = match presk_precision::envelope::Envelope::load(env_path) {
        Ok(e) => e,
        Err(m) => return fail("solve", m),
    };
    let (Some(spec), Some(weights), Some(data)) = (&env.step, &env.weights, &env.data) else {
        eprintln!("presk solve: --step needs an envelope written by `presk calibrate --step`, and {} is of an entry", env_path.display());
        return ExitCode::from(2);
    };
    // The step's functions are the `.fwd`s and `.bwd`s storing and the
    // `.vjp`s recomputing: the envelope's sites are of one policy. One
    // written before the policy was recorded says it by its sites' names.
    let stored = env.sites.keys().any(|s| s.rsplit_once('/').is_some_and(|(f, _)| f.ends_with(".bwd") || f.ends_with(".fwd")));
    let calibrated = spec.remat.as_deref().unwrap_or(if stored { "store" } else { "recompute" });
    if calibrated != crate::remat().name() {
        eprintln!("presk solve: {} was calibrated under --remat {calibrated}: solve it under the same policy", env_path.display());
        return ExitCode::from(2);
    }
    let weights = PathBuf::from(weights);
    let data_path = PathBuf::from(&data.path);
    let a = StepArgs { file, consts: &env.consts, weights: &weights, warmup: spec.warmup, data: Some(&data_path), from: data.from, samples: data.samples.unwrap_or(1), steps: Some(spec.total), seed: spec.seed, bits };
    // Every function is tried in `bf16`: the fusion groups kept (D94).
    let ctx = match context("solve", &a, true) {
        Ok(c) => c,
        Err(c) => return c,
    };
    describe(&ctx.setup, &a, "solve");
    let mut refs = References { ctx: &ctx, bits, native: false, by_input: BTreeMap::new() };
    if let Err(m) = refs.get(Dtype::F64) {
        return fail("solve", m);
    }
    let menu = presk_precision::solve::Menu::default();
    let groups = presk_precision::solve::step_groups(&env, &menu, machine(profile).as_ref());
    let mut error: Option<String> = None;
    let mut runs = 0usize;
    let mut measure = |r: &Realization| -> f64 {
        runs += 1;
        match refs.gap(r) {
            Ok(g) => g.update,
            Err(m) => {
                error.get_or_insert(m);
                f64::INFINITY
            }
        }
    };
    let sol = presk_precision::solve::solve_step(groups, budget, &menu, Dtype::F64, &mut measure, verify);
    if let Some(m) = error {
        return fail("solve", m);
    }
    let unit = if profile.is_some() { "s (predicted)" } else { "bytes" };
    println!("presk solve: the step under an update budget of {budget:.2e} (‖θ̂′−θ′‖₂/‖θ′−θ‖₂), {} functions, {runs} native steps measured", sol.groups.len());
    println!("{:<30} {:>6} {:>12} {:>11} {:>11} {:>8}", "function", "sites", "f64 cost", "gap f32", "gap bf16", "chosen");
    for g in &sol.groups {
        let gap = |d| g.gap.get(&d).map(|x: &f64| format!("{x:.2e}")).unwrap_or_else(|| "-".into());
        println!("{:<30} {:>6} {:>12.3e} {:>11} {:>11} {:>8}", g.name, g.sites, g.cost.get(&Dtype::F64).copied().unwrap_or(0.0), gap(Dtype::F32), gap(Dtype::Bf16), sol.chosen[&g.name].name());
    }
    let n = |d: Dtype| sol.chosen.values().filter(|x| **x == d).count();
    println!("  chosen      {} f64, {} f32, {} bf16", n(Dtype::F64), n(Dtype::F32), n(Dtype::Bf16));
    println!("  cost        {:.4e} against {:.4e} all-f64 ({:.2}x), {unit}", sol.cost, sol.baseline, sol.speedup());
    println!("  estimate    {:.3e} (√Σ of the chosen functions' isolated gaps: it proposes, it does not guarantee)", sol.estimate);
    match sol.measured {
        Some(g) => {
            let (grad, floor) = refs.gap(&sol.realization).map(|x| (x.grad, x.floor)).unwrap_or((f64::NAN, f64::NAN));
            println!("  measured    {g:.3e} update gap (storage floor {floor:.3e}), {grad:.3e} gradient gap, after {} widening round(s) — {}", sol.rounds, if g <= budget { "inside the budget" } else { "OUTSIDE the budget" });
        }
        None => println!("  measured    not checked (--no-verify)"),
    }
    if let Some(path) = out {
        if let Some(d) = path.parent().filter(|d| !d.as_os_str().is_empty())
            && let Err(e) = std::fs::create_dir_all(d)
        {
            return fail("solve", e);
        }
        let mut r = sol.realization.clone();
        r.envelope = Some(env_path.display().to_string());
        if let Err(e) = std::fs::write(path, r.to_json()) {
            return fail("solve", e);
        }
        println!("presk solve: wrote {} — train with it with `presk train --realize @{}`", path.display(), path.display());
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The recipe narrows the forward and backward functions and leaves the
    /// whole optimizer — its step and its schedule — and the master weights
    /// wide, as practitioners do.
    #[test]
    fn the_recipe_keeps_the_optimizer_wide() {
        let fns: Vec<String> = ["Linear.apply", "Linear.apply.vjp", "loss.vjp", "adamw.step", "adamw.lr"].iter().map(|s| s.to_string()).collect();
        let r = recipe(&fns, "adamw.step");
        assert_eq!(r.default, Dtype::F32);
        assert_eq!(r.dtype_at("Linear.apply.vjp/%3"), Dtype::Bf16);
        assert_eq!(r.dtype_at("loss.vjp/%0"), Dtype::Bf16);
        assert_eq!(r.dtype_at("adamw.step/%3"), Dtype::F32);
        assert_eq!(r.dtype_at("adamw.lr/%1"), Dtype::F32);
    }
}
