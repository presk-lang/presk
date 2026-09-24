//! The tools of the RA4 test (plan 0.7.0, step 5): `presk plan` chooses a
//! layout per nest statically, from the layout cost model; `presk autotune`
//! measures every point of every nest's layout space, asserting that each
//! computes the bits of the default; `presk rank` compares the two over two
//! days of measurement, with the plan's rule for which pairs a measurement
//! distinguishes (§3.6).
//!
//! All three read an entry the same way: run once natively under its default
//! layout with the kernel backend capturing the first launch of every nest at
//! every shape ([`presk_kernel::Launch`]), then lower each nest again under
//! every layout of its space. The sweep is per nest, not over the product of
//! nests: kernels of one entry are launched one after another, so a nest's
//! time does not depend on another nest's layout.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use presk_kernel::cost::{Machine, planned_cost};
use presk_kernel::exec::{output_buf, run_kernel};
use presk_kernel::plan::{launch_kind, lower_launch};
use presk_kernel::{Kernel, KernelExec, Launch, LaunchOp, Threads};
use presk_opt::profile::{LayoutFit, group_key};
use presk_rt::layout::{BlockLoop, KBlock, Split};
use presk_rt::{Buf, Layout, NestKind};
use serde_json::{Value as Json, json};

use super::{color, core, parse_dims, render_all};

/// Which layouts a sweep covers.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Space {
    /// The reference's space (`Layout::space`) times every split.
    Full,
    /// The plan's daily subset: the tiles `{1, 4} × {8, 16}`, no k-block and
    /// 128, the orders `IJ JI KIJ IKJ IJK`, the splits none, `(0, p_M)` and
    /// `(1, p_M)`; every lane width, pack and pass count.
    Small,
}

impl Space {
    pub fn parse(s: &str) -> Result<Space, String> {
        match s {
            "full" => Ok(Space::Full),
            "small" => Ok(Space::Small),
            other => Err(format!("--space {other}: `full` or `small`")),
        }
    }
}

/// The entry, run once natively under its default layout, its launches
/// captured.
pub struct Captured {
    pub launches: Vec<Launch>,
    pub machine: Machine,
    pub seconds: f64,
}

/// Runs `entry` once natively with the kernel backend capturing, and returns
/// the launches of the nests `nest` selects (every nest when `None`).
pub fn capture(file: &Path, entry: &str, dims: &[String], consts: &[String], plan: &presk_rt::Plan, seed: u64, nest: Option<&str>) -> Result<Captured, ExitCode> {
    let t0 = Instant::now();
    let dims = parse_dims(dims)?;
    let (sm, prog, mut core) = core(file, consts, false)?;
    let diags = presk_core::ad::transform(&mut core);
    eprint!("{}", render_all(&diags, &sm, color()));
    if presk_syntax::diag::has_errors(&diags) {
        return Err(ExitCode::from(1));
    }
    let realize = &plan.realization;
    let opts = presk_interp::RunOptions { entry: entry.to_string(), dims, seed, bits: 64, digits: 12 };
    let fail = |m: String| {
        eprint!("{}", render_all(&[presk_syntax::Diagnostic::error("E7001", m)], &sm, color()));
        ExitCode::from(1)
    };
    let (func, fdims, inputs) = presk_interp::run::prepare_run(&core, &prog, &opts).map_err(|e| fail(e.message))?;
    let mut args: Vec<presk_rt::Value> = inputs.iter().map(|(_, v)| presk_rt::Value::from_interp(v).map(|v| v.realize(realize.default))).collect::<presk_rt::Result<_>>().map_err(|e| fail(e.message))?;
    realize.quantize_args(&mut args).map_err(|e| fail(e.message))?;
    let mut b = presk_cranelift::native_backend(&core, &prog, realize, seed).map_err(|e| fail(e.message))?;
    b.capture = Some(Vec::new());
    let p = presk_rt::Backend::prepare(&mut b, func, &fdims, realize).map_err(|e| fail(e.message))?;
    presk_rt::Backend::call(&mut b, &p, args).map_err(|e| fail(e.message))?;
    let mut launches = b.capture.take().unwrap_or_default();
    if let Some(n) = nest {
        launches.retain(|l| l.nest == n);
        if launches.is_empty() {
            eprintln!("presk: the entry launches no nest `{n}`");
            return Err(ExitCode::from(2));
        }
    }
    Ok(Captured { launches, machine: Machine::detect(), seconds: t0.elapsed().as_secs_f64() })
}

/// Point 0 of a nest: its default layout, scalar, on one worker, no pack —
/// the kernel every other point is compared with, bit for bit.
pub fn point_zero(kind: &NestKind) -> Layout {
    let d = Layout::default_for(kind);
    Layout { lanes: Some(1), split: Some(Split::None), pack: d.pack.map(|_| false), ..d }
}

/// The layouts of a nest's space that lower at this launch, each with its
/// kernels: `space` narrowed to the daily subset if asked, times the splits
/// the machine allows, each split checked against the kernel (condition 8).
/// Point 0 comes first.
pub fn points(l: &Launch, space: Space, workers: usize) -> Vec<(Layout, Vec<Kernel>)> {
    let kind = launch_kind(l);
    let mut base = Layout::space(&kind);
    if space == Space::Small {
        let orders: [&[BlockLoop]; 5] = [&[BlockLoop::I, BlockLoop::J], &[BlockLoop::J, BlockLoop::I], &[BlockLoop::K, BlockLoop::I, BlockLoop::J], &[BlockLoop::I, BlockLoop::K, BlockLoop::J], &[BlockLoop::I, BlockLoop::J, BlockLoop::K]];
        base.retain(|p| {
            p.tile.is_none_or(|(mi, nj)| [1, 4].contains(&mi) && [8, 16].contains(&nj))
                && p.kblock.is_none_or(|k| matches!(k, KBlock::None | KBlock::Width(128)))
                && p.order.as_ref().is_none_or(|o| orders.contains(&o.as_slice()))
        });
    }
    let mut splits = vec![Split::None];
    let depths: &[usize] = if space == Space::Small { &[0, 1] } else { &[0, 1, 2] };
    let mut counts = if space == Space::Small { vec![workers] } else { vec![workers / 2, workers] };
    counts.retain(|p| *p >= 2);
    counts.dedup();
    for d in depths {
        for p in &counts {
            splits.push(Split::At { depth: *d, workers: *p });
        }
    }
    let zero = point_zero(&kind);
    let mut out: Vec<(Layout, Vec<Kernel>)> = Vec::new();
    if let Ok(ks) = lower_launch(l, &zero, Some(workers)) {
        out.push((zero.clone(), ks));
    }
    for b in &base {
        for s in &splits {
            let p = Layout { split: Some(*s), ..b.clone() };
            if p == zero {
                continue;
            }
            if let Ok(ks) = lower_launch(l, &p, Some(workers)) {
                out.push((p, ks));
            }
        }
    }
    out
}

/// The model's prediction for a point: the sum over its kernels, `None` if a
/// kernel's group was not calibrated.
pub fn predict(fit: &LayoutFit, machine: &Machine, dtype: presk_rt::Dtype, kernels: &[Kernel]) -> Option<f64> {
    kernels.iter().map(|k| {
        let c = planned_cost(k, machine);
        fit.predict(&c, &group(dtype, k, c.workers))
    }).sum()
}

/// A kernel's group: its type, lanes, and the workers its split states.
fn group(dtype: presk_rt::Dtype, k: &Kernel, effective: usize) -> String {
    let stated = match k.threads {
        Threads::At { workers, .. } => workers,
        Threads::One => 1,
        Threads::Default => effective,
    };
    group_key(dtype.name(), k.lanes, stated)
}

/// `a[2, 6, 256, 64] b[…]`: a launch's operand shapes.
fn shape(l: &Launch) -> String {
    l.operands.iter().map(|o| format!("[{}]", o.axes.iter().map(|a| a.len.to_string()).collect::<Vec<_>>().join(", "))).collect::<Vec<_>>().join(" ")
}

fn load_fit(profile: Option<&Path>) -> Result<Option<(LayoutFit, Option<Machine>)>, ExitCode> {
    let Some(path) = profile else { return Ok(None) };
    match presk_opt::Profile::load(path) {
        Ok(p) => match p.layout {
            Some(f) => Ok(Some((f, p.machine_desc))),
            None => {
                eprintln!("presk: {} has no layout cost model: calibrate it with `presk bench --calibrate {} --space calib`", path.display(), path.display());
                Err(ExitCode::from(2))
            }
        },
        Err(m) => {
            eprintln!("presk: --profile: {m}");
            Err(ExitCode::from(2))
        }
    }
}

// ----- presk plan -------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn plan(file: &Path, entry: &str, dims: &[String], consts: &[String], base: &presk_rt::Plan, profile: &Path, nest: Option<&str>, space: Space, cost: bool, out: Option<&Path>, seed: u64) -> ExitCode {
    let t0 = Instant::now();
    let (fit, stored) = match load_fit(Some(profile)) {
        Ok(Some(x)) => x,
        Ok(None) => unreachable!("a profile was given"),
        Err(c) => return c,
    };
    let captured = match capture(file, entry, dims, consts, base, seed, nest) {
        Ok(c) => c,
        Err(c) => return c,
    };
    // The machine the model was fitted on; the one the plan is for decides
    // which worker counts are well-formed.
    let model_machine = stored.unwrap_or_else(|| captured.machine.clone());
    let t_cost = Instant::now();
    let mut layout: BTreeMap<String, Layout> = BTreeMap::new();
    let mut rows: Vec<String> = Vec::new();
    let mut total_points = 0usize;
    for l in &captured.launches {
        let pts = points(l, space, captured.machine.workers);
        total_points += pts.len();
        let mut scored: Vec<(f64, &Layout)> = pts.iter().filter_map(|(p, ks)| predict(&fit, &model_machine, l.dtype, ks).map(|t| (t, p))).collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let Some((best_t, best)) = scored.first().copied() else { continue };
        let kept = scored.iter().filter(|(t, _)| *t <= best_t * (1.0 + fit.u)).count();
        if cost {
            for (t, p) in &scored {
                rows.push(format!("  {:<28} {:>12.2} µs  {p}", l.nest, t * 1e6));
            }
        }
        rows.push(format!("{:<28} {:<24} {:>5} points  {:>10.2} µs  {kept:>3} within 1 + u  {best}", l.nest, shape(l), pts.len(), best_t * 1e6));
        // One layout per nest: the first shape a nest launches at decides it.
        layout.entry(l.nest.clone()).or_insert_with(|| best.clone());
    }
    let cost_seconds = t_cost.elapsed().as_secs_f64();
    println!("{:<28} {:<24} {:>12} {:>13}  {:>17}  layout", "nest", "operands", "space", "predicted", "undecided");
    for r in &rows {
        println!("{r}");
    }
    println!("-- {} nests, {total_points} points costed in {cost_seconds:.3} s, the entry captured in {:.3} s; u = {:.3}", captured.launches.len(), captured.seconds, fit.u);
    println!("-- wall time {:.3} s", t0.elapsed().as_secs_f64());
    if let Some(path) = out {
        let p = presk_rt::Plan { realization: base.realization.clone(), realization_ref: base.realization_ref.clone(), profile: Some(profile.display().to_string()), layout };
        if let Err(e) = std::fs::write(path, p.to_json()) {
            eprintln!("presk plan: {}: {e}", path.display());
            return ExitCode::from(2);
        }
        println!("-- plan written to {}", path.display());
    }
    ExitCode::SUCCESS
}

// ----- presk autotune ---------------------------------------------------------

/// The time of one run of a point's kernels on the captured inputs, its
/// output, on one executor. A pack's copy runs first, its output the
/// contraction's operand 1.
fn run_point(exec: &mut dyn KernelExec, idx: usize, l: &Launch, kernels: &[Kernel]) -> Result<(f64, Buf), String> {
    let inputs: Vec<&Buf> = l.inputs.iter().collect();
    let t0 = Instant::now();
    let out = match kernels {
        [k] => {
            let mut out = output_buf(k);
            exec.run(idx, k, &inputs, &mut out).map_err(|e| e.message)?;
            out
        }
        [pack, k] => {
            let mut panel = output_buf(pack);
            exec.run(idx, pack, &inputs[1..2], &mut panel).map_err(|e| e.message)?;
            let mut out = output_buf(k);
            exec.run(idx + 1, k, &[inputs[0], &panel], &mut out).map_err(|e| e.message)?;
            out
        }
        _ => return Err("a point lowers to one kernel or two".into()),
    };
    Ok((t0.elapsed().as_secs_f64(), out))
}

fn bits(b: &Buf) -> Vec<u64> {
    b.to_f64().into_iter().map(|x| x.to_bits()).collect()
}

fn interp_out(l: &Launch, kernels: &[Kernel]) -> Buf {
    let inputs: Vec<&Buf> = l.inputs.iter().collect();
    match kernels {
        [k] => {
            let mut out = output_buf(k);
            run_kernel(k, &inputs, &mut out);
            out
        }
        [pack, k] => {
            let mut panel = output_buf(pack);
            run_kernel(pack, &inputs[1..2], &mut panel);
            let mut out = output_buf(k);
            run_kernel(k, &[inputs[0], &panel], &mut out);
            out
        }
        _ => unreachable!("one kernel or two"),
    }
}

fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() { 0.0 } else { v[v.len() / 2] }
}

/// Which points of each nest a sweep takes, and what it does with them.
pub struct SweepOpts {
    /// Timed launches per point after the warm-up. Zero runs each point once,
    /// untimed, for its bits: the check of statement L, split over processes.
    pub repeat: usize,
    /// Also check every point's bits on the kernel interpreter.
    pub interp: bool,
    /// A sample instead of the whole space: the model's `top` best points and
    /// `random` others drawn by the seed (both zero: every point).
    pub top: usize,
    pub random: usize,
    /// `(k, n)`: the points whose place in the selection is `k` modulo `n`.
    /// Every shard runs point 0, the reference, and shard 0 records it and
    /// the 0.6.0 default.
    pub shard: Option<(usize, usize)>,
}

/// FNV-1a, a hash that does not change between builds: the random part of a
/// sample is the points whose keys hash lowest under the seed.
fn fnv(s: &str, seed: u64) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ seed;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// The indices of the points a sweep takes, point 0 first.
fn select(pts: &[(Layout, Vec<Kernel>)], predicted: &[Option<f64>], opts: &SweepOpts, seed: u64) -> Vec<usize> {
    let mut sel: Vec<usize> = (0..pts.len()).collect();
    if opts.top > 0 || opts.random > 0 {
        let mut rest: Vec<usize> = (1..pts.len()).collect();
        rest.sort_by(|a, b| predicted[*a].unwrap_or(f64::INFINITY).total_cmp(&predicted[*b].unwrap_or(f64::INFINITY)));
        let mut others = rest.split_off(opts.top.min(rest.len()));
        others.sort_by_key(|i| fnv(&pts[*i].0.key(), seed));
        others.truncate(opts.random);
        sel = [vec![0], rest, others].concat();
        sel.sort();
    }
    if let Some((k, n)) = opts.shard {
        sel = sel.into_iter().enumerate().filter(|(pos, i)| *i == 0 || pos % n == k).map(|(_, i)| i).collect();
    }
    sel
}

/// A launch's primitive, `ein`, `dot`, `softmax`…, or `fused`.
fn op_name(l: &Launch) -> String {
    match &l.what {
        LaunchOp::Prim(p) => p.name().split(['[', ' ']).next().unwrap_or("").to_string(),
        LaunchOp::Fused { .. } => "fused".into(),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn autotune(file: &Path, entry: &str, dims: &[String], consts: &[String], base: &presk_rt::Plan, profile: Option<&Path>, nest: Option<&str>, space: Space, opts: &SweepOpts, out: &Path, seed: u64) -> ExitCode {
    let t0 = Instant::now();
    let fit = match load_fit(profile) {
        Ok(f) => f,
        Err(c) => return c,
    };
    if (opts.top > 0 || opts.random > 0) && fit.is_none() {
        eprintln!("presk autotune: --top and --random choose by the model's predictions: they need --profile");
        return ExitCode::from(2);
    }
    let captured = match capture(file, entry, dims, consts, base, seed, nest) {
        Ok(c) => c,
        Err(c) => return c,
    };
    if let Err(e) = std::fs::create_dir_all(out) {
        eprintln!("presk autotune: {}: {e}", out.display());
        return ExitCode::from(2);
    }
    let model_machine = fit.as_ref().and_then(|(_, m)| m.clone()).unwrap_or_else(|| captured.machine.clone());
    let mut exec = match presk_cranelift::NativeExec::new() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("presk autotune: native backend unavailable: {}", e.message);
            return ExitCode::from(1);
        }
    };
    let config = consts.iter().chain(dims).cloned().collect::<Vec<_>>().join(",");
    let first_shard = opts.shard.is_none_or(|(k, _)| k == 0);
    let workers = captured.machine.workers;
    let mut next_idx = 0usize;
    let (mut sweep_seconds, mut cost_seconds, mut interp_seconds) = (0.0f64, 0.0f64, 0.0f64);
    let (mut total, mut baselines, mut failures) = (0usize, 0usize, 0usize);
    for l in &captured.launches {
        let tc = Instant::now();
        let pts = points(l, space, workers);
        let predict_all = |ks: &[Kernel]| fit.as_ref().and_then(|(f, _)| predict(f, &model_machine, l.dtype, ks));
        let predicted: Vec<Option<f64>> = pts.iter().map(|(_, ks)| predict_all(ks)).collect();
        cost_seconds += tc.elapsed().as_secs_f64();
        let sel = select(&pts, &predicted, opts, seed);
        // 0.6.0's point: the default layout, its split left to the executor.
        let default = Layout::default_for(&launch_kind(l));
        let baseline = if first_shard { lower_launch(l, &default, Some(workers)).ok() } else { None };
        let mut items: Vec<(&Layout, &[Kernel], Option<f64>, bool)> = sel.iter().map(|i| (&pts[*i].0, pts[*i].1.as_slice(), predicted[*i], false)).collect();
        if let Some(ks) = &baseline {
            items.push((&default, ks.as_slice(), predict_all(ks), true));
        }
        let op = op_name(l);
        let ts = Instant::now();
        let file_name = format!("{}.jsonl", l.nest.replace(['/', '%'], "_"));
        let mut lines: Vec<String> = Vec::new();
        let mut reference: Option<(Vec<u64>, Option<Vec<u64>>)> = None;
        for (layout, kernels, pred, is_baseline) in items {
            let idx = next_idx;
            next_idx += kernels.len();
            // One warm-up launch (it compiles), then `repeat` timed ones.
            let mut samples = Vec::with_capacity(opts.repeat);
            let mut last = None;
            for i in 0..=opts.repeat {
                match run_point(&mut exec, idx, l, kernels) {
                    Ok((t, o)) => {
                        if i > 0 {
                            samples.push(t);
                        }
                        last = Some(o);
                    }
                    Err(m) => {
                        eprintln!("presk autotune: {} under {layout}: {m}", l.nest);
                        return ExitCode::from(1);
                    }
                }
            }
            let native_bits = bits(&last.expect("at least one run"));
            let ti = Instant::now();
            let interp_bits = opts.interp.then(|| bits(&interp_out(l, kernels)));
            interp_seconds += ti.elapsed().as_secs_f64();
            // Statement L: every point computes point 0's bits, on each executor.
            let is_zero = reference.is_none();
            let identical = match &reference {
                None => {
                    reference = Some((native_bits.clone(), interp_bits.clone()));
                    true
                }
                Some((n, i)) => *n == native_bits && (interp_bits.is_none() || *i == interp_bits),
            };
            if !identical {
                failures += 1;
                eprintln!("presk autotune: {} under {layout}: the output differs from point 0's", l.nest);
            }
            if is_zero && !first_shard {
                continue;
            }
            let med = median(&samples);
            let mad = median(&samples.iter().map(|t| (t - med).abs()).collect::<Vec<_>>());
            // Per kernel (a pack's copy first): the model's inputs, its group
            // and the terms of its prediction.
            let per_kernel: Vec<Json> = kernels
                .iter()
                .map(|k| {
                    let c = planned_cost(k, &model_machine);
                    let g = group(l.dtype, k, c.workers);
                    let terms = fit.as_ref().and_then(|(f, _)| f.terms(&c, &g));
                    json!({ "group": g, "cost": c, "terms": terms, "bound": terms.map(|t| t.bound()) })
                })
                .collect();
            lines.push(json!({
                "entry": entry, "config": config, "nest": l.nest, "op": op, "shape": shape(l), "dtype": l.dtype.name(),
                "layout": layout.to_json(), "key": if is_baseline { "0.6.0-default".to_string() } else { layout.key() }, "baseline": is_baseline,
                "timed": opts.repeat > 0, "seconds": samples, "median": med, "mad": mad,
                "predicted": pred, "kernels": per_kernel,
                "identical": identical, "interp_checked": opts.interp,
            }).to_string());
            if is_baseline {
                baselines += 1;
            } else {
                total += 1;
            }
        }
        sweep_seconds += ts.elapsed().as_secs_f64();
        let path = out.join(&file_name);
        if let Err(e) = std::fs::write(&path, lines.join("\n") + "\n") {
            eprintln!("presk autotune: {}: {e}", path.display());
            return ExitCode::from(2);
        }
        println!("{:<28} {:<24} {:>5} of {:>5} points  {}", l.nest, shape(l), sel.len(), pts.len(), path.display());
    }
    let sample = (opts.top > 0 || opts.random > 0).then(|| json!({ "top": opts.top, "random": opts.random, "seed": seed }));
    let summary = json!({
        "entry": entry, "config": config, "consts": consts, "dims": dims, "realization": base.realization.label(),
        "space": if space == Space::Small { "small" } else { "full" }, "repeat": opts.repeat, "timed": opts.repeat > 0,
        "sample": sample, "shard": opts.shard.map(|(k, n)| json!([k, n])),
        "points": total, "baselines": baselines, "sweep_seconds": sweep_seconds - interp_seconds, "cost_seconds": cost_seconds, "interp_seconds": interp_seconds,
        "capture_seconds": captured.seconds, "wall_seconds": t0.elapsed().as_secs_f64(), "machine": captured.machine,
        "u": fit.as_ref().map(|(f, _)| f.u), "interp_checked": opts.interp, "not_identical": failures,
    });
    let _ = std::fs::write(out.join("sweep.json"), serde_json::to_string_pretty(&summary).unwrap_or_default());
    let verb = if opts.repeat > 0 { "measured" } else { "run once, untimed," };
    println!("-- {total} points of {} nests {verb} in {:.1} s, costed in {cost_seconds:.3} s; every output bit-identical to point 0{}: {}", captured.launches.len(), sweep_seconds - interp_seconds, if opts.interp { ", natively and on the kernel interpreter" } else { ", natively" }, if failures == 0 { "yes".to_string() } else { format!("NO, {failures} points differ") });
    if failures > 0 { ExitCode::from(1) } else { ExitCode::SUCCESS }
}

// ----- presk rank -------------------------------------------------------------

/// One measured point, as `presk autotune` wrote it.
struct Point {
    layout: BTreeMap<String, Json>,
    /// Its place in the nest's file, which is the space's order: `presk
    /// plan` breaks a tie between two predictions by it.
    order: usize,
    median: f64,
    mad: f64,
    predicted: Option<f64>,
    identical: bool,
    interp: bool,
}

/// A nest at one shape and type: its points, and 0.6.0's point.
#[derive(Default)]
struct Nest {
    entry: String,
    config: String,
    op: String,
    points: BTreeMap<String, Point>,
    baseline: Option<Point>,
}

/// A day's nests, keyed by (nest, operand shapes, type).
type Day = BTreeMap<(String, String, String), Nest>;

/// One `sweep.json`. The decision cost compares, over the same points, what
/// choosing statically takes — one run of the entry to capture its
/// launches, then lowering and costing every point, which is what `presk
/// plan` does — with what the exhaustive sweep takes: the whole of `presk
/// autotune` but the interpreter's check.
struct Sweep {
    entry: String,
    config: String,
    realization: String,
    sampled: bool,
    points: u64,
    static_seconds: f64,
    exhaustive_seconds: f64,
    not_identical: u64,
    u: Option<f64>,
}

/// Every point under `dir`, which may hold one sweep or one per
/// configuration in subdirectories, or the shards of a check.
fn read_day(dir: &Path) -> Result<(Day, Vec<Sweep>), String> {
    fn walk(dir: &Path, out: &mut Day, sweeps: &mut Vec<Sweep>) -> Result<(), String> {
        let mut entries: Vec<_> = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?.flatten().map(|e| e.path()).collect();
        entries.sort();
        for p in entries {
            if p.is_dir() {
                walk(&p, out, sweeps)?;
                continue;
            }
            if p.file_name().is_some_and(|n| n == "sweep.json") {
                let s: Json = std::fs::read_to_string(&p).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Json::Null);
                let f = |k: &str| s[k].as_f64().unwrap_or(0.0);
                let st = |k: &str| s[k].as_str().unwrap_or("").to_string();
                sweeps.push(Sweep {
                    entry: st("entry"),
                    config: st("config"),
                    realization: st("realization"),
                    sampled: !s["sample"].is_null() || !s["shard"].is_null() || s["timed"] == Json::Bool(false),
                    points: s["points"].as_u64().unwrap_or(0),
                    static_seconds: f("capture_seconds") + f("cost_seconds"),
                    exhaustive_seconds: f("wall_seconds") - f("interp_seconds"),
                    not_identical: s["not_identical"].as_u64().unwrap_or(0),
                    u: s["u"].as_f64(),
                });
                continue;
            }
            if p.extension().is_none_or(|x| x != "jsonl") {
                continue;
            }
            let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            for (order, line) in text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
                let v: Json = serde_json::from_str(line).map_err(|e| format!("{}: {e}", p.display()))?;
                let s = |k: &str| v[k].as_str().unwrap_or("").to_string();
                let layout: BTreeMap<String, Json> = v["layout"].as_object().map(|m| m.iter().map(|(k, x)| (k.clone(), x.clone())).collect()).unwrap_or_default();
                let point = Point { layout, order, median: v["median"].as_f64().unwrap_or(0.0), mad: v["mad"].as_f64().unwrap_or(0.0), predicted: v["predicted"].as_f64(), identical: v["identical"] != Json::Bool(false), interp: v["interp_checked"] == Json::Bool(true) };
                let n = out.entry((s("nest"), s("shape"), s("dtype"))).or_default();
                (n.entry, n.config, n.op) = (s("entry"), s("config"), s("op"));
                if v["baseline"] == Json::Bool(true) {
                    n.baseline = Some(point);
                } else {
                    n.points.insert(s("key"), point);
                }
            }
        }
        Ok(())
    }
    let (mut out, mut sweeps) = (Day::new(), Vec::new());
    walk(dir, &mut out, &mut sweeps)?;
    Ok((out, sweeps))
}

/// The one decision two layouts differ in, if they differ in one.
fn close(a: &BTreeMap<String, Json>, b: &BTreeMap<String, Json>) -> Option<String> {
    let keys: BTreeSet<&String> = a.keys().chain(b.keys()).collect();
    let diff: Vec<&String> = keys.into_iter().filter(|k| a.get(*k) != b.get(*k)).collect();
    match diff.as_slice() {
        [k] => Some((*k).clone()),
        _ => None,
    }
}

#[derive(Default, Clone, Copy)]
struct Tally {
    pairs: usize,
    distinguished: usize,
    right: usize,
    decided: usize,
    decided_right: usize,
    /// Distinguished pairs the model predicts equal: not ranked, so not right.
    tied: usize,
}

impl Tally {
    fn add(&mut self, o: Tally) {
        self.pairs += o.pairs;
        self.distinguished += o.distinguished;
        self.right += o.right;
        self.decided += o.decided;
        self.decided_right += o.decided_right;
        self.tied += o.tied;
    }
    fn accuracy(&self) -> Option<f64> {
        (self.distinguished > 0).then(|| self.right as f64 / self.distinguished as f64)
    }
    fn cells(&self) -> String {
        let pct = |a: usize, b: usize| if b == 0 { "—".to_string() } else { format!("{:.1} %", 100.0 * a as f64 / b as f64) };
        format!("{} | {} | {} | {} | {} | {}", self.pairs, self.distinguished, pct(self.right, self.distinguished), pct(self.decided_right, self.decided), pct(self.distinguished - self.decided, self.distinguished), pct(self.tied, self.distinguished))
    }
}

/// What `presk rank` decides the statements of plan 0.7.0 §3.7 on: M and O
/// on `entry` at `at` in `dtype`, N on every configuration of `entry` swept
/// whole, P on `gemm` at `gemm_at` in `f32`, L on everything, with the
/// interpreter's check read from `check` when it was run apart.
pub struct RankOpts {
    pub check: Option<std::path::PathBuf>,
    pub entry: String,
    pub at: String,
    pub dtype: String,
    pub gemm: String,
    pub gemm_at: String,
}

impl Default for RankOpts {
    fn default() -> RankOpts {
        RankOpts { check: None, entry: "SelfAttention.apply".into(), at: "B=16,T=256".into(), dtype: "f32".into(), gemm: "gemm".into(), gemm_at: "N=1024".into() }
    }
}

/// One nest's results, kept for the statements.
struct Row {
    entry: String,
    config: String,
    op: String,
    nest: String,
    dtype: String,
    tally: Tally,
    top1: f64,
    speedup: Option<f64>,
}

fn verdict(holds: bool) -> &'static str {
    if holds { "**holds**" } else { "**fails**" }
}

pub fn rank(day1: &Path, day2: &Path, out: &Path, opts: &RankOpts) -> ExitCode {
    let read = |d: &Path| read_day(d).map_err(|m| eprintln!("presk rank: {m}"));
    let Ok((d1, s1)) = read(day1) else { return ExitCode::from(2) };
    let Ok((d2, s2)) = read(day2) else { return ExitCode::from(2) };
    let check = match &opts.check {
        Some(c) => match read(c) {
            Ok(x) => Some(x),
            Err(()) => return ExitCode::from(2),
        },
        None => None,
    };
    let one_day = day1 == day2;
    let u = s1.iter().chain(&s2).find_map(|s| s.u).unwrap_or(0.0);
    let mut md = String::new();
    md.push_str("# RA4: static ranking against measurement\n\n");
    if one_day {
        md.push_str(&format!("One sweep, `{}`: a close pair differs in one decision, and it is *distinguished* when its medians differ by more than twice the larger MAD (question 65 asks for two sweeps; this is the rule applied to one).", day1.display()));
    } else {
        md.push_str(&format!("Days: `{}` and `{}`. A close pair differs in one decision; it is *distinguished* when its medians differ by more than twice the larger MAD on both days, with the same sign (question 65).", day1.display(), day2.display()));
    }
    md.push_str(&format!(" *Right*: the model orders it as the measurement does; a pair it predicts equal (*tied*) is not ranked, so not right (question 70). *Decided*: its predicted ratio lies outside 1 ± u, u = {u:.3}. The model's pick is `presk plan`'s: the best prediction, a tie broken by the space's order. Times are the mean of the two days' medians. *vs 0.6.0*: 0.6.0's default layout, its split left to the executor, over the best point measured.\n\n"));
    md.push_str("| nest | config | operands | type | points | close pairs | distinguished | right | right, decided | undecided | tied | Spearman | top-1 regret | top-k regret (k) | vs 0.6.0 |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    let mut classes: BTreeMap<String, Tally> = BTreeMap::new();
    let mut per_nest_classes: Vec<(String, String, String, BTreeMap<String, Tally>)> = Vec::new();
    let mut all = Tally::default();
    let mut rows: Vec<Row> = Vec::new();
    for ((nest, shape, dtype), n1) in &d1 {
        let Some(n2) = d2.get(&(nest.clone(), shape.clone(), dtype.clone())) else { continue };
        let (p1, p2) = (&n1.points, &n2.points);
        // The points both days measured and the model predicted.
        let keys: Vec<&String> = p1.keys().filter(|k| p2.contains_key(*k) && p1[*k].predicted.is_some()).collect();
        if keys.is_empty() {
            continue;
        }
        let mean = |k: &String| (p1[k].median + p2[k].median) / 2.0;
        let mut nest_classes: BTreeMap<String, Tally> = BTreeMap::new();
        let mut total = Tally::default();
        for (i, a) in keys.iter().enumerate() {
            for b in &keys[i + 1..] {
                let Some(class) = close(&p1[*a].layout, &p1[*b].layout) else { continue };
                let mut t = Tally { pairs: 1, ..Default::default() };
                let sep = |d: &BTreeMap<String, Point>| {
                    let (x, y) = (&d[*a], &d[*b]);
                    let gap = x.median - y.median;
                    (gap.abs() > 2.0 * x.mad.max(y.mad)).then_some(gap.signum())
                };
                if let (Some(g1), Some(g2)) = (sep(p1), sep(p2))
                    && g1 == g2
                {
                    t.distinguished = 1;
                    let (pa, pb) = (p1[*a].predicted.unwrap(), p1[*b].predicted.unwrap());
                    // A tie orders nothing: it is not right, whatever the
                    // measurement says.
                    let right = pa != pb && (pa - pb).signum() == g1;
                    t.right = right as usize;
                    t.tied = (pa == pb) as usize;
                    if (pa / pb - 1.0).abs() > u && (pb / pa - 1.0).abs() > u {
                        t.decided = 1;
                        t.decided_right = right as usize;
                    }
                }
                nest_classes.entry(class.clone()).or_default().add(t);
                classes.entry(class).or_default().add(t);
                total.add(t);
            }
        }
        all.add(total);
        let pred: Vec<f64> = keys.iter().map(|k| p1[*k].predicted.unwrap()).collect();
        let meas: Vec<f64> = keys.iter().map(|k| mean(k)).collect();
        let rho = presk_opt::profile::spearman(&pred, &meas);
        let best = meas.iter().cloned().fold(f64::INFINITY, f64::min);
        // The model's pick is `presk plan`'s: the best prediction, a tie
        // broken by the space's order.
        let pick = (0..keys.len()).min_by(|x, y| pred[*x].total_cmp(&pred[*y]).then(p1[keys[*x]].order.cmp(&p1[keys[*y]].order)));
        let top1 = pick.map_or(f64::NAN, |i| meas[i] / best);
        let best_pred = pred.iter().cloned().fold(f64::INFINITY, f64::min);
        let kept: Vec<usize> = (0..keys.len()).filter(|i| pred[*i] <= best_pred * (1.0 + u)).collect();
        let topk = kept.iter().map(|i| meas[*i]).fold(f64::INFINITY, f64::min) / best;
        let speedup = match (&n1.baseline, &n2.baseline) {
            (Some(b1), Some(b2)) if best > 0.0 => Some((b1.median + b2.median) / 2.0 / best),
            _ => None,
        };
        let vs = speedup.map_or("—".to_string(), |s| format!("{s:.2}×"));
        md.push_str(&format!("| `{nest}` | {} | {shape} | {dtype} | {} | {} | {rho:.3} | {top1:.2} | {topk:.2} ({}) | {vs} |\n", n1.config, keys.len(), total.cells(), kept.len()));
        per_nest_classes.push((nest.clone(), n1.config.clone(), dtype.clone(), nest_classes));
        rows.push(Row { entry: n1.entry.clone(), config: n1.config.clone(), op: n1.op.clone(), nest: nest.clone(), dtype: dtype.clone(), tally: total, top1, speedup });
    }
    md.push_str(&format!("| **all** | | | | | {} | | | | |\n\n", all.cells()));
    md.push_str(&statements(opts, &rows, &d1, &s1, &s2, check.as_ref(), one_day));
    md.push_str("## Per decision class\n\n| class | close pairs | distinguished | right | right, decided | undecided | tied |\n|---|---|---|---|---|---|---|\n");
    for (c, t) in &classes {
        md.push_str(&format!("| {c} | {} |\n", t.cells()));
    }
    md.push_str("\n## Per nest and class\n\n| nest | config | type | class | close pairs | distinguished | right | right, decided | undecided | tied |\n|---|---|---|---|---|---|---|---|---|---|\n");
    for (nest, config, dtype, cs) in &per_nest_classes {
        for (c, t) in cs {
            md.push_str(&format!("| `{nest}` | {config} | {dtype} | {c} | {} |\n", t.cells()));
        }
    }
    md.push_str("\n## Decision cost\n\nStatic: the entry run once to capture its launches, then every point lowered and costed (`presk plan`). Exhaustive: every point lowered, compiled and timed (`presk autotune`, without the interpreter's check). A sampled sweep is marked: its exhaustive side is not the whole space.\n\n| day | entry | config | type | points | exhaustive (s) | static (s) | ratio |\n|---|---|---|---|---|---|---|---|\n");
    for (name, sweeps) in [(day1, &s1), (day2, &s2)] {
        for s in sweeps.iter() {
            md.push_str(&format!("| `{}` | `{}` | {} | {} | {}{} | {:.1} | {:.3} | {:.0} |\n", name.display(), s.entry, s.config, s.realization, s.points, if s.sampled { " (sample)" } else { "" }, s.exhaustive_seconds, s.static_seconds, s.exhaustive_seconds / s.static_seconds.max(1e-9)));
        }
        if one_day {
            break;
        }
    }
    if let Err(e) = std::fs::write(out, &md) {
        eprintln!("presk rank: {}: {e}", out.display());
        return ExitCode::from(2);
    }
    print!("{md}");
    println!("-- written to {}", out.display());
    ExitCode::SUCCESS
}

/// The statements of plan 0.7.0 §3.7, decided on what the days hold.
fn statements(opts: &RankOpts, rows: &[Row], d1: &Day, s1: &[Sweep], s2: &[Sweep], check: Option<&(Day, Vec<Sweep>)>, one_day: bool) -> String {
    let mut md = String::from("## Statements (plan 0.7.0 §3.7)\n\n| | statement | measured | verdict |\n|---|---|---|---|\n");
    // L: every point of day 1 checked on the interpreter somewhere, and no
    // output differing anywhere.
    let differing: u64 = s1.iter().chain(s2).chain(check.iter().flat_map(|(_, s)| s)).map(|s| s.not_identical).sum();
    let checked = |nest: &(String, String, String), key: Option<&String>| -> bool {
        let in_day = |d: &Day| {
            d.get(nest).is_some_and(|n| match key {
                Some(k) => n.points.get(k).is_some_and(|p| p.interp && p.identical),
                None => n.baseline.as_ref().is_some_and(|p| p.interp && p.identical),
            })
        };
        in_day(d1) || check.is_some_and(|(c, _)| in_day(c))
    };
    let (mut total, mut covered) = (0usize, 0usize);
    for (k, n) in d1 {
        for key in n.points.keys() {
            total += 1;
            covered += checked(k, Some(key)) as usize;
        }
        if n.baseline.is_some() {
            total += 1;
            covered += checked(k, None) as usize;
        }
    }
    let l = if differing > 0 { verdict(false).to_string() } else if covered == total { verdict(true).to_string() } else { "not decided: not every point was checked on the interpreter".to_string() };
    md.push_str(&format!("| L | every point bit-identical to point 0, natively and on the kernel interpreter | {covered} of {total} points of day 1 checked on the interpreter; {differing} outputs differing, natively or on the interpreter, in any sweep | {l} |\n"));
    // M: each ein of the entry at the configuration, > 90 %, at least 10³
    // distinguished pairs (plan §7: fewer is inconclusive).
    let at = |r: &&Row| r.entry == opts.entry && r.config == opts.at && r.dtype == opts.dtype;
    let eins: Vec<&Row> = rows.iter().filter(at).filter(|r| r.op == "ein").collect();
    let m = if eins.is_empty() {
        "not measured".to_string()
    } else if eins.iter().any(|r| r.tally.distinguished >= 1000 && r.tally.accuracy().unwrap_or(0.0) <= 0.9) {
        verdict(false).to_string()
    } else if eins.iter().all(|r| r.tally.distinguished >= 1000) {
        verdict(true).to_string()
    } else {
        "inconclusive: fewer than 10³ distinguished pairs".to_string()
    };
    let m_cells: Vec<String> = eins.iter().map(|r| format!("`{}` {} of {}", r.nest, r.tally.accuracy().map_or("—".to_string(), |a| format!("{:.1} %", 100.0 * a)), r.tally.distinguished)).collect();
    md.push_str(&format!("| M | more than 90 % of the distinguished close pairs ranked right, each `ein` of `{}` at {}, {} | {} | {m} |\n", opts.entry, opts.at, opts.dtype, m_cells.join("; ")));
    // N: every configuration of the entry swept whole, on each day.
    let whole: Vec<&Sweep> = s1.iter().chain(if one_day { &[][..] } else { s2 }).filter(|s| s.entry == opts.entry && !s.sampled).collect();
    let ratio = |s: &Sweep| s.exhaustive_seconds / s.static_seconds.max(1e-9);
    let n = if whole.is_empty() { "not measured".to_string() } else { verdict(whole.iter().all(|s| ratio(s) >= 1000.0)).to_string() };
    let n_cells: Vec<String> = whole.iter().map(|s| format!("{} {} {:.0}×", s.config, s.realization, ratio(s))).collect();
    md.push_str(&format!("| N | choosing statically at least 10³× cheaper than the exhaustive sweep of the same points, `{}` | {} | {n} |\n", opts.entry, n_cells.join("; ")));
    // O: every contraction nest of the entry at the configuration.
    let contractions: Vec<&Row> = rows.iter().filter(at).filter(|r| r.op == "ein" || r.op == "dot").collect();
    let o = if contractions.is_empty() { "not measured".to_string() } else { verdict(contractions.iter().all(|r| r.top1 <= 1.25)).to_string() };
    let o_cells: Vec<String> = contractions.iter().map(|r| format!("`{}` {:.2}", r.nest, r.top1)).collect();
    md.push_str(&format!("| O | the model's pick within 1.25× of the best measured, every contraction nest of `{}` at {}, {} | {} | {o} |\n", opts.entry, opts.at, opts.dtype, o_cells.join("; ")));
    // P: the GEMM's best point against 0.6.0's.
    let gemm: Vec<&Row> = rows.iter().filter(|r| r.entry == opts.gemm && r.config == opts.gemm_at && r.dtype == "f32" && r.op == "dot").collect();
    let p = match gemm.first().and_then(|r| r.speedup) {
        Some(s) => format!("{s:.2}× | {}", verdict(s >= 3.0)),
        None => "— | not measured".to_string(),
    };
    md.push_str(&format!("| P | the best point of the GEMM space at least 3× the 0.6.0 default, `{}` at {}, f32 | {p} |\n\n", opts.gemm, opts.gemm_at));
    md
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §3.6's rule on two synthetic days: a pair is close when its layouts
    /// differ in one decision, distinguished when both days separate it by
    /// more than twice the larger MAD with the same sign, decided when the
    /// predicted ratio lies outside 1 ± u.
    #[test]
    fn rank_counts_the_pairs_both_days_distinguish() {
        let dir = std::env::temp_dir().join(format!("presk-rank-{}", std::process::id()));
        // (name, tile, lanes, day-one median, day-two median, prediction)
        let points = [
            ("a", [1, 8], 1, 10.0, 10.0, 10.0),
            ("b", [1, 8], 4, 5.0, 5.0, 4.0),   // a–b: lanes, distinguished, right, decided
            ("c", [4, 8], 4, 6.0, 4.0, 5.0),   // b–c: tile, the sign flips between the days
            ("d", [1, 8], 8, 5.05, 5.05, 4.5), // b–d: lanes, within the noise; a–d right, decided
            ("e", [1, 16], 1, 8.0, 8.0, 11.0), // a–e: tile, distinguished, wrong, undecided
        ];
        for (day, col) in [("day1", 0), ("day2", 1)] {
            std::fs::create_dir_all(dir.join(day)).unwrap();
            let line = |k: &str, layout: Json, median: f64, pred: f64, baseline: bool| json!({ "entry": "gemm", "config": "N=1024", "op": "dot", "nest": "f/%2", "shape": "[4, 4]", "dtype": "f32", "key": k, "layout": layout, "median": median, "mad": 0.1, "predicted": pred, "baseline": baseline, "identical": true, "interp_checked": col == 0 }).to_string();
            let mut lines: Vec<String> = points.iter().map(|(k, tile, lanes, m1, m2, pred)| line(k, json!({ "tile": tile, "lanes": lanes }), if col == 0 { *m1 } else { *m2 }, *pred, false)).collect();
            // 0.6.0's point, three times slower than the best measured.
            lines.push(line("0.6.0-default", json!({ "tile": [1, 8], "lanes": 4 }), 15.0, 4.0, true));
            std::fs::write(dir.join(day).join("f__2.jsonl"), lines.join("\n")).unwrap();
            std::fs::write(dir.join(day).join("sweep.json"), json!({ "entry": "gemm", "config": "N=1024", "realization": "f32", "points": 5, "u": 0.2, "capture_seconds": 0.1, "cost_seconds": 0.1, "wall_seconds": 2.0, "interp_seconds": 1.0, "sample": null, "shard": null, "timed": true, "not_identical": 0 }).to_string()).unwrap();
        }
        let out = dir.join("ra4.md");
        assert_eq!(rank(&dir.join("day1"), &dir.join("day2"), &out, &RankOpts::default()), ExitCode::SUCCESS);
        let md = std::fs::read_to_string(&out).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        // Five close pairs (a–c, b–e, c–d, c–e, d–e differ in two decisions),
        // three distinguished, two ordered right, both decided ones right.
        assert!(md.contains("| **all** | | | | | 5 | 3 | 66.7 % | 100.0 % | 33.3 % | 0.0 % |"), "{md}");
        assert!(md.contains("| lanes | 3 | 2 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |"), "{md}");
        assert!(md.contains("| tile | 2 | 1 | 0.0 % | — | 100.0 % | 0.0 % |"), "{md}");
        // The model picks b, whose mean time 5 is the best measured, and
        // keeps d within 1 + u of it; 0.6.0's point takes 15.
        assert!(md.contains("| 1.00 | 1.00 (2) | 3.00× |"), "{md}");
        // Statement L: every point of day one checked on the interpreter
        // (the baseline too); P: 3× at 1024; the attention was not measured.
        assert!(md.contains("| 6 of 6 points of day 1 checked on the interpreter; 0 outputs differing"), "{md}");
        assert!(md.contains("| 3.00× | **holds** |"), "{md}");
        assert!(md.contains("| M |") && md.contains("| not measured |"), "{md}");
        // One sweep of 5 points: 1 s exhaustive (the interpreter's second
        // excluded) against 0.2 s static.
        assert!(md.contains("| `gemm` | N=1024 | f32 | 5 | 1.0 | 0.200 | 5 |"), "{md}");
    }

    /// Two points the model predicts equal: the pair is tied, so not right,
    /// and the model's pick is the first in the space's order, as `presk
    /// plan`'s is, not the first key.
    #[test]
    fn a_tie_is_not_right_and_the_pick_follows_the_space() {
        let dir = std::env::temp_dir().join(format!("presk-rank-tie-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let line = |k: &str, tile: [u32; 2], median: f64| json!({ "nest": "f/%2", "shape": "[4]", "dtype": "f32", "key": k, "layout": { "tile": tile }, "median": median, "mad": 0.01, "predicted": 1.0 }).to_string();
        std::fs::write(dir.join("f.jsonl"), [line("z", [1, 4], 3.0), line("a", [1, 16], 1.0)].join("\n")).unwrap();
        let out = dir.join("ra4.md");
        assert_eq!(rank(&dir, &dir, &out, &RankOpts::default()), ExitCode::SUCCESS);
        let md = std::fs::read_to_string(&out).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(md.contains("| 2 | 1 | 1 | 0.0 % | — | 100.0 % | 100.0 % |"), "{md}");
        assert!(md.contains("| 3.00 | 1.00 (2) |"), "the pick is z, three times the best: {md}");
    }

    /// A sample keeps point 0, the model's best, and a draw that does not
    /// depend on the order of the space; shards split it and all run point 0.
    #[test]
    fn a_sample_and_its_shards() {
        let pts: Vec<(Layout, Vec<Kernel>)> = (1..=20).map(|n| (Layout { lanes: Some(n), ..Layout::default() }, vec![])).collect();
        let predicted: Vec<Option<f64>> = (0..20).map(|i| Some(100.0 - i as f64)).collect();
        let opts = |top, random, shard| SweepOpts { repeat: 0, interp: false, top, random, shard };
        let s = select(&pts, &predicted, &opts(3, 4, None), 7);
        assert_eq!(s.len(), 8);
        assert!(s.starts_with(&[0]) && s.contains(&19) && s.contains(&18) && s.contains(&17), "{s:?}");
        assert_eq!(s, select(&pts, &predicted, &opts(3, 4, None), 7), "the draw is deterministic");
        assert_eq!(select(&pts, &predicted, &opts(0, 0, None), 7).len(), 20);
        let shards: Vec<Vec<usize>> = (0..3).map(|k| select(&pts, &predicted, &opts(0, 0, Some((k, 3))), 7)).collect();
        assert!(shards.iter().all(|s| s[0] == 0));
        let mut union: Vec<usize> = shards.concat();
        union.sort();
        union.dedup();
        assert_eq!(union, (0..20).collect::<Vec<_>>());
        assert_eq!(shards.iter().map(|s| s.len()).sum::<usize>(), 20 + 2, "point 0 twice more");
    }
}
