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

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use presk_kernel::cost::{Machine, planned_cost_under};
use presk_kernel::exec::{output_buf, run_kernel};
use presk_kernel::plan::{launch_kind, lower_launch, lower_launch_splits, split_as};
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

/// The entry, run once under its default layout, its launches captured.
pub struct Captured {
    pub launches: Vec<Launch>,
    pub machine: Machine,
    pub seconds: f64,
}

/// How an entry is run to capture its launches.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Capture {
    /// Natively: every launch's inputs hold the values the entry computes,
    /// which timing a point on them needs (`presk autotune`).
    Run,
    /// The host program only (0.8.0 step 1, D65): every launch is lowered
    /// and recorded, and no kernel is compiled or run, so the inputs of a
    /// launch after the first are zeros of their shapes. Choosing statically
    /// needs the launches' shapes, not their values (`presk plan`, `presk
    /// reprice`, and the static side of statement N).
    Shapes,
}

/// An executor that runs nothing ([`Capture::Shapes`]): every output stays
/// the zeros it was allocated as.
struct NoRun;

impl KernelExec for NoRun {
    fn name(&self) -> &'static str {
        "shapes"
    }

    fn run(&mut self, _: usize, _: &Kernel, _: &[&Buf], _: &mut Buf) -> presk_rt::Result<()> {
        Ok(())
    }
}

/// An input of [`Capture::Shapes`], built from its shape: a tensor as zeros
/// in the realization's type, a module's parameters by its initializer on
/// `b`, which runs no kernel, so that they are zeros of their shapes too, and
/// a scalar or a key as `presk run` synthesizes it. Nothing is computed in
/// exact arithmetic.
fn zeros(b: &mut presk_kernel::KernelBackend, realize: &presk_rt::Realization, s: &presk_interp::run::Shaped) -> presk_rt::Result<presk_rt::Value> {
    use presk_interp::run::Shaped;
    use presk_rt::{Tensor, Value};
    let size = |axes: &[presk_interp::value::Axis]| axes.iter().map(|a| a.len).product::<usize>();
    Ok(match s {
        Shaped::Real(axes) => Value::Real(Tensor { axes: axes.clone(), data: Buf::zeros(realize.default, size(axes)) }),
        Shaped::Idx { bound, axes } => Value::Idx { bound: *bound, t: Tensor { axes: axes.clone(), data: Buf::I64(vec![0; size(axes)]) } },
        Shaped::Value(v) => Value::from_interp(v)?,
        Shaped::Tuple(parts) => Value::Tuple(parts.iter().map(|p| zeros(b, realize, p)).collect::<presk_rt::Result<_>>()?),
        Shaped::Params { init, dims, key } => {
            let p = presk_rt::Backend::prepare(b, *init, dims, realize)?;
            presk_rt::Backend::call(b, &p, vec![Value::Key(key.clone())])?
        }
    })
}

/// Runs `entry` once with the kernel backend capturing, natively or on the
/// host program alone, and returns the launches of the nests `nest` selects
/// (every nest when `None`).
#[allow(clippy::too_many_arguments)]
pub fn capture(file: &Path, entry: &str, dims: &[String], consts: &[String], plan: &presk_rt::Plan, seed: u64, nest: Option<&str>, mode: Capture) -> Result<Captured, ExitCode> {
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
    let mut b = match mode {
        Capture::Run => presk_cranelift::native_backend(&core, &prog, realize, seed).map_err(|e| fail(e.message))?,
        Capture::Shapes => {
            // The native backend's defaults, its executor aside.
            let mut b = presk_kernel::KernelBackend::with_exec(&core, &prog, realize, seed, Box::new(NoRun));
            b.default_layout.lanes = None;
            b
        }
    };
    // A realization with integer weights quantizes drawn values; zeros would
    // have no scale.
    let (func, fdims, args) = if mode == Capture::Shapes && realize.quant.is_empty() {
        let (func, fdims, shaped) = presk_interp::run::prepare_shapes(&core, &prog, &opts).map_err(|e| fail(e.message))?;
        let args = shaped.iter().map(|(_, s)| zeros(&mut b, realize, s)).collect::<presk_rt::Result<Vec<_>>>().map_err(|e| fail(e.message))?;
        (func, fdims, args)
    } else {
        let (func, fdims, inputs) = presk_interp::run::prepare_run(&core, &prog, &opts).map_err(|e| fail(e.message))?;
        let mut args: Vec<presk_rt::Value> = inputs.iter().map(|(_, v)| presk_rt::Value::from_interp(v).map(|v| v.realize(realize.default))).collect::<presk_rt::Result<_>>().map_err(|e| fail(e.message))?;
        realize.quantize_args(&mut args).map_err(|e| fail(e.message))?;
        (func, fdims, args)
    };
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
    // Each layout lowered once for all its splits (0.8.0 step 1).
    for b in &base {
        for (s, ks) in splits.iter().zip(lower_launch_splits(l, b, &splits, Some(workers))) {
            let p = Layout { split: Some(*s), ..b.clone() };
            if p == zero {
                continue;
            }
            if let Ok(ks) = ks {
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
        let c = planned_cost_under(k, machine, &fit.rules);
        fit.predict(&c, &group(dtype, k, c.workers))
    }).sum()
}

/// What a point's line records per kernel (a pack's copy first): the model's
/// inputs, its group and the terms of its prediction. `presk autotune` and
/// `presk reprice` both write it.
fn kernel_records(fit: Option<&LayoutFit>, machine: &Machine, dtype: presk_rt::Dtype, kernels: &[Kernel]) -> Vec<Json> {
    kernels
        .iter()
        .map(|k| {
            let c = planned_cost_under(k, machine, &fit.map(|f| f.rules).unwrap_or_default());
            let g = group(dtype, k, c.workers);
            let terms = fit.and_then(|f| f.terms(&c, &g));
            json!({ "group": g, "cost": c, "terms": terms, "bound": terms.map(|t| t.bound()) })
        })
        .collect()
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
    let captured = match capture(file, entry, dims, consts, base, seed, nest, Capture::Shapes) {
        Ok(c) => c,
        Err(c) => return c,
    };
    // The machine the model was fitted on; the one the plan is for decides
    // which worker counts are well-formed.
    let model_machine = stored.unwrap_or_else(|| captured.machine.clone());
    let t_cost = Instant::now();
    // Each launch's points the model prices, best first, and its rows.
    type Costed<'a> = (&'a Launch, Vec<(f64, Layout)>, Vec<String>);
    let mut costed: Vec<Costed> = Vec::new();
    let mut unpriced: Vec<&Launch> = Vec::new();
    let mut total_points = 0usize;
    for l in &captured.launches {
        let pts = points(l, space, captured.machine.workers);
        let space_size = pts.len();
        total_points += space_size;
        let mut scored: Vec<(f64, Layout)> = pts.into_iter().filter_map(|(p, ks)| predict(&fit, &model_machine, l.dtype, &ks).map(|t| (t, p))).collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let Some((best_t, best)) = scored.first() else {
            // Points none of which a group of the model prices: a type the
            // profile was not calibrated in (0.9.0, D82).
            if space_size > 0 {
                unpriced.push(l);
            }
            continue;
        };
        let kept = scored.iter().filter(|(t, _)| *t <= best_t * (1.0 + fit.u)).count();
        let mut rows = Vec::new();
        if cost {
            for (t, p) in &scored {
                rows.push(format!("  {:<28} {:>12.2} µs  {p}", l.nest, t * 1e6));
            }
        }
        rows.push(format!("{:<28} {:<24} {:>5} points  {:>10.2} µs  {kept:>3} within 1 + u  {best}", l.nest, shape(l), space_size, best_t * 1e6));
        costed.push((l, scored, rows));
    }
    let types: Vec<String> = {
        let mut t: Vec<String> = fit.pi.keys().filter_map(|g| g.split('+').next()).map(String::from).collect();
        t.dedup();
        t
    };
    if costed.is_empty() && !unpriced.is_empty() {
        let dtypes: Vec<String> = unpriced.iter().map(|l| l.dtype.to_string()).collect::<BTreeSet<_>>().into_iter().collect();
        eprintln!("presk plan: the profile's layout model prices no layout of `{entry}` in {}: {} was calibrated in {} only, and a plan stating no layout would be the default one", dtypes.join(", "), profile.display(), types.join(" and "));
        return ExitCode::from(2);
    }
    // One layout per nest, since a plan states one per name. A nest launched
    // at one shape takes that shape's best. At several, the first shape's
    // best may not suit another (a tile that does not divide it, condition
    // 3), and `presk run` would refuse the plan: the layout is the one the
    // model prices at every shape whose predicted time, each shape's weighed
    // by the times the entry launches it, is least (0.9.0, D82).
    let mut layout: BTreeMap<String, Layout> = BTreeMap::new();
    let mut rows: Vec<String> = Vec::new();
    let nests: Vec<&str> = {
        let mut n: Vec<&str> = Vec::new();
        for (l, _, _) in &costed {
            if !n.contains(&l.nest.as_str()) {
                n.push(&l.nest);
            }
        }
        n
    };
    let mut summary: HashMap<&str, String> = HashMap::new();
    for nest in &nests {
        // A nest the model cannot price at one of its shapes is left to the
        // default layout, which suits every shape.
        if unpriced.iter().any(|l| l.nest == *nest) {
            continue;
        }
        let group: Vec<&Costed> = costed.iter().filter(|(l, _, _)| l.nest == *nest).collect();
        if let [(_, scored, _)] = group.as_slice() {
            layout.insert(nest.to_string(), scored[0].1.clone());
            continue;
        }
        let others: Vec<HashMap<String, f64>> = group[1..].iter().map(|(_, s, _)| s.iter().map(|(t, p)| (p.key(), *t)).collect()).collect();
        let weigh = |ts: &[f64]| -> f64 { ts.iter().zip(&group).map(|(t, (l, _, _))| t * l.count as f64).sum() };
        let own = weigh(&group.iter().map(|(_, s, _)| s[0].0).collect::<Vec<_>>());
        let mut chosen: Option<(f64, &Layout)> = None;
        for (t0, p) in &group[0].1 {
            let key = p.key();
            let ts = std::iter::once(Some(*t0)).chain(others.iter().map(|m| m.get(&key).copied())).collect::<Option<Vec<f64>>>();
            let Some(ts) = ts else { continue };
            let total = weigh(&ts);
            if chosen.is_none_or(|(b, _)| total < b) {
                chosen = Some((total, p));
            }
        }
        let counts = group.iter().map(|(l, _, _)| l.count.to_string()).collect::<Vec<_>>().join(" + ");
        // A line for a nest whose shapes do not all choose the layout it
        // takes.
        match chosen {
            Some((total, p)) => {
                layout.insert(nest.to_string(), p.clone());
                if group.iter().any(|(_, s, _)| s[0].1 != *p) {
                    summary.insert(nest, format!("= {nest} at {} shapes, {counts} launches: {p}; {:.2} µs over them, {:.2} µs at each shape's own best", group.len(), total * 1e6, own * 1e6));
                }
            }
            None => {
                summary.insert(nest, format!("= {nest} at {} shapes: no layout the model prices at every one, left to the default", group.len()));
            }
        }
    }
    for (i, (l, _, r)) in costed.iter().enumerate() {
        rows.extend(r.iter().cloned());
        let last = !costed[i + 1..].iter().any(|(m, _, _)| m.nest == l.nest);
        if last && let Some(line) = summary.get(l.nest.as_str()) {
            rows.push(line.clone());
        }
    }
    let cost_seconds = t_cost.elapsed().as_secs_f64();
    println!("{:<28} {:<24} {:>12} {:>13}  {:>17}  layout", "nest", "operands", "space", "predicted", "undecided");
    for r in &rows {
        println!("{r}");
    }
    let all_nests = captured.launches.iter().map(|l| l.nest.as_str()).collect::<BTreeSet<_>>().len();
    let shapes = if all_nests == captured.launches.len() { String::new() } else { format!(" at {} shapes", captured.launches.len()) };
    println!("-- {all_nests} nests{shapes}, {total_points} points costed in {cost_seconds:.3} s, the entry captured in {:.3} s; u = {:.3}", captured.seconds, fit.u);
    if !unpriced.is_empty() {
        let mut names: Vec<String> = Vec::new();
        for l in &unpriced {
            let n = unpriced.iter().filter(|m| m.nest == l.nest).count();
            let name = if n > 1 { format!("{} ({}, {n} shapes)", l.nest, l.dtype) } else { format!("{} ({})", l.nest, l.dtype) };
            if !names.contains(&name) {
                names.push(name);
            }
        }
        println!("-- left to the default layout, the profile being calibrated in {} only: {}", types.join(" and "), names.join(", "));
    }
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
    let captured = match capture(file, entry, dims, consts, base, seed, nest, Capture::Run) {
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
            let per_kernel = kernel_records(fit.as_ref().map(|(f, _)| f), &model_machine, l.dtype, kernels);
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
    let wall_seconds = t0.elapsed().as_secs_f64();
    // What choosing statically takes to capture the same launches: the host
    // program alone (D65). Statement N's static side is this and the
    // costing; it is timed after the sweep's wall time, which it is not part of.
    let static_capture = match capture(file, entry, dims, consts, base, seed, nest, Capture::Shapes) {
        Ok(c) => c,
        Err(c) => return c,
    };
    let sample = (opts.top > 0 || opts.random > 0).then(|| json!({ "top": opts.top, "random": opts.random, "seed": seed }));
    let summary = json!({
        "entry": entry, "config": config, "consts": consts, "dims": dims, "realization": base.realization.label(),
        "space": if space == Space::Small { "small" } else { "full" }, "repeat": opts.repeat, "timed": opts.repeat > 0,
        "sample": sample, "shard": opts.shard.map(|(k, n)| json!([k, n])),
        "points": total, "baselines": baselines, "sweep_seconds": sweep_seconds - interp_seconds, "cost_seconds": cost_seconds, "interp_seconds": interp_seconds,
        "capture_seconds": captured.seconds, "static_capture_seconds": static_capture.seconds, "wall_seconds": wall_seconds, "machine": captured.machine,
        "u": fit.as_ref().map(|(f, _)| f.u), "interp_checked": opts.interp, "not_identical": failures,
    });
    let _ = std::fs::write(out.join("sweep.json"), serde_json::to_string_pretty(&summary).unwrap_or_default());
    let verb = if opts.repeat > 0 { "measured" } else { "run once, untimed," };
    println!("-- {total} points of {} nests {verb} in {:.1} s, costed in {cost_seconds:.3} s; every output bit-identical to point 0{}: {}", captured.launches.len(), sweep_seconds - interp_seconds, if opts.interp { ", natively and on the kernel interpreter" } else { ", natively" }, if failures == 0 { "yes".to_string() } else { format!("NO, {failures} points differ") });
    if failures > 0 { ExitCode::from(1) } else { ExitCode::SUCCESS }
}

// ----- presk reprice ----------------------------------------------------------

/// A JSON object whose fields are kept as they were written: `presk reprice`
/// replaces some and copies the others byte for byte, where parsing a float
/// and writing it again need not give back the digits read.
type Fields = BTreeMap<String, Box<serde_json::value::RawValue>>;

/// A launch's kernels under a layout, or why it does not lower.
type Lowered = Result<Vec<Kernel>, String>;

fn raw(v: &Json) -> Box<serde_json::value::RawValue> {
    serde_json::value::to_raw_value(v).expect("a JSON value serializes")
}

/// The directories under `dir` holding a `sweep.json`, relative to `root`.
fn sweep_dirs(root: &Path, dir: &Path, out: &mut Vec<std::path::PathBuf>) -> Result<(), String> {
    let mut entries: Vec<_> = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?.flatten().map(|e| e.path()).collect();
    entries.sort();
    if entries.iter().any(|p| p.file_name().is_some_and(|n| n == "sweep.json")) {
        out.push(dir.strip_prefix(root).unwrap_or(dir).to_path_buf());
    }
    for p in entries.iter().filter(|p| p.is_dir()) {
        sweep_dirs(root, p, out)?;
    }
    Ok(())
}

/// Re-prices recorded measurements under another profile (plan 0.8.0 §3.4,
/// step 1). For each configuration of `entry` under `day` — a directory with
/// the `sweep.json` of a `presk autotune` — the entry is captured once at the
/// constants, dims and realization the sweep records, and every recorded
/// point is lowered again from its layout, against the worker count the
/// sweep ran with, and priced under `profile`. Its line is written to the
/// same place under `out` with its prediction and its per-kernel records
/// (cost, group, terms, bound) replaced; every other field — the times, the
/// MADs, whether its bits matched — is copied as written. `sweep.json` is
/// copied with `u` replaced and a `repriced` record added, and the day's
/// `day.txt` as it is. Nothing is measured.
pub fn reprice(file: &Path, entry: &str, day: &Path, profile: &Path, out: &Path, seed: u64) -> ExitCode {
    let t0 = Instant::now();
    let fail = |m: String| {
        eprintln!("presk reprice: {m}");
        ExitCode::from(2)
    };
    let (fit, stored) = match load_fit(Some(profile)) {
        Ok(Some(x)) => x,
        Ok(None) => unreachable!("a profile was given"),
        Err(c) => return c,
    };
    if let (Ok(a), Ok(b)) = (day.canonicalize(), out.canonicalize())
        && a == b
    {
        return fail(format!("--out {} is the day itself; a repriced day goes elsewhere", out.display()));
    }
    let mut dirs = Vec::new();
    if let Err(m) = sweep_dirs(day, day, &mut dirs) {
        return fail(m);
    }
    let (mut configs, mut total) = (0usize, 0usize);
    for rel in &dirs {
        let (src, dst) = (day.join(rel), out.join(rel));
        let sweep_text = match std::fs::read_to_string(src.join("sweep.json")) {
            Ok(t) => t,
            Err(e) => return fail(format!("{}: {e}", src.join("sweep.json").display())),
        };
        let (sweep, mut sweep_fields): (Json, Fields) = match (serde_json::from_str(&sweep_text), serde_json::from_str(&sweep_text)) {
            (Ok(v), Ok(f)) => (v, f),
            (Err(e), _) | (_, Err(e)) => return fail(format!("{}: {e}", src.join("sweep.json").display())),
        };
        if sweep["entry"] != entry {
            continue;
        }
        let strings = |k: &str| sweep[k].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect::<Vec<_>>()).unwrap_or_default();
        let (consts, dims) = (strings("consts"), strings("dims"));
        let Some(realization) = sweep["realization"].as_str() else { return fail(format!("{}: no realization recorded", src.display())) };
        // The machine the sweep ran on: its worker count decided which splits
        // were well-formed, and it priced the points when the profile has no
        // machine of its own.
        let machine: Machine = match serde_json::from_value(sweep["machine"].clone()) {
            Ok(m) => m,
            Err(e) => return fail(format!("{}: the machine the sweep ran on: {e}", src.join("sweep.json").display())),
        };
        let plan = presk_rt::Plan::from_realization(super::parse_realize(realization));
        let captured = match capture(file, entry, &dims, &consts, &plan, seed, None, Capture::Shapes) {
            Ok(c) => c,
            Err(c) => return c,
        };
        let model_machine = stored.clone().unwrap_or_else(|| machine.clone());
        let launches: BTreeMap<(String, String, String), &Launch> = captured.launches.iter().map(|l| ((l.nest.clone(), shape(l), l.dtype.name().to_string()), l)).collect();
        if let Err(e) = std::fs::create_dir_all(&dst) {
            return fail(format!("{}: {e}", dst.display()));
        }
        let tc = Instant::now();
        let mut files: Vec<_> = std::fs::read_dir(&src).map(|r| r.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "jsonl")).collect()).unwrap_or_default();
        files.sort();
        let mut points = 0usize;
        // Each layout lowered once for every split recorded with it, by nest,
        // shape, type and the layout's key without its split.
        let mut unsplit: BTreeMap<(String, String, String, String), Lowered> = BTreeMap::new();
        for path in &files {
            let text = match std::fs::read_to_string(path) {
                Ok(t) => t,
                Err(e) => return fail(format!("{}: {e}", path.display())),
            };
            let mut lines = Vec::new();
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                let (v, mut fields): (Json, Fields) = match (serde_json::from_str(line), serde_json::from_str(line)) {
                    (Ok(v), Ok(f)) => (v, f),
                    (Err(e), _) | (_, Err(e)) => return fail(format!("{}: {e}", path.display())),
                };
                let s = |k: &str| v[k].as_str().unwrap_or("").to_string();
                let key = (s("nest"), s("shape"), s("dtype"));
                let Some(l) = launches.get(&key) else {
                    return fail(format!("{}: the entry, captured at {} in {realization}, launches no nest `{}` at {} in {}", path.display(), sweep["config"].as_str().unwrap_or(""), key.0, key.1, key.2));
                };
                let layout = match Layout::from_json(&v["layout"]) {
                    Ok(x) => x,
                    Err(m) => return fail(format!("{}: `{}`: {m}", path.display(), s("key"))),
                };
                let kernels = match layout.split {
                    // 0.6.0's point: the split left to the executor.
                    None => lower_launch(l, &layout, Some(machine.workers)),
                    Some(split) => {
                        let base = Layout { split: Some(Split::None), ..layout.clone() };
                        let once = unsplit.entry((key.0.clone(), key.1.clone(), key.2.clone(), base.key())).or_insert_with(|| lower_launch(l, &base, Some(machine.workers)));
                        once.as_ref().map_err(|e| e.clone()).and_then(|ks| split_as(ks, split, Some(machine.workers)))
                    }
                };
                let kernels = match kernels {
                    Ok(k) => k,
                    Err(m) => return fail(format!("{}: `{}` under {layout} does not lower: {m}", path.display(), key.0)),
                };
                fields.insert("predicted".into(), raw(&json!(predict(&fit, &model_machine, l.dtype, &kernels))));
                fields.insert("kernels".into(), raw(&Json::Array(kernel_records(Some(&fit), &model_machine, l.dtype, &kernels))));
                lines.push(serde_json::to_string(&fields).expect("raw fields serialize"));
                points += 1;
            }
            let to = dst.join(path.file_name().expect("a file"));
            if let Err(e) = std::fs::write(&to, lines.join("\n") + "\n") {
                return fail(format!("{}: {e}", to.display()));
            }
        }
        let cost_seconds = tc.elapsed().as_secs_f64();
        sweep_fields.insert("u".into(), raw(&json!(fit.u)));
        sweep_fields.insert("repriced".into(), raw(&json!({ "profile": profile.display().to_string(), "u": fit.u, "points": points, "capture_seconds": captured.seconds, "cost_seconds": cost_seconds })));
        if let Err(e) = std::fs::write(dst.join("sweep.json"), serde_json::to_string_pretty(&sweep_fields).expect("raw fields serialize")) {
            return fail(format!("{}: {e}", dst.display()));
        }
        println!("{:<40} {:>5} points re-priced in {cost_seconds:.3} s, the entry captured in {:.3} s", rel.display(), points, captured.seconds);
        configs += 1;
        total += points;
    }
    if configs == 0 {
        return fail(format!("{} holds no sweep of `{entry}`", day.display()));
    }
    if let Ok(t) = std::fs::read_to_string(day.join("day.txt"))
        && let Err(e) = std::fs::write(out.join("day.txt"), t)
    {
        return fail(format!("{}: {e}", out.display()));
    }
    println!("-- {total} points of {configs} configurations re-priced under {}, u = {:.3}, in {:.1} s; times copied", profile.display(), fit.u, t0.elapsed().as_secs_f64());
    ExitCode::SUCCESS
}

// ----- presk refit --recost -----------------------------------------------------

/// The type and the product a calibration sample's signature names: `kernel
/// dot (%0 : f32[16, 4096], %1 : f32[4096, 4096]) -> …` is `f32` and `I =
/// 16, K = 4096, J = 4096`.
fn product_of(signature: &str) -> Option<(String, [usize; 3])> {
    let operand = |name: &str| -> Option<(String, Vec<usize>)> {
        let rest = &signature[signature.find(&format!("{name} : "))? + name.len() + 3..];
        let (dtype, rest) = rest.split_once('[')?;
        let dims = rest[..rest.find(']')?].split(", ").map(|d| d.parse().ok()).collect::<Option<Vec<usize>>>()?;
        Some((dtype.to_string(), dims))
    };
    let ((dtype, a), (_, b)) = (operand("%0")?, operand("%1")?);
    match (a.as_slice(), b.as_slice()) {
        ([i, k], [k2, j]) if k == k2 => Some((dtype, [*i, *k, *j])),
        _ => None,
    }
}

/// The calibration's samples costed again (plan 0.8.0 step 4, 4b): each
/// sample's layout lowered at the product its signature names, on `entry`
/// of `file` (`experiments/matmul.psk`, whose dimensions are `I`, `K` and
/// `J`), and costed on `machine` — the profile's own description, not the
/// one this process runs on — under `rules`. Its time, layout and group are
/// kept, the kernel lowered checked against its signature and its group.
/// Nothing is measured; the entry is captured once per product, its shapes
/// only.
pub fn recost(samples: &[presk_opt::profile::PlannedSample], file: &Path, entry: &str, machine: &Machine, rules: &presk_kernel::cost::CostRules) -> Result<Vec<presk_opt::profile::PlannedSample>, String> {
    let mut launches: BTreeMap<String, Launch> = BTreeMap::new();
    let mut out = Vec::with_capacity(samples.len());
    for s in samples {
        if !launches.contains_key(&s.signature) {
            let Some((dtype, [i, k, j])) = product_of(&s.signature) else { return Err(format!("`{}` is not a product's signature", s.signature)) };
            let realize = presk_rt::Realization::parse(&dtype).map_err(|m| format!("`{}`: {m}", s.signature))?;
            let dims = [format!("I={i}"), format!("K={k}"), format!("J={j}")];
            let captured = capture(file, entry, &dims, &[], &presk_rt::Plan::from_realization(realize), 0, None, Capture::Shapes).map_err(|_| format!("{} could not be captured at {}", file.display(), dims.join(" ")))?;
            let mut products = captured.launches.into_iter().filter(|l| matches!(launch_kind(l), NestKind::Contraction(_)));
            let (Some(l), None) = (products.next(), products.next()) else { return Err(format!("`{entry}` of {} must launch one contraction", file.display())) };
            launches.insert(s.signature.clone(), l);
        }
        let l = &launches[&s.signature];
        let layout = Layout::from_key(&s.layout)?;
        let kernels = lower_launch(l, &layout, Some(machine.workers)).map_err(|m| format!("`{}` under {layout} does not lower: {m}", s.signature))?;
        let k = kernels.last().expect("a launch lowers to a kernel");
        let signature = k.to_string().lines().next().unwrap_or("").trim_end_matches(" {").to_string();
        if signature != s.signature {
            return Err(format!("{layout} lowers to `{signature}`, and the sample was `{}`", s.signature));
        }
        let cost = planned_cost_under(k, machine, rules);
        let g = group(l.dtype, k, cost.workers);
        if g != s.group {
            return Err(format!("`{}` under {layout} is in the group {g}, and the sample was in {}", s.signature, s.group));
        }
        out.push(presk_opt::profile::PlannedSample { cost, ..s.clone() });
    }
    Ok(out)
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
    /// Its timed launches, added up.
    timed_seconds: f64,
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
/// choosing statically takes — capturing the entry's launches, then
/// lowering and costing every point, which is what `presk plan` does — with
/// what the exhaustive sweep takes: the whole of `presk autotune` but the
/// interpreter's check. Since 0.8.0 step 1 (D65) the capture is of the
/// shapes alone, on the host program; a sweep recorded before has only the
/// capture that runs the entry, and is costed with it, as 0.7.0 was.
struct Sweep {
    entry: String,
    config: String,
    realization: String,
    sampled: bool,
    points: u64,
    static_seconds: f64,
    /// The static side with the capture that runs the entry, when the
    /// static side proper is the capture of shapes (D65).
    run_static_seconds: Option<f64>,
    exhaustive_seconds: f64,
    /// The points' part of the exhaustive side: lowered already, each
    /// compiled, launched `repeat + 1` times and its bits compared.
    sweep_seconds: f64,
    repeat: u64,
    not_identical: u64,
    u: Option<f64>,
}

/// A configuration's sweep on one day, as statement Q counts a point's
/// measured time (plan 0.8.0 §3.6, question 72): its launches, the warm-up
/// and the timed ones, each at the point's median, and its time outside
/// them (compiling it, comparing its bits). The sweep does not record the
/// second point by point: it is the sweep's time less its lines' launches,
/// over its lines, as the dry run estimates it (D72), so that the nests'
/// points and 0.6.0's add up to the sweep's time.
struct Timing {
    /// Statement N's static side, the whole configuration's.
    static_seconds: f64,
    launches: f64,
    outside: f64,
    sampled: bool,
}

/// Each configuration a day swept, keyed by (entry, configuration, type).
fn timings(day: &Day, sweeps: &[Sweep]) -> BTreeMap<(String, String, String), Timing> {
    sweeps
        .iter()
        .map(|s| {
            let launches = (s.repeat + 1) as f64;
            let lines: Vec<&Point> = day.iter().filter(|((_, _, t), n)| n.entry == s.entry && n.config == s.config && *t == s.realization).flat_map(|(_, n)| n.points.values().chain(&n.baseline)).collect();
            let kernels: f64 = lines.iter().map(|p| launches * p.median).sum();
            let outside = if lines.is_empty() { 0.0 } else { (s.sweep_seconds - kernels) / lines.len() as f64 };
            ((s.entry.clone(), s.config.clone(), s.realization.clone()), Timing { static_seconds: s.static_seconds, launches, outside, sampled: s.sampled })
        })
        .collect()
}

/// Statement Q on one contraction nest (plan 0.8.0 §3.6): measure only the
/// candidates, the points predicted within 1 + u of the model's best.
struct Hybrid {
    candidates: usize,
    /// The best candidate over the best point measured, the mean of the
    /// two days' medians.
    regret: f64,
    days: Vec<HybridDay>,
}

/// What the candidates cost on one day.
struct HybridDay {
    static_seconds: f64,
    /// Their measured time, and the nest's exhaustive sweep (`Timing`).
    candidates: f64,
    exhaustive: f64,
    /// D71's count: the candidates' timed launches over the nest's.
    share: f64,
}

impl HybridDay {
    /// Question 72's count: the static side, charged whole to the nest, and
    /// the candidates' measured time, over the exhaustive sweep of the
    /// nest's points.
    fn cost(&self) -> f64 {
        (self.static_seconds + self.candidates) / self.exhaustive
    }
}

impl Hybrid {
    fn new(keys: &[&String], kept: &[usize], regret: f64, days: &[(&Timing, &BTreeMap<String, Point>)]) -> Hybrid {
        let days = days
            .iter()
            .map(|(t, p)| {
                let time = |i: &usize| t.launches * p[keys[*i]].median + t.outside;
                let timed = |i: &usize| p[keys[*i]].timed_seconds;
                let all: Vec<usize> = (0..keys.len()).collect();
                HybridDay {
                    static_seconds: t.static_seconds,
                    candidates: kept.iter().map(time).sum(),
                    exhaustive: all.iter().map(time).sum(),
                    share: kept.iter().map(timed).sum::<f64>() / all.iter().map(timed).sum::<f64>(),
                }
            })
            .collect();
        Hybrid { candidates: kept.len(), regret, days }
    }
    /// The costlier day's.
    fn cost(&self) -> f64 {
        self.days.iter().map(HybridDay::cost).fold(0.0, f64::max)
    }
    fn share(&self) -> f64 {
        self.days.iter().map(|d| d.share).fold(0.0, f64::max)
    }
    fn holds(&self) -> bool {
        self.regret <= 1.05 && self.cost() <= 0.10
    }
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
                    static_seconds: s["static_capture_seconds"].as_f64().unwrap_or(f("capture_seconds")) + f("cost_seconds"),
                    run_static_seconds: s["static_capture_seconds"].as_f64().map(|_| f("capture_seconds") + f("cost_seconds")),
                    exhaustive_seconds: f("wall_seconds") - f("interp_seconds"),
                    sweep_seconds: f("sweep_seconds"),
                    repeat: s["repeat"].as_u64().unwrap_or(0),
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
                let point = Point { layout, order, median: v["median"].as_f64().unwrap_or(0.0), mad: v["mad"].as_f64().unwrap_or(0.0), timed_seconds: v["seconds"].as_array().map_or(0.0, |a| a.iter().filter_map(Json::as_f64).sum()), predicted: v["predicted"].as_f64(), identical: v["identical"] != Json::Bool(false), interp: v["interp_checked"] == Json::Bool(true) };
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
/// interpreter's check read from `check` when it was run apart. Plan
/// 0.8.0's Q, on every contraction nest of every configuration swept whole,
/// is decided on days that record the capture of shapes (D65), and on
/// earlier ones when `statement_q` asks.
pub struct RankOpts {
    pub check: Option<std::path::PathBuf>,
    /// A tag every day must have been measured at or after (plan 0.8.0 §3.4).
    pub frozen: Option<String>,
    pub entry: String,
    pub at: String,
    pub dtype: String,
    pub gemm: String,
    pub gemm_at: String,
    pub statement_q: bool,
}

impl Default for RankOpts {
    fn default() -> RankOpts {
        RankOpts { check: None, frozen: None, entry: "SelfAttention.apply".into(), at: "B=16,T=256".into(), dtype: "f32".into(), gemm: "gemm".into(), gemm_at: "N=1024".into(), statement_q: false }
    }
}

/// Plan 0.8.0's freeze (§3.4): every commit the day's `day.txt` names — one
/// per run of `experiments/ra4.sh` into it — descends from `tag` in the
/// repository at `repo`, and none is marked `-dirty`, a working tree that
/// differed from its commit.
fn check_frozen(repo: &Path, day: &Path, tag: &str) -> Result<(), String> {
    let text = std::fs::read_to_string(day.join("day.txt")).map_err(|e| format!("{}: {e}: a day records the commit it ran on in day.txt", day.join("day.txt").display()))?;
    let is_commit = |l: &str| {
        let hash = l.strip_suffix("-dirty").unwrap_or(l);
        (7..=40).contains(&hash.len()) && hash.bytes().all(|b| b.is_ascii_hexdigit())
    };
    let commits: Vec<&str> = text.lines().map(str::trim).filter(|l| is_commit(l)).collect();
    if commits.is_empty() {
        return Err(format!("{}: no commit recorded", day.join("day.txt").display()));
    }
    for c in commits {
        if let Some(hash) = c.strip_suffix("-dirty") {
            return Err(format!("{} ran on {hash} with uncommitted changes", day.display()));
        }
        let run = std::process::Command::new("git").arg("-C").arg(repo).args(["merge-base", "--is-ancestor", tag, c]).output().map_err(|e| format!("git: {e}"))?;
        match run.status.code() {
            Some(0) => {}
            Some(1) => return Err(format!("{} ran on {c}, which does not descend from {tag}", day.display())),
            _ => return Err(format!("{} ran on {c}: {}", day.display(), String::from_utf8_lossy(&run.stderr).trim())),
        }
    }
    Ok(())
}

/// One nest's results, kept for the statements.
struct Row {
    entry: String,
    config: String,
    op: String,
    nest: String,
    dtype: String,
    points: usize,
    tally: Tally,
    top1: f64,
    speedup: Option<f64>,
    /// Statement Q, on a contraction nest of a configuration swept whole.
    q: Option<Hybrid>,
}

fn verdict(holds: bool) -> &'static str {
    if holds { "**holds**" } else { "**fails**" }
}

pub fn rank(day1: &Path, day2: &Path, out: &Path, opts: &RankOpts) -> ExitCode {
    if let Some(tag) = &opts.frozen {
        for d in [day1, day2] {
            if let Err(m) = check_frozen(Path::new("."), d, tag) {
                eprintln!("presk rank --frozen {tag}: {m}");
                return ExitCode::from(1);
            }
        }
    }
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
    // Days recorded before the capture of shapes (0.7.0's) keep 0.7.0's
    // decision-cost table, and decide Q only when asked.
    let shapes = s1.iter().chain(&s2).any(|s| s.run_static_seconds.is_some());
    let (t1, t2) = (timings(&d1, &s1), timings(&d2, &s2));
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
        let config = (n1.entry.clone(), n1.config.clone(), dtype.clone());
        let q = match (t1.get(&config), t2.get(&config)) {
            (Some(a), Some(b)) if !a.sampled && !b.sampled && (n1.op == "ein" || n1.op == "dot") => Some(Hybrid::new(&keys, &kept, topk, &[(a, p1), (b, p2)][..if one_day { 1 } else { 2 }])),
            _ => None,
        };
        let speedup = match (&n1.baseline, &n2.baseline) {
            (Some(b1), Some(b2)) if best > 0.0 => Some((b1.median + b2.median) / 2.0 / best),
            _ => None,
        };
        let vs = speedup.map_or("—".to_string(), |s| format!("{s:.2}×"));
        md.push_str(&format!("| `{nest}` | {} | {shape} | {dtype} | {} | {} | {rho:.3} | {top1:.2} | {topk:.2} ({}) | {vs} |\n", n1.config, keys.len(), total.cells(), kept.len()));
        per_nest_classes.push((nest.clone(), n1.config.clone(), dtype.clone(), nest_classes));
        rows.push(Row { entry: n1.entry.clone(), config: n1.config.clone(), op: n1.op.clone(), nest: nest.clone(), dtype: dtype.clone(), points: keys.len(), tally: total, top1, speedup, q });
    }
    md.push_str(&format!("| **all** | | | | | {} | | | | |\n\n", all.cells()));
    md.push_str(&statements(opts, &rows, &d1, &s1, &s2, check.as_ref(), one_day, shapes || opts.statement_q));
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
    if shapes {
        md.push_str("\n## Decision cost\n\nStatic: the entry's launches captured from their shapes, on the host program alone (no input drawn, no kernel compiled or run), then every point lowered and costed (`presk plan`; plan 0.8.0 §3.6, D65). Exhaustive: every point lowered, compiled and timed (`presk autotune`, without the interpreter's check). *Entry run*: the static side with the capture that runs the entry instead, as 0.7.0 measured it. A sampled sweep is marked: its exhaustive side is not the whole space.\n\n| day | entry | config | type | points | exhaustive (s) | static (s) | ratio | static, entry run (s) | ratio, entry run |\n|---|---|---|---|---|---|---|---|---|---|\n");
    } else {
        md.push_str("\n## Decision cost\n\nStatic: the entry run once to capture its launches, then every point lowered and costed (`presk plan`). Exhaustive: every point lowered, compiled and timed (`presk autotune`, without the interpreter's check). A sampled sweep is marked: its exhaustive side is not the whole space.\n\n| day | entry | config | type | points | exhaustive (s) | static (s) | ratio |\n|---|---|---|---|---|---|---|---|\n");
    }
    for (name, sweeps) in [(day1, &s1), (day2, &s2)] {
        for s in sweeps.iter() {
            md.push_str(&format!("| `{}` | `{}` | {} | {} | {}{} | {:.1} | {:.3} | {:.0} |", name.display(), s.entry, s.config, s.realization, s.points, if s.sampled { " (sample)" } else { "" }, s.exhaustive_seconds, s.static_seconds, s.exhaustive_seconds / s.static_seconds.max(1e-9)));
            if shapes {
                md.push_str(&match s.run_static_seconds {
                    Some(r) => format!(" {r:.3} | {:.0} |", s.exhaustive_seconds / r.max(1e-9)),
                    None => " — | — |".to_string(),
                });
            }
            md.push('\n');
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

/// The statements of plan 0.7.0 §3.7, decided on what the days hold, and
/// plan 0.8.0's Q when `with_q`.
#[allow(clippy::too_many_arguments)]
fn statements(opts: &RankOpts, rows: &[Row], d1: &Day, s1: &[Sweep], s2: &[Sweep], check: Option<&(Day, Vec<Sweep>)>, one_day: bool, with_q: bool) -> String {
    let mut md = String::from(if with_q { "## Statements (plan 0.7.0 §3.7; Q, plan 0.8.0 §3.6)" } else { "## Statements (plan 0.7.0 §3.7)" });
    md.push_str("\n\n| | statement | measured | verdict |\n|---|---|---|---|\n");
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
    // A configuration too slow to sweep whole is swept on a sample, and its
    // M is inconclusive whatever its pairs say (plan 0.8.0 §3.5); O is
    // marked, its best point being the sample's.
    let sampled = s1.iter().chain(s2).any(|s| s.entry == opts.entry && s.config == opts.at && s.realization == opts.dtype && s.sampled);
    let mark = if sampled { " (sample)" } else { "" };
    let m = if eins.is_empty() {
        "not measured".to_string()
    } else if sampled {
        "inconclusive: swept on a sample (plan 0.8.0 §3.5)".to_string()
    } else if eins.iter().any(|r| r.tally.distinguished >= 1000 && r.tally.accuracy().unwrap_or(0.0) <= 0.9) {
        verdict(false).to_string()
    } else if eins.iter().all(|r| r.tally.distinguished >= 1000) {
        verdict(true).to_string()
    } else {
        "inconclusive: fewer than 10³ distinguished pairs".to_string()
    };
    let m_cells: Vec<String> = eins.iter().map(|r| format!("`{}` {} of {}{}", r.nest, r.tally.accuracy().map_or("—".to_string(), |a| format!("{:.1} %", 100.0 * a)), r.tally.distinguished, mark)).collect();
    md.push_str(&format!("| M | more than 90 % of the distinguished close pairs ranked right, each `ein` of `{}` at {}, {} | {} | {m} |\n", opts.entry, opts.at, opts.dtype, m_cells.join("; ")));
    // N: every configuration of the entry swept whole, on each day.
    let whole: Vec<&Sweep> = s1.iter().chain(if one_day { &[][..] } else { s2 }).filter(|s| s.entry == opts.entry && !s.sampled).collect();
    let ratio = |s: &Sweep| s.exhaustive_seconds / s.static_seconds.max(1e-9);
    let n = if whole.is_empty() { "not measured".to_string() } else { verdict(whole.iter().all(|s| ratio(s) >= 1000.0)).to_string() };
    let n_cells: Vec<String> = whole.iter().map(|s| format!("{} {} {:.0}×{}", s.config, s.realization, ratio(s), s.run_static_seconds.map_or(String::new(), |r| format!(" ({:.0}× with the entry run)", s.exhaustive_seconds / r.max(1e-9))))).collect();
    md.push_str(&format!("| N | choosing statically at least 10³× cheaper than the exhaustive sweep of the same points, `{}` | {} | {n} |\n", opts.entry, n_cells.join("; ")));
    // O: every contraction nest of the entry at the configuration.
    let contractions: Vec<&Row> = rows.iter().filter(at).filter(|r| r.op == "ein" || r.op == "dot").collect();
    let o = if contractions.is_empty() { "not measured".to_string() } else { verdict(contractions.iter().all(|r| r.top1 <= 1.25)).to_string() };
    let o_cells: Vec<String> = contractions.iter().map(|r| format!("`{}` {:.2}{mark}", r.nest, r.top1)).collect();
    md.push_str(&format!("| O | the model's pick within 1.25× of the best measured, every contraction nest of `{}` at {}, {} | {} | {o} |\n", opts.entry, opts.at, opts.dtype, o_cells.join("; ")));
    // P: the GEMM's best point against 0.6.0's.
    let gemm: Vec<&Row> = rows.iter().filter(|r| r.entry == opts.gemm && r.config == opts.gemm_at && r.dtype == "f32" && r.op == "dot").collect();
    let p = match gemm.first().and_then(|r| r.speedup) {
        Some(s) => format!("{s:.2}× | {}", verdict(s >= 3.0)),
        None => "— | not measured".to_string(),
    };
    md.push_str(&format!("| P | the best point of the GEMM space at least 3× the 0.6.0 default, `{}` at {}, f32 | {p} |\n", opts.gemm, opts.gemm_at));
    if with_q {
        md.push_str(&statement_q(rows, one_day));
    }
    md.push('\n');
    md
}

/// Plan 0.8.0's statement Q: its row, closing the statements' table, then
/// its table per contraction nest.
fn statement_q(rows: &[Row], one_day: bool) -> String {
    let nests: Vec<(&Row, &Hybrid)> = rows.iter().filter_map(|r| r.q.as_ref().map(|h| (r, h))).collect();
    let name = |r: &Row| format!("`{}` {} {}", r.nest, r.config, r.dtype);
    let worst = |f: &dyn Fn(&Hybrid) -> f64| nests.iter().max_by(|a, b| f(a.1).total_cmp(&f(b.1)));
    let (measured, q) = match (worst(&|h| h.regret), worst(&Hybrid::cost), worst(&Hybrid::share)) {
        (Some((r1, h1)), Some((r2, h2)), Some((r3, h3))) => (
            format!("{} contraction nests: the best candidate within {:.2}× of the best measured at worst ({}); the static side and the candidates' measured time at most {:.1} % of the nest's exhaustive sweep ({}); their share of its timed launches at most {:.2} % ({}, D71's count)", nests.len(), h1.regret, name(r1), 100.0 * h2.cost(), name(r2), 100.0 * h3.share(), name(r3)),
            verdict(nests.iter().all(|(_, h)| h.holds())).to_string(),
        ),
        _ => ("—".to_string(), "not measured".to_string()),
    };
    let mut md = format!("| Q | measuring only the points predicted within 1 + u of the model's best finds one within 1.05× of the best measured, at no more than 10 % of the exhaustive sweep's time (question 72: the static side and the candidates' measured time), every contraction nest of every configuration swept whole | {measured} | {q} |\n");
    md.push_str(&format!("\n## Statement Q, per contraction nest\n\n*Candidates*: the points predicted within 1 + u of the model's best. *Regret*: the best candidate over the best point measured, the mean of the two days' medians. *Cost* (question 72): statement N's static side, the whole configuration's, charged to each nest, and the candidates' measured time, over the exhaustive sweep of the nest's points. A point's measured time is its launches, the warm-up and the timed ones, at its median, and its time outside them (compiling it, comparing its bits), which a sweep records only as a whole: the sweep's time less its points' launches, over its points, 0.6.0's included (the dry run's estimate, D72). *Kernel share* (D71's count): the candidates' timed launches over the nest's. {}\n\n| nest | config | type | points | candidates | regret | cost | static, candidates, exhaustive (s) | kernel share | verdict |\n|---|---|---|---|---|---|---|---|---|---|\n", if one_day { "One day." } else { "Each day apart, day one first." }));
    for (r, h) in &nests {
        let join = |f: &dyn Fn(&HybridDay) -> String| h.days.iter().map(f).collect::<Vec<_>>().join(" / ");
        md.push_str(&format!(
            "| `{}` | {} | {} | {} | {} | {:.2} | {} | {} | {} | {} |\n",
            r.nest,
            r.config,
            r.dtype,
            r.points,
            h.candidates,
            h.regret,
            join(&|d| format!("{:.1} %", 100.0 * d.cost())),
            join(&|d| format!("{:.2}, {:.2}, {:.1}", d.static_seconds, d.candidates, d.exhaustive)),
            join(&|d| format!("{:.2} %", 100.0 * d.share)),
            verdict(h.holds()),
        ));
    }
    md
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tutorial() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tutorial/03_attention.psk")
    }

    fn f32_plan() -> presk_rt::Plan {
        presk_rt::Plan::from_realization(presk_rt::Realization::parse("f32").unwrap())
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("presk-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// 0.8.0 step 1: a layout lowered once for all its splits gives, point
    /// for point and in the same order, the kernels one `lower_launch` per
    /// point gives (0.7.0's way); and the traffic into three caches computed
    /// in one walk is each capacity's traffic computed alone.
    #[test]
    fn a_layout_is_lowered_once_for_all_its_splits() {
        let dims = ["B=2".to_string(), "T=16".to_string()];
        let captured = capture(&tutorial(), "attend", &dims, &[], &f32_plan(), 0, None, Capture::Shapes).unwrap();
        let workers = 10;
        let mut compared = 0;
        for l in &captured.launches {
            let kind = launch_kind(l);
            let splits: Vec<Split> = [Split::None].into_iter().chain([0, 1, 2].into_iter().flat_map(|d| [5, 10].map(|p| Split::At { depth: d, workers: p }))).collect();
            let zero = point_zero(&kind);
            let mut one_by_one: Vec<(Layout, Vec<Kernel>)> = lower_launch(l, &zero, Some(workers)).ok().map(|ks| (zero.clone(), ks)).into_iter().collect();
            for b in Layout::space(&kind) {
                for s in &splits {
                    let p = Layout { split: Some(*s), ..b.clone() };
                    if p != zero
                        && let Ok(ks) = lower_launch(l, &p, Some(workers))
                    {
                        one_by_one.push((p, ks));
                    }
                }
            }
            let batched = points(l, Space::Full, workers);
            assert_eq!(batched, one_by_one, "{}", l.nest);
            for (_, ks) in &batched {
                for k in ks {
                    let caps = [64 << 10, 2 << 20, 3 << 20];
                    assert_eq!(presk_kernel::cost::traffic_at(k, caps), caps.map(|c| presk_kernel::cost::traffic(k, c)), "{}", k.name);
                }
            }
            compared += batched.len();
        }
        assert!(compared > 500, "{compared} points");
    }

    /// D65: a capture of the host program alone, on zero inputs, records the
    /// launches a native run records — the same nests at the same shapes,
    /// types and orders, their input buffers of the same kinds and lengths —
    /// on every entry of `make plan-check`, in three types.
    #[test]
    fn a_capture_of_shapes_launches_what_a_run_launches() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let entries = std::fs::read_to_string(root.join("tests/plan/entries.txt")).unwrap();
        let sig = |l: &Launch| format!("{} {:?} {:?} {:?} {:?} {:?} {:?}", l.nest, l.what, l.operands, l.dtype, l.order, l.accum, l.inputs.iter().map(|b| (std::mem::discriminant(b), b.len())).collect::<Vec<_>>());
        let mut launches = 0;
        for line in entries.lines().filter(|l| !l.trim().is_empty()) {
            let [file, entry, args] = line.splitn(3, '|').collect::<Vec<_>>()[..] else { panic!("{line}") };
            let words: Vec<&str> = args.split_whitespace().collect();
            let pick = |flag: &str| words.windows(2).filter(|w| w[0] == flag).map(|w| w[1].to_string()).collect::<Vec<_>>();
            let (dims, consts) = (pick("--dim"), pick("--const"));
            for t in ["f32", "f64", "bf16"] {
                let plan = presk_rt::Plan::from_realization(presk_rt::Realization::parse(t).unwrap());
                let run = capture(&root.join(file), entry, &dims, &consts, &plan, 0, None, Capture::Run).unwrap();
                let shapes = capture(&root.join(file), entry, &dims, &consts, &plan, 0, None, Capture::Shapes).unwrap();
                assert_eq!(shapes.launches.iter().map(sig).collect::<Vec<_>>(), run.launches.iter().map(sig).collect::<Vec<_>>(), "{entry} in {t}");
                launches += run.launches.len();
            }
        }
        assert!(launches > 100, "{launches} launches compared");
    }

    /// A profile whose layout model prices every group of `lanes` × `workers`,
    /// its rates scaled by `speed`.
    fn profile(path: &Path, speed: f64, u: f64) {
        let n = Machine::detect().workers;
        let groups: serde_json::Map<String, Json> = [1, 2, 4, 8, 16].iter().flat_map(|lanes| (1..=n).map(move |w| (group_key("f32", *lanes, w), json!(1e9 * speed * *lanes as f64)))).collect();
        let layout = json!({ "lambda": 1e-6, "spawn": 2e-5, "beta_l2": 1e11 * speed, "beta_l3": 5e10 * speed, "beta_mem": 2e10 * speed, "pi": groups, "u": u, "spearman": 0.9, "samples": [] });
        let p = json!({ "version": 1, "dtype": "f32", "machine": "test", "pi": 1e9, "beta": 1e10, "lambda": 1e-6, "spearman": { "counts": 0.0, "work": 0.0, "roofline": 0.0 }, "kernels": [], "layout": layout });
        std::fs::write(path, p.to_string()).unwrap();
    }

    /// `presk reprice` under the profile a sweep was priced with writes the
    /// sweep's lines back byte for byte; under another it replaces the
    /// predictions and the per-kernel records and copies every other field,
    /// the times first, as written.
    #[test]
    fn reprice_copies_the_times_and_replaces_the_predictions() {
        let dir = scratch("reprice");
        let (p1, p2) = (dir.join("p1.json"), dir.join("p2.json"));
        profile(&p1, 1.0, 0.2);
        profile(&p2, 3.0, 0.3);
        let day = dir.join("day");
        let dims = ["B=2".to_string(), "T=16".to_string()];
        let opts = SweepOpts { repeat: 1, interp: false, top: 0, random: 0, shard: None };
        assert_eq!(autotune(&tutorial(), "attend", &dims, &[], &f32_plan(), Some(&p1), None, Space::Small, &opts, &day.join("B2_T16_f32"), 0), ExitCode::SUCCESS);
        std::fs::write(day.join("day.txt"), "a day\n").unwrap();
        let (same, other) = (dir.join("same"), dir.join("other"));
        assert_eq!(reprice(&tutorial(), "attend", &day, &p1, &same, 0), ExitCode::SUCCESS);
        assert_eq!(reprice(&tutorial(), "attend", &day, &p2, &other, 0), ExitCode::SUCCESS);
        let files: Vec<_> = std::fs::read_dir(day.join("B2_T16_f32")).unwrap().flatten().map(|e| e.file_name()).filter(|n| n.to_string_lossy().ends_with(".jsonl")).collect();
        assert!(!files.is_empty());
        let (mut lines, mut moved) = (0, 0);
        for f in &files {
            let rec = std::fs::read_to_string(day.join("B2_T16_f32").join(f)).unwrap();
            assert_eq!(std::fs::read_to_string(same.join("B2_T16_f32").join(f)).unwrap(), rec, "{f:?}: the same profile, the same lines");
            for (a, b) in rec.lines().zip(std::fs::read_to_string(other.join("B2_T16_f32").join(f)).unwrap().lines()) {
                let (a, b): (Fields, Fields) = (serde_json::from_str(a).unwrap(), serde_json::from_str(b).unwrap());
                assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
                for (k, v) in &a {
                    if k != "predicted" && k != "kernels" {
                        assert_eq!(v.get(), b[k].get(), "{f:?}: `{k}` copied as written");
                    }
                }
                moved += (a["predicted"].get() != b["predicted"].get()) as usize;
                lines += 1;
            }
        }
        assert_eq!(moved, lines, "every prediction moves with the rates");
        let sweep = |d: &Path| serde_json::from_str::<Json>(&std::fs::read_to_string(d.join("B2_T16_f32/sweep.json")).unwrap()).unwrap();
        assert_eq!((sweep(&day)["u"].as_f64(), sweep(&other)["u"].as_f64()), (Some(0.2), Some(0.3)));
        assert_eq!(sweep(&other)["repriced"]["profile"], json!(p2.display().to_string()));
        assert_eq!(sweep(&other)["sweep_seconds"], sweep(&day)["sweep_seconds"]);
        assert_eq!(std::fs::read_to_string(other.join("day.txt")).unwrap(), "a day\n");
        // The day itself is never overwritten, and an entry it did not sweep is refused.
        assert_eq!(reprice(&tutorial(), "attend", &day, &p2, &day, 0), ExitCode::from(2));
        assert_eq!(reprice(&tutorial(), "scores", &day, &p2, &dir.join("none"), 0), ExitCode::from(2));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// `presk rank --frozen TAG`: a day ran at the tag or after it, on a
    /// clean working tree, or it is refused.
    #[test]
    fn a_frozen_rank_refuses_a_day_measured_before_the_tag() {
        let dir = scratch("frozen");
        let git = |args: &[&str]| {
            let o = std::process::Command::new("git").arg("-C").arg(&dir).args(["-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]).args(args).output().unwrap();
            assert!(o.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&o.stderr));
            String::from_utf8(o.stdout).unwrap().trim().to_string()
        };
        git(&["init", "-q"]);
        let commit = |m: &str| {
            git(&["commit", "-q", "--allow-empty", "-m", m]);
            git(&["rev-parse", "--short", "HEAD"])
        };
        let before = commit("the model under development");
        let frozen = commit("the freeze");
        git(&["tag", "ra4-model"]);
        let after = commit("a fix that re-prices the development set identically");
        let day = |name: &str, text: Option<String>| {
            let d = dir.join(name);
            std::fs::create_dir_all(&d).unwrap();
            if let Some(t) = text {
                std::fs::write(d.join("day.txt"), t).unwrap();
            }
            check_frozen(&dir, &d, "ra4-model")
        };
        let run = |c: &str| format!("2026-10-01T21:00:00+02:00\n 21:00:00 up 1 day\nSelfAttention.apply  B=16,T=256 f32, cores 5-9,15-19\n{c}\n");
        assert_eq!(day("at", Some(run(&frozen))), Ok(()));
        assert_eq!(day("after", Some(run(&after) + &run(&frozen))), Ok(()));
        assert!(day("before", Some(run(&after) + &run(&before))).unwrap_err().contains("does not descend from ra4-model"));
        assert!(day("dirty", Some(run(&format!("{after}-dirty")))).unwrap_err().contains("uncommitted changes"));
        assert!(day("none", None).unwrap_err().contains("day.txt"));
        assert!(day("unknown", Some(run("0123abc"))).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

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

    /// D65: a sweep that records the capture of shapes is costed with it,
    /// the entry-run figure beside it; one that does not keeps 0.7.0's table.
    #[test]
    fn the_static_side_is_the_capture_of_shapes() {
        let dir = scratch("rank-shapes");
        let line = |k: &str, lanes: u32, median: f64| json!({ "entry": "gemm", "config": "N=1024", "op": "dot", "nest": "f/%2", "shape": "[4]", "dtype": "f32", "key": k, "layout": { "lanes": lanes }, "median": median, "mad": 0.01, "predicted": median }).to_string();
        std::fs::write(dir.join("f.jsonl"), [line("a", 1, 2.0), line("b", 4, 1.0)].join("\n")).unwrap();
        let sweep = |shapes: Option<f64>| {
            let mut v = json!({ "entry": "gemm", "config": "N=1024", "realization": "f32", "points": 2, "u": 0.2, "capture_seconds": 0.1, "cost_seconds": 0.1, "wall_seconds": 2.0, "interp_seconds": 1.0, "sample": null, "shard": null, "timed": true, "not_identical": 0 });
            if let Some(t) = shapes {
                v["static_capture_seconds"] = json!(t);
            }
            std::fs::write(dir.join("sweep.json"), v.to_string()).unwrap();
            let out = dir.join("ra4.md");
            assert_eq!(rank(&dir, &dir, &out, &RankOpts { entry: "gemm".into(), at: "N=1024".into(), ..RankOpts::default() }), ExitCode::SUCCESS);
            std::fs::read_to_string(&out).unwrap()
        };
        // Exhaustive 1 s; static 0.05 + 0.1 s, or 0.1 + 0.1 s with the entry run.
        let md = sweep(Some(0.05));
        assert!(md.contains("| exhaustive (s) | static (s) | ratio | static, entry run (s) | ratio, entry run |"), "{md}");
        assert!(md.contains("| 2 | 1.0 | 0.150 | 7 | 0.200 | 5 |"), "{md}");
        assert!(md.contains("N=1024 f32 7× (5× with the entry run)"), "{md}");
        let md = sweep(None);
        assert!(md.contains("| exhaustive (s) | static (s) | ratio |\n") && md.contains("| 2 | 1.0 | 0.200 | 5 |\n"), "{md}");
        assert!(md.contains("N=1024 f32 5× |"), "{md}");
        std::fs::remove_dir_all(&dir).unwrap();
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

    /// Plan 0.8.0 §3.5: M's configuration swept on a sample leaves M
    /// inconclusive, whatever its pairs say, and marks M's and O's cells; the
    /// same days swept whole are judged on their pairs.
    #[test]
    fn a_sampled_configuration_leaves_m_inconclusive() {
        let dir = scratch("rank-sample");
        let line = |k: &str, tile: [u32; 2], median: f64| json!({ "entry": "SelfAttention.apply", "config": "B=16,T=256", "op": "ein", "nest": "SelfAttention.apply/%13", "shape": "[4]", "dtype": "f32", "key": k, "layout": { "tile": tile }, "median": median, "mad": 0.01, "predicted": median }).to_string();
        std::fs::write(dir.join("f.jsonl"), [line("a", [1, 4], 2.0), line("b", [1, 8], 1.0)].join("\n")).unwrap();
        let rank_with = |sample: Json| {
            let sweep = json!({ "entry": "SelfAttention.apply", "config": "B=16,T=256", "realization": "f32", "points": 2, "u": 0.2, "capture_seconds": 0.1, "cost_seconds": 0.1, "wall_seconds": 2.0, "interp_seconds": 0.0, "sample": sample, "shard": null, "timed": true, "not_identical": 0 });
            std::fs::write(dir.join("sweep.json"), sweep.to_string()).unwrap();
            let out = dir.join("ra4.md");
            assert_eq!(rank(&dir, &dir, &out, &RankOpts::default()), ExitCode::SUCCESS);
            std::fs::read_to_string(&out).unwrap()
        };
        let md = rank_with(json!({ "top": 8, "random": 32, "seed": 0 }));
        assert!(md.contains("| `SelfAttention.apply/%13` 100.0 % of 1 (sample) | inconclusive: swept on a sample (plan 0.8.0 §3.5) |"), "{md}");
        assert!(md.contains("| `SelfAttention.apply/%13` 1.00 (sample) | **holds** |"), "{md}");
        let md = rank_with(Json::Null);
        assert!(md.contains("| `SelfAttention.apply/%13` 100.0 % of 1 | inconclusive: fewer than 10³ distinguished pairs |"), "{md}");
        assert!(md.contains("| `SelfAttention.apply/%13` 1.00 | **holds** |"), "{md}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Plan 0.8.0's statement Q, question 72's cost: the static side and the
    /// candidates' measured time over the nest's exhaustive sweep, a point's
    /// time being its launches at its median and its share of the sweep's
    /// time outside them. Decided on days that record the capture of shapes,
    /// on earlier ones when asked, and left out of their reports otherwise.
    #[test]
    fn statement_q_measures_only_the_candidates() {
        let dir = scratch("rank-q");
        // (key, tile, median, prediction), one timed launch a point: with u
        // = 0.2 the candidates are a, the model's pick, and b, the best.
        let rank_with = |b_pred: f64, shapes: Option<f64>, statement_q: bool| {
            let points = [("a", [1, 4], 1.2, 1.0), ("b", [1, 8], 1.0, b_pred), ("c", [4, 8], 40.0, 3.0)];
            let lines: Vec<String> = points.iter().map(|(k, tile, median, pred)| json!({ "entry": "SelfAttention.apply", "config": "B=16,T=256", "op": "ein", "nest": "SelfAttention.apply/%13", "shape": "[4]", "dtype": "f32", "key": k, "layout": { "tile": tile }, "seconds": [median], "median": median, "mad": 0.01, "predicted": pred }).to_string()).collect();
            std::fs::write(dir.join("f.jsonl"), lines.join("\n")).unwrap();
            // Two launches a point, 84.4 s, and 0.2 s a point outside them.
            let mut sweep = json!({ "entry": "SelfAttention.apply", "config": "B=16,T=256", "realization": "f32", "points": 3, "u": 0.2, "repeat": 1, "sweep_seconds": 85.0, "capture_seconds": 0.1, "cost_seconds": 0.15, "wall_seconds": 90.0, "interp_seconds": 0.0, "sample": null, "shard": null, "timed": true, "not_identical": 0 });
            if let Some(t) = shapes {
                sweep["static_capture_seconds"] = json!(t);
            }
            std::fs::write(dir.join("sweep.json"), sweep.to_string()).unwrap();
            let out = dir.join("ra4.md");
            assert_eq!(rank(&dir, &dir, &out, &RankOpts { statement_q, ..RankOpts::default() }), ExitCode::SUCCESS);
            std::fs::read_to_string(&out).unwrap()
        };
        // Static 0.05 + 0.15 s; the candidates 2 × (1.2 + 1.0) + 2 × 0.2 =
        // 4.8 s; the nest 85 s: 5.9 %. Their launches 2.2 of 42.2 s.
        let md = rank_with(1.1, Some(0.05), false);
        assert!(md.contains("## Statements (plan 0.7.0 §3.7; Q, plan 0.8.0 §3.6)"), "{md}");
        assert!(md.contains("| 1 contraction nests: the best candidate within 1.00× of the best measured at worst (`SelfAttention.apply/%13` B=16,T=256 f32); the static side and the candidates' measured time at most 5.9 % of the nest's exhaustive sweep"), "{md}");
        assert!(md.contains("at most 5.21 % (`SelfAttention.apply/%13` B=16,T=256 f32, D71's count) | **holds** |"), "{md}");
        assert!(md.contains("| `SelfAttention.apply/%13` | B=16,T=256 | f32 | 3 | 2 | 1.00 | 5.9 % | 0.20, 4.80, 85.0 | 5.21 % | **holds** |"), "{md}");
        // b out of the candidates: a alone, 1.2× the best, though cheap.
        let md = rank_with(2.0, Some(0.05), false);
        assert!(md.contains("| `SelfAttention.apply/%13` | B=16,T=256 | f32 | 3 | 1 | 1.20 | 3.3 % | 0.20, 2.60, 85.0 | 2.84 % | **fails** |"), "{md}");
        // A day recorded before the capture of shapes: no Q, unless asked,
        // and then with the capture that runs the entry, 0.1 + 0.15 s.
        let md = rank_with(1.1, None, false);
        assert!(md.contains("## Statements (plan 0.7.0 §3.7)\n") && !md.contains("| Q |") && !md.contains("Statement Q"), "{md}");
        let md = rank_with(1.1, None, true);
        assert!(md.contains("| 3 | 2 | 1.00 | 5.9 % | 0.25, 4.80, 85.0 | 5.21 % | **holds** |"), "{md}");
        std::fs::remove_dir_all(&dir).unwrap();
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

    /// Plan 0.8.0 step 4 (4b): a calibration's samples, costed again from
    /// their layouts at the products their signatures name, come back as
    /// they were recorded; under a rule their costs move and nothing else
    /// does; a sample the program cannot have produced is refused.
    #[test]
    fn a_calibration_is_costed_again_as_it_was_recorded() {
        assert_eq!(product_of("kernel dot (%0 : f32[16, 4096], %1 : f32[4096, 4096]) -> %2 : f32[16, 4096]"), Some(("f32".to_string(), [16, 4096, 4096])));
        assert_eq!(product_of("kernel add (%0 : f32[4, 3], %1 : f32[4, 3]) -> %2 : f32[4, 3]"), None);
        let file = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../experiments/matmul.psk");
        let machine = Machine { workers: 4, l1: 64 << 10, l2: 2 << 20, l3: 8 << 20, l3_sharing: 4, vector_registers: 32, vector_bytes: 16, source: "test".into(), line: 64 };
        let keys = ["form=tiled,tile=1x4,kblock=none,order=IJ,lanes=1,split=none", "form=tiled,tile=2x8,kblock=32,order=KIJ,lanes=1,split=(1, 2)", "form=tiled,tile=4x16,kblock=none,order=JI,lanes=1,split=(0, 2)"];
        // Recorded as `presk bench --space calib` records them, times made up.
        let mut samples = Vec::new();
        for (t, lanes) in [("f32", 4), ("f64", 2)] {
            let plan = presk_rt::Plan::from_realization(presk_rt::Realization::parse(t).unwrap());
            let captured = capture(&file, "matmul", &["I=8".into(), "K=64".into(), "J=32".into()], &[], &plan, 0, None, Capture::Shapes).unwrap();
            let l = captured.launches.iter().find(|l| matches!(launch_kind(l), NestKind::Contraction(_))).unwrap();
            for key in keys.iter().flat_map(|k| [k.to_string(), k.replace("lanes=1", &format!("lanes={lanes}"))]) {
                let layout = Layout::from_key(&key).unwrap();
                let kernels = lower_launch(l, &layout, Some(machine.workers)).unwrap();
                let k = kernels.last().unwrap();
                let cost = presk_kernel::cost::planned_cost(k, &machine);
                let signature = k.to_string().lines().next().unwrap().trim_end_matches(" {").to_string();
                samples.push(presk_opt::profile::PlannedSample { signature, layout: layout.key(), group: group(l.dtype, k, cost.workers), cost, seconds: 1e-3 * (samples.len() + 1) as f64 });
            }
        }
        let none = presk_kernel::cost::CostRules::default();
        assert_eq!(recost(&samples, &file, "matmul", &machine, &none).unwrap(), samples);
        let over = recost(&samples, &file, "matmul", &machine, &presk_kernel::cost::CostRules::parse("overhead").unwrap()).unwrap();
        for (a, b) in over.iter().zip(&samples) {
            assert!(a.cost.work > b.cost.work && a.cost.traffic == b.cost.traffic, "{}", a.layout);
            assert_eq!((&a.signature, &a.layout, &a.group, a.seconds), (&b.signature, &b.layout, &b.group, b.seconds));
        }
        let mut wrong = samples[0].clone();
        wrong.group = group_key("f32", 4, 1);
        assert!(recost(&[wrong], &file, "matmul", &machine, &none).unwrap_err().contains("group"));
    }

    fn root() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    /// `presk plan` into `out`, under `realize`, with a synthetic profile of
    /// `f32` groups: its exit code and the plan written.
    fn plan_under(name: &str, file: &str, entry: &str, dims: &[&str], consts: &[&str], realize: &str) -> (ExitCode, Option<presk_rt::Plan>) {
        let d = scratch(name);
        let p = d.join("profile.json");
        profile(&p, 1.0, 0.2);
        let out = d.join("plan.json");
        let dims: Vec<String> = dims.iter().map(|s| s.to_string()).collect();
        let consts: Vec<String> = consts.iter().map(|s| s.to_string()).collect();
        let base = presk_rt::Plan::from_realization(presk_rt::Realization::parse(realize).unwrap());
        let code = plan(&root().join(file), entry, &dims, &consts, &base, &p, None, Space::Full, false, Some(&out), 0);
        let written = out.exists().then(|| presk_rt::Plan::parse(&format!("@{}", out.display())).unwrap());
        (code, written)
    }

    /// 0.9.0 (D82): the MLP launches `Linear.apply/%3` at D→4D and at 4D→D.
    /// The first shape's best layout does not lower at the second, which is
    /// the plan `presk run` refused; the plan states one that lowers at both,
    /// and the nests launched at one shape keep their own best.
    #[test]
    fn a_nest_at_several_shapes_takes_a_layout_every_shape_takes() {
        let (file, dims) = ("tests/examples/mlp/mlp.psk", ["D=4", "B=2"]);
        let (code, written) = plan_under("plan-shapes", file, "mse_grad", &dims, &[], "f32");
        assert_eq!(code, ExitCode::SUCCESS);
        let written = written.expect("a plan is written");
        let dims: Vec<String> = dims.iter().map(|s| s.to_string()).collect();
        let captured = capture(&root().join(file), "mse_grad", &dims, &[], &f32_plan(), 0, None, Capture::Shapes).unwrap();
        let workers = captured.machine.workers;
        let shapes: Vec<&Launch> = captured.launches.iter().filter(|l| l.nest == "Linear.apply/%3").collect();
        assert_eq!(shapes.len(), 2, "Linear.apply/%3 at two shapes");
        assert_eq!(shapes.iter().map(|l| l.count).collect::<Vec<_>>(), [3, 3], "each launched three times (twice by the forward pass the gradient recomputes)");
        // The first shape's own best does not lower at the second.
        let d = scratch("plan-shapes-fit");
        profile(&d.join("p.json"), 1.0, 0.2);
        let (fit, _) = load_fit(Some(&d.join("p.json"))).unwrap().unwrap();
        let best = |l: &Launch| {
            let mut s: Vec<(f64, Layout)> = points(l, Space::Full, workers).into_iter().filter_map(|(p, ks)| predict(&fit, &captured.machine, l.dtype, &ks).map(|t| (t, p))).collect();
            s.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            s.remove(0).1
        };
        assert!(lower_launch(shapes[1], &best(shapes[0]), Some(workers)).is_err(), "the first shape's best lowers at the second: the test does not test the rule");
        // Every nest's layout lowers at every shape of it.
        for l in &captured.launches {
            if let Some(layout) = written.layout.get(&l.nest) {
                assert!(lower_launch(l, layout, Some(workers)).is_ok(), "{} at {}: {layout}", l.nest, shape(l));
            }
        }
        // A nest at one shape keeps its best.
        for l in captured.launches.iter().filter(|l| captured.launches.iter().filter(|m| m.nest == l.nest).count() == 1) {
            assert_eq!(written.layout.get(&l.nest), Some(&best(l)), "{}", l.nest);
        }
    }

    /// 0.9.0 (D82): a type the profile was not calibrated in is refused, and
    /// no plan is written, where `presk plan` wrote an empty plan; a
    /// realization with sites of such a type leaves them to the default.
    #[test]
    fn a_type_the_profile_cannot_price_is_refused_or_left_out() {
        let minigpt = ["V=64", "D=16", "H=2", "L=2"];
        let dims = ["B=2", "T=8"];
        let (code, written) = plan_under("plan-bf16", "tests/examples/minigpt/minigpt.psk", "MiniGPT.apply", &dims, &minigpt, "bf16");
        assert_eq!(code, ExitCode::from(2));
        assert!(written.is_none(), "a plan was written");
        let mixed = format!("@{}", root().join("tests/plan/mixed-realization.json").display());
        let (code, written) = plan_under("plan-mixed", "tests/examples/minigpt/minigpt.psk", "MiniGPT.apply", &dims, &minigpt, &mixed);
        assert_eq!(code, ExitCode::SUCCESS);
        let written = written.expect("a plan is written");
        let base = presk_rt::Plan::from_realization(presk_rt::Realization::parse(&mixed).unwrap());
        let dims: Vec<String> = dims.iter().map(|s| s.to_string()).collect();
        let consts: Vec<String> = minigpt.iter().map(|s| s.to_string()).collect();
        let captured = capture(&root().join("tests/examples/minigpt/minigpt.psk"), "MiniGPT.apply", &dims, &consts, &base, 0, None, Capture::Shapes).unwrap();
        let other: BTreeSet<&str> = captured.launches.iter().filter(|l| l.dtype != presk_rt::Dtype::F32).map(|l| l.nest.as_str()).collect();
        assert!(!other.is_empty(), "the mixed realization has sites of other types");
        for n in &other {
            assert!(!written.layout.contains_key(*n), "{n}: a layout the profile cannot price");
        }
        assert!(!written.layout.is_empty(), "the f32 sites have layouts");
    }
}
