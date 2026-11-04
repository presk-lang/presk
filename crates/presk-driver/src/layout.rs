//! The layouts of an entry's nests (plan 0.7.0 step 5, moved from
//! `presk-cli` in 0.11.0): the entry captured once, its launches lowered
//! under every layout of their space, each point priced by a profile's
//! layout cost model, and one layout chosen per nest — what `presk plan`
//! prints and writes, and what `presk autotune`, `presk reprice` and the
//! C export's SCoP mode read.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::time::Instant;

use presk_core::ad::Remat;
use presk_kernel::cost::{Machine, planned_cost_under};
use presk_kernel::plan::{launch_kind, lower_launch, lower_launch_splits};
use presk_kernel::{Kernel, KernelExec, Launch, Threads};
use presk_opt::profile::{LayoutFit, group_key};
use presk_rt::layout::{BlockLoop, KBlock, Split};
use presk_rt::{Buf, Layout, NestKind};

use crate::entry::{interp_diag, run_diag};
use crate::{Report, Result, Stop};

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
    pub fn parse(s: &str) -> std::result::Result<Space, String> {
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
pub(crate) struct NoRun;

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
        Shaped::Idx { bound, axes } => Value::Idx { bound: *bound, t: Tensor { axes: axes.clone(), data: Buf::I64(vec![0; size(axes)].into()) } },
        Shaped::Value(v) => Value::from_interp(v)?,
        Shaped::Tuple(parts) => Value::Tuple(parts.iter().map(|p| zeros(b, realize, p)).collect::<presk_rt::Result<_>>()?),
        Shaped::Params { init, dims, key } => {
            let p = presk_rt::Backend::prepare(b, *init, dims, realize)?;
            presk_rt::Backend::call(b, &p, vec![Value::Key(key.clone())])?
        }
    })
}

/// A module's parameters as zeros of their shapes, in `f64`: its initializer
/// on the kernel backend with no executor, as [`Capture::Shapes`] builds a
/// capture's (0.17.0 step 7, D180: the tree a checkpoint is read against,
/// [`crate::entry::read_given_params`]).
pub fn params_zeros(core: &presk_core::Program, prog: &presk_types::elab::ElabProgram, init: presk_core::FuncId, dims: &presk_interp::eval::Dims, key: &presk_interp::key::Key) -> presk_rt::Result<presk_rt::Value> {
    let realize = presk_rt::Realization::uniform(presk_rt::Dtype::F64);
    let mut b = presk_kernel::KernelBackend::with_exec(core, prog, &realize, Box::new(NoRun));
    b.default_layout.lanes = None;
    let p = presk_rt::Backend::prepare(&mut b, init, dims, &realize)?;
    presk_rt::Backend::call(&mut b, &p, vec![presk_rt::Value::Key(key.clone())])
}

/// Runs `entry` once with the kernel backend capturing, natively or on the
/// host program alone, and returns the launches of the nests `nests` names
/// (every nest when it is empty; several since 0.10.0 step 2).
#[allow(clippy::too_many_arguments)]
pub fn capture(file: &Path, entry: &str, dims: &[String], consts: &[String], plan: &presk_rt::Plan, seed: u64, nests: &[String], mode: Capture, remat: Remat, report: &mut dyn Report) -> Result<Captured> {
    capture_as(file, entry, dims, consts, plan, seed, nests, mode, remat, presk_core::inference::Mode::Inference, report)
}

/// [`capture`] of the program of a mode (0.16.0, D164): `Training` for the
/// plan of a training run, whose program keeps its masks and so names its
/// nests otherwise than the inference reading does.
#[allow(clippy::too_many_arguments)]
pub fn capture_as(file: &Path, entry: &str, dims: &[String], consts: &[String], plan: &presk_rt::Plan, seed: u64, nests: &[String], mode: Capture, remat: Remat, lowering: presk_core::inference::Mode, report: &mut dyn Report) -> Result<Captured> {
    let t0 = Instant::now();
    let dims = crate::front::parse_dims(dims, report)?;
    let (sm, prog, _, mut core) = crate::front::typed_core_as(file, consts, false, lowering, report)?;
    let realize = &plan.realization;
    let mut diags = crate::ad::differentiate(&mut core, remat, realize.groups_visible());
    diags.extend(crate::ad::stored_vjps(&core, realize));
    report.diagnostics(&diags, &sm);
    if presk_syntax::diag::has_errors(&diags) {
        return Err(Stop(1));
    }
    let opts = presk_interp::RunOptions { entry: entry.to_string(), dims, seed, bits: 64, digits: 12 };
    let mut fail = |d: presk_syntax::Diagnostic| {
        report.diagnostics(&[d], &sm);
        Stop(1)
    };
    let mut b = match mode {
        Capture::Run => presk_cranelift::native_backend(&core, &prog, realize).map_err(|e| fail(run_diag(e.code, e.message, e.span)))?,
        Capture::Shapes => {
            // The native backend's defaults, its executor aside.
            let mut b = presk_kernel::KernelBackend::with_exec(&core, &prog, realize, Box::new(NoRun));
            b.default_layout.lanes = None;
            b
        }
    };
    // A realization with integer weights quantizes drawn values; zeros would
    // have no scale. Parameters in their own types are zeros of those types.
    let (func, fdims, args) = if mode == Capture::Shapes && realize.quant.is_empty() {
        let (func, fdims, shaped) = presk_interp::run::prepare_shapes(&core, &prog, &opts).map_err(|e| fail(interp_diag(e)))?;
        let mut args = shaped.iter().map(|(_, s)| zeros(&mut b, realize, s)).collect::<presk_rt::Result<Vec<_>>>().map_err(|e| fail(run_diag(e.code, e.message, e.span)))?;
        realize.store_params(&mut args).map_err(|e| fail(run_diag(e.code, e.message, e.span)))?;
        (func, fdims, args)
    } else {
        let (func, fdims, inputs) = presk_interp::run::prepare_run(&core, &prog, &opts).map_err(|e| fail(interp_diag(e)))?;
        let given: Vec<presk_rt::Value> = inputs.iter().map(|(_, v)| presk_rt::Value::from_interp(v)).collect::<presk_rt::Result<_>>().map_err(|e| fail(run_diag(e.code, e.message, e.span)))?;
        let args = realize.realize_args(&given).map_err(|e| fail(run_diag(e.code, e.message, e.span)))?;
        (func, fdims, args)
    };
    b.capture = Some(Vec::new());
    let p = presk_rt::Backend::prepare(&mut b, func, &fdims, realize).map_err(|e| fail(run_diag(e.code, e.message, e.span)))?;
    presk_rt::Backend::call(&mut b, &p, args).map_err(|e| fail(run_diag(e.code, e.message, e.span)))?;
    let mut launches = b.capture.take().unwrap_or_default();
    if !nests.is_empty() {
        launches.retain(|l| nests.contains(&l.nest));
        if let Some(n) = nests.iter().find(|n| !launches.iter().any(|l| l.nest == **n)) {
            report.line(&format!("presk: the entry launches no nest `{n}`"));
            return Err(Stop(2));
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

/// A kernel's group: its type, lanes, and the workers its split states.
pub fn group(dtype: presk_rt::Dtype, k: &Kernel, effective: usize) -> String {
    let stated = match k.threads {
        Threads::At { workers, .. } => workers,
        Threads::One => 1,
        Threads::Default => effective,
    };
    group_key(dtype.name(), k.lanes, stated)
}

/// `a[2, 6, 256, 64] b[…]`: a launch's operand shapes.
pub fn shape(l: &Launch) -> String {
    l.operands.iter().map(|o| format!("[{}]", o.axes.iter().map(|a| a.len.to_string()).collect::<Vec<_>>().join(", "))).collect::<Vec<_>>().join(" ")
}

/// A profile's layout cost model and the machine it was fitted on.
pub fn load_fit(profile: Option<&Path>, report: &mut dyn Report) -> Result<Option<(LayoutFit, Option<Machine>)>> {
    let Some(path) = profile else { return Ok(None) };
    match presk_opt::Profile::load(path) {
        Ok(p) => match p.layout {
            Some(f) => Ok(Some((f, p.machine_desc))),
            None => {
                report.line(&format!("presk: {} has no layout cost model: calibrate it with `presk bench --calibrate {} --space calib`", path.display(), path.display()));
                Err(Stop(2))
            }
        },
        Err(m) => {
            report.line(&format!("presk: --profile: {m}"));
            Err(Stop(2))
        }
    }
}

/// One launch's points, priced: what `presk plan --cost` lists.
pub struct Costed {
    /// The launch, an index into [`Captured::launches`].
    pub launch: usize,
    /// The points of its space that lower at this launch.
    pub space: usize,
    /// The points the model prices, best first.
    pub scored: Vec<(f64, Layout)>,
    /// How many are within `1 + u` of the best: the model cannot tell them
    /// apart.
    pub kept: usize,
}

/// The layouts chosen for an entry's nests, and why.
pub struct Planned {
    /// Every launch the model prices, in the entry's order.
    pub costed: Vec<Costed>,
    /// The launches it prices at no point: a type the profile was not
    /// calibrated in (0.9.0, D82); their nests keep the default layout.
    pub unpriced: Vec<usize>,
    /// One layout per nest.
    pub layout: BTreeMap<String, Layout>,
    /// For a nest launched at several shapes whose shapes do not all choose
    /// the layout it takes, the line that says what it took.
    pub summary: HashMap<String, String>,
    /// The products that take the `add` that alone reads them as an
    /// epilogue.
    pub epilogues: Vec<String>,
    /// The points of every launch's space.
    pub total_points: usize,
    /// The types the profile's layout model was calibrated in.
    pub types: Vec<String>,
    /// The time the pricing and the choice took.
    pub seconds: f64,
}

/// No layout of the entry is priced: its types, and the profile's.
pub struct Unpriced {
    pub dtypes: Vec<String>,
    pub types: Vec<String>,
}

/// Prices every point of every launch under `fit` on `model_machine` (the
/// machine the model was fitted on), and chooses one layout per nest: the
/// best of a nest launched at one shape; at several, the layout the model
/// prices at every shape whose predicted time, each shape's weighed by the
/// times the entry launches it, is least (0.9.0, D82), since the first
/// shape's best may not suit another. Then the epilogues (0.10.0 step 7).
pub fn choose(fit: &LayoutFit, model_machine: &Machine, captured: &Captured, space: Space) -> std::result::Result<Planned, Unpriced> {
    let t_cost = Instant::now();
    let mut costed: Vec<Costed> = Vec::new();
    let mut unpriced: Vec<usize> = Vec::new();
    let mut total_points = 0usize;
    for (i, l) in captured.launches.iter().enumerate() {
        let pts = points(l, space, captured.machine.workers);
        let space_size = pts.len();
        total_points += space_size;
        let mut scored: Vec<(f64, Layout)> = pts.into_iter().filter_map(|(p, ks)| predict(fit, model_machine, l.dtype, &ks).map(|t| (t, p))).collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let Some((best_t, _)) = scored.first() else {
            // Points none of which a group of the model prices: a type the
            // profile was not calibrated in (0.9.0, D82).
            if space_size > 0 {
                unpriced.push(i);
            }
            continue;
        };
        let kept = scored.iter().filter(|(t, _)| *t <= best_t * (1.0 + fit.u)).count();
        costed.push(Costed { launch: i, space: space_size, scored, kept });
    }
    let types: Vec<String> = {
        let mut t: Vec<String> = fit.pi.keys().filter_map(|g| g.split('+').next()).map(String::from).collect();
        t.dedup();
        t
    };
    if costed.is_empty() && !unpriced.is_empty() {
        let dtypes: Vec<String> = unpriced.iter().map(|i| captured.launches[*i].dtype.to_string()).collect::<BTreeSet<_>>().into_iter().collect();
        return Err(Unpriced { dtypes, types });
    }
    let launch = |c: &Costed| &captured.launches[c.launch];
    // One layout per nest, since a plan states one per name.
    let mut layout: BTreeMap<String, Layout> = BTreeMap::new();
    let nests: Vec<&str> = {
        let mut n: Vec<&str> = Vec::new();
        for c in &costed {
            if !n.contains(&launch(c).nest.as_str()) {
                n.push(&launch(c).nest);
            }
        }
        n
    };
    let mut summary: HashMap<String, String> = HashMap::new();
    for nest in &nests {
        // A nest the model cannot price at one of its shapes is left to the
        // default layout, which suits every shape.
        if unpriced.iter().any(|i| captured.launches[*i].nest == *nest) {
            continue;
        }
        let group: Vec<&Costed> = costed.iter().filter(|c| launch(c).nest == *nest).collect();
        if let [c] = group.as_slice() {
            layout.insert(nest.to_string(), c.scored[0].1.clone());
            continue;
        }
        let others: Vec<HashMap<String, f64>> = group[1..].iter().map(|c| c.scored.iter().map(|(t, p)| (p.key(), *t)).collect()).collect();
        let weigh = |ts: &[f64]| -> f64 { ts.iter().zip(&group).map(|(t, c)| t * launch(c).count as f64).sum() };
        let own = weigh(&group.iter().map(|c| c.scored[0].0).collect::<Vec<_>>());
        let mut chosen: Option<(f64, &Layout)> = None;
        for (t0, p) in &group[0].scored {
            let key = p.key();
            let ts = std::iter::once(Some(*t0)).chain(others.iter().map(|m| m.get(&key).copied())).collect::<Option<Vec<f64>>>();
            let Some(ts) = ts else { continue };
            let total = weigh(&ts);
            if chosen.is_none_or(|(b, _)| total < b) {
                chosen = Some((total, p));
            }
        }
        let counts = group.iter().map(|c| launch(c).count.to_string()).collect::<Vec<_>>().join(" + ");
        // A line for a nest whose shapes do not all choose the layout it
        // takes.
        match chosen {
            Some((total, p)) => {
                layout.insert(nest.to_string(), p.clone());
                if group.iter().any(|c| c.scored[0].1 != *p) {
                    summary.insert(nest.to_string(), format!("= {nest} at {} shapes, {counts} launches: {p}; {:.2} µs over them, {:.2} µs at each shape's own best", group.len(), total * 1e6, own * 1e6));
                }
            }
            None => {
                summary.insert(nest.to_string(), format!("= {nest} at {} shapes: no layout the model prices at every one, left to the default", group.len()));
            }
        }
    }
    // The epilogues (plan 0.10.0 step 7): a product whose every launch can
    // take the `add` that alone reads its value takes it, over the layout
    // chosen without it. It saves the `add`'s launch and its pass over the
    // output and adds no arithmetic; the model prices a nest alone, so it
    // has no term for it, and no prediction moves.
    let mut epilogues: Vec<String> = Vec::new();
    for l in &captured.launches {
        if epilogues.contains(&l.nest) {
            continue;
        }
        let same: Vec<&Launch> = captured.launches.iter().filter(|m| m.nest == l.nest).collect();
        if !same.iter().all(|m| m.epilogue.is_some()) {
            continue;
        }
        let with = Layout { epilogue: Some(true), ..layout.get(&l.nest).cloned().unwrap_or_default() };
        if same.iter().all(|m| lower_launch(m, &with, Some(captured.machine.workers)).is_ok()) {
            layout.insert(l.nest.clone(), with);
            epilogues.push(l.nest.clone());
        }
    }
    Ok(Planned { costed, unpriced, layout, summary, epilogues, total_points, types, seconds: t_cost.elapsed().as_secs_f64() })
}

/// The kernels an entry launches, as `presk dump --stage kernel` lists them.
pub struct Listing {
    /// Every distinct kernel, in the order of its first launch.
    pub kernels: Vec<Kernel>,
    /// The launches a call makes.
    pub launches: u64,
    /// The launches fusion saved.
    pub fused: u64,
    /// The realization's label.
    pub label: String,
}

/// The kernels `entry` launches under `plan`, found from its shapes alone
/// (plan 0.11.0 step 2, X4): the host program runs on an executor that runs
/// no kernel ([`Capture::Shapes`]'s), so that nothing is computed and the
/// listing of a model takes the time of its host, where until 0.11.0 the
/// kernel interpreter ran every kernel (minutes on the course GPT). The
/// kernels are lowered at the entry's shapes, which are its values' only
/// part they depend on, so the listing is the one the run gave. The
/// parameters are zeros of their shapes, drawn by their initializers on a
/// backend of their own, whose kernels are not the entry's; a realization
/// with integer weights gets the drawn parameters, since zeros have no
/// scale. The backend's defaults are the kernel interpreter's, as before:
/// scalar, unless the plan states lanes.
pub fn kernels(core: &presk_core::Program, prog: &presk_types::elab::ElabProgram, entry: &str, dims: Vec<(String, u64)>, plan: &presk_rt::Plan, seed: u64) -> presk_rt::Result<Listing> {
    listed(core, prog, entry, dims, plan, seed, false).map(|h| h.listing)
}

/// The kernels of a call and its host.
pub struct Hosted {
    pub listing: Listing,
    /// The host program, or why the recorder refused it.
    pub recording: std::result::Result<presk_kernel::record::Recording, String>,
    /// The host program when the recorder refused only its result, which a
    /// C program cannot write (a key, a host scalar, integer weights): what
    /// `presk memory` counts (plan 0.15.0 step 1, D148). `None` when
    /// `recording` holds it, or when the recorder refused while recording.
    pub counted: Option<presk_kernel::record::Recording>,
    /// The entry's arguments' names.
    pub names: Vec<String>,
    /// When the first argument is a parameter tree: its leaves' paths and
    /// their elements, at the entry's dimensions.
    pub params: Vec<(String, usize)>,
    /// Every launch of the call: its kernel's place in the listing and its
    /// nest (plan 0.13.0 step 1, D134).
    pub launched: Vec<presk_kernel::Launched>,
}

impl Hosted {
    /// The host program as recorded, whether or not the C export takes it.
    pub fn counted(&self) -> Option<&presk_kernel::record::Recording> {
        self.recording.as_ref().ok().or(self.counted.as_ref())
    }
}

/// [`kernels`], and the host program of the call that launched them, as the
/// C export records it ([`presk_kernel::record`]): which site launches which
/// kernel, and the structure of the calls and folds (plan 0.11.0 step 3).
/// The recording is `None` with the reason when the recorder refuses the
/// entry, as the C export would.
pub fn kernels_with_host(core: &presk_core::Program, prog: &presk_types::elab::ElabProgram, entry: &str, dims: Vec<(String, u64)>, plan: &presk_rt::Plan, seed: u64) -> presk_rt::Result<Hosted> {
    listed(core, prog, entry, dims, plan, seed, true)
}

fn listed(core: &presk_core::Program, prog: &presk_types::elab::ElabProgram, entry: &str, dims: Vec<(String, u64)>, plan: &presk_rt::Plan, seed: u64, record: bool) -> presk_rt::Result<Hosted> {
    let realize = &plan.realization;
    let opts = presk_interp::RunOptions { entry: entry.to_string(), dims, seed, bits: 64, digits: 12 };
    let mut b = presk_kernel::KernelBackend::with_exec(core, prog, realize, Box::new(NoRun));
    b.draws = false;
    b.set_plan(plan.clone());
    let (func, fdims, args, names) = if realize.quant.is_empty() {
        let (func, fdims, shaped) = presk_interp::run::prepare_shapes(core, prog, &opts)?;
        let mut init = presk_kernel::KernelBackend::with_exec(core, prog, realize, Box::new(NoRun));
        init.draws = false;
        let mut args = shaped.iter().map(|(_, s)| zeros(&mut init, realize, s)).collect::<presk_rt::Result<Vec<_>>>()?;
        realize.store_params(&mut args)?;
        (func, fdims, args, shaped.iter().map(|(n, _)| n.clone()).collect::<Vec<String>>())
    } else {
        let (func, fdims, inputs) = presk_interp::run::prepare_run(core, prog, &opts)?;
        let mut args: Vec<presk_rt::Value> = inputs.iter().map(|(_, v)| presk_rt::Value::from_interp(v)).collect::<presk_rt::Result<_>>()?;
        realize.store_params(&mut args)?;
        (func, fdims, args, inputs.iter().map(|(n, _)| n.clone()).collect::<Vec<String>>())
    };
    let params = match args.first() {
        Some(theta @ presk_rt::Value::Record(_)) => presk_rt::quant::param_paths(theta).into_iter().filter_map(|path| crate::gap::param_leaf(theta, &path).map(|t| (path, t.data.len()))).collect(),
        _ => Vec::new(),
    };
    let p = presk_rt::Backend::prepare(&mut b, func, &fdims, realize)?;
    if record {
        b.recorder = Some(Box::new(presk_kernel::record::Recorder::named(names.clone())));
        b.launch_log = Some(Vec::new());
    }
    presk_rt::Backend::call(&mut b, &p, args)?;
    let launched = b.launch_log.take().unwrap_or_default();
    let (recording, counted) = match b.recorder.take().map(|r| r.finish_counted()) {
        Some((Some(rec), Ok(()))) => (Ok(rec), None),
        Some((rec, Err(e))) => (Err(e.message), rec),
        Some((None, Ok(()))) | None => (Err("not recorded".into()), None),
    };
    Ok(Hosted { listing: Listing { kernels: std::mem::take(&mut b.kernels), launches: b.launches, fused: b.fused, label: b.realize.label() }, recording, counted, names, params, launched })
}

/// One call of a training step recorded (plan 0.15.0 step 2, D149): its
/// function, its dimensions by name, the host and its launches.
pub struct StepCall {
    pub func: String,
    pub dims: Vec<(String, u64)>,
    pub hosted: Hosted,
}

/// The calls `presk train` makes for one step (`presk_train::train`), each
/// recorded as [`kernels_with_host`] records an entry, on one backend that
/// runs no kernel, in training mode: the objective's `.value_vjp` on the
/// parameters, a key and a batch at `dims`, then `adamw.lr`, then
/// `adamw.step` on the parameters, the state `adamw.init` gives them and the
/// gradient. The parameters are zeros of their shapes, the state too, as
/// the listing's are; a step's buffers depend on its shapes only.
pub fn step_with_host(core: &presk_core::Program, prog: &presk_types::elab::ElabProgram, fns: &presk_train::TrainingFns, dims: Vec<(String, u64)>, plan: &presk_rt::Plan, seed: u64) -> presk_rt::Result<Vec<StepCall>> {
    use presk_rt::{Backend, Tensor, Value};
    let realize = &plan.realization;
    let objective = core.func(fns.objective).name.clone();
    let opts = presk_interp::RunOptions { entry: objective, dims: dims.clone(), seed, bits: 64, digits: 12 };
    let (_, fdims, shaped) = presk_interp::run::prepare_shapes(core, prog, &opts)?;
    let mut init = presk_kernel::KernelBackend::with_exec(core, prog, realize, Box::new(NoRun));
    init.draws = false;
    let mut args = shaped.iter().map(|(_, s)| zeros(&mut init, realize, s)).collect::<presk_rt::Result<Vec<_>>>()?;
    realize.store_params(&mut args)?;
    let mut names: Vec<String> = shaped.iter().map(|(n, _)| n.clone()).collect();
    let mut b = presk_kernel::KernelBackend::with_exec(core, prog, realize, Box::new(NoRun));
    b.draws = false;
    b.set_plan(plan.clone());
    b.set_training(true);
    let named = |f: presk_core::FuncId| -> presk_interp::eval::Dims { core.func(f).dims.iter().filter_map(|v| fdims.iter().find(|(w, _)| prog.dim_name(**w) == prog.dim_name(*v)).map(|(_, n)| (*v, *n))).collect() };
    let mut out = Vec::new();
    let mut record = |b: &mut presk_kernel::KernelBackend, f: presk_core::FuncId, d: presk_interp::eval::Dims, args: Vec<Value>, names: Vec<String>| -> presk_rt::Result<Value> {
        let p = b.prepare(f, &d, realize)?;
        let params = match args.first() {
            Some(theta @ Value::Record(_)) => presk_rt::quant::param_paths(theta).into_iter().filter_map(|path| crate::gap::param_leaf(theta, &path).map(|t| (path, t.data.len()))).collect(),
            _ => Vec::new(),
        };
        b.recorder = Some(Box::new(presk_kernel::record::Recorder::named(names.clone())));
        b.launch_log = Some(Vec::new());
        let (launches, fused) = (b.launches, b.fused);
        let v = b.call(&p, args)?;
        let launched = b.launch_log.take().unwrap_or_default();
        let (recording, counted) = match b.recorder.take().map(|r| r.finish_counted()) {
            Some((Some(rec), Ok(()))) => (Ok(rec), None),
            Some((rec, Err(e))) => (Err(e.message), rec),
            Some((None, Ok(()))) | None => (Err("not recorded".into()), None),
        };
        let dims: Vec<(String, u64)> = d.iter().map(|(v, n)| (prog.dim_name(*v), *n)).collect();
        let listing = Listing { kernels: b.kernels.clone(), launches: b.launches - launches, fused: b.fused - fused, label: b.realize.label() };
        out.push(StepCall { func: core.func(f).name.clone(), dims, hosted: Hosted { listing, recording, counted, names, params, launched } });
        Ok(v)
    };
    // The objective's value and its gradient, the first of the cotangents.
    args.push(Value::Real(Tensor::scalar_f64(1.0)));
    names.push("seed".into());
    let theta = args[0].clone();
    let vv = record(&mut b, fns.value_vjp, fdims.clone(), args, names)?;
    // As `presk_train::value_and_gradient` takes it.
    let g = match vv {
        Value::Tuple(mut parts) if parts.len() >= 2 => {
            parts.truncate(2);
            parts.pop().unwrap()
        }
        other => return presk_rt::err("E7001", format!("value_vjp returned a {}", other.kind())),
    };
    let d_opt = named(fns.opt_step);
    let p = b.prepare(fns.opt_init, &d_opt, realize)?;
    let state = b.call(&p, vec![theta.clone()])?;
    record(&mut b, fns.opt_lr, presk_interp::eval::Dims::new(), vec![Value::Nat(1), Value::Nat(1)], vec!["t".into(), "total".into()])?;
    record(&mut b, fns.opt_step, d_opt, vec![theta, state, g, Value::Nat(1), Value::Nat(1)], vec!["θ".into(), "s".into(), "g".into(), "t".into(), "total".into()])?;
    Ok(out)
}

/// One kernel of a point, priced: its group, the model's inputs and the
/// terms of its prediction (plan 0.11.0 step 2), what `presk autotune` and
/// `presk reprice` record per kernel and `presk explain` prints.
pub struct KernelTerms {
    pub group: String,
    pub cost: presk_kernel::cost::PlannedCost,
    /// `None` when the model has no group for the kernel.
    pub terms: Option<presk_opt::profile::Terms>,
}

/// The terms of a point's prediction, kernel by kernel (a pack's copy
/// first): [`predict`] is the sum of their totals.
pub fn point_terms(fit: &LayoutFit, machine: &Machine, dtype: presk_rt::Dtype, kernels: &[Kernel]) -> Vec<KernelTerms> {
    kernels
        .iter()
        .map(|k| {
            let cost = planned_cost_under(k, machine, &fit.rules);
            let group = group(dtype, k, cost.workers);
            let terms = fit.terms(&cost, &group);
            KernelTerms { group, cost, terms }
        })
        .collect()
}

/// The terms of `layout` at launch `l`, lowered again on `workers`; `None`
/// when the layout does not lower there.
pub fn terms_at(fit: &LayoutFit, machine: &Machine, l: &Launch, layout: &Layout, workers: usize) -> Option<Vec<KernelTerms>> {
    lower_launch(l, layout, Some(workers)).ok().map(|ks| point_terms(fit, machine, l.dtype, &ks))
}

