//! `presk memory` (plan 0.15.0 step 1, D148): what one call of an entry
//! needs in memory, from the program and its realization, never from a
//! process's resident size.
//!
//! The entry is recorded as the C export records it, on an executor that
//! runs no kernel ([`crate::layout::kernels_with_host`]), with the
//! recorder's refusals of what a C program cannot write as its result
//! lifted. From the recording ([`presk_kernel::memory::Lifetimes`]):
//! - the **parameters**, each leaf of the entry's first argument at its
//!   stored type, integer weights with their scales;
//! - the **inputs**, every argument's buffers, held by the caller;
//! - the **peak of live bytes**: the buffers the call makes, each from its
//!   first write to its last read, at the operation where they weigh most,
//!   and what is needed, the inputs and the peak;
//! - per **call**, by its path (each call's `let`, the iterations of a fold
//!   taken together, as the network diagram's boxes are, D141): its share
//!   of the peak, the bytes of what it made that are live there, and its
//!   own peak, the most what it made weighs while it runs.
//!
//! Beside them, the arena of the C program `presk export --target c`
//! writes, as the export computes it: its functions' frames, each holding
//! what it allocates until it returns (D80), so it is at least the peak
//! (the user's answer to the plan's question 1). And the size of a
//! straight-line host whose every buffer is placed at an offset by its
//! life, first fit, the check that the lifetimes are sound (M1): `presk
//! export --target c --straight-line --placed` writes it, and `make
//! c-conformance` runs it.

use std::collections::BTreeMap;

use presk_kernel::memory::Lifetimes;
use presk_kernel::record::{Made, Op, Recording, Sym, leaves};
use presk_syntax::diag::has_errors;

use crate::layout::Hosted;
use crate::{Report, Result, Stop};

/// A leaf of the parameters at rest.
#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub path: String,
    pub elements: usize,
    /// Its stored type: `f64`, `f32`, `bf16`, `int8`, `int4`.
    pub store: String,
    pub bytes: usize,
}

/// A call's part of the peak.
#[derive(Clone, Debug, PartialEq)]
pub struct CallMemory {
    /// Its path, the folds' iterations taken together.
    pub path: Vec<String>,
    /// Its function, and its path in the entry's parameters when it is
    /// given some (`blocks.attn`).
    pub func: String,
    pub params: Option<String>,
    /// How many times it runs in one call of the entry.
    pub runs: usize,
    /// The bytes of what it made that are live at the entry's peak.
    pub share: usize,
    /// The most what it made weighs while it runs, and the first operation
    /// where it does.
    pub own: usize,
    pub own_at: usize,
    /// The first operation inside it: the calls are listed in this order.
    pub first: usize,
}

/// The C program's arena, as `presk export --target c` prints it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arena {
    /// The arena: the inputs, and the entry's outputs and deepest chain of
    /// frames, each allocation rounded to 64 bytes.
    pub total: usize,
    /// The inputs' part of it.
    pub inputs: usize,
    /// The straight-line host's, which releases nothing.
    pub flat: usize,
}

impl Arena {
    /// Beside the inputs.
    pub fn beside(&self) -> usize {
        self.total - self.inputs
    }
}

/// What one call of the entry needs.
#[derive(Clone, Debug)]
pub struct Memory {
    pub entry: String,
    pub dims: Vec<(String, u64)>,
    pub realize: String,
    pub params: Vec<Param>,
    pub lifetimes: Lifetimes,
    /// The recording's operations' sites, for the peak's.
    pub sites: Vec<String>,
    pub launches: usize,
    pub calls: Vec<CallMemory>,
    /// The C program's arena, or why there is none.
    pub arena: std::result::Result<Arena, String>,
    /// The placed straight-line host's buffers, rounded to 64 bytes.
    pub placed: usize,
    /// The recording counted, which `--symbolic` compares its grid's with.
    pub recording: Option<Recording>,
}

/// The rounding of every allocation of the C programs.
pub const ALIGN: usize = 64;

impl Memory {
    pub fn parameters(&self) -> (usize, usize) {
        (self.params.iter().map(|p| p.elements).sum(), self.params.iter().map(|p| p.bytes).sum())
    }

    /// The inputs and the peak.
    pub fn needs(&self) -> usize {
        self.lifetimes.inputs + self.lifetimes.peak
    }

    /// The site of the peak's operation.
    pub fn peak_site(&self) -> &str {
        self.sites.get(self.lifetimes.at).map(String::as_str).unwrap_or("")
    }

    /// The call the peak's operation is in, as a reader names it, with the
    /// iterations of the folds it is inside; `None` in the entry's body.
    pub fn peak_call(&self) -> Option<String> {
        let path = self.lifetimes.paths.get(self.lifetimes.at)?;
        let c = self.calls.iter().find(|c| c.path == strip(path))?;
        let its: Vec<&str> = path.iter().filter_map(|p| p.split_once('[').map(|(_, i)| i.trim_end_matches(']'))).collect();
        let name = c.params.clone().unwrap_or_else(|| format!("`{}`", c.func));
        Some(if its.is_empty() { name } else { format!("{name}, iteration {}", its.join(", ")) })
    }

    /// What the entry's own body made that is live at the peak: the peak
    /// less its calls' shares.
    pub fn own_share(&self) -> usize {
        self.lifetimes.peak - self.calls.iter().filter(|c| c.path.len() == 1).map(|c| c.share).sum::<usize>()
    }
}

/// The site an operation is at: a launch's nest, or the `let` that made its
/// buffer; a host scalar's arithmetic or read at the operation before it.
pub fn sites(rec: &Recording) -> Vec<String> {
    let mut out = Vec::with_capacity(rec.ops.len());
    let mut last = String::new();
    for op in &rec.ops {
        let s = match op {
            Op::Launch { site, .. } => site.clone(),
            Op::Fill { buf, .. } | Op::Scalar { buf, .. } | Op::Copy { dst: buf, .. } | Op::Sample { out: buf, .. } => rec.bufs[*buf].site.clone(),
            Op::Hyper { .. } | Op::Read { .. } => last.clone(),
        };
        last = s.clone();
        out.push(s);
    }
    out
}

/// A path with the folds' iterations taken together, as the diagram's boxes
/// are.
pub fn strip(path: &[String]) -> Vec<String> {
    path.iter().map(|c| c.split_once('[').map_or(c.as_str(), |(s, _)| s).to_string()).collect()
}

/// The leaves of the entry's first argument, as buffers: their paths in the
/// parameters (`h.params`, in leaf order) by the input buffer each is.
fn param_bufs(rec: &Recording, h: &Hosted) -> BTreeMap<usize, String> {
    let mut out = BTreeMap::new();
    for (b, rb) in rec.bufs.iter().enumerate() {
        if let Made::Input { arg: 0, leaf } = rb.made
            && let Some((p, _)) = h.params.get(leaf)
        {
            out.insert(b, p.clone());
        }
    }
    out
}

/// The parameters at rest: each leaf of the entry's first argument when it
/// is a parameter tree.
fn params(rec: &Recording, h: &Hosted) -> Vec<Param> {
    param_bufs(rec, h)
        .into_iter()
        .map(|(b, path)| {
            let rb = &rec.bufs[b];
            Param { path, elements: rb.len, store: rb.store.name().to_string(), bytes: presk_kernel::memory::bytes(rec, b) }
        })
        .collect()
}

/// Each call's function, its parameters' path and its runs, by its path.
fn call_names(rec: &Recording, pbufs: &BTreeMap<usize, String>) -> BTreeMap<Vec<String>, (String, Option<String>, usize)> {
    let mut out: BTreeMap<Vec<String>, (String, Option<String>, usize)> = BTreeMap::new();
    let tree = presk_kernel::memory::call_paths(rec);
    for (c, path) in tree.iter().enumerate() {
        let Some(path) = path else { continue };
        if path.is_empty() {
            continue;
        }
        let call = &rec.calls[c];
        // The parameters it is given: the common prefix of the paths of the
        // parameter leaves its first argument views.
        let mut prefix: Option<Vec<String>> = None;
        if let Some(Sym::Record(_)) = call.args.first() {
            for leaf in leaves(&call.args[0]) {
                if let Sym::Tensor(v) = leaf
                    && let Some(p) = pbufs.get(&v.buf)
                {
                    let parts: Vec<String> = p.split('.').map(String::from).collect();
                    prefix = Some(match prefix {
                        None => parts[..parts.len().saturating_sub(1)].to_vec(),
                        Some(q) => q.iter().zip(&parts).take_while(|(a, b)| a == b).map(|(a, _)| a.clone()).collect(),
                    });
                }
            }
        }
        let e = out.entry(strip(path)).or_insert((call.func.clone(), prefix.filter(|p| !p.is_empty()).map(|p| p.join(".")), 0));
        e.2 += 1;
    }
    out
}

/// The memory of the call `h` recorded, with the C program's arena `arena`.
pub fn of_hosted(entry: &str, dims: &[(String, u64)], h: &Hosted, rec: &Recording, arena: std::result::Result<Arena, String>) -> Memory {
    let lifetimes = Lifetimes::of(rec);
    let pbufs = param_bufs(rec, h);
    let names = call_names(rec, &pbufs);
    let n = rec.ops.len();
    // Each buffer's writer's path, the iterations together.
    let writer: Vec<Option<Vec<String>>> = lifetimes.lives.iter().map(|l| l.first.filter(|_| !l.input).map(|f| strip(&lifetimes.paths[f]))).collect();
    let ops: Vec<Vec<String>> = lifetimes.paths.iter().map(|p| strip(p)).collect();
    let mut calls = Vec::new();
    for (path, (func, params, runs)) in &names {
        let mut delta = vec![0i64; n + 1];
        let mut share = 0;
        for (b, w) in writer.iter().enumerate() {
            let Some(w) = w else { continue };
            if w.len() < path.len() || w[..path.len()] != path[..] {
                continue;
            }
            let l = lifetimes.lives[b];
            let f = l.first.unwrap_or(0);
            delta[f] += l.bytes as i64;
            delta[l.last + 1] -= l.bytes as i64;
            if f <= lifetimes.at && l.last >= lifetimes.at {
                share += l.bytes;
            }
        }
        let (mut cur, mut own, mut own_at, mut first) = (0i64, 0usize, 0usize, None);
        for k in 0..n {
            cur += delta[k];
            let inside = ops[k].len() >= path.len() && ops[k][..path.len()] == path[..];
            if inside {
                first.get_or_insert(k);
            }
            if inside && cur as usize > own {
                (own, own_at) = (cur as usize, k);
            }
        }
        calls.push(CallMemory { path: path.clone(), func: func.clone(), params: params.clone(), runs: *runs, share, own, own_at, first: first.unwrap_or(n) });
    }
    calls.sort_by_key(|c| (c.first, c.path.len()));
    let (_, placed) = lifetimes.place(ALIGN);
    Memory {
        entry: entry.to_string(),
        dims: dims.to_vec(),
        realize: h.listing.label.clone(),
        params: params(rec, h),
        sites: sites(rec),
        launches: rec.ops.iter().filter(|o| matches!(o, Op::Launch { .. })).count(),
        calls,
        arena,
        placed,
        lifetimes,
        recording: Some(rec.clone()),
    }
}

/// The options of `presk memory`.
#[derive(Clone, Copy)]
pub struct Options<'a> {
    pub file: &'a std::path::Path,
    pub entry: &'a str,
    pub dims: &'a [String],
    pub consts: &'a [String],
    /// The realization, `f64` (the default), `f32`, `bf16` or `@FILE`.
    pub realize: &'a str,
    /// A realization plan, `@FILE`, in place of `realize`.
    pub plan: Option<&'a str>,
    pub seed: u64,
    pub remat: presk_core::ad::Remat,
}

/// `presk memory`: the entry of `o` recorded as `presk export --target c`
/// records it, its program differentiated and its plan taken as the export
/// takes them, so that the arena beside the peak is the one the export
/// prints.
pub fn memory(o: &Options, report: &mut dyn Report) -> Result<Memory> {
    counted(o, true, report)
}

/// [`memory`], the C program's arena computed only when `arena`.
fn counted(o: &Options, arena: bool, report: &mut dyn Report) -> Result<Memory> {
    let dims = crate::front::parse_dims(o.dims, report)?;
    let (sm, prog, _typed, mut core) = crate::front::typed_core(o.file, o.consts, false, report)?;
    let plan = match o.plan {
        Some(spec) => presk_rt::Plan::parse(spec),
        None => presk_rt::Realization::parse(o.realize).map(presk_rt::Plan::from_realization),
    };
    let plan = match plan {
        Ok(p) => p,
        Err(m) => {
            report.line(&format!("presk: {m}"));
            return Err(Stop(2));
        }
    };
    let mut diags = crate::ad::differentiate(&mut core, o.remat, plan.realization.groups_visible());
    diags.extend(crate::ad::stored_vjps(&core, &plan.realization));
    report.diagnostics(&diags, &sm);
    if has_errors(&diags) {
        return Err(Stop(1));
    }
    if let Err(errs) = presk_core::verify(&core) {
        for e in errs {
            report.line(&format!("presk: internal error (PSK-Core verification after AD): {e}"));
        }
        return Err(Stop(2));
    }
    crate::plan::check_plan(&plan, &core, &sm, report)?;
    let h = match crate::layout::kernels_with_host(&core, &prog, o.entry, dims.clone(), &plan, o.seed) {
        Ok(h) => h,
        Err(e) => {
            report.diagnostics(&[crate::entry::run_diag(e.code, e.message, e.span)], &sm);
            return Err(Stop(1));
        }
    };
    let (_, fdims) = match crate::entry::entry_dims(&core, &prog, o.entry, &dims) {
        Ok(x) => x,
        Err(e) => {
            report.diagnostics(&[crate::entry::interp_diag(e)], &sm);
            return Err(Stop(1));
        }
    };
    // Named as `presk export` names them.
    let named: Vec<(String, u64)> = fdims.iter().map(|(v, n)| (prog.dim_name(*v), *n)).collect();
    let m = if arena { of(o.entry, &named, &h, o.file, &plan.realization) } else { h.counted().map(|rec| of_hosted(o.entry, &named, &h, rec, Err("not counted".into()))) };
    match m {
        Some(m) => Ok(m),
        None => {
            report.line(&format!("presk memory: `{}` cannot be recorded: {}", o.entry, h.recording.as_ref().err().map(String::as_str).unwrap_or("no recording")));
            Err(Stop(1))
        }
    }
}

/// The memory of the call `h` recorded, the C program's arena computed by
/// the C export from the same recording when it takes it; `None` when the
/// recorder refused while recording.
pub fn of(entry: &str, dims: &[(String, u64)], h: &Hosted, file: &std::path::Path, realize: &presk_rt::Realization) -> Option<Memory> {
    let rec = h.counted()?;
    let arena = match &h.recording {
        Ok(rec) => c_arena(entry, dims, h, rec, file, realize),
        Err(m) => Err(m.clone()),
    };
    Some(of_hosted(entry, dims, h, rec, arena))
}

/// The arena `presk export --target c` gives the recording, without writing
/// its files.
fn c_arena(entry: &str, dims: &[(String, u64)], h: &Hosted, rec: &Recording, file: &std::path::Path, realize: &presk_rt::Realization) -> std::result::Result<Arena, String> {
    let meta = presk_export_c::host::Meta {
        file: file.display().to_string(),
        entry: entry.to_string(),
        dims: dims.to_vec(),
        consts: Vec::new(),
        realize: h.listing.label.clone(),
        seed: 0,
        weights: None,
        version: env!("CARGO_PKG_VERSION").to_string(),
        commit: None,
        exact: false,
        plan: None,
        threads: 1,
        // The weights it quantizes as it reads them, as `presk export` names
        // them.
        quant: realize.quant.iter().map(|(p, q)| (p.clone(), if q.scheme == presk_rt::QuantScheme::Pow2 { "pow2" } else { "absmax" }.to_string())).collect(),
    };
    // Where the program reads its weights from changes no allocation.
    let args: Vec<presk_export_c::host::Arg> = h.names.iter().map(|n| presk_export_c::host::Arg { name: n.clone(), weights: false }).collect();
    let x = presk_export_c::host::export(&meta, &h.listing.kernels, rec, &args, None)?;
    Ok(Arena { total: x.arena, inputs: x.arena_inputs, flat: x.arena_flat })
}

/// A term of a training step's peak, its bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Term {
    pub name: &'static str,
    pub bytes: usize,
}

/// One training step's memory under one `--remat` (plan 0.15.0 step 2,
/// D149): the calls `presk train` makes, each counted as an entry is, and
/// the step composed from them, the loop holding the parameters and the
/// optimizer's state across.
#[derive(Clone, Debug)]
pub struct StepMemory {
    pub remat: presk_core::ad::Remat,
    /// `loss.value_vjp`, `adamw.lr`, `adamw.step`, each with its memory.
    pub calls: Vec<Memory>,
    /// The parameters at their stored types, the optimizer's state (AdamW's
    /// two moments), the gradient, the gradient's other arguments (a key,
    /// the batch, the seed), each in bytes.
    pub params: usize,
    pub moments: usize,
    pub gradients: usize,
    pub others: usize,
    /// What the gradient's forward sweep made that is still live where its
    /// reverse sweep starts, at the first operation that reads the seed
    /// cotangent: the values kept for the backward pass.
    pub kept: usize,
    /// That operation, when one reads the seed.
    pub reverse_at: Option<usize>,
    /// Each call's phase: what the loop and the call hold while it runs,
    /// at the call's peak.
    pub phases: Vec<usize>,
    /// The step's peak: the largest phase, and its terms, which sum to it.
    pub peak: usize,
    pub peak_call: usize,
    pub terms: Vec<Term>,
}

/// The bytes of the inputs of argument `arg` of a recording.
fn arg_bytes(rec: &Recording, arg: usize) -> usize {
    rec.bufs.iter().enumerate().filter(|(_, b)| matches!(b.made, Made::Input { arg: a, .. } if a == arg)).map(|(i, _)| presk_kernel::memory::bytes(rec, i)).sum()
}

/// The step composed from its calls, as `presk_train::train` makes it: the
/// gradient (`value_vjp`) runs on `θ`, a key, the batch and the seed while
/// the loop holds the optimizer's state; the learning rate while it holds
/// `θ`, the state and the gradient; the update (`adamw.step`) on all three.
pub fn step_of(remat: presk_core::ad::Remat, calls: &[crate::layout::StepCall], file: &std::path::Path, realize: &presk_rt::Realization) -> Option<StepMemory> {
    let (vv, lr, up) = (calls.first()?, calls.get(1)?, calls.get(2)?);
    let mems: Vec<Memory> = calls.iter().map(|c| of(&c.func, &c.dims, &c.hosted, file, realize)).collect::<Option<_>>()?;
    let (rv, ru) = (vv.hosted.counted()?, up.hosted.counted()?);
    let params = arg_bytes(rv, 0);
    let moments = arg_bytes(ru, 1);
    let gradients = arg_bytes(ru, 2);
    let others = mems[0].lifetimes.inputs - params;
    // The reverse sweep starts where the seed cotangent, the gradient's
    // last argument, is first read.
    let seed = rv.bufs.iter().rposition(|b| matches!(b.made, Made::Input { .. }));
    let reverse_at = seed.and_then(|s| rv.ops.iter().position(|o| match o {
        Op::Launch { inputs, .. } => inputs.iter().any(|v| v.buf == s),
        Op::Copy { src, .. } | Op::Read { src, .. } | Op::Sample { src, .. } => src.buf == s,
        _ => false,
    }));
    let l0 = &mems[0].lifetimes;
    let kept_at = |k: usize| -> usize { l0.lives.iter().filter(|x| !x.input && x.first.is_some_and(|f| f < reverse_at.unwrap_or(0)) && x.last >= k).map(|x| x.bytes).sum() };
    let kept = reverse_at.map(|r| kept_at(r)).unwrap_or(0);
    let phases = vec![params + others + moments + mems[0].lifetimes.peak, params + moments + gradients + mems[1].lifetimes.peak, mems[2].lifetimes.inputs + mems[2].lifetimes.peak];
    let peak_call = (0..3).max_by_key(|i| (phases[*i], std::cmp::Reverse(*i))).unwrap_or(0);
    let peak = phases[peak_call];
    let m = &mems[peak_call];
    let at = m.lifetimes.at;
    let outs: usize = m.lifetimes.lives.iter().filter(|x| x.output && x.first.is_some_and(|f| f <= at)).map(|x| x.bytes).sum();
    let terms = match peak_call {
        0 => {
            let kept_here = if reverse_at.is_some_and(|r| at >= r) { kept_at(at) } else { 0 };
            let live = m.lifetimes.peak;
            vec![
                Term { name: "parameters", bytes: params },
                Term { name: "moments", bytes: moments },
                Term { name: "the batch, the key and the seed", bytes: others },
                Term { name: "kept for the backward", bytes: kept_here },
                Term { name: "the gradient made so far", bytes: outs },
                Term { name: "the rest", bytes: live - kept_here - outs },
            ]
        }
        1 => vec![Term { name: "parameters", bytes: params }, Term { name: "moments", bytes: moments }, Term { name: "the gradient", bytes: gradients }, Term { name: "the rest", bytes: m.lifetimes.peak }],
        _ => {
            let live = m.lifetimes.peak;
            vec![
                Term { name: "parameters", bytes: arg_bytes(ru, 0) },
                Term { name: "moments", bytes: moments },
                Term { name: "the gradient", bytes: gradients },
                Term { name: "the step's other arguments", bytes: m.lifetimes.inputs - arg_bytes(ru, 0) - moments - gradients },
                Term { name: "the new parameters and moments", bytes: outs },
                Term { name: "the rest", bytes: live - outs },
            ]
        }
    };
    let _ = lr;
    Some(StepMemory { remat, calls: mems, params, moments, gradients, others, kept, reverse_at, phases, peak, peak_call, terms })
}

/// The objective of the file's training run, by its name in PSK-Core, when
/// the file has one: the entry whose step `presk memory --step` counts.
pub fn objective(core: &presk_core::Program, typed: &presk_types::check::TypedProgram) -> Option<String> {
    let run = typed.training.first()?;
    core.funcs.iter().find(|f| matches!(f.origin, presk_core::ir::Origin::Source(id) if id == run.objective)).map(|f| f.name.clone())
}

/// The options of `presk memory --step`.
pub struct StepOptions<'a> {
    pub file: &'a std::path::Path,
    pub dims: &'a [String],
    pub consts: &'a [String],
    pub realize: &'a str,
    pub plan: Option<&'a str>,
    pub seed: u64,
}

/// `presk memory --step`: the file's training run's step at `--dim`, under
/// `store` and under `recompute`, its functions prepared as `presk train`
/// prepares them.
pub fn step(o: &StepOptions, report: &mut dyn Report) -> Result<Vec<StepMemory>> {
    use presk_core::ad::Remat;
    let dims = crate::front::parse_dims(o.dims, report)?;
    let plan = match o.plan {
        Some(spec) => presk_rt::Plan::parse(spec),
        None => presk_rt::Realization::parse(o.realize).map(presk_rt::Plan::from_realization),
    };
    let plan = match plan {
        Ok(p) => p,
        Err(m) => {
            report.line(&format!("presk: {m}"));
            return Err(Stop(2));
        }
    };
    let (sm, prog, typed, lowered) = crate::front::typed_core(o.file, o.consts, false, report)?;
    let mut out = Vec::new();
    for remat in [Remat::Store, Remat::Recompute] {
        let mut core = lowered.clone();
        let diags = crate::ad::differentiate(&mut core, remat, plan.realization.groups_visible());
        report.diagnostics(&diags, &sm);
        if has_errors(&diags) {
            return Err(Stop(1));
        }
        let fns = match presk_train::prepare_training_as(&mut core, &prog, &typed, crate::ad::options(remat, plan.realization.groups_visible())) {
            Ok(f) => f,
            Err(e) => {
                report.diagnostics(&[presk_syntax::Diagnostic::error(e.code, e.message)], &sm);
                return Err(Stop(1));
            }
        };
        let plan = presk_train::plan_for_loop(&plan, &core, &fns);
        crate::plan::check_plan(&plan, &core, &sm, report)?;
        let calls = match crate::layout::step_with_host(&core, &prog, &fns, dims.clone(), &plan, o.seed) {
            Ok(c) => c,
            Err(e) => {
                report.diagnostics(&[crate::entry::run_diag(e.code, e.message, e.span)], &sm);
                return Err(Stop(1));
            }
        };
        match step_of(remat, &calls, o.file, &plan.realization) {
            Some(s) => out.push(s),
            None => {
                report.line("presk memory --step: a call of the step could not be recorded");
                return Err(Stop(1));
            }
        }
    }
    Ok(out)
}

// ----- the polynomial (plan 0.15.0 step 3, D150) -----------------------------

use presk_types::dim::Rational;

/// A polynomial in the entry's dimensions: each term's exponents, one per
/// dimension in [`Symbolic::names`]' order, and its coefficient.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Poly {
    pub terms: BTreeMap<Vec<u32>, Rational>,
}

impl Poly {
    pub fn constant(c: usize, k: usize) -> Poly {
        let mut p = Poly::default();
        if c != 0 {
            p.terms.insert(vec![0; k], Rational::from_integer(c.into()));
        }
        p
    }

    /// Its value at `x`, one value per dimension.
    pub fn eval(&self, x: &[u64]) -> Rational {
        let mut sum = Rational::from_integer(0.into());
        for (e, c) in &self.terms {
            let mut t = c.clone();
            for (xi, ei) in x.iter().zip(e) {
                for _ in 0..*ei {
                    t *= Rational::from_integer((*xi).into());
                }
            }
            sum += t;
        }
        sum
    }

    pub fn add(&self, o: &Poly) -> Poly {
        let mut p = self.clone();
        for (e, c) in &o.terms {
            *p.terms.entry(e.clone()).or_insert_with(|| Rational::from_integer(0.into())) += c;
        }
        p.terms.retain(|_, c| *c != Rational::from_integer(0.into()));
        p
    }

    fn sub(&self, o: &Poly) -> Poly {
        let neg = Poly { terms: o.terms.iter().map(|(e, c)| (e.clone(), -c.clone())).collect() };
        self.add(&neg)
    }

    /// Whether `self ≥ o` wherever every dimension is at least 1, for a
    /// reason the coefficients show: written in `yᵢ = xᵢ − 1`, every
    /// coefficient of the difference is non-negative.
    pub fn dominates(&self, o: &Poly) -> bool {
        self.sub(o).shifted().terms.values().all(|c| *c >= Rational::from_integer(0.into()))
    }

    /// The same polynomial in `yᵢ = xᵢ − 1`: each `xᵢᵉ` expanded as
    /// `Σⱼ C(e, j) yᵢʲ`.
    fn shifted(&self) -> Poly {
        let mut p = Poly::default();
        for (e, c) in &self.terms {
            // Every exponent vector at most `e`, with its binomial weight.
            let mut acc: Vec<(Vec<u32>, Rational)> = vec![(Vec::new(), c.clone())];
            for ei in e {
                let mut next = Vec::new();
                for (v, w) in &acc {
                    let mut binom: u64 = 1;
                    for j in 0..=*ei {
                        let mut v2 = v.clone();
                        v2.push(j);
                        next.push((v2, w * Rational::from_integer(binom.into())));
                        binom = binom * u64::from(*ei - j) / u64::from(j + 1);
                    }
                }
                acc = next;
            }
            for (v, w) in acc {
                *p.terms.entry(v).or_insert_with(|| Rational::from_integer(0.into())) += w;
            }
        }
        p.terms.retain(|_, c| *c != Rational::from_integer(0.into()));
        p
    }

    /// Its terms as the printer takes them, the highest degree first.
    pub fn terms(&self, names: &[String]) -> Vec<(Rational, Vec<(String, u32)>)> {
        let mut ts: Vec<(&Vec<u32>, &Rational)> = self.terms.iter().collect();
        ts.sort_by(|a, b| b.0.iter().sum::<u32>().cmp(&a.0.iter().sum::<u32>()).then(b.0.cmp(a.0)));
        ts.into_iter().map(|(e, c)| (c.clone(), names.iter().cloned().zip(e.iter().copied()).filter(|(_, k)| *k > 0).collect())).collect()
    }
}

/// The polynomial through `values` on the grid `xs` (one list of three
/// points per dimension), of degree two at most in each dimension, by
/// interpolating along each dimension in turn. `values` is indexed by the
/// grid's points, the first dimension's index the slowest.
fn interpolate(xs: &[Vec<u64>], values: &[Rational]) -> Poly {
    let k = xs.len();
    let n = 3usize;
    let mut v: Vec<Rational> = values.to_vec();
    let r = |x: u64| Rational::from_integer(x.into());
    for (a, pts) in xs.iter().enumerate() {
        // The coefficients of the parabola through three points.
        let (x0, x1, x2) = (r(pts[0]), r(pts[1]), r(pts[2]));
        let stride = n.pow((k - 1 - a) as u32);
        for base in 0..v.len() {
            if (base / stride) % n != 0 {
                continue;
            }
            let (y0, y1, y2) = (v[base].clone(), v[base + stride].clone(), v[base + 2 * stride].clone());
            // Newton's form, then its monomial coefficients.
            let d01 = (&y1 - &y0) / (&x1 - &x0);
            let d12 = (&y2 - &y1) / (&x2 - &x1);
            let d012 = (&d12 - &d01) / (&x2 - &x0);
            let c2 = d012.clone();
            let c1 = &d01 - &d012 * (&x0 + &x1);
            let c0 = &y0 - &d01 * &x0 + &d012 * &x0 * &x1;
            v[base] = c0;
            v[base + stride] = c1;
            v[base + 2 * stride] = c2;
        }
    }
    let mut p = Poly::default();
    for (i, c) in v.into_iter().enumerate() {
        if c == r(0) {
            continue;
        }
        let e: Vec<u32> = (0..k).map(|a| ((i / n.pow((k - 1 - a) as u32)) % n) as u32).collect();
        p.terms.insert(e, c);
    }
    p
}

/// What one call needs in memory, as polynomials in the entry's dimensions:
/// each buffer's bytes, interpolated from recordings on a grid, and what
/// follows from them.
#[derive(Clone, Debug)]
pub struct Symbolic {
    /// The dimensions, as `--dim` names them, and the three values each
    /// took on the grid.
    pub names: Vec<String>,
    pub grid: Vec<Vec<u64>>,
    /// The point the polynomials were checked at, off the grid.
    pub check: Vec<u64>,
    /// Three points off the grid and what a recording counts there: the
    /// peak, its operation, the inputs and the outputs, for a reader of the
    /// polynomials to check its own arithmetic against (plan 0.15.0 step
    /// 4, M4).
    pub checks: Vec<Check>,
    /// Each buffer's bytes; the lifetimes are the document's.
    pub bufs: Vec<Poly>,
    pub inputs: Poly,
    pub outputs: Poly,
    /// The live bytes at each operation.
    pub live: Vec<Poly>,
    /// The operations whose live bytes no other's dominate, with them: the
    /// peak is their largest.
    pub peaks: Vec<(usize, Poly)>,
}

impl Symbolic {
    /// The peak at `x`, and its operation.
    pub fn peak_at(&self, x: &[u64]) -> (Rational, usize) {
        self.peaks.iter().map(|(k, p)| (p.eval(x), *k)).max_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1))).unwrap_or((Rational::from_integer(0.into()), 0))
    }
}

/// What a recording off the grid counts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Check {
    pub at: Vec<u64>,
    pub peak: usize,
    pub op: usize,
    pub inputs: usize,
    pub outputs: usize,
}

/// The values a dimension takes on the grid and at the check: multiples of
/// the document's value, and of its divisors, in this order.
fn candidates(v0: u64) -> Vec<u64> {
    let mut out: Vec<u64> = Vec::new();
    for (num, den) in [(1, 1), (2, 1), (3, 1), (4, 1), (1, 2), (3, 2), (5, 1), (1, 4), (6, 1)] {
        if (v0 * num) % den == 0 {
            let x = v0 * num / den;
            if x >= 1 && !out.contains(&x) {
                out.push(x);
            }
        }
    }
    out
}

/// One recording's structure, which every point of the grid must share:
/// the operations' kinds and sites, the buffers' makers and sites.
fn structure(rec: &Recording) -> Vec<String> {
    let mut out: Vec<String> = rec.ops.iter().map(|o| match o {
        Op::Launch { site, inputs, .. } => format!("launch {site} {}", inputs.len()),
        Op::Fill { buf, .. } => format!("fill {buf}"),
        Op::Scalar { buf, .. } => format!("scalar {buf}"),
        Op::Copy { dst, src, .. } => format!("copy {dst} {}", src.buf),
        Op::Hyper { out, .. } => format!("hyper {out}"),
        Op::Read { out, src } => format!("read {out} {}", src.buf),
        Op::Sample { out, .. } => format!("sample {out}"),
    }).collect();
    out.extend(rec.bufs.iter().map(|b| format!("{:?} {}", b.made, b.site)));
    out
}

/// `presk memory --symbolic`: the entry of `o` recorded at the document's
/// dimensions and on a grid of three values per dimension (the others at
/// the document's), each buffer's bytes interpolated, and checked at one
/// more point. Every axis of a buffer is a linear form in the dimensions
/// (chapter 2), so a buffer's elements are a product of such forms; the
/// interpolation is exact when no dimension appears in more than two of a
/// buffer's axes, which the check would show otherwise.
pub fn symbolic(o: &Options, base: &Memory, report: &mut dyn Report) -> Result<Symbolic> {
    let given = crate::front::parse_dims(o.dims, report)?;
    let names: Vec<String> = given.iter().map(|(n, _)| n.clone()).collect();
    let k = names.len();
    // A recording at `x`, quietly, without the C arena, each point once.
    let seen: std::cell::RefCell<BTreeMap<Vec<u64>, Option<Memory>>> = std::cell::RefCell::new(BTreeMap::new());
    let record = |x: &[u64]| -> Option<Memory> {
        if let Some(m) = seen.borrow().get(x) {
            return m.clone();
        }
        let dims: Vec<String> = names.iter().zip(x).map(|(n, v)| format!("{n}={v}")).collect();
        let oo = Options { dims: &dims, ..*o };
        let mut quiet = crate::Collect::default();
        let m = counted(&oo, false, &mut quiet).ok();
        seen.borrow_mut().insert(x.to_vec(), m.clone());
        m
    };
    let refuse = |m: String, report: &mut dyn Report| -> Stop {
        report.line(&format!("presk memory --symbolic: {m}"));
        Stop(1)
    };
    let v0: Vec<u64> = given.iter().map(|(_, v)| *v).collect();
    // Three values per dimension that record, the others at the document's.
    let mut grid: Vec<Vec<u64>> = Vec::new();
    let mut check: Vec<u64> = Vec::new();
    for a in 0..k {
        let mut ok = Vec::new();
        for c in candidates(v0[a]) {
            let mut x = v0.clone();
            x[a] = c;
            if c == v0[a] || record(&x).is_some() {
                ok.push(c);
            }
            if ok.len() == 4 {
                break;
            }
        }
        if ok.len() < 4 {
            return Err(refuse(format!("`{}` takes fewer than four values the entry records at ({:?})", names[a], ok), report));
        }
        check.push(ok[3]);
        ok.truncate(3);
        grid.push(ok);
    }
    let shape = structure(base.recording.as_ref().expect("a memory keeps its recording"));
    // The grid, the first dimension's index the slowest.
    let points: Vec<Vec<u64>> = (0..3usize.pow(k as u32)).map(|i| (0..k).map(|a| grid[a][(i / 3usize.pow((k - 1 - a) as u32)) % 3]).collect()).collect();
    let mut lens: Vec<Vec<Rational>> = vec![Vec::new(); base.lifetimes.lives.len()];
    for x in &points {
        let m = if *x == v0 { base.clone() } else { record(x).ok_or_else(|| refuse(format!("the entry does not record at {}", show(&names, x)), report))? };
        if structure(m.recording.as_ref().expect("a memory keeps its recording")) != shape {
            return Err(refuse(format!("the entry's operations change with its dimensions (at {}): a dimension counts a fold's iterations or decides a layout, and the sizes are not one polynomial", show(&names, x)), report));
        }
        for (b, l) in m.lifetimes.lives.iter().enumerate() {
            lens[b].push(Rational::from_integer(l.bytes.into()));
        }
    }
    let bufs: Vec<Poly> = lens.iter().map(|v| interpolate(&grid, v)).collect();
    let l = &base.lifetimes;
    let sum = |pick: &dyn Fn(usize) -> bool| -> Poly { (0..bufs.len()).filter(|b| pick(*b)).fold(Poly::default(), |acc, b| acc.add(&bufs[b])) };
    let inputs = sum(&|b| l.lives[b].input);
    let outputs = sum(&|b| l.lives[b].output);
    let live: Vec<Poly> = (0..l.live.len()).map(|op| sum(&|b| !l.lives[b].input && l.lives[b].first.is_some_and(|f| f <= op) && l.lives[b].last >= op)).collect();
    // The candidates for the peak: no other's live bytes dominate theirs.
    let mut peaks: Vec<(usize, Poly)> = Vec::new();
    for (op, p) in live.iter().enumerate() {
        if peaks.iter().any(|(_, q)| q.dominates(p)) {
            continue;
        }
        peaks.retain(|(_, q)| !p.dominates(q));
        peaks.push((op, p.clone()));
    }
    let mut s = Symbolic { names: names.clone(), grid, check: check.clone(), checks: Vec::new(), bufs, inputs, outputs, live, peaks };
    // The check, off the grid, and two more points: every dimension at its
    // check value in turn with the given values.
    let mut at = vec![check.clone()];
    for a in 0..k.min(2) {
        let mut x = v0.clone();
        x[a] = check[a];
        if !at.contains(&x) {
            at.push(x);
        }
    }
    for x in at {
        let Some(m) = record(&x) else { continue };
        let want = Rational::from_integer(m.lifetimes.peak.into());
        if s.peak_at(&x).0 != want || s.inputs.eval(&x) != Rational::from_integer(m.lifetimes.inputs.into()) || s.outputs.eval(&x) != Rational::from_integer(m.lifetimes.outputs.into()) {
            return Err(refuse(format!("at {} the polynomials give a peak of {} where the recording has {}: a dimension appears in more than two axes of a buffer", show(&names, &x), s.peak_at(&x).0, want), report));
        }
        s.checks.push(Check { at: x, peak: m.lifetimes.peak, op: m.lifetimes.at, inputs: m.lifetimes.inputs, outputs: m.lifetimes.outputs });
    }
    if s.checks.first().map(|c| &c.at) != Some(&check) {
        return Err(refuse(format!("the entry does not record at {}", show(&names, &check)), report));
    }
    Ok(s)
}

fn show(names: &[String], x: &[u64]) -> String {
    names.iter().zip(x).map(|(n, v)| format!("{n} = {v}")).collect::<Vec<_>>().join(", ")
}
