//! `presk export --target c` (plan 0.9.0 step 3, D79): an entry at bound
//! dimensions as a C program that computes the native backend's bits.
//!
//! The entry's arguments are made as `presk run` makes them (the same seed,
//! the same synthesis at 256 bits, `--weights` replacing the first). The
//! entry then runs once on the native backend with a recorder on its host
//! ([`presk_kernel::record`]), switched on for that call only; the recording
//! and the backend's kernels become the program (`presk_export_c::host`).
//! The same run gives the native outputs, and the reference interpreter the
//! exact ones, which `conformance/` holds with the inputs.
//!
//! Under a plan (`--plan @P.json`, step 5, D81) the native run is the plan's,
//! checked as `presk run` checks it, so the program launches the plan's
//! kernels, divided among its workers; `--threads N` is the native
//! executor's worker count, as for `presk run`, which decides the kernels a
//! plan leaves to the executor's default.
//!
//! With `--serve` (step 6, D83) the entry is the generation function of the
//! file's `inference serve`, as `presk serve` resolves it (or `--entry`), at
//! one sequence (`B = 1`) and the window `--dim T=N`: it is recorded, then
//! its greedy copy (every `sample` an `argmax`) on the same backend, and the
//! program holds both and `generate.c`, `presk serve`'s loop over them.

use std::path::{Path, PathBuf};

use presk_rt::{Backend, Buf, Value};
use presk_syntax::diag::{Diagnostic, has_errors};

use presk_core::ad::Remat;
use presk_syntax::SourceMap;

use crate::{Report, Result, Stop};

/// The options of `presk export --target c`.
pub struct Options<'a> {
    pub file: &'a Path,
    /// The entry; with `serve`, the generation function when not the file's.
    pub entry: Option<&'a str>,
    pub dims: &'a [String],
    pub consts: &'a [String],
    pub realize: &'a str,
    /// A realization plan, `@FILE`, in place of `realize`.
    pub plan: Option<&'a str>,
    pub weights: Option<PathBuf>,
    pub seed: u64,
    /// Whether to leave the exact reference out of `conformance/`.
    pub no_exact: bool,
    /// Whether to write step 3's straight-line host, the structured one's
    /// oracle.
    pub straight_line: bool,
    /// Whether to export the generation loop (step 6).
    pub serve: bool,
    pub out: &'a Path,
    /// The command's `--remat`.
    pub remat: Remat,
}

/// The precision of the exact reference and of the inputs' synthesis: `presk
/// run`'s default, so that the inputs are the ones it draws.
const BITS: u32 = 256;

pub fn export(o: &Options, report: &mut dyn Report) -> Result<()> {
    let mut dims = crate::front::parse_dims(o.dims, report)?;
    let (sm, prog, typed, mut core) = crate::front::typed_core(o.file, o.consts, false, report)?;
    // The plan, or the realization with the default layout, as `presk run`
    // takes them: a gradient keeps its fusion groups when its realization
    // can see them.
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

    // ----- with --serve, the generation function and its greedy copy --------
    let fns = if o.serve {
        match presk_serve::prepare_serve(&mut core, &prog, &typed, o.entry) {
            Ok(f) => Some(f),
            Err(e) => return Err(fail(report, &sm, e.code, e.message)),
        }
    } else {
        None
    };
    let entry = match (&fns, o.entry) {
        (Some(f), _) => core.func(f.generate).name.clone(),
        (None, Some(e)) => e.to_string(),
        (None, None) => unreachable!("--entry is checked"),
    };
    if let Some(f) = &fns {
        // presk serve passes the whole context, growing, unless the
        // parameters fix its length: a C program has one shape.
        let init_dims: Vec<String> = core.func(f.init).dims.iter().map(|d| prog.dim_name(*d)).collect();
        if !init_dims.contains(&f.seq_dim) {
            return Err(fail(report, &sm, "E8009", format!("`{entry}` is served over the whole context, which grows by a token at every step: `{}`'s parameters do not fix its length `{}`, and a C program has one shape. `--serve` exports a model whose parameters fix it, served over a sliding window of that many tokens", f.model, f.seq_dim)));
        }
        match dims.iter().find(|(n, _)| *n == f.batch_dim) {
            Some((_, 1)) => {}
            Some((n, v)) => {
                report.line(&format!("presk export --serve: `--dim {n}={v}`, but presk serve generates one sequence: {n} is 1"));
                return Err(Stop(2));
            }
            None => dims.push((f.batch_dim.clone(), 1)),
        }
    }
    let realize = plan.realization.clone();
    crate::plan::check_plan(&plan, &core, &sm, report)?;

    // ----- the arguments, as `presk run` makes them -------------------------
    let opts = presk_interp::RunOptions { entry: entry.clone(), dims, seed: o.seed, bits: BITS, digits: 12 };
    let (func, fdims, inputs) = match presk_interp::run::prepare_run(&core, &prog, &opts) {
        Ok(x) => x,
        Err(e) if e.message.contains("unbound dimension") || e.message.contains("open prefix") => {
            return Err(fail(report, &sm, "E8003", format!("`{entry}` is exported at bound dimensions only: {}", e.message)));
        }
        Err(e) => {
            report.diagnostics(&[crate::entry::interp_diag(e)], &sm);
            return Err(Stop(1));
        }
    };
    let names: Vec<String> = inputs.iter().map(|(n, _)| n.clone()).collect();
    let mut args: Vec<Value> = match inputs.iter().map(|(_, v)| Value::from_interp(v)).collect::<presk_rt::Result<_>>() {
        Ok(a) => a,
        Err(e) => return Err(fail(report, &sm, e.code, e.message)),
    };
    // The first argument, the parameters of a module, is the weights: read
    // from the checkpoint given, or written from the ones synthesized, as
    // `presk train` writes its checkpoints (`params M`, M the module's
    // name). A stack or a composition of modules, which `presk train` does
    // not train, is an input like the others.
    let params_module = core.func(func).params.first().and_then(|p| match &p.ty {
        presk_types::ty::Ty::Params(presk_types::ty::ModRef::Inst { def, .. }) => Some(prog.module(*def).name.clone()),
        _ => None,
    });
    let weights_type = match (&o.weights, &params_module) {
        (Some(w), Some(_)) => {
            match presk_rt::load_params(w, &args[0]) {
                Ok(v) => args[0] = v,
                Err(e) => return Err(fail(report, &sm, e.code, format!("{}: {}", w.display(), e.message))),
            }
            match presk_data::Reader::open(w) {
                Ok(r) => Some(r.header().sample_ty.clone()),
                Err(e) => return Err(fail(report, &sm, e.code, format!("{}: {}", w.display(), e.message))),
            }
        }
        (Some(_), None) => {
            report.line(&format!("presk export: --weights: the first argument of `{entry}` is not a parameter tree"));
            return Err(Stop(2));
        }
        (None, Some(m)) => Some(format!("params {m}")),
        (None, None) => None,
    };
    // The arguments as read, before a realization quantizes the weights:
    // what `conformance/` holds, since the C program quantizes them as it
    // reads them, as `presk run` does (step 7). Saved after quantization,
    // they would be `f32(q·s)`, whose scales need not be the same.
    let read = args.clone();
    if let Err(e) = realize.quantize_args(&mut args) {
        return Err(fail(report, &sm, e.code, e.message));
    }

    // ----- the native run, recorded -----------------------------------------
    let mut b = match presk_cranelift::native_backend(&core, &prog, &realize) {
        Ok(b) => b,
        Err(e) => {
            report.line(&format!("presk export: the native backend is unavailable, and the C program is its copy: {}", e.message));
            return Err(Stop(2));
        }
    };
    b.set_plan(plan.clone());
    let native = b.prepare(func, &fdims, &realize).and_then(|p| {
        b.recorder = Some(Box::new(presk_kernel::record::Recorder::named(names.clone())));
        b.call(&p, args.clone())
    });
    let recorder = b.recorder.take();
    let native = match native {
        Ok(v) => v,
        Err(e) => return Err(fail(report, &sm, e.code, e.message)),
    };
    let rec = finish(recorder, report, &sm)?;
    // The greedy copy, on the same backend, so that the two share kernels,
    // with the temperature `presk serve` gives it, 1.
    let greedy = match &fns {
        Some(f) => {
            let mut gargs = args.clone();
            if f.temperature {
                gargs[if f.key { 3 } else { 2 }] = Value::F64(1.0);
            }
            let ran = b.prepare(f.greedy, &fdims, &realize).and_then(|p| {
                b.recorder = Some(Box::new(presk_kernel::record::Recorder::named(names.clone())));
                b.call(&p, gargs)
            });
            let recorder = b.recorder.take();
            if let Err(e) = ran {
                return Err(fail(report, &sm, e.code, e.message));
            }
            Some(finish(recorder, report, &sm)?)
        }
        None => None,
    };

    // ----- the exact reference ------------------------------------------------
    // At a model's full size it takes hours: `--no-exact` leaves it out.
    let exact = if o.no_exact {
        None
    } else {
        let mut ib = presk_rt::InterpBackend::new(&core, &prog, BITS);
        match ib.prepare(func, &fdims, &realize).and_then(|p| ib.call(&p, args.clone())) {
            Ok(v) => Some(v),
            Err(e) => return Err(fail(report, &sm, e.code, e.message)),
        }
    };

    // ----- the program ----------------------------------------------------------
    let dims_named: Vec<(String, u64)> = fdims.iter().map(|(v, n)| (prog.dim_name(*v), *n)).collect();
    let weights_abs = o.weights.as_ref().map(|w| std::fs::canonicalize(w).unwrap_or_else(|_| w.clone()).display().to_string());
    let meta = presk_export_c::host::Meta {
        file: o.file.display().to_string(),
        entry: entry.clone(),
        dims: dims_named,
        consts: o.consts.to_vec(),
        realize: realize.label(),
        seed: o.seed,
        weights: weights_abs,
        version: env!("CARGO_PKG_VERSION").to_string(),
        commit: commit(o.file),
        exact: exact.is_some(),
        plan: o.plan.map(|p| p.trim_start_matches('@').to_string()),
        threads: b.exec.workers().unwrap_or(1),
        quant: realize.quant.iter().map(|(p, q)| (p.clone(), if q.scheme == presk_rt::QuantScheme::Pow2 { "pow2" } else { "absmax" }.to_string())).collect(),
    };
    let arg_roles: Vec<presk_export_c::host::Arg> = names.iter().enumerate().map(|(i, n)| presk_export_c::host::Arg { name: n.clone(), weights: i == 0 && params_module.is_some() }).collect();
    let serve = fns.as_ref().map(|f| presk_export_c::host::Serve {
        model: f.model.clone(),
        vocab: match &args[if f.key { 2 } else { 1 }] {
            Value::Idx { bound, .. } => *bound,
            _ => 0,
        },
        seq_dim: f.seq_dim.clone(),
        window: fdims.iter().find(|(v, _)| prog.dim_name(**v) == f.seq_dim).map(|(_, n)| *n).unwrap_or(0),
        key: f.key.then_some(1),
        ctx: if f.key { 2 } else { 1 },
        temperature: f.temperature.then_some(if f.key { 3 } else { 2 }),
        default_temperature: f.default_temperature.unwrap_or(1.0),
        greedy: core.func(f.greedy).name.clone(),
    });
    let exported = match (&serve, &greedy) {
        (Some(s), Some(g)) => presk_export_c::host::export_serve(&meta, &b.kernels, &rec, g, s, &arg_roles, weights_type.as_deref(), o.straight_line),
        _ if o.straight_line => presk_export_c::host::export_flat(&meta, &b.kernels, &rec, &arg_roles, weights_type.as_deref()),
        _ => presk_export_c::host::export(&meta, &b.kernels, &rec, &arg_roles, weights_type.as_deref()),
    };
    let exported = match exported {
        Ok(x) => x,
        Err(m) => {
            report.line(&format!("presk: internal error of the C export: {m}"));
            return Err(Stop(2));
        }
    };
    if let Err(m) = write_all(o, &exported, &read, params_module.as_deref(), &native, exact.as_ref()) {
        report.line(&format!("presk export: {m}"));
        return Err(Stop(2));
    }
    let split = if exported.split == 0 { "on one thread".to_string() } else { format!("{} of them divided among up to {} threads", exported.split, exported.max_workers) };
    let host = if o.straight_line {
        "straight-line".to_string()
    } else {
        let plural = |n: usize| if n == 1 { "" } else { "s" };
        format!("{} function{} besides the entry's and {} loop{}", exported.functions, plural(exported.functions), exported.loops, plural(exported.loops))
    };
    let arena = if o.straight_line { format!("{} bytes", exported.arena) } else { format!("{} bytes ({} released nothing)", exported.arena, exported.arena_flat) };
    let generate = match &serve {
        Some(s) => format!(", with `{}` and `generate`, presk serve's loop", s.greedy),
        None => String::new(),
    };
    let quantized = match exported.quantized {
        0 => String::new(),
        n => format!(", {n} lea{} of integer weights quantized as {} read", if n == 1 { "f" } else { "ves" }, if n == 1 { "it is" } else { "they are" }),
    };
    report.line(&format!(
        "presk export: `{}` at {} in {}{}{quantized}: {} launches ({} dead, {split}) of {} kernels, {host}, an arena of {arena}{generate}, into {}; `make -C {} check` builds it and compares it with the native backend",
        entry,
        if meta.dims.is_empty() { "no dimension".to_string() } else { meta.dims.iter().map(|(n, v)| format!("{n} = {v}")).collect::<Vec<_>>().join(", ") },
        meta.realize,
        meta.plan.as_ref().map(|p| format!(" under the plan {p}")).unwrap_or_default(),
        exported.launches,
        exported.dead,
        exported.kernels,
        o.out.display(),
        o.out.display()
    ));
    Ok(())
}

/// The options of `presk export --target c --mode scop` (step 8, D85).
pub struct ScopOptions<'a> {
    pub file: &'a Path,
    pub entry: &'a str,
    pub dims: &'a [String],
    pub consts: &'a [String],
    pub realize: &'a str,
    pub seed: u64,
    pub out: &'a Path,
    /// The command's `--remat`.
    pub remat: Remat,
}

/// `presk export --target c --mode scop` (plan 0.9.0 §3.7, step 8, D85):
/// the entry's nests captured from their shapes as `presk plan` captures
/// them (D65), each lowered at point zero ([`crate::layout::point_zero`]) and
/// printed as a standalone C function between `#pragma scop` and `#pragma
/// endscop` (`presk_export_c::scop`), with a driver that times each one.
/// Each nest also runs natively at point zero on the inputs the driver
/// draws, and `conformance/native.bin` holds its outputs, which `make check`
/// compares with the C's, bit for bit.
pub fn export_scop(o: &ScopOptions, report: &mut dyn Report) -> Result<()> {
    use presk_export_c::scop;
    let realize = match presk_rt::Realization::parse(o.realize) {
        Ok(r) => r,
        Err(m) => {
            report.line(&format!("presk: {m}"));
            return Err(Stop(2));
        }
    };
    if !realize.quant.is_empty() {
        report.line(&format!("presk export --mode scop: `{}` quantizes weights, and the SCoP mode prints nests over arrays of reals and indices: a nest reading integer weights loads `f32(q)·s` from packed integers and their scales", o.realize));
        return Err(Stop(2));
    }
    let plan = presk_rt::Plan::from_realization(realize.clone());
    let captured = crate::layout::capture(o.file, o.entry, o.dims, o.consts, &plan, o.seed, &[], crate::layout::Capture::Shapes, o.remat, report)?;
    // One function per nest, shape and type, named after the nest.
    let mut taken: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut nests: Vec<scop::Nest> = Vec::new();
    for l in &captured.launches {
        let zero = crate::layout::point_zero(&presk_kernel::plan::launch_kind(l));
        let mut kernels = match presk_kernel::plan::lower_launch(l, &zero, None) {
            Ok(k) => k,
            Err(m) => {
                report.line(&format!("presk: internal error: `{}` does not lower at point zero: {m}", l.nest));
                return Err(Stop(2));
            }
        };
        let (Some(kernel), true) = (kernels.pop(), kernels.is_empty()) else {
            report.line(&format!("presk: internal error: `{}` lowers to several kernels at point zero", l.nest));
            return Err(Stop(2));
        };
        let base = presk_export_c::host::c_ident(&l.nest);
        let n = taken.entry(base.clone()).or_insert(0);
        *n += 1;
        let name = if *n == 1 { base } else { format!("{base}_{n}") };
        let shape = l.operands.iter().map(|op| format!("[{}]", op.axes.iter().map(|a| a.len.to_string()).collect::<Vec<_>>().join(", "))).collect::<Vec<_>>().join(" ");
        nests.push(scop::Nest { name, nest: l.nest.clone(), shape, launches: l.count, kernel });
    }
    // Each nest natively at point zero, on the inputs the driver draws.
    let mut exec = match presk_cranelift::NativeExec::new() {
        Ok(e) => e,
        Err(e) => {
            report.line(&format!("presk export: the native backend is unavailable, and `conformance/native.bin` is its outputs: {}", e.message));
            return Err(Stop(2));
        }
    };
    let mut native: Vec<u8> = Vec::new();
    for (i, n) in nests.iter().enumerate() {
        let k = &n.kernel;
        let bufs: Vec<Buf> = scop::draw_inputs(k, o.seed, i)
            .into_iter()
            .map(|d| match d {
                scop::Drawn::Real(v) => Buf::of(k.dtype, &v),
                scop::Drawn::Idx(v) => Buf::I64(v.into()),
            })
            .collect();
        let refs: Vec<&Buf> = bufs.iter().collect();
        let mut out = presk_kernel::exec::output_buf(k);
        if let Err(e) = presk_kernel::KernelExec::run(&mut exec, i, k, &refs, &mut out) {
            report.line(&format!("presk export: `{}` on the native backend: {}", n.nest, e.message));
            return Err(Stop(2));
        }
        match &out {
            Buf::F64(v) => v.iter().for_each(|x| native.extend(x.to_le_bytes())),
            Buf::F32(v) => v.iter().for_each(|x| native.extend(x.to_le_bytes())),
            Buf::Bf16(v) => v.iter().for_each(|x| native.extend(x.to_le_bytes())),
            Buf::I64(v) => v.iter().for_each(|x| native.extend(x.to_le_bytes())),
            other => {
                report.line(&format!("presk: internal error: `{}` gives a {} output", n.nest, other.dtype()));
                return Err(Stop(2));
            }
        }
    }
    let dims = crate::front::parse_dims(o.dims, report)?;
    let meta = scop::Meta {
        file: o.file.display().to_string(),
        entry: o.entry.to_string(),
        dims,
        consts: o.consts.to_vec(),
        realize: realize.label(),
        seed: o.seed,
        version: env!("CARGO_PKG_VERSION").to_string(),
        commit: commit(o.file),
    };
    let files = match scop::export(&meta, &nests) {
        Ok(f) => f,
        Err(m) => {
            report.line(&format!("presk: internal error of the SCoP mode: {m}"));
            return Err(Stop(2));
        }
    };
    let written = (|| -> std::result::Result<(), String> {
        for dir in ["nests", "conformance"] {
            let d = o.out.join(dir);
            std::fs::create_dir_all(&d).map_err(|e| format!("cannot create {}: {e}", d.display()))?;
        }
        for (name, text) in &files {
            let path = o.out.join(name);
            std::fs::write(&path, text).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        }
        let path = o.out.join("conformance/native.bin");
        std::fs::write(&path, &native).map_err(|e| format!("cannot write {}: {e}", path.display()))
    })();
    if let Err(m) = written {
        report.line(&format!("presk export: {m}"));
        return Err(Stop(2));
    }
    let statics = nests.iter().filter(|n| scop::static_control(&n.kernel).is_ok()).count();
    let names = nests.iter().map(|n| n.nest.as_str()).collect::<std::collections::BTreeSet<_>>().len();
    report.line(&format!(
        "presk export: `{}` at {} in {}: {names} nests at {} shapes at point zero, a C function each, {statics} of them static-control parts and {} not (`nests.json` says why), into {}; `make -C {} check` builds them and compares each with the native backend",
        o.entry,
        if meta.dims.is_empty() { "no dimension".to_string() } else { meta.dims.iter().map(|(n, v)| format!("{n} = {v}")).collect::<Vec<_>>().join(", ") },
        meta.realize,
        nests.len(),
        nests.len() - statics,
        o.out.display(),
        o.out.display()
    ));
    Ok(())
}

/// An error of the export as a diagnostic, reported: exit 1.
fn fail(report: &mut dyn Report, sm: &SourceMap, code: &'static str, m: String) -> Stop {
    report.diagnostics(&[Diagnostic::error(code, m)], sm);
    Stop(1)
}

/// The recording a recorder made, or the error it found.
fn finish(recorder: Option<Box<presk_kernel::record::Recorder>>, report: &mut dyn Report, sm: &SourceMap) -> Result<presk_kernel::record::Recording> {
    match recorder.map(|r| r.finish()) {
        Some(Ok(r)) => Ok(r),
        Some(Err(r)) if r.code == presk_kernel::record::INTERNAL => {
            report.line(&format!("presk: internal error of the recorder: {}", r.message));
            Err(Stop(2))
        }
        Some(Err(r)) => Err(fail(report, sm, r.code, r.message)),
        None => unreachable!("the recorder was set"),
    }
}

/// The commit of the repository holding `file`, `-dirty` when its tree has
/// changes; `None` outside a repository.
fn commit(file: &Path) -> Option<String> {
    let dir = file.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let git = |args: &[&str]| std::process::Command::new("git").arg("-C").arg(dir).args(args).output().ok().filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    let head = git(&["rev-parse", "--short", "HEAD"])?;
    let dirty = git(&["status", "--porcelain"]).is_some_and(|s| !s.is_empty());
    Some(if dirty { format!("{head}-dirty") } else { head })
}

fn collect(v: &Value, out: &mut Vec<Value>) {
    match v {
        Value::Tuple(vs) => vs.iter().for_each(|x| collect(x, out)),
        Value::Record(fs) => fs.iter().for_each(|(_, x)| collect(x, out)),
        leaf => out.push(leaf.clone()),
    }
}

/// The tensor leaves of `values` as one sample of a closed type: reals as
/// `f64` (exact inputs rounded once, as the native backend rounds them
/// before its own conversion), indices as they are.
fn sample(values: &[&Value]) -> Vec<presk_data::Value> {
    let mut leaves = Vec::new();
    for v in values {
        collect(v, &mut leaves);
    }
    leaves
        .into_iter()
        .filter_map(|l| match l {
            Value::Real(t) => Some(presk_data::Value::Real(t.data.to_f64())),
            Value::Idx { t, .. } => Some(presk_data::Value::Idx(match &t.data {
                Buf::I64(v) => v.iter().map(|x| *x as u64).collect(),
                other => other.to_f64().iter().map(|x| *x as u64).collect(),
            })),
            _ => None,
        })
        .collect()
}

fn write_data(path: &Path, f: &presk_export_c::host::DataFile, leaves: Vec<presk_data::Value>) -> std::result::Result<(), String> {
    let layout = presk_data::Layout::from_type_text(&f.ty).map_err(|e| format!("{}: {e}", path.display()))?;
    let v = if leaves.len() == 1 && !f.ty.starts_with('(') { leaves.into_iter().next().unwrap() } else { presk_data::Value::Tuple(leaves) };
    v.check(&layout).map_err(|e| format!("{}: {e}", path.display()))?;
    presk_data::write(path, &f.ty, &layout, std::slice::from_ref(&v)).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(())
}

fn write_all(o: &Options, x: &presk_export_c::host::Exported, args: &[Value], params: Option<&str>, native: &Value, exact: Option<&Value>) -> std::result::Result<(), String> {
    let conf = o.out.join("conformance");
    std::fs::create_dir_all(&conf).map_err(|e| format!("cannot create {}: {e}", conf.display()))?;
    for (name, text) in &x.files {
        let path = o.out.join(name);
        std::fs::write(&path, text).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    }
    // The weights synthesized, as `presk train` writes a checkpoint; a
    // checkpoint given is read where it is.
    if let (Some(m), None) = (params, &o.weights) {
        let path = conf.join("weights.psk.d");
        presk_rt::save_params(&path, m, &args[0]).map_err(|e| format!("{}: {}", path.display(), e.message))?;
    }
    let rest: Vec<&Value> = args.iter().skip(usize::from(params.is_some())).collect();
    if let Some(f) = &x.inputs {
        write_data(&conf.join("inputs.psk.d"), f, sample(&rest))?;
    }
    write_data(&conf.join("native.psk.d"), &x.outputs, sample(&[native]))?;
    let exact_path = conf.join("exact.psk.d");
    match exact {
        Some(e) => write_data(&exact_path, &x.outputs, sample(&[e]))?,
        None if exact_path.exists() => std::fs::remove_file(&exact_path).map_err(|e| format!("cannot remove {}: {e}", exact_path.display()))?,
        None => {}
    }
    Ok(())
}
