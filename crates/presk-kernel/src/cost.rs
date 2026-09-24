//! What a kernel costs before it runs: its *work* (operations of the loop
//! nest, counted from the IR at its concrete shapes) and its *traffic* (bytes
//! of every buffer it reads and writes). These are the `W` and `Q` of the
//! cost-model document (§1, §3), computed rather than measured; the measured
//! side is [`TimedExec`], which times every launch of an executor so that
//! `presk bench --calibrate` can fit a machine profile to them.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::Instant;

use presk_rt::{Buf, RtError};
use serde::{Deserialize, Serialize};

use crate::host::KernelExec;
use crate::ir::{BinOp, BufKind, Dtype, Expr, Kernel, Stmt, Threads, UnOp};

type Result<T> = std::result::Result<T, RtError>;

/// Operations weighed by kind: arithmetic 1, a transcendental function 8
/// (roughly the cost of a `libm` call in multiplies).
pub const TRANSCENDENTAL_OPS: f64 = 8.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KernelCost {
    /// Weighted operation count over the whole loop nest.
    pub work: f64,
    /// Bytes read and written: every input buffer once, the output once.
    pub bytes: f64,
}

fn elem_bytes(kind: BufKind, dtype: Dtype) -> f64 {
    match (kind, dtype) {
        (BufKind::Idx, _) => 8.0,
        // The integers; the scales are one per channel and are not counted.
        (BufKind::Quant { bits, .. }, _) => bits as f64 / 8.0,
        (BufKind::Real, Dtype::F64) => 8.0,
        (BufKind::Real, Dtype::F32) => 4.0,
        (BufKind::Real, Dtype::Bf16) => 2.0,
    }
}

fn expr_ops(e: &Expr) -> f64 {
    match e {
        Expr::F(_) | Expr::I(_) | Expr::Iv(_) | Expr::Sv(_) | Expr::Si(_) => 0.0,
        // Address arithmetic is not work.
        Expr::Load(_, _) | Expr::LoadI(_, _) => 0.0,
        // A lane is an element, so a vector operation is `width` of them.
        Expr::Vv(_) | Expr::VLoad(_, _) => 0.0,
        Expr::VSplat(x) => expr_ops(x),
        Expr::VBin(_, a, b) => 1.0 + expr_ops(a) + expr_ops(b),
        Expr::Bin(op, a, b) => {
            let own = match op {
                crate::ir::BinOp::IAdd | crate::ir::BinOp::IMul => 0.0,
                _ => 1.0,
            };
            own + expr_ops(a) + expr_ops(b)
        }
        Expr::Un(op, a) => {
            let own = match op {
                UnOp::Neg | UnOp::Abs => 1.0,
                UnOp::Exp | UnOp::Log | UnOp::Tanh | UnOp::Sqrt | UnOp::Sin | UnOp::Cos | UnOp::Erf => TRANSCENDENTAL_OPS,
            };
            own + expr_ops(a)
        }
        Expr::Cmp(_, a, b) => 1.0 + expr_ops(a) + expr_ops(b),
        Expr::Select(c, a, b) => 1.0 + expr_ops(c) + expr_ops(a) + expr_ops(b),
        Expr::ToF(a) | Expr::ToI(a) => expr_ops(a),
    }
}

fn stmts_ops(stmts: &[Stmt], lanes: f64) -> f64 {
    stmts
        .iter()
        .map(|s| match s {
            Stmt::For { bound, body, .. } => *bound as f64 * stmts_ops(body, lanes),
            Stmt::Set { value, .. } | Stmt::SetI { value, .. } | Stmt::SetL { value, .. } => expr_ops(value),
            // A vector statement covers `lanes` elements, so it is `lanes`
            // times the work of one. Counting it as one made the fitted `π`
            // of a vector kernel come out *below* its scalar form's, which
            // is the opposite of what the hardware does.
            Stmt::VSet { value, .. } => lanes * expr_ops(value),
            Stmt::VStore { index, value, .. } => lanes * expr_ops(value) + index.iter().map(expr_ops).sum::<f64>(),
            Stmt::Store { index, value, .. } | Stmt::StoreI { index, value, .. } => expr_ops(value) + index.iter().map(expr_ops).sum::<f64>(),
            // The branch is counted as taken: an upper bound.
            Stmt::If { cond, then } => expr_ops(cond) + stmts_ops(then, lanes),
        })
        .sum()
}

/// The work and traffic of a kernel at its shapes.
pub fn kernel_cost(k: &Kernel) -> KernelCost {
    let bytes = k.inputs.iter().map(|b| b.numel() as f64 * elem_bytes(b.kind, k.dtype)).sum::<f64>() + k.output.numel() as f64 * elem_bytes(k.output.kind, k.dtype);
    KernelCost { work: stmts_ops(&k.body, k.lanes as f64), bytes }
}

/// Below this many weighted operations the executor's default split keeps a
/// kernel on one worker: spawning costs tens of microseconds, which only a
/// big kernel earns back. A split a plan states is not held to it.
pub const PARALLEL_MIN_WORK: f64 = 4.0e6;

// ----- the machine description and the planned cost (0.7.0 step 4) ----------------

/// The machine description `M` of the language reference (§6.8): what a cost
/// model may know of the machine besides its fitted constants. Read from the
/// operating system for the first CPU the process may run on, so that a run
/// pinned to some cores describes those cores: on falicon a pinned X925 has
/// 2 MiB of L2 where `cpu0`, an A725, has 512 KiB (D60).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Machine {
    /// Workers a split may use, `p_M`: the CPUs the process may run on.
    pub workers: usize,
    /// Data cache of one worker, in bytes, private: level 1 and level 2.
    pub l1: usize,
    pub l2: usize,
    /// The level-3 cache, in bytes, and how many of the process's CPUs share
    /// one: a worker's share is `l3 / min(p, l3_sharing)`.
    pub l3: usize,
    pub l3_sharing: usize,
    /// Vector registers, and the bytes of one.
    pub vector_registers: usize,
    pub vector_bytes: usize,
    /// Where the sizes were read (`cpu5`), or `defaults`.
    pub source: String,
}

impl Machine {
    /// The machine this process runs on.
    pub fn detect() -> Machine {
        let allowed = allowed_cpus();
        // As the native executor counts them: `PRESK_THREADS` (`--threads`),
        // else the CPUs the process may run on.
        let workers = std::env::var("PRESK_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
            .unwrap_or(allowed.len().max(1))
            .max(1);
        let vector_registers = if cfg!(target_arch = "x86_64") { 16 } else { 32 };
        let mut m = Machine { workers, l1: 32 << 10, l2: 1 << 20, l3: 8 << 20, l3_sharing: workers, vector_registers, vector_bytes: 16, source: "defaults".into() };
        let Some(cpu) = allowed.first().copied() else { return m };
        let dir = format!("/sys/devices/system/cpu/cpu{cpu}/cache");
        let Ok(entries) = std::fs::read_dir(&dir) else { return m };
        let read = |p: &std::path::Path, f: &str| std::fs::read_to_string(p.join(f)).map(|t| t.trim().to_string()).unwrap_or_default();
        for e in entries.flatten() {
            let p = e.path();
            if !p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("index")) {
                continue;
            }
            let (level, kind, size) = (read(&p, "level"), read(&p, "type"), parse_size(&read(&p, "size")));
            let Some(size) = size else { continue };
            match (level.as_str(), kind.as_str()) {
                ("1", "Data") => m.l1 = size,
                ("2", "Data" | "Unified") => m.l2 = size,
                ("3", "Data" | "Unified") => {
                    m.l3 = size;
                    let shared = parse_cpu_list(&read(&p, "shared_cpu_list"));
                    m.l3_sharing = shared.iter().filter(|c| allowed.contains(c)).count().max(1);
                }
                _ => {}
            }
        }
        m.source = format!("cpu{cpu}");
        m
    }

    /// The level-3 cache one of `p` workers may count on.
    pub fn l3_per_worker(&self, p: usize) -> usize {
        self.l3 / p.min(self.l3_sharing).max(1)
    }
}

/// `64K`, `2048K`, `8M`: bytes.
fn parse_size(s: &str) -> Option<usize> {
    let (num, unit) = s.split_at(s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len()));
    let n: usize = num.parse().ok()?;
    Some(match unit {
        "" => n,
        "K" => n << 10,
        "M" => n << 20,
        "G" => n << 30,
        _ => return None,
    })
}

/// `5-9,15-19`: the CPUs.
fn parse_cpu_list(s: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => {
                if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
                    out.extend(a..=b);
                }
            }
            None => out.extend(part.parse::<usize>().ok()),
        }
    }
    out
}

/// The CPUs this process may run on (`Cpus_allowed_list`), in order.
fn allowed_cpus() -> Vec<usize> {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|t| t.lines().find_map(|l| l.strip_prefix("Cpus_allowed_list:").map(|v| parse_cpu_list(v.trim()))))
        .unwrap_or_default()
}

/// What the layout cost model reads of a kernel (plan 0.7.0 §3.5): the work
/// and traffic of **one worker's** part of it, its traffic split by the cache
/// level it comes from, and the register pressure of its busiest innermost
/// loop. Two tilings of one nest have the same `work` and differ in
/// `traffic`; that is the difference 0.6.0's [`KernelCost`] could not see.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlannedCost {
    /// One worker's weighted operations, spilled registers included.
    pub work: f64,
    /// 0.6.0's figure: every input once, the output once.
    pub bytes: f64,
    /// One worker's bytes from L2, from L3 and from memory.
    pub traffic: [f64; 3],
    /// Real and vector variables live in the busiest innermost loop, and how
    /// many of them exceed the machine's vector registers.
    pub regs: usize,
    pub spilled: usize,
    /// The split the cost is of: the workers, and the depth of the loop.
    pub workers: usize,
    pub depth: usize,
}

/// The planned cost of a kernel on a machine. The split is the kernel's own
/// ([`Threads`]); the executor's default is modelled as the executor decides
/// it — the outermost parallel loop over every worker, above
/// [`PARALLEL_MIN_WORK`]. A worker's part is the kernel with the loop at the
/// split's depth cut to a worker's share of its iterations.
pub fn planned_cost(k: &Kernel, m: &Machine) -> PlannedCost {
    let (depth, p) = match k.threads {
        Threads::At { depth, workers } => (depth, k.parallel_at(depth).map_or(1, |n| workers.min(n).max(1))),
        Threads::One => (0, 1),
        Threads::Default => match k.parallel_outer() {
            Some(n) if m.workers > 1 && n >= 2 && kernel_cost(k).work >= PARALLEL_MIN_WORK => (0, m.workers.min(n)),
            _ => (0, 1),
        },
    };
    let worker = if p > 1 { share(k, depth, p) } else { k.clone() };
    let traffic = [traffic(&worker, m.l1), traffic(&worker, m.l2), traffic(&worker, m.l3_per_worker(p))];
    let (regs, iterations) = pressure(&worker.body, 1.0);
    let spilled = regs.saturating_sub(m.vector_registers);
    // Two level-1 accesses per iteration of the busiest loop and per spilled
    // register, counted as operations.
    let spill = 2.0 * spilled as f64 * iterations;
    PlannedCost { work: kernel_cost(&worker).work + spill, bytes: kernel_cost(k).bytes, traffic, regs, spilled, workers: p, depth }
}

/// The kernel with the loop at `depth` cut to one of `p` workers' share.
fn share(k: &Kernel, depth: usize, p: usize) -> Kernel {
    fn cut(stmts: &mut [Stmt], depth: usize, p: usize) {
        if let [Stmt::For { bound, body, .. }] = stmts {
            if depth == 0 {
                *bound = bound.div_ceil(p);
            } else {
                cut(body, depth - 1, p);
            }
        }
    }
    let mut w = k.clone();
    cut(&mut w.body, depth, p);
    w
}

/// Real and vector variables referenced in the innermost loop that
/// references the most, and how many times that loop's body runs.
fn pressure(stmts: &[Stmt], trips: f64) -> (usize, f64) {
    type Vars = std::collections::BTreeSet<(bool, usize)>;
    fn vars(e: &Expr, out: &mut Vars) {
        match e {
            Expr::Sv(v) => {
                out.insert((false, *v));
            }
            Expr::Vv(v) => {
                out.insert((true, *v));
            }
            Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) | Expr::VBin(_, a, b) => {
                vars(a, out);
                vars(b, out);
            }
            Expr::Un(_, a) | Expr::ToF(a) | Expr::ToI(a) | Expr::VSplat(a) => vars(a, out),
            Expr::Select(c, a, b) => {
                vars(c, out);
                vars(a, out);
                vars(b, out);
            }
            Expr::Load(_, ix) | Expr::LoadI(_, ix) | Expr::VLoad(_, ix) => ix.iter().for_each(|x| vars(x, out)),
            Expr::F(_) | Expr::I(_) | Expr::Iv(_) | Expr::Si(_) => {}
        }
    }
    // The variables of the statements outside any loop, through `if`s.
    fn here(stmts: &[Stmt], out: &mut Vars) {
        for s in stmts {
            match s {
                Stmt::For { .. } => {}
                Stmt::Set { var, value } => {
                    out.insert((false, *var));
                    vars(value, out);
                }
                Stmt::VSet { var, value } => {
                    out.insert((true, *var));
                    vars(value, out);
                }
                Stmt::SetI { value, .. } | Stmt::SetL { value, .. } | Stmt::Store { value, .. } | Stmt::StoreI { value, .. } | Stmt::VStore { value, .. } => vars(value, out),
                Stmt::If { cond, then } => {
                    vars(cond, out);
                    here(then, out);
                }
            }
        }
    }
    let mut best = (0usize, 0.0f64);
    let mut innermost = true;
    for s in stmts {
        if let Stmt::For { bound, body, .. } = s {
            innermost = false;
            let inner = pressure(body, trips * *bound as f64);
            if inner.0 > best.0 {
                best = inner;
            }
        }
    }
    if innermost {
        let mut v = Vars::new();
        here(stmts, &mut v);
        best = (v.len(), trips);
    }
    best
}

/// `c + Σ a_v·v` over loop variables, a loop variable a `SetL` assigns being
/// replaced by its definition; `None` for an index that is not affine (a
/// loaded value, a real).
fn affine(e: &Expr, defs: &HashMap<usize, Expr>) -> Option<(i64, HashMap<usize, i64>)> {
    match e {
        Expr::I(c) => Some((*c, HashMap::new())),
        Expr::Iv(v) => match defs.get(v) {
            Some(d) => affine(d, defs),
            None => Some((0, HashMap::from([(*v, 1)]))),
        },
        Expr::Bin(BinOp::IAdd, a, b) => {
            let (ca, mut va) = affine(a, defs)?;
            let (cb, vb) = affine(b, defs)?;
            for (v, c) in vb {
                *va.entry(v).or_insert(0) += c;
            }
            Some((ca + cb, va))
        }
        Expr::Bin(BinOp::IMul, a, b) => {
            let ((ca, va), (cb, vb)) = (affine(a, defs)?, affine(b, defs)?);
            match (va.is_empty(), vb.is_empty()) {
                (true, _) => Some((ca * cb, vb.into_iter().map(|(v, c)| (v, c * ca)).collect())),
                (_, true) => Some((ca * cb, va.into_iter().map(|(v, c)| (v, c * cb)).collect())),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Every access to a buffer in `stmts`: its index and how many consecutive
/// elements it spans along the last axis (its lanes, or one).
fn accesses(stmts: &[Stmt], buf: usize, lanes: usize, out: &mut Vec<(Vec<Expr>, usize)>) {
    fn expr(e: &Expr, buf: usize, lanes: usize, out: &mut Vec<(Vec<Expr>, usize)>) {
        match e {
            Expr::Load(b, ix) | Expr::LoadI(b, ix) => {
                if *b == buf {
                    out.push((ix.clone(), 1));
                }
            }
            Expr::VLoad(b, ix) => {
                if *b == buf {
                    out.push((ix.clone(), lanes));
                }
            }
            Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) | Expr::VBin(_, a, b) => {
                expr(a, buf, lanes, out);
                expr(b, buf, lanes, out);
            }
            Expr::Un(_, a) | Expr::ToF(a) | Expr::ToI(a) | Expr::VSplat(a) => expr(a, buf, lanes, out),
            Expr::Select(c, a, b) => {
                expr(c, buf, lanes, out);
                expr(a, buf, lanes, out);
                expr(b, buf, lanes, out);
            }
            Expr::F(_) | Expr::I(_) | Expr::Iv(_) | Expr::Sv(_) | Expr::Si(_) | Expr::Vv(_) => {}
        }
    }
    for s in stmts {
        match s {
            Stmt::For { body, .. } => accesses(body, buf, lanes, out),
            Stmt::If { cond, then } => {
                expr(cond, buf, lanes, out);
                accesses(then, buf, lanes, out);
            }
            Stmt::Store { buf: b, index, value } | Stmt::StoreI { buf: b, index, value } => {
                if *b == buf {
                    out.push((index.clone(), 1));
                }
                expr(value, buf, lanes, out);
            }
            Stmt::VStore { buf: b, index, value } => {
                if *b == buf {
                    out.push((index.clone(), lanes));
                }
                expr(value, buf, lanes, out);
            }
            Stmt::Set { value, .. } | Stmt::SetI { value, .. } | Stmt::SetL { value, .. } | Stmt::VSet { value, .. } => expr(value, buf, lanes, out),
        }
    }
}

/// The loops of `stmts`, variable and bound, and the loop variables their
/// `SetL`s assign.
fn loops(stmts: &[Stmt], bounds: &mut HashMap<usize, usize>, defs: &mut HashMap<usize, Expr>) {
    for s in stmts {
        match s {
            Stmt::For { var, bound, body } => {
                bounds.insert(*var, *bound);
                loops(body, bounds, defs);
            }
            Stmt::If { then, .. } => loops(then, bounds, defs),
            Stmt::SetL { var, value } => {
                defs.insert(*var, value.clone());
            }
            _ => {}
        }
    }
}

/// The bytes of buffer `buf` that `stmts` touch while the loops enclosing
/// them stay fixed: per axis, the span of the index over the loops inside
/// `stmts`, the union over every access, capped by the axis; the whole axis
/// for an index that is not affine.
fn footprint(k: &Kernel, stmts: &[Stmt], buf: usize, defs: &HashMap<usize, Expr>) -> f64 {
    let spec = if buf == k.out_buf() { &k.output } else { &k.inputs[buf] };
    let mut acc = Vec::new();
    accesses(stmts, buf, k.lanes, &mut acc);
    if acc.is_empty() {
        return 0.0;
    }
    let mut inside = HashMap::new();
    loops(stmts, &mut inside, &mut HashMap::new());
    let rank = spec.dims.len();
    let mut elems = 1.0;
    for a in 0..rank {
        let (mut lo, mut hi, mut whole) = (i64::MAX, i64::MIN, false);
        for (ix, span) in &acc {
            let Some((c, coef)) = ix.get(a).and_then(|e| affine(e, defs)) else {
                whole = true;
                break;
            };
            // Loops outside `stmts` are fixed, at zero: every access sees the
            // same value of them.
            let (mut l, mut h) = (c, c);
            for (v, co) in coef {
                if let Some(n) = inside.get(&v) {
                    let reach = co * (*n as i64 - 1);
                    if reach < 0 { l += reach } else { h += reach }
                }
            }
            if a + 1 == rank {
                h += *span as i64 - 1;
            }
            lo = lo.min(l);
            hi = hi.max(h);
        }
        let extent = if whole { spec.dims[a] } else { ((hi - lo + 1).max(1) as usize).min(spec.dims[a].max(1)) };
        elems *= extent as f64;
    }
    elems * elem_bytes(spec.kind, k.dtype)
}

/// The bytes a kernel brings into a cache of `capacity` bytes, every buffer
/// counted apart (plan 0.7.0 §3.5): a buffer whose footprint inside a loop
/// fits is brought in once per iteration of the loops enclosing it, and
/// otherwise the loop's body is looked at once per iteration of the loop.
pub fn traffic(k: &Kernel, capacity: usize) -> f64 {
    let mut defs = HashMap::new();
    loops(&k.body, &mut HashMap::new(), &mut defs);
    let cap = capacity as f64;
    fn walk(k: &Kernel, stmts: &[Stmt], buf: usize, times: f64, cap: f64, defs: &HashMap<usize, Expr>) -> f64 {
        let fp = footprint(k, stmts, buf, defs);
        if fp == 0.0 {
            return 0.0;
        }
        if fp <= cap {
            return times * fp;
        }
        stmts
            .iter()
            .map(|s| match s {
                Stmt::For { bound, body, .. } => {
                    let whole = footprint(k, std::slice::from_ref(s), buf, defs);
                    if whole <= cap { times * whole } else { walk(k, body, buf, times * *bound as f64, cap, defs) }
                }
                Stmt::If { then, .. } => walk(k, then, buf, times, cap, defs),
                other => times * footprint(k, std::slice::from_ref(other), buf, defs),
            })
            .sum()
    }
    (0..=k.inputs.len()).map(|b| walk(k, &k.body, b, 1.0, cap, &defs)).sum()
}

/// One timed launch: the kernel's index in the backend and the seconds it took.
pub type Samples = Rc<RefCell<Vec<(usize, f64)>>>;

/// An executor that times every launch of another one. The first launch of
/// a kernel includes its compilation on a JIT executor; the consumer drops
/// it (`presk bench --calibrate` takes the median of the later ones).
pub struct TimedExec {
    inner: Box<dyn KernelExec>,
    pub samples: Samples,
}

impl TimedExec {
    pub fn new(inner: Box<dyn KernelExec>) -> (Self, Samples) {
        let samples: Samples = Rc::new(RefCell::new(Vec::new()));
        (TimedExec { inner, samples: samples.clone() }, samples)
    }
}

impl KernelExec for TimedExec {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn workers(&self) -> Option<usize> {
        self.inner.workers()
    }

    fn run(&mut self, idx: usize, k: &Kernel, inputs: &[&Buf], out: &mut Buf) -> Result<()> {
        let t0 = Instant::now();
        let r = self.inner.run(idx, k, inputs, out);
        self.samples.borrow_mut().push((idx, t0.elapsed().as_secs_f64()));
        r
    }
}

/// Median of the launches of each kernel, the first one excluded when there
/// are several (warm-up, compilation).
pub fn median_per_kernel(samples: &[(usize, f64)], nkernels: usize) -> Vec<Option<f64>> {
    let mut per: Vec<Vec<f64>> = vec![Vec::new(); nkernels];
    for (i, t) in samples {
        if *i < nkernels {
            per[*i].push(*t);
        }
    }
    per.into_iter()
        .map(|mut v| {
            if v.len() > 1 {
                v.remove(0);
            }
            if v.is_empty() {
                return None;
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            Some(v[v.len() / 2])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BufSpec, Stmt};
    use crate::lower::{Contraction, Operand, emit_contraction};
    use presk_rt::layout::{BlockLoop, KBlock};
    use presk_rt::{Axis, Layout, ReductionOrder};

    /// A square `f32` GEMM of side `n`, lowered under `layout`.
    fn gemm(n: usize, layout: Layout) -> Kernel {
        let t = |a: usize, b: usize| Operand { axes: vec![Axis { name: None, len: a }, Axis { name: None, len: b }], kind: BufKind::Real };
        let ins = [t(n, n), t(n, n)];
        let c = Contraction::of(&presk_core::PrimOp::Dot, &ins, ReductionOrder::Sequential).unwrap();
        let kind = presk_rt::NestKind::Contraction(c.kind(Dtype::F32, Dtype::F32));
        let mut k = emit_contraction(&c, Dtype::F32, &ins, &layout.resolve(&kind)).unwrap().kernel;
        // One worker, so that two layouts compare as kernels and not as
        // splits: the executor's default would split one and not the other.
        k.threads = Threads::One;
        k
    }

    fn falicon() -> Machine {
        Machine { workers: 10, l1: 64 << 10, l2: 2 << 20, l3: 8 << 20, l3_sharing: 5, vector_registers: 32, vector_bytes: 16, source: "test".into() }
    }

    #[test]
    fn two_k_blocks_of_one_nest_differ_in_traffic_and_not_in_work() {
        let m = falicon();
        let plain = planned_cost(&gemm(512, Layout::default()), &m);
        let blocked = planned_cost(&gemm(512, Layout { kblock: Some(KBlock::Width(128)), order: Some(vec![BlockLoop::K, BlockLoop::I, BlockLoop::J]), ..Layout::default() }), &m);
        assert_eq!(plain.work, blocked.work, "a k-block moves no operation");
        assert_ne!(plain.traffic, blocked.traffic, "a k-block moves data: {:?} against {:?}", plain.traffic, blocked.traffic);
        // A taller register tile reads the right operand fewer times.
        let tall = planned_cost(&gemm(512, Layout { tile: Some((4, 16)), ..Layout::default() }), &m);
        assert_eq!(plain.work, tall.work);
        assert!(tall.traffic[0] < plain.traffic[0], "[4, 16] {:?} against [1, 8] {:?}", tall.traffic, plain.traffic);
        // In four lanes its micro-kernel holds 16 vector accumulators, 4
        // broadcasts and 4 columns of the right operand: 24 of 32 registers.
        // Without lanes, 64 accumulators, 4 and 16: 84, and 52 spill.
        assert_eq!((tall.regs, tall.spilled, plain.spilled), (24, 0, 0));
        let scalar = planned_cost(&gemm(512, Layout { tile: Some((4, 16)), lanes: Some(1), ..Layout::default() }), &m);
        assert_eq!((scalar.regs, scalar.spilled), (84, 52));
        assert!(scalar.work > tall.work, "spills are charged as work");
    }

    #[test]
    fn a_kernel_that_fits_reads_every_byte_once() {
        let mut k = gemm(16, Layout::default());
        k.threads = Threads::Default;
        let c = kernel_cost(&k);
        let p = planned_cost(&k, &falicon());
        assert_eq!(p.traffic, [c.bytes; 3], "three 16x16 f32 matrices fit every level");
        assert_eq!((p.workers, p.depth), (1, 0), "too small for the default to split");
        let mut big = gemm(512, Layout::default());
        big.threads = Threads::Default;
        assert_eq!(planned_cost(&big, &falicon()).workers, 10, "the default splits a big kernel over every worker");
    }

    #[test]
    fn a_split_costs_one_workers_share() {
        let mut k = gemm(512, Layout::default());
        let one = planned_cost(&k, &falicon());
        k.threads = Threads::At { depth: 0, workers: 4 };
        let four = planned_cost(&k, &falicon());
        assert_eq!(four.workers, 4);
        assert_eq!(four.work * 4.0, one.work, "each worker a quarter of the rows");
    }

    #[test]
    fn sysfs_sizes_and_cpu_lists_are_read() {
        assert_eq!(parse_size("64K"), Some(64 << 10));
        assert_eq!(parse_size("8M"), Some(8 << 20));
        assert_eq!(parse_size("2048K"), Some(2 << 20));
        assert_eq!(parse_cpu_list("5-9,15-19"), vec![5, 6, 7, 8, 9, 15, 16, 17, 18, 19]);
        assert_eq!(parse_cpu_list("0"), vec![0]);
        let m = Machine::detect();
        assert!(m.workers >= 1 && m.l1 > 0 && m.l2 >= m.l1 && m.l3_sharing >= 1, "{m:?}");
    }

    #[test]
    fn add_kernel_work_and_bytes() {
        let spec = BufSpec { dims: vec![2, 3], kind: BufKind::Real, names: vec![None, None] };
        let load = |b: usize| Expr::Load(b, vec![Expr::Iv(0), Expr::Iv(1)]);
        let k = Kernel {
            name: "add".into(),
            dtype: Dtype::F64,
            inputs: vec![spec.clone(), spec.clone()],
            output: spec,
            nloops: 2,
            nreal: 0,
            nint: 0,
            nvec: 0,
            lanes: 1,
            threads: Default::default(),
            body: vec![Stmt::For { var: 0, bound: 2, body: vec![Stmt::For { var: 1, bound: 3, body: vec![Stmt::Store { buf: 2, index: vec![Expr::Iv(0), Expr::Iv(1)], value: Expr::Bin(BinOp::Add, Box::new(load(0)), Box::new(load(1))) }] }] }],
        };
        let c = kernel_cost(&k);
        assert_eq!(c.work, 6.0);
        assert_eq!(c.bytes, 3.0 * 6.0 * 8.0);
        assert_eq!(median_per_kernel(&[(0, 5.0), (0, 1.0), (0, 3.0), (0, 2.0)], 1), vec![Some(2.0)]);
        assert_eq!(median_per_kernel(&[(0, 5.0)], 1), vec![Some(5.0)]);
    }
}
