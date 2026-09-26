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

/// The register loads and stores a kernel executes (plan 0.8.0 §3.1, step
/// 2): every `Load`, `LoadI`, `VLoad`, `Store`, `StoreI` and `VStore`, once
/// whatever its lanes, times the trip counts of the loops enclosing it. A
/// broadcast of a loaded scalar is part of its load, and a load inside an
/// index (a gather's) is a load. A branch is counted as taken, as work is.
pub fn kernel_loads(k: &Kernel) -> f64 {
    fn expr(e: &Expr) -> f64 {
        match e {
            Expr::Load(_, ix) | Expr::LoadI(_, ix) | Expr::VLoad(_, ix) => 1.0 + ix.iter().map(expr).sum::<f64>(),
            Expr::F(_) | Expr::I(_) | Expr::Iv(_) | Expr::Sv(_) | Expr::Si(_) | Expr::Vv(_) => 0.0,
            Expr::VSplat(a) | Expr::Un(_, a) | Expr::ToF(a) | Expr::ToI(a) => expr(a),
            Expr::VBin(_, a, b) | Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => expr(a) + expr(b),
            Expr::Select(c, a, b) => expr(c) + expr(a) + expr(b),
        }
    }
    fn stmts(ss: &[Stmt]) -> f64 {
        ss.iter()
            .map(|s| match s {
                Stmt::For { bound, body, .. } => *bound as f64 * stmts(body),
                Stmt::Set { value, .. } | Stmt::SetI { value, .. } | Stmt::SetL { value, .. } | Stmt::VSet { value, .. } => expr(value),
                Stmt::Store { index, value, .. } | Stmt::StoreI { index, value, .. } | Stmt::VStore { index, value, .. } => 1.0 + expr(value) + index.iter().map(expr).sum::<f64>(),
                Stmt::If { cond, then } => expr(cond) + stmts(then),
            })
            .sum()
    }
    stmts(&k.body)
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
    /// The level-1 data cache's line, in bytes (0.8.0 step 4); zero in a
    /// description recorded before, which [`CostRules::lines`] cannot price.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub line: usize,
}

fn is_zero(n: &usize) -> bool {
    *n == 0
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
        let mut m = Machine { workers, l1: 32 << 10, l2: 1 << 20, l3: 8 << 20, l3_sharing: workers, vector_registers, vector_bytes: 16, source: "defaults".into(), line: 64 };
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
                ("1", "Data") => {
                    m.l1 = size;
                    if let Ok(line) = read(&p, "coherency_line_size").parse::<usize>() {
                        m.line = line;
                    }
                }
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

    /// The level-1 data cache's line of `cpu` (`cpu5`, a description's
    /// [`source`](Machine::source)), as the operating system reports it: for
    /// a description recorded before the line size was (0.8.0 step 4).
    pub fn line_of(cpu: &str) -> Option<usize> {
        let dir = std::fs::read_dir(format!("/sys/devices/system/cpu/{cpu}/cache")).ok()?;
        let read = |p: &std::path::Path, f: &str| std::fs::read_to_string(p.join(f)).map(|t| t.trim().to_string()).unwrap_or_default();
        dir.flatten().map(|e| e.path()).find(|p| read(p, "level") == "1" && read(p, "type") == "Data").and_then(|p| read(&p, "coherency_line_size").parse().ok())
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
/// `traffic` and, since 0.8.0 step 2, in `loads`: the register loads and
/// stores a register tile saves.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlannedCost {
    /// One worker's weighted operations, a load weighing nothing. Until
    /// 0.8.0 step 2 it included `spill`, and a profile written before still
    /// holds samples whose `work` does, with `spill` zero.
    pub work: f64,
    /// 0.6.0's figure: every input once, the output once.
    pub bytes: f64,
    /// One worker's bytes from L2, from L3 and from memory.
    pub traffic: [f64; 3],
    /// Real and vector variables live in the busiest innermost loop, and how
    /// many of them exceed the machine's vector registers.
    pub regs: usize,
    pub spilled: usize,
    /// One worker's register loads and stores ([`kernel_loads`]; plan 0.8.0
    /// §3.1). Zero in a sample recorded before 0.8.0 step 2.
    #[serde(default)]
    pub loads: f64,
    /// The two accesses per spilled register per iteration of the busiest
    /// loop: charged as work by a model without a load term, as 0.7.0's
    /// was, and as loads by one with it.
    #[serde(default)]
    pub spill: f64,
    /// The split the cost is of: the workers, and the depth of the loop.
    pub workers: usize,
    pub depth: usize,
}

/// The rules of the planned cost a profile may name beyond 0.7.0's (plan
/// 0.8.0 step 4, 4c), each off by default: a profile that names none prices
/// every kernel as before, bit for bit, as a profile without `ρ` does (D66).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CostRules {
    /// Footprints in whole cache lines of the machine's line size.
    #[serde(default, skip_serializing_if = "is_false")]
    pub lines: bool,
    /// A loop's body resident when the sum of its buffers' footprints fits,
    /// not each buffer apart.
    #[serde(default, skip_serializing_if = "is_false")]
    pub shared: bool,
    /// Every loop iteration one operation of work.
    #[serde(default, skip_serializing_if = "is_false")]
    pub overhead: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

impl CostRules {
    pub fn is_none(&self) -> bool {
        *self == CostRules::default()
    }

    /// `none`, or the rules by name, comma-separated: `lines`, `shared`,
    /// `overhead`.
    pub fn parse(s: &str) -> std::result::Result<CostRules, String> {
        let mut r = CostRules::default();
        for name in s.split(',').map(str::trim).filter(|n| !n.is_empty() && *n != "none") {
            match name {
                "lines" => r.lines = true,
                "shared" => r.shared = true,
                "overhead" => r.overhead = true,
                other => return Err(format!("--rules {other}: `none`, or among `lines`, `shared`, `overhead`")),
            }
        }
        Ok(r)
    }

    /// The rules by name, `none` when there is none.
    pub fn name(&self) -> String {
        let names: Vec<&str> = [(self.lines, "lines"), (self.shared, "shared"), (self.overhead, "overhead")].into_iter().filter(|(on, _)| *on).map(|(_, n)| n).collect();
        if names.is_empty() { "none".into() } else { names.join(",") }
    }
}

/// The planned cost of a kernel on a machine, under 0.7.0's rules. The split
/// is the kernel's own ([`Threads`]); the executor's default is modelled as
/// the executor decides it — the outermost parallel loop over every worker,
/// above [`PARALLEL_MIN_WORK`]. A worker's part is the kernel with the loop
/// at the split's depth cut to a worker's share of its iterations.
pub fn planned_cost(k: &Kernel, m: &Machine) -> PlannedCost {
    planned_cost_under(k, m, &CostRules::default())
}

/// [`planned_cost`] under the rules a profile names.
pub fn planned_cost_under(k: &Kernel, m: &Machine, rules: &CostRules) -> PlannedCost {
    let (depth, p) = match k.threads {
        Threads::At { depth, workers } => (depth, k.parallel_at(depth).map_or(1, |n| workers.min(n).max(1))),
        Threads::One => (0, 1),
        Threads::Default => match k.parallel_outer() {
            Some(n) if m.workers > 1 && n >= 2 && kernel_cost(k).work >= PARALLEL_MIN_WORK => (0, m.workers.min(n)),
            _ => (0, 1),
        },
    };
    let worker = if p > 1 { share(k, depth, p) } else { k.clone() };
    let traffic = traffic_under(&worker, [m.l1, m.l2, m.l3_per_worker(p)], rules, m.line);
    let (regs, iterations) = pressure(&worker.body, 1.0);
    let spilled = regs.saturating_sub(m.vector_registers);
    // Two level-1 accesses per iteration of the busiest loop and per spilled
    // register.
    let spill = 2.0 * spilled as f64 * iterations;
    let work = kernel_cost(&worker).work;
    let work = if rules.overhead { work + loop_iterations(&worker.body) } else { work };
    PlannedCost { work, bytes: kernel_cost(k).bytes, traffic, regs, spilled, loads: kernel_loads(&worker), spill, workers: p, depth }
}

/// The iterations of every loop of `stmts`, each counted once however deep
/// ([`CostRules::overhead`]): its increment, compare and branch weigh one
/// operation. A branch is counted as taken, as work is.
fn loop_iterations(stmts: &[Stmt]) -> f64 {
    stmts
        .iter()
        .map(|s| match s {
            Stmt::For { bound, body, .. } => *bound as f64 * (1.0 + loop_iterations(body)),
            Stmt::If { then, .. } => loop_iterations(then),
            _ => 0.0,
        })
        .sum()
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
/// for an index that is not affine. With `line` ([`CostRules::lines`]), in
/// whole lines of that many bytes: along the contiguous axis, the span
/// rounded up to the line, so that an access whose loops do not walk that
/// axis brings a line per element.
fn footprint(k: &Kernel, stmts: &[Stmt], buf: usize, defs: &HashMap<usize, Expr>, line: Option<f64>) -> f64 {
    let spec = if buf == k.out_buf() { &k.output } else { &k.inputs[buf] };
    let mut acc = Vec::new();
    accesses(stmts, buf, k.lanes, &mut acc);
    if acc.is_empty() {
        return 0.0;
    }
    let mut inside = HashMap::new();
    loops(stmts, &mut inside, &mut HashMap::new());
    let rank = spec.dims.len();
    let (mut elems, mut contiguous) = (1.0, 1.0);
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
        match line {
            Some(_) if a + 1 == rank => contiguous = extent as f64,
            _ => elems *= extent as f64,
        }
    }
    let bytes = elem_bytes(spec.kind, k.dtype);
    match line {
        None => elems * bytes,
        Some(l) => elems * ((contiguous * bytes / l).ceil() * l),
    }
}

/// The bytes a kernel brings into a cache of `capacity` bytes, every buffer
/// counted apart (plan 0.7.0 §3.5): a buffer whose footprint inside a loop
/// fits is brought in once per iteration of the loops enclosing it, and
/// otherwise the loop's body is looked at once per iteration of the loop.
pub fn traffic(k: &Kernel, capacity: usize) -> f64 {
    traffic_at(k, [capacity])[0]
}

/// [`traffic`] into caches of each of `capacities`. A footprint does not
/// depend on the capacity, so each capacity's walk reads the footprints the
/// others computed (0.8.0 step 1): the same terms, added in the same order,
/// as that many calls of `traffic`.
pub fn traffic_at<const N: usize>(k: &Kernel, capacities: [usize; N]) -> [f64; N] {
    traffic_under(k, capacities, &CostRules::default(), 0)
}

/// [`traffic_at`] under the rules a profile names (plan 0.8.0 step 4, 4c),
/// 0.7.0's walk when it names none. [`CostRules::lines`] counts every
/// footprint in lines of `line` bytes. [`CostRules::shared`] holds a list of
/// statements, or a loop, resident when the footprints of every buffer in
/// it, summed, fit, where 0.7.0 asks it of each buffer apart. Under either,
/// a loop that does not fit while one of its iterations does brings what
/// consecutive iterations share once — the lines they share along the
/// contiguous axis, a buffer the loop does not move — since the one
/// iteration between two uses leaves it resident: the loop brings the
/// lesser of its footprint and its iterations' footprints summed.
pub fn traffic_under<const N: usize>(k: &Kernel, capacities: [usize; N], rules: &CostRules, line: usize) -> [f64; N] {
    // `presk refit --recost` records the line size of a description written
    // before 0.8.0 step 4 when it costs under `lines`.
    assert!(!rules.lines || line > 0, "the rule `lines` needs the machine's line size, and the description has none");
    let mut defs = HashMap::new();
    loops(&k.body, &mut HashMap::new(), &mut defs);
    let mut w = Walk { k, defs, line: rules.lines.then_some(line as f64), shared: rules.shared, reuse: rules.lines || rules.shared, memo: HashMap::new() };
    capacities.map(|c| {
        let per_buf: Vec<f64> = (0..=k.inputs.len()).map(|b| w.walk(&k.body, b, 1.0, c as f64)).collect();
        per_buf.into_iter().sum()
    })
}

/// The walk of [`traffic_under`] over one kernel, its footprints memoized by
/// the statements (where they start in the kernel, how many) and the buffer.
struct Walk<'a> {
    k: &'a Kernel,
    defs: HashMap<usize, Expr>,
    line: Option<f64>,
    shared: bool,
    reuse: bool,
    memo: HashMap<(usize, usize, usize), f64>,
}

impl<'a> Walk<'a> {
    fn fp(&mut self, stmts: &'a [Stmt], buf: usize) -> f64 {
        let (k, defs, line) = (self.k, &self.defs, self.line);
        *self.memo.entry((stmts.as_ptr() as usize, stmts.len(), buf)).or_insert_with(|| footprint(k, stmts, buf, defs, line))
    }

    /// What must fit for `stmts` to be resident, `f` being `buf`'s footprint
    /// in them: `f`, or every buffer's summed under [`CostRules::shared`].
    fn held(&mut self, stmts: &'a [Stmt], f: f64) -> f64 {
        if self.shared { (0..=self.k.inputs.len()).map(|b| self.fp(stmts, b)).sum() } else { f }
    }

    fn walk(&mut self, stmts: &'a [Stmt], buf: usize, times: f64, cap: f64) -> f64 {
        let f = self.fp(stmts, buf);
        if f == 0.0 {
            return 0.0;
        }
        if self.held(stmts, f) <= cap {
            return times * f;
        }
        let mut terms = Vec::with_capacity(stmts.len());
        for s in stmts {
            terms.push(match s {
                Stmt::For { bound, body, .. } => {
                    let one = std::slice::from_ref(s);
                    let whole = self.fp(one, buf);
                    if self.held(one, whole) <= cap {
                        times * whole
                    } else {
                        let each = self.fp(body, buf);
                        if self.reuse && self.held(body, each) <= cap { times * whole.min(*bound as f64 * each) } else { self.walk(body, buf, times * *bound as f64, cap) }
                    }
                }
                Stmt::If { then, .. } => self.walk(then, buf, times, cap),
                other => times * self.fp(std::slice::from_ref(other), buf),
            });
        }
        terms.into_iter().sum()
    }
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
        Machine { workers: 10, l1: 64 << 10, l2: 2 << 20, l3: 8 << 20, l3_sharing: 5, vector_registers: 32, vector_bytes: 16, source: "test".into(), line: 64 }
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
        // Since 0.8.0 step 2 the spill is recorded apart from the work, which
        // a model without the load term adds back (D66). Per k the scalar
        // tile loads 4 + 16 values and the one in four lanes 4 + 16/4; each
        // of the (512/4)·(512/16) blocks then stores 64 values or 16 vectors.
        assert_eq!((scalar.work, tall.spill), (tall.work, 0.0));
        assert!(scalar.spill > 0.0);
        let blocks = (512.0 / 4.0) * (512.0 / 16.0);
        assert_eq!((scalar.loads, tall.loads), (blocks * (512.0 * 20.0 + 64.0), blocks * (512.0 * 8.0 + 16.0)));
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

    /// A row-major `add` of two `f32[rows, cols]`, the rows outermost.
    fn add(rows: usize, cols: usize) -> Kernel {
        let spec = BufSpec { dims: vec![rows, cols], kind: BufKind::Real, names: vec![None, None] };
        let load = |b: usize| Expr::Load(b, vec![Expr::Iv(0), Expr::Iv(1)]);
        let store = Stmt::Store { buf: 2, index: vec![Expr::Iv(0), Expr::Iv(1)], value: Expr::Bin(BinOp::Add, Box::new(load(0)), Box::new(load(1))) };
        let body = vec![Stmt::For { var: 0, bound: rows, body: vec![Stmt::For { var: 1, bound: cols, body: vec![store] }] }];
        Kernel { name: "add".into(), dtype: Dtype::F32, inputs: vec![spec.clone(), spec.clone()], output: spec, nloops: 2, nreal: 0, nint: 0, nvec: 0, lanes: 1, threads: Threads::One, body }
    }

    /// Plan 0.8.0 step 4 (4c): each rule a profile may name moves the term
    /// it is about, and a profile naming none prices as 0.7.0's rules do.
    #[test]
    fn each_rule_of_the_planned_cost_moves_its_term_and_none_moves_anything() {
        let m = falicon();
        let none = CostRules::default();
        let [lines, shared, overhead] = ["lines", "shared", "overhead"].map(|r| CostRules::parse(r).unwrap());
        assert_eq!((CostRules::parse("overhead,lines").unwrap().name(), none.name()), ("lines,overhead".to_string(), "none".to_string()));
        assert!(CostRules::parse("none").unwrap().is_none() && CostRules::parse("lanes").is_err());
        // No rule: 0.7.0's cost, whatever the line size (zero in a machine
        // described before 0.8.0 step 4).
        for k in [gemm(256, Layout::default()), gemm(512, Layout { tile: Some((4, 16)), ..Layout::default() }), add(64, 1024)] {
            assert_eq!(planned_cost_under(&k, &Machine { line: 0, ..m.clone() }, &none), planned_cost(&k, &m), "{}", k.name);
        }
        // The loop overhead: an operation per iteration of each loop, 64 of
        // the rows and 64 · 1024 of the columns, and nothing else.
        let a = add(64, 1024);
        let (plain, over) = (planned_cost(&a, &m), planned_cost_under(&a, &m, &overhead));
        assert_eq!((over.work, over.traffic, over.loads), (plain.work + 64.0 * 1025.0, plain.traffic, plain.loads));
        // A stream along the contiguous axis brings its bytes once in lines
        // and with its buffers counted together: its rows are whole lines,
        // and consecutive elements share one, even where a row does not fit.
        for r in [&lines, &shared] {
            assert_eq!(traffic_under(&a, [1 << 10, 64 << 10], r, 64), traffic_at(&a, [1 << 10, 64 << 10]), "{}", r.name());
        }
        // An IJ product of side 2048 with a [1, 4] tile walks the right
        // operand's 2048 rows under each tile, 16 bytes of a line in each:
        // 128 KiB of lines, more than L1, so that the next tile, which reads
        // the next 16 bytes of the same lines, brings them again. In lines
        // each (i, tile, k) brings 64 bytes of it where the elements counted
        // 16. The left operand and the output are whole rows either way. A
        // [1, 16] tile reads whole lines, and the rule does not move it.
        let ij = |nj: usize| gemm(2048, Layout { tile: Some((1, nj)), kblock: Some(KBlock::None), order: Some(vec![BlockLoop::I, BlockLoop::J]), ..Layout::default() });
        let (narrow, wide) = (ij(4), ij(16));
        let (el, ln) = (planned_cost(&narrow, &m), planned_cost_under(&narrow, &m, &lines));
        assert_eq!(ln.traffic[0] - el.traffic[0], 2048.0 * 512.0 * 2048.0 * 48.0, "{:?} against {:?}", ln.traffic, el.traffic);
        assert_eq!(planned_cost_under(&wide, &m, &lines), planned_cost(&wide, &m));
        // Buffers together: a JI product of side 1024 with a [4, 16] tile
        // walks a strip of the right operand, 1024 rows of 64 bytes, under
        // each tile of rows. Alone it fits L1 and is read once per strip;
        // with the left operand's rows and the output's tile beside it, it
        // does not, and is read again under every tile of rows.
        let ji = gemm(1024, Layout { tile: Some((4, 16)), kblock: Some(KBlock::None), order: Some(vec![BlockLoop::J, BlockLoop::I]), ..Layout::default() });
        let (apart, together) = (planned_cost(&ji, &m), planned_cost_under(&ji, &m, &shared));
        assert!(together.traffic[0] > 4.0 * apart.traffic[0], "{:?} against {:?}", together.traffic, apart.traffic);
        assert_eq!((together.work, together.loads), (apart.work, apart.loads));
    }

    #[test]
    fn sysfs_sizes_and_cpu_lists_are_read() {
        assert_eq!(parse_size("64K"), Some(64 << 10));
        assert_eq!(parse_size("8M"), Some(8 << 20));
        assert_eq!(parse_size("2048K"), Some(2 << 20));
        assert_eq!(parse_cpu_list("5-9,15-19"), vec![5, 6, 7, 8, 9, 15, 16, 17, 18, 19]);
        assert_eq!(parse_cpu_list("0"), vec![0]);
        let m = Machine::detect();
        assert!(m.workers >= 1 && m.l1 > 0 && m.l2 >= m.l1 && m.l3_sharing >= 1 && m.line >= 16, "{m:?}");
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
