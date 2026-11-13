//! The native host's time by phase (plan 0.10.0 step 5, `presk bench
//! --host-profile`): where a call goes between the kernels and the host
//! program that launches them, so that the host's fixes (step 6) are taken
//! in the order of what they can save.
//!
//! The host ([`crate::KernelBackend`]) reads the clock around each region of
//! its work and adds the time to the region's phase; the executor times what
//! of a launch is not the kernel's code ([`Overhead`]). Only the clock is
//! read: every buffer, kernel and launch is the one an unprofiled call makes,
//! so the outputs are the same bits. A region is timed where it is, and
//! regions do not nest: a phase's time is its own. What no region covers —
//! the interpreter of the host program, matching its `let`s and filling its
//! environment — is the call's time less every phase, by difference.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use presk_rt::{Buf, Value};

/// A region of the host's work, or the kernels' (plan 0.10.0 §3.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// The kernels' own code: each launch timed around the executor, as
    /// `presk bench --calibrate` times it, less what the executor reports
    /// besides (compiling, dispatching, the threads).
    Kernel,
    /// A `let`'s site named (`<function>/%<let>`) and its type found.
    Site,
    /// A function's fusion groups, cloned at each of its calls.
    Groups,
    /// A nest's kind, and its layout in the plan, checked and resolved.
    Layout,
    /// A launch's cache key formatted, and the kernel looked up by it.
    Key,
    /// A kernel lowered and compiled, at its first launch.
    Lower,
    /// A launch's output buffer allocated, zeroed.
    Alloc,
    /// A launch's inputs converted to its kernel's type.
    Convert,
    /// Buffers freed: a function's values when it returns, a launch's
    /// converted inputs.
    Free,
    /// A stacked tree's layer sliced; a scan's results stacked; a slice set
    /// or a gradient scattered into a stack. Views and parts since step 6,
    /// which copy nothing (`presk_rt::Elems::view`, `stack`).
    Slice,
    /// Axes merged, split or reshaped: the data copied under its new axes.
    View,
    /// The entry's arguments: the caller's copy, then each leaf in the
    /// realization's type.
    Entry,
    /// A launch's operands read out of the environment, then dropped. The
    /// phases of group 4 copied what they read until step 6; a value's
    /// elements are shared since (`presk_rt::Elems`), and their bytes are
    /// what they still copy.
    Operands,
    /// The same operands taken again as tensors, then dropped.
    AsTensor,
    /// A call's arguments, a closure's captured values.
    Args,
    /// Tuples, records, projections and fields; a tree's leaves collected
    /// and rebuilt around a primitive mapped over them.
    Tuples,
    /// A fold's stacked and extra arguments, and a scan's accumulators kept.
    Fold,
    /// The executor's dispatch: the buffers' addresses, a split's size.
    Dispatch,
    /// A divided launch's threads started and joined: its time less its
    /// longest worker's.
    Threads,
    /// Dropout masks, drawn on the host (ones when not training).
    Dropout,
    /// The other primitives the host computes itself: initializers,
    /// `sample`, `zeros_like`, `split_key`.
    HostPrim,
}

impl Phase {
    pub const ALL: [Phase; 21] = [
        Phase::Kernel,
        Phase::Site,
        Phase::Groups,
        Phase::Layout,
        Phase::Key,
        Phase::Lower,
        Phase::Alloc,
        Phase::Convert,
        Phase::Free,
        Phase::Slice,
        Phase::View,
        Phase::Entry,
        Phase::Operands,
        Phase::AsTensor,
        Phase::Args,
        Phase::Tuples,
        Phase::Fold,
        Phase::Dispatch,
        Phase::Threads,
        Phase::Dropout,
        Phase::HostPrim,
    ];

    fn index(self) -> usize {
        self as usize
    }

    /// The phase's key in `presk bench --host-profile`'s table: its group,
    /// a dot, its name.
    pub fn key(self) -> &'static str {
        match self {
            Phase::Kernel => "kernels.code",
            Phase::Site => "find.site",
            Phase::Groups => "find.groups",
            Phase::Layout => "find.layout",
            Phase::Key => "find.key",
            Phase::Lower => "find.lower",
            Phase::Alloc => "buffers.alloc",
            Phase::Convert => "buffers.convert",
            Phase::Free => "buffers.free",
            Phase::Slice => "views.slice",
            Phase::View => "views.axes",
            Phase::Entry => "copies.entry",
            Phase::Operands => "copies.operands",
            Phase::AsTensor => "copies.as-tensor",
            Phase::Args => "copies.args",
            Phase::Tuples => "copies.tuples",
            Phase::Fold => "copies.fold",
            Phase::Dispatch => "threads.dispatch",
            Phase::Threads => "threads.start-join",
            Phase::Dropout => "left.dropout",
            Phase::HostPrim => "left.host-prims",
        }
    }

    pub fn what(self) -> &'static str {
        match self {
            Phase::Kernel => "the kernels' own code",
            Phase::Site => "a let's site named, its type found",
            Phase::Groups => "a function's fusion groups cloned at its call",
            Phase::Layout => "a nest's kind, its layout in the plan",
            Phase::Key => "the cache key formatted, the kernel looked up",
            Phase::Lower => "a kernel lowered and compiled (first launch)",
            Phase::Alloc => "output buffers allocated",
            Phase::Convert => "inputs converted to the kernel's type",
            Phase::Free => "values freed when their function returns",
            Phase::Slice => "stacked layers sliced, scans' results stacked",
            Phase::View => "axes merged, split, reshaped (copied)",
            Phase::Entry => "the entry's arguments typed",
            Phase::Operands => "a launch's operands read from the environment",
            Phase::AsTensor => "the same operands taken again as tensors",
            Phase::Args => "a call's arguments, a closure's captures",
            Phase::Tuples => "tuples, records, fields, a tree's leaves",
            Phase::Fold => "a fold's arguments, a scan's accumulators",
            Phase::Dispatch => "the executor's dispatch of a launch",
            Phase::Threads => "a divided launch's threads started and joined",
            Phase::Dropout => "dropout masks drawn on the host",
            Phase::HostPrim => "other primitives the host computes",
        }
    }
}

/// What of one launch the executor spent outside the kernel's code, in
/// seconds, read by the host after the launch when it profiles.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Overhead {
    /// Compiling the kernel, at its first launch.
    pub compile: f64,
    /// Everything before the kernel's code starts: its addresses, its split.
    pub dispatch: f64,
    /// A divided launch's time less its longest worker's: its threads'
    /// start, their join, and the stagger of their starts.
    pub threads: f64,
    /// The threads the launch ran on (1: on the calling thread).
    pub workers: usize,
}

impl Overhead {
    /// The kernel's own code in a launch of `seconds`: what is left of it
    /// besides the compiling, the dispatch and the threads.
    pub fn kernel(&self, seconds: f64) -> f64 {
        (seconds - self.compile - self.dispatch - self.threads).max(0.0)
    }
}

/// One launch of a call as `presk bench --host-profile --launches` lists it
/// (plan 0.17.2 step 4, D200): which kernel, launched by which nest, its own
/// code's time, and where its buffers are.
#[derive(Clone, Debug, PartialEq)]
pub struct LaunchRow {
    /// The kernel's place in the backend's list.
    pub kernel: usize,
    /// The nest, `"<function>/%<let>"`.
    pub nest: String,
    /// The kernel's own code, in seconds, as [`Phase::Kernel`] counts it.
    pub seconds: f64,
    /// The threads the launch ran on.
    pub workers: usize,
    /// Each input as the kernel read it, its first element's address and its
    /// bytes ([`buf_span`]).
    pub inputs: Vec<(usize, u64)>,
    /// The output, the same.
    pub output: (usize, u64),
}

/// The host's time by phase over some calls.
#[derive(Clone, Debug, Default)]
pub struct HostProfile {
    pub seconds: [f64; Phase::ALL.len()],
    /// The regions timed in each phase.
    pub count: [u64; Phase::ALL.len()],
    /// The bytes each phase copied, allocated, converted or freed.
    pub bytes: [u64; Phase::ALL.len()],
    /// Launches, and those divided among threads with the threads they took.
    pub launches: u64,
    pub divided: u64,
    pub threads: u64,
    /// The process's minor page faults over the call, pages touched for the
    /// first time, when the caller counts them ([`minor_faults`]).
    pub faults: Option<u64>,
    /// When set, a row per launch, in the call's order (`--launches`, plan
    /// 0.17.2 step 4): the caller sets it before each call it wants listed.
    pub rows: Option<Vec<LaunchRow>>,
}

/// A profile shared between a backend and its caller, which reads it
/// between calls (`presk train --host-profile` after each step).
pub type SharedProfile = Rc<RefCell<HostProfile>>;

impl HostProfile {
    pub fn shared() -> SharedProfile {
        Rc::new(RefCell::new(HostProfile::default()))
    }

    pub fn add(&mut self, p: Phase, seconds: f64, bytes: u64) {
        self.seconds[p.index()] += seconds;
        self.count[p.index()] += 1;
        self.bytes[p.index()] += bytes;
    }

    /// Time added to a phase whose region was counted already (a copy's
    /// free, after the copy).
    pub fn add_time(&mut self, p: Phase, seconds: f64) {
        self.seconds[p.index()] += seconds;
    }

    /// A launch's executor time: the kernel's code, and what the executor
    /// reported besides.
    pub fn launch(&mut self, seconds: f64, o: Overhead) {
        self.launches += 1;
        if o.workers > 1 {
            self.divided += 1;
            self.threads += o.workers as u64;
        }
        if o.compile > 0.0 {
            self.add(Phase::Lower, o.compile, 0);
        }
        self.add(Phase::Dispatch, o.dispatch, 0);
        if o.workers > 1 {
            self.add(Phase::Threads, o.threads, 0);
        }
        self.add(Phase::Kernel, o.kernel(seconds), 0);
    }

    pub fn seconds(&self, p: Phase) -> f64 {
        self.seconds[p.index()]
    }

    /// The bytes a phase copied, allocated, converted or freed.
    pub fn bytes_of(&self, p: Phase) -> u64 {
        self.bytes[p.index()]
    }

    /// Every phase's time.
    pub fn total(&self) -> f64 {
        self.seconds.iter().sum()
    }
}

/// The process's minor page faults so far, from `/proc/self/stat` (Linux;
/// `None` elsewhere): read by `--host-profile` before and after a call, never
/// inside one.
pub fn minor_faults() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/stat").ok()?;
    // The fields after the command's closing parenthesis start at the third,
    // the state; the tenth is `minflt`.
    let rest = &s[s.rfind(')')? + 2..];
    rest.split_whitespace().nth(7)?.parse().ok()
}

/// The time since `t0`, when the host profiles.
pub(crate) fn since(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64()
}

/// The bytes a buffer holds.
pub fn buf_bytes(b: &Buf) -> u64 {
    (match b {
        Buf::Exact(v) => std::mem::size_of_val(&v[..]),
        // The length alone: reading a stack's elements would lay its parts
        // out in one allocation.
        Buf::F64(v) => v.len() * 8,
        Buf::F32(v) => v.len() * 4,
        Buf::Bf16(v) => v.len() * 2,
        Buf::I64(v) => v.len() * 8,
        Buf::Quant(q) => std::mem::size_of_val(q.data.as_slice()) + std::mem::size_of_val(q.scales.as_slice()),
    }) as u64
}

/// A buffer as a kernel reads it: its first element's address and its bytes.
/// Read after the launch, when a stack's parts are laid out already.
pub fn buf_span(b: &Buf) -> (usize, u64) {
    let at = match b {
        Buf::Exact(_) => 0,
        Buf::F64(v) => v.as_ptr() as usize,
        Buf::F32(v) => v.as_ptr() as usize,
        Buf::Bf16(v) => v.as_ptr() as usize,
        Buf::I64(v) => v.as_ptr() as usize,
        Buf::Quant(q) => q.data.as_ptr() as usize,
    };
    (at, buf_bytes(b))
}

/// The set strides of the data caches of the CPU the process last ran on:
/// the bytes between two addresses of one set, its sets times its line
/// (plan 0.17.2 step 4, D200). From Linux's sysfs; `None` elsewhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetStrides {
    pub cpu: usize,
    pub l1: u64,
    pub l2: u64,
}

pub fn set_strides() -> Option<SetStrides> {
    let s = std::fs::read_to_string("/proc/self/stat").ok()?;
    // The thirty-ninth field, `processor`: the CPU last run on.
    let rest = &s[s.rfind(')')? + 2..];
    let cpu: usize = rest.split_whitespace().nth(36)?.parse().ok()?;
    let stride = |level: &str| -> Option<u64> {
        for i in 0..8 {
            let d = format!("/sys/devices/system/cpu/cpu{cpu}/cache/index{i}");
            let read = |f: &str| std::fs::read_to_string(format!("{d}/{f}")).ok().map(|v| v.trim().to_string());
            if read("level")?.as_str() == level && read("type")?.as_str() != "Instruction" {
                return Some(read("number_of_sets")?.parse::<u64>().ok()? * read("coherency_line_size")?.parse::<u64>().ok()?);
            }
        }
        None
    };
    Some(SetStrides { cpu, l1: stride("1")?, l2: stride("2")? })
}

/// The transparent huge pages' mode (`always`, `madvise`, `never`) and the
/// process's anonymous memory in huge pages, in kB (Linux; `None`
/// elsewhere).
pub fn huge_pages() -> (Option<String>, Option<u64>) {
    let mode = std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled").ok().and_then(|s| s.split('[').nth(1)?.split(']').next().map(str::to_string));
    let anon = std::fs::read_to_string("/proc/self/smaps_rollup").ok().and_then(|s| s.lines().find_map(|l| l.strip_prefix("AnonHugePages:")?.trim().strip_suffix("kB")?.trim().parse().ok()));
    (mode, anon)
}

/// The bytes a value's tensors hold.
pub fn value_bytes(v: &Value) -> u64 {
    match v {
        Value::Real(t) | Value::Idx { t, .. } => buf_bytes(&t.data),
        Value::Tuple(vs) | Value::Stacked(vs) => vs.iter().map(value_bytes).sum(),
        Value::Record(fs) => fs.iter().map(|(_, v)| value_bytes(v)).sum(),
        _ => 0,
    }
}

/// The bytes a clone of a value copies: its exact buffers' only, since a
/// machine buffer's elements are shared by its clones (plan 0.10.0 step 6).
pub fn copied_bytes(v: &Value) -> u64 {
    match v {
        Value::Real(t) | Value::Idx { t, .. } => match &t.data {
            Buf::Exact(_) => buf_bytes(&t.data),
            _ => 0,
        },
        Value::Tuple(vs) | Value::Stacked(vs) => vs.iter().map(copied_bytes).sum(),
        Value::Record(fs) => fs.iter().map(|(_, v)| copied_bytes(v)).sum(),
        _ => 0,
    }
}

/// The bytes dropping `values` together frees: each allocation every one of
/// whose sharers is among them, counted once (plan 0.10.0 step 6).
pub fn freed_bytes<'a>(values: impl IntoIterator<Item = &'a Value>) -> u64 {
    use std::collections::HashMap;
    // An allocation's address: (its sharers among `values`, its sharers in
    // all, its bytes).
    let mut blocks: HashMap<usize, (usize, usize, u64)> = HashMap::new();
    let mut owned = 0;
    fn walk(v: &Value, blocks: &mut HashMap<usize, (usize, usize, u64)>, owned: &mut u64) {
        match v {
            Value::Real(t) | Value::Idx { t, .. } => {
                let b = match &t.data {
                    Buf::F64(e) => Some(e.block()),
                    Buf::F32(e) => Some(e.block()),
                    Buf::Bf16(e) => Some(e.block()),
                    Buf::I64(e) => Some(e.block()),
                    _ => None,
                };
                match b {
                    Some((at, refs, bytes)) => blocks.entry(at).or_insert((0, refs, bytes as u64)).0 += 1,
                    None => *owned += buf_bytes(&t.data),
                }
            }
            Value::Tuple(vs) | Value::Stacked(vs) => vs.iter().for_each(|x| walk(x, blocks, owned)),
            Value::Record(fs) => fs.iter().for_each(|(_, x)| walk(x, blocks, owned)),
            _ => {}
        }
    }
    for v in values {
        walk(v, &mut blocks, &mut owned);
    }
    owned + blocks.values().filter(|(here, all, _)| here == all).map(|(_, _, b)| b).sum::<u64>()
}

/// The table `presk bench --host-profile` and `presk train --host-profile`
/// print: each phase's mean a call, its share of the call, and its bytes,
/// in the groups of plan 0.10.0 §3.3, then what is left by difference.
/// `calls`: each call's profile and its own time in seconds, measured by
/// the caller around the whole call.
pub fn table(calls: &[(HostProfile, f64)], unit: &str) -> String {
    use std::fmt::Write;
    let n = calls.len().max(1) as f64;
    let mean = |f: &dyn Fn(&HostProfile) -> f64| calls.iter().map(|(p, _)| f(p)).sum::<f64>() / n;
    let call = calls.iter().map(|(_, s)| s).sum::<f64>() / n;
    let mut out = String::new();
    let _ = writeln!(out, "{:<20} {:>11} {:>8} {:>11} {:>11}  what", "phase", format!("ms a {unit}"), "share", format!("MB a {unit}"), format!("n a {unit}"));
    let row = |out: &mut String, key: &str, s: f64, mb: Option<f64>, count: Option<f64>, what: &str| {
        let mb = mb.filter(|m| *m > 0.0).map_or("—".to_string(), |m| format!("{m:.1}"));
        let count = count.map_or("—".to_string(), |c| format!("{c:.0}"));
        let _ = writeln!(out, "{key:<20} {:>11.3} {:>7.2} % {mb:>11} {count:>11}  {what}", s * 1e3, 100.0 * s / call);
    };
    let groups: [(&str, &str, &[Phase]); 6] = [
        ("kernels", "the kernels' code", &[Phase::Kernel]),
        ("find", "1. finding the kernel", &[Phase::Site, Phase::Groups, Phase::Layout, Phase::Key, Phase::Lower]),
        ("buffers", "2. buffers allocated, converted, freed", &[Phase::Alloc, Phase::Convert, Phase::Free]),
        ("views", "3. stacked trees sliced, views materialized", &[Phase::Slice, Phase::View]),
        ("copies", "4. arguments, tuples, fields, accumulators (copied until step 6)", &[Phase::Entry, Phase::Operands, Phase::AsTensor, Phase::Args, Phase::Tuples, Phase::Fold]),
        ("threads", "5. a launch dispatched, its threads started and joined", &[Phase::Dispatch, Phase::Threads]),
    ];
    for (key, what, phases) in groups {
        let s: f64 = phases.iter().map(|p| mean(&|q| q.seconds(*p))).sum();
        let launches = (key == "kernels").then(|| mean(&|q| q.launches as f64));
        row(&mut out, key, s, None, launches, what);
        if phases.len() > 1 || key != "kernels" {
            for p in phases {
                let count = match p {
                    Phase::Threads => mean(&|q| q.threads as f64),
                    _ => mean(&|q| q.count[p.index()] as f64),
                };
                row(&mut out, &format!("  {}", p.key()), mean(&|q| q.seconds(*p)), Some(mean(&|q| q.bytes[p.index()] as f64) / 1e6), Some(count), p.what());
            }
        }
    }
    let measured = mean(&|q| q.total());
    let rest = call - measured;
    let left = [Phase::Dropout, Phase::HostPrim];
    row(&mut out, "left", left.iter().map(|p| mean(&|q| q.seconds(*p))).sum::<f64>() + rest, None, None, "6. what is left");
    for p in left {
        row(&mut out, &format!("  {}", p.key()), mean(&|q| q.seconds(p)), Some(mean(&|q| q.bytes[p.index()] as f64) / 1e6), Some(mean(&|q| q.count[p.index()] as f64)), p.what());
    }
    row(&mut out, "  left.interpreter", rest, None, None, "the host program's interpreter, by difference");
    row(&mut out, "total", call, None, None, &format!("a {unit}, timed around it"));
    let kernels = [Phase::Kernel, Phase::Dispatch, Phase::Threads].iter().map(|p| mean(&|q| q.seconds(*p))).sum::<f64>();
    let faults: Vec<u64> = calls.iter().filter_map(|(p, _)| p.faults).collect();
    if !faults.is_empty() {
        let _ = writeln!(out, "-- {:.0} minor page faults a {unit} (pages touched for the first time, {:.1} MB of 4 KB pages)", faults.iter().sum::<u64>() as f64 / faults.len() as f64, faults.iter().sum::<u64>() as f64 / faults.len() as f64 * 4096.0 / 1e6);
    }
    let _ = writeln!(
        out,
        "-- {:.0} launches a {unit}, {:.0} divided among {:.0} threads; the kernels as `--calibrate` times them (code, dispatch, threads) {:.3} ms, the host {:.3} ms",
        mean(&|q| q.launches as f64),
        mean(&|q| q.divided as f64),
        mean(&|q| q.threads as f64),
        kernels * 1e3,
        (call - kernels) * 1e3
    );
    out
}

/// A buffer `--launches` lists the address of: 64 KiB or more, the X925's
/// L1d. A smaller one is counted, not listed.
pub const LARGE: u64 = 64 * 1024;

/// The table `presk bench --host-profile --launches` prints after the
/// phases' (plan 0.17.2 step 4, D200): each launch of a call in its order,
/// its kernel, its own code's median and fastest over `calls`, its threads,
/// its nest; under it each of its buffers of [`LARGE`] bytes or more, at the
/// last call's virtual address and that address modulo 4 KiB, 64 KiB and the
/// set strides of the CPU's L1d and L2 (`strides`). `kernels`: the names of
/// the backend's kernels, by place.
pub fn launch_table(calls: &[(HostProfile, f64)], kernels: &[String], strides: Option<SetStrides>) -> String {
    use std::fmt::Write;
    let listed: Vec<&Vec<LaunchRow>> = calls.iter().filter_map(|(p, _)| p.rows.as_ref()).collect();
    let mut out = String::new();
    let Some(last) = listed.last() else {
        return out;
    };
    let same = listed.iter().all(|r| r.len() == last.len() && r.iter().zip(last.iter()).all(|(a, b)| a.kernel == b.kernel));
    let at = match strides {
        Some(s) => format!("cpu {}'s L1d and L2 set strides, {} KiB and {} KiB", s.cpu, s.l1 / 1024, s.l2 / 1024),
        None => "no cache's set stride (no sysfs)".to_string(),
    };
    let _ = writeln!(out, "-- the launches of a call timed by phase, in order (`--launches`): each kernel's own code, the median and the fastest of {} calls, and its buffers of {} KiB or more at the last call's virtual addresses, modulo 4 KiB, 64 KiB and {at}", listed.len(), LARGE / 1024);
    if !same {
        let _ = writeln!(out, "-- the calls launched different kernels: the last call's alone");
    }
    let calls_of: Vec<&Vec<LaunchRow>> = if same { listed.clone() } else { vec![last] };
    let moved = (0..last.len()).filter(|&i| calls_of.iter().any(|r| r[i].inputs.iter().chain(std::iter::once(&r[i].output)).zip(last[i].inputs.iter().chain(std::iter::once(&last[i].output))).any(|(a, b)| a.1 >= LARGE && a.0 != b.0))).count();
    let _ = writeln!(out, "-- {moved} of {} launches had a large buffer at another address in another call", last.len());
    let _ = writeln!(out, "{:>6}  {:<24} {:>11} {:>11} {:>7}  nest", "launch", "kernel", "µs median", "µs fastest", "threads");
    let _ = writeln!(out, "{:>6}  {:<24} {:>11} {:>18} {:>6} {:>6} {:>7} {:>7}", "", "  buffer", "bytes", "address", "%4K", "%64K", "%L1", "%L2");
    for (i, row) in last.iter().enumerate() {
        let mut t: Vec<f64> = calls_of.iter().map(|r| r[i].seconds).collect();
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let name = format!("k{} {}", row.kernel, kernels.get(row.kernel).map_or("", String::as_str));
        let _ = writeln!(out, "{:>6}  {:<24} {:>11.3} {:>11.3} {:>7}  `{}`", i + 1, name, t[t.len() / 2] * 1e6, t[0] * 1e6, row.workers, row.nest);
        let bufs = row.inputs.iter().enumerate().map(|(j, b)| (format!("in{j}"), *b)).chain(std::iter::once(("out".to_string(), row.output)));
        for (what, (addr, bytes)) in bufs.filter(|(_, (_, b))| *b >= LARGE) {
            let m = |n: u64| (addr as u64 % n).to_string();
            let (l1, l2) = strides.map_or(("—".to_string(), "—".to_string()), |s| (m(s.l1), m(s.l2)));
            let _ = writeln!(out, "{:>6}  {:<24} {:>11} {:>#18x} {:>6} {:>6} {:>7} {:>7}", "", format!("  {what}"), bytes, addr, m(4096), m(65536), l1, l2);
        }
    }
    out
}

/// Whether two values are the same bits: what `--host-profile` checks of a
/// call with the timers and one without.
pub fn same_bits(a: &Value, b: &Value) -> bool {
    fn bufs(a: &Buf, b: &Buf) -> bool {
        match (a, b) {
            (Buf::F64(x), Buf::F64(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()),
            (Buf::F32(x), Buf::F32(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()),
            (Buf::Bf16(x), Buf::Bf16(y)) => x == y,
            (Buf::I64(x), Buf::I64(y)) => x == y,
            (Buf::Exact(x), Buf::Exact(y)) => x == y,
            (Buf::Quant(x), Buf::Quant(y)) => x.data == y.data && x.scales.iter().zip(&y.scales).all(|(p, q)| p.to_bits() == q.to_bits()),
            _ => false,
        }
    }
    match (a, b) {
        (Value::Real(x), Value::Real(y)) => x.axes == y.axes && bufs(&x.data, &y.data),
        (Value::Idx { bound: m, t: x }, Value::Idx { bound: n, t: y }) => m == n && x.axes == y.axes && bufs(&x.data, &y.data),
        (Value::Tuple(xs), Value::Tuple(ys)) | (Value::Stacked(xs), Value::Stacked(ys)) => xs.len() == ys.len() && xs.iter().zip(ys).all(|(x, y)| same_bits(x, y)),
        (Value::Record(xs), Value::Record(ys)) => xs.len() == ys.len() && xs.iter().zip(ys).all(|((m, x), (n, y))| m == n && same_bits(x, y)),
        (Value::F64(x), Value::F64(y)) => x.to_bits() == y.to_bits(),
        (x, y) => x == y,
    }
}
