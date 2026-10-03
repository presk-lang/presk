//! A kernel executor over the printed C (plan 0.9.0 step 1, D77). The
//! kernels of a set of runs are printed, compiled by the C compiler with the
//! export's flags and `-Wall -Wextra -Werror` into a shared library, loaded
//! with `dlopen`, and called through the native backend's convention
//! (`presk_export_c::entry_fn`). A `KernelBackend` over this executor runs
//! an entry with its kernels in C, and its outputs are compared bit for bit
//! with the native backend's.
//!
//! The transcendental functions are the export's own, `presk_libm.c`
//! (step 2, D78), compiled into the same library. The tests of the functions
//! themselves (`tests/libm.rs`) and of the runtime (`tests/runtime.rs`)
//! load their C through [`So`] too.

#![allow(dead_code)]

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use presk_kernel::{Kernel, KernelExec, Threads};
use presk_rt::{Buf, RtError, Value};

/// A kernel's entry point: the buffer pointers, then the range.
type EntryFn = unsafe extern "C" fn(*const *mut u8, i64, i64);

/// The C compiler: `CC`, or `cc`.
pub fn cc() -> String {
    std::env::var("CC").unwrap_or_else(|_| "cc".into())
}

/// The flags every file is compiled with: the export's, and warnings as
/// errors (statement C3).
pub fn cflags() -> Vec<&'static str> {
    let mut f: Vec<&str> = presk_export_c::CFLAGS.to_vec();
    f.extend_from_slice(presk_export_c::WARNINGS);
    f.push("-Werror");
    f
}

/// The kernels of a set of runs, compiled and loaded.
pub struct CLib {
    /// Each kernel's entry point, by the kernel's C function with its name
    /// left out: two kernels that print the same are one.
    fns: HashMap<String, EntryFn>,
    pub dir: PathBuf,
    /// Distinct kernels, and those calling a transcendental function.
    pub kernels: usize,
    pub transcendental: usize,
    pub compile_secs: f64,
}

// SAFETY: the function pointers are into a library that is never closed, and
// calling them shares nothing but the buffers each call is handed.
unsafe impl Send for CLib {}
unsafe impl Sync for CLib {}

/// The key of a kernel: its C function under a fixed name.
fn key(k: &Kernel) -> String {
    presk_export_c::kernel_fn(k, "k").unwrap_or_else(|e| panic!("{e}"))
}

static BUILDS: AtomicUsize = AtomicUsize::new(0);

/// Runs the commands, as many at a time as there are cores, and panics with
/// the compiler's report on the first that fails.
fn run_all(cmds: Vec<(String, Command)>) {
    let par = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let mut pending = cmds.into_iter();
    let mut running: Vec<(String, std::process::Child)> = Vec::new();
    loop {
        while running.len() < par {
            let Some((what, mut c)) = pending.next() else { break };
            let child = c.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn().unwrap_or_else(|e| panic!("{}: {e} (the C export's tests need a C compiler; set CC)", cc()));
            running.push((what, child));
        }
        if running.is_empty() {
            return;
        }
        let (what, child) = running.remove(0);
        let out = child.wait_with_output().unwrap();
        assert!(out.status.success(), "{what}:\n{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    }
}

/// A shared library compiled from C files with the export's flags and
/// `-Werror`, linked with `-lm`, and loaded with `dlopen`. It is never
/// closed, so the pointers [`So::sym`] returns stay valid.
pub struct So {
    handle: *mut libc::c_void,
    pub path: PathBuf,
}

// SAFETY: a handle to a library that is never closed.
unsafe impl Send for So {}
unsafe impl Sync for So {}

/// A fresh directory for a test's C, `target/tmp/presk-export-c/<tag>`, with
/// the runtime's files the export copies: `presk_rt.h`, `presk_rt.c` and
/// `presk_libm.c`.
pub fn workdir(tag: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("presk-export-c").join(tag);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for (name, text) in presk_export_c::RUNTIME {
        std::fs::write(dir.join(name), text).unwrap();
    }
    dir
}

impl So {
    /// Compiles `sources` with [`cflags`], as many at a time as there are
    /// cores, into objects beside them, links them into a library in `dir`
    /// and loads it.
    pub fn build(dir: &Path, sources: &[PathBuf]) -> So {
        So::build_with(dir, sources, &cflags())
    }

    /// The same, with other flags.
    pub fn build_with(dir: &Path, sources: &[PathBuf], flags: &[&str]) -> So {
        let objects: Vec<PathBuf> = sources.iter().map(|s| s.with_extension("o")).collect();
        let cmds = sources
            .iter()
            .zip(&objects)
            .map(|(s, o)| {
                let mut c = Command::new(cc());
                c.args(flags).arg("-fPIC").arg("-c").arg(s).arg("-o").arg(o);
                (format!("{} {}", cc(), s.display()), c)
            })
            .collect();
        run_all(cmds);
        // A fresh name per build: `dlopen` returns the library already
        // loaded under a path, whatever the file holds now.
        let so = dir.join(format!("libpresk{}.so", BUILDS.fetch_add(1, Ordering::Relaxed)));
        let mut link = Command::new(cc());
        link.arg("-shared").arg("-o").arg(&so).args(&objects).arg("-lm");
        run_all(vec![(format!("linking {}", so.display()), link)]);
        let path = CString::new(so.as_os_str().as_bytes()).unwrap();
        // SAFETY: a library we just built, whose initializers are the C
        // runtime's; it is never closed.
        let handle = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        assert!(!handle.is_null(), "dlopen {}: {}", so.display(), unsafe { CStr::from_ptr(libc::dlerror()) }.to_string_lossy());
        So { handle, path: so }
    }

    /// The address of `name` in the library.
    pub fn sym(&self, name: &str) -> *mut libc::c_void {
        let n = CString::new(name).unwrap();
        // SAFETY: a handle `dlopen` returned and nobody closed.
        let p = unsafe { libc::dlsym(self.handle, n.as_ptr()) };
        assert!(!p.is_null(), "{name} is not in {}", self.path.display());
        p
    }
}

impl CLib {
    /// Prints every distinct kernel of `kernels`, compiles them with
    /// `presk_libm.c` under `target/tmp/presk-export-c/<tag>`, links them
    /// into one shared library and loads it.
    pub fn build<'a>(tag: &str, kernels: impl IntoIterator<Item = &'a Kernel>) -> CLib {
        CLib::build_with(tag, kernels, &cflags())
    }

    /// The same, every file compiled with `flags` in place of the export's.
    pub fn build_with<'a>(tag: &str, kernels: impl IntoIterator<Item = &'a Kernel>, flags: &[&str]) -> CLib {
        let started = Instant::now();
        let dir = workdir(tag);

        let mut seen: HashMap<String, usize> = HashMap::new();
        let mut distinct: Vec<&Kernel> = Vec::new();
        for k in kernels {
            if let std::collections::hash_map::Entry::Vacant(e) = seen.entry(key(k)) {
                e.insert(distinct.len());
                distinct.push(k);
            }
        }
        let transcendental = distinct.iter().filter(|k| !presk_export_c::calls(k).is_empty()).count();

        // Files of a few dozen kernels, compiled in parallel.
        const PER_FILE: usize = 48;
        let mut sources = vec![dir.join("presk_libm.c")];
        for (c, chunk) in distinct.chunks(PER_FILE).enumerate() {
            let mut text = String::from("#include \"presk_rt.h\"\n\n");
            for (j, k) in chunk.iter().enumerate() {
                let i = c * PER_FILE + j;
                text.push_str(&presk_export_c::kernel_fn(k, &format!("k{i}")).unwrap());
                text.push('\n');
                text.push_str(&presk_export_c::entry_fn(k, &format!("k{i}"), &format!("presk_k{i}")).unwrap());
                text.push('\n');
            }
            let path = dir.join(format!("kernels{c}.c"));
            std::fs::write(&path, text).unwrap();
            sources.push(path);
        }
        let so = So::build_with(&dir, &sources, flags);
        let mut fns = HashMap::new();
        for (key, i) in seen {
            // SAFETY: `presk_k<i>` was printed by `entry_fn` with this signature.
            let f = unsafe { std::mem::transmute::<*mut libc::c_void, EntryFn>(so.sym(&format!("presk_k{i}"))) };
            fns.insert(key, f);
        }
        CLib { fns, dir, kernels: distinct.len(), transcendental, compile_secs: started.elapsed().as_secs_f64() }
    }
}

/// How a kernel's `[lo, hi)` is cut: the whole loop in one call, the pieces
/// the native backend gives its workers, or one iteration a call. Pieces are
/// called last first, since no order may matter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Partition {
    Whole,
    Workers,
    Singles,
}

pub const PARTITIONS: [Partition; 3] = [Partition::Whole, Partition::Workers, Partition::Singles];

/// The executor. It has `workers` workers for the plan's checks (condition
/// 5), as the native one it is compared with, and runs every range on the
/// calling thread.
pub struct CExec {
    lib: Arc<CLib>,
    cache: Vec<Option<EntryFn>>,
    pub partition: Partition,
    pub workers: usize,
}

impl CExec {
    pub fn new(lib: Arc<CLib>, partition: Partition, workers: usize) -> CExec {
        CExec { lib, cache: vec![], partition, workers }
    }

    fn ranges(&self, k: &Kernel, n: usize) -> Vec<(i64, i64)> {
        let n = n as i64;
        let mut r: Vec<(i64, i64)> = match self.partition {
            Partition::Whole => vec![(0, n)],
            Partition::Workers => {
                let w = match k.threads {
                    Threads::At { workers, .. } => workers,
                    Threads::Default | Threads::One => self.workers,
                }
                .clamp(1, n.max(1) as usize) as i64;
                (0..w).map(|t| (n * t / w, n * (t + 1) / w)).filter(|(lo, hi)| lo < hi).collect()
            }
            Partition::Singles => (0..n).map(|i| (i, i + 1)).collect(),
        };
        r.reverse();
        if r.is_empty() { vec![(0, 0)] } else { r }
    }
}

fn ptr(b: &Buf) -> Result<*mut u8, RtError> {
    Ok(match b {
        Buf::F64(v) => v.as_ptr() as *mut u8,
        Buf::F32(v) => v.as_ptr() as *mut u8,
        Buf::Bf16(v) => v.as_ptr() as *mut u8,
        Buf::I64(v) => v.as_ptr() as *mut u8,
        // Integer weights: their integers here, their scales after the
        // output (`CExec::run`), as the native backend passes them.
        Buf::Quant(q) => q.data.as_ptr() as *mut u8,
        Buf::Exact(_) => return Err(RtError { code: "E10001", message: "the C executor: an exact buffer".into(), span: None }),
    })
}

/// The scales of the inputs of integer weights, in input order: the
/// pointers after the output (`presk_export_c::entry_fn`).
fn scales(k: &Kernel, inputs: &[&Buf]) -> Vec<*mut u8> {
    inputs
        .iter()
        .zip(&k.inputs)
        .filter_map(|(b, spec)| match (b, spec.kind) {
            (Buf::Quant(q), presk_kernel::BufKind::Quant { .. }) => Some(q.scales.as_ptr() as *mut u8),
            _ => None,
        })
        .collect()
}

impl KernelExec for CExec {
    fn name(&self) -> &'static str {
        "c"
    }

    fn workers(&self) -> Option<usize> {
        Some(self.workers)
    }

    fn run(&mut self, idx: usize, k: &Kernel, inputs: &[&Buf], out: &mut Buf) -> presk_rt::Result<()> {
        if self.cache.len() <= idx {
            self.cache.resize(idx + 1, None);
        }
        let f = match self.cache[idx] {
            Some(f) => f,
            None => {
                let f = *self.lib.fns.get(&key(k)).unwrap_or_else(|| panic!("kernel `{}` was not built into {}", k.name, self.lib.dir.display()));
                self.cache[idx] = Some(f);
                f
            }
        };
        let mut ptrs: Vec<*mut u8> = inputs.iter().map(|b| ptr(b)).collect::<Result<_, _>>()?;
        let ranges = match presk_export_c::ranged(k) {
            Some((_, n)) => self.ranges(k, n),
            None => vec![(0, 0)],
        };
        // Each piece alone, on an output of sentinels: the elements it
        // changes are its own, and no other piece may change one of them.
        // A range that bounded the wrong loop, or none, would recompute
        // elements, give the same bits, and be a race between threads.
        if ranges.len() > 1 {
            let mut owner: Vec<Option<usize>> = vec![None; out.len()];
            for (j, &(lo, hi)) in ranges.iter().enumerate() {
                let probe = sentinels(out);
                let mut with = ptrs.clone();
                with.push(ptr(&probe)?);
                with.extend(scales(k, inputs));
                // SAFETY: as below, on a buffer of the output's type and size.
                unsafe { f(with.as_ptr(), lo, hi) };
                for e in changed(&probe) {
                    if let Some(i) = owner[e] {
                        panic!("kernel `{}`: the ranges {:?} and {:?} both write element {e}", k.name, ranges[i], (lo, hi));
                    }
                    owner[e] = Some(j);
                }
            }
        }
        ptrs.push(ptr(out)?);
        ptrs.extend(scales(k, inputs));
        for (lo, hi) in ranges {
            // SAFETY: the kernel's loop bounds are the buffers' dimensions, as
            // for the native backend, which runs the same loops; the ranges
            // are disjoint pieces of a parallel loop.
            unsafe { f(ptrs.as_ptr(), lo, hi) };
        }
        Ok(())
    }
}

/// The sentinel of an output's type, as raw bits: a value no kernel stores.
/// A signalling NaN, which arithmetic makes quiet, as a `bf16` store does;
/// an index no shape reaches.
fn sentinel(b: &Buf) -> u64 {
    match b {
        Buf::F64(_) => 0x7ff4_dead_beef_0001,
        Buf::F32(_) => 0x7fa0_beef,
        Buf::Bf16(_) => 0x7f81,
        Buf::I64(_) => (i64::MIN + 0xbeef) as u64,
        Buf::Exact(_) | Buf::Quant(_) => unreachable!("an output is a machine buffer"),
    }
}

/// A buffer like `b`, every element the sentinel.
fn sentinels(b: &Buf) -> Buf {
    let s = sentinel(b);
    match b {
        Buf::F64(x) => Buf::F64(vec![f64::from_bits(s); x.len()].into()),
        Buf::F32(x) => Buf::F32(vec![f32::from_bits(s as u32); x.len()].into()),
        Buf::Bf16(x) => Buf::Bf16(vec![s as u16; x.len()].into()),
        Buf::I64(x) => Buf::I64(vec![s as i64; x.len()].into()),
        Buf::Exact(_) | Buf::Quant(_) => unreachable!("an output is a machine buffer"),
    }
}

/// The elements of a buffer of sentinels that a kernel changed.
fn changed(b: &Buf) -> Vec<usize> {
    let s = sentinel(b);
    raw_buf(b).iter().enumerate().filter(|(_, x)| **x != s).map(|(i, _)| i).collect()
}

fn raw_buf(b: &Buf) -> Vec<u64> {
    match b {
        Buf::F64(x) => x.iter().map(|y| y.to_bits()).collect(),
        Buf::F32(x) => x.iter().map(|y| y.to_bits() as u64).collect(),
        Buf::Bf16(x) => x.iter().map(|y| *y as u64).collect(),
        Buf::I64(x) => x.iter().map(|y| *y as u64).collect(),
        Buf::Exact(_) | Buf::Quant(_) => panic!("an exact or integer-weight result"),
    }
}

/// Every leaf's raw bits, in order: what "bit-identical" compares.
pub fn raw(v: &Value) -> Vec<u64> {
    match v {
        Value::Real(t) | Value::Idx { t, .. } => raw_buf(&t.data),
        Value::Tuple(xs) | Value::Stacked(xs) => xs.iter().flat_map(raw).collect(),
        Value::Record(xs) => xs.iter().flat_map(|(_, x)| raw(x)).collect(),
        Value::Nat(n) => vec![*n],
        Value::Int(i) => vec![*i as u64],
        Value::F64(x) => vec![x.to_bits()],
        Value::Bool(b) => vec![*b as u64],
        Value::Key(_) => vec![],
    }
}
