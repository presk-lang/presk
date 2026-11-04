//! The LLVM backend (plan 0.17.0 step 3, D176; route (a) of the plan's
//! question 1, the user's choice): each kernel printed as C by the C
//! export's printer, compiled by LLVM's `clang -O3 -mcpu=native` with the
//! export's other flags ([`cflags`]), loaded with `dlopen`, and called
//! through the native backend's
//! convention. The host, the worker pool, the plans' splits and the
//! profiling are the native backend's own ([`NativeExec::with_compiler`]),
//! so the two backends differ by their kernels' code alone.
//!
//! **The bits are the native backend's.** The printer is statement C1's:
//! the C it prints computes the native backend's bits under `-std=c11
//! -ffp-contract=off` and without fast-math, whatever the optimization
//! level, since no flag lets the compiler reorder or fuse an operation. A
//! one-lane kernel is printed with the lanes the native backend gives it
//! ([`presk_cranelift::laned`]), which change no bit either. The
//! transcendental functions are the export's, `presk_libm.c` (musl 1.1.22's,
//! bit-identical to the `libm` crate, D78): compiled into the same unit as a
//! kernel that calls one, so that clang may inline it, and once into an
//! object the other kernels' libraries link.
//!
//! **The cache.** A kernel's library is kept under a hash of everything it
//! is built from: the compiler's version line, the flags, and the C text,
//! which is stored beside it and compared on every hit, so a collision
//! recompiles. The directory is `PRESK_LLVM_CACHE`, else
//! `$XDG_CACHE_HOME/presk/llvm`, else `~/.cache/presk/llvm`. A build is
//! written under a name of its own and renamed into place, so two processes
//! or threads building the same kernel do not see each other's halves.
//!
//! **The compiler** is `PRESK_CLANG`, else the `clang` of `make llvm` in
//! the workspace this crate was built from (`.llvm/bin/clang`). Without it
//! the backend is unavailable, and says so.

use std::ffi::{CStr, CString};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use presk_core::Program;
use presk_cranelift::{KernelCompiler, KernelFn, NativeExec};
use presk_kernel::{Kernel, KernelBackend};
use presk_rt::{Realization, Result, RtError};
use presk_types::elab::ElabProgram;

/// The flags every kernel and `presk_libm.c` are compiled with: the C
/// export's ([`presk_export_c::CFLAGS`]) at `-O3` in place of `-O2`, for the
/// machine the backend runs on (`-mcpu=native`, as Cranelift compiles for
/// its host), position-independent for a shared library. None of them lets
/// the compiler reorder or fuse a floating-point operation:
/// `-ffp-contract=off` stays, and no fast-math flag is given.
pub fn cflags() -> Vec<&'static str> {
    let mut f: Vec<&'static str> = presk_export_c::CFLAGS.iter().map(|&x| if x == "-O2" { "-O3" } else { x }).collect();
    f.push("-mcpu=native");
    f.push("-fPIC");
    f
}

/// What `-mcpu=native` means here: the target CPU and features clang's
/// driver resolves (`-###`), part of every cache key, so that a cache
/// shared between machines never hands one the code of another.
fn native_target(cc: &Path) -> String {
    let out = Command::new(cc).args(["-mcpu=native", "-###", "-c", "-x", "c", "/dev/null", "-o", "/dev/null"]).output();
    let text = out.map(|o| String::from_utf8_lossy(&o.stderr).into_owned()).unwrap_or_default();
    let words: Vec<&str> = text.split_whitespace().collect();
    words.windows(2).filter(|w| w[0] == "\"-target-cpu\"" || w[0] == "\"-target-feature\"").map(|w| w[1]).collect::<Vec<_>>().join(" ")
}

fn err(message: String) -> RtError {
    RtError { code: "E10001", message, span: None }
}

/// The compiler: `PRESK_CLANG`, else `make llvm`'s in this workspace.
pub fn clang() -> PathBuf {
    match std::env::var_os("PRESK_CLANG") {
        Some(p) => PathBuf::from(p),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.llvm/bin/clang"),
    }
}

/// The compiler's version line, or why there is none: whether the backend
/// is available.
pub fn available() -> std::result::Result<String, String> {
    let cc = clang();
    let out = Command::new(&cc).arg("--version").output().map_err(|e| format!("no C compiler at {} ({e}; `make llvm`, or set PRESK_CLANG)", cc.display()))?;
    if !out.status.success() {
        return Err(format!("{} --version failed", cc.display()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).lines().next().unwrap_or("").to_string())
}

/// The cache's directory.
pub fn cache_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("PRESK_LLVM_CACHE") {
        return PathBuf::from(p);
    }
    if let Some(p) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(p).join("presk/llvm");
    }
    match std::env::var_os("HOME") {
        Some(h) => PathBuf::from(h).join(".cache/presk/llvm"),
        None => std::env::temp_dir().join("presk-llvm"),
    }
}

/// FNV-1a over 64 bits: a hash that is the same in every build of this
/// crate, which the cache's names need (`std`'s hasher may change). The
/// text stored beside each entry settles a collision.
fn fnv(parts: &[&[u8]]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for p in parts {
        for &b in p.iter() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        // A separator, so that the parts' boundaries count.
        h ^= 0xff;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// One of the C export's runtime files by name (`presk_rt.h`,
/// `presk_libm.c`).
fn runtime(name: &str) -> &'static str {
    presk_export_c::RUNTIME.iter().find(|(n, _)| *n == name).map(|(_, t)| *t).expect("a runtime file of the C export")
}

static BUILDS: AtomicU64 = AtomicU64::new(0);

/// A fresh directory beside `final_`, for a build to rename into place.
fn scratch(final_: &Path) -> PathBuf {
    let n = BUILDS.fetch_add(1, Ordering::Relaxed);
    final_.with_extension(format!("tmp-{}-{n}", std::process::id()))
}

fn run(mut c: Command, what: &str) -> Result<()> {
    let out = c.output().map_err(|e| err(format!("{what}: {e}")))?;
    if !out.status.success() {
        return Err(err(format!("{what}:\n{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))));
    }
    Ok(())
}

/// A library loaded and never closed, so the functions it gives stay valid.
struct Lib(*mut libc::c_void);

impl Lib {
    fn open(path: &Path) -> Result<Lib> {
        let c = CString::new(path.as_os_str().as_bytes()).map_err(|e| err(e.to_string()))?;
        // SAFETY: a library this crate built from the printer's C, whose
        // initializers are the C runtime's; it is never closed.
        let h = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if h.is_null() {
            // SAFETY: `dlerror` after a failed `dlopen` returns a message.
            let m = unsafe { CStr::from_ptr(libc::dlerror()) }.to_string_lossy().into_owned();
            return Err(err(format!("dlopen {}: {m}", path.display())));
        }
        Ok(Lib(h))
    }

    fn sym(&self, name: &str) -> Result<*mut libc::c_void> {
        let n = CString::new(name).map_err(|e| err(e.to_string()))?;
        // SAFETY: a handle `dlopen` returned and nobody closed.
        let p = unsafe { libc::dlsym(self.0, n.as_ptr()) };
        if p.is_null() {
            return Err(err(format!("`{name}` is not in the kernel's library")));
        }
        Ok(p)
    }
}

/// What the compiler did, for `presk bench` and NB4: kernels compiled and
/// found in the cache, and the seconds spent compiling.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub compiled: usize,
    pub cached: usize,
    pub seconds: f64,
}

/// The compiler of kernels the native executor uses in place of Cranelift.
pub struct LlvmCompiler {
    clang: PathBuf,
    /// The compiler's version line and the target `-mcpu=native` resolves.
    version: String,
    cache: PathBuf,
    flags: Vec<&'static str>,
    /// `presk_libm.c`'s object, built once.
    libm: Option<PathBuf>,
    libs: Vec<Lib>,
    pub stats: Stats,
}

impl LlvmCompiler {
    pub fn new() -> Result<LlvmCompiler> {
        let version = format!("{} {}", available().map_err(err)?, native_target(&clang()));
        let cache = cache_dir();
        std::fs::create_dir_all(&cache).map_err(|e| err(format!("{}: {e}", cache.display())))?;
        Ok(LlvmCompiler { clang: clang(), version, cache, flags: cflags(), libm: None, libs: Vec::new(), stats: Stats::default() })
    }

    /// The key of what a file is built from.
    fn key(&self, text: &str) -> String {
        format!("{:016x}", fnv(&[self.version.as_bytes(), self.flags.join(" ").as_bytes(), text.as_bytes()]))
    }

    /// `presk_libm.c` compiled, from the cache or now.
    fn libm(&mut self) -> Result<PathBuf> {
        if let Some(p) = &self.libm {
            return Ok(p.clone());
        }
        let key = self.key(&format!("{}{}", runtime("presk_rt.h"), runtime("presk_libm.c")));
        let dir = self.cache.join(format!("libm-{key}"));
        let obj = dir.join("presk_libm.o");
        if !obj.exists() {
            let tmp = scratch(&dir);
            std::fs::create_dir_all(&tmp).map_err(|e| err(e.to_string()))?;
            for (name, text) in presk_export_c::RUNTIME {
                std::fs::write(tmp.join(name), text).map_err(|e| err(e.to_string()))?;
            }
            let mut c = Command::new(&self.clang);
            c.args(&self.flags).arg("-c").arg(tmp.join("presk_libm.c")).arg("-o").arg(tmp.join("presk_libm.o"));
            run(c, &format!("{} presk_libm.c", self.clang.display()))?;
            if std::fs::rename(&tmp, &dir).is_err() {
                // Another build got there first: its object is the same.
                let _ = std::fs::remove_dir_all(&tmp);
            }
        }
        self.libm = Some(obj.clone());
        Ok(obj)
    }
}

impl KernelCompiler for LlvmCompiler {
    fn name(&self) -> &'static str {
        "llvm"
    }

    fn compile(&mut self, k: &Kernel) -> Result<KernelFn> {
        let started = Instant::now();
        let laned = presk_cranelift::laned(k);
        let k = laned.as_ref().unwrap_or(k);
        // A kernel that calls a transcendental function is compiled with
        // `presk_libm.c` in its own unit, so that clang can inline the
        // function into the loop; the same functions, so the same bits.
        let with_libm = !presk_export_c::calls(k).is_empty();
        let mut text = String::from(runtime("presk_rt.h"));
        if with_libm {
            text.push_str("\n/* The C export's functions (presk_libm.c). */\n");
            text.push_str(runtime("presk_libm.c"));
        }
        text.push_str("\n/* The kernel (printed by presk_export_c::kernel_fn). */\n");
        text.push_str(&presk_export_c::kernel_fn(k, "kernel").map_err(|e| err(e.to_string()))?);
        text.push('\n');
        text.push_str(&presk_export_c::entry_fn(k, "kernel", "presk_kernel").map_err(|e| err(e.to_string()))?);
        text.push('\n');
        let key = self.key(&text);
        let dir = self.cache.join(&key[..2]).join(&key);
        let (c, so) = (dir.join("kernel.c"), dir.join("kernel.so"));
        let hit = so.exists() && std::fs::read_to_string(&c).is_ok_and(|t| t == text);
        if hit {
            self.stats.cached += 1;
        } else {
            let libm = if with_libm { None } else { Some(self.libm()?) };
            let tmp = scratch(&dir);
            std::fs::create_dir_all(&tmp).map_err(|e| err(e.to_string()))?;
            std::fs::write(tmp.join("kernel.c"), &text).map_err(|e| err(e.to_string()))?;
            // `presk_libm.c` includes the header the text begins with; its
            // guard makes the second inclusion empty.
            std::fs::write(tmp.join("presk_rt.h"), runtime("presk_rt.h")).map_err(|e| err(e.to_string()))?;
            let mut cc = Command::new(&self.clang);
            cc.args(&self.flags).arg("-c").arg(tmp.join("kernel.c")).arg("-o").arg(tmp.join("kernel.o"));
            run(cc, &format!("{} kernel `{}`", self.clang.display(), k.name))?;
            let mut ld = Command::new(&self.clang);
            ld.arg("-shared").arg("-o").arg(tmp.join("kernel.so")).arg(tmp.join("kernel.o"));
            if let Some(libm) = &libm {
                ld.arg(libm);
            }
            ld.arg("-lm");
            run(ld, &format!("linking kernel `{}`", k.name))?;
            if let Some(parent) = dir.parent() {
                std::fs::create_dir_all(parent).map_err(|e| err(e.to_string()))?;
            }
            if std::fs::rename(&tmp, &dir).is_err() {
                // Another build of the same text got there first; a stale
                // entry whose text differs (a collision) is replaced.
                if !std::fs::read_to_string(&c).is_ok_and(|t| t == text) {
                    let _ = std::fs::remove_dir_all(&dir);
                    std::fs::rename(&tmp, &dir).map_err(|e| err(format!("{}: {e}", dir.display())))?;
                } else {
                    let _ = std::fs::remove_dir_all(&tmp);
                }
            }
            self.stats.compiled += 1;
        }
        let lib = Lib::open(&so)?;
        let p = lib.sym("presk_kernel")?;
        self.libs.push(lib);
        self.stats.seconds += started.elapsed().as_secs_f64();
        // SAFETY: `presk_kernel` was printed by `entry_fn` with this
        // signature, the native backend's.
        Ok(unsafe { std::mem::transmute::<*mut libc::c_void, KernelFn>(p) })
    }
}

/// A kernel backend whose kernels are compiled by LLVM (`--backend llvm`),
/// with the native backend's default lanes.
pub fn llvm_backend<'p>(prog: &'p Program, elab: &'p ElabProgram, realize: &Realization) -> Result<KernelBackend<'p>> {
    let exec = NativeExec::with_compiler(Box::new(LlvmCompiler::new()?))?;
    let mut b = KernelBackend::with_exec(prog, elab, realize, Box::new(exec));
    b.default_layout.lanes = None;
    Ok(b)
}

/// The same with an explicit number of workers.
pub fn llvm_backend_with_threads<'p>(prog: &'p Program, elab: &'p ElabProgram, realize: &Realization, threads: usize) -> Result<KernelBackend<'p>> {
    let mut exec = NativeExec::with_compiler(Box::new(LlvmCompiler::new()?))?;
    exec.threads = threads.max(1);
    let mut b = KernelBackend::with_exec(prog, elab, realize, Box::new(exec));
    b.default_layout.lanes = None;
    Ok(b)
}
