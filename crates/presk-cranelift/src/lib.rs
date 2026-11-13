//! The Cranelift backend (plan 0.3.0, step 3): every PSK-Kernel v0 kernel is
//! compiled once to machine code by Cranelift's JIT and called through a
//! function pointer; the host program is the kernel backend's. Buffers are
//! passed as an array of pointers, loops become basic blocks, `sqrt` is
//! native and the other transcendental functions and `erf` are calls into
//! the `libm` crate, a port of musl's (D76). The kernel interpreter calls
//! the same crate for every one of them since 0.16.0 (D159), so that the two
//! executors compute the same bits; until then it called it for `erf` only,
//! and Rust's standard library, the system's C library, for the others.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

mod lanes;
pub mod pool;
pub use pool::Pool;

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, SourceLoc, Type, Value as CV, types};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};
use presk_core::Program;
use presk_kernel::ir::{BinOp, BufKind, CmpOp, Dtype, Expr, Kernel, Stmt, Threads, UnOp};
use presk_kernel::{KernelBackend, KernelExec, Overhead};
use presk_rt::{Buf, Realization, RtError};
use presk_types::elab::ElabProgram;

type Result<T> = std::result::Result<T, RtError>;

fn err<T>(message: impl Into<String>) -> Result<T> {
    Err(RtError { code: "E10001", message: message.into(), span: None })
}

/// A compiled kernel: the buffer pointers, then the half-open range of its
/// outermost loop. A sequential call passes the whole range.
/// A compiled kernel: the table of buffers (the inputs, the output, the
/// scales of integer inputs), then the range of the divided loop.
pub type KernelFn = unsafe extern "C" fn(*const *mut u8, i64, i64);

/// A compiler of kernels to [`KernelFn`], which the executor uses in place
/// of Cranelift (plan 0.17.0 step 3, D176): the LLVM backend's, through the
/// C printer and `clang`. The host, the pool, the splits and the profiling
/// stay this executor's, so the two backends differ by their code alone.
pub trait KernelCompiler {
    /// The executor's name with this compiler, as the reports print it.
    fn name(&self) -> &'static str;
    /// Kernel `k`'s code, alive for the executor's lifetime.
    fn compile(&mut self, k: &Kernel) -> Result<KernelFn>;
}

/// The buffer table, sent to a worker thread. Safe because the threads of one
/// launch share read-only inputs and write disjoint elements of the output:
/// that is what `Kernel::parallel_at` establishes before we split.
#[derive(Clone, Copy)]
struct Ptrs(*const *mut u8);

impl Ptrs {
    /// Taking the whole wrapper (not its field) is what makes the closure
    /// capture a `Send` value.
    fn get(self) -> *const *mut u8 {
        self.0
    }
}
// SAFETY: see `Ptrs`.
unsafe impl Send for Ptrs {}
// SAFETY: see `Ptrs`; the pool's workers share the table, each reading it.
unsafe impl Sync for Ptrs {}

/// Below this many weighted operations a kernel runs on one thread by default
/// (`presk_kernel::cost::PARALLEL_MIN_WORK`, which the cost model reads too).
use presk_kernel::cost::PARALLEL_MIN_WORK;

/// The JIT: compiles kernels on first use and runs the machine code.
pub struct NativeExec {
    /// Workers for a parallel kernel; `PRESK_THREADS` overrides the core count.
    pub threads: usize,
    /// Launches that were split, for the tests and the manual.
    pub threaded: u64,
    module: JITModule,
    ctx: cranelift_codegen::Context,
    fbc: FunctionBuilderContext,
    compiled: Vec<Option<KernelFn>>,
    libm: HashMap<String, FuncId>,
    pub compiled_count: usize,
    /// Whether the host profiles (`presk bench --host-profile`): each launch
    /// then times its compilation, its dispatch and its threads into `last`.
    profiling: bool,
    last: Overhead,
    /// The workers beside the calling thread, `threads − 1`, made at the
    /// first divided launch (plan 0.10.0 step 6).
    pool: Option<Pool>,
    /// Whether each kernel's machine code is kept, with the line of the
    /// kernel each byte compiles from (plan 0.13.0 step 5): only for
    /// [`machine_code`], so that a run compiles as it always has.
    keep: bool,
    kept: Vec<Option<MachineCode>>,
    /// Whether each kernel's bytes are kept without the lines, for
    /// [`plain_code`].
    plain: bool,
    plain_kept: Vec<Vec<u8>>,
    /// The size of the last kernel compiled, in bytes.
    last_size: usize,
    /// The compiler used in place of Cranelift, when there is one.
    external: Option<Box<dyn KernelCompiler>>,
    /// The seconds compiling took, timed whether the host profiles or not
    /// (plan 0.17.2 step 4, D200), and the kernels compiled by `run`.
    compile_s: f64,
    compiles: u64,
}

/// The machine code of a compiled kernel (plan 0.13.0 step 5, D138): its
/// bytes as Cranelift emitted them, before the JIT links them (since plan
/// 0.14.0, D145: the linked bytes hold the address of a `libm` function,
/// which differs from one process to the next), the bytes each line of the
/// kernel's text compiles to, Cranelift's text of its instructions, and the
/// fields the linker writes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MachineCode {
    pub bytes: Vec<u8>,
    /// `(start, end, line)`: the half-open range of bytes, and the line of
    /// the kernel's printed text (from 1, its header) they compile from,
    /// sorted, covering every byte once. The header's line holds the
    /// prologue, the closing brace's the return.
    pub ranges: Vec<(u32, u32, u32)>,
    /// The ranges of bytes Cranelift attributed to no line, which `ranges`
    /// gives to the line of the code before them (the function's prologue,
    /// before any line, to its header).
    pub given: Vec<(u32, u32)>,
    /// Cranelift's text of the instructions it emitted, the register
    /// allocator's result (`Context::set_disasm`).
    pub disasm: String,
    /// `(start, end, target)`: each field the linker writes, zero in
    /// `bytes`, and the symbol whose address it receives (`exp`).
    pub relocs: Vec<(u32, u32, String)>,
}

/// The machine code of each kernel, compiled as `presk run --backend
/// native` compiles it, with each byte's line (plan 0.13.0 step 5). Nothing
/// is run.
pub fn machine_code(kernels: &[Kernel]) -> Result<Vec<MachineCode>> {
    let mut x = NativeExec::new()?;
    x.keep = true;
    for (j, k) in kernels.iter().enumerate() {
        x.compile(j, k)?;
    }
    Ok(x.kept.into_iter().map(|m| m.expect("every kernel compiled")).collect())
}

/// The machine code of each kernel compiled as a run compiles it, without
/// the lines, before it is linked as [`machine_code`]'s is: the test that
/// the lines change no byte.
pub fn plain_code(kernels: &[Kernel]) -> Result<Vec<Vec<u8>>> {
    let mut x = NativeExec::new()?;
    x.plain = true;
    for (j, k) in kernels.iter().enumerate() {
        x.compile(j, k)?;
    }
    Ok(x.plain_kept)
}

impl NativeExec {
    pub fn new() -> Result<NativeExec> {
        let mut flags = settings::builder();
        flags.set("use_colocated_libcalls", "false").map_err(|e| RtError { code: "E10001", message: e.to_string(), span: None })?;
        flags.set("is_pic", "false").map_err(|e| RtError { code: "E10001", message: e.to_string(), span: None })?;
        flags.set("opt_level", "speed").map_err(|e| RtError { code: "E10001", message: e.to_string(), span: None })?;
        let isa = cranelift_native::builder().map_err(|e| RtError { code: "E10001", message: format!("cranelift: {e}"), span: None })?.finish(settings::Flags::new(flags)).map_err(|e| RtError { code: "E10001", message: e.to_string(), span: None })?;
        let mut jb = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        jb.symbol("exp", libm::exp as *const u8);
        jb.symbol("log", libm::log as *const u8);
        jb.symbol("tanh", libm::tanh as *const u8);
        jb.symbol("sin", libm::sin as *const u8);
        jb.symbol("cos", libm::cos as *const u8);
        jb.symbol("erf", libm::erf as *const u8);
        jb.symbol("expf", libm::expf as *const u8);
        jb.symbol("logf", libm::logf as *const u8);
        jb.symbol("tanhf", libm::tanhf as *const u8);
        jb.symbol("sinf", libm::sinf as *const u8);
        jb.symbol("cosf", libm::cosf as *const u8);
        jb.symbol("erff", libm::erff as *const u8);
        let module = JITModule::new(jb);
        let ctx = module.make_context();
        // `PRESK_THREADS` overrides the core count; 1 disables the split, which
        // is how the tests check that threading changes nothing but the time.
        let threads = std::env::var("PRESK_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
            .unwrap_or(1)
            .max(1);
        Ok(NativeExec { module, ctx, fbc: FunctionBuilderContext::new(), compiled: vec![], libm: HashMap::new(), compiled_count: 0, threads, threaded: 0, profiling: false, last: Overhead::default(), pool: None, keep: false, kept: Vec::new(), plain: false, plain_kept: Vec::new(), last_size: 0, external: None, compile_s: 0.0, compiles: 0 })
    }

    /// The executor with `c` compiling its kernels in place of Cranelift.
    pub fn with_compiler(c: Box<dyn KernelCompiler>) -> Result<NativeExec> {
        let mut x = NativeExec::new()?;
        x.external = Some(c);
        Ok(x)
    }

    /// The pool, of `threads − 1` workers: made, or made again when
    /// `threads` changed since.
    fn pool(&mut self) -> &mut Pool {
        let n = self.threads.saturating_sub(1);
        if self.pool.as_ref().is_none_or(|p| p.len() != n) {
            self.pool = Some(Pool::new(n));
        }
        self.pool.as_mut().expect("just made")
    }

    fn libm_fn(&mut self, name: &str, ty: Type) -> Result<FuncId> {
        if let Some(id) = self.libm.get(name) {
            return Ok(*id);
        }
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ty));
        sig.returns.push(AbiParam::new(ty));
        let id = self.module.declare_function(name, Linkage::Import, &sig).map_err(|e| RtError { code: "E10001", message: e.to_string(), span: None })?;
        self.libm.insert(name.to_string(), id);
        Ok(id)
    }

    fn compile(&mut self, idx: usize, k: &Kernel) -> Result<KernelFn> {
        // The loop a split divides, and a scalar kernel's lane-wise loops
        // run over lanes ([`lanes`]): the same bits, the kernel's text and
        // lines unchanged.
        let split = ranged(k).map(|(d, _)| d);
        let laned = lanes::natural(k, split);
        let k = laned.as_ref().unwrap_or(k);
        // Cranelift's vector types are 128 bits wide, so the lane count a
        // kernel asks for has to be the one the machine type gives: four
        // `f32` or two `f64`. `bf16` is storage-only and would need a
        // conversion per lane, which is step 5's business.
        let vty = match (k.dtype, k.lanes) {
            (_, 1) => None,
            (Dtype::F32, 4) => Some(types::F32X4),
            (Dtype::F64, 2) => Some(types::F64X2),
            (d, n) => return err(format!("the native backend has no {n}-lane vector for {d}")),
        };
        // `bf16` is a *storage* type: the loop nest is generated in `f32` and
        // only the loads and stores narrow (D34). So the arithmetic type and
        // the element size part company here, and nowhere else.
        let fty = match k.dtype {
            Dtype::F64 => types::F64,
            Dtype::F32 | Dtype::Bf16 => types::F32,
        };
        let real_bytes = k.dtype.bytes() as i64;
        // Declare the libm functions the kernel needs before building.
        let mut needed: Vec<&'static str> = Vec::new();
        collect_libm(&k.body, &mut needed);
        // In the names' order, so that the function declares them, and its
        // text numbers them, the same way in every process (D145).
        let mut libm_ids: std::collections::BTreeMap<&'static str, FuncId> = std::collections::BTreeMap::new();
        for n in needed {
            let name = match (k.dtype, n) {
                (Dtype::F64, n) => n.to_string(),
                (Dtype::F32 | Dtype::Bf16, n) => format!("{n}f"),
            };
            libm_ids.insert(n, self.libm_fn(&name, fty)?);
        }
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(types::I64));
        sig.params.push(AbiParam::new(types::I64));
        sig.params.push(AbiParam::new(types::I64));
        let fid = self.module.declare_function(&format!("k{idx}"), Linkage::Local, &sig).map_err(|e| RtError { code: "E10001", message: e.to_string(), span: None })?;
        self.ctx.func.signature = sig;
        let keep = self.keep;
        self.ctx.set_disasm(keep);
        {
            let mut b = FunctionBuilder::new(&mut self.ctx.func, &mut self.fbc);
            // The prologue is the kernel's header line's, line 1.
            if keep {
                b.set_srcloc(SourceLoc::new(1));
            }
            let entry = b.create_block();
            b.append_block_params_for_function_params(entry);
            b.switch_to_block(entry);
            b.seal_block(entry);
            let (bufs_ptr, lo, hi) = (b.block_params(entry)[0], b.block_params(entry)[1], b.block_params(entry)[2]);
            let nbufs = k.inputs.len() + 1;
            let bufs: Vec<CV> = (0..nbufs).map(|i| b.ins().load(types::I64, MemFlags::trusted(), bufs_ptr, (8 * i) as i32)).collect();
            let loops: Vec<Variable> = (0..k.nloops).map(|_| b.declare_var(types::I64)).collect();
            let reals: Vec<Variable> = (0..k.nreal).map(|_| b.declare_var(fty)).collect();
            let ints: Vec<Variable> = (0..k.nint).map(|_| b.declare_var(types::I64)).collect();
            let zero_i = b.ins().iconst(types::I64, 0);
            for v in loops.iter().chain(ints.iter()) {
                b.def_var(*v, zero_i);
            }
            let zero_f = fconst(&mut b, fty, 0.0);
            for v in &reals {
                b.def_var(*v, zero_f);
            }
            let strides: Vec<Vec<usize>> = k.inputs.iter().map(|s| s.strides()).chain(std::iter::once(k.output.strides())).collect();
            let esize: Vec<i64> = k
                .inputs
                .iter()
                .map(|s| match s.kind {
                    BufKind::Idx => 8,
                    BufKind::Quant { .. } => 1,
                    BufKind::Wide(d) => d.bytes() as i64,
                    BufKind::Real => real_bytes,
                })
                .chain(std::iter::once(if k.output.kind == BufKind::Idx { 8 } else { real_bytes }))
                .collect();
            // The scales of each integer input, in the pointer slots after the
            // output, in input order (`NativeExec::run` puts them there).
            let mut slot = nbufs;
            let scales: Vec<Option<(CV, Vec<usize>)>> = k
                .inputs
                .iter()
                .map(|s| match s.kind {
                    BufKind::Quant { .. } => {
                        let p = b.ins().load(types::I64, MemFlags::trusted(), bufs_ptr, (8 * slot) as i32);
                        slot += 1;
                        Some((p, s.scale_strides()))
                    }
                    _ => None,
                })
                .collect();
            let mut refs: HashMap<&'static str, cranelift_codegen::ir::FuncRef> = HashMap::new();
            for (n, id) in &libm_ids {
                refs.insert(n, self.module.declare_func_in_func(*id, b.func));
            }
            let vecs: Vec<Variable> = match vty {
                Some(t) => (0..k.nvec).map(|_| b.declare_var(t)).collect(),
                None => Vec::new(),
            };
            let nibbles: Vec<bool> = k.inputs.iter().map(|s| matches!(s.kind, BufKind::Quant { bits: 4, .. })).collect();
            let wide: Vec<Option<Dtype>> = k.inputs.iter().map(|s| if let BufKind::Wide(d) = s.kind { Some(d) } else { None }).collect();
            let affine_loops = affine_loops(&k.body, k.nloops);
            let mut g = Gen { b, fty, vty, narrow: k.dtype == Dtype::Bf16, bufs, strides, esize, scales, nibbles, wide, loops, reals, ints, out: k.inputs.len(), depth_of: vec![0; k.nloops], idepth_of: vec![0; k.nint], depth: 0, promo: None, vecs, libm: refs, keep, line: 1, affine_loops, guarded: 0 };
            // The loop a split divides — the outermost by default, the one at
            // the depth the plan states (0.7.0 step 3) — runs over the range
            // the caller gives; the loops enclosing it run in full. One
            // compiled kernel serves the sequential call and every worker.
            match split {
                Some(depth) => g.range_at(&k.body, depth, lo, hi),
                None => g.stmts(&k.body),
            }
            // The return is the closing brace's, the text's last line.
            g.next_line();
            g.b.ins().return_(&[]);
            g.b.finalize();
        }
        self.module.define_function(fid, &mut self.ctx).map_err(|e| RtError { code: "E10001", message: format!("cranelift: {e}"), span: None })?;
        // The bytes before the JIT links them, and the fields it writes, when
        // they are kept: the same in every process (D145).
        let want = keep || self.plain;
        let named: HashMap<u32, &str> = self.libm.iter().map(|(n, id)| (id.as_u32(), n.as_str())).collect();
        let params = &self.ctx.func.params;
        let code = self.ctx.compiled_code().map(|c| {
            let relocs: Vec<(u32, u32, String)> = if want {
                c.buffer
                    .relocs()
                    .iter()
                    .map(|r| {
                        let size = match r.kind {
                            cranelift_codegen::binemit::Reloc::Abs8 => 8,
                            _ => 4,
                        };
                        let target = match &r.target {
                            cranelift_codegen::FinalizedRelocTarget::ExternalName(cranelift_codegen::ir::ExternalName::User(u)) => params.user_named_funcs().get(*u).and_then(|n| named.get(&n.index).map(|s| s.to_string())).unwrap_or_else(|| r.target.display(Some(params))),
                            t => t.display(Some(params)),
                        };
                        (r.offset, r.offset + size, target)
                    })
                    .collect()
            } else {
                Vec::new()
            };
            (c.code_buffer().len(), c.buffer.get_srclocs_sorted().iter().map(|l| (l.start, l.end, l.loc.bits())).collect::<Vec<_>>(), c.vcode.clone().unwrap_or_default(), if want { c.code_buffer().to_vec() } else { Vec::new() }, relocs)
        });
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions().map_err(|e| RtError { code: "E10001", message: e.to_string(), span: None })?;
        let ptr = self.module.get_finalized_function(fid);
        self.last_size = code.as_ref().map_or(0, |c| c.0);
        if let Some((len, locs, disasm, bytes, relocs)) = code {
            if self.plain {
                self.plain_kept.push(bytes.clone());
            }
            if keep {
                let (ranges, given) = cover(len as u32, &locs);
                if self.kept.len() <= idx {
                    self.kept.resize(idx + 1, None);
                }
                self.kept[idx] = Some(MachineCode { bytes, ranges, given, disasm, relocs });
            }
        }
        self.compiled_count += 1;
        // SAFETY: the function was built with the signature `fn(*const *mut u8, i64, i64)`.
        Ok(unsafe { std::mem::transmute::<*const u8, KernelFn>(ptr) })
    }
}

impl Default for NativeExec {
    fn default() -> Self {
        NativeExec::new().expect("cranelift for the host")
    }
}

/// `(start, end, line)` ranges, and `(start, end)` ones.
type Covered = (Vec<(u32, u32, u32)>, Vec<(u32, u32)>);

/// Every byte of `len` in one range with a line: Cranelift's ranges as it
/// gives them, and each byte it attributed to no line given to the line of
/// the range before it, or to the header's (1) before the first; adjacent
/// ranges of one line merged. And the ranges so given.
fn cover(len: u32, locs: &[(u32, u32, u32)]) -> Covered {
    let mut out: Vec<(u32, u32, u32)> = Vec::new();
    let mut gaps = Vec::new();
    let mut at = 0;
    let push = |out: &mut Vec<(u32, u32, u32)>, a: u32, b: u32, line: u32| {
        if a >= b {
            return;
        }
        match out.last_mut() {
            Some(last) if last.1 == a && last.2 == line => last.1 = b,
            _ => out.push((a, b, line)),
        }
    };
    for &(s, e, line) in locs.iter().filter(|l| l.2 != u32::MAX) {
        if s > at {
            let prev = out.last().map_or(1, |l| l.2);
            gaps.push((at, s));
            push(&mut out, at, s, prev);
        }
        push(&mut out, s.max(at), e, line);
        at = at.max(e);
    }
    if len > at {
        let prev = out.last().map_or(1, |l| l.2);
        gaps.push((at, len));
        push(&mut out, at, len, prev);
    }
    (out, gaps)
}

fn collect_libm(stmts: &[Stmt], out: &mut Vec<&'static str>) {
    fn expr(e: &Expr, out: &mut Vec<&'static str>) {
        match e {
            Expr::Un(op, a) => {
                let n = match op {
                    UnOp::Exp => Some("exp"),
                    UnOp::Log => Some("log"),
                    UnOp::Tanh => Some("tanh"),
                    UnOp::Sin => Some("sin"),
                    UnOp::Cos => Some("cos"),
                    UnOp::Erf => Some("erf"),
                    _ => None,
                };
                if let Some(n) = n
                    && !out.contains(&n)
                {
                    out.push(n);
                }
                expr(a, out);
            }
            Expr::Load(_, ix) | Expr::LoadI(_, ix) => ix.iter().for_each(|x| expr(x, out)),
            Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
                expr(a, out);
                expr(b, out);
            }
            Expr::Select(c, a, b) => {
                expr(c, out);
                expr(a, out);
                expr(b, out);
            }
            Expr::ToF(a) | Expr::ToI(a) => expr(a, out),
            _ => {}
        }
    }
    for s in stmts {
        match s {
            Stmt::For { body, .. } => collect_libm(body, out),
            Stmt::Set { value, .. } | Stmt::SetI { value, .. } | Stmt::SetL { value, .. } | Stmt::VSet { value, .. } => expr(value, out),
            Stmt::VStore { value, .. } => expr(value, out),
            Stmt::Store { index, value, .. } | Stmt::StoreI { index, value, .. } => {
                index.iter().for_each(|x| expr(x, out));
                expr(value, out);
            }
            Stmt::If { cond, then } => {
                expr(cond, out);
                collect_libm(then, out);
            }
        }
    }
}

fn fconst(b: &mut FunctionBuilder, ty: Type, x: f64) -> CV {
    if ty == types::F32 { b.ins().f32const(x as f32) } else { b.ins().f64const(x) }
}

struct Gen<'a> {
    b: FunctionBuilder<'a>,
    fty: Type,
    bufs: Vec<CV>,
    strides: Vec<Vec<usize>>,
    esize: Vec<i64>,
    /// The kernel's vector type, when it has lanes.
    vty: Option<Type>,
    /// One Cranelift variable per vector variable of the kernel.
    vecs: Vec<Variable>,
    /// Real buffers hold `bf16`: loads widen and stores round.
    narrow: bool,
    /// For an integer input, the pointer to its scales and their strides.
    scales: Vec<Option<(CV, Vec<usize>)>>,
    /// Which integer inputs are `int4`, two to a byte (0.6.0 step 5c).
    nibbles: Vec<bool>,
    /// For an input of reals narrower than the kernel's type, that type
    /// (plan 0.12.0 step 2): its loads widen it.
    wide: Vec<Option<Dtype>>,
    loops: Vec<Variable>,
    reals: Vec<Variable>,
    ints: Vec<Variable>,
    /// The output buffer's slot, after the inputs.
    out: usize,
    /// The depth of the loop each loop variable belongs to, or that assigns
    /// it, and of the statement that assigns each integer variable; and the
    /// depth being generated (plan 0.17.0 step 2, D175).
    depth_of: Vec<usize>,
    idepth_of: Vec<usize>,
    depth: usize,
    /// The output element a loop keeps in a variable (plan 0.17.0 step 2,
    /// D175): its index and the variable, while that loop is generated.
    promo: Option<(Vec<Expr>, Variable)>,
    libm: HashMap<&'static str, cranelift_codegen::ir::FuncRef>,
    /// Whether instructions carry the line of the kernel's text they
    /// compile (plan 0.13.0 step 5), and the line reached, in the order the
    /// text prints them.
    keep: bool,
    line: u32,
    /// The loop variables whose every value is affine in the loops' own
    /// (a `For`'s, or a `SetL`'s of such terms), and how many conditions
    /// enclose the code being generated: what decides whether a load may
    /// move ([`Gen::input_flags`]).
    affine_loops: Vec<bool>,
    guarded: usize,
}

impl Gen<'_> {
    /// The next line of the kernel's text: what the instructions emitted
    /// from now on compile.
    fn next_line(&mut self) {
        self.line += 1;
        if self.keep {
            self.b.set_srcloc(SourceLoc::new(self.line));
        }
    }

    /// The address of element `index` of buffer `buf`: the buffer's pointer
    /// plus each index's byte stride times it. The terms are added from the
    /// outermost loop's to the innermost's, the constant ones folded into
    /// the first, so that every partial sum the inner loop does not move is
    /// computed outside it (Cranelift's loop-invariant motion), and the inner
    /// loop's term comes last, where the target can fold it into the access
    /// (plan 0.17.0 step 2, D175). Integer arithmetic: no bit of a value
    /// depends on the order.
    fn addr(&mut self, buf: usize, index: &[Expr]) -> CV {
        let esize = self.esize[buf];
        let mut konst = 0i64;
        let mut terms: Vec<(usize, &Expr, i64)> = Vec::new();
        for (e, s) in index.iter().zip(self.strides[buf].iter()) {
            if *s == 0 {
                continue;
            }
            match constant(e) {
                Some(c) => konst += c * *s as i64 * esize,
                None => terms.push((self.depth_at(e), e, *s as i64 * esize)),
            }
        }
        terms.sort_by_key(|(d, _, _)| *d);
        let mut a = self.bufs[buf];
        if konst != 0 {
            a = self.b.ins().iadd_imm(a, konst);
        }
        for (_, e, s) in terms {
            let i = self.gen_i(e);
            let t = if s == 1 { i } else { self.b.ins().imul_imm(i, s) };
            a = self.b.ins().iadd(a, t);
        }
        a
    }

    /// The depth of the innermost loop expression `e` moves with: 0 for a
    /// constant, the loop's depth for a loop variable, the assigning
    /// statement's for an integer variable, the deepest for a load.
    fn depth_at(&self, e: &Expr) -> usize {
        match e {
            Expr::F(_) | Expr::I(_) => 0,
            Expr::Iv(v) => self.depth_of[*v],
            Expr::Si(v) => self.idepth_of[*v],
            Expr::Bin(_, x, y) | Expr::Cmp(_, x, y) | Expr::VBin(_, x, y) => self.depth_at(x).max(self.depth_at(y)),
            Expr::Un(_, x) | Expr::ToF(x) | Expr::ToI(x) | Expr::VSplat(x) => self.depth_at(x),
            Expr::Select(c, x, y) => self.depth_at(c).max(self.depth_at(x)).max(self.depth_at(y)),
            Expr::Load(..) | Expr::LoadI(..) | Expr::VLoad(..) | Expr::Sv(_) | Expr::Vv(_) => usize::MAX,
        }
    }

    /// Element `ix` of an integer input as the realization denotes it,
    /// `f32(q)·s` — one `f32` multiplication, as `presk_rt::QBuf::value`
    /// computes it, so the JIT and the kernel interpreter load the same bits —
    /// widened exactly when the kernel computes in `f64`. `lane` moves along
    /// the last axis, for the lanes of a vector load.
    /// Whether buffer `buf` holds integer weights. The output never does, and
    /// a kernel may read its own output back.
    fn is_quant(&self, buf: usize) -> bool {
        matches!(self.scales.get(buf), Some(Some(_)))
    }

    /// The type of buffer `buf` when it holds reals narrower than the
    /// kernel's, which its loads widen. The output never does.
    fn wide(&self, buf: usize) -> Option<Dtype> {
        self.wide.get(buf).copied().flatten()
    }

    /// Element `ix` of an input of narrower reals, moved `lane` elements
    /// along the last axis, widened exactly to the kernel's type: a `bf16`
    /// is the top sixteen bits of its `f32`, and an `f32` is an `f64`.
    fn load_wide(&mut self, buf: usize, index: &[Expr], lane: i64) -> CV {
        let d = self.wide(buf).expect("a narrower input");
        let a = self.addr(buf, index);
        let x = match d {
            Dtype::Bf16 => {
                let h = self.b.ins().load(types::I16, MemFlags::trusted(), a, (2 * lane) as i32);
                let w = self.b.ins().uextend(types::I32, h);
                let up = self.b.ins().ishl_imm(w, 16);
                self.b.ins().bitcast(types::F32, MemFlags::new(), up)
            }
            _ => self.b.ins().load(types::F32, MemFlags::trusted(), a, (4 * lane) as i32),
        };
        if self.fty == types::F64 { self.b.ins().fpromote(types::F64, x) } else { x }
    }

    fn load_quant(&mut self, buf: usize, index: &[Expr], lane: i64) -> CV {
        let (sp, sstrides) = self.scales[buf].clone().expect("an integer input has scales");
        let q = if self.nibbles.get(buf).copied().unwrap_or(false) {
            // `int4`: element `e` is the low nibble of byte `e/2` when `e` is
            // even and the high one when it is odd, two's complement — the
            // packing of `presk_rt::quant`. Shift it to the top of an `i32`
            // and back arithmetically, which sign-extends four bits.
            let mut e = self.b.ins().iconst(types::I64, lane);
            for (x, s) in index.iter().zip(self.strides[buf].clone()) {
                let i = self.gen_i(x);
                let t = self.b.ins().imul_imm(i, s as i64);
                e = self.b.ins().iadd(e, t);
            }
            let byte = self.b.ins().ushr_imm(e, 1);
            let a = self.b.ins().iadd(self.bufs[buf], byte);
            let b = self.b.ins().load(types::I8, MemFlags::trusted(), a, 0);
            let b = self.b.ins().uextend(types::I32, b);
            let odd = self.b.ins().band_imm(e, 1);
            let odd = self.b.ins().ireduce(types::I32, odd);
            let shift = self.b.ins().imul_imm(odd, 4);
            let top = self.b.ins().iconst(types::I32, 28);
            let up = self.b.ins().isub(top, shift);
            let hi = self.b.ins().ishl(b, up);
            let q = self.b.ins().sshr_imm(hi, 28);
            self.b.ins().fcvt_from_sint(types::F32, q)
        } else {
            let a = self.addr(buf, index);
            let q = self.b.ins().load(types::I8, MemFlags::trusted(), a, lane as i32);
            let q = self.b.ins().sextend(types::I32, q);
            self.b.ins().fcvt_from_sint(types::F32, q)
        };
        let mut off = self.b.ins().iconst(types::I64, 0);
        for (e, s) in index.iter().zip(sstrides.clone()) {
            if s == 0 {
                continue;
            }
            let i = self.gen_i(e);
            let t = self.b.ins().imul_imm(i, s as i64);
            off = self.b.ins().iadd(off, t);
        }
        let last = sstrides.last().copied().unwrap_or(0) as i64;
        let bytes = self.b.ins().imul_imm(off, 4);
        let sa = self.b.ins().iadd(sp, bytes);
        let s = self.b.ins().load(types::F32, MemFlags::trusted(), sa, (4 * last * lane) as i32);
        let v = self.b.ins().fmul(q, s);
        if self.fty == types::F64 { self.b.ins().fpromote(types::F64, v) } else { v }
    }

    /// `f32` to the sixteen bits of the nearest `bf16`, ties to even — the
    /// same function as `presk_rt::f32_to_bf16`, so the kernel interpreter and
    /// the JIT store the same bits and the conformance suite compares like
    /// with like.
    fn round_bf16(&mut self, x: CV) -> CV {
        let bits = self.b.ins().bitcast(types::I32, MemFlags::new(), x);
        let lsb = self.b.ins().ushr_imm(bits, 16);
        let lsb = self.b.ins().band_imm(lsb, 1);
        let bias = self.b.ins().iadd_imm(lsb, 0x7fff);
        let rounded = self.b.ins().iadd(bits, bias);
        let shifted = self.b.ins().ushr_imm(rounded, 16);
        // NaN must stay NaN: the bias above could carry it to an infinity.
        let quiet = self.b.ins().ushr_imm(bits, 16);
        let quiet = self.b.ins().bor_imm(quiet, 0x0040);
        let is_nan = self.b.ins().fcmp(FloatCC::NotEqual, x, x);
        let pick = self.b.ins().select(is_nan, quiet, shifted);
        self.b.ins().ireduce(types::I16, pick)
    }

    /// A vector expression. Each lane is an independent output element, so
    /// nothing here may reorder an addition; the instructions below are
    /// lane-wise by construction (D42).
    fn gen_v(&mut self, e: &Expr) -> CV {
        let vty = self.vty.expect("a vector expression in a scalar kernel");
        match e {
            Expr::Vv(v) => self.b.use_var(self.vecs[*v]),
            // Integer lanes are converted one at a time: the same arithmetic
            // as the scalar load, lane by lane, so the vector kernel stays
            // bit-identical to the scalar one (statement H).
            // Four integer lanes in one go. `int8`: four bytes, sign-extended twice to
            // `i32x4`, converted, and multiplied by four scales — the same
            // `f32(q)·s` in each lane as the scalar load computes.
            Expr::VLoad(buf, ix) if self.is_quant(*buf) && vty == types::F32X4 => {
                let (sp, sstrides) = self.scales[*buf].clone().expect("an integer input has scales");
                let v = if self.nibbles[*buf] {
                    // Four `int4` lanes: the 32-bit word at byte `e/2`,
                    // shifted so lane 0's nibble is at the bottom; then each
                    // lane's nibble moved to the top of its `i32` by a
                    // per-lane multiplication and brought back down
                    // arithmetically, which sign-extends it. The word may
                    // reach two bytes past the last element, which is why
                    // `int4` buffers carry four bytes of padding.
                    let mut e = self.b.ins().iconst(types::I64, 0);
                    for (x, s) in ix.iter().zip(self.strides[*buf].clone()) {
                        let i = self.gen_i(x);
                        let t = self.b.ins().imul_imm(i, s as i64);
                        e = self.b.ins().iadd(e, t);
                    }
                    let byte = self.b.ins().ushr_imm(e, 1);
                    let a = self.b.ins().iadd(self.bufs[*buf], byte);
                    let w = self.b.ins().load(types::I32, MemFlags::trusted(), a, 0);
                    let odd = self.b.ins().band_imm(e, 1);
                    let odd = self.b.ins().ireduce(types::I32, odd);
                    let shift = self.b.ins().imul_imm(odd, 4);
                    let w = self.b.ins().ushr(w, shift);
                    let v = self.b.ins().splat(types::I32X4, w);
                    let mut bytes = Vec::with_capacity(16);
                    for j in 0..4u32 {
                        bytes.extend_from_slice(&(1u32 << (28 - 4 * j)).to_le_bytes());
                    }
                    let c = self.b.func.dfg.constants.insert(cranelift_codegen::ir::ConstantData::from(bytes));
                    let m = self.b.ins().vconst(types::I32X4, c);
                    let v = self.b.ins().imul(v, m);
                    self.b.ins().sshr_imm(v, 28)
                } else {
                    let a = self.addr(*buf, ix);
                    let w = self.b.ins().load(types::I32, MemFlags::trusted(), a, 0);
                    let v = self.b.ins().scalar_to_vector(types::I32X4, w);
                    let v = self.b.ins().bitcast(types::I8X16, MemFlags::new().with_endianness(cranelift_codegen::ir::Endianness::Little), v);
                    let v = self.b.ins().swiden_low(v);
                    self.b.ins().swiden_low(v)
                };
                let q = self.b.ins().fcvt_from_sint(types::F32X4, v);
                let mut off = self.b.ins().iconst(types::I64, 0);
                for (e, st) in ix.iter().zip(sstrides.clone()) {
                    if st == 0 {
                        continue;
                    }
                    let i = self.gen_i(e);
                    let t = self.b.ins().imul_imm(i, st as i64);
                    off = self.b.ins().iadd(off, t);
                }
                let bytes = self.b.ins().imul_imm(off, 4);
                let sa = self.b.ins().iadd(sp, bytes);
                // Contiguous scales when the lanes run along a channel axis,
                // one scale for all four when they run along `over`.
                let s = if sstrides.last().copied().unwrap_or(0) == 1 {
                    self.b.ins().load(types::F32X4, MemFlags::new(), sa, 0)
                } else {
                    let s = self.b.ins().load(types::F32, MemFlags::trusted(), sa, 0);
                    self.b.ins().splat(types::F32X4, s)
                };
                self.b.ins().fmul(q, s)
            }
            // Four `bf16` lanes into `f32`: eight bytes, each sixteen bits
            // moved to the top of its lane; two `f32` lanes into `f64`: eight
            // bytes, promoted. The same widening as the scalar load.
            Expr::VLoad(buf, ix) if self.wide(*buf) == Some(Dtype::Bf16) && vty == types::F32X4 => {
                let a = self.addr(*buf, ix);
                let w = self.b.ins().load(types::I64, MemFlags::trusted(), a, 0);
                let v = self.b.ins().scalar_to_vector(types::I64X2, w);
                let v = self.b.ins().bitcast(types::I16X8, MemFlags::new().with_endianness(cranelift_codegen::ir::Endianness::Little), v);
                let v = self.b.ins().uwiden_low(v);
                let v = self.b.ins().ishl_imm(v, 16);
                self.b.ins().bitcast(types::F32X4, MemFlags::new().with_endianness(cranelift_codegen::ir::Endianness::Little), v)
            }
            Expr::VLoad(buf, ix) if self.wide(*buf) == Some(Dtype::F32) && vty == types::F64X2 => {
                let a = self.addr(*buf, ix);
                let w = self.b.ins().load(types::I64, MemFlags::trusted(), a, 0);
                let v = self.b.ins().scalar_to_vector(types::I64X2, w);
                let v = self.b.ins().bitcast(types::F32X4, MemFlags::new().with_endianness(cranelift_codegen::ir::Endianness::Little), v);
                self.b.ins().fvpromote_low(v)
            }
            Expr::VLoad(buf, ix) if self.wide(*buf).is_some() => {
                let lanes = vty.lane_count() as i64;
                let first = self.load_wide(*buf, ix, 0);
                let mut v = self.b.ins().splat(vty, first);
                for l in 1..lanes {
                    let x = self.load_wide(*buf, ix, l);
                    v = self.b.ins().insertlane(v, x, l as u8);
                }
                v
            }
            Expr::VLoad(buf, ix) if self.is_quant(*buf) => {
                let lanes = vty.lane_count() as i64;
                let first = self.load_quant(*buf, ix, 0);
                let mut v = self.b.ins().splat(vty, first);
                for l in 1..lanes {
                    let x = self.load_quant(*buf, ix, l);
                    v = self.b.ins().insertlane(v, x, l as u8);
                }
                v
            }
            Expr::VLoad(buf, ix) => {
                let a = self.addr(*buf, ix);
                // Unaligned: the column a tile starts at need not be aligned.
                let flags = self.input_flags(*buf, ix, MemFlags::new());
                self.b.ins().load(vty, flags, a, 0)
            }
            Expr::VSplat(x) => {
                let s = self.gen_f(x);
                self.b.ins().splat(vty, s)
            }
            Expr::VBin(op, a, c) => {
                let (x, y) = (self.gen_v(a), self.gen_v(c));
                match op {
                    BinOp::Add | BinOp::IAdd => self.b.ins().fadd(x, y),
                    BinOp::Sub => self.b.ins().fsub(x, y),
                    BinOp::Mul | BinOp::IMul => self.b.ins().fmul(x, y),
                    BinOp::Div => self.b.ins().fdiv(x, y),
                }
            }
            other => {
                let s = self.gen_f(other);
                self.b.ins().splat(vty, s)
            }
        }
    }

    fn gen_f(&mut self, e: &Expr) -> CV {
        match e {
            Expr::F(x) => fconst(&mut self.b, self.fty, *x),
            Expr::Load(buf, ix) if self.promoted(*buf, ix).is_some() => {
                let v = self.promoted(*buf, ix).expect("checked");
                self.b.use_var(v)
            }
            Expr::Load(buf, ix) if self.is_quant(*buf) => self.load_quant(*buf, ix, 0),
            Expr::Load(buf, ix) if self.wide(*buf).is_some() => self.load_wide(*buf, ix, 0),
            Expr::Load(buf, ix) => {
                let a = self.addr(*buf, ix);
                if self.narrow {
                    // A `bf16` is the top sixteen bits of the `f32` it
                    // denotes, so widening is exact and is a shift.
                    let h = self.b.ins().load(types::I16, MemFlags::trusted(), a, 0);
                    let w = self.b.ins().uextend(types::I32, h);
                    let up = self.b.ins().ishl_imm(w, 16);
                    self.b.ins().bitcast(types::F32, MemFlags::new(), up)
                } else {
                    let flags = self.input_flags(*buf, ix, MemFlags::trusted());
                    self.b.ins().load(self.fty, flags, a, 0)
                }
            }
            Expr::Sv(v) => self.b.use_var(self.reals[*v]),
            Expr::Bin(op, x, y) if !matches!(op, BinOp::IAdd | BinOp::IMul) => {
                let (a, c) = (self.gen_f(x), self.gen_f(y));
                match op {
                    BinOp::Add => self.b.ins().fadd(a, c),
                    BinOp::Sub => self.b.ins().fsub(a, c),
                    BinOp::Mul => self.b.ins().fmul(a, c),
                    _ => self.b.ins().fdiv(a, c),
                }
            }
            Expr::Un(op, x) => {
                let a = self.gen_f(x);
                match op {
                    UnOp::Neg => self.b.ins().fneg(a),
                    UnOp::Sqrt => self.b.ins().sqrt(a),
                    UnOp::Abs => self.b.ins().fabs(a),
                    UnOp::Exp | UnOp::Log | UnOp::Tanh | UnOp::Sin | UnOp::Cos | UnOp::Erf => {
                        let name = match op {
                            UnOp::Exp => "exp",
                            UnOp::Log => "log",
                            UnOp::Tanh => "tanh",
                            UnOp::Sin => "sin",
                            UnOp::Cos => "cos",
                            _ => "erf",
                        };
                        let f = self.libm[name];
                        let call = self.b.ins().call(f, &[a]);
                        self.b.inst_results(call)[0]
                    }
                }
            }
            Expr::Select(c, x, y) => {
                let cond = self.gen_bool(c);
                let (a, d) = (self.gen_f(x), self.gen_f(y));
                self.b.ins().select(cond, a, d)
            }
            Expr::ToF(x) => {
                let i = self.gen_i(x);
                self.b.ins().fcvt_from_sint(self.fty, i)
            }
            other => {
                let i = self.gen_i(other);
                self.b.ins().fcvt_from_sint(self.fty, i)
            }
        }
    }

    fn gen_i(&mut self, e: &Expr) -> CV {
        match e {
            Expr::I(i) => self.b.ins().iconst(types::I64, *i),
            Expr::Iv(v) => self.b.use_var(self.loops[*v]),
            Expr::Si(v) => self.b.use_var(self.ints[*v]),
            Expr::LoadI(buf, ix) => {
                let a = self.addr(*buf, ix);
                self.b.ins().load(types::I64, MemFlags::trusted(), a, 0)
            }
            Expr::Bin(BinOp::IAdd, x, y) => {
                let (a, c) = (self.gen_i(x), self.gen_i(y));
                self.b.ins().iadd(a, c)
            }
            Expr::Bin(BinOp::IMul, x, y) => {
                let (a, c) = (self.gen_i(x), self.gen_i(y));
                self.b.ins().imul(a, c)
            }
            Expr::ToI(x) => {
                let a = self.gen_f(x);
                self.b.ins().fcvt_to_sint_sat(types::I64, a)
            }
            Expr::Select(c, x, y) => {
                let cond = self.gen_bool(c);
                let (a, d) = (self.gen_i(x), self.gen_i(y));
                self.b.ins().select(cond, a, d)
            }
            other => {
                let a = self.gen_f(other);
                self.b.ins().fcvt_to_sint_sat(types::I64, a)
            }
        }
    }

    fn gen_bool(&mut self, e: &Expr) -> CV {
        match e {
            Expr::Cmp(op, x, y) => {
                let (a, c) = (self.gen_f(x), self.gen_f(y));
                let cc = match op {
                    CmpOp::Lt => FloatCC::LessThan,
                    CmpOp::Le => FloatCC::LessThanOrEqual,
                    CmpOp::Gt => FloatCC::GreaterThan,
                    CmpOp::Ge => FloatCC::GreaterThanOrEqual,
                    CmpOp::Eq => FloatCC::Equal,
                };
                self.b.ins().fcmp(cc, a, c)
            }
            other => {
                let i = self.gen_i(other);
                let z = self.b.ins().iconst(types::I64, 0);
                self.b.ins().icmp(IntCC::NotEqual, i, z)
            }
        }
    }

    /// `for var in lo..hi { body }` with the bounds as values, so the caller
    /// can hand a worker its slice of the iteration space.
    fn for_range(&mut self, var: usize, lo: CV, hi: CV, body: &[Stmt]) {
        self.for_range_with(var, lo, hi, &mut |g: &mut Self| g.stmts(body), body);
    }

    /// [`Gen::for_range`] with the loop's body generated by `emit`, `body`
    /// the statements it generates (to decide what the loop keeps in a
    /// variable).
    fn for_range_with(&mut self, var: usize, lo: CV, hi: CV, emit: &mut dyn FnMut(&mut Self), body: &[Stmt]) {
        // An output element the loop does not move, which every access to the
        // output inside it reads or writes, lives in a variable for the loop:
        // loaded before it, stored after it, the same operations in the same
        // order between (plan 0.17.0 step 2, D175). Cranelift moves no memory
        // access out of a loop, so without this a sum into the output loads
        // and stores it at every step. A `bf16` output rounds at every store,
        // which the variable would skip, so it keeps its stores.
        let promote = if self.promo.is_none() && !self.narrow { promotable(var, body, self.out) } else { None };
        let Some(ix) = promote else {
            return self.loop_with(var, lo, hi, |g| emit(g));
        };
        let a = self.addr(self.out, &ix);
        let x = self.b.ins().load(self.fty, MemFlags::trusted(), a, 0);
        let v = self.b.declare_var(self.fty);
        self.b.def_var(v, x);
        self.promo = Some((ix.clone(), v));
        self.loop_with(var, lo, hi, |g| emit(g));
        self.promo = None;
        let y = self.b.use_var(v);
        let a = self.addr(self.out, &ix);
        self.b.ins().store(MemFlags::trusted(), y, a, 0);
    }

    /// `for var in 0..bound { body }`, innermost, unrolled by `f`
    /// ([`unroll_factor`]): the body `f` times per iteration, `var` set to
    /// each value in turn, then the iterations left over by the plain loop
    /// (plan 0.17.0 step 2, D175). The same operations in the same order, so
    /// the same bits; Cranelift does not unroll. Each copy's instructions
    /// carry the lines of the body's text, as the loop's do.
    fn unrolled(&mut self, var: usize, bound: usize, body: &[Stmt], f: usize) {
        let main = bound / f * f;
        let line = self.line;
        let zero = self.b.ins().iconst(types::I64, 0);
        let n = self.b.ins().iconst(types::I64, main as i64);
        let v = self.loops[var];
        self.for_range_with(var, zero, n, &mut |g: &mut Self| {
            let base = g.b.use_var(v);
            for t in 0..f {
                if t > 0 {
                    let at = g.b.ins().iadd_imm(base, t as i64);
                    g.b.def_var(v, at);
                }
                g.line = line;
                g.stmts(body);
            }
        }, body);
        if main < bound {
            self.line = line;
            let lo = self.b.ins().iconst(types::I64, main as i64);
            let hi = self.b.ins().iconst(types::I64, bound as i64);
            self.for_range(var, lo, hi, body);
        }
    }

    /// The flags of a load of reals from buffer `buf` at `index`, `plain`
    /// unless the load may move (plan 0.17.0 step 2, D175): an input,
    /// which the kernel never writes (the host lends it and allocates the
    /// output apart), read at an index affine in loop variables whose values
    /// are themselves affine ([`affine_loops`]), outside any condition. Such
    /// an address is in the buffer wherever its loop variables have a
    /// value, so the load can run
    /// anywhere they are defined: Cranelift then treats it as pure, places
    /// it at its first use (where a broadcast takes it as `ld1r`), merges
    /// duplicates and hoists it out of the loops it does not depend on, as
    /// a C compiler does with `restrict`. The values loaded are the same.
    fn input_flags(&self, buf: usize, index: &[Expr], plain: MemFlags) -> MemFlags {
        if buf != self.out && self.guarded == 0 && index.iter().all(|e| affine(e, &self.affine_loops)) {
            plain.with_notrap().with_readonly().with_can_move()
        } else {
            plain
        }
    }

    /// The variable that holds output element `index` of buffer `buf`, when
    /// the loop being generated keeps it in one.
    fn promoted(&self, buf: usize, index: &[Expr]) -> Option<Variable> {
        match &self.promo {
            Some((ix, v)) if buf == self.out && ix.as_slice() == index => Some(*v),
            _ => None,
        }
    }

    /// The nest down to the loop at `depth`, which runs over `lo..hi` only;
    /// every loop enclosing it runs in full (`Kernel::parallel_at`).
    fn range_at(&mut self, stmts: &[Stmt], depth: usize, lo: CV, hi: CV) {
        let [Stmt::For { var, bound, body }] = stmts else { return self.stmts(stmts) };
        self.next_line();
        if depth == 0 {
            return self.for_range(*var, lo, hi, body);
        }
        let zero = self.b.ins().iconst(types::I64, 0);
        let n = self.b.ins().iconst(types::I64, *bound as i64);
        self.loop_with(*var, zero, n, |g| g.range_at(body, depth - 1, lo, hi));
    }

    /// `for var in lo..hi { body }`.
    fn loop_with(&mut self, var: usize, lo: CV, hi: CV, body: impl FnOnce(&mut Self)) {
        self.depth += 1;
        self.depth_of[var] = self.depth;
        let v = self.loops[var];
        self.b.def_var(v, lo);
        let header = self.b.create_block();
        let body_b = self.b.create_block();
        let exit = self.b.create_block();
        self.b.ins().jump(header, &[]);
        self.b.switch_to_block(header);
        let i = self.b.use_var(v);
        // The exit is the branch's first target, so that the dominator
        // tree's children put the body before the exit. Cranelift's
        // elaborator (0.131) visits them in that order and forgets a loop
        // once it has visited a block outside it: with the body second, no
        // value is ever hoisted out of the loop (D175).
        let c = self.b.ins().icmp(IntCC::SignedGreaterThanOrEqual, i, hi);
        self.b.ins().brif(c, exit, &[], body_b, &[]);
        self.b.switch_to_block(body_b);
        self.b.seal_block(body_b);
        body(self);
        // The step and the jump back are the closing brace's.
        self.next_line();
        let i = self.b.use_var(v);
        let next = self.b.ins().iadd_imm(i, 1);
        self.b.def_var(v, next);
        self.b.ins().jump(header, &[]);
        self.b.seal_block(header);
        self.b.switch_to_block(exit);
        self.b.seal_block(exit);
        self.depth -= 1;
    }

    fn stmts(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            self.next_line();
            match s {
                Stmt::For { var, bound, body } => match unroll_factor(*var, *bound, body) {
                    1 => {
                        let zero = self.b.ins().iconst(types::I64, 0);
                        let n = self.b.ins().iconst(types::I64, *bound as i64);
                        self.for_range(*var, zero, n, body);
                    }
                    f => self.unrolled(*var, *bound, body, f),
                },
                Stmt::Set { var, value } => {
                    let x = self.gen_f(value);
                    self.b.def_var(self.reals[*var], x);
                }
                Stmt::SetI { var, value } => {
                    let x = self.gen_i(value);
                    self.idepth_of[*var] = self.depth;
                    self.b.def_var(self.ints[*var], x);
                }
                Stmt::SetL { var, value } => {
                    let x = self.gen_i(value);
                    self.depth_of[*var] = self.depth;
                    self.b.def_var(self.loops[*var], x);
                }
                // Vector kernels are refused before generation begins, so
                // reaching here would mean the guard in `compile` was lost.
                Stmt::VSet { var, value } => {
                    let x = self.gen_v(value);
                    self.b.def_var(self.vecs[*var], x);
                }
                Stmt::VStore { buf, index, value } => {
                    // The lanes are contiguous output columns, so one store
                    // of the whole register writes them all in order.
                    let x = self.gen_v(value);
                    let a = self.addr(*buf, index);
                    self.b.ins().store(MemFlags::new(), x, a, 0);
                }
                Stmt::Store { buf, index, value } if self.promoted(*buf, index).is_some() => {
                    let x = self.gen_f(value);
                    let v = self.promoted(*buf, index).expect("checked");
                    self.b.def_var(v, x);
                }
                Stmt::Store { buf, index, value } => {
                    let x = self.gen_f(value);
                    let a = self.addr(*buf, index);
                    if self.narrow {
                        let h = self.round_bf16(x);
                        self.b.ins().store(MemFlags::trusted(), h, a, 0);
                    } else {
                        self.b.ins().store(MemFlags::trusted(), x, a, 0);
                    }
                }
                Stmt::StoreI { buf, index, value } => {
                    let x = self.gen_i(value);
                    let a = self.addr(*buf, index);
                    self.b.ins().store(MemFlags::trusted(), x, a, 0);
                }
                Stmt::If { cond, then } => {
                    let c = self.gen_bool(cond);
                    let then_b = self.b.create_block();
                    let merge = self.b.create_block();
                    self.b.ins().brif(c, then_b, &[], merge, &[]);
                    self.b.switch_to_block(then_b);
                    self.b.seal_block(then_b);
                    self.guarded += 1;
                    self.stmts(then);
                    self.guarded -= 1;
                    self.next_line();
                    self.b.ins().jump(merge, &[]);
                    self.b.switch_to_block(merge);
                    self.b.seal_block(merge);
                }
            }
        }
    }
}

/// Whether integer expression `e` is affine in the loop variables `ok`
/// marks: constants, those variables, sums and products of them.
fn affine(e: &Expr, ok: &[bool]) -> bool {
    match e {
        Expr::Iv(v) => ok[*v],
        Expr::I(_) => true,
        Expr::Bin(BinOp::IAdd | BinOp::IMul, x, y) => affine(x, ok) && affine(y, ok),
        _ => false,
    }
}

/// The loop variables whose every value is affine in the loops' own: a
/// `For`'s that no `SetL` assigns, and one each of whose `SetL`s assigns
/// it an affine expression of such variables (`i5 = i1·128 + i4`, how an
/// interleaved contraction addresses its terms). A variable a `SetL` gives
/// a value read from data is not.
fn affine_loops(body: &[Stmt], nloops: usize) -> Vec<bool> {
    fn collect<'a>(body: &'a [Stmt], sets: &mut Vec<Vec<&'a Expr>>) {
        for s in body {
            match s {
                Stmt::SetL { var, value } => sets[*var].push(value),
                Stmt::For { body, .. } | Stmt::If { then: body, .. } => collect(body, sets),
                _ => {}
            }
        }
    }
    let mut sets = vec![Vec::new(); nloops];
    collect(body, &mut sets);
    let mut ok: Vec<bool> = sets.iter().map(|v| v.is_empty()).collect();
    loop {
        let next: Vec<bool> = (0..nloops).map(|v| ok[v] || sets[v].iter().all(|e| affine(e, &ok))).collect();
        if next == ok {
            return ok;
        }
        ok = next;
    }
}

/// How many times an innermost loop's body is repeated per iteration, at
/// most.
const UNROLL: usize = 4;

/// The floating-point registers the register allocator has: 32 on AArch64,
/// 16 on x86-64 without AVX-512.
const FLOAT_REGS: usize = if cfg!(target_arch = "aarch64") { 32 } else { 16 };

/// How many times loop `var` of `bound` iterations over `body` is repeated
/// per iteration: [`UNROLL`], two, or one (not unrolled). It is unrolled
/// when it is innermost (no loop inside), its variable assigned by the loop
/// alone, at least two unrolled iterations, and a body of at most 64
/// statements, so that the code grows by a bounded amount. The factor is
/// then the largest whose registers fit: Cranelift keeps the loads in their
/// order and computes the arithmetic where its result is used, at the end
/// of the iteration, so every copy's loads are live at once beside the
/// values the body carries across iterations. A factor that needs more than
/// [`FLOAT_REGS`] spills them in the loop, which costs more than the loop's
/// own instructions save (D175: a register tile of sixteen accumulators). A
/// body that calls `libm` is not unrolled.
fn unroll_factor(var: usize, bound: usize, body: &[Stmt]) -> usize {
    fn count(var: usize, body: &[Stmt]) -> Option<usize> {
        let mut n = 0;
        for s in body {
            n += match s {
                Stmt::For { .. } => return None,
                Stmt::SetL { var: v, .. } if *v == var => return None,
                Stmt::If { then, .. } => 1 + count(var, then)?,
                _ => 1,
            };
        }
        Some(n)
    }
    // A call to `libm` clobbers the registers that are not saved across
    // it, so the copies' values would live on the stack, and the call
    // costs more than the loop's instructions anyway.
    let mut calls = Vec::new();
    collect_libm(body, &mut calls);
    if !calls.is_empty() || !count(var, body).is_some_and(|n| n <= 64) {
        return 1;
    }
    let mut regs = Regs::default();
    regs.stmts(body);
    let carried = regs.live_in.len();
    let loads = regs.loads.len();
    // A few registers for the products and splats in flight.
    let fits = |f: usize| carried + f * loads + 4 <= FLOAT_REGS;
    [UNROLL, 2].into_iter().find(|&f| bound >= 2 * f && fits(f)).unwrap_or(1)
}

/// What [`unroll_factor`] counts over a loop's body: the real and vector
/// variables read before the body assigns them (the values it carries, or
/// reads from outside), and the distinct real loads.
#[derive(Default)]
struct Regs {
    assigned: std::collections::HashSet<(bool, usize)>,
    live_in: std::collections::HashSet<(bool, usize)>,
    loads: Vec<(bool, usize, Vec<Expr>)>,
}

impl Regs {
    fn stmts(&mut self, body: &[Stmt]) {
        for s in body {
            match s {
                Stmt::For { body, .. } => self.stmts(body),
                Stmt::Set { var, value } => {
                    self.expr(value);
                    self.assigned.insert((false, *var));
                }
                Stmt::VSet { var, value } => {
                    self.expr(value);
                    self.assigned.insert((true, *var));
                }
                Stmt::SetI { value, .. } | Stmt::SetL { value, .. } => self.expr(value),
                Stmt::Store { index, value, .. } | Stmt::VStore { index, value, .. } | Stmt::StoreI { index, value, .. } => {
                    index.iter().for_each(|e| self.expr(e));
                    self.expr(value);
                }
                Stmt::If { cond, then } => {
                    self.expr(cond);
                    self.stmts(then);
                }
            }
        }
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Sv(v) | Expr::Vv(v) => {
                let key = (matches!(e, Expr::Vv(_)), *v);
                if !self.assigned.contains(&key) {
                    self.live_in.insert(key);
                }
            }
            Expr::Load(b, ix) | Expr::VLoad(b, ix) => {
                ix.iter().for_each(|x| self.expr(x));
                let key = (matches!(e, Expr::VLoad(..)), *b, ix.clone());
                if !self.loads.contains(&key) {
                    self.loads.push(key);
                }
            }
            Expr::LoadI(_, ix) => ix.iter().for_each(|x| self.expr(x)),
            Expr::Bin(_, x, y) | Expr::Cmp(_, x, y) | Expr::VBin(_, x, y) => {
                self.expr(x);
                self.expr(y);
            }
            Expr::Un(_, x) | Expr::ToF(x) | Expr::ToI(x) | Expr::VSplat(x) => self.expr(x),
            Expr::Select(c, x, y) => {
                self.expr(c);
                self.expr(x);
                self.expr(y);
            }
            Expr::F(_) | Expr::I(_) | Expr::Iv(_) | Expr::Si(_) => {}
        }
    }
}

/// The value of an integer index expression made of constants only.
fn constant(e: &Expr) -> Option<i64> {
    match e {
        Expr::I(c) => Some(*c),
        Expr::Bin(BinOp::IAdd, x, y) => Some(constant(x)?.checked_add(constant(y)?)?),
        Expr::Bin(BinOp::IMul, x, y) => Some(constant(x)?.checked_mul(constant(y)?)?),
        _ => None,
    }
}

/// The index of the output element loop `var`'s body keeps fixed, when it
/// can live in a variable for the loop: every access to the output `out`
/// inside the body, at any depth, is a scalar real load or store at that
/// one index; at least one is a store; and the index reads no variable the
/// body assigns, the loop's own included. A kernel's inputs never share the
/// output's memory (the host lends them and allocates the output), so no
/// other access can reach the element.
fn promotable(var: usize, body: &[Stmt], out: usize) -> Option<Vec<Expr>> {
    let mut acc = Access { out, index: None, stores: 0, bad: false, loops: vec![var], ints: vec![] };
    acc.stmts(body);
    let ix = acc.index.take()?;
    if acc.bad || acc.stores == 0 || ix.iter().any(|e| acc.moves(e)) {
        return None;
    }
    Some(ix)
}

/// What [`promotable`] gathers over a loop's body.
struct Access {
    out: usize,
    index: Option<Vec<Expr>>,
    stores: usize,
    bad: bool,
    /// The loop variables the body assigns, and its integer variables.
    loops: Vec<usize>,
    ints: Vec<usize>,
}

impl Access {
    fn at(&mut self, ix: &[Expr]) {
        match &self.index {
            None => self.index = Some(ix.to_vec()),
            Some(i) if i.as_slice() == ix => {}
            Some(_) => self.bad = true,
        }
    }

    fn stmts(&mut self, body: &[Stmt]) {
        for s in body {
            match s {
                Stmt::For { var, body, .. } => {
                    self.loops.push(*var);
                    self.stmts(body);
                }
                Stmt::SetL { var, value } => {
                    self.loops.push(*var);
                    self.expr(value);
                }
                Stmt::SetI { var, value } => {
                    self.ints.push(*var);
                    self.expr(value);
                }
                Stmt::Set { value, .. } | Stmt::VSet { value, .. } => self.expr(value),
                Stmt::Store { buf, index, value } => {
                    index.iter().for_each(|e| self.expr(e));
                    self.expr(value);
                    if *buf == self.out {
                        self.stores += 1;
                        self.at(index);
                    }
                }
                Stmt::VStore { buf, index, value } | Stmt::StoreI { buf, index, value } => {
                    index.iter().for_each(|e| self.expr(e));
                    self.expr(value);
                    if *buf == self.out {
                        self.bad = true;
                    }
                }
                Stmt::If { cond, then } => {
                    self.expr(cond);
                    self.stmts(then);
                }
            }
        }
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Load(buf, ix) => {
                ix.iter().for_each(|e| self.expr(e));
                if *buf == self.out {
                    self.at(ix);
                }
            }
            Expr::LoadI(buf, ix) | Expr::VLoad(buf, ix) => {
                ix.iter().for_each(|e| self.expr(e));
                if *buf == self.out {
                    self.bad = true;
                }
            }
            Expr::Bin(_, x, y) | Expr::Cmp(_, x, y) | Expr::VBin(_, x, y) => {
                self.expr(x);
                self.expr(y);
            }
            Expr::Un(_, x) | Expr::ToF(x) | Expr::ToI(x) | Expr::VSplat(x) => self.expr(x),
            Expr::Select(c, x, y) => {
                self.expr(c);
                self.expr(x);
                self.expr(y);
            }
            Expr::F(_) | Expr::I(_) | Expr::Iv(_) | Expr::Sv(_) | Expr::Si(_) | Expr::Vv(_) => {}
        }
    }

    /// Whether index expression `e` reads a variable the body assigns.
    fn moves(&self, e: &Expr) -> bool {
        match e {
            Expr::Iv(v) => self.loops.contains(v),
            Expr::Si(v) => self.ints.contains(v),
            Expr::F(_) | Expr::I(_) => false,
            Expr::Bin(_, x, y) | Expr::Cmp(_, x, y) | Expr::VBin(_, x, y) => self.moves(x) || self.moves(y),
            Expr::Un(_, x) | Expr::ToF(x) | Expr::ToI(x) | Expr::VSplat(x) => self.moves(x),
            Expr::Select(c, x, y) => self.moves(c) || self.moves(x) || self.moves(y),
            // A load or a real variable in an index: not a fixed position.
            Expr::Load(..) | Expr::LoadI(..) | Expr::VLoad(..) | Expr::Sv(_) | Expr::Vv(_) => true,
        }
    }
}

fn buf_ptr(b: &Buf) -> Result<*mut u8> {
    Ok(match b {
        Buf::F64(v) => v.as_ptr() as *mut u8,
        Buf::F32(v) => v.as_ptr() as *mut u8,
        Buf::Bf16(v) => v.as_ptr() as *mut u8,
        Buf::I64(v) => v.as_ptr() as *mut u8,
        Buf::Quant(q) => q.data.as_ptr() as *mut u8,
        Buf::Exact(_) => return err("native backend: an exact buffer reached a kernel"),
    })
}

impl KernelExec for NativeExec {
    fn name(&self) -> &'static str {
        self.external.as_ref().map_or("native", |c| c.name())
    }

    fn workers(&self) -> Option<usize> {
        Some(self.threads)
    }

    fn set_profiling(&mut self, on: bool) {
        self.profiling = on;
    }

    fn overhead(&self) -> Overhead {
        self.last
    }

    fn compile_stats(&self) -> (u64, f64) {
        (self.compiles, self.compile_s)
    }

    /// The host's own work in pieces (a training dropout mask, plan 0.10.0
    /// step 6), on the pool, the first piece on the calling thread.
    fn parallel(&mut self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        if n <= 1 || self.threads <= 1 {
            (0..n).for_each(f);
            return;
        }
        self.pool().run(n, f);
    }

    fn run(&mut self, idx: usize, k: &Kernel, inputs: &[&Buf], out: &mut Buf) -> Result<()> {
        // When the host profiles, the clock is read around the compilation,
        // before the kernel's code starts, and on each worker; nothing else
        // differs (plan 0.10.0 step 5).
        let t_in = self.profiling.then(std::time::Instant::now);
        let mut compile = 0.0;
        if self.compiled.len() <= idx {
            self.compiled.resize(idx + 1, None);
        }
        let f = match self.compiled[idx] {
            Some(f) => f,
            None => {
                let t = std::time::Instant::now();
                let f = match &mut self.external {
                    Some(c) => c.compile(k)?,
                    None => self.compile(idx, k)?,
                };
                let s = t.elapsed().as_secs_f64();
                self.compile_s += s;
                self.compiles += 1;
                if self.profiling {
                    compile = s;
                }
                self.compiled[idx] = Some(f);
                f
            }
        };
        let mut ptrs: Vec<*mut u8> = inputs.iter().map(|b| buf_ptr(b)).collect::<Result<_>>()?;
        ptrs.push(match out {
            Buf::F64(v) => v.as_mut_ptr() as *mut u8,
            Buf::F32(v) => v.as_mut_ptr() as *mut u8,
            Buf::Bf16(v) => v.as_mut_ptr() as *mut u8,
            Buf::I64(v) => v.as_mut_ptr() as *mut u8,
            Buf::Exact(_) | Buf::Quant(_) => return err("native backend: an exact or integer output buffer"),
        });
        // The scales of the integer inputs, after the output (`compile`).
        for (b, spec) in inputs.iter().zip(&k.inputs) {
            if let (Buf::Quant(q), BufKind::Quant { .. }) = (b, spec.kind) {
                ptrs.push(q.scales.as_ptr() as *mut u8);
            }
        }
        // Threads over a parallel loop: by default the outermost, over every
        // worker, when the kernel is big enough to pay for spawning; or the
        // loop at the depth the plan states, over its workers, whatever the
        // work (0.7.0 step 3). Each output element is computed by the same
        // inner loops in the same order, so the result is bit-identical to
        // the sequential one.
        let ranged = ranged(k);
        let total = ranged.map_or(0, |(_, n)| n);
        let workers = match k.threads {
            Threads::Default => {
                let work = presk_kernel::cost::kernel_cost(k).work;
                match ranged {
                    Some((_, n)) if self.threads > 1 && n >= 2 && work >= PARALLEL_MIN_WORK => self.threads.min(n),
                    _ => 1,
                }
            }
            Threads::One => 1,
            Threads::At { workers, .. } => workers.min(total).max(1),
        };
        let t_go = self.profiling.then(std::time::Instant::now);
        let dispatch = match (t_in, t_go) {
            (Some(a), Some(b)) => (b.duration_since(a).as_secs_f64() - compile).max(0.0),
            _ => 0.0,
        };
        // SAFETY: the kernel's loop bounds are the buffers' dimensions (checked by the
        // lowering and exercised by the kernel interpreter); inputs are never written.
        if workers <= 1 {
            unsafe { f(ptrs.as_ptr(), 0, total as i64) };
            self.last = Overhead { compile, dispatch, threads: 0.0, workers: 1 };
            return Ok(());
        }
        let total = total as i64;
        let p = Ptrs(ptrs.as_ptr());
        self.threaded += 1;
        // Piece `t` of `workers`, on the pool's worker `t`, the first on
        // this thread (plan 0.10.0 step 6): the pieces the scoped threads of
        // 0.4.0 to 0.10.0 step 5 ran, each by the same code.
        let piece = move |t: usize| {
            let lo = total * t as i64 / workers as i64;
            let hi = total * (t as i64 + 1) / workers as i64;
            // SAFETY: the workers share read-only inputs and write disjoint
            // slices of the output (`Kernel::parallel_at`).
            if lo != hi {
                unsafe { f(p.get(), lo, hi) };
            }
        };
        if let Some(t_go) = t_go {
            // Each piece timed: what the launch takes beyond its longest is
            // its workers' wake, their join, and the stagger of their starts.
            let spans: Vec<AtomicU64> = (0..workers).map(|_| AtomicU64::new(0)).collect();
            self.pool().run(workers, &|t| {
                let a = std::time::Instant::now();
                piece(t);
                spans[t].store(a.elapsed().as_nanos() as u64, Ordering::Relaxed);
            });
            let elapsed = t_go.elapsed().as_secs_f64();
            let longest = spans.iter().map(|s| s.load(Ordering::Relaxed)).max().unwrap_or(0) as f64 * 1e-9;
            self.last = Overhead { compile, dispatch, threads: (elapsed - longest).max(0.0), workers };
            return Ok(());
        }
        self.pool().run(workers, &piece);
        Ok(())
    }
}

/// Kernel `k` with its lane-wise loops over lanes (plan 0.17.0 step 2,
/// D175): what this backend compiles for a one-lane kernel, and what the
/// LLVM backend prints for `clang`. `None` when it has none.
pub fn laned(k: &Kernel) -> Option<Kernel> {
    lanes::natural(k, ranged(k).map(|(d, _)| d))
}

/// The loop a kernel's split divides, `(depth, iterations)`: the one at the
/// depth its plan states, or the outermost when it is parallel (0.6.0's
/// analysis, the executor's default).
fn ranged(k: &Kernel) -> Option<(usize, usize)> {
    match k.threads {
        Threads::At { depth, .. } => k.parallel_at(depth).map(|n| (depth, n)),
        Threads::Default | Threads::One => k.parallel_outer().map(|n| (0, n)),
    }
}

/// A kernel backend whose kernels are JIT-compiled by Cranelift.
pub fn native_backend<'p>(prog: &'p Program, elab: &'p ElabProgram, realize: &Realization) -> Result<KernelBackend<'p>> {
    let mut b = KernelBackend::with_exec(prog, elab, realize, Box::new(NativeExec::new()?));
    // Lanes unstated: each kernel takes its natural width (below).
    b.default_layout.lanes = None;
    Ok(b)
}

/// The lane count a machine type gets by default.
///
/// **On, not off.** A lane is an independent output column, so a vector
/// kernel is bit-identical to its scalar form (D42, asserted by test on both
/// executors): there is no accuracy to trade, no realization to declare and
/// no gap to measure. A transformation that is free and exact has no reason
/// to be opt-in, and leaving it off is how 0.6.0 step 4 came to measure a
/// solved realization against a scalar baseline and under-report it.
///
/// `bf16` stays scalar: a narrow lane needs a conversion per element, which
/// is a different kernel and a different measurement.
pub fn natural_lanes(d: Dtype) -> usize {
    d.natural_lanes()
}

/// The same backend with an explicit number of workers, so a caller can pin it
/// (the tests compare one thread against many, which must agree exactly).
pub fn native_backend_with_threads<'p>(prog: &'p Program, elab: &'p ElabProgram, realize: &Realization, threads: usize) -> Result<KernelBackend<'p>> {
    let mut exec = NativeExec::new()?;
    exec.threads = threads.max(1);
    let mut b = KernelBackend::with_exec(prog, elab, realize, Box::new(exec));
    b.default_layout.lanes = None;
    Ok(b)
}
