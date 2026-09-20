//! The Cranelift backend (plan 0.3.0, step 3): every PSK-Kernel v0 kernel is
//! compiled once to machine code by Cranelift's JIT and called through a
//! function pointer; the host program is the kernel backend's. Buffers are
//! passed as an array of pointers, loops become basic blocks, `sqrt` is
//! native and the other transcendental functions and `erf` are calls into
//! `libm`, the same definitions the kernel interpreter uses.

use std::collections::HashMap;

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Type, Value as CV, types};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};
use presk_core::Program;
use presk_kernel::ir::{BinOp, BufKind, CmpOp, Dtype, Expr, Kernel, Stmt, UnOp};
use presk_kernel::{KernelBackend, KernelExec};
use presk_rt::{Buf, Realization, RtError};
use presk_types::elab::ElabProgram;

type Result<T> = std::result::Result<T, RtError>;

fn err<T>(message: impl Into<String>) -> Result<T> {
    Err(RtError { code: "E10001", message: message.into(), span: None })
}

/// A compiled kernel: the buffer pointers, then the half-open range of its
/// outermost loop. A sequential call passes the whole range.
type KernelFn = unsafe extern "C" fn(*const *mut u8, i64, i64);

/// The buffer table, sent to a worker thread. Safe because the threads of one
/// launch share read-only inputs and write disjoint elements of the output:
/// that is what `Kernel::parallel_outer` establishes before we split.
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

/// Below this many weighted operations a kernel runs on one thread: spawning
/// costs tens of microseconds, which only a big kernel earns back.
const PARALLEL_MIN_WORK: f64 = 4.0e6;

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
        Ok(NativeExec { module, ctx, fbc: FunctionBuilderContext::new(), compiled: vec![], libm: HashMap::new(), compiled_count: 0, threads, threaded: 0 })
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
        let mut libm_ids: HashMap<&'static str, FuncId> = HashMap::new();
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
        {
            let mut b = FunctionBuilder::new(&mut self.ctx.func, &mut self.fbc);
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
            let esize: Vec<i64> = k.inputs.iter().map(|s| if s.kind == BufKind::Idx { 8 } else { real_bytes }).chain(std::iter::once(if k.output.kind == BufKind::Idx { 8 } else { real_bytes })).collect();
            let mut refs: HashMap<&'static str, cranelift_codegen::ir::FuncRef> = HashMap::new();
            for (n, id) in &libm_ids {
                refs.insert(n, self.module.declare_func_in_func(*id, b.func));
            }
            let mut g = Gen { b, fty, narrow: k.dtype == Dtype::Bf16, bufs, strides, esize, loops, reals, ints, libm: refs };
            // The outermost loop runs over the range the caller gives, so one
            // compiled kernel serves the sequential call and every worker.
            match (k.parallel_outer(), k.body.as_slice()) {
                (Some(_), [Stmt::For { var, body, .. }]) => g.for_range(*var, lo, hi, body),
                _ => g.stmts(&k.body),
            }
            g.b.ins().return_(&[]);
            g.b.finalize();
        }
        self.module.define_function(fid, &mut self.ctx).map_err(|e| RtError { code: "E10001", message: format!("cranelift: {e}"), span: None })?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions().map_err(|e| RtError { code: "E10001", message: e.to_string(), span: None })?;
        let ptr = self.module.get_finalized_function(fid);
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
            Stmt::Set { value, .. } | Stmt::SetI { value, .. } | Stmt::SetL { value, .. } => expr(value, out),
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
    /// Real buffers hold `bf16`: loads widen and stores round.
    narrow: bool,
    loops: Vec<Variable>,
    reals: Vec<Variable>,
    ints: Vec<Variable>,
    libm: HashMap<&'static str, cranelift_codegen::ir::FuncRef>,
}

impl Gen<'_> {
    fn addr(&mut self, buf: usize, index: &[Expr]) -> CV {
        let mut off = self.b.ins().iconst(types::I64, 0);
        for (e, s) in index.iter().zip(self.strides[buf].clone()) {
            let i = self.gen_i(e);
            let t = self.b.ins().imul_imm(i, s as i64);
            off = self.b.ins().iadd(off, t);
        }
        let bytes = self.b.ins().imul_imm(off, self.esize[buf]);
        self.b.ins().iadd(self.bufs[buf], bytes)
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

    fn gen_f(&mut self, e: &Expr) -> CV {
        match e {
            Expr::F(x) => fconst(&mut self.b, self.fty, *x),
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
                    self.b.ins().load(self.fty, MemFlags::trusted(), a, 0)
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
        let v = self.loops[var];
        self.b.def_var(v, lo);
        let header = self.b.create_block();
        let body_b = self.b.create_block();
        let exit = self.b.create_block();
        self.b.ins().jump(header, &[]);
        self.b.switch_to_block(header);
        let i = self.b.use_var(v);
        let c = self.b.ins().icmp(IntCC::SignedLessThan, i, hi);
        self.b.ins().brif(c, body_b, &[], exit, &[]);
        self.b.switch_to_block(body_b);
        self.b.seal_block(body_b);
        self.stmts(body);
        let i = self.b.use_var(v);
        let next = self.b.ins().iadd_imm(i, 1);
        self.b.def_var(v, next);
        self.b.ins().jump(header, &[]);
        self.b.seal_block(header);
        self.b.switch_to_block(exit);
        self.b.seal_block(exit);
    }

    fn stmts(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            match s {
                Stmt::For { var, bound, body } => {
                    let zero = self.b.ins().iconst(types::I64, 0);
                    let n = self.b.ins().iconst(types::I64, *bound as i64);
                    self.for_range(*var, zero, n, body);
                }
                Stmt::Set { var, value } => {
                    let x = self.gen_f(value);
                    self.b.def_var(self.reals[*var], x);
                }
                Stmt::SetI { var, value } => {
                    let x = self.gen_i(value);
                    self.b.def_var(self.ints[*var], x);
                }
                Stmt::SetL { var, value } => {
                    let x = self.gen_i(value);
                    self.b.def_var(self.loops[*var], x);
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
                    self.stmts(then);
                    self.b.ins().jump(merge, &[]);
                    self.b.switch_to_block(merge);
                    self.b.seal_block(merge);
                }
            }
        }
    }
}

fn buf_ptr(b: &Buf) -> Result<*mut u8> {
    Ok(match b {
        Buf::F64(v) => v.as_ptr() as *mut u8,
        Buf::F32(v) => v.as_ptr() as *mut u8,
        Buf::Bf16(v) => v.as_ptr() as *mut u8,
        Buf::I64(v) => v.as_ptr() as *mut u8,
        Buf::Exact(_) => return err("native backend: an exact buffer reached a kernel"),
    })
}

impl KernelExec for NativeExec {
    fn name(&self) -> &'static str {
        "native"
    }

    fn run(&mut self, idx: usize, k: &Kernel, inputs: &[&Buf], out: &mut Buf) -> Result<()> {
        if self.compiled.len() <= idx {
            self.compiled.resize(idx + 1, None);
        }
        let f = match self.compiled[idx] {
            Some(f) => f,
            None => {
                let f = self.compile(idx, k)?;
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
            Buf::Exact(_) => return err("native backend: an exact output buffer"),
        });
        // Threads over the outermost axis, when the analysis says the writes are
        // disjoint and the kernel is big enough to pay for spawning. Each output
        // element is computed by the same inner loops in the same order, so the
        // result is bit-identical to the sequential one.
        let parallel = k.parallel_outer();
        let work = presk_kernel::cost::kernel_cost(k).work;
        let workers = match parallel {
            Some(b) if self.threads > 1 && b >= 2 && work >= PARALLEL_MIN_WORK => self.threads.min(b),
            _ => 1,
        };
        // SAFETY: the kernel's loop bounds are the buffers' dimensions (checked by the
        // lowering and exercised by the kernel interpreter); inputs are never written.
        if workers <= 1 {
            let hi = parallel.unwrap_or(0) as i64;
            unsafe { f(ptrs.as_ptr(), 0, hi) };
            return Ok(());
        }
        let total = parallel.unwrap_or(0) as i64;
        let p = Ptrs(ptrs.as_ptr());
        self.threaded += 1;
        std::thread::scope(|s| {
            for t in 0..workers as i64 {
                let lo = total * t / workers as i64;
                let hi = total * (t + 1) / workers as i64;
                if lo == hi {
                    continue;
                }
                // SAFETY: the workers share read-only inputs and write disjoint
                // slices of the output (`Kernel::parallel_outer`).
                s.spawn(move || unsafe { f(p.get(), lo, hi) });
            }
        });
        Ok(())
    }
}

/// A kernel backend whose kernels are JIT-compiled by Cranelift.
pub fn native_backend<'p>(prog: &'p Program, elab: &'p ElabProgram, realize: &Realization, seed: u64) -> Result<KernelBackend<'p>> {
    Ok(KernelBackend::with_exec(prog, elab, realize, seed, Box::new(NativeExec::new()?)))
}

/// The same backend with an explicit number of workers, so a caller can pin it
/// (the tests compare one thread against many, which must agree exactly).
pub fn native_backend_with_threads<'p>(prog: &'p Program, elab: &'p ElabProgram, realize: &Realization, seed: u64, threads: usize) -> Result<KernelBackend<'p>> {
    let mut exec = NativeExec::new()?;
    exec.threads = threads.max(1);
    Ok(KernelBackend::with_exec(prog, elab, realize, seed, Box::new(exec)))
}
