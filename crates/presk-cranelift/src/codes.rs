//! binary16 and `e8m15` in the native backend (plan 0.18.2 step 2, D216): a
//! buffer of either holds its codes ([`presk_rt::Buf::Codes`]), a kernel
//! computes in `f32`, and its loads widen a code exactly and its stores
//! round once, as `bf16`'s do. Cranelift 0.131 has no instruction that
//! converts to or from binary16 on aarch64, so both conversions are
//! integer sequences with one `f32` operation each, as `bf16`'s rounding
//! is. Each gives the bits of `presk_rt::codes`' function on every input,
//! which [`Conversions`] runs alone for the tests (FC1), and so the bits of
//! the kernel interpreter's emulation (FC2).
//!
//! A code is held in an `i32`, in its low 16 or 24 bits. An `e8m15` code
//! takes 3 bytes, read and written as 2 and 1, so a buffer needs no byte
//! past its last element; a kernel of either type has one lane.

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Value as CV, types};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use presk_kernel::ir::Dtype;
use presk_rt::RtError;

/// An `i32` constant, its bits `k`.
fn k32(b: &mut FunctionBuilder, k: u32) -> CV {
    b.ins().iconst(types::I32, i64::from(k))
}

/// The code of the element at `a + off` of a buffer of `d`'s codes, in the
/// low bits of an `i32`. An `e8m15` code is its low 2 bytes and its high
/// one, little-endian; its address need not be even.
pub(crate) fn load(b: &mut FunctionBuilder, d: Dtype, a: CV, off: i32) -> CV {
    if d == Dtype::F16 {
        return b.ins().uload16(types::I32, MemFlags::trusted(), a, off);
    }
    let lo = b.ins().uload16(types::I32, MemFlags::new().with_notrap(), a, off);
    let hi = b.ins().uload8(types::I32, MemFlags::trusted(), a, off + 2);
    let hi = b.ins().ishl_imm(hi, 16);
    b.ins().bor(lo, hi)
}

/// Stores code `c`, an `i32`, as the element at `a + off` of a buffer of
/// `d`'s codes ([`load`]'s layout).
pub(crate) fn store(b: &mut FunctionBuilder, d: Dtype, c: CV, a: CV, off: i32) {
    if d == Dtype::F16 {
        b.ins().istore16(MemFlags::trusted(), c, a, off);
        return;
    }
    b.ins().istore16(MemFlags::new().with_notrap(), c, a, off);
    let hi = b.ins().ushr_imm(c, 16);
    b.ins().istore8(MemFlags::trusted(), hi, a, off + 2);
}

/// The `f32` code `c` of `d` denotes, exactly (`presk_rt::codes::f16_to_f32`
/// and `e8m15_to_f32`).
/// - binary16: the magnitude moved to an `f32`'s exponent and fraction
///   fields, 13 bits up, is the number times 2⁻¹¹², a subnormal one
///   included; a multiplication by 2¹¹² gives it. An infinity's and a NaN's
///   field is all ones, the NaN made quiet, as `fcvt s, h` does.
/// - `e8m15`: the top 24 bits of the `f32`, a shift by 8.
pub(crate) fn widen(b: &mut FunctionBuilder, d: Dtype, c: CV) -> CV {
    if d != Dtype::F16 {
        let up = b.ins().ishl_imm(c, 8);
        return b.ins().bitcast(types::F32, MemFlags::new(), up);
    }
    let sign = b.ins().band_imm(c, 0x8000);
    let sign = b.ins().ishl_imm(sign, 16);
    let mag = b.ins().band_imm(c, 0x7fff);
    let m13 = b.ins().ishl_imm(mag, 13);
    let scaled = b.ins().bitcast(types::F32, MemFlags::new(), m13);
    let two112 = b.ins().f32const(f32::from_bits(0x7780_0000));
    let finite = b.ins().fmul(scaled, two112);
    let finite = b.ins().bitcast(types::I32, MemFlags::new(), finite);
    let all_ones = k32(b, 0x7f80_0000);
    let special = b.ins().bor(m13, all_ones);
    let quiet = k32(b, 0x40_0000);
    let quiet = b.ins().bor(special, quiet);
    let is_nan = b.ins().icmp_imm(IntCC::UnsignedGreaterThan, mag, 0x7c00);
    let special = b.ins().select(is_nan, quiet, special);
    let big = b.ins().icmp_imm(IntCC::UnsignedGreaterThanOrEqual, mag, 0x7c00);
    let m = b.ins().select(big, special, finite);
    let bits = b.ins().bor(m, sign);
    b.ins().bitcast(types::F32, MemFlags::new(), bits)
}

/// The code of `d` nearest the `f32` `x`, ties to even, in an `i32`
/// (`presk_rt::codes::f32_to_f16` and `f32_to_e8m15`). A NaN stays a NaN
/// of its sign, made quiet, with its payload's top bits.
/// - binary16, on the magnitude `a`: below 2⁻¹⁴, where the result is
///   subnormal or zero, `a + 0.5` in `f32`, whose unit in the last place is
///   binary16's least subnormal, so that the addition rounds; from 2⁻¹⁴,
///   `bf16`'s bias trick at bit 13 with the exponent rebiased, a carry out
///   of the largest finite number reaching the infinity's code; from 2¹⁶,
///   the infinity, and past the `f32` infinity, the NaN.
/// - `e8m15`: `bf16`'s bias trick at bit 8.
pub(crate) fn narrow(b: &mut FunctionBuilder, d: Dtype, x: CV) -> CV {
    let bits = b.ins().bitcast(types::I32, MemFlags::new(), x);
    if d != Dtype::F16 {
        let lsb = b.ins().ushr_imm(bits, 8);
        let lsb = b.ins().band_imm(lsb, 1);
        let bias = b.ins().iadd_imm(lsb, 0x7f);
        let rounded = b.ins().iadd(bits, bias);
        let code = b.ins().ushr_imm(rounded, 8);
        let quiet = b.ins().ushr_imm(bits, 8);
        let quiet = b.ins().bor_imm(quiet, 0x4000);
        let is_nan = b.ins().fcmp(FloatCC::NotEqual, x, x);
        return b.ins().select(is_nan, quiet, code);
    }
    let sign_bit = k32(b, 0x8000_0000);
    let sign = b.ins().band(bits, sign_bit);
    let a = b.ins().bxor(bits, sign);
    // Subnormal or zero: rounded by an addition.
    let af = b.ins().bitcast(types::F32, MemFlags::new(), a);
    let half = b.ins().f32const(0.5);
    let s = b.ins().fadd(af, half);
    let s = b.ins().bitcast(types::I32, MemFlags::new(), s);
    let half_bits = k32(b, 0x3f00_0000);
    let sub = b.ins().isub(s, half_bits);
    // Normal: the exponent rebiased from 127 to 15 and the bias trick.
    let odd = b.ins().ushr_imm(a, 13);
    let odd = b.ins().band_imm(odd, 1);
    let rebias = k32(b, 0xc800_0fff);
    let n = b.ins().iadd(a, rebias);
    let n = b.ins().iadd(n, odd);
    let n = b.ins().ushr_imm(n, 13);
    // An infinity, or a NaN, quiet, with its payload's top 9 bits.
    let nan = b.ins().ushr_imm(a, 13);
    let nan = b.ins().bor_imm(nan, 0x7e00);
    let nan = b.ins().band_imm(nan, 0x7fff);
    let inf = k32(b, 0x7c00);
    let f32_inf = k32(b, 0x7f80_0000);
    let is_nan = b.ins().icmp(IntCC::UnsignedGreaterThan, a, f32_inf);
    let special = b.ins().select(is_nan, nan, inf);
    let least_normal = k32(b, 0x3880_0000);
    let tiny = b.ins().icmp(IntCC::UnsignedLessThan, a, least_normal);
    let m = b.ins().select(tiny, sub, n);
    let two16 = k32(b, 0x4780_0000);
    let big = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, a, two16);
    let m = b.ins().select(big, special, m);
    let sign = b.ins().ushr_imm(sign, 16);
    b.ins().bor(m, sign)
}

/// 1 where rounding the `f32` `x` to `d` turns a finite value into an
/// infinity, else 0, in an `i32`: the overflow guard's test (plan 0.18.2
/// step 4, D218; `presk_rt::codes::overflows`). Its magnitude's bits lie
/// from the overflow threshold's (65 520's for binary16, `(2 − 2⁻¹⁶)·2¹²⁷`'s
/// for `e8m15`) up to, not including, the infinity's: one subtraction and
/// one unsigned comparison, a NaN's bits above the range.
pub(crate) fn overflows(b: &mut FunctionBuilder, d: Dtype, x: CV) -> CV {
    let threshold: u32 = if d == Dtype::F16 { 0x477f_f000 } else { 0x7f7f_ff80 };
    let bits = b.ins().bitcast(types::I32, MemFlags::new(), x);
    let a = b.ins().band_imm(bits, 0x7fff_ffff);
    let past = b.ins().iadd_imm(a, -i64::from(threshold));
    let o = b.ins().icmp_imm(IntCC::UnsignedLessThan, past, i64::from(0x7f80_0000 - threshold));
    b.ins().uextend(types::I32, o)
}

/// A loop over `n` elements: `src` to `dst`.
type ConvFn = unsafe extern "C" fn(*const u8, *mut u8, i64);

/// The native backend's conversions compiled alone, over arrays, by the
/// sequences its kernels emit ([`load`], [`store`], [`widen`], [`narrow`]):
/// what the tests hold to `presk_rt::codes` on every input (FC1's native
/// half, plan 0.18.2 step 2, D216); and the overflow guard's test
/// ([`overflows`], step 4, D218).
pub struct Conversions {
    _module: JITModule,
    /// `[binary16, e8m15]`: `f32`s to codes, codes to `f32`s, `f32`s to the
    /// guard's test, a byte each.
    narrow: [ConvFn; 2],
    widen: [ConvFn; 2],
    guard: [ConvFn; 2],
}

/// What a loop of [`Conversions`] computes.
#[derive(Clone, Copy, PartialEq)]
enum Loop {
    Narrow,
    Widen,
    Guard,
}

fn rt(e: impl std::fmt::Display) -> RtError {
    RtError { code: "E10001", message: format!("cranelift: {e}"), span: None }
}

impl Conversions {
    pub fn new() -> Result<Conversions, RtError> {
        let mut flags = settings::builder();
        flags.set("opt_level", "speed").map_err(rt)?;
        let isa = cranelift_native::builder().map_err(rt)?.finish(settings::Flags::new(flags)).map_err(rt)?;
        let mut module = JITModule::new(JITBuilder::with_isa(isa, cranelift_module::default_libcall_names()));
        let mut ctx = module.make_context();
        let mut fbc = FunctionBuilderContext::new();
        let mut ids = Vec::new();
        let loops = [(Dtype::F16, Loop::Narrow), (Dtype::E8M15, Loop::Narrow), (Dtype::F16, Loop::Widen), (Dtype::E8M15, Loop::Widen), (Dtype::F16, Loop::Guard), (Dtype::E8M15, Loop::Guard)];
        for (j, (d, what)) in loops.into_iter().enumerate() {
            let mut sig = module.make_signature();
            for _ in 0..3 {
                sig.params.push(AbiParam::new(types::I64));
            }
            let id = module.declare_function(&format!("c{j}"), Linkage::Local, &sig).map_err(rt)?;
            ctx.func.signature = sig;
            {
                let mut b = FunctionBuilder::new(&mut ctx.func, &mut fbc);
                let entry = b.create_block();
                let header = b.create_block();
                let body = b.create_block();
                let exit = b.create_block();
                b.append_block_params_for_function_params(entry);
                b.append_block_param(header, types::I64);
                b.switch_to_block(entry);
                let (src, dst, n) = (b.block_params(entry)[0], b.block_params(entry)[1], b.block_params(entry)[2]);
                let zero = b.ins().iconst(types::I64, 0);
                b.ins().jump(header, &[zero.into()]);
                b.switch_to_block(header);
                let i = b.block_params(header)[0];
                let more = b.ins().icmp(IntCC::SignedLessThan, i, n);
                b.ins().brif(more, body, &[], exit, &[]);
                b.switch_to_block(body);
                let w = d.bytes() as i64;
                let wide = b.ins().imul_imm(i, 4);
                let narrow_at = b.ins().imul_imm(i, w);
                match what {
                    Loop::Narrow => {
                        let a = b.ins().iadd(src, wide);
                        let x = b.ins().load(types::F32, MemFlags::trusted(), a, 0);
                        let c = narrow(&mut b, d, x);
                        let a = b.ins().iadd(dst, narrow_at);
                        store(&mut b, d, c, a, 0);
                    }
                    Loop::Widen => {
                        let a = b.ins().iadd(src, narrow_at);
                        let c = load(&mut b, d, a, 0);
                        let x = widen(&mut b, d, c);
                        let a = b.ins().iadd(dst, wide);
                        b.ins().store(MemFlags::trusted(), x, a, 0);
                    }
                    Loop::Guard => {
                        let a = b.ins().iadd(src, wide);
                        let x = b.ins().load(types::F32, MemFlags::trusted(), a, 0);
                        let o = overflows(&mut b, d, x);
                        let a = b.ins().iadd(dst, i);
                        b.ins().istore8(MemFlags::trusted(), o, a, 0);
                    }
                }
                let next = b.ins().iadd_imm(i, 1);
                b.ins().jump(header, &[next.into()]);
                b.switch_to_block(exit);
                b.ins().return_(&[]);
                b.seal_all_blocks();
                b.finalize();
            }
            module.define_function(id, &mut ctx).map_err(rt)?;
            module.clear_context(&mut ctx);
            ids.push(id);
        }
        module.finalize_definitions().map_err(rt)?;
        // SAFETY: each function was built with the signature `ConvFn`.
        let f = |j: usize| unsafe { std::mem::transmute::<*const u8, ConvFn>(module.get_finalized_function(ids[j])) };
        let (narrow, widen, guard) = ([f(0), f(1)], [f(2), f(3)], [f(4), f(5)]);
        Ok(Conversions { _module: module, narrow, widen, guard })
    }

    fn slot(d: Dtype) -> usize {
        match d {
            Dtype::F16 => 0,
            Dtype::E8M15 => 1,
            d => panic!("{d} is no type the native backend holds as its codes"),
        }
    }

    /// The codes of `xs` in `d`, `d.bytes()` bytes each, little-endian, as
    /// a kernel's stores write them.
    pub fn narrow(&self, d: Dtype, xs: &[f32]) -> Vec<u8> {
        let mut out = vec![0u8; xs.len() * d.bytes()];
        // SAFETY: `xs` holds `n` `f32`s and `out` `n` codes.
        unsafe { (self.narrow[Self::slot(d)])(xs.as_ptr() as *const u8, out.as_mut_ptr(), xs.len() as i64) };
        out
    }

    /// The overflow guard's test of each of `xs` stored in `d`, 1 where the
    /// store turns a finite value into an infinity, as a kernel's flag
    /// takes it (plan 0.18.2 step 4, D218).
    pub fn overflows(&self, d: Dtype, xs: &[f32]) -> Vec<u8> {
        let mut out = vec![0u8; xs.len()];
        // SAFETY: `xs` holds `n` `f32`s and `out` `n` bytes.
        unsafe { (self.guard[Self::slot(d)])(xs.as_ptr() as *const u8, out.as_mut_ptr(), xs.len() as i64) };
        out
    }

    /// The `f32`s the codes in `codes` denote, as a kernel's loads widen
    /// them.
    pub fn widen(&self, d: Dtype, codes: &[u8]) -> Vec<f32> {
        let n = codes.len() / d.bytes();
        let mut out = vec![0f32; n];
        // SAFETY: `codes` holds `n` codes and `out` `n` `f32`s.
        unsafe { (self.widen[Self::slot(d)])(codes.as_ptr(), out.as_mut_ptr() as *mut u8, n as i64) };
        out
    }
}
