//! FC1's native half (plan 0.18.2 step 2, D216): the native backend's
//! conversions between `f32` and binary16 and between `f32` and `e8m15`,
//! the integer sequences its kernels emit ([`presk_cranelift::codes`]),
//! compiled alone over arrays, give the runtime's bits (`presk_rt::codes`,
//! which FC1 holds to the formats' rounding on every input, D215), NaNs'
//! payloads included.
//!
//! The exhaustive checks are the ignored tests (`cargo test --release -p
//! presk-cranelift --test codes -- --ignored --nocapture`): every `f32`
//! (2³²) narrowed to each format and every `e8m15` code (2²⁴) widened. The
//! tests that run by default take every binary16 code and a sample of the
//! rest.
//!
//! And FC2's native half (the kernel interpreter's emulation's bits,
//! natively, byte for byte, NaNs' payloads included):
//! - at the kernel's level, every primitive of the emulation's suite
//!   lowered in either format, and kernels of `f32`, `f64` and `e8m15`
//!   reading inputs of a narrower type (`BufKind::Wide`: binary16, `e8m15`
//!   and `bf16`, in lanes where the kernel's type has them), on operands
//!   where binary16 overflows and on operands where it does not;
//! - on the test set's programs (`examples/tests/plan/entries.txt`, the
//!   tutorial's loss, the Llama decoder and the course GPT's gradient at
//!   small constants), under each format uniform and beside `f32`, `bf16`
//!   and `f64` (sites stored apart, parameters in their own types);
//! - on training runs: the tutorial's and the regression's, and the course
//!   GPT's first step, whose `f16` update gives NaN (AdamW's ε rounds to
//!   zero) on both executors alike.

use std::path::{Path, PathBuf};
use std::sync::Once;

use presk_core::PrimOp;
use presk_cranelift::{Conversions, NativeExec, native_backend};
use presk_interp::RunOptions;
use presk_kernel::exec::{output_buf, run_emulated_guarded};
use presk_kernel::ir::Dtype;
use presk_kernel::lower::{lower, Operand};
use presk_kernel::{BufKind, Kernel, KernelBackend, KernelExec};
use presk_rt::codes::{e8m15_to_f32, f16_to_f32, f32_to_e8m15, f32_to_f16};
use presk_rt::{Axis, Backend, Buf, Realization, Value};
use presk_syntax::diag::render_all;
use presk_syntax::{SourceMap, parse_file};
use presk_types::check;
use presk_types::elab::{self, Options};

/// The runtime's code of `x` in `d`, little-endian, `d.bytes()` bytes.
fn runtime_code(d: Dtype, x: f32) -> Vec<u8> {
    if d == Dtype::F16 { f32_to_f16(x).to_le_bytes().to_vec() } else { f32_to_e8m15(x).to_le_bytes()[..3].to_vec() }
}

/// The runtime's widening of code `c` of `d`.
fn runtime_value(d: Dtype, c: u32) -> f32 {
    if d == Dtype::F16 { f16_to_f32(c as u16) } else { e8m15_to_f32(c) }
}

/// `xs` narrowed natively against the runtime: the inputs (their bits)
/// that differ, the first eight, and how many.
fn narrowing_differs(conv: &Conversions, d: Dtype, xs: &[f32]) -> (Vec<u32>, u64) {
    let got = conv.narrow(d, xs);
    let w = d.bytes();
    let (mut first, mut n) = (Vec::new(), 0);
    for (i, x) in xs.iter().enumerate() {
        if got[i * w..(i + 1) * w] != runtime_code(d, *x)[..] {
            n += 1;
            if first.len() < 8 {
                first.push(x.to_bits());
            }
        }
    }
    (first, n)
}

/// The overflow guard's test of `xs` stored in `d`, natively, against
/// `presk_rt::codes::overflows` (plan 0.18.2 step 4, D218): the inputs that
/// differ, the first eight, and how many.
fn guard_differs(conv: &Conversions, d: Dtype, xs: &[f32]) -> (Vec<u32>, u64) {
    let got = conv.overflows(d, xs);
    let (mut first, mut n) = (Vec::new(), 0);
    for (x, o) in xs.iter().zip(got) {
        if (o == 1) != presk_rt::codes::overflows(d.format(), f64::from(*x)) || o > 1 {
            n += 1;
            if first.len() < 8 {
                first.push(x.to_bits());
            }
        }
    }
    (first, n)
}

/// Codes `lo..hi` of `d` widened natively against the runtime: the codes
/// that differ, the first eight, and how many.
fn widening_differs(conv: &Conversions, d: Dtype, lo: u32, hi: u32) -> (Vec<u32>, u64) {
    let codes: Vec<u8> = (lo..hi).flat_map(|c| c.to_le_bytes()[..d.bytes()].to_vec()).collect();
    let got = conv.widen(d, &codes);
    let (mut first, mut n) = (Vec::new(), 0);
    for (c, y) in (lo..hi).zip(got) {
        if y.to_bits() != runtime_value(d, c).to_bits() {
            n += 1;
            if first.len() < 8 {
                first.push(c);
            }
        }
    }
    (first, n)
}

/// `f32`s near every binary16 number and every 4093rd `e8m15` code, their
/// midpoints and the overflow thresholds, both signs, and the NaNs and
/// infinities: what the default tests narrow.
fn f32_sample() -> Vec<f32> {
    let mut v: Vec<u32> = (0..1u64 << 32).step_by(65_521).map(|b| b as u32).collect();
    for c in 0..1u32 << 16 {
        let x = f16_to_f32(c as u16);
        if x.is_finite() {
            let b = x.to_bits();
            v.extend([b, b.wrapping_sub(1), b + 1, b + 0x1000, b + 0x0fff, b + 0x1001]);
        }
    }
    for c in (0..1u32 << 24).step_by(4093) {
        let b = c << 8;
        v.extend([b, b + 0x7f, b + 0x80, b + 0x81, b.wrapping_sub(1)]);
    }
    for t in [65_520f32.to_bits(), 0x7f7f_ff80, 0x3880_0000, 0x3300_0000] {
        v.extend((t - 64..=t + 64).flat_map(|b| [b, b | 0x8000_0000]));
    }
    v.extend([0x7f80_0000, 0xff80_0000, 0x7fc0_0000, 0x7f80_0001, 0xffbf_ffff, 0x7fa0_2000, 0x0000_0001, 0x8000_0001]);
    v.into_iter().map(f32::from_bits).collect()
}

fn assert_none(what: &str, (first, n): (Vec<u32>, u64)) {
    assert!(n == 0, "{what}: {n} wrong, the first {}", first.iter().map(|b| format!("{b:#x}")).collect::<Vec<_>>().join(", "));
}

#[test]
fn the_native_conversions_are_the_runtime_s_on_a_sample() {
    let conv = Conversions::new().unwrap();
    let xs = f32_sample();
    assert!(xs.len() > 400_000, "{}", xs.len());
    for d in [Dtype::F16, Dtype::E8M15] {
        assert_none(&format!("f32 to {d}"), narrowing_differs(&conv, d, &xs));
    }
    assert_none("every binary16 code to f32", widening_differs(&conv, Dtype::F16, 0, 1 << 16));
    for lo in (0..1u32 << 24).step_by(1 << 20) {
        assert_none("e8m15 codes to f32", widening_differs(&conv, Dtype::E8M15, lo, lo + 4096));
    }
    for d in [Dtype::F16, Dtype::E8M15] {
        assert_none(&format!("the guard in {d}"), guard_differs(&conv, d, &xs));
    }
    assert_eq!(conv.overflows(Dtype::F16, &[65519.99, 65520.0, -1e9, f32::INFINITY, f32::NAN, f32::MAX]), [0, 1, 1, 0, 0, 1]);
    assert_eq!(conv.overflows(Dtype::E8M15, &[f32::from_bits(0x7f7f_ff7f), f32::from_bits(0x7f7f_ff80), -f32::MAX, f32::NEG_INFINITY]), [0, 1, 1, 0]);
    // A NaN's payload's top bits, made quiet: the sequences' own case.
    assert_eq!(conv.narrow(Dtype::F16, &[f32::from_bits(0xffa0_2000)]), 0xff01u16.to_le_bytes());
    assert_eq!(conv.widen(Dtype::F16, &0x7d01u16.to_le_bytes())[0].to_bits(), 0x7fe0_2000);
}

/// Every `f32`, on every core, in pieces of 2²⁰.
#[test]
#[ignore = "2^32 inputs to each format: seconds, a release build on 20 cores"]
fn every_f32_narrows_natively_as_the_runtime_does() {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()) as u64;
    let pieces = 1u64 << 12;
    for d in [Dtype::F16, Dtype::E8M15] {
        let results: Vec<(Vec<u32>, u64)> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..threads)
                .map(|t| {
                    s.spawn(move || {
                        let conv = Conversions::new().unwrap();
                        let (mut first, mut n) = (Vec::new(), 0);
                        for p in (t..pieces).step_by(threads as usize) {
                            let xs: Vec<f32> = (p << 20..(p + 1) << 20).map(|b| f32::from_bits(b as u32)).collect();
                            let (f, k) = narrowing_differs(&conv, d, &xs);
                            let (g, j) = guard_differs(&conv, d, &xs);
                            n += k + j;
                            first.extend(g.into_iter().take(8usize.saturating_sub(first.len())));
                            first.extend(f.into_iter().take(8usize.saturating_sub(first.len())));
                        }
                        (first, n)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let n: u64 = results.iter().map(|r| r.1).sum();
        eprintln!("every f32 to {d}, narrowed and the guard's test: {n} wrong");
        assert_none(&format!("every f32 to {d}"), (results.into_iter().flat_map(|r| r.0).take(8).collect(), n));
    }
}

#[test]
#[ignore = "2^24 codes: seconds, a release build"]
fn every_e8m15_code_widens_natively_as_the_runtime_does() {
    let conv = Conversions::new().unwrap();
    let mut n = 0;
    for lo in (0..1u32 << 24).step_by(1 << 20) {
        let (first, k) = widening_differs(&conv, Dtype::E8M15, lo, lo + (1 << 20));
        assert_none("e8m15 codes to f32", (first, k));
        n += 1 << 20;
    }
    eprintln!("every e8m15 code widened: {n}, 0 wrong");
}

fn ax(name: &str, len: usize) -> Axis {
    Axis { name: if name.is_empty() { None } else { Some(name.into()) }, len }
}

/// The emulation's suite (`presk-kernel`'s `tests/emulation.rs`): every
/// tensor primitive that reads reals, its real operands of kind `real`.
fn suite(real: BufKind) -> Vec<(PrimOp, Vec<Operand>)> {
    let o = |axes: Vec<Axis>| Operand { axes, kind: real };
    let x = || o(vec![ax("batch", 2), ax("model", 3)]);
    let row = || o(vec![ax("model", 3)]);
    let w = || o(vec![ax("", 3), ax("", 4)]);
    let q = || o(vec![ax("batch", 2), ax("heads", 2), ax("seq", 3), ax("hd", 2)]);
    let k2 = || o(vec![ax("batch", 2), ax("heads", 2), ax("seq2", 3), ax("hd", 2)]);
    let ct2 = || o(vec![ax("batch", 2)]);
    let sm = || o(vec![ax("batch", 2), ax("seq", 3), ax("model", 4)]);
    let logits = || o(vec![ax("batch", 2), ax("vocab", 5)]);
    let targets = || Operand { axes: vec![ax("batch", 2)], kind: BufKind::Idx };
    let wide = || o(vec![ax("batch", 3), ax("model", 8)]);
    let sm8 = || o(vec![ax("batch", 2), ax("seq", 3), ax("model", 8)]);
    vec![
        (PrimOp::Add, vec![x(), x()]),
        (PrimOp::Sub, vec![x(), row()]),
        (PrimOp::Mul, vec![x(), x()]),
        (PrimOp::Div, vec![x(), ct2()]),
        (PrimOp::Neg, vec![x()]),
        (PrimOp::Pow(3), vec![x()]),
        (PrimOp::Exp, vec![x()]),
        (PrimOp::Tanh, vec![x()]),
        (PrimOp::Sin, vec![x()]),
        (PrimOp::Cos, vec![x()]),
        (PrimOp::Abs, vec![x()]),
        (PrimOp::Gelu, vec![x()]),
        (PrimOp::GeluDeriv, vec![x()]),
        (PrimOp::Relu, vec![x()]),
        (PrimOp::Sign, vec![x()]),
        (PrimOp::Sigmoid, vec![x()]),
        (PrimOp::Dot, vec![x(), w()]),
        (PrimOp::Ein { operands: vec![vec!["batch".into(), "heads".into(), "seq".into(), "hd".into()], vec!["batch".into(), "heads".into(), "seq2".into(), "hd".into()]], output: vec!["batch".into(), "heads".into(), "seq".into(), "seq2".into()] }, vec![q(), k2()]),
        (PrimOp::SumAll, vec![x()]),
        (PrimOp::Mean { axis: "model".into() }, vec![x()]),
        (PrimOp::Softmax { axis: "model".into() }, vec![sm()]),
        (PrimOp::CrossEntropy, vec![logits(), targets()]),
        (PrimOp::Max, vec![x(), row()]),
        (PrimOp::SumSquares, vec![x()]),
        // Eight columns: a kernel of `f32` or `f64` reads them in lanes.
        (PrimOp::Dot, vec![wide(), o(vec![ax("", 8), ax("", 8)])]),
        (PrimOp::Softmax { axis: "model".into() }, vec![sm8()]),
    ]
}

/// `n` values in [−1, 1) from `seed`.
fn values(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        })
        .collect()
}

/// The operands of case `i` for a kernel of `d`, its values times `scale`,
/// each rounded once to the type its kind stores it in.
fn operands(i: usize, ins: &[Operand], d: Dtype, scale: f64) -> Vec<Buf> {
    ins.iter()
        .enumerate()
        .map(|(j, o)| {
            let n = o.axes.iter().map(|a| a.len).product();
            let v: Vec<f64> = values(n, (i * 8 + j) as u64).iter().map(|x| x * scale).collect();
            match o.kind {
                BufKind::Idx => Buf::I64((0..n as i64).map(|t| (t * 3 + 1) % 5).collect()),
                BufKind::Wide(w) => Buf::of(w, &v),
                _ => Buf::of(d, &v),
            }
        })
        .collect()
}

/// A buffer's bytes as a kernel's stores leave them.
fn raw(b: &Buf) -> Vec<u8> {
    match b {
        Buf::Codes(_, e) => e.to_vec(),
        Buf::F32(v) => v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect(),
        Buf::F64(v) => v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect(),
        Buf::Bf16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Buf::I64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        other => panic!("a kernel's output of {}", other.dtype()),
    }
}

/// Kernel `k` run natively and emulated on `ins`: the same bytes, and the
/// overflow guard's flag set natively (`E7007`, plan 0.18.2 step 4, D218)
/// exactly where the emulation's is, the output written either way.
/// Whether the emulation counted an infinity or a NaN.
fn same_natively(x: &mut NativeExec, idx: &mut usize, k: &Kernel, ins: &[Buf], what: &str) -> bool {
    let refs: Vec<&Buf> = ins.iter().collect();
    let mut emulated = output_buf(k);
    let (counts, flagged) = run_emulated_guarded(k, &refs, &mut emulated).unwrap_or_else(|e| panic!("{what}: {e}"));
    let mut native = output_buf(k);
    match x.run(*idx, k, &refs, &mut native) {
        Ok(()) => assert!(!flagged, "{what}: the emulation's store overflowed and the native guard did not trip"),
        Err(e) if e.code == "E7007" => assert!(flagged, "{what}: the native guard tripped where no emulated store overflowed: {}", e.message),
        Err(e) => panic!("{what}: {}", e.message),
    }
    *idx += 1;
    assert_eq!(raw(&native), raw(&emulated), "{what}: natively, the kernel interpreter's emulation's bytes");
    !counts.is_zero()
}

#[test]
fn every_primitive_in_either_format_is_the_emulation_s_bits_natively() {
    let mut x = NativeExec::new().unwrap();
    x.threads = 1;
    let (mut idx, mut cases, mut counted, mut laned) = (0, 0, 0, 0);
    // Uniform: every input and the output in the format.
    let mut kinds: Vec<(Dtype, BufKind)> = vec![(Dtype::F16, BufKind::Real), (Dtype::E8M15, BufKind::Real)];
    // Mixed: a kernel reading a narrower type's buffers, widened by its loads.
    for (d, w) in [(Dtype::F32, Dtype::F16), (Dtype::F32, Dtype::E8M15), (Dtype::F64, Dtype::F16), (Dtype::F64, Dtype::E8M15), (Dtype::E8M15, Dtype::F16), (Dtype::E8M15, Dtype::BF16)] {
        kinds.push((d, BufKind::Wide(w)));
    }
    for (d, kind) in kinds {
        for (i, (op, ins)) in suite(kind).iter().enumerate() {
            // A primitive the lowering leaves to the host for such an input.
            let Some(l) = lower(op, ins, d) else {
                assert!(kind != BufKind::Real, "{} in {d}: a host operation", op.name());
                continue;
            };
            let k = l.unwrap_or_else(|e| panic!("{}: {e}", op.name())).kernel;
            laned += usize::from(presk_cranelift::laned(&k).is_some());
            for scale in [1.0, 3e4] {
                let what = format!("{} in {d}, reading {kind:?}, at {scale:e}", op.name());
                counted += usize::from(same_natively(&mut x, &mut idx, &k, &operands(i, ins, d, scale), &what));
                cases += 1;
            }
        }
    }
    eprintln!("{cases} kernels' runs natively the emulation's bits, {counted} with an infinity or a NaN counted, {laned} kernels in lanes");
    assert!(counted > 10 && laned >= 4, "{counted} runs overflowed, {laned} kernels in lanes");
}

fn chdir_root() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let root: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap();
        std::env::set_current_dir(root).unwrap()
    });
}

/// A program after automatic differentiation, as `presk run` has it.
fn prepare(rel: &str, consts: &[(String, u64)]) -> (elab::ElabProgram, presk_core::Program) {
    chdir_root();
    let mut sm = SourceMap::new();
    let id = sm.load(Path::new(rel)).unwrap();
    let (program, diags) = parse_file(&sm, id);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (prog, diags) = elab::elaborate(&mut sm, id, program, &Options { const_overrides: consts.to_vec() });
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (typed, diags) = check::check(&sm, &prog);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (mut core, diags) = presk_core::lower(&prog, &typed);
    assert!(diags.is_empty(), "{}", render_all(&diags, &sm, false));
    let diags = presk_core::ad::transform(&mut core);
    assert!(diags.is_empty(), "{}", render_all(&diags, &sm, false));
    (prog, core)
}

/// `(file, entry, consts, dims)`.
type Case = (String, String, Vec<(String, u64)>, Vec<(String, u64)>);

/// `--const N=V --dim N=V` as the test set's lines give them.
fn case(file: &str, entry: &str, args: &str) -> Case {
    let (mut consts, mut dims) = (Vec::new(), Vec::new());
    let mut words = args.split_whitespace();
    while let (Some(flag), Some(kv)) = (words.next(), words.next()) {
        let (k, v) = kv.split_once('=').unwrap();
        let pair = (k.to_string(), v.parse().unwrap());
        if flag == "--const" { consts.push(pair) } else { dims.push(pair) }
    }
    (file.into(), entry.into(), consts, dims)
}

fn cases() -> Vec<Case> {
    chdir_root();
    let mut out: Vec<Case> = std::fs::read_to_string("examples/tests/plan/entries.txt")
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut p = l.splitn(3, '|');
            case(p.next().unwrap(), p.next().unwrap(), p.next().unwrap_or(""))
        })
        .collect();
    out.push(case("examples/tutorial/07_train.psk", "loss", "--dim B=4"));
    let llama = "--const V=100 --const D=32 --const H=2 --const G=2 --const FF=64 --const L=2 --dim B=2 --dim T=16";
    out.push(case("examples/coverage/llama.psk", "Llama.apply", llama));
    out.push(case("examples/coverage/llama.psk", "loss", llama));
    let gpt = "--const V=92 --const D=16 --const H=2 --const L=1 --dim B=2 --dim T=32";
    out.push(case("examples/experiments/gpt_grad.psk", "loss", gpt));
    out.push(case("examples/experiments/gpt_grad.psk", "loss_grad", gpt));
    out
}

/// Each leaf of `v`, a buffer's bytes as [`raw`] gives them.
fn leaves(v: &Value) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    v.map_leaves(&mut |l| {
        out.push(match l {
            Value::Real(t) => raw(&t.data),
            other => format!("{other:?}").into_bytes(),
        });
        l.clone()
    });
    out
}

/// The realizations a program runs under: each format uniform; each beside
/// `f32` and `bf16`, every third of `sites` stored apart and every
/// parameter in the other format; parameters of `f16`, `e8m15` and `bf16`
/// read by kernels of `f64`, `e8m15` and `f32` (`BufKind::Wide`), and of
/// `f32` by kernels of binary16.
fn realizations(sites: &[String], params: &[String]) -> Vec<Realization> {
    let typed = |default: Dtype, every_third: Option<Dtype>, own: Option<Dtype>| {
        let mut r = Realization::uniform(default);
        if let Some(d) = every_third {
            r.storage = sites.iter().step_by(3).map(|s| (s.clone(), d)).collect();
        }
        if let Some(d) = own {
            r.params = params.iter().map(|p| (p.clone(), d)).collect();
        }
        r
    };
    vec![
        Realization::uniform(Dtype::F16),
        Realization::uniform(Dtype::E8M15),
        typed(Dtype::F32, Some(Dtype::F16), Some(Dtype::E8M15)),
        typed(Dtype::BF16, Some(Dtype::E8M15), Some(Dtype::F16)),
        typed(Dtype::F32, Some(Dtype::E8M15), Some(Dtype::BF16)),
        typed(Dtype::F64, None, Some(Dtype::F16)),
        typed(Dtype::E8M15, Some(Dtype::F32), Some(Dtype::F16)),
        typed(Dtype::F16, Some(Dtype::BF16), Some(Dtype::F32)),
    ]
}

#[test]
fn the_test_set_in_either_format_is_the_emulation_s_bits_natively() {
    let (mut runs, mut guarded) = (0, 0);
    for (file, entry, consts, dims) in cases() {
        let (prog, core) = prepare(&file, &consts);
        let opts = RunOptions { entry: entry.clone(), dims: dims.clone(), seed: 0, bits: 128, digits: 17 };
        let (func, fdims, inputs) = presk_interp::run::prepare_run(&core, &prog, &opts).unwrap();
        let args: Vec<Value> = inputs.iter().map(|(_, v)| Value::from_interp(v).unwrap()).collect();
        let params = match args.first() {
            Some(theta @ Value::Record(_)) => presk_rt::quant::param_paths(theta),
            _ => Vec::new(),
        };
        let sites: Vec<String> = core.funcs.iter().flat_map(|f| f.body.lets.iter().map(move |l| format!("{}/%{}", f.name, l.var.0))).collect();
        for r in realizations(&sites, &params) {
            let what = format!("{file} {entry} under {}", r.label());
            let mut outs = Vec::new();
            for native in [false, true] {
                let mut b: Box<dyn Backend> = if native { Box::new(native_backend(&core, &prog, &r).unwrap()) } else { Box::new(KernelBackend::new(&core, &prog, &r)) };
                let mut a = args.clone();
                r.store_params(&mut a).unwrap_or_else(|e| panic!("{what}: {}", e.message));
                let p = b.prepare(func, &fdims, &r).unwrap_or_else(|e| panic!("{what} on {}: {}", b.name(), e.message));
                // A run the overflow guard stops (D218) is its message, on
                // both executors.
                outs.push(match b.call(&p, a) {
                    Ok(v) => Ok(leaves(&v)),
                    Err(e) if e.code == "E7007" => Err(e.message),
                    Err(e) => panic!("{what} on {}: {}", b.name(), e.message),
                });
            }
            assert!(outs[0] == outs[1], "{what}: natively, not the kernel interpreter's emulation's bytes: {:?}", outs.iter().map(|o| o.as_ref().err()).collect::<Vec<_>>());
            guarded += usize::from(outs[0].is_err());
            runs += 1;
        }
    }
    eprintln!("{runs} runs of the test set natively, each the emulation's bytes; {guarded} stopped by the overflow guard on both, in the same words");
    assert!(runs >= 160, "{runs}");
}

/// A training run of `rel`, `steps` steps, under `r`, natively or on the
/// kernel interpreter: its parameters' bytes and its losses' bits.
fn trained(rel: &str, consts: &[(String, u64)], steps: u64, r: &Realization, native: bool) -> (Vec<Vec<u8>>, Vec<u64>) {
    use presk_train::{TrainOptions, prepare_training, train};
    chdir_root();
    let mut sm = SourceMap::new();
    let id = sm.load(Path::new(rel)).unwrap();
    let (program, _) = parse_file(&sm, id);
    let (prog, _) = elab::elaborate(&mut sm, id, program, &Options { const_overrides: consts.to_vec() });
    let (typed, _) = check::check(&sm, &prog);
    let (mut core, _) = presk_core::lower(&prog, &typed);
    presk_core::ad::transform(&mut core);
    let fns = prepare_training(&mut core, &prog, &typed).unwrap();
    let opts = TrainOptions { data: None, steps: Some(steps), epochs: None, seed: 0, bits: 128, realize: r.clone(), ckpt: None, checkpoints: false };
    let out = if native { train(&mut native_backend(&core, &prog, r).unwrap(), &core, &prog, &typed, &fns, &opts, |_| {}) } else { train(&mut KernelBackend::new(&core, &prog, r), &core, &prog, &typed, &fns, &opts, |_| {}) };
    let out = out.unwrap_or_else(|e| panic!("{rel} under {}: {}", r.label(), e.message));
    (leaves(&out.theta), out.logs.iter().map(|l| l.loss.to_bits()).collect())
}

#[test]
fn training_in_either_format_is_the_emulation_s_bits_natively() {
    let none: Vec<(String, u64)> = Vec::new();
    let gpt = vec![("D".to_string(), 16), ("H".to_string(), 2), ("L".to_string(), 1)];
    for (rel, consts, steps) in [("examples/tutorial/07_train.psk", &none, 20), ("examples/tests/train/mlp_regression.psk", &none, 20), ("examples/tests/train/dropout_reg.psk", &none, 20), ("examples/corpus/nlp/05_gpt.psk", &gpt, 1)] {
        for d in [Dtype::F16, Dtype::E8M15] {
            let r = Realization::uniform(d);
            let (emulated, native) = (trained(rel, consts, steps, &r, false), trained(rel, consts, steps, &r, true));
            assert!(emulated == native, "{rel} in {d}, {steps} steps: natively, not the emulation's parameters and losses");
            eprintln!("{rel} in {d}: {steps} steps, the last loss {:e}", f64::from_bits(*native.1.last().unwrap()));
        }
    }
}
