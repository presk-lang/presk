//! The kernel printer's check (plan 0.9.0 §3.2, step 1; D77): the printed
//! C, compiled with the export's flags and `-Wall -Wextra -Werror`, run on
//! the inputs the native backend's kernels get, gives the native backend's
//! bits on
//! - the 41 primitive cases of `presk-kernel/tests/conformance.rs`, in
//!   `f64`, `f32` and `bf16`, scalar and at each type's natural lane width;
//! - every layout of the four nests of `presk-cranelift/tests/layout.rs`
//!   (linear, scores, values, normalize) in the three types, and the thread
//!   splits that test states (all of them on every point in an `#[ignore]`d
//!   test);
//! - kernels written by hand, for the forms the lowering does not produce
//!   today;
//! - integer weights (step 7, D84): the primitives with every real operand
//!   quantized, in `int8` and `int4`, their scales constant along the first
//!   axis or the last, in the three types, scalar and in lanes; and the four
//!   nests' layouts with an operand quantized;
//!
//! each kernel whose loop a split may divide run over its whole range, over
//! the pieces the native backend gives its workers, and one iteration at a
//! time, each piece writing elements no other piece writes. Kernels calling
//! `exp`, … call `presk_libm.c`'s functions (step 2, D78), compiled into the
//! same library.
//!
//! And the contraction rule (§3.5, rule 1): the primitives' check built in
//! GCC's GNU mode, without `-ffp-contract=off`, moves bits on aarch64.

mod cexec;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Once};
use std::time::Instant;

use cexec::{CExec, CLib, PARTITIONS, raw};
use presk_core::ir::Rhs;
use presk_interp::RunOptions;
use presk_kernel::lower::Operand;
use presk_kernel::{Kernel, KernelBackend};
use presk_rt::layout::Split;
use presk_rt::{Dtype, Layout, Plan, Realization, ReductionOrder, Value};
use presk_syntax::diag::render_all;
use presk_syntax::{SourceMap, parse_file};
use presk_types::check;
use presk_types::elab::{self, ElabProgram, Options};
use rug::Float;

type IValue = presk_interp::Value<Float>;

const DTYPES: [Dtype; 3] = [Dtype::F64, Dtype::F32, Dtype::Bf16];
/// The native runs have four workers, as `layout.rs`'s, so a plan's split
/// takes 2 or 4 (condition 5); the C executor states the same.
const WORKERS: usize = 4;

fn chdir_root() {
    static ONCE: Once = Once::new();
    let root: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap();
    ONCE.call_once(|| std::env::set_current_dir(root).unwrap());
}

fn prepare(rel: &str, consts: &[(&str, u64)]) -> (ElabProgram, presk_core::Program) {
    chdir_root();
    let mut sm = SourceMap::new();
    let id = sm.load(Path::new(rel)).unwrap();
    let (program, diags) = parse_file(&sm, id);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let opts = Options { const_overrides: consts.iter().map(|(n, v)| (n.to_string(), *v)).collect() };
    let (prog, diags) = elab::elaborate(&mut sm, id, program, &opts);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (typed, diags) = check::check(&sm, &prog);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (mut core, diags) = presk_core::lower(&prog, &typed);
    assert!(diags.is_empty(), "{}", render_all(&diags, &sm, false));
    let diags = presk_core::ad::transform(&mut core);
    assert!(diags.is_empty(), "{}", render_all(&diags, &sm, false));
    presk_core::verify(&core).unwrap();
    (prog, core)
}

/// A kernel backend over the C, laid out as the native backend is: lanes
/// unstated, each kernel at its type's natural width, unless `lanes` says.
fn c_backend<'p>(core: &'p presk_core::Program, prog: &'p ElabProgram, r: &Realization, lib: &Arc<CLib>, partition: cexec::Partition, lanes: Option<usize>) -> KernelBackend<'p> {
    let mut b = KernelBackend::with_exec(core, prog, r, Box::new(CExec::new(lib.clone(), partition, WORKERS)));
    b.default_layout.lanes = lanes;
    b
}

fn native_backend<'p>(core: &'p presk_core::Program, prog: &'p ElabProgram, r: &Realization, lanes: Option<usize>) -> KernelBackend<'p> {
    let mut b = presk_cranelift::native_backend_with_threads(core, prog, r, WORKERS).unwrap();
    b.default_layout.lanes = lanes;
    b
}

/// The C of a linear layer's kernels in the three types, as the native
/// backend lowers them (lanes in `f64` and `f32`): the printer's form, for
/// review.
#[test]
fn c_of_a_linear_layer() {
    let (prog, core) = prepare("tutorial/01_linear.psk", &[]);
    let opts = RunOptions { entry: "Linear{4, 8}.apply".into(), dims: vec![("batch".into(), 2)], seed: 0, bits: 64, digits: 12 };
    let mut text = String::new();
    let mut all: Vec<Kernel> = Vec::new();
    for d in DTYPES {
        let r = Realization::uniform(d);
        let mut b = native_backend(&core, &prog, &r, None);
        presk_rt::run_with(&mut b, &core, &prog, &opts, &r).unwrap();
        for (i, k) in b.kernels.iter().enumerate() {
            text.push_str(&presk_export_c::kernel_fn(k, &format!("{d}_k{i}")).unwrap());
            text.push('\n');
        }
        all.extend(b.kernels);
    }
    insta::assert_snapshot!("linear_c", text);
    CLib::build("linear", &all);
}

/// Every tensor primitive, as `presk-kernel`'s conformance suite states them.
fn primitive_cases() -> Vec<(presk_core::PrimOp, Vec<IValue>)> {
    use presk_core::PrimOp;
    let ax = |name: Option<&str>, len: usize| presk_interp::value::Axis { name: name.map(String::from), len };
    let real = |axes: Vec<presk_interp::value::Axis>, seed: u64| -> IValue {
        let n: usize = axes.iter().map(|a| a.len).product();
        let mut s = seed;
        let data = (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                Float::with_val(256, ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0)
            })
            .collect();
        IValue::Real(presk_interp::Tensor { axes, data })
    };
    let idx = |axes: Vec<presk_interp::value::Axis>, bound: u64, seed: u64| -> IValue {
        let n: usize = axes.iter().map(|a| a.len).product();
        IValue::Idx { bound, t: presk_interp::Tensor { axes, data: (0..n as u64).map(|i| ((i * 7 + seed) % bound) as i64).collect() } }
    };
    let x = real(vec![ax(Some("batch"), 2), ax(Some("model"), 3)], 1);
    let y = real(vec![ax(Some("batch"), 2), ax(Some("model"), 3)], 2);
    let row = real(vec![ax(Some("model"), 3)], 3);
    let scalar = real(vec![], 4);
    let w = real(vec![ax(None, 3), ax(None, 4)], 5);
    let q = real(vec![ax(Some("batch"), 2), ax(Some("heads"), 2), ax(Some("seq"), 3), ax(Some("hd"), 2)], 6);
    let k = real(vec![ax(Some("batch"), 2), ax(Some("heads"), 2), ax(Some("seq"), 3), ax(Some("hd"), 2)], 7);
    let table = real(vec![ax(None, 5), ax(None, 3)], 8);
    let ids = idx(vec![ax(Some("batch"), 2), ax(Some("seq"), 4)], 5, 1);
    let logits = real(vec![ax(Some("batch"), 2), ax(Some("vocab"), 5)], 9);
    let targets = idx(vec![ax(Some("batch"), 2)], 5, 2);
    let ct2 = real(vec![ax(Some("batch"), 2)], 10);
    let ct3 = real(vec![ax(Some("batch"), 2), ax(Some("seq"), 4), ax(None, 3)], 11);
    let sm = real(vec![ax(Some("batch"), 2), ax(Some("seq"), 3), ax(Some("model"), 4)], 12);
    vec![
        (PrimOp::Add, vec![x.clone(), y.clone()]),
        (PrimOp::Sub, vec![x.clone(), row.clone()]),
        (PrimOp::Mul, vec![x.clone(), scalar.clone()]),
        (PrimOp::Div, vec![x.clone(), ct2.clone()]),
        (PrimOp::Neg, vec![x.clone()]),
        (PrimOp::Pow(3), vec![x.clone()]),
        (PrimOp::Exp, vec![x.clone()]),
        (PrimOp::Tanh, vec![x.clone()]),
        (PrimOp::Sin, vec![x.clone()]),
        (PrimOp::Cos, vec![x.clone()]),
        (PrimOp::Abs, vec![x.clone()]),
        (PrimOp::Gelu, vec![x.clone()]),
        (PrimOp::GeluDeriv, vec![x.clone()]),
        (PrimOp::Relu, vec![x.clone()]),
        (PrimOp::Heaviside, vec![x.clone()]),
        (PrimOp::Sign, vec![x.clone()]),
        (PrimOp::Sigmoid, vec![x.clone()]),
        (PrimOp::Dot, vec![x.clone(), w.clone()]),
        (PrimOp::DotVjpL, vec![real(vec![ax(Some("batch"), 2), ax(None, 4)], 13), w.clone()]),
        (PrimOp::DotVjpR, vec![x.clone(), real(vec![ax(Some("batch"), 2), ax(None, 4)], 14)]),
        (PrimOp::Ein { operands: vec![vec!["batch".into(), "heads".into(), "seq".into(), "hd".into()], vec!["batch".into(), "heads".into(), "seq2".into(), "hd".into()]], output: vec!["batch".into(), "heads".into(), "seq".into(), "seq2".into()] }, vec![q.clone(), k.clone()]),
        (PrimOp::SumAll, vec![x.clone()]),
        (PrimOp::MeanAll, vec![x.clone()]),
        (PrimOp::Sum { axis: "model".into() }, vec![x.clone()]),
        (PrimOp::Mean { axis: "model".into() }, vec![x.clone()]),
        (PrimOp::MeanVjp { axis: Some("model".into()) }, vec![ct2.clone(), x.clone()]),
        (PrimOp::MeanVjp { axis: None }, vec![scalar.clone(), x.clone()]),
        (PrimOp::Softmax { axis: "model".into() }, vec![x.clone()]),
        (PrimOp::Embed, vec![table.clone(), ids.clone()]),
        (PrimOp::EmbedVjp, vec![table.clone(), ids.clone(), ct3.clone()]),
        (PrimOp::CrossEntropy, vec![logits.clone(), targets.clone()]),
        (PrimOp::CrossEntropyVjp, vec![logits.clone(), targets.clone(), ct2.clone()]),
        (PrimOp::Argmax, vec![logits.clone()]),
        (PrimOp::IndexConst { axis: "seq".into(), index: -1 }, vec![sm.clone()]),
        (PrimOp::ScatterIndex { axis: "seq".into(), index: 1 }, vec![sm.clone(), real(vec![ax(Some("batch"), 2), ax(Some("model"), 4)], 15)]),
        (PrimOp::SplitHeads { h: 2, hd: 2 }, vec![sm.clone()]),
        (PrimOp::MergeHeads { h: 2, hd: 2 }, vec![q.clone()]),
        (PrimOp::SumTo, vec![x.clone(), row.clone()]),
        (PrimOp::BroadcastLike, vec![row.clone(), x.clone()]),
        (PrimOp::SumSquares, vec![x.clone()]),
        (PrimOp::Real, vec![ids.clone()]),
    ]
}

/// A primitive run on the native backend: its type and lanes, the case, its
/// arguments, its result's bits, and whether its kernels call a
/// transcendental function.
type PrimRun = (Dtype, Option<usize>, &'static presk_core::PrimOp, Vec<Value>, Vec<u64>, bool);

/// Every primitive case in the three types, scalar and at each type's
/// natural width (`bf16` has none: one lane), run on the native backend:
/// the runs and their kernels.
fn primitive_runs(prog: &ElabProgram, core: &presk_core::Program, cases: &'static [(presk_core::PrimOp, Vec<IValue>)]) -> (Vec<PrimRun>, Vec<Kernel>) {
    let widths = [Some(1), None];
    let mut runs = Vec::new();
    let mut kernels: Vec<Kernel> = Vec::new();
    for d in DTYPES {
        let r = Realization::uniform(d);
        for lanes in widths {
            for (op, args) in cases {
                let rargs: Vec<Value> = args.iter().map(|v| Value::from_interp(v).unwrap().realize(d)).collect();
                let mut b = native_backend(core, prog, &r, lanes);
                let got = b.prim(op, &rargs, &presk_types::ty::Ty::Nat, &Default::default()).unwrap_or_else(|e| panic!("{} natively: {}", op.name(), e.message));
                let transcendental = b.kernels.iter().any(|k| !presk_export_c::calls(k).is_empty());
                kernels.extend(b.kernels.iter().cloned());
                runs.push((d, lanes, op, rargs, raw(&got), transcendental));
            }
        }
    }
    (runs, kernels)
}

fn cases() -> &'static [(presk_core::PrimOp, Vec<IValue>)] {
    static CASES: std::sync::OnceLock<Vec<(presk_core::PrimOp, Vec<IValue>)>> = std::sync::OnceLock::new();
    CASES.get_or_init(primitive_cases)
}

#[test]
fn the_primitives_are_bit_identical_to_the_native_backend() {
    let started = Instant::now();
    let (prog, core) = prepare("tests/examples/mlp/mlp.psk", &[]);
    assert_eq!(cases().len(), 41);
    let (runs, kernels) = primitive_runs(&prog, &core, cases());
    let lib = Arc::new(CLib::build("primitives", &kernels));
    let mut compared = 0usize;
    let mut through_libm = 0usize;
    for (d, lanes, op, rargs, want, transcendental) in &runs {
        let r = Realization::uniform(*d);
        for partition in PARTITIONS {
            let mut b = c_backend(&core, &prog, &r, &lib, partition, *lanes);
            let got = b.prim(op, rargs, &presk_types::ty::Ty::Nat, &Default::default()).unwrap_or_else(|e| panic!("{} in C: {}", op.name(), e.message));
            assert!(raw(&got) == *want, "{} ({d}, lanes {lanes:?}, {partition:?}): the C moved a bit", op.name());
            compared += 1;
            through_libm += *transcendental as usize;
        }
    }
    eprintln!(
        "primitives: {compared} runs bit-identical to the native backend ({through_libm} through presk_libm.c); {} distinct kernels, {} calling a transcendental; compiled in {:.1} s, {:.1} s in all",
        lib.kernels,
        lib.transcendental,
        lib.compile_secs,
        started.elapsed().as_secs_f64()
    );
}

/// The SCoP mode's form (step 8, D85): every primitive case at point zero
/// (its default layout on one lane and one worker, no pack), in the three
/// types, each distinct kernel printed as a nest's file (`scop::nest_file`)
/// and compiled on its own with the export's flags and `-Werror`, then run
/// on the inputs the SCoP driver draws, against the native backend on the
/// same inputs: the same bits. It reaches forms the test set's nests do not,
/// argmax's selection of two values among them.
#[test]
fn the_primitives_as_scops_are_bit_identical_to_the_native_backend() {
    use presk_export_c::scop;
    let (prog, core) = prepare("tests/examples/mlp/mlp.psk", &[]);
    let zero = Layout { lanes: Some(1), split: Some(Split::None), pack: Some(false), ..Layout::default() };
    let mut kernels: Vec<Kernel> = Vec::new();
    for d in DTYPES {
        let r = Realization::uniform(d);
        for (op, args) in cases() {
            let rargs: Vec<Value> = args.iter().map(|v| Value::from_interp(v).unwrap().realize(d)).collect();
            let mut b = native_backend(&core, &prog, &r, Some(1));
            b.default_layout = zero.clone();
            b.prim(op, &rargs, &presk_types::ty::Ty::Nat, &Default::default()).unwrap_or_else(|e| panic!("{} natively: {}", op.name(), e.message));
            kernels.extend(b.kernels.iter().cloned());
        }
    }
    let mut seen = std::collections::BTreeSet::new();
    kernels.retain(|k| seen.insert(scop::scop_fn(k, "k").unwrap()));
    let dir = cexec::workdir("scop");
    std::fs::create_dir_all(dir.join("nests")).unwrap();
    let meta = scop::Meta { file: "presk-kernel/tests/conformance.rs".into(), entry: "the primitive cases".into(), ..Default::default() };
    let mut sources = vec![dir.join("presk_libm.c")];
    let mut selections = 0;
    for (i, k) in kernels.iter().enumerate() {
        let n = scop::Nest { name: format!("s{i}"), nest: k.name.clone(), shape: String::new(), launches: 1, kernel: k.clone() };
        let mut text = scop::nest_file(&meta, &n).unwrap();
        selections += text.contains("int c0") as usize;
        let args: Vec<String> = (0..=k.inputs.len()).map(|j| format!("b[{j}]")).collect();
        text.push_str(&format!("\nvoid presk_s{i}(void *const *b) {{\n    s{i}({});\n}}\n", args.join(", ")));
        let path = dir.join(format!("nests/s{i}.c"));
        std::fs::write(&path, text).unwrap();
        sources.push(path);
    }
    let so = cexec::So::build(&dir, &sources);
    let mut native = presk_cranelift::NativeExec::new().unwrap();
    native.threads = 1;
    let ptr = |b: &presk_rt::Buf| -> *mut u8 {
        match b {
            presk_rt::Buf::F64(v) => v.as_ptr() as *mut u8,
            presk_rt::Buf::F32(v) => v.as_ptr() as *mut u8,
            presk_rt::Buf::Bf16(v) => v.as_ptr() as *mut u8,
            presk_rt::Buf::I64(v) => v.as_ptr() as *mut u8,
            other => panic!("a {} buffer", other.dtype()),
        }
    };
    let bits = |b: &presk_rt::Buf| raw(&Value::Real(presk_rt::Tensor { axes: vec![], data: b.clone() }));
    let mut statics = 0;
    for (i, k) in kernels.iter().enumerate() {
        let inputs: Vec<presk_rt::Buf> = scop::draw_inputs(k, 0, i)
            .into_iter()
            .map(|d| match d {
                scop::Drawn::Real(v) => presk_rt::Buf::of(k.dtype, &v),
                scop::Drawn::Idx(v) => presk_rt::Buf::I64(v.into()),
            })
            .collect();
        let refs: Vec<&presk_rt::Buf> = inputs.iter().collect();
        let mut want = presk_kernel::exec::output_buf(k);
        presk_kernel::KernelExec::run(&mut native, i, k, &refs, &mut want).unwrap();
        let got = presk_kernel::exec::output_buf(k);
        let mut ptrs: Vec<*mut u8> = inputs.iter().map(ptr).collect();
        ptrs.push(ptr(&got));
        // SAFETY: `presk_s<i>` was printed above with this signature, and
        // its arrays are the kernel's buffers, of the kernel's lengths.
        let f = unsafe { std::mem::transmute::<*mut libc::c_void, unsafe extern "C" fn(*const *mut u8)>(so.sym(&format!("presk_s{i}"))) };
        unsafe { f(ptrs.as_ptr()) };
        assert!(bits(&got) == bits(&want), "{} ({}): the SCoP moved a bit", k.name, k.dtype);
        statics += scop::static_control(k).is_ok() as usize;
    }
    assert!(selections > 0, "no kernel selects two values");
    eprintln!("the primitives as SCoPs: {} distinct kernels at point zero bit-identical to the native backend, {statics} of them static control, {selections} selecting more than one value", kernels.len());
}

/// §3.5, rule 1 (D77, D78): the same check built as GCC builds by default,
/// in a GNU mode and without `-ffp-contract=off` (`-std=gnu11 -O2`), where
/// GCC contracts `a + b*c` into one fused multiply-add. On aarch64, which
/// has the instruction, bits must move, or the check does not test the
/// rule; elsewhere it reports what moved.
#[test]
fn contraction_moves_bits_in_gccs_gnu_mode() {
    let (prog, core) = prepare("tests/examples/mlp/mlp.psk", &[]);
    let (runs, kernels) = primitive_runs(&prog, &core, cases());
    let mut flags = vec!["-std=gnu11", "-O2"];
    flags.extend_from_slice(presk_export_c::WARNINGS);
    flags.push("-Werror");
    let lib = Arc::new(CLib::build_with("primitives-gnu", &kernels, &flags));
    let mut moved: Vec<String> = Vec::new();
    let mut moved_libm = 0;
    for (d, lanes, op, rargs, want, transcendental) in &runs {
        let r = Realization::uniform(*d);
        let mut b = c_backend(&core, &prog, &r, &lib, cexec::Partition::Whole, *lanes);
        let got = b.prim(op, rargs, &presk_types::ty::Ty::Nat, &Default::default()).unwrap_or_else(|e| panic!("{} in C: {}", op.name(), e.message));
        if raw(&got) != *want {
            moved.push(format!("{} ({d}, lanes {lanes:?})", op.name()));
            moved_libm += *transcendental as usize;
        }
    }
    eprintln!(
        "{} {}: built with -std=gnu11 -O2 and no -ffp-contract, {} of {} primitive runs move ({moved_libm} of them through presk_libm.c): {}",
        cexec::cc(),
        std::env::consts::ARCH,
        moved.len(),
        runs.len(),
        moved.join(", ")
    );
    if std::env::consts::ARCH == "aarch64" {
        assert!(!moved.is_empty(), "no bit moved in GCC's GNU mode: the check does not test the contraction rule");
    }
}

/// An operand as integer weights (step 7): its reals quantized to `bits`,
/// the scales constant along axis `over`, `absmax` in `int8` and `pow2` in
/// `int4`, so that both schemes reach the kernels.
fn quantized(v: &Value, bits: u8, over: usize) -> Value {
    let Value::Real(t) = v else { panic!("a real operand") };
    let t64 = presk_rt::Tensor { axes: t.axes.clone(), data: presk_rt::Buf::F64(t.data.to_f64().into()) };
    let scheme = if bits == 8 { presk_rt::QuantScheme::Absmax } else { presk_rt::QuantScheme::Pow2 };
    let q = presk_rt::QTensor::quantize(&t64, &presk_rt::Quant { bits, over: presk_rt::AxisRef::Pos(over as i64), scheme }).unwrap();
    Value::Real(q.to_tensor())
}

/// Whether a kernel loads integer weights.
fn reads_integers(k: &Kernel) -> bool {
    k.inputs.iter().any(|s| matches!(s.kind, presk_kernel::BufKind::Quant { .. }))
}

/// The layouts of the four nests, and the thread splits on every `every`-th
/// point of each space. With `quant`, one operand of each nest holds integer
/// weights of those bits (step 7): the weight of `linear`, the keys of
/// `scores` (which a pack copies), the values of `values`, the scores
/// `normalize` reads, each quantized along its contracted axis, or the
/// softmax's.
fn check_layouts(tag: &str, every: usize) {
    check_layouts_with(tag, 1, every, None)
}

/// The same on every `stride`-th point of each space, with the splits of
/// every `every`-th of those.
fn check_layouts_with(tag: &str, stride: usize, every: usize, quant: Option<u8>) {
    let started = Instant::now();
    let (prog, core) = prepare("tests/plan/contractions.psk", &[]);
    struct Run {
        nest: String,
        d: Dtype,
        fid: presk_core::ir::FuncId,
        fdims: presk_interp::eval::Dims,
        layout: Layout,
        args: Vec<Value>,
        want: Vec<u64>,
    }
    let mut runs: Vec<Run> = Vec::new();
    let mut kernels: Vec<Kernel> = Vec::new();
    let (mut points, mut splits, mut refused) = (0usize, 0usize, 0usize);
    let mut with_integers = 0usize;
    for (entry, var, qarg, qover) in [("linear", 2, 1, 0), ("scores", 2, 1, 3), ("values", 2, 1, 2), ("normalize", 1, 0, 3)] {
        let nest = format!("{entry}/%{var}");
        let func = core.funcs.iter().find(|f| f.name == entry).unwrap();
        let Rhs::Prim { op, .. } = &func.body.lets.iter().find(|l| l.var.0 == var).unwrap().rhs else { panic!("{nest} is a primitive") };
        let opts = RunOptions { entry: entry.into(), dims: vec![], seed: 0, bits: 128, digits: 17 };
        let (fid, fdims, inputs) = presk_interp::run::prepare_run(&core, &prog, &opts).unwrap();
        let operands: Vec<Operand> = inputs
            .iter()
            .map(|(_, v)| match Value::from_interp(v).unwrap() {
                Value::Real(t) => Operand { axes: t.axes.clone(), kind: presk_kernel::BufKind::Real },
                _ => unreachable!(),
            })
            .collect();
        for d in DTYPES {
            let r = Realization::uniform(d);
            let mut args: Vec<Value> = inputs.iter().map(|(_, v)| Value::from_interp(v).unwrap().realize(d)).collect();
            if let Some(bits) = quant {
                args[qarg] = quantized(&Value::from_interp(&inputs[qarg].1).unwrap(), bits, qover);
            }
            let kind = presk_kernel::plan::nest_kind(op, &operands, ReductionOrder::Sequential, d, d.arith());
            let space: Vec<Layout> = Layout::space(&kind).into_iter().step_by(stride).collect();
            let mut layouts: Vec<Layout> = space.clone();
            for l in space.iter().step_by(every) {
                for depth in 0..=2 {
                    for workers in [2, 4] {
                        layouts.push(Layout { split: Some(Split::At { depth, workers }), ..l.clone() });
                    }
                }
            }
            for layout in layouts {
                let mut b = native_backend(&core, &prog, &r, None);
                b.set_plan(Plan { realization: r.clone(), realization_ref: None, profile: None, layout: BTreeMap::from([(nest.clone(), layout.clone())]) });
                let p = presk_rt::Backend::prepare(&mut b, fid, &fdims, &r).unwrap();
                match presk_rt::Backend::call(&mut b, &p, args.clone()) {
                    Ok(out) => {
                        if layout.split.is_some() { splits += 1 } else { points += 1 }
                        with_integers += usize::from(b.kernels.iter().any(reads_integers));
                        kernels.extend(b.kernels.iter().cloned());
                        runs.push(Run { nest: nest.clone(), d, fid, fdims: fdims.clone(), layout, args: args.clone(), want: raw(&out) });
                    }
                    Err(e) => {
                        assert!(layout.split.is_some() && e.code == "E7002" && e.message.contains("condition 8"), "{nest} ({d}) under {layout}: {}", e.message);
                        refused += 1;
                    }
                }
            }
        }
    }
    let native_secs = started.elapsed().as_secs_f64();
    let lib = Arc::new(CLib::build(tag, &kernels));
    let c_started = Instant::now();
    for run in &runs {
        let r = Realization::uniform(run.d);
        for partition in PARTITIONS {
            let mut b = c_backend(&core, &prog, &r, &lib, partition, None);
            b.set_plan(Plan { realization: r.clone(), realization_ref: None, profile: None, layout: BTreeMap::from([(run.nest.clone(), run.layout.clone())]) });
            let p = presk_rt::Backend::prepare(&mut b, run.fid, &run.fdims, &r).unwrap();
            let out = presk_rt::Backend::call(&mut b, &p, run.args.clone()).unwrap_or_else(|e| panic!("{} ({}) under {} in C: {}", run.nest, run.d, run.layout, e.message));
            assert!(raw(&out) == run.want, "{} ({}, {partition:?}): the C moved a bit under {}", run.nest, run.d, run.layout);
        }
    }
    eprintln!(
        "layouts ({tag}): {points} points and {splits} splits bit-identical to the native backend, each under 3 partitions of its range ({refused} splits refused by condition 8); {} distinct kernels, {} calling a transcendental; native {native_secs:.1} s, compiled in {:.1} s, C runs {:.1} s",
        lib.kernels,
        lib.transcendental,
        lib.compile_secs,
        c_started.elapsed().as_secs_f64()
    );
    match quant {
        None => assert!(points > 1000 && splits > 500, "{points} points, {splits} splits"),
        Some(bits) => {
            assert_eq!(with_integers, points + splits, "int{bits}: a run whose kernels load no integer");
            assert!(points > 1000 / stride && splits > 500 / (stride * every), "{points} points, {splits} splits");
        }
    }
}

/// The layouts with an operand of integer weights (step 7), in `int8` and
/// `int4`, the pack of the keys included, in the three types: every third
/// point of each space, and the splits of every fifth of those.
#[test]
fn the_layouts_on_integer_weights_are_bit_identical_to_the_native_backend() {
    check_layouts_with("layouts-int8", 3, 5, Some(8));
    check_layouts_with("layouts-int4", 3, 5, Some(4));
}

/// Every point, and the splits of every point (`cargo test -p
/// presk-export-c -- --ignored`).
#[test]
#[ignore = "every point and every split on integer weights: nine times the default check"]
fn every_layout_on_integer_weights_is_bit_identical_to_the_native_backend() {
    check_layouts_with("layouts-int8-all", 1, 1, Some(8));
    check_layouts_with("layouts-int4-all", 1, 1, Some(4));
}

/// The primitives with every real operand of rank one or more as integer
/// weights (step 7): `int8` and `int4`, the scales constant along the first
/// axis or the last (so that lanes along the last axis read four scales, or
/// one), in the three types, scalar and at each type's natural width. The
/// host marks an integer input on every primitive (`tensor_prim_kinds`).
#[test]
fn the_primitives_on_integer_weights_are_bit_identical_to_the_native_backend() {
    let started = Instant::now();
    let (prog, core) = prepare("tests/examples/mlp/mlp.psk", &[]);
    // A primitive run as `PrimRun`, with what was quantized how.
    let mut runs: Vec<(PrimRun, String)> = Vec::new();
    let mut kernels: Vec<Kernel> = Vec::new();
    for d in DTYPES {
        let r = Realization::uniform(d);
        for lanes in [Some(1), None] {
            for (op, args) in cases() {
                for bits in [8u8, 4] {
                    for last in [false, true] {
                        let rargs: Vec<Value> = args
                            .iter()
                            .map(|v| match Value::from_interp(v).unwrap() {
                                Value::Real(t) if t.rank() >= 1 => quantized(&Value::Real(t.clone()), bits, if last { t.rank() - 1 } else { 0 }),
                                other => other.realize(d),
                            })
                            .collect();
                        if !rargs.iter().any(|v| matches!(v, Value::Real(t) if matches!(t.data, presk_rt::Buf::Quant(_)))) {
                            continue;
                        }
                        let mut b = native_backend(&core, &prog, &r, lanes);
                        let got = b.prim(op, &rargs, &presk_types::ty::Ty::Nat, &Default::default()).unwrap_or_else(|e| panic!("{} natively on int{bits}: {}", op.name(), e.message));
                        assert!(b.kernels.iter().any(reads_integers), "{} ({d}): no kernel reads the integers", op.name());
                        kernels.extend(b.kernels.iter().cloned());
                        runs.push(((d, lanes, op, rargs, raw(&got), false), format!("int{bits}/{}", if last { "last" } else { "first" })));
                    }
                }
            }
        }
    }
    let lib = Arc::new(CLib::build("primitives-int", &kernels));
    let mut compared = 0usize;
    for ((d, lanes, op, rargs, want, _), what) in &runs {
        let r = Realization::uniform(*d);
        for partition in PARTITIONS {
            let mut b = c_backend(&core, &prog, &r, &lib, partition, *lanes);
            let got = b.prim(op, rargs, &presk_types::ty::Ty::Nat, &Default::default()).unwrap_or_else(|e| panic!("{} in C: {}", op.name(), e.message));
            assert!(raw(&got) == *want, "{} ({d}, {what}, lanes {lanes:?}, {partition:?}): the C moved a bit", op.name());
            compared += 1;
        }
    }
    eprintln!("primitives on integer weights: {compared} runs ({} cases) bit-identical to the native backend; {} distinct kernels; {:.1} s", runs.len(), lib.kernels, started.elapsed().as_secs_f64());
}

/// Every layout, and the splits on every fifth point, as `layout.rs` states
/// them.
#[test]
fn every_layout_is_bit_identical_to_the_native_backend() {
    check_layouts("layouts", 5);
}

/// The splits on every point, not every fifth (`cargo test -p
/// presk-export-c -- --ignored`).
#[test]
#[ignore = "every split of every point: five times the splits of the default check"]
fn every_split_of_every_layout_is_bit_identical_to_the_native_backend() {
    check_layouts("layouts-all", 1);
}

/// Forms the lowering does not produce today, written by hand and run on
/// the native executor and in C: a vector store whose value reads the lanes
/// it overwrites, the integer statements, a saturating conversion, a
/// condition, an index output.
#[test]
fn hand_written_kernels_are_bit_identical_to_the_native_backend() {
    use presk_kernel::ir::{BinOp, BufKind, BufSpec, CmpOp, Expr, Stmt, Threads, UnOp};
    use presk_kernel::{Dtype, KernelExec};
    use presk_rt::Buf;
    let b = |e: Expr| Box::new(e);
    let spec = |dims: &[usize], kind: BufKind| BufSpec { dims: dims.to_vec(), kind, names: vec![None; dims.len()] };
    let at = |i: usize, j: i64| vec![Expr::Iv(i), Expr::I(j)];
    // out[i0, j] = 2·x[i0, j], then out[i0, 1..5] <v= out[i0, 0..4] + ½:
    // every lane read before any is written, as the native backend does.
    let shift = Kernel {
        name: "shift".into(),
        dtype: Dtype::F32,
        inputs: vec![spec(&[2, 8], BufKind::Real)],
        output: spec(&[2, 8], BufKind::Real),
        nloops: 2,
        nreal: 0,
        nint: 0,
        nvec: 0,
        lanes: 4,
        body: vec![Stmt::For {
            var: 0,
            bound: 2,
            body: vec![
                Stmt::For { var: 1, bound: 8, body: vec![Stmt::Store { buf: 1, index: vec![Expr::Iv(0), Expr::Iv(1)], value: Expr::Bin(BinOp::Mul, b(Expr::Load(0, vec![Expr::Iv(0), Expr::Iv(1)])), b(Expr::F(2.0))) }] },
                Stmt::VStore { buf: 1, index: at(0, 1), value: Expr::VBin(BinOp::Add, b(Expr::VLoad(1, at(0, 0))), b(Expr::VSplat(b(Expr::F(0.5))))) },
            ],
        }],
        threads: Threads::Default,
    };
    // n0 = int(3·x), i1 = n0 + 1, n1 = x > -¼ ? 2·i1 : y; if x < ½, out = n1 + 7.
    let x = || Expr::Load(0, vec![Expr::Iv(0)]);
    let ints = Kernel {
        name: "ints".into(),
        dtype: Dtype::F64,
        inputs: vec![spec(&[5], BufKind::Real), spec(&[5], BufKind::Idx)],
        output: spec(&[5], BufKind::Idx),
        nloops: 2,
        nreal: 0,
        nint: 2,
        nvec: 0,
        lanes: 1,
        body: vec![Stmt::For {
            var: 0,
            bound: 5,
            body: vec![
                Stmt::SetI { var: 0, value: Expr::ToI(b(Expr::Bin(BinOp::Mul, b(x()), b(Expr::F(3.0))))) },
                Stmt::SetL { var: 1, value: Expr::Bin(BinOp::IAdd, b(Expr::Si(0)), b(Expr::I(1))) },
                Stmt::SetI { var: 1, value: Expr::Select(b(Expr::Cmp(CmpOp::Gt, b(x()), b(Expr::F(-0.25)))), b(Expr::Bin(BinOp::IMul, b(Expr::Iv(1)), b(Expr::I(2)))), b(Expr::LoadI(1, vec![Expr::Iv(0)]))) },
                Stmt::If { cond: Expr::Cmp(CmpOp::Lt, b(x()), b(Expr::F(0.5))), then: vec![Stmt::StoreI { buf: 2, index: vec![Expr::Iv(0)], value: Expr::Bin(BinOp::IAdd, b(Expr::Si(1)), b(Expr::I(7))) }] },
            ],
        }],
        threads: Threads::Default,
    };
    // Constants at the edges: −∞, a negative constant, a double negation.
    let edges = Kernel {
        name: "edges".into(),
        dtype: Dtype::F32,
        inputs: vec![spec(&[5], BufKind::Real)],
        output: spec(&[5], BufKind::Real),
        nloops: 1,
        nreal: 0,
        nint: 0,
        nvec: 0,
        lanes: 1,
        body: vec![Stmt::For {
            var: 0,
            bound: 5,
            body: vec![Stmt::Store {
                buf: 1,
                index: vec![Expr::Iv(0)],
                value: Expr::Select(
                    b(Expr::Cmp(CmpOp::Le, b(x()), b(Expr::F(0.0)))),
                    b(Expr::F(f64::NEG_INFINITY)),
                    b(Expr::Bin(BinOp::Sub, b(Expr::F(1.0e-3)), b(Expr::Bin(BinOp::Mul, b(Expr::Un(UnOp::Neg, b(Expr::Un(UnOp::Neg, b(x()))))), b(Expr::F(-0.1)))))),
                ),
            }],
        }],
        threads: Threads::Default,
    };
    // −10³⁰⁰ saturates to the least `int64_t`; nothing overflows after it,
    // which C leaves undefined where Cranelift wraps.
    let xs = [0.3, -1e300, f64::NAN, 0.1, 1e18];
    let cases: Vec<(Kernel, Vec<Buf>)> = vec![
        (shift, vec![Buf::F32((0..16).map(|i| i as f32 * 0.37 - 2.0).collect())]),
        (ints, vec![Buf::F64(xs.to_vec().into()), Buf::I64(vec![11, 12, 13, 14, 15].into())]),
        (edges, vec![Buf::F32(vec![0.3, -2.0, f32::NAN, 1e-20, 3e38].into())]),
    ];
    let lib = Arc::new(CLib::build("hand", cases.iter().map(|(k, _)| k)));
    let mut native = presk_cranelift::NativeExec::new().unwrap();
    native.threads = 1;
    for (i, (k, inputs)) in cases.iter().enumerate() {
        let refs: Vec<&Buf> = inputs.iter().collect();
        let mut want = presk_kernel::exec::output_buf(k);
        native.run(i, k, &refs, &mut want).unwrap();
        for partition in PARTITIONS {
            let mut got = presk_kernel::exec::output_buf(k);
            CExec::new(lib.clone(), partition, WORKERS).run(i, k, &refs, &mut got).unwrap();
            let bits = |b: &Buf| raw(&Value::Real(presk_rt::Tensor { axes: vec![], data: b.clone() }));
            assert!(bits(&got) == bits(&want), "{} ({partition:?}): C {got:?}, native {want:?}", k.name);
        }
    }
}
