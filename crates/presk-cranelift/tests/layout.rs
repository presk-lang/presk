//! Every point of a nest's layout space is exact (plan 0.7.0, steps 2 and 3;
//! statement L's first form): on the three contraction nests of attention —
//! a `dot` as `Linear` has it, the NT scores `ein`, the NN values `ein` — and
//! its softmax, every point of the space the reference defines (the plan
//! asked for 64 at random; the whole space takes seconds), each lowered to
//! kernels of its own, gives the bits of the default layout, on the kernel
//! interpreter and natively, in `f64`, `f32` and `bf16`. And a thread split
//! at depth 0, 1 or 2 over 2 or 4 workers, stated on a sample of the points,
//! gives them too, natively, wherever the kernel's loop at that depth is
//! parallel — and is refused (`E7002`, condition 8) wherever it is not. The
//! points differ in where running sums live, which memory the loops touch and
//! which worker computes an element; none may move an addition.
//!
//! Since 0.10.0 step 1 (statement T1's first part), the derivatives of `dot`
//! join the check: every nest `dot_vjp_l` and `dot_vjp_r` of the gradients of
//! the MLP, MiniGPT and the course GPT at small constants, at every point of
//! its space, and every gradient with its derivative nests planned at once.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Once;

use presk_core::PrimOp;
use presk_core::ir::Rhs;
use presk_interp::RunOptions;
use presk_kernel::exec::{output_buf, run_kernel};
use presk_kernel::lower::Operand;
use presk_kernel::plan::{launch_kind, lower_launch};
use presk_kernel::{Kernel, KernelBackend, KernelExec, Launch, LaunchOp};
use presk_rt::layout::Split;
use presk_rt::{Buf, Dtype, Layout, NestKind, Plan, Realization, ReductionOrder, Value};
use presk_syntax::diag::render_all;
use presk_syntax::{SourceMap, parse_file};
use presk_types::check;
use presk_types::elab::{self, ElabProgram, Options};

fn chdir_root() {
    static ONCE: Once = Once::new();
    let root: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap();
    ONCE.call_once(|| std::env::set_current_dir(root).unwrap());
}

fn prepare(rel: &str) -> (ElabProgram, presk_core::Program) {
    chdir_root();
    let mut sm = SourceMap::new();
    let id = sm.load(Path::new(rel)).unwrap();
    let (program, diags) = parse_file(&sm, id);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (prog, diags) = elab::elaborate(&mut sm, id, program, &Options::default());
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (typed, diags) = check::check(&sm, &prog);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (core, diags) = presk_core::lower(&prog, &typed);
    assert!(diags.is_empty(), "{}", render_all(&diags, &sm, false));
    presk_core::verify(&core).unwrap();
    (prog, core)
}

fn bits_of(v: &Value) -> Vec<u64> {
    match v {
        Value::Real(t) => t.data.to_f64().into_iter().map(|x| x.to_bits()).collect(),
        _ => panic!("the entries return one tensor"),
    }
}

#[test]
fn every_point_of_a_contractions_layout_space_is_exact() {
    let (prog, core) = prepare("tests/plan/contractions.psk");
    let mut checked = 0usize;
    let mut split_checked = 0usize;
    let mut split_refused = 0usize;
    for (entry, var) in [("linear", 2), ("scores", 2), ("values", 2), ("normalize", 1)] {
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
        for d in [Dtype::F64, Dtype::F32, Dtype::Bf16] {
            let r = Realization::uniform(d);
            let args: Vec<Value> = inputs.iter().map(|(_, v)| Value::from_interp(v).unwrap().realize(d)).collect();
            let kind = presk_kernel::plan::nest_kind(op, &operands, ReductionOrder::Sequential, d, d.arith());
            assert!(!matches!(kind, presk_rt::NestKind::Other), "{nest} is a contraction or a softmax");
            let space = Layout::space(&kind);
            let kernels = std::cell::RefCell::new(std::collections::BTreeSet::new());
            // Native runs have four workers, so a split takes 4 or 2 (condition 5).
            let attempt = |layout: Option<&Layout>, native: bool| -> presk_rt::Result<(Vec<u64>, Vec<presk_kernel::Kernel>)> {
                let mut b = if native { presk_cranelift::native_backend_with_threads(&core, &prog, &r, 4).unwrap() } else { KernelBackend::new(&core, &prog, &r) };
                if let Some(l) = layout {
                    b.set_plan(Plan { realization: r.clone(), realization_ref: None, profile: None, layout: BTreeMap::from([(nest.clone(), l.clone())]) });
                }
                let p = presk_rt::Backend::prepare(&mut b, fid, &fdims, &r).unwrap();
                let out = presk_rt::Backend::call(&mut b, &p, args.clone())?;
                Ok((bits_of(&out), b.kernels.clone()))
            };
            let run = |layout: Option<&Layout>, native: bool| -> Vec<u64> {
                let (bits, ks) = attempt(layout, native).unwrap_or_else(|e| panic!("{nest} ({d}) under {}: {}", layout.map(|l| l.to_string()).unwrap_or_default(), e.message));
                if !native && layout.is_some() {
                    kernels.borrow_mut().insert(ks.iter().map(|k| k.to_string()).collect::<String>());
                }
                bits
            };
            // Each executor against its own default: the two agree bit for bit
            // on a contraction, and not on an exponential, which the native
            // backend takes from libm and the kernel interpreter from Rust's
            // standard library (the conformance suite's caveat).
            let reference = [run(None, false), run(None, true)];
            if matches!(kind, presk_rt::NestKind::Contraction(_)) {
                assert_eq!(reference[0], reference[1], "{nest} ({d}): the two executors disagree at the default");
            }
            for l in &space {
                for native in [false, true] {
                    assert!(reference[native as usize] == run(Some(l), native), "{nest} ({d}, {}): the layout {l} moved a bit", if native { "native" } else { "kernel interpreter" });
                }
                checked += 1;
            }
            // Each point is its own kernel: a layout that was not applied would
            // pass the comparison above trivially.
            assert_eq!(kernels.borrow().len(), space.len(), "{nest} ({d}): {} points lowered to {} distinct kernels", space.len(), kernels.borrow().len());
            eprintln!("{nest} ({d}): {} points, {} distinct kernels, every bit equal", space.len(), kernels.borrow().len());
            // Thread splits, natively, on every fifth point.
            for l in space.iter().step_by(5) {
                for depth in 0..=2 {
                    for workers in [2, 4] {
                        let split = Layout { split: Some(Split::At { depth, workers }), ..l.clone() };
                        match attempt(Some(&split), true) {
                            Ok((bits, ks)) => {
                                assert!(bits == reference[1], "{nest} ({d}): {split} moved a bit");
                                assert!(ks.iter().any(|k| k.threads == presk_kernel::Threads::At { depth, workers }), "{nest} ({d}): {split} was not carried by the kernel");
                                split_checked += 1;
                            }
                            Err(e) => {
                                assert!(e.code == "E7002" && e.message.contains("condition 8"), "{nest} ({d}) under {split}: {}", e.message);
                                split_refused += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 1000, "{checked} points checked");
    eprintln!("splits: {split_checked} bit-identical, {split_refused} refused by condition 8");
    assert!(split_checked > 500 && split_refused > 0, "{split_checked} splits checked, {split_refused} refused");
}

/// A gradient entry's program: elaborated with its constants, lowered, and
/// differentiated, as `presk run` has it.
fn prepare_grad(rel: &str, consts: &[(&str, u64)]) -> (ElabProgram, presk_core::Program) {
    chdir_root();
    let mut sm = SourceMap::new();
    let id = sm.load(Path::new(rel)).unwrap();
    let (program, diags) = parse_file(&sm, id);
    assert!(!presk_syntax::diag::has_errors(&diags), "{}", render_all(&diags, &sm, false));
    let opts = Options { const_overrides: consts.iter().map(|(n, v)| (n.to_string(), *v)).collect() };
    let (prog, diags) = elab::elaborate(&mut sm, id, program, &opts);
    assert!(!presk_syntax::diag::has_errors(&diags), "{}", render_all(&diags, &sm, false));
    let (typed, diags) = check::check(&sm, &prog);
    assert!(!presk_syntax::diag::has_errors(&diags), "{}", render_all(&diags, &sm, false));
    let (mut core, diags) = presk_core::lower(&prog, &typed);
    assert!(!presk_syntax::diag::has_errors(&diags), "{}", render_all(&diags, &sm, false));
    presk_core::verify(&core).unwrap();
    let diags = presk_core::ad::transform(&mut core);
    assert!(!presk_syntax::diag::has_errors(&diags), "{}", render_all(&diags, &sm, false));
    (prog, core)
}

/// The gradients whose derivative nests are checked: the MLP's (`dot_vjp_r`
/// summed over one axis), MiniGPT's (over `batch` and `seq`, `T = 8`) and the
/// course GPT's at `T = 32`, where a k-block of 32 divides `seq`, the
/// innermost summed axis, and not their product alone.
/// A program, its entry, its constants and its dimensions.
type Entry = (&'static str, &'static str, &'static [(&'static str, u64)], &'static [(&'static str, u64)]);

const GRADIENTS: [Entry; 3] = [
    ("tests/examples/mlp/mlp.psk", "mse_grad", &[], &[("D", 4), ("B", 2)]),
    ("tests/plan/minigpt_grad.psk", "loss_grad", &[("V", 64), ("D", 16), ("H", 2), ("L", 2)], &[("B", 2), ("T", 8)]),
    ("experiments/gpt_grad.psk", "loss_grad", &[("V", 92), ("D", 16), ("H", 2), ("L", 1)], &[("B", 2), ("T", 32)]),
];

/// An entry's arguments as `presk run` synthesizes them, in `d`.
fn entry_args(prog: &ElabProgram, core: &presk_core::Program, entry: &str, dims: &[(&str, u64)], d: Dtype) -> (presk_core::FuncId, presk_interp::eval::Dims, Vec<Value>) {
    let opts = RunOptions { entry: entry.into(), dims: dims.iter().map(|(n, v)| (n.to_string(), *v)).collect(), seed: 0, bits: 128, digits: 17 };
    let (fid, fdims, inputs) = presk_interp::run::prepare_run(core, prog, &opts).unwrap();
    (fid, fdims, inputs.iter().map(|(_, v)| Value::from_interp(v).unwrap().realize(d)).collect())
}

/// Every launch of a derivative of `dot` an entry makes natively, at each
/// shape once, as `presk autotune` captures them.
fn derivative_launches(prog: &ElabProgram, core: &presk_core::Program, entry: &str, dims: &[(&str, u64)], d: Dtype) -> Vec<Launch> {
    let r = Realization::uniform(d);
    let (fid, fdims, args) = entry_args(prog, core, entry, dims, d);
    let mut b = presk_cranelift::native_backend_with_threads(core, prog, &r, 4).unwrap();
    b.capture = Some(Vec::new());
    let p = presk_rt::Backend::prepare(&mut b, fid, &fdims, &r).unwrap();
    presk_rt::Backend::call(&mut b, &p, args).unwrap();
    b.capture.take().unwrap().into_iter().filter(|l| matches!(&l.what, LaunchOp::Prim(PrimOp::DotVjpL | PrimOp::DotVjpR))).collect()
}

/// Values of every magnitude from 2⁻⁸ to 2⁸, both signs, drawn from a fixed
/// sequence: summed in another order, they round differently.
fn drawn(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let m = (s >> 11) as f64 / (1u64 << 53) as f64;
            let e = ((s >> 3) % 17) as i32 - 8;
            (if s & 1 == 0 { m } else { -m }) * 2f64.powi(e)
        })
        .collect()
}

/// A point's kernels run on `l`'s inputs: a pack's copy first, its output the
/// contraction's operand 1. `next` numbers the kernels for the native
/// executor, which compiles each index once.
fn run_point(exec: &mut dyn KernelExec, next: &mut usize, l: &Launch, kernels: &[Kernel]) -> Vec<u64> {
    let inputs: Vec<&Buf> = l.inputs.iter().collect();
    let out = match kernels {
        [k] => {
            let mut out = output_buf(k);
            exec.run(*next, k, &inputs, &mut out).unwrap();
            out
        }
        [pack, k] => {
            let mut panel = output_buf(pack);
            exec.run(*next, pack, &inputs[1..2], &mut panel).unwrap();
            let mut out = output_buf(k);
            exec.run(*next + 1, k, &[inputs[0], &panel], &mut out).unwrap();
            out
        }
        _ => panic!("a point lowers to one kernel or two"),
    };
    *next += kernels.len();
    out.to_f64().into_iter().map(|x| x.to_bits()).collect()
}

struct Interp;

impl KernelExec for Interp {
    fn name(&self) -> &'static str {
        "kernel"
    }
    fn run(&mut self, _: usize, k: &Kernel, inputs: &[&Buf], out: &mut Buf) -> presk_rt::Result<()> {
        run_kernel(k, inputs, out);
        Ok(())
    }
}

#[test]
fn every_point_of_a_derivative_nest_is_exact() {
    let mut exec = presk_cranelift::NativeExec::new().unwrap();
    exec.threads = 4;
    let mut next = 0usize;
    let (mut nests, mut checked, mut split_checked, mut split_refused) = (0usize, 0usize, 0usize, 0usize);
    let mut shapes = std::collections::BTreeSet::new();
    for (file, entry, consts, dims) in GRADIENTS {
        let (prog, core) = prepare_grad(file, consts);
        for d in [Dtype::F64, Dtype::F32, Dtype::Bf16] {
            let launches = derivative_launches(&prog, &core, entry, dims, d);
            assert!(launches.len() >= 4, "{entry} ({file}, {d}): {} derivative launches", launches.len());
            for (n, mut l) in launches.into_iter().enumerate() {
                let what = format!("{} {} ({entry}, {d})", l.nest, l.operands.iter().map(|o| presk_kernel::lower::show(&o.axes)).collect::<Vec<_>>().join(" "));
                for (i, b) in l.inputs.iter_mut().enumerate() {
                    *b = Buf::of(d, &drawn(b.to_f64().len(), (n * 2 + i) as u64 + 1));
                }
                let NestKind::Contraction(kind) = launch_kind(&l) else { panic!("{what} is a contraction nest") };
                shapes.insert((l.nest.clone(), kind.i, kind.j, kind.k, kind.k_inner));
                let space = Layout::space(&NestKind::Contraction(kind));
                let default = lower_launch(&l, &Layout::default(), Some(4)).unwrap();
                let reference = run_point(&mut Interp, &mut next, &l, &default);
                assert_eq!(reference, run_point(&mut exec, &mut next, &l, &default), "{what}: the two executors disagree at the default");
                assert!(reference.iter().filter(|b| f64::from_bits(**b) != 0.0).count() * 2 > reference.len(), "{what}: the output is mostly zeros");
                let mut kernels = std::collections::BTreeSet::new();
                for p in &space {
                    let ks = lower_launch(&l, p, Some(4)).unwrap_or_else(|m| panic!("{what} under {p}: {m}"));
                    kernels.insert(ks.iter().map(|k| k.to_string()).collect::<String>());
                    assert!(reference == run_point(&mut Interp, &mut next, &l, &ks), "{what}: the layout {p} moved a bit on the kernel interpreter");
                    assert!(reference == run_point(&mut exec, &mut next, &l, &ks), "{what}: the layout {p} moved a bit natively");
                    checked += 1;
                }
                // Each point is its own kernel.
                assert_eq!(kernels.len(), space.len(), "{what}: {} points lowered to {} distinct kernels", space.len(), kernels.len());
                for p in space.iter().step_by(7) {
                    for depth in 0..=2 {
                        for workers in [2, 4] {
                            let split = Layout { split: Some(Split::At { depth, workers }), ..p.clone() };
                            match lower_launch(&l, &split, Some(4)) {
                                Ok(ks) => {
                                    assert!(reference == run_point(&mut exec, &mut next, &l, &ks), "{what}: {split} moved a bit");
                                    split_checked += 1;
                                }
                                Err(m) => {
                                    assert!(m.contains("condition 8"), "{what} under {split}: {m}");
                                    split_refused += 1;
                                }
                            }
                        }
                    }
                }
                nests += 1;
            }
        }
    }
    eprintln!("{nests} derivative launches ({} nest shapes), {checked} points, every bit equal; splits: {split_checked} bit-identical, {split_refused} refused by condition 8", shapes.len());
    // The course GPT's dot_vjp_r summed over (batch, seq) with k-blocks on seq.
    // (In `Linear.apply.bwd` since the gradient keeps its residuals, step 4.)
    assert!(shapes.iter().any(|(n, _, _, k, ki)| n.starts_with("Linear.apply.bwd") && *k == 64 && *ki == 32), "{shapes:?}");
    assert!(checked > 4000 && split_checked > 2000 && split_refused > 0, "{checked} points, {split_checked} splits checked, {split_refused} refused");
}

/// The gradients with every derivative nest under a stated layout at once,
/// through the host (the pack's two launches, the cache), natively and on the
/// kernel backend: the bits of the default layout. Each round gives each
/// nest the point of its space at a stride, so the rounds differ.
#[test]
fn a_gradient_with_its_derivatives_planned_is_exact() {
    let mut rounds = 0usize;
    for (file, entry, consts, dims) in GRADIENTS {
        let (prog, core) = prepare_grad(file, consts);
        for d in [Dtype::F64, Dtype::F32, Dtype::Bf16] {
            let r = Realization::uniform(d);
            let launches = derivative_launches(&prog, &core, entry, dims, d);
            let (fid, fdims, args) = entry_args(&prog, &core, entry, dims, d);
            let run = |plan: Option<Plan>, native: bool| -> Vec<u64> {
                let mut b = if native { presk_cranelift::native_backend_with_threads(&core, &prog, &r, 4).unwrap() } else { KernelBackend::new(&core, &prog, &r) };
                if let Some(p) = plan {
                    presk_kernel::plan::check_plan(&p, &core).unwrap();
                    b.set_plan(p);
                }
                let p = presk_rt::Backend::prepare(&mut b, fid, &fdims, &r).unwrap();
                let out = presk_rt::Backend::call(&mut b, &p, args.clone()).unwrap_or_else(|e| panic!("{entry} ({d}): {}", e.message));
                let mut bits = Vec::new();
                out.map_leaves(&mut |leaf| {
                    if let Value::Real(t) = leaf {
                        bits.extend(t.data.to_f64().into_iter().map(|x| x.to_bits()));
                    }
                    leaf.clone()
                });
                bits
            };
            let reference = [run(None, false), run(None, true)];
            for round in 0..4 {
                // One layout per nest, which every shape it is launched at takes.
                let mut layout = BTreeMap::new();
                for l in &launches {
                    if layout.contains_key(&l.nest) {
                        continue;
                    }
                    let fits = |p: &Layout| launches.iter().filter(|m| m.nest == l.nest).all(|m| p.check(&launch_kind(m)).is_ok());
                    let space: Vec<Layout> = Layout::space(&launch_kind(l)).into_iter().filter(|p| fits(p)).collect();
                    layout.insert(l.nest.clone(), space[(round * 37 + 11) % space.len()].clone());
                }
                let plan = Plan { realization: r.clone(), realization_ref: None, profile: None, layout };
                for native in [false, true] {
                    assert!(reference[native as usize] == run(Some(plan.clone()), native), "{entry} ({d}, {}): the plan {} moved a bit", if native { "native" } else { "kernel backend" }, plan.to_json());
                }
                rounds += 1;
            }
        }
    }
    assert_eq!(rounds, 36);
}

// ----- the epilogue (plan 0.10.0 step 7) ----------------------------------------
//
// A product whose value one `add` alone reads can add the `add`'s operand in
// its own kernel, once each output element's sum is done. At every point of
// its layout space, the kernel with its epilogue gives the bits of the
// product's kernel followed by the `add`'s: in `f64`, `f32` and `bf16`, on
// the kernel interpreter and natively, under thread splits, on the products
// of `tests/plan/epilogue.psk` and on the derivative nests of the gradients
// given an operand to add; and whole programs under plans that state it give
// the bits of the default.

/// A point's kernels run on `l`'s inputs and, with an epilogue, its
/// operand: a pack's copy first, its output the contraction's operand 1.
fn run_with(exec: &mut dyn KernelExec, next: &mut usize, l: &Launch, kernels: &[Kernel], extra: Option<&Buf>) -> Buf {
    let mut inputs: Vec<&Buf> = l.inputs.iter().collect();
    inputs.extend(extra);
    let out = match kernels {
        [k] => {
            let mut out = output_buf(k);
            exec.run(*next, k, &inputs[..k.inputs.len()], &mut out).unwrap();
            out
        }
        [pack, k] => {
            let mut panel = output_buf(pack);
            exec.run(*next, pack, &inputs[1..2], &mut panel).unwrap();
            let mut out = output_buf(k);
            let mut ins = vec![inputs[0], &panel];
            ins.extend(inputs[2..k.inputs.len()].iter().copied());
            exec.run(*next + 1, k, &ins, &mut out).unwrap();
            out
        }
        _ => panic!("a point lowers to one kernel or two"),
    };
    *next += kernels.len();
    out
}

fn bits(b: &Buf) -> Vec<u64> {
    b.to_f64().into_iter().map(|x| x.to_bits()).collect()
}

/// The reference of a launch with an epilogue: the product's kernel at its
/// default layout, then the `add`'s kernel on its output and the operand.
fn product_then_add(exec: &mut dyn KernelExec, next: &mut usize, l: &Launch) -> Vec<u64> {
    let at = l.epilogue.as_ref().expect("the launch has an epilogue");
    let product = run_with(exec, next, l, &lower_launch(l, &Layout::default(), Some(4)).unwrap(), None);
    let p = Operand { axes: at.epilogue.out.clone(), kind: presk_kernel::BufKind::Real };
    let (ops, ins) = if at.epilogue.first { (vec![p, at.operand.clone()], vec![&product, &at.input]) } else { (vec![at.operand.clone(), p], vec![&at.input, &product]) };
    let add = presk_kernel::lower::lower(&PrimOp::Add, &ops, l.dtype).unwrap().unwrap().kernel;
    let mut out = output_buf(&add);
    exec.run(*next, &add, &ins, &mut out).unwrap();
    *next += 1;
    bits(&out)
}

/// Every point of a launch's space with the epilogue, against the product
/// and the `add` apart, on both executors; and a split on every seventh
/// point natively. Returns (points, splits checked, splits refused).
fn check_epilogue_points(exec: &mut presk_cranelift::NativeExec, next: &mut usize, l: &Launch, what: &str) -> (usize, usize, usize) {
    let reference = product_then_add(&mut Interp, next, l);
    assert_eq!(reference, product_then_add(exec, next, l), "{what}: the two executors disagree on the product and the add");
    assert!(reference.iter().filter(|b| f64::from_bits(**b) != 0.0).count() * 2 > reference.len(), "{what}: the output is mostly zeros");
    let extra = Some(&l.epilogue.as_ref().unwrap().input);
    let space = Layout::space(&launch_kind(l));
    let mut kernels = std::collections::BTreeSet::new();
    for p in &space {
        let p = Layout { epilogue: Some(true), ..p.clone() };
        let ks = lower_launch(l, &p, Some(4)).unwrap_or_else(|m| panic!("{what} under {p}: {m}"));
        kernels.insert(ks.iter().map(|k| k.to_string()).collect::<String>());
        assert!(reference == bits(&run_with(&mut Interp, next, l, &ks, extra)), "{what}: {p} moved a bit on the kernel interpreter");
        assert!(reference == bits(&run_with(exec, next, l, &ks, extra)), "{what}: {p} moved a bit natively");
    }
    assert_eq!(kernels.len(), space.len(), "{what}: {} points lowered to {} distinct kernels", space.len(), kernels.len());
    let (mut ok, mut refused) = (0, 0);
    for p in space.iter().step_by(7) {
        for depth in 0..=2 {
            for workers in [2, 4] {
                let split = Layout { split: Some(Split::At { depth, workers }), epilogue: Some(true), ..p.clone() };
                match lower_launch(l, &split, Some(4)) {
                    Ok(ks) => {
                        assert!(reference == bits(&run_with(exec, next, l, &ks, extra)), "{what}: {split} moved a bit");
                        ok += 1;
                    }
                    Err(m) => {
                        assert!(m.contains("condition 8"), "{what} under {split}: {m}");
                        refused += 1;
                    }
                }
            }
        }
    }
    (space.len(), ok, refused)
}

/// An entry's launches captured natively with no plan, each product noting
/// the epilogue it can take.
fn captured(prog: &ElabProgram, core: &presk_core::Program, entry: &str, dims: &[(&str, u64)], r: &Realization) -> Vec<Launch> {
    let (fid, fdims, args) = entry_args(prog, core, entry, dims, r.default);
    let mut b = presk_cranelift::native_backend_with_threads(core, prog, r, 4).unwrap();
    b.set_plan(Plan::from_realization(r.clone()));
    b.capture = Some(Vec::new());
    let p = presk_rt::Backend::prepare(&mut b, fid, &fdims, r).unwrap();
    presk_rt::Backend::call(&mut b, &p, args).unwrap();
    b.capture.take().unwrap()
}

#[test]
fn every_point_with_its_epilogue_is_exact() {
    let (prog, core) = prepare("tests/plan/epilogue.psk");
    let mut exec = presk_cranelift::NativeExec::new().unwrap();
    exec.threads = 4;
    let mut next = 0usize;
    let (mut points, mut split_ok, mut split_refused) = (0usize, 0usize, 0usize);
    // The three types. A realization accumulating `f32` storage in `f64`,
    // which the kernels do not honour, is refused since 0.10.0 step 8
    // (`a_declared_accumulation_type_the_kernels_do_not_honour_is_refused`).
    let realizations = [Realization::uniform(Dtype::F64), Realization::uniform(Dtype::F32), Realization::uniform(Dtype::Bf16)];
    for entry in ["linear", "bias_first", "scores", "values", "rows"] {
        for (n, r) in realizations.iter().enumerate() {
            let d = r.default;
            let mut launches: Vec<Launch> = captured(&prog, &core, entry, &[], r).into_iter().filter(|l| l.epilogue.is_some()).collect();
            assert_eq!(launches.len(), 1, "{entry} ({}): one product with an epilogue", r.label());
            let mut l = launches.remove(0);
            for (i, b) in l.inputs.iter_mut().enumerate() {
                *b = Buf::of(d, &drawn(b.to_f64().len(), (i + 1) as u64));
            }
            let at = l.epilogue.as_mut().unwrap();
            at.input = Buf::of(d, &drawn(at.input.to_f64().len(), 7));
            assert_eq!(at.epilogue.first, entry != "bias_first");
            let what = format!("{} ({entry}, {})", l.nest, r.label());
            let (p, ok, refused) = check_epilogue_points(&mut exec, &mut next, &l, &what);
            eprintln!("{what}: {p} points with the epilogue, every bit equal to the product and the add");
            points += p;
            split_ok += ok;
            split_refused += refused;
        }
    }
    eprintln!("{points} points; splits: {split_ok} bit-identical, {split_refused} refused by condition 8");
    // 2 674 points in the three types (with the `f64` accumulation of `f32`,
    // refused since 0.10.0 step 8, there were 3 000 and more).
    assert!(points > 2500 && split_ok > 1000 && split_refused > 0, "{points} points, {split_ok} splits, {split_refused} refused");
}

/// The derivative nests of the gradients, each given an operand to add in
/// its epilogue (none of them has one in these programs): the NT
/// `dot_vjp_l` packed, and `dot_vjp_r` summed over several axes, whose last
/// block of `K` is every block variable at its last value. The operand runs
/// along `J`, or is of the output's shape, the order of the `add` alternating.
#[test]
fn every_point_of_a_derivative_nest_with_an_epilogue_is_exact() {
    let mut exec = presk_cranelift::NativeExec::new().unwrap();
    exec.threads = 4;
    let mut next = 0usize;
    let (mut nests, mut points) = (0usize, 0usize);
    let mut several = false;
    for (file, entry, consts, dims) in GRADIENTS {
        let (prog, core) = prepare_grad(file, consts);
        for d in [Dtype::F64, Dtype::F32, Dtype::Bf16] {
            for (n, mut l) in derivative_launches(&prog, &core, entry, dims, d).into_iter().enumerate() {
                for (i, b) in l.inputs.iter_mut().enumerate() {
                    *b = Buf::of(d, &drawn(b.to_f64().len(), (n * 3 + i) as u64 + 1));
                }
                let LaunchOp::Prim(op) = &l.what else { unreachable!() };
                let c = presk_kernel::lower::Contraction::of(op, &l.operands, ReductionOrder::Sequential).unwrap();
                several |= c.ks.len() > 1;
                let axes = if n % 2 == 0 { c.out[c.out.len() - 1..].to_vec() } else { c.out.clone() };
                let len: usize = axes.iter().map(|a| a.len).product();
                l.epilogue = Some(presk_kernel::EpilogueAt {
                    operand: Operand { axes, kind: presk_kernel::BufKind::Real },
                    input: Buf::of(d, &drawn(len, n as u64 + 99)),
                    epilogue: presk_kernel::lower::Epilogue { first: n % 3 != 1, out: c.out.clone() },
                });
                let what = format!("{} ({entry}, {d})", l.nest);
                points += check_epilogue_points(&mut exec, &mut next, &l, &what).0;
                nests += 1;
            }
        }
    }
    eprintln!("{nests} derivative launches, {points} points with an epilogue, every bit equal to the product and the add");
    assert!(several && points > 4000, "{points} points");
}

/// The entries whose products are planned with their epilogues: a forward
/// pass and a gradient of each of the MLP, MiniGPT and the course GPT.
const PROGRAMS: [Entry; 6] = [
    ("tests/examples/mlp/mlp.psk", "mse", &[], &[("D", 4), ("B", 2)]),
    ("tests/examples/mlp/mlp.psk", "mse_grad", &[], &[("D", 4), ("B", 2)]),
    ("tests/examples/minigpt/train_minigpt.psk", "loss", &[("V", 64), ("D", 16), ("H", 2), ("L", 2)], &[("B", 2), ("T", 8)]),
    ("tests/plan/minigpt_grad.psk", "loss_grad", &[("V", 64), ("D", 16), ("H", 2), ("L", 2)], &[("B", 2), ("T", 8)]),
    ("corpus/nlp/05_gpt.psk", "GPT.apply", &[("V", 92), ("D", 16), ("H", 2), ("L", 1)], &[("B", 2), ("T", 32)]),
    ("experiments/gpt_grad.psk", "loss_grad", &[("V", 92), ("D", 16), ("H", 2), ("L", 1)], &[("B", 2), ("T", 32)]),
];

/// Every real leaf's bits.
fn leaf_bits(v: &Value) -> Vec<u64> {
    let mut bits = Vec::new();
    v.map_leaves(&mut |leaf| {
        if let Value::Real(t) = leaf {
            bits.extend(t.data.to_f64().into_iter().map(|x| x.to_bits()));
        }
        leaf.clone()
    });
    bits
}

/// An entry run under a plan (checked first), or none; its output's bits,
/// and the kernels it launched.
fn run_planned(prog: &ElabProgram, core: &presk_core::Program, entry: &str, dims: &[(&str, u64)], r: &Realization, plan: Option<&Plan>, native: bool) -> presk_rt::Result<(Vec<u64>, Vec<Kernel>)> {
    let (fid, fdims, args) = entry_args(prog, core, entry, dims, r.default);
    let mut b = if native { presk_cranelift::native_backend_with_threads(core, prog, r, 4).unwrap() } else { KernelBackend::new(core, prog, r) };
    if let Some(p) = plan {
        presk_kernel::plan::check_plan(p, core)?;
        b.set_plan(p.clone());
    }
    let p = presk_rt::Backend::prepare(&mut b, fid, &fdims, r).unwrap();
    let out = presk_rt::Backend::call(&mut b, &p, args)?;
    Ok((leaf_bits(&out), b.kernels.clone()))
}

/// The plan of a round: every product that can take an epilogue at every
/// shape it is launched at takes it, at the default layout in round 0 and
/// at a point of its space every shape fits in the others; every other
/// contraction nest at such a point too.
fn epilogue_plan(launches: &[Launch], r: &Realization, round: usize) -> Plan {
    let mut layout = BTreeMap::new();
    for l in launches {
        if layout.contains_key(&l.nest) || !matches!(launch_kind(l), NestKind::Contraction(_)) {
            continue;
        }
        let same: Vec<&Launch> = launches.iter().filter(|m| m.nest == l.nest).collect();
        let epilogue = same.iter().all(|m| m.epilogue.is_some()).then_some(true);
        let fits = |p: &Layout| same.iter().all(|m| p.check(&launch_kind(m)).is_ok());
        let space: Vec<Layout> = Layout::space(&launch_kind(l)).into_iter().filter(|p| fits(p)).collect();
        let point = if round == 0 { Layout::default() } else { space[(round * 37 + 11) % space.len()].clone() };
        layout.insert(l.nest.clone(), Layout { epilogue, ..point });
    }
    Plan { realization: r.clone(), realization_ref: None, profile: None, layout }
}

#[test]
fn a_program_with_its_epilogues_planned_is_exact() {
    let mut rounds = 0usize;
    for (file, entry, consts, dims) in PROGRAMS {
        let (prog, core) = prepare_grad(file, consts);
        for d in [Dtype::F64, Dtype::F32, Dtype::Bf16] {
            let r = Realization::uniform(d);
            let launches = captured(&prog, &core, entry, dims, &r);
            assert!(launches.iter().any(|l| l.epilogue.is_some()), "{entry} ({file}, {d}): no product takes an epilogue");
            let reference = [false, true].map(|native| run_planned(&prog, &core, entry, dims, &r, None, native).unwrap().0);
            for round in 0..4 {
                let plan = epilogue_plan(&launches, &r, round);
                for native in [false, true] {
                    let (bits, kernels) = run_planned(&prog, &core, entry, dims, &r, Some(&plan), native).unwrap_or_else(|e| panic!("{entry} ({d}): {}", e.message));
                    assert!(kernels.iter().any(|k| k.name.ends_with("+add")), "{entry} ({d}): no epilogue was launched");
                    assert!(reference[native as usize] == bits, "{entry} ({d}, {}): the plan {} moved a bit", if native { "native" } else { "kernel backend" }, plan.to_json());
                }
                rounds += 1;
            }
        }
    }
    assert_eq!(rounds, 72);
}

/// Mixed realizations of MiniGPT's gradient. Both sites of `Linear.apply`'s
/// product and bias typed apart alike, `bf16` in an `f32` program: the
/// epilogue holds, its operands converted at the kernel's edge. The `add`'s
/// site alone typed apart: refused (`E7002`), by the plan's check and by
/// the backend. (The product's accumulation in `f64`, which held, is
/// refused by the backend since 0.10.0 step 8: the kernels do not honour
/// it.)
#[test]
fn an_epilogue_under_a_mixed_realization() {
    let (file, entry, consts, dims) = PROGRAMS[3];
    let (prog, core) = prepare_grad(file, consts);
    let base = Realization::uniform(Dtype::F32);
    let mut both = base.clone();
    both.storage.insert("Linear.apply/%3".into(), Dtype::Bf16);
    both.storage.insert("Linear.apply/%5".into(), Dtype::Bf16);
    for r in [both] {
        let launches = captured(&prog, &core, entry, dims, &r);
        assert!(launches.iter().any(|l| l.nest == "Linear.apply/%3" && l.epilogue.is_some()), "{}: no epilogue noted", r.to_json());
        for native in [false, true] {
            let reference = run_planned(&prog, &core, entry, dims, &r, None, native).unwrap().0;
            for round in 0..2 {
                let plan = epilogue_plan(&launches, &r, round);
                let (bits, kernels) = run_planned(&prog, &core, entry, dims, &r, Some(&plan), native).unwrap_or_else(|e| panic!("{}: {}", r.to_json(), e.message));
                assert!(kernels.iter().any(|k| k.name == "dot+add"), "{}: no epilogue was launched", r.to_json());
                assert!(reference == bits, "{} (native: {native}): the plan {} moved a bit", r.to_json(), plan.to_json());
            }
        }
    }
    let mut apart = base.clone();
    apart.storage.insert("Linear.apply/%5".into(), Dtype::Bf16);
    let launches = captured(&prog, &core, entry, dims, &apart);
    assert!(launches.iter().all(|l| l.nest != "Linear.apply/%3" || l.epilogue.is_none()), "an epilogue noted across two types");
    let plan = Plan { realization: apart.clone(), realization_ref: None, profile: None, layout: BTreeMap::from([("Linear.apply/%3".to_string(), Layout { epilogue: Some(true), ..Layout::default() })]) };
    let e = presk_kernel::plan::check_plan(&plan, &core).unwrap_err();
    assert!(e.code == "E7002" && e.message.contains("typed bf16"), "{}", e.message);
    let (fid, fdims, args) = entry_args(&prog, &core, entry, dims, Dtype::F32);
    let mut b = presk_cranelift::native_backend_with_threads(&core, &prog, &apart, 4).unwrap();
    b.set_plan(plan);
    let p = presk_rt::Backend::prepare(&mut b, fid, &fdims, &apart).unwrap();
    let e = presk_rt::Backend::call(&mut b, &p, args).unwrap_err();
    assert!(e.code == "E7002" && e.message.contains("typed bf16"), "{}", e.message);
}

/// A declared accumulation type other than the storage's arithmetic type
/// (0.10.0 step 8, D97's finding): `bf16` accumulated in `bf16`, under
/// which condition 7 admitted a spill whose bits differ from the default's
/// since the kernels accumulate in `f32`, and `f32` in `f64`. Refused
/// (`E7002`) by the kernel and native backends' `prepare`, before anything
/// runs; the storage's own arithmetic type, `bf16` in `f32`, is accepted,
/// and the reference interpreter, which reads no realization, runs either.
#[test]
fn a_declared_accumulation_type_the_kernels_do_not_honour_is_refused() {
    let (prog, core) = prepare("tests/plan/contractions.psk");
    let accum = |storage: Dtype, sum: Dtype| {
        let mut r = Realization::uniform(storage);
        r.accum.insert("linear/%2".into(), sum);
        r
    };
    let (fid, fdims, _) = entry_args(&prog, &core, "linear", &[], Dtype::F32);
    for (r, refused) in [(accum(Dtype::Bf16, Dtype::Bf16), true), (accum(Dtype::F32, Dtype::F64), true), (accum(Dtype::Bf16, Dtype::F32), false), (accum(Dtype::F64, Dtype::F64), false)] {
        let mut kernel = presk_kernel::KernelBackend::new(&core, &prog, &r);
        let mut native = presk_cranelift::native_backend_with_threads(&core, &prog, &r, 2).unwrap();
        for b in [&mut kernel as &mut dyn presk_rt::Backend, &mut native] {
            match presk_rt::Backend::prepare(b, fid, &fdims, &r) {
                Err(e) => {
                    assert!(refused, "{}: {}", r.to_json(), e.message);
                    assert!(e.code == "E7002" && e.message.contains("honoured by no kernel"), "{}", e.message);
                }
                Ok(_) => assert!(!refused, "{}: not refused", r.to_json()),
            }
        }
        let mut interp = presk_rt::InterpBackend::new(&core, &prog, 64);
        assert!(presk_rt::Backend::prepare(&mut interp, fid, &fdims, &r).is_ok());
    }
}

/// Where a product cannot take an epilogue, a plan stating one is refused
/// (`E7002`), saying why: its value is the function's result, or the nest
/// is not a product.
#[test]
fn an_epilogue_is_refused_where_it_does_not_apply() {
    let (_, core) = prepare("tests/plan/contractions.psk");
    let r = Realization::uniform(Dtype::F32);
    let plan = |nest: &str| Plan { realization: r.clone(), realization_ref: None, profile: None, layout: BTreeMap::from([(nest.to_string(), Layout { epilogue: Some(true), ..Layout::default() })]) };
    let e = presk_kernel::plan::check_plan(&plan("linear/%2"), &core).unwrap_err();
    assert!(e.code == "E7002" && e.message.contains("is the result"), "{}", e.message);
    let (_, core) = prepare("tests/plan/epilogue.psk");
    let e = presk_kernel::plan::check_plan(&plan("linear/%4"), &core).unwrap_err();
    assert!(e.code == "E7002" && e.message.contains("decision of a contraction nest"), "{}", e.message);
    assert!(presk_kernel::plan::check_plan(&plan("linear/%3"), &core).is_ok());
}
