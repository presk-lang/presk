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

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Once;

use presk_core::ir::Rhs;
use presk_interp::RunOptions;
use presk_kernel::KernelBackend;
use presk_kernel::lower::Operand;
use presk_rt::layout::Split;
use presk_rt::{Dtype, Layout, Plan, Realization, ReductionOrder, Value};
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
                let mut b = if native { presk_cranelift::native_backend_with_threads(&core, &prog, &r, 0, 4).unwrap() } else { KernelBackend::new(&core, &prog, &r, 0) };
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
