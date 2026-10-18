//! `presk memory` (plan 0.15.0 step 1, D148): M1's parts that hold of every
//! recording, on the entries of `tests/plan/entries.txt` and the course GPT's
//! forward pass, gradient and optimizer step. The placed straight-line host
//! and the C program's arena are `make c-conformance`'s (`presk-cli`'s
//! `export_c` test).

use std::path::{Path, PathBuf};

use presk_core::ad::Remat;
use presk_driver::Collect;
use presk_kernel::record::Op;
use presk_rt::Plan;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

struct Case {
    file: String,
    entry: String,
    dims: Vec<String>,
    consts: Vec<String>,
}

fn cases() -> Vec<Case> {
    let text = std::fs::read_to_string(root().join("tests/plan/entries.txt")).unwrap();
    let mut out: Vec<Case> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut p = l.splitn(3, '|');
            let (file, entry, args) = (p.next().unwrap(), p.next().unwrap(), p.next().unwrap_or(""));
            let words: Vec<&str> = args.split_whitespace().collect();
            let (mut dims, mut consts) = (Vec::new(), Vec::new());
            for w in words.chunks(2) {
                match w[0] {
                    "--dim" => dims.push(w[1].to_string()),
                    "--const" => consts.push(w[1].to_string()),
                    other => panic!("{other} in tests/plan/entries.txt"),
                }
            }
            Case { file: file.into(), entry: entry.into(), dims, consts }
        })
        .collect();
    let gpt = |file: &str, entry: &str| Case { file: file.into(), entry: entry.into(), dims: vec!["B=2".into(), "T=16".into()], consts: vec!["D=64".into(), "H=2".into(), "L=2".into()] };
    out.push(gpt("corpus/nlp/05_gpt.psk", "GPT.apply"));
    out.push(gpt("corpus/nlp/05_gpt.psk", "loss"));
    out.push(gpt("experiments/gpt_grad.psk", "loss_grad"));
    out.push(gpt("corpus/nlp/05_gpt.psk", "adamw.step"));
    out
}

fn report(c: &Case, realize: &str, remat: Remat) -> (presk_driver::memory::Memory, presk_driver::layout::Hosted) {
    let file = root().join(&c.file);
    let mut r = Collect::default();
    let o = presk_driver::memory::Options { file: &file, entry: &c.entry, dims: &c.dims, consts: &c.consts, realize, plan: None, seed: 0, remat };
    let m = presk_driver::memory::memory(&o, &mut r).unwrap_or_else(|s| panic!("{}: exit {} {:?}", c.entry, s.0, r.items));
    // The same call recorded again, for its launch log.
    let (_, prog, mut core) = presk_driver::front::core(&file, &c.consts, false, &mut r).unwrap();
    let plan = Plan::from_realization(presk_rt::Realization::parse(realize).unwrap());
    let _ = presk_driver::ad::differentiate(&mut core, remat, plan.realization.groups_visible());
    let dims = presk_driver::front::parse_dims(&c.dims, &mut r).unwrap();
    let h = presk_driver::layout::kernels_with_host(&core, &prog, &c.entry, dims, &plan, 0).unwrap();
    (m, h)
}

/// M1 (iv): the recording's call paths are the launch log's, launch by
/// launch; the peak is the largest live set, at its operation; the calls'
/// shares at the peak and the entry's own sum to it, and a call's share
/// never exceeds its parent's or its own peak; every buffer an operation
/// reads is live there.
#[test]
fn lifetimes_hold_on_every_entry() {
    for c in cases() {
        for (realize, remat) in [("f64", Remat::Store), ("f32", Remat::Recompute), ("bf16", Remat::Store)] {
            let what = format!("{} in {realize} under {remat:?}", c.entry);
            let (m, h) = report(&c, realize, remat);
            let rec = h.counted().unwrap_or_else(|| panic!("{what}: not recorded"));
            let l = &m.lifetimes;
            let launches: Vec<usize> = rec.ops.iter().enumerate().filter(|(_, o)| matches!(o, Op::Launch { .. })).map(|(k, _)| k).collect();
            assert_eq!(launches.len(), h.launched.len(), "{what}: a launch per logged launch");
            for (k, logged) in launches.iter().zip(&h.launched) {
                assert_eq!(l.paths[*k], logged.calls, "{what}: operation {k}'s calls");
            }
            assert_eq!(l.live.iter().max().copied().unwrap_or(0), l.peak, "{what}: the peak");
            assert_eq!(l.live.get(l.at).copied().unwrap_or(0), l.peak, "{what}: the peak's operation");
            let top: usize = m.calls.iter().filter(|x| x.path.len() == 1).map(|x| x.share).sum();
            assert_eq!(top + m.own_share(), l.peak, "{what}: the shares");
            for x in &m.calls {
                assert!(x.share <= x.own, "{what}: {:?}'s share above its own peak", x.path);
                if x.path.len() > 1 {
                    let parent = m.calls.iter().find(|p| p.path == x.path[..x.path.len() - 1]).unwrap_or_else(|| panic!("{what}: {:?} has no parent", x.path));
                    assert!(x.share <= parent.share && x.own <= parent.own, "{what}: {:?} above its parent", x.path);
                }
            }
            for (k, op) in rec.ops.iter().enumerate() {
                let reads: Vec<usize> = match op {
                    Op::Launch { inputs, .. } => inputs.iter().map(|v| v.buf).collect(),
                    Op::Copy { src, .. } | Op::Read { src, .. } | Op::Sample { src, .. } => vec![src.buf],
                    _ => Vec::new(),
                };
                for b in reads {
                    let life = l.lives[b];
                    assert!(life.input || life.first.is_some_and(|f| f <= k) && life.last >= k, "{what}: buffer {b} read at {k} outside its life");
                }
            }
            if let Ok(a) = &m.arena {
                assert!(l.peak <= a.beside(), "{what}: a peak of {} above the arena's {}", l.peak, a.beside());
            }
            assert!(m.placed >= l.peak, "{what}: placed in {} bytes, under the peak {}", m.placed, l.peak);
        }
    }
}

/// The course GPT's numbers of the plan's §2 (D147), which the scratch test
/// that found them measured.
#[test]
fn the_course_gpt_as_the_plan_measured_it() {
    let want = [("GPT.apply", Remat::Store, 149_504, 377_984), ("loss_grad", Remat::Store, 5_117_072, 9_425_216), ("loss_grad", Remat::Recompute, 4_562_944, 8_298_816), ("adamw.step", Remat::Store, 2_709_544, 3_615_616)];
    let all = cases();
    for (entry, remat, peak, arena) in want {
        let c = all.iter().rev().find(|c| c.entry == entry).unwrap();
        let (m, _) = report(c, "f64", remat);
        assert_eq!(m.lifetimes.peak, peak, "{entry} under {remat:?}: the peak");
        assert_eq!(m.arena.as_ref().map(|a| a.beside()), Ok(arena), "{entry} under {remat:?}: the C arena beside the inputs");
    }
}


/// M2 (plan 0.15.0 step 2, D149): the course GPT's training step at `D =
/// 64`, under `store` and `recompute`, in `f64` and `f32`. The parameters
/// are the parameter tree's elements at their stored type; the gradient is
/// their elements in the step's type, the moments twice that; the terms of
/// the step's peak sum to it, the largest of its calls' phases; and
/// `recompute` keeps less for the backward than `store`.
#[test]
fn the_training_step_s_terms() {
    let file = root().join("corpus/nlp/05_gpt.psk");
    let consts: Vec<String> = vec!["D=64".into(), "H=2".into(), "L=2".into()];
    let dims: Vec<String> = vec!["B=2".into(), "T=16".into()];
    for (realize, bytes) in [("f64", 8), ("f32", 4)] {
        let mut r = Collect::default();
        let o = presk_driver::memory::StepOptions { file: &file, dims: &dims, consts: &consts, realize, plan: None, seed: 0 };
        let steps = presk_driver::memory::step(&o, &mut r).unwrap_or_else(|s| panic!("exit {} {:?}", s.0, r.items));
        assert_eq!(steps.len(), 2);
        for s in &steps {
            let what = format!("{realize} under {:?}", s.remat);
            let names: Vec<&str> = s.calls.iter().map(|c| c.entry.as_str()).collect();
            assert_eq!(names, ["loss.value_vjp", "adamw.lr", "adamw.step"], "{what}");
            let (count, at_rest) = s.calls[0].parameters();
            assert_eq!(count, 112_896, "{what}: the course GPT's parameters at D = 64");
            assert_eq!((s.params, at_rest), (count * bytes, count * bytes), "{what}: the parameters");
            assert_eq!(s.gradients, count * bytes, "{what}: the gradient, the parameters' shapes in the step's type");
            assert_eq!(s.moments, 2 * s.gradients, "{what}: the moments");
            assert_eq!(s.terms.iter().map(|t| t.bytes).sum::<usize>(), s.peak, "{what}: the terms sum to the peak");
            assert_eq!(s.phases.iter().max().copied(), Some(s.peak), "{what}: the peak is the largest phase");
            assert_eq!(s.phases[0], s.params + s.others + s.moments + s.calls[0].lifetimes.peak, "{what}: the gradient's phase");
            assert_eq!(s.phases[2], s.calls[2].lifetimes.inputs + s.calls[2].lifetimes.peak, "{what}: the update's phase");
            assert!(s.reverse_at.is_some(), "{what}: the seed cotangent is read");
        }
        assert!(steps[1].kept < steps[0].kept, "{realize}: recompute keeps {} for the backward, store {}", steps[1].kept, steps[0].kept);
        if realize == "f64" {
            // The plan's §2 (D147).
            assert_eq!((steps[0].calls[0].lifetimes.peak, steps[0].peak), (5_117_080, 7_827_104));
            assert_eq!((steps[1].calls[0].lifetimes.peak, steps[1].peak), (4_586_496, 7_296_520));
        }
    }
}

/// M3 (plan 0.15.0 step 3, D150): on every program of 0.14.0's network test
/// (`make plan-check`'s entries, the coverage models at their tiny
/// constants, the course GPT's `loss`), in `f32` as that test draws them,
/// the polynomials evaluated at three assignments of the dimensions besides
/// the document's equal the peak, the inputs and the outputs a recording at
/// those dimensions counts, to the byte; and at the document's dimensions,
/// its own count.
#[test]
fn the_polynomials_count_what_a_recording_counts() {
    let mut all = cases();
    all.truncate(11);
    let tiny = |file: &str, entry: &str, consts: &[&str]| Case { file: file.into(), entry: entry.into(), dims: vec!["B=2".into(), "T=16".into()], consts: consts.iter().map(|c| c.to_string()).collect() };
    all.push(tiny("coverage/bert.psk", "BertSquad.apply", &["V=100", "D=32", "H=2", "L=2"]));
    all.push(tiny("coverage/llama.psk", "Llama.apply", &["V=100", "D=32", "H=2", "G=2", "FF=64", "L=2"]));
    all.push(tiny("corpus/nlp/05_gpt.psk", "loss", &["D=64", "H=2", "L=2"]));
    let mut checked = 0;
    for c in &all {
        let file = root().join(&c.file);
        let mut r = Collect::default();
        let o = presk_driver::memory::Options { file: &file, entry: &c.entry, dims: &c.dims, consts: &c.consts, realize: "f32", plan: None, seed: 0, remat: Remat::Store };
        let m = presk_driver::memory::memory(&o, &mut r).unwrap_or_else(|s| panic!("{}: exit {}", c.entry, s.0));
        let s = presk_driver::memory::symbolic(&o, &m, &mut r).unwrap_or_else(|e| panic!("{}: no polynomial (exit {}): {:?}", c.entry, e.0, r.items));
        let v0: Vec<u64> = c.dims.iter().map(|d| d.split_once('=').unwrap().1.parse().unwrap()).collect();
        let count = |x: &[u64]| -> Option<presk_driver::memory::Memory> {
            let dims: Vec<String> = s.names.iter().zip(x).map(|(n, v)| format!("{n}={v}")).collect();
            presk_driver::memory::memory(&presk_driver::memory::Options { dims: &dims, ..o }, &mut Collect::default()).ok()
        };
        let same = |x: &[u64], m: &presk_driver::memory::Memory| {
            let r = |n: usize| presk_types::dim::Rational::from_integer(n.into());
            assert_eq!(s.peak_at(x).0, r(m.lifetimes.peak), "{} at {x:?}: the peak", c.entry);
            assert_eq!(s.inputs.eval(x), r(m.lifetimes.inputs), "{} at {x:?}: the inputs", c.entry);
            assert_eq!(s.outputs.eval(x), r(m.lifetimes.outputs), "{} at {x:?}: the outputs", c.entry);
        };
        same(&v0, &m);
        // Three assignments off the document's: every dimension times 5,
        // times 7, and in turn times 9 and as given; another multiple where
        // the entry does not record at one.
        let mut points = 0;
        for (j, mul) in [5u64, 7, 9, 11, 13, 6, 10].iter().enumerate() {
            let x: Vec<u64> = v0.iter().enumerate().map(|(a, v)| if j == 2 && a % 2 == 1 { *v } else { v * mul }).collect();
            if let Some(mx) = count(&x) {
                same(&x, &mx);
                points += 1;
            }
            if points == 3 {
                break;
            }
        }
        assert_eq!(points, 3, "{}: three assignments that record", c.entry);
        checked += 1;
    }
    assert_eq!(checked, 14);
}
