//! A realization plan's layout, applied to the kernels of a program (0.7.0
//! step 1): which kind of nest a launch is, which layout this version lowers
//! it with, and the checks a plan passes against the program before anything
//! runs.
//!
//! The layout space of the language reference is larger than what this
//! version lowers. Every decision a plan states is checked against the nest
//! (its kind, its shapes, the reference's conditions); then, if the
//! lowering cannot realize it, it is refused with `E7002` rather than
//! ignored, since a plan whose layout is silently not applied would be timed
//! and reported as if it had been. Since 0.7.0 step 3 every decision of the
//! reference's table has a lowering: a contraction's form, register tile,
//! k-block, loop order, lanes and pack, a softmax's passes, and a thread split
//! at a depth, whose conditions 5 and 8 the kernel backend checks against the
//! executor and the kernel once the kernel exists.

use presk_core::ir::Rhs;
use presk_core::{PrimOp, Program};
use presk_rt::{Layout, NestKind, Plan, ReductionOrder, RtError};

use crate::ir::Dtype;
use crate::host::{Launch, LaunchOp};
use crate::ir::{Kernel, Threads};
use crate::lower::{Contraction, Operand};
use presk_rt::layout::Split;

/// The kind of the nest a primitive is launched as, at these operands.
/// `accum` is the accumulation type the realization states for the nest.
pub fn nest_kind(op: &PrimOp, ins: &[Operand], order: ReductionOrder, dtype: Dtype, accum: Dtype) -> NestKind {
    if let Some(c) = Contraction::of(op, ins, order) {
        return NestKind::Contraction(c.kind(dtype, accum));
    }
    match op {
        PrimOp::Softmax { .. } => NestKind::Softmax { narrow: dtype != dtype.arith() },
        _ => NestKind::Other,
    }
}

/// The layout a nest is lowered with: `stated`, the plan's, checked against
/// the nest (conditions 3, 4, 7 and 9); over `default`, the backend's own;
/// every decision left resolved to the reference's default. A backend default
/// asks for "as wide as the nest allows" and is not held to condition 4.
pub fn lowered(stated: &Layout, default: &Layout, kind: &NestKind) -> Result<Layout, String> {
    stated.check(kind)?;
    Ok(stated.over(default).resolve(kind))
}

/// The thread split a kernel carries: `split` checked against the kernel
/// (condition 8: its loop at that depth is parallel, with two iterations or
/// more) and against `workers`, the executor's `p_M` when it has one
/// (condition 5); or, `lenient`, applied where the kernel admits it and the
/// executor's default elsewhere (a backend's default layout).
pub fn threads_of(split: Option<Split>, k: &Kernel, workers: Option<usize>, lenient: bool) -> Result<Threads, String> {
    match split {
        None => Ok(Threads::Default),
        Some(Split::None) => Ok(Threads::One),
        Some(Split::At { depth, workers: p }) => {
            let fits = matches!(k.parallel_at(depth), Some(n) if n >= 2);
            if lenient {
                return Ok(if fits { Threads::At { depth, workers: p } } else { Threads::Default });
            }
            if let Some(pm) = workers
                && p != pm
                && p != pm / 2
            {
                return Err(format!("{p} workers, where this machine has {pm}: a split takes {pm} or {} (condition 5)", pm / 2));
            }
            match k.parallel_at(depth) {
                Some(n) if n >= 2 => Ok(Threads::At { depth, workers: p }),
                Some(n) => Err(format!("the loop at depth {depth} of `{}` has {n} iteration, and a split needs two (condition 8)", k.name)),
                None => Err(format!("the loop at depth {depth} of `{}` is not parallel: its iterations do not write disjoint elements of the output (condition 8)", k.name)),
            }
        }
    }
}

/// The kind of a captured launch.
pub fn launch_kind(l: &Launch) -> NestKind {
    match &l.what {
        LaunchOp::Prim(op) => nest_kind(op, &l.operands, l.order, l.dtype, l.accum),
        LaunchOp::Fused { .. } => NestKind::Other,
    }
}

/// The kernels of a captured launch under a layout (0.7.0 step 5): the
/// layout checked against the nest, resolved, lowered, and the split set —
/// one kernel, or two for a pack, the copy first. `workers`: the machine's
/// `p_M`, against which a split is checked (condition 5).
pub fn lower_launch(l: &Launch, layout: &Layout, workers: Option<usize>) -> Result<Vec<Kernel>, String> {
    let kind = launch_kind(l);
    layout.check(&kind)?;
    let eff = layout.resolve(&kind);
    let mut kernels = match &l.what {
        LaunchOp::Fused { steps, out } => vec![crate::lower::fused(steps, &l.operands, out, l.dtype)?.kernel],
        LaunchOp::Prim(op) => match (eff.pack, Contraction::of(op, &l.operands, l.order)) {
            (Some(true), Some(c)) => {
                let pack = crate::lower::pack_panel(&c, l.dtype, &l.operands[1])?.kernel;
                let panel = Operand { axes: c.panel_axes(&l.operands[1]), kind: crate::ir::BufKind::Real };
                let body = crate::lower::emit_contraction(&c.packed(), l.dtype, &[l.operands[0].clone(), panel], &eff)?.kernel;
                vec![pack, body]
            }
            _ => vec![crate::lower::lower_planned(op, &l.operands, l.dtype, l.order, &eff).unwrap_or_else(|| Err(format!("`{}`: no kernel", op.name())))?.kernel],
        },
    };
    // Integer weights are marked as the host marks them.
    let main = kernels.len() - 1;
    for (spec, b) in kernels[main].inputs.iter_mut().zip(&l.inputs) {
        if let presk_rt::Buf::Quant(q) = b {
            spec.kind = crate::ir::BufKind::Quant { bits: q.bits, over: q.over };
        }
    }
    kernels[main].threads = threads_of(eff.split, &kernels[main], workers, false)?;
    Ok(kernels)
}

/// The kernels of a captured launch under one layout and each of `splits`
/// (0.8.0 step 1): element `i` is what [`lower_launch`] gives for the layout
/// with `splits[i]`. No lowering reads the split, and the layout's checks do
/// not either: it only sets the main kernel's `threads`. So the layout is
/// lowered once, and each split is that lowering with its threads set and
/// checked as `lower_launch` checks them (conditions 5 and 8).
pub fn lower_launch_splits(l: &Launch, layout: &Layout, splits: &[Split], workers: Option<usize>) -> Vec<Result<Vec<Kernel>, String>> {
    let once = lower_launch(l, &Layout { split: Some(Split::None), ..layout.clone() }, workers);
    splits.iter().map(|s| once.as_ref().map_err(|e| e.clone()).and_then(|ks| split_as(ks, *s, workers))).collect()
}

/// The kernels of a launch lowered with no split, given `split` as
/// [`lower_launch`] gives it: set on the main kernel, checked against it and
/// against `workers`.
pub fn split_as(kernels: &[Kernel], split: Split, workers: Option<usize>) -> Result<Vec<Kernel>, String> {
    let mut kernels = kernels.to_vec();
    let main = kernels.len() - 1;
    kernels[main].threads = threads_of(Some(split), &kernels[main], workers, false)?;
    Ok(kernels)
}

/// Checks a plan against the program it is run on, before anything runs:
/// every nest it names exists and is a nest (condition 1: the root of a
/// fusion group, or a primitive of its own), and states only decisions its
/// primitive can have. The shapes are not known yet, so the reference's
/// other conditions wait for the launch.
pub fn check_plan(plan: &Plan, core: &Program) -> Result<(), RtError> {
    let fail = |nest: &str, m: String| RtError { code: "E7002", message: format!("the plan's layout of `{nest}`: {m}"), span: None };
    for (nest, layout) in &plan.layout {
        let (fname, var) = nest.rsplit_once("/%").ok_or_else(|| fail(nest, "not a nest name, `<function>/%<let>`".into()))?;
        let var: u32 = var.parse().map_err(|_| fail(nest, format!("`%{var}` is not a binding")))?;
        let func = core.funcs.iter().find(|f| f.name == fname).ok_or_else(|| fail(nest, format!("the program has no function `{fname}`")))?;
        let l = func.body.lets.iter().find(|l| l.var.0 == var).ok_or_else(|| fail(nest, format!("`{fname}` has no binding %{var}")))?;
        let groups = presk_realize::plan(func);
        if groups.interior.contains(&l.var) {
            let root = groups.groups.values().find(|g| g.members.contains(&l.var)).map(|g| format!("%{}", g.root.0)).unwrap_or_default();
            return Err(fail(nest, format!("%{var} is inside the fusion group whose root, {root}, is the nest")));
        }
        let op = match &l.rhs {
            _ if groups.groups.contains_key(&l.var) => None,
            Rhs::Prim { op, .. } => Some(op),
            other => {
                let what = match other {
                    Rhs::Call { .. } => "a call",
                    Rhs::Fold { .. } => "a fold",
                    Rhs::Tuple(_) | Rhs::Record(_) => "a tuple or record",
                    Rhs::Proj(..) | Rhs::Field(..) => "a projection",
                    Rhs::Lit(_) => "a literal",
                    _ => "not a primitive",
                };
                return Err(fail(nest, format!("%{var} of `{fname}` is {what}, and only a primitive or a fusion group is a nest")));
            }
        };
        let contraction = matches!(op, Some(PrimOp::Dot | PrimOp::Matmul | PrimOp::Ein { .. }));
        let softmax = matches!(op, Some(PrimOp::Softmax { .. }));
        let named = layout.stated();
        if !contraction && let Some(d) = named.iter().find(|d| ["form", "tile", "kblock", "order", "pack"].contains(d)) {
            return Err(fail(nest, format!("`{d}` is a decision of a contraction nest, and %{var} is {}", describe(op))));
        }
        if !softmax && layout.passes.is_some() {
            return Err(fail(nest, format!("`passes` is a decision of a softmax nest, and %{var} is {}", describe(op))));
        }
    }
    Ok(())
}

/// What a nest of the program is, for a listing: `dot`, `ein`, `softmax`, a
/// fusion group; `?` for a name that is not a nest (which
/// [`check_plan`] reports).
pub fn nest_label(core: &Program, nest: &str) -> String {
    let found = nest.rsplit_once("/%").and_then(|(fname, var)| {
        let var: u32 = var.parse().ok()?;
        let func = core.funcs.iter().find(|f| f.name == fname)?;
        let l = func.body.lets.iter().find(|l| l.var.0 == var)?;
        let groups = presk_realize::plan(func);
        Some(match (&l.rhs, groups.groups.get(&l.var)) {
            (_, Some(g)) => format!("group of {}", g.steps.len()),
            (Rhs::Prim { op, .. }, None) => op.name().split([' ', '[']).next().unwrap_or("").to_string(),
            _ => "?".into(),
        })
    });
    found.unwrap_or_else(|| "?".into())
}

fn describe(op: Option<&PrimOp>) -> String {
    match op {
        None => "a fusion group".into(),
        Some(op) => format!("`{}`", op.name()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use presk_rt::Axis;

    fn t(axes: &[(&str, usize)]) -> Operand {
        Operand { axes: axes.iter().map(|(n, l)| Axis { name: Some(n.to_string()), len: *l }).collect(), kind: crate::ir::BufKind::Real }
    }

    fn names(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn the_attention_contractions_are_recognized() {
        let seq = ReductionOrder::Sequential;
        let q = t(&[("batch", 2), ("heads", 6), ("seq", 16), ("hd", 64)]);
        let k = t(&[("batch", 2), ("heads", 6), ("seq2", 16), ("hd", 64)]);
        let scores = PrimOp::Ein { operands: vec![names(&["batch", "heads", "seq", "hd"]), names(&["batch", "heads", "seq2", "hd"])], output: names(&["batch", "heads", "seq", "seq2"]) };
        let NestKind::Contraction(c) = nest_kind(&scores, &[q, k], seq, Dtype::F32, Dtype::F32) else { panic!("the scores ein is a contraction") };
        assert_eq!((c.dot, c.i, c.j, c.k, c.nt, c.width), (false, 16, 16, 64, true, 4));
        let a = t(&[("batch", 2), ("heads", 6), ("seq", 16), ("seq2", 16)]);
        let v = t(&[("batch", 2), ("heads", 6), ("seq2", 16), ("hd", 64)]);
        let values = PrimOp::Ein { operands: vec![names(&["batch", "heads", "seq", "seq2"]), names(&["batch", "heads", "seq2", "hd"])], output: names(&["batch", "heads", "seq", "hd"]) };
        let NestKind::Contraction(c) = nest_kind(&values, &[a.clone(), v.clone()], seq, Dtype::Bf16, Dtype::F32) else { panic!("the values ein is a contraction") };
        assert_eq!((c.i, c.j, c.k, c.nt, c.accum_is_storage, c.width), (16, 64, 16, false, false, 1));
        // The output in another order is not the pattern.
        let swapped = PrimOp::Ein { operands: vec![names(&["batch", "heads", "seq", "seq2"]), names(&["batch", "heads", "seq2", "hd"])], output: names(&["batch", "heads", "hd", "seq"]) };
        assert_eq!(nest_kind(&swapped, &[a, v], seq, Dtype::F32, Dtype::F32), NestKind::Other);
        // A dot with a rank-two right operand is one; under an interleaved order it is not.
        let (x, w) = (t(&[("batch", 2), ("seq", 16), ("model", 384)]), t(&[("i", 384), ("o", 92)]));
        let NestKind::Contraction(c) = nest_kind(&PrimOp::Dot, &[x.clone(), w.clone()], seq, Dtype::F64, Dtype::F64) else { panic!("a dot is a contraction") };
        assert_eq!((c.dot, c.i, c.j, c.k, c.width), (true, 16, 92, 384, 2));
        assert_eq!(nest_kind(&PrimOp::Dot, &[x, w], ReductionOrder::Interleaved { partials: 4 }, Dtype::F64, Dtype::F64), NestKind::Other);
    }

    #[test]
    fn a_stated_layout_is_checked_then_resolved() {
        use presk_rt::layout::{Contraction as C, Form, KBlock, Split};
        let wide = NestKind::Contraction(C { dot: true, i: 16, j: 384, k: 384, nt: false, accum_is_storage: true, width: 4 });
        let json = |s: &str| Layout::from_json(&serde_json::from_str(s).unwrap()).unwrap();
        let native = Layout::default();
        let interp = Layout { lanes: Some(1), ..Layout::default() };
        assert_eq!(lowered(&Layout::default(), &native, &wide).unwrap().lanes, Some(4));
        assert_eq!(lowered(&Layout::default(), &interp, &wide).unwrap().lanes, Some(1));
        assert_eq!(lowered(&json(r#"{ "lanes": 4 }"#), &interp, &wide).unwrap().lanes, Some(4));
        let l = lowered(&json(r#"{ "tile": [4, 16], "kblock": 32, "order": "KJI", "split": { "depth": 1, "workers": 10 } }"#), &native, &wide).unwrap();
        assert_eq!((l.form, l.tile, l.kblock, l.lanes, l.split), (Some(Form::Tiled), Some((4, 16)), Some(KBlock::Width(32)), Some(4), Some(Split::At { depth: 1, workers: 10 })));
        // The NT scores ein: lanes with the pack only.
        let ein = NestKind::Contraction(C { dot: false, i: 16, j: 16, k: 64, nt: true, accum_is_storage: true, width: 4 });
        assert_eq!(lowered(&json(r#"{ "tile": [1, 8] }"#), &native, &ein).unwrap().lanes, Some(1));
        assert_eq!(lowered(&json(r#"{ "tile": [1, 8], "pack": true }"#), &native, &ein).unwrap().lanes, Some(4));
        assert!(lowered(&json(r#"{ "tile": [1, 8], "lanes": 4 }"#), &native, &ein).unwrap_err().contains("condition 4"));
        // The bf16 dot of odd width: one accumulator per element, and no row-wise form.
        let narrow = NestKind::Contraction(C { dot: true, i: 16, j: 92, k: 384, nt: false, accum_is_storage: false, width: 1 });
        let l = lowered(&Layout::default(), &native, &narrow).unwrap();
        assert_eq!((l.form, l.tile, l.lanes), (Some(Form::Tiled), Some((1, 1)), Some(1)));
        assert!(lowered(&json(r#"{ "form": "row-wise" }"#), &native, &narrow).unwrap_err().contains("condition 7"));
        assert!(lowered(&json(r#"{ "passes": 2 }"#), &native, &NestKind::Softmax { narrow: true }).unwrap_err().contains("condition 7"));
        assert_eq!(lowered(&json(r#"{ "passes": 2 }"#), &native, &NestKind::Softmax { narrow: false }).unwrap().passes, Some(2));
    }

}
