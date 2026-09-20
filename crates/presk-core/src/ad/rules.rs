//! Vector-Jacobian products of the primitives. Given the forward arguments
//! `args` (with types), the result `y` (with type) and its cotangent `ct`,
//! emit the cotangent of each differentiable argument.

use presk_syntax::Span;
use presk_types::ty::Ty;

use super::{Emit, differentiable};
use crate::ir::{Lit, PrimOp, Rhs, VarId};

type Contribs = Vec<(usize, VarId)>;

/// Returns `(argument index, cotangent)` pairs, or a reason when the
/// primitive has no adjoint.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prim_vjp(e: &mut Emit, op: &PrimOp, args: &[VarId], tys: &[Ty], y: VarId, y_ty: &Ty, ct: VarId, span: Span) -> Result<Contribs, String> {
    let real = |e: &mut Emit, n: u64| -> VarId {
        let k = e.emit(Ty::Nat, Rhs::Lit(Lit::Nat(n)), span);
        e.prim(Ty::scalar_real(), PrimOp::Real, vec![k], span)
    };
    // `x` binary with broadcasting: the adjoint is reduced to the operand's shape.
    let sum_to = |e: &mut Emit, x: VarId, i: usize| -> VarId { e.prim(tys[i].clone(), PrimOp::SumTo, vec![x, args[i]], span) };
    let mul = |e: &mut Emit, a: VarId, b: VarId, ty: &Ty| -> VarId { e.prim(ty.clone(), PrimOp::Mul, vec![a, b], span) };
    let div = |e: &mut Emit, a: VarId, b: VarId, ty: &Ty| -> VarId { e.prim(ty.clone(), PrimOp::Div, vec![a, b], span) };
    let neg = |e: &mut Emit, a: VarId, ty: &Ty| -> VarId { e.prim(ty.clone(), PrimOp::Neg, vec![a], span) };
    let yt = y_ty.clone();
    let both = |e: &mut Emit, ca: VarId, cb: VarId| -> Contribs {
        let mut out = Vec::new();
        if differentiable(&tys[0]) {
            out.push((0, sum_to(e, ca, 0)));
        }
        if differentiable(&tys[1]) {
            out.push((1, sum_to(e, cb, 1)));
        }
        out
    };
    Ok(match op {
        PrimOp::Add => both(e, ct, ct),
        PrimOp::Sub => {
            let n = neg(e, ct, &yt);
            both(e, ct, n)
        }
        PrimOp::Mul => {
            let ca = mul(e, ct, args[1], &yt);
            let cb = mul(e, ct, args[0], &yt);
            both(e, ca, cb)
        }
        PrimOp::Div => {
            // y = a / b: ∂a = ct / b, ∂b = −ct · y / b
            let ca = div(e, ct, args[1], &yt);
            let t = mul(e, ct, y, &yt);
            let t = div(e, t, args[1], &yt);
            let cb = neg(e, t, &yt);
            both(e, ca, cb)
        }
        PrimOp::Neg => vec![(0, neg(e, ct, &yt))],
        PrimOp::Pow(n) => match n {
            0 => vec![(0, e.prim(tys[0].clone(), PrimOp::ZerosLike, vec![args[0]], span))],
            1 => vec![(0, ct)],
            n => {
                let k = real(e, *n);
                let p = e.prim(yt.clone(), PrimOp::Pow(n - 1), vec![args[0]], span);
                let kp = mul(e, k, p, &yt);
                vec![(0, mul(e, ct, kp, &yt))]
            }
        },
        PrimOp::Exp => vec![(0, mul(e, ct, y, &yt))],
        PrimOp::Log => vec![(0, div(e, ct, args[0], &yt))],
        PrimOp::Sqrt => {
            let two = real(e, 2);
            let d = mul(e, two, y, &yt);
            vec![(0, div(e, ct, d, &yt))]
        }
        PrimOp::Tanh => {
            // ct · (1 − y²)
            let one = real(e, 1);
            let y2 = mul(e, y, y, &yt);
            let d = e.prim(yt.clone(), PrimOp::Sub, vec![one, y2], span);
            vec![(0, mul(e, ct, d, &yt))]
        }
        PrimOp::Sigmoid => {
            let one = real(e, 1);
            let d = e.prim(yt.clone(), PrimOp::Sub, vec![one, y], span);
            let yd = mul(e, y, d, &yt);
            vec![(0, mul(e, ct, yd, &yt))]
        }
        PrimOp::Sin => {
            let c = e.prim(yt.clone(), PrimOp::Cos, vec![args[0]], span);
            vec![(0, mul(e, ct, c, &yt))]
        }
        PrimOp::Cos => {
            let s = e.prim(yt.clone(), PrimOp::Sin, vec![args[0]], span);
            let ms = neg(e, s, &yt);
            vec![(0, mul(e, ct, ms, &yt))]
        }
        PrimOp::Gelu => {
            let d = e.prim(yt.clone(), PrimOp::GeluDeriv, vec![args[0]], span);
            vec![(0, mul(e, ct, d, &yt))]
        }
        PrimOp::Relu => {
            let d = e.prim(yt.clone(), PrimOp::Heaviside, vec![args[0]], span);
            vec![(0, mul(e, ct, d, &yt))]
        }
        // A retag moves no data: the cotangent goes back under the argument's axes.
        PrimOp::MergeAxes { .. } | PrimOp::SplitAxes { .. } | PrimOp::ReshapeLike => {
            vec![(0, e.prim(tys[0].clone(), PrimOp::ReshapeLike, vec![ct, args[0]], span))]
        }
        PrimOp::Abs => {
            let d = e.prim(yt.clone(), PrimOp::Sign, vec![args[0]], span);
            vec![(0, mul(e, ct, d, &yt))]
        }
        // Hyperparameters and indices carry no adjoint.
        PrimOp::Real | PrimOp::Argmax | PrimOp::Sample | PrimOp::OneHot | PrimOp::Positions { .. } | PrimOp::CausalMask { .. } => vec![],
        PrimOp::Pi | PrimOp::Zeros { .. } | PrimOp::Ones { .. } | PrimOp::Normal { .. } | PrimOp::XavierUniform { .. } | PrimOp::SplitKey { .. } | PrimOp::DropoutMask => vec![],
        PrimOp::Dot | PrimOp::Matmul => {
            let ca = e.prim(tys[0].clone(), PrimOp::DotVjpL, vec![ct, args[1]], span);
            let cb = e.prim(tys[1].clone(), PrimOp::DotVjpR, vec![args[0], ct], span);
            vec![(0, ca), (1, cb)]
        }
        PrimOp::Ein { operands, output } => {
            // ∂operand_j = ein[axes_j](ct[output] · Π_{i≠j} operand_i), broadcast over
            // axes of j that no other factor carries.
            let mut out = Vec::new();
            for j in 0..args.len() {
                let mut ops = vec![output.clone()];
                let mut vs = vec![ct];
                for (i, a) in args.iter().enumerate() {
                    if i != j {
                        ops.push(operands[i].clone());
                        vs.push(*a);
                    }
                }
                let present: Vec<String> = operands[j].iter().filter(|ax| ops.iter().any(|o| o.contains(ax))).cloned().collect();
                let partial = e.prim(tys[j].clone(), PrimOp::Ein { operands: ops, output: present.clone() }, vs, span);
                let cj = if present.len() == operands[j].len() { partial } else { e.prim(tys[j].clone(), PrimOp::BroadcastLike, vec![partial, args[j]], span) };
                out.push((j, cj));
            }
            out
        }
        PrimOp::SumAll | PrimOp::Sum { .. } => vec![(0, e.prim(tys[0].clone(), PrimOp::BroadcastLike, vec![ct, args[0]], span))],
        PrimOp::MeanAll => vec![(0, e.prim(tys[0].clone(), PrimOp::MeanVjp { axis: None }, vec![ct, args[0]], span))],
        PrimOp::Mean { axis } => vec![(0, e.prim(tys[0].clone(), PrimOp::MeanVjp { axis: Some(axis.clone()) }, vec![ct, args[0]], span))],
        PrimOp::Softmax { axis } => {
            // ∂x = y · (ct − Σ_axis(ct · y))
            let t = mul(e, ct, y, &yt);
            let s = e.prim(yt.clone(), PrimOp::Sum { axis: axis.clone() }, vec![t], span);
            let sb = e.prim(yt.clone(), PrimOp::BroadcastLike, vec![s, ct], span);
            let d = e.prim(yt.clone(), PrimOp::Sub, vec![ct, sb], span);
            vec![(0, mul(e, y, d, &yt))]
        }
        PrimOp::Embed => vec![(0, e.prim(tys[0].clone(), PrimOp::EmbedVjp, vec![args[0], args[1], ct], span))],
        PrimOp::CrossEntropy => vec![(0, e.prim(tys[0].clone(), PrimOp::CrossEntropyVjp, vec![args[0], args[1], ct], span))],
        PrimOp::IndexConst { axis, index } => vec![(0, e.prim(tys[0].clone(), PrimOp::ScatterIndex { axis: axis.clone(), index: *index }, vec![args[0], ct], span))],
        PrimOp::SplitHeads { h, hd } => vec![(0, e.prim(tys[0].clone(), PrimOp::MergeHeads { h: *h, hd: *hd }, vec![ct], span))],
        PrimOp::MergeHeads { h, hd } => vec![(0, e.prim(tys[0].clone(), PrimOp::SplitHeads { h: *h, hd: *hd }, vec![ct], span))],
        PrimOp::SliceParams { index } => vec![(0, e.prim(tys[0].clone(), PrimOp::ScatterParams { index: *index }, vec![args[0], ct], span))],
        PrimOp::StackParams { .. } => (0..args.len()).map(|i| (i, e.prim(tys[i].clone(), PrimOp::SliceParams { index: i as u64 }, vec![ct], span))).collect(),
        // Structural primitives of adjoint code: linear, with obvious adjoints.
        PrimOp::ZerosLike => vec![],
        PrimOp::SetField { name } => {
            let z = e.prim(tys[1].clone(), PrimOp::ZerosLike, vec![args[1]], span);
            let cr = e.prim(tys[0].clone(), PrimOp::SetField { name: name.clone() }, vec![ct, z], span);
            let cv = e.emit(tys[1].clone(), Rhs::Field(ct, name.clone()), span);
            vec![(0, cr), (1, cv)]
        }
        PrimOp::SetElem { index } => {
            let z = e.prim(tys[1].clone(), PrimOp::ZerosLike, vec![args[1]], span);
            let cr = e.prim(tys[0].clone(), PrimOp::SetElem { index: *index }, vec![ct, z], span);
            let cv = e.emit(tys[1].clone(), Rhs::Proj(ct, *index), span);
            vec![(0, cr), (1, cv)]
        }
        PrimOp::SumTo => vec![(0, e.prim(tys[0].clone(), PrimOp::BroadcastLike, vec![ct, args[0]], span))],
        PrimOp::BroadcastLike => vec![(0, e.prim(tys[0].clone(), PrimOp::SumTo, vec![ct, args[0]], span))],
        PrimOp::MeanVjp { axis } => {
            // linear in ct: adjoint is the forward mean with the same axis
            let m = match axis {
                None => e.prim(tys[0].clone(), PrimOp::MeanAll, vec![ct], span),
                Some(a) => e.prim(tys[0].clone(), PrimOp::Mean { axis: a.clone() }, vec![ct], span),
            };
            vec![(0, m)]
        }
        PrimOp::SumSquares | PrimOp::SetSlice | PrimOp::DotVjpL | PrimOp::DotVjpR | PrimOp::EmbedVjp | PrimOp::CrossEntropyVjp | PrimOp::ScatterIndex { .. } | PrimOp::ScatterParams { .. } | PrimOp::GeluDeriv | PrimOp::Heaviside | PrimOp::Sign => {
            return Err("second-order derivatives of adjoint primitives are not supported in 0.1.0".into());
        }
    })
}
