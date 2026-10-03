//! Property test: random PSK-Core expressions over two tensors and a scalar
//! are rewritten and evaluated against the original on the interpreter at
//! 128 bits. The rules are exact over ℝ, so the two results may differ only
//! by rounding: 2⁻¹⁰⁰ relative is the bound.

mod common;

use std::collections::BTreeMap;

use common::{prepare, repo_root};
use presk_core::ir::{Block, Func, Let, Lit, Origin, Param, Rhs};
use presk_core::{FuncId, PrimOp, VarId};
use presk_interp::value::{Axis, Tensor};
use presk_interp::{Interp, Value};
use presk_types::ty::Ty;
use proptest::prelude::*;
use rug::Float;

/// A random expression, driven by a byte string (each byte one decision).
#[derive(Clone, Debug)]
enum E {
    A,
    B,
    S,
    Const(u8), // 0, 1, 2, 2.5
    Add(Box<E>, Box<E>),
    Sub(Box<E>, Box<E>),
    Mul(Box<E>, Box<E>),
    Neg(Box<E>),
    Exp(Box<E>),
    LogExp(Box<E>),
    Pow(u64, Box<E>),
    DivExp(Box<E>, Box<E>),
    SqrtSq(Box<E>),
    Sum(Box<E>),
    Mean(Box<E>),
    Scale(Box<E>, Box<E>),
    SumTo(Box<E>),
    Broadcast(Box<E>),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Tensor,
    Scalar,
}

struct Gen<'b> {
    bytes: &'b [u8],
    pos: usize,
}

impl Gen<'_> {
    fn next(&mut self) -> u8 {
        let b = self.bytes.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }

    fn grow(&mut self, kind: Kind, depth: u32, exps: u32) -> E {
        let b = self.next();
        if depth == 0 || b < 40 {
            return match (kind, b % 4) {
                (Kind::Tensor, 0) => E::A,
                (Kind::Tensor, 1) => E::B,
                (Kind::Tensor, 2) => E::A,
                (Kind::Tensor, _) => E::B,
                (Kind::Scalar, 0) => E::S,
                (Kind::Scalar, k) => E::Const(k),
            };
        }
        let d = depth - 1;
        match (kind, b % 14) {
            (k, 0) => E::Add(Box::new(self.grow(k, d, exps)), Box::new(self.grow(k, d, exps))),
            (k, 1) => E::Sub(Box::new(self.grow(k, d, exps)), Box::new(self.grow(k, d, exps))),
            (k, 2) => E::Mul(Box::new(self.grow(k, d, exps)), Box::new(self.grow(k, d, exps))),
            (k, 3) => E::Neg(Box::new(self.grow(k, d, exps))),
            (k, 4) if exps < 2 => E::Exp(Box::new(self.grow(k, d, exps + 1))),
            (k, 5) if exps < 2 => E::LogExp(Box::new(self.grow(k, d, exps + 1))),
            (k, 6) => E::Pow(1 + u64::from(self.next() % 3), Box::new(self.grow(k, d, exps))),
            (k, 7) if exps < 2 => E::DivExp(Box::new(self.grow(k, d, exps)), Box::new(self.grow(k, d, exps + 1))),
            (k, 8) if exps < 2 => E::SqrtSq(Box::new(self.grow(k, d, exps + 1))),
            (Kind::Scalar, 9) => E::Sum(Box::new(self.grow(Kind::Tensor, d, exps))),
            (Kind::Scalar, 10) => E::Mean(Box::new(self.grow(Kind::Tensor, d, exps))),
            (Kind::Tensor, 11) => E::Scale(Box::new(self.grow(Kind::Scalar, d, exps)), Box::new(self.grow(Kind::Tensor, d, exps))),
            (Kind::Tensor, 12) => E::SumTo(Box::new(self.grow(Kind::Tensor, d, exps))),
            (Kind::Tensor, 13) => E::Broadcast(Box::new(self.grow(Kind::Tensor, d, exps))),
            (k, _) => E::Add(Box::new(self.grow(k, d, exps)), Box::new(self.grow(k, d, exps))),
        }
    }
}

struct Emit {
    lets: Vec<Let>,
    next: u32,
    tensor: Ty,
    span: presk_syntax::Span,
}

impl Emit {
    fn push(&mut self, ty: Ty, rhs: Rhs) -> VarId {
        let var = VarId(self.next);
        self.next += 1;
        self.lets.push(Let { var, name: None, ty, rhs, span: self.span });
        var
    }

    fn prim(&mut self, ty: Ty, op: PrimOp, args: Vec<VarId>) -> VarId {
        self.push(ty, Rhs::Prim { op, args })
    }

    fn ty(&self, k: Kind) -> Ty {
        match k {
            Kind::Tensor => self.tensor.clone(),
            Kind::Scalar => Ty::scalar_real(),
        }
    }

    fn emit(&mut self, e: &E, k: Kind) -> VarId {
        let ty = self.ty(k);
        match e {
            E::A => VarId(0),
            E::B => VarId(1),
            E::S => VarId(2),
            E::Const(c) => {
                let lit = match c {
                    0 => self.push(Ty::Nat, Rhs::Lit(Lit::Nat(0))),
                    1 => self.push(Ty::Nat, Rhs::Lit(Lit::Nat(1))),
                    2 => self.push(Ty::Nat, Rhs::Lit(Lit::Nat(2))),
                    _ => self.push(Ty::F64, Rhs::Lit(Lit::Float("2.5".into()))),
                };
                self.prim(Ty::scalar_real(), PrimOp::Real, vec![lit])
            }
            E::Add(a, b) => {
                let (x, y) = (self.emit(a, k), self.emit(b, k));
                self.prim(ty, PrimOp::Add, vec![x, y])
            }
            E::Sub(a, b) => {
                let (x, y) = (self.emit(a, k), self.emit(b, k));
                self.prim(ty, PrimOp::Sub, vec![x, y])
            }
            E::Mul(a, b) => {
                let (x, y) = (self.emit(a, k), self.emit(b, k));
                self.prim(ty, PrimOp::Mul, vec![x, y])
            }
            E::Neg(a) => {
                let x = self.emit(a, k);
                self.prim(ty, PrimOp::Neg, vec![x])
            }
            E::Exp(a) => {
                let x = self.emit(a, k);
                self.prim(ty, PrimOp::Exp, vec![x])
            }
            E::LogExp(a) => {
                let x = self.emit(a, k);
                let e = self.prim(ty.clone(), PrimOp::Exp, vec![x]);
                self.prim(ty, PrimOp::Log, vec![e])
            }
            E::Pow(n, a) => {
                let x = self.emit(a, k);
                self.prim(ty, PrimOp::Pow(*n), vec![x])
            }
            E::DivExp(a, b) => {
                let (x, y) = (self.emit(a, k), self.emit(b, k));
                let e = self.prim(ty.clone(), PrimOp::Exp, vec![y]);
                self.prim(ty, PrimOp::Div, vec![x, e])
            }
            E::SqrtSq(a) => {
                let x = self.emit(a, k);
                let e = self.prim(ty.clone(), PrimOp::Exp, vec![x]);
                let s = self.prim(ty.clone(), PrimOp::Sqrt, vec![e]);
                self.prim(ty, PrimOp::Mul, vec![s, s])
            }
            E::Sum(a) => {
                let x = self.emit(a, Kind::Tensor);
                self.prim(ty, PrimOp::SumAll, vec![x])
            }
            E::Mean(a) => {
                let x = self.emit(a, Kind::Tensor);
                self.prim(ty, PrimOp::MeanAll, vec![x])
            }
            E::Scale(s, a) => {
                let (x, y) = (self.emit(s, Kind::Scalar), self.emit(a, Kind::Tensor));
                self.prim(ty, PrimOp::Mul, vec![x, y])
            }
            E::SumTo(a) => {
                let x = self.emit(a, Kind::Tensor);
                self.prim(ty, PrimOp::SumTo, vec![x, VarId(0)])
            }
            E::Broadcast(a) => {
                let x = self.emit(a, Kind::Tensor);
                self.prim(ty, PrimOp::BroadcastLike, vec![x, VarId(0)])
            }
        }
    }
}

/// The program of `mlp.psk` plus one function `prop(a, b : ℝ[batch: B, model: D], s : ℝ[]) -> …`.
fn program_with(e: &E, kind: Kind) -> (presk_types::elab::ElabProgram, presk_core::Program, FuncId, BTreeMap<presk_types::dim::DimVar, u64>) {
    let (prog, _, mut core) = prepare("tests/examples/mlp/mlp.psk", &[]);
    let mse = core.func(core.find("mse").unwrap()).clone();
    let tensor = mse.params[1].ty.clone();
    let mut em = Emit { lets: vec![], next: 3, tensor: tensor.clone(), span: mse.span };
    let result = em.emit(e, kind);
    let ret = em.ty(kind);
    let f = Func {
        name: "prop".into(),
        origin: Origin::Lambda { parent: core.find("mse").unwrap() },
        dims: mse.dims.clone(),
        params: vec![Param { var: VarId(0), name: "a".into(), ty: tensor.clone() }, Param { var: VarId(1), name: "b".into(), ty: tensor }, Param { var: VarId(2), name: "s".into(), ty: Ty::scalar_real() }],
        ret,
        effects: vec![],
        body: Block { lets: em.lets, result },
        span: mse.span,
    };
    core.funcs.push(f);
    let fid = FuncId(core.funcs.len() as u32 - 1);
    presk_core::verify(&core).unwrap();
    let dims = mse.dims.iter().map(|d| (*d, match prog.dim_name(*d).as_str() { "D" => 3, _ => 2 })).collect();
    (prog, core, fid, dims)
}

fn inputs(seed: u64) -> Vec<Value<Float>> {
    // A small fixed generator: enough to vary the inputs, reproducible.
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 4.0 - 2.0
    };
    let axes = vec![Axis { name: Some("batch".into()), len: 2 }, Axis { name: Some("model".into()), len: 3 }];
    let mut t = || Value::Real(Tensor { axes: axes.clone(), data: (0..6).map(|_| Float::with_val(128, next())).collect() });
    let a = t();
    let b = t();
    let s = Value::Real(Tensor { axes: vec![], data: vec![Float::with_val(128, next())] });
    vec![a, b, s]
}

fn close(x: &Value<Float>, y: &Value<Float>) -> Result<(), String> {
    let (Value::Real(tx), Value::Real(ty)) = (x, y) else { return Err(format!("kinds differ: {} vs {}", x.kind(), y.kind())) };
    if tx.axes != ty.axes {
        return Err(format!("axes differ: {:?} vs {:?}", tx.axes, ty.axes));
    }
    let tol = Float::with_val(128, 2f64.powi(-100));
    for (u, v) in tx.data.iter().zip(&ty.data) {
        if u.is_nan() && v.is_nan() {
            continue;
        }
        let err = Float::with_val(128, u - v).abs();
        let scale = Float::with_val(128, u.clone().abs()).max(&Float::with_val(128, 1));
        if err > Float::with_val(128, &tol * &scale) {
            return Err(format!("{u} vs {v} (err {err})"));
        }
    }
    Ok(())
}

fn check(bytes: &[u8], kind: Kind, seed: u64) -> Result<(), TestCaseError> {
    let e = Gen { bytes, pos: 0 }.grow(kind, 4, 0);
    let (prog, core, fid, dims) = program_with(&e, kind);
    let mut opt = core.clone();
    let report = presk_opt::optimize(&mut opt, &prog, &presk_opt::Options::default());
    presk_core::verify(&opt).map_err(|errs| TestCaseError::fail(errs.join("\n")))?;
    let args = inputs(seed);
    let mut i0: Interp<Float> = Interp::new(&core, &prog, 128);
    let mut i1: Interp<Float> = Interp::new(&opt, &prog, 128);
    let r0 = i0.call(fid, dims.clone(), args.clone()).map_err(|e| TestCaseError::fail(e.message))?;
    let r1 = i1.call(fid, dims, args).map_err(|e| TestCaseError::fail(format!("optimized program fails: {}\n{}", e.message, presk_core::pretty::pretty_func(opt.func(fid), &opt, &prog))))?;
    close(&r0, &r1).map_err(|why| {
        TestCaseError::fail(format!(
            "{e:?}: {why}\n-- original\n{}\n-- rewritten\n{}\n{}",
            presk_core::pretty::pretty_func(core.func(fid), &core, &prog),
            presk_core::pretty::pretty_func(opt.func(fid), &opt, &prog),
            report.summary()
        ))
    })
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::WithSource("proptest-regressions"))), ..ProptestConfig::default() })]

    #[test]
    fn rewritten_tensor_expressions_agree(bytes in proptest::collection::vec(any::<u8>(), 1..40), seed in any::<u64>()) {
        check(&bytes, Kind::Tensor, seed)?;
    }

    #[test]
    fn rewritten_scalar_expressions_agree(bytes in proptest::collection::vec(any::<u8>(), 1..40), seed in any::<u64>()) {
        check(&bytes, Kind::Scalar, seed)?;
    }
}

#[test]
fn a_hand_written_case() {
    // (a − a) + 2·(b·1) + s·0 : exactly 2b.
    let e = E::Add(Box::new(E::Add(Box::new(E::Sub(Box::new(E::A), Box::new(E::A))), Box::new(E::Scale(Box::new(E::Const(2)), Box::new(E::Mul(Box::new(E::B), Box::new(E::Scale(Box::new(E::Const(1)), Box::new(E::B))))))))), Box::new(E::Scale(Box::new(E::S), Box::new(E::Scale(Box::new(E::Const(0)), Box::new(E::A))))));
    let _ = repo_root();
    check(&[], Kind::Tensor, 0).unwrap();
    let (prog, core, fid, _) = program_with(&e, Kind::Tensor);
    let mut opt = core.clone();
    let report = presk_opt::optimize(&mut opt, &prog, &presk_opt::Options::default());
    let before = core.func(fid).body.lets.len();
    let after = opt.func(fid).body.lets.len();
    assert!(after < before, "{before} → {after}\n{}\n{}", presk_core::pretty::pretty_func(opt.func(fid), &opt, &prog), report.summary());
}
