//! The kernel-IR interpreter: executes a [`Kernel`] on runtime buffers in
//! the kernel's machine type, every operation rounded to that type — the
//! first native-arithmetic realization, and the yardstick of the code
//! generator to come.

use presk_rt::Buf;

use crate::ir::{BinOp, BufKind, CmpOp, Expr, Kernel, Stmt, UnOp};

/// A machine real: `f64` or `f32`, every operation in that type.
pub trait Elem: Copy + PartialOrd + std::fmt::Debug {
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    fn un(self, op: UnOp) -> Self;
    fn bin(self, op: BinOp, other: Self) -> Self;
}

impl Elem for f64 {
    fn from_f64(x: f64) -> Self {
        x
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn un(self, op: UnOp) -> Self {
        match op {
            UnOp::Neg => -self,
            UnOp::Exp => self.exp(),
            UnOp::Log => self.ln(),
            UnOp::Tanh => self.tanh(),
            UnOp::Sqrt => self.sqrt(),
            UnOp::Sin => self.sin(),
            UnOp::Cos => self.cos(),
            UnOp::Erf => libm::erf(self),
            UnOp::Abs => self.abs(),
        }
    }
    fn bin(self, op: BinOp, o: Self) -> Self {
        match op {
            BinOp::Add | BinOp::IAdd => self + o,
            BinOp::Sub => self - o,
            BinOp::Mul | BinOp::IMul => self * o,
            BinOp::Div => self / o,
        }
    }
}

impl Elem for f32 {
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn un(self, op: UnOp) -> Self {
        match op {
            UnOp::Neg => -self,
            UnOp::Exp => self.exp(),
            UnOp::Log => self.ln(),
            UnOp::Tanh => self.tanh(),
            UnOp::Sqrt => self.sqrt(),
            UnOp::Sin => self.sin(),
            UnOp::Cos => self.cos(),
            UnOp::Erf => libm::erff(self),
            UnOp::Abs => self.abs(),
        }
    }
    fn bin(self, op: BinOp, o: Self) -> Self {
        match op {
            BinOp::Add | BinOp::IAdd => self + o,
            BinOp::Sub => self - o,
            BinOp::Mul | BinOp::IMul => self * o,
            BinOp::Div => self / o,
        }
    }
}

struct Env<'a, E: Elem> {
    k: &'a Kernel,
    strides: Vec<Vec<usize>>,
    inputs: &'a [&'a Buf],
    out: &'a mut Buf,
    loops: Vec<usize>,
    reals: Vec<E>,
    ints: Vec<i64>,
    /// One register per vector variable, each of `k.lanes` elements.
    vecs: Vec<Vec<E>>,
}

impl<E: Elem> Env<'_, E> {
    fn offset(&self, buf: usize, index: &[Expr]) -> usize {
        let st = &self.strides[buf];
        index.iter().zip(st).map(|(e, s)| self.eval_i(e) as usize * s).sum()
    }

    fn load_f(&self, buf: usize, off: usize) -> E {
        let b: &Buf = if buf == self.k.out_buf() { self.out } else { self.inputs[buf] };
        match b {
            Buf::F64(v) => E::from_f64(v[off]),
            Buf::F32(v) => E::from_f64(v[off] as f64),
            Buf::Bf16(v) => E::from_f64(presk_rt::bf16_to_f32(v[off]) as f64),
            Buf::I64(v) => E::from_f64(v[off] as f64),
            Buf::Exact(v) => E::from_f64(v[off].to_f64()),
            // An integer weight: `f32(q)·s`, one `f32` multiplication, then
            // widened exactly if the kernel computes in `f64` (D44).
            Buf::Quant(q) => E::from_f64(q.value(off) as f64),
        }
    }

    fn load_i(&self, buf: usize, off: usize) -> i64 {
        let b: &Buf = if buf == self.k.out_buf() { self.out } else { self.inputs[buf] };
        match b {
            Buf::I64(v) => v[off],
            other => other.to_f64()[off] as i64,
        }
    }

    fn eval_f(&self, e: &Expr) -> E {
        match e {
            Expr::F(x) => E::from_f64(*x),
            Expr::Load(b, ix) => {
                let off = self.offset(*b, ix);
                self.load_f(*b, off)
            }
            Expr::Sv(v) => self.reals[*v],
            Expr::Bin(op, a, b) => self.eval_f(a).bin(*op, self.eval_f(b)),
            Expr::Un(op, a) => self.eval_f(a).un(*op),
            Expr::Select(c, a, b) => {
                if self.eval_bool(c) {
                    self.eval_f(a)
                } else {
                    self.eval_f(b)
                }
            }
            Expr::ToF(a) => E::from_f64(self.eval_i(a) as f64),
            other => E::from_f64(self.eval_i(other) as f64),
        }
    }

    fn eval_i(&self, e: &Expr) -> i64 {
        match e {
            Expr::I(i) => *i,
            Expr::Iv(v) => self.loops[*v] as i64,
            Expr::Si(v) => self.ints[*v],
            Expr::LoadI(b, ix) => {
                let off = self.offset(*b, ix);
                self.load_i(*b, off)
            }
            Expr::Bin(BinOp::IAdd, a, b) => self.eval_i(a) + self.eval_i(b),
            Expr::Bin(BinOp::IMul, a, b) => self.eval_i(a) * self.eval_i(b),
            Expr::ToI(a) => self.eval_f(a).to_f64() as i64,
            Expr::Select(c, a, b) => {
                if self.eval_bool(c) {
                    self.eval_i(a)
                } else {
                    self.eval_i(b)
                }
            }
            other => self.eval_f(other).to_f64() as i64,
        }
    }

    fn eval_bool(&self, e: &Expr) -> bool {
        match e {
            Expr::Cmp(op, a, b) => {
                let (x, y) = (self.eval_f(a), self.eval_f(b));
                match op {
                    CmpOp::Lt => x < y,
                    CmpOp::Le => x <= y,
                    CmpOp::Gt => x > y,
                    CmpOp::Ge => x >= y,
                    CmpOp::Eq => x == y,
                }
            }
            _ => self.eval_i(e) != 0,
        }
    }

    /// A vector expression, lane by lane. Every lane is an independent
    /// element of the same computation — never a partial sum of one — so the
    /// arithmetic each lane performs is exactly the arithmetic the scalar
    /// form performs for that element, in the same order (D42).
    fn eval_v(&self, e: &Expr) -> Vec<E> {
        let n = self.k.lanes;
        match e {
            Expr::Vv(v) => self.vecs[*v].clone(),
            Expr::VLoad(buf, ix) => {
                let off = self.offset(*buf, ix);
                (0..n).map(|i| self.load_f(*buf, off + i)).collect()
            }
            Expr::VSplat(x) => {
                let s = self.eval_f(x);
                vec![s; n]
            }
            Expr::VBin(op, a, b) => {
                let (x, y) = (self.eval_v(a), self.eval_v(b));
                x.into_iter().zip(y).map(|(p, q)| p.bin(*op, q)).collect()
            }
            // A scalar in a vector position is one lane wide, broadcast.
            other => vec![self.eval_f(other); n],
        }
    }

    fn store_f(&mut self, off: usize, x: E) {
        match self.out {
            Buf::F64(v) => v[off] = x.to_f64(),
            Buf::F32(v) => v[off] = x.to_f64() as f32,
            // Storage only: the value was computed in `f32` and is rounded
            // once, to nearest even, on the way to memory.
            Buf::Bf16(v) => v[off] = presk_rt::f32_to_bf16(x.to_f64() as f32),
            Buf::I64(v) => v[off] = x.to_f64() as i64,
            // Parameters are quantised, results never are.
            Buf::Exact(_) | Buf::Quant(_) => {}
        }
    }

    fn run(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            match s {
                Stmt::For { var, bound, body } => {
                    for i in 0..*bound {
                        self.loops[*var] = i;
                        self.run(body);
                    }
                }
                Stmt::Set { var, value } => {
                    let x = self.eval_f(value);
                    self.reals[*var] = x;
                }
                Stmt::SetI { var, value } => {
                    let x = self.eval_i(value);
                    self.ints[*var] = x;
                }
                Stmt::SetL { var, value } => {
                    let x = self.eval_i(value);
                    self.loops[*var] = x.max(0) as usize;
                }
                Stmt::VSet { var, value } => {
                    let x = self.eval_v(value);
                    self.vecs[*var] = x;
                }
                Stmt::VStore { buf, index, value } => {
                    let off = self.offset(*buf, index);
                    let x = self.eval_v(value);
                    for (i, e) in x.into_iter().enumerate() {
                        self.store_f(off + i, e);
                    }
                }
                Stmt::Store { buf, index, value } => {
                    let off = self.offset(*buf, index);
                    let x = self.eval_f(value);
                    self.store_f(off, x);
                }
                Stmt::StoreI { buf, index, value } => {
                    let off = self.offset(*buf, index);
                    let x = self.eval_i(value);
                    if let Buf::I64(v) = self.out {
                        v[off] = x;
                    }
                }
                Stmt::If { cond, then } => {
                    if self.eval_bool(cond) {
                        self.run(then);
                    }
                }
            }
        }
    }
}

/// Runs `k` on `inputs`, writing the output buffer (allocated by the caller
/// with the kernel's output spec).
pub fn run_kernel(k: &Kernel, inputs: &[&Buf], out: &mut Buf) {
    let mut strides: Vec<Vec<usize>> = k.inputs.iter().map(|s| s.strides()).collect();
    strides.push(k.output.strides());
    match k.dtype {
        crate::ir::Dtype::F64 => {
            let mut env = Env::<f64> { k, strides, inputs, out, loops: vec![0; k.nloops], reals: vec![0.0; k.nreal], ints: vec![0; k.nint], vecs: vec![vec![0.0f64; k.lanes]; k.nvec] };
            env.run(&k.body);
        }
        // `bf16` is a storage type: the loop nest runs in `f32` and only the
        // buffers are narrow, which is the whole of the idiom (D34).
        crate::ir::Dtype::F32 | crate::ir::Dtype::Bf16 => {
            let mut env = Env::<f32> { k, strides, inputs, out, loops: vec![0; k.nloops], reals: vec![0.0; k.nreal], ints: vec![0; k.nint], vecs: vec![vec![0.0f32; k.lanes]; k.nvec] };
            env.run(&k.body);
        }
    }
}

/// A fresh output buffer for a kernel.
pub fn output_buf(k: &Kernel) -> Buf {
    let n = k.output.numel();
    match (k.output.kind, k.dtype) {
        (BufKind::Idx, _) => Buf::I64(vec![0; n]),
        (BufKind::Real | BufKind::Quant { .. }, crate::ir::Dtype::F64) => Buf::F64(vec![0.0; n]),
        (BufKind::Real | BufKind::Quant { .. }, crate::ir::Dtype::F32) => Buf::F32(vec![0.0; n]),
        (BufKind::Real | BufKind::Quant { .. }, crate::ir::Dtype::Bf16) => Buf::Bf16(vec![0u16; n]),
    }
}
