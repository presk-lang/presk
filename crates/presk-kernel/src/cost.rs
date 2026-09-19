//! What a kernel costs before it runs: its *work* (operations of the loop
//! nest, counted from the IR at its concrete shapes) and its *traffic* (bytes
//! of every buffer it reads and writes). These are the `W` and `Q` of the
//! cost-model document (§1, §3), computed rather than measured; the measured
//! side is [`TimedExec`], which times every launch of an executor so that
//! `presk bench --calibrate` can fit a machine profile to them.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use presk_rt::{Buf, RtError};

use crate::host::KernelExec;
use crate::ir::{BufKind, Dtype, Expr, Kernel, Stmt, UnOp};

type Result<T> = std::result::Result<T, RtError>;

/// Operations weighed by kind: arithmetic 1, a transcendental function 8
/// (roughly the cost of a `libm` call in multiplies).
pub const TRANSCENDENTAL_OPS: f64 = 8.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KernelCost {
    /// Weighted operation count over the whole loop nest.
    pub work: f64,
    /// Bytes read and written: every input buffer once, the output once.
    pub bytes: f64,
}

fn elem_bytes(kind: BufKind, dtype: Dtype) -> f64 {
    match (kind, dtype) {
        (BufKind::Idx, _) => 8.0,
        (BufKind::Real, Dtype::F64) => 8.0,
        (BufKind::Real, Dtype::F32) => 4.0,
    }
}

fn expr_ops(e: &Expr) -> f64 {
    match e {
        Expr::F(_) | Expr::I(_) | Expr::Iv(_) | Expr::Sv(_) | Expr::Si(_) => 0.0,
        // Address arithmetic is not work.
        Expr::Load(_, _) | Expr::LoadI(_, _) => 0.0,
        Expr::Bin(op, a, b) => {
            let own = match op {
                crate::ir::BinOp::IAdd | crate::ir::BinOp::IMul => 0.0,
                _ => 1.0,
            };
            own + expr_ops(a) + expr_ops(b)
        }
        Expr::Un(op, a) => {
            let own = match op {
                UnOp::Neg | UnOp::Abs => 1.0,
                UnOp::Exp | UnOp::Log | UnOp::Tanh | UnOp::Sqrt | UnOp::Sin | UnOp::Cos | UnOp::Erf => TRANSCENDENTAL_OPS,
            };
            own + expr_ops(a)
        }
        Expr::Cmp(_, a, b) => 1.0 + expr_ops(a) + expr_ops(b),
        Expr::Select(c, a, b) => 1.0 + expr_ops(c) + expr_ops(a) + expr_ops(b),
        Expr::ToF(a) | Expr::ToI(a) => expr_ops(a),
    }
}

fn stmts_ops(stmts: &[Stmt]) -> f64 {
    stmts
        .iter()
        .map(|s| match s {
            Stmt::For { bound, body, .. } => *bound as f64 * stmts_ops(body),
            Stmt::Set { value, .. } | Stmt::SetI { value, .. } => expr_ops(value),
            Stmt::Store { index, value, .. } | Stmt::StoreI { index, value, .. } => expr_ops(value) + index.iter().map(expr_ops).sum::<f64>(),
            // The branch is counted as taken: an upper bound.
            Stmt::If { cond, then } => expr_ops(cond) + stmts_ops(then),
        })
        .sum()
}

/// The work and traffic of a kernel at its shapes.
pub fn kernel_cost(k: &Kernel) -> KernelCost {
    let bytes = k.inputs.iter().map(|b| b.numel() as f64 * elem_bytes(b.kind, k.dtype)).sum::<f64>() + k.output.numel() as f64 * elem_bytes(k.output.kind, k.dtype);
    KernelCost { work: stmts_ops(&k.body), bytes }
}

/// One timed launch: the kernel's index in the backend and the seconds it took.
pub type Samples = Rc<RefCell<Vec<(usize, f64)>>>;

/// An executor that times every launch of another one. The first launch of
/// a kernel includes its compilation on a JIT executor; the consumer drops
/// it (`presk bench --calibrate` takes the median of the later ones).
pub struct TimedExec {
    inner: Box<dyn KernelExec>,
    pub samples: Samples,
}

impl TimedExec {
    pub fn new(inner: Box<dyn KernelExec>) -> (Self, Samples) {
        let samples: Samples = Rc::new(RefCell::new(Vec::new()));
        (TimedExec { inner, samples: samples.clone() }, samples)
    }
}

impl KernelExec for TimedExec {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn run(&mut self, idx: usize, k: &Kernel, inputs: &[&Buf], out: &mut Buf) -> Result<()> {
        let t0 = Instant::now();
        let r = self.inner.run(idx, k, inputs, out);
        self.samples.borrow_mut().push((idx, t0.elapsed().as_secs_f64()));
        r
    }
}

/// Median of the launches of each kernel, the first one excluded when there
/// are several (warm-up, compilation).
pub fn median_per_kernel(samples: &[(usize, f64)], nkernels: usize) -> Vec<Option<f64>> {
    let mut per: Vec<Vec<f64>> = vec![Vec::new(); nkernels];
    for (i, t) in samples {
        if *i < nkernels {
            per[*i].push(*t);
        }
    }
    per.into_iter()
        .map(|mut v| {
            if v.len() > 1 {
                v.remove(0);
            }
            if v.is_empty() {
                return None;
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            Some(v[v.len() / 2])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinOp, BufSpec, Stmt};

    #[test]
    fn add_kernel_work_and_bytes() {
        let spec = BufSpec { dims: vec![2, 3], kind: BufKind::Real, names: vec![None, None] };
        let load = |b: usize| Expr::Load(b, vec![Expr::Iv(0), Expr::Iv(1)]);
        let k = Kernel {
            name: "add".into(),
            dtype: Dtype::F64,
            inputs: vec![spec.clone(), spec.clone()],
            output: spec,
            nloops: 2,
            nreal: 0,
            nint: 0,
            body: vec![Stmt::For { var: 0, bound: 2, body: vec![Stmt::For { var: 1, bound: 3, body: vec![Stmt::Store { buf: 2, index: vec![Expr::Iv(0), Expr::Iv(1)], value: Expr::Bin(BinOp::Add, Box::new(load(0)), Box::new(load(1))) }] }] }],
        };
        let c = kernel_cost(&k);
        assert_eq!(c.work, 6.0);
        assert_eq!(c.bytes, 3.0 * 6.0 * 8.0);
        assert_eq!(median_per_kernel(&[(0, 5.0), (0, 1.0), (0, 3.0), (0, 2.0)], 1), vec![Some(2.0)]);
        assert_eq!(median_per_kernel(&[(0, 5.0)], 1), vec![Some(5.0)]);
    }
}
