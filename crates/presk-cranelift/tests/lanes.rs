//! The lanes the native backend gives a scalar kernel itself (plan 0.17.0
//! step 2, D175): every kernel below is written with one lane, the backend
//! runs its lane-wise loops over four (`f32`) or two (`f64`), and the bits
//! are the kernel interpreter's, on values that include the infinities, a
//! NaN, signed zeros and subnormals, alone and under a thread split.

use presk_kernel::KernelExec;
use presk_kernel::ir::{BinOp, BufKind, BufSpec, Dtype, Expr, Kernel, Stmt, Threads};
use presk_rt::Buf;

fn b(e: Expr) -> Box<Expr> {
    Box::new(e)
}

fn spec(dims: &[usize]) -> BufSpec {
    BufSpec { dims: dims.to_vec(), kind: BufKind::Real, names: vec![None; dims.len()] }
}

fn iv(v: usize) -> Expr {
    Expr::Iv(v)
}

fn bin(op: BinOp, x: Expr, y: Expr) -> Expr {
    Expr::Bin(op, b(x), b(y))
}

/// `out[i, j] = Σ_k (1 · a[k, j]) · c[k, i]`, the attention gradient's
/// `ein[…, seq2, hd]` (D174): lanes along `j`, the sum over `k` in its order
/// in each lane, `c` broadcast.
fn contraction(d: Dtype, threads: Threads) -> Kernel {
    Kernel {
        name: "ein".into(),
        dtype: d,
        inputs: vec![spec(&[6, 8]), spec(&[6, 4])],
        output: spec(&[4, 8]),
        nloops: 3,
        nreal: 1,
        nint: 0,
        nvec: 0,
        lanes: 1,
        body: vec![Stmt::For {
            var: 0,
            bound: 4,
            body: vec![Stmt::For {
                var: 1,
                bound: 8,
                body: vec![
                    Stmt::Set { var: 0, value: Expr::F(0.0) },
                    Stmt::For {
                        var: 2,
                        bound: 6,
                        body: vec![Stmt::Set {
                            var: 0,
                            value: bin(BinOp::Add, Expr::Sv(0), bin(BinOp::Mul, bin(BinOp::Mul, Expr::F(1.0), Expr::Load(0, vec![iv(2), iv(1)])), Expr::Load(1, vec![iv(2), iv(0)]))),
                        }],
                    },
                    Stmt::Store { buf: 2, index: vec![iv(0), iv(1)], value: Expr::Sv(0) },
                ],
            }],
        }],
        threads,
    }
}

/// `out[i, j] = (x[i, j] − m[i]) / (y[i, j] · s)`, a scalar `s` computed
/// before the loop: lanes along `j`, `m[i]` and `s` broadcast.
fn elementwise(d: Dtype) -> Kernel {
    Kernel {
        name: "fused".into(),
        dtype: d,
        inputs: vec![spec(&[3, 8]), spec(&[3]), spec(&[3, 8])],
        output: spec(&[3, 8]),
        nloops: 2,
        nreal: 1,
        nint: 0,
        nvec: 0,
        lanes: 1,
        body: vec![
            Stmt::Set { var: 0, value: bin(BinOp::Sub, Expr::F(0.75), Expr::Load(1, vec![Expr::I(2)])) },
            Stmt::For {
                var: 0,
                bound: 3,
                body: vec![Stmt::For {
                    var: 1,
                    bound: 8,
                    body: vec![Stmt::Store {
                        buf: 3,
                        index: vec![iv(0), iv(1)],
                        value: bin(BinOp::Div, bin(BinOp::Sub, Expr::Load(0, vec![iv(0), iv(1)]), Expr::Load(1, vec![iv(0)])), bin(BinOp::Mul, Expr::Load(2, vec![iv(0), iv(1)]), Expr::Sv(0))),
                    }],
                }],
            },
        ],
        threads: Threads::Default,
    }
}

/// `out[j] = 0`, then `out[j] = out[j] + x[i, k, j]` over `i` and `k`,
/// `sum_to`'s form: each lane reads and writes its own element.
fn sum_to(d: Dtype) -> Kernel {
    Kernel {
        name: "sum_to".into(),
        dtype: d,
        inputs: vec![spec(&[3, 2, 8])],
        output: spec(&[8]),
        nloops: 4,
        nreal: 0,
        nint: 0,
        nvec: 0,
        lanes: 1,
        body: vec![
            Stmt::For { var: 0, bound: 8, body: vec![Stmt::Store { buf: 1, index: vec![iv(0)], value: Expr::F(0.0) }] },
            Stmt::For {
                var: 1,
                bound: 3,
                body: vec![Stmt::For {
                    var: 2,
                    bound: 2,
                    body: vec![Stmt::For { var: 3, bound: 8, body: vec![Stmt::Store { buf: 1, index: vec![iv(3)], value: bin(BinOp::Add, Expr::Load(1, vec![iv(3)]), Expr::Load(0, vec![iv(1), iv(2), iv(3)])) }] }],
                }],
            },
        ],
        threads: Threads::One,
    }
}

/// Values with every edge of the format among ordinary ones.
fn drawn(n: usize, seed: u64, d: Dtype) -> Buf {
    let edges = [f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 0.0, -0.0, 1e-310, -3e-39, 1e-45];
    let mut s = seed;
    let xs: Vec<f64> = (0..n)
        .map(|i| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            if i % 7 == 3 { edges[(s >> 60) as usize % edges.len()] } else { ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 8.0 }
        })
        .collect();
    match d {
        Dtype::F64 => Buf::F64(xs.into()),
        _ => Buf::F32(xs.iter().map(|&x| x as f32).collect()),
    }
}

fn bits(b: &Buf) -> Vec<u64> {
    match b {
        Buf::F64(v) => v.iter().map(|x| x.to_bits()).collect(),
        Buf::F32(v) => v.iter().map(|x| x.to_bits() as u64).collect(),
        other => panic!("not a real buffer: {other:?}"),
    }
}

#[test]
fn lane_wise_loops_give_the_scalar_kernels_bits() {
    for d in [Dtype::F32, Dtype::F64] {
        let kernels = [
            contraction(d, Threads::Default),
            contraction(d, Threads::At { depth: 0, workers: 3 }),
            elementwise(d),
            sum_to(d),
        ];
        for threads in [1, 3] {
            let mut native = presk_cranelift::NativeExec::new().unwrap();
            native.threads = threads;
            for (i, k) in kernels.iter().enumerate() {
                let inputs: Vec<Buf> = k.inputs.iter().enumerate().map(|(j, s)| drawn(s.numel(), 17 + j as u64, d)).collect();
                let refs: Vec<&Buf> = inputs.iter().collect();
                let mut want = presk_kernel::exec::output_buf(k);
                presk_kernel::exec::run_kernel(k, &refs, &mut want);
                let mut got = presk_kernel::exec::output_buf(k);
                native.run(i, k, &refs, &mut got).unwrap();
                assert_eq!(bits(&got), bits(&want), "{} ({d:?}, {threads} threads, {:?}): the lanes must give the scalar kernel's bits", k.name, k.threads);
            }
        }
    }
}
