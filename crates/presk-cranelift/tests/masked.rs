//! The masked softmax (plan 0.10.0 step 7b, D99) against the additive mask it
//! replaces (`examples/tests/data/masked_softmax.psk`): on every row a mask leaves
//! non-empty, `softmax(axis a, x, where = m)` gives the bits the softmax of
//! `x` with `−∞` at every refused position gives, and so does its gradient.
//! On the reference interpreter at 256 bits; on the kernel interpreter and
//! natively in `f64`, `f32` and `bf16`, in three passes, and in two where a
//! realization admits them (`f64`, `f32`: condition 7). Masks of the scores'
//! last two axes and of the keys alone, drawn with refused first positions;
//! a causal one. A row the mask empties has no value: the interpreter's run
//! is refused.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use presk_cranelift::native_backend;
use presk_interp::Interp;
use presk_kernel::KernelBackend;
use presk_rt::{Axis, Backend, Buf, Dtype, Realization, Tensor, Value};
use presk_syntax::diag::render_all;
use presk_syntax::{SourceMap, parse_file};
use presk_types::check;
use presk_types::elab::{self, ElabProgram, Options};
use rug::Float;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn prepare() -> (ElabProgram, presk_core::Program) {
    let mut sm = SourceMap::new();
    let id = sm.load(&repo_root().join("examples/tests/data/masked_softmax.psk")).unwrap();
    let (program, diags) = parse_file(&sm, id);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (prog, diags) = elab::elaborate(&mut sm, id, program, &Options::default());
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (typed, diags) = check::check(&sm, &prog);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (mut core, diags) = presk_core::lower(&prog, &typed);
    assert!(diags.is_empty(), "{}", render_all(&diags, &sm, false));
    let diags = presk_core::ad::transform(&mut core);
    assert!(diags.is_empty(), "{}", render_all(&diags, &sm, false));
    (prog, core)
}

/// splitmix64: a fixed stream of test inputs.
struct Mix(u64);

impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

const B: usize = 2;
const T: usize = 7;

fn axes(names: &[&str], lens: &[usize]) -> Vec<Axis> {
    names.iter().zip(lens).map(|(n, l)| Axis { name: Some(n.to_string()), len: *l }).collect()
}

/// Scores of `[batch: B, seq: T, seq2: T]`: drawn at a scale, a few ties.
fn scores(m: &mut Mix) -> Vec<f64> {
    let scale = [0.1, 1.0, 8.0][(m.next() % 3) as usize];
    let mut x: Vec<f64> = (0..B * T * T).map(|_| (m.unit() - 0.5) * 2.0 * scale).collect();
    for i in (0..x.len()).step_by(5) {
        x[i] = x[0];
    }
    x
}

/// A mask of `len` positions in rows of `row`, none empty: each position
/// refused with probability one half, the first ones included.
fn mask(m: &mut Mix, len: usize, row: usize) -> Vec<i64> {
    let mut v: Vec<i64> = (0..len).map(|_| (m.next() % 2) as i64).collect();
    for r in v.chunks_mut(row) {
        if r.iter().all(|x| *x == 0) {
            r[(m.next() as usize) % row] = 1;
        }
    }
    v
}

fn causal() -> Vec<i64> {
    (0..T * T).map(|k| i64::from(k % T <= k / T)).collect()
}

/// The arguments of `masked…` and of `additive…`: the scores, and the mask
/// as truth values or as `0` and `−∞`.
fn args(x: &[f64], mk: &[i64], keys: bool, dtype: Dtype) -> (Vec<Value>, Vec<Value>) {
    let (mnames, mlens): (&[&str], Vec<usize>) = if keys { (&["seq2"], vec![T]) } else { (&["seq", "seq2"], vec![T, T]) };
    let x = Value::Real(Tensor { axes: axes(&["batch", "seq", "seq2"], &[B, T, T]), data: Buf::of(dtype, x) });
    let m = Value::Idx { bound: 2, t: Tensor { axes: axes(mnames, &mlens), data: Buf::I64(mk.to_vec().into()) } };
    let add: Vec<f64> = mk.iter().map(|b| if *b != 0 { 0.0 } else { f64::NEG_INFINITY }).collect();
    let a = Value::Real(Tensor { axes: axes(mnames, &mlens), data: Buf::of(dtype, &add) });
    (vec![x.clone(), m], vec![x, a])
}

fn bits(v: &Value) -> Vec<u64> {
    match v {
        Value::Real(t) => t.data.to_f64().iter().map(|x| x.to_bits()).collect(),
        other => panic!("a {}", other.kind()),
    }
}

fn dims(core: &presk_core::Program, prog: &ElabProgram, f: presk_core::FuncId) -> BTreeMap<presk_types::dim::DimVar, u64> {
    core.func(f).dims.iter().map(|d| (*d, if prog.dim_name(*d) == "B" { B as u64 } else { T as u64 })).collect()
}

/// The pairs of entries compared, and whether the mask is of the keys alone.
const PAIRS: [(&str, &str, bool); 3] = [("masked", "additive", false), ("masked_keys", "additive_keys", true), ("masked_grad", "additive_grad", false)];

fn run(b: &mut dyn Backend, core: &presk_core::Program, prog: &ElabProgram, name: &str, args: Vec<Value>, r: &Realization) -> Value {
    let f = core.find(name).unwrap();
    let p = b.prepare(f, &dims(core, prog, f), r).unwrap();
    b.call(&p, args).unwrap()
}

#[test]
fn a_masked_softmax_is_the_additive_masks_bit_for_bit() {
    let (prog, core) = prepare();
    let mut m = Mix(29);
    let mut compared = 0;
    for case in 0..12 {
        let x = scores(&mut m);
        for (masked, additive, keys) in PAIRS {
            let mk = if keys { mask(&mut m, T, T) } else if case == 0 { causal() } else { mask(&mut m, T * T, T) };
            // The reference interpreter, at 256 bits.
            let (am, aa) = args(&x, &mk, keys, Dtype::F64);
            let mut interp: Interp<Float> = Interp::new(&core, &prog, 256);
            let call = |interp: &mut Interp<Float>, name: &str, a: Vec<Value>| {
                let f = core.find(name).unwrap();
                interp.call(f, dims(&core, &prog, f), a.into_iter().map(|v| v.into_interp(256)).collect()).unwrap()
            };
            let (vm, va) = (call(&mut interp, masked, am), call(&mut interp, additive, aa));
            assert_eq!(format!("{vm:?}"), format!("{va:?}"), "{masked} on the interpreter");
            compared += 1;
            for dtype in [Dtype::F64, Dtype::F32, Dtype::Bf16] {
                let r = Realization::uniform(dtype);
                let passes: &[usize] = if dtype == Dtype::Bf16 { &[3] } else { &[3, 2] };
                for &p in passes {
                    let mut kernel = KernelBackend::new(&core, &prog, &r);
                    let mut native = native_backend(&core, &prog, &r).unwrap();
                    if p == 2 {
                        kernel.default_layout.passes = Some(2);
                        native.default_layout.passes = Some(2);
                    }
                    for (what, b) in [("the kernel interpreter", &mut kernel as &mut dyn Backend), ("natively", &mut native as &mut dyn Backend)] {
                        let (am, aa) = args(&x, &mk, keys, dtype);
                        let (vm, va) = (run(b, &core, &prog, masked, am, &r), run(b, &core, &prog, additive, aa, &r));
                        assert_eq!(bits(&vm), bits(&va), "{masked} {what} in {dtype}, {p} passes");
                        compared += 1;
                    }
                    // The forms compared are the ones asked for.
                    let form = if p == 2 { "where, two passes" } else { "where" };
                    for (what, b) in [("the kernel interpreter", &kernel), ("natively", &native)] {
                        assert!(b.kernels.iter().any(|k| k.name == format!("softmax axis seq2 {form}")), "{what}, {p} passes: no `{form}` kernel");
                    }
                }
            }
        }
    }
    eprintln!("{compared} masked softmaxes and gradients, the additive mask's bits");
}

#[test]
fn a_row_the_mask_empties_has_no_value() {
    let (prog, core) = prepare();
    let mut mk = causal();
    mk[2 * T..3 * T].fill(0);
    let (am, _) = args(&scores(&mut Mix(3)), &mk, false, Dtype::F64);
    let mut interp: Interp<Float> = Interp::new(&core, &prog, 256);
    let f = core.find("masked").unwrap();
    let e = interp.call(f, dims(&core, &prog, f), am.into_iter().map(|v| v.into_interp(256)).collect()).unwrap_err();
    assert!(e.message.contains("the mask allows no position of a row along `seq2`"), "{}", e.message);
}
