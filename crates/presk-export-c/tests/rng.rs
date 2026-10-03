//! The keys of the C export and their draws, `c/presk_rt.c` (plan 0.9.0 step
//! 6, D83; plan 0.10.0 step 7b, D99), against the Rust they stand for:
//! - `presk_rng_seed_u64` and `presk_rng_next_u64` against rand_chacha's
//!   `ChaCha20Rng::seed_from_u64` and `next_u64`, 10⁶ draws over 10³ seeds
//!   (special values, drawn ones), each seed read across fifteen refills;
//!   `presk_uniform01` against `presk_interp::key::uniform01` on the same
//!   seeds;
//! - `presk_rng_from_seed` on drawn 32-byte seeds, read by a drawn mix of
//!   `next_u32` and `next_u64`, so that a `u64` straddles two refills;
//! - `presk_key_from_seed`, `presk_key_split` and `presk_key_child` against
//!   `presk_interp::key::Key`'s `from_seed`, `split` and `child`;
//! - `presk_sample` against the native backend's `sample` (the host of
//!   `presk-kernel`), from the same key, on logits in `f64`, `f32` and
//!   `bf16`: drawn at several scales, ties, infinities and NaNs,
//!   vocabularies of one to a thousand tokens.

mod cexec;

use std::ffi::c_void;
use std::path::Path;
use std::sync::Once;
use std::time::Instant;

use cexec::{So, workdir};
use presk_core::PrimOp;
use presk_interp::key::Key;
use presk_rt::{Axis, Buf, Dtype, Realization, Tensor, Value};
use presk_syntax::SourceMap;
use presk_syntax::diag::render_all;
use presk_syntax::parser::parse_file;
use presk_types::check;
use presk_types::elab::{self, ElabProgram, Options};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

/// `presk_rng`.
#[repr(C)]
#[derive(Clone)]
struct CRng {
    key: [u32; 8],
    counter: u64,
    results: [u32; 64],
    index: u32,
}

impl CRng {
    fn new() -> CRng {
        CRng { key: [0; 8], counter: 0, results: [0; 64], index: 0 }
    }
}

/// `presk_dtype`.
const F64: i32 = 0;
const F32: i32 = 1;
const BF16: i32 = 2;

struct Stream {
    _so: So,
    from_seed: unsafe extern "C" fn(*mut CRng, *const u8),
    seed_u64: unsafe extern "C" fn(*mut CRng, u64),
    next_u32: unsafe extern "C" fn(*mut CRng) -> u32,
    next_u64: unsafe extern "C" fn(*mut CRng) -> u64,
    uniform01: unsafe extern "C" fn(*mut CRng) -> f64,
    key_from_seed: unsafe extern "C" fn(*mut [u8; 32], u64),
    key_split: unsafe extern "C" fn(*const [u8; 32], u64, *mut [u8; 32]),
    key_child: unsafe extern "C" fn(*const [u8; 32], u64, *mut [u8; 32]),
    sample: unsafe extern "C" fn(*mut i64, *const c_void, i32, usize, usize, *const [u8; 32]),
}

fn sym<T: Copy>(so: &So, name: &str) -> T {
    assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<*mut c_void>());
    // SAFETY: `name` is declared in presk_rt.h with the signature `T` spells.
    unsafe { std::mem::transmute_copy::<*mut c_void, T>(&so.sym(name)) }
}

impl Stream {
    fn build(tag: &str) -> Stream {
        let dir = workdir(tag);
        let so = So::build(&dir, &[dir.join("presk_rt.c"), dir.join("presk_libm.c")]);
        Stream {
            from_seed: sym(&so, "presk_rng_from_seed"),
            seed_u64: sym(&so, "presk_rng_seed_u64"),
            next_u32: sym(&so, "presk_rng_next_u32"),
            next_u64: sym(&so, "presk_rng_next_u64"),
            uniform01: sym(&so, "presk_uniform01"),
            key_from_seed: sym(&so, "presk_key_from_seed"),
            key_split: sym(&so, "presk_key_split"),
            key_child: sym(&so, "presk_key_child"),
            sample: sym(&so, "presk_sample"),
            _so: so,
        }
    }
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

/// 10³ seeds: special values, small ones, and drawn.
fn seeds() -> Vec<u64> {
    let mut s = vec![0, 1, 2, u64::MAX, u64::MAX - 1, 1 << 63, 0x5a5a_5a5a, 0x5a5a_5a5a_5a5a_5a5a, 0xffff_ffff, 1 << 32];
    s.extend(3..43);
    let mut m = Mix(7);
    while s.len() < 1000 {
        s.push(m.next());
    }
    s
}

#[test]
fn the_stream_is_rand_chachas_on_a_million_draws() {
    let t0 = Instant::now();
    let c = Stream::build("rng-u64");
    let (mut draws, mut refills) = (0usize, 0u64);
    for seed in seeds() {
        let mut want = ChaCha20Rng::seed_from_u64(seed);
        let mut got = CRng::new();
        // SAFETY: a presk_rng the C fills.
        unsafe { (c.seed_u64)(&mut got, seed) };
        for i in 0..1000 {
            let (w, g) = (want.next_u64(), unsafe { (c.next_u64)(&mut got) });
            assert_eq!(g, w, "seed {seed:#x}, draw {i}: {g:#x}, rand_chacha {w:#x}");
            draws += 1;
        }
        refills += got.counter / 4;
        // uniform01 on the same seed.
        let mut want = ChaCha20Rng::seed_from_u64(seed);
        let mut got = CRng::new();
        unsafe { (c.seed_u64)(&mut got, seed) };
        for i in 0..100 {
            let (w, g) = (presk_interp::key::uniform01(&mut want), unsafe { (c.uniform01)(&mut got) });
            assert_eq!(g.to_bits(), w.to_bits(), "seed {seed:#x}, uniform01 {i}: {g}, presk-interp {w}");
        }
    }
    assert!(draws >= 1_000_000);
    eprintln!("{draws} draws of next_u64 over 1000 seeds, {refills} refills of four blocks: rand_chacha's, bit for bit ({:.2} s)", t0.elapsed().as_secs_f64());
}

#[test]
fn a_seed_of_32_bytes_read_by_words_and_pairs_is_rand_chachas() {
    let c = Stream::build("rng-mixed");
    let mut m = Mix(11);
    let mut straddled = 0;
    for s in 0..1000 {
        let mut seed = [0u8; 32];
        for chunk in seed.chunks_exact_mut(8) {
            chunk.copy_from_slice(&m.next().to_le_bytes());
        }
        let mut want = ChaCha20Rng::from_seed(seed);
        let mut got = CRng::new();
        unsafe { (c.from_seed)(&mut got, seed.as_ptr()) };
        for i in 0..300 {
            let wide = m.next() & 1 == 1;
            if wide && got.index == 63 {
                straddled += 1;
            }
            let (w, g) = if wide { (want.next_u64(), unsafe { (c.next_u64)(&mut got) }) } else { (u64::from(want.next_u32()), u64::from(unsafe { (c.next_u32)(&mut got) })) };
            assert_eq!(g, w, "seed {s}, read {i} ({}): {g:#x}, rand_chacha {w:#x}", if wide { "u64" } else { "u32" });
        }
    }
    assert!(straddled > 100, "only {straddled} u64s straddled two refills");
}

#[test]
fn keys_are_presk_interps() {
    let c = Stream::build("rng-keys");
    let mut m = Mix(17);
    let (mut splits, mut children) = (0, 0);
    for seed in seeds() {
        let want = Key::from_seed(seed);
        let mut k = [0u8; 32];
        unsafe { (c.key_from_seed)(&mut k, seed) };
        assert_eq!(k, want.0, "seed {seed:#x}: the key it names");
        // Split: key l of split_key n, every l of n up to 9.
        let n = 1 + (m.next() % 9) as usize;
        for (l, w) in want.split(n).iter().enumerate() {
            let mut g = [0u8; 32];
            unsafe { (c.key_split)(&k, l as u64, &mut g) };
            assert_eq!(g, w.0, "seed {seed:#x}: key {l} of split {n}");
            splits += 1;
        }
        // Children: the first steps, special numbers, drawn ones.
        for t in [0, 1, 2, 3, 255, 256, u64::MAX, 1 << 63, m.next(), m.next()] {
            let mut g = [0u8; 32];
            unsafe { (c.key_child)(&k, t, &mut g) };
            assert_eq!(g, want.child(t).0, "seed {seed:#x}: child {t}");
            children += 1;
        }
    }
    eprintln!("1000 keys of seeds, {splits} splits, {children} children: presk-interp's, byte for byte");
}

fn chdir_root() {
    static ONCE: Once = Once::new();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap();
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
    (prog, core)
}

/// A row of logits: drawn at a scale, with ties, infinities or NaNs.
fn row(m: &mut Mix, vocab: usize) -> Vec<f64> {
    let scale = [1e-3, 0.1, 1.0, 8.0, 100.0, 1e4][(m.next() % 6) as usize];
    let mut r: Vec<f64> = (0..vocab).map(|_| (m.unit() - 0.5) * 2.0 * scale).collect();
    match m.next() % 8 {
        0 if vocab > 1 => {
            // Ties: every other logit the first.
            for i in (0..vocab).step_by(2) {
                r[i] = r[0];
            }
        }
        1 => r.iter_mut().for_each(|x| *x = 0.25),
        2 => r[(m.next() as usize) % vocab] = f64::NEG_INFINITY,
        3 => r[(m.next() as usize) % vocab] = f64::INFINITY,
        4 => r[(m.next() as usize) % vocab] = f64::NAN,
        5 => r.iter_mut().for_each(|x| *x = f64::NEG_INFINITY),
        _ => {}
    }
    r
}

/// The raw data of a buffer and its `presk_dtype`.
fn raw(b: &Buf) -> (*const c_void, i32) {
    match b {
        Buf::F64(v) => (v.as_ptr().cast(), F64),
        Buf::F32(v) => (v.as_ptr().cast(), F32),
        Buf::Bf16(v) => (v.as_ptr().cast(), BF16),
        other => panic!("a {} buffer", other.dtype()),
    }
}

#[test]
fn sample_draws_the_native_backends_tokens() {
    let t0 = Instant::now();
    let c = Stream::build("rng-sample");
    let (prog, core) = prepare("tests/examples/mlp/mlp.psk");
    let mut m = Mix(13);
    let mut rows_drawn = 0;
    let mut special = 0;
    for dtype in [Dtype::F64, Dtype::F32, Dtype::Bf16] {
        for seed in [0u64, 1, 42, 0xdead_beef] {
            let r = Realization::uniform(dtype);
            let mut b = presk_cranelift::native_backend(&core, &prog, &r).unwrap();
            for call in 0..400u64 {
                // A key per call, as a serve gives a token: each draws its
                // rows from the key's first uniforms.
                let key = Key::from_seed(seed).child(call);
                let vocab = [1, 2, 3, 7, 64, 92, 1000][(m.next() % 7) as usize];
                let rows = 1 + (m.next() % 4) as usize;
                let logits: Vec<f64> = (0..rows).flat_map(|_| row(&mut m, vocab)).collect();
                special += logits.iter().filter(|x| !x.is_finite()).count();
                let data = Buf::of(dtype, &logits);
                let t = Tensor { axes: vec![Axis { name: Some("batch".into()), len: rows }, Axis { name: Some("vocab".into()), len: vocab }], data: data.clone() };
                let want = match b.prim(&PrimOp::Sample, &[Value::Key(key.clone()), Value::Real(t)], &presk_types::ty::Ty::F64, &Default::default()).unwrap() {
                    Value::Idx { t, .. } => match t.data {
                        Buf::I64(v) => v.to_vec(),
                        other => panic!("sample gave a {} buffer", other.dtype()),
                    },
                    other => panic!("sample gave a {}", other.kind()),
                };
                let mut out = vec![-1i64; rows];
                let (ptr, dt) = raw(&data);
                unsafe { (c.sample)(out.as_mut_ptr(), ptr, dt, rows, vocab, &key.0) };
                assert_eq!(out, want, "{dtype} seed {seed}, {rows} rows of {vocab}: {logits:?}");
                rows_drawn += rows;
            }
        }
    }
    assert!(special > 100);
    eprintln!("{rows_drawn} rows sampled in f64, f32 and bf16 from 4800 keys, the native backend's tokens ({:.2} s)", t0.elapsed().as_secs_f64());
}
