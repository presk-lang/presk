//! The runtime of the C export, `c/presk_rt.c` (plan 0.9.0 §3.3, §3.4;
//! D78), against the Rust it stands for:
//! - `presk_convert` against `Buf::of(to, b.to_f64())`, the native backend's
//!   conversion at a kernel's edge, bit for bit between every two real
//!   types, `f64 → f32 → bf16`'s double rounding included;
//! - the `.psk.d` reader and writer against `presk-data`: files written in
//!   Rust read in C, into every real type, and files written in C read by
//!   `presk-data` and equal byte for byte to the ones it writes; and each
//!   error of the reader and the writer, with `presk-data`'s code;
//! - the arena: alignment, release, a kept result, exhaustion;
//! - the quantizer (plan 0.9.0 step 7, D84): `presk_quantize` against
//!   `QTensor::quantize` on over 10⁶ weights in each width and scheme, their
//!   integers and scales bit for bit, and `presk_dequantize` against the
//!   native backend's conversion of integer weights for a kernel of each
//!   type.

mod cexec;

use std::ffi::{CStr, CString, c_char, c_void};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

use cexec::{So, workdir};
use presk_data::{Layout, Leaf, Value};
use presk_rt::{Buf, Dtype};

/// `presk_leaf`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct CLeaf {
    bound: u64,
    n: u64,
}

/// `presk_reader`.
#[repr(C)]
struct CReader {
    f: *mut c_void,
    leaves: *const CLeaf,
    nleaves: usize,
    count: u64,
    next: u64,
}

/// `presk_arena`.
#[repr(C)]
struct CArena {
    base: *mut u8,
    size: usize,
    top: usize,
    peak: usize,
}

/// `presk_dtype`: `PRESK_F64`, `PRESK_F32`, `PRESK_BF16`, `PRESK_I64`.
const F64: i32 = 0;
const F32: i32 = 1;
const BF16: i32 = 2;
const I64: i32 = 3;
const REALS: [i32; 3] = [F64, F32, BF16];

/// The runtime's functions, from a library of `presk_rt.c` and
/// `presk_libm.c`.
struct Rt {
    _so: So,
    dir: PathBuf,
    convert: unsafe extern "C" fn(*mut c_void, i32, *const c_void, i32, usize),
    error: unsafe extern "C" fn() -> *const c_char,
    open: unsafe extern "C" fn(*mut CReader, *const c_char, *const c_char, *const CLeaf, usize) -> i32,
    read: unsafe extern "C" fn(*mut CReader, *const *mut c_void, *const i32) -> i32,
    close: unsafe extern "C" fn(*mut CReader),
    load: unsafe extern "C" fn(*const c_char, *const c_char, *const CLeaf, usize, *const *mut c_void, *const i32) -> i32,
    write: unsafe extern "C" fn(*const c_char, *const c_char, *const CLeaf, usize, u64, *const *const c_void, *const i32) -> i32,
    arena_init: unsafe extern "C" fn(*mut CArena, usize) -> i32,
    arena_free: unsafe extern "C" fn(*mut CArena),
    try_alloc: unsafe extern "C" fn(*mut CArena, usize) -> *mut c_void,
    mark: unsafe extern "C" fn(*const CArena) -> usize,
    release: unsafe extern "C" fn(*mut CArena, usize),
    release_keep: unsafe extern "C" fn(*mut CArena, usize, *const c_void, usize) -> *mut c_void,
    quantize: unsafe extern "C" fn(*mut c_void, *mut f32, *const f64, u64, u64, u64, i32, i32),
    dequantize: unsafe extern "C" fn(*mut c_void, i32, *const c_void, *const f32, i32, u64, u64, u64, usize),
}

/// A function of the library, as the pointer type of the field it fills.
fn sym<T: Copy>(so: &So, name: &str) -> T {
    assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<*mut c_void>());
    // SAFETY: `name` is declared in presk_rt.h with the signature `T` spells.
    unsafe { std::mem::transmute_copy::<*mut c_void, T>(&so.sym(name)) }
}

impl Rt {
    fn build(tag: &str) -> Rt {
        let dir = workdir(tag);
        let so = So::build(&dir, &[dir.join("presk_rt.c"), dir.join("presk_libm.c")]);
        Rt {
            dir,
            convert: sym(&so, "presk_convert"),
            error: sym(&so, "presk_error"),
            open: sym(&so, "presk_open"),
            read: sym(&so, "presk_read"),
            close: sym(&so, "presk_close"),
            load: sym(&so, "presk_load"),
            write: sym(&so, "presk_write"),
            arena_init: sym(&so, "presk_arena_init"),
            arena_free: sym(&so, "presk_arena_free"),
            try_alloc: sym(&so, "presk_try_alloc"),
            mark: sym(&so, "presk_mark"),
            release: sym(&so, "presk_release"),
            release_keep: sym(&so, "presk_release_keep"),
            quantize: sym(&so, "presk_quantize"),
            dequantize: sym(&so, "presk_dequantize"),
            _so: so,
        }
    }

    fn error(&self) -> String {
        // SAFETY: a NUL-terminated static buffer.
        unsafe { CStr::from_ptr((self.error)()) }.to_string_lossy().into_owned()
    }
}

fn cpath(p: &Path) -> CString {
    CString::new(p.as_os_str().as_bytes()).unwrap()
}

/// splitmix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// Reals that test a conversion: specials, NaNs with payloads, subnormals
/// of every type, ties of `f64 → f32` and of `f32 → bf16`, the `f64`s next
/// to a `bf16` tie (where rounding through `f32` differs from rounding
/// once), and random bits of every exponent.
fn reals(n: usize, seed: u64) -> Vec<f64> {
    let mut v = vec![
        0.0,
        -0.0,
        1.0,
        -1.5,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
        f64::from_bits(0x7ff8_dead_beef_0001),
        f64::from_bits(0x7ff0_0000_0000_0001), // signalling
        f64::from_bits(1),
        f64::MIN_POSITIVE,
        f64::MAX,
        f32::MAX as f64,
        f32::MAX as f64 * (1.0 + 2f64.powi(-24)), // rounds to f32::MAX or ∞
        f32::MIN_POSITIVE as f64,
        f32::from_bits(1) as f64,
        f32::from_bits(1) as f64 / 2.0,            // a tie with 0 in f32
        f32::from_bits(1) as f64 * 0.75,
    ];
    let mut rng = Rng(seed);
    // Ties: the midpoint of two neighbouring f32s, and of two bf16s, and
    // the f64s either side of the latter by less than an f32 ulp.
    for _ in 0..2000 {
        let f = f32::from_bits((rng.next() as u32) & 0x7f7f_ffff);
        let next = f32::from_bits(f.to_bits() + 1);
        v.push((f as f64 + next as f64) / 2.0);
        let b = f.to_bits() & 0xffff_0000;
        let mid = f32::from_bits(b | 0x8000) as f64; // an f32, halfway between two bf16
        let ulp = f32::from_bits(b | 0x8001) as f64 - mid;
        v.extend([mid, mid + ulp / 4.0, mid - ulp / 4.0, -mid, -(mid + ulp / 4.0)]);
    }
    while v.len() < n {
        let r = rng.next();
        v.push(f64::from_bits(r));
        v.push(f64::from_bits(r & 0x800f_ffff_ffff_ffff)); // subnormal
        let e = 0x380 + (rng.next() % 0x100); // around f32's range
        v.push(f64::from_bits((r & 0x800f_ffff_ffff_ffff) | (e << 52)));
    }
    v.truncate(n);
    v
}

fn dtype(d: i32) -> Dtype {
    match d {
        F64 => Dtype::F64,
        F32 => Dtype::F32,
        BF16 => Dtype::Bf16,
        _ => unreachable!(),
    }
}

/// A real buffer's raw bits.
fn bits(b: &Buf) -> Vec<u64> {
    match b {
        Buf::F64(x) => x.iter().map(|y| y.to_bits()).collect(),
        Buf::F32(x) => x.iter().map(|y| y.to_bits() as u64).collect(),
        Buf::Bf16(x) => x.iter().map(|y| *y as u64).collect(),
        Buf::I64(x) => x.iter().map(|y| *y as u64).collect(),
        _ => unreachable!(),
    }
}

/// The native backend's conversion of a buffer to a kernel of type `to`:
/// lent as it is when it has the type, `Buf::of(to, b.to_f64())` otherwise.
fn native_convert(b: &Buf, to: i32) -> Buf {
    if b.dtype() == dtype(to).name() { b.clone() } else { Buf::of(dtype(to), &b.to_f64()) }
}

/// An empty buffer of `n` elements of a type, and a pointer to its data.
fn buf(d: i32, n: usize) -> Buf {
    match d {
        F64 => Buf::F64(vec![0.0; n]),
        F32 => Buf::F32(vec![0.0; n]),
        BF16 => Buf::Bf16(vec![0; n]),
        _ => Buf::I64(vec![0; n]),
    }
}

fn ptr(b: &mut Buf) -> *mut c_void {
    match b {
        Buf::F64(x) => x.as_mut_ptr() as *mut c_void,
        Buf::F32(x) => x.as_mut_ptr() as *mut c_void,
        Buf::Bf16(x) => x.as_mut_ptr() as *mut c_void,
        Buf::I64(x) => x.as_mut_ptr() as *mut c_void,
        _ => unreachable!(),
    }
}

#[test]
fn conversions_are_the_native_backends() {
    let rt = Rt::build("rt-convert");
    let xs = reals(200_000, 1);
    // The f64s whose rounding to bf16 through f32 differs from a single
    // rounding: the inputs reach the case the double rounding decides.
    let once = |x: f64| -> u16 {
        // f64 → bf16 in one rounding, to nearest, ties to even (finite x
        // inside f32's normal range): x's two bf16 neighbours, found from
        // its f32.
        let f = x as f32;
        let lo = f32::from_bits(f.to_bits() & 0xffff_0000) as f64;
        let hi = f32::from_bits((f.to_bits() & 0xffff_0000) + 0x1_0000) as f64;
        let (dl, dh) = ((x - lo).abs(), (hi - x).abs());
        let pick = if dl < dh || (dl == dh && (f.to_bits() >> 16) & 1 == 0) { lo } else { hi };
        ((pick as f32).to_bits() >> 16) as u16
    };
    let double_rounded = xs.iter().filter(|x| x.is_finite() && x.abs() < 1e30 && x.abs() > 1e-30 && presk_rt::f32_to_bf16(**x as f32) != once(**x)).count();
    assert!(double_rounded > 1000, "only {double_rounded} inputs where rounding through f32 matters");
    let mut pairs = 0;
    for from in REALS {
        let src = Buf::of(dtype(from), &xs);
        for to in REALS {
            let want = native_convert(&src, to);
            let mut got = buf(to, xs.len());
            let mut s = src.clone();
            // SAFETY: both buffers hold xs.len() elements of their types.
            unsafe { (rt.convert)(ptr(&mut got), to, ptr(&mut s), from, xs.len()) };
            let (g, w) = (bits(&got), bits(&want));
            if let Some(i) = (0..g.len()).find(|&i| g[i] != w[i]) {
                panic!("presk_convert {} → {}: element {i} ({:e}, {:#x}): C {:#x}, native {:#x}", dtype(from).name(), dtype(to).name(), xs[i], bits(&src)[i], g[i], w[i]);
            }
            pairs += 1;
        }
    }
    eprintln!("presk_convert: {pairs} pairs of types on {} reals bit-identical to Buf::of, {double_rounded} of them where f64 → f32 → bf16 differs from one rounding", xs.len());
}

/// Sample types to write and read: every idx width, reals, a scalar, an
/// empty leaf, leaves larger than the reader's pieces, a parameter tree.
fn layouts() -> Vec<(String, Layout, u64)> {
    let real = |shape: &[usize]| Layout::Leaf(Leaf::Real { shape: shape.to_vec() });
    let idx = |bound: u64, shape: &[usize]| Layout::Leaf(Leaf::Idx { bound, shape: shape.to_vec() });
    vec![
        ("ℝ[batch: 2, model: 3]".into(), real(&[2, 3]), 1),
        (
            "(ℝ[2, 3], idx[5][4], idx[300][2], idx[70000][3], idx[5000000000][2], ℝ[], ℝ[0, 3], idx[1][2])".into(),
            Layout::Tuple(vec![real(&[2, 3]), idx(5, &[4]), idx(300, &[2]), idx(70000, &[3]), idx(5_000_000_000, &[2]), real(&[]), real(&[0, 3]), idx(1, &[2])]),
            3,
        ),
        ("(ℝ[10000], idx[65536][40000])".into(), Layout::Tuple(vec![real(&[10000]), idx(65536, &[40000])]), 2),
        (
            "params Tiny".into(),
            Layout::Record(vec![
                ("w".into(), real(&[4, 3])),
                ("b".into(), real(&[3])),
                ("blocks".into(), Layout::Record(vec![("ln".into(), real(&[2, 3])), ("head".into(), real(&[3, 5]))])),
            ]),
            1,
        ),
    ]
}

/// Values for a layout's samples: reals from `reals`, indices up to their
/// bound's extremes.
fn samples(layout: &Layout, count: u64, rng: &mut Rng) -> Vec<Value> {
    fn one(l: &Layout, rng: &mut Rng, xs: &mut impl Iterator<Item = f64>) -> Value {
        match l {
            Layout::Leaf(Leaf::Real { shape }) => Value::Real((0..shape.iter().product::<usize>()).map(|_| xs.next().unwrap()).collect()),
            Layout::Leaf(Leaf::Idx { bound, shape }) => Value::Idx(
                (0..shape.iter().product::<usize>())
                    .map(|i| match i % 3 {
                        0 => bound - 1,
                        1 => 0,
                        _ => rng.next() % bound,
                    })
                    .collect(),
            ),
            Layout::Tuple(ts) => Value::Tuple(ts.iter().map(|t| one(t, rng, xs)).collect()),
            Layout::Record(fs) => Value::Record(fs.iter().map(|(n, t)| (n.clone(), one(t, rng, xs))).collect()),
        }
    }
    let seed = rng.next();
    let mut xs = reals(1 << 20, seed).into_iter().cycle();
    (0..count).map(|_| one(layout, rng, &mut xs)).collect()
}

fn leaves(layout: &Layout) -> Vec<CLeaf> {
    layout
        .leaves()
        .iter()
        .map(|l| match l {
            Leaf::Idx { bound, .. } => CLeaf { bound: *bound, n: l.numel() as u64 },
            Leaf::Real { .. } => CLeaf { bound: 0, n: l.numel() as u64 },
        })
        .collect()
}

/// The leaves of a sample as buffers: reals in `f64`, indices in `i64`.
fn sample_bufs(v: &Value) -> Vec<Buf> {
    match v {
        Value::Real(x) => vec![Buf::F64(x.clone())],
        Value::Idx(x) => vec![Buf::I64(x.iter().map(|i| *i as i64).collect())],
        Value::Tuple(ts) => ts.iter().flat_map(sample_bufs).collect(),
        Value::Record(fs) => fs.iter().flat_map(|(_, t)| sample_bufs(t)).collect(),
    }
}

/// A sample with its reals replaced by those of `bufs` (as `f64`).
fn with_reals(v: &Value, bufs: &mut impl Iterator<Item = Vec<f64>>) -> Value {
    match v {
        Value::Real(_) => Value::Real(bufs.next().unwrap()),
        Value::Idx(x) => {
            bufs.next();
            Value::Idx(x.clone())
        }
        Value::Tuple(ts) => Value::Tuple(ts.iter().map(|t| with_reals(t, bufs)).collect()),
        Value::Record(fs) => Value::Record(fs.iter().map(|(n, t)| (n.clone(), with_reals(t, bufs))).collect()),
    }
}

fn file_bytes(p: &Path) -> Vec<u8> {
    std::fs::read(p).unwrap()
}

/// Bitwise equality of two samples, NaNs included.
fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Real(x), Value::Real(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()),
        (Value::Idx(x), Value::Idx(y)) => x == y,
        (Value::Tuple(x), Value::Tuple(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same(p, q)),
        (Value::Record(x), Value::Record(y)) => x.len() == y.len() && x.iter().zip(y).all(|((m, p), (n, q))| m == n && same(p, q)),
        _ => false,
    }
}

#[test]
fn psk_d_files_are_presk_datas() {
    let rt = Rt::build("rt-files");
    let mut rng = Rng(7);
    let mut checked = 0;
    for (k, (ty, layout, count)) in layouts().into_iter().enumerate() {
        let cleaves = leaves(&layout);
        let values = samples(&layout, count, &mut rng);
        let ctype = CString::new(ty.as_str()).unwrap();

        // Rust writes, C reads, in every real type.
        let rust_file = rt.dir.join(format!("rust{k}.psk.d"));
        presk_data::write(&rust_file, &ty, &layout, &values).unwrap();
        for d in REALS {
            let dtypes: Vec<i32> = cleaves.iter().map(|l| if l.bound > 0 { I64 } else { d }).collect();
            let mut r = CReader { f: std::ptr::null_mut(), leaves: std::ptr::null(), nleaves: 0, count: 0, next: 0 };
            // SAFETY: the leaves outlive the reader, which is closed below.
            let e = unsafe { (rt.open)(&mut r, cpath(&rust_file).as_ptr(), ctype.as_ptr(), cleaves.as_ptr(), cleaves.len()) };
            assert_eq!(e, 0, "{ty}: {}", rt.error());
            assert_eq!(r.count, count);
            for (s, v) in values.iter().enumerate() {
                let want: Vec<Buf> = sample_bufs(v).iter().map(|b| if matches!(b, Buf::I64(_)) { b.clone() } else { native_convert(b, d) }).collect();
                let mut got: Vec<Buf> = cleaves.iter().zip(&dtypes).map(|(l, t)| buf(*t, l.n as usize)).collect();
                let ptrs: Vec<*mut c_void> = got.iter_mut().map(ptr).collect();
                // SAFETY: each pointer holds its leaf's elements in its type.
                let e = unsafe { (rt.read)(&mut r, ptrs.as_ptr(), dtypes.as_ptr()) };
                assert_eq!(e, 0, "{ty}, sample {s}: {}", rt.error());
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!(bits(g) == bits(w), "{ty} as {}: sample {s}, leaf {i} read differently", dtype(d).name());
                }
            }
            // A sample past the last.
            let mut none = buf(F64, 1);
            let dummy = vec![ptr(&mut none); cleaves.len()];
            let e = unsafe { (rt.read)(&mut r, dummy.as_ptr(), dtypes.as_ptr()) };
            assert_eq!(e, 9003, "{}", rt.error());
            unsafe { (rt.close)(&mut r) };
            checked += 1;
        }

        // C writes from f64, f32 and bf16 buffers; presk-data reads it, and
        // the bytes are those presk-data writes from the same reals.
        for d in REALS {
            let dtypes: Vec<i32> = cleaves.iter().map(|l| if l.bound > 0 { I64 } else { d }).collect();
            let mut held: Vec<Vec<Buf>> = values.iter().map(|v| sample_bufs(v).iter().map(|b| if matches!(b, Buf::I64(_)) { b.clone() } else { native_convert(b, d) }).collect()).collect();
            let expected: Vec<Value> = values.iter().zip(&held).map(|(v, bs)| with_reals(v, &mut bs.iter().map(|b| b.to_f64()))).collect();
            let src: Vec<*const c_void> = held.iter_mut().flat_map(|bs| bs.iter_mut().map(|b| ptr(b) as *const c_void)).collect();
            let c_file = rt.dir.join(format!("c{k}-{}.psk.d", dtype(d).name()));
            // SAFETY: count × leaves pointers, each to its leaf in its type.
            let e = unsafe { (rt.write)(cpath(&c_file).as_ptr(), ctype.as_ptr(), cleaves.as_ptr(), cleaves.len(), count, src.as_ptr(), dtypes.as_ptr()) };
            assert_eq!(e, 0, "{ty}: {}", rt.error());
            let rust_same = rt.dir.join(format!("rust{k}-{}.psk.d", dtype(d).name()));
            presk_data::write(&rust_same, &ty, &layout, &expected).unwrap();
            assert!(file_bytes(&c_file) == file_bytes(&rust_same), "{ty} from {}: the C file's bytes are not presk-data's", dtype(d).name());
            let mut reader = presk_data::Reader::open(&c_file).unwrap();
            if ty.starts_with("params ") {
                reader = reader.with_layout(layout.clone()).unwrap();
            }
            let back = reader.read_all().unwrap();
            assert!(back.len() == expected.len() && back.iter().zip(&expected).all(|(a, b)| same(a, b)), "{ty}: presk-data reads the C file differently");
            checked += 1;
        }
    }
    eprintln!("{checked} .psk.d round trips between presk_rt.c and presk-data, bit for bit and byte for byte");
}

#[test]
fn psk_d_errors_are_presk_datas() {
    let rt = Rt::build("rt-errors");
    let layout = Layout::Tuple(vec![Layout::Leaf(Leaf::Real { shape: vec![3] }), Layout::Leaf(Leaf::Idx { bound: 5, shape: vec![2] })]);
    let ty = "(ℝ[3], idx[5][2])";
    let cleaves = leaves(&layout);
    let good = rt.dir.join("good.psk.d");
    let v = Value::Tuple(vec![Value::Real(vec![1.0, 2.0, 3.0]), Value::Idx(vec![4, 0])]);
    presk_data::write(&good, ty, &layout, [&v, &v]).unwrap();
    let bytes = file_bytes(&good);
    let head = 4 + 2 + 4 + ty.len() + 8;

    let open = |path: &Path, ty: &str, leaves: &[CLeaf]| -> (i32, String) {
        let mut r = CReader { f: std::ptr::null_mut(), leaves: std::ptr::null(), nleaves: 0, count: 0, next: 0 };
        let t = CString::new(ty).unwrap();
        // SAFETY: as in the round trips.
        let e = unsafe { (rt.open)(&mut r, cpath(path).as_ptr(), t.as_ptr(), leaves.as_ptr(), leaves.len()) };
        unsafe { (rt.close)(&mut r) };
        (e, rt.error())
    };
    let variant = |name: &str, f: &dyn Fn(&mut Vec<u8>)| -> PathBuf {
        let mut b = bytes.clone();
        f(&mut b);
        let p = rt.dir.join(name);
        std::fs::write(&p, b).unwrap();
        p
    };
    let mut cases: Vec<(&str, i32, (i32, String))> = vec![
        ("missing", 9002, open(&rt.dir.join("none.psk.d"), ty, &cleaves)),
        ("short", 9002, open(&variant("short.psk.d", &|b| b.truncate(3)), ty, &cleaves)),
        ("magic", 9002, open(&variant("magic.psk.d", &|b| b[0] = b'Q'), ty, &cleaves)),
        ("version", 9002, open(&variant("version.psk.d", &|b| b[4] = 2), ty, &cleaves)),
        ("header", 9003, open(&variant("header.psk.d", &|b| b.truncate(head - 3)), ty, &cleaves)),
        ("type", 9005, open(&good, "(ℝ[3], idx[6][2])", &cleaves)),
        ("truncated", 9003, open(&variant("trunc.psk.d", &|b| b.truncate(b.len() - 1)), ty, &cleaves)),
        ("longer", 9005, open(&variant("long.psk.d", &|b| b.push(0)), ty, &cleaves)),
        ("leaves of other dimensions", 9005, open(&good, ty, &[CLeaf { bound: 0, n: 2 }, CLeaf { bound: 5, n: 2 }])),
        ("leaves of more elements", 9003, open(&good, ty, &[CLeaf { bound: 0, n: 4 }, CLeaf { bound: 5, n: 2 }])),
    ];
    // Presk-data agrees on the file-level cases it checks.
    for (name, code) in [("short.psk.d", "E9002"), ("magic.psk.d", "E9002"), ("version.psk.d", "E9002"), ("trunc.psk.d", "E9003")] {
        assert_eq!(presk_data::Reader::open(&rt.dir.join(name)).unwrap_err().code, code, "{name}");
    }
    // An index at its bound, in a file: presk-data's writer refuses to
    // write one, so the file is patched.
    let (mut x, mut i) = (vec![0f64; 3], vec![0i64; 2]);
    let dst = [x.as_mut_ptr() as *mut c_void, i.as_mut_ptr() as *mut c_void];
    let t = CString::new(ty).unwrap();
    let load = |path: &Path, dtypes: [i32; 2]| -> (i32, String) {
        // SAFETY: dst holds a leaf of 3 f64 and one of 2 i64.
        let e = unsafe { (rt.load)(cpath(path).as_ptr(), t.as_ptr(), cleaves.as_ptr(), 2, dst.as_ptr(), dtypes.as_ptr()) };
        (e, rt.error())
    };
    let one = rt.dir.join("one.psk.d");
    presk_data::write(&one, ty, &layout, [&v]).unwrap();
    let oob1 = rt.dir.join("oob1.psk.d");
    let mut b = file_bytes(&one);
    b[head + 3 * 8 + 1] = 5;
    std::fs::write(&oob1, b).unwrap();
    cases.push(("index at its bound", 9004, load(&oob1, [F64, I64])));
    cases.push(("two samples to presk_load", 9005, load(&good, [F64, I64])));
    cases.push(("reals held as indices", 9005, load(&one, [I64, I64])));
    let (e, _) = load(&one, [F64, I64]);
    assert_eq!(e, 0);
    assert_eq!((x.clone(), i.clone()), (vec![1.0, 2.0, 3.0], vec![4, 0]));
    // The writer refuses an index at its bound and a negative one, and
    // leaves no file.
    for (name, bad) in [("an index at its bound", 5i64), ("a negative index", -1)] {
        let (mut x, mut i) = (vec![1.0f64, 2.0, 3.0], vec![1i64, bad]);
        let src = [x.as_mut_ptr() as *const c_void, i.as_mut_ptr() as *const c_void];
        let p = rt.dir.join("refused.psk.d");
        // SAFETY: as above.
        let e = unsafe { (rt.write)(cpath(&p).as_ptr(), t.as_ptr(), cleaves.as_ptr(), 2, 1, src.as_ptr(), [F64, I64].as_ptr()) };
        cases.push((name, 9004, (e, rt.error())));
        assert!(!p.exists(), "{name}: a partial file is left");
    }
    for (name, want, (got, msg)) in &cases {
        assert_eq!(got, want, "{name}: {msg}");
        assert!(msg.starts_with(&format!("[E{want}] ")), "{name}: {msg}");
        eprintln!("  {name}: {msg}");
    }
}

#[test]
fn the_arena_bumps_releases_and_keeps() {
    let rt = Rt::build("rt-arena");
    let mut a = CArena { base: std::ptr::null_mut(), size: 0, top: 0, peak: 0 };
    // SAFETY: the arena is used through the functions only, and freed.
    unsafe {
        assert_eq!((rt.arena_init)(&mut a, 1000), 0);
        assert_eq!(a.size, 1024, "rounded up to 64");
        assert_eq!(a.base as usize % 64, 0);
        let p = (rt.try_alloc)(&mut a, 10) as *mut u8;
        assert_eq!(p, a.base);
        let mark = (rt.mark)(&a);
        assert_eq!(mark, 64);
        let q = (rt.try_alloc)(&mut a, 100) as *mut u8;
        assert_eq!(q, a.base.add(64));
        let r = (rt.try_alloc)(&mut a, 64) as *mut u8;
        assert_eq!(r, a.base.add(64 + 128));
        for k in 0..64 {
            *r.add(k) = k as u8;
        }
        assert_eq!(a.top, 256);
        // A function's temporaries go, its result is kept at the mark.
        let kept = (rt.release_keep)(&mut a, mark, r as *const c_void, 64) as *mut u8;
        assert_eq!(kept, a.base.add(64));
        assert!((0..64).all(|k| *kept.add(k) == k as u8));
        assert_eq!((a.top, a.peak), (128, 256));
        (rt.release)(&mut a, mark);
        assert_eq!(a.top, 64);
        // Exhaustion: NULL, and a message; what is left stays usable.
        assert!((rt.try_alloc)(&mut a, 1000).is_null());
        assert!(rt.error().contains("has 960 left, and 1000 were asked"), "{}", rt.error());
        assert!(!(rt.try_alloc)(&mut a, 960).is_null());
        assert_eq!(a.top, 1024);
        assert!((rt.try_alloc)(&mut a, 1).is_null());
        (rt.arena_free)(&mut a);
        assert!(a.base.is_null());
    }
}

/// The scale `QTensor::quantize` gives a channel whose largest magnitude is
/// `m`, as the test builds ties on it.
fn scale_for(m: f64, qmax: f64, pow2: bool) -> f64 {
    if m == 0.0 {
        return 0.0;
    }
    if !pow2 {
        return (m / qmax) as f32 as f64;
    }
    let mut s = 1.0f64;
    while m / (s / 2.0) <= qmax {
        s /= 2.0;
    }
    while m / s > qmax {
        s *= 2.0;
    }
    s as f32 as f64
}

/// The `k` elements of one channel, of one of nine kinds: drawn reals of
/// every exponent (the conversions' test values), ordinary weights, ties
/// `(n + ½)·s` on the channel's own scale, all zeros (signed), subnormals,
/// huge values and infinities, NaNs among weights, reals whose scale falls
/// below every `f32`, or a largest magnitude of exactly `qmax·2ʲ`, where
/// `pow2`'s loops stop on an equality.
fn channel(k: usize, kind: u64, qmax: f64, pow2: bool, rng: &mut Rng, pool: &[f64]) -> Vec<f64> {
    let unit = |rng: &mut Rng| (rng.next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
    match kind {
        0 => (0..k).map(|_| pool[(rng.next() % pool.len() as u64) as usize]).collect(),
        1 => (0..k).map(|_| unit(rng) * 0.05).collect(),
        2 => {
            // The largest magnitude first, then ties on the scale it gives.
            let m = 0.1 + unit(rng).abs();
            let s = scale_for(m, qmax, pow2);
            let mut v = vec![if rng.next() % 2 == 0 { m } else { -m }];
            while v.len() < k {
                let n = (rng.next() % (2 * qmax as u64 - 2)) as f64 - (qmax - 1.0);
                v.push((n + 0.5) * s);
            }
            v
        }
        3 => (0..k).map(|i| if i % 2 == 0 { 0.0 } else { -0.0 }).collect(),
        4 => (0..k).map(|_| f64::from_bits(rng.next() & 0x800f_ffff_ffff_ffff)).collect(),
        5 => (0..k).map(|_| [1e300, -f64::MAX, f64::INFINITY, 3.0e38, -1.0, f64::NEG_INFINITY][(rng.next() % 6) as usize] * unit(rng).abs().max(0.5)).collect(),
        6 => (0..k).map(|i| if i % 3 == 1 { f64::NAN } else { unit(rng) }).collect(),
        7 => (0..k).map(|_| unit(rng) * 1e-45).collect(),
        _ => {
            let m = qmax * 2f64.powi((rng.next() % 41) as i32 - 20);
            (0..k).map(|i| if i == 0 { -m } else { unit(rng) * m }).collect()
        }
    }
}

#[test]
fn quantization_is_presk_rts() {
    use presk_rt::{Axis, AxisRef, QTensor, Quant, QuantScheme, Tensor};
    let rt = Rt::build("rt-quant");
    let pool = reals(100_000, 7);
    // (outer, k, inner): the axis the scales are constant along in the
    // middle, of odd lengths too, so that int4 channels start mid-byte.
    let shapes = [(1, 1, 1), (3, 5, 13), (2, 16, 64), (7, 3, 1), (1, 17, 9), (4, 1, 5), (5, 13, 1), (1, 64, 64), (2, 9, 9), (3, 7, 11)];
    let mut rng = Rng(42);
    let (mut weights, mut ties, mut zero_channels, mut dequantized) = (0usize, 0usize, 0usize, 0usize);
    for width in [8u8, 4] {
        let qmax = ((1 << (width - 1)) - 1) as f64;
        for scheme in [QuantScheme::Absmax, QuantScheme::Pow2] {
            let pow2 = scheme == QuantScheme::Pow2;
            let mut these = 0;
            while these < 1_000_000 {
                for &(outer, k, inner) in &shapes {
                    // One kind per channel, the data row-major over (outer, k, inner).
                    let mut w = vec![0.0; outer * k * inner];
                    for o in 0..outer {
                        for i in 0..inner {
                            let kind = rng.next() % 9;
                            zero_channels += usize::from(kind == 3);
                            for (j, x) in channel(k, kind, qmax, pow2, &mut rng, &pool).into_iter().enumerate() {
                                w[(o * k + j) * inner + i] = x;
                            }
                        }
                    }
                    let axes = vec![Axis { name: None, len: outer }, Axis { name: None, len: k }, Axis { name: None, len: inner }];
                    let t = Tensor { axes, data: Buf::F64(w.clone()) };
                    let want = QTensor::quantize(&t, &Quant { bits: width, over: AxisRef::Pos(1), scheme }).unwrap();
                    let mut q = vec![0u8; want.data.len()];
                    let mut s = vec![0f32; outer * inner];
                    // SAFETY: the buffers are the sizes presk_rt.h states.
                    unsafe { (rt.quantize)(q.as_mut_ptr() as *mut c_void, s.as_mut_ptr(), w.as_ptr(), outer as u64, k as u64, inner as u64, width as i32, i32::from(pow2)) };
                    let what = format!("int{width} {scheme:?} [{outer}, {k}, {inner}]");
                    if let Some(c) = (0..s.len()).find(|&c| s[c].to_bits() != want.scales[c].to_bits()) {
                        panic!("{what}: scale {c}: C {:e}, presk-rt {:e}", s[c], want.scales[c]);
                    }
                    if let Some(b) = (0..q.len()).find(|&b| q[b] != want.data[b]) {
                        panic!("{what}: byte {b}: C {:#04x}, presk-rt {:#04x}, from {:?}", q[b], want.data[b], &w[b * (8 / width as usize)..((b + 1) * (8 / width as usize)).min(w.len())]);
                    }
                    for (e, x) in w.iter().enumerate() {
                        let sc = want.scale(e) as f64;
                        ties += usize::from(sc != 0.0 && (x / sc).abs().fract() == 0.5 && (x / sc).abs() < qmax);
                    }
                    these += w.len();
                    // Dequantized for a kernel of each type, from an element
                    // of the tensor on: the native backend's conversion.
                    let qb = want.to_tensor().data;
                    let from = (rng.next() % w.len() as u64) as usize;
                    let n = w.len() - from;
                    let values = qb.to_f64();
                    for to in REALS {
                        let expect = Buf::of(dtype(to), &values[from..]);
                        let mut got = buf(to, n);
                        // SAFETY: q and s hold the tensor, got n elements.
                        unsafe { (rt.dequantize)(ptr(&mut got), to, q.as_ptr() as *const c_void, s.as_ptr(), width as i32, k as u64, inner as u64, from as u64, n) };
                        assert!(bits(&got) == bits(&expect), "{what}: presk_dequantize to {} from element {from}", dtype(to).name());
                        dequantized += n;
                    }
                }
            }
            weights += these;
        }
    }
    assert!(ties > 100_000 && zero_channels > 1_000, "{ties} ties, {zero_channels} zero channels");
    eprintln!("presk_quantize: {weights} weights in int8 and int4, absmax and pow2, integers and scales bit-identical to QTensor::quantize ({ties} ties at n + ½, {zero_channels} zero channels, subnormals, huge values, NaNs); presk_dequantize: {dequantized} elements into the three types, bit-identical to Buf::of");
}
