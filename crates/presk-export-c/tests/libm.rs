//! The transcendental functions of the C export, `c/presk_libm.c` (plan
//! 0.9.0 §3.5, rule 2; D78), against the `libm` crate, whose functions the
//! native backend calls: each of the twelve, compiled with the export's
//! flags, gives the crate's bits on 10⁶ inputs over its domain. The inputs
//! include the special values (±0, ±∞, NaNs with payloads), subnormals, the
//! thresholds of each function's branches (overflow, underflow, the ends of
//! its approximations, the range reductions of `sin` and `cos`) with their
//! neighbours, multiples of π/4 and of ln 2 / 2 where the reductions change
//! quadrant or exponent, and large arguments, then values drawn over the
//! function's domain and over every finite number.
//!
//! An `#[ignore]`d test runs the six `f32` functions on every `f32`.

mod cexec;

use std::time::Instant;

use cexec::{So, workdir};

const N: usize = 1_000_000;

/// splitmix64: a fixed stream, so that a failure names the same input on
/// every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * ((self.next() >> 11) as f64 / (1u64 << 53) as f64)
    }
}

/// The inputs of a function of one type, as raw bits: a type's width and
/// its special values, the branch thresholds of the function and draws over
/// its domain.
trait Bits {
    const EXP_SHIFT: u32;
    const SIGN: u64;
    fn from_f64(x: f64) -> u64;
    fn specials() -> Vec<u64>;
}

struct D;
struct F;

impl Bits for D {
    const EXP_SHIFT: u32 = 52;
    const SIGN: u64 = 1 << 63;
    fn from_f64(x: f64) -> u64 {
        x.to_bits()
    }
    fn specials() -> Vec<u64> {
        vec![
            0,                     // +0
            0x7ff0_0000_0000_0000, // +∞
            0x7ff8_0000_0000_0000, // the default NaN
            0x7ff8_dead_beef_0001, // a quiet NaN with a payload
            0x7ff0_0000_0000_0001, // a signalling NaN
            0x7ff4_dead_beef_0001, // another, with a payload
            1,                     // the least subnormal
            0x000f_ffff_ffff_ffff, // the greatest subnormal
            0x0010_0000_0000_0000, // the least normal
            0x7fef_ffff_ffff_ffff, // the greatest finite
            1.0f64.to_bits(),
            0.5f64.to_bits(),
            2.0f64.to_bits(),
        ]
    }
}

impl Bits for F {
    const EXP_SHIFT: u32 = 23;
    const SIGN: u64 = 1 << 31;
    fn from_f64(x: f64) -> u64 {
        (x as f32).to_bits() as u64
    }
    fn specials() -> Vec<u64> {
        vec![
            0,
            0x7f80_0000,
            0x7fc0_0000,
            0x7fc0_beef,
            0x7f80_0001,
            0x7fa0_beef,
            1,
            0x007f_ffff,
            0x0080_0000,
            0x7f7f_ffff,
            1.0f32.to_bits() as u64,
            0.5f32.to_bits() as u64,
            2.0f32.to_bits() as u64,
        ]
    }
}

/// A set of inputs being built: each value is added with both signs.
struct Inputs<T: Bits> {
    v: Vec<u64>,
    rng: Rng,
    _t: std::marker::PhantomData<T>,
}

impl<T: Bits> Inputs<T> {
    fn new(seed: u64) -> Self {
        let mut s = Inputs { v: Vec::with_capacity(N), rng: Rng(seed), _t: std::marker::PhantomData };
        for b in T::specials() {
            s.around(b, 4);
        }
        s
    }

    fn push(&mut self, b: u64) {
        let mask = (T::SIGN << 1).wrapping_sub(1);
        let b = b & mask & !T::SIGN;
        self.v.push(b);
        self.v.push(b | T::SIGN);
    }

    /// The magnitude with these bits and its `k` neighbours on each side.
    fn around(&mut self, b: u64, k: u64) {
        let mask = (T::SIGN << 1).wrapping_sub(1);
        for d in 0..=k {
            self.push(b.wrapping_add(d) & mask);
            self.push(b.wrapping_sub(d) & mask);
        }
    }

    fn around_f64(&mut self, x: f64, k: u64) {
        self.around(T::from_f64(x.abs()), k);
    }

    /// A threshold stated on the leading 32 bits of an `f64` (musl's high
    /// word, `hx`), or on the bits of an `f32`: the least magnitude with
    /// them and the greatest, and their neighbours.
    fn word(&mut self, h: u64) {
        if T::EXP_SHIFT == 52 {
            self.around(h << 32, 8);
            self.around((h << 32) | 0xffff_ffff, 8);
        } else {
            self.around(h, 8);
        }
    }

    /// `x·step` for `j` in `range`, each with its neighbours: where a
    /// reduction changes its quotient.
    fn multiples(&mut self, step: f64, range: std::ops::Range<i64>, k: u64) {
        for j in range {
            self.around_f64(j as f64 * step, k);
        }
    }

    /// Every power of two in the type, with its neighbours.
    fn powers_of_two(&mut self) {
        let emax = if T::EXP_SHIFT == 52 { 2046 } else { 254 };
        for e in 1..=emax {
            self.around(e << T::EXP_SHIFT, 2);
        }
        for s in 0..T::EXP_SHIFT {
            self.around(1 << s, 1); // the subnormal powers
        }
    }

    /// Fills up to `N`: 40 % any finite value, 5 % any bits (NaNs of every
    /// payload), the rest uniform over the intervals given, in turn.
    fn fill(mut self, intervals: &[(f64, f64)]) -> Vec<u64> {
        assert!(self.v.len() < N / 4, "{} fixed inputs", self.v.len());
        let exp_all = if T::EXP_SHIFT == 52 { 0x7ff } else { 0xff };
        let mask = (T::SIGN << 1).wrapping_sub(1);
        let fixed = self.v.len();
        let mut i = 0;
        while self.v.len() < N {
            let r = self.rng.next();
            let b = match i % 20 {
                0..=7 => {
                    // any finite value: a random exponent, subnormals included
                    let e = self.rng.next() % exp_all;
                    (r & !(exp_all << T::EXP_SHIFT) & mask) | (e << T::EXP_SHIFT)
                }
                8 => r & mask,
                _ => {
                    let (lo, hi) = intervals[i % intervals.len()];
                    T::from_f64(self.rng.uniform(lo, hi))
                }
            };
            self.v.push(b & mask);
            i += 1;
        }
        assert!(fixed > 0);
        self.v
    }
}

fn exp_inputs<T: Bits>() -> Vec<u64> {
    let mut s = Inputs::<T>::new(1);
    if T::EXP_SHIFT == 52 {
        // musl's thresholds, as its exp.c writes them.
        #[allow(clippy::excessive_precision)]
        let thresholds = [709.782712893383973096, -708.39641853226410622, -745.13321910194110842];
        for x in thresholds {
            s.around_f64(x, 64);
        }
        for h in [0x4086232b, 0x3fd62e42, 0x3ff0a2b2, 0x3e300000] {
            s.word(h);
        }
        s.multiples(std::f64::consts::LN_2 / 2.0, -2200..2100, 4);
    } else {
        for h in [0x42aeac50, 0x42b17218, 0x42cff1b5, 0x3eb17218, 0x3f851592, 0x39000000] {
            s.word(h);
        }
        s.multiples(std::f64::consts::LN_2 / 2.0, -310..260, 4);
    }
    let range = if T::EXP_SHIFT == 52 { (-746.0, 710.0) } else { (-104.0, 89.0) };
    s.fill(&[range, (-1.0, 1.0), (-0.5, 0.5)])
}

fn log_inputs<T: Bits>() -> Vec<u64> {
    let mut s = Inputs::<T>::new(2);
    s.powers_of_two();
    s.around_f64(1.0, 1024);
    if T::EXP_SHIFT == 52 {
        for h in [0x00100000, 0x3fe6a09e, 0x3ff6a09e, 0x7ff00000] {
            s.word(h);
        }
    } else {
        for h in [0x00800000, 0x3f3504f3, 0x3fb504f3, 0x7f800000] {
            s.word(h);
        }
    }
    s.fill(&[(0.0, 4.0), (0.5, 2.0), (0.0, 1e6)])
}

fn tanh_inputs<T: Bits>() -> Vec<u64> {
    let mut s = Inputs::<T>::new(3);
    // tanh's thresholds, and expm1's at half the argument, since tanh calls
    // expm1(2|x|).
    if T::EXP_SHIFT == 52 {
        for h in [0x3fe193ea, 0x40340000, 0x3fd058ae, 0x00100000] {
            s.word(h);
        }
        for h in [0x4043687a, 0x40862e42, 0x3fd62e42, 0x3ff0a2b2, 0x3c900000] {
            s.word(h);
            s.word(h - 0x00100000);
        }
    } else {
        for h in [0x3f0c9f54, 0x41200000, 0x3e82c578, 0x00800000] {
            s.word(h);
        }
        for h in [0x4195b844, 0x42b17217, 0x3eb17218, 0x3f851592, 0x33000000] {
            s.word(h);
            s.word(h - 0x00800000);
        }
    }
    s.multiples(std::f64::consts::LN_2 / 4.0, -200..200, 4);
    s.fill(&[(-25.0, 25.0), (-1.0, 1.0), (-0.3, 0.3)])
}

/// `sin` and `cos` share their inputs: the range reduction's thresholds,
/// multiples of π/4 and of π/2, and large arguments.
fn trig_inputs<T: Bits>() -> Vec<u64> {
    let mut s = Inputs::<T>::new(4);
    if T::EXP_SHIFT == 52 {
        for h in [0x3fe921fb, 0x3e500000, 0x3e46a09e, 0x400f6a7a, 0x4002d97c, 0x401c463b, 0x4015fdbc, 0x4012d97c, 0x401921fb, 0x413921fb, 0x7ff00000] {
            s.word(h);
        }
        // The double nearest a multiple of π/2 (Muller, "Elementary
        // Functions"), where the large reduction must be right to the last
        // bit.
        s.around_f64(6381956970095103.0 * 2f64.powi(797), 8);
        s.multiples(std::f64::consts::FRAC_PI_4, -5000..5000, 2);
        s.multiples(std::f64::consts::FRAC_PI_2 * 1021.0, -1100..1100, 2); // up to 2^20·π/2
    } else {
        for h in [0x3f490fda, 0x39800000, 0x407b53d1, 0x4016cbe3, 0x40e231d5, 0x40afeddf, 0x4dc90fdb, 0x7f800000] {
            s.word(h);
        }
        s.multiples(std::f64::consts::FRAC_PI_4, -5000..5000, 2);
        s.multiples(std::f64::consts::FRAC_PI_2 * 262147.0, -1100..1100, 2); // up to 2^28·π/2
    }
    for e in 0..(if T::EXP_SHIFT == 52 { 1024 } else { 128 }) {
        s.around_f64(2f64.powi(e), 1);
    }
    let medium = if T::EXP_SHIFT == 52 { 1.7e6 } else { 4.3e8 };
    s.fill(&[(-10.0, 10.0), (-medium, medium), (-1.0, 1.0)])
}

fn erf_inputs<T: Bits>() -> Vec<u64> {
    let mut s = Inputs::<T>::new(5);
    if T::EXP_SHIFT == 52 {
        for h in [0x3feb0000, 0x3e300000, 0x3ff40000, 0x4006db6d, 0x40180000, 0x7ff00000] {
            s.word(h);
        }
    } else {
        for h in [0x3f580000, 0x31800000, 0x3fa00000, 0x4036db6d, 0x40c00000, 0x7f800000] {
            s.word(h);
        }
    }
    s.fill(&[(-7.0, 7.0), (-1.3, 1.3), (-0.9, 0.9)])
}

type F64Fn = unsafe extern "C" fn(f64) -> f64;
type F32Fn = unsafe extern "C" fn(f32) -> f32;
type RustF32 = fn(f32) -> f32;

/// The twelve functions, compiled and loaded.
struct Libm {
    so: So,
}

impl Libm {
    fn build(tag: &str) -> Libm {
        let dir = workdir(tag);
        Libm { so: So::build(&dir, &[dir.join("presk_libm.c")]) }
    }

    fn f64(&self, name: &str) -> F64Fn {
        // SAFETY: `presk_<name>` is declared `double (double)` in presk_rt.h.
        unsafe { std::mem::transmute::<*mut libc::c_void, F64Fn>(self.so.sym(&format!("presk_{name}"))) }
    }

    fn f32(&self, name: &str) -> F32Fn {
        // SAFETY: `presk_<name>` is declared `float (float)` in presk_rt.h.
        unsafe { std::mem::transmute::<*mut libc::c_void, F32Fn>(self.so.sym(&format!("presk_{name}"))) }
    }
}

/// What one function's comparison found.
struct Found {
    name: &'static str,
    inputs: usize,
    differ: usize,
    first: Option<String>,
}

fn compare64(name: &'static str, c: F64Fn, rust: fn(f64) -> f64, inputs: &[u64]) -> Found {
    let mut differ = 0;
    let mut first = None;
    for &b in inputs {
        let x = f64::from_bits(b);
        // SAFETY: a pure function of one double.
        let got = unsafe { c(x) };
        let want = rust(x);
        if got.to_bits() != want.to_bits() {
            differ += 1;
            first.get_or_insert_with(|| format!("{name}({x:e}) [{b:#018x}]: C {got:e} [{:#018x}], the crate {want:e} [{:#018x}]", got.to_bits(), want.to_bits()));
        }
    }
    Found { name, inputs: inputs.len(), differ, first }
}

fn compare32(name: &'static str, c: F32Fn, rust: fn(f32) -> f32, inputs: &[u64]) -> Found {
    let mut differ = 0;
    let mut first = None;
    for &b in inputs {
        let x = f32::from_bits(b as u32);
        // SAFETY: a pure function of one float.
        let got = unsafe { c(x) };
        let want = rust(x);
        if got.to_bits() != want.to_bits() {
            differ += 1;
            first.get_or_insert_with(|| format!("{name}({x:e}) [{b:#010x}]: C {got:e} [{:#010x}], the crate {want:e} [{:#010x}]", got.to_bits(), want.to_bits()));
        }
    }
    Found { name, inputs: inputs.len(), differ, first }
}

fn report(found: &[Found]) {
    let bad: Vec<&Found> = found.iter().filter(|f| f.differ > 0).collect();
    for f in found {
        eprintln!("  {:6} {} inputs, {} differ{}", f.name, f.inputs, f.differ, f.first.as_ref().map(|s| format!(", first {s}")).unwrap_or_default());
    }
    assert!(bad.is_empty(), "{} of the twelve functions differ from the libm crate: {}", bad.len(), bad.iter().map(|f| f.first.as_deref().unwrap()).collect::<Vec<_>>().join("; "));
}

#[test]
fn the_twelve_functions_are_bit_identical_to_the_libm_crate() {
    let started = Instant::now();
    let lib = Libm::build("libm");
    let built = started.elapsed().as_secs_f64();
    let (e64, e32) = (exp_inputs::<D>(), exp_inputs::<F>());
    let (l64, l32) = (log_inputs::<D>(), log_inputs::<F>());
    let (h64, h32) = (tanh_inputs::<D>(), tanh_inputs::<F>());
    let (t64, t32) = (trig_inputs::<D>(), trig_inputs::<F>());
    let (r64, r32) = (erf_inputs::<D>(), erf_inputs::<F>());
    let drawn = started.elapsed().as_secs_f64() - built;
    let found = vec![
        compare64("exp", lib.f64("exp"), libm::exp, &e64),
        compare32("expf", lib.f32("expf"), libm::expf, &e32),
        compare64("log", lib.f64("log"), libm::log, &l64),
        compare32("logf", lib.f32("logf"), libm::logf, &l32),
        compare64("tanh", lib.f64("tanh"), libm::tanh, &h64),
        compare32("tanhf", lib.f32("tanhf"), libm::tanhf, &h32),
        compare64("sin", lib.f64("sin"), libm::sin, &t64),
        compare32("sinf", lib.f32("sinf"), libm::sinf, &t32),
        compare64("cos", lib.f64("cos"), libm::cos, &t64),
        compare32("cosf", lib.f32("cosf"), libm::cosf, &t32),
        compare64("erf", lib.f64("erf"), libm::erf, &r64),
        compare32("erff", lib.f32("erff"), libm::erff, &r32),
    ];
    eprintln!(
        "presk_libm.c against the libm crate: {} inputs, built in {built:.1} s, drawn in {drawn:.1} s, compared in {:.1} s",
        found.iter().map(|f| f.inputs).sum::<usize>(),
        started.elapsed().as_secs_f64() - built - drawn
    );
    for f in &found {
        assert_eq!(f.inputs, N, "{}", f.name);
    }
    report(&found);
}

/// The six `f32` functions on every one of the 2³² `f32`s, on every core
/// (`cargo test -p presk-export-c --test libm -- --ignored`).
#[test]
#[ignore = "every f32: 2^32 inputs for each of six functions"]
fn the_f32_functions_are_bit_identical_to_the_libm_crate_on_every_f32() {
    let started = Instant::now();
    let lib = Libm::build("libm-every-f32");
    let fns: [(&'static str, RustF32); 6] = [("expf", libm::expf), ("logf", libm::logf), ("tanhf", libm::tanhf), ("sinf", libm::sinf), ("cosf", libm::cosf), ("erff", libm::erff)];
    let par = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) as u64;
    let mut found = Vec::new();
    for (name, rust) in fns {
        let c = lib.f32(name);
        let chunk = (1u64 << 32).div_ceil(par);
        let parts: Vec<(usize, Option<u32>)> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..par)
                .map(|t| {
                    sc.spawn(move || {
                        let (lo, hi) = (t * chunk, ((t + 1) * chunk).min(1 << 32));
                        let mut differ = 0;
                        let mut first = None;
                        for b in lo..hi {
                            let x = f32::from_bits(b as u32);
                            // SAFETY: a pure function of one float.
                            if unsafe { c(x) }.to_bits() != rust(x).to_bits() {
                                differ += 1;
                                first.get_or_insert(b as u32);
                            }
                        }
                        (differ, first)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let differ = parts.iter().map(|p| p.0).sum();
        let first = parts.iter().find_map(|p| p.1).map(|b| {
            let x = f32::from_bits(b);
            // SAFETY: as above.
            let got = unsafe { c(x) };
            format!("{name}({x:e}) [{b:#010x}]: C {got:e} [{:#010x}], the crate {:e} [{:#010x}]", got.to_bits(), rust(x), rust(x).to_bits())
        });
        found.push(Found { name, inputs: 1 << 32, differ, first });
        eprintln!("  {name}: {:.0} s so far", started.elapsed().as_secs_f64());
    }
    eprintln!("presk_libm.c against the libm crate on every f32: {:.0} s", started.elapsed().as_secs_f64());
    report(&found);
}
