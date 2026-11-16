//! The image of every value of a format through the kernel interpreter's
//! emulation (plan 0.18.0 step 2, D205): the kernel `out = x·2^k`, lowered
//! as the compiler lowers a multiplication by a scalar, run in an arithmetic
//! format, from a buffer of every code of a format of at most 16 bits into a
//! buffer of another format. One line per code, in order: the code and the
//! code of its image, in hexadecimal; the counts on stderr. What a check
//! against another implementation of the formats reads, as MT1's against a
//! machine's simulator does.
//!
//! ```sh
//! cargo run --release -p presk-kernel --example convert -- FROM TO [--arith F] [--scale-exp K]
//! ```
//!
//! FROM is a float or an integer format of at most 16 bits, its codes in
//! order (an integer's in two's complement); TO is any format, an integer's
//! code its two's complement on its bits, a fixed-point number's that of its
//! units; the arithmetic is `f32` unless `--arith` names another, and `k` is
//! 0 unless given. `i16 i8/sat --scale-exp -3` is the rescaling of an `i16`
//! by a right shift of three bits, ties to even, clamped to `i8`.

use std::io::Write;

use presk_core::PrimOp;
use presk_format::{Format, F32};
use presk_kernel::lower::{lower, Operand};
use presk_kernel::{BufKind, Dtype};
use presk_rt::{Axis, Buf};

fn usage() -> ! {
    eprintln!("usage: convert FROM TO [--arith FORMAT] [--scale-exp K]");
    std::process::exit(2);
}

fn format(s: &str) -> Format {
    Format::parse(s).unwrap_or_else(|e| {
        eprintln!("convert: {e}");
        std::process::exit(2)
    })
}

/// The value a code of `f` denotes.
fn value(f: Format, c: u64) -> f64 {
    match f {
        Format::Float(x) => x.decode(c),
        Format::Int(i) if i.signed && c >> (i.bits - 1) & 1 == 1 => (c as i64 - (1i64 << i.bits)) as f64,
        Format::Int(_) => c as f64,
        Format::Fixed(_) | Format::Affine(_) => unreachable!("a fixed-point or an affine source"),
    }
}

/// The code of a value of `f`.
fn code(f: Format, v: f64) -> u64 {
    let mask = |n: i128, bits: u32| (n as u64) & (u64::MAX >> (64 - bits));
    match f {
        Format::Float(x) => x.encode(v).expect("the image is a value of its format"),
        Format::Int(i) => mask(v as i128, i.bits),
        Format::Fixed(q) => mask((v * 2f64.powi(q.frac)) as i128, q.int.bits),
        // `v = (q − z)·s` exactly.
        Format::Affine(a) => mask((v / a.scale()) as i128 + i128::from(a.zero), a.int.bits),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut from, mut to, mut arith, mut k) = (None, None, Format::Float(F32), 0i32);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--arith" => arith = format(it.next().unwrap_or_else(|| usage())),
            "--scale-exp" => k = it.next().and_then(|s| s.parse().ok()).unwrap_or_else(|| usage()),
            _ if from.is_none() => from = Some(format(a)),
            _ if to.is_none() => to = Some(format(a)),
            _ => usage(),
        }
    }
    let (Some(from), Some(to)) = (from, to) else { usage() };
    if matches!(from, Format::Fixed(_) | Format::Affine(_)) || from.bits() > 16 || !to.exact_in_f64() {
        eprintln!("convert: FROM is a float or an integer format of at most 16 bits, TO a format an f64 holds");
        std::process::exit(2);
    }
    let codes: Vec<u64> = (0..1u64 << from.bits()).collect();
    let xs: Vec<f64> = codes.iter().map(|c| value(from, *c)).collect();
    let ins = [Operand { axes: vec![Axis { name: Some("x".into()), len: xs.len() }], kind: BufKind::Real }, Operand { axes: vec![], kind: BufKind::Real }];
    let kernel = lower(&PrimOp::Mul, &ins, Dtype::F32).expect("a multiplication is a kernel").expect("it lowers").kernel;
    let (x, s) = (Buf::in_format(from, &xs), Buf::in_format(Format::Float(F32), &[2f64.powi(k)]));
    let mut out = Buf::in_format(to, &vec![0.0; xs.len()]);
    let counts = presk_kernel::exec::run_kernel_in(&kernel, arith, &[&x, &s], &mut out).unwrap_or_else(|e| {
        eprintln!("convert: {e}");
        std::process::exit(1)
    });
    let width = from.bits().div_ceil(4) as usize;
    let out_width = to.bits().div_ceil(4) as usize;
    let mut w = std::io::BufWriter::new(std::io::stdout().lock());
    for (c, y) in codes.iter().zip(out.to_f64()) {
        writeln!(w, "{c:0width$x} {:0out_width$x}", code(to, y)).unwrap();
    }
    w.flush().unwrap();
    eprintln!("{from} to {to} in {arith}, scale 2^{k}: {} codes, {counts}", codes.len());
}
