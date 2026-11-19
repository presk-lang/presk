//! FC1's C half (plan 0.18.2 step 3, D217): `presk_rt.h`'s conversions
//! between `f32` and binary16, between `f32` and `e8m15`, and from `f64`
//! to each give the runtime's bits (`presk_rt::codes`, which FC1 holds to
//! the formats' rounding on every input, D215), NaNs' payloads included,
//! under each C compiler and in both of binary16's forms: `_Float16`'s
//! casts where the compiler has them, and the integer form
//! (`-DPRESK_F16_INTEGER`).
//!
//! The C is `tests/fc1.c`, a program that converts each job's inputs block
//! by block and prints a hash of each block's outputs; this file computes
//! the same hashes from `presk_rt::codes` and compares them. A program, so
//! that an x86-64 build runs under an emulator as an aarch64 one runs
//! natively.
//!
//! - By default (`make test`), under the C compiler (`CC`, else `cc`): every
//!   binary16 code widened, every binary16 midpoint and the `f64`s on either
//!   side of it narrowed, and every 1009th input of the other jobs.
//! - Ignored (`cargo test --release -p presk-export-c --test codes --
//!   --ignored`): every input, the statement's (2³² `f32`s to each format,
//!   every code, every midpoint of both formats with its neighbours, 2³²
//!   random `f64`s), under `cc`, GCC 16.2 (`.gcc`, `experiments/gcc`),
//!   clang 23 (`make llvm`) and x86-64 through `zig cc` under qemu (`make
//!   zig`, `make qemu`), each where it is installed.
//!
//! Since plan 0.18.2 step 4 (D218), job 7 holds the overflow guard's test,
//! the flag a guarded store's conversion sets, to
//! `presk_rt::codes::overflows` on every `f32`, in both formats.

use std::path::{Path, PathBuf};
use std::process::Command;

use presk_rt::codes;

/// FNV-1a over a word's eight bytes, little-endian: `fc1.c`'s `hash_word`.
fn hash_word(mut h: u64, v: u64) -> u64 {
    for k in 0..8 {
        h ^= (v >> (8 * k)) & 0xff;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

const BLOCKS: [u64; 8] = [256, 256, 1, 16, 1, 8, 128, 256];

/// The value of positive code `c` of the IEEE format of `t` trailing bits
/// and bias `bias`, the overflow threshold for the code past the largest
/// finite: `fc1.c`'s `value`.
fn value(c: u64, t: u32, bias: i32) -> f64 {
    let (field, frac) = (c >> t, c & ((1 << t) - 1));
    let p2 = |e: i32| 2f64.powi(e);
    if field == 0 { frac as f64 * p2(1 - bias - t as i32) } else { ((1u64 << t) | frac) as f64 * p2(field as i32 - bias - t as i32) }
}

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// One block of a job, as `fc1.c` computes it, by `presk_rt::codes`: its
/// count of inputs and its hash.
fn block(job: u32, b: u64, step: u64) -> (u64, u64) {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut n = 0;
    match job {
        0 | 1 => {
            let mut i = b << 24;
            while i < (b + 1) << 24 {
                let x = f32::from_bits(i as u32);
                h = hash_word(h, if job == 0 { u64::from(codes::f32_to_f16(x)) } else { u64::from(codes::f32_to_e8m15(x)) });
                n += 1;
                i += step;
            }
        }
        2 => {
            for c in 0..1u32 << 16 {
                h = hash_word(h, u64::from(codes::f16_to_f32(c as u16).to_bits()));
                n += 1;
            }
        }
        3 => {
            let mut c = b << 20;
            while c < (b + 1) << 20 {
                h = hash_word(h, u64::from(codes::e8m15_to_f32(c as u32).to_bits()));
                n += 1;
                c += step;
            }
        }
        4 | 5 => {
            let (t, bias, top) = if job == 4 { (10, 15, 0x7c00) } else { (15, 127, 0x7f_8000) };
            let (lo, hi, by) = if job == 4 { (0, top, 1) } else { (b << 20, ((b + 1) << 20).min(top), step) };
            let mut c = lo;
            while c < hi {
                let m = (value(c, t, bias) + value(c + 1, t, bias)) / 2.0;
                let (below, above) = (f64::from_bits(m.to_bits() - 1), f64::from_bits(m.to_bits() + 1));
                for x in [m, below, above, -m, -below, -above] {
                    h = hash_word(h, if job == 4 { u64::from(codes::f64_to_f16(x)) } else { u64::from(codes::f64_to_e8m15(x)) });
                    n += 1;
                }
                c += by;
            }
        }
        7 => {
            let mut i = b << 24;
            while i < (b + 1) << 24 {
                let x = f64::from(f32::from_bits(i as u32));
                h = hash_word(h, u64::from(codes::overflows(presk_rt::Dtype::F16.format(), x)) | u64::from(codes::overflows(presk_rt::Dtype::E8M15.format(), x)) << 1);
                n += 1;
                i += step;
            }
        }
        _ => {
            let mut i = b << 24;
            while i < (b + 1) << 24 {
                let (a, r) = (mix(i.wrapping_mul(0x9e37_79b9_7f4a_7c15)), mix(i ^ 0x5851_f42d_4c95_7f2d));
                let e = (r >> 52) % 300;
                for x in [f64::from_bits(a), f64::from_bits((r & (1 << 63)) | (1023 - 160 + e) << 52 | (a & ((1 << 52) - 1)))] {
                    h = hash_word(h, u64::from(codes::f64_to_f16(x)));
                    h = hash_word(h, u64::from(codes::f64_to_e8m15(x)));
                    n += 1;
                }
                i += step;
            }
        }
    }
    (n, h)
}

/// Every block's line, `"job block count hash"`, as `fc1.c` prints them,
/// computed on every core.
fn expected(step: u64) -> Vec<String> {
    let units: Vec<(u32, u64)> = (0..8u32).flat_map(|j| (0..BLOCKS[j as usize]).map(move |b| (j, b))).collect();
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut lines = vec![String::new(); units.len()];
    std::thread::scope(|s| {
        for (part, chunk) in lines.chunks_mut(units.len().div_ceil(cores)).enumerate() {
            let units = &units;
            s.spawn(move || {
                let base = part * units.len().div_ceil(cores);
                for (k, line) in chunk.iter_mut().enumerate() {
                    let (j, b) = units[base + k];
                    let (n, h) = block(j, b, step);
                    *line = format!("{j} {b} {n} {h:016x}");
                }
            });
        }
    });
    lines
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

/// A compiler of `fc1.c`: its name for the report, the command that
/// compiles, and what runs the program (an emulator, or nothing).
struct Compiler {
    name: String,
    cc: Vec<String>,
    runner: Option<PathBuf>,
}

/// `CC` (else `cc`), and with `all` every other compiler installed in the
/// workspace: GCC 16.2, clang 23, and x86-64 through zig under qemu.
fn compilers(all: bool) -> Vec<Compiler> {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let mut v = vec![Compiler { name: cc.clone(), cc: vec![cc], runner: None }];
    if !all {
        return v;
    }
    let r = root();
    for (name, path) in [("GCC 16.2", r.join(".gcc/bin/gcc")), ("clang 23", r.join(".llvm/bin/clang"))] {
        if path.exists() {
            v.push(Compiler { name: name.into(), cc: vec![path.display().to_string()], runner: None });
        } else {
            eprintln!("{name}: {} is absent, not checked", path.display());
        }
    }
    let (zig, qemu) = (r.join(".venv-zig/bin/python"), r.join(".qemu/bin/qemu-x86_64-static"));
    if zig.exists() && qemu.exists() {
        let cc = [zig.display().to_string(), "-m".into(), "ziglang".into(), "cc".into(), "-target".into(), "x86_64-linux-musl".into()].to_vec();
        v.push(Compiler { name: "x86-64 (zig cc, under qemu)".into(), cc, runner: Some(qemu) });
    } else {
        eprintln!("x86-64: zig ({}) or qemu ({}) is absent, not checked", zig.display(), qemu.display());
    }
    v
}

/// `fc1.c` built by `c` in binary16's default form and in its integer
/// form, into `dir`: the programs, by form.
fn build(c: &Compiler, dir: &Path) -> Vec<PathBuf> {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("presk_rt.h"), presk_export_c::PRESK_RT_H).unwrap();
    std::fs::copy(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fc1.c"), dir.join("fc1.c")).unwrap();
    let mut out = Vec::new();
    for (tag, extra) in [("default", None), ("integer", Some("-DPRESK_F16_INTEGER"))] {
        let program = dir.join(format!("fc1-{tag}"));
        let mut cmd = Command::new(&c.cc[0]);
        cmd.args(&c.cc[1..]).args(presk_export_c::CFLAGS).args(presk_export_c::WARNINGS).arg("-Werror").args(extra).arg("-o").arg(&program).arg("fc1.c").arg("-lm").current_dir(dir);
        let o = cmd.output().unwrap_or_else(|e| panic!("{}: {e}", c.name));
        assert!(o.status.success(), "{}, {tag}: {}", c.name, String::from_utf8_lossy(&o.stderr));
        out.push(program);
    }
    out
}

/// The program's lines, run in one process per core, and the form of
/// binary16 it printed.
fn run(c: &Compiler, program: &Path, step: u64) -> (String, Vec<String>) {
    let parts = std::thread::available_parallelism().map_or(4, |n| n.get());
    let children: Vec<_> = (0..parts)
        .map(|p| {
            let mut cmd = match &c.runner {
                Some(r) => {
                    let mut x = Command::new(r);
                    x.arg(program);
                    x
                }
                None => Command::new(program),
            };
            cmd.args([p.to_string(), parts.to_string(), step.to_string()]).stdout(std::process::Stdio::piped()).spawn().unwrap()
        })
        .collect();
    let mut form = String::new();
    let mut lines = Vec::new();
    for child in children {
        let o = child.wait_with_output().unwrap();
        assert!(o.status.success(), "{}: {}", c.name, program.display());
        for l in String::from_utf8(o.stdout).unwrap().lines() {
            match l.strip_prefix("form ") {
                Some(f) => form = f.to_string(),
                None => lines.push(l.to_string()),
            }
        }
    }
    let key = |l: &String| l.split(' ').take(2).map(|x| x.parse::<u64>().unwrap()).collect::<Vec<_>>();
    lines.sort_by_key(key);
    (form, lines)
}

const JOBS: [&str; 8] = ["f32 to binary16", "f32 to e8m15", "binary16 codes to f32", "e8m15 codes to f32", "binary16's midpoints from f64", "e8m15's midpoints from f64", "random f64s to both", "the guard's test of every f32"];

/// FC1 under each compiler, both forms, at `step`: no block's hash differs.
fn holds(all: bool, step: u64) {
    let want = expected(step);
    let total: u64 = want.iter().map(|l| l.split(' ').nth(2).unwrap().parse::<u64>().unwrap()).sum();
    let base = Path::new(env!("CARGO_TARGET_TMPDIR")).join("presk-export-c").join(format!("fc1-{step}"));
    for (i, c) in compilers(all).iter().enumerate() {
        for program in build(c, &base.join(i.to_string())) {
            let (form, got) = run(c, &program, step);
            let differ: Vec<String> = want.iter().zip(&got).filter(|(w, g)| w != g).map(|(w, _)| {
                let job: usize = w.split(' ').next().unwrap().parse().unwrap();
                format!("{} block {}", JOBS[job], w.split(' ').nth(1).unwrap())
            }).collect();
            assert_eq!(got.len(), want.len(), "{} ({form}): {} blocks, {} expected", c.name, got.len(), want.len());
            assert!(differ.is_empty(), "{} ({form}, {}): the C's conversions differ from presk_rt::codes in {}", c.name, program.display(), differ.join(", "));
            eprintln!("{} ({form} binary16, {}): {total} conversions in {} blocks, presk_rt::codes' bits", c.name, program.file_name().unwrap().to_string_lossy(), want.len());
        }
    }
}

#[test]
fn the_c_conversions_are_the_runtime_s_on_a_sample() {
    holds(false, 1009);
}

#[test]
#[ignore = "FC1 on every input under every compiler installed: minutes, a release build on 20 cores"]
fn the_c_conversions_are_the_runtime_s_on_every_input_under_each_compiler() {
    holds(true, 1);
}
