//! Statement P4 of plan 0.13.0 (D138): the native column. Every kernel of
//! the entries `make plan-check` runs, in the three types and under the
//! plans of `examples/tests/plan/`, compiled as `presk run --backend native` compiles
//! it: every byte of its machine code lies in one range of one line of the
//! kernel's printed text, the ranges in order and none overlapping, and the
//! lines change no byte (a run's code is the same).

use std::path::{Path, PathBuf};

use presk_core::ad::Remat;
use presk_driver::Collect;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

/// The kernels of `entry` as `presk dump --stage kernel` lists them.
fn kernels(file: &str, entry: &str, flags: &str, plan: Option<&str>, realize: &str) -> Vec<presk_kernel::Kernel> {
    let w: Vec<&str> = flags.split_whitespace().collect();
    let (mut dims, mut consts) = (Vec::new(), Vec::new());
    for p in w.chunks(2) {
        let (n, v) = p[1].split_once('=').unwrap();
        match p[0] {
            "--dim" => dims.push((n.to_string(), v.parse::<u64>().unwrap())),
            _ => consts.push(p[1].to_string()),
        }
    }
    let plan = presk_driver::plan::parse_plan(plan.map(|p| format!("@{}", root().join(p).display())).as_deref(), realize).unwrap();
    let mut r = Collect::default();
    let (_, prog, mut core) = presk_driver::front::core(&root().join(file), &consts, false, &mut r).unwrap_or_else(|_| panic!("{file}"));
    let diags = presk_driver::ad::differentiate(&mut core, Remat::Store, plan.realization.groups_visible());
    assert!(!presk_syntax::diag::has_errors(&diags), "{file}");
    presk_driver::layout::kernels(&core, &prog, entry, dims, &plan, 0).unwrap_or_else(|e| panic!("{file} {entry}: {}", e.message)).kernels
}

#[test]
fn every_byte_of_every_kernel_has_its_line() {
    let entries = std::fs::read_to_string(root().join("examples/tests/plan/entries.txt")).unwrap();
    let (mut n, mut bytes, mut given, mut prologue, mut tail, mut relocs) = (0usize, 0usize, 0u32, 0u32, 0u32, 0usize);
    for line in entries.lines().filter(|l| !l.trim().is_empty()) {
        let mut parts = line.splitn(3, '|');
        let (file, entry, flags) = (parts.next().unwrap(), parts.next().unwrap(), parts.next().unwrap_or(""));
        let mut runs: Vec<(Option<&str>, &str, &str)> = vec![(None, "f64", flags), (None, "f32", flags), (None, "bf16", flags), (Some("examples/tests/plan/default.json"), "f64", flags)];
        // A tiled contraction in lanes, k-blocked, split and packed, and a
        // two-pass softmax: the attention's plan, at a T its tile divides.
        if entry == "SelfAttention.apply" {
            runs.push((Some("examples/tests/plan/attention.json"), "f32", "--const V=92 --const D=64 --const H=2 --const L=1 --dim B=2 --dim T=16"));
        }
        for (plan, realize, flags) in runs {
            let ks = kernels(file, entry, flags, plan, realize);
            let code = presk_cranelift::machine_code(&ks).unwrap();
            let plain = presk_cranelift::plain_code(&ks).unwrap();
            for (j, (k, m)) in ks.iter().zip(&code).enumerate() {
                let what = format!("{entry} ({file}, {realize}{}) k{j}", plan.map(|p| format!(", {p}")).unwrap_or_default());
                let lines = k.to_string().lines().count() as u32;
                assert_eq!(m.bytes, plain[j], "{what}: the lines changed the code");
                assert!(!m.bytes.is_empty(), "{what}: no code");
                let mut at = 0;
                for &(s, e, l) in &m.ranges {
                    assert!(s == at && e > s, "{what}: the ranges leave a gap or overlap at {at}: {:?}", m.ranges);
                    assert!((1..=lines).contains(&l), "{what}: line {l} of {lines}");
                    at = e;
                }
                assert_eq!(at as usize, m.bytes.len(), "{what}: the ranges stop at {at} of {}", m.bytes.len());
                assert!(!m.disasm.is_empty(), "{what}: no disassembly");
                // The bytes before linking (D145): every field the linker
                // writes is zero, and names a function of `libm`.
                for (a, b, t) in &m.relocs {
                    assert!(m.bytes[*a as usize..*b as usize].iter().all(|x| *x == 0), "{what}: the field {a}–{b} for {t} is not zero before linking");
                    assert!(["exp", "log", "tanh", "sin", "cos", "erf", "expf", "logf", "tanhf", "sinf", "cosf", "erff"].contains(&t.as_str()), "{what}: a relocation to {t}");
                    relocs += 1;
                }
                n += 1;
                bytes += m.bytes.len();
                for &(a, b) in &m.given {
                    if a == 0 {
                        prologue += b - a;
                    } else if b as usize == m.bytes.len() {
                        tail += b - a;
                    } else {
                        given += b - a;
                    }
                }
                if std::env::var("P4_SHOW").is_ok_and(|v| v == what) {
                    eprintln!("{k}\n{:?}\n{:?}\n{}", m.ranges, m.given, m.disasm);
                }
            }
        }
    }
    eprintln!("P4: {n} kernels, {bytes} bytes; Cranelift gave no line to {prologue} bytes of prologues (given to the header), {given} bytes inside the code and {tail} at its end (given to the line before them); {relocs} fields the linker writes, zero before linking");
    assert!(relocs > 0, "no kernel calls libm: the relocations are not tested");
    assert!(n > 200, "only {n} kernels");
}
