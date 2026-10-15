//! Parameters stored in their own types (plan 0.12.0 step 2, D121; statement
//! V2): a realization's `params` rounds a parameter once, to its type, and
//! every nest reads it from there, widened by its loads where the kernel's
//! type is wider (`BufKind::Wide`), with no copy at the kernel's edge.
//!
//! On the test set (`tests/plan/entries.txt`), the tutorial's loss and the
//! Llama decoder at tiny constants, with every parameter typed:
//! - on each executor, the realization gives the bits the same record
//!   without `params` gives on the parameters rounded to their types before
//!   the run, when no parameter is wider than the default (rounded once, and
//!   read exactly);
//! - the kernel interpreter and the native backend agree bit for bit
//!   wherever they agree without `params` (they differ by their libraries'
//!   `exp` and `log` in the last bits, D78).

use std::path::{Path, PathBuf};
use std::sync::Once;

use presk_cranelift::native_backend;
use presk_interp::RunOptions;
use presk_kernel::KernelBackend;
use presk_rt::{Backend, Dtype, Realization, Value};
use presk_syntax::diag::render_all;
use presk_syntax::{SourceMap, parse_file};
use presk_types::check;
use presk_types::elab::{self, ElabProgram, Options};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn chdir_root() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| std::env::set_current_dir(repo_root()).unwrap());
}

fn prepare(rel: &str, consts: &[(String, u64)]) -> (ElabProgram, presk_core::Program) {
    chdir_root();
    let mut sm = SourceMap::new();
    let id = sm.load(Path::new(rel)).unwrap();
    let (program, diags) = parse_file(&sm, id);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let opts = Options { const_overrides: consts.to_vec() };
    let (prog, diags) = elab::elaborate(&mut sm, id, program, &opts);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (typed, diags) = check::check(&sm, &prog);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (mut core, diags) = presk_core::lower(&prog, &typed);
    assert!(diags.is_empty(), "{}", render_all(&diags, &sm, false));
    let diags = presk_core::ad::transform(&mut core);
    assert!(diags.is_empty(), "{}", render_all(&diags, &sm, false));
    presk_core::verify(&core).unwrap();
    (prog, core)
}

struct Case {
    file: String,
    entry: String,
    consts: Vec<(String, u64)>,
    dims: Vec<(String, u64)>,
}

/// `--const N=V --dim N=V` as the test set's lines give them.
fn case(file: &str, entry: &str, args: &str) -> Case {
    let (mut consts, mut dims) = (Vec::new(), Vec::new());
    let mut words = args.split_whitespace();
    while let (Some(flag), Some(kv)) = (words.next(), words.next()) {
        let (k, v) = kv.split_once('=').unwrap();
        let pair = (k.to_string(), v.parse().unwrap());
        if flag == "--const" { consts.push(pair) } else { dims.push(pair) }
    }
    Case { file: file.into(), entry: entry.into(), consts, dims }
}

fn cases() -> Vec<Case> {
    chdir_root();
    let mut out: Vec<Case> = std::fs::read_to_string("tests/plan/entries.txt")
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut p = l.splitn(3, '|');
            case(p.next().unwrap(), p.next().unwrap(), p.next().unwrap_or(""))
        })
        .collect();
    out.push(case("tutorial/07_train.psk", "loss", "--dim B=4"));
    // The Llama decoder at tiny constants: RMSNorm, RoPE's rotation, the
    // grouped attention, SwiGLU, the head (V2's second clause).
    let llama = "--const V=100 --const D=32 --const H=2 --const G=2 --const FF=64 --const L=2 --dim B=2 --dim T=16";
    out.push(case("coverage/llama.psk", "Llama.apply", llama));
    out.push(case("coverage/llama.psk", "loss", llama));
    out
}

/// Runs `entry` under `r` on `b`, on `args` as `presk run` passes them.
fn run(b: &mut dyn Backend, c: &Case, func: presk_core::FuncId, fdims: &presk_interp::eval::Dims, r: &Realization, mut args: Vec<Value>) -> Value {
    r.store_params(&mut args).unwrap_or_else(|e| panic!("{} under {}: {}", c.entry, r.label(), e.message));
    let p = b.prepare(func, fdims, r).unwrap_or_else(|e| panic!("{} under {}: {}", c.entry, r.label(), e.message));
    b.call(&p, args).unwrap_or_else(|e| panic!("{} under {} on {}: {}", c.entry, r.label(), b.name(), e.message))
}

#[test]
fn parameters_in_their_own_types_are_rounded_once_on_every_executor() {
    let (mut typed_cases, mut compared, mut differ) = (0, 0, 0);
    for c in cases() {
        let (prog, core) = prepare(&c.file, &c.consts);
        let opts = RunOptions { entry: c.entry.clone(), dims: c.dims.clone(), seed: 0, bits: 128, digits: 17 };
        let (func, fdims, inputs) = presk_interp::run::prepare_run(&core, &prog, &opts).unwrap();
        let args: Vec<Value> = inputs.iter().map(|(_, v)| Value::from_interp(v).unwrap()).collect();
        let Some(theta @ Value::Record(_)) = args.first() else { continue };
        let paths = presk_rt::quant::param_paths(theta);
        for (default, own) in [(Dtype::F64, Dtype::F32), (Dtype::F64, Dtype::Bf16), (Dtype::F32, Dtype::Bf16), (Dtype::Bf16, Dtype::F32)] {
            let plain = Realization::uniform(default);
            let mut r = plain.clone();
            r.params = paths.iter().map(|p| (p.clone(), own)).collect();
            // The same parameters, rounded to their types before the run.
            let mut rounded = args.clone();
            rounded[0] = r.stored_params(&args[0]).unwrap();
            let what = format!("{} {} under {}", c.file, c.entry, r.label());
            let mut outs = Vec::new();
            for native in [false, true] {
                let mut b: Box<dyn Backend> = if native { Box::new(native_backend(&core, &prog, &r).unwrap()) } else { Box::new(KernelBackend::new(&core, &prog, &r)) };
                let got = run(b.as_mut(), &c, func, &fdims, &r, args.clone());
                if own.bytes() <= default.bytes() {
                    let mut b0: Box<dyn Backend> = if native { Box::new(native_backend(&core, &prog, &plain).unwrap()) } else { Box::new(KernelBackend::new(&core, &prog, &plain)) };
                    let want = run(b0.as_mut(), &c, func, &fdims, &plain, rounded.clone());
                    assert_eq!(got, want, "{what} on {}: not the bits of the parameters rounded once", b.name());
                    outs.push(Some(want));
                } else {
                    outs.push(None);
                }
                outs.push(Some(got));
            }
            // [kernel plain-rounded?, kernel, native plain-rounded?, native]
            let (kp, k, np, n) = (&outs[0], &outs[1], &outs[2], &outs[3]);
            if let (Some(kp), Some(np)) = (kp, np) {
                compared += 1;
                if kp == np {
                    assert_eq!(k, n, "{what}: the kernel interpreter and the native backend differ where they agree on the rounded parameters");
                } else {
                    differ += 1;
                }
            }
            typed_cases += 1;
        }
    }
    eprintln!("{typed_cases} realizations typing every parameter; {compared} compared across the two executors, {differ} where their libraries already differ");
    assert!(typed_cases >= 40, "{typed_cases}");
}

/// V3: a parameter stored in a type no wider than every nest that reads it
/// is never copied at a kernel's edge. On the tutorial's loss and the Llama
/// decoder at tiny constants, under an `f64` default whose nests are all
/// `f64` and under an `f32` one, with every parameter in `bf16` or `f32`,
/// the host's conversion phase (`presk bench --host-profile`'s
/// `buffers.convert`) moves no byte; and each parameter at rest is its
/// elements times its type's width.
#[test]
fn a_parameter_in_its_own_type_is_not_converted_at_an_edge() {
    use presk_kernel::profile::{HostProfile, Phase};
    for c in cases().into_iter().filter(|c| c.file == "tutorial/07_train.psk" || c.file == "coverage/llama.psk") {
        let (prog, core) = prepare(&c.file, &c.consts);
        let opts = RunOptions { entry: c.entry.clone(), dims: c.dims.clone(), seed: 0, bits: 128, digits: 17 };
        let (func, fdims, inputs) = presk_interp::run::prepare_run(&core, &prog, &opts).unwrap();
        let args: Vec<Value> = inputs.iter().map(|(_, v)| Value::from_interp(v).unwrap()).collect();
        let paths = presk_rt::quant::param_paths(&args[0]);
        for (default, own) in [(Dtype::F64, Dtype::Bf16), (Dtype::F64, Dtype::F32), (Dtype::F32, Dtype::Bf16)] {
            let mut r = Realization::uniform(default);
            r.params = paths.iter().map(|p| (p.clone(), own)).collect();
            let stored = r.realize_args(&args).unwrap();
            let mut at_rest = 0;
            stored[0].map_leaves_at(&mut |_, l| {
                if let Value::Real(t) = l {
                    assert_eq!(t.data.dtype(), own.name());
                    at_rest += 1;
                }
                l.clone()
            });
            assert_eq!(at_rest, paths.len());
            let mut b = native_backend(&core, &prog, &r).unwrap();
            let profile = HostProfile::shared();
            b.set_host_profile(Some(profile.clone()));
            let p = Backend::prepare(&mut b, func, &fdims, &r).unwrap();
            Backend::call(&mut b, &p, stored).unwrap();
            let converted = profile.borrow().bytes_of(Phase::Convert);
            assert_eq!(converted, 0, "{} {} under {}: {converted} bytes converted at kernels' edges", c.file, c.entry, r.label());
        }
    }
}
