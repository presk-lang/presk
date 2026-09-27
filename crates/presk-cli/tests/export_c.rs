//! `presk export --target c` (plan 0.9.0 step 3, D79), through the binary.
//!
//! - The options: `--entry` required, `--plan` read as `presk run` reads
//!   it and exclusive with `--realize`, the PyTorch export's options refused
//!   with this target and this target's with PyTorch, nothing written.
//!   (`--plan` and `--threads` were refused with `E8002` until step 5.)
//! - The refusals of an entry, each with its code: unbound dimensions
//!   (`E8003`), an initializer (`E8004`), a key in the result (`E8006`),
//!   integer weights in the result or under a reshape (`E8007`, every
//!   integer weight until step 7), a scalar in the result (`E8008`), and
//!   with `--serve` a model whose parameters do not fix the context's length
//!   (`E8009`). `sample` was refused (`E8005`) until step 6.
//! - Statements C1 and C2 (§3.8): every entry of `tests/plan/entries.txt`
//!   (`next_token`, which samples, since step 6), in `f64`, `f32` and
//!   `bf16`, exported, built by its own
//!   `Makefile` and run by its own `conformance/check.sh` as a subprocess,
//!   its outputs compared with the native backend's bit for bit (C1). Each
//!   program is exported twice, with the structured host (step 4) and with
//!   the straight-line one (`--straight-line`, step 3), whose outputs must
//!   be the same file byte for byte. Each
//!   entry whose first argument is a module's parameters runs twice: on the
//!   parameters `presk run` synthesizes, and with every parameter an
//!   initializer sets to zero drawn instead (`--weights`), since on the
//!   synthesized ones MiniGPT's and the course GPT's head is zero and so
//!   are their logits. C2 is reported: the elements outside the conformance
//!   band of the exact reference, the C program's and the native backend's,
//!   which C1 makes the same.
//! - Every program is built with GCC with OpenMP and without it (statement
//!   C3's part on GCC, `-Wall -Wextra`, no warning) and run both ways; a
//!   program with a launch divided among threads also on one thread
//!   (`OMP_NUM_THREADS=1`): the same bits each time (step 5).
//! - Threads through the binary (step 5): MiniGPT's forward pass under a
//!   plan stating splits at three depths (`tests/export_c/minigpt-splits.json`),
//!   and at a size where the native backend divides kernels by its own rule
//!   (`--threads 4`).
//! - Statement C5 (step 6): `presk export --serve` of `serve_minigpt.psk`,
//!   its `generate` against `presk serve --backend native`, greedy and
//!   sampled at several temperatures and seeds, in the three types, through
//!   both hosts: the same `--json` document, byte for byte. The check is
//!   itself checked by three mutations of `generate.c`, each of which
//!   changes the tokens: a draw skipped, the stream seeded again at each
//!   step, the window one token behind.
//! - Integer weights (step 7, D84): C1 on `tests/quant/odd.psk` in `int8`
//!   and `int4`, `absmax` and `pow2`, in the three types; on MiniGPT's
//!   forward pass and `next_token` with the head, the embeddings and every
//!   matrix quantized, its gradient and an optimizer step, on synthesized
//!   parameters and with their zeros drawn;
//!   at `D = 9`, `H = 3` in `int4`, where a layer's attention matrices are
//!   odd and the host copies their slices to `f64`; with the biases and the
//!   norms' parameters quantized (an `add` that loads integers, a fusion
//!   group that reads their values); under a plan stating splits; C5 with
//!   `--serve`. The check is checked by three mutations of the program, each
//!   of which changes the outputs: a scale one channel off, the two nibbles
//!   of a byte swapped, the scale kept in `f64` when quantizing.
//! - In `#[ignore]`d tests (`make c-conformance`):
//!   - C5 and C1 on the tiny MiniGPT's trained checkpoint,
//!     `out/ckpt/final.psk.d` (`make train-examples`), in the three types;
//!   - the course GPT at its constants (V = 92, D = 384, H = 6, L = 6): with
//!     its zeros drawn at B = 1, T = 8, and with the course's trained
//!     checkpoint at the dimensions it was trained at, B = 1, T = 256, where
//!     the exact reference would take hours (`--no-exact`); and at B = 16,
//!     T = 256, statement C4's size, under the plan `presk plan` gives with
//!     the frozen profile, which divides most launches among ten threads,
//!     and under the default layout, where the native backend divides the
//!     largest by its own rule;
//!   - every entry of the test set under the plan `presk plan` gives with the
//!     frozen profile (`~/presk-data/ra4-model-0.8.0/profile.json`, or
//!     `RA4_FROZEN`), in the three types, MiniGPT's also under the mixed
//!     realization, and on integer weights;
//!   - integer weights on trained checkpoints (step 7): the tiny MiniGPT's
//!     forward pass, `next_token` and C5, and the course GPT at `T = 256`, in
//!     `int8`;
//!   - with `PRESK_C3=1`, the rest of statement C3 on every program: Clang
//!     and the x86-64 cross-compiler, both `zig cc` of `.venv-zig` (`make
//!     zig`), without a warning. On aarch64, Clang's program is linked and
//!     run, and gives the native bits; the x86-64 program is linked (musl,
//!     static) and not run, no x86-64 machine or emulator being at hand.
//!     `zig` ships no OpenMP runtime, so with `-fopenmp` both compile each
//!     file of the program without linking it, against GCC's `omp.h`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("presk-export-c-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn presk(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_presk")).current_dir(root()).args(args).env("NO_COLOR", "1").output().expect("running presk")
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

// ----- options -------------------------------------------------------------

#[test]
fn the_options_of_the_c_target_are_checked_before_anything_is_written() {
    let out = scratch("options");
    let o = out.to_str().unwrap();
    let cases: &[(&[&str], &str)] = &[
        (&["export", "tests/examples/mlp/mlp.psk", "--target", "c", "-o", o], "--entry is required"),
        (&["export", "tests/examples/mlp/mlp.psk", "--target", "c", "--entry", "mse", "--plan", "@absent.json", "-o", o], "presk: absent.json: No such file"),
        (&["export", "tests/examples/mlp/mlp.psk", "--target", "c", "--entry", "mse", "--plan", "@absent.json", "--realize", "f32", "-o", o], "cannot be used with"),
        (&["export", "tests/examples/mlp/mlp.psk", "--target", "c", "--entry", "mse", "--mode", "graph", "-o", o], "`--mode graph` is a mode of the PyTorch export"),
        (&["export", "tests/examples/mlp/mlp.psk", "--target", "c", "--entry", "mse", "--conformance-entry", "mse", "-o", o], "`--conformance-entry` is an option of the PyTorch export"),
        (&["export", "tests/examples/mlp/mlp.psk", "--target", "c", "--entry", "mse", "--train", "-o", o], "`--train` is an option of the PyTorch export"),
        (&["export", "tests/examples/mlp/mlp.psk", "--target", "pytorch", "--entry", "mse", "-o", o], "`--entry` is an option of --target c"),
        (&["export", "tests/examples/mlp/mlp.psk", "--target", "pytorch", "--realize", "f32", "-o", o], "`--realize` is an option of --target c"),
        (&["export", "tests/examples/mlp/mlp.psk", "--target", "pytorch", "--straight-line", "-o", o], "`--straight-line` is an option of --target c"),
        (&["export", SERVE, "--target", "c", "--serve", "--const", "V=64", "--const", "D=16", "--const", "H=2", "--const", "L=2", "--dim", "B=2", "--dim", "T=8", "-o", o], "presk serve generates one sequence"),
    ];
    for (args, expected) in cases {
        let r = presk(args);
        let err = stderr(&r);
        assert_eq!(r.status.code(), Some(2), "{args:?}: {err}");
        assert!(err.contains(expected), "{args:?}: {err}");
        assert!(!out.exists(), "{args:?}: {} was written", out.display());
    }
}

#[test]
fn a_program_with_errors_reports_them_and_writes_nothing() {
    let out = scratch("bad");
    let r = presk(&["export", "tests/diag/shape_rank.psk", "--target", "c", "--entry", "f", "-o", out.to_str().unwrap()]);
    let err = stderr(&r);
    assert_eq!(r.status.code(), Some(1), "{err}");
    assert!(err.contains("[E2003]"), "{err}");
    assert!(!out.exists(), "{} was written", out.display());
}

// ----- refusals ----------------------------------------------------------------

#[test]
fn an_entry_the_c_program_cannot_run_is_refused_with_its_code() {
    let q8 = scratch("q8.json");
    std::fs::write(&q8, r#"{"default":"f32","order":{"kind":"sequential"},"quant":{"w":{"bits":8,"over":0}}}"#).unwrap();
    let q8 = format!("@{}", q8.display());
    let quant = "tests/diag/export_c_quant.psk";
    let cases: Vec<(Vec<&str>, &str)> = vec![
        (vec!["tests/examples/mlp/mlp.psk", "--entry", "mse", "--dim", "D=4"], "[E8003] Error: `mse` is exported at bound dimensions only: unbound dimension B"),
        (vec!["tests/examples/mlp/mlp.psk", "--entry", "Linear{4,4}.init"], "[E8004] Error: the entry calls the initializer `xavier_uniform`"),
        (vec!["tests/diag/export_c_key.psk", "--entry", "halves"], "[E8006] Error: a key reaches the entry's result"),
        (vec![quant, "--entry", "L.same", "--realize", &q8], "[E8007] Error: the entry's result holds integer weights"),
        (vec![quant, "--entry", "L.flat", "--realize", &q8], "[E8007] Error: `merge r, c` reshapes integer weights"),
        (vec!["tests/diag/export_c_scalar.psk", "--entry", "next"], "[E8008] Error: the entry's result holds a nat"),
        (vec!["tests/export_c/serve_untied.psk", "--serve", "--dim", "T=4"], "[E8009] Error: `next` is served over the whole context"),
    ];
    for (args, expected) in cases {
        let out = scratch("refused");
        let mut all = vec!["export", "--target", "c", "-o", out.to_str().unwrap()];
        all.extend(args.iter().copied());
        let r = presk(&all);
        let err = stderr(&r);
        assert_eq!(r.status.code(), Some(1), "{args:?}: {err}");
        assert!(err.contains(expected), "{args:?}: {err}");
        assert!(!out.exists(), "{args:?}: {} was written", out.display());
    }
}

// ----- C1 and C2 -------------------------------------------------------------------

/// The conformance band of `presk-kernel/tests/conformance.rs` (`tol`):
/// `|x − x̂| ≤ ε·(|x| + a)` per element, for a uniform realization with
/// additions in order; a mixed one is given `bf16`'s, its narrowest type's.
fn band(realize: &str) -> (f64, f64) {
    match realize {
        "f64" => (2f64.powi(-40), 2f64.powi(-50)),
        "f32" => (2f64.powi(-15), 2f64.powi(-20)),
        _ => (2f64.powi(-4), 2f64.powi(-5)),
    }
}

/// The reals of a `.psk.d` file of a closed type, in order.
fn reals(path: &Path) -> Vec<f64> {
    fn flat(v: &presk_data::Value, out: &mut Vec<f64>) {
        match v {
            presk_data::Value::Real(d) => out.extend(d),
            presk_data::Value::Idx(d) => out.extend(d.iter().map(|x| *x as f64)),
            presk_data::Value::Tuple(vs) => vs.iter().for_each(|x| flat(x, out)),
            presk_data::Value::Record(fs) => fs.iter().for_each(|(_, x)| flat(x, out)),
        }
    }
    let mut r = presk_data::Reader::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut out = Vec::new();
    for v in r.read_all().unwrap() {
        flat(&v, &mut out);
    }
    out
}

/// A copy of a checkpoint with every value that is exactly zero replaced by
/// a small value drawn from a fixed sequence: a model's biases, shifts and
/// zero-initialized head, so that no output is zero by construction.
fn draw_zeros(from: &Path, to: &Path) {
    let bytes = std::fs::read(from).unwrap();
    let ty_len = u32::from_le_bytes(bytes[6..10].try_into().unwrap()) as usize;
    let data = 10 + ty_len + 8;
    let mut out = bytes[..data].to_vec();
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    for chunk in bytes[data..].chunks_exact(8) {
        let x = f64::from_le_bytes(chunk.try_into().unwrap());
        let y = if x == 0.0 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.2
        } else {
            x
        };
        out.extend_from_slice(&y.to_le_bytes());
    }
    std::fs::write(to, out).unwrap();
}

/// One exported program checked: C1 by its own script, C2 counted.
#[derive(Debug)]
struct Checked {
    what: String,
    /// C1, every way the program was built and run.
    c1: bool,
    /// The straight-line host's program: C1 too, and the same output file.
    flat: bool,
    /// Whether a launch is divided among threads.
    divided: bool,
    elements: usize,
    nonzero: usize,
    /// Elements outside the band: the C program's, the native backend's.
    outside: Option<(usize, usize)>,
    /// The largest `|x − x̂| / (|x| + a)` over the C program's outputs, in
    /// units of the band's `ε`.
    worst: f64,
    secs: f64,
}

/// How an entry is exported: `--realize R` or `--plan @P`, and the type
/// whose band C2 uses (a plan's realization's).
#[derive(Clone, Debug)]
struct Under {
    flags: Vec<String>,
    band: String,
}

fn realized(r: &str) -> Under {
    Under { flags: vec!["--realize".into(), r.into()], band: r.into() }
}

fn planned(plan: &Path, band: &str) -> Under {
    Under { flags: vec!["--plan".into(), format!("@{}", plan.display())], band: band.into() }
}

/// Runs the program built in `dir` as its `check.sh` runs it, into
/// `conformance/<name>.psk.d`, on `threads` threads when given, and compares
/// its outputs with the native backend's.
fn rerun(dir: &Path, name: &str, program: &str, threads: Option<usize>) -> Result<(), String> {
    let check = std::fs::read_to_string(dir.join("conformance/check.sh")).unwrap();
    let line = check.lines().find(|l| l.starts_with("./model ")).expect("check.sh runs ./model");
    let line = format!("{program}{}", &line["./model".len()..]).replace("conformance/out.psk.d", &format!("conformance/{name}.psk.d"));
    let mut cmd = Command::new("sh");
    cmd.arg("-c").arg(&line).current_dir(dir);
    if let Some(n) = threads {
        cmd.env("OMP_NUM_THREADS", n.to_string());
    }
    let run = cmd.output().unwrap();
    if !run.status.success() {
        return Err(format!("{line}: {}", String::from_utf8_lossy(&run.stderr)));
    }
    let same = std::fs::read(dir.join(format!("conformance/{name}.psk.d"))).ok() == std::fs::read(dir.join("conformance/native.psk.d")).ok();
    if same { Ok(()) } else { Err(format!("{line}: other bits than the native backend's")) }
}

/// Whether the program's `Makefile` builds it with OpenMP: whether a launch
/// is divided among threads.
fn divided(dir: &Path) -> bool {
    let make = std::fs::read_to_string(dir.join("Makefile")).unwrap();
    make.lines().any(|l| l.starts_with("OPENMP = -"))
}

/// The program of `dir`, built and run as `check.sh` did, run the other
/// ways (step 5): on one thread when it divides a launch; built the other
/// way by its `Makefile`, without OpenMP or with it, without a warning, and
/// run; and with `PRESK_C3=1`, statement C3's other compilers ([`c3`]).
/// The ways it was run, or the first that fails.
fn other_ways(dir: &Path) -> Result<Vec<&'static str>, String> {
    let mut ways = Vec::new();
    let omp = divided(dir);
    if omp {
        rerun(dir, "one-thread", "./model", Some(1))?;
        ways.push("one thread");
    }
    let other = if omp { "OPENMP=".to_string() } else { format!("OPENMP={}", presk_export_c::OPENMP) };
    let make = Command::new("make").args(["-s", "-B", "-C"]).arg(dir).arg(&other).arg("model").output().unwrap();
    let err = String::from_utf8_lossy(&make.stderr);
    if !make.status.success() || err.contains("warning") {
        return Err(format!("make {other}: {err}"));
    }
    rerun(dir, "other-build", "./model", None)?;
    ways.push(if omp { "without OpenMP" } else { "with OpenMP" });
    if std::env::var("PRESK_C3").is_ok_and(|v| v == "1") {
        c3(dir)?;
        ways.push("Clang and x86-64");
    }
    Ok(ways)
}

/// `zig cc`, the Clang and the x86-64 cross-compiler of statement C3: zig
/// from PyPI in `.venv-zig` (`make zig`).
fn zig() -> Command {
    let python = root().join(".venv-zig/bin/python");
    assert!(python.exists(), "{} is absent: `make zig` installs zig (PRESK_C3=1 asks for statement C3's compilers)", python.display());
    let mut c = Command::new(python);
    c.args(["-m", "ziglang", "cc"]);
    c
}

/// A directory holding GCC's `omp.h` alone, which `zig cc -fopenmp` reads
/// as a system header: zig ships no OpenMP.
fn omp_include() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let found = Command::new("gcc").arg("-print-file-name=include").output().unwrap();
        let header = PathBuf::from(String::from_utf8_lossy(&found.stdout).trim()).join("omp.h");
        assert!(header.is_absolute() && header.exists(), "GCC's omp.h not found: {}", header.display());
        let dir = std::env::temp_dir().join(format!("presk-export-c-{}-omp", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::copy(&header, dir.join("omp.h")).unwrap();
        dir
    })
}

/// zig's cache is shared by every `zig cc`, and runs that each compile
/// several C files deadlock on its locks when many run at once (zig 0.16.0:
/// two processes, each holding one object's lock and waiting for the
/// other's). So each run compiles one file, the objects are linked by a run
/// of their own, and the first program's runs are made alone, which fills
/// the cache with what every link needs (compiler_rt, the C libraries).
static ZIG_WARM: Mutex<bool> = Mutex::new(false);

/// Statement C3 on Clang and x86-64 (`zig cc`), `-Wall -Wextra -std=c11`
/// and the program's own flags, no warning: on aarch64 linked, run and
/// compared bit for bit; for x86_64-linux-musl linked; with `-fopenmp`,
/// the files that print OpenMP regions compiled, not linked.
fn c3(dir: &Path) -> Result<(), String> {
    let mut warm = ZIG_WARM.lock().unwrap();
    if *warm {
        drop(warm);
        return c3_runs(dir);
    }
    let r = c3_runs(dir);
    *warm = r.is_ok();
    r
}

fn c3_runs(dir: &Path) -> Result<(), String> {
    let flags: Vec<String> = presk_export_c::CFLAGS.iter().chain(presk_export_c::WARNINGS).map(|f| f.to_string()).collect();
    // A program exported with --serve has its driver too (step 6).
    let serve = dir.join("generate.c").exists();
    let files: &[&str] = if serve { &["main.c", "model.c", "presk_rt.c", "presk_libm.c", "generate.c"] } else { &["main.c", "model.c", "presk_rt.c", "presk_libm.c"] };
    let omp = omp_include();
    let zig_cc = |what: String, args: Vec<String>| -> Result<(), String> {
        let o = zig().args(&args).current_dir(dir).output().unwrap();
        let err = String::from_utf8_lossy(&o.stderr);
        if !o.status.success() || err.contains("warning") {
            return Err(format!("zig cc, {what}: {err}"));
        }
        Ok(())
    };
    for (target, tag) in [(None, "clang"), (Some("x86_64-linux-musl"), "x86_64")] {
        let t: Vec<String> = target.map(|t| vec!["-target".to_string(), t.to_string()]).unwrap_or_default();
        let mut objects = Vec::new();
        for f in files {
            let object = format!("{}-{tag}.o", f.trim_end_matches(".c"));
            zig_cc(format!("{tag}, {f}"), [t.clone(), flags.clone(), vec!["-c".into(), "-o".into(), object.clone(), f.to_string()]].concat())?;
            objects.push(object);
        }
        let (model, generate): (Vec<String>, Vec<String>) = (objects.iter().filter(|o| !o.starts_with("generate")).cloned().collect(), objects.iter().filter(|o| !o.starts_with("main")).cloned().collect());
        zig_cc(format!("{tag}, linked"), [t.clone(), vec!["-o".into(), format!("model-{tag}")], model, vec!["-lm".into()]].concat())?;
        if serve {
            zig_cc(format!("{tag}, generate linked"), [t.clone(), vec!["-o".into(), format!("generate-{tag}")], generate, vec!["-lm".into()]].concat())?;
        }
        for f in files.iter().filter(|f| ["main.c", "model.c", "generate.c"].contains(f)) {
            let openmp = vec![presk_export_c::OPENMP.into(), "-isystem".into(), omp.display().to_string(), "-c".into(), "-o".into(), "/dev/null".into(), f.to_string()];
            zig_cc(format!("{tag}, {f} with OpenMP"), [t.clone(), flags.clone(), openmp].concat())?;
        }
    }
    rerun(dir, "clang", "./model-clang", None)
}

/// Exports a program into `out` and runs its own script: build, run, `cmp`
/// with the native outputs; then the other ways ([`other_ways`]). Whether C1
/// holds every way.
fn export_one(what: &str, file: &str, entry: &str, args: &[String], under: &Under, extra: &[&str], out: &Path) -> (bool, String) {
    let mut all: Vec<String> = ["export", file, "--target", "c", "--entry", entry].iter().map(|s| s.to_string()).collect();
    all.extend(under.flags.iter().cloned());
    all.push("-o".into());
    all.push(out.display().to_string());
    all.extend(args.iter().cloned());
    all.extend(extra.iter().map(|s| s.to_string()));
    let refs: Vec<&str> = all.iter().map(|s| s.as_str()).collect();
    let r = presk(&refs);
    assert!(r.status.success(), "{what}: presk export failed:\n{}", stderr(&r));
    let check = Command::new("sh").arg(out.join("conformance/check.sh")).env("PRESK", "/nonexistent").output().unwrap();
    let c1 = check.status.success() && String::from_utf8_lossy(&check.stdout).contains("C1 holds");
    // Built with -Wall -Wextra by its Makefile: no warning (C3, GCC's part).
    let built = String::from_utf8_lossy(&check.stderr);
    assert!(!built.contains("warning"), "{what}: the build warns:\n{built}");
    let log = format!("{}{}", String::from_utf8_lossy(&check.stdout), built);
    if !c1 {
        return (false, log);
    }
    match other_ways(out) {
        Ok(_) => (true, log),
        Err(e) => (false, format!("{log}{e}\n")),
    }
}

fn export_and_check(what: &str, file: &str, entry: &str, args: &[String], under: &Under, extra: &[&str], out: &Path) -> Checked {
    let t0 = Instant::now();
    let (c1, log) = export_one(what, file, entry, args, under, extra, out);
    // The straight-line host, the structured one's oracle: the same bytes.
    let flat_dir = out.with_extension("flat");
    let straight: Vec<&str> = extra.iter().copied().chain(std::iter::once("--straight-line")).collect();
    let (flat_c1, flat_log) = export_one(&format!("{what}, straight-line"), file, entry, args, under, &straight, &flat_dir);
    let same = |f: &str| std::fs::read(out.join(f)).ok() == std::fs::read(flat_dir.join(f)).ok();
    let flat = flat_c1 && same("conformance/out.psk.d") && same("conformance/native.psk.d");
    if !flat {
        eprintln!("{what}: the straight-line host's program differs:\n{flat_log}");
    }
    let conf = out.join("conformance");
    let native = reals(&conf.join("native.psk.d"));
    let mut checked = Checked { what: what.to_string(), c1, flat, divided: divided(out), elements: native.len(), nonzero: native.iter().filter(|x| **x != 0.0).count(), outside: None, worst: 0.0, secs: 0.0 };
    if !c1 {
        eprintln!("{what}: C1 fails:\n{log}");
    } else if conf.join("exact.psk.d").exists() {
        let (c, exact) = (reals(&conf.join("out.psk.d")), reals(&conf.join("exact.psk.d")));
        let (eps, a) = band(&under.band);
        // A NaN is outside.
        let inside = |g: f64, x: f64| (g - x).abs() <= eps * (x.abs() + a);
        let out_of = |got: &[f64]| got.iter().zip(&exact).filter(|(g, x)| !inside(**g, **x)).count();
        checked.outside = Some((out_of(&c), out_of(&native)));
        checked.worst = c.iter().zip(&exact).map(|(g, x)| (g - x).abs() / (x.abs() + a)).fold(0.0, f64::max) / eps;
    }
    checked.secs = t0.elapsed().as_secs_f64();
    checked
}

struct Case {
    file: String,
    entry: String,
    args: Vec<String>,
}

fn test_set() -> Vec<Case> {
    let text = std::fs::read_to_string(root().join("tests/plan/entries.txt")).unwrap();
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut p = l.splitn(3, '|');
            let (file, entry, args) = (p.next().unwrap(), p.next().unwrap(), p.next().unwrap_or(""));
            Case { file: file.into(), entry: entry.into(), args: args.split_whitespace().map(String::from).collect() }
        })
        .collect()
}

/// Runs `jobs` on every core, returns their results in order.
fn parallel<T: Send>(jobs: Vec<Box<dyn FnOnce() -> T + Send + '_>>) -> Vec<T> {
    let n = jobs.len();
    let queue = Mutex::new(jobs.into_iter().enumerate().collect::<Vec<_>>());
    let results = Mutex::new((0..n).map(|_| None).collect::<Vec<Option<T>>>());
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(n.max(1));
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                loop {
                    let Some((i, job)) = queue.lock().unwrap().pop() else { break };
                    let r = job();
                    results.lock().unwrap()[i] = Some(r);
                }
            });
        }
    });
    results.into_inner().unwrap().into_iter().map(|r| r.unwrap()).collect()
}

fn report(rows: &[Checked]) {
    eprintln!("{:<52} {:>4} {:>5} {:>7} {:>7} {:>7} {:>12} {:>9} {:>6}", "entry", "C1", "flat", "threads", "reals", "≠ 0", "outside C/N", "worst/ε", "s");
    for r in rows {
        let outside = r.outside.map(|(c, n)| format!("{c}/{n}")).unwrap_or_else(|| "—".into());
        eprintln!("{:<52} {:>4} {:>5} {:>7} {:>7} {:>7} {:>12} {:>9.3} {:>6.1}", r.what, if r.c1 { "ok" } else { "FAIL" }, if r.flat { "same" } else { "DIFF" }, if r.divided { "split" } else { "one" }, r.elements, r.nonzero, outside, r.worst, r.secs);
    }
}

fn assert_c1_c2(rows: &[Checked]) {
    for r in rows {
        assert!(r.c1, "{}: C1 fails", r.what);
        assert!(r.flat, "{}: the straight-line host's outputs are not the structured host's", r.what);
        // C2: outside the band only where the native backend is.
        if let Some((c, n)) = r.outside {
            assert!(c <= n, "{}: {c} elements outside the band, the native backend's {n}", r.what);
        }
    }
}

/// A realization of MiniGPT's programs with seven sites of other types
/// than its default `f32`: `f64` and `bf16` buffers meet at kernels'
/// edges, where the host converts them.
const MIXED: &str = "@tests/plan/mixed-realization.json";

/// C1 and C2 on the test set: every entry, three types, on the synthesized
/// parameters and with their zeros drawn; MiniGPT's five entries also under
/// a mixed realization.
#[test]
fn every_entry_of_the_test_set_computes_the_native_backends_bits() {
    let t0 = Instant::now();
    let base = scratch("matrix");
    let mut jobs: Vec<Box<dyn FnOnce() -> Vec<Checked> + Send>> = Vec::new();
    for case in test_set() {
        let minigpt = case.file.contains("minigpt");
        for realize in ["f64", "f32", "bf16", MIXED] {
            if realize == MIXED && !minigpt {
                continue;
            }
            let label = if realize == MIXED { "mixed" } else { realize };
            let dir = base.join(format!("{}-{label}", case.entry.replace(|c: char| !c.is_ascii_alphanumeric(), "_")));
            let case = Case { file: case.file.clone(), entry: case.entry.clone(), args: case.args.clone() };
            jobs.push(Box::new(move || both_weights(&format!("{} {label}", case.entry), &case, &realized(realize), &[], &dir)));
        }
    }
    let rows: Vec<Checked> = parallel(jobs).into_iter().flatten().collect();
    report(&rows);
    eprintln!("{} programs in {:.1} s", rows.len(), t0.elapsed().as_secs_f64());
    assert_c1_c2(&rows);
    // With its zeros drawn, no entry's outputs are all zero.
    for r in rows.iter().filter(|r| r.what.contains("drawn")) {
        assert!(r.nonzero > 0, "{}: every output is zero", r.what);
    }
    let _ = std::fs::remove_dir_all(&base);
}

/// A program exported on the parameters `presk run` synthesizes and, when
/// the entry's first argument is a module's parameters, with their zeros
/// drawn (`--weights`).
fn both_weights(what: &str, case: &Case, under: &Under, extra: &[&str], dir: &Path) -> Vec<Checked> {
    let mut rows = vec![export_and_check(what, &case.file, &case.entry, &case.args, under, extra, dir)];
    let weights = dir.join("conformance/weights.psk.d");
    if weights.exists() {
        let drawn = dir.with_extension("drawn.psk.d");
        draw_zeros(&weights, &drawn);
        let d = dir.with_extension("drawn");
        let extra: Vec<&str> = extra.iter().copied().chain(["--weights", drawn.to_str().unwrap()]).collect();
        rows.push(export_and_check(&format!("{what}, zeros drawn"), &case.file, &case.entry, &case.args, under, &extra, &d));
    }
    rows
}

/// Threads through the binary (step 5): MiniGPT's forward pass under a plan
/// stating splits at depths 0, 1 and 2 among two and four workers
/// (`--threads 4`, so that four is the machine's worker count, condition 5),
/// and at `D = 64`, `B = 4`, `T = 32`, where the native backend divides the
/// launches whose work reaches its threshold among `--threads 4` workers by
/// its own rule. Each exported and checked as the test set is: C1 every way
/// (on every thread, on one, without OpenMP), C2 where the exact reference
/// is affordable.
#[test]
fn launches_divided_among_threads_compute_the_native_backends_bits() {
    let t0 = Instant::now();
    let base = scratch("threads");
    let minigpt = |dims: &[&str]| Case {
        file: "tests/examples/minigpt/minigpt.psk".into(),
        entry: "MiniGPT.apply".into(),
        args: ["--const", "V=64", "--const", "H=2", "--const", "L=2"].iter().chain(dims).map(|s| s.to_string()).collect(),
    };
    let plan = root().join("tests/export_c/minigpt-splits.json");
    let jobs: Vec<Box<dyn FnOnce() -> Vec<Checked> + Send>> = vec![
        {
            let (base, plan) = (base.clone(), plan.clone());
            Box::new(move || both_weights("MiniGPT.apply f32, splits stated", &minigpt(&["--const", "D=16", "--dim", "B=2", "--dim", "T=8"]), &planned(&plan, "f32"), &["--threads", "4"], &base.join("stated")))
        },
        {
            let base = base.clone();
            Box::new(move || both_weights("MiniGPT.apply f64, D=64 B=4 T=32, the default's", &minigpt(&["--const", "D=64", "--dim", "B=4", "--dim", "T=32"]), &realized("f64"), &["--threads", "4", "--no-exact"], &base.join("default")))
        },
    ];
    let rows: Vec<Checked> = parallel(jobs).into_iter().flatten().collect();
    report(&rows);
    eprintln!("{} programs in {:.1} s", rows.len(), t0.elapsed().as_secs_f64());
    assert_c1_c2(&rows);
    for r in &rows {
        assert!(r.divided, "{}: no launch is divided among threads", r.what);
    }
    let _ = std::fs::remove_dir_all(&base);
}

/// The frozen model's profile, which `presk plan` plans with (0.8.0, D71).
fn frozen_profile() -> PathBuf {
    let p = std::env::var("RA4_FROZEN").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap()).join("presk-data/ra4-model-0.8.0/profile.json"));
    assert!(p.exists(), "{}: the frozen profile is absent (RA4_FROZEN names another)", p.display());
    p
}

/// `presk plan` for an entry under a realization, with the frozen profile,
/// written to `out`. In `bf16` it refuses (D82): the frozen model is
/// calibrated in `f32` and `f64` alone. So the `bf16` plan is the `f32` one
/// in `bf16`, each nest's lanes one and its softmax passes the default, which
/// `bf16` refuses (conditions 4 and 7): the same tiles, orders, k-blocks,
/// packs and splits.
fn plan_of(file: &str, entry: &str, args: &[String], realize: &str, out: &Path) {
    let planned = if realize == "bf16" { "f32" } else { realize };
    let mut all: Vec<String> = ["plan", file, "--entry", entry, "--realize", planned, "--profile"].iter().map(|s| s.to_string()).collect();
    all.push(frozen_profile().display().to_string());
    all.push("--out".into());
    all.push(out.display().to_string());
    all.extend(args.iter().cloned());
    let refs: Vec<&str> = all.iter().map(|s| s.as_str()).collect();
    let r = presk(&refs);
    assert!(r.status.success(), "presk plan {entry} ({planned}): {}", stderr(&r));
    let mut json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(out).unwrap()).unwrap();
    let layouts = json["layout"].as_object().map_or(0, |l| l.len());
    assert!(layouts > 0, "presk plan {entry} ({planned}) states no layout");
    if realize == "bf16" {
        json["realization"]["default"] = "bf16".into();
        for l in json["layout"].as_object_mut().unwrap().values_mut() {
            let l = l.as_object_mut().unwrap();
            if l.contains_key("lanes") {
                l.insert("lanes".into(), 1.into());
            }
            l.remove("passes");
        }
        std::fs::write(out, serde_json::to_string_pretty(&json).unwrap()).unwrap();
    }
}

/// C1 and C2 under the plan `presk plan` gives with the frozen profile, on
/// every entry of the test set, in the three types (`bf16`
/// under the `f32` plan made one, [`plan_of`]), and
/// MiniGPT's under the mixed realization; on the synthesized parameters and
/// with their zeros drawn, through both hosts. At these sizes the plans
/// state tiles, lanes and packs, and no split. Each plan runs as `presk
/// plan` wrote it: until D82 the MLP's was refused by `presk run` itself.
/// `make c-conformance`.
#[test]
#[ignore]
fn every_entry_under_its_plan_computes_the_native_backends_bits() {
    let t0 = Instant::now();
    let base = scratch("planned");
    std::fs::create_dir_all(&base).unwrap();
    let mut jobs: Vec<Box<dyn FnOnce() -> Vec<Checked> + Send>> = Vec::new();
    for case in test_set() {
        let minigpt = case.file.contains("minigpt");
        for realize in ["f64", "f32", "bf16", MIXED] {
            if realize == MIXED && !minigpt {
                continue;
            }
            let label = if realize == MIXED { "mixed" } else { realize };
            let dir = base.join(format!("{}-{label}", case.entry.replace(|c: char| !c.is_ascii_alphanumeric(), "_")));
            let case = Case { file: case.file.clone(), entry: case.entry.clone(), args: case.args.clone() };
            jobs.push(Box::new(move || {
                let plan = dir.with_extension("plan.json");
                plan_of(&case.file, &case.entry, &case.args, realize, &plan);
                let band = if realize == MIXED { "bf16" } else { realize };
                both_weights(&format!("{} {label}, planned", case.entry), &case, &planned(&plan, band), &[], &dir)
            }));
        }
    }
    let rows: Vec<Checked> = parallel(jobs).into_iter().flatten().collect();
    report(&rows);
    eprintln!("{} programs in {:.1} s", rows.len(), t0.elapsed().as_secs_f64());
    assert_c1_c2(&rows);
    let _ = std::fs::remove_dir_all(&base);
}

/// C1 on the course GPT at its constants: with its zeros drawn at B = 1,
/// T = 8, with C2; and with the course's trained checkpoint at the dimensions
/// it was trained at, T = 256, without the exact reference (hours at that
/// size). `make c-conformance`.
#[test]
#[ignore]
fn the_course_gpt_computes_the_native_backends_bits() {
    let t0 = Instant::now();
    let base = scratch("course");
    let file = "corpus/nlp/05_gpt.psk";
    let ckpt = root().join("out/course_probe_f32/final.psk.d");
    let small: Vec<String> = ["--dim", "B=1", "--dim", "T=8"].iter().map(|s| s.to_string()).collect();
    let full: Vec<String> = ["--dim", "B=1", "--dim", "T=256"].iter().map(|s| s.to_string()).collect();
    let wide: Vec<String> = ["--dim", "B=16", "--dim", "T=256"].iter().map(|s| s.to_string()).collect();
    std::fs::create_dir_all(&base).unwrap();
    let mut jobs: Vec<Box<dyn FnOnce() -> Vec<Checked> + Send>> = Vec::new();
    for realize in ["f64", "f32", "bf16"] {
        let (base, small, full, ckpt) = (base.clone(), small.clone(), full.clone(), ckpt.clone());
        jobs.push(Box::new(move || {
            let dir = base.join(format!("gpt-{realize}"));
            let mut rows = vec![export_and_check(&format!("GPT.apply {realize}, B=1 T=8"), file, "GPT.apply", &small, &realized(realize), &[], &dir)];
            let drawn = dir.with_extension("drawn.psk.d");
            draw_zeros(&dir.join("conformance/weights.psk.d"), &drawn);
            rows.push(export_and_check(&format!("GPT.apply {realize}, B=1 T=8, zeros drawn"), file, "GPT.apply", &small, &realized(realize), &["--weights", drawn.to_str().unwrap()], &dir.with_extension("drawn")));
            if ckpt.exists() {
                rows.push(export_and_check(&format!("GPT.apply {realize}, B=1 T=256, trained"), file, "GPT.apply", &full, &realized(realize), &["--weights", ckpt.to_str().unwrap(), "--no-exact"], &dir.with_extension("trained")));
            } else {
                eprintln!("{}: absent, the trained run is skipped (make corpus-gpt writes a checkpoint)", ckpt.display());
            }
            rows
        }));
    }
    // Statement C4's size, B = 16, T = 256, with the trained checkpoint:
    // under the plan `presk plan` gives, which divides most launches among
    // ten workers (the frozen profile's cores); and in `f32` under the
    // default layout, where the native backend divides the kernels above its
    // threshold of work among every core.
    assert!(ckpt.exists(), "{}: absent (make corpus-gpt writes a checkpoint)", ckpt.display());
    for realize in ["f64", "f32", "bf16"] {
        let (base, wide, ckpt) = (base.clone(), wide.clone(), ckpt.clone());
        jobs.push(Box::new(move || {
            let dir = base.join(format!("gpt16-{realize}"));
            let plan = dir.with_extension("plan.json");
            plan_of(file, "GPT.apply", &wide, realize, &plan);
            vec![export_and_check(&format!("GPT.apply {realize}, B=16 T=256, trained, planned"), file, "GPT.apply", &wide, &planned(&plan, realize), &["--weights", ckpt.to_str().unwrap(), "--no-exact"], &dir)]
        }));
    }
    {
        let (base, wide, ckpt) = (base.clone(), wide.clone(), ckpt.clone());
        jobs.push(Box::new(move || vec![export_and_check("GPT.apply f32, B=16 T=256, trained, the default's", file, "GPT.apply", &wide, &realized("f32"), &["--weights", ckpt.to_str().unwrap(), "--no-exact"], &base.join("gpt16-default"))]));
    }
    let rows: Vec<Checked> = parallel(jobs).into_iter().flatten().collect();
    report(&rows);
    eprintln!("{} programs in {:.1} s", rows.len(), t0.elapsed().as_secs_f64());
    assert_c1_c2(&rows);
    for r in rows.iter().filter(|r| r.what.contains("B=16")) {
        assert!(r.divided, "{}: no launch is divided among threads", r.what);
    }
    let _ = std::fs::remove_dir_all(&base);
}

// ----- C5: generation (step 6) ---------------------------------------------------------

/// The generation example, its tiny constants, and a prompt longer than the
/// window `T = 8`, so that a window one token behind still lies in it.
const SERVE: &str = "tests/examples/minigpt/serve_minigpt.psk";
const TINY: [&str; 8] = ["--const", "V=64", "--const", "D=16", "--const", "H=2", "--const", "L=2"];
const PROMPT: &str = "5 1 4 1 5 9 2 6 5 3 5 8";

/// Temperatures (0 greedy) and seeds, and the tokens each generates.
const SETTINGS: [(&str, u64); 5] = [("0", 0), ("0.8", 0), ("0.8", 7), ("0.3", 1), ("1.5", 2)];
const TOKENS: &str = "16";

/// `presk export --target c --serve` of the example into `out`, at `T = 8`,
/// built by its `Makefile` (`model` and `generate`) without a warning.
fn export_serve(realize: &str, weights: Option<&Path>, extra: &[&str], out: &Path) {
    let mut all: Vec<String> = ["export", SERVE, "--target", "c", "--serve", "--dim", "T=8", "--realize", realize, "-o"].iter().map(|s| s.to_string()).collect();
    all.push(out.display().to_string());
    all.extend(TINY.iter().map(|s| s.to_string()));
    if let Some(w) = weights {
        all.push("--weights".into());
        all.push(w.display().to_string());
    }
    all.extend(extra.iter().map(|s| s.to_string()));
    let refs: Vec<&str> = all.iter().map(|s| s.as_str()).collect();
    let r = presk(&refs);
    assert!(r.status.success(), "presk export --serve ({realize}): {}", stderr(&r));
    build_generate(out);
    if std::env::var("PRESK_C3").is_ok_and(|v| v == "1") {
        c3(out).unwrap_or_else(|e| panic!("{}: {e}", out.display()));
    }
}

fn build_generate(dir: &Path) {
    let make = Command::new("make").args(["-s", "-B", "-C"]).arg(dir).output().unwrap();
    let err = String::from_utf8_lossy(&make.stderr);
    assert!(make.status.success() && !err.contains("warning"), "make in {}: {err}", dir.display());
}

/// `presk serve --backend native --json`, as a document.
fn serve_json(realize: &str, weights: &Path, temperature: &str, seed: u64) -> String {
    let seed = seed.to_string();
    let w = weights.display().to_string();
    let mut all: Vec<&str> = vec!["serve", SERVE];
    all.extend(TINY);
    all.extend(["--weights", &w, "--prompt", PROMPT, "--tokens", TOKENS, "--temperature", temperature, "--seed", &seed, "--backend", "native", "--realize", realize, "--dim", "T=8", "--json"]);
    let r = presk(&all);
    assert!(r.status.success(), "presk serve: {}", stderr(&r));
    String::from_utf8(r.stdout).unwrap()
}

/// The exported program's `generate --json`, as a document.
fn generate_json(dir: &Path, weights: &Path, temperature: &str, seed: u64) -> String {
    run_generate(&dir.join("generate"), weights, temperature, seed)
}

fn run_generate(program: &Path, weights: &Path, temperature: &str, seed: u64) -> String {
    let r = Command::new(program).current_dir(root()).args(["--weights", &weights.display().to_string(), "--prompt", PROMPT, "--tokens", TOKENS, "--temperature", temperature, "--seed", &seed.to_string(), "--json"]).output().unwrap();
    assert!(r.status.success(), "{}: {}", program.display(), String::from_utf8_lossy(&r.stderr));
    String::from_utf8(r.stdout).unwrap()
}

/// The tokens of a `--json` document.
fn generated(json: &str) -> Vec<u64> {
    let v: serde_json::Value = serde_json::from_str(json).unwrap_or_else(|e| panic!("{e}: {json}"));
    v["generated"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap()).collect()
}

/// C5 on one program: every setting, `generate`'s document `presk serve`'s
/// byte for byte. The documents, in the order of [`SETTINGS`].
fn c5(what: &str, dir: &Path, realize: &str, weights: &Path) -> Vec<String> {
    let mut docs = Vec::new();
    let clang = dir.join("generate-clang").exists();
    for (t, seed) in SETTINGS {
        let (n, c) = (serve_json(realize, weights, t, seed), generate_json(dir, weights, t, seed));
        assert_eq!(c, n, "{what}, temperature {t}, seed {seed}: generate's tokens are not presk serve's");
        if clang {
            assert_eq!(run_generate(&dir.join("generate-clang"), weights, t, seed), n, "{what}, temperature {t}, seed {seed}: Clang's generate");
        }
        docs.push(c);
    }
    eprintln!("C5 holds: {what}, {} settings, e.g. greedy {:?}", SETTINGS.len(), generated(&docs[0]));
    docs
}

/// Weights with their zeros drawn, from the synthesized ones a first export
/// writes: the example's head is `init = zeros`, and on synthesized
/// parameters its logits are zero, every token a uniform draw.
fn drawn_weights(base: &Path) -> PathBuf {
    let first = base.join("synthesized");
    export_serve("f64", None, &[], &first);
    let drawn = base.join("drawn.psk.d");
    draw_zeros(&first.join("conformance/weights.psk.d"), &drawn);
    drawn
}

/// Statement C5 in `make test`, on the example's parameters with their zeros
/// drawn: in `f64`, `f32` and `bf16`, structured and straight-line, greedy
/// and sampled, every program's `generate` gives `presk serve`'s document.
/// And each of three mutations of the loop changes the tokens.
#[test]
fn generate_gives_presk_serves_tokens() {
    let t0 = Instant::now();
    let base = scratch("serve");
    std::fs::create_dir_all(&base).unwrap();
    let drawn = drawn_weights(&base);
    let jobs: Vec<Box<dyn FnOnce() -> Vec<String> + Send>> = ["f64", "f32", "bf16"]
        .into_iter()
        .flat_map(|r| [(r, false), (r, true)])
        .map(|(realize, flat)| {
            let (base, drawn) = (base.clone(), drawn.clone());
            Box::new(move || {
                let dir = base.join(format!("{realize}{}", if flat { "-flat" } else { "" }));
                let extra: &[&str] = if flat { &["--straight-line"] } else { &[] };
                export_serve(realize, Some(&drawn), extra, &dir);
                // C1 on one call too, the conformance script's.
                let check = Command::new("sh").arg(dir.join("conformance/check.sh")).env("PRESK", "/nonexistent").output().unwrap();
                assert!(String::from_utf8_lossy(&check.stdout).contains("C1 holds"), "{realize}: {}", String::from_utf8_lossy(&check.stderr));
                c5(&format!("{realize}{}, zeros drawn", if flat { ", straight-line" } else { "" }), &dir, realize, &drawn)
            }) as Box<dyn FnOnce() -> Vec<String> + Send>
        })
        .collect();
    let docs = parallel(jobs);
    // The settings generate different tokens: the check sees the
    // temperature and the seed.
    let f64_docs = &docs[0];
    let tokens: Vec<Vec<u64>> = f64_docs.iter().map(|d| generated(d)).collect();
    for i in 0..tokens.len() {
        for j in 0..i {
            assert_ne!(tokens[i], tokens[j], "settings {:?} and {:?} generate the same tokens", SETTINGS[i], SETTINGS[j]);
        }
    }
    // Mutations of generate.c, on the f64 program: each changes the tokens
    // of the settings it concerns, and a draw leaves the greedy tokens.
    let mutations: [(&str, &str, &str, bool); 3] = [
        ("a draw skipped at the second step", "        if (greedy) {", "        if (step == 1)\n            (void)presk_uniform01(&rng);\n        if (greedy) {", false),
        ("the stream seeded again at each step", "        presk_release(&arena, mark);\n", "        presk_release(&arena, mark);\n        presk_rng_init(&rng, seed);\n", false),
        ("the window one token behind", "toks[len - WINDOW + i]", "toks[len - WINDOW - 1 + i]", true),
    ];
    for (k, (what, from, to, greedy_too)) in mutations.iter().enumerate() {
        let dir = base.join(format!("mutation-{k}"));
        let src = base.join("f64");
        let copied = Command::new("cp").arg("-r").arg(&src).arg(&dir).status().unwrap();
        assert!(copied.success());
        let path = dir.join("generate.c");
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.matches(from).count(), 1, "{what}: `{from}` is not once in generate.c");
        std::fs::write(&path, text.replace(from, to)).unwrap();
        build_generate(&dir);
        for (i, (t, seed)) in SETTINGS.iter().enumerate() {
            let got = generated(&generate_json(&dir, &drawn, t, *seed));
            let greedy = *t == "0";
            if greedy && !greedy_too {
                assert_eq!(got, tokens[i], "{what}: the greedy tokens moved");
            } else {
                assert_ne!(got, tokens[i], "{what}: temperature {t}, seed {seed}, the tokens did not move");
            }
        }
        eprintln!("mutation caught: {what}");
    }
    eprintln!("C5: {} programs, {} documents, 3 mutations in {:.1} s", docs.len(), docs.iter().map(|d| d.len()).sum::<usize>(), t0.elapsed().as_secs_f64());
    let _ = std::fs::remove_dir_all(&base);
}

/// Statement C5 on the tiny MiniGPT's trained checkpoint (`make
/// train-examples`), in the three types, through both hosts; and C1 on
/// `next_token` with it, at the test set's `B = 2`, `T = 8`. The trained
/// logits are nearly flat (about 10⁻⁴), so the tokens hardly depend on the
/// temperature, but the draws are the stream's. `make c-conformance`.
#[test]
#[ignore]
fn the_trained_minigpt_generates_presk_serves_tokens() {
    let t0 = Instant::now();
    let ckpt = root().join("out/ckpt/final.psk.d");
    assert!(ckpt.exists(), "{}: absent (make train-examples writes it)", ckpt.display());
    let base = scratch("serve-trained");
    std::fs::create_dir_all(&base).unwrap();
    let mut jobs: Vec<Box<dyn FnOnce() -> Vec<Checked> + Send>> = Vec::new();
    for realize in ["f64", "f32", "bf16"] {
        for flat in [false, true] {
            let (base, ckpt) = (base.clone(), ckpt.clone());
            jobs.push(Box::new(move || {
                let dir = base.join(format!("{realize}{}", if flat { "-flat" } else { "" }));
                let extra: &[&str] = if flat { &["--straight-line"] } else { &[] };
                export_serve(realize, Some(&ckpt), extra, &dir);
                c5(&format!("{realize}{}, trained", if flat { ", straight-line" } else { "" }), &dir, realize, &ckpt);
                Vec::new()
            }));
        }
        let (base, ckpt) = (base.clone(), ckpt.clone());
        jobs.push(Box::new(move || {
            let args: Vec<String> = TINY.iter().chain(&["--dim", "B=2", "--dim", "T=8"]).map(|s| s.to_string()).collect();
            vec![export_and_check(&format!("next_token {realize}, trained"), SERVE, "next_token", &args, &realized(realize), &["--weights", ckpt.to_str().unwrap()], &base.join(format!("next-token-{realize}")))]
        }));
    }
    let rows: Vec<Checked> = parallel(jobs).into_iter().flatten().collect();
    report(&rows);
    assert_c1_c2(&rows);
    eprintln!("C5 and C1 on the trained checkpoint in {:.1} s", t0.elapsed().as_secs_f64());
    let _ = std::fs::remove_dir_all(&base);
}

// ----- integer weights (step 7) ---------------------------------------------------------

/// MiniGPT's constants of the test set.
const MINI: [&str; 12] = ["--const", "V=64", "--const", "D=16", "--const", "H=2", "--const", "L=2", "--dim", "B=2", "--dim", "T=8"];

/// The parameters of MiniGPT (and of `serve_minigpt.psk`) a realization
/// quantizes as integer weights: the head, the embeddings, and every
/// matrix of the blocks, each over the axis it is contracted over (the
/// embeddings over their rows, the one a one-hot contraction reduces).
const MINI_MATRICES: [(&str, i64); 9] = [("head", 0), ("emb.tok", 0), ("emb.pos", 0), ("blocks.attn.wq.w", -2), ("blocks.attn.wk.w", -2), ("blocks.attn.wv.w", -2), ("blocks.attn.wo.w", -2), ("blocks.mlp.l1.w", -2), ("blocks.mlp.l2.w", -2)];

/// Its biases and its norms' parameters: a bias's `add` loads integers, a
/// norm's fusion group reads their values.
const MINI_BIASES: [(&str, i64); 11] = [("blocks.attn.wq.b", -1), ("blocks.attn.wk.b", -1), ("blocks.attn.wv.b", -1), ("blocks.attn.wo.b", -1), ("blocks.mlp.l1.b", -1), ("blocks.mlp.l2.b", -1), ("blocks.ln1.γ", -1), ("blocks.ln1.β", -1), ("blocks.ln2.γ", -1), ("blocks.ln2.β", -1), ("lnf.γ", -1)];

/// The same of `tests/plan/minigpt_grad.psk`, whose model names its
/// embeddings `tok` and `pos`.
const GRAD_MATRICES: [(&str, i64); 9] = [("head", 0), ("tok", 0), ("pos", 0), ("blocks.attn.wq.w", -2), ("blocks.attn.wk.w", -2), ("blocks.attn.wv.w", -2), ("blocks.attn.wo.w", -2), ("blocks.mlp.l1.w", -2), ("blocks.mlp.l2.w", -2)];

/// The course GPT's matrices, embeddings and head.
const GPT_MATRICES: [(&str, i64); 9] = [("tok", 0), ("pos", 0), ("head", 0), ("blocks.attn.wq.w", -2), ("blocks.attn.wk.w", -2), ("blocks.attn.wv.w", -2), ("blocks.attn.wo.w", -2), ("blocks.ff.up.w", -2), ("blocks.ff.down.w", -2)];

/// A realization of `default` quantizing `params` to `bits` with `scheme`,
/// written as `file` in `dir`: `--realize @…`.
fn quantizing(dir: &Path, file: &str, default: &str, bits: u8, scheme: &str, params: &[(&str, i64)]) -> String {
    let quant: serde_json::Map<String, serde_json::Value> = params.iter().map(|(p, over)| (p.to_string(), serde_json::json!({ "bits": bits, "over": over, "scheme": scheme }))).collect();
    let r = serde_json::json!({ "default": default, "order": { "kind": "sequential" }, "quant": quant });
    std::fs::create_dir_all(dir).unwrap();
    let path = dir.join(file);
    std::fs::write(&path, serde_json::to_string_pretty(&r).unwrap()).unwrap();
    format!("@{}", path.display())
}

/// Whether an exported program reads integer weights: a leaf of `presk_q8`
/// or `presk_q4` in its `_in` struct.
fn quantized_leaves(dir: &Path) -> usize {
    let h = std::fs::read_to_string(dir.join("model.h")).unwrap();
    h.lines().filter(|l| l.trim_start().starts_with("presk_q8 ") || l.trim_start().starts_with("presk_q4 ")).count()
}

/// C1 and C2 on integer weights (step 7): `odd.psk` in both widths and both
/// schemes; MiniGPT's forward pass and `next_token` with its matrices
/// quantized; its gradient (`loss_grad`, whose scan and backward fold slice
/// the integers) and an optimizer step (`adamw.step`, whose fusion groups
/// read their values); at `D = 9`, `H = 3`, where the host copies a layer's
/// odd `int4` matrices to `f64`; with its biases and norms quantized; under
/// a plan stating splits; in the three types, on synthesized parameters and
/// with their zeros drawn, through both hosts.
#[test]
fn integer_weights_compute_the_native_backends_bits() {
    let t0 = Instant::now();
    let base = scratch("quant");
    let r = base.join("realizations");
    let mut jobs: Vec<Box<dyn FnOnce() -> Vec<Checked> + Send>> = Vec::new();
    let mini = |extra: &[&str]| -> Vec<String> { MINI.iter().chain(extra).map(|s| s.to_string()).collect() };
    // odd.psk: rows of 13, so every other int4 row starts mid-byte.
    for (bits, scheme) in [(8u8, "absmax"), (8, "pow2"), (4, "absmax"), (4, "pow2")] {
        for d in ["f64", "f32", "bf16"] {
            let realize = quantizing(&r, &format!("odd-{bits}-{scheme}-{d}.json"), d, bits, scheme, &[("w", 0)]);
            let dir = base.join(format!("odd-{bits}-{scheme}-{d}"));
            let what = format!("L.apply int{bits}/{scheme} {d}");
            jobs.push(Box::new(move || vec![export_and_check(&what, "tests/quant/odd.psk", "L.apply", &["--dim".into(), "B=3".into()], &realized(&realize), &[], &dir)]));
        }
    }
    // MiniGPT: its forward pass and a sampled step, both widths.
    for (bits, scheme) in [(8u8, "absmax"), (4, "pow2")] {
        for d in ["f64", "f32", "bf16"] {
            let realize = quantizing(&r, &format!("mini-{bits}-{d}.json"), d, bits, scheme, &MINI_MATRICES);
            let (dir, args) = (base.join(format!("mini-{bits}-{d}")), mini(&[]));
            let case = Case { file: "tests/examples/minigpt/minigpt.psk".into(), entry: "MiniGPT.apply".into(), args };
            let what = format!("MiniGPT.apply int{bits}/{scheme} {d}");
            let under = realized(&realize);
            jobs.push(Box::new(move || both_weights(&what, &case, &Under { band: d.into(), ..under }, &[], &dir)));
        }
    }
    for (bits, scheme, d) in [(8u8, "absmax", "f32"), (4, "pow2", "bf16")] {
        let realize = quantizing(&r, &format!("next-{bits}-{d}.json"), d, bits, scheme, &MINI_MATRICES);
        let case = Case { file: SERVE.into(), entry: "next_token".into(), args: mini(&[]) };
        let (dir, what, under) = (base.join(format!("next-{bits}-{d}")), format!("next_token int{bits}/{scheme} {d}"), realized(&realize));
        jobs.push(Box::new(move || both_weights(&what, &case, &Under { band: d.into(), ..under }, &[], &dir)));
    }
    // A gradient and an optimizer step on integer weights.
    for (file, entry, params) in [("tests/plan/minigpt_grad.psk", "loss_grad", &GRAD_MATRICES), ("tests/examples/minigpt/train_minigpt.psk", "adamw.step", &MINI_MATRICES)] {
        let realize = quantizing(&r, &format!("{entry}-8-f32.json"), "f32", 8, "absmax", params);
        let case = Case { file: file.into(), entry: entry.into(), args: mini(&[]) };
        let (dir, what, under) = (base.join(entry.replace('.', "_")), format!("{entry} int8/absmax f32"), realized(&realize));
        jobs.push(Box::new(move || both_weights(&what, &case, &Under { band: "f32".into(), ..under }, &[], &dir)));
    }
    // D = 9, H = 3: attention's matrices are 81 per layer, odd, so their
    // int4 slices are copies in f64; the MLP's, 324, stay integers.
    for d in ["f32", "bf16"] {
        let realize = quantizing(&r, &format!("odd-layers-{d}.json"), d, 4, "absmax", &MINI_MATRICES);
        let args: Vec<String> = ["--const", "V=64", "--const", "D=9", "--const", "H=3", "--const", "L=2", "--dim", "B=2", "--dim", "T=8"].iter().map(|s| s.to_string()).collect();
        let case = Case { file: "tests/examples/minigpt/minigpt.psk".into(), entry: "MiniGPT.apply".into(), args };
        let (dir, what, under) = (base.join(format!("odd-layers-{d}")), format!("MiniGPT.apply D=9 H=3 int4 {d}"), realized(&realize));
        jobs.push(Box::new(move || both_weights(&what, &case, &Under { band: d.into(), ..under }, &[], &dir)));
    }
    // The biases and the norms' parameters.
    for d in ["f32", "bf16"] {
        let realize = quantizing(&r, &format!("biases-{d}.json"), d, 8, "absmax", &MINI_BIASES);
        let case = Case { file: "tests/examples/minigpt/minigpt.psk".into(), entry: "MiniGPT.apply".into(), args: mini(&[]) };
        let (dir, what, under) = (base.join(format!("biases-{d}")), format!("MiniGPT.apply biases and norms int8 {d}"), realized(&realize));
        jobs.push(Box::new(move || both_weights(&what, &case, &Under { band: d.into(), ..under }, &[], &dir)));
    }
    // Under a plan stating splits at three depths, its realization
    // quantizing the matrices.
    {
        let mut plan: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(root().join("tests/export_c/minigpt-splits.json")).unwrap()).unwrap();
        let q = quantizing(&r, "splits-quant.json", "f32", 8, "absmax", &MINI_MATRICES);
        let quant: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(q.trim_start_matches('@')).unwrap()).unwrap();
        plan["realization"]["quant"] = quant["quant"].clone();
        let path = r.join("splits-plan.json");
        std::fs::write(&path, serde_json::to_string_pretty(&plan).unwrap()).unwrap();
        let case = Case { file: "tests/examples/minigpt/minigpt.psk".into(), entry: "MiniGPT.apply".into(), args: mini(&[]) };
        let dir = base.join("splits");
        jobs.push(Box::new(move || both_weights("MiniGPT.apply int8 f32, splits stated", &case, &planned(&path, "f32"), &["--threads", "4"], &dir)));
    }
    let rows: Vec<Checked> = parallel(jobs).into_iter().flatten().collect();
    report(&rows);
    eprintln!("{} programs in {:.1} s", rows.len(), t0.elapsed().as_secs_f64());
    assert_c1_c2(&rows);
    for r in rows.iter().filter(|r| r.what.contains("drawn") || r.what.starts_with("L.apply")) {
        assert!(r.nonzero > 0, "{}: every output is zero", r.what);
    }
    // Each program reads integers; the odd layers' copies are in its host.
    for dir in std::fs::read_dir(&base).unwrap().flatten().map(|e| e.path()).filter(|p| p.join("model.h").exists()) {
        assert!(quantized_leaves(&dir) > 0, "{}: no leaf of integer weights", dir.display());
    }
    let model = std::fs::read_to_string(base.join("odd-layers-f32/model.c")).unwrap();
    assert!(model.contains("presk_dequantize(") && model.contains("PRESK_F64"), "D = 9: no slice copied to f64");
    assert!(base.join("splits").join("Makefile").exists() && divided(&base.join("splits")), "the plan divides no launch");
    let _ = std::fs::remove_dir_all(&base);
}

/// A copy of MiniGPT's checkpoint whose head (its first leaf, `ℝ[16, 64]`,
/// quantized over its rows) has in each column one weight exactly
/// `3.5·s`, `s` the column's scale in `int8` `absmax`, `f32(m/127)`: a tie
/// of `w / s`, which presk-rt rounds to even, and which `w / (m/127)`, the
/// scale not rounded to `f32`, moves off the half to one side or the other.
fn tie_head(from: &Path, to: &Path) {
    let bytes = std::fs::read(from).unwrap();
    let ty_len = u32::from_le_bytes(bytes[6..10].try_into().unwrap()) as usize;
    let data = 10 + ty_len + 8;
    let (rows, cols) = (16, 64);
    let mut w: Vec<f64> = bytes[data..data + 8 * rows * cols].chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    for c in 0..cols {
        let m = (0..rows).map(|r| w[r * cols + c].abs()).fold(0.0, f64::max);
        let s = (m / 127.0) as f32 as f64;
        let r = (0..rows).find(|r| w[r * cols + c].abs() < m).unwrap();
        w[r * cols + c] = 3.5 * s;
    }
    let mut out = bytes.clone();
    for (i, x) in w.iter().enumerate() {
        out[data + 8 * i..data + 8 * (i + 1)].copy_from_slice(&x.to_le_bytes());
    }
    std::fs::write(to, out).unwrap();
}

/// The check checked (step 7): three mutations of an exported program on
/// integer weights, each of which must change its outputs, which C1
/// compares. On MiniGPT's forward pass with its zeros drawn, in `f32`:
/// - a scale one channel off (after quantizing, the head's first channel
///   takes its neighbour's scale), in `int8`;
/// - the two nibbles of a byte swapped (`presk_int4` reads the other), in
///   `int4`;
/// - the scale kept in `f64` when quantizing (`w / (m/qmax)` rounded, where
///   presk-rt divides by the scale rounded to `f32`), in `int8` `absmax`.
///   Drawn weights reach a tie of `w / s` too rarely for it to move an
///   integer, so the head is given one in each column ([`tie_head`]).
#[test]
fn a_program_on_integer_weights_that_departs_from_presk_rt_fails_c1() {
    let base = scratch("quant-mutations");
    let r = base.join("realizations");
    // (what, bits, file, text, what it becomes): each once in the file.
    let mutations: [(&'static str, u8, &'static str, &'static str, &'static str); 3] = [
        ("a scale one channel off", 8, "main.c", "        in.theta_head = (presk_q8){q, s};\n", "        s[0] = s[1];\n        in.theta_head = (presk_q8){q, s};\n"),
        ("the two nibbles of a byte swapped", 4, "presk_rt.h", "unsigned n = e % 2 ? b >> 4 : b & 0x0fu;", "unsigned n = e % 2 ? b & 0x0fu : b >> 4;"),
        ("the scale kept in f64 when quantizing", 8, "presk_rt.c", "const double sc = s[c];", "const double sc = m == 0.0 || scheme == PRESK_POW2 ? s[c] : m / qmax;"),
    ];
    let jobs: Vec<Box<dyn FnOnce() -> &'static str + Send>> = mutations
        .into_iter()
        .map(|(what, bits, file, from, to)| {
            let realize = quantizing(&r, &format!("int{bits}.json"), "f32", bits, "absmax", &MINI_MATRICES);
            let dir = base.join(format!("int{bits}-{}", what.replace(' ', "-")));
            Box::new(move || {
                // The program on the parameters' zeros drawn (or tied, for
                // the scale's rounding), checked as it is exported.
                let args: Vec<String> = MINI.iter().map(|s| s.to_string()).collect();
                let file_psk = "tests/examples/minigpt/minigpt.psk";
                let (ok, log) = export_one(what, file_psk, "MiniGPT.apply", &args, &realized(&realize), &[], &dir);
                assert!(ok, "{what}: {log}");
                let weights = dir.with_extension("weights.psk.d");
                draw_zeros(&dir.join("conformance/weights.psk.d"), &weights);
                if file == "presk_rt.c" {
                    tie_head(&weights.clone(), &weights);
                }
                let d = dir.with_extension("mutated");
                let (ok, log) = export_one(what, file_psk, "MiniGPT.apply", &args, &realized(&realize), &["--weights", weights.to_str().unwrap()], &d);
                assert!(ok, "{what}: C1 fails before the mutation: {log}");
                let path = d.join(file);
                let text = std::fs::read_to_string(&path).unwrap();
                assert_eq!(text.matches(from).count(), 1, "{what}: `{from}` is not once in {file}");
                std::fs::write(&path, text.replace(from, to)).unwrap();
                let check = Command::new("sh").arg(d.join("conformance/check.sh")).env("PRESK", "/nonexistent").output().unwrap();
                let err = String::from_utf8_lossy(&check.stderr);
                assert!(!check.status.success() && err.contains("C1 fails"), "{what}: the outputs did not change:\n{}{err}", String::from_utf8_lossy(&check.stdout));
                what
            }) as Box<dyn FnOnce() -> &'static str + Send>
        })
        .collect();
    for what in parallel(jobs) {
        eprintln!("mutation caught: {what}");
    }
    let _ = std::fs::remove_dir_all(&base);
}

/// Statement C5 on integer weights (step 7): `presk export --serve` under a
/// realization quantizing the matrices, `generate` quantizing them as it
/// reads them, as `presk serve` does; greedy and sampled, with the zeros
/// drawn, both hosts in `int8` `f32`, the structured one in `int4` `bf16`.
#[test]
fn generate_on_integer_weights_gives_presk_serves_tokens() {
    let t0 = Instant::now();
    let base = scratch("serve-quant");
    std::fs::create_dir_all(&base).unwrap();
    let drawn = drawn_weights(&base);
    let r = base.join("realizations");
    let jobs: Vec<Box<dyn FnOnce() -> Vec<String> + Send>> = [(8u8, "absmax", "f32", false), (8, "absmax", "f32", true), (4, "pow2", "bf16", false)]
        .into_iter()
        .map(|(bits, scheme, d, flat)| {
            let (base, drawn) = (base.clone(), drawn.clone());
            let realize = quantizing(&r, &format!("int{bits}-{d}.json"), d, bits, scheme, &MINI_MATRICES);
            Box::new(move || {
                let dir = base.join(format!("int{bits}-{d}{}", if flat { "-flat" } else { "" }));
                let extra: &[&str] = if flat { &["--straight-line"] } else { &[] };
                export_serve(&realize, Some(&drawn), extra, &dir);
                assert!(quantized_leaves(&dir) == MINI_MATRICES.len(), "{}: not every matrix is integers", dir.display());
                c5(&format!("int{bits}/{scheme} {d}{}, zeros drawn", if flat { ", straight-line" } else { "" }), &dir, &realize, &drawn)
            }) as Box<dyn FnOnce() -> Vec<String> + Send>
        })
        .collect();
    let docs = parallel(jobs);
    eprintln!("C5 on integer weights: {} programs, {} documents in {:.1} s", docs.len(), docs.iter().map(|d| d.len()).sum::<usize>(), t0.elapsed().as_secs_f64());
    let _ = std::fs::remove_dir_all(&base);
}

/// Integer weights on trained checkpoints (step 7), `make c-conformance`:
/// the tiny MiniGPT's (`make train-examples`) in `int8` and `int4`, its
/// forward pass, `next_token` and C5; the course GPT's at the dimensions it
/// was trained at, `T = 256`, in `int8`, without the exact reference; and
/// MiniGPT's forward pass under the plan `presk plan` gives for its
/// quantized realization with the frozen profile.
#[test]
#[ignore]
fn trained_models_on_integer_weights_compute_the_native_backends_bits() {
    let t0 = Instant::now();
    let tiny = root().join("out/ckpt/final.psk.d");
    let course = root().join("out/course_probe_f32/final.psk.d");
    assert!(tiny.exists(), "{}: absent (make train-examples writes it)", tiny.display());
    assert!(course.exists(), "{}: absent (make corpus-gpt writes it)", course.display());
    let base = scratch("quant-trained");
    let r = base.join("realizations");
    let mut jobs: Vec<Box<dyn FnOnce() -> Vec<Checked> + Send>> = Vec::new();
    let mini: Vec<String> = MINI.iter().map(|s| s.to_string()).collect();
    for (bits, scheme, d) in [(8u8, "absmax", "f64"), (8, "absmax", "f32"), (4, "pow2", "f32"), (8, "absmax", "bf16")] {
        let realize = quantizing(&r, &format!("mini-{bits}-{d}.json"), d, bits, scheme, &MINI_MATRICES);
        for (file, entry) in [("tests/examples/minigpt/minigpt.psk", "MiniGPT.apply"), (SERVE, "next_token")] {
            let (base, tiny, mini, realize) = (base.clone(), tiny.clone(), mini.clone(), realize.clone());
            jobs.push(Box::new(move || {
                let dir = base.join(format!("{}-{bits}-{d}", entry.replace('.', "_")));
                vec![export_and_check(&format!("{entry} int{bits}/{scheme} {d}, trained"), file, entry, &mini, &Under { band: d.into(), ..realized(&realize) }, &["--weights", tiny.to_str().unwrap()], &dir)]
            }));
        }
        let (base, tiny, realize) = (base.clone(), tiny.clone(), realize.clone());
        jobs.push(Box::new(move || {
            let dir = base.join(format!("serve-{bits}-{d}"));
            export_serve(&realize, Some(&tiny), &[], &dir);
            c5(&format!("int{bits}/{scheme} {d}, trained"), &dir, &realize, &tiny);
            Vec::new()
        }));
    }
    for d in ["f32", "bf16"] {
        let realize = quantizing(&r, &format!("gpt-8-{d}.json"), d, 8, "absmax", &GPT_MATRICES);
        let (base, course) = (base.clone(), course.clone());
        jobs.push(Box::new(move || {
            let full: Vec<String> = ["--dim", "B=1", "--dim", "T=256"].iter().map(|s| s.to_string()).collect();
            vec![export_and_check(&format!("GPT.apply int8 {d}, B=1 T=256, trained"), "corpus/nlp/05_gpt.psk", "GPT.apply", &full, &Under { band: d.into(), ..realized(&realize) }, &["--weights", course.to_str().unwrap(), "--no-exact"], &base.join(format!("gpt-8-{d}")))]
        }));
    }
    for d in ["f64", "f32"] {
        let realize = quantizing(&r, &format!("mini-plan-{d}.json"), d, 8, "absmax", &MINI_MATRICES);
        let (base, tiny, mini) = (base.clone(), tiny.clone(), mini.clone());
        jobs.push(Box::new(move || {
            let dir = base.join(format!("planned-{d}"));
            let plan = dir.with_extension("plan.json");
            plan_of("tests/examples/minigpt/minigpt.psk", "MiniGPT.apply", &mini, &realize, &plan);
            let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&plan).unwrap()).unwrap();
            assert!(json["realization"]["quant"].as_object().is_some_and(|q| q.len() == MINI_MATRICES.len()), "presk plan dropped the quant map");
            vec![export_and_check(&format!("MiniGPT.apply int8 {d}, trained, planned"), "tests/examples/minigpt/minigpt.psk", "MiniGPT.apply", &mini, &planned(&plan, d), &["--weights", tiny.to_str().unwrap()], &dir)]
        }));
    }
    let rows: Vec<Checked> = parallel(jobs).into_iter().flatten().collect();
    report(&rows);
    eprintln!("{} programs on integer weights, trained, in {:.1} s", rows.len(), t0.elapsed().as_secs_f64());
    assert_c1_c2(&rows);
    let _ = std::fs::remove_dir_all(&base);
}
