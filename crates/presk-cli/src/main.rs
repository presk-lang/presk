//! `presk` — the Presk command-line tool.
//!
//! Subcommands follow the prospective interface of the specification
//! (`presk check | run | export`); `dump` is a development aid.
//! Exit codes: 0 success, 1 diagnostics reported, 2 not implemented / usage,
//! 3 not executable in this version.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use presk_syntax::diag::{has_errors, render_all};
use presk_syntax::{SourceMap, lexer, parse_file};
use presk_types::check as typing;
use presk_types::check::TypedProgram;
use presk_types::elab::{self, Options};

#[derive(Parser)]
#[command(name = "presk", version, about = "Presk compiler (0.4.0: front-end, reference interpreter, native backend with fusion and threads, e-graph optimizer, training, serving, PyTorch export)")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Type-check a program: shapes, regularity classes, effects.
    Check {
        file: PathBuf,
        /// Override a `const` (may be repeated): --const H=2
        #[arg(long = "const", value_name = "NAME=N")]
        consts: Vec<String>,
        /// Print the typed signatures of every declaration.
        #[arg(long)]
        summary: bool,
        /// Machine-readable diagnostics.
        #[arg(long)]
        json: bool,
    },
    /// Evaluate an entry point on the reference interpreter.
    Run {
        file: PathBuf,
        /// Entry point: a top-level fn or `Module{args}.fn`.
        #[arg(long)]
        entry: Option<String>,
        #[arg(long = "dim", value_name = "NAME=N")]
        dims: Vec<String>,
        #[arg(long = "const", value_name = "NAME=N")]
        consts: Vec<String>,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// MPFR precision of every intermediate.
        #[arg(long, default_value_t = 256)]
        bits: u32,
        /// Significant digits when printing.
        #[arg(long, default_value_t = 12)]
        digits: u32,
        #[arg(long)]
        json: bool,
        #[arg(long)]
        verbose: bool,
        /// Unroll `stack{L}` into L calls instead of a fold (debugging aid).
        #[arg(long)]
        unroll_stack: bool,
        /// The executor: the reference interpreter (native backends come with 0.3.0).
        #[arg(long, value_enum, default_value_t = BackendKind::Interp)]
        backend: BackendKind,
        /// Machine type of a native realization (ignored by the interpreter).
        #[arg(long, value_enum, default_value_t = RealizeArg::F64)]
        realize: RealizeArg,
        /// Rewrite the program with the e-graph optimizer before running it (`presk-opt`).
        #[arg(long)]
        opt: bool,
        /// With --opt: extract by the calibrated roofline of this machine profile (`presk bench --calibrate`).
        #[arg(long, value_name = "FILE.json")]
        profile: Option<PathBuf>,
        /// Workers a parallel kernel may use on the native backend (default: the cores).
        #[arg(long)]
        threads: Option<usize>,
        /// Take the entry's last argument from this data file instead of synthesizing it;
        /// its `batch` axis is bound to the number of samples read.
        #[arg(long, value_name = "FILE.psk.d")]
        data: Option<PathBuf>,
        /// With --data: read only the first N samples.
        #[arg(long)]
        samples: Option<usize>,
    },
    /// Export a program to another framework.
    Export {
        file: PathBuf,
        #[arg(long, value_enum)]
        target: ExportTarget,
        #[arg(long, value_enum, default_value_t = ExportMode::Readable)]
        mode: ExportMode,
        #[arg(short, long, value_name = "DIR")]
        out: PathBuf,
        /// Insert executable shape assertions.
        #[arg(long = "shapes", value_name = "check")]
        shapes: Option<String>,
        #[arg(long = "stack", value_enum, default_value_t = StackMode::Loop)]
        stack: StackMode,
        /// Also generate test_conformance.py for this entry (needs --dim/--const as for `run`);
        /// `train:K` replays K training steps of the interpreter.
        #[arg(long, value_name = "ENTRY")]
        conformance_entry: Option<String>,
        /// Also write train.py and pskd.py: the file's `training run` as a PyTorch loop.
        #[arg(long)]
        train: bool,
        /// Also write serve.py and pskd.py: the file's `inference serve` as a PyTorch generation loop.
        #[arg(long)]
        serve: bool,
        #[arg(long = "dim", value_name = "NAME=N")]
        dims: Vec<String>,
        #[arg(long = "const", value_name = "NAME=N")]
        consts: Vec<String>,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Rewrite the program with the e-graph optimizer before computing the conformance data.
        #[arg(long)]
        opt: bool,
        /// With --opt: the machine profile for the extraction cost.
        #[arg(long, value_name = "FILE.json")]
        profile: Option<PathBuf>,
    },
    /// Dump an intermediate representation (development aid).
    Dump {
        file: PathBuf,
        #[arg(long, value_enum)]
        stage: DumpStage,
        /// Unroll `stack{L}` into L calls instead of a fold (debugging aid).
        #[arg(long)]
        unroll_stack: bool,
        /// For `--stage kernel`: the entry to run (as `presk run`), with --dim/--const/--realize.
        #[arg(long)]
        entry: Option<String>,
        #[arg(long = "dim", value_name = "NAME=N")]
        dims: Vec<String>,
        #[arg(long = "const", value_name = "NAME=N")]
        consts: Vec<String>,
        #[arg(long, value_enum, default_value_t = RealizeArg::F64)]
        realize: RealizeArg,
        /// For `--stage core-opt`: extract by the roofline of this machine profile.
        #[arg(long, value_name = "FILE.json")]
        profile: Option<PathBuf>,
    },
    /// Run the `training run` of a file on the reference interpreter.
    Train {
        file: PathBuf,
        #[arg(long = "const", value_name = "NAME=N")]
        consts: Vec<String>,
        /// Override the run's data file.
        #[arg(long, value_name = "FILE.psk.d")]
        data: Option<PathBuf>,
        /// Stop after N steps (the schedule's horizon becomes N).
        #[arg(long)]
        steps: Option<u64>,
        /// Override the run's `epochs`.
        #[arg(long)]
        epochs: Option<u64>,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// MPFR precision of every intermediate.
        #[arg(long, default_value_t = 256)]
        bits: u32,
        /// Print one line every N steps (0: only the last).
        #[arg(long, default_value_t = 1)]
        log_every: u64,
        /// Override the checkpoint directory (`final.psk.d` is written there).
        #[arg(long, value_name = "DIR")]
        ckpt: Option<PathBuf>,
        /// Machine-readable log on stdout.
        #[arg(long)]
        json: bool,
        /// The executor: the reference interpreter (native backends come with 0.3.0).
        #[arg(long, value_enum, default_value_t = BackendKind::Interp)]
        backend: BackendKind,
        /// Machine type of a native realization (ignored by the interpreter).
        #[arg(long, value_enum, default_value_t = RealizeArg::F64)]
        realize: RealizeArg,
        /// Rewrite the program with the e-graph optimizer before training (`presk-opt`).
        #[arg(long)]
        opt: bool,
        /// Workers a parallel kernel may use on the native backend (default: the cores).
        #[arg(long)]
        threads: Option<usize>,
        /// With --opt: the machine profile for the extraction cost.
        #[arg(long, value_name = "FILE.json")]
        profile: Option<PathBuf>,
    },
    /// Generate tokens with the file's `inference serve`: its model, weights and generation function.
    Serve {
        file: PathBuf,
        #[arg(long = "const", value_name = "NAME=N")]
        consts: Vec<String>,
        /// The weights (a `params M` .psk.d file); default: the block's `weights`.
        #[arg(long, value_name = "FILE.psk.d")]
        weights: Option<PathBuf>,
        /// The prompt: token indices separated by spaces.
        #[arg(long, value_name = "\"t1 t2 …\"")]
        prompt: String,
        /// Tokens to generate.
        #[arg(long, default_value_t = 16)]
        tokens: u64,
        /// The generation function (default: the file's, found by its signature).
        #[arg(long, value_name = "FN")]
        entry: Option<String>,
        /// Dimensions of the model not fixed by the prompt (the `T` of a positional embedding).
        #[arg(long = "dim", value_name = "NAME=N")]
        dims: Vec<String>,
        /// Sampling temperature; 0 is greedy (argmax). Default: the function's own default.
        #[arg(long)]
        temperature: Option<f64>,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// MPFR precision of the reference interpreter.
        #[arg(long, default_value_t = 256)]
        bits: u32,
        #[arg(long, value_enum, default_value_t = BackendKind::Interp)]
        backend: BackendKind,
        #[arg(long, value_enum, default_value_t = RealizeArg::F64)]
        realize: RealizeArg,
        /// Machine-readable output on stdout.
        #[arg(long)]
        json: bool,
        /// Rewrite the program with the e-graph optimizer before generating (`presk-opt`).
        #[arg(long)]
        opt: bool,
        /// With --opt: the machine profile for the extraction cost.
        #[arg(long, value_name = "FILE.json")]
        profile: Option<PathBuf>,
    },
    /// Time an entry on every backend (the reference at --bits, the kernel interpreter, native f64 and f32).
    Bench {
        file: PathBuf,
        #[arg(long)]
        entry: String,
        #[arg(long = "dim", value_name = "NAME=N")]
        dims: Vec<String>,
        #[arg(long = "const", value_name = "NAME=N")]
        consts: Vec<String>,
        /// Precision of the reference interpreter in the table.
        #[arg(long, default_value_t = 64)]
        bits: u32,
        /// Calls per backend; the median is reported.
        #[arg(long, default_value_t = 5)]
        repeat: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Rewrite the program with the e-graph optimizer before timing it (`presk-opt`).
        #[arg(long)]
        opt: bool,
        /// Workers a parallel kernel may use on the native backend (default: the cores).
        #[arg(long)]
        threads: Option<usize>,
        /// With --opt: the machine profile for the extraction cost.
        #[arg(long, value_name = "FILE.json")]
        profile: Option<PathBuf>,
        /// Time every kernel the entry launches natively, fit the roofline (π, β, λ) and write the profile;
        /// an existing file's kernels are kept and refitted with the new ones.
        #[arg(long, value_name = "FILE.json")]
        calibrate: Option<PathBuf>,
        /// The realization to calibrate (kernels are timed in this machine type).
        #[arg(long, value_enum, default_value_t = RealizeArg::F64)]
        realize: RealizeArg,
    },
    /// Produce and inspect `.psk.d` data files.
    Data {
        #[command(subcommand)]
        command: DataCommand,
    },
}

#[derive(Subcommand)]
enum DataCommand {
    /// Pack a text file (one sample per line, whitespace-separated values) into a .psk.d of the given type.
    Pack {
        /// The sample type, in Presk syntax: '(idx[64][seq: 8], idx[64][seq: 8])', 'ℝ[model: 4]'.
        #[arg(long = "type", value_name = "TYPE")]
        ty: String,
        input: PathBuf,
        #[arg(short, long, value_name = "FILE.psk.d")]
        out: PathBuf,
    },
    /// Build a character vocabulary from a UTF-8 text file: a JSON array of tokens, index `i` denoting token `i`.
    Chars {
        input: PathBuf,
        #[arg(short, long, value_name = "FILE.json")]
        out: PathBuf,
    },
    /// Encode a text file against a vocabulary into a whitespace-separated token stream (feeds `lm-windows`).
    Encode {
        input: PathBuf,
        #[arg(long, value_name = "FILE.json")]
        vocab: PathBuf,
        #[arg(short, long, value_name = "FILE.txt")]
        out: PathBuf,
    },
    /// Language-modelling windows (x, y = x shifted by one) of a whitespace-separated token stream.
    LmWindows {
        /// Vocabulary size V: every token is below V.
        #[arg(long)]
        vocab: u64,
        /// Window length T.
        #[arg(long)]
        seq: usize,
        /// Distance between the starts of two windows (default: T, non-overlapping).
        #[arg(long)]
        stride: Option<usize>,
        input: PathBuf,
        #[arg(short, long, value_name = "FILE.psk.d")]
        out: PathBuf,
    },
    /// Print the header and the first samples of a .psk.d file.
    Info {
        file: PathBuf,
        /// Number of samples to print.
        #[arg(long, default_value_t = 3)]
        samples: usize,
    },
    /// Write the samples of a .psk.d file as JSON (nested lists).
    Export {
        file: PathBuf,
        #[arg(short, long, value_name = "FILE.json")]
        out: PathBuf,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum ExportTarget {
    Pytorch,
}

#[derive(Clone, Copy, ValueEnum)]
enum ExportMode {
    Readable,
    Graph,
}

#[derive(Clone, Copy, ValueEnum)]
enum StackMode {
    Loop,
    Unroll,
}

#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum BackendKind {
    /// The reference interpreter (MPFR at --bits).
    Interp,
    /// PSK-Kernel v0 interpreted in the --realize machine type.
    Kernel,
    /// PSK-Kernel v0 compiled to machine code by Cranelift (in process).
    Native,
}

/// `--threads N`: how many workers a parallel kernel may use. Read by the
/// native executor; 1 keeps every kernel on one thread.
fn set_threads(n: Option<usize>) {
    if let Some(n) = n {
        // SAFETY: set before any backend is built and before any thread is
        // spawned, from the single-threaded `main`.
        unsafe { std::env::set_var("PRESK_THREADS", n.max(1).to_string()) };
    }
}

/// Runs `f` with the chosen backend over `core`.
fn with_backend<R>(kind: BackendKind, core: &presk_core::Program, prog: &elab::ElabProgram, bits: u32, seed: u64, realize: presk_rt::Realize, f: impl FnOnce(&mut dyn presk_rt::Backend) -> R) -> R {
    match kind {
        BackendKind::Interp => f(&mut presk_rt::InterpBackend::new(core, prog, bits, seed)),
        BackendKind::Kernel => f(&mut presk_kernel::KernelBackend::new(core, prog, realize, seed)),
        BackendKind::Native => match presk_cranelift::native_backend(core, prog, realize, seed) {
            Ok(mut b) => f(&mut b),
            Err(e) => {
                eprintln!("presk: the native backend is unavailable: {}", e.message);
                f(&mut presk_kernel::KernelBackend::new(core, prog, realize, seed))
            }
        },
    }
}

/// `--opt [--profile FILE]`.
struct Opt {
    on: bool,
    profile: Option<PathBuf>,
}

impl Opt {
    /// Rewrite `core` with the e-graph optimizer when `--opt` is on; one
    /// summary line on stderr (the whole report with `--verbose` or `dump
    /// --stage core-opt`). `dims` are the run's dimensions, for the roofline.
    fn apply(&self, core: &mut presk_core::Program, prog: &elab::ElabProgram, verbose: bool, dims: &[(String, u64)]) -> Result<(), ExitCode> {
        if !self.on {
            return Ok(());
        }
        let profile = match &self.profile {
            Some(path) => match presk_opt::Profile::load(path) {
                Ok(p) => Some(p),
                Err(m) => {
                    eprintln!("presk: --profile: {m}");
                    return Err(ExitCode::from(2));
                }
            },
            None => None,
        };
        let opts = presk_opt::Options { limits: presk_opt::Limits::default(), profile, dims: dims.iter().cloned().collect(), default_dim: None };
        let report = presk_opt::optimize(core, prog, &opts);
        let summary = report.summary();
        if verbose {
            eprint!("{summary}");
        } else {
            eprintln!("{}", summary.lines().next().unwrap_or(""));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum RealizeArg {
    F64,
    F32,
}

impl RealizeArg {
    fn to_rt(self) -> presk_rt::Realize {
        match self {
            RealizeArg::F64 => presk_rt::Realize::F64,
            RealizeArg::F32 => presk_rt::Realize::F32,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum DumpStage {
    Tokens,
    Ast,
    Elab,
    Core,
    CoreAfterAd,
    /// PSK-Core after AD and the e-graph optimizer (`presk-opt`); the report goes to stderr.
    CoreOpt,
    /// The fusion groups of every function (`presk-sched`): what shares a loop nest.
    Sched,
    /// The PSK-Kernel v0 kernels launched by an entry (needs --entry, --dim/--const as `presk run`).
    Kernel,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Check { file, consts, summary, json } => check(&file, &consts, summary, json),
        Command::Dump { file, stage, unroll_stack, entry, dims, consts, realize, profile } => dump(&file, stage, unroll_stack, entry.as_deref(), &dims, &consts, realize.to_rt(), profile),
        Command::Run { file, entry, dims, consts, seed, bits, digits, json, verbose, unroll_stack, backend, realize, opt, profile, data, samples, threads } => { set_threads(threads); run(&file, entry.as_deref(), &dims, &consts, seed, bits, digits as usize, json, verbose, unroll_stack, backend, realize.to_rt(), Opt { on: opt, profile }, data, samples) }
        Command::Export { file, target: _, mode, out, shapes, stack, conformance_entry, train, serve, dims, consts, seed, opt, profile } => export(&file, mode, &out, shapes.as_deref(), stack, conformance_entry.as_deref(), train, serve, &dims, &consts, seed, Opt { on: opt, profile }),
        Command::Serve { file, consts, weights, prompt, tokens, entry, dims, temperature, seed, bits, backend, realize, json, opt, profile } => {
            let mut prompt_tokens = Vec::new();
            for t in prompt.split_whitespace() {
                match t.parse::<u64>() {
                    Ok(n) => prompt_tokens.push(n),
                    Err(_) => {
                        eprintln!("presk serve: --prompt takes token indices separated by spaces, not `{t}`");
                        return ExitCode::from(2);
                    }
                }
            }
            let mut named = std::collections::BTreeMap::new();
            for d in &dims {
                match d.split_once('=').and_then(|(n, v)| v.parse::<u64>().ok().map(|v| (n.to_string(), v))) {
                    Some((n, v)) => {
                        named.insert(n, v);
                    }
                    None => {
                        eprintln!("presk serve: --dim takes NAME=N, not `{d}`");
                        return ExitCode::from(2);
                    }
                }
            }
            let opts = presk_serve::ServeOptions { weights, prompt: prompt_tokens, tokens, seed, bits, realize: realize.to_rt(), temperature, dims: named };
            serve(&file, &consts, entry.as_deref(), opts, backend, json, Opt { on: opt, profile })
        }
        Command::Train { file, consts, data, steps, epochs, seed, bits, log_every, ckpt, json, backend, realize, opt, profile, threads } => { set_threads(threads); train(&file, &consts, presk_train::TrainOptions { data, steps, epochs, seed, bits, realize: realize.to_rt(), ckpt, checkpoints: true }, log_every, json, backend, Opt { on: opt, profile }) }
        Command::Bench { file, entry, dims, consts, bits, repeat, seed, opt, profile, calibrate, realize, threads } => { set_threads(threads); bench(&file, &entry, &dims, &consts, bits, repeat, seed, Opt { on: opt, profile }, calibrate.as_deref(), realize.to_rt()) }
        Command::Data { command } => match data(command) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                let usage = e.code == "E9003" && e.message.contains("No such file");
                eprint!("{}", render_all(&[presk_syntax::Diagnostic::error(e.code, e.message)], &SourceMap::new(), color()));
                ExitCode::from(if usage { 2 } else { 1 })
            }
        },
    }
}

/// `presk serve`: the generation loop of the file's `inference serve`.
fn serve(file: &Path, consts: &[String], entry: Option<&str>, opts: presk_serve::ServeOptions, backend: BackendKind, json: bool, opt: Opt) -> ExitCode {
    let (sm, prog, typed) = match typed_front(file, consts) {
        Err(c) => return c,
        Ok(x) => x,
    };
    if typed.inference.is_empty() {
        eprintln!("presk serve: {} has no `inference serve`", file.display());
        return ExitCode::from(2);
    }
    let (mut core, diags) = presk_core::lower(&prog, &typed);
    eprint!("{}", render_all(&diags, &sm, color()));
    if has_errors(&diags) {
        return ExitCode::from(1);
    }
    let diags = presk_core::ad::transform(&mut core);
    eprint!("{}", render_all(&diags, &sm, color()));
    if has_errors(&diags) {
        return ExitCode::from(1);
    }
    let fns = match presk_serve::prepare_serve(&mut core, &prog, &typed, entry) {
        Ok(f) => f,
        Err(e) => {
            eprint!("{}", render_all(&[presk_syntax::Diagnostic::error(e.code, e.message)], &sm, color()));
            return ExitCode::from(1);
        }
    };
    let serve_dims: Vec<(String, u64)> = opts.dims.iter().map(|(n, v)| (n.clone(), *v)).collect();
    if let Err(c) = opt.apply(&mut core, &prog, false, &serve_dims) {
        return c;
    }
    let on_token = |_step: u64, _tok: u64| {};
    match with_backend(backend, &core, &prog, opts.bits, opts.seed, opts.realize, |b| presk_serve::serve(b, &core, &prog, &fns, &opts, on_token)) {
        Err(e) => {
            let usage = e.code == "E9003";
            let mut d = presk_syntax::Diagnostic::error(e.code, e.message);
            if let Some(s) = e.span {
                d = d.label(s, "while evaluating this");
            }
            eprint!("{}", render_all(&[d], &sm, color()));
            ExitCode::from(if usage { 2 } else { 1 })
        }
        Ok(out) => {
            if json {
                let j = serde_json::json!({
                    "model": out.model,
                    "weights": out.weights.display().to_string(),
                    "prompt": out.prompt,
                    "generated": out.generated,
                    "window": out.window,
                    "greedy": out.greedy,
                    "vocab": out.vocab,
                });
                println!("{}", serde_json::to_string_pretty(&j).unwrap());
            } else {
                let ctx = match out.window {
                    Some(w) => format!("context window {w}"),
                    None => "whole context".to_string(),
                };
                eprintln!("presk serve: {} from {}, prompt of {} tokens, {} generated, {}, {}", out.model, out.weights.display(), out.prompt.len(), out.generated.len(), if out.greedy { "greedy" } else { "sampled" }, ctx);
                println!("{}", out.generated.iter().map(u64::to_string).collect::<Vec<_>>().join(" "));
            }
            ExitCode::SUCCESS
        }
    }
}

/// `presk bench`: one entry timed on every backend.
#[allow(clippy::too_many_arguments)]
fn bench(file: &Path, entry: &str, dims: &[String], consts: &[String], bits: u32, repeat: usize, seed: u64, opt: Opt, calibrate: Option<&Path>, realize: presk_rt::Realize) -> ExitCode {
    let dims = match parse_dims(dims) {
        Ok(d) => d,
        Err(c) => return c,
    };
    let (sm, prog, mut core) = match core(file, consts, false) {
        Err(c) => return c,
        Ok(x) => x,
    };
    let diags = presk_core::ad::transform(&mut core);
    eprint!("{}", render_all(&diags, &sm, color()));
    if has_errors(&diags) {
        return ExitCode::from(1);
    }
    if let Err(c) = opt.apply(&mut core, &prog, false, &dims) {
        return c;
    }
    let opts = presk_interp::RunOptions { entry: entry.to_string(), dims, seed, bits, digits: 12 };
    let (func, fdims, inputs) = match presk_interp::run::prepare_run(&core, &prog, &opts) {
        Ok(x) => x,
        Err(e) => {
            eprint!("{}", render_all(&[presk_syntax::Diagnostic::error("E7001", e.message)], &sm, color()));
            return ExitCode::from(1);
        }
    };
    let args: Vec<presk_rt::Value> = inputs.iter().map(|(_, v)| presk_rt::Value::from_interp(v).unwrap()).collect();
    if let Some(path) = calibrate {
        return calibrate_profile(&core, &prog, func, &fdims, &args, realize, repeat.max(3), seed, path);
    }
    let time = |name: &str, b: &mut dyn presk_rt::Backend, realize: presk_rt::Realize| -> Option<(String, f64, f64)> {
        let p = b.prepare(func, &fdims, realize).ok()?;
        let mut samples = Vec::new();
        for _ in 0..repeat.max(1) {
            let t0 = std::time::Instant::now();
            if let Err(e) = b.call(&p, args.clone()) {
                eprintln!("presk bench: {name}: {}", e.message);
                return None;
            }
            samples.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some((name.to_string(), samples[samples.len() / 2], samples[0]))
    };
    let mut rows = Vec::new();
    if let Some(r) = time(&format!("interp {bits} bits"), &mut presk_rt::InterpBackend::new(&core, &prog, bits, seed), presk_rt::Realize::F64) {
        rows.push(r);
    }
    if let Some(r) = time("kernel f64", &mut presk_kernel::KernelBackend::new(&core, &prog, presk_rt::Realize::F64, seed), presk_rt::Realize::F64) {
        rows.push(r);
    }
    for (name, realize) in [("native f64", presk_rt::Realize::F64), ("native f32", presk_rt::Realize::F32)] {
        match presk_cranelift::native_backend(&core, &prog, realize, seed) {
            Ok(mut b) => {
                if let Some(r) = time(name, &mut b, realize) {
                    rows.push(r);
                }
            }
            Err(e) => eprintln!("presk bench: native backend unavailable: {}", e.message),
        }
    }
    let base = rows.first().map(|r| r.1).unwrap_or(1.0);
    println!("{:<18} {:>12} {:>12} {:>9}", "backend", "median ms", "best ms", "speedup");
    for (name, med, best) in &rows {
        println!("{name:<18} {med:>12.3} {best:>12.3} {:>8.1}x", base / med);
    }
    ExitCode::SUCCESS
}

/// `presk bench --calibrate`: time every kernel of the entry natively, fit
/// the roofline and write (or extend) the profile.
#[allow(clippy::too_many_arguments)]
fn calibrate_profile(core: &presk_core::Program, prog: &elab::ElabProgram, func: presk_core::FuncId, fdims: &presk_interp::eval::Dims, args: &[presk_rt::Value], realize: presk_rt::Realize, repeat: usize, seed: u64, path: &Path) -> ExitCode {
    let native = match presk_cranelift::NativeExec::new() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("presk bench: native backend unavailable: {}", e.message);
            return ExitCode::from(1);
        }
    };
    let (timed, samples) = presk_kernel::cost::TimedExec::new(Box::new(native));
    let mut backend = presk_kernel::KernelBackend::with_exec(core, prog, realize, seed, Box::new(timed));
    let p = match presk_rt::Backend::prepare(&mut backend, func, fdims, realize) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("presk bench: {}", e.message);
            return ExitCode::from(1);
        }
    };
    for _ in 0..repeat {
        if let Err(e) = presk_rt::Backend::call(&mut backend, &p, args.to_vec()) {
            eprintln!("presk bench: {}", e.message);
            return ExitCode::from(1);
        }
    }
    let medians = presk_kernel::cost::median_per_kernel(&samples.borrow(), backend.kernels.len());
    let dtype = backend.dtype.to_string();
    let mut kernels: Vec<presk_opt::profile::Sample> = Vec::new();
    for (k, t) in backend.kernels.iter().zip(medians) {
        let Some(seconds) = t else { continue };
        let c = presk_kernel::cost::kernel_cost(k);
        let signature = k.to_string().lines().next().unwrap_or("").trim_end_matches(" {").to_string();
        kernels.push(presk_opt::profile::Sample { signature, name: k.name.clone(), work: c.work, bytes: c.bytes, seconds });
    }
    // Merge with an existing profile of the same realization: new measurements win.
    let mut merged: Vec<presk_opt::profile::Sample> = match presk_opt::Profile::load(path) {
        Ok(old) if old.dtype == dtype => old.kernels.into_iter().filter(|o| !kernels.iter().any(|n| n.signature == o.signature)).collect(),
        Ok(old) => {
            eprintln!("presk bench: {} was calibrated for {}, not {dtype}: starting over", path.display(), old.dtype);
            vec![]
        }
        Err(_) => vec![],
    };
    let kept = merged.len();
    merged.extend(kernels.iter().cloned());
    let machine = format!("{} {}", std::env::consts::OS, std::env::consts::ARCH);
    let profile = presk_opt::profile::fit(&merged, &dtype, &machine, &|s| presk_opt::cost::weight_of_name(&s.name));
    println!("{:<44} {:>12} {:>12} {:>11} {:>11}", "kernel", "work", "bytes", "measured µs", "predicted µs");
    for s in &kernels {
        let sig = if s.signature.len() > 44 { format!("{}…", &s.signature[..s.signature.char_indices().nth(43).map_or(s.signature.len(), |(i, _)| i)]) } else { s.signature.clone() };
        println!("{sig:<44} {:>12.0} {:>12.0} {:>11.2} {:>11.2}", s.work, s.bytes, s.seconds * 1e6, profile.predict(s.work, s.bytes) * 1e6);
    }
    println!("-- {} kernels timed ({} kept from {}), {} calls, {dtype}", kernels.len(), kept, path.display(), repeat);
    println!("-- roofline: π = {:.3e} ops/s, β = {:.3e} bytes/s, λ = {:.2} µs per launch", profile.pi, profile.beta, profile.lambda * 1e6);
    println!("-- Spearman with the measured times: operation counts {:.3}, work {:.3}, roofline {:.3}", profile.spearman.counts, profile.spearman.work, profile.spearman.roofline);
    match profile.save(path) {
        Ok(()) => {
            println!("-- profile written to {}", path.display());
            ExitCode::SUCCESS
        }
        Err(m) => {
            eprintln!("presk bench: {m}");
            ExitCode::from(2)
        }
    }
}

/// `presk train`: the loop of the file's `training run` (crate `presk-train`).
fn train(file: &Path, consts: &[String], opts: presk_train::TrainOptions, log_every: u64, json: bool, backend: BackendKind, opt: Opt) -> ExitCode {
    let (sm, prog, typed) = match typed_front(file, consts) {
        Err(c) => return c,
        Ok(x) => x,
    };
    if typed.training.is_empty() {
        eprintln!("presk train: {} has no `training run`", file.display());
        return ExitCode::from(2);
    }
    let (mut core, diags) = presk_core::lower(&prog, &typed);
    eprint!("{}", render_all(&diags, &sm, color()));
    if has_errors(&diags) {
        return ExitCode::from(1);
    }
    let diags = presk_core::ad::transform(&mut core);
    eprint!("{}", render_all(&diags, &sm, color()));
    if has_errors(&diags) {
        return ExitCode::from(1);
    }
    let mut lines: Vec<serde_json::Value> = Vec::new();
    let on_step = |l: &presk_train::StepLog| {
        if json {
            lines.push(serde_json::json!({ "step": l.step, "epoch": l.epoch, "loss": l.loss, "lr": l.lr, "grad_norm": l.grad_norm }));
        } else if log_every > 0 && l.step % log_every == 0 {
            eprintln!("{}", presk_train::format_step(l, 0).replace("/0", ""));
        }
    };
    let fns = match presk_train::prepare_training(&mut core, &prog, &typed) {
        Ok(f) => f,
        Err(e) => {
            eprint!("{}", render_all(&[presk_syntax::Diagnostic::error(e.code, e.message)], &sm, color()));
            return ExitCode::from(1);
        }
    };
    if let Err(c) = opt.apply(&mut core, &prog, false, &[]) {
        return c;
    }
    let mut stats = (0u64, 0u64);
    match with_backend(backend, &core, &prog, opts.bits, opts.seed, opts.realize, |b| {
        let out = presk_train::train(b, &core, &prog, &typed, &fns, &opts, on_step);
        stats = presk_rt::Backend::launch_stats(b);
        out
    }) {
        Err(e) => {
            let usage = e.code == "E9002";
            let mut d = presk_syntax::Diagnostic::error(e.code, e.message);
            if let Some(s) = e.span {
                d = d.label(s, "while evaluating this");
            }
            eprint!("{}", render_all(&[d], &sm, color()));
            ExitCode::from(if usage { 2 } else { 1 })
        }
        Ok(out) => {
            if json {
                let j = serde_json::json!({
                    "model": out.model,
                    "steps": out.logs.len(),
                    "total": out.total,
                    "log": lines,
                    "final": out.final_path.as_ref().map(|p| p.display().to_string()),
                    "checkpoints": out.checkpoints.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
                });
                println!("{}", serde_json::to_string_pretty(&j).unwrap());
            } else {
                if let Some(last) = out.logs.last()
                    && (log_every == 0 || last.step % log_every != 0)
                {
                    eprintln!("{}", presk_train::format_step(last, out.total));
                }
                let (launched, fused) = stats;
                if launched > 0 && !out.logs.is_empty() {
                    let steps = out.logs.len() as u64;
                    let before = launched + fused;
                    eprintln!(
                        "presk train: {} kernel launches a step, {} without fusion ({:.0}% removed)",
                        launched / steps,
                        before / steps,
                        100.0 * fused as f64 / before as f64
                    );
                }
                match &out.final_path {
                    Some(p) => println!("presk train: {} steps of {}, final loss {:.6}, weights in {}", out.logs.len(), out.model, out.logs.last().map(|l| l.loss).unwrap_or(f64::NAN), p.display()),
                    None => println!("presk train: {} steps of {}, final loss {:.6} (no checkpoint directory: give --ckpt or a `checkpoint … to` clause)", out.logs.len(), out.model, out.logs.last().map(|l| l.loss).unwrap_or(f64::NAN)),
                }
            }
            ExitCode::SUCCESS
        }
    }
}

/// `presk data …`: the `.psk.d` container (crate `presk-data`).
fn data(command: DataCommand) -> Result<(), presk_data::DataError> {
    use presk_data::{DataError, Layout, Reader, Value};
    let read_text = |p: &Path| std::fs::read_to_string(p).map_err(|e| DataError { code: "E9003", message: format!("{}: {e}", p.display()) });
    match command {
        DataCommand::Pack { ty, input, out } => {
            let layout = Layout::from_type_text(&ty)?;
            let text = read_text(&input)?;
            let mut samples = Vec::new();
            for (i, line) in text.lines().enumerate() {
                if line.trim().is_empty() || line.trim_start().starts_with("--") {
                    continue;
                }
                samples.push(Value::parse_text(&layout, line).map_err(|e| DataError { code: e.code, message: format!("{}:{}: {}", input.display(), i + 1, e.message) })?);
            }
            let n = presk_data::write(&out, &ty, &layout, &samples)?;
            eprintln!("presk data: wrote {} ({n} samples of type {ty})", out.display());
        }
        DataCommand::Chars { input, out } => {
            let text = read_text(&input)?;
            let vocab = presk_data::char_vocab(&text);
            let json = serde_json::to_string(&vocab).unwrap();
            std::fs::write(&out, json).map_err(|e| DataError { code: "E9003", message: format!("{}: {e}", out.display()) })?;
            eprintln!("presk data: wrote {} ({} characters over {} of text)", out.display(), vocab.len(), text.chars().count());
        }
        DataCommand::Encode { input, vocab, out } => {
            let words: Vec<String> = serde_json::from_str(&read_text(&vocab)?).map_err(|e| DataError { code: "E9004", message: format!("{}: not a vocabulary: {e}", vocab.display()) })?;
            let text = read_text(&input)?;
            let tokens = presk_data::encode(&text, &words)?;
            let mut s = String::with_capacity(tokens.len() * 3);
            for (i, t) in tokens.iter().enumerate() {
                if i > 0 {
                    s.push(' ');
                }
                s.push_str(&t.to_string());
            }
            std::fs::write(&out, s).map_err(|e| DataError { code: "E9003", message: format!("{}: {e}", out.display()) })?;
            eprintln!("presk data: wrote {} ({} tokens over a vocabulary of {})", out.display(), tokens.len(), words.len());
        }
        DataCommand::LmWindows { vocab, seq, stride, input, out } => {
            let text = read_text(&input)?;
            let tokens: Vec<u64> = text
                .split_whitespace()
                .map(|t| t.parse::<u64>().map_err(|_| DataError { code: "E9004", message: format!("{}: `{t}` is not a token", input.display()) }))
                .collect::<Result<_, _>>()?;
            let (ty, layout, samples) = presk_data::lm_windows(&tokens, vocab, seq, stride.unwrap_or(seq))?;
            let n = presk_data::write(&out, &ty, &layout, &samples)?;
            eprintln!("presk data: wrote {} ({n} windows of type {ty} from {} tokens)", out.display(), tokens.len());
        }
        DataCommand::Info { file, samples } => {
            let mut r = Reader::open(&file)?;
            print!("{}", presk_data::info(&mut r, samples)?);
        }
        DataCommand::Export { file, out } => {
            let mut r = Reader::open(&file)?;
            let layout = r.layout()?.clone();
            let h = r.header().clone();
            let vals: Vec<serde_json::Value> = r.read_all()?.iter().map(|v| v.to_json(&layout)).collect();
            let j = serde_json::json!({ "type": h.sample_ty, "count": h.count, "samples": vals });
            std::fs::write(&out, serde_json::to_string_pretty(&j).unwrap()).map_err(|e| DataError { code: "E9003", message: format!("{}: {e}", out.display()) })?;
            eprintln!("presk data: wrote {}", out.display());
        }
    }
    Ok(())
}

fn load(file: &Path) -> Result<(SourceMap, presk_syntax::FileId), ExitCode> {
    let mut sm = SourceMap::new();
    match sm.load(file) {
        Ok(id) => Ok((sm, id)),
        Err(e) => {
            eprintln!("presk: cannot read {}: {e}", file.display());
            Err(ExitCode::from(2))
        }
    }
}

fn color() -> bool {
    std::io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none()
}

/// `NAME=N` pairs of `--const`.
fn parse_consts(consts: &[String]) -> Result<Vec<(String, u64)>, ExitCode> {
    let mut out = Vec::new();
    for c in consts {
        match c.split_once('=').and_then(|(n, v)| v.trim().parse::<u64>().ok().map(|v| (n.trim().to_string(), v))) {
            Some(pair) => out.push(pair),
            None => {
                eprintln!("presk: `--const {c}`: expected NAME=N with N a natural number");
                return Err(ExitCode::from(2));
            }
        }
    }
    Ok(out)
}

/// Diagnostics as JSON: `{code, severity, message, labels: [{file, line, col, start, end, text}], notes, help}`.
fn diags_json(diags: &[presk_syntax::Diagnostic], sm: &SourceMap) -> serde_json::Value {
    serde_json::Value::Array(
        diags
            .iter()
            .map(|d| {
                let labels: Vec<serde_json::Value> = d
                    .labels
                    .iter()
                    .map(|l| {
                        let f = sm.file(l.span.file);
                        let (line, col) = f.line_col(l.span.start);
                        serde_json::json!({"file": f.name, "line": line, "col": col, "start": l.span.start, "end": l.span.end, "text": l.text})
                    })
                    .collect();
                serde_json::json!({
                    "code": d.code,
                    "severity": match d.severity { presk_syntax::Severity::Error => "error", presk_syntax::Severity::Warning => "warning" },
                    "message": d.message,
                    "labels": labels,
                    "notes": d.notes,
                    "help": d.help,
                })
            })
            .collect(),
    )
}

/// Parse and elaborate; report all diagnostics. Returns the program when
/// there is no error.
fn front(file: &Path, consts: &[String]) -> Result<(SourceMap, elab::ElabProgram), ExitCode> {
    let const_overrides = parse_consts(consts)?;
    let (mut sm, id) = load(file)?;
    let (program, mut diags) = parse_file(&sm, id);
    if has_errors(&diags) {
        eprint!("{}", render_all(&diags, &sm, color()));
        return Err(ExitCode::from(1));
    }
    let (prog, more) = elab::elaborate(&mut sm, id, program, &Options { const_overrides });
    diags.extend(more);
    eprint!("{}", render_all(&diags, &sm, color()));
    if has_errors(&diags) { Err(ExitCode::from(1)) } else { Ok((sm, prog)) }
}

/// Parse, elaborate and type; report diagnostics.
fn typed_front(file: &Path, consts: &[String]) -> Result<(SourceMap, elab::ElabProgram, TypedProgram), ExitCode> {
    let (sm, prog) = front(file, consts)?;
    let (typed, diags) = typing::check(&sm, &prog);
    eprint!("{}", render_all(&diags, &sm, color()));
    if has_errors(&diags) { Err(ExitCode::from(1)) } else { Ok((sm, prog, typed)) }
}

/// The complete static check. With `json`, nothing is rendered: every
/// diagnostic of every stage is collected and printed as one JSON document.
fn check(file: &Path, consts: &[String], summary: bool, json: bool) -> ExitCode {
    let mut all: Vec<presk_syntax::Diagnostic> = Vec::new();
    let mut sm = SourceMap::new();
    let code = check_stages(file, consts, summary && !json, json, &mut sm, &mut all);
    if json {
        let ok = !has_errors(&all) && code == ExitCode::SUCCESS;
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({"file": file.display().to_string(), "ok": ok, "diagnostics": diags_json(&all, &sm)})).unwrap());
    }
    code
}

/// Run the stages, rendering diagnostics as they come (or collecting them
/// when `quiet`). `sm` receives the source map so that collected diagnostics
/// can be located afterwards.
fn check_stages(file: &Path, consts: &[String], summary: bool, quiet: bool, sm_out: &mut SourceMap, all: &mut Vec<presk_syntax::Diagnostic>) -> ExitCode {
    let mut emit = |diags: &[presk_syntax::Diagnostic], sm: &SourceMap| {
        if quiet {
            all.extend(diags.iter().cloned());
        } else {
            eprint!("{}", render_all(diags, sm, color()));
        }
    };
    let const_overrides = match parse_consts(consts) {
        Ok(c) => c,
        Err(code) => return code,
    };
    let (mut sm, id) = match load(file) {
        Ok(x) => x,
        Err(code) => return code,
    };
    let (program, diags) = parse_file(&sm, id);
    emit(&diags, &sm);
    if has_errors(&diags) {
        *sm_out = sm;
        return ExitCode::from(1);
    }
    let (prog, diags) = elab::elaborate(&mut sm, id, program, &Options { const_overrides });
    emit(&diags, &sm);
    if has_errors(&diags) {
        *sm_out = sm;
        return ExitCode::from(1);
    }
    let (typed, diags) = typing::check(&sm, &prog);
    emit(&diags, &sm);
    if has_errors(&diags) {
        *sm_out = sm;
        return ExitCode::from(1);
    }
    // Regularity classes and effects are analyzed on PSK-Core.
    let (core, diags) = presk_core::lower(&prog, &typed);
    emit(&diags, &sm);
    if has_errors(&diags) {
        *sm_out = sm;
        return ExitCode::from(1);
    }
    if let Err(errs) = presk_core::verify(&core) {
        for e in errs {
            eprintln!("presk: internal error (PSK-Core verification): {e}");
        }
        *sm_out = sm;
        return ExitCode::from(2);
    }
    let (analysis, diags) = presk_core::analyze(&core, &prog);
    emit(&diags, &sm);
    let failed = has_errors(&diags);
    if summary && !failed {
        print!("{}", typing::summary(&prog, &typed));
        print!("-- regularity and effects --\n{}", presk_core::analysis::summary(&core, &analysis));
        if !typed.training.is_empty() {
            print!("-- training --\n{}", typing::training::summary(&prog, &typed.training));
        }
        if !typed.inference.is_empty() {
            print!("-- inference --\n{}", typing::inference::summary(&prog, &typed.inference));
        }
    }
    *sm_out = sm;
    if failed { ExitCode::from(1) } else { ExitCode::SUCCESS }
}

/// Lower to PSK-Core and verify the result.
fn core(file: &Path, consts: &[String], unroll_stack: bool) -> Result<(SourceMap, elab::ElabProgram, presk_core::Program), ExitCode> {
    let (sm, prog, typed) = typed_front(file, consts)?;
    let (core, diags) = presk_core::lower::lower_with(&prog, &typed, &presk_core::lower::Options { unroll_stack });
    eprint!("{}", render_all(&diags, &sm, color()));
    if has_errors(&diags) {
        return Err(ExitCode::from(1));
    }
    if let Err(errs) = presk_core::verify(&core) {
        for e in errs {
            eprintln!("presk: internal error (PSK-Core verification): {e}");
        }
        return Err(ExitCode::from(2));
    }
    Ok((sm, prog, core))
}

#[allow(clippy::too_many_arguments)]
fn dump(file: &Path, stage: DumpStage, unroll_stack: bool, entry: Option<&str>, dims: &[String], consts: &[String], realize: presk_rt::Realize, profile: Option<PathBuf>) -> ExitCode {
    if let DumpStage::Kernel = stage {
        let Some(entry) = entry else {
            eprintln!("presk dump --stage kernel: --entry is required (the kernels are specialized on the entry's shapes)");
            return ExitCode::from(2);
        };
        let dims = match parse_dims(dims) {
            Ok(d) => d,
            Err(c) => return c,
        };
        let (sm, prog, mut core) = match core(file, consts, unroll_stack) {
            Err(c) => return c,
            Ok(x) => x,
        };
        let diags = presk_core::ad::transform(&mut core);
        eprint!("{}", render_all(&diags, &sm, color()));
        if has_errors(&diags) {
            return ExitCode::from(1);
        }
        let opts = presk_interp::RunOptions { entry: entry.to_string(), dims, seed: 0, bits: 64, digits: 12 };
        let mut backend = presk_kernel::KernelBackend::new(&core, &prog, realize, 0);
        if let Err(e) = presk_rt::run_with(&mut backend, &core, &prog, &opts, realize) {
            eprint!("{}", render_all(&[presk_syntax::Diagnostic::error(e.code, e.message)], &sm, color()));
            return ExitCode::from(1);
        }
        let before = backend.launches + backend.fused;
        let pct = if before > 0 { 100.0 * backend.fused as f64 / before as f64 } else { 0.0 };
        println!("-- {} kernels, {} launches, {}", backend.kernels.len(), backend.launches, backend.dtype);
        println!("-- fusion: {before} launches without it, {} saved ({pct:.0}%)", backend.fused);
        for k in &backend.kernels {
            print!("{k}");
        }
        return ExitCode::SUCCESS;
    }
    let (sm, id) = match load(file) {
        Ok(x) => x,
        Err(c) => return c,
    };
    match stage {
        DumpStage::Kernel => unreachable!("handled above"),
        DumpStage::Tokens => {
            let lexed = lexer::lex(id, sm.text(id));
            print!("{}", lexer::dump_tokens(&lexed.tokens));
            eprint!("{}", render_all(&lexed.diags, &sm, color()));
            if has_errors(&lexed.diags) { ExitCode::from(1) } else { ExitCode::SUCCESS }
        }
        DumpStage::Ast => {
            let (program, diags) = parse_file(&sm, id);
            println!("{program:#?}");
            eprint!("{}", render_all(&diags, &sm, color()));
            if has_errors(&diags) { ExitCode::from(1) } else { ExitCode::SUCCESS }
        }
        DumpStage::Elab => match front(file, &[]) {
            Err(c) => c,
            Ok((_, prog)) => {
                print!("{}", elab::dump(&prog));
                ExitCode::SUCCESS
            }
        },
        DumpStage::Core => match core(file, &[], unroll_stack) {
            Err(c) => c,
            Ok((_, prog, core)) => {
                print!("{}", presk_core::pretty(&core, &prog));
                ExitCode::SUCCESS
            }
        },
        DumpStage::Sched => match core(file, consts, unroll_stack) {
            Err(c) => c,
            Ok((sm, _prog, mut core)) => {
                let diags = presk_core::ad::transform(&mut core);
                eprint!("{}", render_all(&diags, &sm, color()));
                if has_errors(&diags) {
                    return ExitCode::from(1);
                }
                let (mut lets, mut saved, mut groups) = (0usize, 0usize, 0usize);
                for f in &core.funcs {
                    let plan = presk_sched::plan(f);
                    lets += f.body.lets.len();
                    saved += plan.saved();
                    groups += plan.groups.len();
                    if !plan.groups.is_empty() {
                        print!("{}", presk_sched::dump(f, &plan));
                    }
                }
                println!("-- {} functions, {lets} lets, {groups} groups, {saved} lets fused away", core.funcs.len());
                ExitCode::SUCCESS
            }
        },
        DumpStage::CoreAfterAd | DumpStage::CoreOpt => match core(file, consts, unroll_stack) {
            Err(c) => c,
            Ok((sm, prog, mut core)) => {
                let diags = presk_core::ad::transform(&mut core);
                eprint!("{}", render_all(&diags, &sm, color()));
                if has_errors(&diags) {
                    return ExitCode::from(1);
                }
                if let Err(errs) = presk_core::verify(&core) {
                    for e in errs {
                        eprintln!("presk: internal error (PSK-Core verification after AD): {e}");
                    }
                    return ExitCode::from(2);
                }
                if let DumpStage::CoreOpt = stage {
                    let dims = parse_dims(dims).unwrap_or_default();
                    if let Err(c) = (Opt { on: true, profile }).apply(&mut core, &prog, true, &dims) {
                        return c;
                    }
                }
                print!("{}", presk_core::pretty(&core, &prog));
                ExitCode::SUCCESS
            }
        },
    }
}

/// `NAME=N` pairs of `--dim`.
fn parse_dims(dims: &[String]) -> Result<Vec<(String, u64)>, ExitCode> {
    parse_consts(dims).inspect_err(|_| eprintln!("presk: (in `--dim`)"))
}

#[allow(clippy::too_many_arguments)]
fn run(file: &Path, entry: Option<&str>, dims: &[String], consts: &[String], seed: u64, bits: u32, digits: usize, json: bool, verbose: bool, unroll_stack: bool, backend: BackendKind, realize: presk_rt::Realize, opt: Opt, data: Option<PathBuf>, samples: Option<usize>) -> ExitCode {
    let dims = match parse_dims(dims) {
        Ok(d) => d,
        Err(c) => return c,
    };
    let (sm, prog, mut core) = match core(file, consts, unroll_stack) {
        Err(c) => return c,
        Ok(x) => x,
    };
    let Some(entry) = entry else {
        // No entry: a `training run` is not executable in 0.1.0; otherwise list the candidates.
        if !prog.units[prog.main].training.is_empty() {
            eprintln!("presk run: this file has a `training run`: use `presk train` to execute it, or --entry to evaluate one function");
            return ExitCode::from(2);
        }
        let candidates: Vec<String> = core.funcs.iter().filter(|f| matches!(f.origin, presk_core::ir::Origin::Source(_))).map(|f| f.name.clone()).collect();
        eprintln!("presk run: --entry is required; functions: {}", candidates.join(", "));
        return ExitCode::from(2);
    };
    let diags = presk_core::ad::transform(&mut core);
    eprint!("{}", render_all(&diags, &sm, color()));
    if has_errors(&diags) {
        return ExitCode::from(1);
    }
    if let Err(errs) = presk_core::verify(&core) {
        for e in errs {
            eprintln!("presk: internal error (PSK-Core verification after AD): {e}");
        }
        return ExitCode::from(2);
    }
    if let Err(c) = opt.apply(&mut core, &prog, verbose, &dims) {
        return c;
    }
    let opts = presk_interp::RunOptions { entry: entry.to_string(), dims, seed, bits, digits };
    let evaluate = |b: &mut dyn presk_rt::Backend| match &data {
        Some(path) => presk_train::run_on_data(b, &core, &prog, &opts, realize, path, samples).map_err(|e| presk_rt::RtError { code: e.code, message: e.message, span: e.span }),
        None => presk_rt::run_with(b, &core, &prog, &opts, realize),
    };
    match with_backend(backend, &core, &prog, bits, seed, realize, evaluate) {
        Err(e) => {
            let mut d = presk_syntax::Diagnostic::error(e.code, e.message);
            if let Some(s) = e.span {
                d = d.label(s, "while evaluating this");
            }
            eprint!("{}", render_all(&[d], &sm, color()));
            ExitCode::from(1)
        }
        Ok(out) => {
            if json {
                let inputs: serde_json::Map<String, serde_json::Value> = out.inputs.iter().map(|(n, v)| (n.clone(), presk_interp::run::to_json(v, digits))).collect();
                let j = serde_json::json!({
                    "entry": out.entry,
                    "seed": seed,
                    "bits": bits,
                    "dims": out.dims.iter().map(|(n, v)| (n.clone(), serde_json::json!(v))).collect::<serde_json::Map<_, _>>(),
                    "inputs": inputs,
                    "result": presk_interp::run::to_json(&out.result, digits),
                });
                println!("{}", serde_json::to_string_pretty(&j).unwrap());
            } else {
                if verbose {
                    println!("-- entry {} with {} at {} bits, seed {}", out.entry, out.dims.iter().map(|(n, v)| format!("{n} = {v}")).collect::<Vec<_>>().join(", "), bits, seed);
                    for (n, v) in &out.inputs {
                        println!("{n} : {} = {}", presk_interp::run::describe(v), presk_interp::run::render(v, digits, 0));
                    }
                    println!("-- result");
                }
                println!("{} = {}", presk_interp::run::describe(&out.result), presk_interp::run::render(&out.result, digits, 0));
            }
            ExitCode::SUCCESS
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn export(file: &Path, mode: ExportMode, out: &Path, shapes: Option<&str>, stack: StackMode, conformance_entry: Option<&str>, train: bool, serve: bool, dims: &[String], consts: &[String], seed: u64, opt: Opt) -> ExitCode {
    let graph = matches!(mode, ExportMode::Graph);
    if graph && conformance_entry.is_none_or(|e| e.starts_with("train:")) {
        eprintln!("presk export: --mode graph needs --conformance-entry with a module entry (e.g. MiniGPT.apply): the graphs are built from its inputs");
        return ExitCode::from(2);
    }
    if let Some(s) = shapes
        && s != "check"
    {
        eprintln!("presk export: --shapes accepts only `check`");
        return ExitCode::from(2);
    }
    let (sm, prog, typed) = match typed_front(file, consts) {
        Err(c) => return c,
        Ok(x) => x,
    };
    let opts = presk_export_pytorch::Options {
        shapes_check: shapes.is_some(),
        stack: match stack {
            StackMode::Loop => presk_export_pytorch::StackMode::Loop,
            StackMode::Unroll => presk_export_pytorch::StackMode::Unroll,
        },
    };
    let (files, diags) = presk_export_pytorch::export(&sm, &prog, &typed, &opts);
    eprint!("{}", render_all(&diags, &sm, color()));
    if has_errors(&diags) {
        return ExitCode::from(1);
    }
    if let Err(e) = std::fs::create_dir_all(out) {
        eprintln!("presk export: cannot create {}: {e}", out.display());
        return ExitCode::from(2);
    }
    let mut written = Vec::new();
    for f in &files {
        let path = out.join(&f.name);
        if let Err(e) = std::fs::write(&path, &f.content) {
            eprintln!("presk export: cannot write {}: {e}", path.display());
            return ExitCode::from(2);
        }
        written.push(path.display().to_string());
    }
    // `--serve`: the inference block as serve.py + pskd.py.
    if serve {
        let Some(sv) = typed.inference.first() else {
            eprintln!("presk export: --serve: {} has no `inference serve`", file.display());
            return ExitCode::from(2);
        };
        let presk_types::ty::ModRef::Inst { def, .. } = &sv.model else {
            eprintln!("presk export: --serve: the model must be a plain module");
            return ExitCode::from(2);
        };
        let extra: Vec<String> = presk_export_pytorch::extra_dims(&prog, *def).iter().map(|v| prog.dim_name(*v)).collect();
        let (generate, temperature, seq_dim) = match sv.generate {
            Some(f) => {
                let f = prog.function(f);
                let seq = f.params.get(1).and_then(|p| p.ty.as_ref()).and_then(|t| match t {
                    presk_types::ty::Ty::Idx { shape, .. } => shape.axes.iter().find(|a| a.name.as_deref() == Some("seq")).and_then(|a| a.len.vars().next()).map(|v| prog.dim_name(v)),
                    _ => None,
                });
                let temp = f.params.get(2).and_then(|p| p.default.as_ref()).and_then(|e| match e {
                    presk_syntax::ast::Expr::Lit { lit: presk_syntax::ast::Lit::Float(x), .. } => Some(x.clone()),
                    _ => None,
                });
                (Some(f.name.clone()), if f.params.len() > 2 { Some(temp.unwrap_or_else(|| "1.0".into())) } else { None }, seq)
            }
            None => (None, None, None),
        };
        let spec = presk_export_pytorch::ServeSpec {
            source: file.display().to_string(),
            model_stem: presk_export_pytorch::stem(&prog, prog.module(*def).unit),
            class: prog.module(*def).name.clone(),
            ctor_dims: extra,
            main_stem: presk_export_pytorch::stem(&prog, prog.main),
            generate,
            temperature,
            seq_dim,
            weights: sv.weights.clone(),
            layout: presk_export_pytorch::layout_py(&prog.params_tree(&sv.model)),
            ignored: sv.ignored.clone(),
        };
        for f in presk_export_pytorch::serve_files(&spec) {
            let path = out.join(&f.name);
            if let Err(e) = std::fs::write(&path, &f.content) {
                eprintln!("presk export: cannot write {}: {e}", path.display());
                return ExitCode::from(2);
            }
            written.push(path.display().to_string());
        }
    }
    // `--train` (or `train:K`): the training run as train.py + pskd.py, with the data file copied.
    let train_k: Option<u64> = conformance_entry.and_then(|e| e.strip_prefix("train:")).map(|k| k.parse().unwrap_or(0));
    if train || train_k.is_some() {
        let Some(run) = typed.training.first() else {
            eprintln!("presk export: --train: {} has no `training run`", file.display());
            return ExitCode::from(2);
        };
        let Some(pipe) = &run.data else {
            eprintln!("presk export: --train: the training run has no `data`");
            return ExitCode::from(2);
        };
        let presk_types::ty::ModRef::Inst { def, .. } = &run.model else {
            eprintln!("presk export: --train: the model must be a plain module");
            return ExitCode::from(2);
        };
        let by_name: std::collections::BTreeMap<String, u64> = run.dims.iter().map(|(v, n)| (prog.dim_name(*v), *n)).collect();
        let extra: Vec<String> = presk_export_pytorch::extra_dims(&prog, *def).iter().map(|v| by_name.get(&prog.dim_name(*v)).map(u64::to_string).unwrap_or_else(|| prog.dim_name(*v))).collect();
        let ctor = format!("{}({})", prog.module(*def).name, extra.join(", "));
        let items = prog.units[prog.main].optimizers.get(&run.optimizer).cloned().unwrap_or_default();
        let o = presk_core::lower::optim::AdamW::from_items(&items);
        let spec = presk_export_pytorch::TrainSpec {
            source: file.display().to_string(),
            model_stem: presk_export_pytorch::stem(&prog, prog.module(*def).unit),
            ctor: ctor.clone(),
            objective_stem: presk_export_pytorch::stem(&prog, prog.main),
            objective: prog.function(run.objective).name.clone(),
            lr: o.lr.clone(),
            betas: (o.beta1.clone(), o.beta2.clone()),
            eps: o.eps.clone(),
            weight_decay: o.weight_decay.clone(),
            grad_clip: o.grad_clip.clone(),
            schedule: match o.schedule {
                presk_core::lower::optim::Schedule::Constant => presk_export_pytorch::Schedule::Constant,
                presk_core::lower::optim::Schedule::Linear { warmup } => presk_export_pytorch::Schedule::Linear { warmup },
                presk_core::lower::optim::Schedule::Cosine { warmup } => presk_export_pytorch::Schedule::Cosine { warmup },
            },
            batch: pipe.batch,
            shuffle: pipe.shuffle,
            epochs: run.epochs.unwrap_or(1),
            checkpoint_every: run.checkpoint.as_ref().map(|(k, _)| *k),
            checkpoint_dir: run.checkpoint.as_ref().map(|(_, d)| d.clone()),
            dims: by_name.iter().map(|(n, v)| (n.clone(), *v)).collect(),
            comments: vec![format!("The `parallel` and `precision` clauses are reproduced as comments in {}.py.", presk_export_pytorch::stem(&prog, prog.main))],
        };
        for f in presk_export_pytorch::train_files(&spec) {
            let path = out.join(&f.name);
            if let Err(e) = std::fs::write(&path, &f.content) {
                eprintln!("presk export: cannot write {}: {e}", path.display());
                return ExitCode::from(2);
            }
            written.push(path.display().to_string());
        }
        let data_copy = out.join("data.psk.d");
        if let Err(e) = std::fs::copy(&pipe.path, &data_copy) {
            eprintln!("presk export: cannot copy the data file {}: {e}", pipe.path.display());
            return ExitCode::from(2);
        }
        written.push(data_copy.display().to_string());
        // `train:K`: replay K steps of the interpreter.
        if let Some(k) = train_k {
            if k == 0 {
                eprintln!("presk export: --conformance-entry train:K needs K ≥ 1");
                return ExitCode::from(2);
            }
            let (mut core, diags) = presk_core::lower(&prog, &typed);
            eprint!("{}", render_all(&diags, &sm, color()));
            let diags = presk_core::ad::transform(&mut core);
            eprint!("{}", render_all(&diags, &sm, color()));
            if has_errors(&diags) {
                return ExitCode::from(1);
            }
            let topts = presk_train::TrainOptions { data: None, steps: Some(k), epochs: None, seed, bits: 256, realize: presk_rt::Realize::F64, ckpt: None, checkpoints: false };
            let fns = match presk_train::prepare_training(&mut core, &prog, &typed) {
                Ok(f) => f,
                Err(e) => {
                    eprint!("{}", render_all(&[presk_syntax::Diagnostic::error(e.code, e.message)], &sm, color()));
                    return ExitCode::from(1);
                }
            };
            if let Err(c) = opt.apply(&mut core, &prog, false, &[]) {
                return c;
            }
            let tout = match presk_train::train(&mut presk_rt::InterpBackend::new(&core, &prog, 256, seed), &core, &prog, &typed, &fns, &topts, |_| {}) {
                Ok(t) => t,
                Err(e) => {
                    eprint!("{}", render_all(&[presk_syntax::Diagnostic::error(e.code, e.message)], &sm, color()));
                    return ExitCode::from(1);
                }
            };
            // The same K steps on the native backend (same seed: same θ₀ and batches),
            // for the experiments chapter's native points.
            let mut native_files: Vec<(String, presk_rt::Value)> = Vec::new();
            for (realize, name) in [(presk_rt::Realize::F64, "native_f64"), (presk_rt::Realize::F32, "native_f32")] {
                let o = presk_train::TrainOptions { realize, ..topts.clone() };
                match presk_cranelift::native_backend(&core, &prog, realize, seed) {
                    Ok(mut b) => match presk_train::train(&mut b, &core, &prog, &typed, &fns, &o, |_| {}) {
                        Ok(t) => native_files.push((name.to_string(), t.theta)),
                        Err(e) => eprintln!("presk export: native {name}: {}", e.message),
                    },
                    Err(e) => eprintln!("presk export: native backend unavailable: {}", e.message),
                }
            }
            let cdir = out.join("conformance");
            if let Err(e) = std::fs::create_dir_all(&cdir) {
                eprintln!("presk export: cannot create {}: {e}", cdir.display());
                return ExitCode::from(2);
            }
            let inputs = serde_json::json!({ "θ": presk_interp::run::to_json(&tout.theta0.to_interp(256), 17) });
            std::fs::write(cdir.join("inputs.json"), serde_json::to_string_pretty(&inputs).unwrap()).ok();
            std::fs::write(cdir.join("expected.json"), serde_json::to_string_pretty(&presk_interp::run::to_json(&tout.theta.to_interp(256), 17)).unwrap()).ok();
            std::fs::write(cdir.join("batches.json"), serde_json::to_string(&tout.batches).unwrap()).ok();
            for (name, theta) in &native_files {
                std::fs::write(cdir.join(format!("{name}.json")), serde_json::to_string_pretty(&presk_interp::run::to_json(&theta.to_interp(53), 17)).unwrap()).ok();
            }
            let data = presk_export_pytorch::ConformanceData {
                entry: format!("train:{k}"),
                module_file: presk_export_pytorch::stem(&prog, prog.main),
                class: None,
                function: None,
                theta_modules: vec![],
                optimizer: None,
                train: Some(presk_export_pytorch::TrainConformance { k, ctor, objective: spec.objective.clone(), rtol: 1e-6, atol: 1e-9 }),
                graph: false,
                inputs: vec!["θ".into()],
                seed,
                bits: 256,
                rtol: 1e-9,
                atol: 1e-12,
            };
            let test_path = out.join("test_conformance.py");
            std::fs::write(&test_path, presk_export_pytorch::test_file(&data)).ok();
            written.push(test_path.display().to_string());
            written.extend(["inputs.json", "expected.json", "batches.json"].iter().map(|f| cdir.join(f).display().to_string()));
            written.extend(native_files.iter().map(|(n, _)| cdir.join(format!("{n}.json")).display().to_string()));
            eprintln!("presk export: wrote {}", written.join(", "));
            return ExitCode::SUCCESS;
        }
    }
    // Conformance data: run the interpreter on the entry and write the test.
    if let Some(entry) = conformance_entry {
        let dims = match parse_dims(dims) {
            Ok(d) => d,
            Err(c) => return c,
        };
        let (mut core, diags) = presk_core::lower(&prog, &typed);
        eprint!("{}", render_all(&diags, &sm, color()));
        let diags = presk_core::ad::transform(&mut core);
        eprint!("{}", render_all(&diags, &sm, color()));
        if has_errors(&diags) || presk_core::verify(&core).is_err() {
            return ExitCode::from(1);
        }
        if let Err(c) = opt.apply(&mut core, &prog, false, &dims) {
            return c;
        }
        let ropts = presk_interp::RunOptions { entry: entry.to_string(), dims, seed, bits: 256, digits: 17 };
        let run = match presk_interp::run(&core, &prog, &ropts) {
            Ok(r) => r,
            Err(e) => {
                eprint!("{}", render_all(&[presk_syntax::Diagnostic::error("E7001", e.message)], &sm, color()));
                return ExitCode::from(1);
            }
        };
        let cdir = out.join("conformance");
        if let Err(e) = std::fs::create_dir_all(&cdir) {
            eprintln!("presk export: cannot create {}: {e}", cdir.display());
            return ExitCode::from(2);
        }
        let inputs: serde_json::Map<String, serde_json::Value> = run.inputs.iter().map(|(n, v)| (n.clone(), presk_interp::run::to_json(v, 17))).collect();
        std::fs::write(cdir.join("inputs.json"), serde_json::to_string_pretty(&serde_json::Value::Object(inputs)).unwrap()).ok();
        std::fs::write(cdir.join("expected.json"), serde_json::to_string_pretty(&presk_interp::run::to_json(&run.result, 17)).unwrap()).ok();
        // What the test needs: the entry's kind and how to build its modules.
        let func = core.func(run.func);
        let mut theta_mods: Vec<(String, String)> = Vec::new();
        let mut optimizer: Option<presk_export_pytorch::OptimizerConformance> = None;
        let names = prog.names();
        let dim_val = |e: &presk_types::dim::DimExpr| -> String {
            // Any variable given on the command line, by name: the dims of the
            // entry and the constructor dims of the modules it mentions.
            let map: std::collections::BTreeMap<_, _> = e
                .vars()
                .filter_map(|d| run.dims.iter().find(|(n, _)| *n == prog.dim_name(d)).map(|(_, x)| (d, presk_types::dim::rat(*x as i64))))
                .collect();
            e.eval(&map).map(|r| r.to_integer().to_string()).unwrap_or_else(|| e.display(&*names.dim))
        };
        let ctor = |def: presk_types::ty::ModuleId, args: &Vec<presk_types::dim::DimExpr>| -> String {
            let mut a: Vec<String> = args.iter().map(&dim_val).collect();
            for v in presk_export_pytorch::extra_dims(&prog, def) {
                a.push(dim_val(&presk_types::dim::DimExpr::var(v)));
            }
            format!("{}({})", prog.module(def).name, a.join(", "))
        };
        let (class, function) = match &func.origin {
            presk_core::ir::Origin::Source(fid) => {
                let sig = prog.function(*fid);
                match sig.owner {
                    Some(mid) => {
                        // Module entry: constructor arguments are the module's dimension values.
                        let m = prog.module(mid);
                        let mut vars: Vec<presk_types::dim::DimVar> = m.typarams.clone();
                        vars.extend(presk_export_pytorch::extra_dims(&prog, mid));
                        let args: Vec<u64> = vars.iter().map(|v| run.dims.iter().find(|(n, _)| *n == prog.dim_name(*v)).map(|(_, x)| *x).unwrap_or(0)).collect();
                        (Some((m.name.clone(), args)), None)
                    }
                    None => {
                        // Function entry: module-typed parameters are constructed from their instance.
                        let mods: Vec<(String, String)> = sig
                            .params
                            .iter()
                            .filter_map(|p| match &p.ty {
                                Some(presk_types::ty::Ty::Module(presk_types::ty::ModRef::Inst { def, args })) => Some((p.name.clone(), ctor(*def, args))),
                                _ => None,
                            })
                            .collect();
                        theta_mods = sig
                            .params
                            .iter()
                            .filter_map(|p| match &p.ty {
                                Some(presk_types::ty::Ty::Params(presk_types::ty::ModRef::Inst { def, args })) => Some((p.name.clone(), ctor(*def, args))),
                                _ => None,
                            })
                            .collect();
                        (None, Some((sig.name.clone(), mods)))
                    }
                }
            }
            presk_core::ir::Origin::Optimizer { name, model } => {
                // Optimizer entry: the model's constructor and the block's hyperparameters.
                if let presk_types::ty::ModRef::Inst { def, args } = model
                    && let Some(items) = prog.units[prog.main].optimizers.get(name)
                {
                    let o = presk_core::lower::optim::AdamW::from_items(items);
                    let schedule = match o.schedule {
                        presk_core::lower::optim::Schedule::Constant => ("constant".to_string(), 0),
                        presk_core::lower::optim::Schedule::Linear { warmup } => ("linear".to_string(), warmup),
                        presk_core::lower::optim::Schedule::Cosine { warmup } => ("cosine".to_string(), warmup),
                    };
                    optimizer = Some(presk_export_pytorch::OptimizerConformance { ctor: ctor(*def, args), lr: o.lr, betas: (o.beta1, o.beta2), eps: o.eps, weight_decay: o.weight_decay, grad_clip: o.grad_clip, schedule });
                }
                (None, None)
            }
            _ => (None, None),
        };
        let mut inputs_order: Vec<String> = run.inputs.iter().map(|(n, _)| n.clone()).collect();
        if let Some((_, mods)) = &function {
            // Module-typed parameters are not inputs of the interpreter; put them back in call order.
            let sig_names: Vec<String> = match &func.origin {
                presk_core::ir::Origin::Source(fid) => prog.function(*fid).params.iter().map(|p| p.name.clone()).collect(),
                _ => vec![],
            };
            inputs_order = sig_names.into_iter().filter(|n| mods.iter().any(|(m, _)| m == n) || run.inputs.iter().any(|(i, _)| i == n)).collect();
        }
        let data = presk_export_pytorch::ConformanceData {
            entry: entry.to_string(),
            module_file: presk_export_pytorch::stem(&prog, prog.main),
            class,
            function,
            theta_modules: theta_mods,
            optimizer,
            train: None,
            graph,
            inputs: inputs_order,
            seed,
            bits: 256,
            rtol: 1e-9,
            atol: 1e-12,
        };
        let test_path = out.join("test_conformance.py");
        std::fs::write(&test_path, presk_export_pytorch::test_file(&data)).ok();
        written.push(test_path.display().to_string());
        written.push(cdir.join("inputs.json").display().to_string());
        written.push(cdir.join("expected.json").display().to_string());
        if graph {
            match presk_export_pytorch::graph_driver(&data) {
                Some(f) => {
                    let path = out.join(&f.name);
                    std::fs::write(&path, &f.content).ok();
                    written.push(path.display().to_string());
                }
                None => {
                    eprintln!("presk export: --mode graph needs a module entry (`Module.apply`), not `{entry}`");
                    return ExitCode::from(2);
                }
            }
        }
    }
    eprintln!("presk export: wrote {}", written.join(", "));
    ExitCode::SUCCESS
}
