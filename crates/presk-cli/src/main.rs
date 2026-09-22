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

mod step;

#[derive(Parser)]
#[command(
    name = "presk",
    version,
    about = concat!(
        "Presk compiler (",
        env!("CARGO_PKG_VERSION"),
        ": front-end, reference interpreter, native backend, e-graph optimizer, training, serving, precision budgets, PyTorch export)"
    )
)]
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
        /// Propagate the rounding error of the entry this envelope was
        /// measured on, and print the bound. Conditional on the envelope,
        /// which is named in the output (`precision` chapter).
        #[arg(long, value_name = "ENVELOPE.json")]
        precision: Option<PathBuf>,
        /// The realization the bound is about.
        #[arg(long, default_value = "f64")]
        realize: String,
        /// Print the bound at every site, widest first.
        #[arg(long, default_value_t = 0)]
        sites: usize,
        /// The rule for sums and contractions: `cond` (the condition
        /// number, the default), `normwise`, `k` (0.5.0's) or `rss` (an
        /// estimate, not a bound); `all` tabulates the four.
        #[arg(long, default_value = "cond")]
        model: String,
        /// Also run the realization natively on the envelope's own inputs and
        /// weights, and print its measured distance to the exact reference
        /// beside the bound.
        #[arg(long)]
        measure: bool,
    },
    /// Evaluate an entry point on the reference interpreter or natively.
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
        /// The executor: the reference interpreter by default.
        #[arg(long, value_enum, default_value_t = BackendKind::Interp)]
        backend: BackendKind,
        /// The realization: `f64`, `f32`, `bf16`, or `@FILE` for a solved one.
        #[arg(long, default_value = "f64")]
        realize: String,
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
        /// With --data: start at sample I.
        #[arg(long, default_value_t = 0)]
        from: usize,
        /// Trained parameters for the entry's first argument (a `params M`
        /// file, of the dimensions the entry is run at).
        #[arg(long, value_name = "ckpt/final.psk.d")]
        weights: Option<PathBuf>,
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
        /// For `--stage sensitivity`: how many sites to list.
        #[arg(long, default_value_t = 20)]
        sites: usize,
        /// For `--stage sensitivity`: start from these trained parameters.
        /// A zero-initialized head makes every upstream gradient exactly
        /// zero, so a ranking taken at initialization is vacuous.
        #[arg(long, value_name = "ckpt/final")]
        weights: Option<PathBuf>,
        /// For `--stage kernel`: the entry to run (as `presk run`), with --dim/--const/--realize.
        #[arg(long)]
        entry: Option<String>,
        #[arg(long = "dim", value_name = "NAME=N")]
        dims: Vec<String>,
        #[arg(long = "const", value_name = "NAME=N")]
        consts: Vec<String>,
        #[arg(long, default_value = "f64")]
        realize: String,
        /// For `--stage core-opt`: extract by the roofline of this machine profile.
        #[arg(long, value_name = "FILE.json")]
        profile: Option<PathBuf>,
    },
    /// Run the `training run` of a file, on the reference interpreter or natively.
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
        /// The executor: the reference interpreter by default.
        #[arg(long, value_enum, default_value_t = BackendKind::Interp)]
        backend: BackendKind,
        /// The realization: `f64`, `f32`, `bf16`, or `@FILE` for a solved one.
        #[arg(long, default_value = "f64")]
        realize: String,
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
        #[arg(long, default_value = "f64")]
        realize: String,
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
    /// Time an entry on every backend (the reference at --bits, the kernel interpreter, native f64, f32 and bf16).
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
        #[arg(long, default_value = "f64")]
        realize: String,
        /// Time the native backend only. The reference interpreter's row is
        /// exact arbitrary precision, so at a thousand cubed it is not
        /// affordable and its presence is what stops a large benchmark from
        /// finishing at all.
        #[arg(long)]
        native_only: bool,
    },
    /// Measure the magnitude envelope of an entry, or one training step under each realization (--step).
    ///
    /// Runs an entry on the *reference* interpreter and records the largest
    /// `|x|` at each site. Static intervals cannot supply that scale for a
    /// network, so it is measured, and every bound quoting the envelope
    /// quotes its sample count with it (decision D35).
    Calibrate {
        file: PathBuf,
        /// Required, and the only mode: this subcommand exists for the
        /// precision analysis. The roofline is `presk bench --calibrate`.
        #[arg(long)]
        precision: bool,
        #[arg(long)]
        entry: Option<String>,
        #[arg(long = "dim", value_name = "NAME=N")]
        dims: Vec<String>,
        #[arg(long = "const", value_name = "NAME=N")]
        consts: Vec<String>,
        /// Fill the entry's last argument from this `.psk.d` file.
        #[arg(long)]
        data: Option<PathBuf>,
        /// Samples to fold in (with --data, the size of the batch).
        #[arg(long)]
        samples: Option<usize>,
        /// Skip this many samples first: two disjoint ranges give two
        /// independent measurements of the same program.
        #[arg(long, default_value_t = 0)]
        from: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Precision of the reference interpreter.
        #[arg(long, default_value_t = 256)]
        bits: u32,
        #[arg(long, value_name = "FILE.json")]
        out: Option<PathBuf>,
        /// Report how far this envelope and another disagree. A ratio near 1
        /// means the magnitudes are a property of the program rather than of
        /// the sample drawn, which is what a bound quoting one needs.
        #[arg(long, value_name = "FILE.json")]
        compare: Option<PathBuf>,
        /// Trained parameters for the entry's first argument. A trained
        /// network is another function than its initialization, and a
        /// zero-initialized head makes a loss `ln V` whatever else happens.
        #[arg(long, value_name = "ckpt/final.psk.d")]
        weights: Option<PathBuf>,
        /// Measure one step of the file's `training run` instead of an
        /// entry (0.6.0 step 7b): the state is `--weights` plus `--warmup`
        /// steps in `f64`, the batch is `--samples` samples of `--data` (the
        /// run's own by default) from `--from`, and the envelope is of the
        /// reverse sweep's and the optimizer's sites.
        #[arg(long)]
        step: bool,
        /// With --step: warm-up steps in `f64` before the measured one, so
        /// that AdamW's second moment is not the zero of initialisation.
        #[arg(long, default_value_t = 10)]
        warmup: u64,
        /// With --step: the schedule's horizon (the run's by default).
        #[arg(long)]
        steps: Option<u64>,
        /// With --step: measure these realizations' update and gradient gaps
        /// natively against the exact step (repeatable). `recipe` is mixed
        /// precision as practitioners use it: `bf16` everywhere but the
        /// optimizer, which keeps the master weights in `f32`.
        #[arg(long = "realize", value_name = "f64|f32|bf16|recipe|@FILE")]
        realize: Vec<String>,
        /// With --step: the machine profile, to print each realization's
        /// predicted time.
        #[arg(long, value_name = "FILE.json")]
        profile: Option<PathBuf>,
        /// With --step: `exact` (the reference interpreter) or `f64` (a
        /// native `f64` step, for a model too large for the exact one).
        #[arg(long, default_value = "exact")]
        reference: String,
    },
    /// Solve for a realization under a declared error budget.
    Solve {
        file: PathBuf,
        /// The magnitude envelope the bound and the estimate are read from.
        #[arg(long, value_name = "ENVELOPE.json")]
        precision: PathBuf,
        /// Relative error budget for the entry's output.
        #[arg(long, default_value_t = 1e-3)]
        budget: f64,
        /// Trained parameters: a sensitivity ranking at initialization is
        /// vacuous, so without these every site scores zero and the solver
        /// narrows by traffic alone.
        #[arg(long, value_name = "ckpt/final.psk.d")]
        weights: Option<PathBuf>,
        /// Do not verify by measurement (the proposal only).
        #[arg(long)]
        no_verify: bool,
        #[arg(long, default_value_t = 128)]
        bits: u32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Write the solved realization here; `--realize @FILE` then runs it.
        #[arg(long, value_name = "FILE.json")]
        out: Option<PathBuf>,
        /// List this many assigned sites.
        #[arg(long, default_value_t = 8)]
        sites: usize,
        /// The machine profile, so the objective is **time** and not traffic
        /// (`presk bench --calibrate`). Without it the solver minimises bytes
        /// moved, which is what 0.5.0 did and why its assignments were slower.
        #[arg(long, value_name = "FILE.json")]
        profile: Option<PathBuf>,
        /// Also offer every matrix parameter as `int8` or `int4` (0.6.0 step
        /// 5d), under the budget the float assignment leaves.
        #[arg(long)]
        quant: bool,
        /// With --quant: `absmax` scales, or `pow2`.
        #[arg(long, default_value = "absmax")]
        scheme: String,
        /// Solve one training step under a budget on its **update**, the
        /// envelope being `presk calibrate --step`'s (0.6.0 step 7b): every
        /// function of the step measured alone in `f32` and `bf16`, greedy
        /// by predicted time per unit of gap, then verified.
        #[arg(long)]
        step: bool,
    },
    /// Quantise parameters as a realization's `quant` map says, and measure
    /// the gap in two parts: the weights' rounding and the arithmetic.
    Quantize {
        file: PathBuf,
        /// The realization, usually a file with a `quant` map (`@FILE`); its
        /// float part runs the arithmetic. Without a `quant` map the
        /// representation part is zero and the gap is the float one.
        #[arg(long, default_value = "f32")]
        realize: String,
        /// List the parameters, their paths and shapes, and stop.
        #[arg(long)]
        list: bool,
        /// The entry to measure on; its first argument is the parameters.
        #[arg(long)]
        entry: String,
        #[arg(long = "dim", value_name = "NAME=N")]
        dims: Vec<String>,
        #[arg(long = "const", value_name = "NAME=N")]
        consts: Vec<String>,
        /// Trained parameters. A zero-initialized head makes a loss `ln V`
        /// whatever the other weights are, so a gap measured at
        /// initialization says little.
        #[arg(long, value_name = "ckpt/final.psk.d")]
        weights: Option<PathBuf>,
        /// The executor of the arithmetic part.
        #[arg(long, value_enum, default_value_t = BackendKind::Native)]
        backend: BackendKind,
        /// Take the entry's last argument from this data file, as `presk run
        /// --data` does, instead of synthesizing it: a trained model's loss
        /// on random tokens is not the loss it was trained for.
        #[arg(long, value_name = "FILE.psk.d")]
        data: Option<PathBuf>,
        /// With --data: how many samples, from which one.
        #[arg(long)]
        samples: Option<usize>,
        #[arg(long, default_value_t = 0)]
        from: usize,
        #[arg(long, default_value_t = 128)]
        bits: u32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
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
    /// The distance between two files of the same type, over their real
    /// values: `‖a − b‖₂`, and relative to `‖a‖₂`. Two checkpoints' distance
    /// is how far two training runs ended apart (0.6.0 step 7b).
    Diff {
        a: PathBuf,
        b: PathBuf,
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
fn with_backend<R>(kind: BackendKind, core: &presk_core::Program, prog: &elab::ElabProgram, bits: u32, seed: u64, realize: &presk_rt::Realization, f: impl FnOnce(&mut dyn presk_rt::Backend) -> R) -> R {
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

/// `--realize f64|f32|bf16|@FILE`. The three names are uniform realizations,
/// what the flag has always meant; `@FILE` reads the object a solver writes
/// (D34). Parsing is deferred to `presk_rt::Realization::parse` so that one
/// spelling serves the CLI, the solver and the conformance report.
fn parse_realize(spec: &str) -> presk_rt::Realization {
    match presk_rt::Realization::parse(spec) {
        Ok(r) => r,
        Err(m) => {
            eprintln!("presk: {m}");
            std::process::exit(2);
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
    /// `‖∂L/∂v‖₁` at every site of an entry, read off one reverse sweep
    /// (needs --entry, --dim/--const as `presk run`; `--data` for real
    /// batches). Ranks where a precision budget should *not* be spent.
    Sensitivity,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Check { file, consts, summary, json, precision, realize, sites, model, measure } => match precision {
            Some(env) => check_precision(&file, &env, &parse_realize(&realize), sites, &model, measure),
            None => check(&file, &consts, summary, json),
        },
        Command::Dump { file, stage, unroll_stack, entry, dims, consts, realize, profile, weights, sites } => dump(&file, stage, unroll_stack, entry.as_deref(), &dims, &consts, &parse_realize(&realize), profile, weights, sites),
        Command::Run { file, entry, dims, consts, seed, bits, digits, json, verbose, unroll_stack, backend, realize, opt, profile, data, samples, from, weights, threads } => { set_threads(threads); run(&file, entry.as_deref(), &dims, &consts, seed, bits, digits as usize, json, verbose, unroll_stack, backend, &parse_realize(&realize), Opt { on: opt, profile }, data, samples, from, weights) }
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
            let opts = presk_serve::ServeOptions { weights, prompt: prompt_tokens, tokens, seed, bits, realize: parse_realize(&realize), temperature, dims: named };
            serve(&file, &consts, entry.as_deref(), opts, backend, json, Opt { on: opt, profile })
        }
        Command::Train { file, consts, data, steps, epochs, seed, bits, log_every, ckpt, json, backend, realize, opt, profile, threads } => { set_threads(threads); train(&file, &consts, presk_train::TrainOptions { data, steps, epochs, seed, bits, realize: parse_realize(&realize), ckpt, checkpoints: true }, log_every, json, backend, Opt { on: opt, profile }) }
        Command::Calibrate { file, precision: _, step: true, consts, data, samples, from, seed, bits, out, weights, warmup, steps, realize, profile, reference, .. } => {
            let native_reference = match reference.as_str() {
                "exact" => false,
                "f64" => true,
                other => {
                    eprintln!("presk calibrate: unknown --reference `{other}`; expected exact or f64");
                    return ExitCode::from(2);
                }
            };
            let Some(weights) = weights else {
                eprintln!("presk calibrate: --step needs --weights: a step from initialization measures an untrained network");
                return ExitCode::from(2);
            };
            let a = step::StepArgs { file: &file, consts: &consts, weights: &weights, warmup, data: data.as_deref(), from, samples: samples.unwrap_or(2), steps, seed, bits };
            step::calibrate(&a, &realize, out.as_deref(), profile.as_deref(), native_reference)
        }
        Command::Calibrate { file, precision, entry, dims, consts, data, samples, from, seed, bits, out, compare, weights, steps, realize, profile, reference, .. } => {
            if steps.is_some() || !realize.is_empty() || profile.is_some() || reference != "exact" {
                eprintln!("presk calibrate: --steps, --realize, --profile and --reference are for a training step: add --step");
                return ExitCode::from(2);
            }
            calibrate_precision(&file, precision, entry.as_deref(), &dims, &consts, data.as_deref(), samples, from, seed, bits, out.as_deref(), compare.as_deref(), weights.as_deref())
        }
        Command::Quantize { file, realize, list, entry, dims, consts, weights, backend, data, samples, from, bits, seed } => quantize(&file, &parse_realize(&realize), list, &entry, &dims, &consts, weights.as_deref(), backend, data.as_deref().map(|d| (d, from, samples)), bits, seed),
        Command::Solve { file, precision, budget, no_verify, bits, out, profile, step: true, .. } => step::solve(&file, &precision, budget, bits, !no_verify, out.as_deref(), profile.as_deref()),
        Command::Solve { file, precision, budget, weights, no_verify, bits, seed, out, sites, profile, quant, scheme, .. } => {
            let scheme = match scheme.as_str() {
                "absmax" => presk_rt::QuantScheme::Absmax,
                "pow2" => presk_rt::QuantScheme::Pow2,
                other => {
                    eprintln!("presk solve: unknown --scheme `{other}`; expected absmax or pow2");
                    return ExitCode::from(2);
                }
            };
            solve_precision(&file, &precision, budget, weights.as_deref(), !no_verify, bits, seed, out.as_deref(), sites, profile, quant.then_some(scheme))
        }
        Command::Bench { file, entry, dims, consts, bits, repeat, seed, opt, profile, calibrate, realize, threads, native_only } => { set_threads(threads); bench(&file, &entry, &dims, &consts, bits, repeat, seed, Opt { on: opt, profile }, calibrate.as_deref(), &parse_realize(&realize), native_only) }
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
    match with_backend(backend, &core, &prog, opts.bits, opts.seed, &opts.realize, |b| presk_serve::serve(b, &core, &prog, &fns, &opts, on_token)) {
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

/// `presk solve`: a budget in, a realization out (plan 0.5.0 §3.4).
#[allow(clippy::too_many_arguments)]
fn solve_precision(file: &Path, env_path: &Path, budget: f64, weights: Option<&Path>, verify: bool, bits: u32, seed: u64, out: Option<&Path>, list: usize, profile: Option<PathBuf>, quant: Option<presk_rt::QuantScheme>) -> ExitCode {
    use presk_precision::{sensitivity, solve};
    let env = match presk_precision::envelope::Envelope::load(env_path) {
        Ok(e) => e,
        Err(m) => {
            eprintln!("presk solve: {m}");
            return ExitCode::from(1);
        }
    };
    let consts = env.consts.clone();
    let (sm, prog, mut core) = match core(file, &consts, false) {
        Err(c) => return c,
        Ok(x) => x,
    };
    let diags = presk_core::ad::transform(&mut core);
    eprint!("{}", render_all(&diags, &sm, color()));
    if has_errors(&diags) {
        return ExitCode::from(1);
    }
    let Ok((entry, _, _)) = presk_interp::run::resolve_entry(&core, &prog, &env.entry) else {
        eprintln!("presk solve: `{}` is not an entry of {}", env.entry, file.display());
        return ExitCode::from(1);
    };
    let opts = presk_interp::RunOptions { entry: env.entry.clone(), dims: env.dims.clone(), seed, bits, digits: 17 };
    // The envelope's own inputs: synthesized, or the data slice it was
    // measured on (0.6.0 step 6). A checkpoint that does not load is an
    // error, never a fallback to the initial parameters: 0.5.0 read a
    // `T = 256` checkpoint at `T = 32` without a word (D48).
    let (_, fdims, args) = match entry_args(&core, &prog, &opts, env.data.as_ref(), weights) {
        Ok(x) => x,
        Err(m) => {
            eprintln!("presk solve: cannot bind the inputs of `{}`: {m}", env.entry);
            return ExitCode::from(1);
        }
    };

    // Where the budget should *not* be spent: one reverse sweep.
    let (vjp, adjoints, _) = presk_core::ad::vjp_with_adjoints(&mut core, entry);
    let mut sb = presk_rt::InterpBackend::new(&core, &prog, bits, seed);
    sb.interp.sums = Some(Default::default());
    let mut vargs = args.clone();
    vargs.push(presk_rt::Value::Real(presk_rt::Tensor { axes: vec![], data: presk_rt::Buf::F64(vec![1.0]) }));
    let one = presk_rt::Realization::uniform(presk_rt::Dtype::F64);
    // The same sweep's cotangent of the parameters, `∂L/∂θ`, prices the
    // parameter menu (0.6.0 step 5d).
    let (sens, grad_theta) = match presk_rt::Backend::prepare(&mut sb, vjp, &fdims, &one).and_then(|p| presk_rt::Backend::call(&mut sb, &p, vargs)) {
        Ok(g) => (sensitivity::from_run(&core, entry, &adjoints, &sb.interp.sums.take().unwrap_or_default()), params_cotangent(&g, &args[0])),
        Err(e) => {
            eprintln!("presk solve: the objective does not differentiate: {}", e.message);
            return ExitCode::from(1);
        }
    };

    // The exact value, to measure candidates against.
    let mut exact = presk_rt::InterpBackend::new(&core, &prog, bits, seed);
    let reference = match presk_rt::Backend::prepare(&mut exact, entry, &fdims, &one).and_then(|p| presk_rt::Backend::call(&mut exact, &p, args.clone())) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("presk solve: {}", e.message);
            return ExitCode::from(1);
        }
    };
    let ref_f64 = leaf_max(&reference);
    let output_site = format!("{}/%{}", core.func(entry).name, core.func(entry).body.result.0);

    let mut measure = |r: &presk_rt::Realization| -> f64 {
        let mut b = presk_kernel::KernelBackend::new(&core, &prog, r, seed);
        let mut a: Vec<presk_rt::Value> = args.iter().map(|v| v.realize(r.default)).collect();
        if r.quantize_args(&mut a).is_err() {
            return f64::INFINITY;
        }
        match presk_rt::Backend::prepare(&mut b, entry, &fdims, r).and_then(|p| presk_rt::Backend::call(&mut b, &p, a)) {
            Ok(v) => leaf_gap(&reference, &v),
            Err(_) => f64::INFINITY,
        }
    };
    let menu = solve::Menu::default();
    // Price *time* when a machine profile is available, and traffic
    // otherwise. 0.5.0 minimised traffic and lost: on a compute-bound site
    // narrowing buys nothing and the conversions cost.
    struct Roofline(presk_opt::Profile);
    impl solve::Machine for Roofline {
        fn seconds(&self, d: presk_rt::Dtype, work: f64, bytes: f64) -> f64 {
            self.0.predict_for(d.name(), work, bytes)
        }
    }
    let machine: Box<dyn solve::Machine> = match profile.as_deref().map(presk_opt::Profile::load) {
        Some(Ok(p)) => Box::new(Roofline(p)),
        _ => Box::new(solve::Traffic),
    };
    let mut sol = if verify {
        solve::solve(&env, &sens, &output_site, budget, &menu, presk_rt::Dtype::F64, machine.as_ref(), Some(&mut measure))
    } else {
        solve::solve(&env, &sens, &output_site, budget, &menu, presk_rt::Dtype::F64, machine.as_ref(), None)
    };
    // The parameter menu, on top of the float assignment and under what it
    // left of the budget (0.6.0 step 5d). Verified, the measurement is of
    // the whole realization, so the budget is the whole budget.
    let qsol = match (quant, &grad_theta) {
        (Some(scheme), Some(g)) => {
            // Ranks as declared, so a stacked vector is not taken for a matrix.
            let declared: std::collections::BTreeMap<String, usize> = match &core.func(entry).params.first().map(|p| &p.ty) {
                Some(presk_types::ty::Ty::Params(m)) => prog.params_tree(m).declared_ranks().into_iter().collect(),
                _ => Default::default(),
            };
            let cands = presk_precision::quant::candidates(&args[0], g, &declared, sol.realization.default.bytes(), scheme);
            let q = if verify {
                presk_precision::quant::solve(cands, sol.budget_abs, &sol.realization, scheme, Some(&mut measure))
            } else {
                let left = (sol.budget_abs - sol.estimated_error).max(0.0);
                presk_precision::quant::solve(cands, left, &sol.realization, scheme, None)
            };
            sol.realization.quant = q.quant.clone();
            Some(q)
        }
        (Some(_), None) => {
            eprintln!("presk solve: --quant needs the parameters' gradient, and the reverse sweep gave none");
            return ExitCode::from(1);
        }
        _ => None,
    };

    println!("presk solve: `{}` under budget {:.2e} relative ({:.3e} absolute, |output| {:.3e})", env.entry, budget, sol.budget_abs, ref_f64);
    println!("  sites       {} narrowable, {} assigned away from f64", sol.considered, sol.realization.storage.len());
    println!("  cost        {:.4e} against {:.4e} all-f64 ({:.2}x), relaxation floor {:.4e}", sol.bytes, sol.baseline_bytes, sol.speedup(), sol.relaxation_bytes);
    println!("  proposal    {:.4e} before measurement widened anything (time when --profile is given, bytes otherwise)", sol.proposed_bytes);
    println!("  estimate    {:.3e} (sensitivity-weighted, first order: it ranks, it does not guarantee)", sol.estimated_error);
    match sol.measured {
        Some(g) => println!("  measured    {:.3e} against the exact reference, after {} widening round(s) — {}", g, sol.rounds, if g <= sol.budget_abs { "inside the budget" } else { "OUTSIDE the budget" }),
        None => println!("  measured    not checked (--no-verify)"),
    }
    if let Some(q) = &qsol {
        let (n8, n4) = (q.chosen.values().filter(|b| **b == Some(8)).count(), q.chosen.values().filter(|b| **b == Some(4)).count());
        println!("  parameters  {} of {} matrices quantised ({n8} int8, {n4} int4): {} bytes → {} ({:.2}x less)", n8 + n4, q.candidates.len(), q.bytes_before, q.bytes_after, q.bytes_before as f64 / q.bytes_after.max(1) as f64);
        println!("  estimate    {:.3e} (√Σ(∂L/∂w·(ŵ−w))², first order, roundings taken as independent)", q.estimated);
        match q.measured {
            Some(g) => println!("  measured    {:.3e} against the exact reference with the weights quantised, after {} widening round(s) — {}", g, q.rounds, if g <= sol.budget_abs { "inside the budget" } else { "OUTSIDE the budget" }),
            None => println!("  measured    not checked (--no-verify)"),
        }
        if list > 0 {
            println!("{:<28} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}", "parameter", "bits", "elements", "bound 8", "Σgδ 8", "rss 8", "bound 4", "Σgδ 4", "rss 4");
            for c in &q.candidates {
                let b = q.chosen[&c.path].map(|b| format!("int{b}")).unwrap_or_else(|| sol.realization.default.name().into());
                println!("{:<28} {:>6} {:>10} {:>10.2e} {:>10.2e} {:>10.2e} {:>10.2e} {:>10.2e} {:>10.2e}", c.path, b, c.elems, c.est8, c.signed[0], c.rss[0], c.est4, c.signed[1], c.rss[1]);
            }
            let sum = |f: &dyn Fn(&presk_precision::quant::Candidate) -> f64| q.candidates.iter().map(f).sum::<f64>();
            println!("{:<28} {:>6} {:>10} {:>10.2e} {:>10.2e} {:>10.2e} {:>10.2e} {:>10.2e} {:>10.2e}", "all, combined", "", "", sum(&|c| c.est8), sum(&|c| c.signed[0]).abs(), sum(&|c| c.rss[0] * c.rss[0]).sqrt(), sum(&|c| c.est4), sum(&|c| c.signed[1]).abs(), sum(&|c| c.rss[1] * c.rss[1]).sqrt());
        }
    }
    if let Some((_, bytes)) = solve::exhaustive(&env, &sens, &output_site, budget, &menu, presk_rt::Dtype::F64, machine.as_ref(), 200_000) {
        println!("  optimal     {:.4e} bytes by exhaustive search under the same estimate — the greedy proposal is {:.3}x that", bytes, sol.proposed_bytes / bytes.max(1.0));
    } else {
        println!("  optimal     not searched (the space is too large); the relaxation floor above is the lower bound");
    }
    if list > 0 && !sol.realization.storage.is_empty() {
        let mut rows: Vec<(&String, &presk_rt::Dtype)> = sol.realization.storage.iter().collect();
        rows.sort_by_key(|(k, _)| k.as_str());
        println!("{:<30} {:>8} {:>12} {:>12}", "site", "dtype", "elements", "‖∂L/∂v‖₁");
        for (site, d) in rows.iter().take(list) {
            println!("{site:<30} {:>8} {:>12} {:>12.3e}", d.name(), env.elems(site), sens.get(site).unwrap_or(0.0));
        }
    }
    if let Some(path) = out {
        if let Some(d) = path.parent().filter(|d| !d.as_os_str().is_empty())
            && let Err(e) = std::fs::create_dir_all(d)
        {
            eprintln!("presk solve: {e}");
            return ExitCode::from(1);
        }
        if let Err(e) = std::fs::write(path, sol.realization.to_json()) {
            eprintln!("presk solve: {e}");
            return ExitCode::from(1);
        }
        println!("presk solve: wrote {} — run it with `--realize @{}`", path.display(), path.display());
    }
    ExitCode::SUCCESS
}

/// `presk quantize`: the quantised parameters, and the gap of the
/// realization split in two (plan 0.6.0 §3.3, D44). The *representation*
/// part is the exact denotation on the dequantised weights against the
/// exact denotation on the true ones — what rounding the weights costs, with
/// no arithmetic error in it at all. The *arithmetic* part is the machine run
/// on the dequantised weights against the former. Only a compiler that keeps
/// an exact reference can separate the two.
#[allow(clippy::too_many_arguments)]
fn quantize(file: &Path, realize: &presk_rt::Realization, list: bool, entry: &str, dims: &[String], consts: &[String], weights: Option<&Path>, backend: BackendKind, data: Option<(&Path, usize, Option<usize>)>, bits: u32, seed: u64) -> ExitCode {
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
    let fail = |e: presk_rt::RtError| {
        eprint!("{}", render_all(&[presk_syntax::Diagnostic::error(e.code, format!("presk quantize: {}", e.message))], &sm, color()));
        ExitCode::from(1)
    };
    let opts = presk_interp::RunOptions { entry: entry.to_string(), dims, seed, bits, digits: 17 };
    let (func, fdims, mut args) = match data {
        Some((path, from, samples)) => match presk_train::args_on_data(&core, &prog, &opts, path, from, samples) {
            Ok((func, fdims, _, args)) => (func, fdims, args),
            Err(e) => return fail(presk_rt::RtError { code: e.code, message: e.message, span: e.span }),
        },
        None => {
            let (func, fdims, inputs) = match presk_interp::run::prepare_run(&core, &prog, &opts) {
                Ok(x) => x,
                Err(e) => return fail(e.into()),
            };
            match inputs.iter().map(|(_, v)| presk_rt::Value::from_interp(v)).collect() {
                Ok(a) => (func, fdims, a),
                Err(e) => return fail(e),
            }
        }
    };
    if !matches!(args.first(), Some(presk_rt::Value::Record(_))) {
        eprintln!("presk quantize: the first argument of `{entry}` is not a parameter tree");
        return ExitCode::from(2);
    }
    match weights {
        Some(path) => match presk_rt::load_params(path, &args[0]) {
            Ok(v) => args[0] = v,
            Err(e) => return fail(e),
        },
        None => eprintln!("presk quantize: no --weights, so the parameters are the initial ones"),
    }
    let theta = args[0].clone();
    // Every leaf read once in f64, the checkpoint's precision.
    let theta = theta.realize(presk_rt::Dtype::F64);

    if list {
        println!("presk quantize: the parameters of `{entry}`");
        let leaves = presk_rt::quant::param_paths(&theta);
        for p in leaves {
            let shape = param_leaf(&theta, &p).map(|t| t.axes.iter().map(|a| match &a.name {
                Some(n) => format!("{n}: {}", a.len),
                None => a.len.to_string(),
            }).collect::<Vec<_>>().join(", ")).unwrap_or_default();
            println!("  {p:<28} [{shape}]");
        }
        println!("name them in the realization's `quant` map, e.g. \"quant\": {{ \"<path>\": {{ \"bits\": 8, \"over\": -2 }} }}");
        return ExitCode::SUCCESS;
    }
    let qs = match presk_rt::quant::quantize_params(&theta, &realize.quant) {
        Ok(q) => q,
        Err(e) => return fail(e),
    };

    println!("presk quantize: {} parameter(s) of `{entry}`, {} at {} bits", qs.len(), if weights.is_some() { "trained" } else { "initial" }, bits);
    if !qs.is_empty() {
        println!("{:<28} {:>11} {:>6} {:>9} {:>11} {:>11} {:>11} {:>9}", "parameter", "scheme", "over", "channels", "max scale", "max |w−ŵ|", "max s/2", "bytes");
    }
    let (mut before, mut after) = (0usize, 0usize);
    for (path, q) in &qs {
        let w = param_leaf(&theta, path).map(|t| t.data.to_f64()).unwrap_or_default();
        let presk_rt::Buf::F32(d) = q.dequantize().data else { unreachable!() };
        let err = w.iter().zip(&d).map(|(a, b)| (a - *b as f64).abs()).fold(0.0, f64::max);
        let over = q.quant.over.to_string();
        let smax = q.scales.iter().copied().fold(0f32, f32::max);
        println!("{path:<28} {:>11} {over:>6} {:>9} {smax:>11.3e} {err:>11.3e} {:>11.3e} {:>9}", q.quant.name(), q.scales.len(), q.half_step(), q.bytes());
        before += 4 * q.numel();
        after += q.bytes();
    }
    if !qs.is_empty() {
        println!("  storage     {before} bytes in f32 → {after} quantised ({:.2}x less)", before as f64 / after.max(1) as f64);
    }
    // Every parameter as this realization stores it: integers where it
    // quantises, its default machine type elsewhere.
    let all: usize = presk_rt::quant::param_paths(&theta).iter().map(|p| match qs.get(p) {
        Some(q) => q.bytes(),
        None => param_leaf(&theta, p).map_or(0, |t| t.numel() * realize.default.bytes()),
    }).sum();
    println!("  weights     {all} bytes, every parameter as the realization stores it");

    let deq = presk_rt::quant::dequantize_params(&theta, &qs);
    let one = presk_rt::Realization::uniform(presk_rt::Dtype::F64);
    let exact_on = |theta: &presk_rt::Value| -> presk_rt::Result<presk_rt::Value> {
        let mut b = presk_rt::InterpBackend::new(&core, &prog, bits, seed);
        let mut a = args.clone();
        a[0] = theta.clone();
        let p = presk_rt::Backend::prepare(&mut b, func, &fdims, &one)?;
        presk_rt::Backend::call(&mut b, &p, a)
    };
    let exact = match exact_on(&theta) {
        Ok(v) => v,
        Err(e) => return fail(e),
    };
    // Nothing quantised: nothing to represent, and one exact run fewer.
    let exact_q = if qs.is_empty() {
        exact.clone()
    } else {
        match exact_on(&deq) {
            Ok(v) => v,
            Err(e) => return fail(e),
        }
    };
    // The realization itself: the integer weights, loaded by the kernels as
    // `f32(q)·s` (step 5b), and the float part for everything else.
    let machine = with_backend(backend, &core, &prog, bits, seed, realize, |b| {
        let mut a: Vec<presk_rt::Value> = args.iter().map(|v| v.realize(realize.default)).collect();
        a[0] = presk_rt::quant::quantized_leaves(&a[0], &qs);
        let p = b.prepare(func, &fdims, realize)?;
        b.call(&p, a).map(|v| (v, b.name()))
    });
    let (machine, backend) = match machine {
        Ok(v) => v,
        Err(e) => return fail(e),
    };
    let scale = leaf_max(&exact).max(f64::MIN_POSITIVE);
    let (rep, ari, tot) = (leaf_gap(&exact, &exact_q), leaf_gap(&exact_q, &machine), leaf_gap(&exact, &machine));
    println!("  output      |exact| {scale:.6e}, realization {} on the {backend} backend", realize.label());
    println!("  representation  {rep:.3e} ({:.2e} relative)  exact on ŵ against exact on w: the weights' rounding alone", rep / scale);
    println!("  arithmetic      {ari:.3e} ({:.2e} relative)  the machine run on ŵ against exact on ŵ", ari / scale);
    println!("  total           {tot:.3e} ({:.2e} relative)  the machine run on ŵ against exact on w", tot / scale);
    ExitCode::SUCCESS
}

/// The real tensor at `path` in a parameter tree.
fn param_leaf<'v>(theta: &'v presk_rt::Value, path: &str) -> Option<&'v presk_rt::Tensor> {
    let mut v = theta;
    for part in path.split('.') {
        v = match v {
            presk_rt::Value::Record(_) => v.field(part)?,
            presk_rt::Value::Tuple(xs) => xs.get(part.parse::<usize>().ok()?)?,
            _ => return None,
        };
    }
    match v {
        presk_rt::Value::Real(t) => Some(t),
        _ => None,
    }
}

/// The cotangent of the parameters in a vjp's result: the result itself when
/// it is shaped like `theta`, else the first component of a tuple that is.
fn params_cotangent(out: &presk_rt::Value, theta: &presk_rt::Value) -> Option<presk_rt::Value> {
    let paths = presk_rt::quant::param_paths(theta);
    if presk_rt::quant::param_paths(out) == paths {
        return Some(out.clone());
    }
    match out {
        presk_rt::Value::Tuple(xs) => xs.iter().find(|x| presk_rt::quant::param_paths(x) == paths).cloned(),
        _ => None,
    }
}

/// The largest `|x|` over the leaves of a runtime value.
fn leaf_max(v: &presk_rt::Value) -> f64 {
    match v {
        presk_rt::Value::Real(t) => t.data.to_f64().into_iter().map(f64::abs).filter(|x| x.is_finite()).fold(0.0, f64::max),
        presk_rt::Value::Tuple(xs) => xs.iter().map(leaf_max).fold(0.0, f64::max),
        presk_rt::Value::Record(xs) => xs.iter().map(|(_, x)| leaf_max(x)).fold(0.0, f64::max),
        _ => 0.0,
    }
}

/// The largest absolute difference over the leaves of two runtime values.
fn leaf_gap(a: &presk_rt::Value, b: &presk_rt::Value) -> f64 {
    match (a, b) {
        (presk_rt::Value::Real(x), presk_rt::Value::Real(y)) => x
            .data
            .to_f64()
            .into_iter()
            .zip(y.data.to_f64())
            .map(|(p, q)| (p - q).abs())
            .filter(|d| d.is_finite())
            .fold(0.0, f64::max),
        (presk_rt::Value::Tuple(xs), presk_rt::Value::Tuple(ys)) => xs.iter().zip(ys).map(|(x, y)| leaf_gap(x, y)).fold(0.0, f64::max),
        (presk_rt::Value::Record(xs), presk_rt::Value::Record(ys)) => xs.iter().zip(ys).map(|((_, x), (_, y))| leaf_gap(x, y)).fold(0.0, f64::max),
        _ => 0.0,
    }
}

/// `presk dump --stage sensitivity`: one reverse sweep, and what it says
/// about where a budget should go (plan 0.5.0 §3.3).
/// The inputs are the synthesized ones of a fixed seed rather than a data
/// file: a sensitivity *ranking* is driven by the architecture — which values
/// the output depends on and through how many operations — and not by which
/// tokens happen to be in a batch. The magnitudes would move; the order does
/// not.
#[allow(clippy::too_many_arguments)]
fn dump_sensitivity(file: &Path, entry: Option<&str>, dims: &[String], consts: &[String], weights: Option<&Path>, top: usize, seed: u64, bits: u32) -> ExitCode {
    let dims_p = match parse_dims(dims) {
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
    let Some(entry) = entry else {
        eprintln!("presk dump --stage sensitivity: --entry is required");
        return ExitCode::from(2);
    };
    let Ok((f, _, _)) = presk_interp::run::resolve_entry(&core, &prog, entry) else {
        eprintln!("presk dump: `{entry}` is not an entry of {}", file.display());
        return ExitCode::from(1);
    };
    if !matches!(core.func(f).ret, presk_types::ty::Ty::Real(ref s) if s.axes.is_empty() && s.prefix.is_none()) {
        eprintln!("presk dump --stage sensitivity: `{entry}` must return a scalar `ℝ[]`, since a sensitivity is `∂L/∂v`");
        return ExitCode::from(2);
    }
    // The reverse sweep already computes the adjoint of every intermediate;
    // this only asks where each one ended up.
    let (vjp, adjoints, diags) = presk_core::ad::vjp_with_adjoints(&mut core, f);
    eprint!("{}", render_all(&diags, &sm, color()));
    if has_errors(&diags) {
        return ExitCode::from(1);
    }
    let opts = presk_interp::RunOptions { entry: entry.into(), dims: dims_p, seed, bits, digits: 17 };
    let Ok((_, fdims, inputs)) = presk_interp::run::prepare_run(&core, &prog, &opts) else {
        eprintln!("presk dump: cannot bind the inputs of `{entry}`");
        return ExitCode::from(1);
    };
    let mut backend = presk_rt::InterpBackend::new(&core, &prog, bits, seed);
    backend.interp.sums = Some(Default::default());
    // The vjp takes the forward arguments and a cotangent; seeding it with 1
    // is what makes the adjoints `∂L/∂v` rather than a scaled multiple.
    let mut args: Vec<presk_rt::Value> = inputs.iter().map(|(_, v)| presk_rt::Value::from_interp(v).unwrap()).collect();
    if let Some(path) = weights {
        match presk_rt::load_params(path, &args[0]) {
            Ok(v) => args[0] = v,
            Err(e) => {
                eprintln!("presk dump: {}", e.message);
                return ExitCode::from(1);
            }
        }
    }
    args.push(presk_rt::Value::Real(presk_rt::Tensor { axes: vec![], data: presk_rt::Buf::F64(vec![1.0]) }));
    let one = presk_rt::Realization::uniform(presk_rt::Dtype::F64);
    let p = match presk_rt::Backend::prepare(&mut backend, vjp, &fdims, &one) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("presk dump: {}", e.message);
            return ExitCode::from(1);
        }
    };
    if let Err(e) = presk_rt::Backend::call(&mut backend, &p, args) {
        eprintln!("presk dump: {}", e.message);
        return ExitCode::from(1);
    }
    let sums = backend.interp.sums.take().unwrap_or_default();
    let s = presk_precision::sensitivity::from_run(&core, f, &adjoints, &sums);
    println!("-- sensitivity of `{entry}`: ‖∂L/∂v‖₁ at each site, one reverse sweep at {bits} bits");
    println!("{:<34} {:>14}", "module", "Σ ‖∂L/∂v‖₁");
    for (m, v) in s.by_module() {
        println!("{m:<34} {v:>14.4e}");
    }
    println!("\n{:<34} {:>14}", "site", "‖∂L/∂v‖₁");
    for (site, v) in s.ranked().into_iter().take(top) {
        println!("{site:<34} {v:>14.4e}");
    }
    ExitCode::SUCCESS
}

/// `presk check --precision ENVELOPE.json`: the bound the analysis of plan
/// 0.5.0 §3.2 gives for the entry the envelope was measured on — by the rule
/// `--model` names, or by all four side by side — and, with `--measure`, the
/// distance the realization actually puts between the execution and the
/// exact reference, on the envelope's own inputs and weights (0.6.0 step 6).
fn check_precision(file: &Path, env_path: &Path, realize: &presk_rt::Realization, sites: usize, model: &str, measure: bool) -> ExitCode {
    use presk_precision::propagate::Model;
    let models: Vec<Model> = match model {
        "all" => Model::ALL.to_vec(),
        m => match Model::parse(m) {
            Some(m) => vec![m],
            None => {
                eprintln!("presk check: --model is one of cond, normwise, k, rss, all; not `{m}`");
                return ExitCode::from(2);
            }
        },
    };
    let env = match presk_precision::envelope::Envelope::load(env_path) {
        Ok(e) => e,
        Err(m) => {
            eprintln!("presk check: {m}");
            return ExitCode::from(1);
        }
    };
    let consts: Vec<String> = env.consts.clone();
    let (sm, prog, mut core) = match core(file, &consts, false) {
        Err(c) => return c,
        Ok(x) => x,
    };
    let diags = presk_core::ad::transform(&mut core);
    eprint!("{}", render_all(&diags, &sm, color()));
    if has_errors(&diags) {
        return ExitCode::from(1);
    }
    let Ok((entry, _, _)) = presk_interp::run::resolve_entry(&core, &prog, &env.entry) else {
        eprintln!("presk check: `{}` is not an entry of {}", env.entry, file.display());
        return ExitCode::from(1);
    };
    // The same shapes the magnitudes were measured at: `k` in a contraction's
    // rule is a shape, so a bound computed at other dimensions is a bound
    // about another program.
    let named: std::collections::BTreeMap<String, u64> = env.dims.iter().cloned().collect();
    let dims: std::collections::BTreeMap<presk_types::dim::DimVar, u64> =
        core.func(entry).dims.iter().filter_map(|d| named.get(&prog.dim_name(*d)).map(|n| (*d, *n))).collect();

    // The measured gap first, since every row is compared with it.
    let measured = if measure {
        let opts = presk_interp::RunOptions { entry: env.entry.clone(), dims: env.dims.clone(), seed: 0, bits: env.bits, digits: 17 };
        let weights = env.weights.as_ref().map(PathBuf::from);
        let (func, fdims, args) = match entry_args(&core, &prog, &opts, env.data.as_ref(), weights.as_deref()) {
            Ok(x) => x,
            Err(m) => {
                eprintln!("presk check: {m}");
                return ExitCode::from(1);
            }
        };
        let one = presk_rt::Realization::uniform(presk_rt::Dtype::F64);
        let mut exact = presk_rt::InterpBackend::new(&core, &prog, env.bits, 0);
        let reference = presk_rt::Backend::prepare(&mut exact, func, &fdims, &one).and_then(|p| presk_rt::Backend::call(&mut exact, &p, args.clone()));
        let mut b = presk_kernel::KernelBackend::new(&core, &prog, realize, 0);
        let mut a: Vec<presk_rt::Value> = args.iter().map(|v| v.realize(realize.default)).collect();
        let got = realize
            .quantize_args(&mut a)
            .and_then(|_| presk_rt::Backend::prepare(&mut b, func, &fdims, realize))
            .and_then(|p| presk_rt::Backend::call(&mut b, &p, a));
        match (reference, got) {
            (Ok(r), Ok(g)) => Some(leaf_gap(&r, &g)),
            (Err(e), _) | (_, Err(e)) => {
                eprintln!("presk check: {}", e.message);
                return ExitCode::from(1);
            }
        }
    } else {
        None
    };

    let runs: Vec<presk_precision::propagate::Analysis> =
        models.iter().map(|m| presk_precision::propagate::propagate_with(&core, &prog, entry, &dims, &env, realize, *m)).collect();
    let a = &runs[0];
    println!("presk check --precision: `{}` under {}", env.entry, realize.label());
    println!("  magnitude   {:.4e}", a.output.mag);
    for r in &runs {
        let what = if r.model.is_bound() { "bound   " } else { "estimate" };
        let ratio = match measured {
            Some(g) if g > 0.0 => format!("  {:>9.2e}× the measured gap", r.output.err / g),
            _ => String::new(),
        };
        println!("  {what} {:<9} {:.4e}  ({:.2e} relative){ratio}", r.model.name(), r.output.err, r.output.relative());
    }
    if let Some(g) = measured {
        println!("  measured             {:.4e}  ({:.2e} relative)  native against the exact reference", g, if a.output.mag > 0.0 { g / a.output.mag } else { f64::NAN });
    }
    println!(
        "  envelope    {} over {} samples, {} sites, {}",
        env_path.display(),
        env.samples,
        a.sites.len(),
        env.weights.as_deref().map(|w| format!("weights {w}")).unwrap_or_else(|| "initial weights".into())
    );
    if !a.uncovered.is_empty() {
        println!("  uncovered   {} site(s) the envelope does not mention, first {}", a.uncovered.len(), a.uncovered.first().map(|s| s.as_str()).unwrap_or("-"));
    }
    if !a.unbounded.is_empty() {
        println!("  unbounded   {} site(s) divide by something with no measured floor: {}", a.unbounded.len(), a.unbounded.first().map(|s| s.as_str()).unwrap_or("-"));
    }
    if let Some(r) = runs.iter().find(|r| !r.fallback.is_empty()) {
        println!("  fallback    {} sum(s) with no recorded terms took the `k` rule, first {} (an envelope from before 0.6.0 step 6?)", r.fallback.len(), r.fallback[0]);
    }
    if sites > 0 {
        let mut rows: Vec<(&String, &presk_precision::propagate::Bound)> = a.sites.iter().collect();
        rows.sort_by(|x, y| y.1.err.partial_cmp(&x.1.err).unwrap_or(std::cmp::Ordering::Equal));
        println!("{:<30} {:>12} {:>12} {:>12} {:>12}", "site", "magnitude", a.model.name(), "relative", "condition");
        for (site, b) in rows.iter().take(sites) {
            let c = env.condition(site).map(|c| format!("{c:.3e}")).unwrap_or_else(|| "-".into());
            println!("{site:<30} {:>12.3e} {:>12.3e} {:>12.3e} {c:>12}", b.mag, b.err, b.relative());
        }
    }
    ExitCode::SUCCESS
}

/// `presk calibrate --precision`: the magnitude envelope of plan 0.5.0 §3.2.
#[allow(clippy::too_many_arguments)]
fn calibrate_precision(
    file: &Path,
    precision: bool,
    entry: Option<&str>,
    dims: &[String],
    consts: &[String],
    data: Option<&Path>,
    samples: Option<usize>,
    from: usize,
    seed: u64,
    bits: u32,
    out: Option<&Path>,
    compare: Option<&Path>,
    weights: Option<&Path>,
) -> ExitCode {
    if !precision {
        eprintln!("presk calibrate: --precision is required (the roofline is `presk bench --calibrate`)");
        return ExitCode::from(2);
    }
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
    let Some(entry) = entry else {
        eprintln!("presk calibrate: --entry is required");
        return ExitCode::from(2);
    };
    let opts = presk_interp::RunOptions { entry: entry.into(), dims, seed, bits, digits: 17 };
    let mut backend = presk_rt::InterpBackend::new(&core, &prog, bits, seed);
    backend.interp.magnitudes = Some(Default::default());
    backend.interp.minima = Some(Default::default());
    backend.interp.elems = Some(Default::default());
    backend.interp.ops = Some(Default::default());
    backend.interp.terms = Some(Default::default());
    let today = presk_precision::envelope::today();
    let mut env = presk_precision::envelope::Envelope::new(&file.display().to_string(), entry, bits, &today);
    env.dims = opts.dims.clone();
    env.consts = consts.to_vec();
    env.weights = weights.map(|w| w.display().to_string());
    env.data = data.map(|d| presk_precision::envelope::DataSlice { path: d.display().to_string(), from, samples });
    let run = entry_args(&core, &prog, &opts, env.data.as_ref(), weights).and_then(|(func, fdims, args)| {
        let one = presk_rt::Realization::uniform(presk_rt::Dtype::F64);
        presk_rt::Backend::prepare(&mut backend, func, &fdims, &one).and_then(|p| presk_rt::Backend::call(&mut backend, &p, args)).map_err(|e| e.message)
    });
    if let Err(m) = run {
        eprintln!("presk calibrate: {m}");
        return ExitCode::from(1);
    }
    let m = backend.interp.magnitudes.take().unwrap_or_default();
    env.observe(&m);
    env.observe_minima(&backend.interp.minima.take().unwrap_or_default());
    env.observe_elems(&backend.interp.elems.take().unwrap_or_default());
    env.observe_ops(&backend.interp.ops.take().unwrap_or_default());
    env.observe_terms(&backend.interp.terms.take().unwrap_or_default());
    env.samples = samples.unwrap_or(1).max(1);
    println!("presk calibrate: {} sites over {} samples of `{entry}` at {bits} bits", env.sites.len(), env.samples);
    let mut widest: Vec<(&String, &f64)> = env.sites.iter().collect();
    widest.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (site, v) in widest.iter().take(5) {
        println!("  {site:<28} {v:.4e}");
    }
    // The sums whose terms cancel most: the condition number the bound now
    // uses in place of `k` (0.6.0 step 6).
    let mut conds: Vec<(&String, f64)> = env.terms.keys().filter_map(|s| env.condition(s).map(|c| (s, c))).collect();
    conds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if !conds.is_empty() {
        println!("presk calibrate: {} sums and contractions, worst condition numbers Σ|xᵢwᵢ| / |Σxᵢwᵢ|:", conds.len());
        for (site, c) in conds.iter().take(5) {
            println!("  {site:<28} {c:.4e}");
        }
    }
    if let Some(other) = compare {
        match presk_precision::envelope::Envelope::load(other) {
            Ok(o) => {
                let a = env.agreement(&o);
                println!(
                    "presk calibrate: against {}: worst ratio {:.3}x at {} over {} sites ({} zero in both, {} in only one)",
                    other.display(),
                    a.worst,
                    if a.at.is_empty() { "-" } else { &a.at },
                    a.compared,
                    a.both_zero,
                    a.only_one
                );
            }
            Err(m) => {
                eprintln!("presk calibrate: {m}");
                return ExitCode::from(1);
            }
        }
    }
    if let Some(path) = out
        && let Err(m) = env.save(path)
    {
        eprintln!("presk calibrate: {m}");
        return ExitCode::from(1);
    }
    if let Some(path) = out {
        println!("presk calibrate: wrote {}", path.display());
    }
    ExitCode::SUCCESS
}

/// The entry of `opts` and its arguments at the seam: the last one read from
/// a data slice when there is one, the first one from a checkpoint when there
/// is one, the rest synthesized as `presk run` does. The same inputs for the
/// envelope and for a gap measured beside the bound it supports.
type EntryArgs = (presk_core::FuncId, presk_interp::eval::Dims, Vec<presk_rt::Value>);

fn entry_args(core: &presk_core::Program, prog: &elab::ElabProgram, opts: &presk_interp::RunOptions, data: Option<&presk_precision::envelope::DataSlice>, weights: Option<&Path>) -> Result<EntryArgs, String> {
    let (func, fdims, mut args) = match data {
        Some(d) => {
            let (func, fdims, _, args) = presk_train::args_on_data(core, prog, opts, Path::new(&d.path), d.from, d.samples).map_err(|e| e.message)?;
            (func, fdims, args)
        }
        None => {
            let (func, fdims, inputs) = presk_interp::run::prepare_run(core, prog, opts).map_err(|e| e.message)?;
            let args = inputs.iter().map(|(_, v)| presk_rt::Value::from_interp(v)).collect::<presk_rt::Result<Vec<_>>>().map_err(|e| e.message)?;
            (func, fdims, args)
        }
    };
    // A checkpoint that does not load is an error, never a fallback to the
    // initial parameters (D48).
    if let Some(w) = weights {
        args[0] = presk_rt::load_params(w, &args[0]).map_err(|e| format!("{}: {}", w.display(), e.message))?;
    }
    Ok((func, fdims, args))
}

/// `presk bench`: one entry timed on every backend.
#[allow(clippy::too_many_arguments)]
fn bench(file: &Path, entry: &str, dims: &[String], consts: &[String], bits: u32, repeat: usize, seed: u64, opt: Opt, calibrate: Option<&Path>, realize: &presk_rt::Realization, native_only: bool) -> ExitCode {
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
    let time = |name: &str, b: &mut dyn presk_rt::Backend, realize: &presk_rt::Realization| -> Option<(String, f64, f64)> {
        let p = b.prepare(func, &fdims, realize).ok()?;
        // The inputs are built by the reference interpreter, as exact reals.
        // Rounding them to the machine type, and quantising the weights of a
        // quantised realization, is loading a checkpoint, not a step, so both
        // happen once, outside the timing (0.6.0 step 5b; before it, every
        // row paid a conversion from MPFR per call).
        let mut args: Vec<presk_rt::Value> = args.iter().map(|v| if b.name() == "interp" { v.clone() } else { v.realize(realize.default) }).collect();
        if let Err(e) = realize.quantize_args(&mut args) {
            eprintln!("presk bench: {name}: {}", e.message);
            return None;
        }
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
    let uniform = |d| presk_rt::Realization::uniform(d);
    let f64r = uniform(presk_rt::Dtype::F64);
    let mut rows = Vec::new();
    if !native_only {
        if let Some(r) = time(&format!("interp {bits} bits"), &mut presk_rt::InterpBackend::new(&core, &prog, bits, seed), &f64r) {
            rows.push(r);
        }
        if let Some(r) = time("kernel f64", &mut presk_kernel::KernelBackend::new(&core, &prog, &f64r, seed), &f64r) {
            rows.push(r);
        }
    }
    let mut want: Vec<(String, presk_rt::Realization)> =
        [presk_rt::Dtype::F64, presk_rt::Dtype::F32, presk_rt::Dtype::Bf16].iter().map(|d| (format!("native {d}"), uniform(*d))).collect();
    // A realization given on the command line is timed too, whatever it is:
    // a solved assignment or a declared reduction order is exactly the thing
    // one wants a row for.
    if !realize.is_uniform() || !want.iter().any(|(_, r)| r == realize) {
        want.push((format!("native {}", realize.label()), realize.clone()));
    }
    for (name, r) in &want {
        // The natural lane count of the machine type: Cranelift's vector
        // types are 128 bits, so four `f32` or two `f64`. A lane is an
        // independent output column, so the vector row is bit-identical to
        // the scalar one and the only thing that differs is the time.
        let lanes = match r.default {
            presk_rt::Dtype::F32 => 4,
            presk_rt::Dtype::F64 => 2,
            presk_rt::Dtype::Bf16 => 1,
        };
        for (suffix, n) in [("", 1usize), (" +lanes", lanes)] {
            if n == 1 && !suffix.is_empty() {
                continue;
            }
            match presk_cranelift::native_backend(&core, &prog, r, seed) {
                Ok(mut b) => {
                    b.lanes = n;
                    if let Some(row) = time(&format!("{name}{suffix}"), &mut b, r) {
                        rows.push(row);
                    }
                }
                Err(e) => eprintln!("presk bench: native backend unavailable: {}", e.message),
            }
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
fn calibrate_profile(core: &presk_core::Program, prog: &elab::ElabProgram, func: presk_core::FuncId, fdims: &presk_interp::eval::Dims, args: &[presk_rt::Value], realize: &presk_rt::Realization, repeat: usize, seed: u64, path: &Path) -> ExitCode {
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
    let dtype = backend.realize.label();
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
    let mut profile = presk_opt::profile::fit(&merged, &dtype, &machine, &|s| presk_opt::cost::weight_of_name(&s.name));
    // The same kernels timed under the *other* realizations of this machine,
    // so one profile can price a mixed assignment (0.6.0 step 3). A narrow
    // type's π is not a wide type's — that is the whole reason lanes exist —
    // and a cost model meant to compare machine types cannot have one π.
    for d in [presk_rt::Dtype::F64, presk_rt::Dtype::F32, presk_rt::Dtype::Bf16] {
        let r = presk_rt::Realization::uniform(d);
        let label = r.label();
        if label == dtype {
            continue;
        }
        for lanes in [1usize, if d == presk_rt::Dtype::F32 { 4 } else if d == presk_rt::Dtype::F64 { 2 } else { 1 }] {
            let key = if lanes > 1 { format!("{label}+lanes") } else { label.clone() };
            if profile.others.contains_key(&key) {
                continue;
            }
            let Ok(native) = presk_cranelift::NativeExec::new() else { continue };
            let (timed, samples) = presk_kernel::cost::TimedExec::new(Box::new(native));
            let mut b = presk_kernel::KernelBackend::with_exec(core, prog, &r, seed, Box::new(timed));
            b.lanes = lanes;
            let Ok(p) = presk_rt::Backend::prepare(&mut b, func, fdims, &r) else { continue };
            for _ in 0..repeat {
                if presk_rt::Backend::call(&mut b, &p, args.to_vec()).is_err() {
                    break;
                }
            }
            let medians = presk_kernel::cost::median_per_kernel(&samples.borrow(), b.kernels.len());
            let rows: Vec<presk_opt::profile::Sample> = b
                .kernels
                .iter()
                .zip(medians)
                .filter_map(|(k, t)| {
                    let seconds = t?;
                    let c = presk_kernel::cost::kernel_cost(k);
                    let signature = k.to_string().lines().next().unwrap_or("").trim_end_matches(" {").to_string();
                    Some(presk_opt::profile::Sample { signature, name: k.name.clone(), work: c.work, bytes: c.bytes, seconds })
                })
                .collect();
            if rows.len() < 3 {
                continue;
            }
            let f = presk_opt::profile::fit(&rows, &key, &machine, &|s| presk_opt::cost::weight_of_name(&s.name));
            profile.others.insert(key, presk_opt::profile::Fit { pi: f.pi, beta: f.beta, lambda: f.lambda });
        }
    }
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
    match with_backend(backend, &core, &prog, opts.bits, opts.seed, &opts.realize, |b| {
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
        DataCommand::Diff { a, b } => {
            let (mut ra, mut rb) = (Reader::open(&a)?, Reader::open(&b)?);
            if ra.header().sample_ty != rb.header().sample_ty || ra.len() != rb.len() {
                return Err(DataError { code: "E9005", message: format!("{} and {} are not of the same type and length", a.display(), b.display()) });
            }
            fn reals(v: &presk_data::Value, out: &mut Vec<f64>) {
                match v {
                    presk_data::Value::Real(x) => out.extend(x),
                    presk_data::Value::Idx(_) => {}
                    presk_data::Value::Tuple(ts) => ts.iter().for_each(|t| reals(t, out)),
                    presk_data::Value::Record(fs) => fs.iter().for_each(|(_, t)| reals(t, out)),
                }
            }
            let all = |r: &mut Reader| -> Result<Vec<f64>, DataError> {
                if r.header().sample_ty.starts_with("params ") {
                    return r.params_reals();
                }
                let mut out = Vec::new();
                for v in r.read_all()? {
                    reals(&v, &mut out);
                }
                Ok(out)
            };
            let (xa, xb) = (all(&mut ra)?, all(&mut rb)?);
            let d = xa.iter().zip(&xb).map(|(p, q)| (p - q) * (p - q)).sum::<f64>().sqrt();
            let n = xa.iter().map(|p| p * p).sum::<f64>().sqrt();
            println!("presk data: ‖a − b‖₂ {d:.6e}   ‖a‖₂ {n:.6e}   relative {:.6e}   over {} reals", d / n.max(f64::MIN_POSITIVE), xa.len());
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
fn dump(file: &Path, stage: DumpStage, unroll_stack: bool, entry: Option<&str>, dims: &[String], consts: &[String], realize: &presk_rt::Realization, profile: Option<PathBuf>, weights: Option<PathBuf>, sites: usize) -> ExitCode {
    if let DumpStage::Sensitivity = stage {
        return dump_sensitivity(file, entry, dims, consts, weights.as_deref(), sites, 0, 128);
    }
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
        println!("-- {} kernels, {} launches, {}", backend.kernels.len(), backend.launches, backend.realize.label());
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
        DumpStage::Kernel | DumpStage::Sensitivity => unreachable!("handled above"),
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
fn run(file: &Path, entry: Option<&str>, dims: &[String], consts: &[String], seed: u64, bits: u32, digits: usize, json: bool, verbose: bool, unroll_stack: bool, backend: BackendKind, realize: &presk_rt::Realization, opt: Opt, data: Option<PathBuf>, samples: Option<usize>, from: usize, weights: Option<PathBuf>) -> ExitCode {
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
    let evaluate = |b: &mut dyn presk_rt::Backend| match (&data, &weights) {
        (Some(path), None) => presk_train::run_on_data_from(b, &core, &prog, &opts, realize.clone(), path, from, samples).map_err(|e| presk_rt::RtError { code: e.code, message: e.message, span: e.span }),
        (None, None) => presk_rt::run_with(b, &core, &prog, &opts, realize),
        // Trained parameters: the arguments built as above, the first one
        // read from the checkpoint (0.6.0 step 7, a validation loss).
        (_, Some(w)) => {
            let (func, fdims, inputs, mut args) = match &data {
                Some(path) => presk_train::args_on_data(&core, &prog, &opts, path, from, samples).map_err(|e| presk_rt::RtError { code: e.code, message: e.message, span: e.span })?,
                None => {
                    let (func, fdims, inputs) = presk_interp::run::prepare_run(&core, &prog, &opts)?;
                    let args = inputs.iter().map(|(_, v)| presk_rt::Value::from_interp(v)).collect::<presk_rt::Result<Vec<_>>>()?;
                    (func, fdims, inputs, args)
                }
            };
            args[0] = presk_rt::load_params(w, &args[0])?;
            realize.quantize_args(&mut args)?;
            let p = b.prepare(func, &fdims, realize)?;
            let result = b.call(&p, args)?;
            let dims_named: Vec<(String, u64)> = fdims.iter().map(|(v, n)| (prog.dim_name(*v), *n)).collect();
            Ok(presk_interp::RunOutput { entry: opts.entry.clone(), func, dims: dims_named, inputs, result: result.to_interp(opts.bits) })
        }
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
            let topts = presk_train::TrainOptions { data: None, steps: Some(k), epochs: None, seed, bits: 256, realize: presk_rt::Realization::uniform(presk_rt::Dtype::F64), ckpt: None, checkpoints: false };
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
            for (realize, name) in [(presk_rt::Realization::uniform(presk_rt::Dtype::F64), "native_f64"), (presk_rt::Realization::uniform(presk_rt::Dtype::F32), "native_f32")] {
                let o = presk_train::TrainOptions { realize: realize.clone(), ..topts.clone() };
                match presk_cranelift::native_backend(&core, &prog, &realize, seed) {
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
