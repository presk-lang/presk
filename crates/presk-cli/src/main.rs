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

mod ra4;
mod step;

#[derive(Parser)]
#[command(
    name = "presk",
    version,
    about = concat!(
        "Presk compiler (",
        env!("CARGO_PKG_VERSION"),
        ": front-end, reference interpreter, native backend, e-graph optimizer, training, serving, precision budgets, layout plans, PyTorch and C exports, a compilation as a document)"
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
        /// Print the result alone: `--json` without `inputs`, `--verbose`
        /// without the inputs' lines. For a checkpoint too large to print
        /// (Llama 3.2 1B's 1.2·10⁹ parameters), which is then never held in
        /// exact arithmetic to be printed.
        #[arg(long)]
        result_only: bool,
        /// Unroll `stack{L}` into L calls instead of a fold (debugging aid).
        #[arg(long)]
        unroll_stack: bool,
        /// The executor: the reference interpreter by default.
        #[arg(long, value_enum, default_value_t = BackendKind::Interp)]
        backend: BackendKind,
        /// The realization: `f64`, `f32`, `bf16`, or `@FILE` for a solved one.
        #[arg(long, default_value = "f64")]
        realize: String,
        /// A realization plan, `@FILE`: its realization and a layout per nest
        /// (`realize` chapter); `--realize` is the plan with the default layout.
        #[arg(long, value_name = "@FILE", conflicts_with = "realize")]
        plan: Option<String>,
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
        /// What a gradient does with its forward pass's values: `store` them
        /// for its reverse sweep, or `recompute` them inside each callee's
        /// `.vjp` (0.10.0 step 4). The same bits either way.
        #[arg(long, value_enum, default_value_t = RematArg::Store)]
        remat: RematArg,
    },
    /// Export a program to another framework (PyTorch), or an entry as a C program.
    Export {
        file: PathBuf,
        #[arg(long, value_enum)]
        target: ExportTarget,
        /// PyTorch: `readable` (the default) or `graph`. C: `scop`, each nest of the entry at
        /// point zero between `#pragma scop` and `#pragma endscop`, with a driver.
        #[arg(long, value_enum)]
        mode: Option<ExportMode>,
        #[arg(short, long, value_name = "DIR")]
        out: PathBuf,
        /// Insert executable shape assertions.
        #[arg(long = "shapes", value_name = "check")]
        shapes: Option<String>,
        /// PyTorch: `loop` (the default) or `unroll`.
        #[arg(long = "stack", value_enum)]
        stack: Option<StackMode>,
        /// Also generate test_conformance.py for this entry (needs --dim/--const as for `run`);
        /// `train:K` replays K training steps of the interpreter.
        #[arg(long, value_name = "ENTRY")]
        conformance_entry: Option<String>,
        /// Also write train.py and pskd.py: the file's `training run` as a PyTorch loop.
        #[arg(long)]
        train: bool,
        /// Also write serve.py and pskd.py: the file's `inference serve` as a PyTorch generation loop.
        /// C: the generation function and its greedy copy, with `generate`, presk serve's loop.
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
        /// C: the entry to export, as `presk run --entry` names it.
        #[arg(long)]
        entry: Option<String>,
        /// C: the realization, `f64` (the default), `f32`, `bf16`, or `@FILE`.
        #[arg(long)]
        realize: Option<String>,
        /// C: trained parameters for the entry's first argument (a `params M`
        /// checkpoint, of the dimensions the entry is exported at).
        #[arg(long, value_name = "ckpt/final.psk.d")]
        weights: Option<PathBuf>,
        /// C: a realization plan, `@FILE` (`presk plan --out`): its realization and a
        /// layout per nest, its splits the program's threads.
        #[arg(long, value_name = "@FILE", conflicts_with = "realize")]
        plan: Option<String>,
        /// C: the native executor's workers (default: the cores), which divide the
        /// kernels a plan leaves to its default, as for `presk run`.
        #[arg(long)]
        threads: Option<usize>,
        /// C: leave the exact reference's outputs out of `conformance/` (the
        /// reference interpreter takes hours at a model's full size).
        #[arg(long)]
        no_exact: bool,
        /// C: write the host straight-line, every operation in order in one
        /// function and nothing released (plan 0.9.0 step 3), the structured
        /// host's oracle.
        #[arg(long)]
        straight_line: bool,
        /// C, with --straight-line: every buffer at an offset by its life,
        /// two buffers live at once never overlapping, the check that
        /// `presk memory`'s lifetimes are sound (plan 0.15.0 step 1). Hidden:
        /// `make c-conformance` runs it.
        #[arg(long, hide = true, requires = "straight_line")]
        placed: bool,
        /// C: the run the program is recorded from and checked against on
        /// the LLVM backend rather than the native one, so that its check
        /// is the LLVM backend's bits (plan 0.17.0 step 3, NB0). Hidden:
        /// `make llvm-conformance` runs it.
        #[arg(long, hide = true)]
        llvm: bool,
        /// What a gradient does with its forward pass's values: `store` them
        /// for its reverse sweep, or `recompute` them inside each callee's
        /// `.vjp` (0.10.0 step 4). The same bits either way.
        #[arg(long, value_enum, default_value_t = RematArg::Store)]
        remat: RematArg,
    },
    /// Dump an intermediate representation (development aid).
    ///
    /// With `--format latex|unicode`, the typed program (`--stage elab`) or
    /// PSK-Core after differentiation (`--stage core-after-ad`) read as the
    /// equations of the language reference's chapter 3.
    Dump {
        file: PathBuf,
        #[arg(long, value_enum)]
        stage: DumpStage,
        /// `text`, the stage's own listing; `latex` or `unicode`, its
        /// functions as equations (plan 0.13.0 step 4); for `--stage
        /// network`, `json`, `svg` or `tikz` (plan 0.14.0 step 3).
        #[arg(long, value_enum, default_value_t = DumpFormat::Text)]
        format: DumpFormat,
        /// For `--stage network --format svg|tikz`: the opening drawn, 0 the
        /// entry alone, 1 its first box opened (the default), then a stack,
        /// else the first box, opened at each level; `all`, every box
        /// opened, the exploded view (plan 0.14.1).
        #[arg(long, value_parser = parse_level)]
        level: Option<Level>,
        /// For `--stage network --format svg|tikz`: the box called NAME
        /// (`blocks`) opened by itself, without what is around it.
        #[arg(long = "box", value_name = "NAME", conflicts_with = "level")]
        boxed: Option<String>,
        /// For `--stage network --format svg|tikz`: each box coloured by its
        /// share of the peak of live bytes, as the page's memory overlay
        /// colours it (`presk memory`, plan 0.15.0 step 4).
        #[arg(long)]
        memory: bool,
        /// With `--format latex|unicode`: only this module.
        #[arg(long)]
        module: Option<String>,
        /// With `--format latex|unicode`: only this function (`apply`,
        /// `Linear.apply`, `loss.bwd`).
        #[arg(long)]
        function: Option<String>,
        /// With `--format latex`: a document that compiles alone, the
        /// reference's macros in its preamble.
        #[arg(long)]
        standalone: bool,
        /// Unroll `stack{L}` into L calls instead of a fold (debugging aid).
        #[arg(long)]
        unroll_stack: bool,
        /// The program of which mode (0.16.0, D160): `inference`, every
        /// command's but a training run's, `dropout` the identity; or
        /// `training`, the program as written, its masks drawn by a
        /// training run.
        #[arg(long, value_enum, default_value_t = ModeArg::Inference)]
        mode: ModeArg,
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
        /// A realization plan, `@FILE`: its realization and a layout per nest
        /// (`realize` chapter); `--realize` is the plan with the default layout.
        #[arg(long, value_name = "@FILE", conflicts_with = "realize")]
        plan: Option<String>,
        /// For `--stage core-opt`: extract by the roofline of this machine profile.
        #[arg(long, value_name = "FILE.json")]
        profile: Option<PathBuf>,
        /// What a gradient does with its forward pass's values: `store` them
        /// for its reverse sweep, or `recompute` them inside each callee's
        /// `.vjp` (0.10.0 step 4). The same bits either way.
        #[arg(long, value_enum, default_value_t = RematArg::Store)]
        remat: RematArg,
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
        /// A realization plan, `@FILE`: its realization and a layout per nest
        /// (`realize` chapter); `--realize` is the plan with the default layout.
        #[arg(long, value_name = "@FILE", conflicts_with = "realize")]
        plan: Option<String>,
        /// Rewrite the program with the e-graph optimizer before training (`presk-opt`).
        #[arg(long)]
        opt: bool,
        /// Workers a parallel kernel may use on the native backend (default: the cores).
        #[arg(long)]
        threads: Option<usize>,
        /// With --opt: the machine profile for the extraction cost.
        #[arg(long, value_name = "FILE.json")]
        profile: Option<PathBuf>,
        /// What a gradient does with its forward pass's values: `store` them
        /// for its reverse sweep, or `recompute` them inside each callee's
        /// `.vjp` (0.10.0 step 4). The same bits either way.
        #[arg(long, value_enum, default_value_t = RematArg::Store)]
        remat: RematArg,
        /// With `--backend native`: time the host by phase in every step, as
        /// `presk bench --host-profile` does a call, and print the table of a
        /// step, the first left out (plan 0.10.0 step 5). Only the clock is
        /// read: the run's weights are the same bits.
        #[arg(long)]
        host_profile: bool,
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
    /// Choose a realization plan statically: every nest's layout space costed by the profile's layout model.
    Plan {
        file: PathBuf,
        #[arg(long)]
        entry: String,
        #[arg(long = "dim", value_name = "NAME=N")]
        dims: Vec<String>,
        #[arg(long = "const", value_name = "NAME=N")]
        consts: Vec<String>,
        /// The realization the plan is for.
        #[arg(long, default_value = "f32")]
        realize: String,
        /// The program of which mode (0.16.0, D164): `inference`, for
        /// `presk run`, `bench`, `serve` and the C export; or `training`,
        /// for `presk train --plan`, whose program keeps its masks and
        /// names its nests as written.
        #[arg(long, value_enum, default_value_t = ModeArg::Inference)]
        mode: ModeArg,
        /// A calibrated profile with a layout cost model (`presk bench --calibrate FILE --space calib`).
        #[arg(long, value_name = "FILE.json")]
        profile: PathBuf,
        /// Plan these nests only (`<function>/%<let>`), the option repeated or the names separated by commas.
        #[arg(long, value_name = "NEST", value_delimiter = ',')]
        nest: Vec<String>,
        /// `full` (the reference's space) or `small` (the daily subset).
        #[arg(long, default_value = "full")]
        space: String,
        /// Print the prediction of every point, not only the best.
        #[arg(long)]
        cost: bool,
        /// Write the plan here.
        #[arg(long, value_name = "PLAN.json")]
        out: Option<PathBuf>,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// What a gradient does with its forward pass's values: `store` them
        /// for its reverse sweep, or `recompute` them inside each callee's
        /// `.vjp` (0.10.0 step 4). The same bits either way.
        #[arg(long, value_enum, default_value_t = RematArg::Store)]
        remat: RematArg,
    },
    /// One compilation as a document: every stage's representation and the reasons for its decisions, in DIR/explain.md (plan 0.11.0).
    Explain {
        file: PathBuf,
        #[arg(long)]
        entry: String,
        #[arg(long = "dim", value_name = "NAME=N")]
        dims: Vec<String>,
        #[arg(long = "const", value_name = "NAME=N")]
        consts: Vec<String>,
        /// Solve for a realization under this relative budget, as `presk solve --budget`.
        #[arg(long, conflicts_with_all = ["realize", "plan"])]
        budget: Option<f64>,
        /// With --budget: the solve's menu, as `presk solve --menu` (plan
        /// 0.17.0 step 4): `f64,f32,bf16` by default, `int8` and `int4` the
        /// integer rungs a parameter of rank two or more is offered besides.
        #[arg(long, value_name = "RUNGS", requires = "budget")]
        menu: Option<String>,
        /// A realization given: `f64`, `f32`, `bf16` or `@FILE.json`.
        #[arg(long, conflicts_with = "plan")]
        realize: Option<String>,
        /// A realization plan given, `@PLAN.json`, its realization with it.
        #[arg(long, value_name = "@PLAN.json")]
        plan: Option<String>,
        /// A calibrated profile: the solver prices time, and the layouts are planned.
        #[arg(long, value_name = "FILE.json")]
        profile: Option<PathBuf>,
        /// Worker threads of the native executor, which the plan's splits are counted on.
        #[arg(long, value_name = "N")]
        threads: Option<usize>,
        /// What a gradient does with its forward pass's values: `store` them
        /// for its reverse sweep, or `recompute` them inside each callee's
        /// `.vjp` (0.10.0 step 4). The same bits either way.
        #[arg(long, value_enum, default_value_t = RematArg::Store)]
        remat: RematArg,
        /// Rewrite the program with the e-graph optimizer.
        #[arg(long)]
        opt: bool,
        /// Trained weights in place of the initial ones.
        #[arg(long, value_name = "FILE.psk.d")]
        weights: Option<PathBuf>,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// The exact reference's precision.
        #[arg(long, default_value_t = 128)]
        bits: u32,
        /// An envelope given (`presk calibrate --precision --out`) in place of measuring one.
        #[arg(long, value_name = "ENVELOPE.json")]
        envelope: Option<PathBuf>,
        /// No exact reference: no envelope measured, no gap, no conformance outputs.
        #[arg(long)]
        no_exact: bool,
        /// The targets, separated by commas: `native`, `c`, `pytorch`.
        #[arg(long, default_value = "native")]
        target: String,
        /// Also run it: natively against the exact reference, a call timed, each kernel beside its prediction, the host by phase.
        #[arg(long)]
        measure: bool,
        /// A site to trace through every stage (`<function>/%<let>`), the option repeated; by default the largest shares of the estimate and the slowest predicted nests.
        #[arg(long = "site", value_name = "SITE")]
        sites: Vec<String>,
        /// What to write besides `explain.md`: `json`, the document's facts and the provenance table as values, `explain.json` (plan 0.13.0 step 2); `html`, the pipeline view `explain.html` too (step 3).
        #[arg(long, value_name = "md|json|html")]
        format: Option<String>,
        /// The directory the document, its listings and its artifacts are written to.
        #[arg(long, value_name = "DIR")]
        out: PathBuf,
    },
    /// What one call of an entry needs in memory: its parameters, the peak of its live bytes, the C program's arena beside them (plan 0.15.0).
    ///
    /// The entry is recorded as `presk export --target c` records it, on an
    /// executor that runs no kernel; each buffer the call makes is live from
    /// its first write to its last read. Counted bytes, never a process's
    /// resident size.
    Memory {
        file: PathBuf,
        #[arg(long, required_unless_present = "step")]
        entry: Option<String>,
        /// The training run's step in place of an entry: the calls `presk
        /// train` makes, the parameters and AdamW's moments held across,
        /// under `--remat store` and `recompute` side by side (plan 0.15.0
        /// step 2).
        #[arg(long, conflicts_with_all = ["entry", "remat"])]
        step: bool,
        /// Also the inputs, the outputs and the peak as polynomials in the
        /// entry's dimensions, each buffer's bytes interpolated on a grid of
        /// recordings and checked off it (plan 0.15.0 step 3).
        #[arg(long, conflicts_with = "step")]
        symbolic: bool,
        #[arg(long = "dim", value_name = "NAME=N")]
        dims: Vec<String>,
        #[arg(long = "const", value_name = "NAME=N")]
        consts: Vec<String>,
        /// The realization: `f64` (the default), `f32`, `bf16` or `@FILE.json`.
        #[arg(long, conflicts_with = "plan")]
        realize: Option<String>,
        /// A realization plan, `@PLAN.json`, its realization with it.
        #[arg(long, value_name = "@PLAN.json")]
        plan: Option<String>,
        /// What a gradient does with its forward pass's values: `store` them
        /// for its reverse sweep, or `recompute` them inside each callee's
        /// `.vjp` (0.10.0 step 4).
        #[arg(long, value_enum)]
        remat: Option<RematArg>,
        /// The calls shown, this many levels deep.
        #[arg(long, default_value_t = 2)]
        depth: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Measure every point of every nest's layout space on the entry's own buffers, asserting each computes point 0's bits.
    Autotune {
        file: PathBuf,
        #[arg(long)]
        entry: String,
        #[arg(long = "dim", value_name = "NAME=N")]
        dims: Vec<String>,
        #[arg(long = "const", value_name = "NAME=N")]
        consts: Vec<String>,
        #[arg(long, default_value = "f32")]
        realize: String,
        /// `full` (the reference's space) or `small` (the daily subset).
        #[arg(long, default_value = "full")]
        space: String,
        /// Timed launches per point, after one warm-up; 0 runs each point once, untimed, for its bits.
        #[arg(long, default_value_t = 7)]
        repeat: usize,
        /// Also record the layout cost model's prediction of every point.
        #[arg(long, value_name = "FILE.json")]
        profile: Option<PathBuf>,
        /// Measure these nests only, the option repeated or the names separated by commas; recorded in sweep.json.
        #[arg(long, value_name = "NEST", value_delimiter = ',')]
        nest: Vec<String>,
        /// Also check every point's bits on the kernel interpreter (slow at large shapes).
        #[arg(long)]
        interp: bool,
        /// A sample: the model's N best points per nest (with --random; needs --profile).
        #[arg(long, value_name = "N", default_value_t = 0)]
        top: usize,
        /// A sample: N other points per nest, drawn by --seed.
        #[arg(long, value_name = "N", default_value_t = 0)]
        random: usize,
        /// Only the points whose place is K modulo N (point 0 in every shard), to split a check over processes.
        #[arg(long, value_name = "K/N")]
        shard: Option<String>,
        /// The day's directory: one JSON line per point, per nest, and sweep.json.
        #[arg(long, value_name = "DIR")]
        out: PathBuf,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// What a gradient does with its forward pass's values: `store` them
        /// for its reverse sweep, or `recompute` them inside each callee's
        /// `.vjp` (0.10.0 step 4). The same bits either way.
        #[arg(long, value_enum, default_value_t = RematArg::Store)]
        remat: RematArg,
    },
    /// Price the points of a recorded day again under another profile, their times copied: nothing is measured.
    Reprice {
        file: PathBuf,
        /// The entry whose sweeps to re-price; the day's other sweeps are left.
        #[arg(long)]
        entry: String,
        /// The day `presk autotune` wrote: one sweep, or one per configuration in subdirectories.
        #[arg(long, value_name = "DIR")]
        day: PathBuf,
        /// The profile whose layout cost model prices the points.
        #[arg(long, value_name = "FILE.json")]
        profile: PathBuf,
        /// Where the re-priced day goes, laid out as the day is.
        #[arg(long, value_name = "DIR")]
        out: PathBuf,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// What a gradient does with its forward pass's values: `store` them
        /// for its reverse sweep, or `recompute` them inside each callee's
        /// `.vjp` (0.10.0 step 4). The same bits either way.
        #[arg(long, value_enum, default_value_t = RematArg::Store)]
        remat: RematArg,
    },
    /// Rank accuracy of the layout cost model on two days of measurement, the statements of plan 0.7.0, and plan 0.8.0's Q.
    Rank {
        day1: PathBuf,
        /// The second day; the first again for one sweep.
        day2: PathBuf,
        #[arg(long, value_name = "FILE.md")]
        out: PathBuf,
        /// A check of statement L run apart (`autotune --repeat 0 --interp --shard`).
        #[arg(long, value_name = "DIR")]
        check: Option<PathBuf>,
        /// Refuse a day whose `day.txt` names a commit that does not descend from this tag, or a modified working tree (plan 0.8.0's freeze; run in the repository).
        #[arg(long, value_name = "TAG")]
        frozen: Option<String>,
        /// The entry statements M, N and O are about.
        #[arg(long, default_value = "SelfAttention.apply")]
        entry: String,
        /// Its configuration for M and O, as autotune records it (constants, then dims).
        #[arg(long, default_value = "B=16,T=256")]
        at: String,
        /// Its type for M and O.
        #[arg(long = "type", default_value = "f32")]
        dtype: String,
        /// The GEMM entry statement P is about, and its configuration.
        #[arg(long, default_value = "gemm")]
        gemm: String,
        #[arg(long, default_value = "N=1024")]
        gemm_at: String,
        /// Decide plan 0.8.0's statement Q on days recorded before the capture of shapes too (0.7.0's); days that record it always do.
        #[arg(long)]
        statement_q: bool,
    },
    /// Fit a profile's layout cost model again on its samples, in another form: nothing is measured.
    Refit {
        profile: PathBuf,
        /// The load term: `group`, `machine` or `none` (0.7.0's model).
        #[arg(long, value_name = "TERM", default_value = "group")]
        load_term: String,
        /// How the terms combine: `max`, `sum` or `pnorm`, 0.8.0's frozen form.
        #[arg(long, value_name = "FORM", default_value = "pnorm")]
        combine: String,
        /// Where the refitted profile goes; the profile itself when omitted.
        #[arg(long, value_name = "FILE.json")]
        out: Option<PathBuf>,
        /// Fit every form under every load term on the same samples and print their constants and u; write nothing.
        #[arg(long)]
        compare: bool,
        /// Cost every sample again first, its time kept: its layout lowered at its product on this program (examples/experiments/matmul.psk) and costed on the profile's machine description.
        #[arg(long, value_name = "FILE.psk")]
        recost: Option<PathBuf>,
        /// The product entry of --recost's program, with dimensions I, K and J.
        #[arg(long, default_value = "matmul")]
        entry: String,
        /// The rules of the planned cost --recost costs under: `none`, or among `lines`, `shared`, `overhead`, comma-separated. By default, the profile's own.
        #[arg(long, value_name = "RULES", requires = "recost")]
        rules: Option<String>,
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
        /// Calls per backend before the timed ones, not timed: the native
        /// backend compiles its kernels on its first call (the PyTorch
        /// baseline at 0.9.0, D87).
        #[arg(long, default_value_t = 0)]
        warmup: usize,
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
        /// Time the native backend's host by phase (plan 0.10.0 step 5): under
        /// the plan, or the realization's default layout, each call's time split
        /// between the kernels and what the host does around them — finding a
        /// kernel, its buffers, slices and views, copies, threads — with
        /// `--repeat` calls timed so and as many not, alternated after at least
        /// one warm-up call, which measures the timers' own cost. The two kinds
        /// of call give the same bits, which is checked.
        #[arg(long, conflicts_with = "calibrate")]
        host_profile: bool,
        /// With --host-profile: list each launch of a call timed by phase, in
        /// the call's order (plan 0.17.2 step 4): its kernel, its own code's
        /// median and fastest over those calls, and its buffers of 64 KiB or
        /// more at their addresses, modulo 4 KiB, 64 KiB and the set strides
        /// of the CPU's L1d and L2. Without it, nothing is timed per launch.
        #[arg(long, requires = "host_profile")]
        launches: bool,
        /// With --calibrate: `calib` times the entry's one contraction (a GEMM)
        /// under a spanning set of layouts and fits the layout cost model into
        /// the profile (0.7.0 step 4); without it, the roofline of every kernel.
        #[arg(long, value_name = "calib", requires = "calibrate")]
        space: Option<String>,
        /// With --space calib: the layout cost model's load term (plan 0.8.0 §3.1), its rate fitted
        /// per group as π (`group`), once for the machine (`machine`), or absent, 0.7.0's model (`none`).
        #[arg(long, value_name = "TERM", default_value = "group")]
        load_term: String,
        /// With --space calib: how the model's terms combine (plan 0.8.0 §3.2): `max`, `sum` of the
        /// in-core and memory terms, or `pnorm` with its exponent fitted, 0.8.0's frozen form.
        #[arg(long, value_name = "FORM", default_value = "pnorm")]
        combine: String,
        /// With --space calib: leave out the layouts stated on fewer workers than this. `make
        /// calibrate-ra4` passes 2 at N = 2048, whose one-worker layouts took three quarters of its
        /// time (0.8.0 step 4, D67).
        #[arg(long, value_name = "N", default_value_t = 1)]
        min_workers: usize,
        /// With --space calib: the rules of the planned cost the samples are costed under (0.8.0
        /// step 4, D70): `none`, or among `lines`, `shared`, `overhead`. By default, the rules of the
        /// profile's samples, and for a new profile the frozen model's, `lines,overhead`.
        #[arg(long, value_name = "RULES")]
        rules: Option<String>,
        /// The realization to calibrate (kernels are timed in this machine type).
        #[arg(long, default_value = "f64")]
        realize: String,
        /// A realization plan, `@FILE`: its realization and a layout per nest
        /// (`realize` chapter); `--realize` is the plan with the default layout.
        #[arg(long, value_name = "@FILE", conflicts_with = "realize")]
        plan: Option<String>,
        /// Time the native backend only. The reference interpreter's row is
        /// exact arbitrary precision, so at a thousand cubed it is not
        /// affordable and its presence is what stops a large benchmark from
        /// finishing at all.
        #[arg(long)]
        native_only: bool,
        /// With --plan: time the plan's row alone, the native backend under the
        /// plan, and not the uniform realizations' rows (0.9.0 step 5: the C
        /// export's speed against the native backend's, statement C4). With
        /// --calibrate: the plan's kernels alone, not the same kernels again
        /// under the other uniform realizations (plan 0.17.0 step 1).
        #[arg(long, requires = "plan")]
        plan_only: bool,
        /// The native rows, and --calibrate, on the LLVM backend: the native
        /// backend's host with each kernel printed as C and compiled by
        /// LLVM's clang -O3 (plan 0.17.0 step 3). The same bits.
        #[arg(long)]
        llvm: bool,
        /// Trained parameters for the entry's first argument (a `params M` file,
        /// of the dimensions the entry is timed at), as for `presk run`.
        #[arg(long, value_name = "ckpt/final.psk.d")]
        weights: Option<PathBuf>,
        /// What a gradient does with its forward pass's values: `store` them
        /// for its reverse sweep, or `recompute` them inside each callee's
        /// `.vjp` (0.10.0 step 4). The same bits either way.
        #[arg(long, value_enum, default_value_t = RematArg::Store)]
        remat: RematArg,
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
        /// What a gradient does with its forward pass's values: `store` them
        /// for its reverse sweep, or `recompute` them inside each callee's
        /// `.vjp` (0.10.0 step 4). The same bits either way.
        #[arg(long, value_enum, default_value_t = RematArg::Store)]
        remat: RematArg,
    },
    /// Solve for a realization under a declared error budget.
    Solve {
        file: PathBuf,
        /// The magnitude envelope the bound and the estimate are read from.
        #[arg(long, value_name = "ENVELOPE.json")]
        precision: PathBuf,
        /// Relative error budget for the entry's output: on its largest
        /// magnitude, and on its largest elementwise gap when it is a tensor.
        #[arg(long, default_value_t = 1e-3)]
        budget: f64,
        /// An absolute error budget instead, on the output's largest
        /// elementwise gap (0.12.0): ε\* from a model's margins is a number
        /// on its logits.
        #[arg(long, value_name = "EPS", conflicts_with = "budget")]
        budget_abs: Option<f64>,
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
        /// The menu the solve offers, comma-separated (plan 0.17.0 step 4,
        /// D177): `f64`, `f32` and `bf16`, the default, and `int8` and
        /// `int4`, which offer a parameter of rank two or more its integers
        /// under both schemes (`int8/absmax`, `int4/pow2`… for one), scaled
        /// along the axis its readers contract, in the solve itself rather
        /// than in a pass after it as `--quant` does.
        #[arg(long, value_name = "RUNGS")]
        menu: Option<String>,
        /// The estimate's rounding term (D155): `measured`, the envelope's
        /// largest `|fl_d(x) − x|` over the sampled values, which is 0 where
        /// every value is exact in the type, the default since 0.16.0 (an
        /// envelope written before D155 has none, and keeps `unit`); or
        /// `unit`, `u(d)·max|x|` at each site, the solver of 0.12.0–0.15.0.
        #[arg(long, value_enum, default_value_t = RoundingArg::Measured)]
        rounding: RoundingArg,
        /// After the measurement passes, narrow by measurement (D155):
        /// `measured` narrows each decision one step, the largest saving
        /// first, and keeps it while the measured gap stays inside the
        /// budget divided by `--margin`, in up to three passes; `heldout`
        /// does so, then undoes the narrowings kept, the last first, until
        /// the gap on the samples after the envelope's (`--heldout`) is
        /// inside it too (D161); `off`, the default, does not.
        #[arg(long, value_enum, default_value_t = NarrowArg::Off)]
        narrow: NarrowArg,
        /// The proposal's estimate (D161): `sum`, the sites' first-order
        /// terms added and aimed at the budget, the default; or `rss`, their
        /// root-sum-square, the roundings taken as independent, aimed at
        /// the budget divided by `--margin`.
        #[arg(long, value_enum, default_value_t = EstimateArg::Sum)]
        estimate: EstimateArg,
        /// The factor kept between the budget and what is spent in sample
        /// (D161): the target of `--estimate rss` and the threshold of the
        /// narrowings, the budget divided by it. The default solve does not
        /// read it.
        #[arg(long, default_value_t = 2.5)]
        margin: f64,
        /// With `--narrow heldout`: how many samples after the envelope's to
        /// hold out (default: as many as the envelope's).
        #[arg(long)]
        heldout: Option<usize>,
        /// Rule 2 (0.16.1, D169): a value is offered only when its first
        /// narrowing, `f64` to `f32`, saves at least this fraction of the
        /// entry's all-`f64` traffic, in bytes even under `--profile`; 0
        /// turns the rule off. A value of one element is never offered
        /// (rule 1).
        #[arg(long, value_name = "θ", default_value_t = 1e-3)]
        min_gain: f64,
        /// The sensitivity sweep's arithmetic (plan 0.17.1, D185): `exact`,
        /// the reference interpreter at `--bits`, the default; or `f64`, the
        /// same evaluator in binary64, for a model whose sweep in MPFR does
        /// not fit in memory (2.7 times an exact run's). The sweep ranks; the
        /// measurement against the exact reference decides either way.
        #[arg(long, value_enum, default_value_t = SweepArg::Exact)]
        sweep: SweepArg,
        /// Where each measurement runs (plan 0.17.1, D185): `kernel`, the
        /// kernel interpreter, the default; or `native`, the native backend,
        /// which computes its bits, the same gaps about twenty times faster.
        #[arg(long, value_enum, default_value_t = MeasureArg::Kernel)]
        measure: MeasureArg,
        /// Solve one training step under a budget on its **update**, the
        /// envelope being `presk calibrate --step`'s (0.6.0 step 7b): every
        /// function of the step measured alone in `f32` and `bf16`, greedy
        /// by predicted time per unit of gap, then verified.
        #[arg(long)]
        step: bool,
        /// With --step (plan 0.17.0 step 5): `function`, one type per
        /// function of the step, J′'s solver and the default; or `value`,
        /// one type per stored value, each measured alone, the master
        /// weights and the moments kept in the default type.
        #[arg(long, value_enum, default_value_t = GrainArg::Function, requires = "step")]
        grain: GrainArg,
        /// What a gradient does with its forward pass's values: `store` them
        /// for its reverse sweep, or `recompute` them inside each callee's
        /// `.vjp` (0.10.0 step 4). The same bits either way.
        #[arg(long, value_enum, default_value_t = RematArg::Store)]
        remat: RematArg,
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
    /// Readable PyTorch: the typed program printed as Python.
    Pytorch,
    /// A C program of one entry: its kernels and its host, computing the
    /// native backend's bits (plan 0.9.0; needs --entry).
    C,
}

#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum ExportMode {
    Readable,
    Graph,
    /// C only (plan 0.9.0 step 8): the entry's nests at point zero, each a
    /// standalone C function between `#pragma scop` and `#pragma endscop`.
    Scop,
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
    /// PSK-Kernel v0 printed as C and compiled by LLVM's clang -O3, on the
    /// native backend's host (plan 0.17.0 step 3).
    Llvm,
}

/// `--threads N`: how many workers a parallel kernel may use. Read by the
/// native executor; 1 keeps every kernel on one thread.
/// `--remat`: what a gradient does with its forward pass's values (plan
/// 0.10.0 §3.2, question 76).
#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum RematArg {
    /// Keep them: each `.vjp` a `.bwd` after a `.fwd`, no forward pass twice.
    Store,
    /// Compute them again inside each callee's `.vjp`, as before 0.10.0 step 4.
    Recompute,
}

/// The estimate's rounding term of `presk solve` (D155).
#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum RoundingArg {
    /// `u(d)·max|x|`: the unit roundoff times the magnitude, as every row of
    /// record was solved until 0.16.0.
    Unit,
    /// The envelope's measured `max |fl_d(x) − x|`, the default (D157).
    Measured,
}

/// The grain `presk solve --step` decides at (plan 0.17.0 step 5, D178).
#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum GrainArg {
    /// One type per function of the step (D52), J′'s solver.
    Function,
    /// One type per stored value of the step.
    Value,
}

/// Whether `presk solve` narrows by measurement after it passes (D155).
#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum NarrowArg {
    /// No narrowing after the measurement: the solver of record.
    Off,
    /// Each decision narrowed one step and kept while the measured gap fits
    /// the budget over the margin.
    Measured,
    /// The same, then stopped early on held-out samples (D161).
    Heldout,
}

/// The arithmetic of `presk solve`'s sensitivity sweep (plan 0.17.1, D185).
#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum SweepArg {
    /// The reference interpreter at `--bits`: the solver of record.
    Exact,
    /// The same evaluator in binary64.
    F64,
}

/// Where `presk solve`'s measurements run (plan 0.17.1, D185).
#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum MeasureArg {
    /// The kernel interpreter: the solver of record.
    Kernel,
    /// The native backend, with the kernel interpreter's bits.
    Native,
}

/// The estimate of `presk solve`'s proposal (D161).
#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum EstimateArg {
    /// The sum of the first-order terms, aimed at the budget.
    Sum,
    /// Their root-sum-square, aimed at the budget over the margin.
    Rss,
}

/// The policy of the command running, set once from `main` before anything
/// differentiates; `store` when the command has no `--remat`.
static REMAT: std::sync::OnceLock<presk_core::ad::Remat> = std::sync::OnceLock::new();

/// The mode the CLI lowers in (0.16.0, D160): the inference reading for
/// every command but `presk dump --mode training` and `presk plan --mode
/// training` (D164); a training run lowers its own program, as written.
static MODE: std::sync::OnceLock<presk_core::inference::Mode> = std::sync::OnceLock::new();

fn set_mode(m: ModeArg) {
    let _ = MODE.set(match m {
        ModeArg::Inference => presk_core::inference::Mode::Inference,
        ModeArg::Training => presk_core::inference::Mode::Training,
    });
}

/// `presk dump --mode`.
#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum ModeArg {
    /// The inference reading: `dropout` the identity, no mask.
    Inference,
    /// The program as written, as a training run compiles it.
    Training,
}

fn set_remat(r: RematArg) {
    let _ = REMAT.set(match r {
        RematArg::Store => presk_core::ad::Remat::Store,
        RematArg::Recompute => presk_core::ad::Remat::Recompute,
    });
}

pub(crate) fn remat() -> presk_core::ad::Remat {
    REMAT.get().copied().unwrap_or_default()
}

/// The mode the command lowers in: the inference reading unless the command
/// set another (`presk dump --mode`, `presk plan --mode`).
pub(crate) fn mode() -> presk_core::inference::Mode {
    MODE.get().copied().unwrap_or(presk_core::inference::Mode::Inference)
}

/// Automatic differentiation under the command's `--remat` (the driver's
/// [`presk_driver::ad::differentiate`]).
pub(crate) fn differentiate(core: &mut presk_core::Program, groups: bool) -> Vec<presk_syntax::Diagnostic> {
    presk_driver::ad::differentiate(core, remat(), groups)
}

/// The command's `--remat`, and the fusion groups kept or not.
pub(crate) fn ad_options(groups: bool) -> presk_core::ad::Options {
    presk_driver::ad::options(remat(), groups)
}

pub(crate) use presk_driver::ad::stored_vjps;

fn set_threads(n: Option<usize>) {
    if let Some(n) = n {
        // SAFETY: set before any backend is built and before any thread is
        // spawned, from the single-threaded `main`.
        unsafe { std::env::set_var("PRESK_THREADS", n.max(1).to_string()) };
    }
}

/// The command-line tool's report: standard error (`presk_driver::Stderr`).
pub(crate) fn stderr() -> presk_driver::Stderr {
    presk_driver::Stderr { color: color() }
}

/// The exit code of a step that stopped.
pub(crate) fn exit(stop: presk_driver::Stop) -> ExitCode {
    ExitCode::from(stop.0)
}

/// `--backend llvm` without its compiler: refused (plan 0.17.0 §3), rather
/// than run on another backend than the one asked for.
fn require_llvm() {
    if let Err(m) = presk_llvm::available() {
        eprintln!("presk: --backend llvm: {m}");
        std::process::exit(1);
    }
}

/// The native backend, or with `llvm` the LLVM backend on the native host
/// (`presk bench --llvm`).
fn compiled_backend<'p>(llvm: bool, core: &'p presk_core::Program, prog: &'p elab::ElabProgram, r: &presk_rt::Realization) -> presk_rt::Result<presk_kernel::KernelBackend<'p>> {
    if llvm {
        require_llvm();
        presk_llvm::llvm_backend(core, prog, r)
    } else {
        presk_cranelift::native_backend(core, prog, r)
    }
}

/// Runs `f` with the chosen backend (the driver's
/// [`presk_driver::backend::with_backend`]).
fn with_backend<R>(kind: BackendKind, core: &presk_core::Program, prog: &elab::ElabProgram, bits: u32, plan: &presk_rt::Plan, f: impl FnOnce(&mut dyn presk_rt::Backend) -> R) -> R {
    let kind = match kind {
        BackendKind::Interp => presk_driver::backend::BackendKind::Interp,
        BackendKind::Kernel => presk_driver::backend::BackendKind::Kernel,
        BackendKind::Native => presk_driver::backend::BackendKind::Native,
        BackendKind::Llvm => {
            require_llvm();
            presk_driver::backend::BackendKind::Llvm
        }
    };
    presk_driver::backend::with_backend(kind, core, prog, bits, plan, &mut stderr(), f)
}

/// `--opt [--profile FILE]`.
struct Opt {
    on: bool,
    profile: Option<PathBuf>,
}

impl Opt {
    /// The driver's [`presk_driver::opt::Opt::apply`], reported on standard
    /// error.
    fn apply(&self, core: &mut presk_core::Program, prog: &elab::ElabProgram, verbose: bool, dims: &[(String, u64)]) -> Result<(), ExitCode> {
        presk_driver::opt::Opt { on: self.on, profile: self.profile.clone() }.apply(core, prog, verbose, dims, &mut stderr()).map(|_| ()).map_err(exit)
    }

    /// [`Opt::apply`] for a command that runs `plan`: a realization or a
    /// plan that names sites of a function the optimizer rewrote is refused
    /// (`E7002`, plan 0.13.0 step 1, D134).
    #[allow(clippy::too_many_arguments)]
    fn apply_for(&self, core: &mut presk_core::Program, prog: &elab::ElabProgram, verbose: bool, dims: &[(String, u64)], plan: &presk_rt::Plan, sm: &presk_syntax::SourceMap) -> Result<(), ExitCode> {
        presk_driver::opt::Opt { on: self.on, profile: self.profile.clone() }.apply_for(core, prog, verbose, dims, plan, sm, &mut stderr()).map(|_| ()).map_err(exit)
    }
}

/// `--realize f64|f32|bf16|@FILE` ([`presk_driver::plan::parse_realize`]);
/// a usage error ends the command.
fn parse_realize(spec: &str) -> presk_rt::Realization {
    match presk_driver::plan::parse_realize(spec) {
        Ok(r) => r,
        Err(m) => {
            eprintln!("presk: {m}");
            std::process::exit(2);
        }
    }
}

/// `--plan @FILE`, or the plan `--realize` means
/// ([`presk_driver::plan::parse_plan`]); a usage error ends the command.
fn parse_plan(plan: Option<&str>, realize: &str) -> presk_rt::Plan {
    match presk_driver::plan::parse_plan(plan, realize) {
        Ok(p) => p,
        Err(m) => {
            eprintln!("presk: {m}");
            std::process::exit(2);
        }
    }
}

/// [`presk_driver::plan::check_plan`] (`E7002`), reported on standard error.
fn check_plan(plan: &presk_rt::Plan, core: &presk_core::Program, sm: &SourceMap) -> Result<(), ExitCode> {
    presk_driver::plan::check_plan(plan, core, sm, &mut stderr()).map_err(exit)
}

#[derive(Clone, Copy, ValueEnum)]
enum DumpFormat {
    Text,
    Latex,
    Unicode,
    Json,
    Svg,
    Tikz,
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
    /// The fusion groups of every function (`presk-realize`): what shares a loop nest.
    #[value(alias = "sched")]
    Realize,
    /// The PSK-Kernel v0 kernels launched by an entry (needs --entry, --dim/--const as `presk run`).
    Kernel,
    /// `‖∂L/∂v‖₁` at every site of an entry, read off one reverse sweep
    /// (needs --entry, --dim/--const as `presk run`; `--data` for real
    /// batches). Ranks where a precision budget should *not* be spent.
    Sensitivity,
    /// The network diagram of an entry, as `presk explain` draws it (needs
    /// --entry, --dim/--const as `presk run`): text, or --format
    /// json|svg|tikz.
    Network,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Check { file, consts, summary, json, precision, realize, sites, model, measure } => match precision {
            Some(env) => check_precision(&file, &env, &parse_realize(&realize), sites, &model, measure),
            None => check(&file, &consts, summary, json),
        },
        Command::Dump { file, stage, format, level, boxed, memory, module, function, standalone, unroll_stack, mode, entry, dims, consts, realize, plan, profile, weights, sites, remat } => {
            set_remat(remat);
            set_mode(mode);
            if let DumpStage::Network = stage {
                return dump_network(&file, format, (level, boxed.as_deref(), memory), standalone, entry.as_deref(), &dims, &consts, &realize, plan.as_deref(), profile, weights);
            }
            match format {
                DumpFormat::Text => dump(&file, stage, unroll_stack, entry.as_deref(), &dims, &consts, &parse_plan(plan.as_deref(), &realize), plan.is_some(), profile, weights, sites),
                DumpFormat::Json | DumpFormat::Svg | DumpFormat::Tikz => {
                    eprintln!("presk dump: --format json|svg|tikz is the network diagram's: --stage network");
                    ExitCode::from(2)
                }
                DumpFormat::Latex | DumpFormat::Unicode => {
                    let f = if let DumpFormat::Latex = format { presk_print_math::Format::Latex } else { presk_print_math::Format::Unicode };
                    dump_math(&file, stage, f, &presk_print_math::Select { module, function }, standalone, &consts, unroll_stack, plan.as_deref().map(|_| parse_plan(plan.as_deref(), &realize)).as_ref())
                }
            }
        }
        Command::Run { file, entry, dims, consts, seed, bits, digits, json, verbose, result_only, unroll_stack, backend, realize, plan, opt, profile, data, samples, from, weights, threads, remat } => { set_threads(threads); set_remat(remat); run(&file, entry.as_deref(), &dims, &consts, seed, bits, digits as usize, (json, verbose, result_only), unroll_stack, backend, &parse_plan(plan.as_deref(), &realize), Opt { on: opt, profile }, data, samples, from, weights) }
        Command::Export { file, target: ExportTarget::C, mode: Some(ExportMode::Scop), out, shapes, stack, conformance_entry, train, serve, dims, consts, seed, opt, profile, entry, realize, weights, plan, threads, no_exact, straight_line, placed: _, llvm: _, remat } => {
            set_remat(remat);
            let pytorch = [("--shapes", shapes.is_some()), ("--stack", stack.is_some()), ("--conformance-entry", conformance_entry.is_some()), ("--train", train), ("--opt", opt), ("--profile", profile.is_some())];
            if let Some((flag, _)) = pytorch.iter().find(|(_, given)| *given) {
                eprintln!("presk export: `{flag}` is an option of the PyTorch export, not of --target c");
                return ExitCode::from(2);
            }
            // Point zero, one lane, one worker, and no host (plan 0.9.0 §3.7).
            let refused = [
                ("--plan", plan.is_some(), "every nest is at point zero, its default layout, before any plan (a plan's realization is `--realize @FILE`)"),
                ("--threads", threads.is_some(), "every nest runs on one worker, without OpenMP"),
                ("--straight-line", straight_line, "the nests have no host"),
                ("--serve", serve, "the nests have no host (`--entry` names a generation function to export its nests)"),
                ("--weights", weights.is_some(), "a nest's shapes do not depend on the weights, and the driver draws its inputs"),
                ("--no-exact", no_exact, "the nests are compared with the native backend, not with the exact reference"),
            ];
            if let Some((flag, _, why)) = refused.iter().find(|(_, given, _)| *given) {
                eprintln!("presk export --mode scop: `{flag}` does not apply: {why}");
                return ExitCode::from(2);
            }
            let Some(entry) = entry else {
                eprintln!("presk export --target c --mode scop: --entry is required (its nests are captured as `presk plan` captures them)");
                return ExitCode::from(2);
            };
            presk_driver::export_c::export_scop(&presk_driver::export_c::ScopOptions { file: &file, entry: &entry, dims: &dims, consts: &consts, realize: realize.as_deref().unwrap_or("f64"), seed, out: &out, remat: crate::remat() }, &mut stderr()).map_or_else(exit, |()| ExitCode::SUCCESS)
        }
        Command::Export { file, target: ExportTarget::C, mode, out, shapes, stack, conformance_entry, train, serve, dims, consts, seed, opt, profile, entry, realize, weights, plan, threads, no_exact, straight_line, placed, llvm, remat } => {
            set_remat(remat);
            if let Some(m) = mode {
                eprintln!("presk export: `--mode {}` is a mode of the PyTorch export; --target c has `--mode scop`", if m == ExportMode::Graph { "graph" } else { "readable" });
                return ExitCode::from(2);
            }
            let pytorch = [("--shapes", shapes.is_some()), ("--stack", stack.is_some()), ("--conformance-entry", conformance_entry.is_some()), ("--train", train), ("--opt", opt), ("--profile", profile.is_some())];
            set_threads(threads);
            if llvm {
                require_llvm();
            }
            export_c(&file, &out, entry.as_deref(), &dims, &consts, realize.as_deref().unwrap_or("f64"), plan.as_deref(), weights, seed, (no_exact, straight_line, placed, serve, llvm), &pytorch)
        }
        Command::Export { file, target: ExportTarget::Pytorch, mode, out, shapes, stack, conformance_entry, train, serve, dims, consts, seed, opt, profile, entry, realize, weights, plan, threads, no_exact, straight_line, placed: _, llvm: _, remat } => {
            set_remat(remat);
            let c_only = [("--mode scop", mode == Some(ExportMode::Scop)), ("--entry", entry.is_some()), ("--realize", realize.is_some()), ("--weights", weights.is_some()), ("--plan", plan.is_some()), ("--threads", threads.is_some()), ("--no-exact", no_exact), ("--straight-line", straight_line)];
            if let Some((flag, _)) = c_only.iter().find(|(_, given)| *given) {
                let entry = if *flag == "--entry" { "; the PyTorch export's entry is --conformance-entry" } else { "" };
                eprintln!("presk export: `{flag}` is an option of --target c{entry}");
                return ExitCode::from(2);
            }
            export(&file, mode.unwrap_or(ExportMode::Readable), &out, shapes.as_deref(), stack.unwrap_or(StackMode::Loop), conformance_entry.as_deref(), train, serve, &dims, &consts, seed, Opt { on: opt, profile })
        }
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
        Command::Train { file, consts, data, steps, epochs, seed, bits, log_every, ckpt, json, backend, realize, plan, opt, profile, threads, remat, host_profile } => {
            set_remat(remat);
            set_threads(threads);
            if host_profile && !matches!(backend, BackendKind::Native | BackendKind::Llvm) {
                eprintln!("presk train: --host-profile times the native backend's host: add --backend native");
                return ExitCode::from(2);
            }
            let plan = parse_plan(plan.as_deref(), &realize);
            train(&file, &consts, presk_train::TrainOptions { data, steps, epochs, seed, bits, realize: plan.realization.clone(), ckpt, checkpoints: true }, &plan, log_every, json, backend, Opt { on: opt, profile }, host_profile)
        }
        Command::Calibrate { file, precision: _, step: true, consts, data, samples, from, seed, bits, out, weights, warmup, steps, realize, profile, reference, remat, .. } => {
            set_remat(remat);
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
        Command::Calibrate { file, precision, entry, dims, consts, data, samples, from, seed, bits, out, compare, weights, steps, realize, profile, reference, remat, .. } => {
            set_remat(remat);
            if steps.is_some() || !realize.is_empty() || profile.is_some() || reference != "exact" {
                eprintln!("presk calibrate: --steps, --realize, --profile and --reference are for a training step: add --step");
                return ExitCode::from(2);
            }
            calibrate_precision(&file, precision, entry.as_deref(), &dims, &consts, data.as_deref(), samples, from, seed, bits, out.as_deref(), compare.as_deref(), weights.as_deref())
        }
        Command::Quantize { file, realize, list, entry, dims, consts, weights, backend, data, samples, from, bits, seed } => quantize(&file, &parse_realize(&realize), list, &entry, &dims, &consts, weights.as_deref(), backend, data.as_deref().map(|d| (d, from, samples)), bits, seed),
        Command::Solve { file, precision, budget, no_verify, bits, out, profile, step: true, grain, remat, .. } => {
            set_remat(remat);
            let grain = match grain {
                GrainArg::Function => presk_precision::solve::Grain::Function,
                GrainArg::Value => presk_precision::solve::Grain::Value,
            };
            step::solve(&file, &precision, budget, bits, !no_verify, out.as_deref(), profile.as_deref(), grain)
        }
        Command::Solve { file, precision, budget, budget_abs, weights, no_verify, bits, seed, out, sites, profile, quant, scheme, menu, rounding, narrow, estimate, margin, heldout, min_gain, sweep, measure, remat, .. } => {
            set_remat(remat);
            let scheme = match scheme.as_str() {
                "absmax" => presk_rt::QuantScheme::Absmax,
                "pow2" => presk_rt::QuantScheme::Pow2,
                other => {
                    eprintln!("presk solve: unknown --scheme `{other}`; expected absmax or pow2");
                    return ExitCode::from(2);
                }
            };
            let budget = match budget_abs {
                Some(eps) => presk_precision::solve::Budget::Absolute(eps),
                None => presk_precision::solve::Budget::Relative(budget),
            };
            let options = presk_precision::solve::Options {
                rounding: match rounding {
                    RoundingArg::Unit => presk_precision::solve::RoundingTerm::Unit,
                    RoundingArg::Measured => presk_precision::solve::RoundingTerm::Measured,
                },
                narrow: match narrow {
                    NarrowArg::Off => presk_precision::solve::Narrow::Off,
                    NarrowArg::Measured => presk_precision::solve::Narrow::Measured,
                    NarrowArg::Heldout => presk_precision::solve::Narrow::Heldout,
                },
                estimate: match estimate {
                    EstimateArg::Sum => presk_precision::solve::Estimate::Sum,
                    EstimateArg::Rss => presk_precision::solve::Estimate::Rss,
                },
                margin,
                heldout,
                min_gain,
            };
            if !(margin.is_finite() && margin >= 1.0) {
                eprintln!("presk solve: --margin must be a number of at least 1, not {margin}");
                return ExitCode::from(2);
            }
            if !(min_gain.is_finite() && (0.0..1.0).contains(&min_gain)) {
                eprintln!("presk solve: --min-gain is a fraction of the entry's traffic, at least 0 and below 1, not {min_gain}");
                return ExitCode::from(2);
            }
            let menu = match menu.as_deref().map(presk_precision::solve::Menu::parse).transpose() {
                Ok(m) => m.unwrap_or_default(),
                Err(m) => {
                    eprintln!("presk solve: --menu: {m}");
                    return ExitCode::from(2);
                }
            };
            if quant && !menu.1.is_empty() {
                eprintln!("presk solve: --quant quantizes in a pass after the solve, and --menu's integers in the solve itself: one or the other");
                return ExitCode::from(2);
            }
            let exec = presk_driver::precision::SolveExec {
                sweep: match sweep {
                    SweepArg::Exact => presk_driver::precision::Sweep::Exact,
                    SweepArg::F64 => presk_driver::precision::Sweep::F64,
                },
                measure: match measure {
                    MeasureArg::Kernel => presk_driver::precision::Measure::Kernel,
                    MeasureArg::Native => presk_driver::precision::Measure::Native,
                },
            };
            solve_precision(&file, &precision, budget, weights.as_deref(), !no_verify, bits, seed, out.as_deref(), sites, profile, quant.then_some(scheme), menu, options, exec)
        }
        Command::Plan { file, entry, dims, consts, realize, mode, profile, nest, space, cost, out, seed, remat } => match ra4::Space::parse(&space).inspect(|_| {
            set_remat(remat);
            set_mode(mode);
        }) {
            Ok(sp) => ra4::plan(&file, &entry, &dims, &consts, &presk_rt::Plan::from_realization(parse_realize(&realize)), &profile, &nest, sp, cost, out.as_deref(), seed),
            Err(m) => {
                eprintln!("presk plan: {m}");
                ExitCode::from(2)
            }
        },
        Command::Memory { file, entry: None, step: true, dims, consts, realize, plan, depth: _, seed, .. } => {
            let o = presk_driver::memory::StepOptions { file: &file, dims: &dims, consts: &consts, realize: realize.as_deref().unwrap_or("f64"), plan: plan.as_deref(), seed };
            match presk_driver::memory::step(&o, &mut stderr()) {
                Ok(steps) => {
                    for l in presk_driver::text::step(&steps) {
                        println!("{l}");
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => exit(e),
            }
        }
        Command::Memory { file, entry, dims, consts, realize, plan, remat, depth, seed, symbolic, .. } => {
            set_remat(remat.unwrap_or(RematArg::Store));
            let entry = entry.unwrap_or_default();
            let o = presk_driver::memory::Options { file: &file, entry: &entry, dims: &dims, consts: &consts, realize: realize.as_deref().unwrap_or("f64"), plan: plan.as_deref(), seed, remat: crate::remat() };
            match presk_driver::memory::memory(&o, &mut stderr()) {
                Ok(m) => {
                    for l in presk_driver::text::memory(&m, depth) {
                        println!("{l}");
                    }
                    if symbolic {
                        match presk_driver::memory::symbolic(&o, &m, &mut stderr()) {
                            Ok(s) => {
                                let render = |p: &presk_driver::memory::Poly| presk_print_math::core::poly(&p.terms(&s.names)).render(presk_print_math::Format::Unicode);
                                for l in presk_driver::text::symbolic(&m, &s, &render) {
                                    println!("{l}");
                                }
                            }
                            Err(e) => return exit(e),
                        }
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => exit(e),
            }
        }
        Command::Explain { file, entry, dims, consts, budget, menu, realize, plan, profile, threads, remat, opt, weights, seed, bits, envelope, no_exact, target, measure, sites, format, out } => {
            let menu = match menu.as_deref().map(presk_precision::solve::Menu::parse).transpose() {
                Ok(m) => m.unwrap_or_default(),
                Err(m) => {
                    eprintln!("presk explain: --menu: {m}");
                    return ExitCode::from(2);
                }
            };
            set_threads(threads);
            set_remat(remat);
            let (json, html) = match format.as_deref() {
                None | Some("md") => (false, false),
                Some("json") => (true, false),
                Some("html") => (true, true),
                Some(other) => {
                    eprintln!("presk explain: --format {other}: `md` (the document alone, the default), `json` (and `explain.json` beside it) or `html` (and `explain.html`, the pipeline view, too)");
                    return ExitCode::from(2);
                }
            };
            let targets = match presk_explain::Target::parse_list(&target) {
                Ok(t) => t,
                Err(m) => {
                    eprintln!("presk explain: {m}");
                    return ExitCode::from(2);
                }
            };
            let o = presk_explain::Options { file, entry, dims, consts, budget, menu, realize, plan, profile, threads, remat: crate::remat(), opt, weights, seed, bits, envelope, no_exact, targets, measure, sites, json, html, out };
            match presk_explain::explain(&o, &mut stderr()) {
                Ok(e) => {
                    let listings = if e.listings == 0 { String::new() } else { format!(", {} listing{} under listings/", e.listings, if e.listings == 1 { "" } else { "s" }) };
                    println!("presk explain: wrote {} ({} sections{listings}, {} artifact{})", e.document.display(), e.sections, e.artifacts.len(), if e.artifacts.len() == 1 { "" } else { "s" });
                    ExitCode::SUCCESS
                }
                Err(stop) => exit(stop),
            }
        }
        Command::Autotune { file, entry, dims, consts, realize, space, repeat, profile, nest, interp, top, random, shard, out, seed, remat } => {
            set_remat(remat);
            let shard = match shard.as_deref().map(|s| s.split_once('/').and_then(|(k, n)| Some((k.parse::<usize>().ok()?, n.parse::<usize>().ok()?)))) {
                None => None,
                Some(Some((k, n))) if k < n => Some((k, n)),
                Some(_) => {
                    eprintln!("presk autotune: --shard K/N with K < N");
                    return ExitCode::from(2);
                }
            };
            match ra4::Space::parse(&space) {
                Ok(sp) => ra4::autotune(&file, &entry, &dims, &consts, &presk_rt::Plan::from_realization(parse_realize(&realize)), profile.as_deref(), &nest, sp, &ra4::SweepOpts { repeat, interp, top, random, shard }, &out, seed),
                Err(m) => {
                    eprintln!("presk autotune: {m}");
                    ExitCode::from(2)
                }
            }
        }
        Command::Reprice { file, entry, day, profile, out, seed, remat } => {
            set_remat(remat);
            ra4::reprice(&file, &entry, &day, &profile, &out, seed)
        }
        Command::Rank { day1, day2, out, check, frozen, entry, at, dtype, gemm, gemm_at, statement_q } => ra4::rank(&day1, &day2, &out, &ra4::RankOpts { check, frozen, entry, at, dtype, gemm, gemm_at, statement_q }),
        Command::Refit { profile, load_term, combine, out, compare, recost, entry, rules } => match (presk_opt::profile::LoadTerm::parse(&load_term), presk_opt::profile::Combine::parse(&combine), rules.as_deref().map(presk_kernel::cost::CostRules::parse).transpose()) {
            (Ok(l), Ok(c), Ok(r)) => refit(&profile, l, c, out.as_deref(), compare, recost.as_deref().map(|f| (f, entry.as_str(), r))),
            (Err(m), _, _) | (_, Err(m), _) | (_, _, Err(m)) => {
                eprintln!("presk refit: {m}");
                ExitCode::from(2)
            }
        },
        Command::Bench { file, entry, dims, consts, bits, repeat, warmup, seed, opt, profile, calibrate, host_profile, launches, space, load_term, combine, min_workers, rules, realize, plan, threads, native_only, plan_only, llvm, weights, remat } => {
            set_remat(remat);
            set_threads(threads);
            let rules = rules.as_deref().map(presk_kernel::cost::CostRules::parse).transpose();
            let model = match (presk_opt::profile::LoadTerm::parse(&load_term), presk_opt::profile::Combine::parse(&combine), rules) {
                (Ok(l), Ok(c), Ok(r)) => (l, c, min_workers, r),
                (Err(m), _, _) | (_, Err(m), _) | (_, _, Err(m)) => {
                    eprintln!("presk bench: {m}");
                    return ExitCode::from(2);
                }
            };
            let calibrate = match (calibrate.as_deref(), space.as_deref()) {
                (Some(p), Some("calib")) => Some((p, Some(model))),
                (Some(_), Some(other)) => {
                    eprintln!("presk bench: --space {other}: the one space is `calib`");
                    return ExitCode::from(2);
                }
                (Some(p), None) => Some((p, None)),
                (None, _) => None,
            };
            bench(&file, &entry, &dims, &consts, bits, (repeat, warmup), seed, Opt { on: opt, profile }, calibrate, (host_profile, launches), &parse_plan(plan.as_deref(), &realize), plan.is_some(), (native_only, plan_only, llvm), weights.as_deref())
        }
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
    // An inference serve is no training run: its program is the inference
    // reading, with no mask (0.16.0, D160).
    let (mut core, diags) = presk_core::lower::lower_with(&prog, &typed, &presk_core::lower::Options { unroll_stack: false, mode: presk_core::inference::Mode::Inference });
    eprint!("{}", render_all(&diags, &sm, color()));
    if has_errors(&diags) {
        return ExitCode::from(1);
    }
    let mut diags = differentiate(&mut core, opts.realize.groups_visible());
    diags.extend(stored_vjps(&core, &opts.realize));
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
    if let Err(c) = opt.apply_for(&mut core, &prog, false, &serve_dims, &presk_rt::Plan::from_realization(opts.realize.clone()), &sm) {
        return c;
    }
    let on_token = |_step: u64, _tok: u64| {};
    match with_backend(backend, &core, &prog, opts.bits, &presk_rt::Plan::from_realization(opts.realize.clone()), |b| presk_serve::serve(b, &core, &prog, &fns, &opts, on_token)) {
        Err(e) => {
            let usage = e.code == "E9003";
            let d = run_diag(e.code, e.message, e.span);
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
                    "compile_s": out.compile_s,
                    "compute_s": out.compute_s,
                    "compilations": out.compilations,
                    "compiled_kernels": out.kernels,
                });
                println!("{}", serde_json::to_string_pretty(&j).unwrap());
            } else {
                let ctx = match out.window {
                    Some(w) => format!("context window {w}"),
                    None => "whole context".to_string(),
                };
                let plural = |n: u64, what: &str| format!("{n} {what}{}", if n == 1 { "" } else { "s" });
                eprintln!(
                    "presk serve: {} from {}, prompt of {} tokens, {} generated, {}, {}; compiled in {:.3} s ({}, {}), computed in {:.3} s",
                    out.model,
                    out.weights.display(),
                    out.prompt.len(),
                    out.generated.len(),
                    if out.greedy { "greedy" } else { "sampled" },
                    ctx,
                    out.compile_s,
                    plural(out.compilations, "length"),
                    plural(out.kernels, "kernel"),
                    out.compute_s
                );
                println!("{}", out.generated.iter().map(u64::to_string).collect::<Vec<_>>().join(" "));
            }
            ExitCode::SUCCESS
        }
    }
}

/// `presk solve`: a budget in, a realization out (plan 0.5.0 §3.4).
#[allow(clippy::too_many_arguments)]
fn solve_precision(file: &Path, env_path: &Path, budget: presk_precision::solve::Budget, weights: Option<&Path>, verify: bool, bits: u32, seed: u64, out: Option<&Path>, list: usize, profile: Option<PathBuf>, quant: Option<presk_rt::QuantScheme>, menu: presk_precision::solve::Menu, options: presk_precision::solve::Options, exec: presk_driver::precision::SolveExec) -> ExitCode {
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
    // Its candidates are `bf16` too: the groups kept.
    let diags = differentiate(&mut core, true);
    eprint!("{}", render_all(&diags, &sm, color()));
    if has_errors(&diags) {
        return ExitCode::from(1);
    }
    let args = presk_driver::precision::SolveArgs { budget, weights, verify, bits, seed, profile: profile.as_deref(), quant, menu, options, exec };
    let solved = match presk_driver::precision::solve(&mut core, &prog, &env, &args) {
        Ok(s) => s,
        Err(e) => {
            use presk_driver::precision::SolveError as E;
            let (message, code) = match e {
                E::NotEntry => (format!("presk solve: `{}` is not an entry of {}", env.entry, file.display()), 1),
                E::NotScalar => (format!("presk solve: `{}` must return a real tensor, since the sites are ranked by its sensitivity `∂y/∂v`", env.entry), 2),
                E::Inputs(m) => (format!("presk solve: cannot bind the inputs of `{}`: {m}", env.entry), 1),
                E::NoGradient(m) => (format!("presk solve: the objective does not differentiate: {m}"), 1),
                E::Reference(m) => (format!("presk solve: {m}"), 1),
                E::QuantWithoutGradient => ("presk solve: --quant needs the parameters' gradient, and the reverse sweep gave none".to_string(), 1),
                E::NoHeldout(m) => (format!("presk solve: --narrow heldout needs samples after the envelope's: {m}"), 2),
            };
            eprintln!("{message}");
            return ExitCode::from(code);
        }
    };
    for line in presk_driver::text::solve(&env, budget, &solved, list) {
        println!("{line}");
    }
    let sol = &solved.solution;
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
    let mut diags = differentiate(&mut core, realize.groups_visible());
    diags.extend(stored_vjps(&core, &realize));
    eprint!("{}", render_all(&diags, &sm, color()));
    if has_errors(&diags) {
        return ExitCode::from(1);
    }
    let fail = |e: presk_rt::RtError| {
        eprint!("{}", render_all(&[presk_syntax::Diagnostic::error(e.code, format!("presk quantize: {}", e.message))], &sm, color()));
        ExitCode::from(1)
    };
    let opts = presk_interp::RunOptions { entry: entry.to_string(), dims, seed, bits, digits: 17 };
    // With a checkpoint, the parameters are described and not drawn (D180).
    let given = weights.is_some();
    let (func, fdims, mut args, described) = match data {
        Some((path, from, samples)) => {
            let made = if given { presk_train::args_on_data_given_params(&core, &prog, &opts, path, from, samples) } else { presk_train::args_on_data(&core, &prog, &opts, path, from, samples).map(|a| (a, None)) };
            match made {
                Ok(((func, fdims, _, args), described)) => (func, fdims, args, described),
                Err(e) => return fail(presk_rt::RtError { code: e.code, message: e.message, span: e.span }),
            }
        }
        None => {
            let made = if given { presk_interp::run::prepare_run_given_params(&core, &prog, &opts) } else { presk_interp::run::prepare_run(&core, &prog, &opts).map(|(f, d, i)| (f, d, i, None)) };
            let (func, fdims, inputs, described) = match made {
                Ok(x) => x,
                Err(e) => return fail(e.into()),
            };
            match inputs.iter().map(|(_, v)| presk_rt::Value::from_interp(v)).collect() {
                Ok(a) => (func, fdims, a, described),
                Err(e) => return fail(e),
            }
        }
    };
    if let (Some(path), Some(s)) = (weights, &described) {
        match presk_driver::entry::read_given_params(&core, &prog, s, path) {
            Ok(v) => args[0] = v,
            Err(e) => return fail(e),
        }
    }
    if !matches!(args.first(), Some(presk_rt::Value::Record(_))) {
        eprintln!("presk quantize: the first argument of `{entry}` is not a parameter tree");
        return ExitCode::from(2);
    }
    match weights {
        Some(_) if described.is_some() => {}
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
        let mut b = presk_rt::InterpBackend::new(&core, &prog, bits);
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
    let machine = with_backend(backend, &core, &prog, bits, &presk_rt::Plan::from_realization(realize.clone()), |b| {
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

use presk_driver::gap::{leaf_gap, leaf_max, param_leaf};

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
    // No realization runs a gradient here: the groups kept, as by default.
    let diags = differentiate(&mut core, true);
    eprint!("{}", render_all(&diags, &sm, color()));
    if has_errors(&diags) {
        return ExitCode::from(1);
    }
    let Some(entry) = entry else {
        eprintln!("presk dump --stage sensitivity: --entry is required");
        return ExitCode::from(2);
    };
    let s = match presk_driver::precision::sensitivity(&mut core, &prog, &sm, entry, dims_p, seed, bits, weights, &mut stderr()) {
        Ok(s) => s,
        Err(e) => {
            use presk_driver::precision::SensitivityError as E;
            match e {
                E::NotEntry => eprintln!("presk dump: `{entry}` is not an entry of {}", file.display()),
                E::NotScalar => {
                    eprintln!("presk dump --stage sensitivity: `{entry}` must return a scalar `ℝ[]`, since a sensitivity is `∂L/∂v`");
                    return ExitCode::from(2);
                }
                E::Diagnostics => {}
                E::Inputs => eprintln!("presk dump: cannot bind the inputs of `{entry}`"),
                E::Run(m) => eprintln!("presk dump: {m}"),
            }
            return ExitCode::from(1);
        }
    };
    for line in presk_driver::text::sensitivity(&s, entry, bits, top) {
        println!("{line}");
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
    let mut diags = differentiate(&mut core, realize.groups_visible());
    diags.extend(stored_vjps(&core, &realize));
    eprint!("{}", render_all(&diags, &sm, color()));
    if has_errors(&diags) {
        return ExitCode::from(1);
    }
    // The bound at the envelope's shapes, a module entry's own arguments
    // among them (D108, finding 1), and the gap measured beside it.
    let bounded = match presk_driver::precision::bound(&core, &prog, &env, realize, &models, measure) {
        Ok(b) => b,
        Err(presk_driver::precision::BoundError::NotEntry) => {
            eprintln!("presk check: `{}` is not an entry of {}", env.entry, file.display());
            return ExitCode::from(1);
        }
        Err(presk_driver::precision::BoundError::Measure(m)) => {
            eprintln!("presk check: {m}");
            return ExitCode::from(1);
        }
    };
    for line in presk_driver::text::bound(&env, &env_path.display().to_string(), &realize.label(), &bounded, sites) {
        println!("{line}");
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
    // No realization runs a gradient here: the groups kept, as by default.
    let diags = differentiate(&mut core, true);
    eprint!("{}", render_all(&diags, &sm, color()));
    if has_errors(&diags) {
        return ExitCode::from(1);
    }
    let Some(entry) = entry else {
        eprintln!("presk calibrate: --entry is required");
        return ExitCode::from(2);
    };
    // `--samples` counts the samples read from `--data`; without it the
    // entry runs on the one input `--seed` synthesizes, which an envelope
    // must not call N samples (D108, finding 2).
    if samples.is_some() && data.is_none() {
        eprintln!("presk calibrate: --samples counts the samples read from --data; without --data the entry runs on the one input --seed synthesizes (bind a batch axis with --dim to measure several rows)");
        return ExitCode::from(2);
    }
    let c = presk_driver::precision::Calibrate { file, entry, dims, consts, data, samples, from, seed, bits, weights };
    let env = match presk_driver::precision::calibrate(&core, &prog, &c) {
        Ok(e) => e,
        Err(m) => {
            eprintln!("presk calibrate: {m}");
            return ExitCode::from(1);
        }
    };
    for line in presk_driver::text::calibrate(&env, entry, bits) {
        println!("{line}");
    }
    for line in presk_driver::text::rounding(&env) {
        println!("{line}");
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

/// `presk bench`: one entry timed on every backend.
#[allow(clippy::too_many_arguments)]
/// `calibrate`: the profile to write, and for the layout cost model
/// (`--space calib`) its load term, its combination and the fewest workers
/// a layout of the space is stated on.
#[allow(clippy::type_complexity)]
fn bench(file: &Path, entry: &str, dims: &[String], consts: &[String], bits: u32, (repeat, warmup): (usize, usize), seed: u64, opt: Opt, calibrate: Option<(&Path, Option<(presk_opt::profile::LoadTerm, presk_opt::profile::Combine, usize, Option<presk_kernel::cost::CostRules>)>)>, (host_profile, launches): (bool, bool), plan: &presk_rt::Plan, planned: bool, (native_only, plan_only, llvm): (bool, bool, bool), weights: Option<&Path>) -> ExitCode {
    let realize = &plan.realization;
    let dims = match parse_dims(dims) {
        Ok(d) => d,
        Err(c) => return c,
    };
    let (sm, prog, mut core) = match core(file, consts, false) {
        Err(c) => return c,
        Ok(x) => x,
    };
    // Its rows time `bf16` too, unless the plan's is the only one.
    let mut diags = differentiate(&mut core, if calibrate.is_some() || plan_only || host_profile { realize.groups_visible() } else { true });
    diags.extend(stored_vjps(&core, realize));
    eprint!("{}", render_all(&diags, &sm, color()));
    if has_errors(&diags) {
        return ExitCode::from(1);
    }
    if let Err(c) = opt.apply_for(&mut core, &prog, false, &dims, plan, &sm) {
        return c;
    }
    if let Err(c) = check_plan(plan, &core, &sm) {
        return c;
    }
    let opts = presk_interp::RunOptions { entry: entry.to_string(), dims, seed, bits, digits: 12 };
    // With a checkpoint, the parameters are described and not drawn (D180).
    let made = if weights.is_some() { presk_interp::run::prepare_run_given_params(&core, &prog, &opts) } else { presk_interp::run::prepare_run(&core, &prog, &opts).map(|(f, d, i)| (f, d, i, None)) };
    let (func, fdims, inputs, described) = match made {
        Ok(x) => x,
        Err(e) => {
            eprint!("{}", render_all(&[interp_diag(e)], &sm, color()));
            return ExitCode::from(1);
        }
    };
    let mut args: Vec<presk_rt::Value> = inputs.iter().map(|(_, v)| presk_rt::Value::from_interp(v).unwrap()).collect();
    if let Some(w) = weights {
        let read = match &described {
            Some(s) => Some(presk_driver::entry::read_given_params(&core, &prog, s, w)),
            None => args.first().map(|a| presk_rt::load_params(w, a)),
        };
        match read {
            Some(Ok(v)) => args[0] = v,
            Some(Err(e)) => {
                eprint!("{}", render_all(&[presk_syntax::Diagnostic::error(e.code, format!("{}: {}", w.display(), e.message))], &sm, color()));
                return ExitCode::from(1);
            }
            None => {
                eprintln!("presk bench: --weights: `{entry}` takes no argument");
                return ExitCode::from(2);
            }
        }
    }
    if let Some((path, layouts)) = calibrate {
        if let Some(model) = layouts {
            return calibrate_layouts(&core, &prog, func, &fdims, &args, plan, repeat.max(3), path, model);
        }
        return calibrate_profile(&core, &prog, func, &fdims, &args, plan, repeat.max(3), path, plan_only, llvm);
    }
    if host_profile {
        return bench_host_profile(&core, &prog, (entry, func, &fdims), &args, (plan, planned), (repeat, warmup), launches);
    }
    let time = |name: &str, b: &mut dyn presk_rt::Backend, realize: &presk_rt::Realization| -> Option<(String, f64, f64)> {
        match presk_driver::bench::time_call(b, func, &fdims, realize, &args, warmup, repeat) {
            Ok((median, best)) => Some((name.to_string(), median, best)),
            Err(presk_driver::bench::TimeError::Prepare(_)) => None,
            Err(presk_driver::bench::TimeError::Run(m)) => {
                eprintln!("presk bench: {name}: {m}");
                None
            }
        }
    };
    let uniform = |d| presk_rt::Realization::uniform(d);
    let f64r = uniform(presk_rt::Dtype::F64);
    let side = if llvm { "llvm" } else { "native" };
    let mut rows = Vec::new();
    if !native_only && !plan_only {
        if let Some(r) = time(&format!("interp {bits} bits"), &mut presk_rt::InterpBackend::new(&core, &prog, bits), &f64r) {
            rows.push(r);
        }
        if let Some(r) = time("kernel f64", &mut presk_kernel::KernelBackend::new(&core, &prog, &f64r), &f64r) {
            rows.push(r);
        }
    }
    let mut want: Vec<(String, presk_rt::Realization)> =
        [presk_rt::Dtype::F64, presk_rt::Dtype::F32, presk_rt::Dtype::Bf16].iter().map(|d| (format!("{side} {d}"), uniform(*d))).collect();
    // A realization given on the command line is timed too, whatever it is:
    // a solved assignment or a declared reduction order is exactly the thing
    // one wants a row for.
    if !realize.is_uniform() || !want.iter().any(|(_, r)| r == realize) {
        want.push((format!("{side} {}", realize.label()), realize.clone()));
    }
    if plan_only {
        want.clear();
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
            match compiled_backend(llvm, &core, &prog, r) {
                Ok(mut b) => {
                    b.set_default_lanes(n);
                    if let Some(row) = time(&format!("{name}{suffix}"), &mut b, r) {
                        rows.push(row);
                    }
                }
                Err(e) => eprintln!("presk bench: native backend unavailable: {}", e.message),
            }
        }
    }
    // A plan is timed as it states, on the native backend; its realization's
    // row above runs the default layout, so the two rows differ by the
    // layout alone.
    if planned {
        match compiled_backend(llvm, &core, &prog, realize) {
            Ok(mut b) => {
                b.set_plan(plan.clone());
                if let Some(row) = time(&format!("{side} {} plan", realize.label()), &mut b, realize) {
                    rows.push(row);
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

/// A layout cost model's constants, as `presk bench --calibrate --space
/// calib` and `presk refit` print them.
fn print_layout_fit(f: &presk_opt::profile::LayoutFit) {
    let rules = if f.rules.is_none() { String::new() } else { format!(", rules {}", f.rules.name()) };
    println!("-- fit ({}, {}{rules}): λ = {:.2} µs, spawn = {:.2} µs, β_L2 = {:.3e}, β_L3 = {:.3e}, β_mem = {:.3e} bytes/s", f.combine.name(), if f.rho.is_empty() { "no load term" } else { "load term" }, f.lambda * 1e6, f.spawn * 1e6, f.beta_l2, f.beta_l3, f.beta_mem);
    for (g, pi) in &f.pi {
        println!("--      π[{g}] = {pi:.3e} ops/s");
    }
    for (g, rho) in &f.rho {
        println!("--      ρ[{g}] = {rho:.3e} loads/s");
    }
    println!("-- uncertainty u = {:.3} (median relative residual over {} samples), Spearman {:.3}", f.u, f.samples.len(), f.spearman);
}

/// `presk refit` (plan 0.8.0 step 3): the layout cost model fitted again on
/// the samples a profile holds, under another load term or combination, so
/// that every form is fitted on the same calibration. Samples recorded
/// before 0.8.0 step 2 have no load count and are left out, as a
/// calibration leaves them out.
///
/// With `recost` (0.8.0 step 4, 4b), every sample is costed again first —
/// its layout lowered at its product on the program given, costed on the
/// profile's machine description under the rules given — its time kept, so
/// that a rule of the planned cost is tried on the definitive calibration
/// without measuring it again. The fit then records those rules.
#[allow(clippy::type_complexity)]
fn refit(path: &Path, load: presk_opt::profile::LoadTerm, combine: presk_opt::profile::Combine, out: Option<&Path>, compare: bool, recost: Option<(&Path, &str, Option<presk_kernel::cost::CostRules>)>) -> ExitCode {
    use presk_opt::profile::{Combine, LoadTerm, fit_layout};
    let mut profile = match presk_opt::Profile::load(path) {
        Ok(p) => p,
        Err(m) => {
            eprintln!("presk refit: {m}");
            return ExitCode::from(2);
        }
    };
    let Some(layout) = profile.layout.take() else {
        eprintln!("presk refit: {} has no layout cost model: calibrate it with `presk bench --calibrate {} --space calib`", path.display(), path.display());
        return ExitCode::from(2);
    };
    let mut samples = layout.samples;
    let mut rules = layout.rules;
    if let Some((file, entry, r)) = recost {
        // Under the rules given, or the profile's own.
        let r = r.unwrap_or(rules);
        let Some(mut machine) = profile.machine_desc.clone() else {
            eprintln!("presk refit --recost: {} has no machine description to cost on", path.display());
            return ExitCode::from(2);
        };
        if r.lines && machine.line == 0 {
            // A description recorded before 0.8.0 step 4 has no line size.
            let Some(line) = presk_kernel::cost::Machine::line_of(&machine.source) else {
                eprintln!("presk refit --recost: {}'s machine description has no line size, and the operating system gives none for {}", path.display(), machine.source);
                return ExitCode::from(2);
            };
            println!("-- {}'s machine description has no line size: {line} bytes, as the operating system gives {}'s", path.display(), machine.source);
            machine.line = line;
        }
        let t0 = std::time::Instant::now();
        let fresh = match ra4::recost(&samples, file, entry, &machine, &r) {
            Ok(f) => f,
            Err(m) => {
                eprintln!("presk refit --recost: {m}");
                return ExitCode::from(2);
            }
        };
        let moved = |f: &dyn Fn(&presk_kernel::cost::PlannedCost) -> Vec<f64>| samples.iter().zip(&fresh).filter(|(a, b)| f(&a.cost) != f(&b.cost)).count();
        let same = samples.iter().zip(&fresh).filter(|(a, b)| a.cost == b.cost).count();
        println!(
            "-- {} samples re-costed under the rules {} in {:.1} s, times kept: {same} costs identical, {} not (work {}, traffic {}, loads {}, spill {})",
            fresh.len(),
            r.name(),
            t0.elapsed().as_secs_f64(),
            fresh.len() - same,
            moved(&|c| vec![c.work]),
            moved(&|c| c.traffic.to_vec()),
            moved(&|c| vec![c.loads]),
            moved(&|c| vec![c.spill, c.regs as f64, c.spilled as f64]),
        );
        (samples, rules) = (fresh, r);
        profile.machine_desc = Some(machine);
    }
    let before = samples.len();
    samples.retain(|s| s.cost.loads > 0.0);
    if samples.len() < before {
        eprintln!("presk refit: {} samples recorded before 0.8.0 step 2 left out: they have no load count", before - samples.len());
    }
    if samples.is_empty() {
        eprintln!("presk refit: {} has no sample to fit", path.display());
        return ExitCode::from(2);
    }
    if compare {
        println!("{:<8} {:<16} {:>6} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10} {:>8}", "load", "combination", "u", "Spearman", "λ µs", "spawn µs", "β_L2", "β_L3", "β_mem", "fit s");
        for load in [LoadTerm::None, LoadTerm::Group, LoadTerm::Machine] {
            for combine in [Combine::Max, Combine::Sum, Combine::Pnorm(1.0)] {
                let t0 = std::time::Instant::now();
                let f = fit_layout(samples.clone(), load, combine);
                let name = match load {
                    LoadTerm::None => "none",
                    LoadTerm::Group => "group",
                    LoadTerm::Machine => "machine",
                };
                println!("{:<8} {:<16} {:>6.3} {:>8.3} {:>8.2} {:>10.2} {:>10.3e} {:>10.3e} {:>10.3e} {:>8.1}", name, f.combine.name(), f.u, f.spearman, f.lambda * 1e6, f.spawn * 1e6, f.beta_l2, f.beta_l3, f.beta_mem, t0.elapsed().as_secs_f64());
            }
        }
        println!("-- {} samples of {}; nothing written", samples.len(), path.display());
        return ExitCode::SUCCESS;
    }
    let mut fitted = fit_layout(samples, load, combine);
    fitted.rules = rules;
    print_layout_fit(&fitted);
    profile.layout = Some(fitted);
    let to = out.unwrap_or(path);
    match profile.save(to) {
        Ok(()) => {
            println!("-- profile written to {}", to.display());
            ExitCode::SUCCESS
        }
        Err(m) => {
            eprintln!("presk refit: {m}");
            ExitCode::from(2)
        }
    }
}

/// `presk bench --host-profile` (plan 0.10.0 step 5): the native backend's
/// host timed by phase, under the plan, or the realization's default layout
/// without one. Each call is timed as `presk bench` times it, the copy of
/// its arguments and the drop of its result included; `repeat` calls with
/// the timers and `repeat` without, alternated, after at least one warm-up
/// call (the first compiles); then one call of each kind, their outputs
/// compared bit for bit. `launches`: the timed calls list each launch too
/// (plan 0.17.2 step 4, D200).
#[allow(clippy::type_complexity)]
fn bench_host_profile(core: &presk_core::Program, prog: &elab::ElabProgram, (entry, func, fdims): (&str, presk_core::FuncId, &presk_interp::eval::Dims), args: &[presk_rt::Value], (plan, planned): (&presk_rt::Plan, bool), (repeat, warmup): (usize, usize), launches: bool) -> ExitCode {
    match presk_driver::bench::host_profile(core, prog, func, fdims, args, plan, planned, repeat, warmup, launches) {
        Ok(h) => {
            print!("{}", presk_driver::text::host_profile(entry, &plan.realization.label(), planned, &h));
            if h.same { ExitCode::SUCCESS } else { ExitCode::from(1) }
        }
        Err(m) => {
            eprintln!("presk bench: {m}");
            ExitCode::from(1)
        }
    }
}

/// `presk bench --calibrate`: time every kernel of the entry natively, fit
/// the roofline and write (or extend) the profile. `plan_only`: the plan's
/// kernels alone, the profile without the other realizations' fits.
#[allow(clippy::too_many_arguments)]
fn calibrate_profile(core: &presk_core::Program, prog: &elab::ElabProgram, func: presk_core::FuncId, fdims: &presk_interp::eval::Dims, args: &[presk_rt::Value], plan: &presk_rt::Plan, repeat: usize, path: &Path, plan_only: bool, llvm: bool) -> ExitCode {
    let timed = if llvm {
        require_llvm();
        presk_llvm::LlvmCompiler::new().and_then(|c| presk_cranelift::NativeExec::with_compiler(Box::new(c))).map_err(|e| format!("the LLVM backend is unavailable: {}", e.message)).and_then(|x| presk_driver::bench::kernel_times_with(Box::new(x), core, prog, func, fdims, args, plan, repeat))
    } else {
        presk_driver::bench::kernel_times(core, prog, func, fdims, args, plan, repeat)
    };
    let times = match timed {
        Ok(t) => t,
        Err(m) => {
            eprintln!("presk bench: {m}");
            return ExitCode::from(1);
        }
    };
    // The kernels' share of a call: each kernel's median launch times its
    // launches a call, so that a first launch's compilation is not counted;
    // the call's own time (`presk bench` without --calibrate) less this is
    // the host's (the PyTorch baseline at 0.9.0, D87).
    let (per_call_launches, per_call_seconds) = (times.per_call_launches(), times.per_call_seconds());
    let dtype = times.label.clone();
    let mut kernels: Vec<presk_opt::profile::Sample> = Vec::new();
    for (k, t) in times.kernels.iter().zip(times.medians.iter().copied()) {
        let Some(seconds) = t else { continue };
        let c = presk_kernel::cost::kernel_cost(k);
        let signature = k.to_string().lines().next().unwrap_or("").trim_end_matches(" {").to_string();
        kernels.push(presk_opt::profile::Sample { signature, name: k.name.clone(), work: c.work, bytes: c.bytes, seconds });
    }
    // Merge with an existing profile of the same realization: new measurements
    // win. The layout cost model's fit and machine description, if the file
    // has them (`--space calib`), are kept whatever the realization.
    let old = presk_opt::Profile::load(path).ok();
    let (kept_layout, kept_machine) = old.as_ref().map(|o| (o.layout.clone(), o.machine_desc.clone())).unwrap_or((None, None));
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
    (profile.layout, profile.machine_desc) = (kept_layout, kept_machine);
    // The same kernels timed under the *other* realizations of this machine,
    // so one profile can price a mixed assignment (0.6.0 step 3). A narrow
    // type's π is not a wide type's — that is the whole reason lanes exist —
    // and a cost model meant to compare machine types cannot have one π.
    // Under `--plan-only` the plan's kernels are all that is asked: timing
    // them again under every other realization's default layout takes most
    // of the run on one thread (plan 0.17.0 step 1).
    let others: &[presk_rt::Dtype] = if plan_only { &[] } else { &[presk_rt::Dtype::F64, presk_rt::Dtype::F32, presk_rt::Dtype::Bf16] };
    for &d in others {
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
            let mut b = presk_kernel::KernelBackend::with_exec(core, prog, &r, Box::new(timed));
            b.set_default_lanes(lanes);
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
    println!("-- a call: {per_call_launches} launches, the kernels' time {:.3} ms (each kernel's median launch times its launches)", per_call_seconds * 1e3);
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

/// `presk bench --calibrate FILE --space calib` (0.7.0 step 4): the entry's
/// one contraction — a GEMM — timed natively under a set of layouts that
/// spans the layout cost model's terms (two register tiles, no k-block and a
/// k-block of 128 in every loop order, one lane and the type's width, one
/// worker, half the machine's and all of it), each kernel's planned cost
/// computed on the machine's description, and the model fitted on every
/// sample the profile holds, this run's and earlier ones (other sizes, other
/// types). The fit's median relative residual is the model's uncertainty `u`.
#[allow(clippy::too_many_arguments)]
fn calibrate_layouts(core: &presk_core::Program, prog: &elab::ElabProgram, func: presk_core::FuncId, fdims: &presk_interp::eval::Dims, args: &[presk_rt::Value], plan: &presk_rt::Plan, repeat: usize, path: &Path, (load, combine, min_workers, rules): (presk_opt::profile::LoadTerm, presk_opt::profile::Combine, usize, Option<presk_kernel::cost::CostRules>)) -> ExitCode {
    use presk_rt::layout::{BlockLoop, Form, KBlock, Split};
    let machine = presk_kernel::cost::Machine::detect();
    let realize = &plan.realization;
    let dtype = realize.default;
    // The inputs rounded to the machine type once, as the backend would round
    // them at every call, outside the kernels it times: at 2048 that was
    // most of a calibration's wall time (0.8.0 step 3, D67).
    let args: Vec<presk_rt::Value> = args.iter().map(|v| v.realize(dtype)).collect();
    let args = args.as_slice();
    let f = core.func(func);
    let nests: Vec<String> = f
        .body
        .lets
        .iter()
        .filter(|l| matches!(&l.rhs, presk_core::ir::Rhs::Prim { op: presk_core::PrimOp::Dot | presk_core::PrimOp::Matmul | presk_core::PrimOp::Ein { .. }, .. }))
        .map(|l| format!("{}/%{}", f.name, l.var.0))
        .collect();
    let [nest] = nests.as_slice() else {
        eprintln!("presk bench --space calib: the entry must have one contraction (a GEMM); `{}` has {}", f.name, nests.len());
        return ExitCode::from(2);
    };
    let pm = machine.workers;
    let mut workers = vec![1, pm / 2, pm];
    workers.retain(|w| *w >= 1);
    workers.dedup();
    // `--min-workers`: at 2048 the one-worker layouts are left out, three
    // quarters of that size's time (0.8.0 step 4, D67).
    workers.retain(|w| *w >= min_workers);
    if workers.is_empty() {
        eprintln!("presk bench --space calib: --min-workers {min_workers} leaves no layout on {pm} workers");
        return ExitCode::from(2);
    }
    let mut lanes = vec![1, dtype.natural_lanes()];
    lanes.dedup();
    let orders3: [[BlockLoop; 3]; 6] = [
        [BlockLoop::I, BlockLoop::J, BlockLoop::K],
        [BlockLoop::I, BlockLoop::K, BlockLoop::J],
        [BlockLoop::J, BlockLoop::I, BlockLoop::K],
        [BlockLoop::J, BlockLoop::K, BlockLoop::I],
        [BlockLoop::K, BlockLoop::I, BlockLoop::J],
        [BlockLoop::K, BlockLoop::J, BlockLoop::I],
    ];
    // Plan 0.8.0 §3.3: no k-block in the orders IJ and JI, a k-block of 128
    // in every order of IJK, and of 32 in IJK and KIJ.
    let mut blocks: Vec<(KBlock, Vec<BlockLoop>)> = vec![(KBlock::None, vec![BlockLoop::I, BlockLoop::J]), (KBlock::None, vec![BlockLoop::J, BlockLoop::I])];
    blocks.extend(orders3.iter().map(|o| (KBlock::Width(128), o.to_vec())));
    blocks.extend([&orders3[0], &orders3[4]].iter().map(|o| (KBlock::Width(32), o.to_vec())));
    println!("-- {} on {} ({}): L1d {} KiB, L2 {} KiB, L3 {} KiB shared by {}, {} workers, {} vector registers", nest, machine.source, realize.label(), machine.l1 >> 10, machine.l2 >> 10, machine.l3 >> 10, machine.l3_sharing, pm, machine.vector_registers);
    if min_workers > 1 {
        println!("-- layouts on {} workers only (--min-workers {min_workers})", workers.iter().map(|w| w.to_string()).collect::<Vec<_>>().join(", "));
    }
    // The rules of the planned cost (0.8.0 step 4, D70): those given, or
    // those the profile's samples were costed under, which a run that keeps
    // them must share.
    let kept = presk_opt::Profile::load(path).ok().and_then(|p| p.layout).filter(|l| !l.samples.is_empty()).map(|l| l.rules);
    let rules = match (rules, kept) {
        (Some(r), Some(k)) if r != k => {
            eprintln!("presk bench --space calib: {}'s samples were costed under the rules {}, not {}: re-cost them first (`presk refit --recost`), or calibrate into a new profile", path.display(), k.name(), r.name());
            return ExitCode::from(2);
        }
        // A new profile is costed under the frozen model's rules (D71).
        (r, k) => r.or(k).unwrap_or(presk_kernel::cost::CostRules { lines: true, overhead: true, ..Default::default() }),
    };
    if !rules.is_none() {
        println!("-- costed under the rules {}", rules.name());
    }
    let mut fresh: Vec<presk_opt::profile::PlannedSample> = Vec::new();
    let mut roofline: Vec<presk_opt::profile::Sample> = Vec::new();
    let mut skipped = 0usize;
    // Five register tiles, so that the load rate is told apart from π: their
    // loads per multiply-add span 2 to 0.3125 in four lanes (plan 0.8.0 §3.3).
    for tile in [(1usize, 4usize), (1, 8), (2, 8), (2, 16), (4, 16)] {
        for (kblock, order) in &blocks {
            for l in &lanes {
                for w in &workers {
                    let depth = if order[0] == BlockLoop::K { 1 } else { 0 };
                    let split = if *w == 1 { Split::None } else { Split::At { depth, workers: *w } };
                    let layout = presk_rt::Layout { form: Some(Form::Tiled), tile: Some(tile), kblock: Some(*kblock), order: Some(order.clone()), lanes: Some(*l), split: Some(split), ..Default::default() };
                    let p = presk_rt::Plan { realization: realize.clone(), realization_ref: None, profile: None, layout: std::collections::BTreeMap::from([(nest.clone(), layout.clone())]) };
                    let Ok(native) = presk_cranelift::NativeExec::new() else {
                        eprintln!("presk bench: native backend unavailable");
                        return ExitCode::from(1);
                    };
                    let (timed, samples) = presk_kernel::cost::TimedExec::new(Box::new(native));
                    let mut b = presk_kernel::KernelBackend::with_exec(core, prog, realize, Box::new(timed));
                    b.set_plan(p);
                    let Ok(prepared) = presk_rt::Backend::prepare(&mut b, func, fdims, realize) else { continue };
                    let mut failed = None;
                    for _ in 0..=repeat {
                        if let Err(e) = presk_rt::Backend::call(&mut b, &prepared, args.to_vec()) {
                            failed = Some(e.message);
                            break;
                        }
                    }
                    if let Some(m) = failed {
                        // A layout ill-formed at these shapes (a tile or k-block
                        // that does not divide): not a point of this space.
                        eprintln!("presk bench: {layout}: {m}");
                        skipped += 1;
                        continue;
                    }
                    let medians = presk_kernel::cost::median_per_kernel(&samples.borrow(), b.kernels.len());
                    let Some((k, seconds)) = b.kernels.iter().zip(medians).find(|(k, _)| k.name == "dot" || k.name.starts_with("ein[")).and_then(|(k, t)| t.map(|t| (k.clone(), t))) else { continue };
                    let cost = presk_kernel::cost::planned_cost_under(&k, &machine, &rules);
                    let signature = k.to_string().lines().next().unwrap_or("").trim_end_matches(" {").to_string();
                    let kc = presk_kernel::cost::kernel_cost(&k);
                    roofline.push(presk_opt::profile::Sample { signature: signature.clone(), name: k.name.clone(), work: kc.work, bytes: kc.bytes, seconds });
                    // The group is the plan's decision: the stated workers,
                    // whatever share of them a short loop keeps busy (which
                    // the planned cost already counts).
                    let stated = match k.threads {
                        presk_kernel::Threads::At { workers, .. } => workers,
                        presk_kernel::Threads::One => 1,
                        presk_kernel::Threads::Default => cost.workers,
                    };
                    fresh.push(presk_opt::profile::PlannedSample { signature, layout: layout.key(), group: presk_opt::profile::group_key(dtype.name(), k.lanes, stated), cost, seconds });
                }
            }
        }
    }
    if fresh.is_empty() {
        eprintln!("presk bench --space calib: no layout could be timed");
        return ExitCode::from(1);
    }
    // Earlier samples of the profile stay, except those this run measured again.
    let machine_label = format!("{} {}", std::env::consts::OS, std::env::consts::ARCH);
    let mut profile = match presk_opt::Profile::load(path) {
        Ok(p) => p,
        Err(_) => presk_opt::profile::fit(&roofline, &realize.label(), &machine_label, &|s| presk_opt::cost::weight_of_name(&s.name)),
    };
    let mut all: Vec<presk_opt::profile::PlannedSample> = profile.layout.take().map(|l| l.samples).unwrap_or_default();
    all.retain(|o| !fresh.iter().any(|n| n.signature == o.signature && n.layout == o.layout));
    // A contraction always loads: a sample without a load count was recorded
    // before 0.8.0 step 2, its spilled registers inside its work, and does
    // not mix with the others.
    let before = all.len();
    all.retain(|o| o.cost.loads > 0.0);
    if all.len() < before {
        eprintln!("presk bench: {} samples of {} recorded before 0.8.0 step 2 dropped: they have no load count", before - all.len(), path.display());
    }
    let kept = all.len();
    all.extend(fresh.iter().cloned());
    let mut fitted = presk_opt::profile::fit_layout(all, load, combine);
    fitted.rules = rules;
    println!("{:<64} {:<18} {:>12} {:>12}", "layout", "group", "measured µs", "predicted µs");
    for s in &fresh {
        let predicted = fitted.predict(&s.cost, &s.group).unwrap_or(f64::NAN);
        println!("{:<64} {:<18} {:>12.1} {:>12.1}", s.layout, s.group, s.seconds * 1e6, predicted * 1e6);
    }
    println!("-- {} layouts timed ({skipped} ill-formed at these shapes), {kept} samples kept from {}, {} calls each", fresh.len(), path.display(), repeat);
    print_layout_fit(&fitted);
    profile.machine_desc = Some(machine);
    profile.layout = Some(fitted);
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
#[allow(clippy::too_many_arguments)]
fn train(file: &Path, consts: &[String], mut opts: presk_train::TrainOptions, plan: &presk_rt::Plan, log_every: u64, json: bool, backend: BackendKind, opt: Opt, host_profile: bool) -> ExitCode {
    let (sm, prog, typed) = match typed_front(file, consts) {
        Err(c) => return c,
        Ok(x) => x,
    };
    if typed.training.is_empty() {
        eprintln!("presk train: {} has no `training run`", file.display());
        return ExitCode::from(2);
    }
    // A training run's program is as written, its masks drawn (0.16.0,
    // D160); under a realization solved outside training, which types no
    // mask, each takes its product's operand's type.
    let (mut core, diags) = presk_core::lower(&prog, &typed);
    eprint!("{}", render_all(&diags, &sm, color()));
    if has_errors(&diags) {
        return ExitCode::from(1);
    }
    let typed_masks = presk_train::masks_typed(plan, &core);
    let plan = &typed_masks;
    let diags = differentiate(&mut core, plan.realization.groups_visible());
    eprint!("{}", render_all(&diags, &sm, color()));
    if has_errors(&diags) {
        return ExitCode::from(1);
    }
    let mut lines: Vec<serde_json::Value> = Vec::new();
    // `--host-profile`: the host's phases of each step, taken after it.
    let profile = host_profile.then(presk_kernel::HostProfile::shared);
    let mut profiled: Vec<(presk_kernel::HostProfile, f64)> = Vec::new();
    // The page faults between two steps' ends: a step's, and what the loop
    // does between steps (the batch, the arguments' copy).
    let mut faults = presk_kernel::profile::minor_faults();
    let on_step = |l: &presk_train::StepLog| {
        if let Some(p) = &profile {
            let mut snap = std::mem::take(&mut *p.borrow_mut());
            let now = presk_kernel::profile::minor_faults();
            snap.faults = faults.zip(now).map(|(a, b)| b - a);
            faults = now;
            profiled.push((snap, l.ms / 1e3));
        }
        if json {
            lines.push(serde_json::json!({ "step": l.step, "epoch": l.epoch, "loss": l.loss, "lr": l.lr, "grad_norm": l.grad_norm, "ms": l.ms }));
        } else if log_every > 0 && l.step % log_every == 0 {
            eprintln!("{}", presk_train::format_step(l, 0).replace("/0", ""));
        }
    };
    let fns = match presk_train::prepare_training_as(&mut core, &prog, &typed, ad_options(plan.realization.groups_visible())) {
        Ok(f) => f,
        Err(e) => {
            eprint!("{}", render_all(&[presk_syntax::Diagnostic::error(e.code, e.message)], &sm, color()));
            return ExitCode::from(1);
        }
    };
    eprint!("{}", render_all(&stored_vjps(&core, &plan.realization), &sm, color()));
    // The loop calls the objective's `.value_vjp`: the plan's types and
    // layouts of the `.vjp` and of the objective hold for it (0.10.0 step 3).
    let plan = &presk_train::plan_for_loop(plan, &core, &fns);
    opts.realize = plan.realization.clone();
    if let Err(c) = opt.apply_for(&mut core, &prog, false, &[], plan, &sm) {
        return c;
    }
    if let Err(c) = check_plan(plan, &core, &sm) {
        return c;
    }
    let mut stats = (0u64, 0u64);
    let run = match &profile {
        // The native backend (or the LLVM one) built here, to be given the
        // profile.
        Some(p) => match compiled_backend(backend == BackendKind::Llvm, &core, &prog, &plan.realization) {
            Ok(mut b) => {
                b.set_plan(plan.clone());
                b.set_host_profile(Some(p.clone()));
                let out = presk_train::train(&mut b, &core, &prog, &typed, &fns, &opts, on_step);
                stats = presk_rt::Backend::launch_stats(&b);
                out
            }
            Err(e) => {
                eprintln!("presk train: the native backend is unavailable: {}", e.message);
                return ExitCode::from(1);
            }
        },
        None => with_backend(backend, &core, &prog, opts.bits, plan, |b| {
            let out = presk_train::train(b, &core, &prog, &typed, &fns, &opts, on_step);
            stats = presk_rt::Backend::launch_stats(b);
            out
        }),
    };
    if profile.is_some() && profiled.len() > 1 {
        eprintln!("presk train --host-profile: the host by phase, the mean of steps 2 to {} (the first left out)", profiled.len());
        eprint!("{}", presk_kernel::profile::table(&profiled[1..], "step"));
    }
    match run {
        Err(e) => {
            let usage = e.code == "E9002";
            let d = run_diag(e.code, e.message, e.span);
            eprint!("{}", render_all(&[d], &sm, color()));
            ExitCode::from(if usage { 2 } else { 1 })
        }
        Ok(out) => {
            if json {
                let j = serde_json::json!({
                    "model": out.model,
                    "steps": out.logs.len(),
                    "total": out.total,
                    "step_ms": presk_train::median_step_ms(&out.logs),
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
                if let Some(ms) = presk_train::median_step_ms(&out.logs) {
                    eprintln!("presk train: {ms:.1} ms a step, median of {} (the first excluded)", out.logs.len() - 1);
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
    presk_driver::front::load(file, &mut stderr()).map_err(exit)
}

fn color() -> bool {
    std::io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none()
}

pub(crate) use presk_driver::entry::{interp_diag, run_diag};

/// `NAME=N` pairs of `--const`.
fn parse_consts(consts: &[String]) -> Result<Vec<(String, u64)>, ExitCode> {
    presk_driver::front::parse_consts(consts, &mut stderr()).map_err(exit)
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

/// Parse and elaborate ([`presk_driver::front::front`]).
fn front(file: &Path, consts: &[String]) -> Result<(SourceMap, elab::ElabProgram), ExitCode> {
    presk_driver::front::front(file, consts, &mut stderr()).map_err(exit)
}

/// Parse, elaborate and type ([`presk_driver::front::typed_front`]).
fn typed_front(file: &Path, consts: &[String]) -> Result<(SourceMap, elab::ElabProgram, TypedProgram), ExitCode> {
    presk_driver::front::typed_front(file, consts, &mut stderr()).map_err(exit)
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

/// Lower to PSK-Core and verify the result ([`presk_driver::front::core`]).
fn core(file: &Path, consts: &[String], unroll_stack: bool) -> Result<(SourceMap, elab::ElabProgram, presk_core::Program), ExitCode> {
    presk_driver::front::typed_core_as(file, consts, unroll_stack, mode(), &mut stderr()).map(|(sm, prog, _, core)| (sm, prog, core)).map_err(exit)
}

#[allow(clippy::too_many_arguments)]
fn dump(file: &Path, stage: DumpStage, unroll_stack: bool, entry: Option<&str>, dims: &[String], consts: &[String], plan: &presk_rt::Plan, planned: bool, profile: Option<PathBuf>, weights: Option<PathBuf>, sites: usize) -> ExitCode {
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
        let mut diags = differentiate(&mut core, plan.realization.groups_visible());
        diags.extend(stored_vjps(&core, &plan.realization));
        eprint!("{}", render_all(&diags, &sm, color()));
        if has_errors(&diags) {
            return ExitCode::from(1);
        }
        if let Err(c) = check_plan(plan, &core, &sm) {
            return c;
        }
        // The kernels from the entry's shapes, nothing computed (0.11.0,
        // X4); the kernel interpreter's default is scalar, so a dump without
        // a plan, or with one that states no lanes, prints 0.6.0's kernels.
        let listing = match presk_driver::layout::kernels(&core, &prog, entry, dims, plan, 0) {
            Ok(l) => l,
            Err(e) => {
                eprint!("{}", render_all(&[presk_syntax::Diagnostic::error(e.code, e.message)], &sm, color()));
                return ExitCode::from(1);
            }
        };
        for line in presk_driver::text::kernel_heading(&listing) {
            println!("{line}");
        }
        for k in &listing.kernels {
            print!("{k}");
        }
        return ExitCode::SUCCESS;
    }
    let (sm, id) = match load(file) {
        Ok(x) => x,
        Err(c) => return c,
    };
    match stage {
        DumpStage::Kernel | DumpStage::Sensitivity | DumpStage::Network => unreachable!("handled above"),
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
        // `--const` binds the constants here too (until 0.11.0 step 7 the two
        // stages ignored it, D110).
        DumpStage::Elab => match front(file, consts) {
            Err(c) => c,
            Ok((sm, prog)) => {
                // A module that includes itself has no parameter tree to print.
                let cycles = typing::module_cycles(&prog);
                if !cycles.is_empty() {
                    eprint!("{}", render_all(&cycles, &sm, color()));
                    return ExitCode::from(1);
                }
                print!("{}", elab::dump(&prog));
                ExitCode::SUCCESS
            }
        },
        DumpStage::Core => match core(file, consts, unroll_stack) {
            Err(c) => c,
            Ok((_, prog, core)) => {
                print!("{}", presk_core::pretty(&core, &prog));
                ExitCode::SUCCESS
            }
        },
        DumpStage::Realize => match core(file, consts, unroll_stack) {
            Err(c) => c,
            Ok((sm, _prog, mut core)) => {
                let mut diags = differentiate(&mut core, plan.realization.groups_visible());
                diags.extend(stored_vjps(&core, &plan.realization));
                eprint!("{}", render_all(&diags, &sm, color()));
                if has_errors(&diags) {
                    return ExitCode::from(1);
                }
                let (mut lets, mut saved, mut groups) = (0usize, 0usize, 0usize);
                for f in &core.funcs {
                    let plan = presk_realize::plan(f);
                    lets += f.body.lets.len();
                    saved += plan.saved();
                    groups += plan.groups.len();
                    if !plan.groups.is_empty() {
                        print!("{}", presk_realize::dump(f, &plan));
                    }
                }
                println!("-- {} functions, {lets} lets, {groups} groups, {saved} lets fused away", core.funcs.len());
                if planned {
                    if let Err(c) = check_plan(plan, &core, &sm) {
                        return c;
                    }
                    let profile = plan.profile.as_deref().map(|p| format!(", chosen against {p}")).unwrap_or_default();
                    println!("-- plan: realization {}{profile}; {} nests with a layout, every other nest at its default", plan.realization.label(), plan.layout.len());
                    let width = plan.layout.keys().map(|n| n.chars().count()).max().unwrap_or(0);
                    for (nest, layout) in &plan.layout {
                        println!("  {nest:<width$}  {:<14}  {layout}", presk_kernel::plan::nest_label(&core, nest));
                    }
                }
                ExitCode::SUCCESS
            }
        },
        DumpStage::CoreAfterAd | DumpStage::CoreOpt => match core(file, consts, unroll_stack) {
            Err(c) => c,
            Ok((sm, prog, mut core)) => {
                let mut diags = differentiate(&mut core, plan.realization.groups_visible());
                diags.extend(stored_vjps(&core, &plan.realization));
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

/// `--level` of `presk dump --stage network`: a depth, or `all`.
#[derive(Clone, Copy, Debug)]
enum Level {
    Depth(usize),
    All,
}

fn parse_level(s: &str) -> Result<Level, String> {
    if s == "all" {
        return Ok(Level::All);
    }
    s.parse().map(Level::Depth).map_err(|_| format!("`{s}`: a level is a number or `all`"))
}

/// `presk dump --stage network` (plan 0.14.0 step 3, D143): the network
/// diagram of an entry, as `presk explain` draws it. Every stage runs as for
/// the document, on the native target and without the exact reference; the
/// directory its artifacts would go to is removed afterwards.
#[allow(clippy::too_many_arguments)]
fn dump_network(file: &Path, format: DumpFormat, (level, boxed, memory): (Option<Level>, Option<&str>, bool), standalone: bool, entry: Option<&str>, dims: &[String], consts: &[String], realize: &str, plan: Option<&str>, profile: Option<PathBuf>, weights: Option<PathBuf>) -> ExitCode {
    let Some(entry) = entry else {
        eprintln!("presk dump --stage network: --entry is required (the diagram is an entry's, its arrows at the entry's shapes)");
        return ExitCode::from(2);
    };
    if matches!(format, DumpFormat::Latex | DumpFormat::Unicode) {
        eprintln!("presk dump --stage network: --format text, json, svg or tikz");
        return ExitCode::from(2);
    }
    let out = std::env::temp_dir().join(format!("presk-network-{}", std::process::id()));
    let o = presk_explain::Options {
        file: file.to_path_buf(),
        entry: entry.to_string(),
        dims: dims.to_vec(),
        consts: consts.to_vec(),
        budget: None,
        menu: Default::default(),
        realize: plan.is_none().then(|| realize.to_string()),
        plan: plan.map(String::from),
        profile,
        threads: None,
        remat: crate::remat(),
        opt: false,
        weights,
        seed: 0,
        bits: 128,
        envelope: None,
        no_exact: true,
        targets: vec![presk_explain::Target::Native],
        measure: false,
        sites: Vec::new(),
        json: true,
        html: false,
        out: out.clone(),
    };
    let got = presk_explain::network(&o, &mut stderr());
    let _ = std::fs::remove_dir_all(&out);
    let n = match got {
        Ok(n) => n,
        Err(stop) => return exit(stop),
    };
    let shares = n.memory;
    let Some(mut n) = n.network else {
        eprintln!("presk dump --stage network: `{entry}` has no diagram");
        return ExitCode::from(1);
    };
    if memory {
        if shares.is_empty() {
            eprintln!("presk dump --stage network --memory: `{entry}`'s host could not be recorded, so its memory is not counted");
            return ExitCode::from(1);
        }
        n.memory = shares;
    }
    let v = match boxed {
        Some(name) => match presk_diagram::layout::find_box(&n, name) {
            Some(b) => presk_diagram::layout::alone(&n, b),
            None => {
                eprintln!("presk dump --stage network: no box called `{name}`");
                return ExitCode::from(1);
            }
        },
        None => match level.unwrap_or(Level::Depth(1)) {
            Level::Depth(k) => presk_diagram::layout::view(&n, &presk_diagram::layout::level(&n, k)),
            Level::All => presk_diagram::layout::view(&n, &presk_diagram::layout::everything(&n)),
        },
    };
    let v = &v;
    let command = std::iter::once("presk".to_string()).chain(std::env::args().skip(1)).collect::<Vec<_>>().join(" ");
    match format {
        DumpFormat::Json => println!("{}", presk_diagram::json::network(&n)),
        DumpFormat::Svg => print!("{}", presk_diagram::svg::svg(&n, v, false)),
        DumpFormat::Tikz => print!("{}", presk_diagram::tikz::tikz(&n, v, standalone, &command)),
        _ => print!("{}", presk_diagram::text::outline(&n)),
    }
    ExitCode::SUCCESS
}

/// `presk dump --format latex|unicode` (plan 0.13.0 step 4): the typed
/// program, or PSK-Core after differentiation, as the equations of the
/// reference's chapter 3. A form the printer does not write is printed as
/// its source text and named on standard error.
#[allow(clippy::too_many_arguments)]
fn dump_math(file: &Path, stage: DumpStage, f: presk_print_math::Format, sel: &presk_print_math::Select, standalone: bool, consts: &[String], unroll_stack: bool, plan: Option<&presk_rt::Plan>) -> ExitCode {
    let main = file.file_name().and_then(|s| s.to_str()).unwrap_or("");
    match stage {
        DumpStage::Elab => {
            let (sm, prog, typed) = match typed_front(file, consts) {
                Ok(x) => x,
                Err(c) => return c,
            };
            let l = presk_print_math::typed_program(&sm, &prog, &typed);
            print!("{}", presk_print_math::listing(&l, f, sel, main, standalone));
            for (sp, text) in &l.gaps {
                let (line, _) = sm.file(sp.file).line_col(sp.start);
                eprintln!("presk dump: {}:{line}: `{text}` printed as its source text: the printer does not write this form", sm.file(sp.file).name);
            }
            ExitCode::SUCCESS
        }
        DumpStage::CoreAfterAd => {
            let (sm, prog, mut core) = match core(file, consts, unroll_stack) {
                Ok(x) => x,
                Err(c) => return c,
            };
            let groups = plan.is_some_and(|p| p.realization.groups_visible());
            let (diags, links) = presk_driver::ad::differentiate_traced(&mut core, remat(), groups);
            eprint!("{}", render_all(&diags, &sm, color()));
            if has_errors(&diags) {
                return ExitCode::from(1);
            }
            let ds = presk_print_math::core::core_program(&core, &prog, &links);
            print!("{}", presk_print_math::core::listing(&ds, f, sel, main, standalone));
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("presk dump: --format latex|unicode reads --stage elab or --stage core-after-ad");
            ExitCode::from(2)
        }
    }
}

/// `NAME=N` pairs of `--dim`.
fn parse_dims(dims: &[String]) -> Result<Vec<(String, u64)>, ExitCode> {
    presk_driver::front::parse_dims(dims, &mut stderr()).map_err(exit)
}

#[allow(clippy::too_many_arguments)]
fn run(file: &Path, entry: Option<&str>, dims: &[String], consts: &[String], seed: u64, bits: u32, digits: usize, (json, verbose, result_only): (bool, bool, bool), unroll_stack: bool, backend: BackendKind, plan: &presk_rt::Plan, opt: Opt, data: Option<PathBuf>, samples: Option<usize>, from: usize, weights: Option<PathBuf>) -> ExitCode {
    let realize = &plan.realization;
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
    let mut diags = differentiate(&mut core, plan.realization.groups_visible());
    diags.extend(stored_vjps(&core, &plan.realization));
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
    if let Err(c) = opt.apply_for(&mut core, &prog, verbose, &dims, plan, &sm) {
        return c;
    }
    if let Err(c) = check_plan(plan, &core, &sm) {
        return c;
    }
    let opts = presk_interp::RunOptions { entry: entry.to_string(), dims, seed, bits, digits };
    let evaluate = |b: &mut dyn presk_rt::Backend| match (&data, &weights) {
        (Some(path), None) => presk_train::run_on_data_from(b, &core, &prog, &opts, realize.clone(), path, from, samples).map_err(|e| presk_rt::RtError { code: e.code, message: e.message, span: e.span }),
        (None, None) => presk_rt::run_with(b, &core, &prog, &opts, realize),
        // Trained parameters: the arguments built as above, the first one
        // read from the checkpoint (0.6.0 step 7, a validation loss).
        (_, Some(w)) => {
            // The parameters are described and not drawn, then read from the
            // checkpoint against a tree of their shape (0.17.0 step 7, D180).
            let ((func, fdims, inputs, mut args), described) = match &data {
                Some(path) => presk_train::args_on_data_given_params(&core, &prog, &opts, path, from, samples).map_err(|e| presk_rt::RtError { code: e.code, message: e.message, span: e.span })?,
                None => {
                    let (func, fdims, inputs, described) = presk_interp::run::prepare_run_given_params(&core, &prog, &opts)?;
                    let args = inputs.iter().map(|(_, v)| presk_rt::Value::from_interp(v)).collect::<presk_rt::Result<Vec<_>>>()?;
                    ((func, fdims, inputs, args), described)
                }
            };
            args[0] = match &described {
                Some(s) => presk_driver::entry::read_given_params(&core, &prog, s, w)?,
                None => presk_rt::load_params(w, &args[0])?,
            };
            // What `--json` and `--verbose` print as the parameters is the
            // checkpoint's, as it is read and before a `quant` map rounds it,
            // not the initialization synthesized above (D118's follow-up).
            // Built only to be printed: the synthesized tree is dropped
            // first, so a model's parameters are never held twice more.
            // `--result-only` prints the result alone, and nothing is built.
            let mut inputs = inputs;
            if (json || verbose) && !result_only {
                inputs[0].1 = presk_interp::Value::Nat(0);
                inputs[0].1 = args[0].to_interp(opts.bits);
            }
            realize.store_params(&mut args)?;
            let p = b.prepare(func, &fdims, realize)?;
            let result = b.call(&p, args)?;
            let dims_named: Vec<(String, u64)> = fdims.iter().map(|(v, n)| (prog.dim_name(*v), *n)).collect();
            Ok(presk_interp::RunOutput { entry: opts.entry.clone(), func, dims: dims_named, inputs, result: result.to_interp(opts.bits) })
        }
    };
    match with_backend(backend, &core, &prog, bits, plan, evaluate) {
        Err(e) => {
            let d = run_diag(e.code, e.message, e.span);
            eprint!("{}", render_all(&[d], &sm, color()));
            ExitCode::from(1)
        }
        Ok(out) => {
            // Printed by the entry's signature: a `bool[S]` value as truth
            // values (D99's addendum), though it is indices below 2 at run time.
            let f = core.func(out.func);
            let param_ty = |i: usize| f.params.get(i).map_or(&presk_types::Ty::Nat, |p| &p.ty);
            if json {
                let inputs: serde_json::Map<String, serde_json::Value> = if result_only { serde_json::Map::new() } else { out.inputs.iter().enumerate().map(|(i, (n, v))| (n.clone(), presk_interp::run::to_json_typed(v, param_ty(i), digits))).collect() };
                let mut j = serde_json::json!({
                    "entry": out.entry,
                    "seed": seed,
                    "bits": bits,
                    "dims": out.dims.iter().map(|(n, v)| (n.clone(), serde_json::json!(v))).collect::<serde_json::Map<_, _>>(),
                    "inputs": inputs,
                    "result": presk_interp::run::to_json_typed(&out.result, &f.ret, digits),
                });
                if result_only && let Some(o) = j.as_object_mut() {
                    o.remove("inputs");
                }
                println!("{}", serde_json::to_string_pretty(&j).unwrap());
            } else {
                if verbose {
                    println!("-- entry {} with {} at {} bits, seed {}", out.entry, out.dims.iter().map(|(n, v)| format!("{n} = {v}")).collect::<Vec<_>>().join(", "), bits, seed);
                    for (i, (n, v)) in out.inputs.iter().enumerate().filter(|_| !result_only) {
                        println!("{n} : {} = {}", presk_interp::run::describe_typed(v, param_ty(i)), presk_interp::run::render_typed(v, param_ty(i), digits, 0));
                    }
                    println!("-- result");
                }
                println!("{} = {}", presk_interp::run::describe_typed(&out.result, &f.ret), presk_interp::run::render_typed(&out.result, &f.ret, digits, 0));
            }
            ExitCode::SUCCESS
        }
    }
}

/// `presk export --target c` (plan 0.9.0 step 3, D79): the options first,
/// then the entry (`presk_driver::export_c::export`). `E8002`, which refused the target
/// (steps 0–2) and then `--plan` and `--threads` (steps 3–4), is no longer
/// emitted (step 5, D81). With `--serve` (step 6, D83) the entry is the
/// file's generation function unless `--entry` names another, as for
/// `presk serve`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn export_c(file: &Path, out: &Path, entry: Option<&str>, dims: &[String], consts: &[String], realize: &str, plan: Option<&str>, weights: Option<PathBuf>, seed: u64, (no_exact, straight_line, placed, serve, llvm): (bool, bool, bool, bool, bool), pytorch: &[(&str, bool)]) -> ExitCode {
    if let Some((flag, _)) = pytorch.iter().find(|(_, given)| *given) {
        eprintln!("presk export: `{flag}` is an option of the PyTorch export, not of --target c");
        return ExitCode::from(2);
    }
    if entry.is_none() && !serve {
        eprintln!("presk export --target c: --entry is required (the program is one entry at bound dimensions, as `presk run` runs it), or --serve for the file's generation function");
        return ExitCode::from(2);
    }
    presk_driver::export_c::export(&presk_driver::export_c::Options { file, entry, dims, consts, realize, plan, weights, seed, no_exact, straight_line, placed, serve, llvm, out, remat: remat() }, &mut stderr()).map_or_else(exit, |()| ExitCode::SUCCESS)
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
        let (generate, key, temperature, seq_dim) = match sv.generate {
            Some(f) => {
                let f = prog.function(f);
                // The context follows a key when the function takes one (plan 0.10.0 step 7b).
                let key = matches!(f.params.get(1).and_then(|p| p.ty.as_ref()), Some(presk_types::ty::Ty::Key));
                let c = if key { 2 } else { 1 };
                let seq = f.params.get(c).and_then(|p| p.ty.as_ref()).and_then(|t| match t {
                    presk_types::ty::Ty::Idx { shape, .. } => shape.axes.iter().find(|a| a.name.as_deref() == Some("seq")).and_then(|a| a.len.vars().next()).map(|v| prog.dim_name(v)),
                    _ => None,
                });
                let temp = f.params.get(c + 1).and_then(|p| p.default.as_ref()).and_then(|e| match e {
                    presk_syntax::ast::Expr::Lit { lit: presk_syntax::ast::Lit::Float(x), .. } => Some(x.clone()),
                    _ => None,
                });
                (Some(f.name.clone()), key, if f.params.len() > c + 1 { Some(temp.unwrap_or_else(|| "1.0".into())) } else { None }, seq)
            }
            None => (None, false, None, None),
        };
        let spec = presk_export_pytorch::ServeSpec {
            source: file.display().to_string(),
            model_stem: presk_export_pytorch::stem(&prog, prog.module(*def).unit),
            class: prog.module(*def).name.clone(),
            ctor_dims: extra,
            main_stem: presk_export_pytorch::stem(&prog, prog.main),
            generate,
            key,
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
        // `fn loss(θ, k : key, batch) uses Rand`: the loop passes `None` for
        // the key and turns the `training` switch on (D54).
        let osig = prog.function(run.objective);
        let key = matches!(typed.param_ty.get(&(run.objective, 1)).cloned().or_else(|| osig.params.get(1).and_then(|p| p.ty.clone())), Some(presk_types::ty::Ty::Key));
        let rand = osig.effects.iter().any(|e| e == "Rand");
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
            key,
            rand,
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
            // No realization: the groups kept, as by default.
            let diags = differentiate(&mut core, true);
            eprint!("{}", render_all(&diags, &sm, color()));
            if has_errors(&diags) {
                return ExitCode::from(1);
            }
            let topts = presk_train::TrainOptions { data: None, steps: Some(k), epochs: None, seed, bits: 256, realize: presk_rt::Realization::uniform(presk_rt::Dtype::F64), ckpt: None, checkpoints: false };
            let fns = match presk_train::prepare_training_as(&mut core, &prog, &typed, ad_options(true)) {
                Ok(f) => f,
                Err(e) => {
                    eprint!("{}", render_all(&[presk_syntax::Diagnostic::error(e.code, e.message)], &sm, color()));
                    return ExitCode::from(1);
                }
            };
            if let Err(c) = opt.apply(&mut core, &prog, false, &[]) {
                return c;
            }
            let tout = match presk_train::train(&mut presk_rt::InterpBackend::new(&core, &prog, 256), &core, &prog, &typed, &fns, &topts, |_| {}) {
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
                match presk_cranelift::native_backend(&core, &prog, &realize) {
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
                train: Some(presk_export_pytorch::TrainConformance { k, ctor, objective: spec.objective.clone(), key: spec.key, rtol: 1e-6, atol: 1e-9 }),
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
        // No realization: the groups kept, as by default.
        let diags = differentiate(&mut core, true);
        eprint!("{}", render_all(&diags, &sm, color()));
        if has_errors(&diags) {
            return ExitCode::from(1);
        }
        // An internal error, reported as every command reports it (until
        // 0.11.0 step 7 the export exited 1 without a word, D110).
        if let Err(errs) = presk_core::verify(&core) {
            for e in errs {
                eprintln!("presk: internal error (PSK-Core verification after AD): {e}");
            }
            return ExitCode::from(2);
        }
        if let Err(c) = opt.apply(&mut core, &prog, false, &dims) {
            return c;
        }
        let ropts = presk_interp::RunOptions { entry: entry.to_string(), dims, seed, bits: 256, digits: 17 };
        let run = match presk_interp::run(&core, &prog, &ropts) {
            Ok(r) => r,
            Err(e) => {
                eprint!("{}", render_all(&[interp_diag(e)], &sm, color()));
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
