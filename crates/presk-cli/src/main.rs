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
#[command(name = "presk", version, about = "Presk compiler (0.2.0: front-end, reference interpreter, training, PyTorch export)")]
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
        #[arg(long = "dim", value_name = "NAME=N")]
        dims: Vec<String>,
        #[arg(long = "const", value_name = "NAME=N")]
        consts: Vec<String>,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Dump an intermediate representation (development aid).
    Dump {
        file: PathBuf,
        #[arg(long, value_enum)]
        stage: DumpStage,
        /// Unroll `stack{L}` into L calls instead of a fold (debugging aid).
        #[arg(long)]
        unroll_stack: bool,
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

#[derive(Clone, Copy, ValueEnum)]
enum DumpStage {
    Tokens,
    Ast,
    Elab,
    Core,
    CoreAfterAd,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Check { file, consts, summary, json } => check(&file, &consts, summary, json),
        Command::Dump { file, stage, unroll_stack } => dump(&file, stage, unroll_stack),
        Command::Run { file, entry, dims, consts, seed, bits, digits, json, verbose, unroll_stack } => run(&file, entry.as_deref(), &dims, &consts, seed, bits, digits as usize, json, verbose, unroll_stack),
        Command::Export { file, target: _, mode, out, shapes, stack, conformance_entry, train, dims, consts, seed } => export(&file, mode, &out, shapes.as_deref(), stack, conformance_entry.as_deref(), train, &dims, &consts, seed),
        Command::Train { file, consts, data, steps, epochs, seed, bits, log_every, ckpt, json } => train(&file, &consts, presk_train::TrainOptions { data, steps, epochs, seed, bits, ckpt, checkpoints: true }, log_every, json),
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

/// `presk train`: the loop of the file's `training run` (crate `presk-train`).
fn train(file: &Path, consts: &[String], opts: presk_train::TrainOptions, log_every: u64, json: bool) -> ExitCode {
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
    match presk_train::train(&mut core, &prog, &typed, &opts, on_step) {
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

fn dump(file: &Path, stage: DumpStage, unroll_stack: bool) -> ExitCode {
    let (sm, id) = match load(file) {
        Ok(x) => x,
        Err(c) => return c,
    };
    match stage {
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
        DumpStage::CoreAfterAd => match core(file, &[], unroll_stack) {
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
fn run(file: &Path, entry: Option<&str>, dims: &[String], consts: &[String], seed: u64, bits: u32, digits: usize, json: bool, verbose: bool, unroll_stack: bool) -> ExitCode {
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
    let opts = presk_interp::RunOptions { entry: entry.to_string(), dims, seed, bits, digits };
    match presk_interp::run(&core, &prog, &opts) {
        Err(e) => {
            let mut d = presk_syntax::Diagnostic::error("E7001", e.message);
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
fn export(file: &Path, mode: ExportMode, out: &Path, shapes: Option<&str>, stack: StackMode, conformance_entry: Option<&str>, train: bool, dims: &[String], consts: &[String], seed: u64) -> ExitCode {
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
            let topts = presk_train::TrainOptions { data: None, steps: Some(k), epochs: None, seed, bits: 256, ckpt: None, checkpoints: false };
            let tout = match presk_train::train(&mut core, &prog, &typed, &topts, |_| {}) {
                Ok(t) => t,
                Err(e) => {
                    eprint!("{}", render_all(&[presk_syntax::Diagnostic::error(e.code, e.message)], &sm, color()));
                    return ExitCode::from(1);
                }
            };
            let cdir = out.join("conformance");
            if let Err(e) = std::fs::create_dir_all(&cdir) {
                eprintln!("presk export: cannot create {}: {e}", cdir.display());
                return ExitCode::from(2);
            }
            let inputs = serde_json::json!({ "θ": presk_interp::run::to_json(&tout.theta0, 17) });
            std::fs::write(cdir.join("inputs.json"), serde_json::to_string_pretty(&inputs).unwrap()).ok();
            std::fs::write(cdir.join("expected.json"), serde_json::to_string_pretty(&presk_interp::run::to_json(&tout.theta, 17)).unwrap()).ok();
            std::fs::write(cdir.join("batches.json"), serde_json::to_string(&tout.batches).unwrap()).ok();
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
