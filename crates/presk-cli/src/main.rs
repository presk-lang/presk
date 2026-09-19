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
#[command(name = "presk", version, about = "Presk compiler (0.1.0: front-end, reference interpreter, PyTorch export)")]
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
        /// Also generate test_conformance.py for this entry (needs --dim/--const as for `run`).
        #[arg(long, value_name = "ENTRY")]
        conformance_entry: Option<String>,
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
        Command::Dump { file, stage } => dump(&file, stage),
        Command::Run { file, entry, dims, consts, seed, bits, digits, json, verbose } => run(&file, entry.as_deref(), &dims, &consts, seed, bits, digits as usize, json, verbose),
        Command::Export { file, target: _, mode, out, shapes, stack, conformance_entry, dims, consts, seed } => export(&file, mode, &out, shapes.as_deref(), stack, conformance_entry.as_deref(), &dims, &consts, seed),
    }
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
    }
    *sm_out = sm;
    if failed { ExitCode::from(1) } else { ExitCode::SUCCESS }
}

/// Lower to PSK-Core and verify the result.
fn core(file: &Path, consts: &[String]) -> Result<(SourceMap, elab::ElabProgram, presk_core::Program), ExitCode> {
    let (sm, prog, typed) = typed_front(file, consts)?;
    let (core, diags) = presk_core::lower(&prog, &typed);
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

fn dump(file: &Path, stage: DumpStage) -> ExitCode {
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
        DumpStage::Core => match core(file, &[]) {
            Err(c) => c,
            Ok((_, prog, core)) => {
                print!("{}", presk_core::pretty(&core, &prog));
                ExitCode::SUCCESS
            }
        },
        DumpStage::CoreAfterAd => match core(file, &[]) {
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
fn run(file: &Path, entry: Option<&str>, dims: &[String], consts: &[String], seed: u64, bits: u32, digits: usize, json: bool, verbose: bool) -> ExitCode {
    let dims = match parse_dims(dims) {
        Ok(d) => d,
        Err(c) => return c,
    };
    let (sm, prog, mut core) = match core(file, consts) {
        Err(c) => return c,
        Ok(x) => x,
    };
    let Some(entry) = entry else {
        // No entry: a `training run` is not executable in 0.1.0; otherwise list the candidates.
        if !prog.units[prog.main].training.is_empty() {
            eprintln!("presk run: training run: not executable in 0.1.0 (no data reader, no optimizer semantics); use --entry to evaluate a function");
            return ExitCode::from(3);
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
fn export(file: &Path, mode: ExportMode, out: &Path, shapes: Option<&str>, stack: StackMode, conformance_entry: Option<&str>, dims: &[String], consts: &[String], seed: u64) -> ExitCode {
    if matches!(mode, ExportMode::Graph) {
        eprintln!("presk export: --mode graph is not available in 0.1.0 (only --mode readable)");
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
    }
    eprintln!("presk export: wrote {}", written.join(", "));
    ExitCode::SUCCESS
}
