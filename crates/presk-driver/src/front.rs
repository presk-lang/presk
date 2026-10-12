//! The front of the pipeline: a file read, parsed, elaborated, typed and
//! lowered to PSK-Core, with the `--const` and `--dim` pairs it is given.

use std::path::Path;

use presk_syntax::diag::has_errors;
use presk_syntax::{FileId, SourceMap, parse_file};
use presk_types::check::{self as typing, TypedProgram};
use presk_types::elab::{self, ElabProgram, Options};

use crate::{Report, Result, Stop};

/// The file, read into a fresh source map.
pub fn load(file: &Path, report: &mut dyn Report) -> Result<(SourceMap, FileId)> {
    let mut sm = SourceMap::new();
    match sm.load(file) {
        Ok(id) => Ok((sm, id)),
        Err(e) => {
            report.line(&format!("presk: cannot read {}: {e}", file.display()));
            Err(Stop(2))
        }
    }
}

/// `NAME=N` pairs of `--const`.
pub fn parse_consts(consts: &[String], report: &mut dyn Report) -> Result<Vec<(String, u64)>> {
    let mut out = Vec::new();
    for c in consts {
        match c.split_once('=').and_then(|(n, v)| v.trim().parse::<u64>().ok().map(|v| (n.trim().to_string(), v))) {
            Some(pair) => out.push(pair),
            None => {
                report.line(&format!("presk: `--const {c}`: expected NAME=N with N a natural number"));
                return Err(Stop(2));
            }
        }
    }
    Ok(out)
}

/// `NAME=N` pairs of `--dim`.
pub fn parse_dims(dims: &[String], report: &mut dyn Report) -> Result<Vec<(String, u64)>> {
    parse_consts(dims, report).inspect_err(|_| report.line("presk: (in `--dim`)"))
}

/// Parse and elaborate; report all diagnostics. Returns the program when
/// there is no error.
pub fn front(file: &Path, consts: &[String], report: &mut dyn Report) -> Result<(SourceMap, ElabProgram)> {
    let const_overrides = parse_consts(consts, report)?;
    let (mut sm, id) = load(file, report)?;
    let (program, mut diags) = parse_file(&sm, id);
    if has_errors(&diags) {
        report.diagnostics(&diags, &sm);
        return Err(Stop(1));
    }
    let (prog, more) = elab::elaborate(&mut sm, id, program, &Options { const_overrides });
    diags.extend(more);
    report.diagnostics(&diags, &sm);
    if has_errors(&diags) { Err(Stop(1)) } else { Ok((sm, prog)) }
}

/// Parse, elaborate and type; report diagnostics.
pub fn typed_front(file: &Path, consts: &[String], report: &mut dyn Report) -> Result<(SourceMap, ElabProgram, TypedProgram)> {
    let (sm, prog) = front(file, consts, report)?;
    let (typed, diags) = typing::check(&sm, &prog);
    report.diagnostics(&diags, &sm);
    if has_errors(&diags) { Err(Stop(1)) } else { Ok((sm, prog, typed)) }
}

/// Lower to PSK-Core and verify the result.
pub fn core(file: &Path, consts: &[String], unroll_stack: bool, report: &mut dyn Report) -> Result<(SourceMap, ElabProgram, presk_core::Program)> {
    typed_core(file, consts, unroll_stack, report).map(|(sm, prog, _, core)| (sm, prog, core))
}

/// [`core()`], and the typed program it was lowered from.
pub fn typed_core(file: &Path, consts: &[String], unroll_stack: bool, report: &mut dyn Report) -> Result<(SourceMap, ElabProgram, TypedProgram, presk_core::Program)> {
    let (sm, prog, typed) = typed_front(file, consts, report)?;
    let (core, diags) = presk_core::lower::lower_with(&prog, &typed, &presk_core::lower::Options { unroll_stack });
    report.diagnostics(&diags, &sm);
    if has_errors(&diags) {
        return Err(Stop(1));
    }
    if let Err(errs) = presk_core::verify(&core) {
        for e in errs {
            report.line(&format!("presk: internal error (PSK-Core verification): {e}"));
        }
        return Err(Stop(2));
    }
    Ok((sm, prog, typed, core))
}
