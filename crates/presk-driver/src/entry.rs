//! The entry a command runs: its function, its dimensions, its arguments.

use std::collections::BTreeMap;
use std::path::Path;

use presk_core::FuncId;
use presk_interp::eval::Dims;
use presk_interp::run::resolve_entry;
use presk_precision::envelope::DataSlice;
use presk_syntax::{Diagnostic, Span};
use presk_types::elab::ElabProgram;

/// The entry `entry` names and its dimensions: a module's arguments, as
/// `LayerNorm{8}.apply` gives them, then `named`, by name among the
/// function's own dimensions, as `presk run` binds `--dim` (a name that is
/// not the function's is left out). Until 0.11.0, `presk check --precision`
/// kept only `named`, so that a module entry's dimensions were unbound and
/// a sum over them was priced at a default length (D108, finding 1).
pub fn entry_dims(core: &presk_core::Program, prog: &ElabProgram, entry: &str, named: &[(String, u64)]) -> std::result::Result<(FuncId, Dims), presk_interp::RunError> {
    let (func, mut dims, _) = resolve_entry(core, prog, entry)?;
    let named: BTreeMap<&str, u64> = named.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    for d in &core.func(func).dims {
        if let Some(n) = named.get(prog.dim_name(*d).as_str()) {
            dims.insert(*d, *n);
        }
    }
    Ok((func, dims))
}

/// An entry and its arguments at the seam.
pub type EntryArgs = (FuncId, Dims, Vec<presk_rt::Value>);

/// The entry of `opts` and its arguments at the seam: the last one read from
/// a data slice when there is one, the first one from a checkpoint when there
/// is one, the rest synthesized as `presk run` does. The same inputs for the
/// envelope and for a gap measured beside the bound it supports.
pub fn entry_args(core: &presk_core::Program, prog: &ElabProgram, opts: &presk_interp::RunOptions, data: Option<&DataSlice>, weights: Option<&Path>) -> std::result::Result<EntryArgs, String> {
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
    // The reader's message names the file when it could not open it; the
    // path is put in front of the others (until 0.11.0 step 7 it was put in
    // front of every one, so that a missing file was named twice).
    if let Some(w) = weights {
        let path = w.display().to_string();
        args[0] = presk_rt::load_params(w, &args[0]).map_err(|e| if e.message.starts_with(&path) { e.message } else { format!("{path}: {}", e.message) })?;
    }
    Ok((func, fdims, args))
}

/// A run's error as a diagnostic, at the binding it was evaluating; an entry
/// refused at its dimensions (`E7003`) at the operation undefined there.
pub fn run_diag(code: &'static str, message: String, span: Option<Span>) -> Diagnostic {
    let d = Diagnostic::error(code, message);
    match span {
        Some(s) if code == presk_interp::defined::CODE => d.label(s, "undefined at the dimensions given"),
        Some(s) => d.label(s, "while evaluating this"),
        None => d,
    }
}

/// The same for the interpreter's error, `E7001` unless it names its code.
pub fn interp_diag(e: presk_interp::RunError) -> Diagnostic {
    run_diag(e.code.unwrap_or("E7001"), e.message, e.span)
}
