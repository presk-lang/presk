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
    // With a checkpoint, the parameters are described and not drawn (D180).
    let given = weights.is_some();
    let (func, fdims, mut args, described) = match data {
        Some(d) => {
            let ((func, fdims, _, args), described) = if given {
                presk_train::args_on_data_given_params(core, prog, opts, Path::new(&d.path), d.from, d.samples).map_err(|e| e.message)?
            } else {
                (presk_train::args_on_data(core, prog, opts, Path::new(&d.path), d.from, d.samples).map_err(|e| e.message)?, None)
            };
            (func, fdims, args, described)
        }
        None => {
            let (func, fdims, inputs, described) = if given {
                presk_interp::run::prepare_run_given_params(core, prog, opts).map_err(|e| e.message)?
            } else {
                let (func, fdims, inputs) = presk_interp::run::prepare_run(core, prog, opts).map_err(|e| e.message)?;
                (func, fdims, inputs, None)
            };
            let args = inputs.iter().map(|(_, v)| presk_rt::Value::from_interp(v)).collect::<presk_rt::Result<Vec<_>>>().map_err(|e| e.message)?;
            (func, fdims, args, described)
        }
    };
    // A checkpoint that does not load is an error, never a fallback to the
    // initial parameters (D48).
    // The reader's message names the file when it could not open it; the
    // path is put in front of the others (until 0.11.0 step 7 it was put in
    // front of every one, so that a missing file was named twice).
    if let Some(w) = weights {
        let path = w.display().to_string();
        let read = match &described {
            Some(s) => read_given_params(core, prog, s, w),
            None => presk_rt::load_params(w, &args[0]),
        };
        args[0] = read.map_err(|e| if e.message.starts_with(&path) { e.message } else { format!("{path}: {}", e.message) })?;
    }
    Ok((func, fdims, args))
}

/// The parameters a checkpoint gives, read against a tree of the
/// initializer's shape that is never drawn (0.17.0 step 7, D180). The
/// initializer runs on the kernel backend with no executor
/// ([`crate::layout::params_zeros`]), so the tree is zeros in `f64` and
/// nothing is computed in exact arithmetic; the drawn parameters have the
/// same tree and axes (the test `given_params`), so the checkpoint reads to
/// the same values. `described` is what
/// [`presk_interp::run::prepare_run_given_params`] gives.
pub fn read_given_params(core: &presk_core::Program, prog: &ElabProgram, described: &presk_interp::run::Shaped, weights: &Path) -> presk_rt::Result<presk_rt::Value> {
    let presk_interp::run::Shaped::Params { init, dims, key } = described else {
        return presk_rt::err("E7001", "the entry's first argument is not a module's parameters");
    };
    let template = crate::layout::params_zeros(core, prog, *init, dims, key)?;
    presk_rt::load_params(weights, &template)
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
