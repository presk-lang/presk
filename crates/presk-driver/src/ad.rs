//! Automatic differentiation, as the commands apply it.

use std::collections::{BTreeSet, HashSet};

use presk_core::ad::{Options, Remat};
use presk_syntax::Diagnostic;

/// Automatic differentiation under `remat`, keeping the fusion groups as the
/// recomputing `.vjp`s had them when `groups`: when a realization the
/// command runs can see them (`Realization::groups_visible`, 0.10.0 step
/// 4b, D94). A command that runs several realizations keeps them when any
/// of them can.
pub fn differentiate(core: &mut presk_core::Program, remat: Remat, groups: bool) -> Vec<Diagnostic> {
    presk_core::ad::transform_as(core, options(remat, groups))
}

/// The options [`differentiate`] passes.
pub fn options(remat: Remat, groups: bool) -> Options {
    Options { remat, groups }
}

/// `W0004` (0.10.0 step 4b, D94): a realization that types a `.vjp`, by
/// function or by site, where the program stores its gradient types
/// nothing of it: that `.vjp` only calls `.fwd` and `.bwd` (D92). Unless it
/// names them too, as `presk solve --step` does under `store`, it was
/// written for the recomputing `.vjp`s. Not an error, since a realization
/// written for a training step is also run on forward entries.
pub fn stored_vjps(core: &presk_core::Program, r: &presk_rt::Realization) -> Vec<Diagnostic> {
    let names: HashSet<&str> = core.funcs.iter().map(|f| f.name.as_str()).collect();
    let sites = r.storage.keys().chain(r.accum.keys()).filter_map(|s| s.rsplit_once('/').map(|(f, _)| f));
    let typed: BTreeSet<&str> = r.functions.keys().map(String::as_str).chain(sites).collect();
    let stored = |f: &str| f.strip_suffix(".vjp").is_some_and(|g| names.contains(format!("{g}.bwd").as_str()) && !typed.contains(format!("{g}.bwd").as_str()) && !typed.contains(format!("{g}.fwd").as_str()));
    let named: Vec<&str> = typed.iter().copied().filter(|f| stored(f)).collect();
    named
        .into_iter()
        .map(|v| {
            let g = v.strip_suffix(".vjp").expect("a .vjp");
            let fwd = if names.contains(format!("{g}.fwd").as_str()) { format!("`{g}.fwd`") } else { format!("`{g}` itself") };
            // No label, so no help either: the advice is in the message.
            Diagnostic::warning("W0004", format!("the realization types `{v}`, which this stored gradient only calls (its forward pass is {fwd}, its sweep `{g}.bwd`), so the type applies to nothing: a realization solved under `--remat recompute`, or before 0.10.0 step 4, runs with `--remat recompute`; under `--remat store`, solve it again"))
        })
        .collect()
}
