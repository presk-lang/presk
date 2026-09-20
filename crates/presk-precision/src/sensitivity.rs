//! Sensitivity by differentiation: the *ranking* side.
//!
//! The propagated bound of [`crate::propagate`] is sound and, on a real
//! model, pessimistic by many orders of magnitude (D39). It is therefore the
//! wrong instrument for *allocating* a budget: allocating by a pessimistic
//! bound spends the budget where the bound is loose rather than where the
//! error matters.
//!
//! The right instrument is the one this compiler already has. For a scalar
//! output `L`, the sensitivity of `L` to a perturbation of the value `v` is
//!
//! ```text
//! s_v = ‖∂L/∂v‖₁
//! ```
//!
//! and a reverse sweep computes exactly that for **every** intermediate on
//! the way to the parameters — it is what a reverse sweep is — and then
//! discards all of them but the parameters'. So no new differentiation and
//! no perturbation per site is needed: the adjoints are already computed, and
//! [`presk_core::ad::AdjointMap`] says which variable of the generated `.vjp`
//! holds each one.
//!
//! # What it is and is not
//!
//! `s_v` is a first-order estimate, not a bound: it says how much `L` moves
//! for a small perturbation of `v`, which is what a budget allocation needs
//! and is not what a guarantee needs. **The two must never be confused**: the
//! solver proposes with `s_v` and verifies with a bound, which is what the
//! specification means by calling precision assignment an inverse Herbie.
//!
//! # What it predicts
//!
//! Mixed-precision practice says to keep the embedding table and the output
//! head wide and that the down-projection can be narrow. That is folklore,
//! arrived at empirically. If this analysis reproduces it, the folklore has
//! an explanation; if it contradicts it, that is a result. Either way the
//! ranking is printed rather than asserted.

use std::collections::BTreeMap;

use presk_core::ad::AdjointMap;
use presk_core::{FuncId, Program};

/// `‖∂L/∂v‖₁` per site of the differentiated function.
#[derive(Clone, Debug, Default)]
pub struct Sensitivity {
    /// Site (`"<function>/%<let>"`) to its 1-norm adjoint.
    pub sites: BTreeMap<String, f64>,
}

impl Sensitivity {
    pub fn get(&self, site: &str) -> Option<f64> {
        self.sites.get(site).copied()
    }

    /// Sites by decreasing sensitivity.
    pub fn ranked(&self) -> Vec<(&str, f64)> {
        let mut v: Vec<(&str, f64)> = self.sites.iter().map(|(k, x)| (k.as_str(), *x)).collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        v
    }

    /// The same, summed per module — `SelfAttention`, `LayerNorm`, … — which
    /// is the grain mixed-precision advice is actually given at.
    pub fn by_module(&self) -> Vec<(String, f64)> {
        let mut m: BTreeMap<String, f64> = BTreeMap::new();
        for (site, s) in &self.sites {
            let module = site.split('/').next().unwrap_or(site).to_string();
            *m.entry(module).or_insert(0.0) += s;
        }
        let mut v: Vec<(String, f64)> = m.into_iter().collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        v
    }
}

/// Read the adjoints of `entry` off a run of its `.vjp`.
///
/// `sums` is what the interpreter recorded while running the vjp: the 1-norm
/// at every site of every function it touched. This maps the vjp's sites back
/// to the forward ones the adjoints belong to.
pub fn from_run(core: &Program, entry: FuncId, adjoints: &AdjointMap, sums: &BTreeMap<String, f64>) -> Sensitivity {
    let mut out = Sensitivity::default();
    for (fwd, (vjp, table)) in adjoints {
        let fwd_name = core.func(*fwd).name.clone();
        let vjp_name = core.func(*vjp).name.clone();
        for (fv, av) in table {
            let vjp_site = format!("{vjp_name}/%{}", av.0);
            let Some(s) = sums.get(&vjp_site) else { continue };
            let fwd_site = format!("{fwd_name}/%{}", fv.0);
            // A function called from several places contributes several
            // adjoints to one site; the largest is the one that decides.
            let e = out.sites.entry(fwd_site).or_insert(0.0);
            if *s > *e {
                *e = *s;
            }
        }
    }
    let _ = entry;
    out
}
