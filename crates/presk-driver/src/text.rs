//! What the commands print, as lines (plan 0.11.0 step 3): `presk
//! calibrate --precision`, `presk dump --stage sensitivity`, `presk check
//! --precision`, `presk solve` and `presk plan`. The command-line tool
//! prints them; `presk explain` quotes them, so that a document's figures
//! are the commands' own, character for character (X3). A line that holds a
//! time or a path the document does not print is the command's alone, and
//! stays in the command.

use presk_precision::envelope::Envelope;
use presk_precision::sensitivity::Sensitivity;

use crate::layout::{Captured, Planned, shape};
use crate::precision::{Bounded, Solved};

/// `presk calibrate --precision`: the sites, the five largest magnitudes, and
/// the sums whose terms cancel most, with their condition numbers, the
/// numbers the bound uses in place of `k` (0.6.0 step 6).
pub fn calibrate(env: &Envelope, entry: &str, bits: u32) -> Vec<String> {
    let mut out = vec![format!("presk calibrate: {} sites over {} samples of `{entry}` at {bits} bits", env.sites.len(), env.samples)];
    let mut widest: Vec<(&String, &f64)> = env.sites.iter().collect();
    widest.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (site, v) in widest.iter().take(5) {
        out.push(format!("  {site:<28} {v:.4e}"));
    }
    let mut conds: Vec<(&String, f64)> = env.terms.keys().filter_map(|s| env.condition(s).map(|c| (s, c))).collect();
    conds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if !conds.is_empty() {
        out.push(format!("presk calibrate: {} sums and contractions, worst condition numbers Σ|xᵢwᵢ| / |Σxᵢwᵢ|:", conds.len()));
        for (site, c) in conds.iter().take(5) {
            out.push(format!("  {site:<28} {c:.4e}"));
        }
    }
    out
}

/// `presk dump --stage sensitivity`: the sums by module, then the `top`
/// sites, the largest first, after an empty line.
pub fn sensitivity(s: &Sensitivity, entry: &str, bits: u32, top: usize) -> Vec<String> {
    let mut out = vec![format!("-- sensitivity of `{entry}`: ‖∂L/∂v‖₁ at each site, one reverse sweep at {bits} bits"), format!("{:<34} {:>14}", "module", "Σ ‖∂L/∂v‖₁")];
    for (m, v) in s.by_module() {
        out.push(format!("{m:<34} {v:>14.4e}"));
    }
    out.push(String::new());
    out.push(format!("{:<34} {:>14}", "site", "‖∂L/∂v‖₁"));
    for (site, v) in s.ranked().into_iter().take(top) {
        out.push(format!("{site:<34} {v:>14.4e}"));
    }
    out
}

/// `presk check --precision`: the output's bound by every rule run, the gap
/// when measured, the envelope (`env_path` as the command names it), and the
/// `sites` largest bounds of the first rule.
pub fn bound(env: &Envelope, env_path: &str, realize: &str, b: &Bounded, sites: usize) -> Vec<String> {
    let (measured, runs) = (b.measured, &b.runs);
    let a = &runs[0];
    let mut out = vec![format!("presk check --precision: `{}` under {realize}", env.entry), format!("  magnitude   {:.4e}", a.output.mag)];
    for r in runs {
        let what = if r.model.is_bound() { "bound   " } else { "estimate" };
        let ratio = match measured {
            Some(g) if g > 0.0 => format!("  {:>9.2e}× the measured gap", r.output.err / g),
            _ => String::new(),
        };
        out.push(format!("  {what} {:<9} {:.4e}  ({:.2e} relative){ratio}", r.model.name(), r.output.err, r.output.relative()));
    }
    if let Some(g) = measured {
        // Measured on the kernel interpreter, which computes the native
        // backend's bits (U5); until 0.11.0 step 7 the line said "native".
        out.push(format!("  measured             {:.4e}  ({:.2e} relative)  the kernel interpreter against the exact reference", g, if a.output.mag > 0.0 { g / a.output.mag } else { f64::NAN }));
    }
    out.push(format!("  envelope    {env_path} over {} samples, {} sites, {}", env.samples, a.sites.len(), env.weights.as_deref().map(|w| format!("weights {w}")).unwrap_or_else(|| "initial weights".into())));
    if !a.uncovered.is_empty() {
        out.push(format!("  uncovered   {} site(s) the envelope does not mention, first {}", a.uncovered.len(), a.uncovered.first().map(|s| s.as_str()).unwrap_or("-")));
    }
    if !a.unbounded.is_empty() {
        out.push(format!("  unbounded   {} site(s) divide by something with no measured floor: {}", a.unbounded.len(), a.unbounded.first().map(|s| s.as_str()).unwrap_or("-")));
    }
    if let Some(r) = runs.iter().find(|r| !r.fallback.is_empty()) {
        out.push(format!("  fallback    {} sum(s) with no recorded terms took the `k` rule, first {} (an envelope from before 0.6.0 step 6?)", r.fallback.len(), r.fallback[0]));
    }
    if sites > 0 {
        let mut rows: Vec<(&String, &presk_precision::propagate::Bound)> = a.sites.iter().collect();
        rows.sort_by(|x, y| y.1.err.partial_cmp(&x.1.err).unwrap_or(std::cmp::Ordering::Equal));
        out.push(format!("{:<30} {:>12} {:>12} {:>12} {:>12}", "site", "magnitude", a.model.name(), "relative", "condition"));
        for (site, b) in rows.iter().take(sites) {
            let c = env.condition(site).map(|c| format!("{c:.3e}")).unwrap_or_else(|| "-".into());
            out.push(format!("{site:<30} {:>12.3e} {:>12.3e} {:>12.3e} {c:>12}", b.mag, b.err, b.relative()));
        }
    }
    out
}

/// `presk solve`: the budget, the sites, the cost, the estimate and the
/// measured gap, the parameter menu's with `--quant`, the exhaustive
/// search's cost, and with `list` the sites assigned away from the default.
pub fn solve(env: &Envelope, budget: f64, solved: &Solved, list: usize) -> Vec<String> {
    let (sol, qsol, sens, ref_f64) = (&solved.solution, &solved.quant, &solved.sens, solved.reference_max);
    let mut out = vec![
        format!("presk solve: `{}` under budget {:.2e} relative ({:.3e} absolute, |output| {:.3e})", env.entry, budget, sol.budget_abs, ref_f64),
        format!("  sites       {} narrowable, {} assigned away from f64", sol.considered, sol.realization.storage.len()),
        format!("  cost        {:.4e} against {:.4e} all-f64 ({:.2}x), relaxation floor {:.4e}", sol.bytes, sol.baseline_bytes, sol.speedup(), sol.relaxation_bytes),
        format!("  proposal    {:.4e} before measurement widened anything (time when --profile is given, bytes otherwise)", sol.proposed_bytes),
        format!("  estimate    {:.3e} (sensitivity-weighted, first order: it ranks, it does not guarantee)", sol.estimated_error),
    ];
    match sol.measured {
        Some(g) => out.push(format!("  measured    {:.3e} against the exact reference, after {} widening round(s) — {}", g, sol.rounds, if g <= sol.budget_abs { "inside the budget" } else { "OUTSIDE the budget" })),
        None => out.push("  measured    not checked (--no-verify)".to_string()),
    }
    if let Some(q) = &qsol {
        let (n8, n4) = (q.chosen.values().filter(|b| **b == Some(8)).count(), q.chosen.values().filter(|b| **b == Some(4)).count());
        out.push(format!("  parameters  {} of {} matrices quantised ({n8} int8, {n4} int4): {} bytes → {} ({:.2}x less)", n8 + n4, q.candidates.len(), q.bytes_before, q.bytes_after, q.bytes_before as f64 / q.bytes_after.max(1) as f64));
        out.push(format!("  estimate    {:.3e} (√Σ(∂L/∂w·(ŵ−w))², first order, roundings taken as independent)", q.estimated));
        match q.measured {
            Some(g) => out.push(format!("  measured    {:.3e} against the exact reference with the weights quantised, after {} widening round(s) — {}", g, q.rounds, if g <= sol.budget_abs { "inside the budget" } else { "OUTSIDE the budget" })),
            None => out.push("  measured    not checked (--no-verify)".to_string()),
        }
        if list > 0 {
            out.push(format!("{:<28} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}", "parameter", "bits", "elements", "bound 8", "Σgδ 8", "rss 8", "bound 4", "Σgδ 4", "rss 4"));
            for c in &q.candidates {
                let b = q.chosen[&c.path].map(|b| format!("int{b}")).unwrap_or_else(|| sol.realization.default.name().into());
                out.push(format!("{:<28} {:>6} {:>10} {:>10.2e} {:>10.2e} {:>10.2e} {:>10.2e} {:>10.2e} {:>10.2e}", c.path, b, c.elems, c.est8, c.signed[0], c.rss[0], c.est4, c.signed[1], c.rss[1]));
            }
            let sum = |f: &dyn Fn(&presk_precision::quant::Candidate) -> f64| q.candidates.iter().map(f).sum::<f64>();
            out.push(format!("{:<28} {:>6} {:>10} {:>10.2e} {:>10.2e} {:>10.2e} {:>10.2e} {:>10.2e} {:>10.2e}", "all, combined", "", "", sum(&|c| c.est8), sum(&|c| c.signed[0]).abs(), sum(&|c| c.rss[0] * c.rss[0]).sqrt(), sum(&|c| c.est4), sum(&|c| c.signed[1]).abs(), sum(&|c| c.rss[1] * c.rss[1]).sqrt()));
        }
    }
    // Until 0.11.0 the line said bytes of a time too, and divided by at
    // least one, which made the ratio of two times 0.000x.
    if let Some(cost) = solved.optimal {
        let (unit, ratio) = if solved.time { ("s", if cost > 0.0 { sol.proposed_bytes / cost } else { 1.0 }) } else { ("bytes", sol.proposed_bytes / cost.max(1.0)) };
        out.push(format!("  optimal     {cost:.4e} {unit} by exhaustive search under the same estimate — the greedy proposal is {ratio:.3}x that"));
    } else {
        out.push("  optimal     not searched (the space is too large); the relaxation floor above is the lower bound".to_string());
    }
    if list > 0 && !sol.realization.storage.is_empty() {
        let mut rows: Vec<(&String, &presk_rt::Dtype)> = sol.realization.storage.iter().collect();
        rows.sort_by_key(|(k, _)| k.as_str());
        out.push(format!("{:<30} {:>8} {:>12} {:>12}", "site", "dtype", "elements", "‖∂L/∂v‖₁"));
        for (site, d) in rows.iter().take(list) {
            out.push(format!("{site:<30} {:>8} {:>12} {:>12.3e}", d.name(), env.elems(site), sens.get(site).unwrap_or(0.0)));
        }
    }
    out
}

/// `presk plan`'s table: its heading, each launch's row (with `cost`, every
/// point it priced first), and the line of a nest whose shapes do not all
/// choose the layout it takes, after its last launch.
pub fn plan_rows(captured: &Captured, planned: &Planned, cost: bool) -> Vec<String> {
    let mut rows = vec![format!("{:<28} {:<24} {:>12} {:>13}  {:>17}  layout", "nest", "operands", "space", "predicted", "undecided")];
    for (i, c) in planned.costed.iter().enumerate() {
        let l = &captured.launches[c.launch];
        if cost {
            for (t, p) in &c.scored {
                rows.push(format!("  {:<28} {:>12.2} µs  {p}", l.nest, t * 1e6));
            }
        }
        let (best_t, best) = &c.scored[0];
        rows.push(format!("{:<28} {:<24} {:>5} points  {:>10.2} µs  {:>3} within 1 + u  {best}", l.nest, shape(l), c.space, best_t * 1e6, c.kept));
        let last = !planned.costed[i + 1..].iter().any(|m| captured.launches[m.launch].nest == l.nest);
        if last && let Some(line) = planned.summary.get(&l.nest) {
            rows.push(line.clone());
        }
    }
    rows
}

/// `presk plan`'s totals: the nests and the points costed, with the time the
/// pricing and the capture took when `seconds` is given (the command's
/// line; a document has no time), the epilogues, and the nests the profile
/// could not price.
pub fn plan_totals(captured: &Captured, planned: &Planned, u: f64, seconds: bool) -> Vec<String> {
    let mut out = Vec::new();
    let all_nests = captured.launches.iter().map(|l| l.nest.as_str()).collect::<std::collections::BTreeSet<_>>().len();
    let shapes = if all_nests == captured.launches.len() { String::new() } else { format!(" at {} shapes", captured.launches.len()) };
    if seconds {
        out.push(format!("-- {all_nests} nests{shapes}, {} points costed in {:.3} s, the entry captured in {:.3} s; u = {u:.3}", planned.total_points, planned.seconds, captured.seconds));
    } else {
        out.push(format!("-- {all_nests} nests{shapes}, {} points costed; u = {u:.3}", planned.total_points));
    }
    let epilogues = &planned.epilogues;
    if !epilogues.is_empty() {
        out.push(format!("-- {} product{} the `add` that alone reads {} as {} epilogue: {}", epilogues.len(), if epilogues.len() == 1 { " takes" } else { "s take" }, if epilogues.len() == 1 { "it" } else { "them" }, if epilogues.len() == 1 { "its" } else { "their" }, epilogues.join(", ")));
    }
    let unpriced: Vec<&presk_kernel::Launch> = planned.unpriced.iter().map(|i| &captured.launches[*i]).collect();
    if !unpriced.is_empty() {
        let mut names: Vec<String> = Vec::new();
        for l in &unpriced {
            let n = unpriced.iter().filter(|m| m.nest == l.nest).count();
            let name = if n > 1 { format!("{} ({}, {n} shapes)", l.nest, l.dtype) } else { format!("{} ({})", l.nest, l.dtype) };
            if !names.contains(&name) {
                names.push(name);
            }
        }
        out.push(format!("-- left to the default layout, the profile being calibrated in {} only: {}", planned.types.join(" and "), names.join(", ")));
    }
    out
}

/// `presk dump --stage kernel`'s two first lines: the kernels and launches
/// under the realization, and what fusion saved.
pub fn kernel_heading(listing: &crate::layout::Listing) -> Vec<String> {
    let before = listing.launches + listing.fused;
    let pct = if before > 0 { 100.0 * listing.fused as f64 / before as f64 } else { 0.0 };
    vec![format!("-- {} kernels, {} launches, {}", listing.kernels.len(), listing.launches, listing.label), format!("-- fusion: {before} launches without it, {} saved ({pct:.0}%)", listing.fused)]
}

/// `presk bench --host-profile`: the call with and without the timers, the
/// two outputs compared, the page faults, and the table of the phases.
pub fn host_profile(entry: &str, realize: &str, planned: bool, h: &crate::bench::HostProfiled) -> String {
    let med = |v: &[f64]| {
        let mut v = v.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (v[v.len() / 2], v[0], v[v.len() - 1])
    };
    let (pm, plo, phi) = med(&h.plain);
    let (tm, tlo, thi) = med(&h.timed.iter().map(|(_, s)| *s).collect::<Vec<_>>());
    let mut out = format!(
        "presk bench --host-profile: `{entry}` on the native backend, {realize} {}, {} threads; {} calls timed by phase and {} not, alternated, after {} warm-up\n",
        if planned { "under the plan" } else { "at the default layout" },
        h.workers,
        h.timed.len(),
        h.plain.len(),
        h.warmup
    );
    out.push_str(&format!(
        "-- a call: {:.3} ms without the timers (median; {:.3}–{:.3}), {:.3} ms with them ({:.3}–{:.3}): the timers' cost {:+.3} ms, {:+.2} %\n",
        pm * 1e3,
        plo * 1e3,
        phi * 1e3,
        tm * 1e3,
        tlo * 1e3,
        thi * 1e3,
        (tm - pm) * 1e3,
        100.0 * (tm - pm) / pm
    ));
    out.push_str(&format!("-- the outputs of a call with the timers and of one without: {}\n", if h.same { "the same bits" } else { "DIFFERENT" }));
    if !h.plain_faults.is_empty() {
        out.push_str(&format!("-- without the timers, {:.0} minor page faults a call\n", h.plain_faults.iter().sum::<u64>() as f64 / h.plain_faults.len() as f64));
    }
    out.push_str(&presk_kernel::profile::table(&h.timed, "call"));
    out
}
