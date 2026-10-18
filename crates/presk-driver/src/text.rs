//! What the commands print, as lines (plan 0.11.0 step 3): `presk
//! calibrate --precision`, `presk dump --stage sensitivity`, `presk check
//! --precision`, `presk solve`, `presk plan` and `presk memory` (plan
//! 0.15.0 step 1). The command-line tool
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
pub fn solve(env: &Envelope, budget: presk_precision::solve::Budget, solved: &Solved, list: usize) -> Vec<String> {
    use presk_precision::solve::Budget;
    let (sol, qsol, sens, ref_f64) = (&solved.solution, &solved.quant, &solved.sens, solved.reference_max);
    // A tensor entry's budget is on its largest elementwise gap (plan
    // 0.12.0 step 4).
    let over = if solved.outputs > 1 { format!(" on the largest elementwise gap of its {} outputs", solved.outputs) } else { String::new() };
    let head = match budget {
        Budget::Relative(b) => format!("presk solve: `{}` under budget {:.2e} relative{over} ({:.3e} absolute, |output| {:.3e})", env.entry, b, sol.budget_abs, ref_f64),
        Budget::Absolute(b) => format!("presk solve: `{}` under budget {:.3e} absolute{over} ({:.2e} relative to |output| {:.3e})", env.entry, b, b / ref_f64.max(f64::MIN_POSITIVE), ref_f64),
    };
    let params: Vec<&presk_precision::solve::SiteCost> = solved.breakdown.iter().filter(|c| c.param).collect();
    let mut out = vec![head, format!("  sites       {} narrowable, {} assigned away from f64", sol.considered - params.len(), sol.realization.storage.len())];
    if !params.is_empty() {
        let typed = |d: presk_rt::Dtype| sol.realization.params.values().filter(|t| **t == d).count();
        let at_rest = |c: &&presk_precision::solve::SiteCost| c.elems * c.dtype.bytes() as f64;
        out.push(format!(
            "  parameters  {} offered, {} stored away from f64 ({} f32, {} bf16): {:.4e} bytes at rest against {:.4e} in f64",
            params.len(),
            sol.realization.params.len(),
            typed(presk_rt::Dtype::F32),
            typed(presk_rt::Dtype::Bf16),
            params.iter().map(at_rest).sum::<f64>(),
            params.iter().map(|c| c.elems * 8.0).sum::<f64>()
        ));
    }
    let per = if solved.outputs > 1 { "first order, of a mean output element" } else { "first order" };
    out.extend([
        format!("  cost        {:.4e} against {:.4e} all-f64 ({:.2}x), relaxation floor {:.4e}", sol.bytes, sol.baseline_bytes, sol.speedup(), sol.relaxation_bytes),
        format!("  proposal    {:.4e} before measurement widened anything (time when --profile is given, bytes otherwise)", sol.proposed_bytes),
        format!("  estimate    {:.3e} (sensitivity-weighted, {per}: it ranks, it does not guarantee)", sol.estimated_error),
    ]);
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
    if list > 0 && !sol.realization.params.is_empty() {
        out.push(format!("{:<30} {:>8} {:>12} {:>12}", "parameter", "dtype", "elements", "‖∂L/∂θ‖₁"));
        for c in params.iter().filter(|c| sol.realization.params.contains_key(&c.name)).take(list) {
            out.push(format!("{:<30} {:>8} {:>12} {:>12.3e}", c.name, c.dtype.name(), c.elems, c.sens));
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

/// `presk memory` (plan 0.15.0 step 1): the parameters, the inputs, the peak
/// of live bytes and what is needed, the C program's arena beside them, and
/// the calls' shares of the peak, the deepest `depth` levels of calls.
pub fn memory(m: &crate::memory::Memory, depth: usize) -> Vec<String> {
    let l = &m.lifetimes;
    let dims = if m.dims.is_empty() { "no dimension".to_string() } else { m.dims.iter().map(|(n, v)| format!("{n} = {v}")).collect::<Vec<_>>().join(", ") };
    let ops = l.live.len();
    let mut out = vec![format!("presk memory: `{}` at {dims} in {}: {ops} operations, {} launches", m.entry, m.realize, m.launches)];
    let (count, bytes) = m.parameters();
    if !m.params.is_empty() {
        let mut by: Vec<(String, usize)> = Vec::new();
        for p in &m.params {
            match by.iter_mut().find(|(s, _)| *s == p.store) {
                Some(e) => e.1 += p.elements,
                None => by.push((p.store.clone(), p.elements)),
            }
        }
        let types = by.iter().map(|(s, n)| format!("{n} in {s}")).collect::<Vec<_>>().join(", ");
        out.push(format!("  parameters  {bytes} bytes: {count} elements, {types}"));
    }
    out.push(format!("  inputs      {} bytes{}", l.inputs, if m.params.is_empty() { "" } else { ", the parameters among them" }));
    let at = match (ops, m.peak_call()) {
        (0, _) => String::new(),
        (_, Some(c)) => format!(", at operation {} of {ops}, `{}` in {c}", l.at + 1, m.peak_site()),
        (_, None) => format!(", at operation {} of {ops}, `{}`", l.at + 1, m.peak_site()),
    };
    out.push(format!("  peak        {} bytes live{at}", l.peak));
    out.push(format!("  outputs     {} bytes", l.outputs));
    out.push(format!("  needs       {} bytes: the inputs and the peak", m.needs()));
    match &m.arena {
        Ok(a) => {
            let ratio = if l.peak == 0 { String::new() } else { format!(", {:.2} times the peak", a.beside() as f64 / l.peak as f64) };
            out.push(format!("  C arena     {} bytes, {} beside the inputs{ratio}: the frames of `presk export --target c`, each function's buffers held until it returns ({} released nothing)", a.total, a.beside(), a.flat));
        }
        Err(e) => out.push(format!("  C arena     none: {e}")),
    }
    out.push(format!("  placed      {} bytes beside the inputs: every buffer at an offset by its life, rounded to {} (`--straight-line --placed`)", m.placed, crate::memory::ALIGN));
    let shown: Vec<&crate::memory::CallMemory> = m.calls.iter().filter(|c| c.path.len() <= depth).collect();
    if !shown.is_empty() {
        let n = m.calls.len();
        out.push(if n == 1 { "presk memory: 1 call, its share of the peak and its own peak".to_string() } else { format!("presk memory: {n} calls, their shares of the peak and their own peaks") });
        let names: Vec<String> = shown.iter().map(|c| format!("{}{}{}", "  ".repeat(c.path.len() - 1), c.params.clone().unwrap_or_else(|| format!("`{}`", c.func)), if c.runs > 1 { format!(" ×{}", c.runs) } else { String::new() })).collect();
        let itself = format!("`{}` itself", m.entry);
        let w = names.iter().chain(std::iter::once(&itself)).map(|n| n.chars().count()).max().unwrap_or(0).max(4);
        out.push(format!("  {:<w$}  {:>12}  {:>12}", "call", "share", "own peak"));
        out.push(format!("  {:<w$}  {:>12}", itself, m.own_share()));
        for (c, n) in shown.iter().zip(&names) {
            out.push(format!("  {n:<w$}  {:>12}  {:>12}", c.share, c.own, w = w + n.len() - n.chars().count()));
        }
    }
    out
}

/// `presk memory --step` (plan 0.15.0 step 2): a training step under each
/// `--remat`, side by side, its terms, each call's phase and the step's
/// peak; then the peak's terms under each.
pub fn step(steps: &[crate::memory::StepMemory]) -> Vec<String> {
    let Some(first) = steps.first() else { return Vec::new() };
    let c0 = &first.calls[0];
    let dims = if c0.dims.is_empty() { "no dimension".to_string() } else { c0.dims.iter().map(|(n, v)| format!("{n} = {v}")).collect::<Vec<_>>().join(", ") };
    let names: Vec<&str> = first.calls.iter().map(|c| c.entry.as_str()).collect();
    let mut out = vec![format!("presk memory --step: the training run's step at {dims} in {}, as presk train makes it: `{}`", c0.realize, names.join("`, then `"))];
    let remat = |s: &crate::memory::StepMemory| if s.remat == presk_core::ad::Remat::Store { "store" } else { "recompute" };
    let mut rows: Vec<(String, Vec<String>)> = vec![
        ("parameters".into(), steps.iter().map(|s| s.params.to_string()).collect()),
        ("moments, AdamW's m and v".into(), steps.iter().map(|s| s.moments.to_string()).collect()),
        ("the gradient".into(), steps.iter().map(|s| s.gradients.to_string()).collect()),
        ("the batch, the key and the seed".into(), steps.iter().map(|s| s.others.to_string()).collect()),
        ("kept for the backward".into(), steps.iter().map(|s| s.kept.to_string()).collect()),
    ];
    for (i, n) in names.iter().enumerate() {
        rows.push((format!("`{n}`: its peak"), steps.iter().map(|s| s.calls[i].lifetimes.peak.to_string()).collect()));
        rows.push((format!("`{n}`: with what the loop holds"), steps.iter().map(|s| s.phases[i].to_string()).collect()));
    }
    rows.push(("the step's peak".into(), steps.iter().map(|s| s.peak.to_string()).collect()));
    let w = rows.iter().map(|(n, _)| n.chars().count()).max().unwrap_or(0);
    out.push(format!("  {:<w$}{}", "", steps.iter().map(|s| format!("{:>12}", remat(s))).collect::<String>()));
    for (n, v) in rows {
        let pad = w + n.len() - n.chars().count();
        out.push(format!("  {n:<pad$}{}", v.iter().map(|x| format!("{x:>12}")).collect::<String>()));
    }
    for s in steps {
        let m = &s.calls[s.peak_call];
        let terms = s.terms.iter().map(|t| format!("{} {}", t.name, t.bytes)).collect::<Vec<_>>().join(", ");
        out.push(format!("presk memory --step: under {}, the peak is `{}`'s, at its operation {}: {terms}", remat(s), m.entry, m.lifetimes.at + 1));
    }
    out
}

/// `presk memory --symbolic` (plan 0.15.0 step 3): the inputs, the outputs
/// and the peak as polynomials in the entry's dimensions, `render` printing
/// one (the printer of the denotation's form, `presk-print-math`), the
/// peak the largest of its candidates when no one dominates the others.
pub fn symbolic(m: &crate::memory::Memory, s: &crate::memory::Symbolic, render: &dyn Fn(&crate::memory::Poly) -> String) -> Vec<String> {
    let grid = s.names.iter().zip(&s.grid).map(|(n, g)| format!("{n} ∈ {{{}}}", g.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "))).collect::<Vec<_>>().join(", ");
    let at = |x: &[u64]| s.names.iter().zip(x).map(|(n, v)| format!("{n} = {v}")).collect::<Vec<_>>().join(", ");
    let dims = if s.names.is_empty() { "no dimension".to_string() } else { s.names.join(" and ") };
    let mut out = vec![format!(
        "presk memory --symbolic: `{}` in {}, in {dims}: each buffer's bytes interpolated on {}, checked at {}",
        m.entry,
        m.realize,
        if grid.is_empty() { "one point".to_string() } else { grid },
        if s.check.is_empty() { "it".to_string() } else { at(&s.check) }
    )];
    out.push(format!("  inputs      {}", render(&s.inputs)));
    out.push(format!("  outputs     {}", render(&s.outputs)));
    let site = |k: usize| m.sites.get(k).map(String::as_str).unwrap_or("");
    match &s.peaks[..] {
        [(k, p)] => out.push(format!("  peak        {}, at operation {} (`{}`)", render(p), k + 1, site(*k))),
        many => {
            out.push(format!("  peak        the largest of {}, none of which dominates another:", many.len()));
            for (k, p) in many {
                out.push(format!("                {}, at operation {} (`{}`)", render(p), k + 1, site(*k)));
            }
        }
    }
    let x: Vec<u64> = m.dims.iter().filter(|(n, _)| s.names.contains(n)).map(|(_, v)| *v).collect();
    if x.len() == s.names.len() {
        let x: Vec<u64> = s.names.iter().map(|n| m.dims.iter().find(|(d, _)| d == n).map(|(_, v)| *v).unwrap_or(0)).collect();
        out.push(format!("  at {}: the peak {}, the inputs {}, as counted", at(&x), s.peak_at(&x).0, s.inputs.eval(&x)));
    }
    out
}
