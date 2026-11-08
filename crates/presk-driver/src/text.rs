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
        // backend's bits (U5), the transcendental functions included since
        // 0.16.0 (D159); until 0.11.0 step 7 the line said "native".
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

/// The lines `rules` of `presk solve` (0.16.1, D169): the values of one
/// element, the values under rule 2's threshold with their first step's
/// gain, and what the two forgo at most against the realization's traffic.
/// Nothing when the rules leave every value offered.
pub fn rules(sol: &presk_precision::solve::Solution) -> Vec<String> {
    use presk_precision::solve::Rule;
    let o = &sol.offers;
    if o.excluded.is_empty() {
        return Vec::new();
    }
    let names = |r: Rule, gain: bool| o.by(r).map(|e| if gain { format!("{} {:.3e}", e.name, e.gain) } else { e.name.clone() }).collect::<Vec<_>>().join(", ");
    let mut out = Vec::new();
    let one = o.by(Rule::OneElement).count();
    if one > 0 {
        out.push(format!("  rules       {one} of one element not offered (rule 1): {}", names(Rule::OneElement, false)));
    }
    let low = o.by(Rule::MinGain).count();
    if low > 0 {
        let lead = if one > 0 { "            " } else { "  rules     " };
        out.push(format!(
            "{lead}  {low} not offered whose first narrowing saves under θ·B = {:.0e} × {:.4e} = {:.4e} bytes (rule 2, --min-gain), the bytes it saves: {}",
            o.min_gain,
            o.base,
            o.threshold(),
            names(Rule::MinGain, true)
        ));
    }
    let share = if sol.traffic > 0.0 { o.forgone() / sol.traffic } else { 0.0 };
    out.push(format!("              at most {:.4e} bytes forgone (their saving in the menu's narrowest type), {:.2} % of the realization's traffic, {:.4e} bytes", o.forgone(), 100.0 * share, sol.traffic));
    out
}

/// The line `menu` of `presk solve` with `--menu`'s integers (D177): the
/// rungs, how many parameters were priced in them, and on which axes their
/// scales are, the one their readers contract or −2.
fn menu_line(solved: &Solved) -> Option<String> {
    let m = &solved.menu;
    if m.1.is_empty() {
        return None;
    }
    let rungs: Vec<String> = m.rungs().map(|r| r.name()).collect();
    let priced: Vec<&presk_precision::solve::ParamSite> = solved.params.iter().filter(|p| !p.ints.is_empty()).collect();
    let contracted = priced.iter().filter(|p| p.ints.iter().all(|q| q.contracted)).count();
    Some(format!(
        "  menu        {}: {} parameter(s) of rank two or more priced in integers, {contracted} scaled along the axis their readers contract, {} along −2 (D177)",
        rungs.join(", "),
        priced.len(),
        priced.len() - contracted
    ))
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
    // The values the rules of 0.16.1 do not offer (D169) are counted apart.
    let offers = &sol.offers;
    let (out_sites, out_params) = (offers.excluded.iter().filter(|e| !e.param).count(), offers.excluded.iter().filter(|e| e.param).count());
    let stored = sol.considered - params.len();
    let of = |n: usize, out: usize, what: &str| if out > 0 { format!(" ({n} {what}, {out} not offered)") } else { String::new() };
    let mut out = vec![head, format!("  sites       {} narrowable, {} assigned away from f64{}", stored - out_sites, sol.realization.storage.len(), of(stored, out_sites, "stored values"))];
    if !params.is_empty() {
        let typed = |d: presk_rt::Dtype| sol.realization.params.values().filter(|t| **t == d).count();
        // The one menu's integers (D177), said when it decided some: a
        // `--quant` realization's are the second pass's, below.
        let ints: Vec<&&presk_precision::solve::SiteCost> = params.iter().filter(|c| c.int.is_some()).collect();
        let int_typed = |bits: u8| ints.iter().filter(|c| c.int.as_ref().is_some_and(|q| q.bits == bits)).count();
        let int_part = if ints.is_empty() { String::new() } else { format!(", {} int8, {} int4", int_typed(8), int_typed(4)) };
        out.push(format!(
            "  parameters  {} offered{}, {} stored away from f64 ({} f32, {} bf16{int_part}): {:.4e} bytes at rest against {:.4e} in f64",
            params.len() - out_params,
            if out_params > 0 { format!(" of {}", params.len()) } else { String::new() },
            sol.realization.params.len() + ints.len(),
            typed(presk_rt::Dtype::F32),
            typed(presk_rt::Dtype::Bf16),
            params.iter().map(|c| c.bytes).sum::<f64>(),
            params.iter().map(|c| c.elems * 8.0).sum::<f64>()
        ));
        if let Some(m) = menu_line(solved) {
            out.push(m);
        }
    }
    out.extend(rules(sol));
    let per = if solved.outputs > 1 { "first order, of a mean output element" } else { "first order" };
    out.extend([
        format!("  cost        {:.4e} against {:.4e} all-f64 ({:.2}x), relaxation floor {:.4e}", sol.bytes, sol.baseline_bytes, sol.speedup(), sol.relaxation_bytes),
    ]);
    // The cost per sample (D157), said where the envelope has several.
    if !env.per_sample.is_empty() {
        out.push("  per sample  the activations at one sample's elements and the parameters once, the same cost at any number of samples (D157)".to_string());
    } else if !env.priced_per_sample() {
        out.push("  per sample  no: an envelope of several samples written before 0.16.0, without their per-sample counts, prices the activations at its batch".to_string());
    }
    out.extend([
        format!("  proposal    {:.4e} before measurement widened anything (time when --profile is given, bytes otherwise)", sol.proposed_bytes),
        format!("  estimate    {:.3e} (sensitivity-weighted, {per}: it ranks, it does not guarantee)", sol.estimated_error),
    ]);
    match sol.measured {
        Some(g) => out.push(format!("  measured    {:.3e} against the exact reference, after {} widening round(s) — {}", g, sol.rounds, if g <= sol.budget_abs { "inside the budget" } else { "OUTSIDE the budget" })),
        None => out.push("  measured    not checked (--no-verify)".to_string()),
    }
    // The options of D155: the rounding term, whichever it is (the measured
    // one the default since 0.16.0, D157), and the narrowing when it is on.
    match sol.options.rounding {
        presk_precision::solve::RoundingTerm::Measured => out.push(format!(
            "  rounding    measured: each term s·max|fl_d(x) − x|, the sampled values rounded (D155), at the {} sites the envelope records it for and every parameter; s·u·|v| elsewhere",
            env.rounding.len()
        )),
        presk_precision::solve::RoundingTerm::Unit => out.push("  rounding    unit: each term s·u(d)·|v|, the solver of 0.12.0–0.15.0 (--rounding unit)".to_string()),
    }
    // The ways to spend the budget of D161, said when they are on.
    if sol.options.estimate == presk_precision::solve::Estimate::Rss {
        out.push(format!(
            "  rss         the proposal's estimate √Σ term², the roundings taken as independent, aimed at the budget over the margin {}: {:.3e} (D161)",
            sol.options.margin,
            sol.budget_abs / sol.options.margin
        ));
    }
    if sol.options.narrow != presk_precision::solve::Narrow::Off {
        out.push(format!(
            "  narrowing   {} decision(s) narrowed one step by measurement in {} pass(es), each kept under the budget over the margin {}, {:.3e}; {} measurements in all (D155, D161)",
            sol.narrowed, sol.narrow_passes, sol.options.margin, sol.budget_abs / sol.options.margin, sol.measurements
        ));
    }
    // The executors of D185, said when they are not the solver of record's.
    if solved.exec != crate::precision::SolveExec::default() {
        let sweep = match solved.exec.sweep {
            crate::precision::Sweep::Exact => "the sweep on the reference interpreter",
            crate::precision::Sweep::F64 => "the sweep in binary64 (--sweep f64)",
        };
        let measure = match solved.exec.measure {
            crate::precision::Measure::Kernel => "the measurements on the kernel interpreter",
            crate::precision::Measure::Native => "the measurements on the native backend, the kernel interpreter's bits (--measure native)",
        };
        out.push(format!("  executors   {sweep}; {measure}; the exact reference at its bits either way (D185)"));
    }
    if sol.options.narrow == presk_precision::solve::Narrow::Heldout {
        let which = sol.options.heldout.map(|m| format!("{m} after the envelope's")).unwrap_or_else(|| "as many after the envelope's as its own".into());
        match sol.heldout {
            Some(g) if g <= sol.budget_abs / sol.options.margin => out.push(format!("  held out    {} narrowing(s) undone until the held-out samples ({which}) were inside the threshold: {:.3e} there (D161)", sol.undone, g)),
            Some(g) => out.push(format!("  held out    every narrowing undone ({}), the held-out samples ({which}) still at {:.3e}, above the threshold (D161)", sol.undone, g)),
            None => out.push("  held out    not measured: no narrowing ran".to_string()),
        }
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
    // The one menu's pricing of every parameter in integers (D177), at
    // rest: the term each rung would add, beside its float type's.
    if list > 0 && !solved.menu.1.is_empty() && solved.params.iter().any(|p| !p.ints.is_empty()) {
        let rungs: Vec<String> = solved.menu.1.iter().map(|(bits, scheme)| presk_precision::solve::Rung::Int { bits: *bits, scheme: *scheme }.name()).collect();
        let mut head = format!("{:<28} {:>5} {:>10} {:>10}", "parameter", "axis", "‖∂L/∂θ‖₁", "f32");
        for r in &rungs {
            head.push_str(&format!(" {r:>10}"));
        }
        out.push(head);
        for p in solved.params.iter().filter(|p| !p.ints.is_empty()) {
            let axis = p.ints.first().map(|q| format!("{}{}", q.axis, if q.contracted { "" } else { "*" })).unwrap_or_default();
            let f32_term = p.round.map(|r| p.sens * r.f32).unwrap_or(f64::NAN);
            let mut line = format!("{:<28} {:>5} {:>10.3e} {:>10.3e}", p.path, axis, p.sens, f32_term);
            for (bits, scheme) in &solved.menu.1 {
                match p.ints.iter().find(|q| q.bits == *bits && q.scheme == *scheme) {
                    Some(q) => line.push_str(&format!(" {:>10.3e}", p.sens * q.round)),
                    None => line.push_str(&format!(" {:>10}", "-")),
                }
            }
            out.push(line);
        }
        out.push("  (each rung's term at rest, s·max|ŵ − w|, and f32's, s·max|fl(w) − w|; * scaled along −2, its readers contracting no single axis)".to_string());
    }
    if list > 0 && (!sol.realization.params.is_empty() || params.iter().any(|c| c.int.is_some())) {
        out.push(format!("{:<30} {:>8} {:>12} {:>12}", "parameter", "dtype", "elements", "‖∂L/∂θ‖₁"));
        for c in params.iter().filter(|c| sol.realization.params.contains_key(&c.name) || c.int.is_some()).take(list) {
            out.push(format!("{:<30} {:>8} {:>12} {:>12.3e}", c.name, c.rung.name(), c.elems, c.sens));
        }
    }
    out
}

/// `presk calibrate --precision`'s rounding (D155): beside each site's
/// magnitude, its largest rounding error in `f32` and `bf16` and that over
/// the unit-roundoff term `u·max|x|`, for the sites where the two differ
/// most and those exact in `bf16`, after [`calibrate`]'s lines, by the
/// command and by `presk explain`'s envelope section alike (X3).
pub fn rounding(env: &Envelope) -> Vec<String> {
    use presk_rt::Dtype;
    if env.rounding.is_empty() {
        return Vec::new();
    }
    let ratio = |site: &str| -> f64 {
        let (m, r) = (env.get(site).unwrap_or(0.0), env.rounding(site, Dtype::Bf16).unwrap_or(0.0));
        if m > 0.0 { r / (Dtype::Bf16.unit_roundoff() * m) } else { 1.0 }
    };
    let mut sites: Vec<&String> = env.rounding.keys().filter(|s| env.get(s).is_some_and(|m| m > 0.0)).collect();
    sites.sort_by(|a, b| ratio(a).partial_cmp(&ratio(b)).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b)));
    let exact = sites.iter().filter(|s| env.rounding(s, Dtype::Bf16) == Some(0.0)).count();
    let under = sites.iter().filter(|s| ratio(s) < 0.1).count();
    let mut out = vec![format!(
        "presk calibrate: the largest rounding error of each site's values (D155): {} sites, {exact} exact in bf16, {under} below a tenth of u·max|x| there",
        sites.len()
    )];
    out.push(format!("  {:<28} {:>11} {:>11} {:>11} {:>9}", "site", "max|x|", "f32", "bf16", "bf16/u|x|"));
    for site in sites.iter().take(10) {
        out.push(format!(
            "  {:<28} {:>11.4e} {:>11.4e} {:>11.4e} {:>9.3}",
            site,
            env.get(site).unwrap_or(0.0),
            env.rounding(site, Dtype::F32).unwrap_or(0.0),
            env.rounding(site, Dtype::Bf16).unwrap_or(0.0),
            ratio(site)
        ));
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
