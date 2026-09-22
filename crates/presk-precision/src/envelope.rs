//! The magnitude envelope: how large the values of a program actually get.
//!
//! The first-order error analysis of the next step needs, for every value, a
//! bound on `|v|` as well as on its distance to the exact value: an absolute
//! error is meaningless without a scale, and a *relative* one is unbounded
//! wherever a result passes near zero. Static interval arithmetic cannot
//! supply the scale for a neural network — weights are unbounded and the
//! intervals grow with depth until they say nothing — so it is **measured**,
//! on the exact interpreter, over a stated number of batches.
//!
//! Every bound computed from an envelope is therefore conditional on it, and
//! is printed with it (decision D35). That is a weaker claim than it first
//! sounds, and stating it in the tool's own output is the answer rather than
//! a footnote: presk bounds the distance between an execution and the
//! denotation *over an observed input domain*.
//!
//! What makes the claim usable is that the envelope is **stable**: two
//! disjoint sets of samples give magnitudes within a small factor of each
//! other. That is an assumption the whole analysis rests on, so
//! [`Envelope::agreement`] measures it rather than assuming it.

use std::collections::BTreeMap;

/// `max |x|` per site, with what it was measured on.
///
/// A site is `"<function>/%<let>"`, the same key a [`crate::realization`]
/// assigns a type to, so an envelope and a realization can be read together.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Envelope {
    /// The program the magnitudes were measured on.
    pub program: String,
    /// The entry that was run.
    pub entry: String,
    /// How many samples were folded in. A bound quoting this envelope quotes
    /// this number with it.
    pub samples: usize,
    /// The day it was measured, `YYYY-MM-DD` in UTC ([`today`]).
    pub measured: String,
    /// Precision of the reference interpreter, in bits.
    pub bits: u32,
    /// The dimensions and constant overrides the magnitudes were measured
    /// at. A bound must be computed at the same shapes — `k` in a
    /// contraction's rule is a shape — so they travel with the envelope
    /// rather than being supplied again and possibly differently.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dims: Vec<(String, u64)>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub consts: Vec<String>,
    pub sites: BTreeMap<String, f64>,
    /// The smallest non-zero `|x|` at each site, when one was seen. A
    /// quotient's error grows as its divisor shrinks, so a maximum alone
    /// cannot bound one (0.5.0 step 4). Absent from envelopes written before
    /// that step, which is why it defaults to empty rather than failing.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub minima: BTreeMap<String, f64>,
    /// Real elements at each site. Traffic is elements times bytes per
    /// element, so this is what a narrower type actually saves.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub elems: BTreeMap<String, u64>,
    /// The primitive each site computes, so the solver can price *time* and
    /// not only traffic: the roofline needs `W`, and `W` is elements times
    /// the weight of the operation (0.6.0 step 4).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub ops: BTreeMap<String, String>,
    /// At each sum and contraction, what its terms add up to without
    /// cancelling (0.6.0 step 6, D51). `k·max|x|·max|w|` is what a rule
    /// knows without it; this is what the program actually does. Absent
    /// from envelopes written before that step, and a site without it falls
    /// back to the `k` rule.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub terms: BTreeMap<String, Terms>,
    /// The weights the magnitudes were measured on, when they were not the
    /// program's initial ones. A trained network is a different function,
    /// and a bound or a gap quoted beside this envelope has to be about the
    /// same one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weights: Option<String>,
    /// The data file the entry's last argument was read from, and which
    /// samples: a gap measured beside the bound has to be on the same inputs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<DataSlice>,
    /// When the envelope is of a training **step** rather than an entry
    /// (0.6.0 step 7b): how its state was built, so that `presk solve
    /// --step` rebuilds the same state and measures against the same exact
    /// step. `entry` is then `"step"` and the sites are the reverse sweep's
    /// and the optimizer's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<StepSpec>,
}

/// The state of a measured training step: a checkpoint, `warmup` steps of
/// the run's pipeline in `f64`, then step `warmup + 1` of `total` on
/// `data`'s samples (the envelope's `data`).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StepSpec {
    pub warmup: u64,
    pub total: u64,
    pub seed: u64,
}

/// `samples` entries of a `.psk.d` file, starting at `from`.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DataSlice {
    pub path: String,
    pub from: usize,
    pub samples: Option<usize>,
}

/// The sums of one sum's or contraction's terms, each a maximum over output
/// elements and samples.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Terms {
    /// `Σ|xᵢwᵢ|`, the numerator of the condition number `Σ|xᵢwᵢ| / |Σxᵢwᵢ|`.
    pub abs: f64,
    /// `√Σ(xᵢwᵢ)²`: what the terms' errors add up to when they are
    /// independent, the `√k` of a statistical model.
    pub rss: f64,
    /// Per operand, the factor its error is multiplied by in a sound
    /// normwise rule: `Σ|wᵢ|` for `x` in `Σxᵢwᵢ`. Empty for a plain sum.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub norms: Vec<f64>,
}

/// Today's date in UTC, `YYYY-MM-DD`: the day an envelope is measured on.
/// Until 0.6.0 the CLI wrote a constant here instead.
pub fn today() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let (y, m, d) = civil_from_days((secs / 86_400) as i64);
    format!("{y:04}-{m:02}-{d:02}")
}

/// The proleptic Gregorian date of a day counted from 1970-01-01 (H. Hinnant's
/// `civil_from_days`): shifted to 400-year eras whose years start on 1 March,
/// so that a leap day ends its year.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

impl Envelope {
    pub fn new(program: &str, entry: &str, bits: u32, measured: &str) -> Envelope {
        Envelope { program: program.into(), entry: entry.into(), samples: 0, measured: measured.into(), bits, dims: Vec::new(), consts: Vec::new(), sites: BTreeMap::new(), minima: BTreeMap::new(), elems: BTreeMap::new(), ops: BTreeMap::new(), terms: BTreeMap::new(), weights: None, data: None, step: None }
    }

    /// Fold one run's magnitudes in: the envelope of a set of samples is the
    /// pointwise maximum, since it must cover every one of them.
    pub fn observe(&mut self, m: &BTreeMap<String, f64>) {
        for (site, v) in m {
            let e = self.sites.entry(site.clone()).or_insert(0.0);
            if v > e {
                *e = *v;
            }
        }
        self.samples += 1;
    }

    /// Fold one run's minima in: the pointwise *minimum*, since a floor must
    /// hold for every sample just as the ceiling must.
    pub fn observe_minima(&mut self, m: &BTreeMap<String, f64>) {
        for (site, v) in m {
            let e = self.minima.entry(site.clone()).or_insert(f64::INFINITY);
            if v < e {
                *e = *v;
            }
        }
    }

    /// Fold one run's element counts in.
    pub fn observe_elems(&mut self, m: &BTreeMap<String, u64>) {
        for (site, n) in m {
            let e = self.elems.entry(site.clone()).or_insert(0);
            if n > e {
                *e = *n;
            }
        }
    }

    pub fn observe_ops(&mut self, m: &BTreeMap<String, String>) {
        for (site, op) in m {
            self.ops.entry(site.clone()).or_insert_with(|| op.clone());
        }
    }

    /// Fold one run's sums of terms in, pointwise maximum.
    pub fn observe_terms(&mut self, m: &BTreeMap<String, presk_interp::TermSums>) {
        for (site, t) in m {
            let e = self.terms.entry(site.clone()).or_default();
            e.abs = e.abs.max(t.abs);
            e.rss = e.rss.max(t.rss);
            if e.norms.len() < t.norms.len() {
                e.norms.resize(t.norms.len(), 0.0);
            }
            for (a, b) in e.norms.iter_mut().zip(&t.norms) {
                *a = a.max(*b);
            }
        }
    }

    /// The sums of a site's terms, if they were recorded.
    pub fn terms(&self, site: &str) -> Option<&Terms> {
        self.terms.get(site).filter(|t| t.abs.is_finite() && t.rss.is_finite())
    }

    /// The condition number at a site as far as an envelope can say it,
    /// `max Σ|xᵢwᵢ| / max |Σxᵢwᵢ|`. Both are maxima over output elements,
    /// possibly different ones, so this is a summary of the site and not
    /// the worst element's condition number, which is infinite wherever a
    /// sum passes through zero.
    pub fn condition(&self, site: &str) -> Option<f64> {
        let t = self.terms(site)?;
        let m = self.get(site)?;
        (m > 0.0).then(|| t.abs / m)
    }

    /// The primitive a site computes, if it was recorded.
    pub fn op(&self, site: &str) -> Option<&str> {
        self.ops.get(site).map(|s| s.as_str())
    }

    /// Real elements at a site, or zero when unknown.
    pub fn elems(&self, site: &str) -> u64 {
        self.elems.get(site).copied().unwrap_or(0)
    }

    pub fn get(&self, site: &str) -> Option<f64> {
        self.sites.get(site).copied()
    }

    /// The smallest non-zero `|x|` seen at a site, if any.
    pub fn floor(&self, site: &str) -> Option<f64> {
        self.minima.get(site).copied().filter(|x| x.is_finite() && *x > 0.0)
    }

    /// How far two envelopes of the same program disagree.
    ///
    /// Returns the worst ratio `max(a, b) / min(a, b)` over the sites both
    /// mention, the site where it occurs, and how many sites only one of
    /// them has. A ratio near 1 means the magnitudes are a property of the
    /// program rather than of the sample that happened to be drawn — which
    /// is exactly what a bound quoting an envelope needs to be true.
    ///
    /// Sites whose magnitude is zero in both are skipped: a value that is
    /// identically zero (a zero-initialized head before its first update)
    /// agrees trivially and would otherwise divide by zero.
    pub fn agreement(&self, other: &Envelope) -> Agreement {
        let mut worst = 1.0f64;
        let mut at = String::new();
        let mut compared = 0usize;
        let mut both_zero = 0usize;
        for (site, a) in &self.sites {
            let Some(b) = other.sites.get(site) else { continue };
            if *a == 0.0 && *b == 0.0 {
                both_zero += 1;
                continue;
            }
            compared += 1;
            let (lo, hi) = if a < b { (*a, *b) } else { (*b, *a) };
            let r = if lo > 0.0 { hi / lo } else { f64::INFINITY };
            if r > worst {
                worst = r;
                at = site.clone();
            }
        }
        let only_one = self.sites.keys().filter(|k| !other.sites.contains_key(*k)).count()
            + other.sites.keys().filter(|k| !self.sites.contains_key(*k)).count();
        Agreement { worst, at, compared, both_zero, only_one }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("an envelope serializes")
    }

    pub fn load(path: &std::path::Path) -> Result<Envelope, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("{} is not an envelope: {e}", path.display()))
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        if let Some(d) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(d).map_err(|e| format!("cannot create {}: {e}", d.display()))?;
        }
        std::fs::write(path, self.to_json()).map_err(|e| format!("cannot write {}: {e}", path.display()))
    }
}

/// The result of comparing two envelopes of the same program.
#[derive(Clone, Debug, PartialEq)]
pub struct Agreement {
    /// Worst ratio between the two, over the sites both mention.
    pub worst: f64,
    /// The site where that worst ratio occurs.
    pub at: String,
    /// How many sites were compared.
    pub compared: usize,
    /// How many are identically zero in both — a zero-initialized head and
    /// whatever is downstream of it. They agree trivially and are excluded
    /// from the ratio rather than counted as perfect agreement.
    pub both_zero: usize,
    /// How many sites only one of the two mentions. A program whose control
    /// flow does not depend on data has none, and any at all is worth seeing.
    pub only_one: usize,
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_envelope_is_the_pointwise_maximum() {
        let mut e = Envelope::new("p.psk", "loss", 128, "2026-09-20");
        e.observe(&BTreeMap::from([("f/%1".into(), 2.0), ("f/%2".into(), 1.0)]));
        e.observe(&BTreeMap::from([("f/%1".into(), 1.0), ("f/%2".into(), 7.0)]));
        assert_eq!(e.get("f/%1"), Some(2.0), "it must cover every sample, not the last");
        assert_eq!(e.get("f/%2"), Some(7.0));
        assert_eq!(e.samples, 2);
        // The file form is what a bound will quote, so it must round trip.
        let back: Envelope = serde_json::from_str(&e.to_json()).unwrap();
        assert_eq!(back, e);
    }

    #[test]
    fn agreement_measures_the_ratio_and_ignores_trivial_zeros() {
        let mk = |a: f64, b: f64, z: f64| {
            let mut e = Envelope::new("p.psk", "loss", 128, "2026-09-20");
            e.observe(&BTreeMap::from([("f/%1".into(), a), ("f/%2".into(), b), ("f/%3".into(), z)]));
            e
        };
        let x = mk(1.0, 4.0, 0.0);
        let y = mk(1.1, 5.0, 0.0);
        let a = x.agreement(&y);
        assert_eq!(a.compared, 2, "the site that is zero in both is not compared");
        assert_eq!(a.both_zero, 1);
        assert_eq!(a.only_one, 0);
        assert!((a.worst - 1.25).abs() < 1e-12, "worst is 5/4, not 1.1/1");
        assert_eq!(a.at, "f/%2");
        // Symmetric: which way round the comparison runs cannot matter.
        assert_eq!(y.agreement(&x).worst, a.worst);
        // A site present in one only is reported rather than silently dropped.
        let mut z = x.clone();
        z.sites.insert("f/%4".into(), 1.0);
        assert_eq!(z.agreement(&x).only_one, 1);
    }

    #[test]
    fn the_measured_day_is_a_calendar_date() {
        // Days since 1970-01-01, checked against Python's `datetime`: the
        // epoch, the day before it, a leap day and the day after, the day
        // this was written, and 2100, which is not a leap year.
        for (days, want) in [(0, (1970, 1, 1)), (-1, (1969, 12, 31)), (11_016, (2000, 2, 29)), (11_017, (2000, 3, 1)), (20_718, (2026, 9, 22)), (47_541, (2100, 3, 1))] {
            assert_eq!(civil_from_days(days), want, "day {days}");
        }
        let t = today();
        assert!(t.len() == 10 && t.as_bytes()[4] == b'-' && t.as_bytes()[7] == b'-' && t.as_str() >= "2026-09-22", "{t}");
    }
}
