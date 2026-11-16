//! A machine's numbers, as the solver offers them (plan 0.18.0 step 5,
//! D208): `presk solve --target FILE` reads a target description and offers
//! exactly its menu.
//!
//! A description says what a machine stores and computes in, without its
//! name: the float formats a value is stored in, the affine formats an
//! activation is, by the scheme its scale is computed from the envelope by
//! (`absmax`, `pow2`, `minmax`), the formats a parameter is stored in and the
//! integer weights it reads, the format a nest computes in by the format its
//! value is stored in, and its *contraction modes*: by its operands' formats,
//! the format a sum's products are rounded to and the one its additions are.
//! One order sums every sum, the machine's (a systolic array's tiles along
//! the summed axis are `chunked`). A machine whose matrix unit multiplies
//! E4M3 values into binary16 sums, or 8-bit integers into a wrapping 32-bit
//! one, and whose vector unit computes in binary16:
//!
//! ```json
//! {
//!   "name": "accelerator",
//!   "storage": ["f16", "float(p=4,emin=-6,emax=8,specials=none,underflow=+0)"],
//!   "scaled": [{"int": "i8/sat", "scheme": "absmax"}],
//!   "quant": ["int8"],
//!   "arith": {"*": "f16"},
//!   "contractions": [
//!     {"name": "fp8", "operands": ["float(p=4,emin=-6,emax=8,specials=none,underflow=+0)", "float(p=4,emin=-6,emax=8,specials=none,underflow=+0)"], "product": "f16", "accum": "f16"},
//!     {"name": "int8", "operands": ["int", "int"], "product": "i32", "accum": "i32"}
//!   ],
//!   "order": {"kind": "chunked", "size": 32}
//! }
//! ```
//!
//! An operand's format in a mode is a float format's name, `int` for the
//! integer codes of an affine activation or an integer weight, or `*` for
//! any. A contraction whose operands match no mode is not in the target's
//! types: the solver does not offer the formats that leave it so. A target
//! with no mode computes every sum in its nest's arithmetic format.

use std::collections::BTreeMap;

use presk_format::{Float, Format};
use presk_rt::{Dtype, QuantScheme, ReductionOrder, ScaleScheme};

/// A machine's numbers.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Target {
    /// What the report calls it.
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub about: Option<String>,
    /// The float formats a stored value may take, an activation's and, but
    /// where `params` says otherwise, a parameter's.
    pub storage: Vec<Dtype>,
    /// The affine formats an activation may be stored in, each by the
    /// scheme its scale is computed by from the envelope at its site.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scaled: Vec<ScaleScheme>,
    /// The float formats a parameter is stored in at rest; `storage` where
    /// absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Vec<Dtype>>,
    /// The integer weights it reads, as `presk solve --menu` spells them:
    /// `int8`, `int4/pow2`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub quant: Vec<String>,
    /// The format a nest computes in, by the name of the format its value
    /// is stored in, `affine` for an affine one, `*` for any other.
    pub arith: BTreeMap<String, Format>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub contractions: Vec<Mode>,
    /// The order every sum is added in.
    #[serde(default = "sequential")]
    pub order: ReductionOrder,
}

fn sequential() -> ReductionOrder {
    ReductionOrder::Sequential
}

/// A contraction mode: how a sum of products of operands of these formats
/// is computed.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mode {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Each operand's format: a float's name, `int` or `*`.
    pub operands: Vec<String>,
    pub product: Format,
    pub accum: Format,
}

/// What a contraction's operand is to a mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A value read as a float of this format.
    Float(Float),
    /// The integer codes of an affine value or of an integer weight.
    Int,
}

impl Mode {
    /// Whether operands of these kinds are computed in this mode.
    pub fn takes(&self, kinds: &[Kind]) -> bool {
        self.operands.len() == kinds.len()
            && self.operands.iter().zip(kinds).all(|(o, k)| match (o.as_str(), k) {
                ("*", _) => true,
                ("int", Kind::Int) => true,
                ("int", Kind::Float(_)) => false,
                (name, Kind::Float(f)) => Float::parse(name).is_ok_and(|g| g == *f),
                (_, Kind::Int) => false,
            })
    }

    /// Its name, or its operands', products' and sums' formats.
    pub fn label(&self) -> String {
        match &self.name {
            Some(n) => n.clone(),
            None => format!("{} → {}×, {}+", self.operands.join(" · "), self.product, self.accum),
        }
    }
}

impl Target {
    /// The description in a file, checked.
    pub fn load(path: &std::path::Path) -> Result<Target, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let t: Target = serde_json::from_str(&text).map_err(|e| format!("{} is not a target description: {e}", path.display()))?;
        t.check().map_err(|e| format!("{}: {e}", path.display()))
    }

    /// A description is a menu: some storage format, each mode's operands
    /// formats, its products' and sums' formats numbers an operation is
    /// rounded to, each scheme a scale, each weight `--menu`'s.
    pub fn check(self) -> Result<Target, String> {
        if self.storage.is_empty() {
            return Err("a target stores its values in some float format: `storage` is empty".into());
        }
        for s in &self.scaled {
            s.resolve(1.0, Some((-1.0, 1.0))).map_err(|e| format!("`scaled`: {e}"))?;
        }
        self.weights()?;
        if !self.arith.contains_key("*") && !self.storage.iter().all(|d| self.arith.contains_key(d.name())) {
            return Err("`arith` names the format every stored format computes in, or `*` for those it does not name".into());
        }
        for m in &self.contractions {
            for o in &m.operands {
                if o != "*" && o != "int" {
                    Float::parse(o).map_err(|e| format!("a mode's operand `{o}`: {e}"))?;
                }
            }
            for f in [m.product, m.accum] {
                if matches!(f, Format::Affine(_)) {
                    return Err(format!("the mode {} rounds an operation to {f}: an affine format is a stored value's", m.label()));
                }
            }
        }
        Ok(self)
    }

    /// The integer weights, `(bits, scheme)`.
    pub fn weights(&self) -> Result<Vec<(u8, QuantScheme)>, String> {
        if self.quant.is_empty() {
            return Ok(Vec::new());
        }
        let menu = crate::solve::Menu::parse(&format!("f32,{}", self.quant.join(","))).map_err(|e| format!("`quant`: {e}"))?;
        Ok(menu.1)
    }

    /// The formats a parameter is stored in.
    pub fn param_floats(&self) -> &[Dtype] {
        self.params.as_deref().unwrap_or(&self.storage)
    }

    /// Its widest storage format, which a value the solver does not offer
    /// keeps: the most bytes, then the most precise.
    pub fn widest(&self) -> Dtype {
        *self.storage.iter().max_by_key(|d| (d.bytes(), d.format().p)).expect("a target stores in some format")
    }

    /// The format a nest whose value is stored in `stored` computes in, if
    /// the target says one.
    pub fn arith_for(&self, stored: Format) -> Option<Format> {
        let key = match stored {
            Format::Affine(_) => "affine".to_string(),
            other => other.name().into_owned(),
        };
        self.arith.get(&key).or_else(|| self.arith.get("*")).copied()
    }

    /// The first mode that takes these operands.
    pub fn mode_for(&self, kinds: &[Kind]) -> Option<&Mode> {
        self.contractions.iter().find(|m| m.takes(kinds))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const E4M3: &str = "float(p=4,emin=-6,emax=8,specials=none,underflow=+0)";

    #[test]
    fn a_description_reads_and_is_checked() {
        let text = format!(
            r#"{{"name": "t", "storage": ["f16", "{E4M3}"], "scaled": [{{"int": "i8/sat", "scheme": "absmax"}}], "quant": ["int8"], "arith": {{"*": "f16"}},
                "contractions": [{{"name": "fp8", "operands": ["{E4M3}", "{E4M3}"], "product": "f16", "accum": "f16"}}, {{"operands": ["int", "int"], "product": "i32", "accum": "i32"}}],
                "order": {{"kind": "chunked", "size": 32}}}}"#
        );
        let t: Target = serde_json::from_str::<Target>(&text).unwrap().check().unwrap();
        assert_eq!(t.widest().name(), "f16");
        assert_eq!(t.weights().unwrap(), [(8, QuantScheme::Absmax), (8, QuantScheme::Pow2)], "`int8` is both schemes, as in --menu");
        let e4 = Kind::Float(Float::parse(E4M3).unwrap());
        assert_eq!(t.mode_for(&[e4, e4]).unwrap().label(), "fp8");
        assert_eq!(t.mode_for(&[Kind::Int, Kind::Int]).unwrap().label(), "int · int → i32×, i32+");
        assert!(t.mode_for(&[e4, Kind::Int]).is_none());
        assert!(t.mode_for(&[Kind::Float(presk_format::F16), e4]).is_none());
        assert_eq!(t.arith_for(Format::Float(presk_format::F16)), Some(Format::parse("f16").unwrap()));
        assert_eq!(t.order, ReductionOrder::Chunked { size: 32 });
        // What a description may not say.
        for bad in [r#"{"name": "t", "storage": [], "arith": {"*": "f16"}}"#, r#"{"name": "t", "storage": ["f16"], "arith": {"f32": "f32"}}"#, r#"{"name": "t", "storage": ["f16"], "arith": {"*": "f16"}, "quant": ["int3"]}"#, r#"{"name": "t", "storage": ["f16"], "arith": {"*": "f16"}, "colour": 1}"#] {
            assert!(serde_json::from_str::<Target>(bad).map_err(|e| e.to_string()).and_then(Target::check).is_err(), "{bad}");
        }
    }
}
