//! A parameter's floating-point storage type (plan 0.12.0, D119, D120; the
//! reference's chapter 4, W18): the record's member `params`, keyed by path
//! like `quant`, and its conditions.

use presk_rt::{Axis, AxisRef, Buf, Dtype, Quant, QuantScheme, Realization, Tensor, Value};

fn leaf(n: usize) -> Value {
    Value::Real(Tensor { axes: vec![Axis { name: None, len: n }, Axis { name: None, len: 3 }], data: Buf::F64((0..n * 3).map(|i| i as f64 / 7.0).collect()) })
}

fn tree() -> Value {
    Value::Record(vec![("tok".into(), leaf(4)), ("blocks".into(), Value::Record(vec![("up".into(), Value::Record(vec![("w".into(), leaf(5))]))])), ("head".into(), leaf(2))])
}

#[test]
fn a_typed_realization_survives_its_file_form() {
    let mut r = Realization::uniform(Dtype::F64);
    r.params.insert("blocks.up.w".into(), Dtype::Bf16);
    r.params.insert("head".into(), Dtype::F32);
    let text = r.to_json();
    assert!(text.contains("\"params\": {") && text.contains("\"blocks.up.w\": \"bf16\""), "{text}");
    let back: Realization = serde_json::from_str(&text).unwrap();
    assert_eq!(back, r);
    assert!(!r.is_uniform());
    assert_eq!(r.label(), "f64+sequential+2params");
    // A record without the member, every record before 0.12.0, has none, and
    // one that has none writes none.
    let older: Realization = serde_json::from_str(r#"{"default":"f64","order":{"kind":"sequential"}}"#).unwrap();
    assert!(older.params.is_empty() && older.is_uniform());
    assert!(!Realization::uniform(Dtype::F32).to_json().contains("params"));
}

#[test]
fn typed_parameters_meet_the_records_conditions() {
    let mut r = Realization::uniform(Dtype::F64);
    r.params.insert("blocks.up.w".into(), Dtype::Bf16);
    r.params.insert("tok".into(), Dtype::F32);
    assert!(r.check_params(None).is_ok());
    assert!(r.check_params(Some(&tree())).is_ok());

    // `f64` is the default's to state: a parameter's own type is narrower.
    let mut wide = r.clone();
    wide.params.insert("head".into(), Dtype::F64);
    let e = wide.check_params(None).unwrap_err();
    assert_eq!(e.code, "E7001");
    assert!(e.message.contains("`head`") && e.message.contains("`f32` or `bf16`"), "{}", e.message);

    // Typed or quantised, not both.
    let mut both = r.clone();
    both.quant.insert("tok".into(), Quant { bits: 8, over: AxisRef::Pos(-2), scheme: QuantScheme::Absmax });
    let e = both.check_params(None).unwrap_err();
    assert_eq!(e.code, "E7001");
    assert!(e.message.contains("`tok`") && e.message.contains("not both"), "{}", e.message);

    // A misspelt path fails loudly and names the paths there are.
    let mut typo = r.clone();
    typo.params.insert("blocks.up.W".into(), Dtype::F32);
    assert!(typo.check_params(None).is_ok(), "the paths are checked where the parameters are known");
    let e = typo.check_params(Some(&tree())).unwrap_err();
    assert_eq!(e.code, "E7001");
    assert!(e.message.contains("blocks.up.W") && e.message.contains("tok, blocks.up.w, head"), "{}", e.message);
    let mut args = vec![tree()];
    assert_eq!(typo.store_params(&mut args).unwrap_err().message, e.message);
}
