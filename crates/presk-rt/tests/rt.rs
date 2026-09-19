//! The runtime seam: values cross it exactly, realizations round once,
//! shapes materialize at bound dimensions.

use presk_rt::{Axis, Buf, Realize, Tensor, Value};
use rug::Float;

#[test]
fn exact_values_round_trip() {
    let x = Float::with_val(256, 1) / Float::with_val(256, 3);
    let iv = presk_interp::Value::Record(vec![("w".into(), presk_interp::Value::Real(presk_interp::Tensor { axes: vec![Axis { name: Some("model".into()), len: 1 }], data: vec![x.clone()] }))]);
    let v = Value::from_interp(&iv).unwrap();
    let Value::Real(t) = v.field("w").unwrap() else { panic!() };
    assert!(matches!(t.data, Buf::Exact(_)));
    let back = v.to_interp(256);
    let presk_interp::Value::Record(fs) = &back else { panic!() };
    let presk_interp::Value::Real(t2) = &fs[0].1 else { panic!() };
    assert_eq!(t2.data[0], x, "256-bit value unchanged across the seam");
    assert_eq!(t2.axes[0].name.as_deref(), Some("model"));
}

#[test]
fn realizations_round_once() {
    let x = Float::with_val(256, 1) / Float::with_val(256, 3);
    let v = Value::Real(Tensor { axes: vec![], data: Buf::Exact(vec![x]) });
    let Value::Real(t64) = v.realize(Realize::F64) else { panic!() };
    let Value::Real(t32) = v.realize(Realize::F32) else { panic!() };
    assert_eq!(t64.data, Buf::F64(vec![1.0 / 3.0]));
    assert_eq!(t32.data, Buf::F32(vec![1.0f32 / 3.0]));
    assert_eq!(v.scalar_f64(), 1.0 / 3.0);
    let tree = Value::Tuple(vec![v.clone(), Value::Real(Tensor::scalar_f64(2.0))]);
    assert!((tree.sum_squares_f64() - (1.0 / 9.0 + 4.0)).abs() < 1e-15);
}

#[test]
fn indices_and_scalars() {
    let iv = presk_interp::Value::Tuple(vec![
        presk_interp::Value::Idx { bound: 64, t: presk_interp::Tensor { axes: vec![Axis { name: Some("seq".into()), len: 2 }], data: vec![3, 7] } },
        presk_interp::Value::Nat(5),
        presk_interp::Value::F64(0.25),
    ]);
    let v = Value::from_interp(&iv).unwrap();
    assert_eq!(Value::from_interp(&v.to_interp(53)).unwrap(), v, "round trip through the interpreter's values");
    let Value::Tuple(parts) = &v else { panic!() };
    assert_eq!(parts[0].kind(), "index tensor");
    assert_eq!(parts[1].scalar_f64(), 5.0);
}
