//! The runtime seam: values cross it exactly, realizations round once,
//! shapes materialize at bound dimensions.

use presk_rt::{Axis, Buf, Dtype, Tensor, Value};
use rug::Float;

#[test]
fn exact_values_round_trip() {
    let x = Float::with_val(256, 1) / Float::with_val(256, 3);
    let iv = presk_interp::Value::record(vec![("w".into(), presk_interp::Value::Real(presk_interp::Tensor { axes: vec![Axis { name: Some("model".into()), len: 1 }], data: vec![x.clone()] }))]);
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
    let Value::Real(t64) = v.realize(Dtype::F64) else { panic!() };
    let Value::Real(t32) = v.realize(Dtype::F32) else { panic!() };
    assert_eq!(t64.data, Buf::F64(vec![1.0 / 3.0].into()));
    assert_eq!(t32.data, Buf::F32(vec![1.0f32 / 3.0].into()));
    assert_eq!(v.scalar_f64(), 1.0 / 3.0);
    let tree = Value::Tuple(vec![v.clone(), Value::Real(Tensor::scalar_f64(2.0))]);
    assert!((tree.sum_squares_f64() - (1.0 / 9.0 + 4.0)).abs() < 1e-15);
}

#[test]
fn indices_and_scalars() {
    let iv = presk_interp::Value::tuple(vec![
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

/// The realization is an object, and a round trip through its file form must
/// give the same object back — otherwise `--realize @FILE` and the solver
/// that writes the file disagree about what was asked for (D34).
#[test]
fn a_realization_survives_its_file_form() {
    use presk_rt::{Dtype, Realization, ReductionOrder};
    let mut r = Realization::uniform(Dtype::F32);
    r.storage.insert("loss/%17".into(), Dtype::Bf16);
    r.storage.insert("loss/%23".into(), Dtype::F64);
    r.accum.insert("loss/%17".into(), Dtype::F32);
    r.order = ReductionOrder::Interleaved { partials: 8 };
    r.envelope = Some("out/envelope.json".into());
    let back: Realization = serde_json::from_str(&r.to_json()).unwrap();
    assert_eq!(back, r);

    // The defaults a site inherits, and the one that is not the storage type.
    assert_eq!(back.dtype_at("loss/%17"), Dtype::Bf16);
    assert_eq!(back.dtype_at("loss/%99"), Dtype::F32, "an unmentioned site takes the default");
    assert_eq!(back.accum_at("loss/%23"), Dtype::F64, "an unmentioned accumulator follows its storage");
    assert_eq!(back.accum_at("loss/%42"), Dtype::F32);

    // By function (0.6.0 step 7b): a site takes its function's type unless
    // it is named itself, and a file written before the field still loads.
    let mut f = back.clone();
    f.functions.insert("adamw.step".into(), Dtype::Bf16);
    f.functions.insert("loss".into(), Dtype::F64);
    assert_eq!(f.dtype_at("adamw.step/%3"), Dtype::Bf16);
    assert_eq!(f.dtype_at("Linear.apply.vjp/%3"), Dtype::F32, "another function takes the default");
    assert_eq!(f.dtype_at("loss/%17"), Dtype::Bf16, "a site named itself overrides its function");
    assert_eq!(f.dtype_at("loss/%99"), Dtype::F64);
    let again: Realization = serde_json::from_str(&f.to_json()).unwrap();
    assert_eq!(again, f);
    assert!(!again.to_json().is_empty() && !Realization::uniform(Dtype::F32).to_json().contains("functions"));

    // A uniform realization is the thing every earlier version could write,
    // and it must still be recognisable as one.
    let u = Realization::uniform(Dtype::F64);
    assert!(u.is_uniform() && u.label() == "f64");
    assert!(!back.is_uniform());
    assert!(ReductionOrder::Blocked { width: 8 }.is_exact(), "blocking keeps the order (D32)");
    assert!(!ReductionOrder::Interleaved { partials: 8 }.is_exact(), "partials reassociate (D31)");
}

/// When a fusion group can change a bit (plan 0.10.0 step 4b, D94): a
/// `bf16` site anywhere, or a site typed apart from its function; not a
/// uniform `f32` or `f64` realization, nor one by function, nor a site
/// named with the type it would have anyway.
#[test]
fn a_realization_sees_the_groups_under_bf16_or_a_site_apart() {
    use presk_rt::{AxisRef, Dtype, Quant, QuantScheme, Realization};
    assert!(!Realization::uniform(Dtype::F64).groups_visible());
    assert!(!Realization::uniform(Dtype::F32).groups_visible());
    assert!(Realization::uniform(Dtype::Bf16).groups_visible());
    let mut f = Realization::uniform(Dtype::F64);
    f.functions.insert("Linear.apply.vjp".into(), Dtype::F32);
    f.functions.insert("adamw.step".into(), Dtype::F32);
    assert!(!f.groups_visible(), "by function, every site of a function has its type");
    let mut same = f.clone();
    same.storage.insert("Linear.apply.vjp/%3".into(), Dtype::F32);
    same.storage.insert("loss/%5".into(), Dtype::F64);
    assert!(!same.groups_visible(), "a site named with the type it has anyway");
    let mut apart = f.clone();
    apart.storage.insert("Linear.apply.vjp/%3".into(), Dtype::F64);
    assert!(apart.groups_visible(), "a site typed apart from its function");
    let mut fwd = Realization::uniform(Dtype::F32);
    fwd.storage.insert("Block.apply/%7".into(), Dtype::F64);
    assert!(fwd.groups_visible(), "a forward site apart, which `Block.apply.fwd/%7` takes too");
    let mut b = f.clone();
    b.functions.insert("adamw.lr".into(), Dtype::Bf16);
    assert!(b.groups_visible());
    let mut q = Realization::uniform(Dtype::F32);
    q.quant.insert("blocks.ff.up.w".into(), Quant { bits: 8, over: AxisRef::Pos(-2), scheme: QuantScheme::Absmax });
    q.accum.insert("loss/%3".into(), Dtype::F64);
    assert!(!q.groups_visible(), "integer weights and accumulators are the contractions'");
}

/// `bf16` is a prefix of `f32`, so widening is exact and narrowing is one
/// round to nearest even. The JIT emits the same function in integer ops, so
/// this is also what pins the two executors together.
#[test]
fn bf16_rounds_to_nearest_even() {
    use presk_rt::{bf16_to_f32, f32_to_bf16};
    for x in [1.0f32, -3.75, 0.1, 1e-8, 123456.0, 0.0, -0.0] {
        let y = bf16_to_f32(f32_to_bf16(x));
        assert!((y - x).abs() <= 0.005 * x.abs().max(1e-30), "{x} -> {y}");
    }
    // Exactly representable values must not move at all.
    for x in [1.0f32, 2.0, -0.5, 3.0, 256.0] {
        assert_eq!(bf16_to_f32(f32_to_bf16(x)), x, "{x} is a bf16 already");
    }
    // A tie rounds to the even significand, not away from zero.
    let tie = f32::from_bits(0x3F808000); // 1 + 2^-8, halfway between two bf16
    assert_eq!(f32_to_bf16(tie), 0x3F80, "ties go to even");
    assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan(), "NaN stays NaN");
}

/// A `params` file records its model's name, not its leaves' shapes, so it
/// must be exactly the size the program's layout implies. A file from a
/// model of other dimensions is refused, never decoded from shifted offsets
/// (0.6.0 step 5d, D48: `pos : ℝ[T, D]` trained at `T = 256` and opened at
/// `T = 32` read every later leaf from the wrong place).
#[test]
fn a_checkpoint_of_other_dimensions_is_refused() {
    use presk_rt::{Axis, Buf, Tensor, Value, load_params, save_params};
    let leaf = |n: usize| Value::Real(Tensor { axes: vec![Axis { name: None, len: n }, Axis { name: None, len: 2 }], data: Buf::F64((0..2 * n).map(|i| i as f64).collect()) });
    let tree = |t: usize| Value::Record(vec![("pos".into(), leaf(t)), ("w".into(), leaf(3))]);
    let dir = std::env::temp_dir().join(format!("presk-rt-ckpt-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("big.psk.d");
    save_params(&path, "M", &tree(8)).unwrap();
    assert_eq!(load_params(&path, &tree(8)).unwrap(), tree(8), "the same dimensions load");
    assert_eq!(load_params(&path, &tree(4)).unwrap_err().code, "E9005", "a longer file is a model of other dimensions");
    assert_eq!(load_params(&path, &tree(16)).unwrap_err().code, "E9003", "a shorter one is truncated");
}
