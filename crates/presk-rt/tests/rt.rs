//! The runtime seam: values cross it exactly, realizations round once,
//! shapes materialize at bound dimensions.

use presk_rt::{Axis, Buf, Dtype, Tensor, Value};
use rug::Float;

#[test]
fn exact_values_round_trip() {
    let x = Float::with_val(256, 1) / Float::with_val(256, 3);
    let iv = presk_interp::Value::record(vec![("w".into(), presk_interp::Value::Real(presk_interp::Tensor { axes: vec![Axis { name: Some("model".into()), len: 1 }], data: vec![x.clone()].into() }))]);
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
    let v = Value::Real(Tensor { axes: vec![], data: Buf::Exact(vec![x].into()) });
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
        presk_interp::Value::Idx { bound: 64, t: presk_interp::Tensor { axes: vec![Axis { name: Some("seq".into()), len: 2 }], data: vec![3, 7].into() } },
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
    r.storage.insert("loss/%17".into(), Dtype::BF16);
    r.storage.insert("loss/%23".into(), Dtype::F64);
    r.accum.insert("loss/%17".into(), Dtype::F32.into());
    r.order = ReductionOrder::Interleaved { partials: 8 };
    r.envelope = Some("out/envelope.json".into());
    let back: Realization = serde_json::from_str(&r.to_json()).unwrap();
    assert_eq!(back, r);

    // The defaults a site inherits, and the one that is not the storage type.
    assert_eq!(back.dtype_at("loss/%17"), Dtype::BF16);
    assert_eq!(back.dtype_at("loss/%99"), Dtype::F32, "an unmentioned site takes the default");
    assert_eq!(back.accum_at("loss/%23"), Dtype::F64.into(), "an unmentioned accumulator follows its storage");
    assert_eq!(back.accum_at("loss/%42"), Dtype::F32.into());

    // By function (0.6.0 step 7b): a site takes its function's type unless
    // it is named itself, and a file written before the field still loads.
    let mut f = back.clone();
    f.functions.insert("adamw.step".into(), Dtype::BF16);
    f.functions.insert("loss".into(), Dtype::F64);
    assert_eq!(f.dtype_at("adamw.step/%3"), Dtype::BF16);
    assert_eq!(f.dtype_at("Linear.apply.vjp/%3"), Dtype::F32, "another function takes the default");
    assert_eq!(f.dtype_at("loss/%17"), Dtype::BF16, "a site named itself overrides its function");
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
    assert!(Realization::uniform(Dtype::BF16).groups_visible());
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
    b.functions.insert("adamw.lr".into(), Dtype::BF16);
    assert!(b.groups_visible());
    let mut q = Realization::uniform(Dtype::F32);
    q.quant.insert("blocks.ff.up.w".into(), Quant { bits: 8, over: AxisRef::Pos(-2), scheme: QuantScheme::Absmax });
    q.accum.insert("loss/%3".into(), Dtype::F64.into());
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

/// A type is a format (plan 0.18.0 step 1, D204): `f64`, `f32` and `bf16`
/// keep every property, name, order and spelling they had as an enum. Since
/// step 3 (D206) a realization stores a value in any float format, named
/// by its alias or its parameters; the native types are the three, which
/// the solver's menu alone keeps to (`E7004`), and a realization that
/// names another is one the native backends refuse (`E7005`).
#[test]
fn types_are_formats() {
    use presk_rt::Realization;
    assert_eq!([Dtype::F64.format(), Dtype::F32.format(), Dtype::BF16.format()], [presk_format::F64, presk_format::F32, presk_format::BF16]);
    assert_eq!(Dtype::ALL.map(|d| d.bytes()), [8, 4, 2]);
    assert_eq!(Dtype::ALL.map(|d| d.natural_lanes()), [2, 4, 1]);
    assert_eq!(Dtype::ALL.map(|d| d.arith()), [Dtype::F64, Dtype::F32, Dtype::F32]);
    assert_eq!(Dtype::ALL.map(|d| d.unit_roundoff().to_bits()), [(f64::EPSILON / 2.0).to_bits(), (f32::EPSILON as f64 / 2.0).to_bits(), 2f64.powi(-8).to_bits()]);
    assert!(Dtype::F64 < Dtype::F32 && Dtype::F32 < Dtype::BF16);
    assert_eq!(Dtype::ALL.map(|d| format!("{d:?} {d}")), ["F64 f64", "F32 f32", "Bf16 bf16"]);
    assert_eq!(serde_json::to_string(&Dtype::ALL).unwrap(), r#"["f64","f32","bf16"]"#);
    assert_eq!(serde_json::from_str::<Vec<Dtype>>(r#"["bf16","f64"]"#).unwrap(), [Dtype::BF16, Dtype::F64]);
    assert_eq!(Dtype::of_name("float(p=8,emin=-126,emax=127)"), Ok(Dtype::BF16));
    assert!(Dtype::ALL.iter().all(|d| d.is_native()));
    // Every other float format is a type the kernel interpreter emulates.
    let e4m3 = "float(p=4,emin=-6,emax=8,specials=none,underflow=+0)";
    for name in ["f16", "e4m3fn", e4m3] {
        let d = Dtype::of_name(name).unwrap();
        assert!(!d.is_native() && d.name() == name && d.to_string() == name && Dtype::parse(name) == Some(d), "{name}");
        assert_eq!(serde_json::to_string(&d).unwrap(), format!("\"{name}\""));
        assert!(Dtype::native(d.format()).unwrap_err().contains("[E7004]"), "{name}");
        assert_eq!(Realization::parse(name).unwrap(), Realization::uniform(d));
        let file = format!(r#"{{"default": "f32", "storage": {{"f/%1": "{name}"}}, "order": {{"kind": "sequential"}}}}"#);
        let r: Realization = serde_json::from_str(&file).unwrap();
        assert_eq!((r.dtype_at("f/%1"), r.dtype_at("f/%2")), (d, Dtype::F32));
        let e = r.check_native("the native backend").unwrap_err();
        assert!(e.code == "E7005" && e.message.starts_with(&format!("the realization stores `f/%1` in {name}: the native backend computes in f64, f32 and bf16")), "{}", e.message);
        // binary16 and `e8m15` for an executor that holds their codes, the
        // native backend since plan 0.18.2 step 2 (D216).
        match r.check_executor("the native backend", true) {
            Ok(()) => assert!(d.coded(), "{name}"),
            Err(e) => assert!(!d.coded() && e.code == "E7005" && e.message.contains("the native backend computes in f64, f32, bf16, f16 and e8m15"), "{name}: {}", e.message),
        }
    }
    let r = Realization::uniform(Dtype::E8M15);
    assert!(r.check_native("the C export").is_err() && r.check_executor("the native backend", true).is_ok() && r.emulates());
    assert_eq!(Dtype::of_name("f16").unwrap().arith(), Dtype::F32, "f32 holds f16: an f16 value computes in f32, as bf16 does");
    assert_eq!(Buf::of(Dtype::of_name(e4m3).unwrap(), &[1.0, 3.0, 17.0, 1e6, -0.0]).to_f64(), [1.0, 3.0, 16.0, 480.0, 0.0], "rounded once, saturated, underflowing to +0");
    assert_eq!(Realization::parse("f8").unwrap_err(), "unknown realization `f8`; expected f64, f32, bf16, another float format's name or @FILE");
}

/// What a nest computes in (plan 0.18.0 step 3, D206): its arithmetic
/// format, its storage type's arithmetic type unless the realization says
/// otherwise; the products of its sums and its sums in that format unless
/// it says otherwise; each read through a stored gradient's forward pass as
/// the storage type is. A realization stating any is one the native
/// backends compute only where it is what they compute anyway.
#[test]
fn a_nest_computes_in_its_formats() {
    use presk_format::Format;
    use presk_rt::{Realization, ReductionOrder};
    let f16 = Format::Float(presk_format::F16);
    let json = r#"{"default": "f32", "storage": {"g/%2": "bf16"}, "arith": {"g/%1": "f16"}, "product": {"g/%1": "f16"}, "accum": {"g/%1": "fixed(i64,frac=18)", "g/%2": "f32"}, "order": {"kind": "chunked", "size": 32}}"#;
    let r: Realization = serde_json::from_str(json).unwrap();
    assert_eq!((r.arith_at("g/%1"), r.product_at("g/%1"), r.accum_at("g/%1")), (f16, f16, Format::parse("fixed(i64,frac=18)").unwrap()));
    // Unmentioned: the storage type's arithmetic type, `bf16`'s `f32`.
    assert_eq!((r.arith_at("g/%2"), r.product_at("g/%2"), r.accum_at("g/%2")), (Dtype::F32.into(), Dtype::F32.into(), Dtype::F32.into()));
    assert_eq!((r.arith_at("g/%3"), r.accum_at("g/%3")), (Dtype::F32.into(), Dtype::F32.into()));
    // A stored gradient's forward pass reads `g`'s.
    assert_eq!((r.arith_at("g.fwd/%1"), r.accum_at("g.fwd/%1")), (f16, r.accum_at("g/%1")));
    assert_eq!(r.order, ReductionOrder::Chunked { size: 32 });
    assert_eq!((r.order.name(), r.order.is_exact()), ("chunked(32)".to_string(), false));
    assert!(!r.is_uniform() && r.emulates() && r.label() == "f32+chunked(32)+1sites+1arith+1product+2accum", "{}", r.label());
    assert_eq!(serde_json::from_str::<Realization>(&r.to_json()).unwrap(), r);
    let e = r.check_native("the native backend").unwrap_err();
    assert!(e.message.starts_with("the realization computes `g/%1` in f16, where its storage type's arithmetic type is f32"), "{}", e.message);
    // The formats a native backend computes in anyway are native.
    let mut native = Realization::uniform(Dtype::BF16);
    native.arith.insert("g/%1".into(), Dtype::F32.into());
    native.accum.insert("g/%1".into(), Dtype::F32.into());
    native.order = ReductionOrder::Chunked { size: 4 };
    assert!(native.check_native("the native backend").is_ok() && !native.emulates());
}

/// A value stored in an affine format (plan 0.18.0 step 4, D207): the
/// realization records its format, or the scheme an envelope computes it
/// by, which every executor refuses until `presk scales` has (`E7006`); a
/// gradient's forward pass reads its base's; a native backend refuses it.
#[test]
fn a_value_is_stored_in_an_affine_format() {
    use presk_format::Affine;
    use presk_rt::{Realization, Scale, ScaleRule};
    let json = r#"{"default": "f32", "scaled": {"g/%1": "affine(u8/sat,scale=0.025,zero=12)", "g/%2": {"int": "i8/sat", "scheme": "absmax"}, "g/%3": {"int": "u8/sat", "scheme": "minmax", "tie": "away", "rounds": "x/s+zero", "quotient": "f32"}}, "product": {"g/%4": "i32"}, "accum": {"g/%4": "i32"}, "order": {"kind": "sequential"}}"#;
    let r: Realization = serde_json::from_str(json).unwrap();
    let a = Affine::parse("affine(u8/sat,scale=0.025,zero=12)").unwrap();
    assert_eq!(r.scaled_at("g/%1").unwrap(), Some(a));
    assert_eq!(r.scaled_at("g.fwd/%1").unwrap(), Some(a));
    assert_eq!(r.scaled_at("g/%4").unwrap(), None);
    assert_eq!(serde_json::from_str::<Realization>(&r.to_json()).unwrap(), r);
    let e = r.scaled_at("g/%2").unwrap_err();
    assert_eq!(e.code, "E7006");
    assert!(e.message.contains("`g/%2` in i8/sat with a scale computed by `absmax`") && e.message.contains("presk scales"), "{}", e.message);
    assert_eq!(r.check_scales().unwrap_err().code, "E7006");
    assert!(!r.is_uniform() && r.emulates() && r.groups_visible());
    assert_eq!(r.label(), "f32+sequential+1product+1accum+3scaled");
    let e = r.check_native("the native backend").unwrap_err();
    assert!(e.message.contains("rounds the products of `g/%4` in i32"), "{}", e.message);
    // The schemes, from a site's magnitudes.
    let (Scale::Scheme(abs), Scale::Scheme(mm)) = (&r.scaled["g/%2"], &r.scaled["g/%3"]) else { panic!("schemes") };
    assert_eq!(abs.scheme, ScaleRule::Absmax);
    assert_eq!(abs.resolve(2.54, None).unwrap(), Affine::absmax(presk_format::Int::parse("i8/sat").unwrap(), 2.54).unwrap());
    assert!(mm.resolve(3.0, None).unwrap_err().contains("--range"));
    let m = mm.resolve(3.0, Some((-1.0, 3.0))).unwrap();
    assert_eq!(m.name(), "affine(u8/sat,scale=0.015686275,zero=64,tie=away,rounds=x/s+zero,quotient=f32)");
    // Resolved, it runs where the formats are emulated, and nowhere else.
    let mut ok = r.clone();
    ok.scaled.insert("g/%2".into(), Scale::Static(abs.resolve(2.54, None).unwrap()));
    ok.scaled.insert("g/%3".into(), Scale::Static(m));
    ok.product.clear();
    ok.accum.clear();
    assert!(ok.check_scales().is_ok());
    let e = ok.check_native("the native backend").unwrap_err();
    assert!(e.message.starts_with("the realization stores `g/%1` in affine(u8/sat,scale=0.025,zero=12)"), "{}", e.message);
    for bad in [r#"{"default": "f32", "scaled": {"g/%1": "affine(i8,scale=0.5)"}, "order": {"kind": "sequential"}}"#, r#"{"default": "f32", "scaled": {"g/%1": {"int": "i8/sat", "scheme": "median"}}, "order": {"kind": "sequential"}}"#, r#"{"default": "f32", "scaled": {"g/%1": {"int": "i8/sat", "scheme": "absmax", "colour": "red"}}, "order": {"kind": "sequential"}}"#] {
        assert!(serde_json::from_str::<Realization>(bad).is_err(), "{bad}");
    }
}
