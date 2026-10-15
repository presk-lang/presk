//! Integer quantisation of parameters (plan 0.6.0 step 5, D44): the scales,
//! the error rule, the packing, and the parameter paths a realization names.

use std::collections::BTreeMap;

use presk_rt::quant::{dequantize_params, param_paths, quantize_params};
use presk_rt::{Axis, AxisRef, Buf, Dtype, QTensor, Quant, QuantScheme, Realization, Tensor, Value};

fn ax(name: Option<&str>, len: usize) -> Axis {
    Axis { name: name.map(Into::into), len }
}

/// A deterministic tensor with magnitudes spread over several octaves, signs
/// mixed, and no two channels alike.
fn weights(axes: Vec<Axis>) -> Tensor {
    let n: usize = axes.iter().map(|a| a.len).product();
    let v = (0..n).map(|i| ((i * 7919 % 101) as f64 - 50.0) / 37.0 * (1.0 + (i % 5) as f64)).collect();
    Tensor { axes, data: Buf::F64(v) }
}

fn q(bits: u8, over: AxisRef, scheme: QuantScheme) -> Quant {
    Quant { bits, over, scheme }
}

/// `|w − f32(q·s)| ≤ s/2`, plus one `f32` rounding of the product, and the
/// largest magnitude of every channel lands on `±qmax`.
#[test]
fn absmax_meets_its_error_rule() {
    for bits in [8u8, 4] {
        let t = weights(vec![ax(None, 16), ax(None, 6)]);
        let qt = QTensor::quantize(&t, &q(bits, AxisRef::Pos(-2), QuantScheme::Absmax)).unwrap();
        assert_eq!(qt.scales.len(), 6, "one scale per output column");
        let w = t.data.to_f64();
        let Buf::F32(d) = qt.dequantize().data else { panic!() };
        let qmax = qt.quant.qmax() as i8;
        for (e, (x, y)) in w.iter().zip(&d).enumerate() {
            let s = qt.scale(e) as f64;
            assert!((x - *y as f64).abs() <= s / 2.0 + (*y as f64).abs() * 2f64.powi(-24), "int{bits} element {e}: {x} vs {y}, s = {s}");
            assert!(qt.int(e).abs() <= qmax);
        }
        for c in 0..6 {
            let top = (0..16).map(|k| qt.int(k * 6 + c).abs()).max().unwrap();
            assert_eq!(top, qmax, "int{bits}: channel {c} uses its full range");
        }
    }
}

/// Under `pow2` the product `q·s` is exact in `f32`, so the dequantised value
/// is `q·s` with no rounding and the error is `s/2` exactly as stated.
#[test]
fn pow2_scales_are_exact() {
    let t = weights(vec![ax(Some("model"), 12), ax(Some("vocab"), 5)]);
    let qt = QTensor::quantize(&t, &q(8, AxisRef::Name("model".into()), QuantScheme::Pow2)).unwrap();
    let w = t.data.to_f64();
    let Buf::F32(d) = qt.dequantize().data else { panic!() };
    for (e, (x, y)) in w.iter().zip(&d).enumerate() {
        let s = qt.scale(e) as f64;
        assert_eq!(s.log2().fract(), 0.0, "a power of two");
        assert_eq!(*y as f64, qt.int(e) as f64 * s, "q·s is exact");
        assert!((x - *y as f64).abs() <= s / 2.0);
    }
    for c in 0..5 {
        let m = (0..12).map(|k| w[k * 5 + c].abs()).fold(0.0, f64::max);
        let s = qt.scales[c] as f64;
        assert!(m / s <= 127.0 && m / (s / 2.0) > 127.0, "the smallest power of two that fits: m = {m}, s = {s}");
    }
}

#[test]
fn int4_packs_two_to_a_byte() {
    // Odd length, both signs, the extremes.
    let v = vec![7.0, -7.0, 0.0, 3.0, -1.0, 5.0, -4.0];
    let t = Tensor { axes: vec![ax(None, 7), ax(None, 1)], data: Buf::F64(v.clone().into()) };
    let qt = QTensor::quantize(&t, &q(4, AxisRef::Pos(0), QuantScheme::Absmax)).unwrap();
    assert_eq!(qt.data.len(), 4);
    assert_eq!(qt.scales, vec![1.0]);
    let ints: Vec<i8> = (0..7).map(|e| qt.int(e)).collect();
    assert_eq!(ints, vec![7, -7, 0, 3, -1, 5, -4]);
    assert_eq!(qt.dequantize().data, Buf::F32(v.iter().map(|x| *x as f32).collect()));
}

/// The zero-initialized head: a channel of zeros stores zeros and a zero
/// scale, and never divides by it.
#[test]
fn a_zero_channel_stays_zero() {
    let t = Tensor { axes: vec![ax(None, 3), ax(None, 2)], data: Buf::F64(vec![0.0, 1.0, 0.0, -2.0, 0.0, 0.5].into()) };
    let qt = QTensor::quantize(&t, &q(8, AxisRef::Pos(0), QuantScheme::Absmax)).unwrap();
    assert_eq!(qt.scales[0], 0.0);
    let Buf::F32(d) = qt.dequantize().data else { panic!() };
    assert_eq!([d[0], d[2], d[4]], [0.0, 0.0, 0.0]);
    assert!(d.iter().all(|x| x.is_finite()));
}

/// `stack{L}` adds a leading axis; counting the contracted axis from the end
/// still finds it, and every layer gets scales of its own.
#[test]
fn stacked_parameters_get_scales_per_layer() {
    let mut t = weights(vec![ax(None, 2), ax(None, 4), ax(None, 3)]);
    if let Buf::F64(v) = &mut t.data {
        // The second layer is the first, a hundred times larger.
        for i in 0..12 {
            v[12 + i] = v[i] * 100.0;
        }
    }
    let qt = QTensor::quantize(&t, &q(8, AxisRef::Pos(-2), QuantScheme::Absmax)).unwrap();
    assert_eq!(qt.over, 1);
    assert_eq!(qt.scales.len(), 6);
    for c in 0..3 {
        assert!((qt.scales[3 + c] / qt.scales[c] - 100.0).abs() < 1e-3, "layer 2's scale follows layer 2's weights");
    }
}

#[test]
fn a_bad_axis_is_refused() {
    let t = weights(vec![ax(Some("a"), 2), ax(Some("a"), 2)]);
    assert!(QTensor::quantize(&t, &q(8, AxisRef::Name("a".into()), QuantScheme::Absmax)).is_err(), "ambiguous name");
    assert!(QTensor::quantize(&t, &q(8, AxisRef::Name("b".into()), QuantScheme::Absmax)).is_err(), "missing name");
    assert!(QTensor::quantize(&t, &q(8, AxisRef::Pos(2), QuantScheme::Absmax)).is_err(), "out of range");
    assert!(QTensor::quantize(&t, &q(6, AxisRef::Pos(0), QuantScheme::Absmax)).is_err(), "six bits");
}

fn tree() -> Value {
    let leaf = |n| Value::Real(weights(vec![ax(None, n), ax(None, 3)]));
    Value::Record(vec![
        ("tok".into(), leaf(4)),
        ("blocks".into(), Value::Record(vec![("up".into(), Value::Record(vec![("w".into(), leaf(5)), ("b".into(), Value::Real(weights(vec![ax(None, 3)])))]))])),
        ("pair".into(), Value::Tuple(vec![leaf(2), leaf(2)])),
    ])
}

#[test]
fn parameters_are_named_by_path() {
    assert_eq!(param_paths(&tree()), vec!["tok", "blocks.up.w", "blocks.up.b", "pair.0", "pair.1"]);
}

/// Only the named leaves change, and they change to `f32(q·s)`.
#[test]
fn dequantised_parameters_replace_only_their_leaves() {
    let theta = tree();
    let mut m = BTreeMap::new();
    m.insert("blocks.up.w".to_string(), q(8, AxisRef::Pos(-2), QuantScheme::Absmax));
    m.insert("pair.1".to_string(), q(4, AxisRef::Pos(0), QuantScheme::Pow2));
    let qs = quantize_params(&theta, &m).unwrap();
    let out = dequantize_params(&theta, &qs);
    assert_eq!(out.field("tok"), theta.field("tok"));
    let Some(Value::Record(up)) = out.field("blocks").and_then(|b| b.field("up")) else { panic!() };
    let Value::Real(w) = &up[0].1 else { panic!() };
    assert_eq!(w.data, qs["blocks.up.w"].dequantize().data);
    assert_eq!(&up[1].1, theta.field("blocks").unwrap().field("up").unwrap().field("b").unwrap());

    // What an entry runs on keeps the integers (step 5b), and reads as the
    // same values everywhere.
    let mut r = Realization::uniform(Dtype::F32);
    r.quant = m;
    let run_on = r.stored_params(&theta).unwrap();
    let Some(Value::Real(qw)) = run_on.field("blocks").and_then(|b| b.field("up")).and_then(|u| u.field("w")) else { panic!() };
    assert!(matches!(qw.data, Buf::Quant(_)));
    assert_eq!(qw.data.to_f64(), w.data.to_f64());
    assert_eq!(run_on.sum_squares_f64(), out.sum_squares_f64());
    assert_eq!(Realization::uniform(Dtype::F32).stored_params(&theta).unwrap(), theta);
}

/// A misspelt path must fail loudly, never quantise nothing.
#[test]
fn an_unknown_parameter_is_refused() {
    let mut m = BTreeMap::new();
    m.insert("blocks.up.W".to_string(), q(8, AxisRef::Pos(-2), QuantScheme::Absmax));
    let e = quantize_params(&tree(), &m).unwrap_err();
    assert!(e.message.contains("blocks.up.W") && e.message.contains("blocks.up.w"), "{}", e.message);
}

#[test]
fn a_quantised_realization_survives_its_file_form() {
    let mut r = Realization::uniform(Dtype::F32);
    r.quant.insert("head".into(), q(8, AxisRef::Pos(-2), QuantScheme::Absmax));
    r.quant.insert("tok".into(), q(4, AxisRef::Name("model".into()), QuantScheme::Pow2));
    let text = r.to_json();
    assert!(text.contains("\"over\": -2") && text.contains("\"over\": \"model\""), "{text}");
    let back: Realization = serde_json::from_str(&text).unwrap();
    assert_eq!(back, r);
    assert!(!r.is_uniform());
    assert_eq!(r.label(), "f32+sequential+2quant");
    // The scheme defaults to absmax, and a file written before step 5 still loads.
    let old: Realization = serde_json::from_str(r#"{"default":"f64","order":{"kind":"sequential"},"quant":{"w":{"bits":8,"over":0}}}"#).unwrap();
    assert_eq!(old.quant["w"].scheme, QuantScheme::Absmax);
    let older: Realization = serde_json::from_str(r#"{"default":"f64","order":{"kind":"sequential"}}"#).unwrap();
    assert!(older.quant.is_empty());
}

/// The placement of the scale (plan 0.6.0 §3.3). A contraction over the
/// axis the scales are constant along can apply the scale to each loaded
/// weight — the denotation — or once to the finished sum. With `pow2` scales
/// the two are **bit-identical**, because multiplying by a power of two is
/// exact in `f32` and so commutes with every rounding of the sum; with
/// `absmax` scales they are not, which is what makes `pow2` the scheme under
/// which the cheaper placement is free.
#[test]
fn pow2_scales_can_be_applied_after_the_sum() {
    let (k, n) = (64, 16);
    let t = weights(vec![ax(None, k), ax(None, n)]);
    let x: Vec<f32> = (0..k).map(|i| ((i * 31 % 17) as f32 - 8.0) / 3.0).collect();
    let differs = |scheme| {
        let qt = QTensor::quantize(&t, &q(8, AxisRef::Pos(0), scheme)).unwrap();
        let mut differ = 0;
        for c in 0..n {
            let s = qt.scales[c];
            // On load: Σ x_k · f32(q_k·s), in index order.
            let on_load = (0..k).fold(0f32, |acc, j| acc + x[j] * (qt.int(j * n + c) as f32 * s));
            // After the sum: s · Σ x_k · q_k, in index order.
            let after = s * (0..k).fold(0f32, |acc, j| acc + x[j] * qt.int(j * n + c) as f32);
            if on_load.to_bits() != after.to_bits() {
                differ += 1;
            }
        }
        differ
    };
    assert_eq!(differs(QuantScheme::Pow2), 0, "pow2: the two placements are bit-identical");
    assert!(differs(QuantScheme::Absmax) > 0, "absmax: they differ, so the test above is not vacuous");
}

/// A layer's slice of a stacked parameter stays quantised — `int4` too, when
/// the slice starts on a byte — and reads as the same values as the layer's
/// part of the whole. An odd-length `int4` slice would split a byte, and is
/// refused, so the caller falls back to the values.
#[test]
fn stacked_slices_stay_quantised() {
    use presk_rt::QBuf;
    for (bits, rows) in [(8u8, 4usize), (4, 4), (4, 3)] {
        let t = weights(vec![ax(None, 3), ax(None, rows), ax(None, 3)]);
        let Buf::Quant(q) = QTensor::quantize(&t, &q(bits, AxisRef::Pos(-2), QuantScheme::Absmax)).unwrap().to_tensor().data else { panic!() };
        let per = rows * 3;
        for i in 0..3 {
            let s: Option<QBuf> = q.slice_leading(i, 3);
            if bits == 4 && per % 2 == 1 {
                assert!(s.is_none(), "an odd int4 slice would split a byte");
                continue;
            }
            let s = s.unwrap();
            assert_eq!(s.over, 0);
            let got: Vec<f32> = (0..per).map(|e| s.value(e)).collect();
            let want: Vec<f32> = (0..per).map(|e| q.value(i * per + e)).collect();
            assert_eq!(got, want, "int{bits}, layer {i}");
        }
    }
}

/// `pow2` on a channel whose largest magnitude is tiny (plan 0.9.0 step 7,
/// D84). The scale loops started at `2^⌈log2(m/qmax)⌉`, which `powi` makes 0
/// when `m/qmax` is at most 2⁻¹⁰²⁴, and never ended; they now start at 1.
/// Wherever the old start
/// ended, the new one gives the same scale: both are the smallest power of
/// two with `m/s ≤ qmax`.
#[test]
fn pow2_scales_of_tiny_channels_end() {
    let old = |m: f64, qmax: f64| -> Option<f32> {
        let mut s = 2f64.powi((m / qmax).log2().ceil() as i32);
        let mut steps = 0;
        while m / (s / 2.0) <= qmax {
            s /= 2.0;
            steps += 1;
        }
        while m / s > qmax {
            s *= 2.0;
            steps += 1;
            if steps > 10_000 {
                return None;
            }
        }
        Some(s as f32)
    };
    let (mut compared, mut hung) = (0, 0);
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    for i in 0..200_000u64 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        // Every exponent, subnormals and infinity included.
        let m = match i {
            0 => f64::from_bits(1),
            1 => f64::MIN_POSITIVE,
            2 => f64::MAX,
            3 => f64::INFINITY,
            _ => f64::from_bits(x & 0x7fff_ffff_ffff_ffff).min(f64::INFINITY),
        };
        if m.is_nan() || m == 0.0 {
            continue;
        }
        for bits in [8u8, 4] {
            let t = Tensor { axes: vec![ax(None, 2), ax(None, 1)], data: Buf::F64(vec![m, -m / 3.0].into()) };
            let qt = QTensor::quantize(&t, &q(bits, AxisRef::Pos(0), QuantScheme::Pow2)).unwrap();
            match old(m, ((1 << (bits - 1)) - 1) as f64) {
                Some(s) => {
                    assert_eq!(qt.scales[0].to_bits(), s.to_bits(), "int{bits}, m = {m:e}");
                    compared += 1;
                }
                None => {
                    assert_eq!(qt.scales[0], 0.0, "int{bits}, m = {m:e}: a scale below every f32");
                    assert_eq!(qt.int(0), 0);
                    hung += 1;
                }
            }
        }
    }
    assert!(compared > 300_000 && hung > 500, "{compared} scales compared, {hung} that never ended before");
}
