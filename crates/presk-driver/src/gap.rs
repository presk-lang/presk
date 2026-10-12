//! Distances between runtime values, and the parameters inside them.

use presk_rt::{Tensor, Value};

/// The real tensor at `path` in a parameter tree.
pub fn param_leaf<'v>(theta: &'v Value, path: &str) -> Option<&'v Tensor> {
    let mut v = theta;
    for part in path.split('.') {
        v = match v {
            Value::Record(_) => v.field(part)?,
            Value::Tuple(xs) => xs.get(part.parse::<usize>().ok()?)?,
            _ => return None,
        };
    }
    match v {
        Value::Real(t) => Some(t),
        _ => None,
    }
}

/// The cotangent of the parameters in a vjp's result: the result itself when
/// it is shaped like `theta`, else the first component of a tuple that is.
pub fn params_cotangent(out: &Value, theta: &Value) -> Option<Value> {
    let paths = presk_rt::quant::param_paths(theta);
    if presk_rt::quant::param_paths(out) == paths {
        return Some(out.clone());
    }
    match out {
        Value::Tuple(xs) => xs.iter().find(|x| presk_rt::quant::param_paths(x) == paths).cloned(),
        _ => None,
    }
}

/// A non-finite element counts as +∞: `f64::max` would pass over a NaN.
fn or_infinite(x: f64) -> f64 {
    if x.is_finite() { x } else { f64::INFINITY }
}

/// The largest `|x|` over the leaves of a runtime value; +∞ when an element
/// is not finite.
pub fn leaf_max(v: &Value) -> f64 {
    match v {
        Value::Real(t) => t.data.to_f64().into_iter().map(|x| or_infinite(x.abs())).fold(0.0, f64::max),
        Value::Tuple(xs) => xs.iter().map(leaf_max).fold(0.0, f64::max),
        Value::Record(xs) => xs.iter().map(|(_, x)| leaf_max(x)).fold(0.0, f64::max),
        _ => 0.0,
    }
}

/// The largest absolute difference over the leaves of two runtime values. A
/// difference that is not finite, an element that overflowed or became NaN
/// against a finite reference, makes the gap +∞, as the reference's gap is
/// (its chapter 4); until 0.10.0 step 8 such an element was left out, so a
/// realization that overflowed was measured on the rest and could be
/// accepted. Since 0.10.0 step 7b no tensor holds `−∞`, the only infinity
/// a legitimate value had.
pub fn leaf_gap(a: &Value, b: &Value) -> f64 {
    match (a, b) {
        (Value::Real(x), Value::Real(y)) => x.data.to_f64().into_iter().zip(y.data.to_f64()).map(|(p, q)| or_infinite((p - q).abs())).fold(0.0, f64::max),
        (Value::Tuple(xs), Value::Tuple(ys)) => xs.iter().zip(ys).map(|(x, y)| leaf_gap(x, y)).fold(0.0, f64::max),
        (Value::Record(xs), Value::Record(ys)) => xs.iter().zip(ys).map(|((_, x), (_, y))| leaf_gap(x, y)).fold(0.0, f64::max),
        _ => 0.0,
    }
}
