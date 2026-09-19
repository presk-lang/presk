//! The primitives: elementwise arithmetic with broadcasting, contractions,
//! reductions, indexing, initializers, and the adjoint primitives of AD.

use presk_core::PrimOp;
use presk_types::ty::Ty;
use rand_core::RngCore;

use crate::eval::{Dims, Interp, Result, err};
use crate::key::{Key, normal, uniform01};
use crate::real::Real;
use crate::value::{Axis, Indices, Tensor, Value};

/// How the axes of an operand map into a target axis list: trailing
/// positional alignment when names agree, else alignment by name.
fn align(src: &[Axis], target: &[Axis]) -> Option<Vec<usize>> {
    if src.len() > target.len() {
        return None;
    }
    let off = target.len() - src.len();
    let positional = src.iter().enumerate().all(|(i, a)| {
        let t = &target[off + i];
        a.len == t.len && (a.name.is_none() || t.name.is_none() || a.name == t.name)
    });
    if positional {
        return Some((0..src.len()).map(|i| off + i).collect());
    }
    let mut out = Vec::new();
    for a in src {
        let name = a.name.as_ref()?;
        let p = target.iter().position(|t| t.name.as_deref() == Some(name) && t.len == a.len)?;
        out.push(p);
    }
    Some(out)
}

/// Materialize `x` broadcast to `target`.
fn broadcast_to<T: Clone>(x: &Tensor<T>, target: &[Axis]) -> Option<Tensor<T>> {
    let map = align(&x.axes, target)?;
    let xs = x.strides();
    let dims: Vec<usize> = target.iter().map(|a| a.len).collect();
    let mut data = Vec::with_capacity(dims.iter().product());
    for idx in Indices::new(dims) {
        let off: usize = map.iter().zip(&xs).map(|(p, s)| idx[*p] * s).sum();
        data.push(x.data[off].clone());
    }
    Some(Tensor { axes: target.to_vec(), data })
}

/// Sum `x` down to `target` (the adjoint of `broadcast_to`).
fn sum_to<R: Real>(x: &Tensor<R>, target: &[Axis], zero: &R) -> Option<Tensor<R>> {
    let map = align(target, &x.axes)?;
    let tstrides = {
        let t = Tensor { axes: target.to_vec(), data: Vec::<R>::new() };
        t.strides()
    };
    let n: usize = target.iter().map(|a| a.len).product();
    let mut data = vec![zero.clone(); n];
    for (flat, idx) in x.indices().enumerate() {
        let off: usize = map.iter().zip(&tstrides).map(|(p, s)| idx[*p] * s).sum();
        data[off] = data[off].add(&x.data[flat]);
    }
    Some(Tensor { axes: target.to_vec(), data })
}

/// The result axes of an elementwise binary operation.
fn result_axes(a: &[Axis], b: &[Axis]) -> Option<Vec<Axis>> {
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let map = align(short, long)?;
    let mut out = long.to_vec();
    for (i, p) in map.iter().enumerate() {
        if out[*p].name.is_none() {
            out[*p].name = short[i].name.clone();
        }
    }
    Some(out)
}

impl<R: Real> Interp<'_, R> {
    fn tensor<'v>(&self, v: &'v Value<R>, what: &str) -> Result<&'v Tensor<R>> {
        match v {
            Value::Real(t) => Ok(t),
            other => err(format!("{what}: expected a real tensor, found a {}", other.kind())),
        }
    }

    fn idx<'v>(&self, v: &'v Value<R>, what: &str) -> Result<(u64, &'v Tensor<i64>)> {
        match v {
            Value::Idx { bound, t } => Ok((*bound, t)),
            other => err(format!("{what}: expected an index tensor, found a {}", other.kind())),
        }
    }

    fn binary(&self, a: &Value<R>, b: &Value<R>, f: &dyn Fn(&R, &R) -> R, what: &str) -> Result<Value<R>> {
        // Hyperparameter arithmetic.
        match (a, b) {
            (Value::Nat(x), Value::Nat(y)) => {
                let (x, y) = (*x as f64, *y as f64);
                let r = f(&R::from_f64(x, self.prec), &R::from_f64(y, self.prec)).to_f64();
                return Ok(if r >= 0.0 && r.fract() == 0.0 { Value::Nat(r as u64) } else { Value::Int(r as i64) });
            }
            (Value::F64(x), Value::F64(y)) => return Ok(Value::F64(f(&R::from_f64(*x, self.prec), &R::from_f64(*y, self.prec)).to_f64())),
            // A scalar against a tree applies to every leaf (the optimizer's `lr · g`).
            (Value::Record(_) | Value::Tuple(_), Value::Real(t)) if t.rank() == 0 => {
                return Ok(a.map_leaves(&mut |x| self.binary(x, b, f, what).unwrap_or_else(|_| x.clone())));
            }
            (Value::Real(t), Value::Record(_) | Value::Tuple(_)) if t.rank() == 0 => {
                return Ok(b.map_leaves(&mut |y| self.binary(a, y, f, what).unwrap_or_else(|_| y.clone())));
            }
            (Value::Record(_) | Value::Tuple(_), _) | (_, Value::Record(_) | Value::Tuple(_)) => {
                return a.zip_leaves(b, &mut |x, y| self.binary(x, y, f, what).map_err(|e| e.message)).map_err(|m| crate::eval::RunError { message: m, span: None });
            }
            _ => {}
        }
        let (ta, tb) = (self.tensor(a, what)?, self.tensor(b, what)?);
        let axes = result_axes(&ta.axes, &tb.axes).ok_or_else(|| crate::eval::RunError { message: format!("{what}: cannot broadcast {} with {}", show_axes(&ta.axes), show_axes(&tb.axes)), span: None })?;
        let (xa, xb) = (broadcast_to(ta, &axes).unwrap(), broadcast_to(tb, &axes).unwrap());
        Ok(Value::Real(Tensor { axes, data: xa.data.iter().zip(&xb.data).map(|(x, y)| f(x, y)).collect() }))
    }

    fn unary(&self, a: &Value<R>, f: &dyn Fn(&R) -> R, what: &str) -> Result<Value<R>> {
        match a {
            Value::Record(_) | Value::Tuple(_) => Ok(a.map_leaves(&mut |x| self.unary(x, f, what).unwrap_or_else(|_| x.clone()))),
            Value::Nat(n) => Ok(Value::F64(f(&R::from_u64(*n, self.prec)).to_f64())),
            Value::Int(n) => Ok(Value::F64(f(&R::from_i64(*n, self.prec)).to_f64())),
            Value::F64(x) => Ok(Value::F64(f(&R::from_f64(*x, self.prec)).to_f64())),
            _ => Ok(Value::Real(self.tensor(a, what)?.map(f))),
        }
    }

    fn gelu(&self, x: &R) -> R {
        // x · Φ(x), Φ(x) = (1 + erf(x/√2)) / 2
        x.mul(&self.phi(x))
    }

    fn phi(&self, x: &R) -> R {
        let sqrt2 = self.real(2).sqrt();
        let one = self.real(1);
        one.add(&x.div(&sqrt2).erf()).div(&self.real(2))
    }

    fn gelu_deriv(&self, x: &R) -> R {
        // Φ(x) + x·φ(x), φ(x) = exp(−x²/2) / √(2π)
        let pi = self.pi();
        let two = self.real(2);
        let dens = x.mul(x).div(&two).neg().exp().div(&two.mul(&pi).sqrt());
        self.phi(x).add(&x.mul(&dens))
    }

    fn pi(&self) -> R {
        // 4·atan(1) is not in the trait; use MPFR-quality series: π = 16·atan(1/5) − 4·atan(1/239)
        // computed through ln? Simpler and exact enough: use the double constant when
        // the precision is ≤ 53 bits, else Machin's formula with a Taylor series.
        machin_pi::<R>(self.prec)
    }

    /// Evaluate one primitive on values (for tests and tools).
    pub fn prim_public(&mut self, op: &PrimOp, args: &[Value<R>]) -> Result<Value<R>> {
        self.prim(op, args, &Ty::Nat, &Dims::new())
    }

    pub(crate) fn prim(&mut self, op: &PrimOp, args: &[Value<R>], _ty: &Ty, dims: &Dims) -> Result<Value<R>> {
        let a = |i: usize| -> Result<&Value<R>> { args.get(i).ok_or_else(|| crate::eval::RunError { message: format!("`{}`: missing argument {i}", op.name()), span: None }) };
        let name = op.name();
        match op {
            PrimOp::Add => self.binary(a(0)?, a(1)?, &|x, y| x.add(y), "add"),
            PrimOp::Sub => self.binary(a(0)?, a(1)?, &|x, y| x.sub(y), "sub"),
            PrimOp::Mul => self.binary(a(0)?, a(1)?, &|x, y| x.mul(y), "mul"),
            PrimOp::Div => self.binary(a(0)?, a(1)?, &|x, y| x.div(y), "div"),
            PrimOp::Neg => self.unary(a(0)?, &|x| x.neg(), "neg"),
            PrimOp::Pow(n) => self.unary(a(0)?, &|x| x.pow_u(*n), "pow"),
            PrimOp::Exp => self.unary(a(0)?, &|x| x.exp(), "exp"),
            PrimOp::Log => self.unary(a(0)?, &|x| x.ln(), "log"),
            PrimOp::Tanh => self.unary(a(0)?, &|x| x.tanh(), "tanh"),
            PrimOp::Sqrt => self.unary(a(0)?, &|x| x.sqrt(), "sqrt"),
            PrimOp::Sin => self.unary(a(0)?, &|x| x.sin(), "sin"),
            PrimOp::Cos => self.unary(a(0)?, &|x| x.cos(), "cos"),
            PrimOp::Abs => self.unary(a(0)?, &|x| x.abs(), "abs"),
            PrimOp::Gelu => self.unary(a(0)?, &|x| self.gelu(x), "gelu"),
            PrimOp::GeluDeriv => self.unary(a(0)?, &|x| self.gelu_deriv(x), "gelu'"),
            PrimOp::Relu => self.unary(a(0)?, &|x| if x.is_negative() { self.real(0) } else { x.clone() }, "relu"),
            PrimOp::Heaviside => self.unary(a(0)?, &|x| if x.is_negative() || x.is_zero() { self.real(0) } else { self.real(1) }, "heaviside"),
            PrimOp::Sign => self.unary(a(0)?, &|x| if x.is_negative() { self.real(1).neg() } else if x.is_zero() { self.real(0) } else { self.real(1) }, "sign"),
            PrimOp::Sigmoid => self.unary(a(0)?, &|x| self.real(1).div(&self.real(1).add(&x.neg().exp())), "sigmoid"),
            PrimOp::Pi => Ok(self.scalar(R::pi(self.prec))),
            PrimOp::Real => match a(0)? {
                Value::Nat(n) => Ok(self.scalar(R::from_u64(*n, self.prec))),
                Value::Int(n) => Ok(self.scalar(R::from_i64(*n, self.prec))),
                Value::F64(x) => Ok(self.scalar(R::from_f64(*x, self.prec))),
                Value::Idx { t, .. } => Ok(Value::Real(t.map(|i| R::from_i64(*i, self.prec)))),
                Value::Real(t) => Ok(Value::Real(t.clone())),
                other => err(format!("real: cannot inject a {}", other.kind())),
            },
            PrimOp::Dot | PrimOp::Matmul => {
                let (x, w) = (self.tensor(a(0)?, &name)?, self.tensor(a(1)?, &name)?);
                self.dot(x, w)
            }
            PrimOp::DotVjpL => {
                // ct[..., N…] · b[K, N…] → a[..., K]
                let (ct, b) = (self.tensor(a(0)?, &name)?, self.tensor(a(1)?, &name)?);
                let k = b.rank() - 1;
                let bt = self.move_to_back(b, 0); // [N…, K]
                self.contract_trailing(ct, &bt, k)
            }
            PrimOp::DotVjpR => {
                // a[..., K], ct[..., N…] → b[K, N…] summing over the leading axes
                let (x, ct) = (self.tensor(a(0)?, &name)?, self.tensor(a(1)?, &name)?);
                let lead = x.rank() - 1;
                self.contract_leading(x, ct, lead)
            }
            PrimOp::Ein { operands, output } => {
                let ts: Vec<Tensor<R>> = args.iter().enumerate().map(|(i, v)| Ok(self.tensor(v, &name)?.with_names(&operands[i].iter().map(|n| Some(n.clone())).collect::<Vec<_>>()))).collect::<Result<_>>()?;
                self.ein(&ts, output)
            }
            PrimOp::SumAll | PrimOp::MeanAll => {
                let t = self.tensor(a(0)?, &name)?;
                let mut acc = self.real(0);
                for x in &t.data {
                    acc = acc.add(x);
                }
                if matches!(op, PrimOp::MeanAll) {
                    acc = acc.div(&self.real(t.numel() as u64));
                }
                Ok(self.scalar(acc))
            }
            PrimOp::Sum { axis } | PrimOp::Mean { axis } => {
                let t = self.tensor(a(0)?, &name)?;
                let p = t.axis_pos(axis).ok_or_else(|| crate::eval::RunError { message: format!("{name}: no axis `{axis}` in {}", show_axes(&t.axes)), span: None })?;
                let mut r = self.reduce_axis(t, p);
                if matches!(op, PrimOp::Mean { .. }) {
                    let n = self.real(t.axes[p].len as u64);
                    r = r.map(|x| x.div(&n));
                }
                Ok(Value::Real(r))
            }
            PrimOp::MeanVjp { axis } => {
                let (ct, tmpl) = (self.tensor(a(0)?, &name)?, self.tensor(a(1)?, &name)?);
                let n = match axis {
                    None => tmpl.numel() as u64,
                    Some(ax) => tmpl.axes.iter().find(|x| x.name.as_deref() == Some(ax)).map(|x| x.len as u64).unwrap_or(1),
                };
                let b = broadcast_to(ct, &tmpl.axes).ok_or_else(|| crate::eval::RunError { message: format!("mean_vjp: cannot broadcast {} to {}", show_axes(&ct.axes), show_axes(&tmpl.axes)), span: None })?;
                let nn = self.real(n);
                Ok(Value::Real(b.map(|x| x.div(&nn))))
            }
            PrimOp::Softmax { axis } => {
                let t = self.tensor(a(0)?, &name)?;
                let p = t.axis_pos(axis).ok_or_else(|| crate::eval::RunError { message: format!("softmax: no axis `{axis}` in {}", show_axes(&t.axes)), span: None })?;
                Ok(Value::Real(self.softmax(t, p)))
            }
            PrimOp::Embed => {
                let table = self.tensor(a(0)?, &name)?;
                let (_, idx) = self.idx(a(1)?, &name)?;
                if table.rank() != 2 {
                    return err("embed: the table must have rank 2");
                }
                let (v, d) = (table.axes[0].len, table.axes[1].len);
                let mut axes = idx.axes.clone();
                axes.push(Axis { name: None, len: d });
                let mut data = Vec::with_capacity(idx.numel() * d);
                for &i in &idx.data {
                    if i < 0 || i as usize >= v {
                        return err(format!("embed: index {i} out of range [0, {v})"));
                    }
                    data.extend_from_slice(&table.data[i as usize * d..(i as usize + 1) * d]);
                }
                Ok(Value::Real(Tensor { axes, data }))
            }
            PrimOp::EmbedVjp => {
                let table = self.tensor(a(0)?, &name)?;
                let (_, idx) = self.idx(a(1)?, &name)?;
                let ct = self.tensor(a(2)?, &name)?;
                let d = table.axes[1].len;
                let mut out = self.zeros_like_tensor(table);
                for (p, &i) in idx.data.iter().enumerate() {
                    for j in 0..d {
                        let o = i as usize * d + j;
                        out.data[o] = out.data[o].add(&ct.data[p * d + j]);
                    }
                }
                Ok(Value::Real(out))
            }
            PrimOp::CrossEntropy => {
                let logits = self.tensor(a(0)?, &name)?;
                let (_, y) = self.idx(a(1)?, &name)?;
                let v = logits.axes.last().ok_or_else(|| crate::eval::RunError { message: "cross_entropy: scalar logits".into(), span: None })?.len;
                let sm = self.softmax(logits, logits.rank() - 1);
                let mut data = Vec::with_capacity(y.numel());
                for (p, &t) in y.data.iter().enumerate() {
                    if t < 0 || t as usize >= v {
                        return err(format!("cross_entropy: target {t} out of range [0, {v})"));
                    }
                    data.push(sm.data[p * v + t as usize].ln().neg());
                }
                Ok(Value::Real(Tensor { axes: y.axes.clone(), data }))
            }
            PrimOp::CrossEntropyVjp => {
                let logits = self.tensor(a(0)?, &name)?;
                let (_, y) = self.idx(a(1)?, &name)?;
                let ct = self.tensor(a(2)?, &name)?;
                let v = logits.axes.last().unwrap().len;
                let mut sm = self.softmax(logits, logits.rank() - 1);
                let one = self.real(1);
                for (p, &t) in y.data.iter().enumerate() {
                    let o = p * v + t as usize;
                    sm.data[o] = sm.data[o].sub(&one);
                    for j in 0..v {
                        sm.data[p * v + j] = sm.data[p * v + j].mul(&ct.data[p]);
                    }
                }
                Ok(Value::Real(sm))
            }
            PrimOp::Argmax => {
                let t = self.tensor(a(0)?, &name)?;
                let v = t.axes.last().ok_or_else(|| crate::eval::RunError { message: "argmax: scalar argument".into(), span: None })?.len;
                let n = t.numel() / v.max(1);
                let data: Vec<i64> = (0..n).map(|p| (0..v).max_by(|&i, &j| t.data[p * v + i].cmp(&t.data[p * v + j])).unwrap_or(0) as i64).collect();
                Ok(Value::Idx { bound: v as u64, t: Tensor { axes: t.axes[..t.rank() - 1].to_vec(), data } })
            }
            PrimOp::Sample => {
                let t = self.tensor(a(0)?, &name)?;
                let v = t.axes.last().ok_or_else(|| crate::eval::RunError { message: "sample: scalar argument".into(), span: None })?.len;
                let sm = self.softmax(t, t.rank() - 1);
                let n = t.numel() / v.max(1);
                let mut data = Vec::with_capacity(n);
                for p in 0..n {
                    let u = uniform01(&mut self.sample_rng);
                    let mut acc = 0.0;
                    let mut pick = v - 1;
                    for i in 0..v {
                        acc += sm.data[p * v + i].to_f64();
                        if u < acc {
                            pick = i;
                            break;
                        }
                    }
                    data.push(pick as i64);
                }
                Ok(Value::Idx { bound: v as u64, t: Tensor { axes: t.axes[..t.rank() - 1].to_vec(), data } })
            }
            PrimOp::IndexConst { axis, index } => match a(0)? {
                Value::Real(t) => Ok(Value::Real(self.select(t, axis, *index)?)),
                Value::Idx { bound, t } => Ok(Value::Idx { bound: *bound, t: self.select(t, axis, *index)? }),
                other => err(format!("index: on a {}", other.kind())),
            },
            PrimOp::ScatterIndex { axis, index } => {
                let (tmpl, ct) = (self.tensor(a(0)?, &name)?, self.tensor(a(1)?, &name)?);
                let p = tmpl.axis_pos(axis).ok_or_else(|| crate::eval::RunError { message: format!("scatter_index: no axis `{axis}`"), span: None })?;
                let len = tmpl.axes[p].len as i64;
                let i = if *index < 0 { len + index } else { *index } as usize;
                let mut out = self.zeros_like_tensor(tmpl);
                let strides = tmpl.strides();
                for (flat, idx) in ct.indices().enumerate() {
                    let mut full: Vec<usize> = idx.clone();
                    full.insert(p, i);
                    let off: usize = full.iter().zip(&strides).map(|(x, s)| x * s).sum();
                    out.data[off] = ct.data[flat].clone();
                }
                Ok(Value::Real(out))
            }
            PrimOp::SplitHeads { h, hd } => {
                let t = self.tensor(a(0)?, &name)?;
                let r = t.rank();
                if r < 2 || t.axes[r - 1].len != (*h * *hd) as usize {
                    return err(format!("split_heads: expected a trailing axis of length {}", h * hd));
                }
                // [..., seq, model] → [..., heads, seq, hd]
                let lead = &t.axes[..r - 2];
                let seq = t.axes[r - 2].clone();
                let mut axes = lead.to_vec();
                axes.push(Axis { name: Some("heads".into()), len: *h as usize });
                axes.push(seq.clone());
                axes.push(Axis { name: Some("hd".into()), len: *hd as usize });
                let mut out = Tensor { axes, data: Vec::with_capacity(t.numel()) };
                let nlead: usize = lead.iter().map(|x| x.len).product::<usize>().max(1);
                for l in 0..nlead {
                    for hh in 0..*h as usize {
                        for s in 0..seq.len {
                            for d in 0..*hd as usize {
                                out.data.push(t.data[(l * seq.len + s) * (*h * *hd) as usize + hh * *hd as usize + d].clone());
                            }
                        }
                    }
                }
                Ok(Value::Real(out))
            }
            PrimOp::MergeHeads { h, hd } => {
                let t = self.tensor(a(0)?, &name)?;
                let r = t.rank();
                if r < 3 {
                    return err("merge_heads: expected [..., heads, seq, hd]");
                }
                let lead = &t.axes[..r - 3];
                let seq = t.axes[r - 2].clone();
                let mut axes = lead.to_vec();
                axes.push(seq.clone());
                axes.push(Axis { name: Some("model".into()), len: (*h * *hd) as usize });
                let mut out = Tensor { axes, data: Vec::with_capacity(t.numel()) };
                let nlead: usize = lead.iter().map(|x| x.len).product::<usize>().max(1);
                let (h, hd) = (*h as usize, *hd as usize);
                for l in 0..nlead {
                    for s in 0..seq.len {
                        for hh in 0..h {
                            for d in 0..hd {
                                out.data.push(t.data[((l * h + hh) * seq.len + s) * hd + d].clone());
                            }
                        }
                    }
                }
                Ok(Value::Real(out))
            }
            PrimOp::Positions { axis, len } => {
                let n = self.dim(len, dims)?;
                Ok(Value::Idx { bound: n, t: Tensor { axes: vec![Axis { name: Some(axis.clone()), len: n as usize }], data: (0..n as i64).collect() } })
            }
            PrimOp::CausalMask { axis, len } => {
                let n = self.dim(len, dims)? as usize;
                let (zero, ninf) = (self.real(0), R::neg_infinity(self.prec));
                let mut data = Vec::with_capacity(n * n);
                for i in 0..n {
                    for j in 0..n {
                        data.push(if j <= i { zero.clone() } else { ninf.clone() });
                    }
                }
                Ok(Value::Real(Tensor { axes: vec![Axis { name: Some(axis.clone()), len: n }, Axis { name: Some(format!("{axis}2")), len: n }], data }))
            }
            PrimOp::Zeros { ty } | PrimOp::Ones { ty } => {
                let axes = self.axes_of(shape_of(ty)?, dims)?;
                let n: usize = axes.iter().map(|x| x.len).product();
                let v = if matches!(op, PrimOp::Zeros { .. }) { self.real(0) } else { self.real(1) };
                Ok(Value::Real(Tensor { axes, data: vec![v; n] }))
            }
            PrimOp::Normal { ty, std } => {
                let axes = self.axes_of(shape_of(ty)?, dims)?;
                let key = self.key_of(a(0)?)?;
                let std: f64 = std.parse().unwrap_or(1.0);
                let mut rng = key.rng();
                let n: usize = axes.iter().map(|x| x.len).product();
                let data = (0..n).map(|_| R::from_f64(normal(&mut rng) * std, self.prec)).collect();
                Ok(Value::Real(Tensor { axes, data }))
            }
            PrimOp::XavierUniform { ty } => {
                let axes = self.axes_of(shape_of(ty)?, dims)?;
                let key = self.key_of(a(0)?)?;
                let (fan_in, fan_out) = match axes.len() {
                    0 => (1, 1),
                    1 => (axes[0].len, axes[0].len),
                    _ => (axes[0].len, axes[axes.len() - 1].len),
                };
                let bound = (6.0 / (fan_in + fan_out) as f64).sqrt();
                let mut rng = key.rng();
                let n: usize = axes.iter().map(|x| x.len).product();
                let data = (0..n).map(|_| R::from_f64((2.0 * uniform01(&mut rng) - 1.0) * bound, self.prec)).collect();
                Ok(Value::Real(Tensor { axes, data }))
            }
            PrimOp::SplitKey { n } => {
                let key = self.key_of(a(0)?)?;
                Ok(Value::Tuple(key.split(*n).into_iter().map(Value::Key).collect()))
            }
            PrimOp::SliceParams { index } => Ok(self.slice_tree(a(0)?, *index as usize)),
            PrimOp::SetSlice => {
                let (stacked, value) = (a(0)?, a(2)?);
                let i = match a(1)? {
                    Value::Nat(i) => *i as usize,
                    other => return err(format!("set_slice: index is a {}", other.kind())),
                };
                let mut leaves: Vec<Value<R>> = Vec::new();
                value.map_leaves(&mut |l| {
                    leaves.push(l.clone());
                    l.clone()
                });
                let mut k = 0;
                Ok(stacked.map_leaves(&mut |leaf| {
                    let v = leaves.get(k).cloned();
                    k += 1;
                    match (leaf, v) {
                        (Value::Real(t), Some(Value::Real(v))) if t.rank() >= 1 && i < t.axes[0].len => {
                            let chunk = t.data.len() / t.axes[0].len;
                            let mut out = t.clone();
                            out.data[i * chunk..(i + 1) * chunk].clone_from_slice(&v.data[..chunk.min(v.data.len())]);
                            Value::Real(out)
                        }
                        (leaf, _) => leaf.clone(),
                    }
                }))
            }
            PrimOp::StackParams { n } => {
                if args.len() != *n as usize {
                    return err("stack_params: wrong number of parts");
                }
                self.stack_trees(args)
            }
            PrimOp::ScatterParams { index } => {
                let (tmpl, ct) = (a(0)?, a(1)?);
                let mut ct_leaves: Vec<Value<R>> = Vec::new();
                ct.map_leaves(&mut |l| {
                    ct_leaves.push(l.clone());
                    l.clone()
                });
                let mut k = 0;
                Ok(tmpl.map_leaves(&mut |leaf| {
                    let c = ct_leaves.get(k).cloned();
                    k += 1;
                    match (leaf, c) {
                        (Value::Real(t), Some(Value::Real(c))) => {
                            let mut out = self.zeros_like_tensor(t);
                            let slice = c.numel();
                            let start = *index as usize * slice;
                            out.data[start..start + slice].clone_from_slice(&c.data);
                            Value::Real(out)
                        }
                        (other, _) => other.clone(),
                    }
                }))
            }
            PrimOp::SumSquares => {
                let mut acc = self.real(0);
                let mut add = |leaf: &Value<R>| {
                    if let Value::Real(t) = leaf {
                        for x in &t.data {
                            acc = acc.add(&x.mul(x));
                        }
                    }
                    leaf.clone()
                };
                a(0)?.map_leaves(&mut add);
                Ok(self.scalar(acc))
            }
            PrimOp::ZerosLike => Ok(a(0)?.map_leaves(&mut |leaf| match leaf {
                Value::Real(t) => Value::Real(self.zeros_like_tensor(t)),
                other => other.clone(),
            })),
            PrimOp::SetField { name: field } => match a(0)? {
                Value::Record(fs) => Ok(Value::Record(fs.iter().map(|(n, v)| (n.clone(), if n == field { a(1).unwrap().clone() } else { v.clone() })).collect())),
                other => err(format!("set_field on a {}", other.kind())),
            },
            PrimOp::SetElem { index } => match a(0)? {
                Value::Tuple(vs) => Ok(Value::Tuple(vs.iter().enumerate().map(|(i, v)| if i == *index { a(1).unwrap().clone() } else { v.clone() }).collect())),
                other => err(format!("set_elem on a {}", other.kind())),
            },
            PrimOp::SumTo => {
                let (x, tmpl) = (self.tensor(a(0)?, &name)?, self.tensor(a(1)?, &name)?);
                let zero = self.real(0);
                sum_to(x, &tmpl.axes, &zero).map(Value::Real).ok_or_else(|| crate::eval::RunError { message: format!("sum_to: cannot reduce {} to {}", show_axes(&x.axes), show_axes(&tmpl.axes)), span: None })
            }
            PrimOp::BroadcastLike => {
                let (x, tmpl) = (self.tensor(a(0)?, &name)?, self.tensor(a(1)?, &name)?);
                broadcast_to(x, &tmpl.axes).map(Value::Real).ok_or_else(|| crate::eval::RunError { message: format!("broadcast_like: cannot broadcast {} to {}", show_axes(&x.axes), show_axes(&tmpl.axes)), span: None })
            }
        }
    }

    // ----- tensor helpers ------------------------------------------------

    fn dot(&self, x: &Tensor<R>, w: &Tensor<R>) -> Result<Value<R>> {
        if x.rank() == 0 || w.rank() == 0 {
            return err("dot: both operands need at least one axis");
        }
        let k = x.axes[x.rank() - 1].len;
        if w.axes[0].len != k {
            return err(format!("dot: contracted axes differ ({k} vs {})", w.axes[0].len));
        }
        let mut axes = x.axes[..x.rank() - 1].to_vec();
        axes.extend(w.axes[1..].iter().cloned());
        let rows = x.numel() / k.max(1);
        let cols = w.numel() / k.max(1);
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                let mut acc = self.real(0);
                for i in 0..k {
                    acc = acc.add(&x.data[r * k + i].mul(&w.data[i * cols + c]));
                }
                data.push(acc);
            }
        }
        Ok(Value::Real(Tensor { axes, data }))
    }

    /// Move axis `p` to the last position.
    fn move_to_back(&self, t: &Tensor<R>, p: usize) -> Tensor<R> {
        let mut perm: Vec<usize> = (0..t.rank()).filter(|&i| i != p).collect();
        perm.push(p);
        self.permute(t, &perm)
    }

    fn permute(&self, t: &Tensor<R>, perm: &[usize]) -> Tensor<R> {
        let axes: Vec<Axis> = perm.iter().map(|&i| t.axes[i].clone()).collect();
        let strides = t.strides();
        let dims: Vec<usize> = axes.iter().map(|a| a.len).collect();
        let mut data = Vec::with_capacity(t.numel());
        for idx in Indices::new(dims) {
            let off: usize = perm.iter().zip(&idx).map(|(&src, &i)| strides[src] * i).sum();
            data.push(t.data[off].clone());
        }
        Tensor { axes, data }
    }

    /// Contract the last `k` axes of `x` with the first `k` axes of `y`
    /// (`y` given as [contracted…, rest…]): result x_lead ++ y_rest.
    fn contract_trailing(&self, x: &Tensor<R>, y: &Tensor<R>, k: usize) -> Result<Value<R>> {
        let kx: usize = x.axes[x.rank() - k..].iter().map(|a| a.len).product();
        let ky: usize = y.axes[..k].iter().map(|a| a.len).product();
        if kx != ky {
            return err("contraction: sizes differ");
        }
        let mut axes = x.axes[..x.rank() - k].to_vec();
        axes.extend(y.axes[k..].iter().cloned());
        let rows = x.numel() / kx.max(1);
        let cols = y.numel() / ky.max(1);
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                let mut acc = self.real(0);
                for i in 0..kx {
                    acc = acc.add(&x.data[r * kx + i].mul(&y.data[i * cols + c]));
                }
                data.push(acc);
            }
        }
        Ok(Value::Real(Tensor { axes, data }))
    }

    /// Contract the first `lead` axes of `x` with the first `lead` axes of `y`:
    /// result x_rest ++ y_rest.
    fn contract_leading(&self, x: &Tensor<R>, y: &Tensor<R>, lead: usize) -> Result<Value<R>> {
        let lx: usize = x.axes[..lead].iter().map(|a| a.len).product();
        let ly: usize = y.axes[..lead].iter().map(|a| a.len).product();
        if lx != ly {
            return err("contraction: leading sizes differ");
        }
        let mut axes = x.axes[lead..].to_vec();
        axes.extend(y.axes[lead..].iter().cloned());
        let rx = x.numel() / lx.max(1);
        let ry = y.numel() / ly.max(1);
        let mut data = Vec::with_capacity(rx * ry);
        for i in 0..rx {
            for j in 0..ry {
                let mut acc = self.real(0);
                for l in 0..lx {
                    acc = acc.add(&x.data[l * rx + i].mul(&y.data[l * ry + j]));
                }
                data.push(acc);
            }
        }
        Ok(Value::Real(Tensor { axes, data }))
    }

    /// Einstein summation over named axes.
    fn ein(&self, ts: &[Tensor<R>], output: &[String]) -> Result<Value<R>> {
        // Lengths by name.
        let mut names: Vec<(String, usize)> = Vec::new();
        for t in ts {
            for a in &t.axes {
                let n = a.name.clone().ok_or_else(|| crate::eval::RunError { message: "ein: unnamed axis".into(), span: None })?;
                match names.iter().find(|(m, _)| *m == n) {
                    Some((_, l)) if *l != a.len => return err(format!("ein: axis `{n}` has lengths {l} and {}", a.len)),
                    Some(_) => {}
                    None => names.push((n, a.len)),
                }
            }
        }
        for o in output {
            if !names.iter().any(|(n, _)| n == o) {
                return err(format!("ein: output axis `{o}` appears in no operand"));
            }
        }
        let contracted: Vec<(String, usize)> = names.iter().filter(|(n, _)| !output.contains(n)).cloned().collect();
        let out_axes: Vec<Axis> = output.iter().map(|o| Axis { name: Some(o.clone()), len: names.iter().find(|(n, _)| n == o).unwrap().1 }).collect();
        // For each operand, the position of each of its axes in (output ++ contracted).
        let all: Vec<String> = output.iter().cloned().chain(contracted.iter().map(|(n, _)| n.clone())).collect();
        let maps: Vec<Vec<usize>> = ts.iter().map(|t| t.axes.iter().map(|a| all.iter().position(|n| Some(n) == a.name.as_ref()).unwrap()).collect()).collect();
        let strides: Vec<Vec<usize>> = ts.iter().map(|t| t.strides()).collect();
        let cdims: Vec<usize> = contracted.iter().map(|(_, l)| *l).collect();
        let mut data = Vec::with_capacity(out_axes.iter().map(|a| a.len).product());
        for oidx in Indices::new(out_axes.iter().map(|a| a.len).collect()) {
            let mut acc = self.real(0);
            for cidx in Indices::new(cdims.clone()) {
                let full: Vec<usize> = oidx.iter().chain(cidx.iter()).copied().collect();
                let mut prod = self.real(1);
                for (t, (map, st)) in ts.iter().zip(maps.iter().zip(&strides)) {
                    let off: usize = map.iter().zip(st).map(|(p, s)| full[*p] * s).sum();
                    prod = prod.mul(&t.data[off]);
                }
                acc = acc.add(&prod);
            }
            data.push(acc);
        }
        Ok(Value::Real(Tensor { axes: out_axes, data }))
    }

    fn reduce_axis(&self, t: &Tensor<R>, p: usize) -> Tensor<R> {
        let mut axes = t.axes.clone();
        let n = axes.remove(p).len;
        let strides = t.strides();
        let out_dims: Vec<usize> = axes.iter().map(|a| a.len).collect();
        let mut data = Vec::with_capacity(out_dims.iter().product());
        for idx in Indices::new(out_dims) {
            let mut acc = self.real(0);
            for i in 0..n {
                let mut full = idx.clone();
                full.insert(p, i);
                let off: usize = full.iter().zip(&strides).map(|(x, s)| x * s).sum();
                acc = acc.add(&t.data[off]);
            }
            data.push(acc);
        }
        Tensor { axes, data }
    }

    fn softmax(&self, t: &Tensor<R>, p: usize) -> Tensor<R> {
        let n = t.axes[p].len;
        let strides = t.strides();
        let mut out = t.clone();
        let mut outer: Vec<usize> = t.dims();
        outer.remove(p);
        for idx in Indices::new(outer) {
            let offs: Vec<usize> = (0..n)
                .map(|i| {
                    let mut full = idx.clone();
                    full.insert(p, i);
                    full.iter().zip(&strides).map(|(x, s)| x * s).sum()
                })
                .collect();
            let mut m = t.data[offs[0]].clone();
            for &o in &offs[1..] {
                if t.data[o].cmp(&m) == std::cmp::Ordering::Greater {
                    m = t.data[o].clone();
                }
            }
            let exps: Vec<R> = offs.iter().map(|&o| t.data[o].sub(&m).exp()).collect();
            let mut z = self.real(0);
            for e in &exps {
                z = z.add(e);
            }
            for (o, e) in offs.iter().zip(exps) {
                out.data[*o] = e.div(&z);
            }
        }
        out
    }

    fn select<T: Clone>(&self, t: &Tensor<T>, axis: &str, index: i64) -> Result<Tensor<T>> {
        let p = t.axis_pos(axis).ok_or_else(|| crate::eval::RunError { message: format!("index: no axis `{axis}` in {}", show_axes(&t.axes)), span: None })?;
        let len = t.axes[p].len as i64;
        let i = if index < 0 { len + index } else { index };
        if i < 0 || i >= len {
            return err(format!("index {index} out of range for axis `{axis}` of length {len}"));
        }
        Ok(self.select_pos(t, p, i as usize))
    }

    fn select_pos<T: Clone>(&self, t: &Tensor<T>, p: usize, i: usize) -> Tensor<T> {
        let mut axes = t.axes.clone();
        axes.remove(p);
        let strides = t.strides();
        let dims: Vec<usize> = axes.iter().map(|a| a.len).collect();
        let mut data = Vec::with_capacity(dims.iter().product());
        for idx in Indices::new(dims) {
            let mut full = idx.clone();
            full.insert(p, i);
            let off: usize = full.iter().zip(&strides).map(|(x, s)| x * s).sum();
            data.push(t.data[off].clone());
        }
        Tensor { axes, data }
    }

    /// Stack trees leaf by leaf along a new leading axis.
    /// Slice `i` along the leading axis of every tensor leaf of a stacked value.
    pub(crate) fn slice_tree(&self, v: &Value<R>, i: usize) -> Value<R> {
        v.map_leaves(&mut |leaf| match leaf {
            Value::Real(t) if t.rank() >= 1 => Value::Real(self.select_pos(t, 0, i)),
            other => other.clone(),
        })
    }

    pub(crate) fn stack_trees(&self, parts: &[Value<R>]) -> Result<Value<R>> {
        match &parts[0] {
            Value::Real(first) => {
                let mut axes = vec![Axis { name: None, len: parts.len() }];
                axes.extend(first.axes.iter().cloned());
                let mut data = Vec::with_capacity(first.numel() * parts.len());
                for p in parts {
                    let t = self.tensor(p, "stack_params")?;
                    data.extend(t.data.iter().cloned());
                }
                Ok(Value::Real(Tensor { axes, data }))
            }
            Value::Record(fs) => {
                let mut out = Vec::new();
                for (i, (n, _)) in fs.iter().enumerate() {
                    let subs: Vec<Value<R>> = parts.iter().map(|p| match p {
                        Value::Record(g) => g[i].1.clone(),
                        other => other.clone(),
                    }).collect();
                    out.push((n.clone(), self.stack_trees(&subs)?));
                }
                Ok(Value::Record(out))
            }
            Value::Tuple(vs) => {
                let mut out = Vec::new();
                for i in 0..vs.len() {
                    let subs: Vec<Value<R>> = parts.iter().map(|p| match p {
                        Value::Tuple(g) => g[i].clone(),
                        other => other.clone(),
                    }).collect();
                    out.push(self.stack_trees(&subs)?);
                }
                Ok(Value::Tuple(out))
            }
            other => Ok(other.clone()),
        }
    }
}

fn shape_of(ty: &Ty) -> Result<&presk_types::ty::Shape> {
    match ty {
        Ty::Real(s) | Ty::Idx { shape: s, .. } => Ok(s),
        _ => err("initializer without a tensor type"),
    }
}

pub(crate) fn show_axes(axes: &[Axis]) -> String {
    let parts: Vec<String> = axes.iter().map(|a| match &a.name {
        Some(n) => format!("{n}: {}", a.len),
        None => a.len.to_string(),
    }).collect();
    format!("[{}]", parts.join(", "))
}

/// π by Machin's formula, π = 16·atan(1/5) − 4·atan(1/239), with the Taylor
/// series of atan; converges in a few hundred terms at 256 bits.
fn machin_pi<R: Real>(prec: u32) -> R {
    fn atan_inv<R: Real>(n: u64, prec: u32) -> R {
        let x = R::from_u64(1, prec).div(&R::from_u64(n, prec));
        let x2 = x.mul(&x);
        let mut term = x.clone();
        let mut acc = R::from_u64(0, prec);
        let mut k: u64 = 1;
        let eps = R::from_f64(2f64.powi(-(prec as i32) - 8), prec);
        loop {
            let t = term.div(&R::from_u64(k, prec));
            acc = if (k / 2) % 2 == 0 { acc.add(&t) } else { acc.sub(&t) };
            term = term.mul(&x2);
            k += 2;
            if t.abs().cmp(&eps) == std::cmp::Ordering::Less {
                break;
            }
        }
        acc
    }
    let a = atan_inv::<R>(5, prec).mul(&R::from_u64(16, prec));
    let b = atan_inv::<R>(239, prec).mul(&R::from_u64(4, prec));
    a.sub(&b)
}

#[allow(dead_code)]
fn _key_use(_: &Key) {}
#[allow(dead_code)]
fn _rng_use(r: &mut rand_chacha::ChaCha20Rng) -> u64 {
    r.next_u64()
}
