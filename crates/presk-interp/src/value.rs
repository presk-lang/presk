//! Runtime values: tensors with named axes, indices, hyperparameters, keys,
//! tuples, parameter records and function values.

use std::sync::Arc;

use presk_core::FuncId;

use crate::key::Key;
use crate::real::Real;

#[derive(Clone, Debug, PartialEq)]
pub struct Axis {
    pub name: Option<String>,
    pub len: usize,
}

/// A dense row-major tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor<T> {
    pub axes: Vec<Axis>,
    pub data: Vec<T>,
}

impl<T: Clone> Tensor<T> {
    pub fn scalar(x: T) -> Self {
        Tensor { axes: vec![], data: vec![x] }
    }

    pub fn rank(&self) -> usize {
        self.axes.len()
    }

    pub fn numel(&self) -> usize {
        self.axes.iter().map(|a| a.len).product()
    }

    pub fn dims(&self) -> Vec<usize> {
        self.axes.iter().map(|a| a.len).collect()
    }

    pub fn strides(&self) -> Vec<usize> {
        let mut s = vec![1; self.axes.len()];
        for i in (0..self.axes.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * self.axes[i + 1].len;
        }
        s
    }

    pub fn axis_pos(&self, name: &str) -> Option<usize> {
        self.axes.iter().position(|a| a.name.as_deref() == Some(name))
    }

    /// Flat offset of a multi-index.
    pub fn offset(&self, idx: &[usize]) -> usize {
        self.strides().iter().zip(idx).map(|(s, i)| s * i).sum()
    }

    pub fn map<U: Clone>(&self, f: impl Fn(&T) -> U) -> Tensor<U> {
        Tensor { axes: self.axes.clone(), data: self.data.iter().map(f).collect() }
    }

    /// Iterate over all multi-indices in row-major order.
    pub fn indices(&self) -> Indices {
        Indices::new(self.dims())
    }

    /// Rename the axes positionally.
    pub fn with_names(&self, names: &[Option<String>]) -> Tensor<T> {
        let mut t = self.clone();
        for (a, n) in t.axes.iter_mut().zip(names) {
            a.name = n.clone();
        }
        t
    }
}

/// Row-major multi-index iterator.
pub struct Indices {
    dims: Vec<usize>,
    cur: Vec<usize>,
    done: bool,
}

impl Indices {
    pub fn new(dims: Vec<usize>) -> Self {
        let done = dims.contains(&0);
        Indices { cur: vec![0; dims.len()], dims, done }
    }
}

impl Iterator for Indices {
    type Item = Vec<usize>;
    fn next(&mut self) -> Option<Vec<usize>> {
        if self.done {
            return None;
        }
        let out = self.cur.clone();
        let mut i = self.dims.len();
        loop {
            if i == 0 {
                self.done = true;
                break;
            }
            i -= 1;
            self.cur[i] += 1;
            if self.cur[i] < self.dims[i] {
                break;
            }
            self.cur[i] = 0;
        }
        Some(out)
    }
}

/// A binary operation on leaves, for [`Value::zip_leaves`].
pub type LeafCombiner<'a, R> = dyn FnMut(&Value<R>, &Value<R>) -> Result<Value<R>, String> + 'a;

/// The elements of a tuple or a record, shared rather than owned. The
/// evaluator clones a value at every use, and a parameter record can hold
/// hundreds of millions of exact reals — BERT-large at MLPerf's constants
/// would have held about eight copies of its 335M parameters, some 180 GB
/// (0.6.0 step 8) — so cloning a tree copies pointers and never a tensor. Build with
/// [`Value::tuple`] and [`Value::record`]; read through `Deref`.
pub type Shared<T> = Arc<Vec<T>>;

#[derive(Clone, Debug)]
pub enum Value<R: Real> {
    Real(Tensor<R>),
    Idx { bound: u64, t: Tensor<i64> },
    Nat(u64),
    Int(i64),
    F64(f64),
    Bool(bool),
    Key(Key),
    Tuple(Shared<Value<R>>),
    Record(Shared<(String, Value<R>)>),
    Closure { f: FuncId, captured: Vec<Value<R>> },
    /// A leaf with no axis to stack along (a key, a scalar, a function
    /// value), stacked by a scanned `fold`: one value per layer, as a stacked
    /// tensor holds one slice per layer. Its type is the leaf's own
    /// ([`presk_core::ir::stack_ty`] leaves such a type unchanged).
    Stacked(Shared<Value<R>>),
}

impl<R: Real> Value<R> {
    pub fn tuple(vs: Vec<Value<R>>) -> Self {
        Value::Tuple(Arc::new(vs))
    }

    pub fn record(fs: Vec<(String, Value<R>)>) -> Self {
        Value::Record(Arc::new(fs))
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Value::Real(_) => "real tensor",
            Value::Idx { .. } => "index tensor",
            Value::Nat(_) => "nat",
            Value::Int(_) => "int",
            Value::F64(_) => "f64",
            Value::Bool(_) => "bool",
            Value::Key(_) => "key",
            Value::Tuple(_) => "tuple",
            Value::Record(_) => "record",
            Value::Closure { .. } => "function",
            Value::Stacked(_) => "stacked value",
        }
    }

    pub fn field(&self, name: &str) -> Option<&Value<R>> {
        match self {
            Value::Record(fs) => fs.iter().find(|(n, _)| n == name).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Apply `f` to every tensor leaf of a tree (records, tuples), keeping the structure.
    pub fn map_leaves(&self, f: &mut dyn FnMut(&Value<R>) -> Value<R>) -> Value<R> {
        match self {
            Value::Tuple(vs) => Value::tuple(vs.iter().map(|v| v.map_leaves(f)).collect()),
            Value::Record(fs) => Value::record(fs.iter().map(|(n, v)| (n.clone(), v.map_leaves(f))).collect()),
            leaf => f(leaf),
        }
    }

    /// Combine two trees of the same structure leaf by leaf.
    pub fn zip_leaves(&self, other: &Value<R>, f: &mut LeafCombiner<'_, R>) -> Result<Value<R>, String> {
        match (self, other) {
            (Value::Tuple(a), Value::Tuple(b)) if a.len() == b.len() => Ok(Value::tuple(a.iter().zip(b.iter()).map(|(x, y)| x.zip_leaves(y, f)).collect::<Result<_, _>>()?)),
            (Value::Record(a), Value::Record(b)) if a.len() == b.len() => {
                let mut out = Vec::new();
                for (x, y) in a.iter().zip(b.iter()) {
                    if x.0 != y.0 {
                        return Err(format!("record fields differ: `{}` vs `{}`", x.0, y.0));
                    }
                    out.push((x.0.clone(), x.1.zip_leaves(&y.1, f)?));
                }
                Ok(Value::record(out))
            }
            (Value::Tuple(_), _) | (Value::Record(_), _) | (_, Value::Tuple(_)) | (_, Value::Record(_)) => Err(format!("structure mismatch: {} vs {}", self.kind(), other.kind())),
            (a, b) => f(a, b),
        }
    }
}
