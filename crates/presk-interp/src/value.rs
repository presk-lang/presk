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
    pub data: Data<T>,
}

/// A tensor's elements, shared by its clones (plan 0.10.1 step 2, D105).
///
/// The evaluator clones a value at every read, and an exact real at 128
/// bits is 64 bytes: until 0.10.1 a tensor owned its elements, so a field
/// read of a weight, a call's argument and a fold's slice of a stacked
/// parameter each copied them, and the seam to the other executors copied
/// every parameter once more. One block of the Llama 2 decoder at the 7B
/// width held its parameters twice in `presk run` and could not be solved
/// in 119 GB (D104). A `Data` holds the elements behind a reference count,
/// as the native host's buffers do since 0.10.0 step 6 (`presk_rt::Elems`):
/// a clone is a reference, and the elements are freed with the last one.
///
/// **A view.** A `Data` is a range of an allocation: the slice of a stacked
/// layer along its leading axis is the range of its layer
/// ([`Data::view`]), not a copy.
///
/// **Exact.** Nothing writes elements another value can see: a write
/// through `DerefMut` to elements another value shares, or to a range of a
/// larger allocation, copies them first, so a value never changes under a
/// reader.
pub struct Data<T> {
    buf: Arc<Vec<T>>,
    lo: usize,
    len: usize,
}

impl<T> Data<T> {
    /// Elements `lo..lo + len` of these, without a copy.
    pub fn view(&self, lo: usize, len: usize) -> Data<T> {
        assert!(lo + len <= self.len, "a view of {len} elements at {lo} of {}", self.len);
        Data { buf: Arc::clone(&self.buf), lo: self.lo + lo, len }
    }

    /// Whether these are every element of an allocation no other value
    /// shares, so that a write needs no copy and a drop frees them. A weak
    /// reference ([`Data::downgrade`]) counts as a sharer: what it may hand
    /// out again must not change.
    pub fn is_unique(&self) -> bool {
        self.lo == 0 && self.len == self.buf.len() && Arc::strong_count(&self.buf) == 1 && Arc::weak_count(&self.buf) == 0
    }

    /// A reference that does not keep the elements alive: the interpreter's
    /// shared zeros ([`crate::Interp`]'s `zeros_like`).
    pub fn downgrade(&self) -> WeakData<T> {
        WeakData { buf: Arc::downgrade(&self.buf), lo: self.lo, len: self.len }
    }

    /// Whether `self` and `other` are the same elements of the same
    /// allocation.
    pub fn same(&self, other: &Data<T>) -> bool {
        Arc::ptr_eq(&self.buf, &other.buf) && self.lo == other.lo && self.len == other.len
    }
}

/// A [`Data`] that does not keep its elements alive.
pub struct WeakData<T> {
    buf: std::sync::Weak<Vec<T>>,
    lo: usize,
    len: usize,
}

impl<T> WeakData<T> {
    /// The elements, if a value still holds them.
    pub fn upgrade(&self) -> Option<Data<T>> {
        Some(Data { buf: self.buf.upgrade()?, lo: self.lo, len: self.len })
    }
}

impl<T: Clone> Data<T> {
    /// The elements as a vector of their own: moved out when no other value
    /// shares them, copied otherwise.
    pub fn into_vec(self) -> Vec<T> {
        if self.is_unique() {
            return Arc::try_unwrap(self.buf).unwrap_or_else(|b| (*b).clone());
        }
        self.to_vec()
    }
}

impl<T> Clone for Data<T> {
    fn clone(&self) -> Self {
        Data { buf: Arc::clone(&self.buf), lo: self.lo, len: self.len }
    }
}

impl<T> Default for Data<T> {
    fn default() -> Self {
        Data::from(Vec::new())
    }
}

impl<T> std::ops::Deref for Data<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.buf[self.lo..self.lo + self.len]
    }
}

/// A write copies the elements first when another value shares them, or
/// when they are a range of a larger allocation.
impl<T: Clone> std::ops::DerefMut for Data<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        if !self.is_unique() {
            *self = Data::from(self.to_vec());
        }
        Arc::get_mut(&mut self.buf).expect("elements just made one's own")
    }
}

impl<T> From<Vec<T>> for Data<T> {
    fn from(v: Vec<T>) -> Self {
        let len = v.len();
        Data { buf: Arc::new(v), lo: 0, len }
    }
}

impl<T> FromIterator<T> for Data<T> {
    fn from_iter<I: IntoIterator<Item = T>>(it: I) -> Self {
        Data::from(it.into_iter().collect::<Vec<T>>())
    }
}

impl<'a, T> IntoIterator for &'a Data<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T: PartialEq> PartialEq for Data<T> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Data<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (**self).fmt(f)
    }
}

impl<T: Clone> Tensor<T> {
    pub fn scalar(x: T) -> Self {
        Tensor { axes: vec![], data: vec![x].into() }
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
