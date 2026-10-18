//! What a call needs in memory (plan 0.15.0 step 1, D148): the lifetimes of
//! a [`Recording`]'s buffers, the bytes live at each of its operations, and
//! their peak.
//!
//! - **A buffer's life** runs from its first write (a launch's output, a
//!   fill, a scalar, a copy, a `sample`) to its last read (a launch's input,
//!   a copy's source, a scalar read back, `sample`'s logits), both included,
//!   so that an operation's inputs and its output are live together. A view
//!   reads its buffer. The entry's inputs are held by its caller, live
//!   before the first operation and after the last; the result's buffers
//!   live to the end.
//! - **Bytes** are a buffer's elements times its stored type's
//!   ([`Store::bytes`]); integer weights take their integers (two `int4`s a
//!   byte) and their scales, an `f32` each. Nothing is rounded: the C
//!   arena's 64-byte alignment is the C program's.
//! - **The live bytes** at operation `k` are those of the buffers made by
//!   the call (every buffer but the inputs) live at `k`; the **peak** is
//!   their largest value, and the first operation at which it is reached.
//!   The inputs are counted apart.
//! - **The call path** of an operation is the calls it is inside, as
//!   [`crate::Launched::calls`] names them (D141): each call's `let`,
//!   `"<function>/%<let>"`, an iteration's with its index,
//!   `"<function>/%<let>[<i>]"`; the entry's body has none. A fold's slices
//!   and its scan's stack are made in its caller.
//!
//! It counts the buffers of the native host's launch sequence as the C
//! export records it, which is what any executor of that sequence holds if
//! it frees a buffer after its last read and keeps buffers whole; not a
//! process's resident size, nor the C program's arena, which holds a
//! function's buffers until it returns (D80) and is the C export's to
//! compute.

use crate::record::{Made, Node, Op, Recording, Store, Sym, leaves};

/// One buffer's life.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Life {
    /// Its first write and its last read, operations of the recording;
    /// `None` for a buffer nothing writes (an input).
    pub first: Option<usize>,
    pub last: usize,
    pub bytes: usize,
    pub input: bool,
    /// A leaf of the result (and not an input).
    pub output: bool,
}

/// The lifetimes of a recording.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lifetimes {
    pub lives: Vec<Life>,
    /// Bytes live at each operation, the inputs left out.
    pub live: Vec<usize>,
    pub peak: usize,
    /// The first operation at which the peak is reached (0 for a call that
    /// makes nothing).
    pub at: usize,
    /// The inputs' bytes, and the result's (not an input's).
    pub inputs: usize,
    pub outputs: usize,
    /// Each operation's call path.
    pub paths: Vec<Vec<String>>,
}

/// The bytes of buffer `b`.
pub fn bytes(rec: &Recording, b: usize) -> usize {
    let rb = &rec.bufs[b];
    match rb.store {
        Store::Quant { bits, k, .. } => {
            let ints = if bits == 4 { rb.len.div_ceil(2) } else { rb.len };
            ints + 4 * (rb.len / k.max(1))
        }
        s => rb.len * s.bytes(),
    }
}

/// What operation `op` reads and writes.
fn touched(op: &Op) -> (Vec<usize>, Option<usize>) {
    match op {
        Op::Launch { inputs, out, .. } => (inputs.iter().map(|v| v.buf).collect(), Some(*out)),
        Op::Fill { buf, .. } | Op::Scalar { buf, .. } => (Vec::new(), Some(*buf)),
        Op::Copy { dst, src, .. } => (vec![src.buf], Some(*dst)),
        Op::Read { src, .. } => (vec![src.buf], None),
        Op::Sample { out, src, .. } => (vec![src.buf], Some(*out)),
        Op::Hyper { .. } => (Vec::new(), None),
    }
}

impl Lifetimes {
    pub fn of(rec: &Recording) -> Lifetimes {
        let n = rec.ops.len();
        let mut lives: Vec<Life> = (0..rec.bufs.len())
            .map(|b| {
                let input = matches!(rec.bufs[b].made, Made::Input { .. });
                Life { first: None, last: 0, bytes: bytes(rec, b), input, output: false }
            })
            .collect();
        for (k, op) in rec.ops.iter().enumerate() {
            let (reads, write) = touched(op);
            for b in reads {
                lives[b].last = lives[b].last.max(k);
            }
            if let Some(b) = write {
                let l = &mut lives[b];
                l.first.get_or_insert(k);
                l.last = l.last.max(k);
            }
        }
        if let Some(res) = &rec.result {
            for leaf in leaves(res) {
                if let Sym::Tensor(v) = leaf
                    && !lives[v.buf].input
                {
                    lives[v.buf].output = true;
                    lives[v.buf].last = n.saturating_sub(1);
                }
            }
        }
        let mut delta = vec![0i64; n + 1];
        for l in &lives {
            if let (false, Some(f)) = (l.input, l.first) {
                delta[f] += l.bytes as i64;
                delta[l.last + 1] -= l.bytes as i64;
            }
        }
        let mut live = Vec::with_capacity(n);
        let (mut cur, mut peak, mut at) = (0i64, 0usize, 0usize);
        for (k, d) in delta.iter().take(n).enumerate() {
            cur += d;
            live.push(cur as usize);
            if cur as usize > peak {
                (peak, at) = (cur as usize, k);
            }
        }
        let inputs = lives.iter().filter(|l| l.input).map(|l| l.bytes).sum();
        let outputs = lives.iter().filter(|l| l.output).map(|l| l.bytes).sum();
        Lifetimes { lives, live, peak, at, inputs, outputs, paths: paths(rec) }
    }

    /// The buffers live at operation `k` that the call made.
    pub fn live_at(&self, k: usize) -> impl Iterator<Item = usize> + '_ {
        self.lives.iter().enumerate().filter(move |(_, l)| !l.input && l.first.is_some_and(|f| f <= k) && l.last >= k).map(|(b, _)| b)
    }

    /// Offsets for every buffer the call makes, two buffers live at once
    /// never overlapping: the largest first, each at the lowest offset, a
    /// multiple of `align`, where it fits beside those already placed whose
    /// lives meet its own (greedy by size). Returns the offsets (`None` for
    /// an input or a buffer nothing writes) and the size they take, at least
    /// the peak rounded.
    pub fn place(&self, align: usize) -> (Vec<Option<usize>>, usize) {
        let up = |x: usize| x.div_ceil(align) * align;
        let mut order: Vec<usize> = (0..self.lives.len()).filter(|b| !self.lives[*b].input && self.lives[*b].first.is_some()).collect();
        order.sort_by_key(|b| (std::cmp::Reverse(up(self.lives[*b].bytes.max(1))), self.lives[*b].first, *b));
        let mut at: Vec<Option<usize>> = vec![None; self.lives.len()];
        let mut placed: Vec<usize> = Vec::new();
        let mut size = 0;
        for b in order {
            let l = self.lives[b];
            let first = l.first.unwrap_or(0);
            // The regions of the placed buffers whose lives overlap this one's.
            let mut busy: Vec<(usize, usize)> = placed
                .iter()
                .filter(|p| {
                    let q = self.lives[**p];
                    q.first.unwrap_or(0) <= l.last && q.last >= first
                })
                .map(|p| (at[*p].unwrap_or(0), at[*p].unwrap_or(0) + up(self.lives[*p].bytes.max(1))))
                .collect();
            busy.sort();
            let need = up(l.bytes.max(1));
            let mut off = 0;
            for (s, e) in busy {
                if off + need <= s {
                    break;
                }
                off = off.max(e);
            }
            at[b] = Some(off);
            size = size.max(off + need);
            placed.push(b);
        }
        (at, size)
    }
}

/// Each call's path, from the recording's tree: the entry's is empty, a
/// call's is its caller's and the caller's `let` that made it, an
/// iteration's the fold's `let` with its index. `None` for a call the tree
/// does not reach.
pub fn call_paths(rec: &Recording) -> Vec<Option<Vec<String>>> {
    let mut out = vec![None; rec.calls.len()];
    if rec.calls.is_empty() {
        return out;
    }
    let mut stack = vec![(0usize, Vec::new())];
    while let Some((c, path)) = stack.pop() {
        let call = &rec.calls[c];
        for node in &call.body {
            match node {
                Node::Op(_) => {}
                Node::Call(j) => {
                    let mut p = path.clone();
                    p.push(format!("{}/%{}", call.func, rec.calls[*j].var.unwrap_or(0)));
                    stack.push((*j, p));
                }
                Node::Fold(f) => {
                    let fold = &rec.folds[*f];
                    for (i, j) in &fold.iterations {
                        let mut p = path.clone();
                        p.push(format!("{}/%{}[{i}]", call.func, fold.var.unwrap_or(0)));
                        stack.push((*j, p));
                    }
                }
            }
        }
        out[c] = Some(path);
    }
    out
}

/// Each operation's call path: its call's, a fold's slices and its scan's
/// stack its caller's.
fn paths(rec: &Recording) -> Vec<Vec<String>> {
    let mut out = vec![Vec::new(); rec.ops.len()];
    for (c, path) in call_paths(rec).into_iter().enumerate() {
        let Some(path) = path else { continue };
        for node in &rec.calls[c].body {
            match node {
                Node::Op(i) => out[*i] = path.clone(),
                Node::Fold(f) => {
                    let fold = &rec.folds[*f];
                    for op in fold.slices.iter().map(|(_, op)| op).chain(&fold.stack) {
                        out[*op] = path.clone();
                    }
                }
                Node::Call(_) => {}
            }
        }
    }
    out
}
