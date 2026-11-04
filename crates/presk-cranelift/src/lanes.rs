//! Lanes the native backend gives a scalar kernel itself (plan 0.17.0
//! step 2, D175): a loop whose iterations write different, contiguous
//! output elements and share nothing is run four (`f32`) or two (`f64`)
//! iterations at a time, one per lane. Each lane does its own iteration's
//! operations in their order, so the bits are the scalar kernel's (D42);
//! GCC does the same to the exported C at `-O2`, which is why a nest the
//! plan leaves at one lane is several times faster there (D174).
//!
//! The rewrite produces the vector form the IR already has (`VSet`,
//! `VStore`, `VLoad`, `VSplat`, `VBin`, `Vv`), one statement for each
//! statement, so the kernel's text and its lines are unchanged. A loop is
//! taken when, for its variable `v`:
//! - its bound is a multiple of the lane count, and it is not the loop a
//!   thread split divides or one enclosing it;
//! - every index that reads `v` reads it in its last component, as `v`
//!   plus terms without it, and in no other component: the lanes are then
//!   contiguous elements, loaded or stored at once;
//! - the output is stored at such an index, the same in every store, and
//!   read only there: each lane reads and writes its own element, which no
//!   other iteration touches (`sum_to`'s `%2[i3] = %2[i3] + %0[…, i3]`);
//! - each real variable the body assigns is assigned before it is read in
//!   every iteration, and read nowhere outside the loop: one per lane, it
//!   carries nothing from an iteration to the next;
//! - the arithmetic that depends on `v` is `+`, `−`, `×` and `÷`, and the
//!   body has no condition, integer variable or loop variable assigned.
//!
//! Everything that does not depend on `v` stays scalar and is broadcast.
//! The loop then runs over `bound / lanes`, its variable scaled by the lane
//! count wherever it is read.

use presk_kernel::ir::{BinOp, BufKind, Expr, Kernel, Stmt};
use presk_rt::Dtype;
use std::collections::{HashMap, HashSet};

/// The kernel with its lane-wise loops vectorized, or `None` when it has
/// none (or is not a scalar kernel of `f32` or `f64`). `split` is the depth
/// of the loop a thread split divides, which keeps its form.
pub(crate) fn natural(k: &Kernel, split: Option<usize>) -> Option<Kernel> {
    if k.lanes != 1 || k.nvec != 0 || !matches!(k.dtype, Dtype::F32 | Dtype::F64) || k.output.kind != BufKind::Real {
        return None;
    }
    let mut r = Rewrite { k, w: k.dtype.natural_lanes(), nvec: 0, done: 0 };
    if r.w < 2 {
        return None;
    }
    let body = r.stmts(&k.body, split, &k.body);
    if r.done == 0 {
        return None;
    }
    Some(Kernel { body, nvec: r.nvec, lanes: r.w, ..k.clone() })
}

struct Rewrite<'a> {
    k: &'a Kernel,
    w: usize,
    nvec: usize,
    done: usize,
}

impl Rewrite<'_> {
    /// `stmts` rewritten. `spine` counts down the single loops to the split
    /// one, which is kept with every loop above it; `all` is the kernel's
    /// body, to look for reads outside a loop.
    fn stmts(&mut self, stmts: &[Stmt], spine: Option<usize>, all: &[Stmt]) -> Vec<Stmt> {
        let on_spine = spine.is_some() && stmts.len() == 1;
        stmts
            .iter()
            .map(|s| match s {
                Stmt::For { var, bound, body } => {
                    let below = if on_spine { spine.and_then(|d| d.checked_sub(1)) } else { None };
                    if !on_spine && bound % self.w == 0 && *bound >= self.w {
                        if let Some(b) = self.lanes(*var, body, all) {
                            self.done += 1;
                            return Stmt::For { var: *var, bound: bound / self.w, body: b };
                        }
                    }
                    Stmt::For { var: *var, bound: *bound, body: self.stmts(body, below, all) }
                }
                Stmt::If { cond, then } => Stmt::If { cond: cond.clone(), then: self.stmts(then, None, all) },
                other => other.clone(),
            })
            .collect()
    }

    /// Loop `v`'s body over lanes, when it can be (the module's conditions).
    fn lanes(&mut self, v: usize, body: &[Stmt], all: &[Stmt]) -> Option<Vec<Stmt>> {
        // The real variables the body assigns, each assigned before it is
        // read, and read nowhere else in the kernel.
        let mut c = Check { assigned: HashSet::new(), ok: true };
        c.stmts(body);
        if !c.ok {
            return None;
        }
        let mut outside = HashSet::new();
        reads_outside(all, body, &mut outside);
        if c.assigned.iter().any(|s| outside.contains(s)) {
            return None;
        }
        // The one index the body stores the output at.
        let mut stores = Vec::new();
        out_stores(body, self.k.out_buf(), &mut stores);
        let store_ix = match stores.split_first() {
            Some((first, rest)) if rest.iter().all(|i| i == first) => Some((*first).clone()),
            Some(_) => return None,
            None => None,
        };
        let map: HashMap<usize, usize> = sorted(&c.assigned).into_iter().enumerate().map(|(i, s)| (s, self.nvec + i)).collect();
        let l = Lanes { v, w: self.w, k: self.k, map: &map, store_ix };
        let b = l.stmts(body)?;
        self.nvec += map.len();
        Some(b)
    }
}

/// The indices `body` stores buffer `out` at.
fn out_stores<'a>(body: &'a [Stmt], out: usize, acc: &mut Vec<&'a Vec<Expr>>) {
    for s in body {
        match s {
            Stmt::For { body, .. } | Stmt::If { then: body, .. } => out_stores(body, out, acc),
            Stmt::Store { buf, index, .. } | Stmt::VStore { buf, index, .. } | Stmt::StoreI { buf, index, .. } if *buf == out => acc.push(index),
            _ => {}
        }
    }
}

fn sorted(s: &HashSet<usize>) -> Vec<usize> {
    let mut v: Vec<usize> = s.iter().copied().collect();
    v.sort();
    v
}

/// The real variables a body assigns, and whether its statements are all
/// of the forms a lane can take (loops, real assignments, stores). Whether
/// each variable is assigned before it is read is [`Lanes`]'s to check, in
/// program order.
struct Check {
    assigned: HashSet<usize>,
    ok: bool,
}

impl Check {
    fn stmts(&mut self, body: &[Stmt]) {
        for s in body {
            match s {
                Stmt::For { body, .. } => self.stmts(body),
                Stmt::Set { var, .. } => {
                    self.assigned.insert(*var);
                }
                Stmt::Store { .. } => {}
                _ => self.ok = false,
            }
        }
    }
}

/// The real variables read in `stmts` outside `inner` (compared by address,
/// `inner` being a slice of the kernel's own body).
fn reads_outside(stmts: &[Stmt], inner: &[Stmt], out: &mut HashSet<usize>) {
    if std::ptr::eq(stmts, inner) {
        return;
    }
    fn expr(e: &Expr, out: &mut HashSet<usize>) {
        match e {
            Expr::Sv(s) => {
                out.insert(*s);
            }
            Expr::Load(_, ix) | Expr::LoadI(_, ix) | Expr::VLoad(_, ix) => ix.iter().for_each(|x| expr(x, out)),
            Expr::Bin(_, x, y) | Expr::Cmp(_, x, y) | Expr::VBin(_, x, y) => {
                expr(x, out);
                expr(y, out);
            }
            Expr::Un(_, x) | Expr::ToF(x) | Expr::ToI(x) | Expr::VSplat(x) => expr(x, out),
            Expr::Select(c, x, y) => {
                expr(c, out);
                expr(x, out);
                expr(y, out);
            }
            Expr::F(_) | Expr::I(_) | Expr::Iv(_) | Expr::Si(_) | Expr::Vv(_) => {}
        }
    }
    for s in stmts {
        match s {
            Stmt::For { body, .. } => reads_outside(body, inner, out),
            Stmt::If { cond, then } => {
                expr(cond, out);
                reads_outside(then, inner, out);
            }
            Stmt::Set { value, .. } | Stmt::SetI { value, .. } | Stmt::SetL { value, .. } | Stmt::VSet { value, .. } => expr(value, out),
            Stmt::Store { index, value, .. } | Stmt::StoreI { index, value, .. } | Stmt::VStore { index, value, .. } => {
                index.iter().for_each(|x| expr(x, out));
                expr(value, out);
            }
        }
    }
}

/// The rewrite of one loop's body over lanes: `map` gives each real
/// variable the body assigns its vector variable.
struct Lanes<'a> {
    v: usize,
    w: usize,
    k: &'a Kernel,
    map: &'a HashMap<usize, usize>,
    /// The index the body stores the output at, the only one it may read
    /// the output at.
    store_ix: Option<Vec<Expr>>,
}

impl Lanes<'_> {
    fn stmts(&self, body: &[Stmt]) -> Option<Vec<Stmt>> {
        let mut assigned = HashSet::new();
        self.stmts_in(body, &mut assigned)
    }

    fn stmts_in(&self, body: &[Stmt], assigned: &mut HashSet<usize>) -> Option<Vec<Stmt>> {
        let mut out = Vec::with_capacity(body.len());
        for s in body {
            out.push(match s {
                Stmt::For { var, bound, body } if *var != self.v => {
                    // Assigned before the nested loop: what its body reads
                    // before assigning is this iteration's value.
                    Stmt::For { var: *var, bound: *bound, body: self.stmts_in(body, assigned)? }
                }
                Stmt::Set { var, value } => {
                    let x = self.vec(value, assigned)?;
                    assigned.insert(*var);
                    Stmt::VSet { var: self.map[var], value: x }
                }
                Stmt::Store { buf, index, value } if *buf == self.k.out_buf() => {
                    if self.coeff_last(index)? != 1 {
                        return None;
                    }
                    let x = self.vec(value, assigned)?;
                    Stmt::VStore { buf: *buf, index: self.scaled_ix(index), value: x }
                }
                _ => return None,
            });
        }
        Some(out)
    }

    /// Whether `e` depends on the lane: reads `v`, or a variable the body
    /// assigns.
    fn dep(&self, e: &Expr) -> bool {
        match e {
            Expr::Iv(u) => *u == self.v,
            Expr::Sv(s) => self.map.contains_key(s),
            Expr::Load(_, ix) | Expr::LoadI(_, ix) => ix.iter().any(|x| self.dep(x)),
            Expr::Bin(_, x, y) | Expr::Cmp(_, x, y) => self.dep(x) || self.dep(y),
            Expr::Un(_, x) | Expr::ToF(x) | Expr::ToI(x) => self.dep(x),
            Expr::Select(c, x, y) => self.dep(c) || self.dep(x) || self.dep(y),
            Expr::F(_) | Expr::I(_) | Expr::Si(_) => false,
            Expr::Vv(_) | Expr::VLoad(..) | Expr::VSplat(_) | Expr::VBin(..) => true,
        }
    }

    /// The real expression `e` as a vector: the lane-wise form of what
    /// depends on the lane, a broadcast of what does not.
    fn vec(&self, e: &Expr, assigned: &HashSet<usize>) -> Option<Expr> {
        // The output read anywhere but at the element this iteration
        // stores: an element another iteration writes, in an order the
        // lanes do not keep.
        if self.reads_out_elsewhere(e) {
            return None;
        }
        if !self.dep(e) {
            return Some(Expr::VSplat(Box::new(self.scaled(e))));
        }
        match e {
            Expr::Sv(s) => {
                // A variable of the body read before this iteration assigns
                // it carries a value from the previous one.
                if !assigned.contains(s) {
                    return None;
                }
                Some(Expr::Vv(self.map[s]))
            }
            Expr::Load(buf, ix) if *buf == self.k.out_buf() => Some(Expr::VLoad(*buf, self.scaled_ix(ix))),
            Expr::Load(buf, ix) => {
                let real = *buf < self.k.inputs.len() && self.k.inputs[*buf].kind == BufKind::Real;
                if !real || self.coeff_last(ix)? != 1 {
                    return None;
                }
                Some(Expr::VLoad(*buf, self.scaled_ix(ix)))
            }
            Expr::Bin(op @ (BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div), x, y) => {
                Some(Expr::VBin(*op, Box::new(self.vec(x, assigned)?), Box::new(self.vec(y, assigned)?)))
            }
            _ => None,
        }
    }

    /// Whether `e` reads the output at another index than the body's store.
    fn reads_out_elsewhere(&self, e: &Expr) -> bool {
        match e {
            Expr::Load(b, ix) if *b == self.k.out_buf() => self.store_ix.as_ref() != Some(ix) || ix.iter().any(|x| self.reads_out_elsewhere(x)),
            Expr::Load(_, ix) | Expr::LoadI(_, ix) | Expr::VLoad(_, ix) => ix.iter().any(|x| self.reads_out_elsewhere(x)),
            Expr::Bin(_, x, y) | Expr::Cmp(_, x, y) | Expr::VBin(_, x, y) => self.reads_out_elsewhere(x) || self.reads_out_elsewhere(y),
            Expr::Un(_, x) | Expr::ToF(x) | Expr::ToI(x) | Expr::VSplat(x) => self.reads_out_elsewhere(x),
            Expr::Select(c, x, y) => self.reads_out_elsewhere(c) || self.reads_out_elsewhere(x) || self.reads_out_elsewhere(y),
            Expr::F(_) | Expr::I(_) | Expr::Iv(_) | Expr::Sv(_) | Expr::Si(_) | Expr::Vv(_) => false,
        }
    }

    /// The coefficient of `v` in the index's last component when no other
    /// component reads it and the last is affine in it.
    fn coeff_last(&self, ix: &[Expr]) -> Option<i64> {
        let (last, rest) = ix.split_last()?;
        if rest.iter().any(|x| self.reads_v(x)) {
            return None;
        }
        self.coeff(last)
    }

    fn reads_v(&self, e: &Expr) -> bool {
        match e {
            Expr::Iv(u) => *u == self.v,
            Expr::Bin(_, x, y) => self.reads_v(x) || self.reads_v(y),
            Expr::LoadI(_, ix) => ix.iter().any(|x| self.reads_v(x)),
            Expr::Un(_, x) | Expr::ToI(x) | Expr::ToF(x) => self.reads_v(x),
            _ => false,
        }
    }

    /// `v`'s coefficient in an integer expression affine in it.
    fn coeff(&self, e: &Expr) -> Option<i64> {
        if !self.reads_v(e) {
            return Some(0);
        }
        match e {
            Expr::Iv(_) => Some(1),
            Expr::Bin(BinOp::IAdd, x, y) => Some(self.coeff(x)? + self.coeff(y)?),
            Expr::Bin(BinOp::IMul, x, y) => match (self.reads_v(x), self.reads_v(y)) {
                (true, false) => Some(self.coeff(x)? * constant(y)?),
                (false, true) => Some(constant(x)? * self.coeff(y)?),
                _ => None,
            },
            _ => None,
        }
    }

    /// `e` with `v` read as `v · w`, the first lane's iteration.
    fn scaled(&self, e: &Expr) -> Expr {
        match e {
            Expr::Iv(u) if *u == self.v => Expr::Bin(BinOp::IMul, Box::new(Expr::Iv(*u)), Box::new(Expr::I(self.w as i64))),
            Expr::Load(b, ix) => Expr::Load(*b, self.scaled_ix(ix)),
            Expr::LoadI(b, ix) => Expr::LoadI(*b, self.scaled_ix(ix)),
            Expr::Bin(op, x, y) => Expr::Bin(*op, Box::new(self.scaled(x)), Box::new(self.scaled(y))),
            Expr::Cmp(op, x, y) => Expr::Cmp(*op, Box::new(self.scaled(x)), Box::new(self.scaled(y))),
            Expr::Un(op, x) => Expr::Un(*op, Box::new(self.scaled(x))),
            Expr::ToF(x) => Expr::ToF(Box::new(self.scaled(x))),
            Expr::ToI(x) => Expr::ToI(Box::new(self.scaled(x))),
            Expr::Select(c, x, y) => Expr::Select(Box::new(self.scaled(c)), Box::new(self.scaled(x)), Box::new(self.scaled(y))),
            other => other.clone(),
        }
    }

    fn scaled_ix(&self, ix: &[Expr]) -> Vec<Expr> {
        ix.iter().map(|x| self.scaled(x)).collect()
    }
}

/// The value of an integer expression made of constants only.
fn constant(e: &Expr) -> Option<i64> {
    match e {
        Expr::I(c) => Some(*c),
        Expr::Bin(BinOp::IAdd, x, y) => Some(constant(x)?.checked_add(constant(y)?)?),
        Expr::Bin(BinOp::IMul, x, y) => Some(constant(x)?.checked_mul(constant(y)?)?),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::natural;
    use presk_kernel::ir::{BinOp, BufKind, BufSpec, Dtype, Expr, Kernel, Stmt, Threads};

    fn spec(dims: &[usize]) -> BufSpec {
        BufSpec { dims: dims.to_vec(), kind: BufKind::Real, names: vec![None; dims.len()] }
    }

    fn kernel(d: Dtype, inputs: &[&[usize]], output: &[usize], nloops: usize, nreal: usize, body: Vec<Stmt>) -> Kernel {
        Kernel { name: "k".into(), dtype: d, inputs: inputs.iter().map(|s| spec(s)).collect(), output: spec(output), nloops, nreal, nint: 0, nvec: 0, lanes: 1, body, threads: Threads::Default }
    }

    fn add(x: Expr, y: Expr) -> Expr {
        Expr::Bin(BinOp::Add, Box::new(x), Box::new(y))
    }

    /// `out[i, j] = x[i, j] + 1`.
    fn elementwise() -> Kernel {
        let body = vec![Stmt::For {
            var: 0,
            bound: 2,
            body: vec![Stmt::For { var: 1, bound: 8, body: vec![Stmt::Store { buf: 1, index: vec![Expr::Iv(0), Expr::Iv(1)], value: add(Expr::Load(0, vec![Expr::Iv(0), Expr::Iv(1)]), Expr::F(1.0)) }] }],
        }];
        kernel(Dtype::F32, &[&[2, 8]], &[2, 8], 2, 0, body)
    }

    #[test]
    fn an_elementwise_loop_runs_over_four_lanes() {
        let k = natural(&elementwise(), None).expect("lane-wise");
        assert_eq!(k.lanes, 4);
        let Stmt::For { body, .. } = &k.body[0] else { panic!() };
        let Stmt::For { bound, body, .. } = &body[0] else { panic!() };
        assert_eq!(*bound, 2);
        assert!(matches!(&body[0], Stmt::VStore { .. }), "{:?}", body[0]);
        // Two lanes in f64.
        let mut k = elementwise();
        k.dtype = Dtype::F64;
        assert_eq!(natural(&k, None).expect("lane-wise").lanes, 2);
    }

    #[test]
    fn the_split_loop_keeps_its_form() {
        // The inner loop is the split one (depth 1): nothing else is lane-wise.
        assert!(natural(&elementwise(), Some(1)).is_none());
        // The outer one is split: the inner one still takes the lanes.
        assert!(natural(&elementwise(), Some(0)).is_some());
    }

    #[test]
    fn a_sum_along_the_lanes_axis_is_not_lane_wise() {
        // s = Σ_j x[i, j]; out[i] = s: `j` carries the sum, and `i` is not
        // the last index of `x`.
        let body = vec![Stmt::For {
            var: 0,
            bound: 4,
            body: vec![
                Stmt::Set { var: 0, value: Expr::F(0.0) },
                Stmt::For { var: 1, bound: 8, body: vec![Stmt::Set { var: 0, value: add(Expr::Sv(0), Expr::Load(0, vec![Expr::Iv(0), Expr::Iv(1)])) }] },
                Stmt::Store { buf: 1, index: vec![Expr::Iv(0)], value: Expr::Sv(0) },
            ],
        }];
        assert!(natural(&kernel(Dtype::F32, &[&[4, 8]], &[4], 2, 1, body), None).is_none());
    }

    #[test]
    fn a_transposed_read_and_a_bound_off_the_lanes_are_not_lane_wise() {
        // out[i, j] = x[j, i]: the lanes would be a gather.
        let body = vec![Stmt::For {
            var: 0,
            bound: 8,
            body: vec![Stmt::For { var: 1, bound: 8, body: vec![Stmt::Store { buf: 1, index: vec![Expr::Iv(0), Expr::Iv(1)], value: Expr::Load(0, vec![Expr::Iv(1), Expr::Iv(0)]) }] }],
        }];
        assert!(natural(&kernel(Dtype::F32, &[&[8, 8]], &[8, 8], 2, 0, body), None).is_none());
        // A bound of 6 is not a multiple of four.
        let body = vec![Stmt::For { var: 0, bound: 6, body: vec![Stmt::Store { buf: 1, index: vec![Expr::Iv(0)], value: Expr::Load(0, vec![Expr::Iv(0)]) }] }];
        assert!(natural(&kernel(Dtype::F32, &[&[6]], &[6], 1, 0, body), None).is_none());
    }

    /// `out[j] = 0` for every j, then `out[j] = out[j] + x[i, j]` over `i`
    /// and `j`, `sum_to`'s form; `read` is the index the sum reads the
    /// output at.
    fn sum_to(read: Expr) -> Kernel {
        let body = vec![
            Stmt::For { var: 0, bound: 8, body: vec![Stmt::Store { buf: 1, index: vec![Expr::Iv(0)], value: Expr::F(0.0) }] },
            Stmt::For {
                var: 1,
                bound: 3,
                body: vec![Stmt::For { var: 2, bound: 8, body: vec![Stmt::Store { buf: 1, index: vec![Expr::Iv(2)], value: add(Expr::Load(1, vec![read]), Expr::Load(0, vec![Expr::Iv(1), Expr::Iv(2)])) }] }],
            },
        ];
        kernel(Dtype::F32, &[&[3, 8]], &[8], 3, 0, body)
    }

    #[test]
    fn the_output_is_read_over_lanes_only_where_they_store_it() {
        let k = natural(&sum_to(Expr::Iv(2)), None).expect("lane-wise");
        let Stmt::For { body, .. } = &k.body[1] else { panic!() };
        let Stmt::For { bound, .. } = &body[0] else { panic!() };
        assert_eq!(*bound, 2);
        // `out[0]`, which the iteration j = 0 writes, and `out[j + 1]`,
        // which the next one writes: the inner loop keeps its form (the
        // zeroing loop still takes the lanes).
        for read in [Expr::I(0), add_i(Expr::Iv(2), Expr::I(1))] {
            let k = natural(&sum_to(read.clone()), None).expect("the zeroing loop");
            let Stmt::For { body, .. } = &k.body[1] else { panic!() };
            let Stmt::For { bound, .. } = &body[0] else { panic!() };
            assert_eq!(*bound, 8, "{read:?}");
        }
    }

    fn add_i(x: Expr, y: Expr) -> Expr {
        Expr::Bin(BinOp::IAdd, Box::new(x), Box::new(y))
    }

    #[test]
    fn a_value_carried_from_the_previous_lane_is_not_lane_wise() {
        // out[j] = s; s = x[j]: each iteration stores the previous one's value.
        let body = vec![Stmt::For {
            var: 0,
            bound: 8,
            body: vec![Stmt::Store { buf: 1, index: vec![Expr::Iv(0)], value: Expr::Sv(0) }, Stmt::Set { var: 0, value: Expr::Load(0, vec![Expr::Iv(0)]) }],
        }];
        assert!(natural(&kernel(Dtype::F32, &[&[8]], &[8], 1, 1, body), None).is_none());
    }
}
