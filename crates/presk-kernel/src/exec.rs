//! The kernel-IR interpreter: executes a [`Kernel`] on runtime buffers in
//! the kernel's machine type, every operation rounded to that type — the
//! first native-arithmetic realization, and the yardstick of the code
//! generator to come.
//!
//! It also emulates a kernel in any format ([`run_kernel_in`], plan 0.18.0
//! step 2, D205): a float format's operations computed in a native type that
//! holds twice its significand and two bits more, then rounded once to it;
//! an integer or a fixed-point format's in native integers, wrapped or
//! clamped; every load, store and constant converted, rounded once, ties to
//! even. An emulated run counts what the formats' overflow rules produce
//! ([`Counts`]).
//!
//! Since step 3 (D206) a kernel carries the formats its operations are
//! rounded to ([`crate::ir::Arith`]): its arithmetic format, its product
//! format for a sum's products of terms and its accumulation format for a
//! sum's additions. [`run_emulated`] runs any kernel so: in one format as
//! step 2 does where its formats are one and hold its buffers' values, and
//! otherwise every value held exactly as an `f64` ([`Mixed`]), each
//! operation's exact result rounded once to its own format.
//!
//! Since step 4 (D207) a kernel may store its result in an affine format,
//! a quantized activation's codes, rounded into it by its rescaling; and
//! where its product format is an integer or a fixed-point one, it reads a
//! value of a scaled format, an affine one or an integer weight, as its
//! integer counted in its scale, so that a sum of their products is the
//! integer sum a machine's integer unit computes ([`Mixed`]'s counts).

use std::cell::Cell;

pub use presk_format::Counts;
use presk_format::{Fixed, Float, Format, Int, Tie, BF16, F32, F64};
use presk_rt::Buf;

use crate::ir::{BinOp, BufKind, CmpOp, Expr, Kernel, Stmt, UnOp};

/// The counts of one run, written through a shared reference.
#[derive(Default)]
pub struct Counter(Cell<Counts>);

impl Counter {
    fn bump(&self, f: impl FnOnce(&mut Counts)) {
        let mut c = self.0.get();
        f(&mut c);
        self.0.set(c);
    }

    /// A result `y` of a format `f`, from operands among which an infinity
    /// (`inf_in`) or a NaN (`nan_in`), `o` whether the overflow rule gave it.
    fn note(&self, f: Format, y: f64, o: bool, inf_in: bool, nan_in: bool) {
        self.bump(|c| c.note(f, y, o, inf_in, nan_in));
    }

    pub fn get(&self) -> Counts {
        self.0.get()
    }
}

/// A machine real: `f64` or `f32`, every operation in that type, or a value
/// of an emulated format. `Ctx` is what an emulated format's operations
/// need, its description and the run's [`Counter`]; nothing for a native
/// type.
pub trait Elem: Copy + PartialOrd + std::fmt::Debug {
    type Ctx;
    /// Whether this is `f64` or `f32` computing natively, whose stores keep
    /// the conversions of every version before 0.18.0.
    const NATIVE: bool;
    /// A real held as an `f64` converted to the arithmetic: a load.
    fn from_f64(x: f64, c: &Self::Ctx) -> Self;
    /// A load of a scaled format's element, whose value `value` is the
    /// integer `n` times the scale `s`: the value, but where the kernel
    /// counts its sums in their terms' unit ([`Mixed`]).
    fn from_scaled(n: f64, s: f64, value: f64, c: &Self::Ctx) -> Self {
        let _ = (n, s);
        Self::from_f64(value, c)
    }
    /// A constant of the kernel, or an index read as a real, converted to
    /// the arithmetic: a load's conversion but where loads are exact
    /// ([`Mixed`]).
    fn constant(x: f64, c: &Self::Ctx) -> Self {
        Self::from_f64(x, c)
    }
    /// The value, rounded to `f64` only for an integer or a fixed-point
    /// number of more than 53 bits.
    fn to_f64(self, c: &Self::Ctx) -> f64;
    fn un(self, op: UnOp, c: &Self::Ctx) -> Self;
    fn bin(self, op: BinOp, other: Self, c: &Self::Ctx) -> Self;
    /// The value converted to the format of the buffer it is stored in,
    /// rounded once, ties to even, as an `f64`.
    fn to_format(self, f: Format, c: &Self::Ctx) -> f64;
    /// Whether storing the value in `f`, binary16 or `e8m15`, turns a
    /// finite value into an infinity: the overflow guard's test (plan
    /// 0.18.2 step 4, D218), on the value as it is, with no side effect. A
    /// realization the native types run computes such a store in `f32`;
    /// an integer, a fixed-point number or a sum counted in its terms' unit
    /// is never one, and none is tested.
    fn overflows(self, f: Float, c: &Self::Ctx) -> bool {
        let _ = (f, c);
        false
    }
}

impl Elem for f64 {
    type Ctx = ();
    const NATIVE: bool = true;
    fn from_f64(x: f64, _: &()) -> Self {
        x
    }
    fn to_f64(self, _: &()) -> f64 {
        self
    }
    fn overflows(self, f: Float, _: &()) -> bool {
        presk_rt::codes::overflows(f, self)
    }
    fn un(self, op: UnOp, _: &()) -> Self {
        match op {
            UnOp::Neg => -self,
            // The `libm` crate's functions, musl's, which the native backend
            // binds and the C export ships (D76, D78), so that the kernel
            // interpreter computes their bits (0.16.0, D159; until then the
            // standard library's, the system's, differed in the last bits).
            UnOp::Exp => libm::exp(self),
            UnOp::Log => libm::log(self),
            UnOp::Tanh => libm::tanh(self),
            UnOp::Sqrt => self.sqrt(),
            UnOp::Sin => libm::sin(self),
            UnOp::Cos => libm::cos(self),
            UnOp::Erf => libm::erf(self),
            UnOp::Abs => self.abs(),
        }
    }
    fn bin(self, op: BinOp, o: Self, _: &()) -> Self {
        match op.plain() {
            BinOp::Add | BinOp::IAdd => self + o,
            BinOp::Sub => self - o,
            BinOp::Mul | BinOp::IMul => self * o,
            BinOp::Div => self / o,
            BinOp::Accum | BinOp::Product => unreachable!("plain"),
        }
    }
    fn to_format(self, f: Format, _: &()) -> f64 {
        f.round_noted(self, Tie::Even).0
    }
}

impl Elem for f32 {
    type Ctx = ();
    const NATIVE: bool = true;
    fn from_f64(x: f64, _: &()) -> Self {
        x as f32
    }
    fn to_f64(self, _: &()) -> f64 {
        self as f64
    }
    fn overflows(self, f: Float, _: &()) -> bool {
        presk_rt::codes::overflows(f, f64::from(self))
    }
    fn un(self, op: UnOp, _: &()) -> Self {
        match op {
            UnOp::Neg => -self,
            UnOp::Exp => libm::expf(self),
            UnOp::Log => libm::logf(self),
            UnOp::Tanh => libm::tanhf(self),
            UnOp::Sqrt => self.sqrt(),
            UnOp::Sin => libm::sinf(self),
            UnOp::Cos => libm::cosf(self),
            UnOp::Erf => libm::erff(self),
            UnOp::Abs => self.abs(),
        }
    }
    fn bin(self, op: BinOp, o: Self, _: &()) -> Self {
        match op.plain() {
            BinOp::Add | BinOp::IAdd => self + o,
            BinOp::Sub => self - o,
            BinOp::Mul | BinOp::IMul => self * o,
            BinOp::Div => self / o,
            BinOp::Accum | BinOp::Product => unreachable!("plain"),
        }
    }
    fn to_format(self, f: Format, _: &()) -> f64 {
        f.round_noted(self as f64, Tie::Even).0
    }
}

/// The NaN an operation gives where the format it is rounded to has none,
/// which saturates it with its sign: the first operand's when it is a NaN,
/// else the second's, else +NaN (0/0, ∞ − ∞, √−1). IEEE 754 leaves the
/// sign of a NaN an operation gives to the machine, and the compiler's own
/// code may change it; a format without NaNs makes it a result, so it is
/// fixed here (plan 0.18.0 step 3, D206). Every other value as it is.
fn nan_of(y: f64, a: f64, b: f64, f: Format) -> f64 {
    if !y.is_nan() || matches!(f, Format::Float(g) if g.has_nan()) {
        y
    } else if a.is_nan() {
        a
    } else if b.is_nan() {
        b
    } else {
        f64::NAN
    }
}

/// A value of a float format computed in the native type `W`, which holds
/// every number of the format and at least 2p + 2 bits of significand: each
/// operation in `W`, then rounded once to the format. For +, −, ×, ÷ and √
/// that is the operation correctly rounded in the format (the double
/// rounding is innocuous: S. A. Figueroa, "When is double rounding
/// innocuous?", 1995); a function (`exp`, `tanh`, …) is `W`'s `libm`
/// function rounded once, an approximation of the kind question 84
/// declares.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Soft<W>(W);

/// What a [`Soft`] computes in: the format, whether it is `W` itself (`f32`
/// or `f64` emulated, every operation native and only counted), the counts.
pub struct SoftCtx {
    f: Float,
    native: bool,
    counts: Counter,
}

impl SoftCtx {
    fn fit<W: Elem<Ctx = ()>>(&self, x: f64, inf_in: bool, nan_in: bool) -> Soft<W> {
        let (y, o) = if self.native { (W::from_f64(x, &()).to_f64(&()), false) } else { self.f.round_noted(x, Tie::Even) };
        self.counts.note(Format::Float(self.f), y, o, inf_in, nan_in);
        Soft(W::from_f64(y, &()))
    }
}

impl<W: Elem<Ctx = ()>> Elem for Soft<W> {
    type Ctx = SoftCtx;
    const NATIVE: bool = false;
    fn from_f64(x: f64, c: &SoftCtx) -> Self {
        c.fit(x, x.is_infinite(), x.is_nan())
    }
    fn to_f64(self, _: &SoftCtx) -> f64 {
        self.0.to_f64(&())
    }
    fn overflows(self, f: Float, _: &SoftCtx) -> bool {
        presk_rt::codes::overflows(f, self.0.to_f64(&()))
    }
    fn un(self, op: UnOp, c: &SoftCtx) -> Self {
        let a = self.0.to_f64(&());
        c.fit(nan_of(self.0.un(op, &()).to_f64(&()), a, a, Format::Float(c.f)), a.is_infinite(), a.is_nan())
    }
    fn bin(self, op: BinOp, o: Self, c: &SoftCtx) -> Self {
        let (a, b) = (self.0.to_f64(&()), o.0.to_f64(&()));
        c.fit(nan_of(self.0.bin(op, o.0, &()).to_f64(&()), a, b, Format::Float(c.f)), a.is_infinite() || b.is_infinite(), a.is_nan() || b.is_nan())
    }
    fn to_format(self, f: Format, c: &SoftCtx) -> f64 {
        let x = self.0.to_f64(&());
        let (y, o) = f.round_noted(x, Tie::Even);
        c.counts.note(f, y, o, x.is_infinite(), x.is_nan());
        y
    }
}

/// A value of an integer format of at most 64 bits, computed exactly in
/// 128 bits and then fitted, wrapped or clamped.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct IntE(i64);

pub struct IntCtx {
    int: Int,
    counts: Counter,
}

impl IntCtx {
    fn fit(&self, n: i128) -> IntE {
        let y = self.int.fit(n);
        self.counts.note(Format::Int(self.int), 0.0, y != n, false, false);
        IntE(y as i64)
    }
}

impl Elem for IntE {
    type Ctx = IntCtx;
    const NATIVE: bool = false;
    fn from_f64(x: f64, c: &IntCtx) -> Self {
        let (n, o) = c.int.round_noted(x, Tie::Even);
        c.counts.note(Format::Int(c.int), 0.0, o, x.is_infinite(), x.is_nan());
        IntE(n as i64)
    }
    fn to_f64(self, _: &IntCtx) -> f64 {
        self.0 as f64
    }
    fn un(self, op: UnOp, c: &IntCtx) -> Self {
        match op {
            UnOp::Neg => c.fit(-(self.0 as i128)),
            UnOp::Abs => c.fit((self.0 as i128).abs()),
            _ => unreachable!("an integer kernel computes `{op:?}` (refused by `exact_ops`)"),
        }
    }
    fn bin(self, op: BinOp, o: Self, c: &IntCtx) -> Self {
        let (a, b) = (self.0 as i128, o.0 as i128);
        match op.plain() {
            BinOp::Add | BinOp::IAdd => c.fit(a + b),
            BinOp::Sub => c.fit(a - b),
            BinOp::Mul | BinOp::IMul => c.fit(a * b),
            BinOp::Div => unreachable!("an integer kernel divides (refused by `exact_ops`)"),
            BinOp::Accum | BinOp::Product => unreachable!("plain"),
        }
    }
    fn to_format(self, f: Format, c: &IntCtx) -> f64 {
        let (y, o) = f.round_scaled(self.0 as i128, 0, Tie::Even);
        c.counts.note(f, y, o, false, false);
        y
    }
}

/// A value of a fixed-point format, as its units `m` (`m·2^-frac`),
/// computed exactly in 128 bits and then fitted: a sum of products on the
/// format's grid is exact while it stays in the range.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct FixE(i128);

pub struct FixCtx {
    fixed: Fixed,
    counts: Counter,
}

impl FixCtx {
    fn fit(&self, n: i128) -> FixE {
        let y = self.fixed.int.fit(n);
        self.counts.note(Format::Fixed(self.fixed), 0.0, y != n, false, false);
        FixE(y)
    }
}

impl Elem for FixE {
    type Ctx = FixCtx;
    const NATIVE: bool = false;
    fn from_f64(x: f64, c: &FixCtx) -> Self {
        let (m, o) = c.fixed.round_noted(x, Tie::Even);
        c.counts.note(Format::Fixed(c.fixed), 0.0, o, x.is_infinite(), x.is_nan());
        FixE(m)
    }
    fn to_f64(self, c: &FixCtx) -> f64 {
        c.fixed.value(self.0)
    }
    fn un(self, op: UnOp, c: &FixCtx) -> Self {
        match op {
            UnOp::Neg => c.fit(-self.0),
            UnOp::Abs => c.fit(self.0.abs()),
            _ => unreachable!("a fixed-point kernel computes `{op:?}` (refused by `exact_ops`)"),
        }
    }
    fn bin(self, op: BinOp, o: Self, c: &FixCtx) -> Self {
        match op.plain() {
            BinOp::Add | BinOp::IAdd => c.fit(self.0 + o.0),
            BinOp::Sub => c.fit(self.0 - o.0),
            // Units of 2^-2frac, rounded to the grid once.
            BinOp::Mul | BinOp::IMul => {
                let (m, over) = c.fixed.round_scaled(self.0 * o.0, -2 * c.fixed.frac, Tie::Even);
                c.counts.note(Format::Fixed(c.fixed), 0.0, over, false, false);
                FixE(m)
            }
            BinOp::Div => unreachable!("a fixed-point kernel divides (refused by `exact_ops`)"),
            BinOp::Accum | BinOp::Product => unreachable!("plain"),
        }
    }
    fn to_format(self, f: Format, c: &FixCtx) -> f64 {
        let (y, o) = f.round_scaled(self.0, -c.fixed.frac, Tie::Even);
        c.counts.note(f, y, o, false, false);
        y
    }
}

/// A value of a kernel whose operations round to several formats (plan
/// 0.18.0 step 3, D206), held exactly as an `f64`: every format it meets
/// holds its values in `f64` ([`Format::exact_in_f64`]). A load is exact;
/// each operation's exact result, its `f64` value and the error an
/// error-free transformation finds, is rounded once to the operation's own
/// format ([`Format::round_exact`]): a sum's addition to
/// the accumulation format, a product of terms to the product format, any
/// other operation and every constant to the arithmetic format. A function
/// (`exp`, `tanh`, …) is the `libm` function of the native type the
/// arithmetic format is emulated in ([`emulated_in`]), rounded once.
///
/// Where the product format is an integer or a fixed-point one (step 4,
/// D207), a value loaded from a scaled format, an affine one or an integer
/// weight, is held as its integer counted in its scale, a *count*: `x`
/// units of `base·2^-frac`. A product of two counts is the product of their
/// integers, counted in the product of their scales and rounded in the
/// product format on that unit's grid; a sum's addition of two counts of
/// one unit, or of a count and zero, is rounded in the accumulation format
/// on its grid; a count is stored rounded once from its exact value. Any
/// other operation reads a count's value, which a load's is exactly in an
/// `f64` and a product's or a sum's need not be: the run is then refused.
#[derive(Clone, Copy, Debug)]
pub struct Mixed {
    x: f64,
    unit: Option<Unit>,
}

/// A count's unit, `base·2^-frac`: `base` a binary32 scale or the product
/// of two, exact in an `f64`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Unit {
    base: f64,
    frac: i32,
}

impl Mixed {
    /// The value, rounded to `f64` for a count no `f64` holds: what a
    /// comparison reads.
    fn approx(self) -> f64 {
        match self.unit {
            None => self.x,
            Some(u) => {
                let (m, e) = count_scaled(self.x, u);
                F64.round_scaled(m, e, Tie::Even).0
            }
        }
    }
}

impl PartialEq for Mixed {
    fn eq(&self, o: &Mixed) -> bool {
        self.approx() == o.approx()
    }
}

impl PartialOrd for Mixed {
    fn partial_cmp(&self, o: &Mixed) -> Option<std::cmp::Ordering> {
        self.approx().partial_cmp(&o.approx())
    }
}

/// `|x| = m·2^e`, `m < 2^53`, for a finite non-zero `x`.
fn decompose(x: f64) -> (u64, i32) {
    let bits = x.abs().to_bits();
    let (be, frac) = ((bits >> 52) as i32, bits & ((1u64 << 52) - 1));
    if be == 0 { (frac, -1074) } else { (frac | (1u64 << 52), be - 1075) }
}

/// A count's value as `m·2^e`, exactly.
fn count_scaled(x: f64, u: Unit) -> (i128, i32) {
    if x == 0.0 || u.base == 0.0 {
        return (0, 0);
    }
    let (mb, eb) = decompose(u.base);
    (x as i128 * mb as i128, eb - u.frac)
}

/// Whether `m·2^e` is an `f64`.
fn in_f64(m: i128, e: i32) -> bool {
    if m == 0 {
        return true;
    }
    let t = m.trailing_zeros();
    let (m, e) = (m.unsigned_abs() >> t, e + t as i32);
    let bits = 128 - m.leading_zeros() as i32;
    bits <= 53 && e >= -1074 && e + bits <= 1024
}

pub struct MixedCtx {
    arith: Format,
    product: Format,
    accum: Format,
    /// Where the arithmetic format's functions are computed: `f32`, else
    /// `f64`.
    w: Float,
    /// Whether the product format counts: an integer or a fixed-point one.
    units: bool,
    /// Whether an operation read a count no `f64` holds.
    inexact: Cell<bool>,
    counts: Counter,
}

impl MixedCtx {
    fn fit(&self, f: Format, x: f64, rest: f64, inf_in: bool, nan_in: bool) -> Mixed {
        let (y, o) = f.round_exact(x, rest, Tie::Even);
        self.counts.note(f, y, o, inf_in, nan_in);
        Mixed { x: y, unit: None }
    }

    /// The value an operation that does not count reads.
    fn real(&self, v: Mixed) -> f64 {
        match v.unit {
            None => v.x,
            Some(u) => {
                let (m, e) = count_scaled(v.x, u);
                if !in_f64(m, e) {
                    self.inexact.set(true);
                }
                F64.round_scaled(m, e, Tie::Even).0
            }
        }
    }

    /// `op` on counts where the formats count it, `None` elsewhere.
    fn counted(&self, op: BinOp, a: Mixed, b: Mixed) -> Option<Mixed> {
        let f = match op {
            BinOp::Product => self.product,
            BinOp::Accum => self.accum,
            _ => return None,
        };
        let (int, frac) = match f {
            Format::Int(i) => (i, 0),
            Format::Fixed(x) => (x.int, x.frac),
            _ => return None,
        };
        // The exact result as `m` units of `base·2^-at`.
        let (m, base, at) = match (op, a.unit, b.unit) {
            (BinOp::Product, Some(ua), Some(ub)) => {
                let base = ua.base * ub.base;
                if ua.base.mul_add(ub.base, -base) != 0.0 {
                    return None;
                }
                (a.x as i128 * b.x as i128, base, ua.frac + ub.frac)
            }
            (BinOp::Accum, Some(ua), Some(ub)) if ua.base == ub.base => {
                let at = ua.frac.max(ub.frac);
                let m = (a.x as i128).checked_shl((at - ua.frac) as u32)?.checked_add((b.x as i128).checked_shl((at - ub.frac) as u32)?)?;
                (m, ua.base, at)
            }
            // Zero is a count of every unit: a sum's start.
            (BinOp::Accum, Some(u), None) if b.x == 0.0 => (a.x as i128, u.base, u.frac),
            (BinOp::Accum, None, Some(u)) if a.x == 0.0 => (b.x as i128, u.base, u.frac),
            _ => return None,
        };
        let (n, o) = int.round_scaled(m, frac - at, Tie::Even);
        self.counts.note(f, 0.0, o, false, false);
        Some(Mixed { x: n as f64, unit: Some(Unit { base, frac }) })
    }
}

impl Elem for Mixed {
    type Ctx = MixedCtx;
    const NATIVE: bool = false;
    fn from_f64(x: f64, _: &MixedCtx) -> Self {
        Mixed { x, unit: None }
    }
    fn from_scaled(n: f64, s: f64, value: f64, c: &MixedCtx) -> Self {
        if c.units { Mixed { x: n, unit: Some(Unit { base: s, frac: 0 }) } } else { Mixed { x: value, unit: None } }
    }
    fn constant(x: f64, c: &MixedCtx) -> Self {
        c.fit(c.arith, x, 0.0, x.is_infinite(), x.is_nan())
    }
    fn to_f64(self, c: &MixedCtx) -> f64 {
        c.real(self)
    }
    fn overflows(self, f: Float, _: &MixedCtx) -> bool {
        self.unit.is_none() && presk_rt::codes::overflows(f, self.x)
    }
    fn un(self, op: UnOp, c: &MixedCtx) -> Self {
        let a = c.real(self);
        let (y, r) = match op {
            UnOp::Neg => (-a, 0.0),
            UnOp::Abs => (a.abs(), 0.0),
            // a − s², exact: the root is above `s` where it is positive,
            // which is all a float reads of the rest.
            UnOp::Sqrt => {
                let s = a.sqrt();
                (s, -s.mul_add(s, -a))
            }
            f if c.w == F32 => ((a as f32).un(f, &()) as f64, 0.0),
            f => (a.un(f, &()), 0.0),
        };
        c.fit(c.arith, nan_of(y, a, a, c.arith), r, a.is_infinite(), a.is_nan())
    }
    fn bin(self, op: BinOp, o: Self, c: &MixedCtx) -> Self {
        if c.units
            && let Some(v) = c.counted(op, self, o)
        {
            return v;
        }
        let (a, b) = (c.real(self), c.real(o));
        let f = match op {
            BinOp::Accum => c.accum,
            BinOp::Product => c.product,
            _ => c.arith,
        };
        let (y, r) = match op.plain() {
            // Knuth's two-sum: `y + e` is `a ± b` exactly. The difference
            // itself is the machine's, so that a NaN keeps the sign the
            // one-format path gives it.
            BinOp::Add | BinOp::IAdd | BinOp::Sub => {
                let (y, b) = if op == BinOp::Sub { (a - b, -b) } else { (a + b, b) };
                let bv = y - a;
                (y, (a - (y - bv)) + (b - bv))
            }
            // The product's error by a fused multiply-add, exact.
            BinOp::Mul | BinOp::IMul => {
                let y = a * b;
                (y, a.mul_add(b, -y))
            }
            // The remainder a − q·b, exact: the quotient is above `q` where
            // it has the divisor's sign, which is all a float, the only
            // format that divides, reads of the rest.
            BinOp::Div => {
                let q = a / b;
                let rem = -q.mul_add(b, -a);
                (q, if b < 0.0 { -rem } else { rem })
            }
            BinOp::Accum | BinOp::Product => unreachable!("plain"),
        };
        c.fit(f, nan_of(y, a, b, f), r, a.is_infinite() || b.is_infinite(), a.is_nan() || b.is_nan())
    }
    fn to_format(self, f: Format, c: &MixedCtx) -> f64 {
        let (y, o, inf_in, nan_in) = match self.unit {
            // A count rounded once from its exact value.
            Some(u) => {
                let (m, e) = count_scaled(self.x, u);
                let (y, o) = f.round_scaled(m, e, Tie::Even);
                (y, o, false, false)
            }
            None => {
                let (y, o) = f.round_noted(self.x, Tie::Even);
                (y, o, self.x.is_infinite(), self.x.is_nan())
            }
        };
        c.counts.note(f, y, o, inf_in, nan_in);
        y
    }
}

struct Env<'a, E: Elem> {
    k: &'a Kernel,
    strides: Vec<Vec<usize>>,
    inputs: &'a [&'a Buf],
    out: &'a mut Buf,
    loops: Vec<usize>,
    reals: Vec<E>,
    ints: Vec<i64>,
    /// One register per vector variable, each of `k.lanes` elements.
    vecs: Vec<Vec<E>>,
    c: E::Ctx,
    /// Whether a store into binary16 or `e8m15` turned a finite value into
    /// an infinity: the overflow guard's flag of this call (plan 0.18.2
    /// step 4, D218), which the executor reads after it.
    overflow: bool,
}

impl<E: Elem> Env<'_, E> {
    fn offset(&self, buf: usize, index: &[Expr]) -> usize {
        let st = &self.strides[buf];
        index.iter().zip(st).map(|(e, s)| self.eval_i(e) as usize * s).sum()
    }

    fn load_f(&self, buf: usize, off: usize) -> E {
        let b: &Buf = if buf == self.k.out_buf() { self.out } else { self.inputs[buf] };
        match b {
            Buf::F64(v) => E::from_f64(v[off], &self.c),
            Buf::F32(v) => E::from_f64(v[off] as f64, &self.c),
            Buf::Bf16(v) => E::from_f64(presk_rt::bf16_to_f32(v[off]) as f64, &self.c),
            Buf::I64(v) => E::from_f64(v[off] as f64, &self.c),
            Buf::Exact(v) => E::from_f64(v[off].to_f64(), &self.c),
            // An integer weight: `f32(q)·s`, one `f32` multiplication, then
            // widened exactly if the kernel computes in `f64` (D44); its
            // integer in its scale where the kernel counts (D207).
            Buf::Quant(q) => E::from_scaled(q.int(off) as f64, q.scales[q.channel(off)] as f64, q.value(off) as f64, &self.c),
            // A quantized activation, `(q − z)·s`, exact: `q − z` is the
            // quotient, exact in an `f64` as the value is.
            Buf::Format(Format::Affine(a), v) => E::from_scaled(v[off] / a.scale(), a.scale(), v[off], &self.c),
            Buf::Format(_, v) => E::from_f64(v[off], &self.c),
            // binary16's or `e8m15`'s code, widened exactly (plan 0.18.2
            // step 1, D215).
            Buf::Codes(f, v) => E::from_f64(presk_rt::codes::get(*f, v, off) as f64, &self.c),
        }
    }

    fn load_i(&self, buf: usize, off: usize) -> i64 {
        let b: &Buf = if buf == self.k.out_buf() { self.out } else { self.inputs[buf] };
        match b {
            Buf::I64(v) => v[off],
            other => other.to_f64()[off] as i64,
        }
    }

    fn eval_f(&self, e: &Expr) -> E {
        match e {
            Expr::F(x) => E::constant(*x, &self.c),
            Expr::Load(b, ix) => {
                let off = self.offset(*b, ix);
                self.load_f(*b, off)
            }
            Expr::Sv(v) => self.reals[*v],
            Expr::Bin(op, a, b) => self.eval_f(a).bin(*op, self.eval_f(b), &self.c),
            Expr::Un(op, a) => self.eval_f(a).un(*op, &self.c),
            Expr::Select(c, a, b) => {
                if self.eval_bool(c) {
                    self.eval_f(a)
                } else {
                    self.eval_f(b)
                }
            }
            Expr::ToF(a) => E::constant(self.eval_i(a) as f64, &self.c),
            other => E::constant(self.eval_i(other) as f64, &self.c),
        }
    }

    fn eval_i(&self, e: &Expr) -> i64 {
        match e {
            Expr::I(i) => *i,
            Expr::Iv(v) => self.loops[*v] as i64,
            Expr::Si(v) => self.ints[*v],
            Expr::LoadI(b, ix) => {
                let off = self.offset(*b, ix);
                self.load_i(*b, off)
            }
            Expr::Bin(BinOp::IAdd, a, b) => self.eval_i(a) + self.eval_i(b),
            Expr::Bin(BinOp::IMul, a, b) => self.eval_i(a) * self.eval_i(b),
            Expr::ToI(a) => self.eval_f(a).to_f64(&self.c) as i64,
            Expr::Select(c, a, b) => {
                if self.eval_bool(c) {
                    self.eval_i(a)
                } else {
                    self.eval_i(b)
                }
            }
            other => self.eval_f(other).to_f64(&self.c) as i64,
        }
    }

    fn eval_bool(&self, e: &Expr) -> bool {
        match e {
            Expr::Cmp(op, a, b) => {
                let (x, y) = (self.eval_f(a), self.eval_f(b));
                match op {
                    CmpOp::Lt => x < y,
                    CmpOp::Le => x <= y,
                    CmpOp::Gt => x > y,
                    CmpOp::Ge => x >= y,
                    CmpOp::Eq => x == y,
                }
            }
            _ => self.eval_i(e) != 0,
        }
    }

    /// A vector expression, lane by lane. Every lane is an independent
    /// element of the same computation — never a partial sum of one — so the
    /// arithmetic each lane performs is exactly the arithmetic the scalar
    /// form performs for that element, in the same order (D42).
    fn eval_v(&self, e: &Expr) -> Vec<E> {
        let n = self.k.lanes;
        match e {
            Expr::Vv(v) => self.vecs[*v].clone(),
            Expr::VLoad(buf, ix) => {
                let off = self.offset(*buf, ix);
                (0..n).map(|i| self.load_f(*buf, off + i)).collect()
            }
            Expr::VSplat(x) => {
                let s = self.eval_f(x);
                vec![s; n]
            }
            Expr::VBin(op, a, b) => {
                let (x, y) = (self.eval_v(a), self.eval_v(b));
                x.into_iter().zip(y).map(|(p, q)| p.bin(*op, q, &self.c)).collect()
            }
            // A scalar in a vector position is one lane wide, broadcast.
            other => vec![self.eval_f(other); n],
        }
    }

    fn store_f(&mut self, off: usize, x: E) {
        let c = &self.c;
        // An emulated value goes into a native buffer as into any other
        // format, rounded once and counted.
        let to = |f: Float| x.to_format(Format::Float(f), c);
        match self.out {
            Buf::F64(v) => v[off] = if E::NATIVE { x.to_f64(c) } else { to(F64) },
            Buf::F32(v) => v[off] = if E::NATIVE { x.to_f64(c) as f32 } else { to(F32) as f32 },
            // Storage only: the value was computed in `f32` and is rounded
            // once, to nearest even, on the way to memory.
            Buf::Bf16(v) => v[off] = presk_rt::f32_to_bf16(if E::NATIVE { x.to_f64(c) as f32 } else { to(BF16) as f32 }),
            Buf::I64(v) => v[off] = x.to_f64(c) as i64,
            Buf::Format(f, v) => v[off] = x.to_format(*f, c),
            // The value rounded once to binary16 or `e8m15`, counted when
            // emulated, and stored as its code (plan 0.18.2 step 1, D215);
            // a finite value that rounds to an infinity sets the guard's
            // flag (step 4, D218).
            Buf::Codes(f, v) => {
                self.overflow |= x.overflows(*f, c);
                presk_rt::codes::put(*f, v, off, if E::NATIVE { x.to_f64(c) } else { x.to_format(Format::Float(*f), c) })
            }
            // Parameters are quantised, results never are.
            Buf::Exact(_) | Buf::Quant(_) => {}
        }
    }

    fn run(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            match s {
                Stmt::For { var, bound, body } => {
                    for i in 0..*bound {
                        self.loops[*var] = i;
                        self.run(body);
                    }
                }
                Stmt::Set { var, value } => {
                    let x = self.eval_f(value);
                    self.reals[*var] = x;
                }
                Stmt::SetI { var, value } => {
                    let x = self.eval_i(value);
                    self.ints[*var] = x;
                }
                Stmt::SetL { var, value } => {
                    let x = self.eval_i(value);
                    self.loops[*var] = x.max(0) as usize;
                }
                Stmt::VSet { var, value } => {
                    let x = self.eval_v(value);
                    self.vecs[*var] = x;
                }
                Stmt::VStore { buf, index, value } => {
                    let off = self.offset(*buf, index);
                    let x = self.eval_v(value);
                    for (i, e) in x.into_iter().enumerate() {
                        self.store_f(off + i, e);
                    }
                }
                Stmt::Store { buf, index, value } => {
                    let off = self.offset(*buf, index);
                    let x = self.eval_f(value);
                    self.store_f(off, x);
                }
                Stmt::StoreI { buf, index, value } => {
                    let off = self.offset(*buf, index);
                    let x = self.eval_i(value);
                    if let Buf::I64(v) = self.out {
                        v[off] = x;
                    }
                }
                Stmt::If { cond, then } => {
                    if self.eval_bool(cond) {
                        self.run(then);
                    }
                }
            }
        }
    }
}

/// Runs `k` on `inputs`, writing the output buffer (allocated by the caller
/// with the kernel's output spec).
pub fn run_kernel(k: &Kernel, inputs: &[&Buf], out: &mut Buf) {
    run_guarded(k, inputs, out);
}

/// [`run_kernel`], and whether a store into binary16 or `e8m15` turned a
/// finite value into an infinity, the overflow guard's flag (plan 0.18.2
/// step 4, D218).
pub fn run_guarded(k: &Kernel, inputs: &[&Buf], out: &mut Buf) -> bool {
    match k.dtype {
        crate::ir::Dtype::F64 => exec::<f64>(k, inputs, out, ()).1,
        // `bf16` is a storage type: the loop nest runs in `f32` and only the
        // buffers are narrow, which is the whole of the idiom (D34).
        crate::ir::Dtype::F32 | crate::ir::Dtype::BF16 => exec::<f32>(k, inputs, out, ()).1,
        // So are binary16 and `e8m15`, held as their codes, each load
        // widened exactly and each store rounded once (plan 0.18.2 step 2,
        // D216): the emulation's bits, as the native backend's.
        d if d.coded() => exec::<f32>(k, inputs, out, ()).1,
        d => d.unreachable(),
    }
}

/// The kernel run, its context and the guard's flag.
fn exec<E: Elem>(k: &Kernel, inputs: &[&Buf], out: &mut Buf, c: E::Ctx) -> (E::Ctx, bool) {
    let mut strides: Vec<Vec<usize>> = k.inputs.iter().map(|s| s.strides()).collect();
    strides.push(k.output.strides());
    let zero = E::constant(0.0, &c);
    let mut env = Env::<E> { k, strides, inputs, out, loops: vec![0; k.nloops], reals: vec![zero; k.nreal], ints: vec![0; k.nint], vecs: vec![vec![zero; k.lanes]; k.nvec], c, overflow: false };
    env.run(&k.body);
    (env.c, env.overflow)
}

/// The native type a float format's operations are computed in by the
/// emulation, `f32` or `f64`: the narrower that holds every number of the
/// format with at least 2p + 2 bits of significand, so that each operation
/// rounded twice is rounded once. `None` for a format of 26 to 52 bits
/// other than `f32` (no native type holds 2p + 2 of its bits).
pub fn emulated_in(f: Float) -> Option<Float> {
    if f == F64 || f == F32 {
        Some(f)
    } else if F32.contains(f) && 2 * f.p + 2 <= F32.p {
        Some(F32)
    } else if 2 * f.p + 2 <= F64.p {
        Some(F64)
    } else {
        None
    }
}

/// Runs `k` emulated in the format `arith` (plan 0.18.0 step 2, D205):
/// every operation rounded to it, every load, store and constant converted,
/// rounded once, ties to even; the buffers in any format, a
/// [`Buf::Format`] holding the values of one, a [`Buf::Codes`] the codes of
/// binary16 or `e8m15`. What the run's formats
/// produced besides rounding is counted. `f64` and `f32` give
/// [`run_kernel`]'s bits for a kernel of that type, counted.
///
/// Refused: a float format no native type emulates exactly
/// ([`emulated_in`]), and, in an integer or a fixed-point format, a
/// division or a function, which no such format has.
pub fn run_kernel_in(k: &Kernel, arith: Format, inputs: &[&Buf], out: &mut Buf) -> Result<Counts, String> {
    in_guarded(k, arith, inputs, out).map(|(c, _)| c)
}

/// [`run_kernel_in`] and the guard's flag ([`run_guarded`]).
fn in_guarded(k: &Kernel, arith: Format, inputs: &[&Buf], out: &mut Buf) -> Result<(Counts, bool), String> {
    Ok(match arith {
        Format::Float(f) => match emulated_in(f) {
            Some(w) if w == F32 => {
                let (c, o) = exec::<Soft<f32>>(k, inputs, out, SoftCtx { f, native: f == F32, counts: Counter::default() });
                (c.counts.get(), o)
            }
            Some(_) => {
                let (c, o) = exec::<Soft<f64>>(k, inputs, out, SoftCtx { f, native: f == F64, counts: Counter::default() });
                (c.counts.get(), o)
            }
            None => return Err(format!("no native type holds 2p + 2 = {} bits of {f}, so its operations cannot be emulated rounded once", 2 * f.p + 2)),
        },
        Format::Int(int) => {
            exact_ops(k, arith)?;
            let (c, o) = exec::<IntE>(k, inputs, out, IntCtx { int, counts: Counter::default() });
            (c.counts.get(), o)
        }
        Format::Fixed(fixed) => {
            exact_ops(k, arith)?;
            let (c, o) = exec::<FixE>(k, inputs, out, FixCtx { fixed, counts: Counter::default() });
            (c.counts.get(), o)
        }
        Format::Affine(_) => return Err(format!("the kernel `{}` computes in {arith}: an affine format is a stored value's, not an operation's", k.name)),
    })
}

/// The format of the values a buffer holds, as a kernel loads them: `None`
/// for indices.
fn held_format(b: &Buf) -> Option<Format> {
    Some(match b {
        Buf::F64(_) | Buf::Exact(_) => Format::Float(F64),
        // An integer weight is loaded as `f32(q)·s`.
        Buf::F32(_) | Buf::Quant(_) => Format::Float(F32),
        Buf::Bf16(_) => Format::Float(BF16),
        Buf::Format(f, _) => *f,
        Buf::Codes(f, _) => Format::Float(*f),
        Buf::I64(_) => return None,
    })
}

/// Runs `k` in the formats it carries (plan 0.18.0 step 3, D206), its
/// output in its storage type, and counts what the formats produced: the
/// kernel interpreter under a realization the native backends refuse.
/// Where the kernel's formats are one and that format holds every value of
/// its buffers, this is [`run_kernel_in`]; a native kernel so run gives
/// [`run_kernel`]'s bits. Otherwise every value is held exactly as an `f64`
/// and each operation rounded once to its own format ([`Mixed`]), which
/// needs every format to be one an `f64` holds.
///
/// Where its product format counts, an integer or a fixed-point one, and
/// it reads a scaled format, it counts their products in their unit
/// ([`Mixed`]), which needs an accumulation format that counts too.
pub fn run_emulated(k: &Kernel, inputs: &[&Buf], out: &mut Buf) -> Result<Counts, String> {
    run_emulated_guarded(k, inputs, out).map(|(c, _)| c)
}

/// [`run_emulated`], and whether a store into binary16 or `e8m15` turned a
/// finite value into an infinity ([`run_guarded`]), which the kernel
/// interpreter makes an error of the run where the realization is one the
/// native types run (plan 0.18.2 step 4, D218).
pub fn run_emulated_guarded(k: &Kernel, inputs: &[&Buf], out: &mut Buf) -> Result<(Counts, bool), String> {
    let a = k.arith;
    if let Some(f) = [a.arith, a.product, a.accum].into_iter().find(|f| matches!(f, Format::Affine(_))) {
        return Err(format!("the kernel `{}` rounds an operation to {f}: an affine format is a stored value's, not an operation's", k.name));
    }
    // A value read back from an affine output would be its rescaled codes:
    // a running sum, the two-pass softmax's exponentials, rounded at every
    // step.
    if let Some(f) = a.store
        && reads_output(k)
    {
        return Err(format!("the kernel `{}` reads back what it stores, and stores in {f}: a value of an affine format is stored once, as a nest ends", k.name));
    }
    let counts = |f: Format| matches!(f, Format::Int(_) | Format::Fixed(_));
    let scaled = inputs.iter().any(|b| matches!(b, Buf::Quant(_) | Buf::Format(Format::Affine(_), _)));
    if counts(a.product) && scaled && !counts(a.accum) {
        return Err(format!("the kernel `{}` counts its products of scaled values in {} and sums them in {}: a sum counted in its terms' unit is summed in an integer or a fixed-point format", k.name, a.product, a.accum));
    }
    let held = |b: &Buf| held_format(b).is_none_or(|f| a.arith.contains(f));
    // The output matters where the kernel reads it back: a sum kept there.
    if a.product == a.arith && a.accum == a.arith && inputs.iter().all(|b| held(b)) && (!reads_output(k) || held(out)) && !(counts(a.product) && scaled) {
        return in_guarded(k, a.arith, inputs, out);
    }
    mixed_guarded(k, inputs, out)
}

/// Runs `k` in the formats it carries with every value held exactly as an
/// `f64` and each operation rounded once to its own format ([`Mixed`]):
/// [`run_emulated`]'s path for a kernel of several formats, on any kernel,
/// so that the tests can hold it to the one-format path.
pub fn run_kernel_mixed(k: &Kernel, inputs: &[&Buf], out: &mut Buf) -> Result<Counts, String> {
    mixed_guarded(k, inputs, out).map(|(c, _)| c)
}

/// [`run_kernel_mixed`] and the guard's flag ([`run_guarded`]).
fn mixed_guarded(k: &Kernel, inputs: &[&Buf], out: &mut Buf) -> Result<(Counts, bool), String> {
    let a = k.arith;
    if let Some(f) = [a.arith, a.product, a.accum].into_iter().find(|f| !f.exact_in_f64()) {
        return Err(format!("the kernel `{}` computes in {} and rounds to {f}, which an f64 does not hold: a nest of several formats is emulated in f64", k.name, a));
    }
    let w = match a.arith {
        Format::Float(f) => emulated_in(f).ok_or_else(|| format!("no native type holds 2p + 2 = {} bits of {f}, so its operations cannot be emulated rounded once", 2 * f.p + 2))?,
        other => {
            exact_ops(k, other)?;
            F64
        }
    };
    let units = matches!(a.product, Format::Int(_) | Format::Fixed(_));
    let (c, o) = exec::<Mixed>(k, inputs, out, MixedCtx { arith: a.arith, product: a.product, accum: a.accum, w, units, inexact: Cell::new(false), counts: Counter::default() });
    if c.inexact.get() {
        return Err(format!("the kernel `{}` reads a sum counted in its terms' unit before it stores it, and an f64 does not hold its value: such a sum is stored as it ends", k.name));
    }
    Ok((c.counts.get(), o))
}

/// Whether `k` loads from its output buffer: a running sum kept there, or a
/// value it stored and reads again.
fn reads_output(k: &Kernel) -> bool {
    let out = k.out_buf();
    fn expr(e: &Expr, out: usize) -> bool {
        match e {
            Expr::Load(b, ix) | Expr::VLoad(b, ix) => *b == out || ix.iter().any(|x| expr(x, out)),
            Expr::LoadI(_, ix) => ix.iter().any(|x| expr(x, out)),
            Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) | Expr::VBin(_, a, b) => expr(a, out) || expr(b, out),
            Expr::Select(c, a, b) => expr(c, out) || expr(a, out) || expr(b, out),
            Expr::Un(_, a) | Expr::ToF(a) | Expr::ToI(a) | Expr::VSplat(a) => expr(a, out),
            Expr::F(_) | Expr::I(_) | Expr::Iv(_) | Expr::Sv(_) | Expr::Si(_) | Expr::Vv(_) => false,
        }
    }
    fn stmts(ss: &[Stmt], out: usize) -> bool {
        ss.iter().any(|s| match s {
            Stmt::For { body, .. } => stmts(body, out),
            Stmt::If { cond, then } => expr(cond, out) || stmts(then, out),
            Stmt::Set { value, .. } | Stmt::SetI { value, .. } | Stmt::SetL { value, .. } | Stmt::VSet { value, .. } => expr(value, out),
            Stmt::Store { index, value, .. } | Stmt::StoreI { index, value, .. } | Stmt::VStore { index, value, .. } => index.iter().any(|x| expr(x, out)) || expr(value, out),
        })
    }
    stmts(&k.body, out)
}

/// Whether every real operation of `k` is one an integer or a fixed-point
/// format has: +, −, ×, a negation, an absolute value.
fn exact_ops(k: &Kernel, arith: Format) -> Result<(), String> {
    fn expr(e: &Expr) -> Option<String> {
        match e {
            Expr::Un(op, a) => match op {
                UnOp::Neg | UnOp::Abs => expr(a),
                _ => Some(format!("{op:?}").to_lowercase()),
            },
            Expr::Bin(BinOp::Div, ..) | Expr::VBin(BinOp::Div, ..) => Some("a division".into()),
            Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) | Expr::VBin(_, a, b) => expr(a).or_else(|| expr(b)),
            Expr::Select(c, a, b) => expr(c).or_else(|| expr(a)).or_else(|| expr(b)),
            Expr::ToF(a) | Expr::ToI(a) | Expr::VSplat(a) => expr(a),
            Expr::Load(_, ix) | Expr::LoadI(_, ix) | Expr::VLoad(_, ix) => ix.iter().find_map(expr),
            Expr::F(_) | Expr::I(_) | Expr::Iv(_) | Expr::Sv(_) | Expr::Si(_) | Expr::Vv(_) => None,
        }
    }
    fn stmts(ss: &[Stmt]) -> Option<String> {
        ss.iter().find_map(|s| match s {
            Stmt::For { body, .. } => stmts(body),
            Stmt::If { cond, then } => expr(cond).or_else(|| stmts(then)),
            Stmt::Set { value, .. } | Stmt::SetI { value, .. } | Stmt::SetL { value, .. } | Stmt::VSet { value, .. } => expr(value),
            Stmt::Store { index, value, .. } | Stmt::StoreI { index, value, .. } | Stmt::VStore { index, value, .. } => index.iter().find_map(expr).or_else(|| expr(value)),
        })
    }
    match stmts(&k.body) {
        Some(op) => Err(format!("the kernel `{}` computes {op}, which {arith} does not have", k.name)),
        None => Ok(()),
    }
}

/// A fresh output buffer for a kernel, of zeros: what `presk autotune` times
/// a point's launches on, as it always has.
pub fn output_buf(k: &Kernel) -> Buf {
    let n = k.output.numel();
    if let (Some(f), BufKind::Real) = (k.arith.store, k.output.kind) {
        return Buf::Format(f, vec![0.0; n].into());
    }
    match k.output.kind {
        BufKind::Idx => Buf::I64(vec![0; n].into()),
        BufKind::Real | BufKind::Quant { .. } | BufKind::Wide(_) => Buf::zeros(k.dtype, n),
    }
}

/// An output buffer for a launch of the host (plan 0.10.0 step 6): one of
/// the free list's of its length, whatever it holds, or a new one; the
/// kernel writes every element before it reads any (`presk_rt::elems`).
pub fn launch_buf(k: &Kernel) -> Buf {
    let n = k.output.numel();
    // A quantized activation's codes (plan 0.18.0 step 4).
    if let (Some(f), BufKind::Real) = (k.arith.store, k.output.kind) {
        return Buf::Format(f, presk_rt::Elems::fresh(n));
    }
    match (k.output.kind, k.dtype) {
        (BufKind::Idx, _) => Buf::I64(presk_rt::Elems::fresh(n)),
        (BufKind::Real | BufKind::Quant { .. } | BufKind::Wide(_), crate::ir::Dtype::F64) => Buf::F64(presk_rt::Elems::fresh(n)),
        (BufKind::Real | BufKind::Quant { .. } | BufKind::Wide(_), crate::ir::Dtype::F32) => Buf::F32(presk_rt::Elems::fresh(n)),
        (BufKind::Real | BufKind::Quant { .. } | BufKind::Wide(_), crate::ir::Dtype::BF16) => Buf::Bf16(presk_rt::Elems::fresh(n)),
        // binary16's or `e8m15`'s codes (plan 0.18.2 step 1, D215).
        (BufKind::Real | BufKind::Quant { .. } | BufKind::Wide(_), d) if d.coded() => Buf::Codes(d.format(), presk_rt::Elems::fresh(n * d.bytes())),
        // A type the kernel interpreter emulates (plan 0.18.0 step 3).
        (BufKind::Real | BufKind::Quant { .. } | BufKind::Wide(_), d) => Buf::Format(Format::Float(d.format()), presk_rt::Elems::fresh(n)),
    }
}
