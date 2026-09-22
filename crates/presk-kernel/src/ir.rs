//! PSK-Kernel v0: one kernel per PSK-Core tensor primitive at concrete
//! shapes and one machine type — a loop nest over buffers with scalar
//! operations. Buffers are row-major; inputs are numbered `0..n`, the output
//! is buffer `n`. The textual form (`Display`) is what `presk dump --stage
//! kernel` prints.

use std::fmt;

/// The machine type of a kernel's real buffers. Defined at the runtime seam
/// since 0.5.0 step 2, because a realization assigns it per site and the
/// backends must agree on what the sites mean.
pub use presk_rt::Dtype;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufKind {
    /// Reals in the kernel's dtype.
    Real,
    /// Indices, `i64`.
    Idx,
    /// A parameter stored as integers with a scale per channel, the scales
    /// constant along axis `over` (0.6.0 step 5b, D44). A `Load` or `VLoad`
    /// of it yields `f32(q)·s` in the kernel's arithmetic type. The lowering
    /// never produces it: it emits ordinary real loads, and the host marks
    /// an input it hands integer weights to, so no kernel shape needs to
    /// know about quantisation.
    Quant { bits: u8, over: usize },
}

impl BufSpec {
    /// The strides of the scales of a [`BufKind::Quant`] buffer, one per
    /// axis, zero along `over`: the scale of element `ix` is at
    /// `Σ ix_a · sstride_a`.
    pub fn scale_strides(&self) -> Vec<usize> {
        let BufKind::Quant { over, .. } = self.kind else { return vec![0; self.dims.len()] };
        let mut s = vec![0; self.dims.len()];
        let mut acc = 1;
        for a in (0..self.dims.len()).rev() {
            if a != over {
                s[a] = acc;
                acc *= self.dims[a];
            }
        }
        s
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufSpec {
    pub dims: Vec<usize>,
    pub kind: BufKind,
    /// Axis names, for the textual form (`batch`, `_`).
    pub names: Vec<Option<String>>,
}

impl BufSpec {
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn strides(&self) -> Vec<usize> {
        let mut s = vec![1; self.dims.len()];
        for i in (0..self.dims.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * self.dims[i + 1];
        }
        s
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    /// Integer arithmetic on index expressions.
    IAdd,
    IMul,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Exp,
    Log,
    Tanh,
    Sqrt,
    Sin,
    Cos,
    Erf,
    Abs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
}

/// Scalar expressions. `Iv`, `Si`, `I`, `LoadI`, `IAdd`/`IMul` and `ToI` are
/// integers; everything else is a real of the kernel's dtype. `Cmp` compares
/// two reals and is used only as the condition of `Select` and `If`.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    F(f64),
    I(i64),
    /// Load a real from buffer `.0` at the index expressions `.1` (one per axis).
    Load(usize, Vec<Expr>),
    LoadI(usize, Vec<Expr>),
    /// A loop variable.
    Iv(usize),
    /// A real scalar variable.
    Sv(usize),
    /// An integer scalar variable.
    Si(usize),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Un(UnOp, Box<Expr>),
    Cmp(CmpOp, Box<Expr>, Box<Expr>),
    Select(Box<Expr>, Box<Expr>, Box<Expr>),
    ToF(Box<Expr>),
    ToI(Box<Expr>),

    // ----- the vector form (0.6.0 step 1) -----
    //
    // A *lane* is an independent element of the same computation, never a
    // partial sum of one: the blocked contraction's accumulators are
    // different output columns, so lanes touch no addition's position and a
    // vector kernel is bit-identical to its scalar form (D42). Nothing here
    // may be used to split a reduction — that would be an `Interleaved`
    // order and owes a declared reassociation.
    /// A vector variable of the kernel's lane width.
    Vv(usize),
    /// `width` contiguous reals from buffer `.0`, the index naming the first.
    VLoad(usize, Vec<Expr>),
    /// The same scalar in every lane.
    VSplat(Box<Expr>),
    /// Lane-wise arithmetic.
    VBin(BinOp, Box<Expr>, Box<Expr>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    For { var: usize, bound: usize, body: Vec<Stmt> },
    Set { var: usize, value: Expr },
    SetI { var: usize, value: Expr },
    /// Assign a *loop* variable — the one an index expression reads through
    /// [`Expr::Iv`]. A `For` is the only other thing that writes one, so this
    /// is what lets a lowering unroll an index: `k = cb·n + t` for a literal
    /// `t`, which is how an interleaved contraction addresses its terms
    /// (0.5.0 step 7). The variable is not a loop bound and carries no
    /// iteration of its own.
    SetL { var: usize, value: Expr },
    /// Assign a vector variable (0.6.0 step 1).
    VSet { var: usize, value: Expr },
    /// Store a vector to `width` contiguous reals, the index naming the first.
    VStore { buf: usize, index: Vec<Expr>, value: Expr },
    Store { buf: usize, index: Vec<Expr>, value: Expr },
    StoreI { buf: usize, index: Vec<Expr>, value: Expr },
    If { cond: Expr, then: Vec<Stmt> },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Kernel {
    /// `add`, `softmax axis seq2`, …: the primitive it realizes.
    pub name: String,
    pub dtype: Dtype,
    pub inputs: Vec<BufSpec>,
    pub output: BufSpec,
    /// Numbers of loop variables, real scalars and integer scalars.
    pub nloops: usize,
    pub nreal: usize,
    pub nint: usize,
    /// Vector variables, and how many lanes each holds. A width of one is a
    /// scalar kernel, which is every kernel before 0.6.0.
    pub nvec: usize,
    pub lanes: usize,
    pub body: Vec<Stmt>,
}

impl Kernel {
    pub fn out_buf(&self) -> usize {
        self.inputs.len()
    }

    /// The bound of the outermost loop when splitting it across threads is
    /// safe. Two conditions: the body is a single loop nest, and every store
    /// indexes the output with that loop's variable *directly*, so different
    /// iterations write different elements. A reduction over the outer axis
    /// fails the second (its store has no such index), and so does a
    /// scatter like `embed_vjp`, whose index is a loaded value.
    ///
    /// Splitting on such an axis is exact: each output element is computed by
    /// the same inner loops in the same order, only on another thread.
    pub fn parallel_outer(&self) -> Option<usize> {
        let [Stmt::For { var, bound, body }] = self.body.as_slice() else { return None };
        let (mut stores, mut ok) = (0usize, true);
        fn walk(stmts: &[Stmt], var: usize, stores: &mut usize, ok: &mut bool) {
            for s in stmts {
                match s {
                    Stmt::For { body, .. } => walk(body, var, stores, ok),
                    Stmt::If { then, .. } => walk(then, var, stores, ok),
                    Stmt::Store { index, .. } | Stmt::StoreI { index, .. } => {
                        *stores += 1;
                        if !index.iter().any(|e| matches!(e, Expr::Iv(v) if *v == var)) {
                            *ok = false;
                        }
                    }
                    // An assignment to the parallel candidate's own
                    // variable would break the disjointness the analysis is
                    // about, so it disqualifies the axis.
                    Stmt::SetL { var: v, .. } if *v == var => *ok = false,
                    Stmt::VStore { index, .. } => {
                        *stores += 1;
                        if !index.iter().any(|e| matches!(e, Expr::Iv(v) if *v == var)) {
                            *ok = false;
                        }
                    }
                    Stmt::Set { .. } | Stmt::SetI { .. } | Stmt::SetL { .. } | Stmt::VSet { .. } => {}
                }
            }
        }
        walk(body, *var, &mut stores, &mut ok);
        if ok && stores > 0 { Some(*bound) } else { None }
    }
}

// ----- textual form ----------------------------------------------------------

fn spec(s: &BufSpec, dtype: Dtype) -> String {
    let axes: Vec<String> = s.dims.iter().zip(&s.names).map(|(d, n)| match n {
        Some(n) => format!("{n}: {d}"),
        None => d.to_string(),
    }).collect();
    let t = match s.kind {
        BufKind::Real => dtype.to_string(),
        BufKind::Idx => "i64".into(),
        BufKind::Quant { bits, over } => format!("int{bits}/{over}"),
    };
    format!("{t}[{}]", axes.join(", "))
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::F(x) => write!(f, "{x}"),
            Expr::I(i) => write!(f, "{i}"),
            Expr::Load(b, ix) | Expr::LoadI(b, ix) => write!(f, "%{b}[{}]", ix.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(", ")),
            Expr::Iv(v) => write!(f, "i{v}"),
            Expr::Sv(v) => write!(f, "s{v}"),
            Expr::Si(v) => write!(f, "n{v}"),
            Expr::Vv(v) => write!(f, "v{v}"),
            Expr::VLoad(b, ix) => write!(f, "%{b}[{}]<v>", ix.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(", ")),
            Expr::VSplat(x) => write!(f, "splat({x})"),
            Expr::VBin(op, a, b) => write!(f, "({a} {} {b})<v>", match op {
                BinOp::Add | BinOp::IAdd => "+",
                BinOp::Sub => "-",
                BinOp::Mul | BinOp::IMul => "*",
                BinOp::Div => "/",
            }),
            Expr::Bin(op, a, b) => {
                let s = match op {
                    BinOp::Add | BinOp::IAdd => "+",
                    BinOp::Sub => "-",
                    BinOp::Mul | BinOp::IMul => "*",
                    BinOp::Div => "/",
                };
                write!(f, "({a} {s} {b})")
            }
            Expr::Un(op, a) => write!(f, "{}({a})", format!("{op:?}").to_lowercase()),
            Expr::Cmp(op, a, b) => {
                let s = match op {
                    CmpOp::Lt => "<",
                    CmpOp::Le => "<=",
                    CmpOp::Gt => ">",
                    CmpOp::Ge => ">=",
                    CmpOp::Eq => "==",
                };
                write!(f, "({a} {s} {b})")
            }
            Expr::Select(c, a, b) => write!(f, "select({c}, {a}, {b})"),
            Expr::ToF(a) => write!(f, "real({a})"),
            Expr::ToI(a) => write!(f, "int({a})"),
        }
    }
}

fn fmt_stmts(stmts: &[Stmt], indent: usize, out: &mut String) {
    let pad = "  ".repeat(indent);
    for s in stmts {
        match s {
            Stmt::For { var, bound, body } => {
                out.push_str(&format!("{pad}for i{var} in 0..{bound} {{\n"));
                fmt_stmts(body, indent + 1, out);
                out.push_str(&format!("{pad}}}\n"));
            }
            Stmt::Set { var, value } => out.push_str(&format!("{pad}s{var} = {value}\n")),
            Stmt::SetI { var, value } => out.push_str(&format!("{pad}n{var} = {value}\n")),
            Stmt::SetL { var, value } => out.push_str(&format!("{pad}i{var} = {value}\n")),
            Stmt::VSet { var, value } => out.push_str(&format!("{pad}v{var} = {value}\n")),
            Stmt::VStore { buf, index, value } => out.push_str(&format!("{pad}%{buf}[{}] <v= {value}\n", index.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(", "))),
            Stmt::Store { buf, index, value } | Stmt::StoreI { buf, index, value } => {
                out.push_str(&format!("{pad}%{buf}[{}] = {value}\n", index.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(", ")));
            }
            Stmt::If { cond, then } => {
                out.push_str(&format!("{pad}if {cond} {{\n"));
                fmt_stmts(then, indent + 1, out);
                out.push_str(&format!("{pad}}}\n"));
            }
        }
    }
}

impl fmt::Display for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ins: Vec<String> = self.inputs.iter().enumerate().map(|(i, s)| format!("%{i} : {}", spec(s, self.dtype))).collect();
        writeln!(f, "kernel {} ({}) -> %{} : {} {{", self.name, ins.join(", "), self.out_buf(), spec(&self.output, self.dtype))?;
        let mut body = String::new();
        fmt_stmts(&self.body, 1, &mut body);
        f.write_str(&body)?;
        f.write_str("}\n")
    }
}
