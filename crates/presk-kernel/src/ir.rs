//! PSK-Kernel v0: one kernel per PSK-Core tensor primitive at concrete
//! shapes and one machine type — a loop nest over buffers with scalar
//! operations. Buffers are row-major; inputs are numbered `0..n`, the output
//! is buffer `n`. The textual form (`Display`) is what `presk dump --stage
//! kernel` prints.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dtype {
    F64,
    F32,
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Dtype::F64 => "f64",
            Dtype::F32 => "f32",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufKind {
    /// Reals in the kernel's dtype.
    Real,
    /// Indices, `i64`.
    Idx,
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
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    For { var: usize, bound: usize, body: Vec<Stmt> },
    Set { var: usize, value: Expr },
    SetI { var: usize, value: Expr },
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
    pub body: Vec<Stmt>,
}

impl Kernel {
    pub fn out_buf(&self) -> usize {
        self.inputs.len()
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
