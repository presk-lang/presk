//! Abstract syntax of one Presk source file. Dimension expressions are
//! ordinary [`Expr`]s; their interpretation as linear forms belongs to
//! `presk-types`.

use crate::source::{FileId, Span};

#[derive(Clone, Debug)]
pub struct Program {
    pub file: FileId,
    pub decls: Vec<Decl>,
}

#[derive(Clone, Debug)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

/// `a.b.c`
#[derive(Clone, Debug)]
pub struct Path {
    pub segs: Vec<Ident>,
    pub span: Span,
}

impl Path {
    pub fn single(id: Ident) -> Self {
        Path { span: id.span, segs: vec![id] }
    }

    pub fn is_single(&self, name: &str) -> bool {
        self.segs.len() == 1 && self.segs[0].name == name
    }

    pub fn dotted(&self) -> String {
        self.segs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(".")
    }
}

#[derive(Clone, Debug)]
pub enum Decl {
    /// `import "path" as alias`
    Import { path: String, alias: Ident, span: Span },
    /// `const NAME [: ty] = expr`
    Const { name: Ident, ty: Option<Ty>, value: Expr, span: Span },
    /// `static assert expr`
    StaticAssert { cond: Expr, span: Span },
    /// `dim NAME [<= expr]`
    Dim { name: Ident, bound: Option<Expr>, span: Span },
    /// `type NAME = ty`
    TypeAlias { name: Ident, ty: Ty, span: Span },
    Module(ModuleDecl),
    Fn(FnDecl),
    /// `optimizer NAME { items }`
    Optimizer { name: Ident, items: Vec<ConfigItem>, span: Span },
    /// `training run { items }`
    Training { items: Vec<ConfigItem>, span: Span },
    /// `inference serve { items }`
    Inference { items: Vec<ConfigItem>, span: Span },
    /// `effect NAME`
    Effect { name: Ident, span: Span },
    /// `model NAME = modexpr`
    Model { name: Ident, body: ModExpr, span: Span },
}

impl Decl {
    pub fn span(&self) -> Span {
        match self {
            Decl::Import { span, .. }
            | Decl::Const { span, .. }
            | Decl::StaticAssert { span, .. }
            | Decl::Dim { span, .. }
            | Decl::TypeAlias { span, .. }
            | Decl::Optimizer { span, .. }
            | Decl::Training { span, .. }
            | Decl::Inference { span, .. }
            | Decl::Effect { span, .. }
            | Decl::Model { span, .. } => *span,
            Decl::Module(m) => m.span,
            Decl::Fn(f) => f.span,
        }
    }

    pub fn name(&self) -> Option<&Ident> {
        match self {
            Decl::Import { alias, .. } => Some(alias),
            Decl::Const { name, .. }
            | Decl::Dim { name, .. }
            | Decl::TypeAlias { name, .. }
            | Decl::Optimizer { name, .. }
            | Decl::Effect { name, .. }
            | Decl::Model { name, .. } => Some(name),
            Decl::Module(m) => Some(&m.name),
            Decl::Fn(f) => Some(&f.name),
            Decl::StaticAssert { .. } | Decl::Training { .. } | Decl::Inference { .. } => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ModuleDecl {
    pub name: Ident,
    pub typarams: Vec<TyParam>,
    pub members: Vec<Member>,
    pub span: Span,
}

/// `{In Out : dim, S : Shape}` — one entry per name.
#[derive(Clone, Debug)]
pub struct TyParam {
    pub name: Ident,
    /// `dim` or `Shape`
    pub kind: Ident,
}

#[derive(Clone, Debug)]
pub enum Member {
    /// `param a, b : ty [init = expr]`
    Param { names: Vec<Ident>, ty: Ty, init: Option<Expr>, span: Span },
    /// `use a, b : modexpr`
    Use { names: Vec<Ident>, module: ModExpr, span: Span },
    Fn(FnDecl),
}

/// A module expression: `Linear{D, D}`, `ff.MLP{D, 4*D}`, `stack{L}(Block)`,
/// `A ∘ B`.
#[derive(Clone, Debug)]
pub enum ModExpr {
    App { path: Path, dim_args: Vec<Expr>, mod_args: Vec<ModExpr>, span: Span },
    Compose { parts: Vec<ModExpr>, span: Span },
}

impl ModExpr {
    pub fn span(&self) -> Span {
        match self {
            ModExpr::App { span, .. } | ModExpr::Compose { span, .. } => *span,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FnDecl {
    pub name: Ident,
    pub typarams: Vec<TyParam>,
    pub params: Vec<Param>,
    pub ret: Option<Ty>,
    /// `uses Rand`
    pub effects: Vec<Ident>,
    pub body: Expr,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub name: Ident,
    pub ty: Option<Ty>,
    pub default: Option<Expr>,
    /// `axis i` parameter form
    pub axis: bool,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum ConfigItem {
    /// `name = expr`, `name : ty = expr` or `name : expr`
    Field { name: Ident, ty: Option<Ty>, value: Expr, span: Span },
    /// `checkpoint every N steps to "path"`
    Checkpoint { every: Expr, to: String, span: Span },
}

#[derive(Clone, Debug)]
pub enum Ty {
    /// `ℝ[axes]`; bare `ℝ` is `ℝ[]`
    Real { axes: Vec<AxisSpec>, span: Span },
    /// `idx[N][axes]`
    Idx { bound: Box<Expr>, axes: Vec<AxisSpec>, span: Span },
    /// `param ty`
    Param { inner: Box<Ty>, span: Span },
    /// `params path`
    Params { module: Path, span: Span },
    /// `nat`, `f64`, `Tokens`, `MLP{D, 4*D}`, `ff.Linear{D, D}`
    Named { path: Path, args: Vec<Expr>, span: Span },
    Tuple { elems: Vec<Ty>, span: Span },
}

impl Ty {
    pub fn span(&self) -> Span {
        match self {
            Ty::Real { span, .. }
            | Ty::Idx { span, .. }
            | Ty::Param { span, .. }
            | Ty::Params { span, .. }
            | Ty::Named { span, .. }
            | Ty::Tuple { span, .. } => *span,
        }
    }
}

#[derive(Clone, Debug)]
pub enum AxisSpec {
    /// `[name:] dimexpr`
    Axis { name: Option<Ident>, dim: Expr },
    /// `...`
    Ellipsis { span: Span },
}

#[derive(Clone, Debug)]
pub enum Expr {
    Lit { lit: Lit, span: Span },
    Var { path: Path },
    /// `let pat = init NEWLINE body`
    Let { pat: Pat, init: Box<Expr>, body: Box<Expr>, span: Span },
    /// `λ pat. body`
    Lambda { pat: Pat, body: Box<Expr>, span: Span },
    Call { callee: Box<Expr>, args: Vec<Arg>, span: Span },
    /// `base.field` on a non-path base
    Field { base: Box<Expr>, field: Ident, span: Span },
    /// `base[sel, …]`
    Index { base: Box<Expr>, sels: Vec<IndexSel>, span: Span },
    /// `base{dim args}`
    Inst { base: Box<Expr>, args: Vec<Expr>, span: Span },
    /// `ein[axes](body)`
    Ein { axes: Vec<Ident>, body: Box<Expr>, span: Span },
    Binary { op: BinOp, lhs: Box<Expr>, rhs: Box<Expr>, span: Span },
    Unary { op: UnOp, expr: Box<Expr>, span: Span },
    /// `expr : ty`
    Ascribe { expr: Box<Expr>, ty: Ty, span: Span },
    Tuple { elems: Vec<Expr>, span: Span },
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Lit { span, .. }
            | Expr::Let { span, .. }
            | Expr::Lambda { span, .. }
            | Expr::Call { span, .. }
            | Expr::Field { span, .. }
            | Expr::Index { span, .. }
            | Expr::Inst { span, .. }
            | Expr::Ein { span, .. }
            | Expr::Binary { span, .. }
            | Expr::Unary { span, .. }
            | Expr::Ascribe { span, .. }
            | Expr::Tuple { span, .. } => *span,
            Expr::Var { path } => path.span,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Pat {
    Ident(Ident),
    Tuple { elems: Vec<Pat>, span: Span },
}

impl Pat {
    pub fn span(&self) -> Span {
        match self {
            Pat::Ident(i) => i.span,
            Pat::Tuple { span, .. } => *span,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Arg {
    Pos(Expr),
    /// `name = expr`
    Named { name: Ident, value: Expr },
    /// `axis name`
    Axis(Ident),
    /// `name: expr` (an axis name with its length, e.g. `positions(seq: T)`)
    NamedAxis { name: Ident, value: Expr },
}

#[derive(Clone, Debug)]
pub enum IndexSel {
    Pos(Expr),
    /// `axis: expr`
    Named { axis: Ident, value: Expr },
}

#[derive(Clone, Debug)]
pub enum Lit {
    Nat(String),
    Float(String),
    Quantity { num: String, unit: String },
    Str(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    /// `|>`
    Pipe,
    /// `|`
    Alt,
    /// `⊗`
    Tensor,
    /// `∘`
    Compose,
    Eq,
    Le,
    Ge,
    Lt,
    Gt,
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    /// `·`
    Dot,
    Pow,
}

impl BinOp {
    pub fn text(self) -> &'static str {
        match self {
            BinOp::Pipe => "|>",
            BinOp::Alt => "|",
            BinOp::Tensor => "⊗",
            BinOp::Compose => "∘",
            BinOp::Eq => "==",
            BinOp::Le => "<=",
            BinOp::Ge => ">=",
            BinOp::Lt => "<",
            BinOp::Gt => ">",
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Rem => "%",
            BinOp::Dot => "·",
            BinOp::Pow => "^",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp {
    Neg,
}
