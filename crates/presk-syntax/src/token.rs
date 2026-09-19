//! Tokens of the Presk surface syntax.

use std::fmt;

use crate::source::Span;

/// Reserved words. `model`, `run`, `serve`, `stream`, `checkpoint`, `every`,
/// `steps` and `to` are *contextual*: they lex as identifiers and the parser
/// recognizes them by their text, because `model` is also an axis name and a
/// configuration field.
///
/// ASCII spellings (both forms lex to the same token; the Unicode form is
/// canonical and is what diagnostics print): `ℝ` = `Real`, `λ` = `\`,
/// `·` = `@`, `⊗` = `><`, `∘` = `<<`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kw {
    Module,
    Param,
    Params,
    Use,
    Fn,
    Const,
    Dim,
    Type,
    Import,
    As,
    Static,
    Assert,
    Optimizer,
    Training,
    Inference,
    Effect,
    Let,
    Uses,
    Init,
    Axis,
    Idx,
    Ein,
    /// `ℝ`
    Real,
    /// `λ`
    Lambda,
}

impl Kw {
    pub fn from_ident(s: &str) -> Option<Kw> {
        Some(match s {
            "module" => Kw::Module,
            "param" => Kw::Param,
            "params" => Kw::Params,
            "use" => Kw::Use,
            "fn" => Kw::Fn,
            "const" => Kw::Const,
            "dim" => Kw::Dim,
            "type" => Kw::Type,
            "import" => Kw::Import,
            "as" => Kw::As,
            "static" => Kw::Static,
            "assert" => Kw::Assert,
            "optimizer" => Kw::Optimizer,
            "training" => Kw::Training,
            "inference" => Kw::Inference,
            "effect" => Kw::Effect,
            "let" => Kw::Let,
            "uses" => Kw::Uses,
            "init" => Kw::Init,
            "axis" => Kw::Axis,
            "idx" => Kw::Idx,
            "ein" => Kw::Ein,
            "ℝ" | "Real" => Kw::Real,
            "λ" => Kw::Lambda,
            _ => return None,
        })
    }

    pub fn text(self) -> &'static str {
        match self {
            Kw::Module => "module",
            Kw::Param => "param",
            Kw::Params => "params",
            Kw::Use => "use",
            Kw::Fn => "fn",
            Kw::Const => "const",
            Kw::Dim => "dim",
            Kw::Type => "type",
            Kw::Import => "import",
            Kw::As => "as",
            Kw::Static => "static",
            Kw::Assert => "assert",
            Kw::Optimizer => "optimizer",
            Kw::Training => "training",
            Kw::Inference => "inference",
            Kw::Effect => "effect",
            Kw::Let => "let",
            Kw::Uses => "uses",
            Kw::Init => "init",
            Kw::Axis => "axis",
            Kw::Idx => "idx",
            Kw::Ein => "ein",
            Kw::Real => "ℝ",
            Kw::Lambda => "λ",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    /// `·` last-axis contraction
    Dot,
    Star,
    Slash,
    Percent,
    Plus,
    Minus,
    Caret,
    EqEq,
    Le,
    Ge,
    Lt,
    Gt,
    Eq,
    Arrow,
    /// `|>`
    Pipe,
    /// `|`
    Bar,
    /// `⊗`
    Tensor,
    /// `∘`
    Compose,
    Period,
    Comma,
    Colon,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    /// `...`
    Ellipsis,
}

impl Op {
    pub fn text(self) -> &'static str {
        match self {
            Op::Dot => "·",
            Op::Star => "*",
            Op::Slash => "/",
            Op::Percent => "%",
            Op::Plus => "+",
            Op::Minus => "-",
            Op::Caret => "^",
            Op::EqEq => "==",
            Op::Le => "<=",
            Op::Ge => ">=",
            Op::Lt => "<",
            Op::Gt => ">",
            Op::Eq => "=",
            Op::Arrow => "->",
            Op::Pipe => "|>",
            Op::Bar => "|",
            Op::Tensor => "⊗",
            Op::Compose => "∘",
            Op::Period => ".",
            Op::Comma => ",",
            Op::Colon => ":",
            Op::LParen => "(",
            Op::RParen => ")",
            Op::LBracket => "[",
            Op::RBracket => "]",
            Op::LBrace => "{",
            Op::RBrace => "}",
            Op::Ellipsis => "...",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokKind {
    /// NFC-normalized identifier.
    Ident(String),
    Kw(Kw),
    /// Digits only; the text is kept (arbitrary size).
    Nat(String),
    /// Decimal with a fraction and/or an exponent; the text is kept.
    Float(String),
    /// A number immediately followed by a unit, e.g. `5ms`, `204GB/s`.
    Quantity { num: String, unit: String },
    Str(String),
    Op(Op),
    /// One or more line breaks.
    Newline,
    Eof,
}

impl fmt::Display for TokKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokKind::Ident(s) => write!(f, "identifier `{s}`"),
            TokKind::Kw(k) => write!(f, "`{}`", k.text()),
            TokKind::Nat(s) | TokKind::Float(s) => write!(f, "number `{s}`"),
            TokKind::Quantity { num, unit } => write!(f, "quantity `{num}{unit}`"),
            TokKind::Str(s) => write!(f, "string {s:?}"),
            TokKind::Op(o) => write!(f, "`{}`", o.text()),
            TokKind::Newline => write!(f, "end of line"),
            TokKind::Eof => write!(f, "end of file"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Token {
    pub kind: TokKind,
    pub span: Span,
}
