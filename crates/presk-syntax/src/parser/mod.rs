//! Recursive-descent parser with Pratt expressions.
//!
//! Layout: the token stream contains `Newline` tokens. Inside `(`, `[` and
//! dimension-argument `{…}` the parser is in *free* mode and newlines are
//! skipped; in function bodies, module bodies and configuration blocks it is
//! in *lines* mode, where an expression ends at a newline unless the next
//! token is a binary operator that cannot begin an expression (see
//! `expr.rs`). Declarations recover at the next declaration keyword at
//! column 0.

mod config;
mod decl;
mod expr;
mod ty;

use crate::ast::{Ident, Program};
use crate::diag::Diagnostic;
use crate::lexer;
use crate::source::{FileId, SourceMap, Span};
use crate::token::{Kw, Op, TokKind, Token};

/// Parse one file. Lexer and parser diagnostics are returned together; the
/// program is returned even when there are errors (with the failing
/// declarations dropped) so that later phases can still report on the rest.
pub fn parse_file(sm: &SourceMap, file: FileId) -> (Program, Vec<Diagnostic>) {
    let lexed = lexer::lex(file, sm.text(file));
    let mut p = Parser { toks: lexed.tokens, pos: 0, free: 0, diags: lexed.diags, file, line_starts: line_starts(sm.text(file)) };
    let decls = p.program();
    (Program { file, decls }, p.diags)
}

fn line_starts(text: &str) -> Vec<u32> {
    let mut v = vec![0];
    for (i, b) in text.bytes().enumerate() {
        if b == b'\n' {
            v.push(i as u32 + 1);
        }
    }
    v
}

/// Marker error: the diagnostic has already been recorded.
#[derive(Debug)]
pub(crate) struct ParseError;

pub(crate) type PResult<T> = Result<T, ParseError>;

pub(crate) struct Parser {
    toks: Vec<Token>,
    pos: usize,
    /// Depth of brackets inside which newlines are insignificant.
    free: u32,
    diags: Vec<Diagnostic>,
    file: FileId,
    line_starts: Vec<u32>,
}

impl Parser {
    // ----- token access -------------------------------------------------

    /// Next token, skipping newlines in free mode.
    pub(crate) fn peek(&mut self) -> &Token {
        if self.free > 0 {
            while matches!(self.toks[self.pos].kind, TokKind::Newline) {
                self.pos += 1;
            }
        }
        &self.toks[self.pos]
    }

    pub(crate) fn peek_kind(&mut self) -> TokKind {
        self.peek().kind.clone()
    }

    /// Token at `pos + n`, ignoring newlines in free mode.
    pub(crate) fn peek_nth(&mut self, n: usize) -> &Token {
        self.peek();
        let mut i = self.pos;
        let mut left = n;
        while left > 0 {
            i += 1;
            if self.free > 0 {
                while matches!(self.toks[i].kind, TokKind::Newline) {
                    i += 1;
                }
            }
            left -= 1;
        }
        &self.toks[i.min(self.toks.len() - 1)]
    }

    pub(crate) fn bump(&mut self) -> Token {
        self.peek();
        let t = self.toks[self.pos].clone();
        if !matches!(t.kind, TokKind::Eof) {
            self.pos += 1;
        }
        t
    }

    pub(crate) fn at(&mut self, kind: &TokKind) -> bool {
        &self.peek().kind == kind
    }

    pub(crate) fn at_op(&mut self, op: Op) -> bool {
        self.at(&TokKind::Op(op))
    }

    pub(crate) fn at_kw(&mut self, kw: Kw) -> bool {
        self.at(&TokKind::Kw(kw))
    }

    /// Contextual keyword: an identifier with this exact text.
    pub(crate) fn at_word(&mut self, word: &str) -> bool {
        matches!(&self.peek().kind, TokKind::Ident(s) if s == word)
    }

    pub(crate) fn eat_op(&mut self, op: Op) -> Option<Token> {
        if self.at_op(op) { Some(self.bump()) } else { None }
    }

    pub(crate) fn eat_kw(&mut self, kw: Kw) -> Option<Token> {
        if self.at_kw(kw) { Some(self.bump()) } else { None }
    }

    pub(crate) fn expect_op(&mut self, op: Op) -> PResult<Token> {
        if let Some(t) = self.eat_op(op) { Ok(t) } else { Err(self.unexpected(&format!("`{}`", op.text()))) }
    }

    pub(crate) fn expect_kw(&mut self, kw: Kw) -> PResult<Token> {
        if let Some(t) = self.eat_kw(kw) { Ok(t) } else { Err(self.unexpected(&format!("`{}`", kw.text()))) }
    }

    pub(crate) fn expect_word(&mut self, word: &str) -> PResult<Token> {
        if self.at_word(word) { Ok(self.bump()) } else { Err(self.unexpected(&format!("`{word}`"))) }
    }

    pub(crate) fn expect_ident(&mut self, what: &str) -> PResult<Ident> {
        match self.peek_kind() {
            TokKind::Ident(name) => {
                let t = self.bump();
                Ok(Ident { name, span: t.span })
            }
            _ => Err(self.unexpected(what)),
        }
    }

    /// Skip newlines regardless of mode (between declarations, after `=`).
    pub(crate) fn skip_newlines(&mut self) {
        while matches!(self.toks[self.pos].kind, TokKind::Newline) {
            self.pos += 1;
        }
    }

    pub(crate) fn at_newline(&self) -> bool {
        matches!(self.toks[self.pos].kind, TokKind::Newline)
    }

    /// Run `f` with newlines insignificant.
    pub(crate) fn free<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        self.free += 1;
        let r = f(self);
        self.free -= 1;
        r
    }

    /// Run `f` in lines mode even inside brackets (module bodies, config blocks).
    pub(crate) fn lines<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        let saved = self.free;
        self.free = 0;
        let r = f(self);
        self.free = saved;
        r
    }

    pub(crate) fn prev_span(&self) -> Span {
        self.toks[self.pos.saturating_sub(1)].span
    }

    pub(crate) fn column(&self, span: Span) -> u32 {
        let line = self.line_starts.partition_point(|&s| s <= span.start).max(1) - 1;
        span.start - self.line_starts[line]
    }

    // ----- errors -------------------------------------------------------

    pub(crate) fn unexpected(&mut self, expected: &str) -> ParseError {
        let t = self.peek().clone();
        let found = t.kind.to_string();
        let span = if matches!(t.kind, TokKind::Newline | TokKind::Eof) { Span::new(self.file, t.span.start as usize, t.span.start as usize) } else { t.span };
        self.diags.push(
            Diagnostic::error("E0010", format!("expected {expected}, found {found}"))
                .label(span, format!("expected {expected} here")),
        );
        ParseError
    }

    pub(crate) fn error_at(&mut self, span: Span, message: impl Into<String>, label: impl Into<String>) -> ParseError {
        self.diags.push(Diagnostic::error("E0011", message).label(span, label));
        ParseError
    }
}
