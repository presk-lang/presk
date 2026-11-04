//! Hand-written lexer.
//!
//! Identifiers follow Unicode XID (so `θ`, `γ`, `ŷ` and the decomposed `x̂`
//! all lex as identifiers) except that `·` (U+00B7, an XID_Continue
//! character) is always the contraction operator. Names are NFC-normalized;
//! spans index the original text. `--` starts a comment. Runs of line breaks
//! produce one `Newline` token; the parser decides where they matter.
//!
//! Every non-ASCII symbol has an ASCII spelling lexed to the same token:
//! `ℝ`/`Real`, `λ`/`\`, `·`/`@`, `⊗`/`><`, `∘`/`<<`.

use unicode_normalization::UnicodeNormalization;

use crate::diag::Diagnostic;
use crate::source::{FileId, Span};
use crate::token::{Kw, Op, TokKind, Token};

pub struct Lexed {
    pub tokens: Vec<Token>,
    pub diags: Vec<Diagnostic>,
}

pub fn lex(file: FileId, text: &str) -> Lexed {
    Lexer { file, text, pos: 0, tokens: vec![], diags: vec![] }.run()
}

struct Lexer<'a> {
    file: FileId,
    text: &'a str,
    pos: usize,
    tokens: Vec<Token>,
    diags: Vec<Diagnostic>,
}

fn is_ident_start(c: char) -> bool {
    c == '_' || (unicode_ident::is_xid_start(c) && c != 'λ' && c != 'ℝ')
}

fn is_ident_continue(c: char) -> bool {
    c != '·' && unicode_ident::is_xid_continue(c)
}

impl<'a> Lexer<'a> {
    fn peek(&self) -> Option<char> {
        self.text[self.pos..].chars().next()
    }

    fn peek2(&self) -> Option<char> {
        let mut it = self.text[self.pos..].chars();
        it.next();
        it.next()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn push(&mut self, kind: TokKind, start: usize) {
        self.tokens.push(Token { kind, span: Span::new(self.file, start, self.pos) });
    }

    fn run(mut self) -> Lexed {
        while let Some(c) = self.peek() {
            let start = self.pos;
            match c {
                '\n' => {
                    self.bump();
                    // A single Newline token per run of blank lines, comments included.
                    if !matches!(self.tokens.last(), Some(Token { kind: TokKind::Newline, .. })) {
                        self.push(TokKind::Newline, start);
                    }
                }
                c if c.is_whitespace() => {
                    self.bump();
                }
                '-' if self.peek2() == Some('-') => {
                    while let Some(c) = self.peek() {
                        if c == '\n' {
                            break;
                        }
                        self.bump();
                    }
                }
                '"' => self.string(start),
                c if c.is_ascii_digit() => self.number(start),
                'λ' => {
                    self.bump();
                    self.push(TokKind::Kw(Kw::Lambda), start);
                }
                'ℝ' => {
                    self.bump();
                    self.push(TokKind::Kw(Kw::Real), start);
                }
                '\\' => {
                    self.bump();
                    self.push(TokKind::Kw(Kw::Lambda), start);
                }
                c if is_ident_start(c) => self.ident(start),
                _ => self.operator(start),
            }
        }
        // Every file ends with a Newline before Eof, which simplifies the parser.
        if !matches!(self.tokens.last(), Some(Token { kind: TokKind::Newline, .. }) | None) {
            let p = self.pos;
            self.push(TokKind::Newline, p);
        }
        let p = self.pos;
        self.push(TokKind::Eof, p);
        Lexed { tokens: self.tokens, diags: self.diags }
    }

    fn ident(&mut self, start: usize) {
        while let Some(c) = self.peek() {
            if is_ident_continue(c) {
                self.bump();
            } else {
                break;
            }
        }
        let raw = &self.text[start..self.pos];
        let name: String = raw.nfc().collect();
        let kind = match Kw::from_ident(&name) {
            Some(kw) => TokKind::Kw(kw),
            None => TokKind::Ident(name),
        };
        self.push(kind, start);
    }

    fn number(&mut self, start: usize) {
        let mut is_float = false;
        self.digits();
        if self.peek() == Some('.') && self.peek2().is_some_and(|c| c.is_ascii_digit()) {
            is_float = true;
            self.bump();
            self.digits();
        }
        if matches!(self.peek(), Some('e' | 'E')) {
            let save = self.pos;
            self.bump();
            if matches!(self.peek(), Some('+' | '-')) {
                self.bump();
            }
            if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                is_float = true;
                self.digits();
            } else {
                self.pos = save; // `e` belongs to a unit, e.g. `3e` is not an exponent without digits
            }
        }
        let num = self.text[start..self.pos].to_string();
        // A unit glued to the number: `5ms`, `3MiB`, `204GB/s`, `192KiB/sm`.
        if self.peek().is_some_and(|c| c.is_alphabetic()) {
            let ustart = self.pos;
            while self.peek().is_some_and(|c| c.is_alphanumeric()) {
                self.bump();
            }
            if self.peek() == Some('/') && self.peek2().is_some_and(|c| c.is_alphabetic()) {
                self.bump();
                while self.peek().is_some_and(|c| c.is_alphanumeric()) {
                    self.bump();
                }
            }
            let unit = self.text[ustart..self.pos].to_string();
            self.push(TokKind::Quantity { num, unit }, start);
        } else if is_float {
            self.push(TokKind::Float(num), start);
        } else {
            self.push(TokKind::Nat(num), start);
        }
    }

    fn digits(&mut self) {
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.bump();
        }
    }

    fn string(&mut self, start: usize) {
        self.bump(); // opening quote
        let mut s = String::new();
        loop {
            match self.bump() {
                Some('"') => break,
                Some('\\') => match self.bump() {
                    Some('n') => s.push('\n'),
                    Some('t') => s.push('\t'),
                    Some(c) => s.push(c),
                    None => break,
                },
                Some('\n') | None => {
                    self.diags.push(
                        Diagnostic::error("E0001", "unterminated string literal")
                            .label(Span::new(self.file, start, self.pos), "string starts here and never closes"),
                    );
                    break;
                }
                Some(c) => s.push(c),
            }
        }
        self.push(TokKind::Str(s), start);
    }

    fn operator(&mut self, start: usize) {
        let c = self.bump().expect("caller checked a char is available");
        let two = self.peek();
        let op = match (c, two) {
            ('.', Some('.')) if self.text[self.pos..].starts_with("..") => {
                self.bump();
                self.bump();
                Op::Ellipsis
            }
            ('.', Some('.')) => {
                self.bump();
                Op::DotDot
            }
            ('=', Some('=')) => {
                self.bump();
                Op::EqEq
            }
            ('<', Some('=')) => {
                self.bump();
                Op::Le
            }
            ('<', Some('<')) => {
                self.bump();
                Op::Compose
            }
            ('>', Some('<')) => {
                self.bump();
                Op::Tensor
            }
            ('>', Some('=')) => {
                self.bump();
                Op::Ge
            }
            ('-', Some('>')) => {
                self.bump();
                Op::Arrow
            }
            ('|', Some('>')) => {
                self.bump();
                Op::Pipe
            }
            ('·', _) | ('@', _) => Op::Dot,
            ('*', _) => Op::Star,
            ('/', _) => Op::Slash,
            ('%', _) => Op::Percent,
            ('+', _) => Op::Plus,
            ('-', _) => Op::Minus,
            ('^', _) => Op::Caret,
            ('<', _) => Op::Lt,
            ('>', _) => Op::Gt,
            ('=', _) => Op::Eq,
            ('|', _) => Op::Bar,
            ('⊗', _) => Op::Tensor,
            ('∘', _) => Op::Compose,
            ('.', _) => Op::Period,
            (',', _) => Op::Comma,
            (':', _) => Op::Colon,
            ('(', _) => Op::LParen,
            (')', _) => Op::RParen,
            ('[', _) => Op::LBracket,
            (']', _) => Op::RBracket,
            ('{', _) => Op::LBrace,
            ('}', _) => Op::RBrace,
            _ => {
                self.diags.push(
                    Diagnostic::error("E0002", format!("unexpected character `{c}` (U+{:04X})", c as u32))
                        .label(Span::new(self.file, start, self.pos), "not part of the Presk syntax"),
                );
                return;
            }
        };
        self.push(TokKind::Op(op), start);
    }
}

/// One line per token: `kind @ start..end`, for `presk dump --stage tokens`.
pub fn dump_tokens(tokens: &[Token]) -> String {
    let mut out = String::new();
    for t in tokens {
        out.push_str(&format!("{:?} @ {:?}\n", t.kind, t.span));
    }
    out
}
