//! Where the steps report, and how they stop.

use presk_syntax::diag::render_all;
use presk_syntax::{Diagnostic, SourceMap};

/// Where a step reports: diagnostics, rendered against the source map, and
/// plain lines (`presk: cannot read …`). The steps report in the order the
/// commands printed before 0.11.0, and a command prints its own output
/// between two steps, so the order of everything a command writes is
/// unchanged.
pub trait Report {
    /// Diagnostics of one stage, even none: the command-line tool renders
    /// the list as it always has, an empty one as nothing.
    fn diagnostics(&mut self, diags: &[Diagnostic], sm: &SourceMap);
    /// One line of text, without its newline.
    fn line(&mut self, text: &str);
    /// Text as it is, its newlines its own.
    fn text(&mut self, text: &str);
}

/// The command-line tool's report: standard error, in colour when it is a
/// terminal and `NO_COLOR` is unset (the caller decides).
pub struct Stderr {
    pub color: bool,
}

impl Report for Stderr {
    fn diagnostics(&mut self, diags: &[Diagnostic], sm: &SourceMap) {
        eprint!("{}", render_all(diags, sm, self.color));
    }

    fn line(&mut self, text: &str) {
        eprintln!("{text}");
    }

    fn text(&mut self, text: &str) {
        eprint!("{text}");
    }
}

/// One thing a step reported, kept for a document.
#[derive(Clone, Debug)]
pub enum Reported {
    /// Diagnostics, rendered without colour, and the diagnostics themselves.
    Diagnostics(String, Vec<Diagnostic>),
    Line(String),
    Text(String),
}

/// A report kept in order, for `presk explain`.
#[derive(Default)]
pub struct Collect {
    pub items: Vec<Reported>,
}

impl Report for Collect {
    fn diagnostics(&mut self, diags: &[Diagnostic], sm: &SourceMap) {
        if !diags.is_empty() {
            self.items.push(Reported::Diagnostics(render_all(diags, sm, false), diags.to_vec()));
        }
    }

    fn line(&mut self, text: &str) {
        self.items.push(Reported::Line(text.to_string()));
    }

    fn text(&mut self, text: &str) {
        self.items.push(Reported::Text(text.to_string()));
    }
}

/// Why a step stopped: the exit code its command returns, 1 when the step
/// reported diagnostics, 2 for a usage error or an internal one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stop(pub u8);
