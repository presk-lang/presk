//! Diagnostics: a renderer-independent description of an error or warning,
//! rendered with ariadne. Messages are phrased in the terms of the source.
//!
//! Error codes: `E0xxx` syntax, `E1xxx` elaboration, `E2xxx` typing, `E3xxx`
//! regularity, `E4xxx` effects, `W0xxx` warnings.

use std::fmt;

use ariadne::{Color, Config, IndexType, Label as ALabel, Report, ReportKind};

use crate::source::{FileId, SourceMap, Span};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Warning,
    Error,
}

#[derive(Clone, Debug)]
pub struct Label {
    pub span: Span,
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct Diagnostic {
    pub code: &'static str,
    pub severity: Severity,
    pub message: String,
    /// The first label is the primary one.
    pub labels: Vec<Label>,
    pub notes: Vec<String>,
    pub help: Option<String>,
}

impl Diagnostic {
    pub fn error(code: &'static str, message: impl Into<String>) -> Self {
        Diagnostic { code, severity: Severity::Error, message: message.into(), labels: vec![], notes: vec![], help: None }
    }

    pub fn warning(code: &'static str, message: impl Into<String>) -> Self {
        Diagnostic { code, severity: Severity::Warning, message: message.into(), labels: vec![], notes: vec![], help: None }
    }

    pub fn label(mut self, span: Span, text: impl Into<String>) -> Self {
        self.labels.push(Label { span, text: text.into() });
        self
    }

    pub fn note(mut self, text: impl Into<String>) -> Self {
        self.notes.push(text.into());
        self
    }

    pub fn help(mut self, text: impl Into<String>) -> Self {
        self.help = Some(text.into());
        self
    }

    pub fn primary_span(&self) -> Option<Span> {
        self.labels.first().map(|l| l.span)
    }

    pub fn is_error(&self) -> bool {
        self.severity == Severity::Error
    }

    /// Render with ariadne. `color` selects ANSI colors; tests use `false`,
    /// which also selects the ASCII character set for stable snapshots.
    pub fn render(&self, sm: &SourceMap, color: bool) -> String {
        let kind = match self.severity {
            Severity::Error => ReportKind::Error,
            Severity::Warning => ReportKind::Warning,
        };
        let primary = self.labels.first().map(|l| l.span);
        let anchor = primary.map(|s| (s.file, s.range())).unwrap_or((FileId(0), 0..0));
        let config = Config::default()
            .with_color(color)
            .with_index_type(IndexType::Byte)
            .with_char_set(if color { ariadne::CharSet::Unicode } else { ariadne::CharSet::Ascii });
        let mut report = Report::build(kind, anchor).with_code(self.code).with_message(&self.message).with_config(config);
        for (i, l) in self.labels.iter().enumerate() {
            let mut label = ALabel::new((l.span.file, l.span.range())).with_message(&l.text).with_order(i as i32);
            if color {
                label = label.with_color(if i == 0 { Color::Red } else { Color::Blue });
            }
            report = report.with_label(label);
        }
        for n in &self.notes {
            report = report.with_note(n);
        }
        if let Some(h) = &self.help {
            report = report.with_help(h);
        }
        let mut out = Vec::new();
        report.finish().write(Cache(sm), &mut out).expect("rendering a diagnostic");
        String::from_utf8(out).expect("ariadne emits UTF-8")
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

struct Cache<'a>(&'a SourceMap);

impl ariadne::Cache<FileId> for Cache<'_> {
    type Storage = String;

    fn fetch(&mut self, id: &FileId) -> Result<&ariadne::Source<String>, impl fmt::Debug> {
        Ok::<_, std::convert::Infallible>(&self.0.file(*id).ariadne)
    }

    fn display<'b>(&self, id: &'b FileId) -> Option<impl fmt::Display + 'b> {
        Some(self.0.file(*id).name.clone())
    }
}

/// Render a list of diagnostics, one after the other.
pub fn render_all(diags: &[Diagnostic], sm: &SourceMap, color: bool) -> String {
    diags.iter().map(|d| d.render(sm, color)).collect::<Vec<_>>().join("")
}

pub fn has_errors(diags: &[Diagnostic]) -> bool {
    diags.iter().any(Diagnostic::is_error)
}
