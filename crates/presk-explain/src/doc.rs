//! The document model of `presk explain` and its Markdown renderer.
//!
//! A document is a title, a header (the parameters and the artifacts) and
//! numbered sections. A section says what its stage does, then shows the
//! stage's representation and its decisions as prose, tables, code and
//! notes, then the commands that print the same thing. Nothing here knows
//! the compiler: the sections are filled by the private module `build` from the
//! driver's values. Markdown is the one renderer; the PDF is made from it
//! by pandoc (`make explain-pdf`), and an HTML renderer would read the same
//! model.

use std::fmt::Write as _;

/// What a figure is: the language's discipline of labels (D35), which
/// every figure of a document carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A fact of the program or of the compilation: a count, a shape.
    Fact,
    /// What the compiler chose: a type, a group, a layout.
    Decision,
    /// What the command was given.
    Parameter,
    /// A sound bound, conditional on the envelope it names.
    Bound,
    /// An estimate: it ranks, it does not guarantee.
    Estimate,
    /// A measurement: on the exact reference (deterministic), or a time
    /// (on a named machine, in one run).
    Measurement,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Fact => "fact",
            Kind::Decision => "decision",
            Kind::Parameter => "parameter",
            Kind::Bound => "bound",
            Kind::Estimate => "estimate",
            Kind::Measurement => "measurement",
        }
    }
}

/// One block of a section.
#[derive(Clone, Debug)]
pub enum Block {
    /// A paragraph, Markdown inline syntax allowed.
    Prose(String),
    /// A table: its heading and its rows, each cell Markdown inline syntax.
    Table { head: Vec<String>, rows: Vec<Vec<String>> },
    /// Code as it is printed, in a fenced block of `lang`. Longer than
    /// [`LONG`] lines, it is a listing: its first [`EXCERPT`] lines in the
    /// body, the whole in `listings/<name>`.
    Code { lang: String, text: String, name: Option<String> },
    /// What the reader must not miss: a quoted block, its first words bold.
    Note { title: String, text: String },
    /// A file of `listings/` the body points to without showing it.
    Attach { name: String, text: String },
}

/// A code block longer than this goes to `listings/`.
pub const LONG: usize = 160;
/// The lines of a long listing kept in the body.
pub const EXCERPT: usize = 60;

/// A numbered section.
#[derive(Clone, Debug, Default)]
pub struct Section {
    /// `6.3`; empty for an unnumbered one.
    pub number: String,
    pub title: String,
    /// What the stage does, in two or three sentences: the internals book's
    /// "Purpose".
    pub purpose: Vec<String>,
    pub blocks: Vec<Block>,
    /// The commands that print what the section shows.
    pub commands: Vec<String>,
    pub sections: Vec<Section>,
}

impl Section {
    pub fn new(number: &str, title: &str) -> Section {
        Section { number: number.into(), title: title.into(), ..Default::default() }
    }

    pub fn purpose(mut self, text: &str) -> Section {
        self.purpose.push(text.into());
        self
    }

    pub fn prose(&mut self, text: impl Into<String>) {
        self.blocks.push(Block::Prose(text.into()));
    }

    pub fn table(&mut self, head: &[&str], rows: Vec<Vec<String>>) {
        self.blocks.push(Block::Table { head: head.iter().map(|h| h.to_string()).collect(), rows });
    }

    pub fn code(&mut self, lang: &str, text: impl Into<String>) {
        self.blocks.push(Block::Code { lang: lang.into(), text: text.into(), name: None });
    }

    /// Code that may be long: past [`LONG`] lines it goes whole to
    /// `listings/<name>`.
    pub fn listing(&mut self, lang: &str, name: &str, text: impl Into<String>) {
        self.blocks.push(Block::Code { lang: lang.into(), text: text.into(), name: Some(name.into()) });
    }

    pub fn note(&mut self, title: &str, text: impl Into<String>) {
        self.blocks.push(Block::Note { title: title.into(), text: text.into() });
    }

    pub fn command(&mut self, c: impl Into<String>) {
        self.commands.push(c.into());
    }
}

/// A whole document.
#[derive(Clone, Debug, Default)]
pub struct Document {
    pub title: String,
    pub subtitle: String,
    /// The header: the parameters, the compiler, the artifacts.
    pub header: Vec<Block>,
    pub sections: Vec<Section>,
}

/// A file the renderer writes beside the document: `listings/<name>`.
pub struct Listing {
    pub name: String,
    pub text: String,
}

/// The document as Markdown, and the listings it refers to.
pub fn markdown(doc: &Document) -> (String, Vec<Listing>) {
    let mut out = String::new();
    let mut listings = Vec::new();
    // pandoc's metadata block; GitLab shows it as a table.
    let _ = writeln!(out, "---\ntitle: \"{}\"\nsubtitle: \"{}\"\n---\n", yaml(&doc.title), yaml(&doc.subtitle));
    for b in &doc.header {
        block(&mut out, b, &mut listings);
    }
    for s in &doc.sections {
        section(&mut out, s, 2, &mut listings);
    }
    (out, listings)
}

fn yaml(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn section(out: &mut String, s: &Section, level: usize, listings: &mut Vec<Listing>) {
    let hashes = "#".repeat(level);
    if s.number.is_empty() {
        let _ = writeln!(out, "{hashes} {}\n", s.title);
    } else {
        let _ = writeln!(out, "{hashes} {} {}\n", s.number, s.title);
    }
    for p in &s.purpose {
        let _ = writeln!(out, "{p}\n");
    }
    for b in &s.blocks {
        block(out, b, listings);
    }
    match s.commands.as_slice() {
        [] => {}
        [c] => {
            let _ = writeln!(out, "*Command:* `{c}`\n");
        }
        cs => {
            let _ = writeln!(out, "*Commands:*\n");
            for c in cs {
                let _ = writeln!(out, "- `{c}`");
            }
            out.push('\n');
        }
    }
    for sub in &s.sections {
        section(out, sub, level + 1, listings);
    }
}

fn block(out: &mut String, b: &Block, listings: &mut Vec<Listing>) {
    match b {
        Block::Prose(p) => {
            let _ = writeln!(out, "{p}\n");
        }
        Block::Table { head, rows } => {
            let _ = writeln!(out, "| {} |", head.iter().map(|h| cell(h)).collect::<Vec<_>>().join(" | "));
            // The dashes give each column its share of the width when pandoc
            // wraps a wide table: the longest cell's length, within bounds, a
            // code cell's counted wider since its font is.
            let len = |c: &String| {
                let n = c.chars().count();
                if c.starts_with('`') { n * 7 / 5 } else { n }
            };
            // A code cell does not wrap: its column may take more.
            let widths: Vec<usize> = (0..head.len())
                .map(|i| {
                    let longest = std::iter::once(&head[i]).chain(rows.iter().filter_map(|r| r.get(i))).max_by_key(|c| len(c)).expect("a heading");
                    len(longest).clamp(10, if longest.starts_with('`') { 36 } else { 24 })
                })
                .collect();
            let _ = writeln!(out, "|{}", widths.iter().map(|w| format!("{}|", "-".repeat(*w))).collect::<String>());
            for r in rows {
                let _ = writeln!(out, "| {} |", r.iter().map(|c| cell(c)).collect::<Vec<_>>().join(" | "));
            }
            out.push('\n');
        }
        Block::Code { lang, text, name } => {
            let text = text.strip_suffix('\n').unwrap_or(text);
            let lines: Vec<&str> = text.lines().collect();
            let fence = if text.contains("```") { "````" } else { "```" };
            match name {
                Some(n) if lines.len() > LONG => {
                    let _ = writeln!(out, "{fence}{lang}\n{}\n{fence}\n", lines[..EXCERPT].join("\n"));
                    let _ = writeln!(out, "*The first {EXCERPT} of {} lines; the whole in `listings/{n}`.*\n", lines.len());
                    listings.push(Listing { name: n.clone(), text: format!("{text}\n") });
                }
                _ => {
                    let _ = writeln!(out, "{fence}{lang}\n{text}\n{fence}\n");
                }
            }
        }
        Block::Attach { name, text } => listings.push(Listing { name: name.clone(), text: format!("{}\n", text.strip_suffix('\n').unwrap_or(text)) }),
        Block::Note { title, text } => {
            let mut first = true;
            for line in text.lines() {
                if first {
                    let _ = writeln!(out, "> **{title}** {line}");
                    first = false;
                } else {
                    let _ = writeln!(out, "> {line}");
                }
            }
            out.push('\n');
        }
    }
}

/// A table cell: a `|` would end it, a newline the row.
fn cell(c: &str) -> String {
    c.replace('|', "\\|").replace('\n', " ")
}
