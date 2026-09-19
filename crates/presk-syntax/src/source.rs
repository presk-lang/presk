//! Source files, byte spans and line/column lookup.

use std::fmt;
use std::ops::Range;

/// Index of a file in a [`SourceMap`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FileId(pub u32);

impl fmt::Debug for FileId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "f{}", self.0)
    }
}

/// A byte range in one source file. Offsets index the original text (not
/// normalized); only identifier *names* are NFC-normalized by the lexer.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    pub file: FileId,
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub fn new(file: FileId, start: usize, end: usize) -> Self {
        Span { file, start: start as u32, end: end as u32 }
    }

    /// The smallest span covering `self` and `other` (same file).
    pub fn to(self, other: Span) -> Span {
        debug_assert_eq!(self.file, other.file);
        Span { file: self.file, start: self.start.min(other.start), end: self.end.max(other.end) }
    }

    pub fn range(self) -> Range<usize> {
        self.start as usize..self.end as usize
    }

    pub fn len(self) -> usize {
        (self.end - self.start) as usize
    }

    pub fn is_empty(self) -> bool {
        self.start == self.end
    }
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

pub struct SourceFile {
    pub name: String,
    pub text: String,
    line_starts: Vec<u32>,
    pub(crate) ariadne: ariadne::Source<String>,
}

impl SourceFile {
    fn new(name: String, text: String) -> Self {
        let mut line_starts = vec![0];
        for (i, b) in text.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push(i as u32 + 1);
            }
        }
        let ariadne = ariadne::Source::from(text.clone());
        SourceFile { name, text, line_starts, ariadne }
    }

    /// 1-based line and column (column in characters) of a byte offset.
    pub fn line_col(&self, offset: u32) -> (usize, usize) {
        let line = self.line_starts.partition_point(|&s| s <= offset).max(1) - 1;
        let start = self.line_starts[line] as usize;
        let col = self.text[start..offset as usize].chars().count();
        (line + 1, col + 1)
    }

    pub fn slice(&self, span: Span) -> &str {
        &self.text[span.range()]
    }
}

/// All the files of one compilation.
#[derive(Default)]
pub struct SourceMap {
    files: Vec<SourceFile>,
}

impl SourceMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, name: impl Into<String>, text: impl Into<String>) -> FileId {
        let id = FileId(self.files.len() as u32);
        self.files.push(SourceFile::new(name.into(), text.into()));
        id
    }

    /// Read a file from disk and add it; the name is the path as given.
    pub fn load(&mut self, path: &std::path::Path) -> std::io::Result<FileId> {
        let text = std::fs::read_to_string(path)?;
        Ok(self.add(path.display().to_string(), text))
    }

    pub fn file(&self, id: FileId) -> &SourceFile {
        &self.files[id.0 as usize]
    }

    pub fn text(&self, id: FileId) -> &str {
        &self.file(id).text
    }

    pub fn slice(&self, span: Span) -> &str {
        self.file(span.file).slice(span)
    }

    /// `name:line:col` of the start of a span.
    pub fn location(&self, span: Span) -> String {
        let f = self.file(span.file);
        let (l, c) = f.line_col(span.start);
        format!("{}:{}:{}", f.name, l, c)
    }

    pub fn files(&self) -> impl Iterator<Item = (FileId, &SourceFile)> {
        self.files.iter().enumerate().map(|(i, f)| (FileId(i as u32), f))
    }
}
