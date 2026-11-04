//! `presk explain` (plan 0.11.0, D113): one compilation as a document. A
//! program and the parameters of one compilation in, a directory out:
//! `explain.md`, every stage's representation and the reasons for the
//! compiler's decisions, the long listings under `listings/`, and the
//! artifacts under `artifacts/` (the envelope, the realization, the plan,
//! the C program, the PyTorch module), each with its provenance.
//!
//! Everything is built over `presk-driver` from the compiler's values; no
//! command's printed text is parsed. Where a document quotes what a command
//! prints, it calls the function the command prints with
//! (`presk_driver::text`), so the two agree character for character (X3).
//! Without `--measure` the document is deterministic: no time, no absolute
//! path in its body, the compiler's version in its header only. With it,
//! §11 holds what one run on this machine measured.

pub mod doc;
mod build;
mod sha256;

use std::path::{Path, PathBuf};

use presk_core::ad::Remat;
use presk_driver::{Report, Stop};

pub use build::reach;
pub use sha256::hex as sha256;

/// A target `--target` names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// The Cranelift backend, in process.
    Native,
    /// `presk export --target c`.
    C,
    /// `presk export --target pytorch`.
    Pytorch,
}

impl Target {
    /// `native,c,pytorch`.
    pub fn parse_list(s: &str) -> Result<Vec<Target>, String> {
        s.split(',')
            .map(|t| match t.trim() {
                "native" => Ok(Target::Native),
                "c" => Ok(Target::C),
                "pytorch" => Ok(Target::Pytorch),
                other => Err(format!("--target {other}: `native`, `c` or `pytorch`, separated by commas")),
            })
            .collect()
    }
}

/// The parameters of one compilation, with the names and defaults of the
/// commands' own flags.
#[derive(Clone, Debug)]
pub struct Options {
    pub file: PathBuf,
    pub entry: String,
    /// `--dim` pairs, as given.
    pub dims: Vec<String>,
    /// `--const` pairs, as given.
    pub consts: Vec<String>,
    /// `--budget ε`: solve for a realization.
    pub budget: Option<f64>,
    /// `--menu`: what the solve offers, the integer rungs among it (plan
    /// 0.17.0 step 4, D177); the float types by default.
    pub menu: presk_precision::solve::Menu,
    /// `--realize f64|f32|bf16|@FILE`: a realization given.
    pub realize: Option<String>,
    /// `--plan @FILE`: a plan given, its realization with it.
    pub plan: Option<String>,
    /// A machine profile: the solver prices time, and the layouts are
    /// planned.
    pub profile: Option<PathBuf>,
    /// `--threads N`, the native executor's workers, which the plan's
    /// splits are counted on; the CLI sets it as every command does.
    pub threads: Option<usize>,
    pub remat: Remat,
    /// `--opt`: the e-graph optimizer.
    pub opt: bool,
    /// Trained weights in place of the initial ones.
    pub weights: Option<PathBuf>,
    pub seed: u64,
    /// The exact reference's precision.
    pub bits: u32,
    /// An envelope given in place of measuring one.
    pub envelope: Option<PathBuf>,
    /// No exact reference: no envelope measured, no gap, no conformance
    /// outputs (at a model's full size the reference takes hours).
    pub no_exact: bool,
    pub targets: Vec<Target>,
    /// `--measure`: the realization run natively against the exact
    /// reference, a call timed, each kernel's time beside its prediction,
    /// the host by phase. The section is the document's only one that
    /// changes from run to run.
    pub measure: bool,
    /// `--site S`: the sites traced through every stage; by default the
    /// three largest shares of the estimate and the three slowest predicted
    /// nests.
    pub sites: Vec<String>,
    /// `--format json`: `explain.json` beside `explain.md`, the document's
    /// facts and the provenance table as values (plan 0.13.0 step 2, D135).
    pub json: bool,
    /// `--format html`: `explain.html` too, the pipeline view, the JSON
    /// inlined (plan 0.13.0 step 3, D136); it implies `json`.
    pub html: bool,
    pub out: PathBuf,
}

/// Where an artifact came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Provenance {
    /// Computed in this run.
    Computed,
    /// Given on the command line: the path as given and its SHA-256.
    Given { path: String, sha256: String },
    /// Not computed, and why.
    Omitted(String),
}

/// One file of `artifacts/`, or one input the document names.
#[derive(Clone, Debug)]
pub struct Artifact {
    pub what: String,
    /// Relative to the output directory when written there; the path as
    /// shown when given.
    pub path: String,
    pub provenance: Provenance,
}

/// What `presk explain` wrote.
pub struct Explained {
    pub document: PathBuf,
    pub sections: usize,
    pub listings: usize,
    pub artifacts: Vec<Artifact>,
}

/// Writes the document of `o` into `o.out`. A step that cannot go on stops
/// the command, as it would stop the command it belongs to; what only one
/// section needs (a target, the sensitivities of an entry that returns no
/// scalar) is noted in that section instead.
pub fn explain(o: &Options, report: &mut dyn Report) -> presk_driver::Result<Explained> {
    if o.opt && (o.profile.is_some() || o.plan.is_some()) {
        report.line("presk explain: --opt with a layout plan: `presk plan` plans the program the optimizer has not rewritten, whose nests the rewritten one may not have; give one or the other");
        return Err(Stop(2));
    }
    // A solve types the sites of the program before the optimizer, which
    // writes each function back with new numbers (D133's finding 1; refused,
    // the user's decision of 2026-10-17, D134).
    if o.opt && o.budget.is_some() {
        report.line("presk explain: --opt with --budget: the solver types the sites of the program before the optimizer, which writes each function it optimizes back with new numbers, so the kernels would run none of its types (D133); give one or the other");
        return Err(Stop(2));
    }
    for d in [o.out.clone(), o.out.join("artifacts")] {
        if let Err(e) = std::fs::create_dir_all(&d) {
            report.line(&format!("presk explain: cannot create {}: {e}", d.display()));
            return Err(Stop(2));
        }
    }
    clear(o);
    let (document, artifacts, json) = build::build(o, report)?;
    let (text, listings) = doc::markdown(&document);
    let dir = o.out.join("listings");
    // A listing of an earlier run that this one does not write would be
    // read as this run's.
    if dir.exists()
        && let Err(e) = std::fs::remove_dir_all(&dir)
    {
        report.line(&format!("presk explain: cannot clear {}: {e}", dir.display()));
        return Err(Stop(2));
    }
    if !listings.is_empty() {
        if let Err(e) = std::fs::create_dir_all(&dir) {
            report.line(&format!("presk explain: cannot create {}: {e}", dir.display()));
            return Err(Stop(2));
        }
        for l in &listings {
            write(&dir.join(&l.name), l.text.as_bytes(), report)?;
        }
    }
    let path = o.out.join("explain.md");
    write(&path, text.as_bytes(), report)?;
    let json_path = o.out.join("explain.json");
    let html_path = o.out.join("explain.html");
    let svg_path = o.out.join("network.svg");
    match &json {
        Some((v, _)) => write(&json_path, (serde_json::to_string_pretty(v).unwrap_or_default() + "\n").as_bytes(), report)?,
        // A JSON of an earlier run would be read as this run's.
        None => {
            let _ = std::fs::remove_file(&json_path);
        }
    }
    match (&json, o.html) {
        (Some((v, _)), true) => write(&html_path, page(v).as_bytes(), report)?,
        _ => {
            let _ = std::fs::remove_file(&html_path);
        }
    }
    // The network diagram's default opening (plan 0.14.0, D141).
    match json.as_ref().and_then(|(_, n)| n.as_ref()) {
        Some(n) => write(&svg_path, presk_diagram::svg::svg(n, &presk_diagram::layout::view(n, &n.default), false).as_bytes(), report)?,
        None => {
            let _ = std::fs::remove_file(&svg_path);
        }
    }
    let sums = checksums(&o.out.join("artifacts"));
    write(&o.out.join("artifacts").join("SHA256SUMS"), sums.as_bytes(), report)?;
    Ok(Explained { document: path, sections: document.sections.len(), listings: listings.len(), artifacts })
}

/// The provenance table of the compilation `o` names (plan 0.13.0 step 1,
/// D134): every stage run as for the document, whose artifacts are written
/// under `o.out` as the document's are; the document itself is not.
pub fn provenance(o: &Options, report: &mut dyn Report) -> presk_driver::Result<presk_provenance::Table> {
    if let Err(e) = std::fs::create_dir_all(o.out.join("artifacts")) {
        report.line(&format!("presk explain: cannot create {}: {e}", o.out.display()));
        return Err(Stop(2));
    }
    build::table(o, report)
}

/// The network diagram of the compilation `o` names (plan 0.14.0, D141),
/// with what it is checked against.
pub struct Network {
    /// The diagram; `None` when the entry does not resolve.
    pub network: Option<presk_diagram::Network>,
    pub table: presk_provenance::Table,
    /// The entry's parameters, each leaf and its elements, when the kernels
    /// were listed.
    pub params: Option<Vec<(String, usize)>>,
    /// The envelope's elements per site, when there is an envelope and it
    /// names the sites of the program the kernels run: not under the
    /// optimizer, and not when differentiation kept a fusion group for the
    /// precision stages alone (D94).
    pub elems: Option<std::collections::BTreeMap<String, u64>>,
    /// Each box's share of the peak of live bytes, from 0 to 1, by its node
    /// (plan 0.15.0 step 4): what `presk dump --stage network --memory`
    /// colours the boxes by.
    pub memory: std::collections::BTreeMap<usize, f64>,
}

/// The network diagram of the compilation `o` names: every stage run as for
/// the document, whose artifacts are written under `o.out`; the document
/// itself is not.
pub fn network(o: &Options, report: &mut dyn Report) -> presk_driver::Result<Network> {
    if let Err(e) = std::fs::create_dir_all(o.out.join("artifacts")) {
        report.line(&format!("presk explain: cannot create {}: {e}", o.out.display()));
        return Err(Stop(2));
    }
    build::network(o, report)
}

/// The pipeline view's template (plan 0.13.0 step 3, D136).
const TEMPLATE: &str = include_str!("../templates/explain.html");

/// KaTeX 0.19.0, vendored (`templates/katex/README.md`): the page's
/// formulas (plan 0.13.0 step 4c, D137).
const KATEX_JS: &str = include_str!("../templates/katex/katex.min.js");
const KATEX_CSS: &str = include_str!("../templates/katex/katex.min.css");
/// The fonts the stylesheet names, by name.
const KATEX_FONTS: [(&str, &[u8]); 20] = [
    ("KaTeX_AMS-Regular", include_bytes!("../templates/katex/fonts/KaTeX_AMS-Regular.woff2")),
    ("KaTeX_Caligraphic-Bold", include_bytes!("../templates/katex/fonts/KaTeX_Caligraphic-Bold.woff2")),
    ("KaTeX_Caligraphic-Regular", include_bytes!("../templates/katex/fonts/KaTeX_Caligraphic-Regular.woff2")),
    ("KaTeX_Fraktur-Bold", include_bytes!("../templates/katex/fonts/KaTeX_Fraktur-Bold.woff2")),
    ("KaTeX_Fraktur-Regular", include_bytes!("../templates/katex/fonts/KaTeX_Fraktur-Regular.woff2")),
    ("KaTeX_Main-Bold", include_bytes!("../templates/katex/fonts/KaTeX_Main-Bold.woff2")),
    ("KaTeX_Main-BoldItalic", include_bytes!("../templates/katex/fonts/KaTeX_Main-BoldItalic.woff2")),
    ("KaTeX_Main-Italic", include_bytes!("../templates/katex/fonts/KaTeX_Main-Italic.woff2")),
    ("KaTeX_Main-Regular", include_bytes!("../templates/katex/fonts/KaTeX_Main-Regular.woff2")),
    ("KaTeX_Math-BoldItalic", include_bytes!("../templates/katex/fonts/KaTeX_Math-BoldItalic.woff2")),
    ("KaTeX_Math-Italic", include_bytes!("../templates/katex/fonts/KaTeX_Math-Italic.woff2")),
    ("KaTeX_SansSerif-Bold", include_bytes!("../templates/katex/fonts/KaTeX_SansSerif-Bold.woff2")),
    ("KaTeX_SansSerif-Italic", include_bytes!("../templates/katex/fonts/KaTeX_SansSerif-Italic.woff2")),
    ("KaTeX_SansSerif-Regular", include_bytes!("../templates/katex/fonts/KaTeX_SansSerif-Regular.woff2")),
    ("KaTeX_Script-Regular", include_bytes!("../templates/katex/fonts/KaTeX_Script-Regular.woff2")),
    ("KaTeX_Size1-Regular", include_bytes!("../templates/katex/fonts/KaTeX_Size1-Regular.woff2")),
    ("KaTeX_Size2-Regular", include_bytes!("../templates/katex/fonts/KaTeX_Size2-Regular.woff2")),
    ("KaTeX_Size3-Regular", include_bytes!("../templates/katex/fonts/KaTeX_Size3-Regular.woff2")),
    ("KaTeX_Size4-Regular", include_bytes!("../templates/katex/fonts/KaTeX_Size4-Regular.woff2")),
    ("KaTeX_Typewriter-Regular", include_bytes!("../templates/katex/fonts/KaTeX_Typewriter-Regular.woff2")),
];

/// Base64, the standard alphabet with its padding (RFC 4648).
fn base64(b: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(b.len().div_ceil(3) * 4);
    for c in b.chunks(3) {
        let n = (u32::from(c[0]) << 16) | (u32::from(*c.get(1).unwrap_or(&0)) << 8) | u32::from(*c.get(2).unwrap_or(&0));
        for k in 0..4 {
            if k <= c.len() {
                out.push(A[(n >> (18 - 6 * k) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// KaTeX's stylesheet with its fonts inlined: each `url(fonts/…woff2)` a
/// `data:` URL, the `.woff` and `.ttf` fallbacks dropped, so that the page
/// fetches nothing.
fn katex_css() -> String {
    let mut css = KATEX_CSS.to_string();
    for (name, bytes) in KATEX_FONTS {
        css = css.replace(&format!(",url(fonts/{name}.woff) format(\"woff\"),url(fonts/{name}.ttf) format(\"truetype\")"), "");
        css = css.replace(&format!("url(fonts/{name}.woff2)"), &format!("url(data:font/woff2;base64,{})", base64(bytes)));
    }
    css
}

/// The pipeline view of a document's JSON: the template with KaTeX and the
/// JSON inlined. A `</` in a string would end the data element early, so it
/// is written `<\/`, and a `<!--` would open a comment, so it is written
/// `\u003c!--`: JSON reads both as the same string.
pub fn page(json: &serde_json::Value) -> String {
    let data = serde_json::to_string(json).unwrap_or_else(|_| "null".into()).replace("</", "<\\/").replace("<!--", "\\u003c!--");
    TEMPLATE.replacen("/*KATEX_CSS*/", &katex_css(), 1).replacen("/*KATEX_JS*/", KATEX_JS, 1).replacen("/*PRESK_DATA*/null", &data, 1)
}


/// Removes what an earlier run wrote under `artifacts/` and this one may
/// not write again, but for the files this run is given.
fn clear(o: &Options) {
    let given: Vec<PathBuf> = [o.envelope.clone(), o.realize.as_deref().and_then(|r| r.strip_prefix('@')).map(PathBuf::from), o.plan.as_deref().map(|p| PathBuf::from(p.trim_start_matches('@')))].into_iter().flatten().filter_map(|p| std::fs::canonicalize(p).ok()).collect();
    let dir = o.out.join("artifacts");
    for name in ["envelope.json", "realization.json", "plan.json", "SHA256SUMS"] {
        let p = dir.join(name);
        if std::fs::canonicalize(&p).is_ok_and(|c| !given.contains(&c)) {
            let _ = std::fs::remove_file(&p);
        }
    }
    for name in ["c", "pytorch"] {
        let _ = std::fs::remove_dir_all(dir.join(name));
    }
}

/// `sha256sum`'s lines for every file under `dir` but this one, by path.
fn checksums(dir: &Path) -> String {
    fn walk(dir: &Path, base: &Path, out: &mut Vec<(String, String)>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, base, out);
            } else if let (Ok(bytes), Ok(rel)) = (std::fs::read(&p), p.strip_prefix(base)) {
                let rel = rel.display().to_string();
                if rel != "SHA256SUMS" {
                    out.push((rel, crate::sha256(&bytes)));
                }
            }
        }
    }
    let mut files = Vec::new();
    walk(dir, dir, &mut files);
    files.sort();
    files.iter().map(|(p, h)| format!("{h}  {p}\n")).collect()
}

fn write(path: &Path, bytes: &[u8], report: &mut dyn Report) -> presk_driver::Result<()> {
    std::fs::write(path, bytes).map_err(|e| {
        report.line(&format!("presk explain: cannot write {}: {e}", path.display()));
        Stop(2)
    })
}

/// A path as a document prints it: a relative one as given, which is
/// relative to the working directory the commands run from; an absolute one
/// relative to that directory when it is under it, else with the home
/// directory as `~`. So a document holds no absolute path of the machine it
/// was made on, but for a file outside both.
pub fn shown(path: &Path) -> String {
    if path.is_relative() {
        let s = path.display().to_string();
        return s.strip_prefix("./").map(String::from).unwrap_or(s);
    }
    if let Ok(cwd) = std::env::current_dir()
        && let Ok(rel) = path.strip_prefix(&cwd)
    {
        return if rel.as_os_str().is_empty() { ".".into() } else { rel.display().to_string() };
    }
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from)
        && let Ok(rel) = path.strip_prefix(&home)
    {
        return format!("~/{}", rel.display());
    }
    path.display().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_is_rfc_4648() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn the_page_fetches_no_font() {
        let css = katex_css();
        assert!(!css.contains("url(fonts/"), "a font left to fetch");
        assert_eq!(css.matches("data:font/woff2;base64,").count(), KATEX_FONTS.len());
    }
}
