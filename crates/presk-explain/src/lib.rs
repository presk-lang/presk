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
    for d in [o.out.clone(), o.out.join("artifacts")] {
        if let Err(e) = std::fs::create_dir_all(&d) {
            report.line(&format!("presk explain: cannot create {}: {e}", d.display()));
            return Err(Stop(2));
        }
    }
    clear(o);
    let (document, artifacts) = build::build(o, report)?;
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
    let sums = checksums(&o.out.join("artifacts"));
    write(&o.out.join("artifacts").join("SHA256SUMS"), sums.as_bytes(), report)?;
    Ok(Explained { document: path, sections: document.sections.len(), listings: listings.len(), artifacts })
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
