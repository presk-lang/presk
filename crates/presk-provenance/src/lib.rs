//! The provenance table (plan 0.13.0 step 1, D134): every element of every
//! stage of one compilation, and for each element a stage made, the elements
//! of the stage before it that it was made from, and how.
//!
//! It is the IRs' DWARF. The IRs do not change: a stage returns its links
//! beside its value, as the optimizer returns its report, and
//! `presk-driver` puts them together. Three things, as DWARF has them:
//!
//! - **an identity per element at every stage** ([`Id`]): a site
//!   `<function>/%<n>` in the three programs, a nest by its root's site, a
//!   kernel by its place in the listing, a span in the source;
//! - **a link per derived element** ([`Link`]): the child, its parents in
//!   the stage before (or a stage further up, when a stage did nothing to
//!   it), and the reason, [`How`], a closed list per stage boundary;
//! - **the source's line table**: every element of the elaborated program
//!   links to its span, and every later element to those through its
//!   links.
//!
//! An element of a stage that has no child at the next is not lost: an
//! [`End`] says why (the optimizer removed it, the host computes it).
//!
//! The crate has no dependency: the stages below it name their elements by
//! the strings the documents print.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

/// A stage of one compilation, in the order the compiler runs them. The
/// precision stages read the program before the optimizer (the user's
/// decision of 2026-10-17).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stage {
    /// The program's text: one element per span an elaborated element has.
    Source,
    /// The elaborated, typed program: its functions, module instances and
    /// the typed expressions the lowering read.
    Elab,
    /// PSK-Core as lowered: functions and `let`s.
    Core,
    /// PSK-Core after differentiation: the program the kernels run, before
    /// the optimizer.
    Ad,
    /// What the precision stages name: a site, with the solver's role, or a
    /// parameter by its path.
    Precision,
    /// PSK-Core after the e-graph optimizer (`--opt`).
    Opt,
    /// The fusion groups: one nest per group, or per primitive alone.
    Fusion,
    /// The layout plan: one per nest the profile planned.
    Layout,
    /// The kernels of the listing, by their place in it.
    Kernel,
    /// The host's launches: a nest at a kernel, so many times a call.
    Launch,
    /// The C program: a kernel's function in `kernels.c`.
    C,
    /// The native code.
    Native,
}

impl Stage {
    pub const ALL: [Stage; 12] = [Stage::Source, Stage::Elab, Stage::Core, Stage::Ad, Stage::Precision, Stage::Opt, Stage::Fusion, Stage::Layout, Stage::Kernel, Stage::Launch, Stage::C, Stage::Native];

    /// The stage's name in a key and in a document.
    pub fn name(self) -> &'static str {
        match self {
            Stage::Source => "source",
            Stage::Elab => "elab",
            Stage::Core => "core",
            Stage::Ad => "ad",
            Stage::Precision => "precision",
            Stage::Opt => "opt",
            Stage::Fusion => "fusion",
            Stage::Layout => "layout",
            Stage::Kernel => "kernel",
            Stage::Launch => "launch",
            Stage::C => "c",
            Stage::Native => "native",
        }
    }
}

/// An element: its stage and its key in it.
///
/// | Stage | Key |
/// |---|---|
/// | source | `<file>:<start>-<end>`, byte offsets |
/// | elab | `fn <name>`, `inst <module{dims}>`, `run <name>`, `expr <file>:<start>-<end>` |
/// | core, ad, opt | `<function>` for a function, `<function>/%<n>` for a `let` |
/// | precision | `<function>/%<n>` for a site, `param <path>` for a parameter |
/// | fusion, layout | the nest, its root's `<function>/%<n>` |
/// | kernel | `k<j>`, its place in the listing |
/// | launch | `<nest> k<j>` |
/// | c | `kernels.c k<n>` |
/// | native | `k<j> <start>-<end>`, byte offsets in the kernel's code |
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Id {
    pub stage: Stage,
    pub key: String,
}

impl Id {
    pub fn new(stage: Stage, key: impl Into<String>) -> Id {
        Id { stage, key: key.into() }
    }
}

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.stage.name(), self.key)
    }
}

/// Why an element was made from its parents: a closed list per stage
/// boundary, the reasons a reader of the compiler is taught.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum How {
    // ----- source → elaborated program ---------------------------------------
    /// A function, a module or a training run defined at the span.
    Defined,
    /// A module instance: the module defined at the span, its dimensions
    /// bound.
    Instantiated,
    /// The typed expression at the span.
    Typed,
    // ----- elaborated program → PSK-Core -------------------------------------
    /// A Core function lowered from its source function, its module's
    /// parameters (`init`), its training run's optimizer, or lifted from a
    /// lambda of the function it was written in.
    LoweredFn { kind: String },
    /// A `let` the lowering made from the typed expression, by the rule of
    /// the expression's kind: `call`, `ein`, `index`, a builtin's name …
    Lowered { rule: String },
    // ----- PSK-Core → differentiated, and to the optimized program -----------
    /// The same `let` under the same number: a function the stage did not
    /// touch. A pass-through, never a copy.
    Unchanged,
    /// `grad v` rewritten to a closure of the `.grad` differentiation
    /// generated.
    Gradient,
    /// A function differentiation generated from the parent function:
    /// `vjp`, `grad`, `fwd`, `bwd`, `value_vjp`, `fold_bwd` …
    Generated { kind: String },
    /// The forward pass's copy of the primal `let` in a generated function:
    /// `kept` in a `.vjp`, `stored` in a `.fwd` for the sweep to read,
    /// `recomputed` for a value, `copied` apart for a residual.
    Primal { mode: String },
    /// A `let` of the reverse sweep, made for the primal by the derivative
    /// of its operation, `rule`: a primitive's name, `call` (the callee's
    /// `.vjp`), `fold`, `tuple`, `field` …
    Adjoint { rule: String },
    /// The sum of two contributions to the primal's adjoint.
    Accumulate,
    /// The adjoint of a parameter nothing flowed to: zeros.
    Zero,
    /// The cotangent 1 a `.grad` seeds its sweep with.
    Seed,
    /// What computes nothing in a generated function: its result's tuple,
    /// the residuals' projections and tuples, a call of a `.fwd` or a
    /// `.bwd`. The parents are the function it serves.
    Plumbing,
    // ----- differentiated → precision ----------------------------------------
    /// A site the precision stages name, with the solver's role: `decision`,
    /// `follows`, `carried`, or `analysed` when no solve ran.
    Site { role: String },
    /// A parameter the realization or the solver types by its path.
    Parameter,
    // ----- differentiated → optimized ----------------------------------------
    /// A `let` the extraction wrote for an e-class: the parents are the
    /// `let`s of that class, which it replaces; `rules` the rewrites that
    /// applied in the function.
    Rewritten { rules: Vec<String> },
    /// A `let` a rewrite introduced: the parents are those of the rewritten
    /// `let` that reads it.
    Introduced,
    // ----- → fusion ----------------------------------------------------------
    /// A fusion group, one loop nest: its root first among the parents, then
    /// its members; `stop`, why the group ends at its root.
    Fused { members: usize, stop: String },
    /// A primitive launched alone: why it is not in its reader's group.
    Alone { stop: String },
    // ----- → layout ----------------------------------------------------------
    /// The layout point the plan chose for the nest, of `points`.
    LaidOut { point: String, points: usize },
    // ----- → kernels and launches --------------------------------------------
    /// The kernel the nests launch: `shared_by` of them.
    Kernel { shared_by: usize },
    /// The nest launched at the kernel, `count` times a call: the nest and
    /// the kernel are the parents.
    Launched { count: usize },
    /// An `add` a contraction's kernel computes as its epilogue.
    Epilogue,
    // ----- → C ---------------------------------------------------------------
    /// The kernel's function in `kernels.c`, lines `first..=last`.
    Function { first: usize, last: usize },
    // ----- → native ----------------------------------------------------------
    /// The machine code the kernel compiled to, each byte with the line of
    /// the kernel it compiles (plan 0.13.0 step 5).
    Code,
}

impl How {
    /// The reason's name, the variant's.
    pub fn name(&self) -> &'static str {
        match self {
            How::Defined => "defined",
            How::Instantiated => "instantiated",
            How::Typed => "typed",
            How::LoweredFn { .. } => "lowered_fn",
            How::Lowered { .. } => "lowered",
            How::Unchanged => "unchanged",
            How::Gradient => "gradient",
            How::Generated { .. } => "generated",
            How::Primal { .. } => "primal",
            How::Adjoint { .. } => "adjoint",
            How::Accumulate => "accumulate",
            How::Zero => "zero",
            How::Seed => "seed",
            How::Plumbing => "plumbing",
            How::Site { .. } => "site",
            How::Parameter => "parameter",
            How::Rewritten { .. } => "rewritten",
            How::Introduced => "introduced",
            How::Fused { .. } => "fused",
            How::Alone { .. } => "alone",
            How::LaidOut { .. } => "laid_out",
            How::Kernel { .. } => "kernel",
            How::Launched { .. } => "launched",
            How::Epilogue => "epilogue",
            How::Function { .. } => "function",
            How::Code => "code",
        }
    }
}

/// One derived element and what it was made from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Link {
    pub child: Id,
    pub parents: Vec<Id>,
    pub how: How,
}

/// An element with no child at the next stage, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct End {
    pub id: Id,
    pub why: String,
}

/// The table of one compilation.
#[derive(Clone, Debug, Default)]
pub struct Table {
    /// Each element's label: the line a document prints for it.
    labels: BTreeMap<Id, String>,
    /// The elements in the order they were added, per stage.
    order: BTreeMap<Stage, Vec<Id>>,
    links: Vec<Link>,
    ends: Vec<End>,
    by_child: BTreeMap<Id, Vec<usize>>,
    by_parent: BTreeMap<Id, Vec<usize>>,
}

impl Table {
    pub fn new() -> Table {
        Table::default()
    }

    /// Adds an element, or relabels one added before.
    pub fn element(&mut self, id: Id, label: impl Into<String>) {
        if !self.labels.contains_key(&id) {
            self.order.entry(id.stage).or_default().push(id.clone());
        }
        self.labels.insert(id, label.into());
    }

    pub fn has(&self, id: &Id) -> bool {
        self.labels.contains_key(id)
    }

    pub fn label(&self, id: &Id) -> Option<&str> {
        self.labels.get(id).map(String::as_str)
    }

    /// Adds a link. Parents are deduplicated, in their first order.
    pub fn link(&mut self, child: Id, parents: Vec<Id>, how: How) {
        let mut seen = BTreeSet::new();
        let parents: Vec<Id> = parents.into_iter().filter(|p| seen.insert(p.clone())).collect();
        let i = self.links.len();
        self.by_child.entry(child.clone()).or_default().push(i);
        for p in &parents {
            self.by_parent.entry(p.clone()).or_default().push(i);
        }
        self.links.push(Link { child, parents, how });
    }

    pub fn end(&mut self, id: Id, why: impl Into<String>) {
        self.ends.push(End { id, why: why.into() });
    }

    /// Every element of `stage`, in the order added.
    pub fn elements(&self, stage: Stage) -> &[Id] {
        self.order.get(&stage).map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn links(&self) -> &[Link] {
        &self.links
    }

    pub fn ends(&self) -> &[End] {
        &self.ends
    }

    /// The links that made `id`.
    pub fn made(&self, id: &Id) -> Vec<&Link> {
        self.by_child.get(id).into_iter().flatten().map(|i| &self.links[*i]).collect()
    }

    /// The links `id` is a parent in.
    pub fn made_from(&self, id: &Id) -> Vec<&Link> {
        self.by_parent.get(id).into_iter().flatten().map(|i| &self.links[*i]).collect()
    }

    /// Why `id` has no child, if an [`End`] says.
    pub fn ended(&self, id: &Id) -> Option<&str> {
        self.ends.iter().find(|e| &e.id == id).map(|e| e.why.as_str())
    }

    /// The elements of `stage` reached from `id` through the links, in the
    /// order found; `id` itself when it is of `stage`.
    pub fn descendants(&self, id: &Id, stage: Stage) -> Vec<Id> {
        self.walk(id, stage, |t, x| t.made_from(x).into_iter().map(|l| l.child.clone()).collect())
    }

    /// The elements of `stage` `id` was made from, through the links.
    pub fn ancestors(&self, id: &Id, stage: Stage) -> Vec<Id> {
        self.walk(id, stage, |t, x| t.made(x).into_iter().flat_map(|l| l.parents.iter().cloned()).collect())
    }

    fn walk(&self, id: &Id, stage: Stage, next: impl Fn(&Table, &Id) -> Vec<Id>) -> Vec<Id> {
        let mut out = Vec::new();
        let mut seen = BTreeSet::new();
        let mut queue = VecDeque::from([id.clone()]);
        while let Some(x) = queue.pop_front() {
            if !seen.insert(x.clone()) {
                continue;
            }
            if x.stage == stage {
                out.push(x);
                continue;
            }
            queue.extend(next(self, &x));
        }
        out
    }

    /// The elements of every stage but the source that no link made: what
    /// statement P2 counts, and must find none of.
    pub fn orphans(&self) -> Vec<&Id> {
        self.labels.keys().filter(|id| id.stage != Stage::Source && !self.by_child.contains_key(*id)).collect()
    }

    /// The elements a link or an end names that the table does not have.
    pub fn dangling(&self) -> Vec<&Id> {
        let mut out: BTreeSet<&Id> = BTreeSet::new();
        for l in &self.links {
            out.extend(std::iter::once(&l.child).chain(&l.parents).filter(|id| !self.labels.contains_key(*id)));
        }
        out.extend(self.ends.iter().map(|e| &e.id).filter(|id| !self.labels.contains_key(*id)));
        out.into_iter().collect()
    }

    /// The links whose child is of `stage` and some parent of a stage after
    /// it: a link drawn backwards, which no stage makes.
    pub fn backwards(&self) -> Vec<&Link> {
        self.links.iter().filter(|l| l.parents.iter().any(|p| p.stage >= l.child.stage)).collect()
    }

    /// Adds every element, link and end of `other`.
    pub fn extend(&mut self, other: Table) {
        for stage in Stage::ALL {
            for id in other.elements(stage) {
                self.element(id.clone(), other.labels[id].clone());
            }
        }
        for l in other.links {
            self.link(l.child, l.parents, l.how);
        }
        self.ends.extend(other.ends);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(s: Stage, k: &str) -> Id {
        Id::new(s, k)
    }

    #[test]
    fn a_thread_is_followed_both_ways() {
        let mut t = Table::new();
        t.element(id(Stage::Source, "0:10-20"), "dot x, w");
        t.element(id(Stage::Elab, "expr 0:10-20"), "dot x w");
        t.element(id(Stage::Core, "f/%3"), "%3 = dot x, w");
        t.element(id(Stage::Fusion, "f/%3"), "dot");
        t.element(id(Stage::Kernel, "k0"), "kernel dot");
        t.link(id(Stage::Elab, "expr 0:10-20"), vec![id(Stage::Source, "0:10-20")], How::Typed);
        t.link(id(Stage::Core, "f/%3"), vec![id(Stage::Elab, "expr 0:10-20")], How::Lowered { rule: "dot".into() });
        t.link(id(Stage::Fusion, "f/%3"), vec![id(Stage::Core, "f/%3")], How::Alone { stop: "a reduction".into() });
        t.link(id(Stage::Kernel, "k0"), vec![id(Stage::Fusion, "f/%3")], How::Kernel { shared_by: 1 });
        assert_eq!(t.descendants(&id(Stage::Core, "f/%3"), Stage::Kernel), vec![id(Stage::Kernel, "k0")]);
        assert_eq!(t.ancestors(&id(Stage::Kernel, "k0"), Stage::Source), vec![id(Stage::Source, "0:10-20")]);
        assert!(t.orphans().is_empty());
        assert!(t.dangling().is_empty());
        assert!(t.backwards().is_empty());
    }

    #[test]
    fn an_element_without_a_link_is_an_orphan() {
        let mut t = Table::new();
        t.element(id(Stage::Core, "f/%3"), "%3");
        t.link(id(Stage::Ad, "f/%3"), vec![id(Stage::Core, "f/%3")], How::Unchanged);
        assert_eq!(t.orphans(), vec![&id(Stage::Core, "f/%3")]);
        assert_eq!(t.dangling(), vec![&id(Stage::Ad, "f/%3")]);
    }
}
