//! The network diagram (plan 0.14.0, D141): a program drawn as boxes and
//! arrows, a view of the provenance table and not a second model.
//!
//! The graph of a function is read off the program the kernels run (the
//! table's stage `ad`, or `opt` under the optimizer):
//! - a **call box** per call of a function: named by the field of the
//!   parameters it is given (`ln1`, `attn`), its instance from their type
//!   (`LayerNorm{64}`); a `fold` over stacked parameters is one box ×L;
//! - an **operation node** per fusion element of the function (a group, or
//!   a primitive alone), and per primitive the host computes;
//! - a **parameter node** per parameter an operation reads (`θ.w`); the
//!   parameters a call box is given are its port, not a node;
//! - the function's **inputs** and its **result**;
//! - an **arrow** per value that flows, through the projections and tuples
//!   the host carries.
//!
//! Every node and arrow names an element of the table ([`Node::element`],
//! [`Edge::site`]): a box its call's `let` (a stack's `fold`) and its
//! callee's function beside it, an operation node its fusion element, an
//! arrow the site of the value it carries, or the entry's function for the
//! entry's inputs (the table has no element for a function's parameters).
//! The drawing adds positions and numbers to them, and nothing else:
//! - a box's formula is the printer's display of its function, an
//!   operation node's its root `let`'s equation;
//! - an arrow's type is the `let`'s, under the call's dimension arguments
//!   and with an open prefix bound from the caller's argument, at the
//!   entry's dimensions, its bytes the elements times the realized type's;
//! - a box's parameters are its subtree of the entry's, each leaf in its
//!   stored type;
//! - a box's work W and bytes Q are the cost model's over the launches
//!   inside its call ([`presk_kernel::Launched::calls`]).
//!
//! The layout ([`layout`]) is layered, top to bottom: each box's graph laid
//! out once in its own frame, and any set of boxes opened by composing the
//! frames (plan 0.14.1, D146). The page, the SVG and the TikZ draw the same
//! composition.
//!
//! The crate has no external dependency.

pub mod build;
pub mod json;
pub mod layout;
pub mod shape;
pub mod svg;
pub mod text;
pub mod tikz;

pub use build::{Inputs, network};

use presk_provenance::Id;

/// What a node of the diagram is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The entry's own frame: the graph drawn first.
    Root,
    /// A call of a function: a box that opens to the callee's graph.
    Call,
    /// A `fold` over stacked parameters, a `stack{L}`: one box ×L that
    /// opens to one layer.
    Stack,
    /// A fusion group: several `let`s that run as one loop nest.
    Group,
    /// A primitive that runs as a loop nest alone.
    Op,
    /// A primitive the host computes: no kernel.
    Host,
    /// A parameter an operation reads.
    Param,
    /// An argument of the function drawn around it.
    Input,
    /// Its result.
    Output,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Root => "root",
            Kind::Call => "call",
            Kind::Stack => "stack",
            Kind::Group => "group",
            Kind::Op => "op",
            Kind::Host => "host",
            Kind::Param => "param",
            Kind::Input => "input",
            Kind::Output => "output",
        }
    }

    /// A box: a node with a graph of its own inside.
    pub fn is_box(self) -> bool {
        matches!(self, Kind::Root | Kind::Call | Kind::Stack)
    }
}

/// The parameters a box is given, or a parameter node is: their count of
/// elements over the subtree, their bytes at rest in their stored types
/// (a quantized leaf's integers and its `f32` scales), and the types.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Params {
    pub count: u64,
    pub bytes: u64,
    /// The stored types of the leaves, each once, in the leaves' order:
    /// `f64`, `bf16`, `int8` …
    pub types: Vec<String>,
}

/// A node.
#[derive(Clone, Debug)]
pub struct Node {
    pub id: usize,
    pub kind: Kind,
    /// The box whose graph it is in; `None` for the root.
    pub parent: Option<usize>,
    /// Its element of the provenance table.
    pub element: Id,
    /// A box's callee's function element.
    pub function: Option<Id>,
    /// A box's callee, by the name PSK-Core gives it.
    pub callee: Option<String>,
    /// What the node is called on the drawing: a box by its field (`ln1`),
    /// an operation by its primitives, an input or a parameter by its name.
    pub label: String,
    /// A box's instance, in the entry's dimensions: `LayerNorm{64}`.
    pub instance: Option<String>,
    /// A stack's count of layers, `L`.
    pub stack: Option<u64>,
    /// How many times the node runs for one run of the box around it, the
    /// layers of the stacks it is inside: its numbers are over all of them.
    pub runs: u64,
    /// A box's path in the entry's parameters, `blocks.attn`.
    pub path: Option<String>,
    /// The formula: a box's function's display, an operation's equation,
    /// in LaTeX (`tex`) and Unicode (`text`).
    pub tex: Option<String>,
    pub text: Option<String>,
    /// A contraction's sum and the order its terms are added in, in LaTeX.
    pub sum: Option<String>,
    pub order: Option<String>,
    pub params: Option<Params>,
    /// The cost model's work and bytes over the launches the node makes,
    /// all its runs included; `None` when the kernels were not listed.
    pub work: Option<f64>,
    pub bytes: Option<f64>,
    /// A function differentiation generated: its kind (`vjp`, `fwd`, `bwd`
    /// …) and the function element it was generated from.
    pub generated: Option<(String, Id)>,
    /// A group's member sites, its root first.
    pub members: Vec<Id>,
    /// An input's place among the function's arguments.
    pub index: Option<usize>,
    /// The nodes drawn inside a box, in the order they were made.
    pub children: Vec<usize>,
    /// Its depth: the root 0, its graph's nodes 1 …
    pub depth: usize,
    /// It carries keys only: a `split_key`, an input of type `key`.
    pub key: bool,
}

/// An arrow: a value that flows from one node to another, in the graph of
/// the box `parent`.
#[derive(Clone, Debug)]
pub struct Edge {
    pub id: usize,
    pub from: usize,
    pub to: usize,
    /// The box whose graph it is in.
    pub parent: usize,
    /// Its place among the arguments of `to` (a box's port), when `to`
    /// reads it as an argument.
    pub arg: Option<usize>,
    /// The site of the value: an element of the table.
    pub site: Id,
    /// Its type, in the entry's dimensions: `ℝ[batch: B, seq: T, model: 64]`.
    pub ty: String,
    /// The same at the entry's dimensions: `ℝ[batch: 2, seq: 16, model: 64]`.
    pub at: String,
    /// Its realized type: `f64`, `f32`, `bf16`, `idx`, `bool`, `key`, or
    /// `params` for a tree of parameters.
    pub dtype: String,
    /// The real elements it holds (a tuple's summed), and its bytes.
    pub elems: u64,
    pub bytes: u64,
    /// It carries keys only.
    pub key: bool,
}

/// The diagram of one entry.
#[derive(Clone, Debug)]
pub struct Network {
    pub entry: String,
    /// The table's stage the graph is read from: `ad` or `opt`.
    pub stage: presk_provenance::Stage,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    /// Each box's graph laid out with its children folded, the entry's
    /// (`0`) among them: what every opening composes ([`layout::view`]).
    pub frames: std::collections::BTreeMap<usize, layout::Frame>,
    /// The boxes open in the opening the page starts from: the entry's
    /// first module box, every stack folded.
    pub default: Vec<usize>,
    /// The parameter count over the entry's parameters.
    pub params: Option<Params>,
    /// The work and bytes over every launch of the entry.
    pub work: Option<f64>,
    pub bytes: Option<f64>,
}

impl Network {
    pub fn node(&self, id: usize) -> &Node {
        &self.nodes[id]
    }

    /// The nodes of `n`'s graph.
    pub fn children(&self, n: usize) -> impl Iterator<Item = &Node> {
        self.nodes[n].children.iter().map(move |c| &self.nodes[*c])
    }

    /// The edges drawn inside `n`.
    pub fn edges_in(&self, n: usize) -> impl Iterator<Item = &Edge> {
        self.edges.iter().filter(move |e| e.parent == n)
    }
}
