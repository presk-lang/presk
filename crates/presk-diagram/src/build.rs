//! The graph of the entry and of every function it calls, each box drawn
//! at its own call, with its numbers.

use std::collections::{BTreeMap, HashMap};

use presk_core::ir::{Func, Let, Rhs};
use presk_core::{FuncId, Program, VarId};
use presk_print_math::{Format, Listing};
use presk_provenance::{How, Id, Stage, Table};
use presk_types::dim::{DimExpr, DimVar};
use presk_types::elab::ElabProgram;
use presk_types::ty::{ModRef, Shape, Ty};

use crate::shape::{self, Ctx};
use crate::{Edge, Kind, Network, Node, Params};

/// What the diagram is drawn from.
pub struct Inputs<'a> {
    pub prog: &'a ElabProgram,
    /// The printer's listing of the typed program, for the boxes' formulas.
    pub listing: Option<&'a Listing>,
    /// The program the kernels run, and the table's stage that names it.
    pub program: &'a Program,
    pub stage: Stage,
    pub table: &'a Table,
    pub entry: &'a str,
    /// The entry's dimensions, and the axes of its open prefix.
    pub dims: &'a BTreeMap<DimVar, u64>,
    pub prefix: &'a [(Option<String>, u64)],
    pub realization: &'a presk_rt::Realization,
    /// The kernels and the launches of a call, when they were listed.
    pub kernels: Option<(&'a [presk_kernel::Kernel], &'a [presk_kernel::Launched])>,
}

/// The diagram of `i.entry`, laid out. `None` when the program has no such
/// function.
pub fn network(i: &Inputs) -> Option<Network> {
    let entry = i.program.find(i.entry)?;
    let mut b = Builder::new(i);
    let func = i.program.func(entry);
    let ctx = Ctx::entry(func, i.prefix);
    let root_el = Id::new(i.stage, func.name.clone());
    let theta = func.params.first().filter(|p| matches!(p.ty, Ty::Params(_)));
    let root = b.node(Node {
        kind: Kind::Root,
        parent: None,
        element: root_el.clone(),
        function: Some(root_el.clone()),
        callee: Some(func.name.clone()),
        label: func.name.clone(),
        instance: None,
        path: theta.map(|_| String::new()),
        depth: 0,
        ..blank(root_el.clone())
    });
    b.formula(root, &func.name);
    if let Some(p) = theta {
        b.nodes[root].params = b.params(&ctx.ty(&p.ty), "");
    }
    // The entry's parameters are its first argument when it is a tree of
    // them; another tree (an optimizer's gradient) is a value.
    let args: Vec<Arg> = func.params.iter().enumerate().map(|(k, p)| Arg { site: root_el.clone(), ty: Some(p.ty.clone()), path: (k == 0 && theta.is_some()).then(String::new) }).collect();
    b.draw(root, entry, &ctx, &args, &[], 1);
    let (work, bytes) = b.costs(&[], None);
    b.nodes[root].work = work;
    b.nodes[root].bytes = bytes;
    let params = b.nodes[root].params.clone();
    let mut n = Network { entry: i.entry.to_string(), stage: i.stage, nodes: b.nodes, edges: b.edges, frames: Default::default(), default: Vec::new(), memory: Default::default(), params, work, bytes };
    crate::layout::lay_out(&mut n);
    Some(n)
}

/// A node with every optional field empty.
fn blank(element: Id) -> Node {
    Node {
        id: 0,
        kind: Kind::Host,
        parent: None,
        element,
        function: None,
        callee: None,
        label: String::new(),
        instance: None,
        stack: None,
        runs: 1,
        path: None,
        tex: None,
        text: None,
        sum: None,
        order: None,
        params: None,
        work: None,
        bytes: None,
        generated: None,
        members: Vec::new(),
        index: None,
        children: Vec::new(),
        depth: 0,
        key: false,
        calls: Vec::new(),
    }
}

/// A value a box is given: the site that holds it at the caller, its type
/// there, and its path in the entry's parameters when it is a subtree.
#[derive(Clone)]
struct Arg {
    site: Id,
    ty: Option<Ty>,
    path: Option<String>,
}

/// A box at its call: the box, its callee, the callee's types there, the
/// arguments it is given, the calls it is inside, and its layers.
type AtCall = (usize, FuncId, Ctx, Vec<Arg>, Vec<String>, u64);

/// Where a value read in a function comes from: the node that makes it,
/// the site that holds it, and its type in the function.
#[derive(Clone)]
struct Src {
    node: usize,
    site: Id,
    ty: Ty,
}

/// A launch, its calls without the folds' iterations.
struct Launch {
    calls: Vec<String>,
    nest: String,
    work: f64,
    bytes: f64,
}

struct Builder<'a> {
    i: &'a Inputs<'a>,
    /// The entry's dimensions, to evaluate with.
    at: BTreeMap<DimVar, DimExpr>,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    launches: Option<Vec<Launch>>,
    /// The printer's displays by function: LaTeX and Unicode.
    displays: HashMap<String, (String, String)>,
}

/// What one `let` of a function is on the drawing.
#[derive(Clone, Copy, PartialEq)]
enum Role {
    /// A host scalar of literals: a hyperparameter, not drawn.
    Constant,
    /// A subtree of the parameters: a box's port, or a parameter node.
    Params,
    /// A projection, a tuple or a record the host carries: arrows pass
    /// through it.
    Carried,
    /// A function value: the box that applies it draws it.
    Closure,
    /// A node.
    Node,
    /// A member of a group, inside its root's node.
    Member,
}

impl<'a> Builder<'a> {
    fn new(i: &'a Inputs<'a>) -> Builder<'a> {
        let launches = i.kernels.map(|(ks, log)| {
            let costs: Vec<presk_kernel::cost::KernelCost> = ks.iter().map(presk_kernel::cost::kernel_cost).collect();
            log.iter()
                .map(|l| Launch {
                    calls: l.calls.iter().map(|c| c.split_once('[').map_or(c.as_str(), |(s, _)| s).to_string()).collect(),
                    nest: l.nest.clone(),
                    work: costs.get(l.kernel).map_or(0.0, |c| c.work),
                    bytes: costs.get(l.kernel).map_or(0.0, |c| c.bytes),
                })
                .collect()
        });
        let mut displays = HashMap::new();
        if let Some(l) = i.listing {
            for u in &l.units {
                for item in &u.items {
                    let ds: Vec<&presk_print_math::Display> = match item {
                        presk_print_math::typed::Item::Fn(d) => vec![d],
                        presk_print_math::typed::Item::Module { fns, .. } => fns.iter().collect(),
                    };
                    for d in ds {
                        displays.insert(d.name.clone(), (presk_print_math::aligned(d), unicode(d)));
                    }
                }
            }
        }
        Builder { i, at: shape::assignment(i.dims), nodes: Vec::new(), edges: Vec::new(), launches, displays }
    }

    fn node(&mut self, mut n: Node) -> usize {
        let id = self.nodes.len();
        n.id = id;
        if let Some(p) = n.parent {
            self.nodes[p].children.push(id);
        }
        self.nodes.push(n);
        id
    }

    fn site(&self, f: &Func, v: VarId) -> Id {
        Id::new(self.i.stage, format!("{}/%{}", f.name, v.0))
    }

    /// A box's formula: the printer's display of its function.
    fn formula(&mut self, n: usize, name: &str) {
        if let Some((t, u)) = self.displays.get(name) {
            self.nodes[n].tex = Some(t.clone());
            self.nodes[n].text = Some(u.clone());
        }
    }

    /// The parameters of the tree, or the leaf, of type `t` (in the entry's
    /// dimensions) at `path`: inside a stack, a layer's slice.
    fn params(&self, t: &Ty, path: &str) -> Option<Params> {
        let leaves = match t {
            Ty::Params(m) => self.i.prog.params_tree(m).leaves(),
            Ty::Real(_) => vec![(String::new(), t.clone())],
            _ => return None,
        };
        let r = self.i.realization;
        let mut p = Params::default();
        for (leaf, ty) in leaves {
            let full = if path.is_empty() {
                leaf.clone()
            } else if leaf.is_empty() {
                path.to_string()
            } else {
                format!("{path}.{leaf}")
            };
            let Ty::Real(s) = &ty else { continue };
            let lens: Option<Vec<u64>> = s.axes.iter().map(|a| shape::eval(&a.len, &self.at)).collect();
            let lens = lens?;
            let numel: u64 = lens.iter().product();
            let (name, bytes) = match r.quant.get(&full) {
                Some(q) => {
                    let axes: Vec<presk_rt::Axis> = s.axes.iter().zip(&lens).map(|(a, n)| presk_rt::Axis { name: a.name.clone(), len: *n as usize }).collect();
                    let over = q.over.resolve(&axes).map_or(1, |k| lens[k]).max(1);
                    let ints = (numel * u64::from(q.bits)).div_ceil(8);
                    (format!("int{}", q.bits), ints + 4 * (numel / over))
                }
                None => {
                    let d = r.params.get(&full).copied().unwrap_or(r.default);
                    (d.name().to_string(), numel * d.bytes() as u64)
                }
            };
            p.count += numel;
            p.bytes += bytes;
            if !p.types.contains(&name) {
                p.types.push(name);
            }
        }
        Some(p)
    }

    /// The work and bytes of the launches inside the calls `path`: those of
    /// the nest `nest` alone when it is given, else all of them.
    fn costs(&self, path: &[String], nest: Option<&str>) -> (Option<f64>, Option<f64>) {
        let Some(ls) = &self.launches else {
            return (None, None);
        };
        let (mut w, mut q) = (0.0, 0.0);
        for l in ls {
            let inside = match nest {
                Some(n) => l.nest == n && l.calls == path,
                None => l.calls.len() >= path.len() && l.calls[..path.len()] == *path,
            };
            if inside {
                w += l.work;
                q += l.bytes;
            }
        }
        (Some(w), Some(q))
    }

    /// The function `f` drawn inside the box `parent`, at the call whose
    /// types are `ctx`, given `args`, inside the calls `calls`, run `runs`
    /// times for one run of the entry.
    fn draw(&mut self, parent: usize, f: FuncId, ctx: &Ctx, args: &[Arg], calls: &[String], runs: u64) {
        let i = self.i;
        let func = i.program.func(f);
        let depth = self.nodes[parent].depth + 1;
        let fsite = |v: VarId| format!("{}/%{}", func.name, v.0);

        // The path in the entry's parameters of each subtree the function
        // holds.
        let mut paths: HashMap<VarId, String> = HashMap::new();
        for (k, p) in func.params.iter().enumerate() {
            if let Some(path) = args.get(k).and_then(|a| a.path.clone()) {
                paths.insert(p.var, path);
            }
        }
        for l in &func.body.lets {
            if let Rhs::Field(x, name) = &l.rhs
                && let Some(p) = paths.get(x).cloned()
            {
                paths.insert(l.var, if p.is_empty() { name.clone() } else { format!("{p}.{name}") });
            }
        }

        // What each `let` is.
        let lets: HashMap<VarId, &Let> = func.body.lets.iter().map(|l| (l.var, l)).collect();
        let param_of: HashMap<VarId, usize> = func.params.iter().enumerate().map(|(k, p)| (p.var, k)).collect();
        let fusion = self.fusion(func);
        let mut role: HashMap<VarId, Role> = HashMap::new();
        for l in &func.body.lets {
            let r = match &l.rhs {
                Rhs::Lit(_) => Role::Constant,
                // A scalar of literals the host computes; one a kernel
                // computes is a node.
                Rhs::Prim { args: a, .. } if scalar(&l.ty) && !fusion.groups.contains_key(&fsite(l.var)) && !fusion.member.contains_key(&fsite(l.var)) && a.iter().all(|v| role.get(v) == Some(&Role::Constant)) => Role::Constant,
                // A subtree or a leaf of the entry's parameters; a value of
                // their type (a gradient) flows like any other.
                Rhs::Field(..) if paths.contains_key(&l.var) => Role::Params,
                Rhs::Proj(..) | Rhs::Tuple(_) | Rhs::Record(_) | Rhs::Field(..) => Role::Carried,
                Rhs::Closure { .. } => Role::Closure,
                Rhs::Prim { .. } if fusion.member.contains_key(&fsite(l.var)) => Role::Member,
                _ => Role::Node,
            };
            role.insert(l.var, r);
        }

        // The inputs, every argument but the parameters, which are a box's
        // port; an input of parameters is made when an operation reads it.
        let mut inputs: HashMap<usize, usize> = HashMap::new();
        for (k, p) in func.params.iter().enumerate() {
            if !matches!(p.ty, Ty::Params(_) | Ty::Fn { .. }) {
                let id = self.input(parent, func, k, args, depth);
                inputs.insert(k, id);
            }
        }

        // The nodes, in the body's order.
        let mut node_of: HashMap<VarId, usize> = HashMap::new();
        let mut closures: HashMap<VarId, (FuncId, Vec<VarId>)> = HashMap::new();
        for l in &func.body.lets {
            if let Rhs::Closure { f: g, captured } = &l.rhs {
                closures.insert(l.var, (*g, captured.clone()));
            }
            if role[&l.var] != Role::Node {
                continue;
            }
            let id = match &l.rhs {
                Rhs::Call { f: g, .. } => self.call_box(parent, func, l, *g, &paths, depth, runs),
                Rhs::Fold { body, n, stacked, .. } => {
                    let count = shape::eval(&n.subst(&ctx.dims), &self.at).unwrap_or(1);
                    let id = self.call_box(parent, func, l, *body, &paths, depth, runs);
                    self.nodes[id].kind = Kind::Stack;
                    self.nodes[id].stack = Some(count);
                    if let Some(s) = stacked.first()
                        && let Some(lp) = lets.get(s)
                        && let Rhs::Field(_, name) = &lp.rhs
                    {
                        self.nodes[id].label = name.clone();
                    }
                    id
                }
                Rhs::Apply { fun, .. } => {
                    let g = closures.get(fun).map(|(g, _)| *g);
                    match g {
                        Some(g) => self.call_box(parent, func, l, g, &paths, depth, runs),
                        None => self.op(parent, func, l, &fusion, depth, runs, calls),
                    }
                }
                _ => self.op(parent, func, l, &fusion, depth, runs, calls),
            };
            node_of.insert(l.var, id);
        }
        // A group's members read through its root's node.
        for (member, root) in &fusion.member {
            if let (Some((_, m)), Some((_, r))) = (member.rsplit_once("/%"), root.rsplit_once("/%"))
                && let (Ok(m), Ok(r)) = (m.parse::<u32>(), r.parse::<u32>())
                && let Some(n) = node_of.get(&VarId(r)).copied()
            {
                node_of.insert(VarId(m), n);
            }
        }

        // Each box's call: its callee's types there, its arguments. A call's
        // result keeps the callee's open prefix (differentiation copies its
        // type), so the callee's bindings are this function's too.
        let mut here = ctx.clone();
        let mut at_call: Vec<AtCall> = Vec::new();
        for l in &func.body.lets {
            let Some(&id) = node_of.get(&l.var) else {
                continue;
            };
            if !self.nodes[id].kind.is_box() || self.nodes[id].element != self.site(func, l.var) {
                continue;
            }
            let ty_of = |v: VarId| -> Option<Ty> { param_of.get(&v).map(|p| func.params[*p].ty.clone()).or_else(|| lets.get(&v).map(|x| x.ty.clone())) };
            let arg_of = |v: VarId, me: &Builder| -> Arg {
                let site = match param_of.get(&v) {
                    Some(p) => args.get(*p).map_or_else(|| Id::new(i.stage, func.name.clone()), |a| a.site.clone()),
                    None => me.site(func, v),
                };
                Arg { site, ty: ty_of(v), path: paths.get(&v).cloned() }
            };
            let mut inner_calls = calls.to_vec();
            inner_calls.push(fsite(l.var));
            let (g, c, given, layers) = match &l.rhs {
                Rhs::Call { f: g, dim_args, args: a } => {
                    let tys: Vec<Option<Ty>> = a.iter().map(|v| ty_of(*v)).collect();
                    let c = here.call(i.program.func(*g), dim_args, &tys);
                    (*g, c, a.iter().map(|v| arg_of(*v, self)).collect::<Vec<Arg>>(), 1)
                }
                Rhs::Fold { body, dim_args, stacked, init, extra, index, .. } => {
                    let mut tys: Vec<Option<Ty>> = Vec::new();
                    let mut given: Vec<Arg> = Vec::new();
                    if *index {
                        tys.push(Some(Ty::Nat));
                        given.push(Arg { site: self.site(func, l.var), ty: Some(Ty::Nat), path: None });
                    }
                    for v in stacked {
                        let t = ty_of(*v).map(|t| slice(&t));
                        tys.push(t.clone());
                        let mut a = arg_of(*v, self);
                        a.ty = t;
                        given.push(a);
                    }
                    for v in std::iter::once(init).chain(extra) {
                        tys.push(ty_of(*v));
                        given.push(arg_of(*v, self));
                    }
                    let c = here.call(i.program.func(*body), dim_args, &tys);
                    (*body, c, given, self.nodes[id].stack.unwrap_or(1))
                }
                Rhs::Apply { fun, args: a } => {
                    let Some((g, cap)) = closures.get(fun).cloned() else {
                        continue;
                    };
                    let all: Vec<VarId> = cap.iter().chain(a).copied().collect();
                    let tys: Vec<Option<Ty>> = all.iter().map(|v| ty_of(*v)).collect();
                    let c = here.apply(i.program.func(g), &tys);
                    (g, c, all.iter().map(|v| arg_of(*v, self)).collect(), 1)
                }
                _ => continue,
            };
            for (p, axes) in &c.prefixes {
                here.prefixes.entry(*p).or_insert_with(|| axes.clone());
            }
            at_call.push((id, g, c, given, inner_calls, layers));
        }
        let ctx = &here;

        // The arrows.
        let mut param_nodes: HashMap<VarId, usize> = HashMap::new();
        let mut cx = Resolve { func, lets: &lets, role: &role, node_of: &node_of, param_of: &param_of, inputs, param_nodes: &mut param_nodes, args, parent, depth, paths: &paths, runs, ctx };
        let mut wanted: Vec<(usize, Option<usize>, Src)> = Vec::new();
        for l in &func.body.lets {
            if role[&l.var] != Role::Node && role[&l.var] != Role::Member {
                continue;
            }
            let to = node_of[&l.var];
            let reads: Vec<(Option<usize>, VarId)> = match &l.rhs {
                Rhs::Call { args: a, .. } => a.iter().copied().enumerate().map(|(k, v)| (Some(k), v)).collect(),
                Rhs::Fold { stacked, init, extra, index, .. } => {
                    let k0 = usize::from(*index);
                    stacked.iter().chain(std::iter::once(init)).chain(extra).copied().enumerate().map(|(k, v)| (Some(k0 + k), v)).collect()
                }
                Rhs::Apply { fun, args: a } => {
                    let cap = closures.get(fun).map(|(_, c)| c.clone()).unwrap_or_default();
                    cap.iter().chain(a).copied().enumerate().map(|(k, v)| (Some(k), v)).collect()
                }
                r => r.vars().into_iter().map(|v| (None, v)).collect(),
            };
            let is_box = self.nodes[to].kind.is_box();
            for (k, v) in reads {
                // The parameters a box is given, a subtree of the entry's,
                // are its port.
                if is_box && cx.paths.contains_key(&v) {
                    continue;
                }
                for s in self.resolve(&mut cx, v) {
                    if s.node != to {
                        wanted.push((to, k, s));
                    }
                }
            }
        }
        let out = self.node(Node { kind: Kind::Output, parent: Some(parent), label: "result".into(), depth, runs, ..blank(self.result_element(func, args)) });
        for s in self.resolve(&mut cx, func.body.result) {
            wanted.push((out, None, s));
        }
        let mut seen: std::collections::HashSet<(usize, usize, Id)> = std::collections::HashSet::new();
        for (to, k, s) in wanted {
            if !seen.insert((s.node, to, s.site.clone())) {
                continue;
            }
            let ty = ctx.ty(&s.ty);
            // A tree of parameters, or a gradient, holds its leaves' elements.
            let (reals, others) = match &ty {
                Ty::Params(_) => (self.params(&ty, "").map_or(0, |p| p.count), 0),
                _ => shape::elements(&ty, &self.at).unwrap_or((0, 0)),
            };
            // A parameter flows in its stored type, which a kernel widens as
            // it loads it (D121); any other value in its site's.
            let dtype = match (&self.nodes[s.node].kind, self.nodes[s.node].params.as_ref()) {
                (Kind::Param, Some(p)) if p.types.len() == 1 => p.types[0].clone(),
                _ => self.dtype(&s.ty, &s.site, func),
            };
            let per = match dtype.as_str() {
                "f64" => 8,
                "f32" => 4,
                "bf16" => 2,
                _ => 0,
            };
            let e = Edge {
                id: self.edges.len(),
                from: s.node,
                to,
                parent,
                arg: k,
                site: s.site.clone(),
                ty: shape::display(&ty, i.prog),
                at: shape::display_at(&ty, i.prog, &self.at),
                dtype,
                elems: reals,
                // A parameter's bytes are its stored ones, a quantized
                // leaf's scales included.
                bytes: match (&self.nodes[s.node].kind, self.nodes[s.node].params.as_ref()) {
                    (Kind::Param, Some(p)) => p.bytes,
                    _ => reals * per + others * 8,
                },
                key: shape::keys_only(&s.ty),
            };
            self.edges.push(e);
        }

        // Each box's own graph, at its call.
        for (id, g, c, given, inner_calls, layers) in at_call {
            let callee = i.program.func(g);
            self.box_numbers(id, &c, callee, &given, &inner_calls, runs * layers);
            if self.nodes[id].kind == Kind::Stack {
                // The box's parameters are its layers', its numbers over them.
                if let Some(p) = &mut self.nodes[id].params {
                    p.count *= layers;
                    p.bytes *= layers;
                }
                self.nodes[id].runs = runs;
            }
            self.draw(id, g, &c, &given, &inner_calls, runs * layers);
        }
    }

    /// The element of a function's result: the `let` it returns, or the
    /// argument it returns as it is.
    fn result_element(&self, func: &Func, args: &[Arg]) -> Id {
        match func.params.iter().position(|p| p.var == func.body.result) {
            Some(k) => args.get(k).map_or_else(|| Id::new(self.i.stage, func.name.clone()), |a| a.site.clone()),
            None => self.site(func, func.body.result),
        }
    }

    /// A box's instance, parameters, formula, origin and costs, at its call.
    fn box_numbers(&mut self, id: usize, c: &Ctx, callee: &Func, given: &[Arg], calls: &[String], runs: u64) {
        let i = self.i;
        self.nodes[id].calls = calls.to_vec();
        let theta = callee.params.iter().zip(given).find(|(p, _)| matches!(p.ty, Ty::Params(_)));
        if let Some((p, a)) = theta {
            let t = c.ty(&p.ty);
            if let Ty::Params(m) = &t {
                self.nodes[id].instance = Some(instance(m, i.prog));
            }
            self.nodes[id].path = a.path.clone();
            self.nodes[id].params = a.path.as_deref().and_then(|path| self.params(&t, path));
        }
        self.formula(id, &callee.name);
        let made = i.table.made(&Id::new(i.stage, callee.name.clone()));
        if let Some(l) = made.iter().find(|l| matches!(l.how, How::Generated { .. })) {
            if let How::Generated { kind } = &l.how {
                self.nodes[id].generated = l.parents.first().map(|p| (kind.clone(), p.clone()));
            }
        }
        let (w, q) = self.costs(calls, None);
        self.nodes[id].work = w;
        self.nodes[id].bytes = q;
        self.nodes[id].runs = runs;
    }

    #[allow(clippy::too_many_arguments)]
    fn call_box(&mut self, parent: usize, func: &Func, l: &Let, callee: FuncId, paths: &HashMap<VarId, String>, depth: usize, runs: u64) -> usize {
        let i = self.i;
        let g = i.program.func(callee);
        // The field of the parameters it is given names it, else its callee.
        let given = match &l.rhs {
            Rhs::Call { args, .. } => args.clone(),
            Rhs::Fold { stacked, .. } => stacked.clone(),
            _ => Vec::new(),
        };
        let field = given.iter().find_map(|v| {
            let p = paths.get(v)?;
            let last = p.rsplit('.').next().unwrap_or(p);
            (!last.is_empty() && func.body.lets.iter().any(|x| x.var == *v && matches!(x.rhs, Rhs::Field(..)))).then(|| last.to_string())
        });
        let label = field.unwrap_or_else(|| g.name.clone());
        self.node(Node { kind: Kind::Call, parent: Some(parent), function: Some(Id::new(i.stage, g.name.clone())), callee: Some(g.name.clone()), label, depth, runs, ..blank(self.site(func, l.var)) })
    }

    #[allow(clippy::too_many_arguments)]
    fn op(&mut self, parent: usize, func: &Func, l: &Let, fusion: &Fusion, depth: usize, runs: u64, calls: &[String]) -> usize {
        let i = self.i;
        let site = format!("{}/%{}", func.name, l.var.0);
        // An operation by its name, without its operands' axes:
        // `ein[batch, heads, seq, seq2]`, not the product it contracts.
        let name = |l: &Let| match &l.rhs {
            Rhs::Prim { op, .. } => {
                let n = op.name();
                match n.find('(') {
                    Some(k) => n[..k].to_string(),
                    None => n,
                }
            }
            Rhs::Call { .. } => "call".into(),
            Rhs::Fold { .. } => "fold".into(),
            Rhs::Apply { .. } => "apply".into(),
            Rhs::Grad(_) => "grad".into(),
            _ => "let".into(),
        };
        let (kind, element, members) = match fusion.groups.get(&site) {
            Some(ms) if ms.len() > 1 => (Kind::Group, Id::new(Stage::Fusion, site.clone()), ms.clone()),
            Some(_) => (Kind::Op, Id::new(Stage::Fusion, site.clone()), vec![site.clone()]),
            None => (Kind::Host, self.site(func, l.var), Vec::new()),
        };
        let label = if members.len() > 1 {
            let names: Vec<String> = members.iter().filter_map(|m| m.rsplit_once("/%").and_then(|(_, n)| n.parse::<u32>().ok())).filter_map(|n| func.body.lets.iter().find(|x| x.var.0 == n)).map(name).collect();
            names.join(" · ")
        } else {
            name(l)
        };
        let e = presk_print_math::core::equation(func, l, i.program, i.prog);
        let tex = format!("{} = {}", e.lhs.render(Format::Latex), e.rhs.render(Format::Latex));
        let text = format!("{} = {}", e.lhs.render(Format::Unicode), e.rhs.render(Format::Unicode));
        let (sum, order) = match presk_print_math::rules::contraction(func, l, i.prog) {
            Some(c) => {
                let partials = match i.realization.order {
                    presk_rt::ReductionOrder::Interleaved { partials } => Some(u64::from(partials)),
                    _ => None,
                };
                (Some(c.render(Format::Latex)), Some(presk_print_math::rules::order(partials, Format::Latex)))
            }
            None => (None, None),
        };
        let id = self.node(Node { kind, parent: Some(parent), label, tex: Some(tex), text: Some(text), sum, order, members: members.iter().map(|m| Id::new(i.stage, m.clone())).collect(), depth, runs, key: shape::keys_only(&l.ty), ..blank(element) });
        let (w, q) = if kind == Kind::Host { (self.launches.as_ref().map(|_| 0.0), self.launches.as_ref().map(|_| 0.0)) } else { self.costs(calls, Some(&site)) };
        self.nodes[id].work = w;
        self.nodes[id].bytes = q;
        id
    }

    fn input(&mut self, parent: usize, func: &Func, k: usize, args: &[Arg], depth: usize) -> usize {
        let p = &func.params[k];
        let element = args.get(k).map_or_else(|| Id::new(self.i.stage, func.name.clone()), |a| a.site.clone());
        let runs = self.nodes[parent].runs;
        self.node(Node { kind: Kind::Input, parent: Some(parent), label: p.name.clone(), index: Some(k), depth, runs, key: shape::keys_only(&p.ty), ..blank(element) })
    }

    /// The fusion groups and primitives alone of a function, as the table
    /// holds them: each nest's root site and its members, and each member's
    /// root.
    fn fusion(&self, func: &Func) -> Fusion {
        let mut f = Fusion::default();
        let prefix = format!("{}/%", func.name);
        for id in self.i.table.elements(Stage::Fusion) {
            if !id.key.starts_with(&prefix) {
                continue;
            }
            let ms: Vec<String> = self.i.table.made(id).iter().flat_map(|l| l.parents.iter().filter(|p| p.stage == self.i.stage).map(|p| p.key.clone())).collect();
            for m in &ms {
                if *m != id.key {
                    f.member.insert(m.clone(), id.key.clone());
                }
            }
            let mut all = vec![id.key.clone()];
            all.extend(ms.into_iter().filter(|m| *m != id.key));
            f.groups.insert(id.key.clone(), all);
        }
        f
    }

    /// Where a value read in a function comes from.
    fn resolve(&mut self, cx: &mut Resolve, v: VarId) -> Vec<Src> {
        let func = cx.func;
        if let Some(k) = cx.param_of.get(&v).copied() {
            let p = &func.params[k];
            let node = match cx.inputs.get(&k) {
                Some(n) => *n,
                None => {
                    let n = self.input(cx.parent, func, k, cx.args, cx.depth);
                    cx.inputs.insert(k, n);
                    n
                }
            };
            let site = cx.args.get(k).map_or_else(|| Id::new(self.i.stage, func.name.clone()), |a| a.site.clone());
            return vec![Src { node, site, ty: p.ty.clone() }];
        }
        let Some(l) = cx.lets.get(&v).copied() else {
            return Vec::new();
        };
        let site = self.site(func, v);
        match cx.role.get(&v).copied() {
            Some(Role::Constant | Role::Closure) | None => Vec::new(),
            Some(Role::Node | Role::Member) => match cx.node_of.get(&v) {
                Some(n) => vec![Src { node: *n, site, ty: l.ty.clone() }],
                None => Vec::new(),
            },
            Some(Role::Params) => {
                let node = match cx.param_nodes.get(&v) {
                    Some(n) => *n,
                    None => {
                        let label = match &l.rhs {
                            Rhs::Field(_, n) => n.clone(),
                            _ => format!("%{}", v.0),
                        };
                        let params = cx.paths.get(&v).and_then(|p| self.params(&cx.ctx.ty(&l.ty), p));
                        let n = self.node(Node { kind: Kind::Param, parent: Some(cx.parent), label, path: cx.paths.get(&v).cloned(), params, depth: cx.depth, runs: cx.runs, ..blank(site.clone()) });
                        cx.param_nodes.insert(v, n);
                        n
                    }
                };
                vec![Src { node, site, ty: l.ty.clone() }]
            }
            Some(Role::Carried) => match &l.rhs {
                Rhs::Tuple(vs) => vs.clone().into_iter().flat_map(|x| self.resolve(cx, x)).collect(),
                Rhs::Record(fs) => fs.clone().into_iter().flat_map(|(_, x)| self.resolve(cx, x)).collect(),
                Rhs::Proj(t, k) => {
                    // A component of a tuple the function built is that
                    // component; of any other value, the value it is read
                    // from, carrying this site.
                    if let Some(tl) = cx.lets.get(t)
                        && let Rhs::Tuple(vs) = &tl.rhs
                        && let Some(x) = vs.get(*k)
                    {
                        return self.resolve(cx, *x);
                    }
                    self.through(cx, *t, site, &l.ty)
                }
                Rhs::Field(x, _) => self.through(cx, *x, site, &l.ty),
                _ => Vec::new(),
            },
        }
    }

    /// A part of `from`, held at `site`: an arrow from wherever `from` comes,
    /// carrying the part.
    fn through(&mut self, cx: &mut Resolve, from: VarId, site: Id, ty: &Ty) -> Vec<Src> {
        let srcs = self.resolve(cx, from);
        match srcs.as_slice() {
            [one] => vec![Src { node: one.node, site, ty: ty.clone() }],
            _ => srcs.into_iter().filter(|s| shape::keys_only(&s.ty) == shape::keys_only(ty)).map(|s| Src { node: s.node, site: site.clone(), ty: ty.clone() }).take(1).collect(),
        }
    }

    /// The realized type of the value at `site`, of type `t`.
    fn dtype(&self, t: &Ty, site: &Id, func: &Func) -> String {
        let _ = func;
        match t {
            Ty::Real(_) => self.i.realization.dtype_at(&site.key).name().to_string(),
            Ty::Tuple(ts) if ts.iter().any(|t| matches!(t, Ty::Real(_))) => self.i.realization.dtype_at(&site.key).name().to_string(),
            Ty::Idx { .. } => "idx".into(),
            Ty::Truth(_) => "bool".into(),
            Ty::Key => "key".into(),
            Ty::Tuple(ts) if !ts.is_empty() && ts.iter().all(|t| matches!(t, Ty::Key)) => "key".into(),
            Ty::Params(_) => self.i.realization.dtype_at(&site.key).name().to_string(),
            _ => "host".into(),
        }
    }
}

/// The fusion groups of a function: each nest's root and members, root
/// first, and each member's root.
#[derive(Default)]
struct Fusion {
    groups: BTreeMap<String, Vec<String>>,
    member: BTreeMap<String, String>,
}

/// What resolving a value in a function needs.
struct Resolve<'r> {
    func: &'r Func,
    lets: &'r HashMap<VarId, &'r Let>,
    role: &'r HashMap<VarId, Role>,
    node_of: &'r HashMap<VarId, usize>,
    param_of: &'r HashMap<VarId, usize>,
    inputs: HashMap<usize, usize>,
    param_nodes: &'r mut HashMap<VarId, usize>,
    args: &'r [Arg],
    parent: usize,
    depth: usize,
    paths: &'r HashMap<VarId, String>,
    runs: u64,
    ctx: &'r Ctx,
}

/// A host scalar: what a literal computes to.
fn scalar(t: &Ty) -> bool {
    match t {
        Ty::F64 | Ty::Nat | Ty::Int | Ty::Rat | Ty::Bool => true,
        Ty::Real(s) => s.prefix.is_none() && s.axes.is_empty(),
        _ => false,
    }
}

/// One slice of a stacked value: a stack's layer, a tensor without its
/// leading axis.
fn slice(t: &Ty) -> Ty {
    match t {
        Ty::Params(ModRef::Stack { inner, .. }) => Ty::Params((**inner).clone()),
        Ty::Real(s) if s.prefix.is_none() && !s.axes.is_empty() => Ty::Real(Shape { prefix: None, axes: s.axes[1..].to_vec() }),
        Ty::Tuple(ts) => Ty::Tuple(ts.iter().map(slice).collect()),
        t => t.clone(),
    }
}

/// An instance as the program writes it: `LayerNorm{64}`, `stack{2}(Block)`.
fn instance(m: &ModRef, prog: &ElabProgram) -> String {
    m.display(&prog.names())
}

/// A display's equations for a terminal, one a line, as `presk dump
/// --format unicode` writes them.
pub fn unicode(d: &presk_print_math::Display) -> String {
    d.eqs
        .iter()
        .map(|e| {
            let mut s = format!("{} = {}", e.lhs.render(Format::Unicode), e.rhs.render(Format::Unicode));
            if let Some(c) = &e.cond {
                s.push_str(&format!("    ({})", c.render(Format::Unicode)));
            }
            s
        })
        .collect::<Vec<_>>()
        .join("\n")
}
