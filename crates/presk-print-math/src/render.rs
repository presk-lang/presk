//! A listing in its two renderings: a LaTeX fragment (or a document that
//! compiles alone) and the same formulas for a terminal.

use crate::m::{Format, M, escape};
use crate::typed::{Display, Equation, Item, Listing};

/// What to print of a listing.
#[derive(Clone, Debug, Default)]
pub struct Select {
    /// Only this module (its name, `Linear`).
    pub module: Option<String>,
    /// Only this function (its name, `Linear.apply` or `apply`).
    pub function: Option<String>,
}

impl Select {
    fn module(&self, name: &str) -> bool {
        self.module.as_deref().is_none_or(|m| m == name)
    }
    fn function(&self, d: &Display) -> bool {
        self.function.as_deref().is_none_or(|f| d.name == f || d.name.rsplit_once('.').is_some_and(|(_, n)| n == f))
    }
}

/// The macros and packages a standalone document needs: the reference's.
pub const PREAMBLE: &str = "\\documentclass[11pt]{article}\n\\usepackage{amsmath,amssymb,stmaryrd}\n\\usepackage[a4paper,margin=2cm]{geometry}\n% The language reference's macros (presk-reference.tex).\n\\newcommand{\\code}[1]{\\text{\\texttt{#1}}}\n\\newcommand{\\den}[1]{\\llbracket #1 \\rrbracket}\n\\setlength{\\parindent}{0pt}\n\\begin{document}\n";

/// The equations of a display as one `aligned` block, for KaTeX on the
/// page and for a reader that pastes one function.
pub fn aligned(d: &Display) -> String {
    let mut s = String::from("\\begin{aligned}\n");
    let n = d.eqs.len();
    for (i, e) in d.eqs.iter().enumerate() {
        s.push_str(&latex_eq(e));
        s.push_str(if i + 1 < n { " \\\\\n" } else { "\n" });
    }
    s.push_str("\\end{aligned}");
    s
}

fn latex_eq(e: &Equation) -> String {
    let mut s = format!("{} &= {}", e.lhs.render(Format::Latex), e.rhs.render(Format::Latex));
    if let Some(c) = &e.cond {
        s.push_str(&format!(" && {}", c.render(Format::Latex)));
    }
    s
}

/// The same for a listing's `alignat*{2}`, whose second column follows
/// the equation instead of being pushed to the margin.
fn listing_eq(e: &Equation) -> String {
    let mut s = format!("{} &= {}", e.lhs.render(Format::Latex), e.rhs.render(Format::Latex));
    if let Some(c) = &e.cond {
        s.push_str(&format!(" &\\qquad& {}", c.render(Format::Latex)));
    }
    s
}

fn unicode_eq(e: &Equation) -> String {
    let mut s = format!("{} = {}", e.lhs.render(Format::Unicode), e.rhs.render(Format::Unicode));
    if let Some(c) = &e.cond {
        s.push_str(&format!("    ({})", c.render(Format::Unicode)));
    }
    s
}

fn join(xs: &[M], f: Format) -> String {
    let sep = if f == Format::Latex { "$, $" } else { ",  " };
    let body = xs.iter().map(|x| x.render(f)).collect::<Vec<_>>().join(sep);
    if f == Format::Latex { format!("${body}$") } else { body }
}

/// The listing of the typed program.
pub fn listing(l: &Listing, f: Format, sel: &Select, main: &str, standalone: bool) -> String {
    let mut out = String::new();
    match f {
        Format::Latex => {
            if standalone {
                out.push_str(PREAMBLE);
            }
            out.push_str(&format!("% presk dump --stage elab --format latex: {main} and its imports, as the equations\n% of the language reference's chapter 3. A fragment: it compiles with the reference's\n% preamble, whose macros \\code and \\den it uses (--standalone: a document).\n"));
        }
        Format::Unicode => out.push_str(&format!("-- presk dump --stage elab --format unicode: {main} and its imports, as the equations\n-- of the language reference's chapter 3\n")),
    }
    for u in &l.units {
        let mut body = String::new();
        for item in &u.items {
            match item {
                Item::Module { head, params, fns, .. } => {
                    let name = match head {
                        M::Braced(h, _) => h.render(Format::Unicode),
                        h => h.render(Format::Unicode),
                    };
                    if !sel.module(&name) {
                        continue;
                    }
                    let shown: Vec<&Display> = fns.iter().filter(|d| sel.function(d)).collect();
                    if shown.is_empty() && sel.function.is_some() {
                        continue;
                    }
                    match f {
                        Format::Latex => {
                            body.push_str(&format!("\n\\paragraph{{Module ${}$.}}", head.render(f)));
                            if !params.is_empty() {
                                body.push_str(&format!(" Its parameters and modules: {}.", join(params, f)));
                            }
                            body.push('\n');
                        }
                        Format::Unicode => {
                            body.push_str(&format!("\nmodule {}\n", head.render(f)));
                            if !params.is_empty() {
                                body.push_str(&format!("  parameters  {}\n", join(params, f)));
                            }
                        }
                    }
                    for d in shown {
                        body.push_str(&display(d, f, true));
                    }
                }
                Item::Fn(d) => {
                    if sel.module.is_some() || !sel.function(d) {
                        continue;
                    }
                    body.push_str(&display(d, f, false));
                }
            }
        }
        let facts = [("Constants", "constants", &u.consts), ("dimensions", "dimensions", &u.dims), ("types", "types", &u.types), ("asserted", "asserted", &u.asserts)];
        if body.is_empty() && (sel.module.is_some() || sel.function.is_some()) {
            continue;
        }
        match f {
            Format::Latex => {
                out.push_str(&format!("\n\\subsection*{{\\texttt{{{}}}}}\n", escape(&u.file)));
                let parts: Vec<String> = facts.iter().filter(|(_, _, xs)| !xs.is_empty()).map(|(w, _, xs)| format!("{w} {}", join(xs, f))).collect();
                if !parts.is_empty() {
                    let mut line = parts.join("; ");
                    if let Some(c) = line.get(..1) {
                        line = c.to_uppercase() + &line[1..];
                    }
                    out.push_str(&format!("{line}.\n"));
                }
            }
            Format::Unicode => {
                out.push_str(&format!("\n== {}\n", u.file));
                for (_, w, xs) in facts {
                    if !xs.is_empty() {
                        out.push_str(&format!("  {w}  {}\n", join(xs, f)));
                    }
                }
            }
        }
        out.push_str(&body);
    }
    if f == Format::Latex && standalone {
        out.push_str("\n\\end{document}\n");
    }
    out
}

fn display(d: &Display, f: Format, member: bool) -> String {
    let mut s = String::new();
    let uses = if d.effects.is_empty() { String::new() } else { format!(", uses {}", d.effects.join(", ")) };
    match f {
        Format::Latex => {
            let name = M::code(&d.name).render(f);
            let what = if member { format!("\n\\noindent ${name}$") } else { format!("\n\\paragraph{{Function ${name}$.}}") };
            s.push_str(&what);
            let of = if member { ", of" } else { " Of" };
            if d.of.is_empty() {
                s.push_str(&format!("{uses}:\n"));
            } else {
                s.push_str(&format!("{of} {}{uses}:\n", join(&d.of, f)));
            }
            s.push_str("\\begin{alignat*}{2}\n");
            let n = d.eqs.len();
            for (i, e) in d.eqs.iter().enumerate() {
                s.push_str(&listing_eq(e));
                s.push_str(if i + 1 < n { " \\\\\n" } else { "\n" });
            }
            s.push_str("\\end{alignat*}\n");
        }
        Format::Unicode => {
            let indent = if member { "  " } else { "" };
            s.push_str(&format!("\n{indent}{}", d.name));
            if !d.of.is_empty() {
                s.push_str(&format!(", of {}", join(&d.of, f)));
            }
            s.push_str(&format!("{uses}\n"));
            for e in &d.eqs {
                s.push_str(&format!("{indent}    {}\n", unicode_eq(e)));
            }
        }
    }
    s
}
