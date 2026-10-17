//! The math tree both entries build, and its two renderings.
//!
//! One tree, so that the LaTeX and the Unicode say the same thing: a node
//! is a form of the reference's chapter 3 (an operation applied, a sum over
//! an index set, the denotation of a function), and each rendering only
//! spells it. Parentheses come from the precedences below, never from the
//! entries.

/// The rendering asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// A LaTeX fragment in the reference's macros (`\code`, `\den`).
    Latex,
    /// The same formulas for a terminal: `∑`, `⟦⟧`, `x[i, k]`.
    Unicode,
}

/// An infix operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum B {
    Add,
    Sub,
    /// `×`: the product of two real tensors, element by element
    /// (definition `den-arith`).
    Mul,
    Div,
    /// `·`: the product of two dimensions (`H · HD`).
    DimMul,
    Rem,
    /// `∘`: a composition of modules.
    Compose,
}

/// A function of analysis, written as analysis writes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F {
    Exp,
    Ln,
    Tanh,
    Sin,
    Cos,
    Sqrt,
    Abs,
}

/// A relation between two sides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum R {
    Eq,
    Le,
    Lt,
    Ge,
    Gt,
    /// `x : τ`
    Has,
    /// `x += r`: a contribution to a cotangent (the reference's table
    /// `core-adjoint`).
    AddTo,
}

/// A tensor type's head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum T {
    Real,
    Idx,
    Bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum M {
    /// A name of the program: a variable, a parameter, a dimension, a
    /// constant.
    Name(String),
    /// A number as the source writes it.
    Num(String),
    /// A word of the language: an operation, a function's path, an axis.
    Code(String),
    /// An operation of chapter 3 applied, `\code{softmax axis}\ a\ x, m`:
    /// its name, its attributes (each group joined by commas), its
    /// operands.
    Op { name: String, attrs: Vec<Vec<M>>, args: Vec<M> },
    Bin(B, Box<M>, Box<M>),
    Neg(Box<M>),
    Pow(Box<M>, Box<M>),
    Fun(F, Box<M>),
    /// `f(a, b)`
    Apply(Box<M>, Vec<M>),
    /// `⟦f⟧`
    Den(Box<M>),
    /// `x[i, j]`
    Index(Box<M>, Vec<M>),
    /// `∑_{a, b} body`
    Sum(Vec<M>, Box<M>),
    /// `∑_{k=0}^{to} body`
    SumTo { var: String, to: Box<M>, body: Box<M> },
    /// Factors side by side: the product of an `ein`'s operands at an index.
    Prod(Vec<M>),
    Tuple(Vec<M>),
    /// `{n₁ = x₁, …}`: a parameter tree.
    Record(Vec<(String, M)>),
    /// `θ.w`
    Field(Box<M>, String),
    /// `(v₁, …) ↦ e`
    MapsTo(Vec<M>, Box<M>),
    /// `∇g`
    Nabla(Box<M>),
    /// `y_ℓ`
    Sub(Box<M>, Box<M>),
    /// `M{D, H}`: a module or a function and its dimensions.
    Braced(Box<M>, Vec<M>),
    Ellipsis,
    /// `a: n`, an axis and its length.
    AxisLen(String, Box<M>),
    /// `ℝ[a: n, …]`, `idx[V][…]`, `bool[…]`.
    Tensor(T, Option<Box<M>>, Vec<M>),
    Rel(R, Box<M>, Box<M>),
    /// Statements side by side: the contributions of one rule.
    Seq(Vec<M>),
    /// Forms written one after the other with nothing between, as an
    /// attribute is: `H\code{x}\mathit{HD}`.
    Cat(Vec<M>),
    /// A fixed form, written as the reference writes it: its LaTeX and its
    /// text.
    Raw { latex: String, text: String },
    /// A form this printer does not write, shown as its source text.
    Gap(String),
}

impl M {
    pub fn name(s: impl Into<String>) -> M {
        M::Name(s.into())
    }
    pub fn code(s: impl Into<String>) -> M {
        M::Code(s.into())
    }
    pub fn num(s: impl Into<String>) -> M {
        M::Num(s.into())
    }
    pub fn op(name: impl Into<String>, attrs: Vec<Vec<M>>, args: Vec<M>) -> M {
        M::Op { name: name.into(), attrs, args }
    }
    pub fn bin(b: B, l: M, r: M) -> M {
        M::Bin(b, Box::new(l), Box::new(r))
    }
    pub fn rel(r: R, l: M, rr: M) -> M {
        M::Rel(r, Box::new(l), Box::new(rr))
    }
    pub fn apply(f: M, args: Vec<M>) -> M {
        M::Apply(Box::new(f), args)
    }
    pub fn den(f: M) -> M {
        M::Den(Box::new(f))
    }
    pub fn index(x: M, i: Vec<M>) -> M {
        M::Index(Box::new(x), i)
    }
    pub fn subscript(x: M, s: M) -> M {
        M::Sub(Box::new(x), Box::new(s))
    }

    /// Whether the tree holds a gap anywhere.
    pub fn has_gap(&self) -> bool {
        let mut found = false;
        self.walk(&mut |m| found |= matches!(m, M::Gap(_)));
        found
    }

    fn walk(&self, f: &mut dyn FnMut(&M)) {
        f(self);
        match self {
            M::Op { attrs, args, .. } => {
                attrs.iter().flatten().for_each(|m| m.walk(f));
                args.iter().for_each(|m| m.walk(f));
            }
            M::Bin(_, a, b) | M::Pow(a, b) | M::Sub(a, b) | M::Rel(_, a, b) => {
                a.walk(f);
                b.walk(f);
            }
            M::Neg(a) | M::Fun(_, a) | M::Den(a) | M::Nabla(a) | M::Field(a, _) | M::AxisLen(_, a) => a.walk(f),
            M::SumTo { to, body, .. } => {
                to.walk(f);
                body.walk(f);
            }
            M::Apply(a, xs) | M::Index(a, xs) | M::Braced(a, xs) | M::MapsTo(xs, a) | M::Sum(xs, a) => {
                a.walk(f);
                xs.iter().for_each(|m| m.walk(f));
            }
            M::Prod(xs) | M::Tuple(xs) | M::Seq(xs) | M::Cat(xs) => xs.iter().for_each(|m| m.walk(f)),
            M::Record(fs) => fs.iter().for_each(|(_, m)| m.walk(f)),
            M::Tensor(_, b, xs) => {
                if let Some(b) = b {
                    b.walk(f);
                }
                xs.iter().for_each(|m| m.walk(f));
            }
            M::Name(_) | M::Num(_) | M::Code(_) | M::Ellipsis | M::Gap(_) | M::Raw { .. } => {}
        }
    }

    /// How tightly the form binds: an operand of lower precedence than its
    /// place asks is parenthesized.
    fn prec(&self, f: Format) -> u8 {
        match self {
            M::Bin(B::Div, ..) if f == Format::Latex => 100,
            M::Num(t) => {
                if t.starts_with('-') {
                    85
                } else if let Some((m, _)) = sci(t) {
                    if m == "1" { 95 } else { 60 }
                } else {
                    100
                }
            }
            M::Pow(..) => 95,
            M::Fun(F::Sqrt | F::Abs, _) => 100,
            M::Fun(..) => 90,
            M::Neg(_) => 85,
            M::Bin(B::Add | B::Sub, ..) => 50,
            M::Bin(B::Compose, ..) => 40,
            M::Bin(..) | M::Prod(_) => 60,
            M::Sum(..) | M::SumTo { .. } => 45,
            M::Op { .. } => 30,
            M::Rel(..) => 20,
            M::MapsTo(..) => 10,
            M::Seq(_) => 5,
            _ => 100,
        }
    }

    /// The tree in the format.
    pub fn render(&self, f: Format) -> String {
        self.go(f, 0)
    }

    fn go(&self, f: Format, min: u8) -> String {
        let s = self.spell(f);
        if self.prec(f) < min {
            match f {
                Format::Latex => format!("({s})"),
                Format::Unicode => format!("({s})"),
            }
        } else {
            s
        }
    }

    fn spell(&self, f: Format) -> String {
        let latex = f == Format::Latex;
        // In a list of several, an operation is parenthesized: its own
        // operands are separated by commas too.
        let list = |xs: &[M], min: u8| {
            let several = xs.len() > 1;
            xs.iter().map(|x| x.go(f, if several && matches!(x, M::Op { .. }) { min.max(31) } else { min })).collect::<Vec<_>>().join(", ")
        };
        match self {
            M::Name(n) => {
                if latex {
                    latex_name(n)
                } else {
                    n.clone()
                }
            }
            M::Num(t) => {
                if latex {
                    latex_num(t)
                } else {
                    t.clone()
                }
            }
            M::Code(c) => {
                if latex {
                    format!("\\code{{{}}}", escape(c))
                } else {
                    c.clone()
                }
            }
            M::Op { name, attrs, args } => {
                let sep = if latex { "\\ " } else { " " };
                let mut s = if latex { format!("\\code{{{}}}", escape(name)) } else { name.clone() };
                for group in attrs {
                    s.push_str(sep);
                    s.push_str(&group.iter().map(|x| x.go(f, 95)).collect::<Vec<_>>().join(", "));
                }
                if !args.is_empty() {
                    s.push_str(sep);
                    s.push_str(&list(args, 95));
                }
                s
            }
            M::Bin(b, l, r) => {
                let (lp, rp) = match b {
                    B::Add | B::Sub => (50, 51),
                    B::Compose => (40, 41),
                    _ => (60, 61),
                };
                let op = match (b, latex) {
                    (B::Add, _) => "+",
                    (B::Sub, true) => "-",
                    (B::Sub, false) => "−",
                    (B::Mul, true) => "\\times",
                    (B::Mul, false) => "×",
                    (B::Div, _) => "/",
                    (B::DimMul, true) => "\\cdot",
                    (B::DimMul, false) => "·",
                    (B::Rem, true) => "\\bmod",
                    (B::Rem, false) => "mod",
                    (B::Compose, true) => "\\circ",
                    (B::Compose, false) => "∘",
                };
                if latex && *b == B::Div {
                    return format!("\\frac{{{}}}{{{}}}", l.go(f, 0), r.go(f, 0));
                }
                // `a / b × c` read either way: the left operand of a
                // product or a quotient is parenthesized when it is the
                // other one.
                let lp = match (b, &**l) {
                    (B::Mul, M::Bin(B::Div, ..)) | (B::Div, M::Bin(B::Mul, ..)) if !latex => 61,
                    _ => lp,
                };
                format!("{} {op} {}", l.go(f, lp), r.go(f, rp))
            }
            M::Neg(x) => format!("{}{}", if latex { "-" } else { "−" }, x.go(f, 86)),
            M::Pow(b, e) => {
                let base = b.go(f, 100);
                if latex {
                    format!("{base}^{{{}}}", e.go(f, 0))
                } else {
                    match &**e {
                        M::Num(t) if t.chars().all(|c| c.is_ascii_digit()) => format!("{base}{}", t.chars().map(superscript).collect::<String>()),
                        e => format!("{base}^({})", e.go(f, 0)),
                    }
                }
            }
            M::Fun(F::Sqrt, x) => {
                if latex {
                    format!("\\sqrt{{{}}}", x.go(f, 0))
                } else if x.prec(f) >= 100 && !matches!(**x, M::Apply(..) | M::Index(..)) {
                    format!("√{}", x.go(f, 0))
                } else {
                    format!("√({})", x.go(f, 0))
                }
            }
            M::Fun(F::Abs, x) => {
                if latex {
                    format!("\\lvert {} \\rvert", x.go(f, 0))
                } else {
                    format!("|{}|", x.go(f, 0))
                }
            }
            M::Fun(g, x) => {
                let name = match g {
                    F::Exp => "exp",
                    F::Ln => "ln",
                    F::Tanh => "tanh",
                    F::Sin => "sin",
                    F::Cos => "cos",
                    F::Sqrt | F::Abs => unreachable!("written above"),
                };
                let head = if latex { format!("\\{name}") } else { name.to_string() };
                if matches!(**x, M::Name(_) | M::Num(_)) && x.prec(f) >= 100 {
                    format!("{head} {}", x.go(f, 0))
                } else {
                    format!("{head}({})", x.go(f, 0))
                }
            }
            M::Apply(g, xs) => format!("{}({})", g.go(f, 100), list(xs, 0)),
            M::Den(x) => {
                if latex {
                    format!("\\den{{{}}}", x.go(f, 0))
                } else {
                    format!("⟦{}⟧", x.go(f, 0))
                }
            }
            M::Index(x, is) => format!("{}[{}]", x.go(f, 100), list(is, 0)),
            M::Sum(over, body) => {
                if latex {
                    format!("\\sum_{{{}}} {}", list(over, 0), body.go(f, 60))
                } else {
                    format!("∑_{{{}}} {}", list(over, 0), body.go(f, 60))
                }
            }
            M::SumTo { var, to, body } => {
                let v = if latex { latex_name(var) } else { var.clone() };
                if latex {
                    format!("\\sum_{{{v}=0}}^{{{}}} {}", to.go(f, 0), body.go(f, 60))
                } else {
                    format!("∑_({v}=0..{}) {}", to.go(f, 0), body.go(f, 60))
                }
            }
            M::Prod(xs) => xs.iter().map(|x| x.go(f, 61)).collect::<Vec<_>>().join(if latex { "\\, " } else { " " }),
            M::Tuple(xs) => format!("({})", list(xs, 0)),
            M::Record(fs) => {
                let body = fs.iter().map(|(n, x)| format!("{} = {}", if latex { latex_name(n) } else { n.clone() }, x.go(f, 0))).collect::<Vec<_>>().join(", ");
                if latex { format!("\\{{{body}\\}}") } else { format!("{{{body}}}") }
            }
            M::Field(x, n) => format!("{}.{}", x.go(f, 100), if latex { latex_name(n) } else { n.clone() }),
            M::MapsTo(ps, body) => {
                let head = if ps.len() == 1 { ps[0].go(f, 100) } else { format!("({})", list(ps, 0)) };
                format!("{head} {} {}", if latex { "\\mapsto" } else { "↦" }, body.go(f, 11))
            }
            M::Nabla(x) => {
                let arg = x.go(f, 100);
                if latex { format!("\\nabla {arg}") } else { format!("∇{arg}") }
            }
            M::Sub(x, s) => {
                let base = x.go(f, 100);
                let s = s.go(f, 0);
                if latex {
                    format!("{base}_{{{s}}}")
                } else if s.chars().all(|c| c.is_ascii_digit()) {
                    format!("{base}{}", s.chars().map(subscript).collect::<String>())
                } else if s.chars().count() == 1 {
                    format!("{base}_{s}")
                } else {
                    format!("{base}_({s})")
                }
            }
            M::Braced(x, xs) => {
                if latex {
                    format!("{}\\{{{}\\}}", x.go(f, 100), list(xs, 0))
                } else {
                    format!("{}{{{}}}", x.go(f, 100), list(xs, 0))
                }
            }
            M::Ellipsis => if latex { "\\ldots".into() } else { "…".into() },
            M::AxisLen(a, n) => {
                if latex {
                    format!("\\code{{{}}}{{:}}\\,{}", escape(a), n.go(f, 0))
                } else {
                    format!("{a}: {}", n.go(f, 0))
                }
            }
            M::Tensor(t, bound, axes) => {
                let head = match (t, latex) {
                    (T::Real, true) => "\\mathbb{R}".to_string(),
                    (T::Real, false) => "ℝ".to_string(),
                    (T::Idx, true) => "\\code{idx}".to_string(),
                    (T::Idx, false) => "idx".to_string(),
                    (T::Bool, true) => "\\code{bool}".to_string(),
                    (T::Bool, false) => "bool".to_string(),
                };
                let bound = bound.as_ref().map(|b| format!("[{}]", b.go(f, 0))).unwrap_or_default();
                format!("{head}{bound}[{}]", list(axes, 0))
            }
            M::Rel(r, l, rr) => {
                let op = match (r, latex) {
                    (R::Eq, _) => "=",
                    (R::Le, true) => "\\le",
                    (R::Le, false) => "≤",
                    (R::Lt, _) => "<",
                    (R::Ge, true) => "\\ge",
                    (R::Ge, false) => "≥",
                    (R::Gt, _) => ">",
                    (R::Has, _) => ":",
                    (R::AddTo, true) => "\\mathrel{{+}{=}}",
                    (R::AddTo, false) => "+=",
                };
                format!("{} {op} {}", l.go(f, 20), rr.go(f, 21))
            }
            M::Cat(xs) => xs.iter().map(|x| x.go(f, 100)).collect::<String>(),
            M::Seq(xs) => xs.iter().map(|x| x.go(f, 6)).collect::<Vec<_>>().join(if latex { " \\quad " } else { ";  " }),
            M::Raw { latex: l, text } => {
                if latex {
                    l.clone()
                } else {
                    text.clone()
                }
            }
            M::Gap(text) => {
                if latex {
                    format!("\\text{{\\texttt{{{}}}}}", escape(text))
                } else {
                    format!("«{text}»")
                }
            }
        }
    }
}

/// A number in scientific notation, as its mantissa and its exponent.
fn sci(t: &str) -> Option<(&str, &str)> {
    let (m, e) = t.split_once(['e', 'E'])?;
    let m = m.strip_suffix(".0").unwrap_or(m);
    Some((m, e.strip_prefix('+').unwrap_or(e)))
}

/// `1e-5` as `10^{-5}`, `3e-4` as `3 \cdot 10^{-4}`: the number the source
/// writes, in the notation of a page.
fn latex_num(t: &str) -> String {
    match sci(t) {
        Some(("1", e)) => format!("10^{{{e}}}"),
        Some((m, e)) => format!("{m} \\cdot 10^{{{e}}}"),
        None => t.to_string(),
    }
}

/// The characters a `\texttt` cannot hold as they are.
pub fn escape(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        match c {
            '_' | '%' | '#' | '&' | '{' | '}' | '$' => {
                out.push('\\');
                out.push(c);
            }
            '^' => out.push_str("\\^{}"),
            '~' => out.push_str("\\~{}"),
            '\\' => out.push_str("\\textbackslash{}"),
            c => out.push(c),
        }
    }
    out
}

fn greek(c: char) -> Option<&'static str> {
    Some(match c {
        'α' => "\\alpha",
        'β' => "\\beta",
        'γ' => "\\gamma",
        'δ' => "\\delta",
        'ε' => "\\varepsilon",
        'ϵ' => "\\epsilon",
        'ζ' => "\\zeta",
        'η' => "\\eta",
        'θ' => "\\theta",
        'ι' => "\\iota",
        'κ' => "\\kappa",
        'λ' => "\\lambda",
        'μ' => "\\mu",
        'ν' => "\\nu",
        'ξ' => "\\xi",
        'π' => "\\pi",
        'ρ' => "\\rho",
        'σ' => "\\sigma",
        'τ' => "\\tau",
        'υ' => "\\upsilon",
        'φ' => "\\varphi",
        'ϕ' => "\\phi",
        'χ' => "\\chi",
        'ψ' => "\\psi",
        'ω' => "\\omega",
        'ℓ' => "\\ell",
        'Γ' => "\\Gamma",
        'Δ' => "\\Delta",
        'Θ' => "\\Theta",
        'Λ' => "\\Lambda",
        'Ξ' => "\\Xi",
        'Π' => "\\Pi",
        'Σ' => "\\Sigma",
        'Υ' => "\\Upsilon",
        'Φ' => "\\Phi",
        'Ψ' => "\\Psi",
        'Ω' => "\\Omega",
        _ => return None,
    })
}

/// A letter with a circumflex, precomposed (`ŷ`), and its base.
fn hatted(c: char) -> Option<char> {
    Some(match c {
        'â' => 'a',
        'ĉ' => 'c',
        'ê' => 'e',
        'ĝ' => 'g',
        'ĥ' => 'h',
        'î' => 'i',
        'ĵ' => 'j',
        'ô' => 'o',
        'ŝ' => 's',
        'û' => 'u',
        'ŵ' => 'w',
        'ŷ' => 'y',
        'ẑ' => 'z',
        _ => return None,
    })
}

/// A name of the program as mathematics: one letter in italic (a Greek one
/// by its macro), a hat kept, trailing digits as a subscript (`k1` is
/// $k_{1}$, `σ2` is $\sigma_{2}$), primes kept, a longer name in
/// `\mathit`; `%3`, a binding PSK-Core does not name, as `\%3`.
pub fn latex_name(n: &str) -> String {
    if let Some(k) = n.strip_prefix('%') {
        return format!("\\%{k}");
    }
    if n == "_" {
        return "\\_".into();
    }
    let primes = n.len() - n.trim_end_matches('\'').len();
    let core = n.trim_end_matches('\'');
    let digits = core.len() - core.trim_end_matches(|c: char| c.is_ascii_digit()).len();
    let (base, digits) = if digits > 0 && digits < core.len() { (&core[..core.len() - digits], &core[core.len() - digits..]) } else { (core, "") };
    let mut chars = base.chars();
    let one = |c: char| -> String {
        if let Some(g) = greek(c) {
            g.to_string()
        } else if let Some(b) = hatted(c) {
            format!("\\hat{{{b}}}")
        } else {
            c.to_string()
        }
    };
    let mut s = match (chars.next(), chars.next()) {
        (Some(c), None) => one(c),
        (Some(c), Some('\u{302}')) if chars.next().is_none() => format!("\\hat{{{}}}", one(c)),
        _ => {
            let inner: String = base.chars().map(|c| if c == '_' { "\\_".to_string() } else { one(c) }).collect();
            format!("\\mathit{{{inner}}}")
        }
    };
    if !digits.is_empty() {
        s = format!("{s}_{{{digits}}}");
    }
    for _ in 0..primes {
        s.push('\'');
    }
    s
}

fn superscript(c: char) -> char {
    match c {
        '0' => '⁰',
        '1' => '¹',
        '2' => '²',
        '3' => '³',
        '4' => '⁴',
        '5' => '⁵',
        '6' => '⁶',
        '7' => '⁷',
        '8' => '⁸',
        '9' => '⁹',
        c => c,
    }
}

fn subscript(c: char) -> char {
    match c {
        '0' => '₀',
        '1' => '₁',
        '2' => '₂',
        '3' => '₃',
        '4' => '₄',
        '5' => '₅',
        '6' => '₆',
        '7' => '₇',
        '8' => '₈',
        '9' => '₉',
        c => c,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_written_as_mathematics() {
        assert_eq!(latex_name("x"), "x");
        assert_eq!(latex_name("γ"), "\\gamma");
        assert_eq!(latex_name("σ2"), "\\sigma_{2}");
        assert_eq!(latex_name("k1"), "k_{1}");
        assert_eq!(latex_name("ŷ"), "\\hat{y}");
        assert_eq!(latex_name("ms"), "\\mathit{ms}");
        assert_eq!(latex_name("x'"), "x'");
        assert_eq!(latex_name("%3"), "\\%3");
        assert_eq!(latex_name("HD"), "\\mathit{HD}");
        assert_eq!(latex_name("lm_head"), "\\mathit{lm\\_head}");
    }

    #[test]
    fn numbers_are_written_as_the_source_writes_them() {
        assert_eq!(M::num("1e-5").render(Format::Latex), "10^{-5}");
        assert_eq!(M::num("3e-4").render(Format::Latex), "3 \\cdot 10^{-4}");
        assert_eq!(M::num("0.5").render(Format::Latex), "0.5");
        assert_eq!(M::num("1e-5").render(Format::Unicode), "1e-5");
    }

    #[test]
    fn an_operation_is_parenthesized_as_an_operand() {
        let dot = M::op("dot", vec![], vec![M::name("x"), M::name("w")]);
        let e = M::bin(B::Add, dot, M::name("b"));
        assert_eq!(e.render(Format::Latex), "(\\code{dot}\\ x, w) + b");
        assert_eq!(e.render(Format::Unicode), "(dot x, w) + b");
        let sq = M::Pow(Box::new(M::bin(B::Sub, M::name("x"), M::name("y"))), Box::new(M::num("2")));
        assert_eq!(M::op("mean", vec![], vec![sq]).render(Format::Unicode), "mean (x − y)²");
    }
}
