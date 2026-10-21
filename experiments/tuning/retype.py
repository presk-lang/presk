#!/usr/bin/env python3
"""The C program's floating-point variables, and the program rewritten for
an assignment of them to `float` (D154): the stand-in for TypeForge, which
CRAFT's variable mode and FloatSmith use to rewrite a source.

  variables EXPORT                     the variables, as CRAFT's initial
                                       configuration (JSON on stdout)
  apply EXPORT ASSIGN.json DIR         the program of EXPORT with the
                                       variables ASSIGN.json lists
                                       (`{"float": [handle, …]}`, or a CRAFT
                                       configuration) in `float`, into DIR
  realization EXPORT ASSIGN.json       the realization of the same
                                       assignment, as presk runs it (JSON
                                       on stdout, the sites the C program
                                       types apart on stderr; see
                                       `realization`)

EXPORT is `presk export --target c` of an entry realized in `f64`. The
standard library only, as experiments/margins.py.

The variables, each named by a handle `FUNCTION:NAME`:
- every buffer `model.c` declares as `double *` and allocates from the
  arena, under the comment that names its site; the program's output
  (`result`) among them;
- the pointers one assignment ties together form one variable: the fold's
  state `s_0` points to the buffer before the loop and then to the loop's
  two buffers, so they share a type, as TypeForge's type-dependence sets
  make them;
- every local `double` of a kernel in `kernels.c`: its accumulators (`s0`,
  the register tile `v0`…`v3`) and the values it holds in registers.

What the rewrite does, for a given set of variables in `float`:
- a declaration's `double` becomes `float`, and its `sizeof(double)` with it;
- a function, host or kernel, is specialized by the types of the pointers
  passed to it, as a C++ template would be: a kernel called with a `float *`
  where another call passes a `double *` is copied under a new name with
  that parameter's type changed, its body unchanged, and every call names
  the copy its arguments select. A host function likewise. A kernel's local
  takes the type the assignment gives it in every copy (it is one
  declaration of the source);
- a `presk_fill` of a `float` buffer writes `PRESK_F32`;
- the output, if `float`, is declared so in `model.h`'s result struct and
  written with `PRESK_F32` by `main.c`, which converts it to the file's
  `f64`.

What it does not retype: the parameters (`theta_*`, read by `main.c` from
the weights file as `f64`, are the program's inputs), the indices
(`int64_t`), the literals (`0.0`, `64.0` stay `double`, so an expression
mixing a `float` with one is computed in `double`, as C's arithmetic
conversions say), and the transcendental functions (`presk_exp`,
`presk_erf` and `sqrt` take and return `double`; a `float` argument is
widened). No `bf16`: plain C has no such type. The arena's size stays the
export's, an upper bound once buffers narrow.
"""

import json
import os
import re
import shutil
import sys

FUNC_RE = re.compile(r"^(static )?void (presk_\w+)\((.*)\)$")
KERN_RE = re.compile(r"^PRESK_KERNEL void (k\d+)\((.*)\) \{$")
DECL_RE = re.compile(r"^(\s*)(const )?double \*(\w+) = (.+);$")
BUFS_RE = re.compile(r"^(\s*)double \*(\w+)\[(\d+)\] = \{(.+)\};$")
SITE_RE = re.compile(r"^\s*/\* (\S+/%\d+)(?: \((.*)\))? \*/$")
KCALL_RE = re.compile(r"\b(k\d+)\((.*)\);")
HCALL_RE = re.compile(r"\b(presk_\w+_apply\w*)\(arena, (.*)\);")
FILL_RE = re.compile(r"presk_fill\((\w+), PRESK_F64,")
ASSIGN_RE = re.compile(r"^\s*(\w+) = (\w+)(\[[^\]]*\])?;$")


def split_args(text):
    """Arguments at the top level of a call's parentheses."""
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
            continue
        depth += ch in "(["
        depth -= ch in ")]"
        cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def base_name(expr):
    m = re.match(r"\s*(\w+)", expr)
    return m.group(1) if m else expr


def param(text):
    """`const double *restrict in0` → (const, base type, name)."""
    m = re.match(r"^(const )?(\w+) \*(?:restrict )?(\w+)$", text.strip())
    if m:
        return (bool(m.group(1)), m.group(2), m.group(3))
    m = re.match(r"^(\w+) (\w+)$", text.strip())
    return (False, m.group(1), m.group(2)) if m else (False, None, text.strip())


class Function:
    def __init__(self, name, params, lines, start, kernel):
        self.name, self.params, self.lines, self.start, self.kernel = name, params, lines, start, kernel


def parse_functions(path, kernel):
    text = open(path).read().split("\n")
    funcs, order, i = {}, [], 0
    pieces = []  # ("text", line) or ("func", name)
    while i < len(text):
        line = text[i]
        m = (KERN_RE if kernel else FUNC_RE).match(line)
        if m:
            name, params = (m.group(1), m.group(2)) if kernel else (m.group(2), m.group(3))
            j = i + (1 if kernel else 2)  # a kernel's `{` ends its line; a host function's is the next
            body = []
            while text[j] != "}":
                body.append(text[j])
                j += 1
            funcs[name] = Function(name, [param(p) for p in split_args(params)], body, i + 1, kernel)
            funcs[name].header = line
            order.append(name)
            pieces.append(("func", name))
            i = j + 1
            continue
        pieces.append(("text", line))
        i += 1
    return funcs, order, pieces


class Program:
    def __init__(self, export):
        self.export = export
        self.host, self.host_order, self.host_pieces = parse_functions(os.path.join(export, "model.c"), False)
        self.kern, self.kern_order, self.kern_pieces = parse_functions(os.path.join(export, "kernels.c"), True)
        self.variables = []  # dicts: handle, name, function, file, line, site, label, members
        self.of = {}  # (function, local name) -> handle
        for fname in self.host_order:
            self._host_vars(self.host[fname])
        for kname in self.kern_order:
            self._kernel_vars(self.kern[kname])

    def _add(self, func, name, file, line, site, label, members):
        h = f"{func}:{name}"
        self.variables.append({"handle": h, "name": name, "function": func, "file": file, "line": line,
                               "site": site, "label": label, "members": members})
        for m in members:
            self.of[(func, m)] = h

    def _host_vars(self, f):
        site, label, alias = None, None, {}
        decls = []
        for k, line in enumerate(f.lines):
            m = SITE_RE.match(line)
            if m:
                site, label = m.group(1), m.group(2)
                continue
            m = DECL_RE.match(line)
            if m:
                name, init = m.group(3), m.group(4)
                if init.startswith("presk_alloc("):
                    decls.append((name, f.start + 1 + k + 1, site, label))
                elif re.fullmatch(r"\w+", init):
                    alias[name] = init  # a pointer that starts as another buffer
                continue
            m = BUFS_RE.match(line)
            if m:
                decls.append((m.group(2), f.start + 1 + k + 1, site, label))
                continue
            m = ASSIGN_RE.match(line)
            if m and m.group(1) in alias:
                alias.setdefault("__joins", []).append((m.group(1), m.group(2)))
        # union the pointers one assignment ties together
        group = {d[0]: d[0] for d in decls}

        def find(x):
            while group.get(x, x) != x:
                x = group[x]
            return x

        for a, b in alias.items():
            if a == "__joins":
                continue
            group[a] = find(b)
        for a, b in alias.get("__joins", []):
            ra, rb = find(a), find(b)
            if ra != rb:
                group[rb] = ra
        roots = {}
        for name, line, s, l in decls:
            roots.setdefault(find(name), []).append((name, line, s, l))
        for name, line, s, l in decls:
            r = find(name)
            if roots[r][0][0] != name:
                continue
            members = [n for n, _, _, _ in roots[r]] + [a for a in alias if a != "__joins" and find(a) == r]
            label_name = "+".join(n for n, _, _, _ in roots[r])
            sites = ", ".join(x for _, _, x, _ in roots[r] if x)
            self._add(f.name, label_name, "model.c", line, sites, l, members)

    def _kernel_vars(self, f):
        for k, line in enumerate(f.lines):
            m = re.match(r"^    double (.+);$", line)
            if not m:
                continue
            for d in split_args(m.group(1)):
                name = re.match(r"(\w+)", d).group(1)
                self._add(f.name, name, "kernels.c", f.start + 1 + k, None, None, [name])

    # ---- what each variable holds ----------------------------------------------

    def _writes(self, f):
        """(output argument's base name, the site that writes it) per call
        of f: a kernel writes the site its comment names, a host function
        the site its own `out` receives."""
        site, out = None, []
        for line in f.lines:
            m = SITE_RE.match(line)
            if m:
                site = m.group(1)
            m = FILL_RE.search(line)
            if m:
                out.append((m.group(1), site))
            m = KCALL_RE.search(line)
            if m:
                k = self.kern[m.group(1)]
                ptrs = [p for p in k.params if p[1] in ("double", "int64_t") and p[2] not in ("lo", "hi")]
                for p, a in zip(ptrs, split_args(m.group(2))):
                    if not p[0]:
                        out.append((base_name(a), site))
            m = HCALL_RE.search(line)
            if m:
                g = self.host[m.group(1)]
                gparams = [p for p in g.params if p[2] != "arena"]
                for p, a in zip(gparams, split_args(m.group(2))):
                    if not p[0]:
                        out.append((base_name(a), ("result", m.group(1))))
        return out

    def result_site(self, fname):
        for name, s in self._writes(self.host[fname]):
            if name == "out":
                return self.result_site(s[1]) if isinstance(s, tuple) else s
        return None

    def value_sites(self):
        """handle → the Presk sites whose values the variable holds: a
        buffer the program's site stores, or a callee's result. Empty for a
        kernel's local, which holds no stored value."""
        out = {v["handle"]: [] for v in self.variables}
        for fname in self.host_order:
            for name, s in self._writes(self.host[fname]):
                h = self.of.get((fname, name))
                if h is None:
                    continue
                site = self.result_site(s[1]) if isinstance(s, tuple) else s
                if site and site not in out[h]:
                    out[h].append(site)
        return out

    # ---- the rewrite ----------------------------------------------------------

    def apply(self, floats, out):
        self.floats = set(floats)
        unknown = self.floats - {v["handle"] for v in self.variables}
        if unknown:
            raise SystemExit(f"retype: no such variable: {sorted(unknown)}")
        self.host_specs = {}  # name -> {sig: new name}
        self.kern_specs = {}
        self.emitted_host = {}  # (name, sig) -> lines
        self.out_float = self.ty("presk_logits", "result") == "float"
        self._host_spec("presk_logits", None)
        os.makedirs(out, exist_ok=True)
        for f in os.listdir(self.export):
            p = os.path.join(self.export, f)
            if os.path.isfile(p) and f not in ("model.c", "kernels.c", "model.h", "main.c", "model"):
                shutil.copy(p, os.path.join(out, f))
        if os.path.isdir(os.path.join(self.export, "conformance")):
            shutil.copytree(os.path.join(self.export, "conformance"), os.path.join(out, "conformance"), dirs_exist_ok=True)
        self._write_host(os.path.join(out, "model.c"))
        self._write_kernels(os.path.join(out, "kernels.c"))
        h = open(os.path.join(self.export, "model.h")).read()
        c = open(os.path.join(self.export, "main.c")).read()
        if self.out_float:
            h = h.replace("const double *result;", "const float *result;")
            c = c.replace("OUTPUTS_DTYPES[] = {PRESK_F64};", "OUTPUTS_DTYPES[] = {PRESK_F32};")
        open(os.path.join(out, "model.h"), "w").write(h)
        open(os.path.join(out, "main.c"), "w").write(c)

    def ty(self, func, name):
        h = self.of.get((func, name))
        return "float" if h in self.floats else "double"

    def _sig_name(self, name, sig):
        return name if all(t == "double" for t in sig if t in ("double", "float")) else name + "_" + "".join(
            {"double": "d", "float": "f"}[t] for t in sig if t in ("double", "float"))

    def _host_spec(self, fname, sig):
        f = self.host[fname]
        reals = [p for p in f.params if p[1] == "double"]
        if sig is None:
            sig = tuple("double" for _ in reals)
        key = (fname, sig)
        if key in self.emitted_host:
            return self._sig_name(fname, sig)
        self.emitted_host[key] = None  # in progress
        env = {}
        for p, t in zip(reals, sig):
            env[p[2]] = t
        lines = []
        for line in f.lines:
            m = DECL_RE.match(line)
            if m and (f.name, m.group(3)) in self.of:
                t = self.ty(f.name, m.group(3))
                env[m.group(3)] = t
                if t == "float":
                    line = line.replace("double *", "float *", 1).replace("sizeof(double)", "sizeof(float)")
                lines.append(line)
                continue
            if m:
                env[m.group(3)] = "double"
            m = BUFS_RE.match(line)
            if m:
                t = self.ty(f.name, m.group(2))
                env[m.group(2)] = t
                if t == "float":
                    line = line.replace("double *", "float *", 1).replace("sizeof(double)", "sizeof(float)")
                lines.append(line)
                continue
            m = FILL_RE.search(line)
            if m and env.get(m.group(1)) == "float":
                line = line.replace("PRESK_F64", "PRESK_F32")
            m = KCALL_RE.search(line)
            if m:
                k = self.kern[m.group(1)]
                args = split_args(m.group(2))
                ptrs = [p for p in k.params if p[1] in ("double", "int64_t") and p[2] not in ("lo", "hi")]
                ksig = tuple(env.get(base_name(a), "double") if p[1] == "double" else "int64_t" for p, a in zip(ptrs, args))
                line = line[: m.start(1)] + self._kern_spec(m.group(1), ksig) + line[m.end(1):]
            m = HCALL_RE.search(line)
            if m:
                g = self.host[m.group(1)]
                args = split_args(m.group(2))
                gparams = [p for p in g.params if p[2] != "arena"]
                gsig = tuple(env.get(base_name(a), "double") for p, a in zip(gparams, args) if p[1] == "double")
                line = line[: m.start(1)] + self._host_spec(m.group(1), gsig) + line[m.end(1):]
            lines.append(line)
        header = f.header
        if any(t == "float" for t in sig):
            parts = []
            it = iter(sig)
            for p in f.params:
                if p[1] == "double":
                    t = next(it)
                    parts.append(("const " if p[0] else "") + f"{t} *{p[2]}")
                else:
                    parts.append(("const " if p[0] else "") + f"{p[1]} *{p[2]}")
            header = f"static void {self._sig_name(fname, sig)}({', '.join(parts)})"
        self.emitted_host[key] = [header, "{"] + lines + ["}"]
        return self._sig_name(fname, sig)

    def _kern_spec(self, kname, sig):
        self.kern_specs.setdefault(kname, set()).add(sig)
        return self._sig_name(kname, sig)

    def _write_host(self, path):
        out = []
        for kind, x in self.host_pieces:
            if kind == "text":
                out.append(x)
                continue
            specs = [(k, v) for k, v in self.emitted_host.items() if k[0] == x]
            if not specs:
                out.append(f"/* {x}: not called by this assignment */")
                out.append("")
                continue
            for n, (_, body) in enumerate(specs):
                if n:
                    out.append("")
                out.extend(body)
        open(path, "w").write("\n".join(out))

    def _write_kernels(self, path):
        out = []
        for kind, x in self.kern_pieces:
            if kind == "text":
                out.append(x)
                continue
            f = self.kern[x]
            sigs = sorted(self.kern_specs.get(x, ()))
            if not sigs:
                out.append(f"/* {x}: not launched by this assignment */")
                continue
            for n, sig in enumerate(sigs):
                if n:
                    out.append("")
                parts, it = [], iter(sig)
                for p in f.params:
                    if p[1] == "double":
                        parts.append(("const " if p[0] else "") + f"{next(it)} *restrict {p[2]}")
                    elif p[1] == "int64_t" and p[2] not in ("lo", "hi"):
                        next(it)
                        parts.append(("const " if p[0] else "") + f"int64_t *restrict {p[2]}")
                    else:
                        parts.append(f"{p[1]} {p[2]}")
                out.append(f"PRESK_KERNEL void {self._sig_name(x, sig)}({', '.join(parts)}) {{")
                for line in f.lines:
                    m = re.match(r"^    double (.+);$", line)
                    if m:
                        ds = split_args(m.group(1))
                        types = [self.ty(x, re.match(r"(\w+)", d).group(1)) for d in ds]
                        if all(t == "double" for t in types):
                            out.append(line)
                        else:
                            out.extend(f"    {t} {d};" for t, d in zip(types, ds))
                        continue
                    out.append(line)
                out.append("}")
        open(path, "w").write("\n".join(out))


def craft_config(prog):
    """CRAFT's initial configuration in variable mode: one
    `change_var_basetype` action per variable, its scope the C function,
    its source the file and line, labelled by its site."""
    actions = []
    for i, v in enumerate(prog.variables):
        a = {"uid": str(i + 1), "action": "change_var_basetype", "handle": v["handle"], "name": v["name"],
             "scope": f"function:<{v['function']}>", "source_info": f"{v['file']}:{v['line']}", "to_type": "float"}
        if v["site"]:
            a["labels"] = [v["site"]]
        actions.append(a)
    return {"version": "1", "tool_id": "presk-retype", "source_files": ["model.c", "kernels.c"], "actions": actions}


def realization(prog, floats):
    """The realization the compiler runs for an assignment of the C
    program's variables: a site in `f32` when every variable holding its
    values is `float`, in `f64` otherwise; the parameters in `f64`, as the
    program reads them. A kernel's local has no site: the realization
    computes a site in its own type, where the C program may accumulate a
    `float` buffer in `double` or the reverse. Also which sites the C
    program types apart (some of their variables `float`, some `double`)."""
    floats = set(floats)
    held = {}
    for h, sites in prog.value_sites().items():
        for s in sites:
            held.setdefault(s, []).append(h in floats)
    storage = {s: "f32" for s, f in sorted(held.items()) if all(f)}
    split = sorted(s for s, f in held.items() if any(f) and not all(f))
    return {"default": "f64", "storage": storage, "order": {"kind": "sequential"}, "params": {}}, split


def floats_of(path):
    cfg = json.load(open(path))
    if "float" in cfg:
        return cfg["float"]
    return [a["handle"] for a in cfg.get("actions", []) if a.get("action") == "change_var_basetype" and a.get("to_type") == "float"]


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "variables":
        json.dump(craft_config(Program(sys.argv[2])), sys.stdout, indent=1, ensure_ascii=False)
        print()
    elif len(sys.argv) == 4 and sys.argv[1] == "realization":
        r, split = realization(Program(sys.argv[2]), floats_of(sys.argv[3]))
        json.dump(r, sys.stdout, indent=2, ensure_ascii=False)
        print()
        print(f"typed apart in the C program: {', '.join(split) if split else 'none'}", file=sys.stderr)
    elif len(sys.argv) == 5 and sys.argv[1] == "apply":
        Program(sys.argv[2]).apply(floats_of(sys.argv[3]), sys.argv[4])
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
