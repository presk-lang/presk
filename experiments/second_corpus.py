#!/usr/bin/env python3
"""The certificate on a second text (presk-project
docs/prompt-second-corpus-2026-10-22.md): the Code pénal through the Code
civil's models, realizations and ε*, what experiments/second_corpus.sh hands
to Python. The standard library only.

  fetch URL MEMBER DEST          the container's zip, MEMBER extracted to DEST;
                                 the sha256 of both on stdout
  clean SRC VOCAB DEST LOG       SRC with every character outside VOCAB (a
                                 `presk data chars` JSON) deleted, into DEST,
                                 its line ends kept; per character deleted, its
                                 code point, its count and its first contexts,
                                 into LOG
  map SRC VOCAB DEST LOG [N]     plan 0.16.1's rule for the third and fourth
                                 texts: the typographic characters mapped
                                 (’ ‘ to ', “ ” to ", … to ..., ‒ ― − to -),
                                 then every character outside VOCAB deleted
                                 (the soft hyphen among them), the line ends
                                 kept; both counts per character into LOG,
                                 with how many fall in the first N characters
                                 of DEST (default 461·256 + 1, the judge set)
  head TOKENS WINDOWS SEQ DEST   the first WINDOWS·SEQ + 1 tokens of TOKENS, the
                                 stream whose `presk data lm-windows --seq SEQ`
                                 is the whole stream's first WINDOWS windows
  same A.json B.json N           whether the first N samples of two `presk data
                                 export`s are equal (exit 1 if not)
  report --civil DIR --penal DIR …
                                 the comparison, in Markdown on stdout
  chunk DIR --first F --window W --position P [--what TSV]
                                 one chunk's rows under several variants
                                 (DIR/<variant>.tsv): each one's gap at one
                                 position beside the rest, in Markdown

The report reads what experiments/margins.sh (or margins_course.sh) wrote in
each directory: margins.json (every realization's largest gap on the
logits, its flips, its bound, the margins' ε*) and native/<name>/*.tsv (per
position the largest gap, the native argmax, the sum of the gaps). A
window's gap is its largest over its positions. ε* is the Code civil's
ε*(10⁻³), corrected (Clopper–Pearson), read from the Code civil's
margins.json; the Code pénal's own is reported beside it and certifies
nothing. The two statements, fixed before the measurement:

  Y1  every realization certified at 10⁻³ on the Code civil's windows (its
      gap there at most ε*) has on the Code pénal's a gap at most the Code
      civil's ε* and flips no decision of the exact model;
  Y2  every realization's largest gap on the Code pénal is within [0.5×, 2×]
      of its largest gap on the Code civil.
"""

import argparse
import hashlib
import io
import json
import os
import re
import sys
import urllib.request
import zipfile

T = 256


def sha(b):
    return hashlib.sha256(b).hexdigest()


def fetch(url, member, dest):
    z = urllib.request.urlopen(url).read()
    with zipfile.ZipFile(io.BytesIO(z)) as f:
        b = f.read(member)
    open(dest + ".part", "wb").write(b)
    os.replace(dest + ".part", dest)
    print(f"{sha(z)}  {url.rsplit('/', 1)[-1]}")
    print(f"{sha(b)}  {member}")


def clean(src, vocab, dest, log):
    text = open(src, "rb").read().decode("utf-8")
    keep = set(json.load(open(vocab)))
    gone = {}
    for i, c in enumerate(text):
        if c not in keep:
            gone.setdefault(c, []).append(i)
    out = "".join(c for c in text if c in keep)
    open(dest + ".part", "wb").write(out.encode("utf-8"))
    os.replace(dest + ".part", dest)
    with open(log, "w") as f:
        f.write("char\tcode point\tcount\tfirst contexts\n")
        for c in sorted(gone, key=lambda c: (-len(gone[c]), c)):
            ctx = " | ".join(repr(text[max(0, i - 25):i + 25]) for i in gone[c][:3])
            f.write(f"{c}\tU+{ord(c):04X}\t{len(gone[c])}\t{ctx}\n")
    n = sum(len(v) for v in gone.values())
    print(f"{len(text)} characters, {len(set(text))} distinct, {len(gone)} not in the vocabulary of {len(keep)}: "
          f"{n} deleted, {len(out)} kept; CRLF {text.count(chr(13) + chr(10))}")


# Plan 0.16.1 §7: the typographic characters of the Code du travail and the
# Code de procédure pénale, mapped onto the Code civil's vocabulary.
MAP = {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2026": "...",
       "\u2012": "-", "\u2015": "-", "\u2212": "-"}


def map_text(src, vocab, dest, log, n=str(461 * T + 1)):
    text = open(src, "rb").read().decode("utf-8")
    keep = set(json.load(open(vocab)))
    assert all(set(v) <= keep for v in MAP.values()), "a mapped character outside the vocabulary"
    n = int(n)
    out, mapped, gone = [], {}, {}
    for i, c in enumerate(text):
        if c in MAP:
            mapped.setdefault(c, []).append((i, len(out)))
            out.extend(MAP[c])
        elif c in keep:
            out.append(c)
        else:
            gone.setdefault(c, []).append((i, len(out)))
    out = "".join(out)
    open(dest + ".part", "wb").write(out.encode("utf-8"))
    os.replace(dest + ".part", dest)
    with open(log, "w") as f:
        f.write(f"what\tchar\tcode point\tcount\tin the first {n} characters\tfirst contexts\n")
        for what, d in (("mapped", mapped), ("deleted", gone)):
            for c in sorted(d, key=lambda c: (-len(d[c]), c)):
                head = sum(1 for _, o in d[c] if o < n)
                ctx = " | ".join(repr(text[max(0, i - 25):i + 25]) for i, _ in d[c][:3])
                f.write(f"{what}\t{c}\tU+{ord(c):04X}\t{len(d[c])}\t{head}\t{ctx}\n")
    count = lambda d, h=False: sum(sum(1 for _, o in v if o < n) if h else len(v) for v in d.values())
    print(f"{len(text)} characters, {len(set(text))} distinct, {len(set(text) - keep)} not in the vocabulary of {len(keep)}; "
          f"{count(mapped)} mapped ({len(mapped)} characters), {count(gone)} deleted ({len(gone)} characters), {len(out)} kept; "
          f"in the first {n}: {count(mapped, True)} mapped, {count(gone, True)} deleted; CRLF {text.count(chr(13) + chr(10))}")


def head(tokens, windows, seq, dest):
    t = open(tokens).read().split()
    n = int(windows) * int(seq) + 1
    assert len(t) >= n, (len(t), n)
    open(dest, "w").write(" ".join(t[:n]) + "\n")
    print(f"{n} of {len(t)} tokens")


def same(a, b, n):
    sa, sb = json.load(open(a))["samples"], json.load(open(b))["samples"]
    n = int(n)
    ok = len(sa) >= n and len(sb) >= n and sa[:n] == sb[:n]
    print(f"the first {n} samples: {'the same' if ok else 'NOT the same'} ({len(sa)} and {len(sb)} samples)")
    sys.exit(0 if ok else 1)


# ----- the report -------------------------------------------------------------

class Run:
    """One directory of margins.sh: its margins.json, its windows, its rows."""

    def __init__(self, d, vocab):
        self.dir = d
        self.m = json.load(open(os.path.join(d, "margins.json")))
        self.rows = {r["name"]: r for r in self.m["rows"]}
        self.val = json.load(open(os.path.join(d, "val.json")))["samples"]
        self.vocab = vocab
        self.eps = float(self.m["star"]["0.001"]["eps_cp"])
        self.n = self.m["n"]
        self.exact_top = []
        ex = os.path.join(d, "exact")
        for c in sorted(f for f in os.listdir(ex) if f.endswith(".tsv")):
            for line in open(os.path.join(ex, c)):
                f = line.split("\t")
                self.exact_top.append((int(f[0]), float(f[2])))
        self._pos = {}

    def positions(self, name):
        """Per position the largest gap, read from native/<name>."""
        if name not in self._pos:
            d = os.path.join(self.dir, "native", name)
            g = []
            for c in sorted(f for f in os.listdir(d) if f.endswith(".tsv")):
                for line in open(os.path.join(d, c)):
                    g.append(float(line.split("\t")[0]))
            assert len(g) == self.n, (d, len(g), self.n)
            self._pos[name] = g
        return self._pos[name]

    def windows(self, name):
        g = self.positions(name)
        return [max(g[i * T:(i + 1) * T]) for i in range(len(g) // T)]

    def worst_position(self, name, w):
        g = self.positions(name)[w * T:(w + 1) * T]
        return max(range(T), key=lambda p: g[p])

    def text(self, w, p, before=36):
        x, y = self.val[w]
        ctx = "".join(self.vocab[t] for t in x[max(0, p - before):p + 1])
        return ctx, self.vocab[y[p]], self.vocab[self.exact_top[w * T + p][0]]


def med(xs):
    s = sorted(xs)
    return s[len(s) // 2]


def e3(x):
    return f"{x:.3e}"


def show(s):
    return s.replace("\r", "␍").replace("\n", "⏎").replace("\xa0", "⍽").replace("|", "\\|").replace("`", "'")


def report(argv):
    ap = argparse.ArgumentParser(prog="second_corpus.py report")
    ap.add_argument("--civil", required=True)
    ap.add_argument("--penal", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--loss-civil", default="")
    ap.add_argument("--loss-penal", default="")
    ap.add_argument("--focus", default="", help="realizations whose worst windows on the second text are shown, comma-separated")
    ap.add_argument("--name", default="Code pénal", help="the second text's name (plan 0.16.1: the third and fourth texts)")
    ap.add_argument("--short", default="pénal", help="its short name, in the table's columns")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--windows-out", default="", help="PREFIX: every realization's window gaps on each text, "
                    "PREFIX-civil.tsv and PREFIX-penal.tsv (a row per window, a column per realization)")
    a = ap.parse_args(argv)
    vocab = json.load(open(a.vocab))
    C, P = Run(a.civil, vocab), Run(a.penal, vocab)
    names = [r["name"] for r in C.m["rows"]]
    missing = [n for n in names if n not in P.rows]
    assert not missing, f"realizations missing on the {a.name}: {missing}"
    if a.windows_out:
        for R, t in ((C, "civil"), (P, "penal")):
            cols = [R.windows(n) for n in names]
            with open(f"{a.windows_out}-{t}.tsv", "w") as f:
                f.write("window\t" + "\t".join(names) + "\n")
                for i in range(len(cols[0])):
                    f.write(f"{i}\t" + "\t".join(f"{c[i]:.6e}" for c in cols) + "\n")
    eps = C.eps
    w = print

    def loss(path):
        if not path or not os.path.exists(path):
            return "—"
        m = re.search(r"exact loss ([0-9.]+)", open(path).read())
        return m.group(1) if m else "—"

    w(f"# The {a.name} through the Code civil's model{': ' + a.label if a.label else ''}")
    w("")
    w(f"`{C.dir}` (the Code civil, of record) against `{P.dir}` (the {a.name}). ε\\* is the Code civil's "
      f"ε\\*(10⁻³), corrected: **{eps:.4e}**.")
    w("")
    w("## The two texts under the exact model")
    w("")
    w(f"| | Code civil | {a.name} |")
    w("|---|---|---|")
    w(f"| windows, positions | {C.m['windows']}, {C.n} | {P.m['windows']}, {P.n} |")
    w(f"| exact loss (nats) | {loss(a.loss_civil)} | {loss(a.loss_penal)} |")
    w(f"| exact accuracy | {C.m['exact_correct']}/{C.n} = {C.m['exact_correct'] / C.n:.6f} | "
      f"{P.m['exact_correct']}/{P.n} = {P.m['exact_correct'] / P.n:.6f} |")
    w(f"| largest \\|logit\\| | {C.m['ymax']:.5f} | {P.m['ymax']:.5f} |")
    w(f"| margins: smallest, median | {float(C.m['smallest']):.4e}, {float(C.m['median']):.4e} | "
      f"{float(P.m['smallest']):.4e}, {float(P.m['median']):.4e} |")
    w(f"| ties | {C.m['ties']} | {P.m['ties']} |")
    for d in ("0.01", "0.001", "0.0001"):
        sc, sp = C.m["star"][d], P.m["star"][d]
        w(f"| ε\\*({float(d):.0e}) corrected (k) | {float(sc['eps_cp']):.4e} ({sc['k_cp']}) | "
          f"{float(sp['eps_cp']):.4e} ({sp['k_cp']}) |")
    cc, cp = dict((e, k) for e, k in C.m["cdf"]), dict((e, k) for e, k in P.m["cdf"])
    for e in sorted(cc, reverse=True):
        w(f"| m < 2·{e:.0e} | {cc[e]} | {cp.get(e, '—')} |")
    ep = float(P.m["star"]["0.001"]["eps_cp"])
    w("")
    w(f"The {a.name}'s own ε\\*(10⁻³) is {ep / eps:.3f} times the Code civil's.")
    for R, t in ((C, "Code civil"), (P, a.name)):
        k = sum(1 for _, m in R.exact_top if m < 2 * eps)
        w(f" On the {t}, {k} of {R.n} margins ({k / R.n:.3e}) are below twice the Code civil's ε\\*.")
    w("")

    w("## Per realization")
    w("")
    w("The gap is the largest |ŷ − y| over every position and logit of the 461 windows; the flips are "
      "`decisions`' natively against the exact argmax; the bound is #{m < 2·gap} on that text; "
      "'≤ ε\\*' compares the gap with the Code civil's ε\\*(10⁻³).")
    w("")
    w(f"| realization | gap, Code civil | gap, {a.name} | ratio | flips, civil | flips, {a.short} | bound, {a.short} | "
      f"≤ ε\\*, civil | ≤ ε\\*, {a.short} | ≤ the {a.short}'s ε\\* | Y1 | Y2 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    y1, y2 = [], []
    ratios = {}
    for n in names:
        rc, rp = C.rows[n], P.rows[n]
        gc, gp = rc["eps_meas"], rp["eps_meas"]
        r = gp / gc
        ratios[n] = r
        cert_c, cert_p = gc <= eps, gp <= eps
        y1ok = (cert_p and rp["flips"] == 0) if cert_c else None
        y2ok = 0.5 <= r <= 2
        if y1ok is not None:
            y1.append((n, y1ok))
        y2.append((n, y2ok))
        w(f"| {n} | {e3(gc)} | {e3(gp)} | {r:.2f}× | {rc['flips']} | {rp['flips']} | {rp['bound']} | "
          f"{'yes' if cert_c else 'no'} | {'yes' if cert_p else 'no'} | {'yes' if gp <= ep else 'no'} | "
          f"{'—' if y1ok is None else ('holds' if y1ok else '**fails**')} | {'holds' if y2ok else '**fails**'} |")
    w("")
    f1 = [n for n, ok in y1 if not ok]
    f2 = [n for n, ok in y2 if not ok]
    w(f"**Y1** on {len(y1)} realizations certified on the Code civil: "
      + ("holds." if not f1 else f"fails on {', '.join(f1)}."))
    w("")
    w(f"**Y2** on {len(y2)} realizations: " + ("holds." if not f2 else f"fails on {', '.join(f2)}."))
    w(f" The ratios run from {min(ratios.values()):.2f}× ({min(ratios, key=ratios.get)}) to "
      f"{max(ratios.values()):.2f}× ({max(ratios, key=ratios.get)}); their median is {med(list(ratios.values())):.2f}×.")
    w("")
    order = sorted(names, key=lambda n: -ratios[n])
    w("Ranked by the ratio: " + ", ".join(f"{n} {ratios[n]:.2f}×" for n in order) + ".")
    w("")

    w("## The windows")
    w("")
    w("A window's gap is its largest over its 256 positions. Among the ten worst windows, 'after (' counts "
      "those whose worst position predicts the character after an opening parenthesis (the input there is `(`), "
      "'after a capital' those whose input there is a capital letter.")
    w("")
    w("| realization | worst window, civil | median window, civil | worst / median | after (, after a capital | "
      f"worst window, {a.short} | median window, {a.short} | worst / median | after (, after a capital | median ratio |")
    w("|---|---|---|---|---|---|---|---|---|---|")

    def paren(R, n):
        wins = R.windows(n)
        top = sorted(range(len(wins)), key=lambda i: -wins[i])[:a.top]
        inp = [vocab[R.val[i][0][R.worst_position(n, i)]] for i in top]
        return f"{sum(1 for c in inp if c == '(')}, {sum(1 for c in inp if c.isupper())}"

    for n in names:
        wc, wp = C.windows(n), P.windows(n)
        ic, ip = max(range(len(wc)), key=lambda i: wc[i]), max(range(len(wp)), key=lambda i: wp[i])
        w(f"| {n} | {e3(wc[ic])} ({ic}) | {e3(med(wc))} | {wc[ic] / med(wc):.1f} | {paren(C, n)} | "
          f"{e3(wp[ip])} ({ip}) | {e3(med(wp))} | {wp[ip] / med(wp):.1f} | {paren(P, n)} | {med(wp) / med(wc):.2f}× |")
    w("")
    for n in [x for x in a.focus.split(",") if x]:
        for R, name in ((P, a.name), (C, "Code civil")):
            wins = R.windows(n)
            top = sorted(range(len(wins)), key=lambda i: -wins[i])[:a.top]
            w(f"**`{n}`, the {a.top} worst windows of the {name}.** The context ends with the input at the worst "
              "position; 'next' is the character to predict, 'top' the exact model's choice, 'margin' its top-two margin.")
            w("")
            w("| window | gap | position | context | next | top | margin |")
            w("|---|---|---|---|---|---|---|")
            for i in top:
                p = R.worst_position(n, i)
                ctx, nxt, tp = R.text(i, p)
                w(f"| {i} | {e3(wins[i])} | {p} | `{show(ctx)}` | `{show(nxt)}` | `{show(tp)}` | "
                  f"{R.exact_top[i * T + p][1]:.3f} |")
            w("")
            g = R.positions(n)
            w(f"Positions above 10⁻⁴: {sum(1 for x in g if x > 1e-4)}; windows within 2× of the worst: "
              f"{sum(1 for x in wins if x >= max(wins) / 2)}.")
            w("")


def chunk(argv):
    """The variants' rows on one chunk of windows: per variant, its gap at
    one position, the largest over that position's window and the largest
    over the chunk's other windows, in Markdown."""
    ap = argparse.ArgumentParser(prog="second_corpus.py chunk")
    ap.add_argument("native", help="the directory of <variant>.tsv rows, one chunk each")
    ap.add_argument("--first", type=int, required=True, help="the chunk's first window")
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--position", type=int, required=True)
    ap.add_argument("--what", default="", help="a TSV of variant names and what each changes")
    a = ap.parse_args(argv)
    what = dict(l.rstrip("\n").split("\t", 1) for l in open(a.what)) if a.what else {}
    k = a.window - a.first
    print(f"| variant | what it changes | gap at window {a.window}, position {a.position} | the rest of window {a.window} | "
          f"the chunk's other windows |")
    print("|---|---|---|---|---|")
    for f in sorted(x for x in os.listdir(a.native) if x.endswith(".tsv")):
        g = [float(line.split("\t")[0]) for line in open(os.path.join(a.native, f))]
        n = f[:-4]
        win = g[k * T:(k + 1) * T]
        rest = max(x for i, x in enumerate(win) if i != a.position)
        other = max(x for i, x in enumerate(g) if i // T != k)
        print(f"| `{n}` | {what.get(n, '')} | {e3(win[a.position])} | {e3(rest)} | {e3(other)} |")


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or [""]
    if cmd == "chunk":
        chunk(rest)
        sys.exit(0)
    if cmd == "fetch":
        fetch(*rest)
    elif cmd == "clean":
        clean(*rest)
    elif cmd == "map":
        map_text(*rest)
    elif cmd == "head":
        head(*rest)
    elif cmd == "same":
        same(*rest)
    elif cmd == "report":
        report(rest)
    else:
        sys.exit(__doc__)
