#!/usr/bin/env python3
"""Course data -> Presk datasets.

`civil_mots.txt` (the words of the Code civil, one per line) becomes the
character vocabulary of the ladder and two text files of token indices, which
`presk data pack` turns into `.psk.d`:

  bigrams.txt   "x y"          one line per (character, next character)
  context3.txt  "c1 c2 c3 y"   one line per (three-character context, next)

Every word is bracketed by the boundary token `.` (index 0), exactly as in the
course notebooks: "abc" gives (., a) (a, b) (b, c) (c, .).
"""
import pathlib, urllib.request

COURSE = "https://www.cri.minesparis.psl.eu/people/silber/cours/2025/nlp"
FILES = {
    "civil_mots.txt": f"{COURSE}/notebooks/001_bigrams/civil_mots.txt",
    "civil.md": f"{COURSE}/notebooks/007_gpt/civil.md",
}

here = pathlib.Path(__file__).parent
for name, url in FILES.items():
    target = here / name
    if not target.exists():
        print(f"downloading {name} from the course site")
        urllib.request.urlretrieve(url, target)

words = here.joinpath("civil_mots.txt").read_text(encoding="utf-8").split()
chars = sorted(set("".join(words)))
itos = ["."] + chars
stoi = {c: i for i, c in enumerate(itos)}
V = len(itos)

here.joinpath("vocab.txt").write_text("".join(itos), encoding="utf-8")

CTX = 3
bi, ctx = [], []
for w in words:
    ids = [0] + [stoi[c] for c in w] + [0]
    for a, b in zip(ids, ids[1:]):
        bi.append(f"{a} {b}")
    window = [0] * CTX
    for t in ids[1:]:
        ctx.append(" ".join(map(str, window)) + f" {t}")
        window = window[1:] + [t]

here.joinpath("bigrams.txt").write_text("\n".join(bi) + "\n", encoding="utf-8")
here.joinpath(f"context{CTX}.txt").write_text("\n".join(ctx) + "\n", encoding="utf-8")
print(f"V = {V}  vocabulary: {''.join(itos)!r}")
print(f"{len(words)} words -> {len(bi)} bigrams, {len(ctx)} context windows of {CTX}")
