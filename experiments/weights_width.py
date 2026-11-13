#!/usr/bin/env python3
"""SR6, the weights' certified width: what `experiments/weights_width.sh`
hands to Python, the standard library only (as `margins.py`).

  round SRC DST --leaves TSV --k K --names JSON
        the checkpoint SRC (a `.psk.d` of version 1, every real an `f64`)
        with the leaves NAMES rounded to K significant bits, written to DST
        in the same form; the other leaves keep their bytes
  leaves DUMP --model GPT --dim T=256 > TSV
        the model's leaves in storage order (`presk dump --stage elab`'s
        block), one `path<TAB>elements` a line
  realization --names JSON --type bf16
        the all-`f32` realization with the leaves NAMES at rest in `bf16`
  smallest MARGINS_JSON --variant M --ks "8 … 24" --delta 1e-3
        the smallest k whose row is certified at δ (none: nothing printed)
  report OUT …
        the report (below)

The rounding is the compiler's for `bf16`, generalized. A stored `f64` reaches
`bf16` through `f32` in the compiler (`presk-rt`'s `Buf::of`:
`f32_to_bf16(x as f32)`), so the script rounds the `f64` to `f32` first, to
nearest, ties to even (C's conversion, as Rust's `as`), and then the `f32`'s
bits to K significant bits as `f32_to_bf16` rounds them to 8: add
2^(23 − K) − 1 and the kept part's last bit, then clear the 24 − K bits below.
K = 8 is `bf16`'s value set, K = 24 `f32`'s; the exponent is `f32`'s
throughout, subnormals included, as `bf16`'s is.
"""

import argparse
import array
import hashlib
import json
import math
import os
import re
import sys

MAGIC = b"PSKD"


def read_psk_d(path):
    """(header bytes, the reals as an array of doubles) of a version-1 file
    of one sample."""
    raw = open(path, "rb").read()
    if raw[:4] != MAGIC:
        sys.exit(f"{path}: not a .psk.d file")
    version = int.from_bytes(raw[4:6], "little")
    if version != 1:
        sys.exit(f"{path}: version {version}; the rounding reads version 1, every real an f64")
    n = int.from_bytes(raw[6:10], "little")
    ty = raw[10:10 + n].decode()
    count = int.from_bytes(raw[10 + n:18 + n], "little")
    if not ty.startswith("params ") or count != 1:
        sys.exit(f"{path}: `{ty}`, {count} samples: not one checkpoint")
    head = raw[:18 + n]
    data = array.array("d")
    data.frombytes(raw[18 + n:])
    if sys.byteorder != "little":
        data.byteswap()
    return head, data


def leaves(argv):
    ap = argparse.ArgumentParser(prog="weights_width.py leaves")
    ap.add_argument("dump")
    ap.add_argument("--model", default="GPT")
    ap.add_argument("--dim", action="append", default=[], help="NAME=N for a dimension the dump leaves symbolic")
    a = ap.parse_args(argv)
    dims = dict(d.split("=") for d in a.dim)
    lines = open(a.dump).read().splitlines()
    start = lines.index(f"{a.model}:")
    for line in lines[start + 1:]:
        if not line.startswith("  "):
            break
        m = re.match(r"\s+(\S+) : ℝ\[(.*)\]$", line)
        if not m:
            sys.exit(f"not a real leaf: {line!r}")
        n = 1
        for ax in m.group(2).split(","):
            v = ax.rpartition(":")[2].strip()
            n *= int(dims.get(v, v))
        print(f"{m.group(1)}\t{n}")


def round_bits(bits, k):
    """An `f32`'s bits rounded to k significant bits, to nearest, ties to
    even: `presk_rt::f32_to_bf16` at k = 8 (and its `bf16_to_f32`)."""
    drop = 24 - k
    if drop == 0:
        return bits
    lsb = (bits >> drop) & 1
    return (((bits + (1 << (drop - 1)) - 1 + lsb) >> drop) << drop) & 0xFFFFFFFF


def round_(argv):
    ap = argparse.ArgumentParser(prog="weights_width.py round")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--leaves", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--names", required=True, help="a JSON list of leaf paths")
    a = ap.parse_args(argv)
    if not 1 <= a.k <= 24:
        sys.exit("k is between 1 and 24")
    names = set(json.loads(a.names))
    table = [(p, int(n)) for p, n in (line.rstrip("\n").split("\t") for line in open(a.leaves))]
    unknown = names - {p for p, _ in table}
    if unknown:
        sys.exit(f"no such leaf: {sorted(unknown)}")
    head, data = read_psk_d(a.src)
    if sum(n for _, n in table) != len(data):
        sys.exit(f"{a.src}: {len(data)} reals against {sum(n for _, n in table)} in the leaves")
    out = array.array("d", data)
    off, rounded, changed, worst = 0, 0, 0, 0.0
    for path, n in table:
        if path in names:
            seg = data[off:off + n]
            if not all(math.isfinite(x) for x in seg):
                sys.exit(f"{path}: a value that is not finite")
            f = array.array("f", seg)            # f64 → f32, to nearest, ties to even
            u = array.array("I")
            u.frombytes(f.tobytes())
            u = array.array("I", (round_bits(b, a.k) for b in u))
            g = array.array("f")
            g.frombytes(u.tobytes())
            for i, (x, y) in enumerate(zip(seg, g)):
                if not math.isfinite(y):
                    sys.exit(f"{path}[{i}]: {x!r} overflows at {a.k} bits")
                out[off + i] = y
                if y != x:
                    changed += 1
                    if x != 0:
                        worst = max(worst, abs(y - x) / abs(x))
            rounded += n
        off += n
    with open(a.dst + ".part", "wb") as fh:
        fh.write(head)
        if sys.byteorder != "little":
            out.byteswap()
        out.tofile(fh)
    os.replace(a.dst + ".part", a.dst)
    print(json.dumps({"k": a.k, "leaves": sorted(names), "rounded": rounded, "changed": changed,
                      "largest_relative_change": worst, "elements": len(data)}))


def realization(argv):
    ap = argparse.ArgumentParser(prog="weights_width.py realization")
    ap.add_argument("--names", required=True)
    ap.add_argument("--type", default="bf16")
    a = ap.parse_args(argv)
    r = {"default": "f32", "order": {"kind": "sequential"}, "params": {p: a.type for p in json.loads(a.names)}}
    print(json.dumps(r, indent=2, ensure_ascii=False))


def smallest(argv):
    ap = argparse.ArgumentParser(prog="weights_width.py smallest")
    ap.add_argument("margins")
    ap.add_argument("--variant", default="M")
    ap.add_argument("--ks", required=True)
    ap.add_argument("--delta", default="0.001")
    a = ap.parse_args(argv)
    rows = {r["name"]: r for r in json.load(open(a.margins))["rows"]}
    for k in sorted(int(x) for x in a.ks.split()):
        r = rows.get(f"{a.variant}-{k}")
        if r and r["certified"][str(float(a.delta))]:
            print(k)
            return


# ---------------------------------------------------------------------------
# The report.


def sci(x, d=3):
    return f"{float(x):.{d}e}"


def same_files(a, b, suffix):
    """Whether the two directories hold the same files ending in SUFFIX,
    byte for byte (and how many)."""
    fa = sorted(f for f in os.listdir(a) if f.endswith(suffix))
    fb = sorted(f for f in os.listdir(b) if f.endswith(suffix))
    if fa != fb or not fa:
        return False, len(fa)
    return all(open(os.path.join(a, f), "rb").read() == open(os.path.join(b, f), "rb").read() for f in fa), len(fa)


def loss_value(path):
    """The value `presk run --entry loss` prints, `ℝ[] = v`."""
    for line in open(path):
        if line.startswith("ℝ[] ="):
            return float(line.split("=")[1])
    raise ValueError(path)


def solve_cost(path):
    """A solve's `cost` line: its all-f64 traffic, in bytes (one sample)."""
    for line in open(path):
        if line.strip().startswith("cost"):
            return float(line.split()[3])
    raise ValueError(path)


def report(argv):
    ap = argparse.ArgumentParser(prog="weights_width.py report")
    ap.add_argument("out")
    ap.add_argument("--scales", required=True)
    ap.add_argument("--ks", required=True)
    ap.add_argument("--variants", default="M A")
    ap.add_argument("--mats", required=True)
    ap.add_argument("--record-f32", required=True, help="SCALE=DIR … the native f32 rows of record")
    ap.add_argument("--solve", required=True, help="SCALE=FILE … a logits solve whose cost line gives the all-f64 traffic")
    ap.add_argument("--delta", default="0.001")
    a = ap.parse_args(argv)
    scales = a.scales.split()
    ks = [int(k) for k in a.ks.split()]
    variants = a.variants.split()
    mats = set(json.loads(a.mats))
    rec = dict(x.split("=", 1) for x in a.record_f32.split())
    solves = dict(x.split("=", 1) for x in a.solve.split())
    d = str(float(a.delta))
    L = []
    w = L.append
    summary = {"scales": {}}
    w("# SR6: the weights' certified width\n")
    w("The course GPT's trained weights rounded to k significant bits (to nearest, ties to even, from the stored `f64` "
      "through `f32` as the compiler rounds to `bf16`), run under `v0.17.1`'s all-`f32` realization natively on the "
      "Code civil's 461 held-out windows (118 016 predictions), against the exact logits of the **unrounded** weights "
      "(`make margins`' and `margins_course.sh`'s of record). Variant M rounds the nine matrices, variant A all "
      "twenty-one parameters. Certified: the largest gap at most ε*(10⁻³) of record. Nothing timed, nothing solved.\n")
    for s in scales:
        base = os.path.join(a.out, s)
        m = json.load(open(os.path.join(base, "margins.json")))
        star = float(m["star"][d]["eps_cp"])
        n = m["n"]
        rows = {r["name"]: r for r in m["rows"]}
        table = [(p, int(c)) for p, c in (line.rstrip("\n").split("\t") for line in open(os.path.join(base, "leaves.tsv")))]
        n_all = sum(c for _, c in table)
        n_mat = sum(c for p, c in table if p in mats)
        all_f64 = solve_cost(solves[s])
        values_f64 = all_f64 - 8 * n_all
        values_f32 = values_f64 / 2

        def at_rest(variant, k):
            r = n_mat if variant == "M" else n_all
            return r * (k + 8) / 8 + (n_all - r) * 4

        exact_loss = loss_value(os.path.join(base, "loss", "exact.txt"))
        sc = {"eps_star": star, "n": n, "parameters": n_all, "matrices": n_mat, "all_f64_traffic": all_f64,
              "exact_loss": exact_loss, "rows": {}, "checks": {}}
        # SR6a
        checks = []
        nat = os.path.join(base, "native")
        ok, cnt = same_files(os.path.join(nat, "f32"), rec[s], ".tsv")
        checks.append(("all-`f32` on the unrounded weights: its rows the record's", ok, cnt, "tsv"))
        for v in variants:
            if f"{v}-24" in rows:
                ok, cnt = same_files(os.path.join(nat, f"{v}-24"), rec[s], ".tsv")
                checks.append((f"{v} at k = 24: its rows the record's all-`f32`", ok, cnt, "tsv"))
                ok, cnt = same_files(os.path.join(nat, f"{v}-24"), os.path.join(nat, "f32"), ".json.sha256")
                checks.append((f"{v} at k = 24: its logits all-`f32`'s on the unrounded weights", ok, cnt, "logits"))
            if f"{v}-8" in rows:
                ok, cnt = same_files(os.path.join(nat, f"{v}-8"), os.path.join(nat, f"bf16p-{v}"), ".json.sha256")
                checks.append((f"{v} at k = 8: its logits the realization's with the same parameters at rest in `bf16`", ok, cnt, "logits"))
                dec_a = open(os.path.join(base, "decisions", f"{v}-8.txt"), "rb").read()
                dec_b = open(os.path.join(base, "decisions", f"bf16p-{v}.txt"), "rb").read()
                checks.append((f"{v} at k = 8: its decisions the same realization's", dec_a == dec_b, 1, "decisions"))
        sc["checks"] = [{"check": c, "holds": ok, "files": cnt, "what": wh} for c, ok, cnt, wh in checks]
        sr6a = all(ok for _, ok, _, _ in checks)
        sc["sr6a"] = sr6a
        w(f"## {'D = 64' if s == 'd64' else 'The course constants'}\n")
        w(f"ε*(10⁻³) = {sci(star, 4)} on {n} predictions. {n_all} parameters, {n_mat} of them in the nine matrices. "
          f"The exact loss of the unrounded weights on two held-out windows: {exact_loss!r}. All-`f64` traffic "
          f"{sci(all_f64, 5)} bytes a sample (the solver's unit, D157, from `{solves[s]}`).\n")
        w("**SR6a, the method's check:**\n")
        for c, ok, cnt, wh in checks:
            w(f"- {c}: {'yes' if ok else '**NO**'} ({cnt} {'chunks' if wh != 'decisions' else 'file'}, {wh} byte for byte)")
        w(f"\nSR6a {'holds' if sr6a else '**fails**'} at this scale.\n")
        w("| row | k | bits an element | at rest (bytes) | traffic (bytes) | ÷ all-`f32` | ÷ all-`f64` | largest gap | ÷ ε* | mean | "
          "predictions changed | bound #{m < 2ε} | certified | loss | rel. gap to the exact loss |")
        w("|" + "---|" * 15)
        f32_rest = 4 * n_all
        f32_traffic = f32_rest + values_f32
        order = ["f32", "bf16p-M", "bf16p-A"] + [f"{v}-{k}" for v in variants for k in ks]
        for name in order:
            r = rows.get(name)
            if not r:
                continue
            if name == "f32":
                v, k, rest = "", 24, f32_rest
                label, bits = "all-`f32`, unrounded", "32"
            elif name.startswith("bf16p-"):
                v = name[-1]
                k, rest = 8, at_rest(v, 8)
                label, bits = f"{v}, `bf16` at rest (the compiler's)", "16 / 32"
            else:
                v, k = name.split("-")[0], int(name.split("-")[1])
                rest = at_rest(v, k)
                label, bits = f"{v}", f"{k + 8}" + (" / 32" if v == "M" else "")
            traffic = rest + values_f32
            lf = os.path.join(base, "loss", f"{name}.txt")
            loss = loss_value(lf) if os.path.exists(lf) else float("nan")
            rel = abs(loss - exact_loss) / abs(exact_loss)
            cert = r["certified"][d]
            sc["rows"][name] = {"variant": v, "k": k, "at_rest": rest, "traffic": traffic, "eps_meas": r["eps_meas"],
                                "mean_gap": r["mean_gap"], "flips": r["flips"], "bound": r["bound"], "certified": cert,
                                "loss": loss, "loss_rel_gap": rel}
            w(f"| {label} | {k} | {bits} | {rest:.0f} | {traffic:.4e} | {traffic / f32_traffic:.3f} | {traffic / all_f64:.3f} | "
              f"{sci(r['eps_meas'])} | {r['eps_meas'] / star:.3f} | {sci(r['mean_gap'], 2)} | {r['flips']} | {r['bound']} | "
              f"{'yes' if cert else 'no'} | {loss!r} | {sci(rel, 2)} |")
        for v in variants:
            cs = [k for k in sorted(ks) if sc["rows"].get(f"{v}-{k}", {}).get("certified")]
            sc[f"smallest_{v}"] = cs[0] if cs else None
            # certified at k and at every wider k of the ladder
            mono = [k for k in sorted(ks) if all(sc["rows"].get(f"{v}-{j}", {}).get("certified") for j in ks if j >= k)]
            sc[f"smallest_{v}_upward"] = mono[0] if mono else None
            w(f"\nVariant {v}: the smallest certified k {sc[f'smallest_{v}'] if cs else 'none'}"
              f"{'' if not cs or mono[:1] == cs[:1] else f' (certified at every wider k from {mono[0] if mono else None})'}.")
        m16 = sc["rows"].get("M-16")
        sc["sr6b"] = m16["certified"] if m16 else None
        if m16:
            w(f"\nAt this scale, M at k = 16: largest gap {sci(m16['eps_meas'])} against ε* {sci(star, 4)}: "
              f"{'certified' if sc['sr6b'] else 'not certified'}.\n")
        else:
            w("\nM at k = 16 is not in this ladder.\n")
        # The Code pénal
        pbase = os.path.join(a.out, f"{s}-penal")
        if os.path.exists(os.path.join(pbase, "margins.json")):
            pm = json.load(open(os.path.join(pbase, "margins.json")))
            w(f"**The Code pénal's 461 windows** ({pm['n']} predictions), against the Code civil's ε* {sci(star, 4)} "
              "(D168's protocol); the bound counts the Code pénal's own margins:\n")
            w("| row | largest gap | ÷ ε* (civil) | flips | bound #{m < 2ε} | certified at the civil ε* |")
            w("|---|---|---|---|---|---|")
            sc["penal"] = {}
            for r in pm["rows"]:
                cert = r["eps_meas"] <= star
                sc["penal"][r["name"]] = {"eps_meas": r["eps_meas"], "flips": r["flips"], "bound": r["bound"], "certified": cert}
                w(f"| {r['name']} | {sci(r['eps_meas'])} | {r['eps_meas'] / star:.3f} | {r['flips']} | {r['bound']} | {'yes' if cert else 'no'} |")
            w("")
        summary["scales"][s] = sc
    sr6a = all(summary["scales"][s]["sr6a"] for s in scales)
    b = [summary["scales"][s]["sr6b"] for s in scales]
    sr6b = None if None in b or set(scales) != {"d64", "course"} else all(b)
    summary["sr6a"], summary["sr6b"] = sr6a, sr6b
    w("## The statements\n")
    w(f"- **SR6a**, the method's check: {'holds' if sr6a else '**fails**'} "
      f"({', '.join(s + (' holds' if summary['scales'][s]['sr6a'] else ' fails') for s in scales)}).")
    w(f"- **SR6b**, three bytes: variant M at k = 16 certified at δ = 10⁻³ at "
      + ", ".join(f"{s} {({True: 'yes', False: 'no', None: 'not run'})[summary['scales'][s]['sr6b']]}" for s in scales)
      + ": " + ("not decided (both scales and k = 16 are needed)" if sr6b is None else "holds" if sr6b else "**fails**")
      + ("" if sr6a else " (not read: SR6a fails)") + ".")
    w("\nThe traffic is a count in the solver's unit, the parameters at rest and one sample's values in `f32`: no kernel "
      "reads a three-byte element, and nothing here is timed.")
    text = "\n".join(L) + "\n"
    open(os.path.join(a.out, "report.md"), "w").write(text)
    json.dump(summary, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print(text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd, *rest = sys.argv[1:]
    {"leaves": leaves, "round": round_, "realization": realization, "smallest": smallest, "report": report}.get(
        cmd, lambda _: sys.exit(__doc__))(rest)
