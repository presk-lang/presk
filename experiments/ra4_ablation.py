#!/usr/bin/env python3
"""The ablation of plan 0.8.0 §3.4, gathered (experiments/ra4_ablation.sh).

    ra4_ablation.py OUT FORM...

Each OUT/FORM holds `presk rank`'s report of the development set priced
under one form (ra4.md) and the re-priced days (out/ra4/day1, day2). The
tables read the reports as `presk rank` writes them; the hybrid chooser of
statement Q is computed here from the re-priced lines, since `presk rank`
does not decide Q before step 7. Nothing here decides a statement: these are
development figures (plan §3.4)."""
import glob
import json
import os
import re
import sys

out, forms = sys.argv[1], sys.argv[2:]

# 0.7.0's statement M is about these nests, at B=16,T=256 in f32; O about
# every contraction nest there.
M_NESTS = ['SelfAttention.apply/%13', 'SelfAttention.apply/%23']
O_NESTS = ['Linear.apply/%3'] + M_NESTS
CLASSES = ['tile', 'order', 'kblock', 'lanes', 'split', 'pack']
CONTRACTIONS = ('dot', 'ein')


def cells(line):
    return [c.strip() for c in line.strip().strip('|').split('|')]


def report(form):
    text = open(os.path.join(out, form, 'ra4.md')).read()
    u = float(re.search(r'u = ([0-9]+\.[0-9]+)', text).group(1))
    nests, classes, verdict = {}, {}, {}
    section = None
    for line in text.splitlines():
        if line.startswith('## '):
            section = line[3:]
            continue
        if not line.startswith('| ') or line.startswith('| nest') or line.startswith('| class') or line.startswith('|---') or line.startswith('| |'):
            continue
        c = cells(line)
        if section is None:
            name = c[0].strip('`*')
            nests[(name, c[1], c[3])] = dict(points=c[4], distinguished=c[6], right=c[7], decided=c[8], tied=c[10], regret=c[12])
        elif section.startswith('Statements'):
            verdict[c[0]] = c[3].strip('*')
        elif section == 'Per decision class':
            classes[c[0]] = dict(distinguished=c[2], right=c[3], decided=c[4], tied=c[6])
    return u, nests, classes, verdict


def lines(form, day):
    by = {}
    for path in glob.glob(os.path.join(out, form, 'out', 'ra4', day, '*', '*.jsonl')):
        for l in open(path):
            d = json.loads(l)
            if d.get('baseline') or not d.get('timed'):
                continue
            by.setdefault((d['nest'], d['config'], d['dtype']), {})[d['key']] = d
    return by


def hybrid(form, u):
    """Statement Q's chooser on the development set: measure only the points
    predicted within 1 + u of the best prediction. Its regret against the best
    measured point, and its share of the sweep's kernel time, per contraction
    nest swept whole (the BERT-shaped sample of 41 points is left out)."""
    d1, d2 = lines(form, 'day1'), lines(form, 'day2')
    rows = {}
    for k, pts in d1.items():
        if k not in d2 or len(pts) < 100 or next(iter(pts.values()))['op'] not in CONTRACTIONS:
            continue
        both = [(p['predicted'], (p['median'] + d2[k][key]['median']) / 2, sum(p['seconds'])) for key, p in pts.items() if key in d2[k]]
        best_pred = min(p for p, _, _ in both)
        best = min(t for _, t, _ in both)
        cand = [(p, t, s) for p, t, s in both if p <= (1 + u) * best_pred]
        rows[k] = dict(n=len(both), candidates=len(cand), regret=min(t for _, t, _ in cand) / best,
                       share=sum(s for _, _, s in cand) / sum(s for _, _, s in both))
    return rows


data = {f: report(f) for f in forms}
hyb = {f: hybrid(f, data[f][0]) for f in forms}

print('# The ablation of plan 0.8.0 §3.4 on the development set\n')
print("0.7.0's ten-core days (`~/presk-data/ra4-0.7.0`, `out/ra4/day1` and `day2`) priced under each form of the layout cost model, "
      "every form fitted on the definitive calibration (D68) by `presk refit`; `0.7.0` is 0.7.0's own profile. "
      'Development figures: they decide nothing (plan §3.4). A form is *load term*-*combination*: '
      '`none` has no load term (0.7.0\'s model), `group` a load rate per group, `machine` one for the machine.'
      + (' A form *+rules* names rules of the planned cost (step 4, 4c): `lines`, `shared`, `overhead`; '
         'the calibration is costed again under them, its times kept, before the form is fitted.' if any('+' in f for f in forms) else '') + '\n')

print('## Statement M\'s nests, and O (B=16,T=256, f32)\n')
print('| form | u | ' + ' | '.join(f'`{n.split("/")[1]}` right (tied)' for n in M_NESTS) + ' | M | O: top-1 regret ' + ', '.join(f'`{n.split("/")[1]}`' for n in O_NESTS) + ' | all nests: right (tied) |')
print('|---|---|' + '---|' * len(M_NESTS) + '---|---|---|')
for f in forms:
    u, nests, _, verdict = data[f]
    m = [nests[(n, 'B=16,T=256', 'f32')] for n in M_NESTS]
    o = [nests[(n, 'B=16,T=256', 'f32')]['regret'] for n in O_NESTS]
    a = nests[('all', '', '')]
    print(f'| `{f}` | {u:.3f} | ' + ' | '.join(f'{x["right"]} ({x["tied"]})' for x in m) + f' | {verdict.get("M", "")} | {", ".join(o)} | {a["right"]} ({a["tied"]}) |')

print('\n## Per decision class, every nest: right (tied)\n')
print('| form | ' + ' | '.join(CLASSES) + ' |')
print('|---|' + '---|' * len(CLASSES))
for f in forms:
    cl = data[f][2]
    print(f'| `{f}` | ' + ' | '.join(f'{cl[c]["right"]} ({cl[c]["tied"]})' if c in cl else '—' for c in CLASSES) + ' |')

print('\n## Every contraction nest swept whole: right (tied)\n')
keys = sorted(k for k in data[forms[0]][1] if k[0] != 'all' and k[0] in O_NESTS + ['gemm/%2'] and k[1] != 'D=1024,H=16,B=1,T=384')
print('| form | ' + ' | '.join(f'`{k[0].split("/")[1]}` {k[1]} {k[2]}' for k in keys) + ' |')
print('|---|' + '---|' * len(keys))
for f in forms:
    n = data[f][1]
    print(f'| `{f}` | ' + ' | '.join(f'{n[k]["right"]} ({n[k]["tied"]})' for k in keys) + ' |')

print('\n## The hybrid chooser (statement Q\'s rule), on the development set\n')
print('Measure only the points predicted within 1 + u of the best prediction: the best of them against the best measured point '
      '(Q asks ≤ 1.05), and their share of the sweep\'s kernel time (Q asks ≤ 10 %). Candidates / points.\n')
hk = sorted(hyb[forms[0]])
print('| form | ' + ' | '.join(f'`{k[0].split("/")[1]}` {k[1]} {k[2]}' for k in hk) + ' |')
print('|---|' + '---|' * len(hk))
for f in forms:
    h = hyb[f]
    print(f'| `{f}` | ' + ' | '.join(f'{h[k]["regret"]:.2f}, {100 * h[k]["share"]:.1f} % ({h[k]["candidates"]}/{h[k]["n"]})' if k in h else '—' for k in hk) + ' |')
