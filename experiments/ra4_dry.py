#!/usr/bin/env python3
"""The dry run of a fresh sweep, sized (experiments/ra4_dry.sh; plan 0.8.0
§3.5, step 5, D72).

    ra4_dry.py choose DIR REPEAT      the nest to time: NEST AT TYPE
    ra4_dry.py estimate DIR REPEAT    the estimate, in Markdown
    ra4_dry.py day REPEAT HOURS DIR…  a day of several dry runs, each
                                      configuration swept whole or on a sample

DIR/plans/<config>/plan.txt is `presk plan --cost` for one configuration:
every point's prediction, then one summary line per launch with the size of
its space. DIR/nest/<config>/ is `presk autotune` of one nest, the dry run.
A sweep launches each point once to warm up and REPEAT times more, so its
time is about (REPEAT + 1) times the sum of the medians, and some time a
point outside the kernels (compiling it, comparing its bits). The estimate
scales the model's predictions by the nest's measured time over its
predicted, and adds its time a point outside the kernels. On 0.7.0's day
one, priced by the frozen model, the longest nest's ratio gives each
configuration's sweep 3 to 8 % short, and the day's 5 % short (D72).

F1 (step 6, D73) has two configurations with other constants, and a dry run
each, its own nest timed: `day` adds their estimates into a day and keeps
whole the configurations that fit in HOURS, the quickest first; the others
are swept on a sample.

T3 (plan 0.10.0, step 2, D90) is one configuration of the course GPT's
gradient restricted to four nests, the derivatives of `dot` (ra4.sh's NEST):
its plans price those nests only, the dry run times the longest of them, and
`day` sizes the day from that one directory."""
import glob
import json
import os
import re
import sys

POINT = re.compile(r'^  (\S+)\s+([0-9.]+) µs  ')
LAUNCH = re.compile(r'^(\S+)\s.*?\s(\d+) points\s+([0-9.]+) µs')
WALL = re.compile(r'^-- wall time ([0-9.]+) s')


def launches(path):
    """Each launch of a plan.txt: (nest, points in its space, the sum of its
    predictions scaled to every point, the plan's wall time)."""
    out, preds, wall = [], [], 0.0
    for line in open(path, encoding='utf-8'):
        m = POINT.match(line)
        if m:
            preds.append(float(m.group(2)) * 1e-6)
            continue
        m = LAUNCH.match(line)
        if m:
            n = int(m.group(2))
            total = sum(preds) * n / len(preds) if preds else 0.0
            out.append((m.group(1), n, total))
            preds = []
            continue
        m = WALL.match(line)
        if m:
            wall = float(m.group(1))
    return out, wall


def configs(d):
    out = []
    for p in sorted(glob.glob(os.path.join(d, 'plans', '*', 'plan.txt'))):
        shape, dtype = open(os.path.join(os.path.dirname(p), 'config.txt')).read().split()
        ls, wall = launches(p)
        out.append((shape, dtype, ls, wall))
    return out


def hms(s):
    if s < 10:
        return f'{s:.1f} s'
    s = round(s)
    if s < 60:
        return f'{s} s'
    if s < 3600:
        return f'{s // 60} min {s % 60:02d} s'
    return f'{s // 3600} h {s % 3600 // 60:02d} min'


def choose(d, repeat):
    best = None
    for shape, dtype, ls, _ in configs(d):
        per = {}
        for nest, _, total in ls:
            per[nest] = per.get(nest, 0.0) + (repeat + 1) * total
        for nest, t in per.items():
            if best is None or t > best[0]:
                best = (t, nest, shape, dtype)
    if best is None:
        sys.exit('ra4_dry.py: no prediction in ' + d)
    print(best[1], best[2], best[3])


def sized(d, repeat):
    """The dry run under d: its nest's sweep, and each configuration
    estimated from it, (label, type, nests, points, predicted, sweep, with
    plan and captures)."""
    sweeps = glob.glob(os.path.join(d, 'nest', '*', 'sweep.json'))
    if len(sweeps) != 1:
        sys.exit(f'ra4_dry.py: {len(sweeps)} sweeps under {d}/nest, one expected')
    s = json.load(open(sweeps[0]))
    lines = [json.loads(l) for f in glob.glob(os.path.join(os.path.dirname(sweeps[0]), '*.jsonl')) for l in open(f)]
    predicted = [p for p in lines if p['predicted'] is not None]
    ratio = sum(p['median'] for p in predicted) / sum(p['predicted'] for p in predicted)
    kernels = sum((repeat + 1) * p['median'] for p in lines)
    outside = (s['sweep_seconds'] - kernels) / len(lines)
    fixed = s['wall_seconds'] - s['sweep_seconds'] + s['static_capture_seconds']
    rows = []
    for shape, dtype, ls, wall in configs(d):
        points = sum(n for _, n, _ in ls)
        pred = sum(t for _, _, t in ls)
        sweep = (repeat + 1) * ratio * pred + outside * (points + len(ls))
        label = f'{s["consts"] and ",".join(s["consts"]) + "," or ""}{shape}'
        rows.append((label, dtype, len(ls), points, pred, sweep, sweep + fixed + wall))
    return {'sweep': s, 'lines': lines, 'nest': lines[0]['nest'], 'ratio': ratio, 'kernels': kernels,
            'outside': outside, 'fixed': fixed, 'rows': rows}


def estimate(d, repeat):
    z = sized(d, repeat)
    s, lines, nest, ratio, kernels = z['sweep'], z['lines'], z['nest'], z['ratio'], z['kernels']
    outside, fixed = z['outside'], z['fixed']
    rows, day = [], 0.0
    for label, dtype, nests, points, pred, sweep, total in z['rows']:
        day += total
        rows.append(f'| {label} | {dtype} | {nests} | {points} | {pred:.3g} | {hms(sweep)} | {hms(total)} |')
    cores = next((l.split('cores ')[-1].strip() for l in open(os.path.join(d, 'nest', 'day.txt')) if 'cores ' in l), '?')
    print(f'# Dry run: `{s["entry"]}`, pinned to {cores}\n')
    print(f'The nest timed: `{nest}` at {s["config"]} in {s["realization"]}, {len(lines)} points (its baseline among them) '
          f'swept in {s["sweep_seconds"]:.1f} s, {kernels:.1f} s of it in the kernels ({repeat + 1} launches a point). '
          f'Measured over predicted, the sum of its medians over the sum of its predictions: **{ratio:.3f}**. '
          f'Outside the kernels: {1e3 * outside:.1f} ms a point. Outside the sweep: {fixed:.1f} s a configuration '
          f'(the capture that runs the entry, the costing, the capture of shapes), and `presk plan`\'s wall time.\n')
    print('Each configuration: its nests (launches), the points of their spaces, the sum of the model\'s predictions, '
          f'the sweep estimated as ({repeat + 1} × ratio × predictions + a point\'s time outside the kernels × points), '
          'and with the time outside the sweep.\n')
    print('| config | type | nests | points | predicted (s) | sweep | with plan and captures |\n|---|---|---|---|---|---|---|')
    print('\n'.join(rows))
    print(f'| **one day** | | | | | | **{hms(day)}** |\n')
    print(f'Each day of the sweep is estimated at {hms(day)}. The ratio is one nest\'s: on 0.7.0\'s days, priced by the '
          'frozen model, the ratios of the contraction nests ranged from 1.16 to 1.60 (D72).')


def day(repeat, hours, dirs):
    """Several dry runs, one per configuration, added into a day: the
    configurations are kept whole, the quickest first, while the day fits in
    `hours`; the others are swept on a sample (plan 0.8.0 §3.5)."""
    rows = []
    for d in dirs:
        z = sized(d, repeat)
        for label, dtype, _, points, pred, _, total in z['rows']:
            rows.append((label, dtype, points, pred, z['nest'], z['ratio'], total))
    used, whole = 0.0, set()
    for r in sorted(rows, key=lambda r: r[6]):
        if used + r[6] <= 3600 * hours:
            used += r[6]
            whole.add((r[0], r[1]))
    print(f'# Dry run: one day of {len(rows)} configurations\n')
    print('Each configuration is estimated from its own dry run: its longest-predicted nest swept, pinned, as a day '
          'sweeps it, and the model\'s predictions scaled by that nest\'s measured time over its predicted (each '
          'directory\'s `dry.md`, whose "one day" is that configuration alone). A configuration is swept whole while '
          f'the day fits in {hours:g} h, the quickest first; the others are swept on a sample, and their M is '
          'inconclusive (plan 0.8.0 §3.5).\n')
    print('| config | type | points | predicted (s) | nest timed | ratio | estimate | swept |\n|---|---|---|---|---|---|---|---|')
    for label, dtype, points, pred, nest, ratio, total in rows:
        swept = 'whole' if (label, dtype) in whole else '**on a sample**'
        print(f'| {label} | {dtype} | {points} | {pred:.3g} | `{nest}` | {ratio:.3f} | {hms(total)} | {swept} |')
    total = sum(r[6] for r in rows)
    print(f'| **one day, every configuration whole** | | | | | | **{hms(total)}** | |\n')
    if len(whole) == len(rows):
        print(f'Each day is estimated at {hms(total)}, within {hours:g} h: every configuration is swept whole.')
    else:
        print(f'Swept whole, a day would take {hms(total)}, more than {hours:g} h: the configurations marked are '
              'swept on a sample, set by a commit before day one.')


if __name__ == '__main__':
    if len(sys.argv) >= 5 and sys.argv[1] == 'day':
        day(int(sys.argv[2]), float(sys.argv[3]), sys.argv[4:])
    elif len(sys.argv) == 4 and sys.argv[1] in ('choose', 'estimate'):
        (choose if sys.argv[1] == 'choose' else estimate)(sys.argv[2], int(sys.argv[3]))
    else:
        sys.exit('usage: ra4_dry.py choose|estimate DIR REPEAT, or ra4_dry.py day REPEAT HOURS DIR...')
