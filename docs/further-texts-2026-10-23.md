# The certificate on a third and a fourth text: the Code du travail and the Code de procédure pénale (plan 0.16.1 step 3, D171)

> The plan's §7 fixed the two texts before any measurement, and so it
> fixed Z3. On the Code du travail and on the Code de procédure pénale,
> each taken alone, D168's Y1 and Y2 must hold for Z2's two realizations
> (V6 and the course logits at ε\*) and for every realization of the
> re-solved curves that is certified on the Code civil. The verdict is per
> text.
>
> The realizations are 0.16.1's (D170). 0.16.0's ran beside them for
> comparison, on D163's binary that wrote their Code civil rows. Nothing
> was calibrated, solved, trained or timed. The script is
> `experiments/further_texts.sh` (`make further-texts`). The outputs are
> in `out/further-<text>/`, `out/margins-<text>{,-0.16.0}/` and
> `out/margins-course-<text>{,-0.16.0}/`, and they are archived in
> presk-project `data/further-texts-0.16.1.tar.xz`.

**The answer.**
- *Y1 holds on both texts, at both scales.* With 0.16.1's realizations
  every realization certified on the Code civil stays inside ε\* and flips
  no decision. 0.16.0's did not: they fail Y1 on both texts at D = 64 and
  on the Code de procédure pénale at the course constants.
- *Y2 holds at D = 64 and fails at the course constants on both texts.*
  - On the Code du travail two re-solved rows fall just below half their
    Code civil gap (0.49× and 0.495×). The Code civil is the harder text
    for them.
  - On the Code de procédure pénale four rows sit at 2.07×, in one window,
    after an opening parenthesis.

  Every gap involved is at `f32`'s level, 1.8·10⁻⁵ at most, 27 times inside
  ε\*.
- **Z3 fails on both texts, on Y2 at the course constants.** It holds at
  D = 64 on both.

Neither text fails at the Code pénal's kind of position, a capital letter
inside running text whose LayerNorm is ill-conditioned (window 261, D168).
The worst positions here are the Code civil's own kind: an article's
heading, an article number, a parenthesis.

## 1. The data

Both texts come from the container that holds `civil.md.zip`
(`…/dila-legi-codes/`), fetched on 2026-10-22. Their sha256 were fixed at
the first fetch, and the script refuses another:
- `travail.md`: `df267bfe…148d`;
- `procedure_penale.md`: `0937f889…a1ae`.

The rule (the plan's §7) maps the typographic characters onto the Code
civil's vocabulary and deletes the rest:
- `’` and `‘` to `'`;
- `“` and `”` to `"`;
- `…` to `...`;
- `‒`, `―` and `−` to `-`.

The soft hyphen is deleted, and so is every other character outside the 92.
`second_corpus.py map` logs both counts per character, and how many fall in
the judge set (`out/further-<text>/mapped.tsv`). Where the Code pénal needed
only deletions (D168, its four `“` `”` deleted), the mapping changes the
text.

| | Code du travail | Code de procédure pénale |
|---|---|---|
| characters, distinct, outside the vocabulary | 7 257 450, 133, 41 | 3 687 224, 120, 28 |
| mapped (characters), in the whole text | 649 (7) | 222 (7) |
| deleted (characters), in the whole text | 48 521 (34) | 16 861 (21) |
| kept; CRLF kept | 7 208 941; 516 | 3 670 365; 191 |
| windows of 256 | 28 159 | 14 337 |
| in the judge set (the first 461 windows, 118 017 characters): mapped, deleted | 0, 0 | 2 (`“ ”`), 4 (`[ ]`) |
| what the judge set covers | art. L1 to L1226-1-3 (213 articles) | the preliminary article to art. 40-2 (111 articles) |

Each judge set is the text's first 461 windows. It is the same stream cut
at 461·256 + 1 tokens, and its samples were checked equal to the whole
file's first 461. Like the texts of record, it has 118 016 positions.

## 2. The texts under the exact model

| | Code civil | Code du travail | Code de procédure pénale |
|---|---|---|---|
| exact loss, D = 64 | 2.7736 | 2.7722 | 2.7328 |
| exact accuracy, D = 64 | 0.2424 | 0.2398 | 0.2466 |
| own ε\*(10⁻³), D = 64 | 4.670·10⁻⁴ | 5.532·10⁻⁴ (1.185×) | 5.381·10⁻⁴ (1.152×) |
| exact loss, course | 2.3307 | 2.3587 | 2.3244 |
| exact accuracy, course | 0.3037 | 0.2956 | 0.2991 |
| own ε\*(10⁻³), course | 4.961·10⁻⁴ | 5.732·10⁻⁴ (1.155×) | 4.864·10⁻⁴ (0.981×) |
| m < 2·10⁻³, course | 220 | 199 | 216 |

The exact model is as good on both texts as on the Code civil, at both
scales: within 0.03 nats and 0.8 points of accuracy. ε\*, the Code civil's,
certifies on both. Each text's own ε\* is reported beside it and certifies
nothing.

## 3. D = 64

The full tables are `out/further-<text>/d64-0.16.1.md` and
`d64-0.16.0.md`. The rows Z3 is posed on, 0.16.0 → 0.16.1, give the gap on
the text and its flips:

| realization | Code civil, 0.16.0 → 0.16.1 | Code du travail | Code de procédure pénale |
|---|---|---|---|
| V6 (`logits-4.67e-4`) | 1.431·10⁻⁴ → 2.883·10⁻⁶ | 1.592·10⁻⁴, **2 flips** → 3.145·10⁻⁶, 0 | 1.493·10⁻⁴, **1 flip** → 3.004·10⁻⁶, 0 |
| the curve's 10⁻⁴ (`d016-1e-4`) | 3.467·10⁻⁴ → 3.090·10⁻⁶ | 3.208·10⁻⁴, **1** → 3.145·10⁻⁶, 0 | 3.065·10⁻⁴, 0 → 2.865·10⁻⁶, 0 |
| 10⁻⁵ | 1.623·10⁻⁴ → 2.883·10⁻⁶ | 1.761·10⁻⁴, **2** → 3.145·10⁻⁶, 0 | 1.667·10⁻⁴, **1** → 3.004·10⁻⁶, 0 |
| 10⁻⁶ | 7.644·10⁻⁵ → 2.948·10⁻⁶ | 7.356·10⁻⁵, **1** → 2.878·10⁻⁶, 0 | 6.284·10⁻⁵, 0 → 2.827·10⁻⁶, 0 |
| 10⁻⁷ | 6.207·10⁻⁷ → 7.339·10⁻⁷ | 0 flips at both versions | 0 flips at both versions |
| v0.10.0's curve, 10⁻⁴ (`curve-1e-4`): **outside Z3**, not re-solved | 7.509·10⁻⁵, the same row at both versions | 7.380·10⁻⁵, **1 flip** | 6.305·10⁻⁵, 0 |

| | Code du travail | Code de procédure pénale |
|---|---|---|
| Y1, 0.16.0's realizations (12 certified) | **fails** on V6 and on the curve's 10⁻⁴, 10⁻⁵ and 10⁻⁶ (and v0.10.0's 10⁻⁴): 1 or 2 flips at near-ties, inside ε\* | **fails** on V6 and on the curve's 10⁻⁵: 1 flip each, inside ε\* |
| Y1, 0.16.1's realizations (12 certified) | **holds** for every row of Z3 (0 flips); v0.10.0's 10⁻⁴, outside Z3, keeps its 1 flip | **holds** (0 flips) |
| Y2, 0.16.0 → 0.16.1 | holds: 0.68–1.50×, median 1.04× → 0.68–1.51×, median 1.02× | holds: 0.77–1.77× → 0.78–1.60×, median 1.00× |

So 0.16.0's V6 would have failed Y1 on both new texts, by one or two
near-tie flips inside ε\*, and 0.16.1's does not. At 0.16.1 the certified
rows measure all-`f32`'s gap, 2.9–3.1·10⁻⁶ on either text, and the bound
#{m < 2 gap} is 0–2. The worst window is 1.3–1.5 times the median, as for
all-`f32`. At D = 64 no position stands out.

## 4. The course constants

The exact logits of each judge set took about 2 h 17 at the course
constants: twelve chunks of four windows at a time, the two texts one after
the other, then regrouped by 32. The full tables are
`out/further-<text>/course-0.16.1.md` and `course-0.16.0.md`.

| realization | Code civil, 0.16.0 → 0.16.1 | Code du travail, 0.16.0 → 0.16.1 | Code de procédure pénale, 0.16.0 → 0.16.1 |
|---|---|---|---|
| the logits at ε\* | 4.228·10⁻⁴ → 8.695·10⁻⁶ | 2.002·10⁻⁴ (0.47×) → 6.505·10⁻⁶ (0.75×) | **5.757·10⁻⁴ (1.16 ε\*)** → 1.797·10⁻⁵ (**2.07×**) |
| the trained curve, 10⁻³ (`ct-1e-3`) | 6.518·10⁻³, 29 flips → 8.695·10⁻⁶ | not certified → 6.505·10⁻⁶ (0.75×) | not certified → 1.797·10⁻⁵ (**2.07×**) |
| 10⁻⁴ | 2.627·10⁻⁴ → 8.695·10⁻⁶ | 1.199·10⁻⁴ (0.46×) → 6.505·10⁻⁶ (0.75×) | 3.547·10⁻⁴ (1.35×) → 1.797·10⁻⁵ (**2.07×**) |
| 10⁻⁵ | 2.059·10⁻⁴ → 1.292·10⁻⁵ | 9.802·10⁻⁵ (0.48×) → 6.400·10⁻⁶ (**0.495×**) | 2.145·10⁻⁴ (1.04×) → 1.441·10⁻⁵ (1.12×) |
| 10⁻⁶ | 1.093·10⁻⁶ → 5.845·10⁻⁷ | 0.40× → 0.87× | 0.90× → 1.07× |
| 10⁻⁷ | 6.263·10⁻⁷ → 1.161·10⁻⁷ | 0.46× → 1.38× | 1.11× → 1.45× |
| D129's curve, 10⁻² | 3.169·10⁻², not certified → 8.859·10⁻⁶ | → 7.380·10⁻⁶ (0.83×) | → 1.758·10⁻⁵ (1.98×) |
| 10⁻³ | 2.629·10⁻⁴ → 8.676·10⁻⁶ | 0.67× → 0.75× | 1.41× → 1.796·10⁻⁵ (**2.07×**) |
| 10⁻⁴ | 2.043·10⁻⁴ → 1.230·10⁻⁵ | 0.48× → 6.038·10⁻⁶ (**0.49×**) | 1.05× → 1.24× |
| 10⁻⁵ to 10⁻⁷ | 5·10⁻⁶ and below | 0.71–1.25× | 0.92–1.05× |
| all-`f32`, for comparison | 1.755·10⁻⁵ | 1.302·10⁻⁵ (0.74×) | 2.333·10⁻⁵ (1.33×) |

Every realization certified on the Code civil has 0 flips on both texts at
0.16.1. Their bounds #{m < 2 gap} are 0 on the Code du travail and at most 2
on the Code de procédure pénale.

| | Code du travail | Code de procédure pénale |
|---|---|---|
| Y1, 0.16.0's realizations (12 certified) | holds | **fails** on the logits at ε\*: 5.757·10⁻⁴ = 1.16 ε\*, 0 flips |
| Y1, 0.16.1's realizations (14 certified) | **holds** | **holds** |
| Y2, 0.16.0's | fails on 7 of 22, low (0.40–0.49×), `int4-pow2` among them | fails on 2 of 22, D129's 10⁻² (0.48×) and `int4-pow2` (2.20×) |
| Y2, 0.16.1's | **fails** on 3 of 22: D129's 10⁻⁴ (0.49×), the trained curve's 10⁻⁵ (0.495×), and `int4-pow2` (0.49×, unmoved, outside Z3) | **fails** on 5 of 22: the logits, the trained curve's 10⁻³ and 10⁻⁴ and D129's 10⁻³ (all 2.07×), and `int4-pow2` (2.20×, unmoved, outside Z3) |

**What carries the tails.**
- *The Code du travail.* The rows that fail Y2 do so because their gap on
  the Code civil is an outlier: window 146 at 1.23–1.29·10⁻⁵, after the
  heading `**Art` of an article. The Code du travail has no position that
  bad for them. Their worst windows there are 205 and 79, after `**Art`
  and `L1142-`, at 6.0–6.4·10⁻⁶, and the median windows of the two texts
  are within 5 % of each other. For the logits at ε\* the ten worst windows of
  the Code du travail are all article headings and article numbers
  (`**Art`, `L1142-`, `L1225-`), and one is after an opening parenthesis.
- *The Code de procédure pénale.* All four rows at 2.07× have their worst in
  window 424, position 79: `aux articles 15 (`, a prediction after an
  opening parenthesis. All-`f32` is also worst there, at 2.333·10⁻⁵, and
  0.16.1's realizations measure less, 1.797·10⁻⁵.
  - The ratio of 2.07× comes from the Code civil's side. There 0.16.1's
    realizations measure half all-`f32`'s.
  - 0.16.0's logits at ε\* measure 5.757·10⁻⁴ in the same window, the
    `bf16` tail D170 removed.
  - The second-worst window, 15, is half the worst (8.9·10⁻⁶).

**The Code pénal's kind of position.** D168's window 261 is "par le
J|ournal officiel": a capital letter inside running text, ill-conditioned
in every type, which LayerNorm's ε in `bf16` amplified 25×. Neither text's
worst positions are of that kind. Among the ten worst windows of the logits
at ε\*, none has its worst position after a capital letter on the Code du
travail. The Code de procédure pénale has four, every one of them at the
start of a paragraph (`⏎⏎I`, `⏎⏎N`), at 5.2–7.2·10⁻⁶. The tails here are
after `(` and after article headings, which are the Code civil's own kinds.
For the same logits the Code civil's ten worst windows are after `(` (406,
402) and after `**Art` (146, 74, 285, 77, 201).

## 5. What it says

- **The certificate's promise holds on unseen text.** It promises the gap
  at most ε\* and so at most δ of the decisions flipped. On the Code du
  travail and the Code de procédure pénale, as on the Code pénal (D170),
  every realization certified on the Code civil stays inside ε\* and flips
  nothing. That is Y1, at both scales, with 0.16.1's realizations.
- **The rules bought this.** 0.16.0's realizations fail Y1 on both new texts
  at D = 64, by one or two near-tie flips inside ε\*, and on the Code de
  procédure pénale at the course constants, at 1.16 ε\*. 0.16.1's do not.
- **Y2, the relative stability of the gap, fails at course scale on both
  texts,** by small factors on either side: 0.49× and 0.495×, and 2.07×.
  The gaps are `f32`'s rounding, 6·10⁻⁶ to 1.8·10⁻⁵. A realization near
  `f32` keeps `f32`'s spread across windows and texts: all-`f32`'s own ratio
  is 0.74× on the Code du travail and 1.33× on the Code de procédure pénale.
  Its gap moves with whichever text holds the worse position, more than 2×
  either way. Y2's band was fixed before the rules, when the gaps were
  `bf16`'s.
- **Z3 fails on both texts, on Y2 at the course constants;** it holds at
  D = 64 on both. Per text:
  - *Code du travail:* Y1 holds and Y2 holds at D = 64; at the course
    constants Y1 holds and Y2 fails, on two rows at 0.49× and 0.495×.
  - *Code de procédure pénale:* Y1 holds and Y2 holds at D = 64; at the
    course constants Y1 holds and Y2 fails, on four rows at 2.07×.

## 6. How to run it again

```sh
make further-texts                               # both texts, every stage
TEXT=travail ./experiments/further_texts.sh data d64 exact course report
```

The stages:
- **data:** the fetch, the sha256 check, the mapping, the tokens, the
  windows and the judge set;
- **d64:** `margins.sh` on 0.16.1's realizations, then 0.16.0's on D163's
  binary, the exact logits shared;
- **exact:** the course exact logits, chunks of 4, regrouped by 32, alone;
- **course:** `margins_course.sh`'s native stages for both versions,
  0.16.0's curve from `out/pre-0.16.1/` (`CURVE`);
- **report:** `second_corpus.py report --name … --short …` against the
  Code civil's rows of the same version.

Outputs present are reused. The night of 2026-10-22 ran
`out/step3-0.16.1/course.sh` after D170's phase A: the exact logits from
21:52 to 02:26, the native stages to 05:28.
