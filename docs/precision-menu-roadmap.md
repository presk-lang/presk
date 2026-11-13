# One menu for the float and the integer types: what it will take

> Logged 2026-10-15 (D126), after plan 0.12.0. Not a plan: the design the
> user left for later when deciding to keep `--quant` a pass on top of the
> float assignment (plan 0.12.0's step 4 asked for one menu; D123 kept
> two). To be turned into a plan when a version takes quantization further
> (an `fp8` type, Petal's `int8`, a GPU's narrow types).

## What runs now (0.12.0)

`presk solve` makes two choices in order:
1. **The float assignment** (D41, D112, D123). The menu is `f64`, `f32`
   and `bf16`, for every stored value and, since 0.12.0, every parameter
   by path. A site's term of the estimate is `s · u(t) · |v|`: `s` the
   1-norm of its cotangent, `u(t)` the unit roundoff of its type, `|v|` the
   envelope's magnitude. The greedy proposal orders the moves by bytes (or
   time) saved per unit of term added; the measurement decides.
2. **`--quant`, on top** (D44, D49). Every parameter of rank two or more
   is offered `int8` and `int4`, over its second-to-last axis. A rung's
   term is the root-sum-square `√Σ (g·δ)²` over the parameter's elements,
   with `g = ∂L/∂w` from the whole cotangent and `δ = ŵ − w` known exactly,
   since the weights are at hand. Each candidate's bytes are counted at
   its float type (D123), and a parameter it quantizes leaves `params`.

## What two passes cannot do

They cannot trade one kind of move against the other. A weight in `int8`
and a nest in `bf16` are never weighed in one order: the integer pass
spends what the float pass left of the budget. On a model where a weight
in `int8` would buy more per unit of error than a nest in `bf16`, the
float pass takes the nest first. Neither the statements of record nor plan
0.12.0's measured this; the curve and the decoder's solve ran without
`--quant`.

## Why one menu is a design, not a merge of two lists

The two terms measure different things.
- *The float term* is a bound per element times a 1-norm: every element's
  rounding is at most `u(t) · |v|`, summed with the sensitivity's weights,
  as if every rounding pushed the same way.
- *The integer term* is a root-sum-square of known moves. D49 measured the
  integer bound `Σ |g|·|δ|` at 230 times the gap on the course GPT, and the
  root-sum-square within a factor of two.

The float estimate overestimates by about as much: `make precision` at
0.12.0 estimated 2.7·10⁻³ and measured 3.9·10⁻⁶ (D124). Put in one
knapsack as they are, an integer rung would look two to three orders of
magnitude cheaper per unit of error than a float one, and the greedy
order would follow the units, not the moves.

## The two ways to one measure

1. **Both as a root-sum-square of first-order moves.** A float rung's move
   per element is `fl_t(w) − w`, as known as an integer rung's for a
   parameter: compute it exactly and price it as `√Σ (g·δ)²` too. For a
   stored value the move is not known before the run, so its term becomes
   `u(t) · √Σ (g_i·v_i)²` over the envelope, which needs the cotangent
   per element of every stored value, not its 1-norm.
   - *Memory:* the sweep's per-site 1-norms would become per-element
     accumulations, D105's peak or more at the decoder's width (the
     cotangent of a parameter is already held whole, D123).
   - *The estimate's meaning changes for every site,* so every solve would
     be re-measured, as 0.11.0 and 0.12.0 did.
2. **Both as bounds.** The integer rung priced by `Σ |g|·|δ|`, the float
   one as now. One unit, no new memory. But D49's 230-fold
   overestimate returns: the solver would almost never quantize, which is
   why D49 left it.

The first is the one worth building. The measurement decides either way,
so an estimate that ranks well matters more than one that bounds.

## What it would take

- per-element cotangents of the stored values in the sweep, or a sampled
  estimate of `√Σ (g_i·v_i)²` per site;
- one move list in `presk_precision::solve` with both kinds of rung, the
  widening order unchanged (D41);
- the solves of record and the decoder run again: the course GPT, the
  curve, BERT-large's two blocks and the decoder, with and without
  `--quant`;
- an X1-style check that a realization without `quant` gives the bits it
  gave.

About a week of work and a day of falicon, by plan 0.12.0's measure.
