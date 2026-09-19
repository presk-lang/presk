# Experiments: exact versus almost

Two experiments that the tool can run today, with what they show and what
they do not. `make experiment` runs both (after `make venv`); the script is
`experiments/exact_vs_almost.py` and takes any output directory of
`presk export … --conformance-entry train:K`.

## 1. Shapes as types

The RMSNorm of the tutorial divides `x : ℝ[..., model: D]` by `ms : ℝ[...]`,
the per-row mean of squares. In Presk the two align by axis name, and the
checker proves it for every `B` and `D`. In plain PyTorch the natural line
`x / torch.sqrt(ms)` broadcasts from the trailing axis: it divides column `j`
by the norm of row `j`. When the batch happens to equal the model width, it
runs without any error.

```text
plain PyTorch, batch == model width: `x / sqrt(ms)` runs without error, max |wrong - right| = 0.773
Presk: `x / sqrt(ms)` with x : ℝ[..., model: D], ms : ℝ[...] aligns by name; the export inserts unsqueeze(-1)
```

The export had to learn the same lesson: it inserts `unsqueeze(-1)` exactly
where Presk's alignment (by leading prefix or by name) and PyTorch's (from the
trailing axis) differ. The `E2003` and `E2006` diagnostics of the tutorial,
with their symbolic counterexamples, are the compile-time side of the same
point: a shape error is a type error, found before the first batch.

## 2. The realization gap

`presk export --conformance-entry train:K` records the initial parameters,
the sample indices of every batch and the parameters after `K` steps of the
reference interpreter at 256 bits. The script replays those `K` steps in
PyTorch, on the same batches, with the same optimizer and schedule, in
float32 and in float64, and measures the largest absolute gap to the exact
parameters. Since 0.3.0 the export also runs the same steps on presk's own
native backend (`--backend native`, `native` chapter) in `f64` and `f32`,
and the script measures those too: four realizations of one program against
one reference.

| Run | Steps | PyTorch float32 | PyTorch float64 | presk `f32` | presk `f64` | Parameter scale |
|---|---|---|---|---|---|---|
| Tutorial tower (`tutorial/07_train.psk`, AdamW, cosine warm-up) | 10 | 3.2e-7 | 4.4e-16 | 2.0e-7 | 4.4e-16 | 1.0 |
| Tiny MiniGPT (`train_minigpt.psk`, weight decay 0.1, clipping 1.0, cosine warm-up) | 20 | 9.6e-7 | 8.9e-16 | 3.7e-7 | 6.7e-16 | 1.0 |

Read against the machine epsilons (6e-8 for float32, 1.1e-16 for float64),
all four arithmetics lose a handful of units in the last place over ten to
twenty AdamW steps on these small problems: float64 is exact to fifteen
digits, float32 to six. Presk's kernels are two to three times closer than
PyTorch's in float32, because they follow the reference operation for
operation (no fused or reassociated kernel); this is a property of naive
kernels that the e-graph pass will have to preserve or account for. Two
remarks on the measurement itself:

- The gap is *absolute*, with the model's scale given for context. An
  elementwise relative gap is meaningless here: some parameters are kept near
  zero by both arithmetics (the zero-initialized head of MiniGPT in its first
  steps), and dividing by them turns rounding noise into enormous ratios.
- What is measured is the arithmetic alone. Shuffling, initialization and the
  optimizer's definition are identical on both sides, because the export
  carries the interpreter's batches and parameters and the optimizer is a
  function of the language tested against PyTorch's (`adamw.step`, rtol
  1e-9). In an ordinary PyTorch workflow there is no reference to measure
  against; the "right" answer is whatever float32 produced.

These are the first two points of the specification's "exact versus almost"
experiment, and of its ε-to-quality curve: the error of a realization is a
number attached to a program, measured, not argued. Longer runs, larger
models, bfloat16 and mixed precision are the obvious next points; they need
the PyTorch side only (`train.py --dtype`), the reference being fixed.

## What these experiments do not show

- **Speed, beyond the first step.** Since 0.3.0 the same program trains
  natively (twenty MiniGPT steps in 0.4 s against half a minute on the
  reference; `native` chapter), but the kernels are naive loop nests, far
  behind PyTorch's BLAS on real shapes. The claim that exact semantics plus
  e-graph rewriting and static cost models yield fast kernels is the
  research program of the rest of 0.3.0 and after, untested so far.
- **Error budgets.** `precision = budget(rel_error = 1e-3)` is parsed and
  checked, not solved: the gap of a given dtype can be measured, dtypes
  cannot yet be chosen from a budget.
- **Verified transformations** (the KV cache), **parallelism**, **energy**:
  declared in the language, not implemented.

What the tool delivers today is the front half of the argument: a language
where shapes, smoothness, randomness and data are checked before anything
runs, and an exact reference that turns numerical precision into a
measurement. The back half, that this discipline buys speed rather than
costing it, is what the next version has to start showing.
