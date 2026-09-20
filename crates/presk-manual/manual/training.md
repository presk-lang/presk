# Training: `presk train`

`presk train FILE.psk` executes the `training run` of a file on the reference
interpreter (0.2.0, step 5). Nothing in the loop is special to the
interpreter: the objective and its gradient are PSK-Core functions (the
gradient by reverse-mode differentiation, see the `language` chapter), the
optimizer is the PSK-Core `adamw.init`, `adamw.lr` and `adamw.step` lowered
from the `optimizer` block, the data pipeline is the one of the `data`
chapter; `presk train` sequences them.

## The loop

```text
θ₀ = M.init(key(seed))          s₀ = adamw.init(θ₀)          t = 0
for epoch in 1..=epochs:
  for batch in pipeline(epoch):                 -- shuffle re-keyed per epoch, batch(B), take
    t += 1
    ℓ = objective(θ, batch)
    g = objective.vjp(θ, batch, 1)              -- the cotangent of θ
    (θ, s) = adamw.step(θ, s, g, t, total)      -- lr(t, total) inside
    log ℓ, lr, ‖g‖;  checkpoint θ every N steps to ckpt/step-<t>.psk.d
write ckpt/final.psk.d
```

- `total = epochs × batches per epoch` is the horizon of the schedule; with
  `--steps N` the loop stops after `N` steps and `total = N`.
- The model's key comes from `--seed`; the shuffle's from its own `seed`.
  Same seed, same data, same `--bits`: same weights on every machine.
- The dimensions of every function called (`B`, `T`, the file's `dim`s) are
  those the checker bound from the pipeline and the data file (`presk check
  --summary`, `dims`); a dimension the run does not fix is an error.
- The loop runs over the runtime seam (`presk_rt::Backend`, 0.3.0 step 1):
  the interpreter is the executor behind `--backend interp`, and its exact
  values cross the seam untouched, so the results are those of 0.2.0 digit
  for digit; native backends plug in at the same point.
- Checkpoints and `final.psk.d` are one-sample `.psk.d` files of type
  `params M`, `f64` leaves (the interpreter's extra precision is rounded
  away on the way out; spec question 31). They are written to the directory
  of the `checkpoint every N steps to "dir"` clause, relative to the `.psk`
  file, or to `--ckpt DIR`; without either, no file is written and the
  command says so.

```text
presk train FILE.psk [--const NAME=N]… [--data FILE.psk.d] [--steps N] [--epochs N]
                     [--seed S] [--bits B] [--log-every N] [--ckpt DIR] [--json]
```

| Flag | Meaning |
|---|---|
| `--const NAME=N` | as for `presk run`: override a constant of the file or its imports (this is how the MiniGPT example trains at a size the interpreter can afford) |
| `--data FILE` | use this data file instead of the run's |
| `--steps N` | stop after `N` steps; the schedule's horizon becomes `N` |
| `--epochs N` | override the run's `epochs` |
| `--seed S`, `--bits B` | as for `presk run` (defaults 0 and 256) |
| `--log-every N` | one line per `N` steps on standard error (default 1; 0 prints the last step only) |
| `--ckpt DIR` | override the checkpoint directory |
| `--json` | on standard output: `{model, steps, total, log: [{step, epoch, loss, lr, grad_norm}], final, checkpoints}` |

Exit codes: 0; 1 a diagnostic (compile time, or `E7001`/`E9xxx` at run
time); 2 no `training run` in the file or an unreadable data file.

## Two runs

A linear regression, `tests/train/mlp_regression.psk`: a `Linear{2, 1}` on
256 samples of `y = 2·x₀ − x₁ + 0.5`, AdamW at `lr = 0.05`, batches of 16:

```sh
presk train tests/train/mlp_regression.psk --steps 200 --log-every 50 --ckpt out/reg_ckpt
```

```text
step 50  epoch 4  loss 0.377759  lr 5.000e-2  |g| 6.6034e-1
step 100  epoch 7  loss 0.014019  lr 5.000e-2  |g| 1.8483e-1
step 150  epoch 10  loss 0.000002  lr 5.000e-2  |g| 2.1463e-3
step 200  epoch 13  loss 0.000000  lr 5.000e-2  |g| 4.7522e-4
presk train: 200 steps of Reg, final loss 0.000000, weights in out/reg_ckpt/final.psk.d
```

The weights end near `(2, −1)` and `0.5` (a test asserts it). The MiniGPT of
the specification, on the example data (`examples/minigpt/data/tokens.psk.d`,
510 windows of 8 tokens over a vocabulary of 64), with its constants reduced:

```sh
presk train tests/examples/minigpt/train_minigpt.psk \
     --const V=64 --const D=16 --const H=2 --const L=2 --steps 20 --bits 64 --ckpt out/ckpt
```

```text
step 5  epoch 1  loss 4.158873  lr 7.500e-7  |g| 5.4769e-1
step 10  epoch 1  loss 4.158840  lr 1.500e-6  |g| 5.4270e-1
step 15  epoch 1  loss 4.158777  lr 2.250e-6  |g| 5.1438e-1
step 20  epoch 2  loss 4.158680  lr 3.000e-6  |g| 5.1596e-1
presk train: 20 steps of MiniGPT, final loss 4.158680, weights in out/ckpt/final.psk.d
```

The loss starts at `ln 64 ≈ 4.1589` (the `head` is zero-initialized, so the
logits are uniform) and decreases slowly because the example's schedule is
`cosine(warmup = 2000)`: at step 20 the rate is `3e-4 · 20/2000`. Twenty steps
take about half a minute at 64 bits: the interpreter is a reference, not a
trainer. The same command with `--backend native` (and `--realize f32` for
single precision) trains the same program on compiled kernels, in 0.4 s here
and within the last digits of the reference (`native` chapter); the PyTorch
export of the run is the other way to train at speed. `make train-examples`
runs the reference commands and the native ones; the 20-step log is a golden
output of the test suite.

## In PyTorch: `presk export --train`

```sh
presk export tests/examples/minigpt/train_minigpt.psk --target pytorch --mode readable --train \
     --conformance-entry train:20 --const V=64 --const D=16 --const H=2 --const L=2 -o out/trainloop_pt
cd out/trainloop_pt && python train.py --steps 20 --dtype float64 --log-every 5 --ckpt ckpt
```

`--train` writes, next to the exported modules and objective, `pskd.py` (a
reader of `.psk.d` files as a `torch.utils.data.Dataset`, with the type
parser for the closed data types), `train.py` (the run as a PyTorch loop:
`DataLoader` with the run's batch size and PyTorch's shuffling seeded by the
run's seed, the model built with the run's dimensions, `torch.optim.AdamW`
with the block's hyperparameters, the schedule as a multiplier set before
every step, `clip_grad_norm_`, epochs or `--steps`, checkpoints with
`torch.save` to the run's directory or `--ckpt`; `--dtype float64` for
comparisons), and a copy of the data file. The same 20 steps take a tenth
of a second where the interpreter takes half a minute:

```text
step 5/20  epoch 1  loss 4.158875  lr 7.500e-07
step 10/20  epoch 1  loss 4.158844  lr 1.500e-06
step 15/20  epoch 1  loss 4.158791  lr 2.250e-06
step 20/20  epoch 2  loss 4.158709  lr 3.000e-06
train.py: 20 steps in 0.1 s, final loss 4.158709, weights in …/ckpt/final.pt
```

The loss differs from the interpreter's in the last digits because the
batches differ: `train.py` shuffles with PyTorch, not with the pipeline's
ChaCha permutation. The **K-step conformance test** removes that difference:
`--conformance-entry train:K` runs `K` steps on the interpreter (256 bits,
no checkpoint) and writes `θ₀`, the parameters after `K` steps and the sample
indices of every batch (`conformance/{inputs,expected,batches}.json`);
`test_conformance.py` loads `θ₀` into the model, replays the same batches
with `train.py`'s optimizer and schedule in float64, and compares `θ_K`. The
export also runs the same `K` steps on the native backend in `f64` and `f32`
and writes `conformance/native_f64.json` and `native_f32.json`, which the
`experiments` chapter reads.
On the tiny MiniGPT, 20 steps with weight decay, clipping and the cosine
warm-up agree to a worst relative gap of `1.3e-10` (tolerance `rtol 1e-6`,
`atol 1e-9`; the test prints the measured gap). This is the "exact versus
almost" experiment of the specification applied to training. The batch axis
is the leading axis of every tensor of the batch in both `train.py` and the
test.

## Randomness per step

An objective may take a key between the parameter
tree and the batch, `fn loss(θ : params M, key : key, batch : …) -> ℝ[]`.
The loop derives it from the run's seed as the step's child key, so a run is
reproducible and every step drops differently; `dropout` draws its mask from
that key (`language` chapter). A training run is the only place where
dropout drops: everywhere else its mask is all ones, the expectation of the
random function. A `stack{L}` fold passes the same extra arguments to each of
its iterations, so the blocks of a stacked module would share one key; giving
them independent ones needs a key derived from the loop index, which the
specification has still to settle (question 33).

## Not in this version

Resuming from a checkpoint, the
`parallel` and `precision` clauses (checked, not executed; reproduced as
comments in the export), optimizers other than `adamw`, `.psk.d` checkpoints
from `train.py` (it writes `.pt`), a training loop in code generation.
