# Presk — Examples

Presk source code illustrating the constructs of the [specification](../report/chapters/02-spec.tex)
(Chapter 3 of the [report](../report/presk-report.pdf); a reference §N.M below and in the programs
is section N.M of the specification, printed 3.N.M in the report).
Each example lives in **its own directory**, with a local `README.md` that explains
what it shows and gives the `presk` commands that go with it.

> **Status.** These files *specify by example*, and they are designed to exercise
> every corner of the type system. The `presk` compiler, in a separate repository,
> keeps a copy of them, which it checks as acceptance tests (`make check-examples`).
> The `presk` commands quoted in each local `README.md` are those of the
> specification: the compiler implements `presk check` and `presk export` under these
> names, training is `presk train` there rather than `presk run`, and `presk build`
> is not implemented.

## Suggested progression

| Directory | What it illustrates |
|---|---|
| [`mlp/`](mlp/) | The bare minimum: `module`, `param`, `fn apply`, composition, typed `grad`. |
| [`layernorm/`](layernorm/) | Reductions over a named axis, vector `param`, numerical stability left to the compiler. |
| [`minigpt/`](minigpt/) | The core: causal attention, Transformer blocks, `stack{N}`, `idx[V]`, embeddings, **training**, **inference** and **build** toward an ARM+GPU SoC. |
| [`concat/`](concat/) | Slices that keep their axis, `concat` along a named axis, `min`, `max` and `clamp`, and a gradient through them. |
| [`rope/`](rope/) | Rotary position embeddings as the Llama models have them: two slices, a `concat`, and Llama 3's frequency scaling with `clamp`. |

## Conventions

- `ℝ[…]` is an **exact** real tensor in the semantics (§2); no `float32` appears in the source.
- Axes are **named** (`batch:`, `seq:`, `model:`, `heads:`, `hd:`); broadcasting and
  `ein[…]` contractions are checked axis by axis (§3.2).
- The integers of the graph are **indices** `idx[N]` (tokens, labels), never mixed
  arithmetically with reals without an explicit `real(…)` (§3.5).
- `let x = …` is an immutable binding (shadowing allowed); there is never any mutation.
- Precision is a **budget** declared at the `training`/`inference` level (§6, §7), not a
  dtype in the code.
