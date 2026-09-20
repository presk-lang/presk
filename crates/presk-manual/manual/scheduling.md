# Scheduling: fusion groups

One kernel per primitive makes every intermediate a round trip through
memory. PSK-Sched v0 (crate `presk-sched`, 0.4.0 plan §3.4) partitions each
PSK-Core body into **fusion groups**: connected sets of elementwise nodes
that share a loop nest, so their intermediates stay in registers and only the
group's result is written. `presk dump --stage sched` prints the partition.

## What a group is

A group is a set of `let`s of one function such that

- every member is an *elementwise* primitive — `add`, `mul`, `exp`, `gelu`,
  `sqrt`, … A reduction ends a group, because its loop nest is not the nest
  of its consumers;
- every member has the same result type, so one loop nest serves them all.
  Inputs may be smaller and broadcast, exactly as the unfused kernels do;
- every *interior* member, that is every member but the root, is read only by
  other members. A value read from outside has to reach memory for the
  outside reader, so it cannot be interior.

Those three conditions are what make the group safe to launch as one kernel,
and they are what the crate's test asserts on the corpus and the examples.
The partition is structural: it depends on the body, not on the shapes or the
realization, so it is computed once per function and reused by every backend
and every dimension.

A value read twice *inside* a group appears twice in the emitted expression,
which trades an arithmetic operation for a round trip through memory. That
trade is bounded: past sixty-four nodes the group stops growing. `pow n` with
`n > 2` never joins a group for the same reason — it expands to `n − 1`
products of its argument.

**Parameter trees.** A group whose type is `params M` is mapped over the
leaves of the tree, one launch per leaf for the whole chain. This is where
most of the saving is: an optimizer step is one long elementwise chain over
every parameter, and without it each step of the chain launches again for
every leaf.

## What it removes

On the corpus GPT at `V=92, D=64, H=2, L=2, B=2, T=32`, in `f32`:

| Entry | Launches without fusion | With | Removed |
|---|---|---|---|
| `loss`, the forward pass | 108 | 81 | 25 % |
| `adamw.step` | 414 | 111 | 73 % |
| a whole training step | 1515 | 884 | 42 % |

`presk train` prints the last row for any run on a kernel backend. The plan
of this version asked for half the launches to go; 42 % is short of it, and
the reason is the one below — the forward and backward passes are where the
launches are, and their chains are cut by reductions and by values that two
consumers read.

The forward pass fuses little because reductions cut its chains: the
`(x − μ)² → mean → +ε → sqrt → divide` of a layer normalization is split at
each `mean`, and `x − μ` is read by both the variance and the division, so it
cannot be interior. The optimizer fuses well because it is one chain of
twenty-odd elementwise steps over a tree of parameters, with nothing reading
its intermediates.

## What it buys, measured

On the tiny MiniGPT, native `f64`, medians of 21 calls, three runs each,
before and after the pass:

| Entry | Before | After |
|---|---|---|
| `MiniGPT.apply` | 0.57–0.61 ms | 0.59–0.61 ms |
| training `loss` | 0.58–0.63 ms | 0.60–0.62 ms |
| `adamw.step` | 1.54–1.60 ms | 1.31–1.33 ms |

The forward pass does not move: its time is in the contractions, which fusion
does not touch, and removing a quarter of its launches removes a quarter of
almost nothing. `adamw.step` gains about 15 %, consistently and outside the
run-to-run spread of ±4 %.

That number says something about the cost model. Removing 303 launches at the
measured λ of 50 ns is 15 µs, one percent of the 1.5 ms — so the saving is
*not* launch overhead. It is traffic: each elementwise step over 200 000
parameters reads and writes 1.6 MB, and fusing twenty steps into six groups
removes tens of megabytes. That is the `Q` term of the roofline, and it is
the first measurement in this project where a compiler pass moved the clock
outside the noise.

The training losses are unchanged to the last digit. Fusion reassociates
nothing: the operations happen in the same order, on the same values, in the
same machine type — only the intermediates stay in registers. Every golden
output of the test suite is byte-identical with the pass on.

## Not in this version

- **Multi-output kernels.** A value read from inside *and* outside a group
  has to be materialized, which splits the group. Letting one kernel write
  several outputs would keep those chains together; it changes the kernel IR,
  its interpreter and the Cranelift backend, and it is the main reason the
  forward pass fuses as little as it does.
- **Reduction fusion.** A `mean` or `softmax` ends a group. Accumulating a
  reduction inside the producer's nest is a different and more powerful
  scheduler, and it is what would let a layer normalization become one
  kernel.
- **Extraction that knows about fusion.** With groups, `Q` is a property of
  the partition, not of a node, so choosing the cheapest term and choosing
  the partition are one problem (`optimization` chapter, question 49). The
  e-graph still extracts by the per-node roofline and the partition is
  computed afterwards, so the cost model under-counts what fusion will save.

```sh
presk dump corpus/nlp/05_gpt.psk --stage sched --const D=64 --const H=2 --const L=2
presk dump corpus/nlp/05_gpt.psk --stage kernel --entry adamw.step --const D=64 --const H=2 --const L=2 \
     --dim B=2 --dim T=32 --realize f32
```
