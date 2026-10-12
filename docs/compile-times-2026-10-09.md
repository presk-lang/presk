> The time the compiler's commands take per model, measured on 2026-10-09 by `make compile-times`
> (`experiments/compile_times.sh`) on presk `bb78cb8` (`presk 0.11.0`),
> on aarch64, 20 cores. Wall seconds of one command each, start-up included: `presk check`;
> `presk calibrate --precision` (the magnitude envelope of the entry on 2 sample(s) at 128 bits);
> `presk solve --budget 1e-3` (a realization proposed and verified by measurement against the reference
> at 128 bits); and the exact reference alone, `presk run --bits 128` on one sample. The first column is
> the model's key, which the paper's `tables/setup.py` reads.

# Compile and solve times, 2026-10-09

| model | constants | presk check (s) | presk solve (s) | presk calibrate (s) | exact reference (s) |
|---|---|---|---|---|---|
| bert-large-L2 | BERT-large at MLPerf's constants, L = 2 of 24, one feature, its SQuAD loss | 0.006 | 10447.634 | 5229.045 | 580.703 |

The outputs of every command are under `out/compile_times/` (not committed; archived in
`presk-project/data/paper-0.11.0.tar.xz`). A time is of the
command as the paper's reader would run it, the front end, the JIT and the reading of the data
included; `presk solve` includes the reference run it verifies against.

`bert-large-L2` was run by `MODELS=bert-large-L2 ./experiments/compile_times.sh` on presk
`bb78cb8`, version 0.11.0's tree, from 17:47 to 22:18 CEST on 2026-10-09, in the paper's runs of
plan 0.11.0's step 7 (decision D117), after `make precision`, the quant and step tables and
`make curve` and before the Llama 2 decoder's solve. Nothing else was measured on the machine
meanwhile; the session edited text, and ran `gas-agents` once for a few seconds at `nice 19` on
an A725 core (`taskset -c 0`).

- **The same commands as D101's row** (`docs/compile-times-2026-10-02.md`), on the same entry,
  `squad_loss`, and the same features. The envelope is D101's but for its date, and the exact
  reference prints the same loss, 6.75736064365.
- **The solve differs, as 0.11.0's step 2b makes it** (D112: the solver offers the values the
  program stores). It put all 25 stored values away from `f64`, 21 in `f32` and 4 in `bf16`:
  1.1641·10⁸ bytes against 2.3596·10⁸ all-`f64`, **2.03×** less traffic. The estimate is
  6.094·10⁻³ against the budget's 6.254·10⁻³, and the measured gap **1.046·10⁻⁵**, with no
  widening round. D101's solve listed 63 sites, 59 in `f32` and 4 in `bf16`, at 2.00×, and
  measured 5.097·10⁻⁷. Of the 25 stored values, 23 have D101's types; two moved:
  `LayerNorm.apply/%2`, the mean μ (768 elements), from `bf16` to `f32`, and
  `SelfAttention.apply/%5`, the query after `split_heads` (786 432 elements), from `f32` to
  `bf16`. The 38 sites no longer offered change no bit when typed; over the 25 the greedy order is
  another, its proposal takes the query to `bf16` and keeps μ in `f32`, and the gap it measures is
  twenty times D101's and 600 times inside the budget.
- **Times and memory.** The solve took 10 447.6 s (D101: 10 463.8), `calibrate` 5 229.0 s
  (5 086.2) and the reference 580.7 s (616.9). By `/usr/bin/time`, the peak resident memory was
  5.0 GB (`calibrate`), 11.1 GB (`solve`) and 4.2 GB (the reference); D101 sampled 6, 42 and 8 GB
  every minute, before 0.10.1 made the exact interpreter hold each value once (D105).
- **The cores.** Every command is one thread, unpinned. Sampled once, `calibrate` was on core 15,
  an X925; the other commands were not sampled. Their times are within 6 % of D101's, which were
  one X925 core's.
