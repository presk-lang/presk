> The time the compiler's commands take per model, measured on 2026-10-13 by `make compile-times`
> (`experiments/compile_times.sh`) on presk `7334e31` (`presk 0.12.0-dev`),
> on aarch64, 20 cores. Wall seconds of one command each, start-up included: `presk check`;
> `presk calibrate --precision` (the magnitude envelope of the entry on 2 sample(s) at 128 bits);
> `presk solve --budget 1e-3` (a realization proposed and verified by measurement against the reference
> at 128 bits); and the exact reference alone, `presk run --bits 128` on one sample. The first column is
> the model's key, which the paper's `tables/setup.py` reads.

# Compile and solve times, 2026-10-13

| model | constants | presk check (s) | presk solve (s) | presk calibrate (s) | exact reference (s) |
|---|---|---|---|---|---|
| bert-large-L2 | BERT-large at MLPerf's constants, L = 2 of 24, one feature, its SQuAD loss | 0.006 | 10112.615 | 4821.561 | 557.211 |

The outputs of every command are under `out/compile_times/` (not committed; archived in
`presk-project/data/paper-0.12.0.tar.xz`). A time is of the
command as the paper's reader would run it, the front end, the JIT and the reading of the data
included; `presk solve` includes the reference run it verifies against.

`bert-large-L2` was run by `MODELS=bert-large-L2 ./experiments/compile_times.sh` on presk
`7334e31`, version 0.12.0's code, from 00:17 to 04:35 CEST on 2026-10-14 (the date above is UTC's),
in plan 0.12.0's step 5 (decision D124), after `make precision`, the quant and step tables and
`make curve` and before the Llama 2 decoder's solve. Nothing else was measured on the machine
meanwhile; the session edited text.

- **The same commands as D101's and D117's rows** (`docs/compile-times-2026-10-02.md`,
  `docs/compile-times-2026-10-09.md`), on the same entry, `squad_loss`, and the same features. The
  envelope is D101's but for its date, and the exact reference prints the same loss,
  6.75736064365.
- **The solve differs, as 0.12.0's step 4 makes it** (D123: the solver offers the functions'
  results and the parameters). It put all 28 stored values away from `f64`, 24 in `f32` and 4 in
  `bf16`, and all 23 parameters, 13 in `f32` (the weights, the embeddings and the norms' γ) and 10
  in `bf16` (the biases and the norms' β): 3.6105·10⁸ bytes against 7.2219·10⁸ all-`f64`, the
  parameters' bytes at rest counted (2.2734·10⁸ against 4.5477·10⁸), **2.00×** less traffic. The
  estimate is 1.895·10⁻³ against the budget's 6.254·10⁻³, and the measured gap **9.208·10⁻⁷**,
  with no widening round. D117's solve offered 25 stored values and no parameter, at 2.03× without
  the weights, and measured 1.046·10⁻⁵. The three stored values added are results, all in `f32`:
  the layer norm's output `LayerNorm.apply/%15`, the linear layer's `Linear.apply/%5` and the loss
  `squad_loss/%9`. Of the 25 others, 23 have D117's types, and two are back at D101's:
  `LayerNorm.apply/%2`, the mean μ, in `bf16`, and `SelfAttention.apply/%5`, the query after
  `split_heads`, in `f32`. The measured gap is about a tenth of D117's, about 7 000 times inside the
  budget.
- **Times and memory.** The solve took 10 112.6 s (D117: 10 447.6), `calibrate` 4 821.6 s
  (5 229.0) and the reference 557.2 s (580.7). By `/usr/bin/time`, the peak resident memory was
  5.0 GB (`calibrate`), 11.1 GB (`solve`) and 4.2 GB (the reference), D117's.
- **The cores.** Every command is one thread, unpinned, and was not sampled. The times are within
  8 % of D117's.
