> The time the compiler's commands take per model, measured on 2026-10-02 by `make compile-times`
> (`experiments/compile_times.sh`) on presk `2e461fb` (`presk 0.10.0`),
> on aarch64, 20 cores. Wall seconds of one command each, start-up included: `presk check`;
> `presk calibrate --precision` (the magnitude envelope of the entry on 2 sample(s) at 128 bits);
> `presk solve --budget 1e-3` (a realization proposed and verified by measurement against the reference
> at 128 bits); and the exact reference alone, `presk run --bits 128` on one sample. The first column is
> the model's key, which the paper's `tables/setup.py` reads.

# Compile and solve times, 2026-10-02

| model | constants | presk check (s) | presk solve (s) | presk calibrate (s) | exact reference (s) |
|---|---|---|---|---|---|
| gpt-course | the course GPT, course constants (D = 384, H = 6, L = 6, T = 256) | 0.007 | 2702.445 | 1497.697 | 178.603 |
| gpt-reduced | the course GPT, reduced constants (D = 64, H = 2, L = 2, T = 256), trained weights | 0.006 | 60.450 | 28.773 | 3.782 |
| minigpt | the tiny MiniGPT (V = 64, D = 16, H = 2, L = 2, B = 2, T = 8) | 0.004 | 0.160 | 0.049 | 0.022 |
| bert-large-L2 | BERT-large at MLPerf's constants, L = 2 of 24, one feature, its SQuAD loss | 0.006 | 10463.836 | 5086.214 | 616.894 |

The outputs of every command are under `out/compile_times/` (not committed). A time is of the
command as the paper's reader would run it, the front end, the JIT and the reading of the data
included; `presk solve` includes the reference run it verifies against.

`gpt-course`, `gpt-reduced` and `minigpt` were run by `MODELS="gpt-course gpt-reduced minigpt" make
compile-times` on presk `2e461fb` from 17:33 to 18:47 CEST on 2026-10-02. `bert-large-L2` was run
apart on presk `e769985`, whose compiler is `2e461fb`'s plus `fbf4b76` (`presk solve` refuses a
non-scalar entry) and whose BERT has a scalar entry, by `MODELS=bert-large-L2 ./experiments/compile_times.sh` into a report of its
own, from 21:53 CEST on 2026-10-02 to 02:24 on 2026-10-03, and its row merged here by hand. Nothing
else ran on the machine (decision D101).

- **`bert-large-L2`'s entry is `squad_loss`**, BERT's SQuAD fine-tuning loss as Google's
  `run_squad.py` computes it: the mean of the start and end cross-entropies against an answer's span.
  It runs on `bert_features.py --span`'s features, which are `make bert`'s with the span added
  (`out/bert/features_loss.psk.d`). `presk solve` ranks the sites by the sensitivity `∂L/∂v` of a
  scalar `L`. `squad`, the entry measured first, returns the logits `ℝ[batch: B, seq: T, ends: 2]`.
  D101's first row was on `squad`, from 18:48 to 21:05 on 2026-10-02: `presk check` 0.005 s,
  `presk calibrate` 5 251.564 s, the exact reference 600.140 s, and `presk solve` refused after
  2 349 s (39 min), in the reverse sweep's first `.vjp`, once the sweep's forward half had run
  (`sum_to: cannot reduce [] to [batch: 2, seq: 384, 2]`). Since then `presk solve` refuses such an
  entry before computing anything. The row above is the four commands on `squad_loss`. Its solve put
  all 63 sites away from `f64` (59 in `f32` and 4 in `bf16`, 2.00× less traffic than all-`f64`)
  and measured 5.097·10⁻⁷ against the budget's 6.254·10⁻³, with no widening round. Sampled every
  minute, the commands' resident memory reached 6 GB (calibrate), 42 GB (solve) and 8 GB (the
  reference).
- **The cores.** Every command is one thread and the script does not pin it. Linux's scheduler,
  which knows the X925 cores' capacity (997) from the A725's (718), kept it on an X925 core. BERT's
  commands, sampled every minute, were on cores 15 (calibrate), 8 (solve) and 6 (the reference), all
  X925, and those of its first row on 5, 16 and 18. `gpt-reduced` pinned to one X925 core
  (`taskset -c 5`) took 0.004, 60.670, 29.230 and 3.846 s, the table's row within 2 %. Pinned to
  one A725 core (`taskset -c 0`) it took 0.006, 110.319, 54.934 and 7.029 s, 1.8× slower. The
  times above are therefore one X925 core's.
