# Presk compiler — thin orchestration over cargo (cargo does the work; never the reverse).
#   make build            debug build of the workspace
#   make test             all tests (unit, snapshot, property)
#   make check-examples   `presk check` on every example program
#   make run-examples     the reference `presk run` commands of the plan (§4)
#   make export-examples  export the example files to out/ (with conformance data)
#   make venv             create .venv with PyTorch, ONNX Runtime (uv) for the conformance tests
#   make conformance      export-examples, then run the generated PyTorch tests in .venv
#   make tutorial         every command of the manual's tutorial chapter, steps 0 to 18 (tutorial/*.psk)
#   make train-examples   presk train on the regression and the tiny MiniGPT, on the reference and natively
#   make serve-examples   presk serve on the tiny MiniGPT's checkpoint: greedy tokens identical on every backend; export --serve
#   make serve-pt          ...and the exported serve.py gives the same tokens (needs the venv)
#   make opt-examples     the e-graph optimizer: its report on the examples, and the same outputs with and without --opt
#   make calibrate        presk bench --calibrate: time the kernels of the tiny MiniGPT natively, fit the roofline profile (out/profile.json)
#   make corpus           the NLP-course corpus (corpus/nlp): the ladder checked, run and trained on the Code civil
#   make corpus-gpt       tokenize the Code civil and train the course GPT (reduced constants: the full size is the wall 0.4.0 has to break)
#   make gpt-trained      the corpus GPT trained 600 steps at D = 64: the weights of the precision targets
#   make precision        envelope, bound, sensitivity and solve on the corpus GPT (0.5.0, 0.6.0 step 6)
#   make curve            the epsilon-to-quality curve: seven budgets solved, each trained from scratch (about half an hour since 0.10.0)
#   make quant-table      int8/int4 gaps split into representation and arithmetic, statement K (0.6.0 step 5d)
#   make step-table       one training step under each realization against the exact step (0.6.0 step 7b)
#   make j-step           statement J′: the solved step at course scale, time and update gap
#   make step-train       600 steps from scratch under the solved step, against the seeds
#   make delta-p          δ_p, the reference's own gap: the interpreter at 128, 256 and 512 bits against itself at 1024 on the course GPT's loss, logits and gradient (W16, D98; about half an hour)
#   make compile-times    the time presk check, calibrate, solve and the exact reference take per model, into docs/compile-times-<date>.md (the paper's setup table; a quiet machine)
#   make foil             MLPerf's accuracy rule and a budget on one model (0.6.0 step 8)
#   make margins          ε from the exact model's margins, every realization's flips (D118)
#   make bert             BERT-large inference at MLPerf's constants (about five hours; L=2 for minutes)
#   make llama            the Llama 2 decoder at the 7B width, one block: exact logits, gaps, split, time (D104, D106; about thirteen hours, up to 64 GB, alone)
#   make experiment       exact versus almost: the realization gap of float32/float64 training (manual, `experiments`)
#   make experiment-gpt   the course GPT trained by presk and by PyTorch on the same data (manual, `experiments` §3)
#   make coverage         presk check on the MLPerf transcriptions, BERT and Llama run at tiny constants (manual, `coverage`)
#   make plan-check       a default plan lowers every example entry to the same kernels (0.7.0 step 1)
#   make c-conformance    the C export: every entry of the plan-check set and the course GPT exported, built, run, bit for bit, under the default layout and under plans, on GCC, Clang and for x86-64; presk serve's tokens from the exported generation loop; integer weights on the trained checkpoints; every nest in the SCoP mode against Polly (0.9.0 steps 3–8)
#   make c-speed          statement C4: the exported course GPT's forward at B = 16 against the native backend's, one thread and every thread (0.9.0 step 5; a quiet machine)
#   make pytorch-baseline the course GPT's forward pass and training step on three sides, native, C and PyTorch, into out/pytorch-baseline/report.md (D87; a quiet machine, about forty minutes)
#   make widening         where the measurement changes the solver's decision: three pairs of small programs (an unshifted softmax, the two-pass variance, a sum of equal terms, each beside the form the estimate judges right), the four rules' bounds and every solve's widening rounds at seven budgets, into out/widening/summary.md (D128; minutes, nothing timed)
#   make pytorch-gap      PyTorch's own gap to the exact model and its flips: the course GPT at D = 64 exported and run by PyTorch eager in float32 and float64 on the 461 held-out windows, against make margins' exact logits, beside the compiler's realizations, into out/pytorch-gap/margins.md (D130; needs make margins; seconds, nothing timed)
#   make gappa            one nest's bound proved in Gappa beside the compiler's bound and the measured gap: the layer norm in f32 at D = 8…64 on eight rows, into out/gappa/summary.md (D108; needs `gappa`, apt's; minutes, nothing timed)
#   make gappa-dot        a contraction's bound proved in Gappa beside the compiler's bound and the measured gap: `dot` of [8, K]·[K, 8] in f32 at K = 8…128 under the sequential order and interleaved(4, 8, 16), into out/gappa-dot/summary.md (D131; needs `gappa`; minutes, nothing timed)
#   make pytorch-decoder  the Llama 2 decoder block at the 7B width, its forward pass natively and in PyTorch eager, ten X925 cores pinned, into out/pytorch-decoder/report.md (D107; alone, about three and a half hours)
#   make host-profile     the native host's time by phase on the course GPT's forward pass, gradient and training step, into out/host-profile/report.md (0.10.0 step 5, D95; a quiet machine)
#   make zig              zig from PyPI in .venv-zig: the Clang and the x86-64 cross-compiler of statement C3 (0.9.0 step 5)
#   make llvm             LLVM's release for Linux on aarch64 in .llvm: the clang with Polly the SCoP mode's nests are checked with (0.9.0 step 8)
#   make calibrate-ra4    the layout cost model's calibration on a GEMM and two thin products, pinned to the RA4 cores (0.7.0 step 4, 0.8.0 steps 3–4)
#   make ra4-small        the RA4 sweep's daily subset on the tutorial's attention, two days back to back, ranked (0.7.0 step 5)
#   make ra4              the RA4 sweep, day one: the course GPT's attention, the GEMM, a BERT-shaped sample (about two hours; 0.7.0 step 6)
#   make ra4-check        statement L apart: every point of day one on the kernel interpreter, sharded over every core (hours)
#   make ra4-20           the secondary sweep on all twenty cores, with its own calibration (out/profile-20.json, out/ra4-20.md)
#   make ra4-day2         day one's sweep again, ranked against it with the check into out/ra4.md
#   make ra4-dev          0.8.0's development set: 0.7.0's days re-priced under a profile and ranked, nothing measured (0.8.0 step 1)
#   make ra4-dev-frozen   the development set under the frozen model's profile, its report checked byte for byte (0.8.0 step 4)
#   make ra4-ablation     the development set priced under every form of the model, fitted on the definitive calibration (0.8.0 step 4)
#   make calibrate-ra4-small  calibrate-ra4 on the ten A725 cores, into out/profile-a725.json (0.8.0 step 5)
#   make ra4-f2-dry       F2's dry run: one nest's sweep timed on the A725 cores, a day sized from it (0.8.0 step 5)
#   make ra4-f2           F2, day one: 0.7.0's shapes on the A725 cores, from a committed tree descending from the freeze's tag
#   make ra4-f2-check     statement L on F2 apart, on the kernel interpreter, sharded over every core (hours)
#   make ra4-f2-day2      F2, day two, on another calendar day (RA4_SAME_DATE=1: on day one's), then ra4-f2-rank: presk rank --frozen into out/ra4-f2.md
#   make ra4-f1-dry       F1's dry run: each configuration's longest nest timed on the X925 cores, a day sized from them (0.8.0 step 6)
#   make ra4-f1           F1, day one: new shapes (F1a, F1b) on the X925 cores, priced by the frozen profile, from a committed tree descending from the tag
#   make ra4-f1-check     statement L on F1 apart, on the kernel interpreter, sharded over every core (hours)
#   make ra4-f1-day2      F1, day two (RA4_SAME_DATE=1: on day one's date), then ra4-f1-rank: presk rank --frozen twice, into out/ra4-f1a.md and out/ra4-f1b.md
#   make ra4-t3-dry       T3's dry run: the derivatives of `dot` in the course GPT's gradient, the nest the model predicts longest swept on the X925 cores, a day sized from it (0.10.0 step 2)
#   make ra4-t3           T3, day one: the four derivative nests swept whole, priced by the frozen profile, from a committed tree descending from the tag
#   make ra4-t3-check     statement L on T3 apart, on the kernel interpreter, sharded over every core (hours)
#   make ra4-t3-day2      T3, day two (RA4_SAME_DATE=1: on day one's date), then ra4-t3-rank: presk rank --frozen into out/ra4-t3.md, O and Q on the derivative nests
#   make realize          the fusion groups of the corpus GPT and the launches they remove (was `make sched`)
#   make bench            presk bench on the tiny MiniGPT (forward, loss, optimizer step)
#   make sync-examples    re-copy tests/examples from presk-specs and show the diff
#   make snap             review pending insta snapshots
#   make doc              build the user manual (crates/presk-manual) and open it
#   make explain          presk explain on the tutorial's loss (a budget of 1e-3, the frozen profile, every target) into out/explain/loss (0.11.0)
#   make explain-pdf      the PDF of a presk explain document, EXPLAIN=DIR (out/explain/loss by default): pandoc and lualatex (TeX Live 2026), DejaVu fonts
#   make explain-manual   the tutorial's document regenerated into the manual's chapter explain_loss, which a test holds current
#   make math-check       the printer of the denotation's standalone LaTeX of the tutorial, the examples and the course GPT, each compiled by LuaLaTeX into out/math/ (0.13.0 step 4)
#   make network-figure   the course GPT's network diagram: level 2 into the manual's chapter explain_network (SVG); level 1 and Block alone into out/network/ (SVG, TikZ, standalone documents) and the report's two figures, $(PRESK_SPECS_DIR)/report/figures/network-gpt{,-block}.tex (0.14.0 step 3)
#   make clean

PRESK_SPECS_DIR ?= ../presk-specs
CARGO ?= cargo
PRESK = $(CARGO) run -q -p presk --

# The cores the RA4 sweep of 0.7.0 is pinned to (`taskset -c`): falicon's ten
# Cortex-X925, CPU part 0xd85 in /proc/cpuinfo; 0-4 and 10-14 are the
# Cortex-A725 (D56). Override on another machine.
RA4_CORES ?= 5-9,15-19
# The ten Cortex-A725 (CPU part 0xd87), the cores of 0.8.0's fresh sweep F2,
# which gets its own calibration (plan 0.8.0 §3.5, D63).
RA4_CORES_SMALL ?= 0-4,10-14

# What a gradient does with its forward pass's values in the training runs of
# the experiments (`--remat`, 0.10.0 step 4): `store`, the default, or
# `recompute`, the policy of the runs D48–D55 recorded (D93).
REMAT ?= store

EXAMPLES = tests/examples/mlp/mlp.psk \
           tests/examples/layernorm/layernorm.psk \
           tests/examples/minigpt/minigpt.psk \
           tests/examples/minigpt/train_minigpt.psk \
           tests/examples/minigpt/serve_minigpt.psk

.PHONY: network-figure math-check margins widening pytorch-gap gappa-dot explain explain-pdf explain-manual gappa delta-p compile-times c-conformance c-speed pytorch-baseline pytorch-decoder host-profile zig llvm calibrate-ra4 ra4-small ra4 ra4-day2 ra4-check ra4-20 ra4-dev ra4-dev-frozen ra4-ablation calibrate-ra4-small ra4-f2-dry ra4-f2 ra4-f2-day2 ra4-f2-rank ra4-f2-check ra4-f1-dry ra4-f1 ra4-f1-day2 ra4-f1-rank ra4-f1-check ra4-t3-dry ra4-t3 ra4-t3-day2 ra4-t3-rank ra4-t3-check plan-check foil bert llama step-table j-step step-train curve quant-table gpt-trained precision build test check-examples run-examples export-examples venv conformance tutorial train-examples experiment experiment-gpt coverage bench sched realize opt-examples calibrate corpus corpus-gpt sync-examples snap doc clean

build:
	$(CARGO) build --workspace

test:
	$(CARGO) test --workspace

check-examples: build
	@for f in $(EXAMPLES); do echo "== presk check $$f"; $(PRESK) check $$f || exit 1; done

run-examples: build
	$(PRESK) run tests/examples/mlp/mlp.psk --entry mse_grad --dim D=4 --dim B=2
	$(PRESK) run tests/examples/layernorm/layernorm.psk --entry 'LayerNorm{8}.apply' --dim batch=3
	$(PRESK) run tests/examples/minigpt/minigpt.psk --entry MiniGPT.apply \
	    --const V=64 --const D=16 --const H=2 --const L=2 --dim B=2 --dim T=8

export-examples: build
	$(PRESK) export tests/examples/mlp/mlp.psk --target pytorch --mode readable --conformance-entry mse_grad --dim D=4 --dim B=2 -o out/mlp_pt
	$(PRESK) export tests/examples/layernorm/layernorm.psk --target pytorch --mode readable --conformance-entry 'LayerNorm{8}.apply' --dim batch=3 -o out/layernorm_pt
	$(PRESK) export tests/examples/minigpt/minigpt.psk --target pytorch --mode readable --shapes=check \
	    --conformance-entry MiniGPT.apply --const V=64 --const D=16 --const H=2 --const L=2 --dim B=2 --dim T=8 -o out/minigpt_pt
	$(PRESK) export tests/examples/minigpt/train_minigpt.psk --target pytorch --mode readable \
	    --conformance-entry loss --const V=64 --const D=16 --const H=2 --const L=2 --dim B=2 --dim T=8 -o out/train_pt
	$(PRESK) export tests/examples/minigpt/train_minigpt.psk --target pytorch --mode readable \
	    --conformance-entry adamw.step --const V=64 --const D=16 --const H=2 --const L=2 --dim B=2 --dim T=8 -o out/adamw_pt
	$(PRESK) export tests/examples/minigpt/train_minigpt.psk --target pytorch --mode readable \
	    --conformance-entry train:20 --const V=64 --const D=16 --const H=2 --const L=2 -o out/trainloop_pt
	$(PRESK) export tests/examples/minigpt/minigpt.psk --target pytorch --mode graph \
	    --conformance-entry MiniGPT.apply --const V=64 --const D=16 --const H=2 --const L=2 --dim B=2 --dim T=8 -o out/graph_pt

venv:
	uv venv .venv --python 3.12
	uv pip install --python .venv/bin/python torch numpy pytest onnx onnxscript onnxruntime

conformance: export-examples
	@for d in out/*/; do \
	    [ -f "$$d/test_conformance.py" ] || continue; \
	    echo "== $$d"; (cd $$d && ../../.venv/bin/python -m pytest -q test_conformance.py) || exit 1; \
	done

# The commands of the tutorial chapter, steps 0 to 18, in order
# (crates/presk-manual/manual/tutorial.md): every output the chapter quotes is
# one of these commands'. The PyTorch conformance tests need `make venv`, and
# step 18 a C compiler (with OpenMP for the planned program's threads).
tutorial: build
	$(PRESK) --help
	$(PRESK) check tests/examples/mlp/mlp.psk
	$(PRESK) check tutorial/01_linear.psk
	! $(PRESK) run tutorial/01_linear.psk
	$(PRESK) run tutorial/01_linear.psk --entry 'Linear{3, 2}.apply' --dim batch=4 --verbose
	$(PRESK) run tutorial/01_linear.psk --entry 'Linear{3, 2}.apply' --dim batch=4 --seed 1 --bits 64 --digits 6
	$(PRESK) check tutorial/02_norm.psk --summary
	$(PRESK) run tutorial/02_norm.psk --entry 'RMSNorm{4}.apply' --dim batch=2 --digits 6
	$(PRESK) check tutorial/03_attention.psk --summary
	! $(PRESK) check tutorial/03_attention.psk --const H=3
	$(PRESK) run tutorial/03_attention.psk --entry attend --dim B=1 --dim T=3 --digits 5
	! $(PRESK) run tutorial/03_attention.psk --entry attend --dim B=1 --dim T=2000
	$(PRESK) check tutorial/04_grad.psk --summary
	$(PRESK) run tutorial/04_grad.psk --entry loss --dim D=3 --dim B=2 --digits 6
	$(PRESK) run tutorial/04_grad.psk --entry loss_grad --dim D=3 --dim B=2 --digits 6
	$(PRESK) run tutorial/05_sample.psk --entry pick --dim B=5
	$(PRESK) run tutorial/05_sample.psk --entry pick --dim B=5 --seed 3
	$(PRESK) check tutorial/06_tower.psk --summary
	$(PRESK) dump tutorial/06_tower.psk --stage elab
	$(PRESK) run tutorial/06_tower.psk --entry Tower.apply --dim batch=2 --digits 6
	$(PRESK) run tutorial/06_tower.psk --entry Tower.apply --dim batch=2 --digits 6 --const L=1
	$(PRESK) dump tutorial/04_grad.psk --stage core | sed -n '/^fn loss{/,/^}/p'
	$(PRESK) dump tutorial/04_grad.psk --stage core-after-ad | grep '^fn'
	$(PRESK) explain tutorial/04_grad.psk --entry loss_grad --dim D=3 --dim B=2 --out out/tutorial_explain
	grep '^## ' out/tutorial_explain/explain.md
	$(PRESK) export tutorial/06_tower.psk --target pytorch --mode readable --shapes=check \
	    --conformance-entry Tower.apply --dim batch=2 -o out/tutorial_tower
	$(PRESK) export tutorial/04_grad.psk --target pytorch --mode readable \
	    --conformance-entry loss_grad --dim D=3 --dim B=2 -o out/tutorial_grad
	@for d in out/tutorial_tower out/tutorial_grad; do echo "== $$d"; (cd $$d && ../../.venv/bin/python -m pytest -q test_conformance.py) || exit 1; done
	$(PRESK) check tests/examples/minigpt/minigpt.psk --summary
	$(PRESK) run tests/examples/minigpt/minigpt.psk --entry MiniGPT.apply \
	     --const V=64 --const D=16 --const H=2 --const L=2 --dim B=2 --dim T=8
	$(PRESK) run tests/examples/minigpt/train_minigpt.psk --entry loss \
	     --const V=64 --const D=16 --const H=2 --const L=2 --dim B=2 --dim T=8
	! $(PRESK) run tests/examples/minigpt/train_minigpt.psk
	$(PRESK) data pack --type '(ℝ[model: 4], ℝ[model: 4])' tutorial/data/pairs.txt -o tutorial/data/pairs.psk.d
	$(PRESK) check tutorial/07_train.psk --summary
	$(PRESK) train tutorial/07_train.psk --steps 100 --log-every 20 --ckpt out/tutorial_ckpt
	$(PRESK) export tutorial/07_train.psk --target pytorch --mode readable --conformance-entry train:10 -o out/tutorial_train
	(cd out/tutorial_train && ../../.venv/bin/python -m pytest -q -s test_conformance.py)
	(cd out/tutorial_train && ../../.venv/bin/python train.py --steps 100 --log-every 25 --ckpt ckpt)
	$(PRESK) bench tutorial/07_train.psk --entry loss --dim B=8
	$(PRESK) train tutorial/07_train.psk --steps 100 --log-every 20 --backend native --ckpt out/tutorial_ckpt_native
	$(PRESK) train tutorial/07_train.psk --steps 100 --log-every 100 --backend native --realize f32 --ckpt out/tutorial_ckpt_native32
	$(PRESK) dump tutorial/04_grad.psk --stage core-opt >/dev/null
	$(PRESK) run tutorial/04_grad.psk --entry loss_grad --dim D=3 --dim B=2 --digits 6 --opt
	rm -f out/tutorial_profile.json
	$(PRESK) bench tutorial/07_train.psk --entry loss --dim B=8 --repeat 7 --calibrate out/tutorial_profile.json
	$(PRESK) bench tutorial/07_train.psk --entry loss --dim B=8 --opt --profile out/tutorial_profile.json
	python3 corpus/nlp/data/prepare.py
	$(PRESK) data chars corpus/nlp/data/civil.md -o corpus/nlp/data/civil_vocab.json
	$(PRESK) data encode corpus/nlp/data/civil.md --vocab corpus/nlp/data/civil_vocab.json -o corpus/nlp/data/civil_tokens.txt
	$(PRESK) data lm-windows --vocab 92 --seq 256 corpus/nlp/data/civil_tokens.txt -o corpus/nlp/data/civil_T256.psk.d
	$(PRESK) data pack --type '(idx[41], idx[41])' corpus/nlp/data/bigrams.txt -o corpus/nlp/data/bigrams.psk.d
	$(PRESK) check corpus/nlp/01_bigram_count.psk --summary
	$(PRESK) run corpus/nlp/01_bigram_count.psk --entry loss --data corpus/nlp/data/bigrams.psk.d --backend native --digits 10
	$(PRESK) check corpus/nlp/05_gpt.psk --summary
	$(PRESK) calibrate --precision tutorial/07_train.psk --entry loss --dim B=4 \
	    --bits 128 --out out/tut_env.json
	$(PRESK) check --precision out/tut_env.json tutorial/07_train.psk --realize f32
	$(PRESK) solve tutorial/07_train.psk --precision out/tut_env.json \
	    --budget 1e-3 --out out/tut_solved.json --sites 0
	$(PRESK) run tutorial/07_train.psk --entry loss --dim B=4 \
	    --backend native --realize @out/tut_solved.json --digits 10
	$(PRESK) calibrate --step tutorial/07_train.psk --weights out/tutorial_ckpt/final.psk.d \
	    --warmup 20 --samples 8 --realize f64 --realize f32 --realize bf16 --realize recipe \
	    --out out/tut_step.json
	$(PRESK) solve --step tutorial/07_train.psk --precision out/tut_step.json \
	    --budget 1e-3 --out out/tut_step_solved.json
	$(PRESK) train tutorial/07_train.psk --steps 100 --log-every 20 --backend native \
	    --realize @out/tut_step_solved.json --ckpt out/tutorial_ckpt_step
	$(PRESK) data diff out/tutorial_ckpt_native/final.psk.d out/tutorial_ckpt_step/final.psk.d
	$(PRESK) train tutorial/07_train.psk --steps 100 --log-every 100 --backend native \
	    --seed 1 --ckpt out/tutorial_ckpt_seed1
	$(PRESK) data diff out/tutorial_ckpt_native/final.psk.d out/tutorial_ckpt_seed1/final.psk.d
	rm -rf out/tut_layout.json out/tut_tune
	$(PRESK) bench experiments/gemm.psk --entry gemm --dim N=256 --realize f32 \
	    --calibrate out/tut_layout.json --space calib --repeat 5
	$(PRESK) bench experiments/gemm.psk --entry gemm --dim N=512 --realize f32 \
	    --calibrate out/tut_layout.json --space calib --repeat 5
	$(PRESK) plan tutorial/03_attention.psk --entry attend --dim B=16 --dim T=256 --realize f32 \
	    --profile out/tut_layout.json --out out/tut_plan.json
	$(PRESK) run tutorial/03_attention.psk --entry attend --dim B=16 --dim T=256 --backend native \
	    --realize f32 > out/tut_attend.txt
	$(PRESK) run tutorial/03_attention.psk --entry attend --dim B=16 --dim T=256 --backend native \
	    --plan @out/tut_plan.json > out/tut_attend_plan.txt
	cmp out/tut_attend.txt out/tut_attend_plan.txt && echo "same output"
	$(PRESK) autotune tutorial/03_attention.psk --entry attend --dim B=16 --dim T=256 --realize f32 \
	    --space small --profile out/tut_layout.json --interp --out out/tut_tune/day1
	$(PRESK) autotune tutorial/03_attention.psk --entry attend --dim B=16 --dim T=256 --realize f32 \
	    --space small --profile out/tut_layout.json --out out/tut_tune/day2
	$(PRESK) rank out/tut_tune/day1 out/tut_tune/day2 --entry attend --out out/tut_rank.md \
	    | sed -n '/^## Statements/,/^| tile/p'
	rm -rf out/tut_c out/tut_c16 out/tut_scop
	$(PRESK) export tutorial/03_attention.psk --target c --entry attend --dim B=1 --dim T=3 -o out/tut_c
	$(MAKE) -C out/tut_c check PRESK=$(CURDIR)/target/debug/presk
	out/tut_c/model out/tut_c/conformance/inputs.psk.d out/tut_c/out.psk.d
	$(PRESK) data info out/tut_c/out.psk.d
	$(PRESK) export tutorial/03_attention.psk --target c --entry attend --dim B=16 --dim T=256 \
	    --plan @out/tut_plan.json -o out/tut_c16
	$(MAKE) -s -C out/tut_c16 check
	OMP_NUM_THREADS=1 $(MAKE) -s -C out/tut_c16 check
	out/tut_c16/model -r 11 out/tut_c16/conformance/inputs.psk.d out/tut_c16/out.psk.d
	OMP_NUM_THREADS=1 out/tut_c16/model -r 11 out/tut_c16/conformance/inputs.psk.d out/tut_c16/out.psk.d
	$(PRESK) export tutorial/03_attention.psk --target c --mode scop --entry attend --dim B=16 --dim T=256 \
	    --realize f32 -o out/tut_scop
	$(MAKE) -s -C out/tut_scop check

train-examples: build
	$(PRESK) train tests/train/mlp_regression.psk --steps 200 --log-every 50 --ckpt out/reg_ckpt
	$(PRESK) train tests/examples/minigpt/train_minigpt.psk --const V=64 --const D=16 --const H=2 --const L=2 \
	    --steps 20 --bits 64 --log-every 5 --ckpt out/ckpt
	$(PRESK) data info out/ckpt/final.psk.d
	$(PRESK) train tests/examples/minigpt/train_minigpt.psk --const V=64 --const D=16 --const H=2 --const L=2 \
	    --steps 20 --log-every 5 --backend native --ckpt out/ckpt_native
	$(PRESK) train tests/examples/minigpt/train_minigpt.psk --const V=64 --const D=16 --const H=2 --const L=2 \
	    --steps 20 --log-every 20 --backend native --realize f32 --ckpt out/ckpt_native32
	$(PRESK) train tests/examples/minigpt/train_minigpt.psk --const V=64 --const D=64 --const H=4 --const L=4 \
	    --steps 100 --log-every 25 --backend native --ckpt out/ckpt_large

# The MLPerf transcriptions of the manual's `coverage` chapter: what the shape
# fragment can say. Both type-check; since 0.6.0 step 8 BERT-large also runs,
# and since D104 the Llama decoder, here at tiny constants on the reference and
# natively (`make bert` runs BERT at MLPerf's own, `make llama` the decoder at
# Llama 2 7B's width).
BERT_TINY = --const V=100 --const D=32 --const H=2 --const L=2 --dim B=2 --dim T=16
LLAMA_TINY = --const V=100 --const D=32 --const H=2 --const G=2 --const FF=64 --const L=2 --dim B=2 --dim T=16
coverage: build
	$(PRESK) check coverage/bert.psk --summary | head -12
	$(PRESK) check coverage/llama.psk --summary | head -12
	$(PRESK) run coverage/bert.psk --entry BertSquad.apply $(BERT_TINY) --bits 128 --digits 6 | cut -c1-120
	$(PRESK) run coverage/bert.psk --entry BertSquad.apply $(BERT_TINY) --backend native --realize f32 --digits 6 | cut -c1-120
	$(PRESK) run coverage/llama.psk --entry Llama.apply $(LLAMA_TINY) --bits 128 --digits 6 | cut -c1-120
	$(PRESK) run coverage/llama.psk --entry Llama.apply $(LLAMA_TINY) --backend native --realize f32 --digits 6 | cut -c1-120
	$(PRESK) run coverage/llama.psk --entry loss $(LLAMA_TINY) --bits 128 --digits 17
	$(PRESK) run coverage/llama.psk --entry loss $(LLAMA_TINY) --backend native --realize f64 --digits 17

# The three experiments of the manual's `experiments` chapter (needs `make venv`).
# The third trains the course GPT against PyTorch on the same model and data:
# the same steps, batches (the program's `training run`) and threads on both
# sides, `f32` on both, dropout dropping on both with different masks (D54);
# each side prints its median step time, the first step excluded. The presk
# side runs the release binary, as the timed experiments do.
GPT_STEPS ?= 6
GPT_THREADS ?= $(shell nproc)
experiment-gpt: build
	$(CARGO) build --release -p presk
	python3 corpus/nlp/data/prepare.py
	$(PRESK) data chars corpus/nlp/data/civil.md -o corpus/nlp/data/civil_vocab.json
	$(PRESK) data encode corpus/nlp/data/civil.md --vocab corpus/nlp/data/civil_vocab.json -o corpus/nlp/data/civil_tokens.txt
	$(PRESK) data lm-windows --vocab 92 --seq 256 corpus/nlp/data/civil_tokens.txt -o corpus/nlp/data/civil_T256.psk.d
	$(PRESK) export corpus/nlp/05_gpt.psk --target pytorch --mode readable --train -o out/gpt_pt
	(cd out/gpt_pt && ../../.venv/bin/python train.py --steps $(GPT_STEPS) --log-every 1 --threads $(GPT_THREADS))
	./target/release/presk train corpus/nlp/05_gpt.psk --steps $(GPT_STEPS) --log-every 1 --backend native --realize f32 --threads $(GPT_THREADS) --remat $(REMAT) --ckpt out/gpt_course

experiment: build
	$(PRESK) export tutorial/07_train.psk --target pytorch --mode readable --conformance-entry train:10 -o out/tutorial_train
	$(PRESK) export tests/examples/minigpt/train_minigpt.psk --target pytorch --mode readable \
	    --conformance-entry train:20 --const V=64 --const D=16 --const H=2 --const L=2 -o out/trainloop_pt
	.venv/bin/python experiments/exact_vs_almost.py out/tutorial_train
	.venv/bin/python experiments/exact_vs_almost.py out/trainloop_pt

# The generation loop: the tiny MiniGPT's checkpoint of `make train-examples`,
# greedy tokens identical on the reference and natively (both realizations).
TINY = --const V=64 --const D=16 --const H=2 --const L=2
SERVE = $(PRESK) serve tests/examples/minigpt/serve_minigpt.psk $(TINY) --weights out/ckpt/final.psk.d \
	    --prompt "1 2 3 4 5 6 7 8" --tokens 8 --temperature 0
serve-examples: build
	$(SERVE) --bits 64 > out/serve_interp.txt
	$(SERVE) --backend native > out/serve_native.txt
	$(SERVE) --backend native --realize f32 > out/serve_native32.txt
	cmp out/serve_interp.txt out/serve_native.txt && cmp out/serve_interp.txt out/serve_native32.txt
	cat out/serve_interp.txt
	$(PRESK) serve tests/examples/minigpt/serve_minigpt.psk $(TINY) --weights out/ckpt/final.psk.d \
	    --prompt "1 2 3 4 5 6 7 8" --tokens 8 --backend native
	$(PRESK) export tests/examples/minigpt/serve_minigpt.psk $(TINY) --target pytorch --mode readable --serve -o out/serve_pt

# The exported serve.py against presk's native greedy tokens (needs `make venv`).
serve-pt: serve-examples
	.venv/bin/python out/serve_pt/serve.py --weights out/ckpt/final.psk.d --prompt "1 2 3 4 5 6 7 8" --tokens 8 --temperature 0 --dtype float64 > out/serve_py.txt
	cmp out/serve_native.txt out/serve_py.txt && cat out/serve_py.txt

# The e-graph optimizer (crate presk-opt, manual chapter `optimization`): the
# report on the examples, then the same outputs with and without --opt on
# the reference and natively (forward, gradient, twenty training steps).
opt-examples: build
	$(PRESK) dump tests/examples/mlp/mlp.psk --stage core-opt 2>&1 >/dev/null | grep -A 20 '^presk-opt'
	$(PRESK) dump tests/examples/minigpt/train_minigpt.psk --stage core-opt $(TINY) 2>&1 >/dev/null | grep -A 30 '^presk-opt'
	$(PRESK) run tests/examples/mlp/mlp.psk --entry mse_grad --dim D=4 --dim B=2 2>/dev/null > out/opt_a.txt
	$(PRESK) run tests/examples/mlp/mlp.psk --entry mse_grad --dim D=4 --dim B=2 --opt 2>/dev/null > out/opt_b.txt
	cmp out/opt_a.txt out/opt_b.txt
	$(PRESK) run tests/examples/minigpt/minigpt.psk --entry MiniGPT.apply $(TINY) --dim B=2 --dim T=8 --backend native 2>/dev/null > out/opt_a.txt
	$(PRESK) run tests/examples/minigpt/minigpt.psk --entry MiniGPT.apply $(TINY) --dim B=2 --dim T=8 --backend native --opt 2>/dev/null > out/opt_b.txt
	cmp out/opt_a.txt out/opt_b.txt
	$(PRESK) run tests/examples/mlp/mlp.psk --entry mse_grad --dim D=4 --dim B=2 --backend native --realize f32 2>/dev/null > out/opt_a.txt
	$(PRESK) run tests/examples/mlp/mlp.psk --entry mse_grad --dim D=4 --dim B=2 --backend native --realize f32 --opt 2>/dev/null > out/opt_b.txt
	cmp out/opt_a.txt out/opt_b.txt
	$(PRESK) train tests/examples/minigpt/train_minigpt.psk $(TINY) --steps 20 --log-every 5 --backend native --ckpt out/ckpt_opt 2>&1 | grep '^step' > out/opt_a.txt
	$(PRESK) train tests/examples/minigpt/train_minigpt.psk $(TINY) --steps 20 --log-every 5 --backend native --ckpt out/ckpt_opt --opt 2>&1 | grep '^step' > out/opt_b.txt
	cmp out/opt_a.txt out/opt_b.txt && cat out/opt_b.txt
	$(PRESK) bench tests/examples/minigpt/train_minigpt.psk --entry adamw.step $(TINY) --dim B=2 --dim T=8 --opt

# The corpus of the NLP course (corpus/nlp, EXPECTED.md): the counting bigram
# in closed form on 67652 real pairs, and the same model fitted by AdamW.
corpus: build
	python3 corpus/nlp/data/prepare.py
	$(PRESK) data pack --type '(idx[41], idx[41])' corpus/nlp/data/bigrams.txt -o corpus/nlp/data/bigrams.psk.d
	$(PRESK) check corpus/nlp/01_bigram_count.psk --summary
	$(PRESK) check corpus/nlp/02_bigram_neural.psk --summary
	$(PRESK) run corpus/nlp/01_bigram_count.psk --entry loss --data corpus/nlp/data/bigrams.psk.d --bits 128 --digits 10
	$(PRESK) run corpus/nlp/01_bigram_count.psk --entry loss --data corpus/nlp/data/bigrams.psk.d --backend native --digits 10
	$(PRESK) train corpus/nlp/02_bigram_neural.psk --backend native --log-every 1000 --ckpt out/nlp_bigram
	$(PRESK) data pack --type '(idx[41][ctx: 3], idx[41])' corpus/nlp/data/context3.txt -o corpus/nlp/data/context3.psk.d
	$(PRESK) check corpus/nlp/03_mlp.psk --summary
	$(PRESK) train corpus/nlp/03_mlp.psk --backend native --log-every 400 --ckpt out/nlp_mlp

# The epsilon-to-quality curve (0.5.0 step 8): one program, one parameter
# varying, every point verified by measurement against the exact reference.
curve: build
	REMAT=$(REMAT) ./experiments/curve.sh out/curve.md

# The gap of every realization split into representation and arithmetic, and
# statement K (plan 0.6.0 step 5d). Needs trained weights: `make gpt-trained`.
quant-table:
	$(CARGO) build --release -p presk
	./experiments/quant_table.sh out/quant_table.md

# The MLPerf comparison (plan 0.6.0 step 8, D53). `foil`: MLPerf's accuracy
# rule and a budget applied to the same realizations of the corpus GPT, on its
# held-out tenth, with what each costs (needs `make curve`'s runs; minutes).
# `bert`: BERT-large inference at MLPerf's constants on its initialization —
# the exact reference of one feature, each realization's gap on the SQuAD
# logits, the split on two blocks, and the time per sequence (about five
# hours, up to 76 GB; `L=2 make bert` for a quick version).
foil:
	$(CARGO) build --release -p presk
	./experiments/foil.sh out/foil.md

# ε from the margins (D118): the course GPT's exact margins on the held-out
# tenth (the reference at 128 bits, once), the largest ε on the logits that
# changes at most a fraction δ of its decisions with confidence 1 − α, and
# every realization of the foil and the curve against it: its gap on the
# logits, its flips and the bound the margins give (needs `make curve`'s
# runs; 5 minutes on falicon). DELTAS, ALPHA, WINDOWS, CHUNK, JOBS: see the
# script.
margins:
	$(CARGO) build --release -p presk
	./experiments/margins.sh
bert:
	$(CARGO) build --release -p presk
	./experiments/bert.sh out/bert.md

# The second model of the paper's RQ5 (D104): the Llama 2 decoder of
# coverage/llama.psk at the 7B configuration's width (G = 32), L of its 32
# blocks, on its initialization and random token ids, by `bert`'s protocol —
# the exact logits of one sequence of 384 tokens, the gaps of seven
# realizations, the split of the whole quantisation scheme, the time — and a
# budget of 1e-3 on its next-token loss solved, timed as `compile-times`
# times it, which `STAGES=solve` asks for: at the tag it did not fit on
# falicon (D104), since 0.10.1 it takes 14 h and 72 GB (D106). Alone on the
# measurement machine: about thirteen hours, and up to 64 GB at L = 1.
llama:
	$(CARGO) build --release -p presk
	./experiments/llama.sh out/llama.md

# A budget on one training step's update (plan 0.6.0 step 7b, D52): the gap
# table at D = 64 (needs `make gpt-trained` and out/gpt_prof.json), statement
# J′ at course scale (about half an hour), and 600 steps from scratch under
# the solved step against D50's seeds (needs `make curve`'s runs).
step-table:
	$(CARGO) build --release -p presk
	REMAT=$(REMAT) ./experiments/step_table.sh out/step_table.md
j-step:
	$(CARGO) build --release -p presk
	REMAT=$(REMAT) ./experiments/j_step.sh out/j_step.md
step-train:
	$(CARGO) build --release -p presk
	REMAT=$(REMAT) ./experiments/step_train.sh out/step_train.md

# δ_p, the gap of the reference interpreter itself (the reference's chapter 4,
# W16, D98): the course GPT's loss, logits and gradient at 128, 256 and 512
# bits against 1024 (1000 for the gradient), each printed at 330 digits and
# compared exactly, into out/delta_p/report.md (needs `make gpt-trained`;
# about half an hour, single-threaded, a value and not a time: any machine).
delta-p:
	$(CARGO) build --release -p presk
	./experiments/delta_p.sh

# The time the compiler's commands take per model, for the paper's setup
# table (presk-paper-001): `presk check`, `presk calibrate --precision`,
# `presk solve --budget 1e-3` and the exact reference on one sample, on the
# course GPT at both constants, the tiny MiniGPT and BERT-large's L = 2
# variant. A timing: a quiet machine, like j-step. MODELS= selects a subset.
compile-times:
	$(CARGO) build --release -p presk
	./experiments/compile_times.sh

# The corpus GPT trained for 600 steps at D = 64, the weights statement K and
# the quantisation table are measured on (a few minutes natively).
gpt-trained: build
	$(PRESK) train corpus/nlp/05_gpt.psk --const D=64 --const H=2 --const L=2 \
	    --steps 600 --epochs 3 --log-every 100 --backend native --realize f32 --remat $(REMAT) --ckpt out/gpt_trained

# The magnitude envelope a precision bound is conditional on (0.5.0 step 3),
# and the property the whole analysis rests on: two disjoint sets of samples
# must give magnitudes within a small factor of each other.
precision: build
	$(PRESK) calibrate --precision corpus/nlp/05_gpt.psk --entry loss \
	    --const D=32 --const H=2 --const L=2 --dim T=256 \
	    --data corpus/nlp/data/civil_T256.psk.d --bits 128 --samples 8 --from 0 \
	    --out out/envelope_a.json
	$(PRESK) calibrate --precision corpus/nlp/05_gpt.psk --entry loss \
	    --const D=32 --const H=2 --const L=2 --dim T=256 \
	    --data corpus/nlp/data/civil_T256.psk.d --bits 128 --samples 8 --from 8 \
	    --out out/envelope_b.json --compare out/envelope_a.json
	$(PRESK) calibrate --precision corpus/nlp/05_gpt.psk --entry loss \
	    --const D=32 --const H=2 --const L=2 --dim T=256 \
	    --data corpus/nlp/data/civil_T256.psk.d --bits 128 --samples 8 --from 64 \
	    --out out/envelope_c.json --compare out/envelope_a.json
	# And the bound that envelope supports, per realization (0.5.0 step 4).
	$(PRESK) check --precision out/envelope_a.json corpus/nlp/05_gpt.psk --realize f64
	$(PRESK) check --precision out/envelope_a.json corpus/nlp/05_gpt.psk --realize f32 --sites 6
	$(PRESK) check --precision out/envelope_a.json corpus/nlp/05_gpt.psk --realize bf16
	# At T = 256, the dimension the weights were trained at: `pos : ℝ[T, D]`
	# makes T part of the parameters, and 0.5.0 opened a T = 256 checkpoint at
	# T = 32 without noticing (D48).
	# On trained weights and two windows of text, the envelope now records
	# each sum's terms as well (0.6.0 step 6); the four rules for sums side by
	# side, each against the gap measured natively on the same inputs.
	$(PRESK) calibrate --precision corpus/nlp/05_gpt.psk --entry loss \
	    --const D=64 --const H=2 --const L=2 --dim T=256 --bits 128 \
	    --data corpus/nlp/data/civil_T256.psk.d --samples 2 --from 0 \
	    --weights out/gpt_trained/final.psk.d --out out/gpt_env.json
	$(PRESK) check --precision out/gpt_env.json corpus/nlp/05_gpt.psk --realize f64 --model all --measure
	$(PRESK) check --precision out/gpt_env.json corpus/nlp/05_gpt.psk --realize f32 --model all --measure
	$(PRESK) check --precision out/gpt_env.json corpus/nlp/05_gpt.psk --realize bf16 --model all --measure
	# Where a budget should go, from one reverse sweep on trained weights
	# (0.5.0 step 5). Without --weights the zero-initialized head makes every
	# upstream gradient exactly zero and the ranking says nothing.
	# The weights are `make gpt-trained`'s: 600 steps, not the 12 of
	# `corpus-gpt`, whose head has barely left zero.
	$(PRESK) dump corpus/nlp/05_gpt.psk --stage sensitivity --entry loss \
	    --const D=64 --const H=2 --const L=2 --dim B=2 --dim T=256 \
	    --weights out/gpt_trained/final.psk.d --sites 8
	# A budget in, a realization out (0.5.0 step 6); then run what it chose.
	$(PRESK) solve corpus/nlp/05_gpt.psk --precision out/gpt_env.json --budget 1e-3 \
	    --weights out/gpt_trained/final.psk.d --out out/solved.json --sites 6
	$(PRESK) check --precision out/gpt_env.json corpus/nlp/05_gpt.psk --realize @out/solved.json --model all --measure
	$(PRESK) run corpus/nlp/05_gpt.psk --entry loss --const D=64 --const H=2 --const L=2 \
	    --dim B=2 --dim T=256 --backend native --realize @out/solved.json --digits 10

# The Code civil as a character-level dataset, then the course GPT. `presk
# check` passes at the course's own constants (D=384, H=6, L=6, T=256); what
# trains here is reduced, because a course-scale step is the wall this version
# exists to break (corpus/nlp/EXPECTED.md).
corpus-gpt: build
	python3 corpus/nlp/data/prepare.py
	$(PRESK) data chars corpus/nlp/data/civil.md -o corpus/nlp/data/civil_vocab.json
	$(PRESK) data encode corpus/nlp/data/civil.md --vocab corpus/nlp/data/civil_vocab.json -o corpus/nlp/data/civil_tokens.txt
	$(PRESK) data lm-windows --vocab 92 --seq 256 corpus/nlp/data/civil_tokens.txt -o corpus/nlp/data/civil_T256.psk.d
	$(PRESK) data info corpus/nlp/data/civil_T256.psk.d --samples 1 | head -5
	$(PRESK) check corpus/nlp/05_gpt.psk --summary | tail -8
	$(PRESK) train corpus/nlp/05_gpt.psk --const D=64 --const H=2 --const L=2 \
	    --steps 12 --log-every 6 --backend native --realize f32 --ckpt out/gpt_tiny

# The cost model (manual chapter `optimization`): every kernel of the three
# entries timed natively, the roofline fitted and written to out/profile.json
# (the second and third calls extend the first), then the entries timed with
# the profile deciding the extraction.
calibrate: build
	rm -f out/profile.json
	$(PRESK) bench tests/examples/minigpt/minigpt.psk --entry MiniGPT.apply $(TINY) --dim B=2 --dim T=8 --repeat 7 --calibrate out/profile.json | tail -3
	$(PRESK) bench tests/examples/minigpt/train_minigpt.psk --entry loss $(TINY) --dim B=2 --dim T=8 --repeat 7 --calibrate out/profile.json | tail -3
	$(PRESK) bench tests/examples/minigpt/train_minigpt.psk --entry adamw.step $(TINY) --dim B=2 --dim T=8 --repeat 7 --calibrate out/profile.json
	$(PRESK) bench tests/examples/minigpt/train_minigpt.psk --entry loss $(TINY) --dim B=2 --dim T=8 --repeat 9 --opt --profile out/profile.json
	$(PRESK) bench tests/examples/minigpt/train_minigpt.psk --entry adamw.step $(TINY) --dim B=2 --dim T=8 --repeat 9 --opt --profile out/profile.json

# The realization pass (manual chapter `realize`): what shares a loop nest on
# the corpus GPT, and the launches it removes from the forward and from the
# optimizer's chain. `make sched` is kept as an alias of `make realize`.
sched: realize
realize: build
	$(PRESK) dump corpus/nlp/05_gpt.psk --stage realize --const D=64 --const H=2 --const L=2 | tail -12
	$(PRESK) dump corpus/nlp/05_gpt.psk --stage kernel --entry loss --const D=64 --const H=2 --const L=2 \
	    --dim B=2 --dim T=32 --realize f32 | grep '^--'
	$(PRESK) dump corpus/nlp/05_gpt.psk --stage kernel --entry adamw.step --const D=64 --const H=2 --const L=2 \
	    --dim B=2 --dim T=32 --realize f32 | grep '^--'
	$(PRESK) bench tests/examples/minigpt/train_minigpt.psk --entry adamw.step $(TINY) --dim B=2 --dim T=8 --repeat 21
	@echo "-- a 1024x1024x1024 f32 matmul, one thread and all cores"
	@printf 'const N : nat = 1024\nfn gemm(a : R[N, N], b : R[N, N]) -> R[N, N] = a . b\n' | sed 's/R\[/ℝ[/g; s/ \. / · /' > out/gemm.psk
	$(PRESK) bench out/gemm.psk --entry gemm --realize f32 --threads 1 --repeat 5 --calibrate out/gemm_prof.json | grep '^kernel dot'
	$(PRESK) bench out/gemm.psk --entry gemm --realize f32 --repeat 5 --calibrate out/gemm_prof.json | grep '^kernel dot'
	@echo "-- threads: one training step of the corpus GPT at 1 and at all cores"
	$(PRESK) train corpus/nlp/05_gpt.psk --const D=64 --const H=2 --const L=2 \
	    --steps 2 --log-every 2 --backend native --realize f32 --threads 1 --ckpt out/gpt_thr
	$(PRESK) train corpus/nlp/05_gpt.psk --const D=64 --const H=2 --const L=2 \
	    --steps 2 --log-every 2 --backend native --realize f32 --ckpt out/gpt_thr

bench: build
	$(PRESK) bench tests/examples/minigpt/minigpt.psk --entry MiniGPT.apply $(TINY) --dim B=2 --dim T=8
	$(PRESK) bench tests/examples/minigpt/train_minigpt.psk --entry loss $(TINY) --dim B=2 --dim T=8
	$(PRESK) bench tests/examples/minigpt/train_minigpt.psk --entry adamw.step $(TINY) --dim B=2 --dim T=8

# 0.7.0 step 4, widened in 0.8.0 step 3 (plan 0.8.0 §3.3, D67): the layout
# cost model's calibration. A GEMM at four sizes and two thin products, bound
# by the traffic from memory, in f32 and f64, each timed under a spanning set
# of layouts (`bench --space calib`: five register tiles, the loop orders with
# and without a k-block, both lane widths, one worker, half and all), pinned
# to the RA4 cores. The profile gains the machine description, the samples,
# costed under the rules RA4_RULES names, and the model fitted in the form
# RA4_LOAD_TERM and RA4_COMBINE say, by default the frozen model's (0.8.0
# step 4, D70–D71): a load rate per group, the p-norm, cache lines and loop
# overhead; then
# `presk refit --compare` prints every form fitted on the same samples. Needs a
# quiet machine; prints how long the measurement took. At the sizes of
# CALIB_NO_ONE_WORKER the one-worker layouts are left out (`--min-workers 2`):
# at 2048 they were three quarters of the time (0.8.0 step 4, D67). A
# calibration is one run: an existing profile is moved to *-prev.json first,
# so that none of its samples stays in the fit.
CALIB_SIZES ?= 256 512 1024 2048
CALIB_NO_ONE_WORKER ?= 2048
CALIB_THIN ?= I=16,K=4096,J=4096 I=4096,K=4096,J=16
RA4_PROFILE ?= out/profile.json
RA4_LOAD_TERM ?= group
RA4_COMBINE ?= pnorm
RA4_RULES ?= lines,overhead
CALIB_MODEL = --load-term $(RA4_LOAD_TERM) --combine $(RA4_COMBINE) --rules $(RA4_RULES)
calibrate-ra4:
	$(CARGO) build --release -p presk
	@mkdir -p out
	@if [ -f $(RA4_PROFILE) ]; then mv $(RA4_PROFILE) $(RA4_PROFILE:.json=-prev.json); echo "-- $(RA4_PROFILE) moved to $(RA4_PROFILE:.json=-prev.json)"; fi
	@start=$$(date +%s); \
	for r in f32 f64; do \
	  for n in $(CALIB_SIZES); do \
	    w=1; for m in $(CALIB_NO_ONE_WORKER); do [ $$m = $$n ] && w=2; done; \
	    echo "-- GEMM N=$$n, $$r, pinned to $(RA4_CORES)"; \
	    taskset -c $(RA4_CORES) ./target/release/presk bench experiments/gemm.psk --entry gemm --dim N=$$n \
	      --realize $$r --calibrate $(RA4_PROFILE) --space calib $(CALIB_MODEL) --min-workers $$w --repeat 5 | grep '^--' || exit 1; \
	  done; \
	  for s in $(CALIB_THIN); do \
	    echo "-- product $$s, $$r, pinned to $(RA4_CORES)"; \
	    taskset -c $(RA4_CORES) ./target/release/presk bench experiments/matmul.psk --entry matmul $$(echo $$s | tr ',' ' ' | sed 's/[^ ]*/--dim &/g') \
	      --realize $$r --calibrate $(RA4_PROFILE) --space calib $(CALIB_MODEL) --repeat 5 | grep '^--' || exit 1; \
	  done; \
	done; \
	echo "-- the calibration measured in $$(( $$(date +%s) - start )) s"
	@./target/release/presk refit $(RA4_PROFILE) --compare

# 0.7.0 step 5: the RA4 sweep (experiments/ra4.sh, D61), pinned to the RA4
# cores. `ra4-small` is the daily subset (`--space small`) on the tutorial's
# attention at B = 1 and 16, T = 256, f32 and f64: day one checks statement L
# natively and on the kernel interpreter, day two natively, then `presk rank`
# writes out/ra4-small.md — about a minute.
RA4_SMALL = FILE=tutorial/03_attention.psk ENTRY=attend SPACE=small
ra4-small:
	$(CARGO) build --release -p presk
	@rm -rf out/ra4-small
	@RA4_CORES=$(RA4_CORES) $(RA4_SMALL) INTERP=1 experiments/ra4.sh out/ra4-small/day1
	@RA4_CORES=$(RA4_CORES) $(RA4_SMALL) INTERP=0 experiments/ra4.sh out/ra4-small/day2 out/ra4-small/day1 out/ra4-small.md

# 0.7.0 step 6: the measurement (D62). A day is three sweeps into one
# directory, pinned, timed, checked natively against point 0: the course
# GPT's attention at the constants of plan §3.4 (B = 1 and 16, T = 256, f32
# and f64, the whole space), the GEMM at 1024 in f32 (statement P), and the
# BERT-large-shaped attention (D = 1024, H = 16, T = 384, B = 1) on the
# model's eight best points per nest and thirty-two at random. `ra4-check`
# runs every point of day one once more, untimed, on the kernel interpreter
# too (statement L), in shards on every core, so it must not overlap a timed
# sweep. `ra4-20` is the secondary sweep on all twenty cores, calibrated on
# them. `ra4-day2` repeats day one and ranks the two, the check included.
RA4_GEMM = FILE=experiments/gemm.psk ENTRY=gemm SHAPES=N=1024 TYPES=f32
RA4_BERT = CONSTS="D=1024 H=16" SHAPES=B=1,T=384 TYPES=f32 EXTRA="--top 8 --random 32"
RA4_JOBS ?= 20
ra4 ra4-day2:
	$(CARGO) build --release -p presk
	@d=out/ra4/$(if $(filter ra4-day2,$@),day2,day1); \
	RA4_CORES=$(RA4_CORES) INTERP=0 experiments/ra4.sh $$d && \
	RA4_CORES=$(RA4_CORES) INTERP=0 $(RA4_GEMM) experiments/ra4.sh $$d && \
	RA4_CORES=$(RA4_CORES) INTERP=0 $(RA4_BERT) experiments/ra4.sh $$d
	@if [ $@ = ra4-day2 ]; then ./target/release/presk rank out/ra4/day1 out/ra4/day2 --out out/ra4.md \
	  $$([ -d out/ra4/check ] && echo --check out/ra4/check) | tail -1; fi

ra4-check:
	$(CARGO) build --release -p presk
	@{ SHAPES=B=16,T=256 SHARDS=40 experiments/ra4_check.sh out/ra4/check; \
	   $(RA4_GEMM) SHARDS=40 experiments/ra4_check.sh out/ra4/check; \
	   SHAPES=B=1,T=256 SHARDS=4 experiments/ra4_check.sh out/ra4/check; \
	   $(RA4_BERT) SHARDS=2 experiments/ra4_check.sh out/ra4/check; } | xargs -d '\n' -n 1 -P $(RA4_JOBS) bash -c
	@echo "-- statement L: every shard of out/ra4/check bit-identical to point 0, natively and on the kernel interpreter"

ra4-20:
	$(MAKE) calibrate-ra4 RA4_CORES=0-19 RA4_PROFILE=out/profile-20.json
	@RA4_CORES=0-19 PROFILE=out/profile-20.json INTERP=0 experiments/ra4.sh out/ra4-20/day1
	@./target/release/presk rank out/ra4-20/day1 out/ra4-20/day1 --out out/ra4-20.md | tail -1

# 0.8.0 step 1: the development set (plan 0.8.0 §3.4, D63, D64). 0.7.0's
# measurements, archived read-only in RA4_DEV, re-priced under a profile by
# `presk reprice` (every recorded point lowered again and costed, its times
# copied) and ranked into out/ra4-dev/ra4.md and ra4-20.md — about a minute,
# nothing measured. Under the default profiles, 0.7.0's, every re-priced line
# must be the archived one and each report the one in docs/, byte for byte.
# An empty RA4_DEV_PROFILE_20 leaves the twenty-core day out, as 0.8.0's
# development does (D68): `make ra4-dev RA4_DEV_PROFILE=P RA4_DEV_PROFILE_20=`.
RA4_DEV ?= $(HOME)/presk-data/ra4-0.7.0
RA4_DEV_PROFILE ?= $(RA4_DEV)/out/profile.json
RA4_DEV_PROFILE_20 ?= $(RA4_DEV)/out/profile-20.json
ra4-dev:
	$(CARGO) build --release -p presk
	@RA4_DEV=$(RA4_DEV) PROFILE=$(RA4_DEV_PROFILE) PROFILE_20=$(RA4_DEV_PROFILE_20) FROZEN=$(RA4_FROZEN) experiments/ra4_dev.sh out/ra4-dev

# 0.8.0 step 4 (D71): the frozen model's profile, the definitive calibration
# re-costed under the rules kept and refitted in the chosen form, archived
# read-only. Under it the development set's report must be
# docs/ra4-dev-model-0.8.0.md, byte for byte: the check a defect fixed after
# the freeze must pass (plan 0.8.0 §3.4).
RA4_FROZEN ?= $(HOME)/presk-data/ra4-model-0.8.0/profile.json
ra4-dev-frozen:
	$(MAKE) ra4-dev RA4_DEV_PROFILE=$(RA4_FROZEN) RA4_DEV_PROFILE_20=

# 0.8.0 step 4: the ablation of plan §3.4 (experiments/ra4_ablation.sh). The
# development set's ten-core days priced under every form of the model, each
# refitted on the definitive calibration (RA4_CALIBRATION, archived, D68),
# beside 0.7.0's own model, into out/ra4-ablation/ablation.md — a few
# minutes, nothing measured. FORMS restricts the forms; a form L-C+R names
# rules of the planned cost, under which the calibration is re-costed first
# (4c): FORMS="group-pnorm group-pnorm+lines,overhead".
RA4_CALIBRATION ?= $(HOME)/presk-data/ra4-0.8.0-calibration/out/profile.json
ra4-ablation:
	$(CARGO) build --release -p presk
	@RA4_DEV=$(RA4_DEV) CALIBRATION=$(RA4_CALIBRATION) experiments/ra4_ablation.sh out/ra4-ablation

# 0.8.0 step 5: F2, the fresh sweep of 0.7.0's shapes on the ten A725 cores
# (plan 0.8.0 §3.5, D72), priced by their own calibration in the frozen form.
# `calibrate-ra4-small` is `calibrate-ra4` pinned to RA4_CORES_SMALL, into
# RA4_PROFILE_SMALL, never out/profile.json. `ra4-f2-dry` times the sweep
# of the nest the model predicts longest and sizes a day from it
# (experiments/ra4_dry.sh) into RA4_F2_DIR/dry. `ra4-f2` and `ra4-f2-day2`
# are the two days, on two calendar dates, from a committed tree descending
# from RA4_TAG (experiments/ra4_fresh.sh checks both before measuring);
# `ra4-f2-check` is statement L on the kernel interpreter, sharded over every
# core, never beside a timed run; day two ends with `ra4-f2-rank`: `presk
# rank --frozen RA4_TAG` into RA4_F2_DIR.md, the check included. The
# configuration is RA4_F2_RUN at the shapes of RA4_F2_SHAPES, each with the
# check's shard count. RA4_TRIAL=1 tries the targets on a tiny configuration
# (other RA4_F2_DIR, RA4_PROFILE_SMALL, RA4_F2_RUN, RA4_F2_SHAPES): no tree,
# tag or calendar check, and a rank without --frozen. Each target refuses to
# start beside another presk process.
RA4_PROFILE_SMALL ?= out/profile-a725.json
RA4_F2_DIR ?= out/ra4-f2
RA4_F2_RUN ?= FILE=experiments/ra4_attention.psk ENTRY=SelfAttention.apply TYPES="f32 f64" SPACE=full
RA4_F2_SHAPES ?= B=1,T=256:4 B=16,T=256:40
RA4_TAG ?= ra4-model-0.8.0
RA4_TRIAL ?=
# RA4_SAME_DATE=1 accepts day two on day one's calendar date (the user's
# decision for F2 and F1, 2026-09-25, D72).
RA4_SAME_DATE ?=
RA4_F2 = RA4_CORES=$(RA4_CORES_SMALL) PROFILE=$(RA4_PROFILE_SMALL) $(RA4_F2_RUN) \
  SHAPES="$(foreach s,$(RA4_F2_SHAPES),$(firstword $(subst :, ,$(s))))"
QUIET = if pgrep -x presk > /dev/null; then echo "-- a presk process is running: a timed run needs a quiet machine"; pgrep -ax presk; exit 1; fi
calibrate-ra4-small:
	@[ "$(abspath $(RA4_PROFILE_SMALL))" != "$(abspath $(RA4_PROFILE))" ] || { echo "-- RA4_PROFILE_SMALL is $(RA4_PROFILE): the A725 cores get a profile of their own"; exit 1; }
	@$(QUIET)
	$(MAKE) calibrate-ra4 RA4_CORES=$(RA4_CORES_SMALL) RA4_PROFILE=$(RA4_PROFILE_SMALL)

ra4-f2-dry:
	$(CARGO) build --release -p presk
	@$(QUIET)
	@$(RA4_F2) experiments/ra4_dry.sh $(RA4_F2_DIR)/dry

ra4-f2 ra4-f2-day2:
	$(CARGO) build --release -p presk
	@$(QUIET)
	@d=$(RA4_F2_DIR)/$(if $(filter ra4-f2-day2,$@),day2,day1); \
	TAG=$(RA4_TAG) TRIAL=$(RA4_TRIAL) SAME_DATE=$(RA4_SAME_DATE) experiments/ra4_fresh.sh $$d $(if $(filter ra4-f2-day2,$@),$(RA4_F2_DIR)/day1) && \
	$(RA4_F2) INTERP=0 experiments/ra4.sh $$d
	@if [ $@ = ra4-f2-day2 ]; then $(MAKE) --no-print-directory ra4-f2-rank; fi

ra4-f2-rank:
	@./target/release/presk rank $(RA4_F2_DIR)/day1 $(RA4_F2_DIR)/day2 --out $(RA4_F2_DIR).md \
	  $(if $(RA4_TRIAL),,--frozen $(RA4_TAG)) $$([ -d $(RA4_F2_DIR)/check ] && echo --check $(RA4_F2_DIR)/check) | tail -1

ra4-f2-check:
	$(CARGO) build --release -p presk
	@$(QUIET)
	@[ ! -e $(RA4_F2_DIR)/check ] || { echo "-- $(RA4_F2_DIR)/check exists: move it aside first"; exit 1; }
	@{ $(foreach s,$(RA4_F2_SHAPES),SHAPES=$(firstword $(subst :, ,$(s))) SHARDS=$(lastword $(subst :, ,$(s))) \
	   PROFILE=$(RA4_PROFILE_SMALL) $(RA4_F2_RUN) experiments/ra4_check.sh $(RA4_F2_DIR)/check;) } | xargs -d '\n' -n 1 -P $(RA4_JOBS) bash -c
	@echo "-- statement L: every shard of $(RA4_F2_DIR)/check bit-identical to point 0, natively and on the kernel interpreter"

# 0.8.0 step 6: F1, the fresh sweep of new shapes on the ten X925 cores
# (plan 0.8.0 §3.5, D73), priced by the frozen model's profile itself,
# RA4_FROZEN (D68: no calibration after the freeze). Two configurations of
# RA4_F1_RUN, each `constants:shape`, go into one day: RA4_F1A, BERT-large's
# attention, and RA4_F1B, a head of 128. `ra4-f1-dry` runs
# experiments/ra4_dry.sh once per configuration, each timing its own longest
# nest, into RA4_F1_DIR/dry/f1a and f1b, then sizes the day into
# RA4_F1_DIR/dry/dry.md (`ra4_dry.py day`): a configuration that does not
# fit in RA4_F1_BUDGET hours a day is swept on a sample, RA4_F1A_SAMPLE or
# RA4_F1B_SAMPLE (ra4.sh's EXTRA, e.g. "--top 8 --random 32"), set by a
# commit before day one so that both days and the check take the same
# points, and its M is inconclusive. `ra4-f1`, `ra4-f1-day2` and
# `ra4-f1-check` are F2's; the check runs RA4_F1B_SHARDS shards of F1b, then
# RA4_F1A_SHARDS of F1a. `presk rank` decides M and O at one configuration a
# run, so `ra4-f1-rank` ranks the two days twice, the check included, into
# RA4_F1_DIR`a`.md (M and O at F1a) and RA4_F1_DIR`b`.md (at F1b); L and N
# are the same in both. RA4_TRIAL=1 as for F2, with other RA4_F1_DIR,
# RA4_F1_RUN, RA4_F1A and RA4_F1B.
RA4_F1_DIR ?= out/ra4-f1
RA4_F1_PROFILE ?= $(RA4_FROZEN)
RA4_F1_RUN ?= FILE=experiments/ra4_attention.psk ENTRY=SelfAttention.apply TYPES=f32 SPACE=full
RA4_F1A ?= D=1024,H=16:B=1,T=384
RA4_F1B ?= D=1024,H=8:B=2,T=512
RA4_F1A_SAMPLE ?=
RA4_F1B_SAMPLE ?=
RA4_F1A_SHARDS ?= 20
RA4_F1B_SHARDS ?= 80
RA4_F1_BUDGET ?= 8
comma := ,
# A configuration's constants (as words), its shape, its `presk rank --at`
# (the sweep's config: constants then dimensions), and its letter.
f1_consts = $(subst $(comma), ,$(word 1,$(subst :, ,$(RA4_F1$(1)))))
f1_shape = $(word 2,$(subst :, ,$(RA4_F1$(1))))
f1_at = $(subst :,$(comma),$(RA4_F1$(1)))
f1_lc = $(subst A,a,$(subst B,b,$(1)))
RA4_F1 = RA4_CORES=$(RA4_CORES) PROFILE=$(RA4_F1_PROFILE) $(RA4_F1_RUN) CONSTS="$(call f1_consts,$(1))" SHAPES=$(call f1_shape,$(1))

ra4-f1-dry:
	$(CARGO) build --release -p presk
	@$(QUIET)
	@rm -rf $(RA4_F1_DIR)/dry
	@$(foreach c,A B,$(call RA4_F1,$(c)) experiments/ra4_dry.sh $(RA4_F1_DIR)/dry/f1$(call f1_lc,$(c)) || exit 1;)
	@python3 experiments/ra4_dry.py day 7 $(RA4_F1_BUDGET) $(RA4_F1_DIR)/dry/f1a $(RA4_F1_DIR)/dry/f1b > $(RA4_F1_DIR)/dry/dry.md
	@cat $(RA4_F1_DIR)/dry/dry.md

ra4-f1 ra4-f1-day2:
	$(CARGO) build --release -p presk
	@$(QUIET)
	@d=$(RA4_F1_DIR)/$(if $(filter ra4-f1-day2,$@),day2,day1); \
	TAG=$(RA4_TAG) TRIAL=$(RA4_TRIAL) SAME_DATE=$(RA4_SAME_DATE) experiments/ra4_fresh.sh $$d $(if $(filter ra4-f1-day2,$@),$(RA4_F1_DIR)/day1) && \
	$(foreach c,A B,$(call RA4_F1,$(c)) EXTRA="$(RA4_F1$(c)_SAMPLE)" INTERP=0 experiments/ra4.sh $$d && )true
	@if [ $@ = ra4-f1-day2 ]; then $(MAKE) --no-print-directory ra4-f1-rank; fi

ra4-f1-rank:
	@$(foreach c,A B,./target/release/presk rank $(RA4_F1_DIR)/day1 $(RA4_F1_DIR)/day2 --out $(RA4_F1_DIR)$(call f1_lc,$(c)).md --at $(call f1_at,$(c)) \
	  $(if $(RA4_TRIAL),,--frozen $(RA4_TAG)) $$([ -d $(RA4_F1_DIR)/check ] && echo --check $(RA4_F1_DIR)/check) > /dev/null || exit 1; \
	  echo "-- the two days ranked into $(RA4_F1_DIR)$(call f1_lc,$(c)).md, M and O at $(call f1_at,$(c))"; \
	  grep '^| [LMNOQ] |' $(RA4_F1_DIR)$(call f1_lc,$(c)).md;)

ra4-f1-check:
	$(CARGO) build --release -p presk
	@$(QUIET)
	@[ ! -e $(RA4_F1_DIR)/check ] || { echo "-- $(RA4_F1_DIR)/check exists: move it aside first"; exit 1; }
	@{ $(foreach c,B A,$(call RA4_F1,$(c)) EXTRA="$(RA4_F1$(c)_SAMPLE)" SHARDS=$(RA4_F1$(c)_SHARDS) \
	   experiments/ra4_check.sh $(RA4_F1_DIR)/check;) } | xargs -d '\n' -n 1 -P $(RA4_JOBS) bash -c
	@echo "-- statement L: every shard of $(RA4_F1_DIR)/check bit-identical to point 0, natively and on the kernel interpreter"

# 0.10.0 step 2: T3, RA4's third fresh test (plan 0.10.0 §3.5, D90). The
# derivatives of `dot` in the course GPT's gradient, contraction nests since
# 0.10.0 step 1 that the frozen model has never seen, swept on the ten X925
# cores at the course constants and priced by the frozen profile itself
# (RA4_T3_PROFILE), unchanged and not refitted. The configuration is
# RA4_T3_ENTRY of RA4_T3_FILE at RA4_T3_SHAPE (after the constants
# RA4_T3_CONSTS, none by default) in RA4_T3_TYPE, and a day is one sweep of
# the nests RA4_T3_NESTS (ra4.sh's NEST): `dot_vjp_l` and `dot_vjp_r` of
# Linear.apply at three shapes each, and of the head. `ra4-t3-dry` prices
# them and sweeps the nest the model predicts longest (experiments/ra4_dry.sh)
# into RA4_T3_DIR/dry, then sizes a day of RA4_T3_BUDGET hours into
# RA4_T3_DIR/dry/day.md; a day that does not fit is swept on a sample,
# RA4_T3_SAMPLE (ra4.sh's EXTRA), only once the user decides it, since Q is
# then undecided. `ra4-t3`, `ra4-t3-day2` and `ra4-t3-check` are F2's, the
# check in RA4_T3_SHARDS shards; `ra4-t3-rank` ranks the two days with
# `presk rank --frozen` into RA4_T3_DIR.md: statements O and Q on the
# derivative nests (M is the `ein`s', and the sweep holds none), L with the
# check, N on the configuration. RA4_TRIAL=1 as for F2, with another
# RA4_T3_DIR and a tiny configuration: RA4_T3_CONSTS="D=32 H=2"
# RA4_T3_SHAPE=B=2,T=16 RA4_T3_SPACE=small.
RA4_T3_DIR ?= out/ra4-t3
RA4_T3_PROFILE ?= $(RA4_FROZEN)
RA4_T3_FILE ?= experiments/gpt_grad.psk
RA4_T3_ENTRY ?= loss_grad
RA4_T3_CONSTS ?=
RA4_T3_SHAPE ?= B=16,T=256
RA4_T3_TYPE ?= f32
RA4_T3_SPACE ?= full
# The nests are the `.bwd`s' since 0.10.0 step 4 (D92): a gradient keeps its
# residuals, and each sweep is its function's `.bwd`, numbered as the `.vjp`
# was. T3's days (D90, archived) name them `.vjp`, the program
# `--remat recompute` differentiates: re-price them with that.
RA4_T3_NESTS ?= Linear.apply.bwd/%11 Linear.apply.bwd/%12 GPT.apply.bwd/%28 GPT.apply.bwd/%29
RA4_T3_SAMPLE ?=
RA4_T3_SHARDS ?= 80
RA4_T3_BUDGET ?= 8
empty :=
space := $(empty) $(empty)
RA4_T3 = RA4_CORES=$(RA4_CORES) PROFILE=$(RA4_T3_PROFILE) FILE=$(RA4_T3_FILE) ENTRY=$(RA4_T3_ENTRY) CONSTS="$(RA4_T3_CONSTS)" \
  SHAPES=$(RA4_T3_SHAPE) TYPES=$(RA4_T3_TYPE) SPACE=$(RA4_T3_SPACE) NEST="$(RA4_T3_NESTS)" EXTRA="$(RA4_T3_SAMPLE)"

ra4-t3-dry:
	$(CARGO) build --release -p presk
	@$(QUIET)
	@$(RA4_T3) experiments/ra4_dry.sh $(RA4_T3_DIR)/dry
	@python3 experiments/ra4_dry.py day 7 $(RA4_T3_BUDGET) $(RA4_T3_DIR)/dry > $(RA4_T3_DIR)/dry/day.md
	@cat $(RA4_T3_DIR)/dry/day.md

ra4-t3 ra4-t3-day2:
	$(CARGO) build --release -p presk
	@$(QUIET)
	@d=$(RA4_T3_DIR)/$(if $(filter ra4-t3-day2,$@),day2,day1); \
	TAG=$(RA4_TAG) TRIAL=$(RA4_TRIAL) SAME_DATE=$(RA4_SAME_DATE) experiments/ra4_fresh.sh $$d $(if $(filter ra4-t3-day2,$@),$(RA4_T3_DIR)/day1) && \
	$(RA4_T3) INTERP=0 experiments/ra4.sh $$d
	@if [ $@ = ra4-t3-day2 ]; then $(MAKE) --no-print-directory ra4-t3-rank; fi

ra4-t3-rank:
	@./target/release/presk rank $(RA4_T3_DIR)/day1 $(RA4_T3_DIR)/day2 --out $(RA4_T3_DIR).md \
	  --entry $(RA4_T3_ENTRY) --at $(subst $(space),$(comma),$(strip $(RA4_T3_CONSTS) $(RA4_T3_SHAPE))) --type $(RA4_T3_TYPE) \
	  $(if $(RA4_TRIAL),,--frozen $(RA4_TAG)) $$([ -d $(RA4_T3_DIR)/check ] && echo --check $(RA4_T3_DIR)/check) > /dev/null || exit 1
	@echo "-- the two days ranked into $(RA4_T3_DIR).md"; grep '^| [LMNOQ] |' $(RA4_T3_DIR).md

ra4-t3-check:
	$(CARGO) build --release -p presk
	@$(QUIET)
	@[ ! -e $(RA4_T3_DIR)/check ] || { echo "-- $(RA4_T3_DIR)/check exists: move it aside first"; exit 1; }
	@$(RA4_T3) SHARDS=$(RA4_T3_SHARDS) experiments/ra4_check.sh $(RA4_T3_DIR)/check | xargs -d '\n' -n 1 -P $(RA4_JOBS) bash -c
	@echo "-- statement L: every shard of $(RA4_T3_DIR)/check bit-identical to point 0, natively and on the kernel interpreter"

# 0.7.0 step 1: a plan with the default layout lowers every entry of
# tests/plan/entries.txt to the kernels it has without one, byte for byte
# (`dump --stage kernel`, f64). PLAN names another plan to compare.
PLAN ?= tests/plan/default.json
plan-check: build
	@mkdir -p out
	@while IFS='|' read -r file entry args; do \
	  $(PRESK) dump --stage kernel "$$file" --entry "$$entry" $$args > out/plan-check.bare || exit 1; \
	  $(PRESK) dump --stage kernel "$$file" --entry "$$entry" $$args --plan @$(PLAN) > out/plan-check.plan || exit 1; \
	  if cmp -s out/plan-check.bare out/plan-check.plan; then echo "same kernels  $$entry ($$file)"; \
	  else echo "DIFFERENT     $$entry ($$file)"; diff out/plan-check.bare out/plan-check.plan | head -20; exit 1; fi; \
	done < tests/plan/entries.txt

# 0.9.0 step 3 (D79): statements C1 and C2 of the C export. Every entry of
# tests/plan/entries.txt (next_token since step 6), in f64, f32 and bf16, on its
# synthesized parameters and with their zeros drawn (also in `make test`),
# and the course GPT at its constants, zeros drawn at T = 8 and with the
# course's trained checkpoint at T = 256 (out/course_probe_f32/final.psk.d):
# each exported, built and run by its own conformance/check.sh, its outputs
# compared with the native backend's bit for bit, and the elements outside
# the conformance band reported. Step 5 (D81): every entry also under the
# plan `presk plan` gives with the frozen profile (RA4_FROZEN), the course
# GPT at B = 16, T = 256 under its plans and under the default layout, its
# launches divided among threads; every program built with GCC with OpenMP
# and without it and run on one thread and on every thread, and with Clang
# and for x86-64 (zig cc, statement C3). Step 6 (D83): statement C5, the
# generation loop exported with --serve against `presk serve --backend
# native`, with the tiny MiniGPT's trained checkpoint (out/ckpt/final.psk.d,
# `make train-examples`) in the three types through both hosts, and C1 on
# next_token with it; generate.c through C3 too. Step 8 (D85): the SCoP
# mode, every nest of the test set in the three types and the course GPT's
# at its constants, built, run against the native backend, and read by
# Polly (LLVM 23.1.2's clang, `make llvm`): one SCoP per static-control nest.
c-conformance: .venv-zig/bin/python .llvm/bin/clang
	PRESK_C3=1 RA4_FROZEN=$(RA4_FROZEN) $(CARGO) test --release -p presk --test export_c --test export_scop -- --include-ignored --nocapture

# 0.9.0 step 8 (D85): Polly, the polyhedral front end the SCoP mode's nests
# are read by. LLVM's release links it into its clang (`-mllvm -polly`). It is
# 1.8 GB, from GitHub, its checksum pinned; only clang and its headers are
# kept, in .llvm, without root (presk-tooling.md, `Polly`). The release for
# Linux on aarch64 only, falicon's; elsewhere, POLLY names a clang with Polly.
LLVM_VERSION = 23.1.2
LLVM_RELEASE = LLVM-$(LLVM_VERSION)-Linux-ARM64
LLVM_SHA256 = 075da47cb832273717d4c7bad4b6b4848d7c154262e9ba0dcc0025426d4073f0
llvm: .llvm/bin/clang
.llvm/bin/clang:
	@test "$$(uname -sm)" = "Linux aarch64" || { echo "make llvm: LLVM's release for Linux on aarch64 only; elsewhere set POLLY to a clang that links Polly"; exit 1; }
	mkdir -p .llvm
	curl -fL -o .llvm/$(LLVM_RELEASE).tar.xz https://github.com/llvm/llvm-project/releases/download/llvmorg-$(LLVM_VERSION)/$(LLVM_RELEASE).tar.xz
	echo "$(LLVM_SHA256)  .llvm/$(LLVM_RELEASE).tar.xz" | sha256sum -c -
	tar -xJf .llvm/$(LLVM_RELEASE).tar.xz -C .llvm --strip-components=1 $(LLVM_RELEASE)/bin/clang $(LLVM_RELEASE)/bin/clang-$(firstword $(subst ., ,$(LLVM_VERSION))) $(LLVM_RELEASE)/lib/clang
	rm .llvm/$(LLVM_RELEASE).tar.xz
	.llvm/bin/clang --version | head -1

# 0.9.0 step 5 (D81): statement C3's Clang and x86-64 cross-compiler, `zig
# cc` from zig's PyPI package, in a virtual environment: no sudo, one pinned
# version, the same on every machine (presk-tooling.md, `zig`).
ZIG_VERSION ?= 0.16.0
zig: .venv-zig/bin/python
.venv-zig/bin/python:
	python3 -m venv .venv-zig
	.venv-zig/bin/pip install -q ziglang==$(ZIG_VERSION)

# 0.9.0 step 5 (D81): statement C4, reported and deciding nothing: the
# exported course GPT's forward pass at B = 16, T = 256 with the trained
# checkpoint, under the plan the frozen profile gives, against `presk bench
# --plan-only`, one thread and every thread, into out/c-speed/report.md.
# Times: a quiet machine, nothing beside it.
c-speed:
	$(CARGO) build --release -p presk
	PROFILE=$(RA4_FROZEN) experiments/c_speed.sh out/c-speed

# The PyTorch baseline at 0.9.0 (D87; the manual's `experiments` §3): the
# course GPT at B = 16, T = 256 in f32, with the checkpoint C4 uses, its
# forward pass and one training step timed on three sides — the native
# backend, the C export and PyTorch (the model `presk export --target
# pytorch` writes, in .venv) — under the default layout and under the plans
# the frozen profile gives, on the ten X925 cores pinned (RA4_CORES) and on
# every core; the distances to the native backend's f64 logits, the
# operations counted, the linear products alone, and where the C program's
# time goes kernel by kernel. Into out/pytorch-baseline/report.md. Needs
# corpus/nlp/data/civil_T256.psk.d (make experiment-gpt) and a quiet machine.
# Since 0.10.0 step 7: CFLAGS builds the C programs with those flags in place
# of their Makefiles' -O2 (`make pytorch-baseline CFLAGS="-O3 -mcpu=native"
# BASELINE_OUT=out/pytorch-baseline-o3`), C1 checked on each.
# (CFLAGS is kept from cargo, whose build scripts would build GMP with it.)
# Since step 7's follow-up: each C program timed twice, under OpenMP's
# default waiting and under C_WAIT, the native pool's (T4's row of record).
BASELINE_OUT ?= out/pytorch-baseline
C_WAIT ?= GOMP_SPINCOUNT=8M
pytorch-baseline:
	env -u CFLAGS $(CARGO) build --release -p presk
	PROFILE=$(RA4_FROZEN) CORES=$(RA4_CORES) CFLAGS="$(CFLAGS)" C_WAIT="$(C_WAIT)" experiments/pytorch_baseline.sh $(BASELINE_OUT)

# PyTorch eager on the Llama 2 decoder block (D107): D100's protocol on one
# block of coverage/llama.psk at the 7B width, the forward pass at 384
# tokens, batch 1 and 20, `f32` — the native backend in `f32` and with
# `int8` weights on every matrix, under the default layouts and under the
# frozen profile's plans, against PyTorch on the module the PyTorch export
# writes (ten threads, and one at batch 1), on the ten X925 cores pinned
# (RA4_CORES); then the native kernels alone under the `f32` plan and the
# host by phase at batch 1. Into out/pytorch-decoder/report.md, outputs
# that exist reused. Needs .venv; alone on the machine: the timed rows about
# an hour and a half, the kernels alone two hours more, a native process
# about 35 GB. `T=16` is its gate (about ten minutes).
DECODER_OUT ?= out/pytorch-decoder
pytorch-decoder:
	$(CARGO) build --release -p presk
	PROFILE=$(RA4_FROZEN) CORES=$(RA4_CORES) experiments/pytorch_decoder.sh $(DECODER_OUT)

# Where the measurement changes the solver's decision (D128): the six entries
# of experiments/widening.psk, calibrated at 128 bits, checked under f64,
# f32 and bf16 (the four rules and the measured gap), solved at seven
# budgets with every move `presk explain` records, and the proposal with
# only the misjudged value widened measured beside each widening solve;
# into out/widening/summary.md. Nothing timed, any machine, a few minutes.
widening:
	$(CARGO) build --release -p presk
	./experiments/widening.sh

# PyTorch's own gap to the exact model (D130): make margins' model exported
# to PyTorch and run in eval mode on the same windows, float32 on ten X925
# cores and on one, float64 on ten, each row's largest gap, flips and bound
# counted by experiments/margins.py report beside the compiler's rows. Needs
# make margins' output (EXTRA as experiments/pytorch_gap.sh says, for the
# solved rows) and the .venv's torch.
pytorch-gap:
	$(CARGO) build --release -p presk
	./experiments/pytorch_gap.sh

# One nest's bound proved in Gappa, beside the compiler's bound and the
# measured gap (D108): `LayerNorm{D}.apply` in f32 at D = 8, 16, 32 and 64
# on eight rows, each width's envelope, the four rules' bounds and the gap
# (`presk calibrate`/`check --precision`), the kernels' order, and the Gappa
# file experiments/gappa_layernorm.py writes from the envelope, queried then
# checked as a proof; into out/gappa/summary.md. Needs `gappa` on the PATH
# (Debian's package; GAPPA= another). Nothing timed, any machine: Gappa
# takes seconds at D = 8 and about ten minutes at D = 64. WIDTHS=, BATCH=,
# HYPS= (sites, input), REALIZE= (f32, f64, bf16) and OUT= select.
gappa:
	$(CARGO) build --release -p presk
	WIDTHS="$(WIDTHS)" BATCH="$(BATCH)" HYPS="$(HYPS)" REALIZE="$(REALIZE)" OUT="$(GAPPA_OUT)" ./experiments/gappa.sh
WIDTHS ?= 8 16 32 64
BATCH ?= 8
HYPS ?= sites input
REALIZE ?= f32
GAPPA_OUT ?= out/gappa

# A contraction's bound proved in Gappa (D131): experiments/gappa_dot.sh, a
# copy of gappa.sh for `dot` of experiments/gappa_dot.psk, one output
# element of [8, K] · [K, 8] in f32 at K = 8, 16, 32, 64, 128 under the
# sequential order and interleaved(n), the hypotheses the operands' and the
# output's magnitudes from the envelope; into out/gappa-dot/summary.md.
# Needs `gappa` on the PATH, or GAPPA= (a source build: the manual's
# dependencies). About fifteen minutes, K = 128 the most.
gappa-dot:
	$(CARGO) build --release -p presk
	GAPPA="$(GAPPA)" ./experiments/gappa_dot.sh
GAPPA ?= gappa

# The native host measured (plan 0.10.0 step 5, D95): `presk bench
# --host-profile` and `presk train --host-profile` on the course GPT under
# the frozen profile's plans, ten X925 cores pinned and every core unpinned.
# Into out/host-profile/report.md. Needs what pytorch-baseline needs, not
# PyTorch, and a quiet machine. CONFIGS=10 runs the pinned configuration
# alone (step 6, each fix measured apart).
host-profile:
	$(CARGO) build --release -p presk
	PROFILE=$(RA4_FROZEN) CORES=$(RA4_CORES) CONFIGS="$(CONFIGS)" experiments/host_profile.sh out/host-profile

sync-examples:
	rsync -a --delete --exclude PROVENANCE.md $(PRESK_SPECS_DIR)/examples/ tests/examples/
	@sed -i.bak "s/^Commit: .*/Commit: $$(git -C $(PRESK_SPECS_DIR) rev-parse --short HEAD)/" tests/examples/PROVENANCE.md \
	  && rm tests/examples/PROVENANCE.md.bak   # -i.bak: the one spelling GNU and BSD sed both accept
	git status --short tests/examples

snap:
	$(CARGO) insta review

doc:
	$(CARGO) doc -p presk-manual --no-deps --open

# presk explain (plan 0.11.0): one compilation as a document. `explain`
# writes the tutorial's `loss` under a budget of 1e-3 and the frozen X925
# profile (its constants, without its samples), on ten threads, every
# target; `explain-pdf` makes EXPLAIN/explain.pdf from EXPLAIN/explain.md
# with pandoc 3.1.3 and lualatex from TeX Live 2026 (TEXLIVE=), in the
# DejaVu fonts, whose monospace face has ℝ, θ, ∂ and the subscripts, and
# fvextra's verbatim, which breaks a long listing's lines (the manual's
# `dependencies` chapter). Nothing timed, any machine.
EXPLAIN ?= out/explain/loss
EXPLAIN_PROFILE ?= crates/presk-explain/tests/data/x925-frozen.json
TEXLIVE ?= /usr/local/texlive/2026/bin/aarch64-linux
explain:
	$(PRESK) explain tutorial/07_train.psk --entry loss --dim B=4 --budget 1e-3 --profile $(EXPLAIN_PROFILE) --threads 10 --target native,c,pytorch --out $(EXPLAIN)

explain-pdf:
	@test -f $(EXPLAIN)/explain.md || $(MAKE) explain
	cd $(EXPLAIN) && PATH="$(TEXLIVE):$$PATH" pandoc explain.md -o explain.pdf --pdf-engine=lualatex \
	  -H $(CURDIR)/crates/presk-explain/pandoc/header.tex -V mainfont="DejaVu Serif" -V sansfont="DejaVu Sans" \
	  -V monofont="DejaVu Sans Mono" -V geometry:margin=2cm -V fontsize=10pt -V colorlinks=true --toc --toc-depth=2
	@echo "wrote $(EXPLAIN)/explain.pdf"

# The printer of the denotation (plan 0.13.0 step 4, D137): the typed program
# and PSK-Core after differentiation of the tutorial, the examples and the
# course GPT, as LaTeX documents that compile alone, each compiled by LuaLaTeX
# into out/math/. Nothing timed, any machine with TeX Live.
MATH_FILES = tutorial/07_train.psk tutorial/03_attention.psk tutorial/04_grad.psk \
             tests/examples/mlp/mlp.psk tests/examples/layernorm/layernorm.psk tests/examples/minigpt/minigpt.psk \
             corpus/nlp/05_gpt.psk
math-check: build
	@mkdir -p out/math
	@for f in $(MATH_FILES); do n=$$(basename $$f .psk); \
	  $(PRESK) dump $$f --stage elab --format latex --standalone > out/math/$$n.tex 2> /dev/null || exit 1; \
	  $(PRESK) dump $$f --stage core-after-ad --format latex --standalone > out/math/$$n-core.tex 2> /dev/null || exit 1; \
	done
	@cd out/math && for t in *.tex; do PATH="$(TEXLIVE):$$PATH" lualatex -interaction=nonstopmode -halt-on-error $$t > /dev/null \
	  || { echo "math-check: $$t does not compile (out/math/$${t%.tex}.log)"; exit 1; }; done
	@echo "math-check: $$(ls out/math/*.pdf | wc -l) documents compile, in out/math/"

explain-manual:
	PRESK_UPDATE_MANUAL=1 $(CARGO) test -q -p presk --test explain the_manual_holds_the_tutorial_s_document

# The course GPT's network diagram (plan 0.14.0 step 3, D143), one layout:
# level 2 into the manual's chapter explain_network, its SVG, which a test of
# presk-explain holds current; level 1 and Block alone, the report's two
# figures (the user's decision of 2026-10-17: each readable on paper), as TikZ
# in presk-specs, each with the command that made it in its first lines.
NETWORK_GPT = corpus/nlp/05_gpt.psk --stage network --entry loss --const D=64 --const H=2 --const L=2 --dim B=2 --dim T=16
network-figure: build
	PRESK_UPDATE_MANUAL=1 $(CARGO) test -q -p presk-explain --test network the_manual_holds_the_course_gpt_s_network
	@mkdir -p out/network
	$(PRESK) dump $(NETWORK_GPT) --level 2 --format svg > out/network/gpt-level2.svg
	$(PRESK) dump $(NETWORK_GPT) --level 1 --format svg > out/network/gpt-level1.svg
	$(PRESK) dump $(NETWORK_GPT) --box blocks --format svg > out/network/gpt-block.svg
	$(PRESK) dump $(NETWORK_GPT) --level 1 --format tikz > out/network/gpt-level1.tex
	$(PRESK) dump $(NETWORK_GPT) --box blocks --format tikz > out/network/gpt-block.tex
	$(PRESK) dump $(NETWORK_GPT) --level 1 --format tikz --standalone > out/network/gpt-level1-standalone.tex
	$(PRESK) dump $(NETWORK_GPT) --box blocks --format tikz --standalone > out/network/gpt-block-standalone.tex
	@mkdir -p $(PRESK_SPECS_DIR)/report/figures
	cp out/network/gpt-level1.tex $(PRESK_SPECS_DIR)/report/figures/network-gpt.tex
	cp out/network/gpt-block.tex $(PRESK_SPECS_DIR)/report/figures/network-gpt-block.tex

clean:
	$(CARGO) clean
