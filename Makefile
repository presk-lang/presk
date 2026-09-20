# Presk compiler — thin orchestration over cargo (cargo does the work; never the reverse).
#   make build            debug build of the workspace
#   make test             all tests (unit, snapshot, property)
#   make check-examples   `presk check` on every example program
#   make run-examples     the reference `presk run` commands of the plan (§4)
#   make export-examples  export the example files to out/ (with conformance data)
#   make venv             create .venv with PyTorch, ONNX Runtime (uv) for the conformance tests
#   make conformance      export-examples, then run the generated PyTorch tests in .venv
#   make tutorial         every command of the manual's tutorial chapter (tutorial/*.psk)
#   make train-examples   presk train on the regression and the tiny MiniGPT, on the reference and natively
#   make serve-examples   presk serve on the tiny MiniGPT's checkpoint: greedy tokens identical on every backend; export --serve
#   make serve-pt          ...and the exported serve.py gives the same tokens (needs the venv)
#   make opt-examples     the e-graph optimizer: its report on the examples, and the same outputs with and without --opt
#   make calibrate        presk bench --calibrate: time the kernels of the tiny MiniGPT natively, fit the roofline profile (out/profile.json)
#   make corpus           the NLP-course corpus (corpus/nlp): the ladder checked, run and trained on the Code civil
#   make corpus-gpt       tokenize the Code civil and train the course GPT (reduced constants: the full size is the wall 0.4.0 has to break)
#   make experiment       exact versus almost: the realization gap of float32/float64 training (manual, `experiments`)
#   make experiment-gpt   the course GPT trained by presk and by PyTorch on the same data (manual, `experiments` §3)
#   make coverage         presk check on the MLPerf transcriptions (manual, `coverage`)
#   make sched            the fusion groups of the corpus GPT and the launches they remove
#   make bench            presk bench on the tiny MiniGPT (forward, loss, optimizer step)
#   make sync-examples    re-copy tests/examples from presk-specs and show the diff
#   make snap             review pending insta snapshots
#   make doc              build the user manual (crates/presk-manual) and open it
#   make clean

PRESK_SPECS_DIR ?= ../presk-specs
CARGO ?= cargo
PRESK = $(CARGO) run -q -p presk --

EXAMPLES = tests/examples/mlp/mlp.psk \
           tests/examples/layernorm/layernorm.psk \
           tests/examples/minigpt/minigpt.psk \
           tests/examples/minigpt/train_minigpt.psk \
           tests/examples/minigpt/serve_minigpt.psk

.PHONY: tutorial-precision curve precision build test check-examples run-examples export-examples venv conformance tutorial train-examples experiment experiment-gpt coverage bench sched opt-examples calibrate corpus corpus-gpt sync-examples snap doc clean

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

# The commands of the tutorial chapter, in order (crates/presk-manual/manual/tutorial.md).
# The PyTorch conformance tests at the end need `make venv`.
tutorial: build
	$(PRESK) check tutorial/01_linear.psk
	$(PRESK) run tutorial/01_linear.psk --entry 'Linear{3, 2}.apply' --dim batch=4 --verbose
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
	$(PRESK) dump tutorial/04_grad.psk --stage core-after-ad | grep '^fn'
	$(PRESK) export tutorial/06_tower.psk --target pytorch --mode readable --shapes=check \
	    --conformance-entry Tower.apply --dim batch=2 -o out/tutorial_tower
	$(PRESK) export tutorial/04_grad.psk --target pytorch --mode readable \
	    --conformance-entry loss_grad --dim D=3 --dim B=2 -o out/tutorial_grad
	@for d in out/tutorial_tower out/tutorial_grad; do echo "== $$d"; (cd $$d && ../../.venv/bin/python -m pytest -q test_conformance.py) || exit 1; done
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
# fragment can say. Nothing is run; the question is whether it type-checks.
coverage: build
	$(PRESK) check coverage/bert.psk --summary | head -12
	$(PRESK) check coverage/llama.psk --summary | head -12

# The three experiments of the manual's `experiments` chapter (needs `make venv`).
# The third trains the course GPT against PyTorch on the same model and data.
experiment-gpt: build
	python3 corpus/nlp/data/prepare.py
	$(PRESK) data chars corpus/nlp/data/civil.md -o corpus/nlp/data/civil_vocab.json
	$(PRESK) data encode corpus/nlp/data/civil.md --vocab corpus/nlp/data/civil_vocab.json -o corpus/nlp/data/civil_tokens.txt
	$(PRESK) data lm-windows --vocab 92 --seq 256 corpus/nlp/data/civil_tokens.txt -o corpus/nlp/data/civil_T256.psk.d
	$(PRESK) export corpus/nlp/05_gpt.psk --target pytorch --mode readable --train -o out/gpt_pt
	(cd out/gpt_pt && ../../.venv/bin/python train.py --steps 20 --log-every 20)
	$(PRESK) train corpus/nlp/05_gpt.psk --steps 2 --log-every 2 --backend native --realize f32 --ckpt out/gpt_course

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

# The Code civil as a character-level dataset, then the course GPT. `presk
# check` passes at the course's own constants (D=384, H=6, L=6, T=256); what
# trains here is reduced, because a course-scale step is the wall this version
# exists to break (corpus/nlp/EXPECTED.md).
# The magnitude envelope a precision bound is conditional on (0.5.0 step 3),
# and the property the whole analysis rests on: two disjoint sets of samples
# must give magnitudes within a small factor of each other.
# The epsilon-to-quality curve (0.5.0 step 8): one program, one parameter
# varying, every point verified by measurement against the exact reference.
# Tutorial step 15: every command of "A budget, and what it buys".
tutorial-precision: build
	$(PRESK) calibrate --precision tutorial/07_train.psk --entry loss --dim B=4 --bits 128 --out out/tut_env.json
	$(PRESK) check --precision out/tut_env.json tutorial/07_train.psk --realize f32
	$(PRESK) solve tutorial/07_train.psk --precision out/tut_env.json --budget 1e-3 --out out/tut_solved.json --sites 0
	$(PRESK) run tutorial/07_train.psk --entry loss --dim B=4 --backend native --realize @out/tut_solved.json --digits 10

curve: build
	./experiments/curve.sh out/curve.md

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
	$(PRESK) calibrate --precision corpus/nlp/05_gpt.psk --entry loss \
	    --const D=64 --const H=2 --const L=2 --dim B=2 --dim T=32 --bits 128 \
	    --out out/gpt_env.json
	# Where a budget should go, from one reverse sweep on trained weights
	# (0.5.0 step 5). Without --weights the zero-initialized head makes every
	# upstream gradient exactly zero and the ranking says nothing.
	$(PRESK) dump corpus/nlp/05_gpt.psk --stage sensitivity --entry loss \
	    --const D=64 --const H=2 --const L=2 --dim B=2 --dim T=32 \
	    --weights out/gpt_tiny/final.psk.d --sites 8
	# A budget in, a realization out (0.5.0 step 6); then run what it chose.
	$(PRESK) solve corpus/nlp/05_gpt.psk --precision out/gpt_env.json --budget 1e-3 \
	    --weights out/gpt_tiny/final.psk.d --out out/solved.json --sites 6
	$(PRESK) run corpus/nlp/05_gpt.psk --entry loss --const D=64 --const H=2 --const L=2 \
	    --dim B=2 --dim T=32 --backend native --realize @out/solved.json --digits 10

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

# The fusion pass (manual chapter `scheduling`): what shares a loop nest on
# the corpus GPT, and the launches it removes from the forward and from the
# optimizer's chain.
sched: build
	$(PRESK) dump corpus/nlp/05_gpt.psk --stage sched --const D=64 --const H=2 --const L=2 | tail -12
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

sync-examples:
	rsync -a --delete --exclude PROVENANCE.md $(PRESK_SPECS_DIR)/examples/ tests/examples/
	@sed -i '' "s/^Commit: .*/Commit: $$(git -C $(PRESK_SPECS_DIR) rev-parse --short HEAD)/" tests/examples/PROVENANCE.md
	git status --short tests/examples

snap:
	$(CARGO) insta review

doc:
	$(CARGO) doc -p presk-manual --no-deps --open

clean:
	$(CARGO) clean
