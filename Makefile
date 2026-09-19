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
#   make experiment       exact versus almost: the realization gap of float32/float64 training (manual, `experiments`)
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

.PHONY: build test check-examples run-examples export-examples venv conformance tutorial train-examples experiment bench opt-examples calibrate sync-examples snap doc clean

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
	@for d in out/*/; do echo "== $$d"; (cd $$d && ../../.venv/bin/python -m pytest -q test_conformance.py) || exit 1; done

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

# The two experiments of the manual's `experiments` chapter (needs `make venv`).
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
