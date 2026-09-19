# Presk compiler — thin orchestration over cargo (cargo does the work; never the reverse).
#   make build            debug build of the workspace
#   make test             all tests (unit, snapshot, property)
#   make check-examples   `presk check` on every example program
#   make run-examples     the reference `presk run` commands of the plan (§4)
#   make export-examples  export the example files to out/ (with conformance data)
#   make venv             create .venv with PyTorch (uv) for the conformance tests
#   make conformance      export-examples, then run the generated PyTorch tests in .venv
#   make tutorial         every command of the manual's tutorial chapter (tutorial/*.psk)
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

.PHONY: build test check-examples run-examples export-examples venv conformance tutorial sync-examples snap doc clean

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

venv:
	uv venv .venv --python 3.12
	uv pip install --python .venv/bin/python torch numpy pytest

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
