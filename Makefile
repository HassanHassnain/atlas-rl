PYTHON ?= python

.PHONY: setup setup-train lint test check smoke datasets audit demo eval-mock clean

setup:            ## CPU-side install (envs, tests, eval pipeline)
	$(PYTHON) -m pip install -e ".[dev]"

setup-train:      ## RUNTIME-ONLY: full training stack for the 3090 box
	$(PYTHON) -m pip install -r requirements-train.txt
	$(PYTHON) -m pip install -e ".[dev]"

lint:             ## static checks
	$(PYTHON) -m ruff check .
	bash -n scripts/*.sh

test:             ## unit + contract tests (determinism, verifiability, anti-hacking)
	$(PYTHON) -m pytest -q

check: lint test  ## fast contributor check

smoke:            ## full CPU pipeline smoke test
	PYTHON="$(PYTHON)" bash scripts/00_smoke_cpu.sh

datasets:         ## build the real train/eval JSONL datasets
	PYTHON="$(PYTHON)" bash scripts/01_build_datasets.sh

audit:            ## reward-hacking audit (CPU, ~1 min)
	PYTHON="$(PYTHON)" bash scripts/22_hacking_audit.sh

eval-mock:        ## mock-backend eval + report (CPU pipeline check)
	$(PYTHON) -m atlas_rl.evaluation.run_eval --model mock:noisy_oracle:0.6 \
		--n-per-env 10 --difficulties 2 3 4 --out results/mock_noisy
	$(PYTHON) -m atlas_rl.evaluation.run_eval --model mock:oracle \
		--n-per-env 10 --difficulties 2 3 4 --out results/mock_oracle
	$(PYTHON) -m atlas_rl.evaluation.report --runs results/mock_oracle results/mock_noisy \
		--baseline results/mock_noisy --out results/mock_report

demo:             ## interactive environment demo (CPU)
	$(PYTHON) -m atlas_rl.demo

clean:
	rm -rf results data/generated .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
