# Convenience targets for the ACG instrument.
PY := ./.venv/bin/python
PIP := ./.venv/bin/pip

.PHONY: help venv serve serve-sglang stop logs smoke single experiment determinism analyze test clean complex

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv: ## create venv and install client deps
	python3 -m venv .venv && $(PIP) install -q -r requirements.txt

serve: ## start vLLM on the MIG slice
	bash docker/serve_vllm.sh

serve-sglang: ## start SGLang on the MIG slice (alternative engine)
	bash docker/serve_sglang.sh

stop: ## stop the model server
	-docker rm -f acg-vllm acg-sglang

logs: ## follow vLLM logs
	docker logs -f acg-vllm

smoke: ## validate determinism + tool-calling against the live server
	$(PY) scripts/smoke_test.py

single: ## run one task end-to-end and draw its ACG (TASK=T02)
	$(PY) scripts/run_single.py --task $(or $(TASK),T02)

experiment: ## run the multi-QA variance study (REPS=8)
	$(PY) scripts/run_experiment.py --tasks all --reps $(or $(REPS),8) --temperature 0.7 --vary-seed

complex: ## run 3–4 hop tasks + draw readable ACG graphs (REPS=8)
	$(PY) scripts/run_complex.py --reps $(or $(REPS),8)

determinism: ## decompose variance: sampling vs serving-batch noise (TASK=T06)
	$(PY) scripts/determinism_check.py --task $(or $(TASK),T06) --reps $(or $(REPS),12)

analyze: ## re-analyze an existing trace file
	$(PY) scripts/analyze.py --trace traces/experiment.jsonl

test: ## run the test suite (live ACG tests skip if server down)
	$(PY) -m pytest tests/ -v -s

clean: ## remove generated traces/figures (keeps data/ and code)
	rm -f traces/*.jsonl traces/*.csv traces/*.json traces/_*.log
	rm -f traces/figures/*.png
