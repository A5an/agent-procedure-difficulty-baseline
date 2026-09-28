# Reproduction of the baseline. `make help` lists the steps.

PY ?= .venv/bin/python
BENCHES := sopbench tau2 tauk_banking
FRESH ?=
FLAG := $(if $(FRESH),--fresh,)

.PHONY: help setup reproduce full data irt embeddings features annotate run evaluate checks table5 compare smoke

help:
	@echo "make setup       Python 3.11 environment with pinned packages (uv if present, else venv + pip)"
	@echo "make reproduce   out-of-fold runs, final evaluation, cross-benchmark test, comparison with reference_results/"
	@echo "                 (uses the inputs shipped in data/, no downloads, no API key; 15 to 20 minutes on a laptop CPU)"
	@echo "make full        everything from the raw releases: download (6 GB), build, IRT, embeddings, features,"
	@echo "                 reproduce, reliability checks, Table 5 of the original paper (several hours)"
	@echo "make annotate    ADeLe levels and human-time estimates with YOUR key (.env); FRESH=1 redoes all of them"
	@echo "make smoke       one call to the LLM, to check the key"

setup:
	@if command -v uv >/dev/null; then uv venv --python 3.11 .venv && uv pip install --python $(PY) -r requirements.txt; \
	else python3.11 -m venv .venv && $(PY) -m pip install -r requirements.txt; fi
	GIT_LFS_SKIP_SMUDGE=1 git submodule update --init  # ADeLe keeps its CSVs in Git LFS; the one file used is fetched on demand

reproduce: run evaluate compare

full: data irt embeddings features run evaluate checks table5 compare

data:
	$(PY) src/download_data.py
	$(PY) src/build_sopbench.py
	$(PY) src/build_tau.py

irt:
	$(PY) src/fit_irt.py

embeddings:
	for b in $(BENCHES); do $(PY) src/embed_features.py $$b length bge ap || exit 1; done

features:
	for b in $(BENCHES); do $(PY) src/adele_annotate.py annotate $$b --offline && $(PY) src/human_time.py $$b --offline || exit 1; done

annotate:
	for b in $(BENCHES); do $(PY) src/adele_annotate.py annotate $$b $(FLAG) && $(PY) src/human_time.py $$b $(FLAG) || exit 1; done
	$(PY) src/adele_annotate.py validate 100 $(FLAG)
	$(PY) src/adele_retest.py $(FLAG)

run:
	for b in $(BENCHES); do $(PY) src/run_baseline.py $$b || exit 1; done

evaluate:
	$(PY) src/evaluate.py
	$(PY) src/lobo.py

checks:
	$(PY) src/check_target.py
	$(PY) src/adele_annotate.py validate 100 --offline
	$(PY) src/adele_retest.py --offline

table5:
	$(PY) src/reproduce_table5.py

compare:
	$(PY) src/compare_reference.py

smoke:
	$(PY) src/llm.py
