#!/bin/zsh
# Re-run the episodes that failed on a transient gcloud token error at the end of runall4 (flash>pro, E8-0).
cd ~/zangir-research/prototype/screen/e8
unset GEMINI_API_KEY
export GOOGLE_CLOUD_PROJECT=$(gcloud config get-value project 2>/dev/null); export LLM_BACKEND=vertex
PY=../ideas2/harness/.venv/bin/python
$PY run_e8.py --arm "flash>pro" --workers 30 2>&1 | grep --line-buffered -v "^Active"
$PY run_e8.py --arm pro --max_tokens 512 --carry_only --tag e80 --workers 3 2>&1 | grep --line-buffered -v "^Active"
echo REST_DONE
