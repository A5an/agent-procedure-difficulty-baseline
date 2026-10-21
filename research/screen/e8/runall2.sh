#!/bin/zsh
# Restart of runall.sh after the 429 problem on the global endpoint (2.5 models now go to regional endpoints).
cd ~/zangir-research/prototype/screen/e8
unset GEMINI_API_KEY
export GOOGLE_CLOUD_PROJECT=$(gcloud config get-value project 2>/dev/null); export LLM_BACKEND=vertex
PY=../ideas2/harness/.venv/bin/python
for a in flash pro "pro>flash" "flash>pro"; do
  $PY run_e8.py --arm "$a" --workers 30 2>&1 | grep --line-buffered -v "^Active"
done
$PY run_e8.py --arm pro --max_tokens 512 --carry_only --tag e80 --workers 3 2>&1 | grep --line-buffered -v "^Active"
echo ALL_DONE
