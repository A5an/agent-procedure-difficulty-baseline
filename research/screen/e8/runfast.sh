#!/bin/zsh
# Non-pro arms in parallel with runall.sh (runall skips cases that are already done).
cd ~/zangir-research/prototype/screen/e8
unset GEMINI_API_KEY
export GOOGLE_CLOUD_PROJECT=$(gcloud config get-value project 2>/dev/null); export LLM_BACKEND=vertex
PY=../ideas2/harness/.venv/bin/python
for a in "lite>g38"; do
  $PY run_e8.py --arm "$a" --workers 8 2>&1 | grep --line-buffered -v "^Active"
done
echo FAST_DONE
