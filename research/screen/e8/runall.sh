#!/bin/zsh
# E8-0 then the seven E8 arms, one after another (PREREG.md). Budget is enforced by the ledger cap in run_e8.py.
cd ~/zangir-research/prototype/screen/e8
unset GEMINI_API_KEY
export GOOGLE_CLOUD_PROJECT=$(gcloud config get-value project 2>/dev/null); export LLM_BACKEND=vertex
PY=../ideas2/harness/.venv/bin/python
$PY run_e8.py --arm pro --max_tokens 512 --carry_only --tag e80 --workers 8 2>&1 | grep -v "^Active"
for a in flash pro "pro>flash" "flash>pro" lite g38 "lite>g38"; do
  $PY run_e8.py --arm "$a" --workers 8 2>&1 | grep -v "^Active"
done
echo ALL_DONE
