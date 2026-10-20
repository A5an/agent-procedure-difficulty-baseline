#!/bin/zsh
cd ~/zangir-research/prototype/screen/e8
unset GEMINI_API_KEY
export GOOGLE_CLOUD_PROJECT=$(gcloud config get-value project 2>/dev/null); export LLM_BACKEND=vertex
PY=../ideas2/harness/.venv/bin/python
$PY run_e8.py --arm "$1" --limit ${2:-2} --tag test --workers 2 2>&1 | grep -v "^Active"
