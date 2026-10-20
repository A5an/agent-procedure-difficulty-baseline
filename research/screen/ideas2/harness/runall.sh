#!/bin/zsh
cd ~/zangir-research/prototype/screen/ideas2/harness
export GOOGLE_CLOUD_PROJECT=$(gcloud config get-value project 2>/dev/null); export LLM_BACKEND=vertex
for m in gemini-2.5-flash gemini-3.8-flash; do
  for a in H0 H1 H2; do
    .venv/bin/python run.py --model $m --arm $a --workers 8 2>&1 | grep -v "^Active"
  done
done
