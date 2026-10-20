#!/bin/zsh
cd ~/zangir-research/prototype/screen/ideas3/paper_pilot
export GOOGLE_CLOUD_PROJECT=$(gcloud config get-value project 2>/dev/null) LLM_BACKEND=vertex
P=~/zangir-research/repo/agent-procedure-difficulty-baseline/.venv/bin/python
for m in gemini-2.5-flash-lite gemini-3.8-flash gemini-2.5-flash; do
  for b in sopbench tau2; do $P collect.py $m $b; done
done
echo ALLDONE
