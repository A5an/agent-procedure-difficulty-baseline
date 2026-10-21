cd ~/zangir-research/prototype/screen/ideas7/cond_ladder
export PYTHONHASHSEED=0
~/zangir-research/repo/agent-procedure-difficulty-baseline/.venv/bin/python run.py > run.log 2>&1 && ~/zangir-research/repo/agent-procedure-difficulty-baseline/.venv/bin/python eval.py > eval.log 2>&1 && ~/zangir-research/repo/agent-procedure-difficulty-baseline/.venv/bin/python score.py > score.log 2>&1
