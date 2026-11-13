#!/bin/zsh
cd ${0:a:h}
until grep -q "e2d finished" e2d_run.log 2>/dev/null; do sleep 60; done
PY=${TAU_PY:?set TAU_PY to the tau2 python}
for ag in haiku sonnet; do $PY run_claude_tib.py --agent $ag --set new --workers 4 > e3b_${ag}.log 2>&1 & done; wait
echo "e3b new finished $(date)"
for ag in haiku sonnet; do $PY run_claude_tib.py --agent $ag --set newseed --workers 4 > e3b_seed_${ag}.log 2>&1 & done; wait
echo "e3b newseed finished $(date)"
