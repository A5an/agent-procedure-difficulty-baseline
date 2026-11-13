#!/bin/zsh
cd ${0:a:h}
PY=${TAU_PY:?set TAU_PY to the tau2 python}
for ag in haiku sonnet; do $PY run_claude_tib.py --agent $ag --set real --sim sonnet --workers 4 > e2d_${ag}.log 2>&1 & done; wait
echo "e2d finished $(date)"
