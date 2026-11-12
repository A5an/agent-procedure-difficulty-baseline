#!/bin/zsh
cd ${0:a:h}
PY=${TAU_PY:-python3}   # python of the TauIndianBankBench venv (tau2 installed)
for ag in haiku sonnet; do $PY run_claude_tib.py --agent $ag --set k2 --sim sonnet --workers 4 > e2c_${ag}.log 2>&1 & done; wait
echo "e2c finished $(date)"
