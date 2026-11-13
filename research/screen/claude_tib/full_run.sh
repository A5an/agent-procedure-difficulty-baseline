#!/bin/zsh
# E1-C full run in the pre-registered order (PREREG.md): Haiku real/k2/k5/foreign, Haiku k0/k1, then the same for Sonnet.
cd ${0:a:h}
PY=${TAU_PY:-python3}   # python of the TauIndianBankBench venv (tau2 installed)
for ag in haiku sonnet; do
  for s in real k2 k5 foreign; do $PY run_claude_tib.py --agent $ag --set $s --workers 4 > full_${s}_${ag}.log 2>&1 & done; wait
  for s in k0 k1; do $PY run_claude_tib.py --agent $ag --set $s --workers 6 > full_${s}_${ag}.log 2>&1 & done; wait
done
echo "full run finished $(date)"
