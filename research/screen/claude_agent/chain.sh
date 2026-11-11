#!/bin/sh
# after real episodes and each probe file: run probe episodes (Exp 8 order)
cd "$(dirname "$0")"
PY=../../data/external/tauindianbank/code/.venv/bin/python
while pgrep -f "run_claude.py sonnet real" >/dev/null; do sleep 30; done
for k in pseudo seed doc; do
  while [ ! -f probes_$k.json ]; do sleep 30; done
  $PY run_claude.py sonnet $k old 6 > run_$k.log 2>&1
done
python3 analyze8.py > analyze8.out 2>&1
