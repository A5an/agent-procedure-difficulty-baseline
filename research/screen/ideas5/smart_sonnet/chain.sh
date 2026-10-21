#!/bin/bash
cd "$(dirname "$0")"
export PYTHONHASHSEED=0
python3 run.py --set real --workers 4 >> run_real.log 2>&1
python3 run.py --set synth --workers 4 >> run_synth.log 2>&1
