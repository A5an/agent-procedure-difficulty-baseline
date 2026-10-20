export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
PY=~/zangir-research/repo/agent-procedure-difficulty-baseline/.venv/bin/python
C=~/zangir-research/prototype/screen/clean
R=~/zangir-research/repo/agent-procedure-difficulty-baseline/data/
cd $C
for v in old A B; do
  if [ $v = old ]; then D=$R; else D=$C/data_$v/; fi
  echo "== $v"; $PY s05_a4.py $D a4_$v.json
done
