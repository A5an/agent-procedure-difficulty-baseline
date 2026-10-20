import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, numpy as np, sys
sys.path.insert(0,ZR.SCREEN + "/ideas2/emulate")
from scipy.stats import spearmanr
import gen, gt
rows=[r for r in json.load(open(ZR.SCREEN + "/ideas2/emulate/validation_rows_gen_v3.json")) if r.get('tool_runs') and r.get('code_label') is not None]
real={}; n={}
for d,p in gen.SEL:
    cs=json.load(open(f"{gt.SOP}/data/{d}_tasks.json"))[p]; real[p]=np.mean([c['action_should_succeed'] for c in cs]); n[p]=len(cs)
syn={p:np.mean([r['code_label'] for r in rows if r['proc']==p]) for _,p in gen.SEL}
x=[real[p] for _,p in gen.SEL]; y=[syn[p] for _,p in gen.SEL]
print(dict(zip([p for _,p in gen.SEL],zip(np.round(x,2),np.round(y,2)))))
print('spearman perform share synth vs real over 10 procedures', spearmanr(x,y))
print('real perform share (10 procs, all variants)', sum(real[p]*n[p] for p in n)/sum(n.values()), 'synthetic', np.mean([r['code_label'] for r in rows]), len(rows))
