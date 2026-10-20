"""Checks (i) carry-out vs refuse agent rank correlation, (ii) between-domain agent rank correlations. Old vs A vs B."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, itertools, numpy as np, pandas as pd
from scipy.stats import spearmanr
REPO=ZR.REPO + "/data/sopbench/"
HERE=ZR.SCREEN + "/clean/"
t=pd.read_csv(REPO+"tasks.csv").set_index("case_id")
def mat(path):
    R=[json.loads(l) for l in open(path)]
    return pd.DataFrame({r["subject_id"]:pd.Series(r["responses"]) for r in R}).T
out={}
for name,p in [("old",REPO+"responses.jsonl"),("A",HERE+"data_A/sopbench/responses.jsonl"),("B",HERE+"data_B/sopbench/responses.jsonl")]:
    M=mat(p); ss=t.loc[M.columns,"should_succeed"]
    co=M.loc[:,ss==1].mean(1); rf=M.loc[:,ss==0].mean(1)
    r=spearmanr(co,rf)[0]
    D=pd.DataFrame({d:M.loc[:,t.loc[M.columns,"domain"]==d].mean(1) for d in sorted(t.domain.unique())})
    pairs=[spearmanr(D[a],D[b])[0] for a,b in itertools.combinations(D.columns,2)]
    out[name]=dict(n_agents=len(M),carry_refuse=r,mean_carry=co.mean(),mean_refuse=rf.mean(),
                   dom_mean=float(np.mean(pairs)),dom_min=float(np.min(pairs)),dom_max=float(np.max(pairs)),
                   dom_pairs={f"{a}~{b}":round(spearmanr(D[a],D[b])[0],3) for a,b in itertools.combinations(D.columns,2)})
    print(name,{k:(round(v,3) if isinstance(v,float) else v) for k,v in out[name].items() if k!="dom_pairs"})
json.dump(out,open(HERE+"checks_i_ii.json","w"),indent=1)
