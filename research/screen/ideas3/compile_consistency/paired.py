import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, numpy as np, pandas as pd
sys.path.insert(0, ZR.SCREEN); sys.path.insert(0, ZR.SCREEN + "/ideas")
import os; os.environ["PYTHONHASHSEED"]="0"
import metrics_extra as X
from pathlib import Path
H=Path(ZR.SCREEN + "/ideas3/compile_consistency/out/run"); RUB=ZR.SCREEN + "/rubric/out/run"
t=X.tasks_of("sopbench"); rng=np.random.default_rng(0); res=[]
for scheme in ["new_procedures","new_domain"]:
    m,cols=X.load(H,"sopbench",scheme,["adele","lad2_adele_cons","cons_cat_adele","lad2_adele"],{"lad2_adele":RUB})
    d={c:X.difficulty(m,c).merge(t[["case_id","domain","procedure_id","b"]],on="case_id") for c in cols}
    base=d["lad2_adele"]; d0=d["lad2_adele_cons"]; 
    for other,name in [("lad2_adele","lad2_adele_cons vs lad2_adele"),("adele","lad2_adele_cons vs adele")]:
        a=d["lad2_adele_cons"][d["lad2_adele_cons"].seed==0].reset_index(drop=True); o=d[other][d[other].seed==0].reset_index(drop=True)
        o=o.set_index("case_id").loc[a.case_id].reset_index(); a["l2"]=o["l"].values
        # case-level: bootstrap over cases, rho within (fold,domain)
        diffs=[]
        for _ in range(300):
            s=a.iloc[rng.integers(0,len(a),len(a))]
            diffs.append(X._rho_cells(s,["fold","domain"],"l","b")-X._rho_cells(s,["fold","domain"],"l2","b"))
        pt=X._rho_cells(a,["fold","domain"],"l","b")-X._rho_cells(a,["fold","domain"],"l2","b")
        out=f"{scheme} case-level {name}: {pt:+.3f} [{np.percentile(diffs,2.5):+.3f},{np.percentile(diffs,97.5):+.3f}] (seed 0, 300 case bootstraps)"
        if scheme=="new_domain":
            pa=a.groupby(["domain","procedure_id"])[["l","l2","b"]].mean().reset_index()
            ps=[]
            procs=pa.procedure_id.unique()
            for _ in range(500):
                pick=rng.choice(procs,len(procs)); s=pd.concat([pa[pa.procedure_id==p] for p in pick])
                ps.append(X._rho_cells(s,["domain"],"l","b")-X._rho_cells(s,["domain"],"l2","b"))
            pt=X._rho_cells(pa,["domain"],"l","b")-X._rho_cells(pa,["domain"],"l2","b")
            out+=f"\n   procedure-level {name}: {pt:+.3f} [{np.percentile(ps,2.5):+.3f},{np.percentile(ps,97.5):+.3f}]"
        print(out)
