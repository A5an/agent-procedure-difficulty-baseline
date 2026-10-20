import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys,json,os
sys.path.insert(0,ZR.SCREEN + "/ideas6/msg_facts")
sys.path.insert(0,ZR.SCREEN + "/ideas")
import sim,sop,pandas as pd
from multiprocessing import Pool
from cpulock import cpu_lock
H=ZR.SCREEN + "/ideas6/msg_facts"
C=ZR.SCREEN + "/ideas2/compile"
CASES={c["case_id"]:c for c in sop.load_cases()}
R={r["case_id"]:r for r in json.load(open(f"{C}/results_A.json"))}
P=json.load(open(f"{C}/programs_t0.json")); FN=sys.argv[1] if len(sys.argv)>1 else "facts.json"; ON=sys.argv[2] if len(sys.argv)>2 else "msg_features.csv"
F=json.load(open(f"{H}/{FN}"))
def f(k): return sim.case_features(CASES[k],P[R[k]["unit"]]["code"],F[k])
if __name__=="__main__":
    ids=[k for k in CASES if F.get(k)]
    with cpu_lock("msg_facts"):
        with Pool(3) as p: rows=p.map(f,ids,chunksize=10)
    df=pd.DataFrame(rows); df.to_csv(f"{H}/{ON}",index=False); print(df.describe().T)
