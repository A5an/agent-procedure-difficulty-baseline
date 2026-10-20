import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys,json,time
sys.path.insert(0,ZR.SCREEN + "/ideas6/msg_facts")
import sim,sop
cases={c["case_id"]:c for c in sop.load_cases()}
R={r["case_id"]:r for r in json.load(open(ZR.SCREEN + "/ideas2/compile/results_A.json"))}
P=json.load(open(ZR.SCREEN + "/ideas2/compile/programs_t0.json"))
F=json.load(open(ZR.SCREEN + "/ideas6/msg_facts/facts.json"))
import random
ids=[k for k in F if F[k]][:] 
random.seed(1); ids=random.sample(ids,12) if len(ids)>12 else ids
for k in ids:
    t=cases[k]; t0=time.time()
    r=sim.case_features(t,P[R[k]["unit"]]["code"],F[k],ndraw=20)
    print(k,"should",t["action_should_succeed"],{a:(round(b,2) if isinstance(b,float) else b) for a,b in r.items() if a!="case_id"},round(time.time()-t0,2))
