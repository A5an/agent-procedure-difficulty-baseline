import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, pandas as pd, numpy as np
from astf import feats, COLS
S = ZR.SCREEN + "/"
files = [S+"ideas2/compile/programs_t0.json", S+"ideas2/compile/programs_t1_s1.json", S+"ideas2/compile/programs_t1_s2.json",
         S+"ideas3/compile_consistency/programs_t1_s3.json", S+"ideas3/compile_consistency/programs_t1_s4.json"]
acc = {}; bad = 0
for fn in files:
    for u, d in json.load(open(fn)).items():
        code = d.get("code") if isinstance(d, dict) else d
        if not code: continue
        try: f = feats(code, u.split("/")[0])
        except Exception: bad += 1; continue
        acc.setdefault(u, []).append(f)
uf = pd.DataFrame({u: pd.DataFrame(v).mean().to_dict() for u, v in acc.items()}).T
res = json.load(open(S+"ideas2/compile/results_A.json"))
cs = pd.DataFrame([{"case_id": r["case_id"], "unit": r["unit"]} for r in res]).set_index("case_id")
cf = cs.join(uf, on="unit"); print("units", len(uf), "parse fails", bad, "cases missing", int(cf[COLS].isna().any(axis=1).sum()))
for c in COLS: cf[c] = cf[c].fillna(cf[c].median())
cf = cf.drop(columns="unit").rename(columns=lambda c: c.replace("cp_", "cmpp_"))
cf.reset_index().to_csv("cmpplus_sopbench.csv", index=False); print(cf.describe().T[["mean","std","min","max"]])
