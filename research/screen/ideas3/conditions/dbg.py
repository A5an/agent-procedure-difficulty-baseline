import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
sys.path.insert(0, ZR.SCREEN); sys.path.insert(0, ZR.SCREEN + "/ideas")
import harness as H
import run_baseline as RB
import cm, pandas as pd, numpy as np
from scipy.stats import spearmanr
tk = pd.read_csv(f"{cm.REPO}/data/sopbench/tasks.csv")
tr, te = RB.splits(tk, "new_domain", 0)[0]
import time; t=time.time()
m = cm.CondModel(tr, fam_models=True); print("fit", time.time()-t, m.models["perf"][4], m.models["fail"][4])
for tn, kw in [("l0", {}), ("l2", {}), ("l2t", dict(truth=True))]:
    z = [m.case_score(c, tn[:2], truth=kw.get("truth", False)) for c in te]
    b = tk.set_index("case_id").loc[te]
    print(tn, np.mean(z), spearmanr(z, b.success_rate_28)[0])
