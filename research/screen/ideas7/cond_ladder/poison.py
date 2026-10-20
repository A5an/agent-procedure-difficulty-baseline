"""No test-procedure outcome enters the COND fit: overwrite OUT for test procedures, scores must be identical."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys; sys.path.insert(0, ZR.SCREEN + "/ideas7/cond_ladder"); sys.path.insert(0, ZR.SCREEN + "/ideas3/conditions")
import numpy as np, cm
from condladder import score, summ, PROC
procs = sorted(PROC.unique()); rng = np.random.default_rng(0); te_p = set(rng.choice(procs, 15, replace=False))
cases = list(cm.C2U); tr = [c for c in cases if PROC[c] not in te_p]; te = [c for c in cases if PROC[c] in te_p]
a = (score(cm.CondModel(tr), te), summ(cm.CondModel(tr), te))
m = cm.OUT.case_id.map(PROC).isin(te_p)
cm.OUT.loc[m, "skip"] = 1 - cm.OUT.loc[m, "skip"]; cm.OUT.loc[m, "should_succeed"] = 1 - cm.OUT.loc[m, "should_succeed"]
b = (score(cm.CondModel(tr), te), summ(cm.CondModel(tr), te))
print("test cases", len(te), "poisoned rows", int(m.sum()), "max abs diff", max(np.abs(a[0]-b[0]).max(), np.abs(a[1]-b[1]).max()))
# also: poisoning TRAIN rows must change the result (sanity that the check can detect)
m2 = cm.OUT.case_id.isin(tr); cm.OUT.loc[m2, "skip"] = 1 - cm.OUT.loc[m2, "skip"]
c = score(cm.CondModel(tr), te); print("train poisoned diff", np.abs(a[0]-c).max())
