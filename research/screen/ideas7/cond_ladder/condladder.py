"""Ladder subclass that fits the written-condition skip model (COND) inside every fold. Copy of nothing in shared code:
only overrides fit() and _X() of rubric/ladder.Ladder."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
S = ZR.SCREEN
sys.path.insert(0, S + "/rubric"); sys.path.insert(0, S + "/ideas3/conditions")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from ladder import Ladder
import cm
PROC = pd.read_csv(ZR.REPO + "/data/sopbench/tasks.csv").set_index("case_id").procedure_id

def summ(model, cases):
    """per case: [max skip_perf, mean skip_perf, log prod(1-skip_perf)] over its conditions"""
    sp = model.sk("perf"); out = []
    for c in cases:
        rows = cm.UNIT_ROWS.get(cm.C2U[c], np.array([], dtype=int))
        if len(rows) == 0: out.append([0.0, 0.0, 0.0]); continue
        s = sp[rows]; out.append([s.max(), s.mean(), float(np.log(1 - s).sum())])
    return np.array(out)

def score(model, cases):
    return cm.Pred("l0", "l0")._z(model, cases)[:, None]

class CondLadder(Ladder):
    def __init__(self, variant, *a, **k):
        super().__init__(*a, **k); self.variant = variant; self.dyn = {}
        self.fn = score if variant == "score" else summ
    def _X(self, ids):
        n, g = super()._X(ids)
        d = np.array([self.dyn[i] for i in ids], float)
        return n, np.hstack([g, d])
    def fit(self, data, train_task_ids):
        tr = list(train_task_ids); te = list(data.test_tasks)
        ptr = PROC.loc[tr].to_numpy()
        assert not (set(PROC.loc[te]) & set(ptr)), "test procedure inside training set"
        z = None; rows = []
        for a_, b_ in GroupKFold(5).split(tr, groups=ptr):
            ia, ib = [tr[i] for i in a_], [tr[i] for i in b_]
            assert not (set(PROC.loc[ia]) & set(PROC.loc[ib])), "inner leak"
            m = cm.CondModel(ia); v = self.fn(m, ib)
            if z is None: z = np.zeros((len(tr), v.shape[1]))
            z[b_] = v
        full = cm.CondModel(tr); self.full_scores = (te, self.fn(full, te))
        self.dyn = dict(zip(tr, z)); self.dyn.update(dict(zip(te, self.full_scores[1])))
        super().fit(data, tr)
