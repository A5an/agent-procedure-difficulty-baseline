"""Descriptive full-data fits (not predictions): type weights of typ_logit and composition diagnostics."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, numpy as np, pandas as pd
from types import SimpleNamespace
import steppred as S
REPO = ZR.REPO + "/data/sopbench"
resp = {json.loads(l)["subject_id"]: json.loads(l)["responses"] for l in open(f"{REPO}/responses.jsonl")}
data = SimpleNamespace(train_abilities=pd.DataFrame(index=list(resp)), responses=resp)
tr = list(S.FEAT.index)
m = S.LogitSteps("typ", S.TYPES, False); m.fit(data, tr)
tab = pd.DataFrame({"type": S.TYPES, "d_logit_per_step": m.d, "mean_count": S.FEAT.mean().values})
print(tab.round(3).to_string(index=False))
tab.to_csv("typ_logit_full_fit.csv", index=False)
# by label strata
t = pd.read_csv(f"{REPO}/tasks.csv").set_index("case_id")
for lab in (0, 1):
    ids = [c for c in tr if t.loc[c, "should_succeed"] == lab]
    mm = S.LogitSteps("typ", S.TYPES, False); mm.fit(data, ids)
    print("should_succeed", lab, dict(zip(S.TYPES, np.round(mm.d, 2))))
print(S.FEAT.describe().loc[["mean", "std", "min", "max"]].round(2).to_string())
print("n steps by label", S.N_STEPS.groupby(t.loc[tr, "should_succeed"].values).mean().round(2).to_dict())
b = pd.read_csv(f"{REPO}/irt/1d_1pl/items.csv", index_col=0).b
print("corr(n steps, difficulty):", np.corrcoef(S.N_STEPS.loc[tr], b.loc[tr])[0, 1].round(3))
