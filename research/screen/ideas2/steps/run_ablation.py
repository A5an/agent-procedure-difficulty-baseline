"""Exploratory ablations of typ_logit (secondary, not part of the fixed primary set)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
sys.path.insert(0, ZR.SCREEN)
sys.path.insert(0, ZR.SCREEN + "/ideas")
sys.path.insert(0, ZR.SCREEN + "/ideas2/steps")
import numpy as np, pandas as pd
import harness as H
from cpulock import cpu_lock
import steppred as S

OUT = ZR.SCREEN + "/ideas2/steps/out_abl"
T = S.TYPES
# structure-only features: number of plain leaves, number of OR groups, the action
st = S._steps
leaf = st[~st.step_type.isin(["or_group", "action"])].groupby("case_id").size().reindex(S.FEAT.index).fillna(0)
STRUCT = pd.DataFrame({"leaf": leaf, "or_group": S.FEAT["or_group"], "action": S.FEAT["action"]})
# shuffled type labels: permute the type of the non-action steps across all steps (keeps per case step counts)
rng = np.random.default_rng(0)
x = st[st.step_type != "action"].copy()
x["step_type"] = rng.permutation(x.step_type.values)
SHUF = x.groupby(["case_id", "step_type"]).size().unstack(fill_value=0).reindex(S.FEAT.index).fillna(0).reindex(columns=T, fill_value=0)
SHUF["action"] = 1.0


def make_preds(src, M):
    d = {"typ_logit": S.LogitSteps("typ_logit", T, False),
         "struct_logit": S.LogitSteps("struct_logit", ["leaf", "or_group", "action"], False, feat=STRUCT),
         "shuf_logit": S.LogitSteps("shuf_logit", T, False, feat=SHUF)}
    for t in T:
        if t != "action":
            d[f"drop_{t}"] = S.LogitSteps(f"drop_{t}", [u for u in T if u != t], False)
    for t in T:
        if t != "action":
            d[f"only_{t}"] = S.LogitSteps(f"only_{t}", [t, "action"], False)
    return d


if __name__ == "__main__":
    with cpu_lock("steps_abl"):
        H.run_all(OUT, make_preds, benches=["sopbench"], schemes=["new_procedures", "new_domain"])
        H.evaluate_all(OUT, {k: k for k in make_preds(None, None)}, combos=[])
