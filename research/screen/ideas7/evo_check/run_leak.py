"""Check 1: leak check for F1. usage: run_leak.py [debug]. Columns f1_same, f1_nomdc, f1_refit (see CHECKS.md)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, os
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas"); sys.path.insert(0, S + "/rubric"); sys.path.insert(0, S + "/ideas7/entropy_combo"); sys.path.insert(0, S + "/ideas3/conditions")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
import harness as H
from cpulock import cpu_lock
from ladder import Ladder
import prompts as P
import cm as CM
I3 = Path(S + "/ideas3"); EVO = Path(S + "/ideas7/evo"); HERE = Path(S + "/ideas7/evo_check"); REPO = H.REPO
def tab(path):
    d = pd.read_csv(path); return d.set_index(d.columns[0])
TYPES = [f"n_{t}" for t in P.STEP_TYPES]; GLOB = ["max_depth", "refusal_conditions", "needs_hidden_data", "p_refusal"]
MDC = "mdc"
TASKS = {b: pd.read_csv(REPO / "data" / b / "tasks.csv").set_index("case_id") for b in ["sopbench", "tau2"]}
UNITS = CM.UNITS; C2U = CM.C2U; cd = CM.cd
PRED = CM.Pred("l0", "l0")

def mdc_of(cm_, cid):
    u = C2U[cid]; g = cd[cd.unit == u]
    best = 0.0; n = len(g)
    for idx, (ridx, st) in enumerate(zip(g.index, g.step_type)):
        pskip = max(float(cm_.pred["perf"][ridx]), float(cm_.pred["fail"][ridx]))
        st = str(st).lower()
        tw = 1.4 if st in ("relation", "datetime") else (1.15 if st == "existence" else 1.0)
        pw = 1.0 + (idx / float(n)) * 0.35 if n else 1.0
        best = max(best, pskip * tw * pw)
    return best

def evo_raw(bench):
    t = tab(EVO / "feat_f1.csv"); return t.loc[TASKS[bench].index]
def centre(df, bench):
    return df - df.groupby(TASKS[bench].domain.loc[df.index]).transform("mean")

class LadderRefit(Ladder):
    """sopbench only: most_diff_check recomputed per fold from a skip model fitted on the training procedures of the fold."""
    def __init__(self, base_extra, raw_other, *a, **k):
        super().__init__(*a, **k); self.base_extra = base_extra; self.raw_other = raw_other
    def fit(self, data, train):
        tr = [c for c in train if c in TASKS["sopbench"].index]; te = [c for c in data.test_tasks if c in TASKS["sopbench"].index]
        grp = TASKS["sopbench"].loc[tr, "procedure_id"].values
        vals = {}
        for a_, b_ in GroupKFold(5).split(tr, groups=grp):
            cmi = CM.CondModel([tr[i] for i in a_])
            for i in b_: vals[tr[i]] = mdc_of(cmi, tr[i])
        cmf = CM.CondModel(tr)
        for c in te: vals[c] = mdc_of(cmf, c)
        s = pd.Series(vals); dom = TASKS["sopbench"].domain.loc[s.index]
        s = s - s.groupby(dom).transform("mean")
        ex = self.base_extra.copy(); ex["evo1_" + MDC] = 0.0
        ex.loc[s.index, "evo1_" + MDC] = s.values
        self.extra = ex
        super().fit(data, train)

def make_preds(src, Mm, bench):
    sop = bench == "sopbench"
    ad = tab(REPO / "data" / bench / "features" / "adele.csv")
    rub = tab(S + f"/rubric/features/rub_proc_{bench}.csv")
    plan = tab(I3 / f"plan_tools/features/plan3_{bench}.csv")
    cmp_ = tab(I3 / "compile_consistency/cons_sopbench.csv")[["len_mean", "branch_mean"]] if sop else None
    raw = evo_raw(bench)
    def ex(ev): return pd.concat([ad, plan] + ([cmp_] if sop else []) + [ev], axis=1)
    def pref(ev): ev = ev.copy(); ev.columns = [f"evo1_{c}" for c in ev.columns]; return ev
    same = pref(centre(raw, bench)); nomdc = pref(centre(raw.drop(columns=["most_diff_check"]), bench))
    out = {"f1_same": Ladder(2, TYPES, GLOB, rub, extra_table=ex(same), name="f1_same"),
           "f1_nomdc": Ladder(2, TYPES, GLOB, rub, extra_table=ex(nomdc), name="f1_nomdc")}
    if sop:
        out["f1_refit"] = LadderRefit(ex(nomdc), None, 2, TYPES, GLOB, rub, extra_table=ex(nomdc), name="f1_refit")
    else:
        out["f1_refit"] = Ladder(2, TYPES, GLOB, rub, extra_table=ex(same), name="f1_refit")
    return out

if __name__ == "__main__":
    dbg = len(sys.argv) > 1
    if dbg:  # sanity: reproduce stored most_diff_check from leave-domain-out skip models
        raw = tab(EVO / "feat_f1.csv")
        for d in ["bank", "hotel"]:
            cmi = CM.CondModel([c for c in TASKS["sopbench"].index if TASKS["sopbench"].loc[c, "domain"] in ["bank", "dmv", "healthcare", "hotel"] and TASKS["sopbench"].loc[c, "domain"] != d])
            ids = [c for c in TASKS["sopbench"].index if TASKS["sopbench"].loc[c, "domain"] == d]
            mine = np.array([mdc_of(cmi, c) for c in ids]); st = raw.loc[ids, "most_diff_check"].values
            print(d, "max abs diff vs stored", float(np.abs(mine - st).max()), "corr", float(np.corrcoef(mine, st)[0, 1]))
        sys.exit()
    out = HERE / "out_leak"
    with cpu_lock("evo_check_leak"):
        H.run_all(out, make_preds, benches=["sopbench", "tau2"], schemes=["new_procedures", "new_domain"])
    print("run done")
