"""Variants fixed before the first run (see report):
 b_clad       reference c_lad = ladder on ADeLe + PLAN(gemini mean3) + CMP (rebuilt here)
 v1_int       c_lad with PLAN replaced by intersection-plan features (same 9 columns)
 v2_son       c_lad + Sonnet plan features (9 extra columns)
 v3_dis       c_lad + disagreement features (10 extra columns)
 x_avg        c_lad with PLAN replaced by mean of Gemini and Sonnet plan features (extra, fixed in advance as validation of the mean idea)
"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas"); sys.path.insert(0, S + "/rubric")
from pathlib import Path
import pandas as pd
import harness as H, metrics_extra as X
from cpulock import cpu_lock
from ladder import Ladder
import prompts as P
I3 = Path(S + "/ideas3"); HERE = Path(__file__).parent; REPO = H.REPO; FEAT = HERE / "features"
def tab(path):
    d = pd.read_csv(path); return d.set_index(d.columns[0])
TYPES = [f"n_{t}" for t in P.STEP_TYPES]; GLOB = ["max_depth", "refusal_conditions", "needs_hidden_data", "p_refusal"]
def make_preds(src, M, bench):
    sop = bench == "sopbench"
    ad = tab(REPO / "data" / bench / "features" / "adele.csv")
    rub = tab(S + f"/rubric/features/rub_proc_{bench}.csv")
    plan = tab(I3 / f"plan_tools/features/plan3_{bench}.csv")
    cmp_ = [tab(I3 / "compile_consistency/cons_sopbench.csv")[["len_mean", "branch_mean"]]] if sop else []
    son, inter, dis, avg = [tab(FEAT / f"{n}_{bench}.csv") for n in ("son", "int", "dis", "avg")]
    def L(name, parts): return Ladder(2, TYPES, GLOB, rub, extra_table=pd.concat(parts, axis=1), name=name)
    return {"b_clad": L("b_clad", [ad, plan] + cmp_),
            "v1_int": L("v1_int", [ad, inter] + cmp_),
            "v2_son": L("v2_son", [ad, plan] + cmp_ + [son.add_prefix("son_")]),
            "v3_dis": L("v3_dis", [ad, plan] + cmp_ + [dis]),
            "x_avg": L("x_avg", [ad, avg] + cmp_)}
V = ["b_clad", "v1_int", "v2_son", "v3_dis", "x_avg"]
if __name__ == "__main__":
    out = HERE / "out"
    with cpu_lock("planners"):
        H.run_all(out, make_preds, benches=["sopbench", "tau2"], schemes=["new_procedures", "new_domain"])
        H.evaluate_all(out, {k: "planners round 6" for k in V})
        X.report(out, V)
    print("run done")
