"""Post hoc (exploratory) ablation of c_lad components. usage: run.py"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas"); sys.path.insert(0, S + "/rubric")
from pathlib import Path
import pandas as pd
import harness as H
from cpulock import cpu_lock
from ladder import Ladder
import prompts as P
I3 = Path(S + "/ideas3"); HERE = Path(S + "/ideas4/ablation"); REPO = H.REPO
def tab(path):
    d = pd.read_csv(path); return d.set_index(d.columns[0])
TYPES = [f"n_{t}" for t in P.STEP_TYPES]; GLOB = ["max_depth", "refusal_conditions", "needs_hidden_data", "p_refusal"]
def make_preds(src, M, bench):
    sop = bench == "sopbench"
    ad = tab(REPO / "data" / bench / "features" / "adele.csv")
    rub = tab(S + f"/rubric/features/rub_proc_{bench}.csv")
    plan = tab(I3 / f"plan_tools/features/plan3_{bench}.csv")
    bt = tab(S + f"/ideas/pairwise/features/bt_{bench}.csv")[["bt"]]
    cmp_ = tab(I3 / "compile_consistency/cons_sopbench.csv")[["len_mean", "branch_mean"]] if sop else None
    def L(name, parts): return Ladder(2, TYPES, GLOB, rub, extra_table=pd.concat(parts, axis=1), name=name)
    return {"a1_adele": L("a1_adele", [ad]),
            "a2_adele_plan": L("a2_adele_plan", [ad, plan]),
            "a3_adele_cmp": L("a3_adele_cmp", [ad] + ([cmp_] if sop else [])),
            "a4_full": L("a4_full", [ad, plan] + ([cmp_] if sop else [])),
            "p5_plan_noadele": L("p5_plan_noadele", [plan]),
            "p6_full_bt": L("p6_full_bt", [ad, plan] + ([cmp_] if sop else []) + [bt])}
if __name__ == "__main__":
    out = HERE / "out"
    with cpu_lock("ablation"):
        H.run_all(out, make_preds, benches=["sopbench", "tau2"], schemes=["new_procedures", "new_domain"])
    print("run done")
