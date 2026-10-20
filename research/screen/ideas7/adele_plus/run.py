"""Harness run: variants replace ADeLe in the baseline (b*) and in c_lad (c*). usage: run.py"""
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
I3 = Path(S + "/ideas3"); HERE = Path(S + "/ideas7/adele_plus"); REPO = H.REPO
def tab(path):
    d = pd.read_csv(path); return d.set_index(d.columns[0])
TYPES = [f"n_{t}" for t in P.STEP_TYPES]; GLOB = ["max_depth", "refusal_conditions", "needs_hidden_data", "p_refusal"]
VARS = ["A", "A0", "B", "C"]
def make_preds(src, M, bench):
    sop = bench == "sopbench"
    rub = tab(S + f"/rubric/features/rub_proc_{bench}.csv")
    plan = tab(I3 / f"plan_tools/features/plan3_{bench}.csv")
    cmp_ = tab(I3 / "compile_consistency/cons_sopbench.csv")[["len_mean", "branch_mean"]] if sop else None
    def L(name, parts): return Ladder(2, TYPES, GLOB, rub, extra_table=pd.concat(parts, axis=1), name=name)
    ad = tab(REPO / "data" / bench / "features" / "adele.csv")
    P_ = {"c_orig": L("c_orig", [ad, plan] + ([cmp_] if sop else []))}
    for v in VARS:
        P_["b" + v] = H.ridge(src["f" + v])
        if v != "A0":
            f = tab(HERE / "features" / f"{v}_{bench}.csv")
            P_["c" + v] = L("c" + v, [f, plan] + ([cmp_] if sop else []))
    return P_
if __name__ == "__main__":
    out = HERE / "out"
    extra = {"f" + v: {b: str(HERE / "features" / f"{v}_{b}.csv") for b in ["sopbench", "tau2"]} for v in VARS}
    with cpu_lock("adele_plus"):
        H.run_all(out, make_preds, extra_csv=extra, benches=["sopbench", "tau2"], schemes=["new_procedures", "new_domain"])
    print("run done")
