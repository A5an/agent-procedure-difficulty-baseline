"""Variants fixed before the first run:
 b_clad  reference c_lad (rebuilt)
 g1_graph  c_lad + 16 graph feature columns (tau2 only, avg of Gemini mean3 and Sonnet plans); sopbench identical to c_lad
 g2_dwplan c_lad with PLAN extended by the depth-weighted counts (g_dw_calls, g_dw_writes) only; sopbench identical to c_lad
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
I3 = Path(S + "/ideas3"); HERE = Path(S + "/ideas7/tau_graph"); REPO = H.REPO
def tab(path):
    d = pd.read_csv(path); return d.set_index(d.columns[0])
TYPES = [f"n_{t}" for t in P.STEP_TYPES]; GLOB = ["max_depth", "refusal_conditions", "needs_hidden_data", "p_refusal"]
V = ["b_clad", "g1_graph", "g2_dwplan"]
def make_preds(src, M, bench):
    sop = bench == "sopbench"
    ad = tab(REPO / "data" / bench / "features" / "adele.csv"); rub = tab(S + f"/rubric/features/rub_proc_{bench}.csv")
    plan = tab(I3 / f"plan_tools/features/plan3_{bench}.csv")
    cmp_ = [tab(I3 / "compile_consistency/cons_sopbench.csv")[["len_mean", "branch_mean"]]] if sop else []
    def L(name, parts): return Ladder(2, TYPES, GLOB, rub, extra_table=pd.concat(parts, axis=1), name=name)
    g1, g2 = [], []
    if not sop:
        G = tab(HERE / "features/graph_avg_tau2.csv").loc[plan.index]
        g1 = [G]; g2 = [G[["g_dw_calls", "g_dw_writes"]]]
    return {"b_clad": L("b_clad", [ad, plan] + cmp_), "g1_graph": L("g1_graph", [ad, plan] + cmp_ + g1), "g2_dwplan": L("g2_dwplan", [ad, plan] + cmp_ + g2)}
if __name__ == "__main__":
    out = HERE / "out"
    with cpu_lock("tau_graph"):
        H.run_all(out, make_preds, benches=["sopbench", "tau2"], schemes=["new_procedures", "new_domain"])
        H.evaluate_all(out, {k: "tau_graph round 7" for k in V})
        X.report(out, V)
    print("run done")
