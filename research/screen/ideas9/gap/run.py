"""Round 9 gap: c_lad plus data-gap features. usage: run.py"""
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
from feats import GAP, COND
I3 = Path(S + "/ideas3"); HERE = Path(S + "/ideas9/gap"); REPO = H.REPO
def tab(path):
    d = pd.read_csv(path); return d.set_index(d.columns[0])
TYPES = [f"n_{t}" for t in P.STEP_TYPES]; GLOB = ["max_depth", "refusal_conditions", "needs_hidden_data", "p_refusal"]
G4 = ["n_fetch", "frac_not_given", "n_cond_fetch"]
def make_preds(src, M, bench):
    sop = bench == "sopbench"
    ad = tab(REPO / "data" / bench / "features" / "adele.csv")
    rub = tab(S + f"/rubric/features/rub_proc_{bench}.csv")
    plan = tab(I3 / f"plan_tools/features/plan3_{bench}.csv")
    cmp_ = tab(I3 / "compile_consistency/cons_sopbench.csv")[["len_mean", "branch_mean"]] if sop else None
    g = tab(HERE / f"gap_{bench}.csv")
    base = [ad, plan] + ([cmp_] if sop else [])
    def L(name, parts): return Ladder(2, TYPES, GLOB, rub, extra_table=pd.concat(parts, axis=1), name=name)
    return {"t0_clad": L("t0_clad", base),
            "g1_clad_gap": L("g1_clad_gap", base + [g[GAP]]),
            "g2_adele_gap": L("g2_adele_gap", [ad, g[GAP]]),
            "g3_clad_cond": L("g3_clad_cond", base + [g[COND]]),
            "g4_clad_3": L("g4_clad_3", base + [g[G4]])}
if __name__ == "__main__":
    with cpu_lock("gap"):
        H.run_all(HERE / "out", make_preds, benches=["sopbench", "tau2"], schemes=["new_procedures", "new_domain"])
    print("run done")
