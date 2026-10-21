"""usage: run.py [debug]. Variants fixed in PREREG.md."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas"); sys.path.insert(0, S + "/rubric"); sys.path.insert(0, S + "/ideas7/cond_ladder")
from pathlib import Path
import pandas as pd
import harness as H
from cpulock import cpu_lock
from ladder import Ladder
import prompts as P
from condladder import CondLadder
I3 = Path(S + "/ideas3"); HERE = Path(S + "/ideas7/cond_ladder"); REPO = H.REPO
def tab(path):
    d = pd.read_csv(path); return d.set_index(d.columns[0])
TYPES = [f"n_{t}" for t in P.STEP_TYPES]; GLOB = ["max_depth", "refusal_conditions", "needs_hidden_data", "p_refusal"]
def make_preds(src, M, bench):
    sop = bench == "sopbench"
    ad = tab(REPO / "data" / bench / "features" / "adele.csv")
    rub = tab(S + f"/rubric/features/rub_proc_{bench}.csv")
    plan = tab(I3 / f"plan_tools/features/plan3_{bench}.csv")
    ex = pd.concat([ad, plan] + ([tab(I3 / "compile_consistency/cons_sopbench.csv")[["len_mean", "branch_mean"]]] if sop else []), axis=1)
    ref = Ladder(2, TYPES, GLOB, rub, extra_table=ex, name="d_clad_ref")
    if not sop:  # tau2: no conditions, variants are plain c_lad
        return {"v1_cond_score": Ladder(2, TYPES, GLOB, rub, extra_table=ex, name="v1"),
                "v2_cond_summ": Ladder(2, TYPES, GLOB, rub, extra_table=ex, name="v2"), "d_clad_ref": ref}
    return {"v1_cond_score": CondLadder("score", 2, TYPES, GLOB, rub, extra_table=ex, name="v1"),
            "v2_cond_summ": CondLadder("summ", 2, TYPES, GLOB, rub, extra_table=ex, name="v2"), "d_clad_ref": ref}
if __name__ == "__main__":
    dbg = len(sys.argv) > 1
    out = HERE / ("out_debug" if dbg else "out")
    with cpu_lock("cond_ladder"):
        H.run_all(out, make_preds, benches=["sopbench"] if dbg else ["sopbench", "tau2"], schemes=["new_domain"] if dbg else ["new_procedures", "new_domain"])
    print("run done", out)
