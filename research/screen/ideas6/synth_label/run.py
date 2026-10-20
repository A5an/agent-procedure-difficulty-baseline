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
I3 = Path(S + "/ideas3"); HERE = Path(S + "/ideas6/synth_label"); REPO = H.REPO
def tab(path):
    d = pd.read_csv(path); return d.set_index(d.columns[0])
TYPES = [f"n_{t}" for t in P.STEP_TYPES]; GLOB = ["max_depth", "refusal_conditions", "needs_hidden_data", "p_refusal"]
F = tab(HERE / "feats.csv")
def make_preds(src, M, bench):
    if bench != "sopbench": return {}
    ad = tab(REPO / "data" / bench / "features" / "adele.csv")
    rub = tab(S + f"/rubric/features/rub_proc_{bench}.csv")
    plan = tab(I3 / f"plan_tools/features/plan3_{bench}.csv")
    cmp_ = tab(I3 / "compile_consistency/cons_sopbench.csv")[["len_mean", "branch_mean"]]
    def L(name, parts): return Ladder(2, TYPES, GLOB, rub, extra_table=pd.concat(parts, axis=1), name=name)
    return {"sl1_feat_adele": L("sl1_feat_adele", [ad, F]),
            "sl2_clad_feat": L("sl2_clad_feat", [ad, plan, cmp_, F]),
            "sl3_clad_p": L("sl3_clad_p", [ad, plan, cmp_, F[["sl_p"]]]),
            "sl0_clad": L("sl0_clad", [ad, plan, cmp_])}
if __name__ == "__main__":
    out = HERE / "out"
    with cpu_lock("synth_label"):
        H.run_all(out, make_preds, benches=["sopbench"], schemes=["new_procedures", "new_domain"])
    print("run done")
