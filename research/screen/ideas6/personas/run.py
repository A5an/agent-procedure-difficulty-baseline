"""Harness run. Variants fixed in advance: pers (ridge on persona features), pers_cat_adele (ridge on persona + ADeLe),
c_lad_pers (c_lad ladder tables [ADeLe, PLAN, CMP] + persona table, built like ideas4/ablation/run.py a4_full)."""
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
I3 = Path(S + "/ideas3"); HERE = Path(S + "/ideas6/personas"); REPO = H.REPO; FEAT = HERE / "features"; CAT = FEAT / "concat"; CAT.mkdir(exist_ok=True)
B = ["sopbench", "tau2"]
def tab(path):
    d = pd.read_csv(path); return d.set_index(d.columns[0])
TYPES = [f"n_{t}" for t in P.STEP_TYPES]; GLOB = ["max_depth", "refusal_conditions", "needs_hidden_data", "p_refusal"]
extra = {"pers": {b: str(FEAT / f"pers_{b}.csv") for b in B}, "pers_cat_adele": {}}
for b in B:
    d = pd.concat([tab(FEAT / f"pers_{b}.csv"), tab(REPO / "data" / b / "features" / "adele.csv")], axis=1)
    p = CAT / f"pers_cat_adele_{b}.csv"; d.reset_index().to_csv(p, index=False); extra["pers_cat_adele"][b] = str(p)
def make_preds(src, M, bench):
    sop = bench == "sopbench"
    ad = tab(REPO / "data" / bench / "features" / "adele.csv"); rub = tab(S + f"/rubric/features/rub_proc_{bench}.csv")
    plan = tab(I3 / f"plan_tools/features/plan3_{bench}.csv"); pers = tab(FEAT / f"pers_{bench}.csv")
    cmp_ = tab(I3 / "compile_consistency/cons_sopbench.csv")[["len_mean", "branch_mean"]] if sop else None
    parts = [ad, plan] + ([cmp_] if sop else []) + [pers]
    return {"pers": H.ridge(src["pers"]), "pers_cat_adele": H.ridge(src["pers_cat_adele"]),
            "c_lad_pers": Ladder(2, TYPES, GLOB, rub, extra_table=pd.concat(parts, axis=1), name="c_lad_pers")}
DESC = {"pers": "ridge on persona-simulation features", "pers_cat_adele": "ridge on persona features + ADeLe", "c_lad_pers": "c_lad ladder + persona table"}
if __name__ == "__main__":
    out = HERE / "out"
    with cpu_lock("personas"):
        H.run_all(out, make_preds, extra_csv=extra, benches=B, schemes=["new_procedures", "new_domain"])
        H.evaluate_all(out, DESC)
        X.report(out, list(DESC), extra_dirs={"a4_full": S + "/ideas4/ablation/out"})
    print("run done")
