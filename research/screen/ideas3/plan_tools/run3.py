"""Variants fixed before the first run:
 plan3          ridge on the mean of 3 plans (9 features) for tau2; SOPBench: plan_tau's 9 features (reused, one t0 plan)
 plan3sd        mean of 3 plans + spread (sd of calls, writes, distinct tools, refuse) for tau2; SOPBench same as plan3
 plan3_cat_adele  one ridge on plan3 + ADeLe
 averages with the baseline: plan3+adele, plan3_cat_adele+adele (equal-weight logit)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
sys.path.insert(0, ZR.SCREEN)
sys.path.insert(0, ZR.SCREEN + "/ideas")
from pathlib import Path
import pandas as pd
import harness as H, metrics_extra as X
from cpulock import cpu_lock
HERE = Path(__file__).parent; FEAT = HERE / "features"; CAT = FEAT / "concat"; CAT.mkdir(exist_ok=True)
REPO = H.REPO; B = ["sopbench", "tau2"]
RUB = Path(ZR.SCREEN + "/rubric")
def adele(b):
    a = pd.read_csv(REPO / "data" / b / "features" / "adele.csv"); return a.set_index(a.columns[0])
extra = {"plan3": {b: str(FEAT / f"plan3_{b}.csv") for b in B}, "plan3sd": {b: str(FEAT / f"plan3sd_{b}.csv") for b in B}, "plan3_cat_adele": {}}
for b in B:
    d = pd.concat([pd.read_csv(FEAT / f"plan3_{b}.csv").set_index("case_id"), adele(b)], axis=1); d.index.name = "case_id"
    d.reset_index().to_csv(CAT / f"plan3_cat_adele_{b}.csv", index=False); extra["plan3_cat_adele"][b] = str(CAT / f"plan3_cat_adele_{b}.csv")
def make_preds(src, M, bench):
    return {k: H.ridge(src[k]) for k in ["plan3", "plan3sd", "plan3_cat_adele"]}
DESC = {"plan3": "ridge on mean of 3 plans with real tool list", "plan3sd": "mean + spread of 3 plans",
        "plan3_cat_adele": "one ridge on plan3 + ADeLe"}
COMBOS = [("plan3_cat_adele", "adele"), ("plan3", "adele"), ("plan3sd", "adele")]
out = HERE / "out" / "run"
with cpu_lock("plan_tools"):
    H.run_all(out, make_preds, extra_csv=extra, benches=B, schemes=["new_procedures", "new_domain"])
    H.evaluate_all(out, DESC, combos=COMBOS[:1])
    X.report(out, list(DESC), extra_dirs={"lad2_adele": str(RUB / "out" / "run")}, combos=COMBOS)
