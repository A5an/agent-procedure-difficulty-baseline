"""Plan features through the harness. Variants fixed in advance:
 plan            ridge on the 9 plan features (L1)
 plan_cat_adele  one ridge on plan + ADeLe
 plan_grp_adele  grouped ridge plan | ADeLe (LLM Judge grid)
 and logit averages of plan and plan_cat_adele with the baseline adele.
Second run (tau2 only, uses GOLD): gold_len, gold_cat_adele."""
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
HERE = Path(__file__).parent
FEAT = HERE / "features"; CAT = FEAT / "concat"; CAT.mkdir(exist_ok=True)
REPO = H.REPO
B = ["sopbench", "tau2"]
RUB = Path(ZR.SCREEN + "/rubric")
def adele(b):
    a = pd.read_csv(REPO / "data" / b / "features" / "adele.csv"); return a.set_index(a.columns[0])
def plan(b): return pd.read_csv(FEAT / f"plan_{b}.csv").set_index("case_id")
mode = sys.argv[1]
if mode == "main":
    extra = {"plan": {b: str(FEAT / f"plan_{b}.csv") for b in B}, "plan_cat_adele": {}}
    for b in B:
        d = pd.concat([plan(b), adele(b)], axis=1); d.index.name = "case_id"
        d.reset_index().to_csv(CAT / f"plan_cat_adele_{b}.csv", index=False)
        extra["plan_cat_adele"][b] = str(CAT / f"plan_cat_adele_{b}.csv")
    def make_preds(src, M, bench):
        grids = H.RB.GroupedRidgePredictor.SOURCE_ALPHA_GRIDS["LLM Judge"]
        return {"plan": H.ridge(src["plan"]), "plan_cat_adele": H.ridge(src["plan_cat_adele"]),
                "plan_grp_adele": H.grouped([src["plan"], src["adele"]], {"plan": grids, "adele": grids})}
    DESC = {"plan": "ridge on 9 plan features", "plan_cat_adele": "one ridge on plan + ADeLe",
            "plan_grp_adele": "grouped ridge plan | ADeLe"}
    COMBOS = [("plan_cat_adele", "adele"), ("plan", "adele"), ("plan_grp_adele", "adele")]
    out = HERE / "out" / "run"; schemes = ["new_procedures", "new_domain"]; bs = B
else:
    g = pd.read_csv(HERE / "gold_tau2.csv").set_index("case_id")
    extra = {"gold_len": {}, "gold_cat_adele": {}}
    gcols = g[["gold_len", "gold_writes"]]
    gcols.reset_index().to_csv(CAT / "gold_tau2.csv", index=False); extra["gold_len"]["tau2"] = str(CAT / "gold_tau2.csv")
    d = pd.concat([gcols, adele("tau2")], axis=1); d.index.name = "case_id"
    d.reset_index().to_csv(CAT / "gold_cat_adele_tau2.csv", index=False); extra["gold_cat_adele"]["tau2"] = str(CAT / "gold_cat_adele_tau2.csv")
    def make_preds(src, M, bench):
        return {"gold_len": H.ridge(src["gold_len"]), "gold_cat_adele": H.ridge(src["gold_cat_adele"])}
    DESC = {"gold_len": "ORACLE USING GOLD: ridge on gold action count and gold write count",
            "gold_cat_adele": "ORACLE USING GOLD: gold counts + ADeLe"}
    COMBOS = []; out = HERE / "out" / "gold"; schemes = ["new_procedures", "new_domain"]; bs = ["tau2"]
with cpu_lock("plan_tau"):
    H.run_all(out, make_preds, extra_csv=extra, benches=bs, schemes=schemes)
    H.evaluate_all(out, DESC, combos=COMBOS[:1] if mode == "main" else [])
    if mode == "main":
        X.report(out, list(DESC), extra_dirs={"lad2_adele": str(RUB / "out" / "run")}, combos=COMBOS)
