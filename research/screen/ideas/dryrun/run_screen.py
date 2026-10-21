"""Dry-run features through the harness (benches sopbench, tau2; schemes new_procedures, new_domain)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
sys.path.insert(0, ZR.SCREEN)
sys.path.insert(0, ZR.SCREEN + "/ideas")
sys.path.insert(0, ZR.SCREEN + "/rubric")
from pathlib import Path
import pandas as pd
import harness as H, metrics_extra as X
from cpulock import cpu_lock
from ladder import Ladder
import prompts as P

HERE = Path(__file__).parent
FEAT = HERE / "features"; CAT = FEAT / "concat"; CAT.mkdir(exist_ok=True)
REPO = H.REPO
B = ["sopbench", "tau2"]
RUB = Path(ZR.SCREEN + "/rubric")
TYPES = [f"n_{t}" for t in P.STEP_TYPES]
GLOB = ["max_depth", "refusal_conditions", "needs_hidden_data", "p_refusal"]

def adele(b):
    a = pd.read_csv(REPO / "data" / b / "features" / "adele.csv"); return a.set_index(a.columns[0])
def dry(b):
    return pd.read_csv(FEAT / f"dry_{b}.csv").set_index("case_id")

extra = {"dry": {b: str(FEAT / f"dry_{b}.csv") for b in B}, "dry_cat_adele": {}, "adele_only": {}}
for b in B:
    d = pd.concat([dry(b), adele(b)], axis=1)
    d.index.name = "case_id"; d.reset_index().to_csv(CAT / f"dry_cat_adele_{b}.csv", index=False)
    extra["dry_cat_adele"][b] = str(CAT / f"dry_cat_adele_{b}.csv")

def make_preds(src, M, bench):
    grids = H.RB.GroupedRidgePredictor.SOURCE_ALPHA_GRIDS["LLM Judge"]
    p = pd.read_csv(RUB / "features" / f"rub_proc_{bench}.csv").set_index("case_id")
    ex = pd.concat([adele(bench), dry(bench)], axis=1)
    return {
        "dry": H.ridge(src["dry"]),
        "dry_cat_adele": H.ridge(src["dry_cat_adele"]),
        "dry_grp_adele": H.grouped([src["dry"], src["adele"]], {"dry": grids, "adele": grids}),
        "lad2_adele_dry": Ladder(2, TYPES, GLOB, p, extra_table=ex, name="lad2_adele_dry"),
    }

DESC = {"dry": "ridge on the 11 dry-run features", "dry_cat_adele": "one ridge on dry-run + ADeLe",
        "dry_grp_adele": "grouped ridge dry-run | ADeLe (LLM Judge grid)",
        "lad2_adele_dry": "ladder L2 + ADeLe with dry-run columns added to the global features"}
COMBOS = [("dry_cat_adele", "adele"), ("dry_cat_adele", "lad2_adele")]
if __name__ == "__main__":
    out = HERE / "out" / "run"
    with cpu_lock("dryrun"):
        H.run_all(out, make_preds, extra_csv=extra, benches=B, schemes=["new_procedures", "new_domain"])
        H.evaluate_all(out, DESC, combos=[COMBOS[0]])
        X.report(out, list(DESC), extra_dirs={"lad2_adele": str(RUB / "out" / "run"), "grp_proc_adele": str(RUB / "out" / "run")},
                 combos=COMBOS)
