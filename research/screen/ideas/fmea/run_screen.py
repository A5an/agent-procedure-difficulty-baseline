import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
from pathlib import Path
sys.path.insert(0, ZR.SCREEN)
sys.path.insert(0, ZR.SCREEN + "/ideas")
import pandas as pd
import harness as H, metrics_extra as X
from cpulock import cpu_lock
HERE = Path(__file__).parent; REPO = H.REPO; B = ["sopbench", "tau2"]; CAT = HERE / "features" / "concat"; CAT.mkdir(exist_ok=True)
extra = {"fmea": {b: str(HERE / "features" / f"fmea_{b}.csv") for b in B}}
paths = {}
for b in B:
    f = pd.read_csv(HERE / "features" / f"fmea_{b}.csv").set_index("case_id")
    a = pd.read_csv(REPO / "data" / b / "features" / "adele.csv"); a = a.set_index(a.columns[0])
    o = pd.concat([f, a], axis=1); p = CAT / f"fmea_cat_adele_{b}.csv"; o.reset_index(names="case_id").to_csv(p, index=False); paths[b] = str(p)
extra["fmea_cat_adele"] = paths

def make_preds(src, M, bench):
    grids = H.RB.GroupedRidgePredictor.SOURCE_ALPHA_GRIDS["LLM Judge"]
    return {"fmea": H.ridge(src["fmea"]), "fmea_cat_adele": H.ridge(src["fmea_cat_adele"]),
            "fmea_grp_adele": H.grouped([src["fmea"], src["adele"]], {"fmea": grids, "adele": grids})}
DESC = {"fmea": "ridge on FMEA features", "fmea_cat_adele": "one ridge on FMEA + ADeLe", "fmea_grp_adele": "grouped ridge FMEA | ADeLe"}
COMBOS = [("fmea_cat_adele", "adele")]
RUB = ZR.SCREEN + "/rubric/out/run"
if __name__ == "__main__":
    out = HERE / "out" / "run"
    with cpu_lock("fmea"):
        H.run_all(out, make_preds, extra_csv=extra, benches=B, schemes=["new_procedures", "new_domain"])
        H.evaluate_all(out, DESC, combos=COMBOS)
        X.report(out, list(DESC), extra_dirs={"lad2_adele": RUB, "grp_proc_adele": RUB},
                 combos=COMBOS + [("fmea_cat_adele", "lad2_adele")])
