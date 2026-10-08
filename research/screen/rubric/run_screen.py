"""Run the rubric feature sets through the shared harness (baseline protocol)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
from pathlib import Path
sys.path.insert(0, ZR.SCREEN)
import pandas as pd
import harness as H
import prompts as P
from ladder import Ladder

HERE = Path(__file__).parent
REPO = H.REPO
B = H.BENCHES
FEAT = HERE / "features"
CAT = FEAT / "concat"
CAT.mkdir(exist_ok=True)
TYPES = [f"n_{t}" for t in P.STEP_TYPES]
GLOB = ["max_depth", "refusal_conditions", "needs_hidden_data", "p_refusal"]

def csv_paths(name):
    return {b: str(FEAT / f"{name}_{b}.csv") for b in B}

def make_concat(name, parts):
    """Concatenate feature CSVs (each: id column first) on the id column."""
    paths = {}
    for b in B:
        dfs = []
        for p in parts:
            f = FEAT / f"{p}_{b}.csv" if p.startswith("rub_") else REPO / "data" / b / "features" / f"{p}.csv"
            d = pd.read_csv(f)
            d = d.rename(columns={d.columns[0]: "case_id"}).set_index("case_id")
            dfs.append(d)
        out = pd.concat(dfs, axis=1)
        path = CAT / f"{name}_{b}.csv"
        out.reset_index().to_csv(path, index=False)
        paths[b] = str(path)
    return paths

extra = {n: csv_paths(n) for n in ["rub_proc", "rub_len", "rub_types", "rub_lara", "rub_rpa"]}
for s in ["proc", "types", "len", "lara", "rpa"]:
    extra[f"cat_{s}_adele"] = make_concat(f"cat_{s}_adele", [f"rub_{s}", "adele"])
extra["cat_full"] = make_concat("cat_full", ["rub_proc", "adele", "human_time"])

TABLES = {}
def table(bench):
    if bench not in TABLES:
        p = pd.read_csv(FEAT / f"rub_proc_{bench}.csv").set_index("case_id")
        a = pd.read_csv(REPO / "data" / bench / "features" / "adele.csv")
        a = a.set_index(a.columns[0])
        TABLES[bench] = (p, a)
    return TABLES[bench]

def make_preds(src, M, bench):
    G = M and None
    grids = H.RB.GroupedRidgePredictor.SOURCE_ALPHA_GRIDS["LLM Judge"]
    p, a = table(bench)
    CSV = H.CSVFeatureSource
    def grp(names, srcs):
        return H.grouped(srcs, {n: grids for n in names})
    out = {
        "rub_proc": H.ridge(src["rub_proc"]),
        "rub_lara": H.ridge(src["rub_lara"]),
        "rub_rpa": H.ridge(src["rub_rpa"]),
        "len_ridge": H.ridge(src["rub_len"]),
        "types_ridge": H.ridge(src["rub_types"]),
        "cat_proc_adele": H.ridge(src["cat_proc_adele"]),
        "cat_lara_adele": H.ridge(src["cat_lara_adele"]),
        "cat_rpa_adele": H.ridge(src["cat_rpa_adele"]),
        "cat_full": H.ridge(src["cat_full"]),
        "grp_proc_adele": grp(["rub_proc", "adele"], [src["rub_proc"], src["adele"]]),
        "grp_lara_adele": grp(["rub_lara", "adele"], [src["rub_lara"], src["adele"]]),
        "grp_rpa_adele": grp(["rub_rpa", "adele"], [src["rub_rpa"], src["adele"]]),
        "grp_full": grp(["rub_proc", "adele", "human_time"], [src["rub_proc"], src["adele"], src["human_time"]]),
        "lad1": Ladder(1, ["n_steps"], [], p, name="lad1"),
        "lad2": Ladder(2, TYPES, GLOB, p, name="lad2"),
        "lad2_adele": Ladder(2, TYPES, GLOB, p, extra_table=a, name="lad2_adele"),
        "lad3": Ladder(3, TYPES, GLOB, p, name="lad3"),
    }
    return out

DESC = {
 "rub_proc": "ridge on the 12 procedure-rubric features",
 "rub_lara": "ridge on LARA D1-D5, weighted mean, level",
 "rub_rpa": "ridge on 13 Wellmann RPA criteria + total",
 "len_ridge": "ridge on total steps only (only-length control, ladder level 1 as ridge)",
 "types_ridge": "ridge on step counts by type (level 2 as ridge)",
 "cat_proc_adele": "one ridge on procedure rubric + ADeLe",
 "cat_lara_adele": "one ridge on LARA + ADeLe",
 "cat_rpa_adele": "one ridge on RPA + ADeLe",
 "cat_full": "one ridge on full procedure rubric + ADeLe + human_time",
 "grp_proc_adele": "grouped ridge procedure rubric | ADeLe",
 "grp_lara_adele": "grouped ridge LARA | ADeLe",
 "grp_rpa_adele": "grouped ridge RPA | ADeLe",
 "grp_full": "grouped ridge procedure rubric | ADeLe | human_time",
 "lad1": "logistic ladder L1: theta_a - delta * n_steps",
 "lad2": "logistic ladder L2: theta_a - sum_t delta_t n_t - v'g (g = 4 global rubric features)",
 "lad2_adele": "ladder L2 with g = rubric globals + 18 ADeLe",
 "lad3": "ladder L3: L2 + per-agent deviations gamma_at on type counts (strong L2)",
}

if __name__ == "__main__":
    out = HERE / "out" / "run"
    H.run_all(out, make_preds, extra_csv=extra)
    H.evaluate_all(out, DESC)
