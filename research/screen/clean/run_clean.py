"""Baseline protocol on cleaned SOPBench through the shared harness (module paths redirected).
  python run_clean.py A|B run     fold IRT refit in results_<v>, sopbench obs for all methods
  python run_clean.py A|B eval    merge unchanged tau2 obs, evaluate pooled
  python run_clean.py old eval    same evaluation on the original 28-agent obs (merged from the repo/screen dirs)
Run with PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 and the repository interpreter."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, shutil, json
from pathlib import Path
sys.path.insert(0, ZR.SCREEN)
sys.path.insert(0, ZR.SCREEN + "/rubric")
sys.path.insert(0, ZR.SCREEN + "/targets")
import harness as H
import numpy as np, pandas as pd
HERE = Path(ZR.SCREEN + "/clean")
SCR = HERE.parent
REPO = H.REPO
var, mode = sys.argv[1], sys.argv[2]
SCH = ["new_procedures", "new_domain"]
RUBF = SCR / "rubric/features"

def make_preds(src, M, bench):
    if var == "B":
        return {"length": H.ridge(src["length"]), "human_time": H.ridge(src["human_time"])}
    from shares import SharesPredictor
    from ladder import Ladder
    import prompts as P
    TYPES = [f"n_{t}" for t in P.STEP_TYPES]
    GLOB = ["max_depth", "refusal_conditions", "needs_hidden_data", "p_refusal"]
    grids = H.RB.GroupedRidgePredictor.SOURCE_ALPHA_GRIDS["LLM Judge"]
    p = pd.read_csv(RUBF / f"rub_proc_{bench}.csv").set_index("case_id")
    a = pd.read_csv(REPO / "data" / bench / "features" / "adele.csv"); a = a.set_index(a.columns[0])
    return {
        "length": H.ridge(src["length"]), "human_time": H.ridge(src["human_time"]),
        "shares_adele": SharesPredictor(src["adele"], eb=False),
        "bb_adele": SharesPredictor(src["adele"], eb=True),
        "lad2_adele": Ladder(2, TYPES, GLOB, p, extra_table=a, name="lad2_adele"),
        "grp_proc_adele": H.grouped({"rub_proc": src["rub_proc"], "adele": src["adele"]} and [src["rub_proc"], src["adele"]],
                                    {n: grids for n in ["rub_proc", "adele"]}),
    }
NAMES = {"length": "text length", "human_time": "human time", "shares_adele": "smoothed shares + ridge ADeLe",
         "bb_adele": "beta-binomial shares + ridge ADeLe", "lad2_adele": "ladder L2 + ADeLe",
         "grp_proc_adele": "grouped ridge rubric | ADeLe"}
COMBOS = [("shares_adele", "adele"), ("lad2_adele", "adele"), ("grp_proc_adele", "adele")]
KEY = ["seed", "fold", "agent", "case_id", "y"]

def merge(bench, scheme, dest, sop_dirs):
    """Column-wise merge of obs files of one scheme from several runs (rows must be identical)."""
    base = None
    for d in sop_dirs:
        f = Path(d) / bench / f"obs_{scheme}.csv.gz"
        m = pd.read_csv(f)
        if base is None:
            base = m
            continue
        assert len(m) == len(base) and (m[KEY].values == base[KEY].values).all(), (f, "rows differ")
        for c in m.columns:
            if c not in base.columns:
                base[c] = m[c]
    dest = Path(dest) / bench; dest.mkdir(parents=True, exist_ok=True)
    base.to_csv(dest / f"obs_{scheme}.csv.gz", index=False)

if var in "AB":
    root, res = HERE / f"data_{var}", HERE / f"results_{var}"
    H.RB.DATA = root; H.E.DATA = root; H.RB.RESULTS = res
    if mode == "run":
        res.mkdir(exist_ok=True)
        H.run_all(res / "_sop", make_preds, extra_csv={"rub_proc": {"sopbench": str(RUBF / "rub_proc_sopbench.csv")}},
                  benches=["sopbench"], schemes=SCH)
        for s in SCH:  # sopbench obs where the evaluator expects them
            merge("sopbench", s, res, [res / "_sop"])
        shutil.copytree(res / "_sop" / "sopbench", res / "sopbench", dirs_exist_ok=True)
    else:
        for s in SCH:
            merge("tau2", s, res, [REPO / "results"] + ([SCR / "targets/run", SCR / "rubric/out/run"] if var == "A" else []))
        H.evaluate_all(res, NAMES, combos=COMBOS if var == "A" else ())
else:  # old numbers with the same evaluator and methods
    res = HERE / "results_old"
    for b in ["sopbench", "tau2"]:
        for s in SCH:
            merge(b, s, res, [REPO / "results", SCR / "targets/run", SCR / "rubric/out/run"])
    H.evaluate_all(res, NAMES, combos=COMBOS)
