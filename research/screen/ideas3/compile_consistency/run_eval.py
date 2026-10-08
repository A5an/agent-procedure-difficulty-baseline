"""Variants fixed in advance (before any evaluation):
 cons            ridge on the 10 consistency features alone
 cons_cat_adele  one ridge on consistency + ADeLe 18 levels
 lad2_adele_cons rubric ladder L2 + ADeLe with the consistency features added to the global features
 combos          logit average with the baseline: (cons, adele), (cons_cat_adele, adele), (lad2_adele_cons, adele), (cons, lad2_adele)
SOPBench only (the method needs SOPBench databases), schemes new_procedures and new_domain."""
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
HERE = Path(ZR.SCREEN + "/ideas3/compile_consistency")
FEAT = HERE / "feat"; FEAT.mkdir(exist_ok=True)
REPO = H.REPO; B = ["sopbench"]
RUB = Path(ZR.SCREEN + "/rubric")
TYPES = [f"n_{t}" for t in P.STEP_TYPES]
GLOB = ["max_depth", "refusal_conditions", "needs_hidden_data", "p_refusal"]
def adele(b):
    a = pd.read_csv(REPO / "data" / b / "features" / "adele.csv"); return a.set_index(a.columns[0])
def cons(b): return pd.read_csv(HERE / "cons_sopbench.csv").set_index("case_id")
extra = {"cons": {"sopbench": str(HERE / "cons_sopbench.csv")}, "cons_cat_adele": {}}
d = pd.concat([cons("sopbench"), adele("sopbench")], axis=1); d.index.name = "case_id"
d.reset_index().to_csv(FEAT / "cons_cat_adele_sopbench.csv", index=False)
extra["cons_cat_adele"]["sopbench"] = str(FEAT / "cons_cat_adele_sopbench.csv")
def make_preds(src, M, bench):
    p = pd.read_csv(RUB / "features" / f"rub_proc_{bench}.csv").set_index("case_id")
    ex = pd.concat([adele(bench), cons(bench)], axis=1)
    return {"cons": H.ridge(src["cons"]), "cons_cat_adele": H.ridge(src["cons_cat_adele"]),
            "lad2_adele_cons": Ladder(2, TYPES, GLOB, p, extra_table=ex, name="lad2_adele_cons")}
DESC = {"cons": "ridge on 10 compile-consistency features", "cons_cat_adele": "one ridge on consistency + ADeLe",
        "lad2_adele_cons": "ladder L2 + ADeLe with consistency columns added to the global features"}
COMBOS = [("cons", "adele"), ("cons_cat_adele", "adele"), ("lad2_adele_cons", "adele"), ("cons", "lad2_adele")]
if __name__ == "__main__":
    out = HERE / "out" / "run"
    with cpu_lock("compile_consistency"):
        H.run_all(out, make_preds, extra_csv=extra, benches=B, schemes=["new_procedures", "new_domain"])
        H.evaluate_all(out, DESC, combos=COMBOS[:3])
        X.report(out, list(DESC), extra_dirs={"lad2_adele": str(RUB / "out" / "run"), "grp_proc_adele": str(RUB / "out" / "run")}, combos=COMBOS)
