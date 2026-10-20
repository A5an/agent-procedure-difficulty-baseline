"""Harness run for paper-plan features. Variants fixed before the first full run (see REPORT text):
 pf_lite, pf_lite_cat_adele, pf_g38, pf_g38_cat_adele, pf_both, pf_both_cat_adele  (ridge, baseline model, per benchmark)
 combos: logit average with adele for pf_lite, pf_g38, pf_both, pf_both_cat_adele."""
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
B = ["sopbench", "tau2"]; REPO = H.REPO
def adele(b):
    a = pd.read_csv(REPO / "data" / b / "features" / "adele.csv"); a = a.rename(columns={a.columns[0]: "case_id"}); return a.set_index("case_id")
extra = {}
for k in ("lite", "g38", "both"):
    extra[f"pf_{k}"] = {b: str(FEAT / f"pf_{k}_{b}.csv") for b in B}
    extra[f"pf_{k}_cat_adele"] = {}
    for b in B:
        d = pd.concat([pd.read_csv(FEAT / f"pf_{k}_{b}.csv").set_index("case_id"), adele(b)], axis=1)
        p = CAT / f"pf_{k}_cat_adele_{b}.csv"; d.reset_index().to_csv(p, index=False); extra[f"pf_{k}_cat_adele"][b] = str(p)
def make_preds(src, M, bench):
    return {n: H.ridge(src[n]) for n in ["pf_lite", "pf_lite_cat_adele", "pf_g38", "pf_g38_cat_adele", "pf_both", "pf_both_cat_adele"]}
DESC = {"pf_lite": "ridge on paper-plan features, flash-lite", "pf_lite_cat_adele": "ridge on flash-lite plan features + ADeLe",
        "pf_g38": "ridge on paper-plan features, gemini-3.8-flash", "pf_g38_cat_adele": "ridge on 3.8-flash plan features + ADeLe",
        "pf_both": "ridge on both models' plan features + disagreement", "pf_both_cat_adele": "pf_both + ADeLe"}
COMBOS = [("pf_lite", "adele"), ("pf_g38", "adele"), ("pf_both", "adele"), ("pf_both_cat_adele", "adele")]
if __name__ == "__main__":
    out = HERE / "out" / "run"
    with cpu_lock("paper_pilot"):
        H.run_all(out, make_preds, extra_csv=extra, benches=B, schemes=["new_procedures", "new_domain"])
        H.evaluate_all(out, DESC, combos=COMBOS)
        X.report(out, list(DESC), combos=COMBOS)
