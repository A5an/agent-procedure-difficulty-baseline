import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import os, sys
from alib import *
import metrics_extra as X
from cpulock import cpu_lock

OUT = HERE / "out"
RUB = ZR.SCREEN + "/rubric/out/run"

def make_preds(s, M, bench):
    return {"anchors": AnchorPredictor(bench, "llm", "anchors"),
            "knn4_beta": AnchorPredictor(bench, "knn4", "knn4_beta")}

if __name__ == "__main__":
    combos = [("anchors", "adele"), ("knn4_beta", "adele"), ("anchors", "lad2_adele")]
    with cpu_lock("anchors"):
        H.run_all(OUT, make_preds, benches=["sopbench", "tau2"], schemes=["new_procedures", "new_domain"])
        H.evaluate_all(OUT, {"anchors": "LLM in-context anchors", "knn4_beta": "mean beta of 4 nearest"},
                       combos=combos[:2])
        X.report(OUT, ["anchors", "knn4_beta"], extra_dirs={"lad2_adele": RUB, "grp_proc_adele": RUB},
                 combos=combos)
