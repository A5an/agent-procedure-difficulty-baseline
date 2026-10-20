import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, shutil
sys.path.insert(0, ZR.SCREEN); sys.path.insert(0, ZR.SCREEN + "/ideas")
sys.path.insert(0, ZR.SCREEN + "/ideas3/conditions")
import harness as H, metrics_extra as X, json
import evaluate as E
from cpulock import cpu_lock
import cm
OUT = ZR.SCREEN + "/ideas3/conditions/out_diag"
COLS = ["l2_plain", "l2_alone", "l2_truth", "l0_prior", "l0_alone", "l0_const", "l2_const"]
DESC = {c: c for c in COLS}
with cpu_lock("conditions_diag"):
    H.run_all(OUT, lambda src, M: cm.make_all(COLS), benches=["sopbench"], schemes=["new_procedures", "new_domain"])
    for base in ["adele", "l2_plain", "l0_const"]:
        E.BASELINE = base
        H.evaluate_all(OUT, {k: v for k, v in DESC.items() if k != "adele"})
        shutil.copy(OUT + "/eval.json", OUT + f"/eval_base_{base}.json")
    E.BASELINE = "adele"
    X.report(OUT, ["adele"] + COLS)
