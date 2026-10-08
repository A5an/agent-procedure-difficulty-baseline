"""Control: the ladder with NO task feature (agent intercepts only, same logistic fit). Isolates the calibration gain."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
from pathlib import Path
sys.path.insert(0, ZR.SCREEN)
import harness as H
import pandas as pd
from ladder import Ladder
HERE = Path(__file__).parent
def make_preds(src, M, bench):
    p = pd.read_csv(HERE / "features" / f"rub_proc_{bench}.csv").set_index("case_id")[["n_steps"]].copy()
    p["zero"] = 0.0
    return {"lad0": Ladder(1, ["zero"], [], p, name="lad0")}
out = HERE / "out" / "run_lad0"
H.run_all(out, make_preds, schemes=["new_procedures", "new_domain"], benches=["sopbench", "tau2"])
r = H.evaluate_all(out, {"lad0": "ladder with no task feature (agent intercepts only)"})
