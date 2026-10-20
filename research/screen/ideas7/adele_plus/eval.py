import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas")
import harness as H, metrics_extra as X
from cpulock import cpu_lock
from pathlib import Path
out = Path(S + "/ideas7/adele_plus/out")
DESC = {k: "adele_plus" for k in ["c_orig","bA","bA0","bB","bC","cA","cB","cC"]}
with cpu_lock("adele_plus"):
    H.evaluate_all(out, DESC)
    X.report(out, list(DESC))
