import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas")
import harness as H, metrics_extra as X
from cpulock import cpu_lock
from pathlib import Path
out = Path(S + "/ideas7/cond_ladder/out")
DESC = {"v1_cond_score": "c_lad + in-fold COND case score", "v2_cond_summ": "c_lad + in-fold COND skip summaries", "d_clad_ref": "c_lad reproduction"}
with cpu_lock("cond_ladder"):
    H.evaluate_all(out, DESC)
    X.report(out, ["adele"] + list(DESC))
