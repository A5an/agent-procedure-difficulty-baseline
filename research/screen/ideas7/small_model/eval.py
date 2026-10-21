import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas")
import harness as H, metrics_extra as X
from cpulock import cpu_lock
from pathlib import Path
out = Path(S + "/ideas7/small_model/out_h")
V = ["v_a_clad_sm", "v_b_clad_sm_pri", "v_c_fam_sm", "d_sm_only"]
with cpu_lock("small_model_eval"):
    H.evaluate_all(out, {k: "small_model" for k in V})
    X.report(out, V, extra_dirs={"c_lad": S + "/ideas3/combo/out_main"})
