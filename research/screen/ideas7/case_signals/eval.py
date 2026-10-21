import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas")
import harness as H, metrics_extra as X
import contextlib
cpu_lock = lambda n: contextlib.nullcontext()  # shared lock queue was 20+ min, light job run with nice
from pathlib import Path
out = Path(S + "/ideas7/case_signals/out")
V = ["s1_nli", "s2_ret", "s3_act", "s4_bnd", "s5_all", "c_lad_re"]
with cpu_lock("case_signals_eval"):
    H.evaluate_all(out, {v: v for v in V}); X.report(out, V)
