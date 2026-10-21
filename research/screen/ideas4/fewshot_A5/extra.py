import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas")
import harness as H, metrics_extra as X
from cpulock import cpu_lock
from pathlib import Path
out = Path(__file__).parent / "out_tau2"
V = ["A","A_adele","c_lad","c_lad_A","A_dm","A_w","A_all","A_len","A_raw"]
with cpu_lock("fewshot_A5"):
    H.evaluate_all(out, {v: v for v in V}, combos=[])
    X.report(out, V, combos=[])
