import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas")
import harness as H
from cpulock import cpu_lock
from pathlib import Path
out = Path(__file__).parent / sys.argv[1]
DESC = {k: k for k in sys.argv[2].split(",")}
with cpu_lock("fewshot_plan"):
    H.evaluate_all(out, DESC, combos=[])
