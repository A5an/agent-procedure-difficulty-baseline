import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas"); sys.path.insert(0, S + "/ideas3/combo")
import harness as H, metrics_extra as X
from cpulock import cpu_lock
from pathlib import Path
import json
HERE = Path(S + "/ideas3/combo"); out = HERE / "out_main"
DESC = {"c_L0": "L0 only", "c_L1": "L0+L1 grouped", "c_lad": "ladder + ADeLe + plan + cmp"}
COMBOS = [("c_L1", "adele"), ("c_lad", "adele")]
with cpu_lock("combo"):
    H.evaluate_all(out, DESC, combos=COMBOS)
    X.report(out, list(DESC), extra_dirs={"lad2_adele": S + "/rubric/out/run"}, combos=COMBOS)
