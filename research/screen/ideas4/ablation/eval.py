import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas")
import harness as H, metrics_extra as X
from cpulock import cpu_lock
from pathlib import Path
out = Path(S + "/ideas4/ablation/out")
DESC = {k: "post hoc ablation" for k in ["a1_adele","a2_adele_plan","a3_adele_cmp","a4_full","p5_plan_noadele","p6_full_bt"]}
with cpu_lock("ablation"):
    H.evaluate_all(out, DESC)
    X.report(out, list(DESC))
