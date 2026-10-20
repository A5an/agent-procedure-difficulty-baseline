import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas")
import harness as H, metrics_extra as X
from cpulock import cpu_lock
from pathlib import Path
out = Path(S + "/ideas6/emu_feat/out")
DESC = {"e1_clad_emu": "c_lad + emulation estimate (L0+E)", "e2_clad_mixplan": "c_lad + plan mixture (L1+E)",
        "e3_l2e_mix": "rubric ladder + precheck mixture (L2+E)", "e4_clad_mixpre": "c_lad + precheck mixture (L2+E)"}
with cpu_lock("emu_feat"):
    H.evaluate_all(out, DESC); X.report(out, list(DESC))
