import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas")
import harness as H, metrics_extra as X
from cpulock import cpu_lock
from pathlib import Path
out = Path(S + "/ideas7/entropy_combo/out")
V = ["v_a_clad_ent", "v_b_allcheap", "v_c_allcheap_emu", "d_clad_fam", "d_a_plain"]
if True:
    H.evaluate_all(out, {k: "entropy_combo" for k in V})
    X.report(out, V, extra_dirs={"c_lad": S + "/ideas3/combo/out_main"})
