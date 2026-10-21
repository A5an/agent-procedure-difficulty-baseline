import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas")
import harness as H, metrics_extra as X
from cpulock import cpu_lock
from pathlib import Path
out = Path(S + "/ideas6/msg_facts/out")
DESC = {"f1_feat_adele": "ADeLe + message-fact features (L1 + L0 priors)", "f2_clad_feat": "c_lad + message-fact features (L1)",
        "f3_clad_pperf": "c_lad + p_perf only (L1)"}
with cpu_lock("msg_facts"):
    H.evaluate_all(out, DESC); X.report(out, list(DESC))
