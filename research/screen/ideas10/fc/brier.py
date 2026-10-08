import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
S = ZR.SCREEN
sys.path.insert(0, S)
import harness as H
from pathlib import Path
out = H.evaluate_all(Path(S + "/ideas10/fc/out"), {"g1_clad_gap": "c_lad+GAP", "f2_all_fc": "c_lad+GAP+SCN+PRIOR+FC", "f0_fc_only": "FC only"})
import json
for k, v in out.items():
    if k == "pooled_new_process": continue
    print(k, {m: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in r.items() if "brier" in kk.lower() or "logloss" in kk.lower() or "log_loss" in kk.lower()} for m, r in v.items() if m in ("adele", "f2_all_fc", "g1_clad_gap")})
