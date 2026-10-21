import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
sys.path.insert(0, ZR.SCREEN)
sys.path.insert(0, ZR.SCREEN + "/ideas")
import harness as H
import metrics_extra as X
from cpulock import cpu_lock
OUT = ZR.SCREEN + "/ideas2/steps/out"
with cpu_lock("steps_extra"):
    r = X.report(OUT, ["adele", "cnt_geo", "cnt_logit", "typ_logit", "typ_logit_a", "typ_prod", "cnt_prod"],
                 combos=[("typ_logit", "adele"), ("cnt_logit", "adele")])
import json
for k in r:
    if k == "summary":
        continue
    print(k)
    for c, v in r[k].items():
        print("  ", c, {a: (round(b, 3) if isinstance(b, float) else [round(x, 3) for x in b]) for a, b in v.items()})
