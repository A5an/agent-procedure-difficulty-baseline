import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, os
sys.path.insert(0, ZR.SCREEN + "/ideas2/cc_agent")
import score as S
out = {}
for arm in ["native", "guard", "native_date", "guard_date"]:
    d = S.load_arm(arm)
    out["CC " + arm] = {c: bool(v["success"]) for c, v in d.items()}
h = json.load(open(ZR.SCREEN + "/ideas2/cc_agent/harness_cases.json"))
for k, v in h.items():
    out[k] = {c: bool(x["success"]) for c, x in v.items()}
json.dump(out, open("modern_cases.json", "w"), indent=1)
print({k: (len(v), round(sum(v.values()) / len(v), 3)) for k, v in out.items()})
