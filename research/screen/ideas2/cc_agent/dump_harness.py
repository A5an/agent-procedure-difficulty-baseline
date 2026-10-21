import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json
sys.argv = ["x"]
sys.path.insert(0, ZR.SCREEN + "/ideas2/harness")
import analyze as A
out = {}
for mo in ["gemini-2.5-flash", "gemini-3.8-flash"]:
    for ar in ["H0", "H1", "H2"]:
        d = A.load_arm(mo, ar, strip=(ar == "H2"))
        out[f"{mo} {ar}"] = d
json.dump(out, open(ZR.SCREEN + "/ideas2/cc_agent/harness_cases.json", "w"))
print({k: len(v) for k, v in out.items()})
