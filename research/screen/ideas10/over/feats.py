import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, re, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).parent
OVER = ["ov_extra_checks", "ov_auth_extra", "ov_nocred", "ov_auth_gap"]
AUTH = re.compile(r"login|auth|verify|identity|password", re.I)
CRED = re.compile(r"password|identification|identity|\bid\b|\bpin\b|passcode", re.I)
import sys; sys.path.insert(0, ZR.SCREEN + "/ideas9/gap")
from gap import inputs
REQ = {it[1]: it[4] for it in inputs()}
def tl(p, kinds=None):
    return [c.get("tool") for c in (p or {}).get("calls", []) if isinstance(c, dict) and c.get("tool") and (kinds is None or c.get("kind") in kinds)]
def feats(cid, v):
    nv = [n for n in v["naive"] if isinstance(n, dict)]; pp = v["policy"] if isinstance(v["policy"], dict) else None
    d = {}
    if nv and pp is not None:
        pt = set(tl(pp)); pauth = any(AUTH.search(t) for t in pt)
        d["ov_extra_checks"] = float(np.mean([len(set(tl(n, {"check", "lookup"})) - pt) for n in nv]))
        d["ov_auth_extra"] = 0.0 if pauth else float(np.mean([any(AUTH.search(t) for t in tl(n)) for n in nv]))
    d["ov_nocred"] = float(not CRED.search(REQ[cid]))
    if "ov_auth_extra" in d: d["ov_auth_gap"] = d["ov_auth_extra"] * d["ov_nocred"]
    return d
if __name__ == "__main__":
    raw = json.load(open(HERE.parent / "prior/prior_raw.json"))
    df = pd.DataFrame([{"case_id": k, "bench": v["bench"], **feats(k, v)} for k, v in raw.items()])
    for b, g in df.groupby("bench"):
        g = g.set_index("case_id")[OVER]; print(b, len(g), g.isna().sum().to_dict(), g.mean().round(3).to_dict())
        g.fillna(g.median()).to_csv(HERE / f"over_{ {'banking': 'tauk_banking'}.get(b, b) }.csv")
