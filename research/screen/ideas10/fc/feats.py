"""fc_raw.json -> fc_<bench>.csv (5 FC features, median-filled per benchmark). usage: feats.py"""
import json, re, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).parent
FC = ["fc_diff", "fc_n_modes", "fc_sum_p", "fc_max_p", "fc_noisy_or"]
def pj(t):
    if not t: return None
    t = t.strip()
    if t.startswith("```"): t = t.split("\n", 1)[1].rsplit("```", 1)[0]
    try: return json.loads(t)
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        try: return json.loads(t[i:j + 1])
        except Exception: return None
def fnum(x):
    try: return float(x)
    except Exception: return np.nan
def feats(t):
    d = pj(t)
    if not isinstance(d, dict): return {}
    ps = [min(max(fnum(m.get("p")), 0.0), 1.0) for m in d.get("failure_modes", []) if isinstance(m, dict) and fnum(m.get("p")) == fnum(m.get("p"))]
    out = {}
    s = fnum(d.get("p_success"))
    if s == s: s = min(max(s, 0.02), 0.98); out["fc_diff"] = float(np.log((1 - s) / s))
    out["fc_n_modes"] = len(ps)
    if ps: out.update(fc_sum_p=float(sum(ps)), fc_max_p=float(max(ps)), fc_noisy_or=float(1 - np.prod([1 - p for p in ps])))
    else: out.update(fc_sum_p=0.0, fc_max_p=0.0, fc_noisy_or=0.0)
    return out
if __name__ == "__main__":
    raw = json.load(open(HERE / "fc_raw.json"))
    rows = [{"case_id": k, **feats(v)} for k, v in raw.items()]
    df = pd.DataFrame(rows).set_index("case_id")
    bench = df.index.to_series().map(lambda c: "sopbench" if c.count("/") == 2 else ("tauk_banking" if c.startswith("banking") else "tau2"))
    for b in ["sopbench", "tau2", "tauk_banking"]:
        g = df[bench == b].reindex(columns=FC)
        print(b, len(g), "missing", g.isna().sum().to_dict())
        g.fillna(g.median()).to_csv(HERE / f"fc_{b}.csv")
