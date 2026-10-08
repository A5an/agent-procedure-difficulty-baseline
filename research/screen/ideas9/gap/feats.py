import json, re, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).parent
ARG = ["n_args", "n_given", "n_derivable", "n_fetch", "n_ask", "n_missing", "frac_not_given"]
COND = ["n_cond", "n_cond_given", "n_cond_fetch", "n_cond_ask", "n_cond_unknown", "frac_cond_not_given"]
GAP = ARG + COND
def parse(t):
    if not t: return None
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    try: return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        try: return json.loads(m.group(0)) if m else None
        except Exception: return None
def feats(p):
    seen = {}; 
    for a in p.get("arguments") or []:
        if isinstance(a, dict): seen.setdefault((str(a.get("tool")), str(a.get("argument"))), str(a.get("source", "")).upper())
    s = list(seen.values()); n = len(s)
    c = [str(x.get("source", "")).upper() for x in (p.get("conditions") or []) if isinstance(x, dict)]; m = len(c)
    f = dict(n_args=n, n_given=s.count("GIVEN"), n_derivable=s.count("DERIVABLE"), n_fetch=s.count("FETCH"), n_ask=s.count("ASK"), n_missing=s.count("MISSING"),
             n_cond=m, n_cond_given=c.count("GIVEN"), n_cond_fetch=c.count("FETCH"), n_cond_ask=c.count("ASK"), n_cond_unknown=c.count("UNKNOWN"))
    f["frac_not_given"] = 1 - f["n_given"] / n if n else np.nan
    f["frac_cond_not_given"] = 1 - f["n_cond_given"] / m if m else np.nan
    return f
def build():
    res = json.load(open(HERE / "gap_raw.json")); rows = {}
    for c, t in res.items():
        p = parse(t); rows[c] = feats(p) if isinstance(p, dict) else None
    out = {}
    for b, pre in [("sopbench", None), ("tau2", None), ("tauk_banking", "banking_knowledge")]:
        ids = [c for c in rows if (c.startswith("banking_knowledge") if b == "tauk_banking" else (not c.startswith("banking_knowledge") and ((c.count("/") == 2) == (b == "sopbench"))))]
        d = pd.DataFrame.from_dict({c: (rows[c] or {}) for c in ids}, orient="index").reindex(columns=GAP)
        miss = int(d.isna().all(axis=1).sum()); d = d.fillna(d.median()); d.index.name = "case_id"
        d.reset_index().to_csv(HERE / f"gap_{b}.csv", index=False); out[b] = (len(d), miss)
    print(out)
if __name__ == "__main__": build()
