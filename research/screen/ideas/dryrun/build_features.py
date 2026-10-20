"""Cache -> features/dry_<bench>.csv (case_id first). Failed parses imputed by benchmark median, counted."""
import json, numpy as np, pandas as pd
import lib
from collections import defaultdict
COLS = ["p_refuse", "p_perform", "n_conditions", "n_violated_by_request", "n_needs_system_data", "n_lookups",
        "n_customer_facts", "ambiguous_request", "ambiguous_policy", "decision_entropy", "refuse_share_sampled"]

def build():
    rec = defaultdict(dict)
    for b, cid, text in lib.statements():   # duplicate prompts share one cached call
        for kind, k in [("main", 0)] + [("dec", k) for k in range(1, 5)]:
            r = lib.one(kind, cid, text, k)
            if r["ok"]:
                rec[cid][(kind, k)] = r["parsed"]
    rep = {}
    for b in ("sopbench", "tau2"):
        ids = pd.read_csv(lib.REPO / "data" / b / "tasks.csv").case_id.tolist()
        rows = {}
        for cid in ids:
            d = rec[cid]; row = dict.fromkeys(COLS, np.nan)
            m = d.get(("main", 0))
            decs = [d[("dec", k)] for k in range(1, 5) if ("dec", k) in d]
            if m:
                p = m["outcome_probs"]; s = sum(float(p[k]) for k in lib.DEC)
                row.update(p_refuse=float(p["refuse"]) / s, p_perform=float(p["perform"]) / s,
                           n_conditions=len(m["conditions"]),
                           n_violated_by_request=sum(c["status"] == "violated_by_request" for c in m["conditions"]),
                           n_needs_system_data=sum(c["status"] == "needs_system_data" for c in m["conditions"]),
                           n_lookups=int(m["n_lookups"]), n_customer_facts=int(m["n_customer_facts"]),
                           ambiguous_request=int(m["ambiguous_request"]), ambiguous_policy=int(m["ambiguous_policy"]))
                top = max(lib.DEC, key=lambda k: float(p[k]))
                alld = decs + [top]
            else:
                alld = decs
            if len(alld) >= 3:
                c = np.array([alld.count(k) for k in lib.DEC], float) / len(alld)
                row["decision_entropy"] = float(-(c[c > 0] * np.log(c[c > 0])).sum())
                row["refuse_share_sampled"] = float(c[1])
            rows[cid] = row
        df = pd.DataFrame.from_dict(rows, orient="index")
        fails = int(df.isna().any(axis=1).sum()); df.index.name = "case_id"
        rep[b] = {"n": len(df), "rows_with_any_missing": fails, "main_missing": int(df.p_refuse.isna().sum()),
                  "entropy_missing": int(df.decision_entropy.isna().sum())}
        df = df.fillna(df.median())
        df.reset_index().to_csv(lib.HERE / "features" / f"dry_{b}.csv", index=False)
    print(json.dumps(rep)); return rep

if __name__ == "__main__":
    build()
