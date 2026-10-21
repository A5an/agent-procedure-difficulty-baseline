import json, sys, pandas as pd
from astf import feats, COLS
for b in ["tau2", "sop"]:
    try: res = json.load(open(f"code_{b}.json"))
    except FileNotFoundError: continue
    rows, bad = {}, 0
    for c, src in res.items():
        try: rows[c] = feats(src, c.split("/")[0])
        except Exception: bad += 1
    df = pd.DataFrame(rows).T; df = df.reindex(res.keys()); df = df.fillna(df.median()); df.index.name = "case_id"
    name = "tau2" if b == "tau2" else "sopbench"
    df.reset_index().to_csv(f"cp_{name}.csv", index=False)
    print(name, len(df), "parse fails", bad); print(df.describe().T[["mean", "std", "min", "max"]])
