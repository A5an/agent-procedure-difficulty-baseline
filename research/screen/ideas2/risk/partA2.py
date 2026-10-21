"""Case-split reliability: is the procedure rate a property of the procedure or of the sampled cases? plus wrong_execution check."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
d = pd.read_csv("outcomes.csv")
EMPTY3 = ["gpt-5-mode_fc", "gpt-5-mini-mode_fc", "gemini-2.5-pro-mode_fc"]
d = d[~d.config.isin(EMPTY3)]
lab = {"unsafe": 0, "wrongful_refusal": 1, "wrong_execution": 1, "process_only": None, "fail": None}
d["fail"] = 1 - d.success
for k in lab: d[k] = (d.otype == k).astype(int) if k != "fail" else d.fail
rng = np.random.default_rng(3); rows = []
for k, l in lab.items():
    z = d if l is None else d[d.should == l]
    cr = z.groupby(["procedure_id", "case_id"])[k].mean().reset_index()
    sizes = cr.groupby("procedure_id").size()
    for mn in (4, 6, 10):
        procs = sizes[sizes >= mn].index
        g = {p: cr[cr.procedure_id == p][k].values for p in procs}
        rs = []
        for _ in range(1000):
            a, b = [], []
            for p in procs:
                v = rng.permutation(g[p]); h = len(v) // 2
                a.append(v[:h].mean()); b.append(v[h:2 * h].mean())
            if np.std(a) > 0 and np.std(b) > 0:
                rs.append(spearmanr(a, b).correlation)
        r = np.nanmean(rs)
        rows.append(dict(rate=k, min_cases=mn, n_procs=len(procs), half_r=round(r, 3), lo=round(np.nanpercentile(rs, 2.5), 3),
                         hi=round(np.nanpercentile(rs, 97.5), 3), spearman_brown=round(2 * r / (1 + r), 3)))
res = pd.DataFrame(rows); print(res.to_string())
res.to_csv("case_split.csv", index=False)
# wrong execution: where does it concentrate
w = d[d.should == 1].groupby("procedure_id").agg(n=("case_id", "nunique"), we=("wrong_execution", "mean"), wr=("wrongful_refusal", "mean"), po=("process_only", "mean"), ok=("success", "mean"))
print(w.sort_values("we", ascending=False).head(12).round(2).to_string())
print("perform-case process_only detail", d[(d.should == 1) & (d.otype == "process_only")][["dirg", "ntce", "cnv", "dbm", "asc"]].mean().round(3).to_dict())
