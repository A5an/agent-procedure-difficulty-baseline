"""Test-retest of the procedure rubric on the 100-task sample (same prompt, two independent calls)."""
import json, numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score
import build_features as B, prompts as P, llmrun as R
s = R.sample100()
ids = [c for _, c in s]
t1 = pd.DataFrame.from_dict({k: B.proc_row(v) for k, v in B.load("proc", 1).items()}, orient="index").reindex(ids)
t2 = pd.DataFrame.from_dict({k: B.proc_row(v) for k, v in B.load("proc", 2).items()}, orient="index").reindex(ids)
ok = t1.notna().all(1) & t2.notna().all(1)
t1, t2 = t1[ok], t2[ok]
res = {"n_tasks": int(ok.sum())}
rows = []
for c in t1.columns:
    a, b = t1[c].to_numpy(), t2[c].to_numpy()
    if c == "p_refusal":
        k = np.nan
    else:
        k = cohen_kappa_score(a.astype(int), b.astype(int)) if len(set(a) | set(b)) > 1 else np.nan
    kw = (cohen_kappa_score(a.astype(int), b.astype(int), weights="linear") if c != "p_refusal" and len(set(a) | set(b)) > 1 else np.nan)
    r = pearsonr(a, b)[0] if a.std() > 0 and b.std() > 0 else np.nan
    sp = spearmanr(a, b)[0] if a.std() > 0 and b.std() > 0 else np.nan
    rows.append({"feature": c, "kappa": k, "kappa_linear_weighted": kw, "pearson": r, "spearman": sp, "identical_share": float((a == b).mean())})
df = pd.DataFrame(rows).round(3)
# pooled kappa over the seven step-type count columns (counts are the categories)
tc = [f"n_{t}" for t in P.STEP_TYPES]
a = t1[tc].to_numpy().astype(int).ravel(); b = t2[tc].to_numpy().astype(int).ravel()
res["pooled_type_count_kappa"] = float(cohen_kappa_score(a, b))
res["pooled_type_count_kappa_linear"] = float(cohen_kappa_score(a, b, weights="linear"))
# label-sequence agreement: the step-type sequence identical?
seq = lambda d: [x["type"] for x in d["steps"]]
p1, p2 = B.load("proc", 1), B.load("proc", 2)
same = [seq(p1[i]) == seq(p2[i]) for i in t1.index]
res["identical_type_sequence_share"] = float(np.mean(same))
res["steps_total_corr"] = float(pearsonr(t1.n_steps, t2.n_steps)[0])
df.to_csv(R.HERE / "out" / "agreement.csv", index=False)
json.dump(res, open(R.HERE / "out" / "agreement.json", "w"), indent=1)
print(df.to_string()); print(res)
