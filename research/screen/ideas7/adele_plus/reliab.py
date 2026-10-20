import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
import lib, run_llm, prompts, build
from lib import DEMANDS
U, dom = run_llm.units(); R = run_llm.retest_set(U)
A, ad, texts = run_llm.anchors(U, dom)
def pf(var, b, tx):
    c = U[(b, tx)][0]
    if var == "C":
        s1, s2 = A[dom[c][:2]]; use = s2 if c in s1 else s1
        return prompts.prompt_C(tx, dom[c][1], [(texts[a], ad.loc[a].to_dict()) for a in use])
    return {"A": prompts.prompt_A, "B": prompts.prompt_B, "INT": prompts.prompt_INT}[var](tx)
def get(var, rep, temp=0.0):
    rows = {}
    for (b, tx) in R:
        t = build.raw(pf(var, b, tx), rep, temp)
        x = {"A": build.vecA, "B": build.vecB, "C": build.vecC, "INT": build.vecINT}[var](t)
        if var == "A": x = None if x[0] is None else (x[0], x[1])
        rows[(b, tx)] = x
    return rows
def frame(rows, which=None):
    d = {}
    for k, x in rows.items():
        if x is None: continue
        v = x[which] if which is not None else x
        d[k] = v
    return pd.DataFrame(d).T[DEMANDS]
def rel(a, b, domains):
    idx = a.index.intersection(b.index); a, b = a.loc[idx].astype(float), b.loc[idx].astype(float)
    dm = pd.Series([dom[U[k][0]][:2] for k in idx], index=idx)
    out = {}
    for d in DEMANDS:
        x, y = a[d], b[d]; ok = x.notna() & y.notna()
        if x[ok].nunique() < 2 or y[ok].nunique() < 2: out[d] = (np.nan, np.nan, np.nan, np.nan); continue
        r = spearmanr(x[ok], y[ok])[0]
        xr, yr = x - x.groupby(dm).transform("mean"), y - y.groupby(dm).transform("mean")
        rw = spearmanr(xr[ok], yr[ok])[0]
        out[d] = (r, rw, float((x[ok] == y[ok]).mean()), float(((x[ok] - y[ok]).abs() <= 1).mean()))
    return pd.DataFrame(out, index=["rho", "rho_within_domain", "exact", "within1"]).T
res = {}
# original ADeLe retest on the repository retest sample (subset of R)
import adele_annotate as AA
second = {}
for b in run_llm.BENCH:
    s = AA.levels(lib.REPO / "data" / "adele_retest" / f"{b}.jsonl"); first = AA.levels(lib.REPO / "data" / b / "adele_responses.jsonl").loc[s.index]
    for c in s.index: second[c] = (first.loc[c], s.loc[c])
orig1 = pd.DataFrame({U_k: second[U[U_k][0]][0] for U_k in R if U[U_k][0] in second}).T
orig2 = pd.DataFrame({U_k: second[U[U_k][0]][1] for U_k in R if U[U_k][0] in second}).T
tabs = {"orig (n=%d)" % len(orig1): rel(orig1, orig2, None)}
pairs = {}
fa0, fa1 = frame(get("A", 0), 0), frame(get("A", 1), 0); fm0, fm1 = frame(get("A", 0), 1), frame(get("A", 1), 1)
pairs["A expected"] = (fa0, fa1); pairs["A0 argmax(same call)"] = (fm0, fm1)
pairs["B checklist"] = (frame(get("B", 0)), frame(get("B", 1))); pairs["C anchored"] = (frame(get("C", 0)), frame(get("C", 1)))
S = [frame(get("INT", r + 10, 0.7)) for r in range(4)]
pairs["INT T=0.7 single sample"] = (S[0], S[1])
pairs["INT T=0.7 mean of 2 vs mean of 2"] = ((S[0] + S[1]) / 2, (S[2] + S[3]) / 2)
for k, (a, b) in pairs.items(): tabs[f"{k} (n={len(a.index.intersection(b.index))})"] = rel(a, b, None)
# same-45-task comparison: restrict variants to the tasks of the original retest
keep = orig1.index
tabs45 = {"orig": rel(orig1, orig2, None)}
for k, (a, b) in pairs.items(): tabs45[k] = rel(a.loc[a.index.intersection(keep)], b.loc[b.index.intersection(keep)], None)
lines = []
def summ(t): return {"mean_rho": float(np.nanmean(t.rho)), "median_rho": float(np.nanmedian(t.rho)), "mean_rho_within_domain": float(np.nanmean(t.rho_within_domain)), "mean_exact": float(np.nanmean(t.exact)), "mean_within1": float(np.nanmean(t.within1)), "n_scales": int(t.rho.notna().sum())}
out = {"all_retest_tasks": {k: summ(t) for k, t in tabs.items()}, "same_tasks_as_original_retest": {k: summ(t) for k, t in tabs45.items()}}
json.dump(out, open(lib.HERE / "reliability.json", "w"), indent=1)
for part in out:
    print(part)
    for k, v in out[part].items(): print(f"  {k:40s}", {a: round(b, 3) for a, b in v.items()})
pd.concat({k: t.rho for k, t in tabs45.items()}, axis=1).round(2).to_csv(lib.HERE / "reliability_per_scale_rho_45.csv")
print(pd.concat({k: t.rho for k, t in tabs45.items()}, axis=1).round(2).to_string())
