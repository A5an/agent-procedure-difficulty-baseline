"""Steps 3 and 5: reliability table, family x type residuals, failure decomposition."""
import json, numpy as np, pandas as pd
from scipy.special import expit, logit
FLAG = ["gpt-5-mode_fc", "gpt-5-mini-mode_fc", "gemini-2.5-pro-mode_fc"]
so = pd.read_csv("step_outcomes.csv.gz")
co = pd.read_csv("case_outcomes.csv.gz")
so = so.merge(co[["agent", "case_id", "success", "should", "tree_ok"]], on=["agent", "case_id"])


def fam(a):
    for k, v in [("claude-3-5", "claude-3.5"), ("claude-3-7", "claude-3.7"), ("gemini-1.5", "gemini-1.5/2.0"), ("gemini-2.0", "gemini-1.5/2.0"),
                 ("gemini-2.5-flash", "gemini-2.5-flash"), ("gemma", "gemma"), ("gpt-4", "gpt-4.x"), ("o4-mini", "o4-mini"), ("llama", "llama"),
                 ("qwen2.5", "qwen2.5"), ("qwen3.5", "qwen3.5"), ("deepseek", "deepseek-r1")]:
        if a.startswith(k):
            return v
    return "flagged"
so["family"] = so.agent.map(fam)
so.loc[so.agent.isin(FLAG), "family"] = "flagged"
TYPES = ["auth", "existence", "threshold", "status", "datetime", "relation", "availability", "or_group", "action"]
M = 10  # shrinkage pseudo-count toward the agent's overall step rate
ok = so[~so.agent.isin(FLAG)]
out = []
# reliability table per agent and type, with shrinkage
rows = []
for a, g in so.groupby("agent"):
    ra = g.step_correct.mean()
    for t in TYPES:
        h = g[g.step_type == t]
        n, k = len(h), h.step_correct.sum()
        rows.append(dict(agent=a, family=fam(a) if a not in FLAG else "flagged", type=t, n=n, raw=k / n if n else np.nan, shrunk=(k + M * ra) / (n + M), overall=ra, flagged=a in FLAG))
tab = pd.DataFrame(rows)
tab.to_csv("reliability_table.csv", index=False)
piv = tab.pivot(index="agent", columns="type", values="shrunk")[TYPES]
piv["overall"] = tab.groupby("agent").overall.first()
piv.round(3).to_csv("reliability_table_wide.csv")
# additive expectation: step type effect x agent effect (logit), residual by family
agents = sorted(ok.agent.unique())
def fit_add(df):
    # agent effect and type effect by simple IPF on logit scale via logistic regression with sklearn-free Newton
    from sklearn.linear_model import LogisticRegression
    X = pd.get_dummies(df[["agent", "step_type"]]).astype(float)
    lr = LogisticRegression(C=1e4, max_iter=300)
    lr.fit(X.values, df.step_correct.values)
    return lr.predict_proba(X.values)[:, 1]
ok = ok.copy()
ok["exp"] = fit_add(ok)
ok["res"] = ok.step_correct - ok.exp
fr = ok.groupby(["family", "step_type"]).res.mean().unstack()[TYPES] * 100
# bootstrap over cases for CI of family x type residual (cases resampled, all agents together)
rng = np.random.default_rng(1)
cases = ok.case_id.unique()
idx = {c: i for i, c in enumerate(cases)}
ok["ci"] = ok.case_id.map(idx)
fams = sorted(ok.family.unique())
cellsum = ok.groupby(["ci", "family", "step_type"]).res.agg(["sum", "count"]).reset_index()
B = 300
boot = np.zeros((B, len(fams), len(TYPES)))
arr_sum = np.zeros((len(cases), len(fams), len(TYPES))); arr_n = np.zeros_like(arr_sum)
for r in cellsum.itertuples():
    arr_sum[r.ci, fams.index(r.family), TYPES.index(r.step_type)] = r.sum
    arr_n[r.ci, fams.index(r.family), TYPES.index(r.step_type)] = r.count
for b in range(B):
    s = rng.integers(0, len(cases), len(cases))
    boot[b] = arr_sum[s].sum(0) / np.maximum(arr_n[s].sum(0), 1)
lo, hi = np.percentile(boot, 2.5, 0) * 100, np.percentile(boot, 97.5, 0) * 100
res_rows = []
for i, f in enumerate(fams):
    for j, t in enumerate(TYPES):
        n = int(arr_n[:, i, j].sum())
        res_rows.append(dict(family=f, type=t, n=n, resid_pp=fr.loc[f, t], lo=lo[i, j], hi=hi[i, j], sig=int(lo[i, j] > 0 or hi[i, j] < 0)))
pd.DataFrame(res_rows).to_csv("family_type_residuals.csv", index=False)
# type means over non-flagged agents
tm = ok.groupby("step_type").step_correct.mean()[TYPES]
print("type means", tm.round(3).to_dict())
print(piv.round(2).to_string())
print(fr.round(1).to_string())
print("significant family x type cells:", int(sum(r["sig"] for r in res_rows)), "of", len(res_rows))
# Overall interaction strength: share of step-level variance explained by agent+type vs agent x type
from sklearn.metrics import log_loss
full = ok.groupby(["agent", "step_type"]).step_correct.transform("mean")
print("log loss agent+type", log_loss(ok.step_correct, ok.exp), "agent x type (in sample)", log_loss(ok.step_correct, np.clip(full, 1e-4, 1 - 1e-4)))
# tool kind x step: tool-less steps etc.
print(ok.groupby("kind").step_correct.mean().round(3).to_dict())

# ---------- step 5: failure decomposition ----------
cs = co.copy()
cs["flag"] = cs.agent.isin(FLAG)
unver = so[(so.step_type != "action") & (so.ordered == False)].groupby(["agent", "case_id"]).size().rename("n_unver")
cs = cs.merge(unver.reset_index(), on=["agent", "case_id"], how="left").fillna({"n_unver": 0})
blk = so[so.blocking == 1].groupby(["agent", "case_id"]).agg(n_blk=("step_id", "size"), n_blk_ver=("ordered", lambda x: int((x == True).sum()))).reset_index()
cs = cs.merge(blk, on=["agent", "case_id"], how="left").fillna({"n_blk": 0, "n_blk_ver": 0})
cs["family"] = cs.agent.map(fam); cs.loc[cs.flag, "family"] = "flagged"
def cat(r):
    if r.success == 1:
        return "success"
    if r.should == 1:
        if r.noint or r.empty:
            return "P0 empty or no run"
        if not r.act_success and not r.act_called:
            return "P1 gave up after a skipped check" if r.n_unver > 0 else "P2 refused despite all checks done"
        if not r.act_success:
            return "P3 action called but rejected by the environment"
        if not r.dirgraph:
            return "P4a acted with a rule-tree check missing" if r.n_unver > 0 else "P4b acted, only a graph-only prerequisite missing"
        if not r.args_ok:
            return "P5 wrong action arguments"
        return "P6 other (side effects, tool errors)"
    else:
        if r.act_success:
            if r.tree_ok == 0 or r.n_blk == 0:
                return "R4 acted, failure not visible in the rule tree"
            if r.n_blk_ver == 0:
                return "R1 acted, skipped the failing check"
            if r.n_blk_ver < r.n_blk:
                return "R2 acted, checked only some failing conditions"
            return "R3 acted although it checked every failing condition"
        if not r.dirgraph:
            return "R5a refused, but a call broke the graph order (e.g. login without lookup)"
        if not r.db_match or not r.constraint_ok:
            return "R5b refused, but side effects changed the database"
        return "R5c refused, other (tool errors)"
cs["cat"] = cs.apply(cat, axis=1)
cs.to_csv("case_categories.csv.gz", index=False)
f = cs[(cs.success == 0)]
def tbl(df, name):
    t = df.groupby(["should", "cat"]).size().rename("n").reset_index()
    t["share_of_failures_in_group"] = t.n / t.groupby("should").n.transform("sum")
    print(name); print(t.round(3).to_string(index=False)); return t
t_all = tbl(f[~f.flag], "ALL non-flagged agents, failures")
t_all.to_csv("failure_decomposition.csv", index=False)
t_flag = tbl(f[f.flag], "flagged agents")
# by family
fam_rows = []
for (fm, sh), g in f[~f.flag].groupby(["family", "should"]):
    for c, n in g.cat.value_counts().items():
        fam_rows.append(dict(family=fm, should=sh, cat=c, n=n, share=n / len(g)))
pd.DataFrame(fam_rows).to_csv("failure_decomposition_by_family.csv", index=False)
nf = cs[~cs.flag]
print("must-perform: failures", int(((nf.should == 1) & (nf.success == 0)).sum()), "of", int((nf.should == 1).sum()))
print("must-refuse: failures", int(((nf.should == 0) & (nf.success == 0)).sum()), "of", int((nf.should == 0).sum()))
json.dump({"n_flag_rows": int(cs.flag.sum())}, open("tmp_meta.json", "w"))
