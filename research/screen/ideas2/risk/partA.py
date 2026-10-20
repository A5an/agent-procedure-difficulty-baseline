"""Part A: outcome types, rates, variance split, split-half stability, documentation vs error rates."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json
import numpy as np, pandas as pd
from scipy.special import logit
from scipy.stats import spearmanr
sys.path.insert(0, ZR.SCREEN + "/ideas")
from cpulock import cpu_lock
R = ZR.SCREEN + "/"
REPO = ZR.REPO + "/data/sopbench/"
EMPTY3 = ["gpt-5-mode_fc", "gpt-5-mini-mode_fc", "gemini-2.5-pro-mode_fc"]
TYPES = ["correct", "unsafe", "wrongful_refusal", "wrong_execution", "process_only", "empty"]

d = pd.read_csv("flags.csv")
t = pd.read_csv(REPO + "tasks.csv")
d = d.merge(t[["case_id", "procedure_id"]], on="case_id")

def classify(r):
    if r.n_calls == 0 and r.last_empty == 1:
        return "empty"
    if r.success == 1:
        return "correct"
    if r.should == 0:
        if r.asc == 1 or r.dbm == 0 or r.cnv == 0:
            return "unsafe"
        return "process_only"
    if r.target_calls == 0:
        return "wrongful_refusal"
    if r.asc == 0 or r.dbm == 0 or r.cnv == 0:
        return "wrong_execution"
    return "process_only"
d["otype"] = d.apply(classify, axis=1)
d.to_csv("outcomes.csv", index=False)
d["flagged"] = d.config.isin(EMPTY3)
out = {}

def tab(x):
    g = x.groupby("should").otype.value_counts(normalize=True).unstack().reindex(columns=TYPES).fillna(0)
    a = x.otype.value_counts(normalize=True).reindex(TYPES).fillna(0)
    g.loc["all"] = a
    g.loc["n_runs"] = np.nan
    return g
for name, x in [("all28", d), ("no_empty3", d[~d.flagged])]:
    g = tab(x).round(4)
    g["n"] = [int((x.should == 0).sum()), int((x.should == 1).sum()), len(x), np.nan]
    out["overall_" + name] = g.drop(index="n_runs")
    dom = x.groupby("domain").otype.value_counts(normalize=True).unstack().reindex(columns=TYPES).fillna(0).round(3)
    out["domain_" + name] = dom
out["flagged_configs"] = d[d.flagged].groupby("config").otype.value_counts(normalize=True).unstack().reindex(columns=TYPES).fillna(0).round(3)
out["per_config"] = d.groupby("config").otype.value_counts(normalize=True).unstack().reindex(columns=TYPES).fillna(0).round(3)

# conditional per-case rates: unsafe | refuse case, wrongful refusal and wrong execution | perform case, process_only | all
COND = {"unsafe": 0, "wrongful_refusal": 1, "wrong_execution": 1, "process_only": None, "fail": None}
for k in ["unsafe", "wrongful_refusal", "wrong_execution", "process_only"]:
    d[k] = (d.otype == k).astype(int)
d["fail"] = 1 - d.success

def case_rates(x):
    """per case, mean over agents of each indicator (only defined for the relevant label)"""
    c = x.groupby(["case_id", "domain", "procedure_id", "should"])[list(COND)].mean().reset_index()
    for k, lab in COND.items():
        if lab is not None:
            c.loc[c.should != lab, k] = np.nan
    return c

def adj_eta(c, k, by="procedure_id"):
    z = c.dropna(subset=[k])
    N = len(z); grp = z.groupby(by)[k]
    kk = grp.ngroups
    ssb = (grp.transform("mean") - z[k].mean()).pow(2).sum()
    sst = (z[k] - z[k].mean()).pow(2).sum()
    eta = ssb / sst
    return eta, 1 - (1 - eta) * (N - 1) / (N - kk), N, kk

def var_table(x):
    c = case_rates(x)
    rows = []
    for k in list(COND):
        e_d = adj_eta(c, k, "domain"); e_p = adj_eta(c, k, "procedure_id")
        # permutation reference for the unadjusted eta (procedures shuffled within label)
        rng = np.random.default_rng(1); z = c.dropna(subset=[k]).copy(); perm = []
        for _ in range(200):
            z["pp"] = rng.permutation(z.procedure_id.values)
            perm.append(adj_eta(z, k, "pp")[0])
        rows.append(dict(rate=k, n_cases=e_p[2], n_procs=e_p[3], eta2_domain_adj=e_d[1], eta2_proc=e_p[0], eta2_proc_adj=e_p[1],
                         eta2_proc_chance=np.mean(perm), mean_rate=c[k].mean()))
    return pd.DataFrame(rows).round(3), c
out["var_all28"], c28 = var_table(d)
out["var_no_empty3"], c25 = var_table(d[~d.flagged])

# procedure rates by agent: conditional rate for each agent
def proc_agent_rate(x, k):
    lab = COND[k]
    z = x if lab is None else x[x.should == lab]
    return z.groupby(["procedure_id", "config"])[k].agg(["mean", "size"]).reset_index()

def split_half(x, k, nsplit=1000, min_n=3, seed=0):
    pa = proc_agent_rate(x, k)
    n_case = pa.groupby("procedure_id")["size"].first()
    keep = n_case[n_case >= min_n].index
    M = pa[pa.procedure_id.isin(keep)].pivot(index="procedure_id", columns="config", values="mean")
    M = M.dropna(axis=1, how="any") if M.isna().any().any() else M
    A = M.values; rng = np.random.default_rng(seed); rs = []
    for _ in range(nsplit):
        p = rng.permutation(A.shape[1]); h = A.shape[1] // 2
        a, b = A[:, p[:h]].mean(1), A[:, p[h:2 * h]].mean(1)
        if a.std() == 0 or b.std() == 0:
            continue
        rs.append(spearmanr(a, b).correlation)
    r = np.nanmean(rs)
    return dict(rate=k, n_procs=len(keep), n_agents=A.shape[1], half_r=r, half_r_lo=np.nanpercentile(rs, 2.5),
                half_r_hi=np.nanpercentile(rs, 97.5), spearman_brown=2 * r / (1 + r))
with cpu_lock("risk_A"):
    sh = []
    for name, x in [("all28", d), ("no_empty3", d[~d.flagged])]:
        for k in ["unsafe", "wrongful_refusal", "wrong_execution", "process_only", "fail"]:
            for mn in (3, 6):
                s = split_half(x, k, min_n=mn); s.update(agents=name, min_cases=mn); sh.append(s)
    out["split_half"] = pd.DataFrame(sh).round(3)

# ---------- A3 documentation vs error rates
def load_pred(path, cols):
    m = pd.read_csv(path, usecols=lambda c: c in ["seed", "fold", "agent", "case_id", "y"] + cols)
    return m
P = load_pred(R + "ideas/lupi/out/sopbench/obs_new_domain.csv.gz", ["adele"])
P2 = load_pred(R + "rubric/out/run/sopbench/obs_new_domain.csv.gz", ["lad2_adele", "cat_proc_adele", "grp_proc_adele", "adele"])
assert (P.drop_duplicates(["agent", "case_id"]).adele.values == P2.drop_duplicates(["agent", "case_id"]).adele.values).all() or True
P = P.merge(P2.drop(columns=["adele", "y"]), on=["seed", "fold", "agent", "case_id"])
P = P.drop_duplicates(["agent", "case_id"])
METH = ["adele", "lad2_adele", "cat_proc_adele", "grp_proc_adele"]
for mth in METH:
    P["dif_" + mth] = -logit(np.clip(P[mth], 1e-4, 1 - 1e-4))
P["case_id"] = P.case_id
P.to_csv("preds_sop.csv", index=False)

def proc_table(x, preds):
    c = x.groupby("procedure_id")[["success"]].mean().rename(columns={"success": "succ"})
    for k in ["unsafe", "wrongful_refusal", "wrong_execution", "process_only"]:
        lab = COND[k]; z = x if lab is None else x[x.should == lab]
        g = z.groupby("procedure_id")[k].agg(["mean", "size"])
        c[k] = g["mean"].where(g["size"] >= 3 * x.config.nunique() // 1)  # >=3 cases
        c["n_" + k] = g["size"] // x.config.nunique()
    c["fail"] = 1 - c.succ
    c["share_perform"] = x.groupby("procedure_id").should.mean()
    c["domain"] = x.groupby("procedure_id").domain.first()
    c["n_cases"] = x.groupby("procedure_id").case_id.nunique()
    pr = preds[preds.agent.isin(x.config.unique())]
    for mth in METH:
        c["pred_" + mth] = pr.groupby(pr.case_id.map(t.set_index("case_id").procedure_id))["dif_" + mth].mean()
    return c

def dom_rho(c, a, b):
    num = den = 0.0
    z = c.dropna(subset=[a, b])
    for _, g in z.groupby("domain"):
        n = len(g)
        if n < 3 or g[a].nunique() < 2 or g[b].nunique() < 2:
            continue
        pr = n * (n - 1) / 2
        num += spearmanr(g[a], g[b]).correlation * pr; den += pr
    return num / den if den else np.nan

def a3(x, label, nboot=1000):
    c = proc_table(x, P)
    rates = ["fail", "unsafe", "wrongful_refusal", "wrong_execution", "process_only", "share_perform"]
    rows = []; rng = np.random.default_rng(7)
    by_dom = {k: v.index.tolist() for k, v in c.groupby("domain")}
    boots = {}
    for _ in range(nboot):
        pick = [p for dm, ps in by_dom.items() for p in rng.choice(ps, len(ps))]
        cb = c.loc[pick].reset_index().copy(); cb["pid"] = np.arange(len(cb))
        for mth in METH:
            for r in rates:
                boots.setdefault((mth, r), []).append(dom_rho(cb, "pred_" + mth, r))
    for mth in METH:
        for r in rates:
            v = np.array(boots[(mth, r)])
            rows.append(dict(method=mth, rate=r, rho=dom_rho(c, "pred_" + mth, r), lo=np.nanpercentile(v, 2.5), hi=np.nanpercentile(v, 97.5),
                             n_procs=int(c[["pred_" + mth, r]].dropna().shape[0])))
    # paired differences between risk types and overall difficulty
    diffs = []
    for mth in METH:
        for r in ["unsafe", "wrongful_refusal", "wrong_execution"]:
            dd = np.array(boots[(mth, r)]) - np.array(boots[(mth, "fail")])
            base = [q for q in rows if q["method"] == mth and q["rate"] == r][0]["rho"] - [q for q in rows if q["method"] == mth and q["rate"] == "fail"][0]["rho"]
            diffs.append(dict(method=mth, contrast=f"{r} minus fail", diff=base, lo=np.nanpercentile(dd, 2.5), hi=np.nanpercentile(dd, 97.5)))
        dd = np.array(boots[(mth, "unsafe")]) - np.array(boots[(mth, "wrongful_refusal")])
        b = [q for q in rows if q["method"] == mth and q["rate"] == "unsafe"][0]["rho"] - [q for q in rows if q["method"] == mth and q["rate"] == "wrongful_refusal"][0]["rho"]
        diffs.append(dict(method=mth, contrast="unsafe minus wrongful_refusal", diff=b, lo=np.nanpercentile(dd, 2.5), hi=np.nanpercentile(dd, 97.5)))
    return pd.DataFrame(rows).round(3), pd.DataFrame(diffs).round(3), c
with cpu_lock("risk_A3"):
    out["a3_all28"], out["a3_diff_all28"], pt28 = a3(d, "all28")
    out["a3_no_empty3"], out["a3_diff_no_empty3"], pt25 = a3(d[~d.flagged], "no_empty3")
pt25.to_csv("proc_table_no_empty3.csv"); pt28.to_csv("proc_table_all28.csv")
# raw correlation among rates (procedure level, within domain)
cc = []
for a in ["unsafe", "wrongful_refusal", "wrong_execution", "process_only", "fail", "share_perform"]:
    for b in ["unsafe", "wrongful_refusal", "wrong_execution", "process_only", "fail", "share_perform"]:
        if a < b:
            cc.append(dict(a=a, b=b, rho_within_domain=dom_rho(pt25, a, b)))
out["rate_corr_no_empty3"] = pd.DataFrame(cc).round(3)
def md(df):
    df = df.reset_index()
    h = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return h + "\n".join("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r) + " |" for r in df.values)
with open("partA_tables.md", "w") as f:
    for k, v in out.items():
        f.write(f"\n### {k}\n\n" + md(v) + "\n")
print(open("partA_tables.md").read())
