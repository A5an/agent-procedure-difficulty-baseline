"""Within-agent metrics with paired bootstrap over procedures (seed 0, 300 draws).
AUC_w  : AUC inside each (fold, agent), mean over groups weighted by size (the protocol's auc_within_config).
LLskill: per agent, 1 - logloss(method)/logloss(share_none), where share_none is the agent's own training share
         (a calibrated per-agent constant); mean over the 28 agents with equal weight."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, numpy as np, pandas as pd
from scipy.stats import rankdata
OUT = ZR.SCREEN + "/pertype/run_abl"
ty = pd.read_csv("types.csv").set_index("case_id")
METH = ["share_none", "constant", "adele"] + [p + x for x in ["fam", "tree", "dom", "famtree"] for p in ["main_", "inter_"]]
rng = np.random.default_rng(7)

def auc_groups(y, s, gid, ng):
    out = np.full(ng, np.nan); w = np.zeros(ng)
    order = np.argsort(gid, kind="stable"); gid, y, s = gid[order], y[order], s[order]
    b = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1], True])
    for a, z in zip(b[:-1], b[1:]):
        yy = y[a:z]; n1 = yy.sum(); n0 = len(yy) - n1
        if n1 and n0:
            r = rankdata(s[a:z]); out[gid[a]] = (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0); w[gid[a]] = len(yy)
    return out, w

def stats(m, cols, agents):
    y = m.y.values
    gid = (m.fold.values * len(agents) + m.agent.map({a: i for i, a in enumerate(agents)}).values).astype(int)
    ng = gid.max() + 1
    res = {}
    ag = m.agent.map({a: i for i, a in enumerate(agents)}).values
    ll = {}
    for c in cols:
        p = np.clip(m[c].values, 1e-4, 1 - 1e-4)
        l = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        ll[c] = np.bincount(ag, l, len(agents)) / np.maximum(np.bincount(ag, minlength=len(agents)), 1)
    for c in cols:
        a, w = auc_groups(y, m[c].values, gid, ng)
        ok = ~np.isnan(a)
        res[c] = dict(auc_w=float(np.average(a[ok], weights=w[ok])), ll_skill_agent=float(np.mean(1 - ll[c] / ll["share_none"])),
                      logloss=float(np.average([ll[c][i] for i in range(len(agents))], weights=np.bincount(ag, minlength=len(agents)))))
    return res

out = {}
for scheme in ["new_procedures", "new_domain"]:
    m = pd.read_csv(f"{OUT}/sopbench/obs_{scheme}.csv.gz")
    m0_ = pd.read_csv(f"{OUT[:-4]}/sopbench/obs_{scheme}.csv.gz")[["seed", "fold", "agent", "case_id", "share_none"]]
    m = m.merge(m0_, on=["seed", "fold", "agent", "case_id"])
    cols = [c for c in METH if c in m]
    agents = sorted(m.agent.unique())
    # point estimates: mean over seeds
    pts = [stats(g, cols, agents) for _, g in m.groupby("seed")]
    point = {c: {k: float(np.mean([p[c][k] for p in pts])) for k in pts[0][c]} for c in cols}
    m0 = m[m.seed == 0].reset_index(drop=True)
    m0["proc"] = ty.loc[m0.case_id, "procedure_id"].values
    idx = {p: g.index.values for p, g in m0.groupby("proc")}
    procs = list(idx)
    boots = {c: [] for c in cols}
    for _ in range(300):
        pick = rng.choice(procs, len(procs))
        s = stats(m0.iloc[np.concatenate([idx[p] for p in pick])], cols, agents)
        for c in cols: boots[c].append(s[c])
    res = {}
    for c in cols:
        r = dict(point[c])
        for k in ["auc_w", "ll_skill_agent", "logloss"]:
            ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            r[k + "_ci"] = ci([b[k] for b in boots[c]])
            if c != "adele":
                r[k + "_minus_adele_ci"] = ci([b[k] - a[k] for b, a in zip(boots[c], boots["adele"])])
            if c.startswith("inter_"):
                mm = "main_" + c[6:]
                r[k + "_minus_main_ci"] = ci([b[k] - a[k] for b, a in zip(boots[c], boots[mm])])
        res[c] = r
    out[scheme] = res
json.dump(out, open("within_abl.json", "w"), indent=1)
for s, r in out.items():
    print(s)
    for c, v in r.items():
        print(f"  {c:15s} aucw {v['auc_w']:.3f} llsk {v['ll_skill_agent']:+.3f} dAUC vs main {[round(x,3) for x in v.get('auc_w_minus_main_ci',[])]} dLLsk vs main {[round(x,3) for x in v.get('ll_skill_agent_minus_main_ci',[])]}")
