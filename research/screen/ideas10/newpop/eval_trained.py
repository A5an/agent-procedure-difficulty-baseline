"""Trained out-of-fold predictions (tau2 new_domain and new_procedures, protocol cells) vs the independent 30-agent population."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import pandas as pd, numpy as np, sys
from scipy.special import logit
from scipy.stats import spearmanr
R = ZR.REPO + "/data/tau2"
t = pd.read_csv(f"{R}/tasks.csv")[["case_id", "domain", "procedure_id"]]
n = pd.read_csv("newpop/newpop_tau2.csv").set_index("case_id"); t["ynew"] = 1 - t.case_id.map(n.newpop_success)
t["b"] = t.case_id.map(pd.read_csv(f"{R}/irt/1d_1pl/items.csv", index_col=0).b)
cols = ["adele", "g1_clad_gap", "c1_all", "f2_all_fc"]
rng = np.random.default_rng(11)
for sch in ["new_procedures", "new_domain"]:
    m = pd.read_csv(f"fc/out/tau2/obs_{sch}.csv.gz").drop_duplicates(["seed", "fold", "case_id"]).drop(columns=["y"])
    for c in cols: m[c] = -logit(m[c].clip(1e-4, 1 - 1e-4))
    m = m.merge(t, on="case_id")
    m = m[m.seed == 0]
    procs = m.procedure_id.unique(); gi = {p: np.where(m.procedure_id.values == p)[0] for p in procs}
    def rho(df, col, tgt):
        num = den = 0
        for _, h in df.groupby(["fold", "domain"]):
            k = len(h)
            if k < 3: continue
            w = k * (k - 1) / 2; r = spearmanr(h[col], h[tgt]).correlation
            if r == r: num += r * w
            den += w
        return num / den
    D = {c: [] for c in cols}
    for _ in range(1000):
        idx = np.concatenate([gi[p] for p in rng.choice(procs, len(procs))]); h = m.iloc[idx]
        for c in cols: D[c].append(rho(h, c, "ynew"))
    for c in cols:
        d = np.array(D[c]); s = f"{sch:15s} {c:12s} vs new pop {rho(m, c, 'ynew'):+.3f} [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]"
        if c != "adele": dd = d - np.array(D["adele"]); s += f"  vs baseline {rho(m,c,'ynew')-rho(m,'adele','ynew'):+.3f} [{np.percentile(dd,2.5):+.3f},{np.percentile(dd,97.5):+.3f}]"
        if c == "f2_all_fc": dd = d - np.array(D["g1_clad_gap"]); s += f"  vs g1 {rho(m,c,'ynew')-rho(m,'g1_clad_gap','ynew'):+.3f} [{np.percentile(dd,2.5):+.3f},{np.percentile(dd,97.5):+.3f}]"
        print(s)
