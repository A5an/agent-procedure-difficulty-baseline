"""Self-family check: does a forecaster rank tasks better for agents of its own family?
Per agent: within-domain Spearman between forecaster difficulty score and the agent's failure rate per task
(domain-size weighted). Then mean by forecaster x agent family, and the interaction with a paired bootstrap over tasks."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
R = ZR.REPO + "/data"
P = pd.read_csv(ZR.SCREEN + "/ideas10/panel/panel_scores.csv", index_col=0)
FC = {"sonnet": "claude", "opus": "claude", "haiku": "claude",
      "gemini-2.5-pro": "gemini", "gemini-3.8-flash": "gemini", "gemini-3.5-flash": "gemini"}
def fam(a):
    a = a.lower()
    for k in ["claude", "gemini", "gpt", "o4", "qwen", "glm", "grok", "kimi"]:
        if a.startswith(k): return {"o4": "gpt"}.get(k, k)
    return "other"
def load(b):
    rows = []
    for line in open(f"{R}/{b}/responses.jsonl"):
        d = json.loads(line)
        for t, v in d["responses"].items():
            rows.append((d["subject_id"], t, 1 - v["successes"] / v["trials"]))
    return pd.DataFrame(rows, columns=["agent", "case", "fail"]).pivot(index="case", columns="agent", values="fail")
def wd(x, y, dom):
    num = den = 0.0
    for d in dom.unique():
        m = (dom == d).to_numpy() & ~np.isnan(x) & ~np.isnan(y)
        if m.sum() < 3 or np.nanstd(x[m]) == 0 or np.nanstd(y[m]) == 0: continue
        num += spearmanr(x[m], y[m])[0] * m.sum(); den += m.sum()
    return num / den if den else np.nan
def table(F, idx):
    Fi, dom = F.loc[idx], P.loc[idx, "dom"]
    out = {}
    for f in FC:
        x = P.loc[idx, f].to_numpy(float)
        for a in F.columns:
            out[(f, a)] = wd(x, Fi[a].to_numpy(float), dom)
    s = pd.Series(out); s.index.names = ["fc", "agent"]; return s
rng = np.random.default_rng(0)
for b in ["tau2", "tauk_banking"]:
    F = load(b); idx = [c for c in F.index if c in P.index and P.loc[c, "bench"] == b]
    F = F.loc[idx]
    s = table(F, idx).reset_index(name="rho")
    s["fc_fam"] = s.fc.map(FC); s["ag_fam"] = s.agent.map(fam)
    print(f"\n== {b}: {len(idx)} tasks, agents by family:", s.drop_duplicates("agent").ag_fam.value_counts().to_dict())
    print(s.pivot_table(index="fc", columns="ag_fam", values="rho").round(4).to_string())
    # own-family advantage per forecaster: rho on own-family agents minus rho on all other agents
    def adv(s):
        r = {}
        for f, g in s.groupby("fc"):
            own = g[g.ag_fam == FC[f]].rho.mean(); oth = g[g.ag_fam != FC[f]].rho.mean(); r[f] = own - oth
        return pd.Series(r)
    a0 = adv(s)
    if s.ag_fam.isin(["gemini"]).any() and s.ag_fam.isin(["claude"]).any():
        # interaction: (claude fc on claude - claude fc on gemini) - (gemini fc on claude - gemini fc on gemini)
        def inter(s):
            m = s.pivot_table(index="fc_fam", columns="ag_fam", values="rho")
            return (m.loc["claude", "claude"] - m.loc["claude", "gemini"]) - (m.loc["gemini", "claude"] - m.loc["gemini", "gemini"])
        i0 = inter(s); bs = []
        for _ in range(300):
            bi = list(rng.choice(idx, len(idx), replace=True))
            Fb = F.loc[bi]; Fb.index = range(len(bi))
            Pb = P.loc[bi].reset_index(drop=True)
            o = {}
            for f in FC:
                x = Pb[f].to_numpy(float)
                for a in F.columns: o[(f, a)] = wd(x, Fb[a].to_numpy(float), Pb.dom)
            sb = pd.Series(o); sb.index.names = ["fc", "agent"]; sb = sb.reset_index(name="rho")
            sb["fc_fam"] = sb.fc.map(FC); sb["ag_fam"] = sb.agent.map(fam); bs.append(inter(sb))
        print(f"own-family interaction (claude-vs-gemini): {i0:+.3f} [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]")
    print("own minus other, by forecaster:", a0.round(4).to_dict())
