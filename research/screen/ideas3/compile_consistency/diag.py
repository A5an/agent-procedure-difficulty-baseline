import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, json, itertools, collections
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = ZR.REPO + "/data/sopbench/"
t = pd.read_csv(REPO + "tasks.csv"); t["b"] = t.case_id.map(pd.read_csv(REPO + "irt/1d_1pl/items.csv", index_col=0).b)
cf = pd.read_csv(HERE + "/cons_sopbench.csv").set_index("case_id")
res = json.load(open(ZR.SCREEN + "/ideas2/compile/results_A.json"))
unit = pd.Series({r["case_id"]: r["unit"] for r in res}); t["unit"] = t.case_id.map(unit)
uf = pd.read_csv(HERE + "/unit_features.csv").set_index("unit")
# compile failure rate on real cases (earlier, L2, gold-scored) for comparison
fail = pd.Series({r["case_id"]: 1 - float(r["success"]) for r in res}); t["cfail"] = t.case_id.map(fail)
pu = t.groupby(["domain", "procedure_id", "unit"]).agg(b=("b", "mean"), n=("b", "size"), cfail=("cfail", "mean")).reset_index().merge(uf, left_on="unit", right_index=True)
pp = pu.groupby(["domain", "procedure_id"]).agg(b=("b", "mean"), cfail=("cfail", "mean"), **{c: (c, "mean") for c in ["dec_dis", "dec_ent", "dec_pair", "tool_pair", "call_pair", "crash", "tool_err", "n_fields", "fields_pair", "len_mean", "len_cv", "branch_mean", "cond_pair", "cond_n_diff", "toolref_pair"]}).reset_index()
pp.to_csv(HERE + "/procedure_table.csv", index=False)
out = []
feats = [c for c in pp.columns if c not in ("domain", "procedure_id", "b")]
def wrho(df, c):
    num = den = 0
    for d, g in df.groupby("domain"):
        n = len(g)
        if n < 3 or g[c].nunique() < 2: continue
        pr = n * (n - 1) / 2; num += spearmanr(g[c], g.b).correlation * pr; den += pr
    return num / den if den else np.nan
out.append("Procedure level (n=%d), difficulty b = mean IRT difficulty (higher = harder). Spearman of feature with b: pooled and within-domain (pair weighted)" % len(pp))
for c in feats:
    out.append(f"  {c:12s} pooled {spearmanr(pp[c], pp.b).correlation:+.3f}  within-domain {wrho(pp, c):+.3f}")
out.append("Unit level (n=%d units, per-unit mean b):" % len(pu))
for c in ["dec_dis", "tool_pair", "crash", "n_fields", "cond_pair", "len_mean", "branch_mean", "cfail"]:
    out.append(f"  {c:12s} pooled {spearmanr(pu[c], pu.b).correlation:+.3f}")
# correlation with real compile failure rate (rho 0.49 earlier claim)
out.append("cfail (real compile failure, gold) vs b at procedure level: %+.3f" % spearmanr(pp.cfail, pp.b).correlation)
for c in ["dec_dis", "tool_pair", "cond_pair", "n_fields"]:
    out.append(f"  {c} vs cfail pooled: {spearmanr(pp[c], pp.cfail).correlation:+.3f}")
# scatter by domain
fig, axs = plt.subplots(2, 4, figsize=(18, 8)); axs = axs.ravel()
for ax, (d, g) in zip(axs, pp.groupby("domain")):
    ax.scatter(g.tool_pair, g.b); ax.set_title(f"{d} rho={spearmanr(g.tool_pair, g.b).correlation:+.2f} n={len(g)}"); ax.set_xlabel("tool-call disagreement"); ax.set_ylabel("mean difficulty b")
axs[-1].scatter(pp.dec_dis, pp.b); axs[-1].set_title(f"all, decision disagreement rho={spearmanr(pp.dec_dis, pp.b).correlation:+.2f}")
plt.tight_layout(); plt.savefig(HERE + "/scatter_by_domain.png", dpi=90)
# which conditions (tools) programs disagree on, by step type
runs = json.load(open(HERE + "/runs.json"))
st = pd.read_csv(ZR.SCREEN + "/ideas2/steps/steps.csv", usecols=["step_type", "tools", "kind"])
tool2type = collections.defaultdict(collections.Counter)
for _, r in st.iterrows():
    for tl in str(r.tools).split("|"): tool2type[tl][r.step_type] += 1
ttype = {k: v.most_common(1)[0][0] for k, v in tool2type.items()}
rel = pd.read_csv(ZR.SCREEN + "/ideas2/steps/reliability_table.csv")
rel_by_type = rel.groupby("type").raw.mean()
# per (unit, tool): share of scenarios where programs disagree on whether the tool is called, and share of programs referencing the tool in code
rows = []
for k, r in runs.items():
    P = [p for p in r["progs"] if p is not None]
    names = set(n for p in P for s in p for n in s["tools"])
    for n in names:
        dis = np.mean([len({n in p[s]["tools"] for p in P}) > 1 for s in range(r["scen"])])
        called = np.mean([n in p[s]["tools"] for p in P for s in range(r["scen"])])
        rows.append({"unit": k, "tool": n, "dis": dis, "called": called, "type": ttype.get(n, "other/unmapped")})
tt = pd.DataFrame(rows); tt.to_csv(HERE + "/tool_disagreement.csv", index=False)
g = tt[tt.called > 0].groupby("type").agg(n=("dis", "size"), mean_dis=("dis", "mean"), share_any_dis=("dis", lambda x: (x > 0).mean())).sort_values("mean_dis", ascending=False)
g["agent_step_reliability"] = g.index.map(rel_by_type)
out.append("\nDisagreement on whether a tool is called, by step type of the tool (tool-unit pairs):\n" + g.round(3).to_string())
gt_ = tt[tt.called > 0].groupby("tool").agg(n=("dis", "size"), mean_dis=("dis", "mean")).query("n>=5").sort_values("mean_dis", ascending=False).head(15)
out.append("\nTop tools by disagreement (n>=5 units):\n" + gt_.round(3).to_string())
# crash/tool error cause: names of tools returning errors
errc = collections.Counter(); 
for k, r in runs.items():
    for p in r["progs"]:
        if p:
            for s in p:
                for n in s["tool_err_names"]: errc[n] += 1
out.append("\nTool-error counts (field-guess failures), top: " + ", ".join(f"{a}:{b}" for a, b in errc.most_common(8)))
# does disagreement for datetime/availability tools coincide with names
dt = tt[tt.tool.str.contains("date|time|avail|schedule|valid|expir|hour|period", case=False) & (tt.called > 0)]
out.append(f"\nTools with date/time/availability words in name: mean dis {dt.dis.mean():.3f} (n={len(dt)}) vs rest {tt[~tt.index.isin(dt.index) & (tt.called>0)].dis.mean():.3f}")
open(HERE + "/diag_out.txt", "w").write("\n".join(out)); print("\n".join(out))
