import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, numpy as np, pandas as pd, csv
from scipy.stats import spearmanr
from pathlib import Path
from feats import GAP
D = Path(ZR.REPO + "/data"); HERE = Path(__file__).parent
S = Path(ZR.SCREEN)
def load(b):
    t = pd.read_csv(D / b / "tasks.csv").set_index("case_id")
    it = pd.read_csv(D / b / "irt/1d_1pl/items.csv"); it = it.set_index(it.columns[0])["b"]
    g = pd.read_csv(HERE / f"gap_{b}.csv").set_index("case_id")
    return t.join(g).join(it.rename("b"))
def within(df, f, key="procedure_id"):
    xs, ys = [], []
    for _, g in df.groupby(key):
        if len(g) >= 4 and g[f].std() > 0: xs += list((g[f] - g[f].mean()) / g[f].std()); ys += list(g.b - g.b.mean())
    return (round(spearmanr(xs, ys)[0], 3), len(xs)) if xs else (None, 0)
def per_dom(df, f):
    r = [spearmanr(g[f], g.b)[0] for _, g in df.groupby("domain") if g[f].std() > 0 and len(g) > 5]
    return round(float(np.mean(r)), 3) if r else None
out = []
for b in ["sopbench", "tau2", "tauk_banking"]:
    df = load(b); print("==", b, len(df), "b-missing", df.b.isna().sum()); df = df.dropna(subset=["b"])
    print("feature means"); print(df[GAP].describe().loc[["mean", "std"]].round(2).T.to_string())
    for f in GAP:
        if df[f].std() == 0: continue
        r = spearmanr(df[f], df.b)[0]
        print(f"{b:13s} {f:20s} all {r:+.3f}  dom-mean {per_dom(df, f)}  within-proc {within(df, f) if b == 'sopbench' else ''}")
# overlap with PLAN
for b, pf in [("sopbench", "plan3_sopbench.csv"), ("tau2", "plan3_tau2.csv")]:
    g = pd.read_csv(HERE / f"gap_{b}.csv").set_index("case_id"); p = pd.read_csv(S / "ideas3/plan_tools/features" / pf).set_index("case_id")
    j = g.join(p); print("== overlap", b)
    C = pd.DataFrame({q: [spearmanr(j[a], j[q])[0] if j[a].std() > 0 and j[q].std() > 0 else np.nan for a in GAP] for q in p.columns}, index=GAP).round(2)
    print(C.to_string())
