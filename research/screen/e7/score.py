"""E7 scoring (PREREG.md): Spearman with IRT difficulty per benchmark, zero-shot D against statement-only comparators."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import e7
REPO = Path(ZR.REPO)
AP = REPO / "third_party/agent-psychometrics"
T5 = REPO / "results/table5_statement_only"
BENCH = ["swebench_verified", "swebench_pro", "gso", "terminalbench"]
ALPHAS = np.logspace(-2, 5, 15)


def p_of(txt):
    if txt is None: return np.nan
    m = re.search(r'"p_success"\s*:\s*([0-9.eE+-]+)', txt)
    return float(m.group(1)) if m else np.nan


def target(b, binary=False):
    sub = "irt_binomial" if (b == "terminalbench" and not binary) else "irt"
    t = pd.read_csv(AP / f"data/{b}/{sub}/1d_1pl/items.csv", index_col=0)["b"]
    t.index = t.index.astype(str)
    return t


def ridge_oof(X, y, seeds=5):
    preds = np.zeros((seeds, len(y)))
    for s in range(seeds):
        for tr, te in KFold(5, shuffle=True, random_state=s).split(X):
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(X[tr], y[tr])
            preds[s, te] = m.predict(X[te])
    return preds


def sp(a, b): return float(spearmanr(a, b).correlation)


def boot(y, preds, B=2000, seed=0):
    """preds: dict name -> (n,) array or (seeds, n) array (rho averaged over seeds). Returns point, CI and paired diffs to D."""
    rng = np.random.default_rng(seed); n = len(y)
    def rho(p, ix): return np.mean([sp(q[ix], y[ix]) for q in (p if p.ndim == 2 else p[None])])
    pt = {k: rho(v, np.arange(n)) for k, v in preds.items()}
    bs = {k: [] for k in preds}
    for _ in range(B):
        ix = rng.integers(0, n, n)
        for k, v in preds.items(): bs[k].append(rho(v, ix))
    out = {}
    for k in preds:
        a = np.array(bs[k]); out[k] = {"rho": pt[k], "ci": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]}
        if k != "D" and "D" in preds:
            d = np.array(bs["D"]) - a
            out[k]["D_minus"] = {"diff": pt["D"] - pt[k], "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return out


def main(B=2000):
    raw = json.load(open(HERE / "e7_raw.json"))
    its = {f'{i["bench"]}::{i["id"]}': i for i in e7.items()}
    res = {}
    for b in BENCH:
        y_all = target(b)
        keys = [k for k in raw if k.startswith(b + "::")]
        df = pd.DataFrame({"id": [k.split("::", 1)[1] for k in keys], "p": [p_of(raw[k]) for k in keys],
                           "len": [len(its[k]["text"]) for k in keys]}).set_index("id")
        df = df[df.index.isin(y_all.index)].dropna()
        df["D"] = np.log((1 - df.p.clip(0.02, 0.98)) / df.p.clip(0.02, 0.98))
        df["y"] = y_all.loc[df.index]
        judge = pd.read_csv(T5 / f"judge/{b}.csv").set_index("instance_id"); judge.index = judge.index.astype(str)
        z = np.load(T5 / f"emb/{b}.npz", allow_pickle=True)
        emb = pd.DataFrame(z["X"], index=[str(t) for t in z["task_ids"]])
        ids = [i for i in df.index if i in judge.index and i in emb.index]
        df = df.loc[ids]; y = df.y.values
        Xj = judge.loc[ids].values.astype(float); Xe = emb.loc[ids].values
        preds = {"D": df.D.values, "length": df["len"].values.astype(float),
                 "judge15 ridge (in-benchmark CV)": ridge_oof(Xj, y), "embedding ridge (in-benchmark CV)": ridge_oof(Xe, y),
                 "judge15 + D ridge (in-benchmark CV)": ridge_oof(np.column_stack([Xj, df.D.values]), y)}
        if (T5 / b / "predictions.csv").exists():
            hp = pd.read_csv(T5 / b / "predictions.csv").set_index("item_id")["diff_pred"]; hp.index = hp.index.astype(str)
            ok = [i for i in ids if i in hp.index]
            assert len(ok) == len(ids), (b, len(ok), len(ids))
            preds["judge15 held-out benchmark (Table 5)"] = hp.loc[ids].values
        r = boot(y, preds, B=B)
        r["_n"] = len(ids); r["_p_missing"] = int(len(keys) - len(df)); r["_mean_p"] = float(df.p.mean())
        if b == "terminalbench":
            yb = target(b, binary=True).reindex(ids)
            r["_D_vs_binary_irt"] = sp(df.D.values[~yb.isna().values], yb.dropna().values)
        res[b] = r
        print(f"== {b} n={len(ids)} mean p={df.p.mean():.2f}")
        for k, v in r.items():
            if k.startswith("_"): print("  ", k, v); continue
            extra = f"  D minus this {v['D_minus']['diff']:+.3f} [{v['D_minus']['ci'][0]:+.3f}, {v['D_minus']['ci'][1]:+.3f}]" if "D_minus" in v else ""
            print(f"   {k:40s} {v['rho']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]{extra}")
        df[["p", "D", "len", "y"]].to_csv(HERE / f"scores_{b}.csv")
    json.dump(res, open(HERE / "e7_score.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
