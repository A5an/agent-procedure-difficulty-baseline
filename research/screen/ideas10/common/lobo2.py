"""Leave-one-benchmark-out with arbitrary feature tables, same model and target as repo src/lobo.py
(their FeatureBasedPredictor: StandardScaler + RidgeCV, target z(beta) per benchmark), plus paired bootstrap
differences between methods over task groups of the held-out benchmark.
usage: from lobo2 import run; run({"name": {"sopbench": df, "tau2": df, "tauk_banking": df}}, held=["tauk_banking"])
Each df is indexed by case_id with numeric columns."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
REPO = ZR.REPO
sys.path.insert(0, REPO + "/src")
_cwd = os.getcwd()
from common import BENCHES, DATA, use_agent_psychometrics
use_agent_psychometrics()
from experiment_new_tasks.feature_predictor import FeatureBasedPredictor
os.chdir(_cwd)
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]

class ArraySource:
    def __init__(self, name, ids, X): self.name = name; self._idx = {t: i for i, t in enumerate(ids)}; self._X = X
    def get_features(self, task_ids): return self._X[[self._idx[t] for t in task_ids]]

def tasks():
    out = []
    for b in BENCHES:
        d = DATA / b; t = pd.read_csv(d / "tasks.csv")[["case_id", "domain", "procedure_id"]]
        beta = t.case_id.map(pd.read_csv(d / "irt/1d_1pl/items.csv", index_col=0).b)
        t["bench"] = b; t["z"] = ((beta - beta.mean()) / beta.std()).values; t["uid"] = b + ":" + t.case_id; out.append(t)
    return pd.concat(out, ignore_index=True)

def repo_feat(name):
    return {b: pd.read_csv(DATA / b / "features" / f"{name}.csv").set_index("task_id") for b in BENCHES}

def run(methods, held=BENCHES, n_boot=1000, base="adele", seed=20261008, within_domain=False):
    T = tasks(); res = {}
    for h in held:
        tr = T[T.bench != h]; te = T[T.bench == h].reset_index(drop=True)
        preds = {}
        for name, per in methods.items():
            if not all(b in per for b in BENCHES): continue
            X = np.vstack([per[b].reindex(T[T.bench == b].case_id).to_numpy(float) for b in BENCHES])
            X = np.where(np.isnan(X), np.nanmedian(X, 0), X)
            src = ArraySource(name, T.uid.tolist(), X)
            p = FeatureBasedPredictor(src, alphas=ALPHAS); p.fit(tr.uid.tolist(), tr.z.values)
            pr = p.predict(te.uid.tolist()); preds[name] = np.array([pr[u] for u in te.uid])
        groups = te.procedure_id.unique(); gi = {g: np.where(te.procedure_id == g)[0] for g in groups}
        rng = np.random.default_rng(seed); draws = {n: [] for n in preds}
        def rho(p, idx):
            if not within_domain: return spearmanr(p[idx], te.z.values[idx]).correlation
            num = den = 0
            dd = te.domain.values[idx]
            for d in np.unique(dd):
                k = idx[dd == d]
                if len(k) < 3: continue
                w = len(k) * (len(k) - 1) / 2; num += spearmanr(p[k], te.z.values[k]).correlation * w; den += w
            return num / den
        for _ in range(n_boot):
            idx = np.concatenate([gi[g] for g in rng.choice(groups, len(groups), replace=True)])
            for n in preds: draws[n].append(rho(preds[n], idx))
        res[h] = {}
        allidx = np.arange(len(te))
        for n in preds:
            d = np.array(draws[n]); r = {"rho": float(rho(preds[n], allidx)), "ci": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}
            for b2 in [base, "length"]:
                if b2 in preds and b2 != n:
                    dd = d - np.array(draws[b2]); r[f"vs_{b2}"] = [float(r["rho"] - rho(preds[b2], allidx)), float(np.nanpercentile(dd, 2.5)), float(np.nanpercentile(dd, 97.5))]
            res[h][n] = r
    return res

def show(res):
    for h, r in res.items():
        print("held out:", h)
        for n, v in sorted(r.items(), key=lambda x: -x[1]["rho"]):
            s = f"  {n:28s} rho {v['rho']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]"
            for k in ["vs_adele", "vs_length"]:
                if k in v: s += f"  {k} {v[k][0]:+.3f} [{v[k][1]:+.3f}, {v[k][2]:+.3f}]"
            print(s)
