"""Model tricks: within-domain centering, cross-benchmark borrowing, stacking. Custom predictors for the shared harness."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
sys.path.insert(0, ZR.SCREEN)
sys.path.insert(0, ZR.SCREEN + "/ideas")
import numpy as np, pandas as pd
from scipy.optimize import nnls
from scipy.special import expit
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import harness as H

REPO = H.REPO
SCR = ZR.SCREEN
ALPHAS = H.RB.ALPHAS
BENCH3 = ["sopbench", "tau2", "tauk_banking"]
STACK_LOG = []  # (bench, n_train, weights)
_cache = {}


def tables(bench):
    if bench in _cache:
        return _cache[bench]
    f = REPO / "data" / bench / "features"
    t = pd.read_csv(REPO / "data" / bench / "tasks.csv").set_index("case_id")
    ad = pd.read_csv(f / "adele.csv").set_index("task_id")
    ht = pd.read_csv(f / "human_time.csv").set_index("task_id")
    ln = pd.read_csv(f / "length.csv").set_index("task_id")
    rb = pd.read_csv(f"{SCR}/rubric/features/rub_proc_{bench}.csv").set_index("case_id")
    b = pd.read_csv(REPO / "data" / bench / "irt/1d_1pl/items.csv", index_col=0)["b"]
    feats = {"adele": ad.astype(float), "rub": rb.astype(float), "htlen": pd.concat([ht, ln], axis=1).astype(float)}
    feats["adele_rub"] = pd.concat([feats["adele"], feats["rub"]], axis=1)
    _cache[bench] = dict(tasks=t, feats=feats, b=b)
    return _cache[bench]


def ridge_fit(X, y, w=None):
    sc = StandardScaler().fit(X)
    m = RidgeCV(alphas=ALPHAS, cv=5)
    m.fit(sc.transform(X), y, sample_weight=w)
    return sc, m


def ridge_pred(fit, X):
    sc, m = fit
    return m.predict(sc.transform(X))


class Trick:
    """mode: plain | wd_center | wd_z | xbench | stack"""

    def __init__(self, bench, mode, feat):
        self.bench, self.mode, self.feat = bench, mode, feat
        self._predicted_difficulties = {}

    def _domain_stats(self, T, F):
        dom = T.loc[F.index, "domain"]
        mu = F.groupby(dom).mean()
        sd = F.groupby(dom).std(ddof=0).replace(0, 1.0).fillna(1.0)
        return dom, mu, sd

    def fit(self, data, tr):
        B = tables(self.bench)
        T = B["tasks"]
        y = data.train_items.loc[tr, "b"].values.astype(float)
        te = list(data.test_tasks)
        if self.mode == "plain":
            F = B["feats"][self.feat]
            fit = ridge_fit(F.loc[tr].values, y)
            pred = ridge_pred(fit, F.loc[te].values)
        elif self.mode in ("wd_center", "wd_z"):
            F = B["feats"][self.feat]
            dom_all, mu_all, sd_all = self._domain_stats(T, F)  # all tasks of the benchmark, texts only
            d_tr = dom_all.loc[tr]
            Ftr = F.loc[tr]
            mu_tr = Ftr.groupby(d_tr).mean()
            sd_tr = Ftr.groupby(d_tr).std(ddof=0).replace(0, 1.0).fillna(1.0)
            Xc = Ftr.values - mu_tr.loc[d_tr].values
            if self.mode == "wd_z":
                Xc = Xc / sd_tr.loc[d_tr].values
            yser = pd.Series(y, index=tr)
            yc = (yser - yser.groupby(d_tr).transform("mean")).values
            fit = ridge_fit(Xc, yc)
            d_te = dom_all.loc[te]
            Xt = F.loc[te].values - mu_all.loc[d_te].values
            if self.mode == "wd_z":
                Xt = Xt / sd_all.loc[d_te].values
            pred = ridge_pred(fit, Xt) + y.mean()
        elif self.mode == "xbench":
            F = B["feats"][self.feat]
            Xs, ys, ws = [F.loc[tr].values], [y], [np.ones(len(tr))]
            for ob in BENCH3:
                if ob == self.bench or {self.bench, ob} == {"tau2", "tauk_banking"}:
                    continue
                O = tables(ob)
                dm = O["tasks"].domain.reindex(O["b"].index)
                bz = O["b"].groupby(dm).transform(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0))
                bz = bz * y.std() + y.mean()
                idx = O["feats"][self.feat].index
                Xs.append(O["feats"][self.feat].values); ys.append(bz.loc[idx].values); ws.append(np.full(len(idx), 0.5))
            fit = ridge_fit(np.vstack(Xs), np.concatenate(ys), np.concatenate(ws))
            pred = ridge_pred(fit, F.loc[te].values)
        elif self.mode == "stack":
            names = ["adele", "rub", "htlen", "adele_rub"]
            groups = T.loc[tr, "procedure_id"].values
            oof = np.zeros((len(tr), len(names)))
            for i, nm in enumerate(names):
                Ftr = B["feats"][nm].loc[tr].values
                for a, b_ in GroupKFold(5).split(Ftr, y, groups):
                    oof[b_, i] = ridge_pred(ridge_fit(Ftr[a], y[a]), Ftr[b_])
            mo, my = oof.mean(0), y.mean()
            w, _ = nnls(oof - mo, y - my)
            icpt = my - w @ mo
            STACK_LOG.append((self.bench, len(tr), w.tolist(), float(icpt)))
            P = np.column_stack([ridge_pred(ridge_fit(B["feats"][nm].loc[tr].values, y), B["feats"][nm].loc[te].values)
                                 for nm in names])
            pred = P @ w + icpt
        self._pred = dict(zip(te, map(float, pred)))
        self._predicted_difficulties = dict(self._pred)

    def predict_probability(self, data, agent, task):
        return float(expit(data.train_abilities.loc[agent, "ability"] - self._pred[task]))


def make_preds(s, M, bench):
    return {
        "wd_center_adele": Trick(bench, "wd_center", "adele"),
        "wd_z_adele": Trick(bench, "wd_z", "adele"),
        "wd_center_adele_rub": Trick(bench, "wd_center", "adele_rub"),
        "xbench_adele": Trick(bench, "xbench", "adele"),
        "xbench_adele_rub": Trick(bench, "xbench", "adele_rub"),
        "stack4": Trick(bench, "stack", None),
        "cat_adele_rub": Trick(bench, "plain", "adele_rub"),
    }
