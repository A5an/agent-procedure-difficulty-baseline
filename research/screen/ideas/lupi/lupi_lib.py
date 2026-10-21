"""LUPI and two-part mixture predictors (custom CVPredictor protocol: fit(data, train_ids), predict_probability)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
REPO = Path(ZR.REPO)
TAUT = Path(ZR.DATA + "/tau2-domains/tasks")
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
DIAG = []  # (kind, bench, test_ids, payload) appended at prediction time, used only for diagnostics after the run


def tau2_label():
    """1 = the gold outcome is no state-changing action or a transfer to a human (refusal/handover case)."""
    idm = json.load(open(REPO / "data/tau2/id_map.json"))
    out = {}
    for d in ["airline", "retail", "telecom"]:
        for t in json.load(open(TAUT / f"{d}_tasks.json")):
            k = f"{d}:{t['id']}"
            if k in idm:
                ec = t.get("evaluation_criteria") or {}
                acts = ec.get("actions") or []
                out[idm[k]] = int((not acts) or any("transfer" in a["name"] for a in acts))
    return out


def load_bench(bench):
    X = pd.read_csv(HERE / f"X_{bench}.csv").set_index("case_id")
    P = pd.read_csv(HERE / f"privileged_{bench}.csv").set_index("case_id").drop(columns=["n_runs"])
    keep = [c for c in P.columns if not c.startswith("term_") or (P[c] > 0).sum() >= 15]  # fixed rule: drop near-empty termination shares
    P = P[keep]
    T = pd.read_csv(REPO / "data" / bench / "tasks.csv").set_index("case_id")
    if bench == "sopbench":
        lab = 1 - T.should_succeed
    else:
        lab = pd.Series(tau2_label()).reindex(T.index)
    return X, P, T.procedure_id, lab.astype(float)


class Base:
    def __init__(self, bench, mode):
        self.bench, self.mode = bench, mode  # mode in x_ridge | lupi | mix2 | lupi_mix2
        self.X, self.P, self.proc, self.lab = load_bench(bench)
        self._predicted_difficulties = {}
        self._cache = None

    # ---- helpers
    def _cv(self, ids):
        g = self.proc.loc[ids].values
        n = len(set(g))
        return list(GroupKFold(min(5, n)).split(ids, groups=g)) if n >= 2 else 5

    def _ridge(self, A, y, ids):
        return RidgeCV(alphas=ALPHAS, cv=self._cv(ids)).fit(A, y)

    def _stage1(self, Xs, S, ids, Xte_s):
        """Return (OOF stage-1 preds on train, stage-1 preds on test) for every signal; alpha by inner grouped CV."""
        k = S.shape[1]
        oof = np.zeros_like(S)
        g = self.proc.loc[ids].values
        n = len(set(g))
        for tr, va in GroupKFold(min(5, n)).split(Xs, groups=g):
            sub = [ids[i] for i in tr]
            m = RidgeCV(alphas=ALPHAS, cv=self._cv(sub)).fit(Xs[tr], S[tr])
            oof[va] = m.predict(Xs[va])
        full = RidgeCV(alphas=ALPHAS, cv=self._cv(ids)).fit(Xs, S)
        return oof, full.predict(Xte_s)

    def fit(self, data, train_task_ids):
        self._predicted_difficulties = {}
        self._cache = None
        self.tr = list(train_task_ids)
        self.te = list(data.test_tasks)
        b = data.train_items.loc[self.tr, "b"].values.astype(float)
        sc = StandardScaler().fit(self.X.loc[self.tr].values)
        Xs, Xte = sc.transform(self.X.loc[self.tr].values), sc.transform(self.X.loc[self.te].values)
        feats_tr, feats_te = Xs, Xte
        if "lupi" in self.mode:
            S = self.P.loc[self.tr]
            S = S.where(S.notna(), S.mean(), axis=1).values
            ss = StandardScaler().fit(S)
            oof, te_pred = self._stage1(Xs, ss.transform(S), self.tr, Xte)
            self.s1_te = te_pred
            self.s1_cols = list(self.P.columns)
            fs = StandardScaler().fit(oof)
            feats_tr = np.hstack([Xs, fs.transform(oof)])
            feats_te = np.hstack([Xte, fs.transform(te_pred)])
            DIAG.append(("s1", self.bench, self.te, te_pred, list(self.P.columns)))
        self.tr_feats, self.te_feats = feats_tr, feats_te
        lab = self.lab.loc[self.tr].values
        self.mix = "mix2" in self.mode and np.nanmin([np.sum(lab == 0), np.sum(lab == 1)]) >= 8
        if self.mix:
            lg = LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1, 10], cv=self._cv(self.tr), penalty="l2", max_iter=2000)
            lg.fit(Xs, lab.astype(int))
            self.pi = lg.predict_proba(Xte)[:, 1]
            DIAG.append(("pi", self.bench, self.te, self.pi, None))
            m0 = lab == 0
            r0 = self._ridge(feats_tr[m0], b[m0], [t for t, k in zip(self.tr, m0) if k])
            r1 = self._ridge(feats_tr[~m0], b[~m0], [t for t, k in zip(self.tr, m0) if not k])
            self.b_perf, self.b_ref = r0.predict(feats_te), r1.predict(feats_te)
        else:
            self.b_hat = self._ridge(feats_tr, b, self.tr).predict(feats_te)

    def predict_probability(self, data, agent_id, task_id):
        j = self.te.index(task_id)
        th = float(data.train_abilities.loc[agent_id, "ability"])
        if self.mix:
            return float(self.pi[j] * expit(th - self.b_ref[j]) + (1 - self.pi[j]) * expit(th - self.b_perf[j]))
        return float(expit(th - self.b_hat[j]))
