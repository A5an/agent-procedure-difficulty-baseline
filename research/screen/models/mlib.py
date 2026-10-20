"""Model parts for the 'models' screen: tree regressors, pairwise ranking, per-agent assessors."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler

import harness as H

M = H.M


def inner_cv(groups, seed=42):
    """Inner folds over training tasks, grouped by procedure (train-only information)."""
    return GroupKFold(5, shuffle=True, random_state=seed)


# ---------------------------------------------------------------- 1. trees on difficulty
class TreeDifficulty:
    """features -> fold-IRT beta with a small grid chosen by inner grouped CV on training tasks."""

    def __init__(self, source, tasks, kind):
        self.source, self.kind = source, kind
        self.group = dict(zip(tasks.case_id, tasks.procedure_id))

    def fit(self, task_ids, b):
        X = self.source.get_features(task_ids)
        g = np.array([self.group[t] for t in task_ids])
        if self.kind == "rf":
            est = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=2)
            grid = {"min_samples_leaf": [5, 20, 50], "max_features": [0.33, 1.0]}
        else:
            est = HistGradientBoostingRegressor(learning_rate=0.05, random_state=0, early_stopping=False)
            grid = {"max_depth": [2, 3], "max_iter": [50, 150], "min_samples_leaf": [20, 50]}
        cv = list(inner_cv(g).split(X, groups=g))
        self.m = GridSearchCV(est, grid, cv=cv, scoring="neg_mean_squared_error", refit=True).fit(X, np.asarray(b, float))
        self.best = self.m.best_params_

    def predict(self, task_ids):
        p = self.m.predict(self.source.get_features(task_ids))
        return {t: float(v) for t, v in zip(task_ids, p)}


# ---------------------------------------------------------------- 2. pairwise ranking loss
def _pairs(dom, b, idx, cap, rng):
    """All ordered-by-difficulty pairs inside a domain among rows idx (subsampled to cap)."""
    I, J = [], []
    for d in np.unique(dom[idx]):
        r = idx[dom[idx] == d]
        i, j = np.triu_indices(len(r), 1)
        I.append(r[i]); J.append(r[j])
    I, J = np.concatenate(I), np.concatenate(J)
    keep = b[I] != b[J]
    I, J = I[keep], J[keep]
    if len(I) > cap:
        s = rng.choice(len(I), cap, replace=False)
        I, J = I[s], J[s]
    return I, J


def _fit_rank(Xs, b, dom, idx, C, cap=20000):
    rng = np.random.default_rng(0)
    I, J = _pairs(dom, b, idx, cap, rng)
    D = Xs[I] - Xs[J]
    y = (b[I] > b[J]).astype(int)
    D2, y2 = np.vstack([D, -D]), np.r_[y, 1 - y]
    return LogisticRegression(C=C, fit_intercept=False, max_iter=2000).fit(D2, y2).coef_[0]


class RankDifficulty:
    """Linear scorer trained with pairwise logistic loss; C by inner CV (pairwise accuracy); scale calibrated to beta."""
    CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    def __init__(self, source, tasks):
        self.source = source
        self.group = dict(zip(tasks.case_id, tasks.procedure_id))
        self.domain = dict(zip(tasks.case_id, tasks.domain))

    def fit(self, task_ids, b):
        b = np.asarray(b, float)
        X = self.source.get_features(task_ids)
        g = np.array([self.group[t] for t in task_ids])
        dom = np.array([self.domain[t] for t in task_ids])
        self.sc = StandardScaler().fit(X)
        Xs = self.sc.transform(X)
        acc = {C: [] for C in self.CS}
        for tr, va in inner_cv(g).split(Xs, groups=g):
            sc = StandardScaler().fit(X[tr]); Xt = sc.transform(X)
            I, J = _pairs(dom, b, va, 10 ** 9, np.random.default_rng(0))
            for C in self.CS:
                w = _fit_rank(Xt, b, dom, tr, C)
                s = Xt @ w
                acc[C].append(np.mean(np.sign(s[I] - s[J]) == np.sign(b[I] - b[J])))
        self.C = max(self.CS, key=lambda C: np.mean(acc[C]))
        self.w = _fit_rank(Xs, b, dom, np.arange(len(b)), self.C)
        s = (Xs @ self.w).reshape(-1, 1)
        self.cal = LinearRegression().fit(s, b)

    def predict(self, task_ids):
        s = (self.sc.transform(self.source.get_features(task_ids)) @ self.w).reshape(-1, 1)
        return {t: float(v) for t, v in zip(task_ids, self.cal.predict(s))}


# ---------------------------------------------------------------- 4. per-agent assessors
def _softplus(z):
    return np.logaddexp(0, z)


def fit_mat(X, K, N, Off, lam, pen_c=1e-3, w0=None):
    """Penalised binomial logistic. rows = agents (own intercept), shared coefficients w.
    z[a,t] = Off[a,t] + c[a] + X[t] @ w ; loss = sum(N*softplus(z) - K*z) + lam/2 |w|^2 + pen_c/2 |c|^2."""
    A, d = K.shape[0], X.shape[1]

    def f(p):
        c, w = p[:A], p[A:]
        Z = Off + c[:, None] + (X @ w)[None, :]
        P = expit(Z)
        G = N * P - K
        loss = (N * _softplus(Z) - K * Z).sum() + 0.5 * lam * w @ w + 0.5 * pen_c * c @ c
        gc = G.sum(1) + pen_c * c
        gw = X.T @ G.sum(0) + lam * w
        return loss, np.r_[gc, gw]

    p0 = np.zeros(A + d) if w0 is None else w0
    r = minimize(f, p0, jac=True, method="L-BFGS-B", options={"maxiter": 300})
    return r.x[:A], r.x[A:]


def _nll(Z, K, N):
    return float((N * _softplus(Z) - K * Z).sum())


class _AgentModel(M._Base):
    """Common machinery: standardised features, success/trial matrices on training tasks."""

    def __init__(self, source, tasks, name):
        super().__init__(source, name)
        self.group = dict(zip(tasks.case_id, tasks.procedure_id))

    def fit(self, data, train_task_ids):
        self._p, self._predicted_difficulties = {}, {}
        tr = list(train_task_ids)
        self.agents, cells = M._cells(data, tr)
        A, T = len(self.agents), len(tr)
        K, N = np.zeros((A, T)), np.zeros((A, T))
        for a, t, k, n in cells:
            K[int(a), int(t)] += k; N[int(a), int(t)] += n
        X = self.source.get_features(tr).astype(float)
        self.sc = StandardScaler().fit(X)
        Xs = self.sc.transform(X)
        g = np.array([self.group[t] for t in tr])
        self.folds = list(inner_cv(g).split(Xs, groups=g))
        self._fit(Xs, K, N)

    def _predict_all(self, data):
        test = list(data.test_tasks)
        Xt = self.sc.transform(self.source.get_features(test).astype(float))
        Z = self._logits(Xt)
        for i, a in enumerate(self.agents):
            for j, t in enumerate(test):
                self._p[(a, t)] = float(expit(Z[i, j]))
        self._set_difficulties(data, test)


class PerAgentLogit(_AgentModel):
    """ADeLe-style assessor: one L2 logistic regression per agent, penalty shared and chosen by inner CV."""
    LAMS = [0.3, 3.0, 30.0, 300.0, 3000.0]

    def _fit_all(self, X, K, N, lam, idx=None):
        A = K.shape[0]
        C, W = np.zeros(A), np.zeros((A, X.shape[1]))
        for a in range(A):
            c, w = fit_mat(X, K[a:a + 1], N[a:a + 1], np.zeros((1, K.shape[1])), lam)
            C[a], W[a] = c[0], w
        return C, W

    def _fit(self, X, K, N):
        score = {l: 0.0 for l in self.LAMS}
        for tr, va in self.folds:
            for l in self.LAMS:
                C, W = self._fit_all(X[tr], K[:, tr], N[:, tr], l)
                Z = C[:, None] + W @ X[va].T
                score[l] += _nll(Z, K[:, va], N[:, va])
        self.lam = min(self.LAMS, key=score.get)
        self.C, self.W = self._fit_all(X, K, N, self.lam)

    def _logits(self, Xt):
        return self.C[:, None] + self.W @ Xt.T


class ShrunkPerAgent(_AgentModel):
    """Pooled logistic with agent intercepts, then per-agent deviations with a stronger L2 penalty."""
    POOL = [1.0, 10.0, 100.0]
    DEV = [10.0, 100.0, 1000.0, 1e5]

    def _fit_pooled(self, X, K, N, lp):
        return fit_mat(X, K, N, np.zeros_like(K), lp)

    def _fit_dev(self, X, K, N, c, w, ld):
        A = K.shape[0]
        Z0 = c[:, None] + (X @ w)[None, :]
        C, W = np.zeros(A), np.zeros((A, X.shape[1]))
        for a in range(A):
            ci, wi = fit_mat(X, K[a:a + 1], N[a:a + 1], Z0[a:a + 1], ld)
            C[a], W[a] = ci[0], wi
        return C, W

    def _fit(self, X, K, N):
        score = {(lp, ld): 0.0 for lp in self.POOL for ld in self.DEV}
        for tr, va in self.folds:
            for lp in self.POOL:
                c, w = self._fit_pooled(X[tr], K[:, tr], N[:, tr], lp)
                for ld in self.DEV:
                    C, W = self._fit_dev(X[tr], K[:, tr], N[:, tr], c, w, ld)
                    Z = (c + C)[:, None] + (w[None, :] + W) @ X[va].T
                    score[(lp, ld)] += _nll(Z, K[:, va], N[:, va])
        self.lp, self.ld = min(score, key=score.get)
        c, w = self._fit_pooled(X, K, N, self.lp)
        C, W = self._fit_dev(X, K, N, c, w, self.ld)
        self.c, self.w, self.C, self.W = c, w, C, W

    def _logits(self, Xt):
        return (self.c + self.C)[:, None] + (self.w[None, :] + self.W) @ Xt.T
