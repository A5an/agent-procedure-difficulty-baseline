"""Per-type ability predictors for the screening harness (no LLM calls).

All models keep the baseline's difficulty beta_hat (ridge on the 18 ADeLe levels, fold-IRT theta) as the base and
add an agent x type term, so any gain over 'adele' comes from the agent-specific part.
  TypeShare   (ii)  literal shrunk shares: P(a, type k) = (S_ak + kappa * s_a) / (N_ak + kappa), s_a = agent's overall
                    training share; kappa by 3-fold CV over training procedures (log-loss). Stand-alone (does not use beta_hat).
  TypeLogit   (iii) logit P = theta_a - beta_hat_t + sum_l W[a, l] x_tl, W ridge-penalised (lambda by the same inner CV),
                    x_t = one-hot of the task's levels in the chosen type systems. Fit on training tasks with the
                    fold-IRT beta as offset (so W is what the 1-D model leaves unexplained for agent a on type l).
  TypeLogit with Q (iv) the same with a fixed task-to-dimension loading matrix Q (3 columns) instead of one-hots:
                    confirmatory MIRT, logit = theta_a + sum_d q_td phi_ad - beta_hat_t.
"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
import harness as H

RB = H.RB
TYPES = pd.read_csv(ZR.SCREEN + "/pertype/types.csv").set_index("case_id")
GRID_LAM = [3, 10, 30, 100, 300, 1000, 3000, 30000]
GRID_KAPPA = [1, 3, 10, 30, 100, 300, 1000, 10000]
_cache = {}


class Ctx:
    pass


def get_ctx(data, train, src):
    key = (id(data), len(train), train[0], train[-1])
    if key in _cache:
        return _cache[key]
    _cache.clear()
    c = Ctx()
    c.agents = list(data.train_abilities.index)
    c.train = list(train)
    c.theta = data.train_abilities.loc[c.agents, "ability"].values
    c.beta = data.train_items.loc[c.train, "b"].values
    A, T = len(c.agents), len(c.train)
    c.Y, c.M = np.zeros((A, T)), np.zeros((A, T))
    for i, a in enumerate(c.agents):
        r = data.responses.get(a, {})
        for j, t in enumerate(c.train):
            if t in r:
                c.Y[i, j] = float(r[t]); c.M[i, j] = 1.0
    c.O = c.theta[:, None] - c.beta[None, :]
    c.groups = TYPES.loc[c.train, "procedure_id"].values
    base = H.ridge(src["adele"])
    base.fit(data, c.train)
    base.predict_probability(data, c.agents[0], data.test_tasks[0])
    c.beta_hat = dict(base._predicted_difficulties)
    c.test = list(data.test_tasks)
    _cache[key] = c
    return c


def _folds(groups, k=3, seed=0):
    u = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    part = {g: i % k for i, g in enumerate(u)}
    f = np.array([part[g] for g in groups])
    return [(np.flatnonzero(f != i), np.flatnonzero(f == i)) for i in range(k)]


def _fit_W(X, Y, M, O, lam):
    A, T = Y.shape
    K = X.shape[1]

    def fg(w):
        W = w.reshape(A, K)
        Z = O + W @ X.T
        P = expit(Z)
        Pc = np.clip(P, 1e-9, 1 - 1e-9)
        ll = -(M * (Y * np.log(Pc) + (1 - Y) * np.log(1 - Pc))).sum() + 0.5 * lam * (W ** 2).sum()
        G = ((P - Y) * M) @ X + lam * W
        return ll, G.ravel()
    r = minimize(fg, np.zeros(A * K), jac=True, method="L-BFGS-B", options={"maxiter": 200})
    return r.x.reshape(A, K)


def _ll(P, Y, M):
    P = np.clip(P, 1e-6, 1 - 1e-6)
    return -(M * (Y * np.log(P) + (1 - Y) * np.log(1 - P))).sum()


class TypeLogit:
    """(iii) with one-hot levels of `systems`, or (iv) with Q columns `qcols`."""

    def __init__(self, src_unused=None, systems=(), qcols=(), name="x", fixed_lam=None):
        self.systems, self.qcols, self.name, self.fixed_lam = list(systems), list(qcols), name, fixed_lam
        self._p, self._predicted_difficulties = {}, {}
        self.src = None

    def bind(self, src):
        self.src = src
        return self

    def design(self, ids, fit=False):
        if self.qcols:
            return TYPES.loc[ids, self.qcols].values.astype(float)
        if fit:
            self.levels = [(s, v) for s in self.systems for v in sorted(TYPES[s].unique())]
        return np.stack([(TYPES.loc[ids, s].values == v).astype(float) for s, v in self.levels], 1)

    def fit(self, data, train_task_ids):
        self._p, self._predicted_difficulties = {}, {}
        c = self.ctx = get_ctx(data, list(train_task_ids), self.src)
        X = self.design(c.train, fit=True)
        if self.fixed_lam is not None:
            self.lam = self.fixed_lam
        else:
            tot = {}
            for tr, va in _folds(c.groups):
                for lam in GRID_LAM:
                    W = _fit_W(X[tr], c.Y[:, tr], c.M[:, tr], c.O[:, tr], lam)
                    P = expit(c.O[:, va] + W @ X[va].T)
                    tot[lam] = tot.get(lam, 0) + _ll(P, c.Y[:, va], c.M[:, va])
            self.lam = min(tot, key=tot.get)
        self.W = _fit_W(X, c.Y, c.M, c.O, self.lam)

    def _predict_all(self, data):
        c = self.ctx
        Xt = self.design(c.test)
        shift = self.W @ Xt.T
        for j, t in enumerate(c.test):
            for i, a in enumerate(c.agents):
                self._p[(a, t)] = float(expit(c.theta[i] - c.beta_hat[t] + shift[i, j]))
        self._predicted_difficulties = {t: float(-np.mean(np.log(np.clip([self._p[(a, t)] for a in c.agents], 1e-4, 1 - 1e-4) /
                                         (1 - np.clip([self._p[(a, t)] for a in c.agents], 1e-4, 1 - 1e-4))))) for t in c.test}

    def predict_probability(self, data, agent_id, task_id):
        if not self._p:
            self._predict_all(data)
        return self._p[(agent_id, task_id)]


class TypeShare:
    def __init__(self, systems, name="x"):
        self.systems, self.name = list(systems), name
        self._p, self._predicted_difficulties = {}, {}
        self.src = None

    def bind(self, src):
        self.src = src
        return self

    def cell_of(self, ids):
        return np.array(["|".join(str(TYPES.loc[i, s]) for s in self.systems) for i in ids])

    def _shares(self, Y, M, cells, kappa, cats):
        s = (Y * M).sum(1) / M.sum(1)
        S = np.stack([(Y * M)[:, cells == k].sum(1) for k in cats], 1)
        N = np.stack([M[:, cells == k].sum(1) for k in cats], 1)
        return (S + kappa * s[:, None]) / (N + kappa), s

    def fit(self, data, train_task_ids):
        self._p = {}
        c = self.ctx = get_ctx(data, list(train_task_ids), self.src)
        cells = self.cell_of(c.train)
        tot = {}
        for tr, va in _folds(c.groups):
            cats = sorted(set(cells[tr]))
            for kappa in GRID_KAPPA:
                Pk, s = self._shares(c.Y[:, tr], c.M[:, tr], cells[tr], kappa, cats)
                idx = {k: i for i, k in enumerate(cats)}
                P = np.stack([Pk[:, idx[k]] if k in idx else s for k in cells[va]], 1)
                tot[kappa] = tot.get(kappa, 0) + _ll(P, c.Y[:, va], c.M[:, va])
        self.kappa = min(tot, key=tot.get)
        self.cats = sorted(set(cells))
        self.Pk, self.s = self._shares(c.Y, c.M, cells, self.kappa, self.cats)

    def predict_probability(self, data, agent_id, task_id):
        if not self._p:
            c = self.ctx
            cells = self.cell_of(c.test)
            idx = {k: i for i, k in enumerate(self.cats)}
            for j, t in enumerate(c.test):
                for i, a in enumerate(c.agents):
                    self._p[(a, t)] = float(self.Pk[i, idx[cells[j]]] if cells[j] in idx else self.s[i])
        return self._p[(agent_id, task_id)]


def _fit_full(X, Y, M, lam_c, lam_w, use_w):
    A, T = Y.shape
    K = X.shape[1]
    n = A + K + (A * K if use_w else 0)

    def fg(w):
        al, c = w[:A], w[A:A + K]
        W = w[A + K:].reshape(A, K) if use_w else 0.0
        Z = al[:, None] + (c @ X.T)[None, :] + (W @ X.T if use_w else 0.0)
        P = expit(Z)
        Pc = np.clip(P, 1e-9, 1 - 1e-9)
        ll = -(M * (Y * np.log(Pc) + (1 - Y) * np.log(1 - Pc))).sum() + 0.5 * lam_c * (c ** 2).sum() + 0.5 * 0.01 * (al ** 2).sum()
        R = (P - Y) * M
        g = np.concatenate([R.sum(1) + 0.01 * al, R.sum(0) @ X + lam_c * c])
        if use_w:
            ll += 0.5 * lam_w * (W ** 2).sum()
            g = np.concatenate([g, (R @ X + lam_w * W).ravel()])
        return ll, g
    r = minimize(fg, np.zeros(n), jac=True, method="L-BFGS-B", options={"maxiter": 300})
    w = r.x
    return w[:A], w[A:A + K], (w[A + K:].reshape(A, K) if use_w else np.zeros((A, K)))


class StandAlone(TypeLogit):
    """logit P = alpha_a + sum_l c_l x_tl [+ sum_l W[a, l] x_tl]; no baseline difficulty. main-effect control vs interaction."""

    def __init__(self, systems, interaction, name="x"):
        super().__init__(systems=systems, name=name)
        self.inter = interaction

    def fit(self, data, train_task_ids):
        self._p, self._predicted_difficulties = {}, {}
        c = self.ctx = get_ctx(data, list(train_task_ids), self.src)
        X = self.design(c.train, fit=True)
        tot = {}
        grid = [(lc, lw) for lc in (1, 10, 100) for lw in ((10, 30, 100, 300, 1000, 10000) if self.inter else (0,))]
        for tr, va in _folds(c.groups):
            for lc, lw in grid:
                al, cc, W = _fit_full(X[tr], c.Y[:, tr], c.M[:, tr], lc, lw, self.inter)
                P = expit(al[:, None] + (cc @ X[va].T)[None, :] + W @ X[va].T)
                tot[(lc, lw)] = tot.get((lc, lw), 0) + _ll(P, c.Y[:, va], c.M[:, va])
        self.hp = min(tot, key=tot.get)
        self.al, self.cc, self.W = _fit_full(X, c.Y, c.M, self.hp[0], self.hp[1], self.inter)

    def _predict_all(self, data):
        c = self.ctx
        Xt = self.design(c.test)
        Z = self.al[:, None] + (self.cc @ Xt.T)[None, :] + self.W @ Xt.T
        for j, t in enumerate(c.test):
            for i, a in enumerate(c.agents):
                self._p[(a, t)] = float(expit(Z[i, j]))


def make_preds(src, M):
    ms = {
        "share_fam": TypeShare(["fam"]), "share_tree": TypeShare(["tree"]), "share_dom": TypeShare(["dom"]),
        "share_famtree": TypeShare(["fam", "tree"]), "share_none": TypeShare([]),
        "logit_fam": TypeLogit(systems=["fam"]), "logit_tree": TypeLogit(systems=["tree"]), "logit_dom": TypeLogit(systems=["dom"]),
        "logit_all": TypeLogit(systems=["fam", "tree", "dom"]),
        "mirt_q3": TypeLogit(qcols=["q_deep", "q_branch", "q_modify"]),
        "logit_all_l10": TypeLogit(systems=["fam", "tree", "dom"], fixed_lam=10), "logit_all_l30": TypeLogit(systems=["fam", "tree", "dom"], fixed_lam=30),
        "mirt_q3_l10": TypeLogit(qcols=["q_deep", "q_branch", "q_modify"], fixed_lam=10),
        "main_all": StandAlone(["fam", "tree", "dom"], False), "inter_all": StandAlone(["fam", "tree", "dom"], True),
        "typebeta": H.ridge(src["onehot"]), "adele_onehot": H.ridge(src["adele_onehot"]),
    }
    for m in ms.values():
        if hasattr(m, "bind"):
            m.bind(src)
    return ms
