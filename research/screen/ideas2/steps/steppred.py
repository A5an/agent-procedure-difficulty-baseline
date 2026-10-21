"""Compositional predictors from rule-tree step lists (labelled "from the rule tree": counts of step types per case).
Features of a case: n_t = number of steps of type t (auth, existence, threshold, status, datetime, relation, availability,
or_group, action). No hidden-state label, no truth values, no outcomes of test tasks.
Fit uses only the training tasks of the fold. Agents flagged for empty answers contribute nothing to the shared
parameters and get their own intercept (or their own r) only.

Predictors (all fixed before the first full run):
  cnt_geo      null (i): P = r_a ** n, n = number of steps, r_a per agent by maximum likelihood (Mittal's geometric law)
  cnt_geo_leaf same with n = number of leaves of the rule tree (num_constraints) + 1
  cnt_logit    logit P = a_a - b * n                      (one shared slope, agent intercepts)
  cnt_logit_a  logit P = a_a - (b + e_a) * n              (agent-specific slope, ridge on e_a)
  typ_logit    logit P = a_a - sum_t d_t n_t              (shared type weights, agent intercepts)
  typ_logit_a  logit P = a_a - sum_t (d_t + e_at) n_t     (agent x type weights, ridge on e_at toward 0)
  typ_prod     P = prod_steps r_{a,type}                  r from observed step correctness, shrunk to the agent's overall rate
  cnt_prod     P = r_a ** n, r_a = observed overall step correctness of the agent (training cases)
"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import numpy as np, pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit

SCREEN = ZR.SCREEN + "/ideas2/steps"
TYPES = ["auth", "existence", "threshold", "status", "datetime", "relation", "availability", "or_group", "action"]
FLAG = {"gpt-5-mode_fc", "gpt-5-mini-mode_fc", "gemini-2.5-pro-mode_fc"}
LAM_D, LAM_E, M_SHRINK = 1.0, 5.0, 10.0

_steps = pd.read_csv(f"{SCREEN}/steps.csv", keep_default_na=False, na_values=[""])
_cnt = _steps.groupby(["case_id", "step_type"]).size().unstack(fill_value=0).reindex(columns=TYPES, fill_value=0)
FEAT = _cnt.astype(float)
N_STEPS = FEAT.sum(1)
_leaf = _steps[~_steps.step_type.isin(["or_group", "action"])].groupby("case_id").size()
N_LEAF = (_leaf.reindex(FEAT.index).fillna(0) + 1.0)
_so = None


def step_outcomes():
    global _so
    if _so is None:
        _so = pd.read_csv(f"{SCREEN}/step_outcomes.csv.gz")
    return _so


def _y(data, a, t):
    v = data.responses.get(a, {}).get(t)
    if v is None:
        return None
    return float(v["successes"] / v["trials"]) if isinstance(v, dict) else float(v)


class _Base:
    name = "base"

    def __init__(self):
        self._predicted_difficulties = {}
        self._P = {}

    def predict_probability(self, data, agent, task):
        return float(np.clip(self._P[(agent, task)], 1e-4, 1 - 1e-4))

    def _finish(self, data, tr, X_all_fn):
        raise NotImplementedError

    def _collect(self, data, tr):
        agents = list(data.train_abilities.index)
        self.agents = agents
        self.tr = list(tr)
        Y = np.full((len(agents), len(tr)), np.nan)
        for i, a in enumerate(agents):
            for j, t in enumerate(tr):
                y = _y(data, a, t)
                if y is not None:
                    Y[i, j] = y
        return agents, Y

    def _set_diff(self, data, tasks):
        for t in tasks:
            ps = np.clip([self._P[(a, t)] for a in self.agents if (a, t) in self._P], 1e-4, 1 - 1e-4)
            self._predicted_difficulties[t] = float(-np.mean(logit(ps)))


class GeoCount(_Base):
    def __init__(self, name, n_series):
        super().__init__(); self.name, self.n_series = name, n_series

    def fit(self, data, tr):
        self._P, self._predicted_difficulties = {}, {}
        agents, Y = self._collect(data, tr)
        n = self.n_series.reindex(tr).values.astype(float)
        self.r = {}
        for i, a in enumerate(agents):
            m = ~np.isnan(Y[i])
            y, nn = Y[i][m], n[m]
            def nll(z):
                r = expit(z)
                p = np.clip(r ** nn, 1e-6, 1 - 1e-6)
                return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
            self.r[a] = float(expit(minimize_scalar(nll, bounds=(-2, 12), method="bounded").x))
        self._te_ready = False

    def predict_probability(self, data, agent, task):
        return float(np.clip(self.r[agent] ** self.n_series.loc[task], 1e-4, 1 - 1e-4))

    @property
    def _predicted_difficulties_(self):
        return {}


class LogitSteps(_Base):
    """logit P = a_agent - x_case . (d + e_agent); x = type counts (or the total count).
    Stage 1 fits unflagged agents (a, d, e with ridge). Stage 2 fits only the intercept of each flagged agent."""
    def __init__(self, name, cols, agent_specific, feat=None):
        super().__init__(); self.name, self.cols, self.agent_specific = name, cols, agent_specific
        self.feat = FEAT if feat is None else feat

    def _x(self, tasks):
        if self.cols == "n":
            return self.feat.loc[tasks].sum(1).values[:, None]
        return self.feat.loc[tasks, self.cols].values

    def fit(self, data, tr):
        self._P, self._predicted_difficulties = {}, {}
        agents, Y = self._collect(data, tr)
        X = self._x(tr)
        self.mu = X.mean(0)
        Xc = X - self.mu
        K = X.shape[1]
        ui = [i for i, a in enumerate(agents) if a not in FLAG]
        A = len(ui)
        Yu = Y[ui]; mask = ~np.isnan(Yu); Y0 = np.where(mask, Yu, 0.0)
        ag = self.agent_specific

        def f(w):
            a = w[:A]; d = w[A:A + K]
            e = w[A + K:].reshape(A, K) if ag else 0.0
            eta = a[:, None] - ((Xc @ d)[None, :] + (e @ Xc.T if ag else 0.0))
            p = expit(eta)
            nll = -np.sum(mask * (Y0 * np.log(np.clip(p, 1e-9, 1)) + (1 - Y0) * np.log(np.clip(1 - p, 1e-9, 1))))
            R = (p - Y0) * mask                      # dNLL/deta
            ga = R.sum(1)
            gd = -(R.sum(0) @ Xc)
            val = nll + LAM_D * np.sum(d ** 2)
            g = [ga, gd + 2 * LAM_D * d]
            if ag:
                val += LAM_E * np.sum(e ** 2)
                g.append((-(R @ Xc) + 2 * LAM_E * e).ravel())
            return val, np.concatenate(g)

        w0 = np.zeros(A + K + (A * K if ag else 0))
        res = minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
        w = res.x
        self.a_ = {agents[i]: w[k] for k, i in enumerate(ui)}
        self.d = w[A:A + K]
        e = w[A + K:].reshape(A, K) if ag else np.zeros((A, K))
        self.e_ = {agents[i]: e[k] for k, i in enumerate(ui)}
        base = Xc @ self.d
        for i, a in enumerate(agents):
            if a in FLAG:
                m = ~np.isnan(Y[i]); y = Y[i][m]; b = base[m]
                fn = lambda z: -np.sum(y * np.log(np.clip(expit(z - b), 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - expit(z - b), 1e-9, 1)))
                self.a_[a] = minimize_scalar(fn, bounds=(-8, 8), method="bounded").x
                self.e_[a] = np.zeros(K)

    def predict_probability(self, data, agent, task):
        x = self._x([task])[0] - self.mu
        eta = self.a_[agent] - x @ (self.d + self.e_[agent])
        return float(np.clip(expit(eta), 1e-4, 1 - 1e-4))


class StepProduct(_Base):
    """P = prod over steps of the shrunk observed step reliability. type_specific=False gives r_a ** n."""
    def __init__(self, name, type_specific):
        super().__init__(); self.name, self.type_specific = name, type_specific

    def fit(self, data, tr):
        self._P, self._predicted_difficulties = {}, {}
        so = step_outcomes()
        so = so[so.case_id.isin(set(tr))]
        self.agents = list(data.train_abilities.index)
        ok = so[~so.agent.isin(FLAG)]
        self.r_type_pool = ok.groupby("step_type").step_correct.mean().reindex(TYPES).fillna(ok.step_correct.mean())
        self.r_pool = ok.step_correct.mean()
        self.r_a, self.r_at = {}, {}
        for a in self.agents:
            g = so[so.agent == a] if a not in FLAG else None
            if g is None or len(g) == 0:
                self.r_a[a] = self.r_pool
                self.r_at[a] = self.r_type_pool.copy()
                continue
            ra = g.step_correct.mean()
            self.r_a[a] = ra
            k = g.groupby("step_type").step_correct.agg(["sum", "count"]).reindex(TYPES).fillna(0)
            self.r_at[a] = (k["sum"] + M_SHRINK * ra) / (k["count"] + M_SHRINK)

    def predict_probability(self, data, agent, task):
        if self.type_specific:
            lr = np.log(np.clip(self.r_at[agent].values, 1e-4, 1)) @ FEAT.loc[task].values
        else:
            lr = np.log(np.clip(self.r_a[agent], 1e-4, 1)) * N_STEPS.loc[task]
        return float(np.clip(np.exp(lr), 1e-4, 1 - 1e-4))


def _with_diffs(p):
    """Wrap so that _predicted_difficulties is filled lazily for test tasks the harness asks about."""
    return p


def make_all():
    return {
        "cnt_geo": GeoCount("cnt_geo", N_STEPS),
        "cnt_geo_leaf": GeoCount("cnt_geo_leaf", N_LEAF),
        "cnt_logit": LogitSteps("cnt_logit", "n", False),
        "cnt_logit_a": LogitSteps("cnt_logit_a", "n", True),
        "typ_logit": LogitSteps("typ_logit", TYPES, False),
        "typ_logit_a": LogitSteps("typ_logit_a", TYPES, True),
        "typ_prod": StepProduct("typ_prod", True),
        "cnt_prod": StepProduct("cnt_prod", False),
    }
