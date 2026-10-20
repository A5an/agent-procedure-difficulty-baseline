"""E1 and its variant: plain (smoothed) outcome shares instead of IRT, inside the baseline protocol."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
sys.path.insert(0, ZR.SCREEN)
import numpy as np
import harness as H
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit, gammaln

OUT = ZR.SCREEN + "/targets/run"


def _bb_logpmf(k, n, a, b):
    return (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) + gammaln(k + a) + gammaln(n - k + b)
            - gammaln(n + a + b) + gammaln(a + b) - gammaln(a) - gammaln(b))


def _eb_prior(k, n):
    """Beta(a,b) maximum-likelihood prior for counts k out of n (beta-binomial marginal)."""
    def nll(x):
        a, b = np.exp(x)
        return -_bb_logpmf(k, n, a, b).sum()
    r = minimize(nll, np.log([1.0, 1.0]), method="Nelder-Mead")
    a, b = np.exp(r.x)
    return float(np.clip(a, 0.05, 50)), float(np.clip(b, 0.05, 50))


class SharesPredictor:
    def __init__(self, source, eb=False, name="shares"):
        self.source, self.eb, self.name = source, eb, name
        self._predicted_difficulties = {}
        self.ridge = H.RB.FeatureBasedPredictor(source, alphas=H.RB.ALPHAS)

    def fit(self, data, train_task_ids):
        tr = list(train_task_ids)
        agents = list(data.train_abilities.index)
        K = np.zeros((len(agents), len(tr)))  # successes
        N = np.zeros_like(K)
        for i, a in enumerate(agents):
            resp = data.responses.get(a, {})
            for j, t in enumerate(tr):
                if t in resp:
                    v = resp[t]
                    k, n = (v["successes"], v["trials"]) if isinstance(v, dict) else (int(v), 1)
                    K[i, j], N[i, j] = k, n
        sa, na = K.sum(1), N.sum(1)          # per agent
        ft, nt = (N - K).sum(0), N.sum(0)    # per task: failures, trials
        if self.eb:
            a1, b1 = _eb_prior(sa, na)       # agent success-share prior
            a2, b2 = _eb_prior(ft, nt)       # task failure-share prior
            self.prior = (a1, b1, a2, b2)
            ps = (sa + a1) / (na + a1 + b1)
            pf = (ft + a2) / (nt + a2 + b2)
        else:
            ps = (sa + 0.5) / (na + 1.0)
            pf = (ft + 0.5) / (nt + 1.0)
        self.strength = dict(zip(agents, logit(ps)))
        d = logit(pf)
        # one constant c by maximum likelihood on training cells
        s = np.array([self.strength[a] for a in agents])[:, None] - d[None, :]
        def nll(c):
            z = s + c
            return -(K * -np.logaddexp(0, -z) + (N - K) * -np.logaddexp(0, z)).sum()
        self.c = float(minimize_scalar(nll, bounds=(-10, 10), method="bounded").x)
        self.ridge.fit(tr, d)
        self._predicted_difficulties = {}

    def predict_probability(self, data, agent, task):
        if task not in self._predicted_difficulties:
            self._predicted_difficulties.update(self.ridge.predict(list(data.test_tasks)))
        return float(expit(self.strength[agent] - self._predicted_difficulties[task] + self.c))


def make_preds(src, M):
    p = {}
    for f in ["adele", "human_time", "length"]:
        p[f"shares_{f}"] = SharesPredictor(src[f], eb=False)
        p[f"bb_{f}"] = SharesPredictor(src[f], eb=True)
    return p


if __name__ == "__main__":
    H.run_all(OUT, make_preds)
    names = {}
    for f in ["adele", "human_time", "length"]:
        names[f"shares_{f}"] = f"smoothed shares (+0.5/+1) + ridge on {f}"
        names[f"bb_{f}"] = f"beta-binomial EB shares + ridge on {f}"
    combos = [(f"shares_adele", "adele"), ("bb_adele", "adele"), ("shares_human_time", "adele"),
              ("shares_adele", "shares_human_time", "adele")]
    H.evaluate_all(OUT, names, combos=combos)
