"""Formula ladder: logistic models on training cells, fitted by L-BFGS with ridge penalties.
 L1: logit P = theta_a - delta * n_steps
 L2: logit P = theta_a - sum_t delta_t n_t - v'g          (g = global rubric features, optional ADeLe)
 L3: logit P = theta_a - sum_t (delta_t + gamma_at) n_t - v'g   (gamma strongly L2-penalised)
Features are standardised on the training tasks. Penalties are picked by inner 5-fold CV over training
tasks (log-loss), never on held-out tasks. Binary or binomial cells."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit


def _cells(data, task_ids):
    agents = list(data.train_abilities.index)
    ai = {a: i for i, a in enumerate(agents)}
    ti = {t: i for i, t in enumerate(task_ids)}
    rows = []
    for a in agents:
        r = data.responses.get(a, {})
        for t in task_ids:
            if t in r:
                v = r[t]
                k, n = (v["successes"], v["trials"]) if isinstance(v, dict) else (int(v), 1)
                rows.append((ai[a], ti[t], k, n))
    return agents, np.array(rows, dtype=float)


class Ladder:
    def __init__(self, level, step_cols, glob_cols, table, extra_table=None, name="ladder"):
        """table: DataFrame indexed by case_id with all rubric columns; extra_table: optional ADeLe columns joining g."""
        self.level, self.step_cols, self.glob_cols = level, list(step_cols), list(glob_cols)
        self.table, self.extra = table, extra_table
        self.name = name
        self._predicted_difficulties = {}
        self.lams = [0.003, 0.03, 0.3, 3.0]
        self.mus = [0.3, 3.0, 30.0]   # penalty on gamma (L3), strong by construction

    def _X(self, ids):
        n = self.table.loc[ids, self.step_cols].to_numpy(float)
        if self.level == 1:
            return n, np.zeros((len(ids), 0))
        g = self.table.loc[ids, self.glob_cols].to_numpy(float) if self.glob_cols else np.zeros((len(ids), 0))
        if self.extra is not None:
            g = np.hstack([g, self.extra.loc[ids].to_numpy(float)])
        return n, g

    def _fit_params(self, N, G, cells, lam, mu, A):
        a, t, k, n = (cells[:, 0].astype(int), cells[:, 1].astype(int), cells[:, 2], cells[:, 3])
        ds, dg = N.shape[1], G.shape[1]
        L3 = self.level == 3
        tot = n.sum()
        sz = A + ds + dg + (A * ds if L3 else 0)

        def unpack(p):
            th = p[:A]; w = p[A:A + ds]; v = p[A + ds:A + ds + dg]
            gam = p[A + ds + dg:].reshape(A, ds) if L3 else None
            return th, w, v, gam

        def f(p):
            th, w, v, gam = unpack(p)
            eta = th[a] - N[t] @ w - (G[t] @ v if dg else 0)
            if L3:
                eta = eta - (N[t] * gam[a]).sum(1)
            ll = k * log_expit(eta) + (n - k) * log_expit(-eta)
            loss = -ll.sum() / tot + lam * (w @ w + v @ v) + 1e-3 * (th @ th)
            r = (k - n * expit(eta)) / tot   # d(-ll)/d eta = -(k - n p)/tot
            gth = np.zeros(A); np.add.at(gth, a, -r)
            gw = N[t].T @ r + 2 * lam * w
            gv = G[t].T @ r + 2 * lam * v if dg else np.zeros(0)
            grads = [gth + 2e-3 * th, gw, gv]
            if L3:
                loss += mu * (gam ** 2).sum() / A
                gg = np.zeros((A, ds)); np.add.at(gg, a, N[t] * r[:, None])
                grads.append((gg + 2 * mu * gam / A).ravel())
                loss_ = loss
            return loss, np.concatenate(grads)
        res = minimize(f, np.zeros(sz), jac=True, method="L-BFGS-B", options={"maxiter": 500})
        return unpack(res.x)

    def _prep(self, ids, fit=False):
        N, G = self._X(ids)
        if fit:
            self.mN, self.sN = N.mean(0), N.std(0) + 1e-9
            self.mG, self.sG = (G.mean(0), G.std(0) + 1e-9) if G.shape[1] else (0, 1)
        return (N - self.mN) / self.sN, ((G - self.mG) / self.sG if G.shape[1] else G)

    @staticmethod
    def _ll(params_eta, k, n):
        return -(k * log_expit(params_eta) + (n - k) * log_expit(-params_eta)).sum() / n.sum()

    def fit(self, data, train_task_ids):
        self._predicted_difficulties = {}
        train = list(train_task_ids)
        self.agents, cells = _cells(data, train)
        A = len(self.agents)
        N, G = self._prep(train, fit=True)
        # inner CV over training tasks
        rng = np.random.RandomState(0)
        fold = rng.randint(0, 5, len(train))
        grid = [(l, m) for l in self.lams for m in (self.mus if self.level == 3 else [0.0])]
        scores = np.zeros(len(grid))
        for fi in range(5):
            trm = fold[cells[:, 1].astype(int)] != fi
            for gi, (l, m) in enumerate(grid):
                th, w, v, gam = self._fit_params(N, G, cells[trm], l, m, A)
                te = cells[~trm]
                t = te[:, 1].astype(int); a = te[:, 0].astype(int)
                eta = th[a] - N[t] @ w - (G[t] @ v if G.shape[1] else 0)
                if gam is not None:
                    eta = eta - (N[t] * gam[a]).sum(1)
                scores[gi] += self._ll(eta, te[:, 2], te[:, 3])
        self.best = grid[int(np.argmin(scores))]
        self.th, self.w, self.v, self.gam = self._fit_params(N, G, cells, self.best[0], self.best[1], A)
        self.ai = {a: i for i, a in enumerate(self.agents)}
        # held-out difficulties
        self._test = None

    def predict_probability(self, data, agent_id, task_id):
        if self._test is None:
            test = list(data.test_tasks)
            Nt, Gt = self._prep(test)
            self._test = {t: i for i, t in enumerate(test)}
            self._eta_base = Nt @ self.w + (Gt @ self.v if Gt.shape[1] else 0)
            self._Nt = Nt
            for t, i in self._test.items():
                b = self._eta_base[i]
                if self.gam is not None:
                    b = b + (Nt[i] * self.gam.mean(0)).sum()
                self._predicted_difficulties[t] = float(b)
        i = self._test[task_id]
        e = self.th[self.ai[agent_id]] - self._eta_base[i]
        if self.gam is not None:
            e -= float(self._Nt[i] @ self.gam[self.ai[agent_id]])
        return float(expit(e))
