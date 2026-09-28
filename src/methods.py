"""Candidate methods that do not go through the fixed-IRT ridge of Agent psychometrics.

Each class follows the agent-psychometrics CVPredictor protocol (fit(data, train_task_ids),
predict_probability(data, agent_id, task_id)) and fills `_predicted_difficulties` for the held-out
tasks so the rank metrics can be computed (higher = harder).

  KNNRouter          past-performance retrieval: success of an agent on a new task = its success on
                     the k most similar training tasks (kNN / few-shot routers)
  AmortizedIRT       1PL IRT whose task difficulty is a linear function of task covariates, trained
                     jointly on the responses. With the ADeLe levels as covariates this is the linear
                     logistic test model (LLTM; Fischer 1973; De Boeck and Wilson 2004): `lltm_adele`
  AmortizedMIRT      multidimensional version: P = sigmoid(a_t . theta_a - b_t) with a_t and b_t
                     produced from the text embedding (IRT-Router style, our reimplementation)

Inputs are only the leak-free task statement features used everywhere else. The router family
follows Li (2025, arXiv 2505.12601) for kNN and IRT-Router (Song et al., ACL 2025) for the
multidimensional model; both are our reimplementations.
"""

import numpy as np
import torch
from scipy.special import logit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _cells(data, task_ids):
    """(agent index, task index, successes, trials) for the given tasks, binary or binomial."""
    agents = list(data.train_abilities.index)
    a_idx = {a: i for i, a in enumerate(agents)}
    t_idx = {t: i for i, t in enumerate(task_ids)}
    rows = []
    for a in agents:
        resp = data.responses.get(a, {})
        for t in task_ids:
            if t in resp:
                v = resp[t]
                k, n = (v["successes"], v["trials"]) if isinstance(v, dict) else (int(v), 1)
                rows.append((a_idx[a], t_idx[t], k, n))
    arr = np.array(rows, dtype=np.float64)
    return agents, arr


class _Base:
    def __init__(self, source, name):
        self.source = source
        self.name = name
        self._predicted_difficulties = {}
        self._p = {}

    def predict_probability(self, data, agent_id, task_id):
        if not self._p:
            self._predict_all(data)
        return float(self._p[(agent_id, task_id)])

    def _set_difficulties(self, data, test):
        agents = list(data.train_abilities.index)
        for t in test:
            ps = np.clip([self._p[(a, t)] for a in agents if (a, t) in self._p], 1e-4, 1 - 1e-4)
            self._predicted_difficulties[t] = float(-np.mean(logit(ps)))


class KNNRouter(_Base):
    def __init__(self, source, name="knn_router", k=10, prior=1.0):
        super().__init__(source, name)
        self.k, self.prior = k, prior

    def fit(self, data, train_task_ids):
        self._p, self._predicted_difficulties = {}, {}
        self.train = list(train_task_ids)
        X = self.source.get_features(self.train).astype(np.float64)
        self.Xtr = X / np.linalg.norm(X, axis=1, keepdims=True)
        self.agents, cells = _cells(data, self.train)
        A, T = len(self.agents), len(self.train)
        self.S = np.zeros((A, T))
        self.N = np.zeros((A, T))
        for a, t, k, n in cells:
            self.S[int(a), int(t)] += k
            self.N[int(a), int(t)] += n
        self.rate = (self.S.sum(1) + 0.5) / (self.N.sum(1) + 1.0)

    def _predict_all(self, data):
        test = list(data.test_tasks)
        X = self.source.get_features(test).astype(np.float64)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        sim = X @ self.Xtr.T
        for j, t in enumerate(test):
            nb = np.argsort(-sim[j])[: self.k]
            w = np.clip(sim[j, nb], 0, None) + 1e-6
            for i, a in enumerate(self.agents):
                s = (w * self.S[i, nb]).sum() + self.prior * self.rate[i]
                n = (w * self.N[i, nb]).sum() + self.prior
                self._p[(a, t)] = s / n
        self._set_difficulties(data, test)


class _Torch(_Base):
    """Shared training loop: binomial likelihood over training cells, Adam, weight decay."""

    def __init__(self, source, name, n_comp=64, dims=1, weight_decay=1e-2, epochs=600, lr=0.02, seed=0):
        super().__init__(source, name)
        self.n_comp, self.dims, self.wd, self.epochs, self.lr, self.seed = n_comp, dims, weight_decay, epochs, lr, seed

    def _features(self, task_ids, fit=False):
        X = self.source.get_features(task_ids).astype(np.float64)
        if fit:
            self.scaler = StandardScaler().fit(X)
            nc = min(self.n_comp, X.shape[0] - 1, X.shape[1])
            self.pca = PCA(nc, random_state=0).fit(self.scaler.transform(X))
            self.post = StandardScaler().fit(self.pca.transform(self.scaler.transform(X)))
        return self.post.transform(self.pca.transform(self.scaler.transform(X)))

    def fit(self, data, train_task_ids):
        self._p, self._predicted_difficulties = {}, {}
        torch.manual_seed(self.seed)
        train = list(train_task_ids)
        X = torch.tensor(self._features(train, fit=True), dtype=torch.float32)
        self.agents, cells = _cells(data, train)
        a = torch.tensor(cells[:, 0], dtype=torch.long)
        t = torch.tensor(cells[:, 1], dtype=torch.long)
        k = torch.tensor(cells[:, 2], dtype=torch.float32)
        n = torch.tensor(cells[:, 3], dtype=torch.float32)
        self._build(len(self.agents), X.shape[1])
        opt = torch.optim.Adam(self._task_params(), lr=self.lr, weight_decay=self.wd)
        opt_a = torch.optim.Adam(self._agent_params(), lr=self.lr)
        for _ in range(self.epochs):
            opt.zero_grad()
            opt_a.zero_grad()
            logits = self._logits(X[t], a)
            loss = -(k * torch.nn.functional.logsigmoid(logits)
                     + (n - k) * torch.nn.functional.logsigmoid(-logits)).sum() / n.sum()
            loss = loss + 1e-3 * self._agent_penalty()
            loss.backward()
            opt.step()
            opt_a.step()

    def _predict_all(self, data):
        test = list(data.test_tasks)
        X = torch.tensor(self._features(test), dtype=torch.float32)
        with torch.no_grad():
            for i, ag in enumerate(self.agents):
                a = torch.full((len(test),), i, dtype=torch.long)
                p = torch.sigmoid(self._logits(X, a)).numpy()
                for j, tt in enumerate(test):
                    self._p[(ag, tt)] = float(p[j])
        self._set_difficulties(data, test)


class AmortizedIRT(_Torch):
    def __init__(self, source, name="amortized_irt", **kw):
        super().__init__(source, name, dims=1, **kw)

    def _build(self, n_agents, d):
        self.theta = torch.zeros(n_agents, requires_grad=True)
        self.lin = torch.nn.Linear(d, 1)

    def _task_params(self):
        return list(self.lin.parameters())

    def _agent_params(self):
        return [self.theta]

    def _agent_penalty(self):
        return (self.theta ** 2).mean()

    def _logits(self, x, a):
        return self.theta[a] - self.lin(x).squeeze(-1)


class AmortizedMIRT(_Torch):
    def __init__(self, source, name="amortized_mirt", dims=4, **kw):
        super().__init__(source, name, dims=dims, **kw)

    def _build(self, n_agents, d):
        self.theta = torch.nn.Parameter(0.1 * torch.randn(n_agents, self.dims))
        self.disc = torch.nn.Linear(d, self.dims)
        self.diff = torch.nn.Linear(d, 1)

    def _task_params(self):
        return list(self.disc.parameters()) + list(self.diff.parameters())

    def _agent_params(self):
        return [self.theta]

    def _agent_penalty(self):
        return (self.theta ** 2).mean()

    def _logits(self, x, a):
        disc = torch.nn.functional.softplus(self.disc(x))
        return (disc * self.theta[a]).sum(-1) - self.diff(x).squeeze(-1)
