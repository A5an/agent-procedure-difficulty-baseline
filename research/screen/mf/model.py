"""Multi-fidelity hierarchical model of agent failure per procedure (SOPBench).

Unit u = (procedure, class), class = perform or refuse case. For agent a and unit u:

    real  : logit P(fail) = mR[a,k] + lR[a,k] * z_u + v_au
    synth : logit P(fail) = mS[a,k] + lS[a,k] * z_u + kap[a,k] * v_au + eps_u

    z_u   ~ N(b0[k] + b1[k] * text_u, 1)   shared difficulty of the unit, all agents
    v_au  ~ N(0, tau[a,k]^2)               this agent's own deviation on this unit, shared by its probes and real runs
    eps_u ~ N(0, se[k]^2)                  how far the LLM-written probes of this unit are off, shared by all agents

Outcomes are binomial counts given the latents. z and eps are integrated on a grid, v by Gauss-Hermite quadrature,
so the marginal likelihood is exact up to the grid and the parameters are fitted by maximum marginal likelihood
(empirical Bayes) on training procedures. Prediction for a held-out unit = posterior mean of the real failure
probability of a target agent given whatever evidence exists for that unit (probes of any agents, a few real pilots).
"""
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
ZG = torch.linspace(-5, 5, 41)
EG = torch.linspace(-3.5, 3.5, 21)
_h, _w = np.polynomial.hermite_e.hermegauss(11)
VH = torch.tensor(_h); VW = torch.tensor(_w / _w.sum())
LOGPHI_E = -0.5 * EG ** 2; LOGPHI_E = LOGPHI_E - torch.logsumexp(LOGPHI_E, 0)


def logsig(x):
    return torch.nn.functional.logsigmoid(x)


class MF:
    def __init__(self, n_agents, use_text=False, l2=0.01, shared_eps=True, agent_dev=True):
        A = n_agents
        self.A, self.use_text, self.l2, self.shared_eps, self.agent_dev = A, use_text, l2, shared_eps, agent_dev
        z = lambda *s: torch.zeros(*s, requires_grad=True)
        self.p = dict(mR=z(A, 2), lR=torch.ones(A, 2, requires_grad=True), mS=z(A, 2), lS=torch.ones(A, 2, requires_grad=True),
                      kap=torch.full((A, 2), 0.5, requires_grad=True), ltau=torch.full((A, 2), -0.5, requires_grad=True),
                      lse=torch.full((2,), -0.5, requires_grad=True), b0=z(2), b1=z(2))

    # ---- core: log-likelihood tensors ----
    def _terms(self, k, nR, fR, nS, fS):
        """per agent log-lik over grid: returns L[a,u,z,e,v] (A,U,Gz,Ge,V). k: (U,) class index."""
        P = self.p
        tau = torch.nn.functional.softplus(P["ltau"])[:, k] if self.agent_dev else torch.zeros(self.A, len(k))
        se = torch.nn.functional.softplus(P["lse"])[k] if self.shared_eps else torch.zeros(len(k))
        z = ZG.view(1, 1, -1, 1, 1); e = EG.view(1, 1, 1, -1, 1); v = VH.view(1, 1, 1, 1, -1)
        g = lambda name: P[name][:, k].view(self.A, -1, 1, 1, 1)
        vv = tau.view(self.A, -1, 1, 1, 1) * v
        lr = g("mR") + g("lR") * z + vv
        ls = g("mS") + g("lS") * z + g("kap") * vv + se.view(1, -1, 1, 1, 1) * e
        c = lambda t: t.view(self.A, -1, 1, 1, 1)
        L = c(fR) * logsig(lr) + c(nR - fR) * logsig(-lr) + c(fS) * logsig(ls) + c(nS - fS) * logsig(-ls)
        return L

    def _logprior_z(self, k, text):
        mu = self.p["b0"][k] + (self.p["b1"][k] * text if self.use_text else 0.0)
        lp = -0.5 * (ZG.view(1, -1) - mu.view(-1, 1)) ** 2
        return lp - torch.logsumexp(lp, 1, keepdim=True)          # (U,Gz)

    def marginal_ll(self, k, text, nR, fR, nS, fS):
        L = self._terms(k, nR, fR, nS, fS)                          # A,U,Gz,Ge,V
        La = torch.logsumexp(L + torch.log(VW).view(1, 1, 1, 1, -1), -1).sum(0)  # U,Gz,Ge
        lp = self._logprior_z(k, text).unsqueeze(-1) + LOGPHI_E.view(1, 1, -1)
        return torch.logsumexp((La + lp).flatten(1), 1)             # U

    def fit(self, data, iters=300, verbose=False):
        k, text, nR, fR, nS, fS = [torch.as_tensor(x) for x in data]
        params = list(self.p.values())
        opt = torch.optim.LBFGS(params, lr=0.5, max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-6)
        n_obs = float((nR.sum() + nS.sum()))

        def closure():
            opt.zero_grad()
            ll = self.marginal_ll(k, text, nR, fR, nS, fS).sum()
            pen = self.l2 * sum(((v - (1.0 if n in ("lR", "lS") else 0.0)) ** 2).sum() for n, v in self.p.items())
            loss = -(ll) / n_obs * 1000 + pen
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            self.train_ll = float(self.marginal_ll(k, text, nR, fR, nS, fS).sum())
        return self

    @torch.no_grad()
    def predict(self, target, k, text, nR, fR, nS, fS):
        """posterior mean P(real fail) of agent `target` on each unit, given evidence counts (A,U)."""
        k, text, nR, fR, nS, fS = [torch.as_tensor(x) for x in (k, text, nR, fR, nS, fS)]
        L = self._terms(k, nR, fR, nS, fS)                          # A,U,Gz,Ge,V
        lw = torch.log(VW).view(1, 1, 1, -1)
        others = [a for a in range(self.A) if a != target]
        Lo = torch.logsumexp(L[others] + lw.unsqueeze(0), -1).sum(0) if others else 0.0   # U,Gz,Ge
        lp = self._logprior_z(k, text).unsqueeze(-1) + LOGPHI_E.view(1, 1, -1)
        post = (Lo + lp).unsqueeze(-1) + L[target] + lw             # U,Gz,Ge,V
        post = post - torch.logsumexp(post.flatten(1), 1).view(-1, 1, 1, 1)
        P = self.p
        tau = torch.nn.functional.softplus(P["ltau"])[target, k] if self.agent_dev else torch.zeros(len(k))
        lr = P["mR"][target, k].view(-1, 1, 1, 1) + P["lR"][target, k].view(-1, 1, 1, 1) * ZG.view(1, -1, 1, 1) + tau.view(-1, 1, 1, 1) * VH.view(1, 1, 1, -1)
        return (post.exp() * torch.sigmoid(lr)).flatten(1).sum(1).numpy()

    def summary(self, names):
        sp = torch.nn.functional.softplus
        out = {}
        for i, a in enumerate(names):
            for kk, cl in enumerate(("refuse", "perform")):
                out[f"{a}/{cl}"] = {n: round(float(self.p[n][i, kk]), 3) for n in ("mR", "lR", "mS", "lS", "kap")}
                out[f"{a}/{cl}"]["tau"] = round(float(sp(self.p["ltau"][i, kk])), 3)
        out["se"] = [round(float(x), 3) for x in sp(self.p["lse"])]
        out["b0"] = [round(float(x), 3) for x in self.p["b0"]]; out["b1"] = [round(float(x), 3) for x in self.p["b1"]]
        return out
