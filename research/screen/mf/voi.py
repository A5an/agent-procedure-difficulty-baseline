"""How many probes to buy, and for which agent: value-of-information probing vs a fixed number (no API calls).

Per held-out procedure (leave-one-domain-out models), probes of flash / g38 / sonnet arrive one at a time from that
agent's recorded probe list (random order per seed). After each probe the model posterior gives P(fail) per agent;
the routing choice is the option with the lowest expected loss (agents or human, as in route.py).

Policies with the same total probe budget B per procedure:
  fixed  : B/3 probes per agent
  voi    : each next probe goes to the agent with the largest expected value of sample information (myopic EVSI:
           expected drop of the minimum expected loss after seeing that probe's outcome)
  voi-stop: like voi, but stops early when the best EVSI is below the probe's price (prices per probe below)
Realized loss uses the real per-procedure failure rates. Paired bootstrap over procedures.
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv = sys.argv[:1]
import eval2 as M
E = M.E

API = {"flash": 0.008, "g38": 0.021, "sonnet": 0.05}
PROBE_USD = {"flash": 0.01, "g38": 0.0165, "sonnet": 0.08}     # per probe episode, rough, from the ledgers
AGS = list(API); TI = [M.AG.index(a) for a in AGS]
units = M.units; PROCS = M.PROCS
REAL_FAIL = {a: M.obs_proc(M.AG.index(a)) for a in AGS}
FOLD_OF = {}
for F in M.folds:
    for p in F.test_units.proc.unique(): FOLD_OF[p] = F


def proc_units(p):
    g = units[units.proc == p]
    return g.u.values, g.perform.values.astype(int)


class State:
    """probe counts for one procedure: per agent, per unit"""
    def __init__(self, p):
        self.p = p; self.u, self.k = proc_units(p)
        self.nS = np.zeros((len(M.AG), len(self.u))); self.fS = np.zeros_like(self.nS)
        self.F = FOLD_OF[p]

    def post(self, nS=None, fS=None):
        nS = self.nS if nS is None else nS; fS = self.fS if fS is None else fS
        z = np.zeros_like(nS); t = units.text_z.values[self.u]
        out = {}
        for a, ti in zip(AGS, TI):
            pu = self.F.models["mf"].predict(ti, self.k, t, z, z, nS, fS)
            d = dict(zip(self.k, pu))
            out[a] = E.W_PERF * d[1] + (1 - E.W_PERF) * d[0] if len(d) == 2 else list(d.values())[0]
            out[a + "_unit"] = dict(zip(self.k, pu))
        return out

    def exp_losses(self, post, F, h, q):
        return np.array([API[a] + F * post[a] for a in AGS] + [h + F * (1 - q)])


def run(policy, B, F, h, q, seed, price=False):
    r = np.random.default_rng(seed)
    loss = np.zeros(len(PROCS)); used = np.zeros(len(PROCS)); usd = np.zeros(len(PROCS))
    L = np.vstack([API[a] + F * REAL_FAIL[a] for a in AGS] + [np.full(len(PROCS), h + F * (1 - q))])
    for i, p in enumerate(PROCS):
        S = State(p); syn = S.F.te_syn[S.F.te_syn.proc == p]
        queue = {a: syn[syn.a == ti].sample(frac=1, random_state=int(r.integers(1e9))) for a, ti in zip(AGS, TI)}
        ptr = {a: 0 for a in AGS}
        def add(a, nS, fS, outcome=None):
            row = queue[a].iloc[ptr[a]]; j = list(S.u).index(row.u); ti = M.AG.index(a)
            nS = nS.copy(); fS = fS.copy(); nS[ti, j] += 1; fS[ti, j] += row.fail if outcome is None else outcome
            return nS, fS, row
        steps = 0
        while steps < B:
            avail = [a for a in AGS if ptr[a] < len(queue[a])]
            if not avail: break
            if policy == "fixed":
                a = min(avail, key=lambda x: (ptr[x], AGS.index(x)))
            else:
                post = S.post(); cur = S.exp_losses(post, F, h, q).min()
                best, best_v = None, -1
                for a_ in avail:
                    row = queue[a_].iloc[ptr[a_]]; j = list(S.u).index(row.u)
                    pf = post[a_ + "_unit"][S.k[j]]
                    ev = 0.0
                    for y, py in ((1, pf), (0, 1 - pf)):
                        nS2, fS2, _ = add(a_, S.nS, S.fS, outcome=y)
                        ev += py * S.exp_losses(S.post(nS2, fS2), F, h, q).min()
                    v = cur - ev - (PROBE_USD[a_] if price else 0.0)
                    if v > best_v: best, best_v = a_, v
                if price and best_v <= 0: break
                a = best
            S.nS, S.fS, _ = add(a, S.nS, S.fS); ptr[a] += 1; steps += 1; usd[i] += PROBE_USD[a]
        ch = S.exp_losses(S.post(), F, h, q).argmin()
        loss[i] = L[ch, i]; used[i] = steps
    return loss, used, usd


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    out = {}
    for F, h, q in ((20, 5, 0.99), (20, 2, 0.95), (50, 5, 0.99)):
        key = f"F={F} h={h} q={q}"; out[key] = {}
        res = {}
        for B in (3, 6, 12):
            for pol in ("fixed", "voi"):
                ls = [run(pol, B, F, h, q, s) for s in range(5)]
                res[(pol, B)] = (np.mean([x[0] for x in ls], 0), np.mean([x[1].mean() for x in ls]), np.mean([x[2].sum() for x in ls]))
        ls = [run("voi", 30, F, h, q, s, price=True) for s in range(5)]
        res[("voi-stop", 30)] = (np.mean([x[0] for x in ls], 0), np.mean([x[1].mean() for x in ls]), np.mean([x[2].sum() for x in ls]))
        P = len(PROCS)
        for (pol, B), (l, u, d) in res.items():
            out[key][f"{pol} B={B}"] = {"loss": float(l.mean()), "probes_per_proc": float(u), "probe_usd_68_procs": float(d)}
        for B in (3, 6, 12):
            dd = res[("voi", B)][0] - res[("fixed", B)][0]
            bs = [dd[rng.integers(0, P, P)].mean() for _ in range(2000)]
            out[key][f"voi - fixed B={B}"] = (float(dd.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
        dd = res[("voi-stop", 30)][0] - res[("fixed", 12)][0]
        bs = [dd[rng.integers(0, P, P)].mean() for _ in range(2000)]
        out[key]["voi-stop - fixed B=12"] = (float(dd.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
        print("\n" + key)
        for k, v in out[key].items(): print("  ", k, v)
    json.dump(out, open(os.path.join(HERE, "voi.json"), "w"), indent=1)
