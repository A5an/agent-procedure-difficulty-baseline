"""Exp 1: progressive autonomy. How many shadow cases does a procedure need before it can be handed to an agent, with and
without documentation probes as the prior? No API calls, recorded outcomes only.

Shadow mode: real cases of the procedure arrive one by one; a human handles each, and every candidate agent is run on it
in the shadow, so we learn each agent's real outcome at the cost of the human's time. The MF model (../mf/model.py)
turns probes (prior) + shadow outcomes into a posterior over each agent's failure rate on this procedure.

Policies per procedure (options: the agents or the human; loss per case = API + F x fail, human h + F(1 - q)):
  fixed m   : shadow m cases, then pick the option with the lowest posterior expected loss
  sequential: after each shadow case, hand over to the best agent once P(agent's loss < human's loss) >= 0.9,
              otherwise keep shadowing, at most M_MAX cases, then pick by expected loss
Realized loss of a decision = loss from the agent's real failure rate on a held-out half of the procedure's cases (the
other half is the shadow pool), so every policy is scored on the same cases within a seed.
Total cost per procedure over V cases = shadow cases at the human's loss + shadow API + (V - m) x realized loss.

  python autonomy.py sop   (68 SOPBench procedures, leave-one-domain-out MF fits, agents flash / g38 / sonnet)
  python autonomy.py tib   (20 TauIndianBank families, leave-one-family-out MF fits, agents g38 / flash)
"""
import json, os, sys, collections
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mf"))
from model import MF, ZG, EG, VH, VW, LOGPHI_E

COSTS = [(20, 5, 0.99), (20, 2, 0.95), (50, 5, 0.99), (10, 5, 0.99)]
M_LIST = [0, 1, 2, 4, 8]
M_MAX, P_RELEASE, SEEDS, VOLS = 8, 0.9, 10, (50, 200)
rng_boot = np.random.default_rng(7)


@torch.no_grad()
def post_samples(m, target, k, nR, fR, nS, fS, S=400, seed=0):
    """posterior samples of the target agent's real failure probability on each unit (U, S)"""
    k, nR, fR, nS, fS = [torch.as_tensor(x) for x in (k, nR, fR, nS, fS)]
    text = torch.zeros(len(k))
    L = m._terms(k, nR, fR, nS, fS)
    lw = torch.log(VW).view(1, 1, 1, -1)
    others = [a for a in range(m.A) if a != target]
    Lo = torch.logsumexp(L[others] + lw.unsqueeze(0), -1).sum(0) if others else 0.0
    lp = m._logprior_z(k, text).unsqueeze(-1) + LOGPHI_E.view(1, 1, -1)
    post = (Lo + lp).unsqueeze(-1) + L[target] + lw
    post = post - torch.logsumexp(post.flatten(1), 1).view(-1, 1, 1, 1)
    P = m.p
    tau = torch.nn.functional.softplus(P["ltau"])[target, k] if m.agent_dev else torch.zeros(len(k))
    lr = P["mR"][target, k].view(-1, 1, 1, 1) + P["lR"][target, k].view(-1, 1, 1, 1) * ZG.view(1, -1, 1, 1) + tau.view(-1, 1, 1, 1) * VH.view(1, 1, 1, -1)
    p = torch.sigmoid(lr).expand(-1, len(ZG), len(EG), len(VH)).flatten(1).numpy()
    w = post.exp().flatten(1).numpy()
    r = np.random.default_rng(seed)
    out = np.empty((len(k), S))
    for u in range(len(k)):
        idx = r.choice(len(w[u]), size=S, p=w[u] / w[u].sum())
        out[u] = p[u, idx]
    return out


class Proc:
    """one held-out procedure: its units, probe counts per agent, real cases per agent (same case ids across agents)"""
    def __init__(self, name, model, agent_idx, units_k, probes, real_cases, api):
        self.name, self.m, self.ai, self.k = name, model, agent_idx, np.array(units_k)
        self.probes, self.cases, self.api = probes, real_cases, api   # probes[a] = list of (unit, fail); cases = list of (unit, {a: fail})

    def evidence(self, n_probe, shadow, seed):
        A, U = self.m.A, len(self.k)
        nS = np.zeros((A, U)); fS = np.zeros_like(nS); nR = np.zeros_like(nS); fR = np.zeros_like(nS)
        r = np.random.default_rng(seed)
        for a, ai in self.ai.items():
            pr = self.probes.get(a, [])
            if n_probe is not None: pr = [pr[i] for i in r.permutation(len(pr))[:n_probe]]
            for u, f in pr: nS[ai, u] += 1; fS[ai, u] += f
            for u, outs in shadow:
                nR[ai, u] += 1; fR[ai, u] += outs[a]
        return nR, fR, nS, fS

    def predicted(self, n_probe, shadow, seed, S=400):
        key = (n_probe, seed, len(shadow))                            # shadow prefix is fixed by (seed, length)
        if not hasattr(self, "_cache"): self._cache = {}
        if key not in self._cache: self._cache[key] = self._predicted(n_probe, shadow, seed, S)
        return self._cache[key]

    def _predicted(self, n_probe, shadow, seed, S):
        nR, fR, nS, fS = self.evidence(n_probe, shadow, seed)
        perf = sum(1 for u, _ in shadow if self.k[u] == 1)
        pi = (perf + 0.35 * 2) / (len(shadow) + 2)                   # class mix of incoming cases, smoothed
        out = {}
        for a, ai in self.ai.items():
            s = post_samples(self.m, ai, self.k, nR, fR, nS, fS, S=S, seed=seed)
            w = np.array([pi if kk == 1 else 1 - pi for kk in self.k]); w = w / w.sum()
            out[a] = (w[:, None] * s).sum(0)                          # samples of the procedure failure rate
        return out


def run_policy(procs, F, h, q, n_probe, policy, m_fixed, seed):
    hum = h + F * (1 - q)
    res = []
    for P in procs:
        r = np.random.default_rng([seed, sum(map(ord, P.name))])
        order = r.permutation(len(P.cases))
        n_pool = len(P.cases) // 2                        # first half: shadow pool, second half: evaluation (same for all policies)
        pool, held = order[:n_pool], order[n_pool:]
        shadow_n = 0
        while True:
            shadow = [P.cases[i] for i in pool[:shadow_n]]
            samp = P.predicted(n_probe, shadow, seed)
            exp_loss = {a: P.api[a] + F * samp[a].mean() for a in samp}
            best = min(exp_loss, key=exp_loss.get)
            if policy == "fixed":
                if shadow_n >= min(m_fixed, n_pool): break
            else:
                p_better = float(np.mean(P.api[best] + F * samp[best] < hum))
                if p_better >= P_RELEASE or shadow_n >= min(M_MAX, n_pool): break
            shadow_n += 1
        rest = [P.cases[i] for i in held]
        real = {a: np.mean([c[1][a] for c in rest]) for a in P.ai}
        if policy == "fixed":
            choice = best if exp_loss[best] < hum else "human"
        else:
            choice = best if (p_better >= P_RELEASE or exp_loss[best] < hum) else "human"
        realized = P.api[choice] + F * real[choice] if choice != "human" else hum
        oracle = min([P.api[a] + F * real[a] for a in P.ai] + [hum])
        truly_better = choice != "human" and realized < hum
        res.append(dict(proc=P.name, m=shadow_n, choice=choice, realized=realized, oracle=oracle,
                        shadow_cost=shadow_n * (hum + sum(P.api.values())), wrong_release=choice != "human" and not truly_better))
    return res


def summarize(res, V):
    tot = np.array([(r["shadow_cost"] + (V - r["m"]) * r["realized"]) / V for r in res])
    return dict(loss_per_case=float(np.mean([r["realized"] for r in res])), total_per_case_V=float(tot.mean()),
                mean_shadow=float(np.mean([r["m"] for r in res])), released=float(np.mean([r["choice"] != "human" for r in res])),
                wrong_release=float(np.mean([r["wrong_release"] for r in res])), oracle=float(np.mean([r["oracle"] for r in res])))


def load_sop():
    sys.argv = sys.argv[:1]
    import eval2 as M2
    E = M2.E
    AG = ["flash", "g38", "sonnet"]; API = {"flash": 0.008, "g38": 0.021, "sonnet": 0.05}
    procs = []
    for F_ in M2.folds:
        mdl = F_.models["mf"]
        for proc, g in F_.test_units.groupby("proc"):
            us = list(g.u.values); loc = {u: i for i, u in enumerate(us)}
            probes = {a: [(loc[r.u], int(r.fail)) for r in F_.te_syn[(F_.te_syn.u.isin(us)) & (F_.te_syn.a == E.AG.index(a))].itertuples()] for a in AG}
            rr = F_.te_real[F_.te_real.u.isin(us)]
            cases = []
            for cid, gg in rr.groupby("case_id"):
                outs = {E.AG[int(x.a)]: int(x.fail) for x in gg.itertuples()}
                if all(a in outs for a in AG): cases.append((loc[gg.u.iloc[0]], outs))
            if len(cases) >= 2:
                procs.append(Proc(proc, mdl, {a: E.AG.index(a) for a in AG}, list(g.perform.values), probes, cases, API))
    return procs


def load_tib():
    T = os.path.join(HERE, "..", "tib")
    sel = json.load(open(os.path.join(T, "selection.json")))
    F = sel["test_families"]; AG = ["g38", "flash"]
    def fam(t):
        return next(f for f in F if f"_{f}_" in t)
    real = {a: {} for a in AG}; probe = {a: collections.defaultdict(list) for a in AG}
    for a in AG:
        for l in open(os.path.join(T, "runs", f"real_{a}.jsonl")):
            r = json.loads(l); real[a][r["task_id"]] = 1 - int(r["reward"] >= 1)
        for l in open(os.path.join(T, "runs", f"probes_{a}.v2.jsonl")):
            r = json.loads(l); probe[a][fam(r["task_id"])].append(1 - int(r["reward"] >= 1))
    res = json.load(open(os.path.join(T, "results_v2.json")))
    api = res["api_usd_per_episode_agent_only"]
    cls = np.array([sel["classes"][f] for f in F])
    nR = np.array([[sum(1 for t in sel["real_tasks"][f]) for f in F] for a in AG], float)
    fR = np.array([[sum(real[a][t] for t in sel["real_tasks"][f]) for f in F] for a in AG], float)
    nS = np.array([[len(probe[a][f]) for f in F] for a in AG], float); fS = np.array([[sum(probe[a][f]) for f in F] for a in AG], float)
    procs = []
    for j, f in enumerate(F):
        tr = [i for i in range(len(F)) if i != j]
        mdl = MF(2).fit((cls[tr], np.zeros(len(tr)), nR[:, tr], fR[:, tr], nS[:, tr], fS[:, tr]))
        cases = [(0, {a: real[a][t] for a in AG}) for t in sel["real_tasks"][f]]
        procs.append(Proc(f, mdl, {"g38": 0, "flash": 1}, [cls[j]], {a: [(0, x) for x in probe[a][f]] for a in AG}, cases, api))
    return procs


if __name__ == "__main__":
    which = sys.argv[1]
    procs = load_sop() if which == "sop" else load_tib()
    m_list = M_LIST if which == "sop" else [0, 1, 2, 3, 4]
    probe_n = [None, 4, 0] if which == "sop" else [None, 0]          # None = all probes, 0 = no probes
    out = {}
    for F, h, q in COSTS:
        key = f"F={F} h={h} q={q}"; out[key] = {}
        for npb in probe_n:
            lab = "no probes" if npb == 0 else ("all probes" if npb is None else f"{npb} probes")
            for mm in m_list:
                rs = [run_policy(procs, F, h, q, npb, "fixed", mm, s) for s in range(SEEDS)]
                out[key][f"{lab}, {mm} shadow"] = {f"V={V}": {k: float(np.mean([summarize(r, V)[k] for r in rs])) for k in summarize(rs[0], V)} for V in VOLS}
            rs = [run_policy(procs, F, h, q, npb, "seq", None, s) for s in range(SEEDS)]
            out[key][f"{lab}, sequential"] = {f"V={V}": {k: float(np.mean([summarize(r, V)[k] for r in rs])) for k in summarize(rs[0], V)} for V in VOLS}
        print("\n" + key, flush=True)
        for k, v in out[key].items():
            s = v["V=200"]
            print(f"  {k:<28} loss/case {s['loss_per_case']:.2f}  total/case V200 {s['total_per_case_V']:.2f}  V50 {v['V=50']['total_per_case_V']:.2f}  shadow {s['mean_shadow']:.1f}  released {s['released']:.2f}  wrong release {s['wrong_release']:.2f}  oracle {s['oracle']:.2f}", flush=True)
    json.dump(out, open(os.path.join(HERE, f"results_{which}.json"), "w"), indent=1)
