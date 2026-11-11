"""Leave-one-domain-out evaluation of the multi-fidelity model against simple probe baselines. No API calls.

For every held-out domain the model is fitted on the other six domains (real + probe outcomes of the four agents), then
predicts the real failure rate of each agent on each held-out (procedure, class) unit from the evidence of the scenario.
Metrics on the held-out real cases: within-domain pair-weighted Spearman of the procedure failure rate (benchmark mix
0.35 perform + 0.65 refuse), case log-loss, Brier, ECE. Intervals: bootstrap over procedures.
"""
import os, sys, json, time, itertools
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import MF

AG = ["lite", "flash", "g38", "sonnet"]
W_PERF = 0.35
rng = np.random.default_rng(20261102)
df = pd.read_csv(os.path.join(HERE, "outcomes.csv"))
prior = pd.read_csv(os.path.join(HERE, "text_prior.csv"))
df = df[df.fid != "public"]
units = df[["proc", "domain", "perform"]].drop_duplicates().sort_values(["proc", "perform"]).reset_index(drop=True)
units["u"] = np.arange(len(units))
units = units.merge(prior, on=["proc", "perform"], how="left")
units["text_prior"] = units.text_prior.fillna(units.text_prior.mean())
units["text_z"] = (units.text_prior - units.text_prior.mean()) / units.text_prior.std()
U = len(units)
uix = {(r.proc, r.perform): r.u for r in units.itertuples()}
df["u"] = [uix[(p, k)] for p, k in zip(df.proc, df.perform)]
df["a"] = df.agent.map({a: i for i, a in enumerate(AG)})
REAL = df[df.fid == "real"].reset_index(drop=True)
SYN = df[df.fid == "synth"].reset_index(drop=True)


def counts(rows, sel_units=None):
    n = np.zeros((len(AG), U)); f = np.zeros((len(AG), U))
    np.add.at(n, (rows.a.values, rows.u.values), 1); np.add.at(f, (rows.a.values, rows.u.values), rows.fail.values)
    return n, f


def subsample_probes(syn, n_per_proc, seed):
    """keep n probes per (agent, procedure), class mix as drawn"""
    if n_per_proc is None: return syn
    r = np.random.default_rng(seed)
    keep = []
    for (a, p), g in syn.groupby(["a", "proc"]):
        idx = g.index.values
        keep.extend(r.choice(idx, size=min(n_per_proc, len(idx)), replace=False))
    return syn.loc[np.sort(keep)]


def proc_rate(p_unit, units_sub):
    """benchmark-mix failure rate per procedure from per-unit rates"""
    out = {}
    for proc, g in units_sub.groupby("proc"):
        d = {int(r.perform): p_unit[r.u] for r in g.itertuples()}
        if len(d) == 2: out[proc] = W_PERF * d[1] + (1 - W_PERF) * d[0]
        else: out[proc] = list(d.values())[0]
    return out


def pw_spearman(pred, obs, dom_of):
    num = den = 0.0
    for d in set(dom_of.values()):
        ps = [p for p in pred if dom_of[p] == d and p in obs]
        if len(ps) < 3: continue
        x = [pred[p] for p in ps]; y = [obs[p] for p in ps]
        if np.ptp(y) < 1e-9: continue
        w = len(ps) * (len(ps) - 1) / 2
        r = spearmanr(x, y).correlation if np.ptp(x) > 1e-9 else 0.0   # a constant prediction carries no ranking
        num += w * r; den += w
    return num / den if den else float("nan")


def case_metrics(p_case, y):
    p = np.clip(p_case, 1e-4, 1 - 1e-4)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    br = (p - y) ** 2
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).sum() for b in range(10) if (bins == b).any()) / len(y)
    return ll, br, ece


class Fold:
    def __init__(self, test_dom):
        self.test_units = units[units.domain == test_dom]
        tr_u = set(units.u) - set(self.test_units.u)
        self.tr_real = REAL[REAL.u.isin(tr_u)]; self.tr_syn = SYN[SYN.u.isin(tr_u)]
        self.te_real = REAL[REAL.u.isin(self.test_units.u)]; self.te_syn = SYN[SYN.u.isin(self.test_units.u)]
        self.models = {}

    def fit(self, name, **kw):
        nR, fR = counts(self.tr_real); nS, fS = counts(self.tr_syn)
        tr = units[~units.u.isin(self.test_units.u)].u.values
        k = units.perform.values; t = units.text_z.values
        m = MF(len(AG), **kw)
        m.fit((k[tr], t[tr], nR[:, tr], fR[:, tr], nS[:, tr], fS[:, tr]))
        self.models[name] = m
        return m

    # ---- baselines fitted on training units ----
    def const(self):
        r = self.tr_real.groupby(["a", "perform"]).fail.mean()
        return lambda a, k: r[(a, k)]

    def calib(self, a, k_cls):
        """logistic map from smoothed own-probe rate to real failure, fitted on training units of agent a, class k"""
        nR, fR = counts(self.tr_real); nS, fS = counts(self.tr_syn)
        tr = units[(~units.u.isin(self.test_units.u)) & (units.perform == k_cls)].u.values
        x = np.log((fS[a, tr] + .5) / (nS[a, tr] - fS[a, tr] + .5))
        ok = (nR[a, tr] > 0) & (nS[a, tr] > 0); x, n, f = x[ok], nR[a, tr][ok], fR[a, tr][ok]
        from scipy.optimize import minimize
        def nll(w):
            z = w[0] + w[1] * x
            return -(f * -np.logaddexp(0, -z) + (n - f) * -np.logaddexp(0, z)).sum() + 0.01 * (w ** 2).sum()
        w = minimize(nll, np.array([0., 1.])).x
        return lambda nS_, fS_: 1 / (1 + np.exp(-(w[0] + w[1] * np.log((fS_ + .5) / (nS_ - fS_ + .5)))))


def evaluate(scen, folds, n_probe=None, probe_agents="own", pilots=0, model_name="mf", seeds=(0,)):
    """returns per-agent dict of metrics for MF and baselines, pooled over the folds"""
    out = {a: {} for a in AG}
    for ti, tgt in enumerate(AG):
        recs = {m: dict(ll=[], br=[], y=[], proc=[], ece_p=[]) for m in ("const", "raw", "calib", model_name)}
        rhos = {m: [] for m in recs}
        for seed in seeds:
            pred_proc = {m: {} for m in recs}; obs_proc = {}; dom_of = {}
            for F in folds:
                te_u = F.test_units.u.values; k = units.perform.values; t = units.text_z.values
                syn = subsample_probes(F.te_syn, n_probe, seed)
                if probe_agents == "own": syn = syn[syn.a == ti]
                elif probe_agents == "others": syn = syn[syn.a != ti]
                elif probe_agents == "none": syn = syn.iloc[0:0]
                real_t = F.te_real[F.te_real.a == ti]
                # pilots: reveal `pilots` real cases of the target per procedure, evaluate on the rest
                if pilots:
                    r = np.random.default_rng(1000 + seed)
                    piv = []
                    for p, g in real_t.groupby("proc"):
                        piv.extend(r.choice(g.index.values, size=min(pilots, len(g) - 1), replace=False) if len(g) > 1 else [])
                    pil = real_t.loc[piv]; ev_real = real_t.drop(index=piv)
                else:
                    pil = real_t.iloc[0:0]; ev_real = real_t
                nS, fS = counts(syn); nR, fR = counts(pil)
                m = F.models[model_name]
                p_mf = m.predict(ti, k[te_u], t[te_u], nR[:, te_u], fR[:, te_u], nS[:, te_u], fS[:, te_u])
                p_unit = {model_name: dict(zip(te_u, p_mf))}
                c = F.const(); p_unit["const"] = {u: c(ti, int(units.perform[u])) for u in te_u}
                # raw and calibrated own-probe rate (fall back to const where the unit has no own probes); pilots pooled into raw
                cal = {kk: F.calib(ti, kk) for kk in (0, 1)}
                nSo, fSo = nS[ti] + nR[ti], fS[ti] + fR[ti]
                p_unit["raw"] = {u: ((fSo[u] + .5) / (nSo[u] + 1) if nSo[u] > 0 else p_unit["const"][u]) for u in te_u}
                p_unit["calib"] = {u: (float(cal[int(units.perform[u])](nS[ti, u], fS[ti, u])) if nS[ti, u] > 0 else p_unit["const"][u]) for u in te_u}
                for mname in recs:
                    pu = np.array([p_unit[mname][u] for u in range(U)] if False else [0.0] * U)
                    for u in te_u: pu[u] = p_unit[mname][u]
                    pr = proc_rate(pu, F.test_units)
                    pred_proc[mname].update(pr)
                    y = ev_real.fail.values.astype(float); pc = pu[ev_real.u.values]
                    ll, br, _ = case_metrics(pc, y)
                    recs[mname]["ll"].extend(ll); recs[mname]["br"].extend(br); recs[mname]["y"].extend(y)
                    recs[mname]["proc"].extend(ev_real.proc.values); recs[mname]["ece_p"].extend(pc)
                ob = {}
                for u in te_u:
                    g = ev_real[ev_real.u == u]
                    if len(g): ob[u] = g.fail.mean()
                obu = np.full(U, np.nan)
                for u, v in ob.items(): obu[u] = v
                for proc, g in F.test_units.groupby("proc"):
                    d = {int(r.perform): obu[r.u] for r in g.itertuples() if not np.isnan(obu[r.u])}
                    if not d: continue
                    obs_proc[proc] = W_PERF * d[1] + (1 - W_PERF) * d[0] if len(d) == 2 else list(d.values())[0]
                    dom_of[proc] = g.domain.iloc[0]
            for mname in recs:
                rhos[mname].append(pw_spearman(pred_proc[mname], obs_proc, dom_of))
        for mname, rr in recs.items():
            p = np.array(rr["ece_p"]); y = np.array(rr["y"])
            _, _, ece = case_metrics(p, y)
            out[tgt][mname] = dict(rho=float(np.mean(rhos[mname])), ll=float(np.mean(rr["ll"])), brier=float(np.mean(rr["br"])), ece=float(ece), n=len(y))
        # bootstrap over procedures: MF minus calib, log-loss
        procs = np.array(recs[model_name]["proc"]); up = np.unique(procs)
        dl = np.array(recs[model_name]["ll"]) - np.array(recs["calib"]["ll"])
        db = np.array(recs[model_name]["br"]) - np.array(recs["const"]["br"])
        idx = {p: np.where(procs == p)[0] for p in up}
        bs_l, bs_b = [], []
        for _ in range(1000):
            pick = rng.choice(up, len(up)); ii = np.concatenate([idx[p] for p in pick])
            bs_l.append(dl[ii].mean()); bs_b.append(db[ii].mean())
        out[tgt]["mf_minus_calib_ll"] = (float(dl.mean()), float(np.percentile(bs_l, 2.5)), float(np.percentile(bs_l, 97.5)))
        out[tgt]["mf_minus_const_brier"] = (float(db.mean()), float(np.percentile(bs_b, 2.5)), float(np.percentile(bs_b, 97.5)))
    return out


def show(title, res, model_name="mf"):
    print(f"\n### {title}")
    print(f"{'agent':<7}{'method':<8}{'rho':>7}{'logloss':>9}{'brier':>8}{'ece':>7}")
    for a in AG:
        for m in ("const", "raw", "calib", model_name):
            r = res[a][m]
            print(f"{a:<7}{m:<8}{r['rho']:7.3f}{r['ll']:9.4f}{r['brier']:8.4f}{r['ece']:7.3f}")
        d = res[a]["mf_minus_calib_ll"]; e = res[a]["mf_minus_const_brier"]
        print(f"{'':<7}MF - calib logloss {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}];  MF - const brier {e[0]:+.4f} [{e[1]:+.4f}, {e[2]:+.4f}]")


if __name__ == "__main__":
    t0 = time.time()
    folds = [Fold(d) for d in sorted(units.domain.unique())]
    variants = {"mf": {}, "mf_text": {"use_text": True}, "mf_noeps": {"shared_eps": False}, "mf_nodev": {"agent_dev": False}}
    which = sys.argv[1:] or list(variants)
    for F in folds:
        for v in which: F.fit(v, **variants[v])
    print(f"fitted {len(folds)} folds x {len(which)} variants in {time.time() - t0:.0f}s")
    print(json.dumps(folds[0].models["mf"].summary(AG), indent=0)[:3000])
    R = {}
    R["own probes, all"] = evaluate("own", folds)
    show("own probes, all (~20 per procedure)", R["own probes, all"])
    R["all agents' probes"] = evaluate("all", folds, probe_agents="all")
    show("probes of all four agents", R["all agents' probes"])
    R["other agents' probes only"] = evaluate("others", folds, probe_agents="others")
    show("probes of the other three agents only (target has no probes)", R["other agents' probes only"])
    R["no evidence"] = evaluate("none", folds, probe_agents="none")
    show("no evidence (prior only)", R["no evidence"])
    for n in (2, 4, 8):
        R[f"own probes n={n}"] = evaluate("own", folds, n_probe=n, seeds=range(5))
        show(f"own probes, {n} per procedure (5 draws)", R[f"own probes n={n}"])
    for m in (1, 2, 4):
        R[f"no probes + {m} pilots"] = evaluate("none", folds, probe_agents="none", pilots=m, seeds=range(5))
        show(f"no probes, {m} real pilot cases per procedure (5 draws)", R[f"no probes + {m} pilots"])
        R[f"own probes + {m} pilots"] = evaluate("own", folds, pilots=m, seeds=range(5))
        show(f"own probes + {m} real pilot cases (5 draws)", R[f"own probes + {m} pilots"])
    for v in which[1:]:
        R[f"{v}: own probes"] = evaluate("own", folds, model_name=v)
        show(f"variant {v}, own probes", R[f"{v}: own probes"], model_name=v)
    json.dump(R, open(os.path.join(HERE, "results_lodo.json"), "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s")
