"""Main numbers for the multi-fidelity model (leave-one-domain-out, no API calls).

A. probe-count curve: rho and log-loss of raw probe rate, calibrated probe rate and the model, n = 1, 2, 4, 8, 12, all
   own probes per procedure, 10 random draws; paired bootstrap over procedures for model minus raw at the same n and
   for the model at n probes minus the raw rate with all probes.
B. ceiling: split-half reliability of the real per-procedure failure rate (how high rho can go at all).
C. routing: per procedure choose flash / g38 / sonnet / human by predicted loss, realized loss on the real cases.
"""
import os, sys, json, collections
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv = sys.argv[:1]
import eval as E

AG, U, units = E.AG, E.U, E.units
rng = np.random.default_rng(11)
folds = [E.Fold(d) for d in sorted(units.domain.unique())]
for F in folds: F.fit("mf")
PROCS = sorted(units.proc.unique()); DOM = dict(zip(units.proc, units.domain))


def unit_preds(ti, n_probe, seed, probe_agents="own"):
    """per-unit predicted real failure for target agent ti: dict method -> array(U)"""
    out = {m: np.full(U, np.nan) for m in ("const", "raw", "calib", "mf")}
    for F in folds:
        te = F.test_units.u.values; k = units.perform.values; t = units.text_z.values
        syn = E.subsample_probes(F.te_syn, n_probe, seed)
        syn = syn[syn.a == ti] if probe_agents == "own" else syn
        nS, fS = E.counts(syn); z = np.zeros_like(nS)
        out["mf"][te] = F.models["mf"].predict(ti, k[te], t[te], z[:, te], z[:, te], nS[:, te], fS[:, te])
        c = F.const(); cal = {kk: F.calib(ti, kk) for kk in (0, 1)}
        for u in te:
            kk = int(units.perform[u]); out["const"][u] = c(ti, kk)
            out["raw"][u] = (fS[ti, u] + .5) / (nS[ti, u] + 1) if nS[ti, u] > 0 else out["const"][u]
            out["calib"][u] = float(cal[kk](nS[ti, u], fS[ti, u])) if nS[ti, u] > 0 else out["const"][u]
    return out


def proc_vec(pu):
    r = E.proc_rate(pu, units)
    return np.array([r[p] for p in PROCS])


REAL = E.REAL
def obs_proc(ti, rows=None):
    rows = REAL[REAL.a == ti] if rows is None else rows
    pu = np.full(U, np.nan)
    for u, g in rows.groupby("u"): pu[u] = g.fail.mean()
    out = []
    for p in PROCS:
        g = units[units.proc == p]
        d = {int(r.perform): pu[r.u] for r in g.itertuples() if not np.isnan(pu[r.u])}
        out.append(E.W_PERF * d[1] + (1 - E.W_PERF) * d[0] if len(d) == 2 else (list(d.values())[0] if d else np.nan))
    return np.array(out)


def pw_rho(pred, obs, idx=None):
    idx = np.arange(len(PROCS)) if idx is None else idx
    num = den = 0.0
    doms = np.array([DOM[PROCS[i]] for i in idx])
    for d in np.unique(doms):
        ii = idx[doms == d]
        x, y = pred[ii], obs[ii]
        ok = ~np.isnan(y) & ~np.isnan(x); x, y = x[ok], y[ok]
        if len(x) < 3 or np.ptp(y) < 1e-9: continue
        r = spearmanr(x, y).correlation if np.ptp(x) > 1e-9 else 0.0
        w = len(x) * (len(x) - 1) / 2; num += w * r; den += w
    return num / den if den else np.nan


def boot_idx(B=1000):
    """resample procedures within domain"""
    by = collections.defaultdict(list)
    for i, p in enumerate(PROCS): by[DOM[p]].append(i)
    for _ in range(B):
        yield np.concatenate([rng.choice(v, len(v)) for v in by.values()])


def case_ll(pu, ti):
    rows = REAL[REAL.a == ti]
    p = np.clip(pu[rows.u.values], 1e-4, 1 - 1e-4); y = rows.fail.values
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)), rows.proc.values


RES = {"curve": {}, "ceiling": {}, "routing": {}}
NS = [1, 2, 4, 8, 12, None]
SEEDS = range(10)
if __name__ == "__main__":
    # ---------------- A: probe-count curve ----------------
    full = {}
    for ti, a in enumerate(AG):
        if a == "lite": continue
        ob = obs_proc(ti)
        allp = unit_preds(ti, None, 0); full[a] = allp
        raw_all = proc_vec(allp["raw"])
        RES["curve"][a] = {}
        for n in NS:
            draws = [unit_preds(ti, n, s) for s in (SEEDS if n else [0])]
            row = {}
            for m in ("raw", "calib", "mf"):
                pv = [proc_vec(d[m]) for d in draws]
                row[m + "_rho"] = float(np.mean([pw_rho(v, ob) for v in pv]))
                row[m + "_ll"] = float(np.mean([case_ll(d[m], ti)[0].mean() for d in draws]))
            # paired bootstrap over procedures, averaged over draws: mf - raw (same n) and mf(n) - raw(all)
            mfv = [proc_vec(d["mf"]) for d in draws]; rawv = [proc_vec(d["raw"]) for d in draws]
            b1, b2 = [], []
            for idx in boot_idx(500):
                m_ = np.mean([pw_rho(v, ob, idx) for v in mfv])
                b1.append(m_ - np.mean([pw_rho(v, ob, idx) for v in rawv])); b2.append(m_ - pw_rho(raw_all, ob, idx))
            row["mf_minus_raw_rho"] = (row["mf_rho"] - row["raw_rho"], float(np.percentile(b1, 2.5)), float(np.percentile(b1, 97.5)))
            row["mf_minus_rawALL_rho"] = (row["mf_rho"] - pw_rho(raw_all, ob), float(np.percentile(b2, 2.5)), float(np.percentile(b2, 97.5)))
            RES["curve"][a][str(n or "all")] = row
            print(a, n or "all", {k: (round(v, 3) if isinstance(v, float) else tuple(round(x, 3) for x in v)) for k, v in row.items()}, flush=True)

    # ---------------- B: ceiling (split-half reliability of the real target) ----------------
    for ti, a in enumerate(AG):
        rows = REAL[REAL.a == ti]; rs = []
        for s in range(200):
            r = np.random.default_rng(s); half = np.zeros(len(rows), bool)   # split within each (procedure, class) unit
            for u in rows.u.unique():
                idx = np.where(rows.u.values == u)[0]; r.shuffle(idx); half[idx[:len(idx) // 2]] = True
            o1, o2 = obs_proc(ti, rows[half]), obs_proc(ti, rows[~half])
            rs.append(pw_rho(o1, o2))
        sh = float(np.nanmean(rs)); sb = 2 * sh / (1 + sh)
        RES["ceiling"][a] = {"split_half": sh, "spearman_brown": sb, "max_rho_vs_noisy_target": float(np.sqrt(sb))}
        print("ceiling", a, RES["ceiling"][a], flush=True)

    # ---------------- C: routing per procedure ----------------
    # loss per case: API cost + F * fail; human: cost h, correct with prob q
    API = {"flash": 0.008, "g38": 0.021, "sonnet": 0.05}      # USD per case, E8 / smart_eval ledgers (sonnet: rough)
    for F_, h, q in ((20, 5, 0.99), (20, 2, 0.95), (50, 5, 0.99), (10, 5, 0.99)):
        key = f"F={F_} human=${h} q={q}"
        real_fail = {a: obs_proc(AG.index(a)) for a in API}
        # case-mix weighted realized loss per procedure for each option (benchmark mix)
        loss_opt = {a: API[a] + F_ * real_fail[a] for a in API}
        loss_opt["human"] = np.full(len(PROCS), h + F_ * (1 - q))
        opts = list(loss_opt)
        L = np.vstack([loss_opt[o] for o in opts])                     # options x procs
        res = {}
        for o in opts: res[f"always {o}"] = float(np.nanmean(loss_opt[o]))
        res["oracle per procedure"] = float(np.nanmean(np.nanmin(L, 0)))
        for n in (2, 4, None):
            for m in ("raw", "mf"):
                choices = []
                for s in (SEEDS if n else [0]):
                    pred = {a: proc_vec(unit_preds(AG.index(a), n, s)[m]) for a in API}
                    pl = np.vstack([API[a] + F_ * pred[a] for a in API] + [np.full(len(PROCS), h + F_ * (1 - q))])
                    ch = pl.argmin(0)
                    choices.append(float(np.nanmean(L[ch, np.arange(len(PROCS))])))
                    if s == 0: share = {o: float(np.mean(ch == i)) for i, o in enumerate(opts)}
                res[f"{m} probes n={n or 'all'}"] = float(np.mean(choices)); res[f"{m} probes n={n or 'all'} shares"] = share
        RES["routing"][key] = res
        print("\nrouting", key)
        for k_, v in res.items(): print("  ", k_, v if isinstance(v, dict) else round(v, 3))
    json.dump(RES, open(os.path.join(HERE, "results_main.json"), "w"), indent=1)
