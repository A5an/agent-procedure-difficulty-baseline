"""Pilot runs instead of none: Bayesian update of case difficulty from k pilot agents' outcomes.
Run: PYTHONHASHSEED=0 python pilot.py [--quick]"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, json, random, time
sys.path.insert(0, ZR.SCREEN)
sys.path.insert(0, ZR.SCREEN + "/ideas")
import numpy as np, pandas as pd
import harness as H
from cpulock import cpu_lock
from scipy.special import expit, log_expit
from sklearn.model_selection import GroupKFold
RB, E = H.RB, H.E
QUICK = "--quick" in sys.argv
OUT = os.path.dirname(os.path.abspath(__file__))
KS = [1, 2, 3, 5]
RULES = ["weakest", "median", "strongest", "adaptive"]
GRID = np.linspace(-12, 12, 481)
NBOOT = 100 if QUICK else 500
SCEN = [("sopbench", "new_procedures"), ("sopbench", "new_domain"), ("tau2", "new_procedures"), ("tau2", "new_domain")]
AP = RB.AP_REPO

def load_matrices(bench, tasks):
    """S, N (all attempts), Y1 (first attempt) as dict agent -> arrays over tasks order."""
    resp = {}
    for line in open(RB.DATA / bench / "responses.jsonl"):
        r = json.loads(line); resp[r["subject_id"]] = r["responses"]
    att = json.load(open(os.path.join(OUT, "tau2_attempts.json"))) if bench == "tau2" else None
    ids = tasks.case_id.tolist(); idx = {c: i for i, c in enumerate(ids)}
    agents = sorted(resp)
    S = np.full((len(ids), len(agents)), np.nan); N = np.zeros((len(ids), len(agents))); Y1 = np.full_like(S, np.nan)
    for j, a in enumerate(agents):
        for c, v in resp[a].items():
            i = idx[c]
            if isinstance(v, dict):
                S[i, j], N[i, j] = v["successes"], v["trials"]
                tr = att[a][c]; Y1[i, j] = float(tr[str(min(map(int, tr)))] if isinstance(next(iter(tr)), str) else tr[min(tr)])
            else:
                S[i, j], N[i, j], Y1[i, j] = v, 1, v
    S = np.nan_to_num(S, nan=0.0)
    assert not np.isnan(S).any() and np.isnan(Y1).sum() == (N == 0).sum()
    return agents, idx, S, N, Y1

def prior_ridge(src, tr, te, btr, groups):
    mk = lambda: RB.FeatureBasedPredictor(src["adele"], alphas=RB.ALPHAS)
    m = mk(); m.fit(tr, btr); mu = np.array([m.predict(te)[t] for t in te])
    res = np.zeros(len(tr)); tr_a = np.array(tr)
    for a, b in GroupKFold(5).split(tr_a, groups=groups):
        mm = mk(); mm.fit(list(tr_a[a]), btr[a]); p = mm.predict(list(tr_a[b]))
        res[b] = btr[b] - np.array([p[t] for t in tr_a[b]])
    return mu, float(res.std(ddof=1))

def lognorm(mu, sd):
    return -0.5 * ((GRID[None, :] - mu[:, None]) / sd) ** 2

def pm(lp):
    w = np.exp(lp - lp.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    return w, w @ GRID

def lik(theta_sel, y):  # [T] theta, [T] outcome -> [T,G]
    d = theta_sel[:, None] - GRID[None, :]
    return np.where(y[:, None] == 1, log_expit(d), log_expit(-d))

def run_fold(lp0, theta, avail, Y1, rule, ks):
    """Return {k: (posterior w, pilot mask [T,A])} for cumulative pilots."""
    T, A = avail.shape; ar = np.arange(T)
    sig = expit(theta[:, None] - GRID[None, :])  # [A,G]
    lp = lp0.copy(); used = np.zeros((T, A), bool); out = {}
    if rule != "adaptive":
        if rule == "weakest": key = np.where(avail, theta[None, :], np.inf)
        elif rule == "strongest": key = np.where(avail, -theta[None, :], np.inf)
        else:
            med = np.array([np.median(theta[avail[t]]) for t in range(T)])
            key = np.where(avail, np.abs(theta[None, :] - med[:, None]), np.inf)
        order = np.argsort(key, axis=1, kind="stable")
    for j in range(1, max(ks) + 1):
        if rule == "adaptive":
            w, _ = pm(lp); p = w @ sig.T
            key = np.where(avail & ~used, np.abs(p - 0.5), np.inf); sel = key.argmin(1)
        else:
            sel = order[:, j - 1]
        assert avail[ar, sel].all() and not np.isnan(Y1[ar, sel]).any()
        used[ar, sel] = True
        lp = lp + lik(theta[sel], Y1[ar, sel])
        if j in ks:
            out[j] = (lp.copy(), used.copy())
    return out

def tgt_of(S, N, mask):
    assert not np.isnan(S).any()
    runs = (N * (~mask)).sum(1); fails = ((N - S) * (~mask)).sum(1)
    return (fails + 0.5) / (runs + 1)

def main():
    rows = {}   # (scen, prior, rule, k) -> list of frames
    ll = {}     # (scen, prior, rule, k) -> list of (seed, ll_prior_sum, ll_post_sum, n)
    p2rows = {}
    for bench, scheme in SCEN:
        tasks = pd.read_csv(RB.DATA / bench / "tasks.csv")
        src = RB.sources(bench)
        agents_all, idx, S_all, N_all, Y1_all = load_matrices(bench, tasks)
        full_b = pd.read_csv(RB.DATA / bench / "irt/1d_1pl/items.csv", index_col=0).b
        dom = tasks.set_index("case_id").domain; proc = tasks.set_index("case_id").procedure_id
        seeds = [0] if scheme == "new_domain" else ([0, 1] if QUICK else list(range(5)))
        scen = f"{bench}|{scheme}"
        for seed in seeds:
            folds = RB.splits(tasks, scheme, seed)
            for k_, (tr, te) in enumerate(folds):
                if QUICK and k_ > 1: break
                fs = 1000 * seed + k_
                random.seed(fs); np.random.seed(fs)
                cwd = os.getcwd(); os.chdir(AP)
                try:
                    import torch, pyro
                    torch.manual_seed(fs); pyro.set_rng_seed(fs)
                    data = RB.load_dataset_for_fold(
                        abilities_path=RB.DATA / bench / "irt/1d_1pl/abilities.csv", items_path=RB.DATA / bench / "irt/1d_1pl/items.csv",
                        responses_path=RB.DATA / bench / "responses.jsonl", train_tasks=tr, test_tasks=te, fold_idx=k_,
                        k_folds=len(folds), split_seed=seed, irt_cache_dir=RB.RESULTS / bench / "irt_splits" / scheme)
                finally:
                    os.chdir(cwd)
                ab = data.train_abilities
                cols = [agents_all.index(a) for a in ab.index]
                theta = ab["ability"].values.astype(float)
                ti = np.array([idx[t] for t in te])
                S, N, Y1 = S_all[ti][:, cols], N_all[ti][:, cols], Y1_all[ti][:, cols]
                avail = N > 0
                btr = data.train_items.loc[tr, "b"].values.astype(float)
                groups = tasks.set_index("case_id").loc[tr, "procedure_id"].values
                mu_doc, sd_doc = prior_ridge(src, tr, te, btr, groups)
                priors = {"doc": (mu_doc, sd_doc), "free": (np.full(len(te), btr.mean()), float(btr.std(ddof=1)))}
                cell = np.array([hash(dom[t]) % 10**6 for t in te]) * 0  # replaced below
                dcode = {d: i for i, d in enumerate(sorted(tasks.domain.unique()))}
                cell = np.array([k_ * 100 + dcode[dom[t]] for t in te])
                bf = full_b.loc[te].values; procs = np.array([proc[t] for t in te])
                sigA = expit(theta[:, None] - GRID[None, :])
                for pname, (mu, sd) in priors.items():
                    lp0 = lognorm(mu, sd)
                    w0, m0 = pm(lp0)
                    p_prior = w0 @ sigA.T
                    # k = 0
                    t0 = tgt_of(S, N, np.zeros_like(avail))
                    def add(key, pred, tgt, refdoc, refown, w_post, mask, p_post_):
                        assert not (np.isnan(pred).any() or np.isnan(tgt).any() or np.isnan(p_post_).any())
                        df = pd.DataFrame({"seed": seed, "cell": cell, "case_id": te, "proc": procs, "pred": pred, "tgt": tgt,
                                           "ref_doc": refdoc, "ref_own": refown, "bf": bf})
                        rows.setdefault((scen,) + key, []).append(df)
                        nonp = avail & ~mask
                        def L(p):
                            p = np.clip(p, 1e-6, 1 - 1e-6)
                            return -(nonp * (S * np.log(p) + (N - S) * np.log(1 - p))).sum()
                        ll.setdefault((scen,) + key, []).append((seed, L(p_prior), L(p_post_), (nonp * N).sum()))
                    add((pname, "none", 0), m0, t0, mu_doc, m0, w0, np.zeros_like(avail), p_prior)
                    for rule in RULES:
                        res = run_fold(lp0, theta, avail, Y1, rule, KS)
                        for k, (lpk, used) in res.items():
                            wk, mk = pm(lpk)
                            add((pname, rule, k), mk, tgt_of(S, N, used), mu_doc, m0, wk, used, wk @ sigA.T)
                            if bench == "sopbench" and rule == "adaptive" and k == 1:
                                p2rows.setdefault((scheme, pname), []).append((seed, k_, te, procs, cell, bf, mu, m0, mk, t0, S, N))
            print(scen, "seed", seed, "done", flush=True)
    return rows, ll, p2rows

def rho_seed(df, col, tgt="tgt"):
    out = []
    for s, g in df.groupby("seed"):
        out.append(E.rank_arrays(g.cell.values, g[col].values, g[tgt].values))
    return float(np.nanmean(out))

def boot(rows, ll, scen_keys):
    """Per scenario bootstrap over procedures (seed 0), same resample for every condition as the evaluator."""
    res = {}
    bi = {"sopbench": 0, "tau2": 1}
    for scen in scen_keys:
        bench = scen.split("|")[0]
        keys = [k for k in rows if k[0] == scen]
        d0 = {}
        for k in keys:
            df = pd.concat(rows[k]); df = df[df.seed == 0].sort_values("proc", kind="stable").reset_index(drop=True)
            d0[k] = df
        ref = d0[keys[0]]
        procs = sorted(ref.proc.unique())
        starts = ref.groupby("proc").indices
        rng = np.random.default_rng([20260928, H.BENCHES.index(bench)])
        picks = [rng.choice(procs, len(procs)) for _ in range(NBOOT)]
        idxs = [np.concatenate([starts[p] for p in pk]) for pk in picks]
        cell = ref.cell.values
        draws = {}
        for k in keys:
            df = d0[k]
            for col in ("pred", "ref_doc", "ref_own"):
                if col == "ref_doc" and k[1] == "doc" or col == "ref_own" and k[1] == "doc" and k[2] == 0:
                    continue
                draws[(k, col)] = np.array([E.rank_arrays(cell[ix], df[col].values[ix], df.tgt.values[ix]) for ix in idxs])
        res[scen] = draws
        print("boot", scen, flush=True)
    return res

if __name__ == "__main__":
    t0 = time.time()
    with cpu_lock("pilot"):
        rows, ll, p2rows = main()
        scen_keys = sorted({k[0] for k in rows})
        draws = boot(rows, ll, scen_keys)
    import pickle
    pickle.dump((rows, ll, p2rows, draws), open(os.path.join(OUT, "raw_quick.pkl" if QUICK else "raw.pkl"), "wb"))
    print("done", time.time() - t0)
