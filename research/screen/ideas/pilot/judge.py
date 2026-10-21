"""Imperfect judge, cheapest pilot, adaptive pilot frequencies. Reuses pilot.py helpers. Run: PYTHONHASHSEED=0 python judge.py"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, re, json, random, time, pickle, collections
sys.path.insert(0, ZR.SCREEN); sys.path.insert(0, ZR.SCREEN + "/ideas")
import numpy as np, pandas as pd
from scipy.special import expit
import pilot as P
from pilot import RB, E, GRID, lognorm, pm, tgt_of, load_matrices, prior_ridge
from cpulock import cpu_lock
EVALS = [0.0, 0.05, 0.1, 0.2, 0.3]; REPS = 5; NB = 200
SMALL = re.compile(r"(?<![\d.])e?[2-8]b(?![a-z0-9])|(?<![a-z])mini|flash|lite|haiku|small")

def lik_e(theta_sel, y, e):
    p = expit(theta_sel[:, None] - GRID[None, :]); p1 = (1 - e) * p + e * (1 - p)
    return np.log(np.clip(np.where(y[:, None] == 1, p1, 1 - p1), 1e-300, None))

def run(lp0, theta, avail, Yo, rule, ks, e_lik, small):
    T, A = avail.shape; ar = np.arange(T)
    sig = expit(theta[:, None] - GRID[None, :]); lp = lp0.copy(); used = np.zeros((T, A), bool); out = {}; sel1 = None
    if rule == "cheapest":
        key = np.where(avail & small[None, :], theta[None, :], np.inf)
        key = np.where(np.isinf(key).all(1, keepdims=True), np.where(avail, theta[None, :], np.inf), key)  # fallback: weakest
        order = np.argsort(key, axis=1, kind="stable")
    for j in range(1, max(ks) + 1):
        if rule == "adaptive":
            w, _ = pm(lp); p = w @ sig.T
            sel = np.where(avail & ~used, np.abs(p - 0.5), np.inf).argmin(1)
        else:
            sel = order[:, j - 1]
        assert avail[ar, sel].all() and not np.isnan(Yo[ar, sel]).any()
        used[ar, sel] = True
        if j == 1: sel1 = sel.copy()
        lp = lp + lik_e(theta[sel], Yo[ar, sel], e_lik)
        if j in ks: out[j] = (lp.copy(), used.copy())
    return out, sel1

def main():
    rows = {}; freq = {}; smallnames = {}
    for bench, scheme in P.SCEN:
        tasks = pd.read_csv(RB.DATA / bench / "tasks.csv"); src = RB.sources(bench)
        agents_all, idx, S_all, N_all, Y1_all = load_matrices(bench, tasks)
        dom = tasks.set_index("case_id").domain; proc = tasks.set_index("case_id").procedure_id
        dcode = {d: i for i, d in enumerate(sorted(tasks.domain.unique()))}
        scen = f"{bench}|{scheme}"
        for seed in ([0] if scheme == "new_domain" else range(5)):
            folds = RB.splits(tasks, scheme, seed)
            for k_, (tr, te) in enumerate(folds):
                fs = 1000 * seed + k_; random.seed(fs); np.random.seed(fs)
                cwd = os.getcwd(); os.chdir(RB.AP_REPO)
                try:
                    import torch, pyro; torch.manual_seed(fs); pyro.set_rng_seed(fs)
                    data = RB.load_dataset_for_fold(abilities_path=RB.DATA / bench / "irt/1d_1pl/abilities.csv", items_path=RB.DATA / bench / "irt/1d_1pl/items.csv",
                        responses_path=RB.DATA / bench / "responses.jsonl", train_tasks=tr, test_tasks=te, fold_idx=k_, k_folds=len(folds), split_seed=seed,
                        irt_cache_dir=RB.RESULTS / bench / "irt_splits" / scheme)
                finally: os.chdir(cwd)
                ab = data.train_abilities; names = list(ab.index); cols = [agents_all.index(a) for a in names]
                theta = ab["ability"].values.astype(float); ti = np.array([idx[t] for t in te])
                S, N, Y1 = S_all[ti][:, cols], N_all[ti][:, cols], Y1_all[ti][:, cols]; avail = N > 0
                small = np.array([bool(SMALL.search(n)) for n in names]); smallnames[bench] = [n for n, s in zip(names, small) if s]
                btr = data.train_items.loc[tr, "b"].values.astype(float)
                mu_doc, sd_doc = prior_ridge(src, tr, te, btr, tasks.set_index("case_id").loc[tr, "procedure_id"].values)
                priors = {"doc": (mu_doc, sd_doc), "free": (np.full(len(te), btr.mean()), float(btr.std(ddof=1)))}
                cell = np.array([k_ * 100 + dcode[dom[t]] for t in te]); procs = np.array([proc[t] for t in te])
                for pname, (mu, sd) in priors.items():
                    lp0 = lognorm(mu, sd); _, m0 = pm(lp0)
                    def add(key, rep, mk, used):
                        rows.setdefault((scen, pname) + key, {}).setdefault(rep, []).append(pd.DataFrame(
                            {"seed": seed, "cell": cell, "proc": procs, "pred": mk, "ref": m0, "tgt": tgt_of(S, N, used)}))
                    jobs = [("adaptive", [1, 2], e, naive) for e in EVALS for naive in ([False, True] if e == 0.2 else [False])] + [("cheapest", [1], 0.0, False)]
                    for rule, ks, e, naive in jobs:
                        for rep in (range(REPS) if e > 0 else [0]):
                            rng = np.random.default_rng([4242, seed, k_, rep])
                            flip = rng.random(Y1.shape) < e
                            Yo = np.where(flip, 1 - Y1, Y1)  # NaN stays NaN
                            out, sel1 = run(lp0, theta, avail, Yo, rule, ks, 0.0 if naive else e, small)
                            for k, (lpk, used) in out.items():
                                _, mk = pm(lpk); add((rule, k, e, naive), rep, mk, used)
                            if rule == "adaptive" and k_ is not None and e == 0.0 and pname == "doc":
                                for a in sel1: freq.setdefault((bench, scheme), collections.Counter())[names[a]] += 1
        print(scen, "done", flush=True)
    return rows, freq, smallnames

def boot(rows):
    res = {}
    for scen in sorted({k[0] for k in rows}):
        bench = scen.split("|")[0]
        keys = [k for k in rows if k[0] == scen]
        d0 = {k: {r: pd.concat(v).query("seed==0").sort_values("proc", kind="stable").reset_index(drop=True) for r, v in rows[k].items()} for k in keys}
        ref = next(iter(d0[keys[0]].values())); procs = sorted(ref.proc.unique()); starts = ref.groupby("proc").indices
        rng = np.random.default_rng([20260928, H_B.index(bench)]); picks = [rng.choice(procs, len(procs)) for _ in range(NB)]
        idxs = [np.concatenate([starts[p] for p in pk]) for pk in picks]; cell = ref.cell.values
        for k in keys:
            res[(k, "pred")] = np.mean([[E.rank_arrays(cell[ix], df.pred.values[ix], df.tgt.values[ix]) for ix in idxs] for df in d0[k].values()], axis=0)
            res[(k, "ref")] = np.mean([[E.rank_arrays(cell[ix], df.ref.values[ix], df.tgt.values[ix]) for ix in idxs] for df in d0[k].values()], axis=0)
        print("boot", scen, flush=True)
    return res
H_B = ["sopbench", "tau2", "tauk_banking"]

if __name__ == "__main__":
    t0 = time.time()
    with cpu_lock("pilot"):
        rows, freq, smallnames = main()
        draws = boot(rows)
    pickle.dump((rows, freq, smallnames, draws), open(os.path.join(P.OUT, "judge_raw.pkl"), "wb"))
    print("done", time.time() - t0)
