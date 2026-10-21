"""Offline cascade simulation: cheap agent A, judge with error e, escalate to strong B. PYTHONHASHSEED=0 python cascade.py"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, json, random, collections
sys.path.insert(0, ZR.SCREEN); sys.path.insert(0, ZR.SCREEN + "/ideas")
import numpy as np, pandas as pd
from scipy.special import expit
import judge as J
import pilot as P
from pilot import RB, GRID, lognorm, pm, load_matrices, prior_ridge
from cpulock import cpu_lock
ES = [0.0, 0.1, 0.2]; TAUS = [0.5, 0.7]; REPS = 20; PRICE_A, PRICE_B = 1.0, 10.0
SCH = [("sopbench", "new_procedures"), ("sopbench", "new_domain"), ("tau2", "new_procedures"), ("tau2", "new_domain")]

def policies(bench, scheme):
    tasks = pd.read_csv(RB.DATA / bench / "tasks.csv"); src = RB.sources(bench)
    agents_all, idx, S_all, N_all, Y1_all = load_matrices(bench, tasks)
    proc = tasks.set_index("case_id").procedure_id
    recs = []
    folds = RB.splits(tasks, scheme, 0)
    for k_, (tr, te) in enumerate(folds):
        fs = k_; random.seed(fs); np.random.seed(fs)
        cwd = os.getcwd(); os.chdir(RB.AP_REPO)
        try:
            import torch, pyro; torch.manual_seed(fs); pyro.set_rng_seed(fs)
            data = RB.load_dataset_for_fold(abilities_path=RB.DATA / bench / "irt/1d_1pl/abilities.csv", items_path=RB.DATA / bench / "irt/1d_1pl/items.csv",
                responses_path=RB.DATA / bench / "responses.jsonl", train_tasks=tr, test_tasks=te, fold_idx=k_, k_folds=len(folds), split_seed=0,
                irt_cache_dir=RB.RESULTS / bench / "irt_splits" / scheme)
        finally: os.chdir(cwd)
        ab = data.train_abilities; names = list(ab.index); cols = [agents_all.index(a) for a in names]
        theta = ab["ability"].values.astype(float); ti = np.array([idx[t] for t in te])
        Y1 = Y1_all[ti][:, cols]; avail = N_all[ti][:, cols] > 0; T = len(te); ar = np.arange(T)
        small = np.array([bool(J.SMALL.search(n)) for n in names])
        keyA = np.where(avail & small[None, :], theta[None, :], np.inf)
        keyA = np.where(np.isinf(keyA).all(1, keepdims=True), np.where(avail, theta[None, :], np.inf), keyA)
        A = keyA.argmin(1)
        B = np.where(avail, -theta[None, :], np.inf).argmin(1)
        assert (A != B).all() and avail[ar, A].all() and avail[ar, B].all()
        top3 = np.argsort(np.where(avail, -theta[None, :], np.inf), axis=1)[:, :3]
        yA, yB = Y1[ar, A], Y1[ar, B]; assert not np.isnan(yA).any() and not np.isnan(yB).any()
        btr = data.train_items.loc[tr, "b"].values.astype(float)
        mu, sd = prior_ridge(src, tr, te, btr, tasks.set_index("case_id").loc[tr, "procedure_id"].values)
        lp0 = lognorm(mu, sd); sig = expit(theta[:, None] - GRID[None, :])  # [A,G]
        sigA = sig[A]  # [T,G]
        procs = np.array([proc[t] for t in te])
        base = dict(proc=procs, yA=yA, yB=yB)
        recs.append(("always_A", 0.0, None, dict(base, succ=yA, esc=np.zeros(T), bfail_esc=np.zeros(T), cost=np.full(T, PRICE_A))))
        recs.append(("always_B", 0.0, None, dict(base, succ=yB, esc=np.ones(T), bfail_esc=1 - yB, cost=np.full(T, PRICE_B))))
        e_ = (yA == 0).astype(float)
        recs.append(("oracle", 0.0, None, dict(base, succ=np.where(yA == 1, yA, yB), esc=e_, bfail_esc=e_ * (1 - yB), cost=PRICE_A + PRICE_B * e_)))
        for e in ES:
            acc = collections.defaultdict(list)
            for rep in range(REPS if e > 0 else 1):
                rng = np.random.default_rng([99, k_, rep]); flip = rng.random(T) < e
                v = np.where(flip, 1 - yA, yA)
                esc1 = (v == 0).astype(float)
                acc["C1"].append((np.where(esc1 == 1, yB, yA), esc1))
                p1 = expit(0)  # placeholder
                lp = lp0 + J.lik_e(theta[A], v, e)
                w, _ = pm(lp); pA = (w * sigA).sum(1)
                psucc = w @ sig.T  # [T,Agents]
                best = np.take_along_axis(top3, np.argmax(np.take_along_axis(psucc, top3, 1), 1)[:, None], 1)[:, 0]
                yBest = Y1[ar, best]; assert not np.isnan(yBest).any()
                for tau in TAUS:
                    esc2 = (pA <= tau).astype(float)
                    acc[f"C2_tau{tau}"].append((np.where(esc2 == 1, yBest, yA), esc2, yBest))
            for pol, lst in acc.items():
                succ = np.mean([x[0] for x in lst], 0); esc = np.mean([x[1] for x in lst], 0)
                yE = yB if pol == "C1" else np.mean([x[2] for x in lst], 0)
                recs.append((pol, e, None, dict(base, succ=succ, esc=esc, bfail_esc=esc * (1 - yE), cost=PRICE_A + PRICE_B * esc)))
    return recs

def summarize(frames, rng, nb=1000):
    df = pd.concat(frames); procs = sorted(df.proc.unique()); g = df.groupby("proc")[["succ", "esc", "bfail_esc", "cost"]].sum(); g["n"] = df.groupby("proc").size()
    def stats(g_):
        n = g_.n.sum(); return dict(success=g_.succ.sum() / n, escalated=g_.esc.sum() / n, cost=g_.cost.sum() / n,
                                    b_fails_among_escalated=(g_.bfail_esc.sum() / g_.esc.sum()) if g_.esc.sum() > 0 else float("nan"))
    pt = stats(g); ar = g.reset_index(drop=True); pi = g.index
    picks = [rng.integers(0, len(ar), len(ar)) for _ in range(nb)]
    return pt, ar, picks, stats

def main():
    allrecs = {}
    for b, s in SCH: allrecs[(b, s)] = policies(b, s); print(b, s, flush=True)
    out = {}
    for bench in ("sopbench", "tau2"):
        pols = collections.defaultdict(list)
        for (b, s), recs in allrecs.items():
            if b != bench: continue
            for pol, e, _, d in recs: pols[(pol, e)].append(pd.DataFrame(d))
        # one set of picks per benchmark (same procedures for all policies); procedure list identical (union over both schemes, sorted)
        rng = np.random.default_rng([5150, 0 if bench == "sopbench" else 1])
        any_df = pd.concat(next(iter(pols.values()))); procs = sorted(any_df.proc.unique()); P_ = len(procs)
        picks = [rng.integers(0, P_, P_) for _ in range(1000)]
        res = {}
        for key, frames in pols.items():
            df = pd.concat(frames); g = df.groupby("proc")[["succ", "esc", "bfail_esc", "cost"]].sum().reindex(procs).fillna(0); g["n"] = df.groupby("proc").size().reindex(procs).fillna(0)
            G = g.values  # cols succ, esc, bfail, cost, n
            tot = G.sum(0); pt = dict(success=tot[0] / tot[4], escalated=tot[1] / tot[4], cost=tot[3] / tot[4], b_fails_among_escalated=(tot[2] / tot[1]) if tot[1] > 0 else None)
            bs = np.array([G[p].sum(0) for p in picks]); sr = bs[:, 0] / bs[:, 4]
            pt["success_ci"] = [float(np.percentile(sr, 2.5)), float(np.percentile(sr, 97.5))]
            res["|".join(map(str, key))] = dict(pt, _sr=sr)
        # paired differences C2 vs C1 and C1 vs always_A
        def diff(a, b): d = res[a]["_sr"] - res[b]["_sr"]; return [float(res[a]["success"] - res[b]["success"]), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        for e in ES:
            for tau in TAUS:
                res[f"C2_tau{tau}|{e}"]["success_minus_C1"] = diff(f"C2_tau{tau}|{e}", f"C1|{e}")
            res[f"C1|{e}"]["success_minus_alwaysA"] = diff(f"C1|{e}", "always_A|0.0")
        for v in res.values(): v.pop("_sr", None)
        out[bench] = {k: {kk: (None if isinstance(vv, float) and np.isnan(vv) else vv) for kk, vv in v.items()} for k, v in res.items()}
    return out

if __name__ == "__main__":
    with cpu_lock("pilot"):
        out = main()
    r = json.load(open(os.path.join(P.OUT, "results.json"))); r["cascade"] = out
    json.dump(r, open(os.path.join(P.OUT, "results.json"), "w"), indent=1)
    for b, o in out.items():
        print(b)
        for k, v in sorted(o.items()):
            print(f"  {k:22s} succ {v['success']:.3f} [{v['success_ci'][0]:.3f},{v['success_ci'][1]:.3f}] esc {v['escalated']:.3f} cost {v['cost']:.2f} Bfail|esc {v['b_fails_among_escalated']}" + (f" vsC1 {v['success_minus_C1']}" if 'success_minus_C1' in v else "") + (f" vsA {v['success_minus_alwaysA']}" if 'success_minus_alwaysA' in v else ""))
