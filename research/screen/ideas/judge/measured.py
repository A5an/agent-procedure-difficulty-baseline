"""Measured pipeline: P1 k=1, pilot fixed to gpt-4.1-mini, outcome from grader vs LLM judge verdicts. Run: PYTHONHASHSEED=0 python measured.py"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, json, random, hashlib, pickle
sys.path.insert(0, ZR.SCREEN); sys.path.insert(0, ZR.SCREEN + "/ideas"); sys.path.append(ZR.SCREEN + "/ideas/pilot")
import numpy as np, pandas as pd
from scipy.special import expit
import pilot as P
from pilot import RB, E, GRID, lognorm, pm, tgt_of, load_matrices, prior_ridge
from cpulock import cpu_lock
import jan as A
PILOT = {"sopbench": "gpt-4.1-mini-mode_fc", "tau2": "gpt-4-1-mini_openai_2024-06-20"}
NB = 1000
def verdicts(think="low"):
    full = json.load(open("full.json")); c = {}
    for l in open(f"cache_{think}.jsonl"): d = json.loads(l); c[d["key"]] = d
    out = {"sopbench": {}, "tau2": {}}
    for r in full:
        k = hashlib.sha256((think + r["prompt"]).encode()).hexdigest()
        if k in c:
            p = A.parse(c[k]["text"])
            if p: out[r["bench"]][r["case_id"]] = (int(p[0]), r["outcome"])
    return out
V = verdicts()
RATES = {}
for b, d in V.items():
    g = np.array([x[1] for x in d.values()]); j = np.array([x[0] for x in d.values()])
    RATES[b] = dict(fpr=float(((j == 1) & (g == 0)).sum() / (g == 0).sum()), fnr=float(((j == 0) & (g == 1)).sum() / (g == 1).sum()), n=len(g), acc=float((g == j).mean()), agree_pass=float(g.mean()))
print(RATES, flush=True)
def lik_plain(theta, y):
    p = expit(theta[:, None] - GRID[None, :]); return np.log(np.clip(np.where(y[:, None] == 1, p, 1 - p), 1e-300, None))
def lik_err(theta, v, fpr, fnr):
    p = expit(theta[:, None] - GRID[None, :]); p1 = p * (1 - fnr) + (1 - p) * fpr
    return np.log(np.clip(np.where(v[:, None] == 1, p1, 1 - p1), 1e-300, None))
VARS = ["k0", "grader", "judge_plain", "judge_err"]
def main():
    rows = {}
    for bench, scheme in P.SCEN:
        tasks = pd.read_csv(RB.DATA / bench / "tasks.csv"); src = RB.sources(bench)
        agents_all, idx, S_all, N_all, Y1_all = load_matrices(bench, tasks)
        dom = tasks.set_index("case_id").domain; proc = tasks.set_index("case_id").procedure_id
        dcode = {d: i for i, d in enumerate(sorted(tasks.domain.unique()))}; scen = f"{bench}|{scheme}"
        vd = V[bench]; fpr, fnr = RATES[bench]["fpr"], RATES[bench]["fnr"]
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
                a = names.index(PILOT[bench]); T = len(te); ar = np.arange(T)
                has = avail[:, a] & np.array([t in vd for t in te])     # pilot ran and a judge verdict exists
                assert has.mean() > 0.95, (scen, has.mean())
                vj = np.array([vd[t][0] if t in vd else 0 for t in te]); gj = np.array([vd[t][1] if t in vd else 0 for t in te])
                assert (gj[has] == Y1[has, a]).all()
                btr = data.train_items.loc[tr, "b"].values.astype(float)
                mu, sd = prior_ridge(src, tr, te, btr, tasks.set_index("case_id").loc[tr, "procedure_id"].values)
                lp0 = lognorm(mu, sd); _, m0 = pm(lp0)
                used = np.zeros_like(avail); used[has, a] = True          # same pilot cells removed from the target in every variant
                tgt = tgt_of(S, N, used)
                cell = np.array([k_ * 100 + dcode[dom[t]] for t in te]); procs = np.array([proc[t] for t in te])
                th = np.full(T, theta[a]); yg = np.where(has, Y1[:, a], 0.0)
                upd = {"k0": np.zeros_like(lp0), "grader": np.where(has[:, None], lik_plain(th, yg), 0),
                       "judge_plain": np.where(has[:, None], lik_plain(th, vj), 0), "judge_err": np.where(has[:, None], lik_err(th, vj, fpr, fnr), 0)}
                for v in VARS:
                    _, mk = pm(lp0 + upd[v])
                    rows.setdefault((scen, v), []).append(pd.DataFrame({"seed": seed, "cell": cell, "proc": procs, "pred": mk, "tgt": tgt}))
        print(scen, "done", flush=True)
    return rows
def boot(rows):
    draws = {}; point = {}
    for scen in sorted({k[0] for k in rows}):
        bench = scen.split("|")[0]; d0 = {v: pd.concat(rows[(scen, v)]).query("seed==0").sort_values("proc", kind="stable").reset_index(drop=True) for v in VARS}
        ref = d0["k0"]; procs = sorted(ref.proc.unique()); starts = ref.groupby("proc").indices
        rng = np.random.default_rng([20261005, ["sopbench", "tau2"].index(bench)]); idxs = [np.concatenate([starts[p] for p in rng.choice(procs, len(procs))]) for _ in range(NB)]
        cell = ref.cell.values
        for v in VARS:
            draws[(scen, v)] = np.array([E.rank_arrays(cell[ix], d0[v].pred.values[ix], d0[v].tgt.values[ix]) for ix in idxs])
            fr = pd.concat(rows[(scen, v)])
            point[(scen, v)] = float(np.mean([E.rank_arrays(g.cell.values, g.pred.values, g.tgt.values) for _, g in fr.groupby("seed")]))
        print("boot", scen, flush=True)
    return draws, point
if __name__ == "__main__":
    with cpu_lock("judge"):
        rows = main(); draws, point = boot(rows)
    SC = [f"{b}|{s}" for b, s in P.SCEN]; ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    R = {"rates": RATES, "scen": {}, "pooled": {}}
    for v in VARS: R["scen"][v] = {s: point[(s, v)] for s in SC}
    def pool(sc, v): return np.mean([point[(s, v)] for s in sc]), np.mean([draws[(s, v)] for s in sc], axis=0)
    for name, sc in (("all4", SC), ("sopbench", SC[:2]), ("tau2", SC[2:])):
        for v in VARS:
            p, d = pool(sc, v); R["pooled"][f"{name}|{v}"] = dict(rho=float(p), ci=ci(d))
        for a, b in (("judge_plain", "grader"), ("judge_err", "grader"), ("judge_plain", "k0"), ("judge_err", "k0"), ("grader", "k0"), ("judge_err", "judge_plain")):
            pa, da = pool(sc, a); pb, db = pool(sc, b); R["pooled"][f"{name}|{a}-{b}"] = dict(diff=float(pa - pb), ci=ci(da - db))
    json.dump(R, open("measured.json", "w"), indent=1)
    for k, v in R["pooled"].items(): print(k, {a: (np.round(b, 3).tolist()) for a, b in v.items()})
    for v in VARS: print(v, {s: round(x, 3) for s, x in R["scen"][v].items()})
