"""Exp 3 analysis (PREREG.md in this folder): seeded probes vs documentation-only probes, same real outcomes."""
import json, sys, collections
import numpy as np, torch
from pathlib import Path
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent; T = HERE.parent
sys.path.insert(0, str(T.parent / "mf"))
from model import MF

rng = np.random.default_rng(20261103)
sel = json.load(open(T / "selection.json")); F = sel["test_families"]; CLS = np.array([sel["classes"][f] for f in F])
AGS = ["g38", "flash"]; MIDX = {"g38": 2, "flash": 1}


def fam(t):
    return next(f for f in F if f"_{f}_" in t)


def load(path):
    d = collections.defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        if r.get("reward") is None: continue
        d[fam(r["task_id"])].append(r)
    return d


REAL = {a: load(T / "runs" / f"real_{a}.jsonl") for a in AGS}
DOC = {a: load(T / "runs" / f"probes_{a}.v2.jsonl") for a in AGS}
SEED = {a: load(T / "runs" / f"seeded_{a}.jsonl") for a in AGS}
fail = lambda rs: [1 - int(r["reward"] >= 1) for r in rs]
miss = lambda rs: [1 - float(np.mean(r["actions_ok"])) if r.get("actions_ok") else 0.0 for r in rs]
real = {a: np.array([np.mean(fail(REAL[a][f])) for f in F]) for a in AGS}

P = torch.load(T / "mf_params.pt"); m = MF(4)
for n in m.p: m.p[n] = P[n].clone()


def mf(a, src):
    nS = np.zeros((4, len(F))); fS = np.zeros_like(nS)
    for j, f in enumerate(F):
        nS[MIDX[a], j] = len(src[a][f]); fS[MIDX[a], j] = sum(fail(src[a][f]))
    z = np.zeros_like(nS)
    return m.predict(MIDX[a], CLS, np.zeros(len(F)), z, z, nS, fS)


def rho(x, y):
    return float(spearmanr(x, y).correlation)


def boot(x, y, x2=None, B=4000):
    out = []
    for _ in range(B):
        i = rng.integers(0, len(F), len(F))
        if np.ptp(x[i]) == 0 or np.ptp(y[i]) == 0 or (x2 is not None and np.ptp(x2[i]) == 0): continue
        out.append(rho(x[i], y[i]) - (rho(x2[i], y[i]) if x2 is not None else 0))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


res = {}
for a in AGS:
    s_rate = np.array([np.mean(fail(SEED[a][f])) if SEED[a][f] else np.nan for f in F])
    d_rate = np.array([np.mean(fail(DOC[a][f])) for f in F])
    s_miss = np.array([np.mean(miss(SEED[a][f])) if SEED[a][f] else np.nan for f in F])
    y = real[a]
    res[a] = {"n_seeded_eps": int(sum(len(v) for v in SEED[a].values())), "seeded_fail_mean": float(np.nanmean(s_rate)),
              "real_fail_mean": float(y.mean()), "doc_fail_mean": float(d_rate.mean()),
              "rho_seeded (primary)": (rho(s_rate, y), *boot(s_rate, y)),
              "rho_doc": (rho(d_rate, y), *boot(d_rate, y)),
              "seeded_minus_doc": (rho(s_rate, y) - rho(d_rate, y), *boot(s_rate, y, d_rate)),
              "rho_seeded_graded": (rho(s_miss, y), *boot(s_miss, y)),
              "rho_mf_seeded": (rho(mf(a, SEED), y), *boot(mf(a, SEED), y)),
              "level: mean |seeded rate - real rate|": float(np.nanmean(np.abs(s_rate - y)))}
    print(a, json.dumps(res[a]))
res["per_family"] = {f: {a: {"real": int(sum(fail(REAL[a][f]))), "doc": int(sum(fail(DOC[a][f]))), "seeded": int(sum(fail(SEED[a][f])))}
                         for a in AGS} for f in F}
L = [json.loads(l) for l in open(T / "ledger.jsonl")]
res["spend_exp3"] = round(sum(r.get("usd", 0) for r in L) - 19.6916, 3)
json.dump(res, open(HERE / "results.json", "w"), indent=1)
for f, d in res["per_family"].items():
    print(f"{f:16s} " + " | ".join(f"{a} real {d[a]['real']}/6 doc {d[a]['doc']}/6 seeded {d[a]['seeded']}/6" for a in AGS))
print("spend exp3", res["spend_exp3"])
