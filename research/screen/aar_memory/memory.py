"""Memory for a new procedure on SOPBench (see PREREG.md). Run: python3 memory.py > memory.out"""
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data" / "sopbench"
rng = np.random.default_rng(0)

rows = list(csv.DictReader(open(D / "tasks.csv")))
ids = [r["case_id"] for r in rows]
z = np.load(D / "features" / "bge_base_en_v1.5.npz", allow_pickle=True)
pos = {t: i for i, t in enumerate(z["task_ids"])}
X = z["X"][[pos[i] for i in ids]].astype(float)
X /= np.linalg.norm(X, axis=1, keepdims=True)
procs = sorted({r["procedure_id"] for r in rows})
pidx = np.array([procs.index(r["procedure_id"]) for r in rows])
pdom = np.array([p.split("/")[0] for p in procs])
R = [json.loads(l) for l in open(D / "responses.jsonl")]
agents = [r["subject_id"] for r in R]
Y = np.array([[r["responses"].get(i, np.nan) for i in ids] for r in R], float)  # agents x cases
Y = np.where(np.isnan(Y), np.nanmean(Y, axis=1, keepdims=True), Y)              # 5 missing cells
A, N, Q = Y.shape[0], len(ids), len(procs)
cases = [np.where(pidx == q)[0] for q in range(Q)]
ncase = np.array([len(c) for c in cases])
rate = np.stack([Y[:, c].mean(1) for c in cases], 1)                             # agents x procs
E_ = np.stack([X[c].mean(0) for c in cases]); E_ /= np.linalg.norm(E_, axis=1, keepdims=True)
SP, SC = E_ @ E_.T, X @ X.T
targets = [q for q in range(Q) if ncase[q] >= 8]
PK, PT = [3, 5, 10, 999], [0.02, 0.05, 0.1, np.inf]
CK, CT = [5, 10, 20], [0.02, 0.05]
NS, MS, TAUS, REPS = [0, 1, 2, 4], [2, 4, 8], [0.5, 0.7], 30


def knn(sims, vals, k, T):
    """sims: (m,), vals: (A, m) -> (A,) weighted mean over top-k."""
    o = np.argsort(-sims)[:k]
    w = np.ones(len(o)) if np.isinf(T) else np.exp((sims[o] - sims[o[0]]) / T)
    return vals[:, o] @ w / w.sum()


def brier_rate(p, r):  # mean case Brier when rate is r and forecast p
    return r * (1 - p) ** 2 + (1 - r) * p ** 2


def proc_memory(q, mem):
    """Pick (k,T) by leave-one-out inside mem, return forecast for q, (A,)."""
    mem = np.array(mem)
    if len(mem) < 2:
        return None
    best, cfg = 1e9, None
    for k in PK:
        for T in PT:
            b = np.mean([brier_rate(knn(SP[m, mem[mem != m]], rate[:, mem[mem != m]], k, T), rate[:, m]).mean()
                         for m in mem])
            if b < best:
                best, cfg = b, (k, T)
    return knn(SP[q, mem], rate[:, mem], *cfg)


def case_cfg(mem):
    """Pick (k,T) for CaseMemory: every memory case is predicted from memory cases of other procedures."""
    idx = np.concatenate([cases[m] for m in mem])
    S = SC[np.ix_(idx, idx)].copy()
    S[pidx[idx][:, None] == pidx[idx][None, :]] = -9
    best, cfg = 1e9, None
    o_all = np.argsort(-S, axis=1)[:, :max(CK)]
    for k in CK:
        o = o_all[:, :k]
        s = np.take_along_axis(S, o, 1)
        for T in CT:
            w = np.exp((s - s[:, :1]) / T); w /= w.sum(1, keepdims=True)
            P = np.einsum("ik,aik->ai", w, Y[:, idx][:, o])
            b = ((P - Y[:, idx]) ** 2).mean()
            if b < best:
                best, cfg = b, (k, T)
    return cfg


def case_memory(test, cand, k, T):
    S = SC[np.ix_(test, cand)]
    o = np.argsort(-S, axis=1)[:, :k]
    s = np.take_along_axis(S, o, 1)
    w = np.exp((s - s[:, :1]) / T); w /= w.sum(1, keepdims=True)
    return np.einsum("ik,aik->ai", w, Y[:, cand][:, o])                           # A x test


def evaluate(q, mem, draws):
    """Return dict method -> (A,) Brier averaged over draws, and regrets."""
    out = {}
    def add(name, val):
        out.setdefault(name, []).append(val)
    mem = [m for m in mem if m != q]
    prior = rate[:, mem].mean(1) if mem else np.full(A, 0.5)
    pm = proc_memory(q, mem)
    if pm is None:
        pm = prior
    ccfg = case_cfg(mem) if len(mem) >= 2 else None
    memcases = np.concatenate([cases[m] for m in mem]) if mem else np.array([], int)
    for perm in draws:
        for n in NS:
            pr, te = perm[:n], perm[4:]                 # test set is the same for every n
            yt = Y[:, te]; rt = yt.mean(1); k = Y[:, pr].sum(1)
            def score(name, p):                          # p: (A,) one forecast for the procedure
                add(f"{name}|n={n}|brier", brier_rate(p, rt))
                for tau in TAUS:
                    add(f"{name}|n={n}|regret{tau}", np.maximum(rt - tau, 0) - (p >= tau) * (rt - tau))
            if n == 0:
                score("Prior", prior); score("ProcMemory", pm)
            for m in MS:
                if n > 0:
                    score(f"Probes+Prior(m={m})", (m * prior + k) / (m + n))
                    score(f"Probes+ProcMemory(m={m})", (m * pm + k) / (m + n))
            if n > 0:
                score("Probes only (k+1)/(n+2)", (k + 1) / (n + 2))
            cand = np.concatenate([memcases, pr]).astype(int)
            if ccfg and len(cand) >= ccfg[0]:
                P = case_memory(te, cand, *ccfg)
                add(f"CaseMemory|n={n}|brier", ((P - yt) ** 2).mean(1))
                for tau in TAUS:
                    add(f"CaseMemory|n={n}|regret{tau}", np.maximum(rt - tau, 0) - (P.mean(1) >= tau) * (rt - tau))
    return {k_: np.mean(v, 0).mean() for k_, v in out.items()}                    # mean over draws and agents


def boot(v, w=None):
    v = np.asarray(v); b = rng.integers(0, len(v), (2000, len(v)))
    m = v[b].mean(1)
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def table(res, title, ref="Prior|n=0|brier"):
    print(f"\n## {title}  ({len(res)} target procedures, {A} agents, {REPS} probe draws)")
    keys = sorted({k.rsplit("|", 1)[0] for r in res for k in r}, key=lambda s: (int(s.split("n=")[1]), s))
    base = np.array([r[ref] for r in res])
    print(f"{'method':38s} {'Brier':>7s} {'skill vs Prior n=0 [95% CI]':>30s} {'regret0.5':>10s} {'regret0.7':>10s}")
    for k in keys:
        b = np.array([r[k + "|brier"] for r in res])
        d, lo, hi = boot(base - b)
        s = lambda x: x / base.mean()
        print(f"{k:38s} {b.mean():7.4f} {s(d):+9.3f} [{s(lo):+.3f}, {s(hi):+.3f}] "
              f"{np.mean([r[k + '|regret0.5'] for r in res]):10.4f} {np.mean([r[k + '|regret0.7'] for r in res]):10.4f}")
    def diff(a, b_):
        d, lo, hi = boot(np.array([r[b_ + "|brier"] - r[a + "|brier"] for r in res]))
        print(f"  Brier({b_}) - Brier({a}) = {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  (positive: first is better)")
    print("paired differences:")
    diff("ProcMemory|n=0", "Prior|n=0"); diff("CaseMemory|n=0", "Prior|n=0")
    for n in (1, 2, 4):
        diff(f"Probes+ProcMemory(m=4)|n={n}", f"Probes+Prior(m=4)|n={n}")
        diff(f"CaseMemory|n={n}", f"Probes+Prior(m=4)|n={n}")
    diff("ProcMemory|n=0", "Probes+Prior(m=4)|n=1"); diff("ProcMemory|n=0", "Probes+Prior(m=4)|n=2")
    diff("Probes+ProcMemory(m=4)|n=1", "Probes+Prior(m=4)|n=2"); diff("Probes+ProcMemory(m=4)|n=2", "Probes+Prior(m=4)|n=4")


draws = {q: [rng.permutation(cases[q]) for _ in range(REPS)] for q in targets}
allp = [q for q in range(Q) if ncase[q] >= 2]
print(f"cases {N}, procedures {Q}, in memory pool {len(allp)}, targets {len(targets)}, agents {A}")
print("noise floor: Brier of the true test rate =",
      round(np.mean([np.mean([brier_rate(Y[:, p[4:]].mean(1), Y[:, p[4:]].mean(1)).mean() for p in draws[q]]) for q in targets]), 4))

res_in = [evaluate(q, allp, draws[q]) for q in targets]
table(res_in, "In-domain memory: all other procedures")
res_ood = [evaluate(q, [m for m in allp if pdom[m] != pdom[q]], draws[q]) for q in targets]
table(res_ood, "New domain: memory holds only procedures of other domains")

print("\n## Stream: memory starts empty, procedures arrive in random order (40 orders), n = 0 probes and n = 2 probes")
bins = [(0, 5), (5, 10), (10, 20), (20, 35), (35, 66)]
acc = {b: {} for b in bins}
for s in range(40):
    order = rng.permutation(allp)
    for t, q in enumerate(order):
        if q not in targets or t == 0:
            continue
        r = evaluate(q, list(order[:t]), draws[q][:5])
        b = next(b for b in bins if b[0] <= t < b[1])
        for k in ("Prior|n=0", "ProcMemory|n=0", "CaseMemory|n=0", "Probes+Prior(m=4)|n=2", "Probes+ProcMemory(m=4)|n=2",
                  "CaseMemory|n=2"):
            if k + "|brier" in r:
                acc[b].setdefault(k, []).append(r[k + "|brier"])
print(f"{'procedures already in memory':30s}" + "".join(f"{k:>30s}" for k in acc[bins[-1]]))
for b in bins:
    print(f"{str(b[0]) + '-' + str(b[1] - 1):30s}" + "".join(f"{np.mean(acc[b].get(k, [np.nan])):30.4f}" for k in acc[bins[-1]]))
json.dump({"in": res_in, "ood": res_ood, "targets": [procs[q] for q in targets]}, open(Path(__file__).parent / "results.json", "w"))
