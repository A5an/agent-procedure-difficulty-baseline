"""E5 copy of ready_proc.py (A5an branch research/2026-11-11-diagnostic-memory): only the family similarity changes (SIM_MODE).
READY-style qualification at the procedure level and matched-budget baselines (see PREREG.md).
Run: python3 ready_proc.py > ready_proc.out   (recorded data only, no LLM calls)"""
import csv, json, math, os, re, sys, collections
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SCR = HERE.parent
ROOT = HERE.parents[2]
TARGETS, MARGIN, NBOOT, MS = [0.90, 0.85], float(os.environ.get("MARGIN", 0.05)), 300, [2, 4, 8]
_cp = {}


def cp_lower(k, n, alpha=0.05):
    """one-sided Clopper-Pearson lower bound: p with P(X >= k | n, p) = alpha"""
    k, n = int(k), int(n)
    if k == 0: return 0.0
    if (k, n) in _cp: return _cp[(k, n)]
    lc = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in range(k, n + 1)]
    def sf(p):
        if p <= 0: return 0.0
        if p >= 1: return 1.0
        return sum(math.exp(c + i * math.log(p) + (n - i) * math.log(1 - p)) for c, i in zip(lc, range(k, n + 1)))
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if sf(mid) < alpha: lo = mid
        else: hi = mid
    _cp[(k, n)] = lo
    return lo


def brier(p, s, n):
    return float((s * (1 - p) ** 2 + (n - s) * p ** 2).sum() / n.sum())


def qualify(ed, sd, nd, eq, sq, nq, target, rng):
    """threshold on dev, outcome on qualification. e = estimate, s = real successes, n = real cases (per procedure)."""
    ed, eq = np.round(ed, 9), np.round(eq, 9)
    t = None
    for c in np.unique(ed):
        a = ed >= c
        if sd[a].sum() / nd[a].sum() >= target + MARGIN - 1e-12:
            t = c; break
    out = {"none": 1.0, "sh_proc": 0.0, "sh_case": 0.0}
    if t is None: return out
    a = eq >= t
    if not a.any(): return out
    s, n = sq[a], nq[a]
    succ = s.sum() / n.sum()
    idx = rng.integers(0, len(s), (NBOOT, len(s)))
    bs = s[idx].sum(1) / n[idx].sum(1)
    cp = cp_lower(s.sum(), n.sum()); bl = float(np.percentile(bs, 5))
    return {"none": 0.0, "sh_proc": float(a.mean()), "sh_case": float(n.sum() / nq.sum()), "succ": float(succ),
            "held": float(succ >= target), "cp": cp, "cp_ok": float(cp >= target), "boot": bl, "boot_ok": float(bl >= target),
            "n_proc": float(a.sum())}


def agg(rows):
    """rows: list of qualify() dicts over splits"""
    o = {k: float(np.mean([r[k] for r in rows])) for k in ("none", "sh_proc", "sh_case")}
    acc = [r for r in rows if r["none"] == 0]
    for k in ("succ", "held", "cp", "boot", "n_proc"):
        o[k] = float(np.mean([r[k] for r in acc])) if acc else float("nan")
    # qualification verdicts counted over all splits (nothing accepted = not qualified)
    for k in ("cp_ok", "boot_ok"):
        o[k] = float(np.sum([r[k] for r in acc]) / len(rows))
    return o


def show(title, res, order):
    print(f"\n{title}")
    print(f"{'estimator':22s} {'Brier':>7s} | " + " | ".join(
        f"target {t:.2f}: none  %proc %case  succ  held  CPlb CP>=t  BSlb BS>=t" for t in TARGETS))
    for e in order:
        r = res[e]
        line = f"{e:22s} {r['brier']:7.4f} | "
        for t in TARGETS:
            q = r[f"{t:.2f}"]
            line += (f"             {q['none']:5.2f} {q['sh_proc']:5.2f} {q['sh_case']:5.2f} {q['succ']:5.3f} {q['held']:5.2f} "
                     f"{q['cp']:5.3f} {q['cp_ok']:5.2f} {q['boot']:5.3f} {q['boot_ok']:5.2f} | ")
        print(line)


def knn(sim, vals, k=5, T=0.05):
    o = np.argsort(-sim)[:k]
    w = np.exp((sim[o] - sim[o[0]]) / T)
    return float(vals[o] @ w / w.sum())


RES = {}

# ------------------------------------------------------------------ TauIndianBank
def tib():
    import torch
    sys.path.insert(0, str(SCR / "mf"))
    from model import MF
    s1 = json.load(open(SCR / "tib" / "selection.json")); s2 = json.load(open(SCR / "combo_rep" / "selection_rep.json"))
    F = s1["test_families"] + s2["test_families"]
    cls = np.array([{**s1["classes"], **s2["classes"]}[f] for f in F]); purpose = {**s1["purpose"], **s2["purpose"]}
    AG = ["g38", "flash"]; MIDX = {"g38": 2, "flash": 1}
    files = {"real": [SCR / "tib/runs/real_{a}.jsonl", SCR / "combo_rep/runs/real_{a}.jsonl"],
             "doc": [SCR / "tib/runs/probes_{a}.v2.jsonl", SCR / "combo_rep/runs/doc_{a}.jsonl"],
             "seeded": [SCR / "tib/runs/seeded_{a}.jsonl", SCR / "combo_rep/runs/seeded_{a}.jsonl"]}

    def load(src, a):
        s, n, seen = np.zeros(len(F)), np.zeros(len(F)), set()
        for p in files[src]:
            for l in open(str(p).format(a=a)):
                r = json.loads(l)
                if r.get("reward") is None or r["task_id"] in seen: continue
                seen.add(r["task_id"])
                j = next(i for i, f in enumerate(F) if f"_{f}_" in r["task_id"])
                n[j] += 1; s[j] += int(r["reward"] >= 1.0)
        return s, n

    # TF-IDF over purpose + family prefix
    MODE = os.environ.get("SIM_MODE", "tfidf_prefix")   # E5: tfidf_prefix (original), tfidf (no family prefix), bge
    docs = [re.findall(r"[a-z0-9]+", purpose[f].lower()) + (["pfx_" + re.match(r"(edga|edgb|hp|sq2|sq|tl)", f).group(1)] if MODE == "tfidf_prefix" else []) for f in F]
    vocab = sorted({w for d in docs for w in d}); df = collections.Counter(w for d in docs for w in set(d))
    X = np.array([[d.count(w) * math.log(len(F) / df[w] + 1) for w in vocab] for d in docs])
    X /= np.linalg.norm(X, axis=1, keepdims=True); SIM = X @ X.T
    if MODE == "bge":
        z = np.load(HERE / "emb_tib.npz", allow_pickle=True); assert list(z["families"]) == F
        SIM = z["X"] @ z["X"].T
    print("similarity:", MODE)

    P = torch.load(SCR / "tib" / "mf_params.pt"); m = MF(4)
    for nme in m.p: m.p[nme] = P[nme].clone()

    rng = np.random.default_rng(0)
    print("=" * 30, "TauIndianBank, 50 families, 6 real tasks each")
    for a in AG:
        sR, nR = load("real", a)
        rate = sR / nR
        print(f"\nagent {a}: real success {sR.sum() / nR.sum():.3f}")
        probes = {src: load(src, a) for src in ("doc", "seeded")}
        mfp = {}
        for src, (k, n) in probes.items():
            nS = np.zeros((4, len(F))); fS = np.zeros_like(nS); nS[MIDX[a]] = n; fS[MIDX[a]] = n - k; z = np.zeros_like(nS)
            mfp[src] = 1 - m.predict(MIDX[a], cls, np.zeros(len(F)), z, z, nS, fS)
            print(f"  {src}: probe success {k.sum() / n.sum():.3f}, probes {int(n.sum())}")
            print("    real success by probe successes k (families, real successes/cases, rate, CP lower bound): " + "; ".join(
                f"k={int(kk)}: {int((k == kk).sum())} fam {int(sR[k == kk].sum())}/{int(nR[k == kk].sum())} = {sR[k == kk].sum() / nR[k == kk].sum():.2f} (lb {cp_lower(sR[k == kk].sum(), nR[k == kk].sum()):.2f})"
                for kk in sorted(set(k))[::-1]))
        names = ["const (0 probes)", "knn text (0 probes)"] + [f"{src} {e}" for src in probes for e in ("raw", "bb", "bb_knn", "mf")]
        acc = {e: {"brier": [], **{f"{t:.2f}": [] for t in TARGETS}} for e in names}
        for sp in range(1000):
            perm = rng.permutation(len(F)); D, Q = perm[:25], perm[25:]
            pi = sR[D].sum() / nR[D].sum()
            kn = np.zeros(len(F))
            for j in Q: kn[j] = knn(SIM[j, D], rate[D])
            for j in D:
                o = D[D != j]; kn[j] = knn(SIM[j, o], rate[o])
            est = {"const (0 probes)": np.full(len(F), pi), "knn text (0 probes)": kn}
            for src, (k, n) in probes.items():
                est[f"{src} raw"] = k / n
                for nm, prior in (("bb", np.full(len(F), pi)), ("bb_knn", kn)):
                    mb = min(MS, key=lambda mm: brier(((mm * prior + k) / (mm + n))[D], sR[D], nR[D]))
                    est[f"{src} {nm}"] = (mb * prior + k) / (mb + n)
                est[f"{src} mf"] = mfp[src]
            for e in names:
                acc[e]["brier"].append(brier(est[e][Q], sR[Q], nR[Q]))
                for t in TARGETS:
                    acc[e][f"{t:.2f}"].append(qualify(est[e][D], sR[D], nR[D], est[e][Q], sR[Q], nR[Q], t, rng))
        res = {e: {"brier": float(np.mean(acc[e]["brier"])), **{f"{t:.2f}": agg(acc[e][f"{t:.2f}"]) for t in TARGETS}} for e in names}
        show(f"TIB / {a}  (always-agent success {sR.sum() / nR.sum():.3f})", res, names)
        RES[f"tib/{a}"] = res
        RES[f"tib/{a}/per_family"] = {f: {"real": [int(sR[j]), int(nR[j])], "doc": [int(probes['doc'][0][j]), int(probes['doc'][1][j])],
                                          "seeded": [int(probes['seeded'][0][j]), int(probes['seeded'][1][j])]} for j, f in enumerate(F)}


# ------------------------------------------------------------------ SOPBench
def sop():
    D_ = ROOT / "data" / "sopbench"
    rows = list(csv.DictReader(open(D_ / "tasks.csv"))); ids = [r["case_id"] for r in rows]
    z = np.load(D_ / "features" / "bge_base_en_v1.5.npz", allow_pickle=True)
    pos = {t: i for i, t in enumerate(z["task_ids"])}
    X = z["X"][[pos[i] for i in ids]].astype(float); X /= np.linalg.norm(X, axis=1, keepdims=True)
    procs = sorted({r["procedure_id"] for r in rows}); pidx = np.array([procs.index(r["procedure_id"]) for r in rows])
    R = [json.loads(l) for l in open(D_ / "responses.jsonl")]
    agents = [r["subject_id"] for r in R]
    Y = np.array([[r["responses"].get(i, np.nan) for i in ids] for r in R], float)
    Y = np.where(np.isnan(Y), np.nanmean(Y, axis=1, keepdims=True), Y)
    Y = (Y >= 0.5).astype(float)                       # 5 imputed cells -> 0/1 so counts stay integer
    A, Qn = Y.shape[0], len(procs)
    cases = [np.where(pidx == q)[0] for q in range(Qn)]
    rate = np.stack([Y[:, c].mean(1) for c in cases], 1)
    E_ = np.stack([X[c].mean(0) for c in cases]); E_ /= np.linalg.norm(E_, axis=1, keepdims=True); SP = E_ @ E_.T
    T_ = np.array([q for q in range(Qn) if len(cases[q]) >= 8]); hist = np.array([q for q in range(Qn) if len(cases[q]) < 8])
    overall = Y.mean(1); order_ag = np.argsort(-overall)
    groups = {"all 28 agents": list(range(A)), "top 7 agents": list(order_ag[:7]), "middle 14 agents": list(order_ag[7:21]),
              "bottom 7 agents": list(order_ag[21:])}
    print("\n" + "=" * 30, f"SOPBench, {len(T_)} target procedures, {A} agents, probes = real held-out cases")
    print("agent overall success: top7 %.3f, middle14 %.3f, bottom7 %.3f" % (overall[order_ag[:7]].mean(), overall[order_ag[7:21]].mean(), overall[order_ag[21:]].mean()))
    rng = np.random.default_rng(1)
    NS = [2, 4, 6]
    names = ["const (0 probes)", "knn memory (0 probes)"] + [f"n={n} {e}" for n in NS for e in ("raw", "bb", "bb_knn")]
    acc = {a: {e: {"brier": [], **{f"{t:.2f}": [] for t in TARGETS}} for e in names} for a in range(A)}
    base = {a: [] for a in range(A)}
    for sp in range(300):
        perm = rng.permutation(len(T_)); Dv, Qv = T_[perm[:14]], T_[perm[14:]]
        known = np.concatenate([hist, Dv])
        order = {q: rng.permutation(cases[q]) for q in T_}
        pi_q = rate[:, known].mean(1)                                    # (A,)
        prior_c, prior_k = np.zeros((A, Qn)), np.zeros((A, Qn))
        for q in T_:
            kn_ = known[known != q]
            o = np.argsort(-SP[q, kn_])[:5]; w = np.exp((SP[q, kn_][o] - SP[q, kn_][o[0]]) / 0.05)
            prior_k[:, q] = rate[:, kn_[o]] @ w / w.sum()
            prior_c[:, q] = rate[:, kn_].mean(1)
        # test cases are the cases beyond the first 6 of the permutation for every n, so n = 2, 4, 6 share one test set
        test = {q: order[q][6:] for q in T_}
        sT = np.stack([Y[:, test[q]].sum(1) for q in T_], 1); nT = np.array([len(test[q]) for q in T_], float)
        col = {q: i for i, q in enumerate(T_)}; dI = np.array([col[q] for q in Dv]); qI = np.array([col[q] for q in Qv])
        pc, pk = prior_c[:, T_], prior_k[:, T_]
        ests = {"const (0 probes)": pc, "knn memory (0 probes)": pk}
        for n in NS:
            k = np.stack([Y[:, order[q][:n]].sum(1) for q in T_], 1)
            ests[f"n={n} raw"] = k / n
            for nm, prior in (("bb", pc), ("bb_knn", pk)):
                cand = [(mm * prior + k) / (mm + n) for mm in MS]
                bd = np.stack([(sT[:, dI] * (1 - c[:, dI]) ** 2 + (nT[dI] - sT[:, dI]) * c[:, dI] ** 2).sum(1) for c in cand])  # (3,A)
                best = bd.argmin(0)
                ests[f"n={n} {nm}"] = np.stack([cand[best[a]][a] for a in range(A)])
        for a in range(A):
            base[a].append(sT[a, qI].sum() / nT[qI].sum())
            for e in names:
                p = ests[e][a]
                acc[a][e]["brier"].append(brier(p[qI], sT[a, qI], nT[qI]))
                for t in TARGETS:
                    acc[a][e][f"{t:.2f}"].append(qualify(p[dI], sT[a, dI], nT[dI], p[qI], sT[a, qI], nT[qI], t, rng))
    for g, ag in groups.items():
        res = {}
        for e in names:
            res[e] = {"brier": float(np.mean([np.mean(acc[a][e]["brier"]) for a in ag]))}
            for t in TARGETS:
                rows_ = [r for a in ag for r in acc[a][e][f"{t:.2f}"]]
                res[e][f"{t:.2f}"] = agg(rows_)
        show(f"SOPBench / {g}  (always-agent success on test cases {np.mean([np.mean(base[a]) for a in ag]):.3f})", res, names)
        RES[f"sop/{g}"] = res


if __name__ == "__main__":
    tib()   # E5 recomputes the TauIndianBank part only
    json.dump(RES, open(HERE / f"results_{os.environ.get('SIM_MODE', 'tfidf_prefix')}.json", "w"), indent=1)
