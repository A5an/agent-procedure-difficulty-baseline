"""Part B: value of documentation measured in pilot runs saved. Exact (hypergeometric, DP) expected decision accuracy."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json
import numpy as np, pandas as pd
from scipy.stats import beta as Beta, hypergeom
from scipy.special import betaln, gammaln
sys.path.insert(0, ZR.SCREEN + "/ideas")
from cpulock import cpu_lock
R = ZR.SCREEN + "/ideas2/risk/"
EMPTY3 = ["gpt-5-mode_fc", "gpt-5-mini-mode_fc", "gemini-2.5-pro-mode_fc"]
TAUS = [0.5, 0.7, 0.8]; NMAX = 12; MS = [0.5, 1, 2, 4, 8, 16, 32, 64, 128]
d = pd.read_csv(R + "outcomes.csv")[["config", "case_id", "domain", "procedure_id", "success"]]
P = pd.read_csv(R + "preds_sop.csv")
t = pd.read_csv(ZR.REPO + "/data/sopbench/tasks.csv")
P["procedure_id"] = P.case_id.map(t.set_index("case_id").procedure_id)
P = P.rename(columns={"agent": "config"})

def build(agents):
    x = d[d.config.isin(agents)]
    g = x.groupby(["domain", "procedure_id", "config"]).success.agg(K="sum", N="size").reset_index()
    for mth in ["adele", "lad2_adele"]:
        q = P[P.config.isin(agents)].groupby(["procedure_id", "config"])[mth].mean().rename("q_" + mth).reset_index()
        g = g.merge(q, on=["procedure_id", "config"])
    return g

def bb_ll(K, N, a, b):
    return betaln(K + a, N - K + b) - betaln(a, b) + gammaln(N + 1) - gammaln(K + 1) - gammaln(N - K + 1)

def fit_m(tr, qcol, sa):
    """prior strength by beta-binomial marginal likelihood on training-domain (procedure, agent) pairs"""
    best, bm = -1e18, MS[0]
    for m in MS:
        mu = np.clip(tr[qcol].values if qcol else tr.config.map(sa).values, 0.02, 0.98)
        ll = bb_ll(tr.K.values, tr.N.values, m * mu, m * (1 - mu)).sum()
        if ll > best: best, bm = ll, m
    return bm

from scipy.stats import betabinom
def post_table(a, b, N, tau):
    """P(K/N >= tau | j pilots, k successes) for j,k in 0..NMAX, remaining N-j cases ~ beta-binomial predictive"""
    need_tot = int(np.ceil(tau * N - 1e-9))
    T = np.full((NMAX + 1, NMAX + 1), np.nan)
    for j in range(0, min(NMAX, N) + 1):
        k = np.arange(0, j + 1); need = need_tot - k; rem = N - j
        pr = np.where(need <= 0, 1.0, 0.0)
        mid = (need > 0) & (need <= rem)
        if rem > 0 and mid.any():
            pr[mid] = betabinom.sf(need[mid] - 1, rem, a + k[mid], b + j - k[mid])
        T[j, :j + 1] = pr
    return T

def acc_curves(g, a0, b0, tau):
    n_pairs = len(g); fixed = np.zeros((n_pairs, NMAX + 1)); seq_n = np.zeros(n_pairs); seq_acc = np.zeros(n_pairs)
    for i, (K, N, a, b) in enumerate(zip(g.K.values, g.N.values, a0, b0)):
        truth = (K / N) >= tau
        T = post_table(a, b, N, tau); nm = min(NMAX, N)
        dist = np.zeros(NMAX + 2); dist[0] = 1.0; alive = dist.copy(); en = sacc = 0.0
        for j in range(0, NMAX + 1):
            jj = min(j, N)
            ks = np.arange(0, jj + 1)
            pr = T[jj, :jj + 1]
            if j <= N:
                fixed[i, j] = (dist[:jj + 1] * ((pr >= 0.5) == truth)).sum()
            else:
                fixed[i, j] = fixed[i, N]
            stop = (pr > 0.9) | (pr < 0.1) | (j >= nm)
            pa = alive[:jj + 1]
            en += (pa * stop).sum() * jj; sacc += (pa * stop * ((pr >= 0.5) == truth)).sum()
            if j >= NMAX: break
            nd = np.zeros(NMAX + 2); na = np.zeros(NMAX + 2)
            if j < N:
                ps = (K - ks) / (N - j)
                ps = np.clip(ps, 0, 1)
                nd[1:jj + 2] += dist[:jj + 1] * ps; nd[:jj + 1] += dist[:jj + 1] * (1 - ps)
                cont = pa * (~stop)
                na[1:jj + 2] += cont * ps; na[:jj + 1] += cont * (1 - ps)
            else:
                nd[:jj + 1] = dist[:jj + 1]
            dist, alive = nd, na
        seq_n[i], seq_acc[i] = en, sacc
    return fixed, seq_n, seq_acc

def run(agents, label, min_cases=1):
    g = build(agents); g = g[g.N >= min_cases].reset_index(drop=True)
    doms = sorted(g.domain.unique()); res = {}
    for tau in TAUS:
        priors = {k: (np.ones(len(g)), np.ones(len(g))) for k in ["flat", "agent", "doc_adele", "doc_lad2_adele"]}
        ms = {}
        for dm in doms:
            te = (g.domain == dm).values; tr = g[~te]
            sa = tr.groupby("config")[["K", "N"]].sum().eval("K / N")
            m_a = fit_m(tr, None, sa); mu_a = np.clip(g[te].config.map(sa).values, 0.02, 0.98)
            priors["agent"][0][te] = m_a * mu_a; priors["agent"][1][te] = m_a * (1 - mu_a)
            ms[(dm, "agent")] = m_a
            for mth in ["adele", "lad2_adele"]:
                m = fit_m(tr, "q_" + mth, sa); mu = np.clip(g[te]["q_" + mth].values, 0.02, 0.98)
                priors["doc_" + mth][0][te] = m * mu; priors["doc_" + mth][1][te] = m * (1 - mu); ms[(dm, mth)] = m
        res[tau] = {}
        for pk, (a0, b0) in priors.items():
            res[tau][pk] = acc_curves(g, a0, b0, tau)
        res[tau]["_ms"] = ms
        res[tau]["_truth"] = (g.K / g.N >= tau).values
    return g, res

def summarize(g, res, label, nboot=1000):
    procs = g.procedure_id.unique(); idx = {p: np.where(g.procedure_id.values == p)[0] for p in procs}
    rng = np.random.default_rng(11); out = []; lines = []
    boots = [rng.choice(len(procs), len(procs)) for _ in range(nboot)]
    def stat(rows, tau, pk):
        f, sn, sa = res[tau][pk]
        return f[rows].mean(0), sn[rows].mean(), sa[rows].mean()
    def needed(curve_flat8, curve):
        # smallest (interpolated) n at which curve >= flat accuracy at n=8
        for n in range(0, NMAX + 1):
            if curve[n] >= curve_flat8:
                if n == 0: return 0.0
                lo, hi = curve[n - 1], curve[n]
                return n - 1 + (curve_flat8 - lo) / (hi - lo) if hi > lo else n
        return np.inf
    for tau in TAUS:
        allrows = np.arange(len(g))
        base = {pk: stat(allrows, tau, pk) for pk in ["flat", "agent", "doc_adele", "doc_lad2_adele"]}
        bs = {pk: [] for pk in base}; need = {pk: [] for pk in base}
        for bi in boots:
            rows = np.concatenate([idx[procs[j]] for j in bi])
            f8 = stat(rows, tau, "flat")[0][8]
            for pk in base:
                s = stat(rows, tau, pk); bs[pk].append(np.r_[s[0], s[1], s[2]])
                need[pk].append(needed(f8, s[0]))
        for pk in base:
            arr = np.array(bs[pk]); pt = np.r_[base[pk][0], base[pk][1], base[pk][2]]
            lo, hi = np.percentile(arr, [2.5, 97.5], axis=0)
            nd = np.array(need[pk]); nd0 = needed(base["flat"][0][8], base[pk][0])
            out.append(dict(sample=label, tau=tau, prior=pk, base_rate_automate=float(res[tau]["_truth"].mean()),
                            **{f"acc_n{n}": pt[n] for n in (0, 1, 2, 3, 5, 8, 12)},
                            **{f"lo_n{n}": lo[n] for n in (0, 1, 2, 3, 5, 8, 12)}, **{f"hi_n{n}": hi[n] for n in (0, 1, 2, 3, 5, 8, 12)},
                            seq_pilots=pt[NMAX + 1], seq_pilots_lo=lo[NMAX + 1], seq_pilots_hi=hi[NMAX + 1],
                            seq_acc=pt[NMAX + 2], seq_acc_lo=lo[NMAX + 2], seq_acc_hi=hi[NMAX + 2],
                            n_to_match_flat8=nd0, n_match_med=float(np.median(nd)), share_boot_not_reached=float(np.isinf(nd).mean())))
        # paired differences vs flat at fixed n
        for pk in ["agent", "doc_adele", "doc_lad2_adele"]:
            for n in (0, 3, 5, 8):
                dd = np.array([b[n] for b in bs[pk]]) - np.array([b[n] for b in bs["flat"]])
                lines.append(dict(sample=label, tau=tau, prior=pk, n=n, diff_vs_flat=base[pk][0][n] - base["flat"][0][n],
                                  lo=np.percentile(dd, 2.5), hi=np.percentile(dd, 97.5)))
            dd = np.array([b[NMAX + 1] for b in bs[pk]]) - np.array([b[NMAX + 1] for b in bs["flat"]])
            lines.append(dict(sample=label, tau=tau, prior=pk, n="seq pilots", diff_vs_flat=base[pk][1] - base["flat"][1], lo=np.percentile(dd, 2.5), hi=np.percentile(dd, 97.5)))
            dd = np.array([b[NMAX + 2] for b in bs[pk]]) - np.array([b[NMAX + 2] for b in bs["flat"]])
            lines.append(dict(sample=label, tau=tau, prior=pk, n="seq acc", diff_vs_flat=base[pk][2] - base["flat"][2], lo=np.percentile(dd, 2.5), hi=np.percentile(dd, 97.5)))
        for pk in ["doc_adele", "doc_lad2_adele"]:
            for n in (0, 1, 3, 5, 8):
                dd = np.array([b[n] for b in bs[pk]]) - np.array([b[n] for b in bs["agent"]])
                lines.append(dict(sample=label, tau=tau, prior=pk + " minus agent", n=n, diff_vs_flat=base[pk][0][n] - base["agent"][0][n], lo=np.percentile(dd, 2.5), hi=np.percentile(dd, 97.5)))
            for nm_, ix in (("seq pilots", NMAX + 1), ("seq acc", NMAX + 2)):
                dd = np.array([b[ix] for b in bs[pk]]) - np.array([b[ix] for b in bs["agent"]])
                bp = base[pk][1] if ix == NMAX + 1 else base[pk][2]; ba = base["agent"][1] if ix == NMAX + 1 else base["agent"][2]
                lines.append(dict(sample=label, tau=tau, prior=pk + " minus agent", n=nm_, diff_vs_flat=bp - ba, lo=np.percentile(dd, 2.5), hi=np.percentile(dd, 97.5)))
    return pd.DataFrame(out), pd.DataFrame(lines)

allag = sorted(d.config.unique()); ag25 = [a for a in allag if a not in EMPTY3]
tabs = []; diffs = []; ms_log = {}
with cpu_lock("risk_B"):
    for label, ags, mc in [("25 agents, all procedures", ag25, 1), ("28 agents, all procedures", allag, 1), ("25 agents, procedures with >=6 cases", ag25, 6)]:
        g, res = run(ags, label, mc)
        a, b = summarize(g, res, label); tabs.append(a); diffs.append(b)
        ms_log[label] = {f"{tau}": {f"{k[0]}|{k[1]}": v for k, v in res[tau]["_ms"].items()} for tau in TAUS[:1]}
        print(label, len(g), "pairs", g.procedure_id.nunique(), "procs", flush=True)
T = pd.concat(tabs); D = pd.concat(diffs)
T.to_csv(R + "partB_table.csv", index=False); D.to_csv(R + "partB_diffs.csv", index=False)
json.dump(ms_log, open(R + "partB_fitted_strengths.json", "w"), indent=1)
pd.set_option("display.width", 250)
print(json.dumps({k:sorted(set(v["0.5"].values())) for k,v in ms_log.items()}))
print(T[["sample", "tau", "prior", "base_rate_automate", "acc_n0", "acc_n1", "acc_n3", "acc_n5", "acc_n8", "acc_n12", "seq_pilots", "seq_acc", "n_to_match_flat8", "share_boot_not_reached"]].round(3).to_string())
print(D.round(3).to_string())
