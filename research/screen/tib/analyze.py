"""Analysis for P0-2 on TauIndianBankBench, written before the first test episode (PREREG.md). Run once.

Primary: Spearman between frozen-MF predicted and real per-family failure (per agent); router over {g38, flash, human}
vs the best constant policy; log-loss of MF vs raw probe rate vs no-evidence MF. Secondary: raw probe rho, MF refit
leave-one-family-out on this benchmark, split-half reliability of the real target, probe validity, cost per episode.
"""
import json, os, sys, collections
import numpy as np, torch
from scipy.stats import spearmanr
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mf"))
from model import MF

rng = np.random.default_rng(20261102)
sel = json.load(open(HERE / "selection.json"))
FAMS = sel["test_families"]; CLS = np.array([sel["classes"][f] for f in FAMS])
AGS = ["g38", "flash"]; MF_AGENTS = ["lite", "flash", "g38", "sonnet"]; MIDX = {"g38": 2, "flash": 1}


def fam_of(tid):
    for f in FAMS:
        if f"_{f}_" in tid: return f
    return None


SCORING = os.environ.get("PROBE_SCORING", "v2")   # deviation (REPORT.md): v2 = benchmark's per-tool compare_args


def outcomes(which, a):
    p = HERE / "runs" / (f"{which}_{a}.v2.jsonl" if which == "probes" and SCORING == "v2" else f"{which}_{a}.jsonl")
    d = collections.defaultdict(list)
    seen = set()
    for l in open(p):
        r = json.loads(l)
        if r.get("reward") is None or r["task_id"] in seen: continue
        seen.add(r["task_id"]); d[fam_of(r["task_id"])].append(1 - int(r["reward"] >= 1.0))
    return d


REAL = {a: outcomes("real", a) for a in AGS}
PROBE = {a: outcomes("probes", a) for a in AGS}
real_fail = {a: np.array([np.mean(REAL[a][f]) if REAL[a][f] else np.nan for f in FAMS]) for a in AGS}

# frozen model
P = torch.load(HERE / "mf_params.pt")
m = MF(len(MF_AGENTS))
for n in m.p: m.p[n] = P[n].clone()


def mf_pred(target, own_only=True):
    nS = np.zeros((4, len(FAMS))); fS = np.zeros_like(nS)
    for a in (AGS if not own_only else [target]):
        for j, f in enumerate(FAMS):
            nS[MIDX[a], j] = len(PROBE[a][f]); fS[MIDX[a], j] = sum(PROBE[a][f])
    z = np.zeros_like(nS)
    return m.predict(MIDX[target], CLS, np.zeros(len(FAMS)), z, z, nS, fS)


def mf_prior(target):
    z = np.zeros((4, len(FAMS)))
    return m.predict(MIDX[target], CLS, np.zeros(len(FAMS)), z, z, z, z)


def raw(target):
    return np.array([(sum(PROBE[target][f]) + .5) / (len(PROBE[target][f]) + 1) for f in FAMS])


def rho_ci(x, y, B=2000):
    ok = ~np.isnan(y); x, y = x[ok], y[ok]
    r = spearmanr(x, y).correlation
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0: bs.append(spearmanr(x[i], y[i]).correlation)
    return float(r), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def ll(pfam, a):
    out = []
    for j, f in enumerate(FAMS):
        p = np.clip(pfam[j], 1e-4, 1 - 1e-4)
        out += [-(y * np.log(p) + (1 - y) * np.log(1 - p)) for y in REAL[a][f]]
    return float(np.mean(out))


def lofo(target):
    """refit on this benchmark: two agents (g38, flash), 19 families real + probes, predict the 20th from its probes"""
    pr = np.zeros(len(FAMS))
    nR = np.array([[len(REAL[a][f]) for f in FAMS] for a in AGS], float); fR = np.array([[sum(REAL[a][f]) for f in FAMS] for a in AGS], float)
    nS = np.array([[len(PROBE[a][f]) for f in FAMS] for a in AGS], float); fS = np.array([[sum(PROBE[a][f]) for f in FAMS] for a in AGS], float)
    ti = AGS.index(target)
    for j in range(len(FAMS)):
        tr = [i for i in range(len(FAMS)) if i != j]
        mm = MF(2).fit((CLS[tr], np.zeros(len(tr)), nR[:, tr], fR[:, tr], nS[:, tr], fS[:, tr]))
        ev_nS = nS[:, [j]].copy(); ev_fS = fS[:, [j]].copy()
        other = 1 - ti; ev_nS[other] = 0; ev_fS[other] = 0
        z = np.zeros_like(ev_nS)
        pr[j] = mm.predict(ti, CLS[[j]], np.zeros(1), z, z, ev_nS, ev_fS)[0]
    return pr


def split_half(a, S=500):
    rs = []
    for s in range(S):
        r = np.random.default_rng(s); x, y = [], []
        for f in FAMS:
            v = np.array(REAL[a][f]); r.shuffle(v); h = len(v) // 2
            x.append(v[:h].mean()); y.append(v[h:].mean())
        x, y = np.array(x), np.array(y)
        if np.ptp(x) > 0 and np.ptp(y) > 0: rs.append(spearmanr(x, y).correlation)
    sh = float(np.mean(rs)); return {"split_half": sh, "spearman_brown": 2 * sh / (1 + sh)}


def cost_per_episode():
    L = [json.loads(l) for l in open(HERE / "ledger.jsonl")]
    by = collections.defaultdict(float)
    for r in L: by[(r.get("tag"), r.get("model"))] += r.get("usd", 0)
    return {f"{k[0]}|{k[1]}": round(v, 4) for k, v in by.items()}, round(sum(r.get("usd", 0) for r in L), 4)


if __name__ == "__main__":
    res = {"n_real": {a: int(sum(len(v) for v in REAL[a].values())) for a in AGS},
           "n_probe_eps": {a: int(sum(len(v) for v in PROBE[a].values())) for a in AGS}}
    preds = {}
    for a in AGS:
        y = real_fail[a]
        preds[a] = {"mf": mf_pred(a), "mf_both": mf_pred(a, own_only=False), "raw": raw(a), "prior": mf_prior(a), "lofo": lofo(a)}
        res[a] = {"real_fail_mean": float(np.nanmean(y)), "probe_fail_mean": float(np.mean([v for f in FAMS for v in PROBE[a][f]])),
                  "rho_mf (primary)": rho_ci(preds[a]["mf"], y), "rho_raw": rho_ci(preds[a]["raw"], y),
                  "rho_mf_both_agents_probes": rho_ci(preds[a]["mf_both"], y), "rho_lofo_refit": rho_ci(preds[a]["lofo"], y),
                  "logloss_mf": ll(preds[a]["mf"], a), "logloss_raw": ll(preds[a]["raw"], a), "logloss_prior": ll(preds[a]["prior"], a),
                  "logloss_lofo": ll(preds[a]["lofo"], a), "reliability": split_half(a)}
    # router
    usd, total = cost_per_episode()
    n_pilot = {a: sum(1 for l in open(HERE / "runs" / f"pilot_{a}.jsonl") if json.loads(l).get("reward") is not None) for a in AGS}
    n_eps = {a: res["n_real"][a] + res["n_probe_eps"][a] + n_pilot[a] for a in AGS}   # the ledger also holds the pilot
    API = {a: (usd.get(f"agent|{ {'g38': 'gemini-3.8-flash', 'flash': 'gemini-2.5-flash'}[a] }", 0) / max(1, n_eps[a])) for a in AGS}
    res["api_usd_per_episode_agent_only"] = API; res["ledger_by_tag"] = usd; res["ledger_total"] = total
    res["routing"] = {}
    for F, h, q in ((20, 5, 0.99), (20, 2, 0.95), (50, 5, 0.99), (10, 5, 0.99)):
        L = np.vstack([API[a] + F * real_fail[a] for a in AGS] + [np.full(len(FAMS), h + F * (1 - q))])
        per = {f"always {o}": L[i] for i, o in enumerate(AGS + ["human"])}
        per["oracle"] = L.min(0)
        for name in ("mf", "raw", "lofo"):
            PL = np.vstack([API[a] + F * preds[a][name] for a in AGS] + [np.full(len(FAMS), h + F * (1 - q))])
            per[f"router {name}"] = L[PL.argmin(0), np.arange(len(FAMS))]
        best = min((k for k in per if k.startswith("always")), key=lambda k: per[k].mean())
        out = {k: float(v.mean()) for k, v in per.items()}; out["best_constant"] = best
        for name in ("mf", "raw", "lofo"):
            d = per[f"router {name}"] - per[best]
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            out[f"router {name} - best constant"] = (float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
        res["routing"][f"F={F} h={h} q={q}"] = out
    # probe validity
    plog = [json.loads(l) for l in open(HERE / "probes_log.jsonl")]
    res["probe_attempts"] = len(plog); res["probe_kept"] = sum(1 for r in plog if r.get("kept"))
    res["probe_drop_reasons"] = collections.Counter((r.get("drop") or "").split(":")[0] for r in plog if not r.get("kept")).most_common()
    res["per_family"] = {f: {a: {"real": REAL[a][f], "probe_fail": PROBE[a][f], "mf": float(preds[a]["mf"][j])} for a in AGS} for j, f in enumerate(FAMS)}
    res["probe_scoring"] = SCORING
    json.dump(res, open(HERE / f"results_{SCORING}.json", "w"), indent=1, default=float)
    for k, v in res.items():
        if k != "per_family": print(k, json.dumps(v, default=float)[:900])
