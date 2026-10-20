"""FROZEN evaluator for the evo search (round 7). Never edited after its SHA-256 is logged. The proposer never sees this file or any label.
score(code, split) -> dict. split 'search' reads only records_search.jsonl and targets_search.csv; split 'heldout' reads the heldout files.
Score = gain in mean within-domain case-level Spearman (target: IRT difficulty b) when the candidate's features are added to the c_lad
out-of-fold predicted difficulty in a ridge, outer CV 5 folds grouped by procedure (3 repeats with different group shuffles), inner
CV 4 folds grouped by procedure picks the ridge penalty. Reference = the same pipeline with the c_lad prediction alone.
Cascade: static check, run on all records (reject > 2% crashes), drop constant features and features with |r| > 0.95 against c_lad
(within-domain demeaned), reject when no feature is left."""
import json, subprocess, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = Path(__file__).parent
PY = sys.executable
GRID = [30.0, 100.0, 300.0, 1000.0, 3000.0, 1e5]
REPEATS = 3; OUTER = 5; INNER = 4

def load(split):
    t = pd.read_csv(HERE / f"targets_{split}.csv")
    return t, str(HERE / f"records_{split}.jsonl")

def run_sandbox(code, rec_path, timeout=120):
    try:
        p = subprocess.run([PY, str(HERE / "sandbox_worker.py"), rec_path], input=code, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if p.returncode != 0: return None, "worker crash: " + p.stderr[-200:]
    try: d = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception: return None, "bad output"
    if "static_error" in d: return None, "static: " + d["static_error"]
    return d, None

def demean(x, dom):
    x = np.asarray(x, float).copy()
    for d in np.unique(dom): m = dom == d; x[m] -= x[m].mean(0)
    return x

def group_folds(groups, k, rng):
    u = np.array(sorted(set(groups))); rng.shuffle(u)
    assign = {g: i % k for i, g in enumerate(u)}
    f = np.array([assign[g] for g in groups]); return f

def fit_pred(Xtr, ytr, btr, Xte, bte, gtr, rng):
    """residual ridge on top of an OLS slope on the c_lad prediction; penalty by inner grouped CV"""
    a = float(btr @ ytr / max(btr @ btr, 1e-12)); rtr = ytr - a * btr
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-9] = 1.0
    Z, Zt = (Xtr - mu) / sd, (Xte - mu) / sd
    fo = group_folds(gtr, INNER, rng); best, ba = np.inf, GRID[-1]
    for al in GRID:
        e = 0.0
        for i in range(INNER):
            tr, va = fo != i, fo == i
            if va.sum() == 0 or tr.sum() < 5: continue
            w = np.linalg.solve(Z[tr].T @ Z[tr] + al * np.eye(Z.shape[1]), Z[tr].T @ (rtr[tr] - rtr[tr].mean()))
            e += np.sum((Z[va] @ w + rtr[tr].mean() - rtr[va]) ** 2)
        if e < best: best, ba = e, al
    w = np.linalg.solve(Z.T @ Z + ba * np.eye(Z.shape[1]), Z.T @ (rtr - rtr.mean()))
    return a * bte + Zt @ w + rtr.mean(), ba

def within_rho(pred, y, dom):
    r = []
    for d in np.unique(dom):
        m = dom == d
        if m.sum() > 5 and np.std(pred[m]) > 0: r.append(spearmanr(pred[m], y[m])[0])
    return float(np.mean(r)), {str(d): float(spearmanr(pred[dom == d], y[dom == d])[0]) for d in np.unique(dom)}

def gain(F, t):
    dom = t.domain.values; grp = t.procedure_id.values
    y = demean(t.y_irt_b.values[:, None], dom)[:, 0]; b = demean(t.clad_oof.values[:, None], dom)[:, 0]
    X = demean(F, dom)
    gs, rf, rb = [], [], []; per_f = {}; per_b = {}
    for rep in range(REPEATS):
        rng = np.random.default_rng(1000 + rep); fo = group_folds(grp, OUTER, rng)
        pf, pb = np.zeros(len(y)), np.zeros(len(y))
        for i in range(OUTER):
            tr, te = fo != i, fo == i
            pf[te], _ = fit_pred(X[tr], y[tr], b[tr], X[te], b[te], grp[tr], rng)
            pb[te] = (b[tr] @ y[tr] / (b[tr] @ b[tr])) * b[te]
        r1, d1 = within_rho(pf, y, dom); r0, d0 = within_rho(pb, y, dom)
        gs.append(r1 - r0); rf.append(r1); rb.append(r0)
        for k, v in d1.items(): per_f.setdefault(k, []).append(v - d0[k])
    return float(np.mean(gs)), float(np.mean(rf)), float(np.mean(rb)), {k: float(np.mean(v)) for k, v in per_f.items()}

def score(code, split="search", return_matrix=False):
    t, rp = load(split)
    out = {"status": "ok", "score": None}
    d, err = run_sandbox(code, rp)
    if err: return {"status": "reject", "reason": err, "score": None}
    ids = [r["id"] for r in d["rows"]]
    if d["errors"] > 0.02 * len(ids): return {"status": "reject", "reason": f"crash on {d['errors']} of {len(ids)} records: {d['err_example']}", "score": None}
    keys = []
    for r in d["rows"]:
        if r["f"]:
            for k in r["f"]:
                if k not in keys: keys.append(k)
    keys = keys[:8]
    if not keys: return {"status": "reject", "reason": "no features returned", "score": None}
    tt = t.set_index("case_id").loc[ids].reset_index()
    F = np.full((len(ids), len(keys)), np.nan)
    for i, r in enumerate(d["rows"]):
        if r["f"]:
            for j, k in enumerate(keys):
                v = r["f"].get(k)
                if v is not None: F[i, j] = v
    F = np.where(np.isnan(F), np.nanmedian(F, 0, keepdims=True) if np.isfinite(F).any() else 0.0, F)
    F = np.nan_to_num(F, nan=0.0)
    F = np.clip(F, -1e6, 1e6)
    dom = tt.domain.values; Fd = demean(F, dom); bd = demean(tt.clad_oof.values[:, None], dom)[:, 0]
    keep, corr = [], []
    for j in range(F.shape[1]):
        if Fd[:, j].std() < 1e-9: continue
        r = abs(np.corrcoef(Fd[:, j], bd)[0, 1]); corr.append(float(r))
        if r > 0.95: continue
        keep.append(j)
    if not keep: return {"status": "reject", "reason": "all features constant or correlated above 0.95 with c_lad", "score": None, "corr_with_clad": corr}
    g, rf, rb, per = gain(F[:, keep], tt)
    out.update(score=g, rho_with=rf, rho_base=rb, per_domain_gain=per, n_features=len(keep), feature_names=[keys[j] for j in keep],
               max_corr_with_clad=float(max(corr)), crash_rate=d["errors"] / len(ids))
    if return_matrix: out["matrix"] = F[:, keep]; out["ids"] = ids; out["domains"] = dom
    return out
