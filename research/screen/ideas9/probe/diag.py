"""Diagnostics (not used for selection). usage: diag.py. Labels b are used only to score; ridge probe is fitted on training groups only."""
import numpy as np, pandas as pd, json
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import feats as F
R = F.R; LAYERS = F.LAYERS
def bvals(bench, ids): return pd.read_csv(R / bench / "irt/1d_1pl/items.csv", index_col=0).loc[ids, "b"].values
def within(score, b, g, minn=5):
    r = []
    for v in np.unique(g):
        m = g == v
        if m.sum() >= minn and np.std(score[m]) > 0: r.append(spearmanr(score[m], b[m])[0])
    return float(np.nanmean(r)), len(r)
out = []; D = {}
for bench in ["sopbench", "tau2", "tauk_banking"]:
    tasks, ids, last, ref, mreq, sc = F.load(bench); b = bvals(bench, ids)
    dom = tasks.domain.values; proc = tasks.procedure_id.values; g = F.group_of(bench, tasks)
    D[bench] = dict(ids=ids, b=b, dom=dom, proc=proc, g=g, last=last, mreq=mreq, sc=sc, tasks=tasks)
    gin = proc if bench == "sopbench" else dom   # within-group = within procedure (SOP), within domain (tau2, banking)
    L = [np.log(sc[:, 2]), np.log(sc[:, 3])]; shift = np.linalg.norm(last - ref, axis=2)
    out.append(f"== {bench}: within-{'procedure' if bench=='sopbench' else 'domain'} Spearman with b (mean over groups, n groups)")
    for nm, s in [("log_nreq", L[0]), ("log_nA", L[1]), ("nll_cond", sc[:, 0]), ("nll_unc", sc[:, 1]), ("pmi", sc[:, 1] - sc[:, 0])] + [(f"shift{l}", shift[:, i]) for i, l in enumerate(LAYERS)]:
        r, n = within(s, b, gin); out.append(f"  {nm:10s} {r:+.3f} ({n})")
    for li, l in enumerate(LAYERS):   # centred last-token PCs per layer (diagnostic only)
        X = F.group_center(last[:, li], g); pcs, _, _ = F.pca(X, 4)
        out.append(f"  centred last L{l}: " + " ".join(f"pc{k+1} {within(pcs[:, k], b, gin)[0]:+.3f}" for k in range(4)))
    X = F.group_center(mreq[:, 3], g); pcs, _, _ = F.pca(X, 4)
    out.append(f"  centred mean-request L24: " + " ".join(f"pc{k+1} {within(pcs[:, k], b, gin)[0]:+.3f}" for k in range(4)))
    # length explained
    len2 = np.column_stack([np.ones(len(ids)), L[0], L[1]])
    for nm, X in [("raw", last[:, 3]), ("centred", F.group_center(last[:, 3], g))]:
        pcs, _, s = F.pca(X, 16); Y = pcs - pcs.mean(0)
        beta = np.linalg.lstsq(len2, Y, rcond=None)[0]; res = Y - len2 @ beta
        r2 = 1 - (res ** 2).sum(0) / (Y ** 2).sum(0)
        out.append(f"  R2 of PC1..16 on [log nreq, log nA] ({nm}): " + " ".join(f"{x:.2f}" for x in r2) + f" | variance-weighted {np.average(r2, weights=s[:16]**2):.2f}")
# ridge probe scores
def feat64(bench):
    d = D[bench]; X = F.group_center(d["last"][:, 3], d["g"]); pcs, _, _ = F.pca(X, 64); return pcs
def fit_pred(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    m = RidgeCV(alphas=np.logspace(0, 4, 9), cv=KFold(5, shuffle=True, random_state=0)).fit((Xtr - mu) / sd, ytr)
    return m.predict((Xte - mu) / sd)
def z(b): return (b - b.mean()) / b.std()
res = {}
out.append("== ridge probe on PCA64 of centred layer-24 last-token state (b-target, train groups only)")
for bench, key in [("sopbench", "proc"), ("tau2", "dom")]:
    d = D[bench]; X = feat64(bench); Xl = np.column_stack([np.log(d["sc"][:, 2]), np.log(d["sc"][:, 3])]); b = z(d["b"]); grp = d[key]
    pr = np.zeros(len(b)); pl = np.zeros(len(b))
    for v in np.unique(grp):
        te = grp == v; pr[te] = fit_pred(X[~te], b[~te], X[te]); pl[te] = fit_pred(Xl[~te], b[~te], Xl[te])
    gin = d["proc"] if bench == "sopbench" else d["dom"]; dd = d["dom"]
    # length-residualised probe score
    A = np.column_stack([np.ones(len(b)), Xl]); rp = pr - A @ np.linalg.lstsq(A, pr, rcond=None)[0]
    wd = lambda s: np.mean([spearmanr(s[dd == v], d["b"][dd == v])[0] for v in np.unique(dd)])
    out.append(f"  {bench} (leave-one-{key}-out): within-{'procedure' if bench=='sopbench' else 'domain'} probe {within(pr, d['b'], gin)[0]:+.3f}, "
               f"length-only ridge {within(pl, d['b'], gin)[0]:+.3f}, probe length-residualised {within(rp, d['b'], gin)[0]:+.3f}; "
               f"within-domain pooled rho: probe {wd(pr):+.3f}, length {wd(pl):+.3f}, resid {wd(rp):+.3f}; corr(probe, length-ridge) {spearmanr(pr, pl)[0]:+.2f}")
    res[bench] = pr
# banking: train on sop+tau2 pooled (b z-scored per bench), and 5-fold within banking
db = D["tauk_banking"]; Xb = feat64("tauk_banking")
# banking features are in a different PCA basis, so project all three benches into one basis: PCA on centred states of all cases (unlabelled)
Xs = [F.group_center(D[k]["last"][:, 3], D[k]["g"]) for k in ["sopbench", "tau2", "tauk_banking"]]
allp, _, _ = F.pca(np.vstack(Xs), 64); n1, n2 = len(Xs[0]), len(Xs[1])
Ptr = allp[:n1 + n2]; ytr = np.concatenate([z(D["sopbench"]["b"]), z(D["tau2"]["b"])]); Pb = allp[n1 + n2:]
pb = fit_pred(Ptr, ytr, Pb); bb = db["b"]
ln = np.column_stack([np.log(db["sc"][:, 2]), np.log(db["sc"][:, 3])])
out.append(f"  banking (n=97), trained on SOPBench+tau2 (joint unlabelled PCA basis): Spearman probe vs b {spearmanr(pb, bb)[0]:+.3f} (p={spearmanr(pb, bb)[1]:.2f}); log_nreq {spearmanr(ln[:,0], bb)[0]:+.3f}; "
           f"nll_cond {spearmanr(db['sc'][:,0], bb)[0]:+.3f}; pmi {spearmanr(db['sc'][:,1]-db['sc'][:,0], bb)[0]:+.3f}")
pcv = np.zeros(len(bb)); plv = np.zeros(len(bb))
for tr, te in KFold(5, shuffle=True, random_state=1).split(Xb):
    pcv[te] = fit_pred(Xb[tr], z(bb)[tr], Xb[te]); plv[te] = fit_pred(ln[tr], z(bb)[tr], ln[te])
out.append(f"  banking within-banking 5-fold ridge: probe {spearmanr(pcv, bb)[0]:+.3f}, length-only {spearmanr(plv, bb)[0]:+.3f}")
for bench in ["sopbench", "tau2"]:
    pass
print("\n".join(out)); open("diag.txt", "w").write("\n".join(out))
