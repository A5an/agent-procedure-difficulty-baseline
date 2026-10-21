"""Diagnostics: zero-shot Spearman with fitted beta, label AUC, per-procedure rho, position bias, agreement."""
import json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import fit as F
REPO = F.REPO
rng = np.random.default_rng(20261005)

def pooled_rho(df, a="bt", b="b"):
    num = den = 0.0
    for _, g in df.groupby("domain"):
        n = len(g)
        if n < 3 or g[a].nunique() < 2: continue
        pr = n*(n-1)/2
        num += spearmanr(g[a], g[b]).correlation*pr; den += pr
    return num/den if den else np.nan

def boot_procs(df, B=500):
    byd = {d: {p: gp for p, gp in g.groupby("procedure_id")} for d, g in df.groupby("domain")}
    out = []
    for _ in range(B):
        parts = []
        for d, pr in byd.items():
            keys = list(pr); pick = rng.choice(len(keys), len(keys))
            parts += [pr[keys[i]].assign(domain=d) for i in pick]
        out.append(pooled_rho(pd.concat(parts)))
    return np.nanpercentile(out, [2.5, 97.5])

def main():
    P = F.load_comparisons()
    res = {}
    S, gam = F.scores(P, "y_soft"); SH, gamh = F.scores(P, "y_hard")
    prim = P[P.repeat_of == ""]
    res["n_comparisons_primary"] = int(len(prim)); res["n_repeat"] = int((P.repeat_of != "").sum())
    L = []
    for bench in ["sopbench", "tau2"]:
        t = pd.read_csv(REPO/"data"/bench/"tasks.csv")
        t["b"] = t.case_id.map(pd.read_csv(REPO/"data"/bench/"irt/1d_1pl/items.csv", index_col=0).b)
        d = t.merge(S[bench][["case_id", "bt"]], on="case_id").merge(SH[bench][["case_id", "bt"]].rename(columns={"bt": "bt_hard"}), on="case_id")
        r = {"zero_shot_rho_soft": pooled_rho(d), "zero_shot_rho_hard": pooled_rho(d, "bt_hard")}
        r["zero_shot_rho_soft_ci_procedures"] = list(boot_procs(d))
        r["per_domain_rho"] = {dm: float(spearmanr(g.bt, g.b).correlation) for dm, g in d.groupby("domain")}
        if bench == "sopbench":
            d["refuse"] = 1 - d.should_succeed
            aucs, ws = {}, {}
            for dm, g in d.groupby("domain"):
                if g.refuse.nunique() == 2: aucs[dm] = roc_auc_score(g.refuse, g.bt); ws[dm] = len(g)
            r["auc_refuse_by_domain"] = aucs
            r["auc_refuse_weighted"] = float(np.average(list(aucs.values()), weights=list(ws.values())))
            # rho with beta inside each procedure, and inside procedure x label
            pr = []
            for p, g in d.groupby("procedure_id"):
                if len(g) >= 5 and g.bt.nunique() > 1 and g.b.nunique() > 1:
                    pr.append((len(g), spearmanr(g.bt, g.b).correlation))
            w = np.array([n*(n-1)/2 for n, _ in pr]); v = np.array([x for _, x in pr])
            r["within_procedure_rho_weighted"] = float((w*v).sum()/w.sum()); r["within_procedure_rho_median"] = float(np.median(v)); r["n_procedures_used"] = len(pr)
            pr2 = []
            for (p, s), g in d.groupby(["procedure_id", "should_succeed"]):
                if len(g) >= 5 and g.bt.nunique() > 1 and g.b.nunique() > 1: pr2.append((len(g), spearmanr(g.bt, g.b).correlation))
            w = np.array([n*(n-1)/2 for n, _ in pr2]); v = np.array([x for _, x in pr2])
            r["within_procedure_label_rho_weighted"] = float((w*v).sum()/w.sum()) if len(pr2) else None
            # rho inside label (domain x label cells)
            num = den = 0
            for (dm, s), g in d.groupby(["domain", "should_succeed"]):
                n = len(g); pr_ = n*(n-1)/2
                if n >= 3: num += spearmanr(g.bt, g.b).correlation*pr_; den += pr_
            r["rho_given_label_domain"] = num/den
        res[bench] = r
        L.append(d.assign(bench=bench))
    # pairs: agreement with truly harder case
    D = pd.concat(L)[["case_id", "b"]].set_index("case_id").b
    q = prim.copy(); q["bA"] = q.a.map(D); q["bB"] = q.b.map(D); q["db"] = q.bA - q.bB
    q["A_true"] = q.db > 0
    big = q[q.db.abs() > 1]
    ag = {}
    for bench, g in big.groupby("bench"):
        ag[bench] = {"n": int(len(g)), "agree": float(((g.harder == "A") == g.A_true).mean())}
        for k, gk in g.groupby("kind"): ag[bench][f"agree_{k}"] = [int(len(gk)), float(((gk.harder == "A") == gk.A_true).mean())]
    # agreement weighted by confidence >=0.9 as extra
    res["agreement_abs_dbeta_gt1"] = ag
    res["agreement_all_pairs"] = {b: float(((g.harder == "A") == g.A_true).mean()) for b, g in q[q.db != 0].groupby("bench")}
    # position bias
    res["share_A_chosen"] = {b: float((g.harder == "A").mean()) for b, g in prim.groupby("bench")}
    res["gamma_soft"] = {f"{k[0]}/{k[1]}": float(v) for k, v in gam.items()}
    res["gamma_hard"] = {f"{k[0]}/{k[1]}": float(v) for k, v in gamh.items()}
    res["mean_confidence"] = float(prim.conf.mean())
    # swapped repeat consistency
    rep = P[P.repeat_of != ""].copy(); o = prim.set_index("pair_id")
    rep["orig_harder_case"] = rep.repeat_of.map(lambda i: (o.loc[int(i)].a if o.loc[int(i)].harder == "A" else o.loc[int(i)].b) if int(i) in o.index else None)
    rep["rep_harder_case"] = np.where(rep.harder == "A", rep.a, rep.b)
    rep = rep[rep.orig_harder_case.notna()]
    res["swap_repeat"] = {"n": int(len(rep)), "same_case_chosen": float((rep.orig_harder_case == rep.rep_harder_case).mean())}
    for b, g in rep.groupby("bench"): res["swap_repeat"][b] = float((g.orig_harder_case == g.rep_harder_case).mean())
    # reliability: split halves of primary comparisons per domain
    rel = {}
    for bench in ["sopbench", "tau2"]:
        t = pd.read_csv(REPO/"data"/bench/"tasks.csv"); rs = []; ws = []
        for dm, g in t.groupby("domain"):
            ids = list(g.case_id); Q = prim[(prim.bench == bench) & (prim.domain == dm)]
            for rep_i in range(5):
                perm = np.random.default_rng(100+rep_i).permutation(len(Q)); h = len(Q)//2
                s1, _ = F.fit_domain(Q.iloc[perm[:h]], ids, "y_soft"); s2, _ = F.fit_domain(Q.iloc[perm[h:]], ids, "y_soft")
                rs.append(np.corrcoef(s1, s2)[0, 1]); ws.append((dm, len(ids), spearmanr(s1, s2).correlation))
        r_ = float(np.mean(rs))
        rel[bench] = {"split_half_pearson": r_, "spearman_brown": 2*r_/(1+r_), "split_half_spearman": float(np.mean([w[2] for w in ws]))}
    res["reliability"] = rel
    json.dump(res, open(F.HERE/"diagnostics.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
if __name__ == "__main__":
    main()
