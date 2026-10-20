import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, pickle, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,ZR.SCREEN + "/ideas/lupi")
import numpy as np, pandas as pd, lupi_lib as L
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, r2_score
D = pickle.load(open("diag.pkl","rb"))
RUB=ZR.SCREEN + "/rubric/features/rub_proc_%s.csv"
lines=[]
for bench in ["sopbench","tau2"]:
    X,P,proc,lab = L.load_bench(bench)
    T = pd.read_csv(L.REPO/"data"/bench/"tasks.csv").set_index("case_id")
    b = pd.read_csv(L.REPO/"data"/bench/"irt/1d_1pl/items.csv", index_col=0).b
    dom = T.domain
    # stage 1 (concatenate all test predictions over all folds/seeds; new_procedures covers each task 5 times, new_domain once)
    rows=[]
    for kind,bn,te,pred,cols in D:
        if bn!=bench or kind!="s1": continue
        nd = dom.loc[te].nunique(); scheme = "new_domain" if nd==1 else "new_procedures"
        for j,c in enumerate(cols):
            rows.append(pd.DataFrame({"scheme":scheme,"sig":c,"case_id":te,"pred":pred[:,j]}))
    S = pd.concat(rows); S["true"]=[P.loc[c,s] for c,s in zip(S.case_id,S.sig)]
    lines.append(f"\n## Stage-1 predictability on test tasks, {bench}\n\n| signal | scheme | R2 | Spearman | Spearman within domain |\n|---|---|---|---|---|")
    for (sg,sc),g in S.groupby(["sig","scheme"]):
        g=g.dropna(); mu,sd=P[sg].mean(),P[sg].std(); r2=r2_score((g.true-mu)/sd,g.pred)
        sp=spearmanr(g.true,g.pred)[0]
        wd=np.nanmean([spearmanr(h.true,h.pred)[0] for _,h in g.groupby(g.case_id.map(dom))])
        lines.append(f"| {sg} | {sc} | {r2:+.3f} | {sp:+.3f} | {wd:+.3f} |")
    # signals vs beta within domain
    lines.append(f"\n## Spearman of each privileged signal with fitted beta, inside domains, {bench}\n\n| signal | mean within-domain rho |\n|---|---|")
    out=[]
    for c in P.columns:
        r=np.nanmean([spearmanr(P.loc[g.index,c], b.loc[g.index])[0] for _,g in T.groupby("domain")])
        out.append((c,r))
    for c,r in sorted(out,key=lambda x:-abs(x[1])): lines.append(f"| {c} | {r:+.3f} |")
    if bench=="sopbench":
        lines.append("\n(for SOPBench, within allow/refuse label cells:)")
        for c in P.columns:
            r=np.nanmean([spearmanr(P.loc[g.index,c], b.loc[g.index])[0] for _,g in T.groupby(["domain","should_succeed"])])
            lines.append(f"- {c}: {r:+.3f}")
    # pi AUC
    rows=[]
    for kind,bn,te,pred,cols in D:
        if bn==bench and kind=="pi":
            nd = dom.loc[te].nunique(); rows.append(pd.DataFrame({"scheme":"new_domain" if nd==1 else "new_procedures","case_id":te,"pi":pred}))
    if rows:
        Pi=pd.concat(rows); Pi["lab"]=lab.loc[Pi.case_id].values; Pi["dom"]=Pi.case_id.map(dom).values
        rub=pd.read_csv(RUB%bench).set_index("case_id").p_refusal; Pi["rub"]=rub.loc[Pi.case_id].values
        lines.append(f"\n## P(refuse) AUC against the label, {bench}\n\n| scheme | logistic pi pooled | logistic pi within domain | rubric p_refusal pooled | rubric within domain |\n|---|---|---|---|---|")
        for sc,g in Pi.groupby("scheme"):
            def wd(col): return np.nanmean([roc_auc_score(h.lab,h[col]) for _,h in g.groupby("dom") if h.lab.nunique()==2])
            lines.append(f"| {sc} | {roc_auc_score(g.lab,g.pi):.3f} | {wd('pi'):.3f} | {roc_auc_score(g.lab,g.rub):.3f} | {wd('rub'):.3f} |")
    lines.append(f"\nlabel share positive ({'refuse' if bench=='sopbench' else 'refuse_or_transfer'}): {lab.mean():.3f}, n={lab.notna().sum()}")
open("diag.md","w").write("\n".join(lines)); print("\n".join(lines))
