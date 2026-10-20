"""Ceiling of the emulated pilot at procedure level, from public SOPBench outcomes (no new agent runs, no LLM calls).
If a synthetic case were as good as a real case of the same procedure, how well would one agent's mean outcome on N such cases
rank the procedures of a domain by real difficulty?"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, numpy as np, pandas as pd, sys
from scipy.stats import spearmanr
REPO=ZR.REPO + "/data/sopbench"
t=pd.read_csv(f"{REPO}/tasks.csv"); b=pd.read_csv(f"{REPO}/irt/1d_1pl/items.csv",index_col=0).b
t["b"]=t.case_id.map(b)
R={}
for l in open(f"{REPO}/responses.jsonl"):
    r=json.loads(l); R[r["subject_id"]]=r["responses"]
agents=list(R)
Y=pd.DataFrame({a:pd.Series(R[a]) for a in agents}).reindex(t.case_id)  # 830 x 28, NaN missing
Y.index=t.index
F=1-Y   # failure
t["perf"]=t.should_succeed.astype(int)
procs=t.groupby(["domain","procedure_id"]).size().reset_index(name="n")
def pair_rho(est, tgt):
    """est, tgt: Series indexed by (domain, procedure_id). pair-weighted within-domain Spearman"""
    d=pd.DataFrame({"e":est,"t":tgt}).reset_index()
    num=den=0
    for _,g in d.groupby("domain"):
        n=len(g)
        if n<3: continue
        pr=n*(n-1)/2
        if g.e.nunique()>1 and g.t.nunique()>1: num+=spearmanr(g.e,g.t).correlation*pr
        den+=pr
    return num/den
rng=np.random.default_rng(0)
grp=t.groupby(["domain","procedure_id"]).groups
tgt_b=t.groupby(["domain","procedure_id"]).b.mean()
def est_full(a, w=None):
    f=F[a]
    if w is None: return f.groupby([t.domain,t.procedure_id]).mean()
    # class-reweighted: w = assumed perform share
    g=f.groupby([t.domain,t.procedure_id,t.perf]).mean().unstack()
    g=g.apply(lambda r: r[1]*w+r[0]*(1-w) if not np.isnan(r[1]) and not np.isnan(r[0]) else np.nan,axis=1)
    return g
def est_boot(a, N, mode, w=0.35, reps=200):
    out=[]
    for _ in range(reps):
        e={}
        for key,idx in grp.items():
            idx=np.array(idx); f=F.loc[idx,a].values; pf=t.loc[idx,"perf"].values
            if mode=="random":
                s=rng.choice(len(idx),size=N,replace=True); e[key]=np.nanmean(f[s])
            else: # stratified N/2 perform, N/2 refuse, reweighted to w
                v=[]
                for c,wt in ((1,w),(0,1-w)):
                    ii=np.where(pf==c)[0]
                    if len(ii)==0: v.append((np.nan,wt)); continue
                    s=rng.choice(ii,size=max(1,N//2),replace=True); v.append((np.nanmean(f[s]),wt))
                e[key]=sum(x*wt for x,wt in v)/sum(wt for x,wt in v if not np.isnan(x))*1 if any(not np.isnan(x) for x,_ in v) else np.nan
                if np.isnan(v[0][0]) or np.isnan(v[1][0]): e[key]=np.nanmean([x for x,_ in v])
        out.append(pair_rho(pd.Series(e).rename_axis(["domain","procedure_id"]), tgt_b))
    return float(np.mean(out)), float(np.std(out))
cheap=["gpt-4.1-mini-mode_fc","gemini-2.0-flash-001-mode_fc","gemini-2.5-flash-mode_fc","gpt-4o-mini-mode_fc","gemma-4-e4b-it-mode_fc","qwen3.5-4b-mode_fc"]
res={"n_proc":int(len(procs))}
res["real_perform_share_vs_b"]=pair_rho(t.groupby(["domain","procedure_id"]).perf.mean(), tgt_b)
res["real_perform_share_vs_b_note"]="procedure-level rho of the real perform share with mean b (what a label-mix-only emulator would reach if the mix were right)"
full={}; 
for a in cheap:
    full[a]={"all_real_cases_raw":pair_rho(est_full(a),tgt_b),
             "all_real_cases_w035":pair_rho(est_full(a,0.35).dropna(),tgt_b.loc[est_full(a,0.35).dropna().index]),
             "N10_random":est_boot(a,10,"random"),"N30_random":est_boot(a,30,"random"),
             "N10_strat_w035":est_boot(a,10,"strat",0.35),"N10_strat_w050":est_boot(a,10,"strat",0.5),"N10_strat_w020":est_boot(a,10,"strat",0.2),
             "N4_random":est_boot(a,4,"random")}
    print(a, full[a], flush=True)
# mean over all 28 agents as the "perfect pilot"
allm=F.mean(axis=1).groupby([t.domain,t.procedure_id]).mean()
res["mean_of_all_28_agents_failure_vs_b"]=pair_rho(allm,tgt_b)
res["by_agent"]=full
json.dump(res,open("ceiling.json","w"),indent=1)
print({k:v for k,v in res.items() if k!="by_agent"})
