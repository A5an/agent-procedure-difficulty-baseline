import json,numpy as np
from common import *
from scipy.stats import spearmanr
from math import comb
rows,tasks,S,N=load_tb(); A,T=S.shape
cells=np.argwhere(N>0); rng=np.random.default_rng(0)
fold=np.zeros(S.shape,int)-1; perm=rng.permutation(len(cells))
for k,idx in enumerate(np.array_split(perm,5)):
    for i,j in cells[idx]: fold[i,j]=k
def smoothed(Str,Ntr,alpha=2.0):
    g=Str.sum()/Ntr.sum()
    a=(Str.sum(1)+alpha*g)/(Ntr.sum(1)+alpha); b=(Str.sum(0)+alpha*g)/(Ntr.sum(0)+alpha)
    lg=lambda x:np.log(x/(1-x))
    return 1/(1+np.exp(-(lg(a)[:,None]+lg(b)[None,:]-lg(g))))
cvp={k:np.zeros(S.shape) for k in ['1pl','shares','2pl','rank2']}
for f in range(5):
    tr=(fold!=f)&(N>0); Str=np.where(tr,S,0);Ntr=np.where(tr,N,0)
    for k in cvp:
        if k=='shares': cvp[k]=np.where(fold==f,smoothed(Str,Ntr),cvp[k])
        else:
            _,p=fit(Str,Ntr,kind=k); cvp[k]=np.where(fold==f,p,cvp[k])
fl={k:[nll(S,N,cvp[k],fold==f) for f in range(5)] for k in cvp}
tot={k:nll(S,N,cvp[k],N>0) for k in cvp}
paired={k:[ (fl[k][f]-fl['1pl'][f]) for f in range(5)] for k in cvp}
# bootstrap over cells for paired difference vs 1pl
def cell_nll(p): 
    p=np.clip(p,1e-6,1-1e-6);return -(S*np.log(p)+(N-S)*np.log(1-p))
boot={}
m=N>0
for k in cvp:
    if k=='1pl':continue
    d=(cell_nll(cvp[k])-cell_nll(cvp['1pl']))[m]; n=N[m]
    bs=[d[ix].sum()/n[ix].sum() for ix in (rng.integers(0,len(d),len(d)) for _ in range(1000))]
    boot[k]=(d.sum()/n.sum(),np.percentile(bs,2.5),np.percentile(bs,97.5))
# Spearman on full data
P,p1=fit(S,N); theta=P['mu'][0]+P['th']; beta=P['be']
ash=S.sum(1)/N.sum(1); tsh=S.sum(0)/N.sum(0)
sp=dict(theta_vs_agent_share=spearmanr(theta,ash)[0],beta_vs_task_share=spearmanr(-beta,tsh)[0],
 agent_share_unweighted=spearmanr(theta,np.where(N>0,S/np.maximum(N,1),np.nan).__array__().mean(1) if False else (S/np.maximum(N,1)).sum(1)/(N>0).sum(1))[0])
# Method 3
def pearson(Sx,Nx,px,mask):
    v=Nx*px*(1-px); return (((Sx-Nx*px)**2)[mask]/np.maximum(v[mask],1e-9)).sum()
mk=N>=2
chi=pearson(S,N,p1,mk); dof=mk.sum()-(A+T)
disp_in=chi/dof
# cross-fitted dispersion (held-out predictions)
disp_cv={k:pearson(S,N,cvp[k],mk)/mk.sum() for k in ['1pl','rank2','2pl']}
P2,p2=fit(S,N,kind='rank2'); disp_r2_in=pearson(S,N,p2,mk)/(mk.sum()-(A+T+2*(A+T)))
# parametric bootstrap null under fitted 1PL: simulate binomial, refit, in-sample ratio and cross-fit ratio
null_in=[];null_cv=[]
r=np.random.default_rng(1)
for b in range(20):
    Sb=r.binomial(N.astype(int),p1).astype(float)
    _,pb=fit(Sb,N); null_in.append(pearson(Sb,N,pb,mk)/dof)
    pc=np.zeros(S.shape)
    for f in range(5):
        tr=(fold!=f)&(N>0); _,pf=fit(np.where(tr,Sb,0),np.where(tr,N,0)); pc=np.where(fold==f,pf,pc)
    null_cv.append(pearson(Sb,N,pc,mk)/mk.sum())
# pass^k with n==5 cells
res3={}
for k in range(1,6):
    sel=(N==5)
    obs=np.mean([comb(int(s),k)/comb(5,k) for s in S[sel]])
    res3[k]=dict(observed=obs,indep_insample=np.mean(p1[sel]**k),indep_crossfit=np.mean(cvp['1pl'][sel]**k),indep_crossfit_rank2=np.mean(cvp['rank2'][sel]**k))
# bootstrap-null for observed pass^k under refit not needed; CI for observed via cell bootstrap
sel=(N==5); ss=S[sel]
for k in range(1,6):
    v=np.array([comb(int(s),k)/comb(5,k) for s in ss]); bs=[v[ix].mean() for ix in (rng.integers(0,len(v),len(v)) for _ in range(1000))]
    res3[k]['obs_ci']=(np.percentile(bs,2.5),np.percentile(bs,97.5))
# beta-binomial reference: dispersion estimate rho from 5-attempt cells (moment): Var(s)=np q (1+(n-1)rho)
phi=disp_cv['1pl']
json.dump(dict(n_cells=int(m.sum()),n_attempts=float(N.sum()),n_n5=int((N==5).sum()),fold_ll=fl,total_ll=tot,paired_fold=paired,boot_vs_1pl=boot,spearman=sp,
 disp=dict(chi=chi,dof=int(dof),insample=disp_in,rank2_insample=disp_r2_in,crossfit=disp_cv,null_insample=(np.mean(null_in),np.std(null_in)),null_crossfit=(np.mean(null_cv),np.std(null_cv))),
 passk=res3),open('m23.json','w'),indent=1,default=float)
print(open('m23.json').read())
