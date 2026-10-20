import pandas as pd,numpy as np,sys
from sklearn.metrics import roc_auc_score
df=pd.read_csv('data.csv');Rl=df[df.kind=='real'].reset_index(drop=True)
dry=pd.read_csv('../../ideas/dryrun/features/dry_sopbench.csv').set_index('case_id')
Rl['dry']=dry.loc[Rl.case_id,'p_refuse'].values
Rl['dom']=Rl.proc.str.split('/').str[0]
def within(p,key):
    a=[];w=[]
    for _,g in Rl.assign(p=p).groupby(key):
        if g.refuse.nunique()==2: a.append(roc_auc_score(g.refuse,g.p));w.append(len(g))
    return np.average(a,weights=w)
def rep(name,p):
    p=np.asarray(p,float);ok=~np.isnan(p)
    print(f'{name:28s} n={ok.sum()} pooled {roc_auc_score(Rl.refuse[ok],p[ok]):.3f} within-dom {within(p,"dom"):.3f} within-proc {within(p,"proc"):.3f}')
rep('dry zero-shot',Rl.dry)
for m in sys.argv[1:]:
    f=pd.read_csv(f'fs_{m}.csv').set_index('case_id');p=f.loc[Rl.case_id,'p'].values;rep(m,p)
    rep(m+'+dry avg',(np.nan_to_num(p,nan=.5)+Rl.dry.values)/2)
