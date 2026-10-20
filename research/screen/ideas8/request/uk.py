import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
from screen import *
import json
R=ZR.DATA + "/SOPBench-d2622008/data/"
st,tk,it=load('sopbench')
rows=[]
for dom in ['bank','dmv','healthcare','hotel','library','online_market','university']:
    d=json.load(open(R+dom+'_tasks.json'))
    for proc,L in d.items():
        for k,t in enumerate(L):
            cid=f'{dom}/{proc}/{k}'
            if cid not in it: continue
            args=t['directed_action_graph']['nodes'][0][1]
            uk=t['user_known']
            nodes=t['directed_action_graph']['nodes']
            cargs=set(); 
            for n in nodes[1:]:
                if isinstance(n,list): cargs|=set(n[1].values())
            missing=[a for a in args if a not in uk]
            # constraint args not known to user (must be fetched)
            cfetch=[a for a in cargs if a not in uk]
            rows.append(dict(cid=cid,dom=dom,proc=proc,n_args=len(args),n_known=len(uk),n_missing=len(missing),frac_missing=len(missing)/max(1,len(args)),
               n_cfetch=len(cfetch),n_cargs=len(cargs),extra_known=len([a for a in uk if a not in args]),b=it[cid],lab=int(t['action_should_succeed'])))
import pandas as pd
df=pd.DataFrame(rows)
print(len(df))
for f in ['n_args','n_known','n_missing','frac_missing','n_cfetch','n_cargs','extra_known']:
    per=[spearmanr(g[f],g.b)[0] for _,g in df.groupby('dom') if g[f].std()>0]
    # within proc
    xs=[];ys=[]
    for _,g in df.groupby('proc'):
        if len(g)>=4 and g[f].std()>0: xs+=list((g[f]-g[f].mean())/g[f].std()); ys+=list(g.b-g.b.mean())
    print(f,'dom mean',round(np.mean(per),3),'within-proc',round(spearmanr(xs,ys)[0],3) if xs else None, 'n_within',len(xs))
print(df.groupby('proc').frac_missing.std().gt(0).sum(),'procs with varying frac_missing')
