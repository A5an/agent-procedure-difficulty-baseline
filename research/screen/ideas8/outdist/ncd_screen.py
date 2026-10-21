import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json,csv,zlib,sys,numpy as np
from scipy.stats import spearmanr
sys.path.insert(0,ZR.SCREEN + "/ideas")
from cpulock import cpu_lock
R=ZR.REPO + "/data/"
C=lambda s:len(zlib.compress(s.encode(),9))
def load(b):
    st={}
    for l in open(R+b+'/statements.jsonl'):
        r=json.loads(l);st[r['case_id']]=r['text']
    it={r['']:float(r['b']) for r in csv.DictReader(open(R+b+'/irt/1d_1pl/items.csv'))}
    tk=list(csv.DictReader(open(R+b+'/tasks.csv')))
    return st,it,tk
def msg(t):
    return t.split('Customer message:',1)[1] if 'Customer message:' in t else t
with cpu_lock("outdist"):
  for b in ['sopbench','tau2','tauk_banking']:
    st,it,tk=load(b)
    rows=[]
    byp={}
    for r in tk: byp.setdefault(r['procedure_id'],[]).append(r['case_id'])
    for p,ids in byp.items():
        for c in ids:
            if c not in it: continue
            x=msg(st[c]); sib=[msg(st[o]) for o in ids if o!=c][:40]
            ctx="\n".join(sib)
            cond=(C(ctx+"\n"+x)-C(ctx))/max(1,C(x))   # conditional compressibility: high=atypical
            rows.append((next(r['domain'] for r in tk if r['case_id']==c),p,c,cond,len(x),it[c]))
    doms=sorted(set(r[0] for r in rows))
    out=[]
    for d in doms:
        s=[r for r in rows if r[0]==d]
        if len(s)<20: continue
        f=np.array([r[3] for r in s]);L=np.array([r[4] for r in s]);y=np.array([r[5] for r in s])
        # within-procedure centered
        fc=f.copy()
        for p in set(r[1] for r in s):
            m=np.array([r[1]==p for r in s]);fc[m]-=f[m].mean()
        out.append((d,len(s),spearmanr(f,y)[0],spearmanr(fc,y)[0],spearmanr(L,y)[0]))
    print(b)
    for o in out: print("  %-18s n=%3d rho_ncd=%+.3f rho_ncd_centered=%+.3f rho_len=%+.3f"%o)
    if out: print("  mean ncd %.3f centered %.3f len %.3f"%tuple(np.mean([o[i] for o in out]) for i in (2,3,4)))
