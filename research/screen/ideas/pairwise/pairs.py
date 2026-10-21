"""Fixed pair design (seed 20261005). Writes pairs.csv: pair_id,bench,domain,a,b,kind,repeat_of."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, random
import pandas as pd
REPO=ZR.REPO
SEED=20261005
rng=random.Random(SEED)
edges={}  # (min,max)->kind
def add(u,v,kind,E):
    if u==v: return False
    k=(min(u,v),max(u,v))
    if k in E: return False
    E[k]=kind; return True
def ring(nodes,deg_rounds,kind,E,ok=lambda u,v:True):
    """deg_rounds rounds of a shuffled ring, skipping disallowed/duplicate pairs (each round adds up to 2 per node)"""
    for _ in range(deg_rounds):
        o=nodes[:]; rng.shuffle(o)
        for i in range(len(o)):
            u,v=o[i],o[(i+1)%len(o)]
            if ok(u,v): add(u,v,kind,E)
rows=[]
for bench in ["sopbench","tau2"]:
    t=pd.read_csv(f"{REPO}/data/{bench}/tasks.csv")
    for dom,g in t.groupby("domain"):
        E={}
        ids=list(g.case_id); proc=dict(zip(g.case_id,g.procedure_id))
        if bench=="sopbench":
            for p,gp in g.groupby("procedure_id"):
                n=list(gp.case_id)
                if len(n)<=5:
                    for i in range(len(n)):
                        for j in range(i+1,len(n)): add(n[i],n[j],"same",E)
                else:
                    ring(n,2,"same",E)  # about 4 same-procedure partners
            # cross-procedure: fill degree up to 8 with ring rounds across procedures
            for _ in range(6):
                deg={c:0 for c in ids}
                for (u,v) in E: deg[u]+=1; deg[v]+=1
                need=[c for c in ids if deg[c]<8]
                if not need: break
                o=need[:]; rng.shuffle(o)
                # pair each needy case with a random different-procedure case of low degree
                for u in o:
                    if deg[u]>=8: continue
                    cand=[c for c in ids if proc[c]!=proc[u] and deg[c]<8 and (min(u,c),max(u,c)) not in E]
                    if not cand: continue
                    v=rng.choice(cand)
                    add(u,v,"cross",E); deg[u]+=1; deg[v]+=1
        else:
            ring(ids,4,"random",E)
        for (u,v),k in sorted(E.items()):
            a,b=(u,v) if rng.random()<0.5 else (v,u)
            rows.append(dict(bench=bench,domain=dom,a=a,b=b,kind=k,repeat_of=""))
df=pd.DataFrame(rows); df.insert(0,"pair_id",range(len(df)))
# swapped-order repeat on a random 1000 pairs (A/B swapped): reliability and position bias
rep=df.sample(1000,random_state=SEED).copy()
rep=rep.rename(columns={"a":"b","b":"a"}); rep["repeat_of"]=rep.pair_id; rep["pair_id"]=range(len(df),len(df)+len(rep))
out=pd.concat([df,rep[df.columns]])
out.to_csv("pairs.csv",index=False)
print(len(df),len(rep)); print(df.groupby(["bench","kind"]).size())
for bench in ["sopbench","tau2"]:
    d=df[df.bench==bench]; c=pd.concat([d.a,d.b]).value_counts(); print(bench,"degree min/mean/max",c.min(),c.mean(),c.max())
