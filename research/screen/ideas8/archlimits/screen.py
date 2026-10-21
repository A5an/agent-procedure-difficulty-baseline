import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, re, json, glob
sys.path.insert(0,ZR.SCREEN + "/ideas")
from cpulock import cpu_lock
import pandas as pd, numpy as np
from scipy.stats import spearmanr
R=ZR.REPO + "/data/"
P=ZR.DATA + "/tau2-domains/tasks/"
STOP=set("the and for you that with this your have will from are not can any all about would like want please hello need what when which their there been into more than also just only some then them they does being".split())
def words(s): return [w for w in re.findall(r"[a-z]{4,}",s.lower()) if w not in STOP]
def feats(msg,pol):
    digs=sum(c.isdigit() for c in msg)
    ids=[t for t in re.findall(r"[A-Za-z0-9_@.\-]{5,}",msg) if any(c.isdigit() for c in t)]
    nums=set(re.findall(r"\d+(?:\.\d+)?",msg))
    w=words(msg); pw=set(words(pol))
    ov=np.mean([x in pw for x in w]) if w else np.nan
    return dict(digits=digs,idtok=len(ids),idlen=sum(len(t) for t in ids),nums=len(nums),lexov=ov,nwords=len(w))
out=[]
with cpu_lock("archlimits"):
  for b in ["sopbench","tau2"]:
    st=pd.read_json(R+b+"/statements.jsonl",lines=True).set_index("case_id")
    t=pd.read_csv(R+b+"/tasks.csv").set_index("case_id")
    it=pd.read_csv(R+b+"/irt/1d_1pl/items.csv",index_col=0)
    rows=[]
    pols={}
    if b=="tau2":
        for d,f in [("airline","airline_policy.md"),("retail","retail_policy.md"),("telecom","telecom_main_policy.md")]:
            try: pols[d]=open(P+f).read()
            except Exception as e: pols[d]=""
    for cid,r in st.iterrows():
        txt=r["text"]
        if b=="sopbench":
            m=re.search(r"Policy for this action:\n(.*?)\nCustomer message:(.*)",txt,re.S)
            pol,msg=(m.group(1),m.group(2)) if m else ("",txt)
        else:
            pol=pols.get(t.loc[cid,"domain"],""); msg=txt
        f=feats(msg,pol); f.update(case=cid,dom=t.loc[cid,"domain"],proc=t.loc[cid,"procedure_id"],beta=it.loc[cid,"b"],bench=b)
        rows.append(f)
    out.append(pd.DataFrame(rows))
d=pd.concat(out)
def rho(g,c): 
    g=g.dropna(subset=[c]); 
    return spearmanr(g[c],g["beta"])[0] if g[c].nunique()>1 and len(g)>5 else np.nan
for c in ["digits","idtok","idlen","nums","lexov","nwords"]:
    d[c+"_pc"]=d[c]-d.groupby("proc")[c].transform("mean")
    print("\n==",c)
    for b,g in d.groupby("bench"):
        r1=g.groupby("dom").apply(lambda x:rho(x,c)); r2=g.groupby("dom").apply(lambda x:rho(x,c+"_pc"))
        print(b,"raw by domain",r1.round(3).to_dict(),"mean",round(r1.mean(),3),"| within-proc",r2.round(3).to_dict(),"mean",round(r2.mean(),3))
