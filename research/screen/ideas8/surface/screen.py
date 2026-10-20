import os
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, re, lzma, zlib
sys.path.insert(0,ZR.SCREEN + "/ideas")
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from cpulock import cpu_lock
from tokenizers import Tokenizer
R=ZR.REPO + "/data/"
import glob
tk=Tokenizer.from_file(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/*/tokenizer.json"))[0])
P=ZR.DATA + "/tau2-domains/tasks/"
pol={"airline":open(P+"airline_policy.md").read(),"retail":open(P+"retail_policy.md").read(),"telecom":open(P+"telecom_main_policy.md").read()}
def lz(s): return len(lzma.compress(s.encode(),preset=6))
def zl(s): return len(zlib.compress(s.encode(),9))
def words(s): return re.findall(r"[A-Za-z']+",s.lower())
STOP=set("the a an of to and or in on for is are be with that this it as at by from your you i my me we our not must have has can will if any all".split())
def feats(policy,req):
    w=words(req); cw=set(x for x in w if x not in STOP); pw=set(x for x in words(policy) if x not in STOP)
    ids=tk.encode(req).ids; ntok=len(ids)
    toks=[tk.decode([i]) for i in ids]
    dig=sum(1 for t in toks if re.search(r"\d",t))
    nums=re.findall(r"\d[\d,\.]*",req)
    idl=re.findall(r"\b\w*[_\d]\w*\b",req)
    lp,lr,lpr=lz(policy),lz(req),lz(policy+"\n"+req)
    return dict(
      icomp_lzma=(lpr-lp)/max(lr,1), icomp_abs=lpr-lp, icomp_zlib=(zl(policy+"\n"+req)-zl(policy))/max(zl(req),1),
      comp_req=lr/max(len(req.encode()),1),
      fert=ntok/max(len(w),1), ntok=ntok, ndigtok=dig, fdigtok=dig/max(ntok,1), nnum=len(nums), nid=len(idl),
      ovl=len(cw&pw)/max(len(cw),1), novl=len(cw-pw), ncw=len(cw))
rows=[]
for b in ["sopbench","tau2"]:
    t=pd.read_csv(R+b+"/tasks.csv"); it=pd.read_csv(R+b+"/irt/1d_1pl/items.csv",index_col=0)
    for l in open(R+b+"/statements.jsonl"):
        r=json.loads(l); cid=r["case_id"]; x=r["text"]; d=t.set_index("case_id").loc[cid]
        if b=="sopbench":
            m=re.search(r"Policy for this action:\n(.*?)\nCustomer message: (.*?)\nAvailable tools",x,re.S)
            policy,req=m.group(1),m.group(2)
        else:
            policy=pol[d.domain]; req=x
        f=feats(policy,req); f.update(bench=b,domain=d.domain,case=cid,beta=it.loc[cid,"b"],ss=d.get("should_succeed",np.nan)); rows.append(f)
df=pd.DataFrame(rows); df.to_csv("features.csv",index=False)
fc=[c for c in df.columns if c not in("bench","domain","case","beta","ss")]
out=[]
for c in fc:
    for b in ["sopbench","tau2"]:
        rs=[];ws=[];rss=[]
        for dom,g in df[df.bench==b].groupby("domain"):
            r=spearmanr(g[c],g.beta)[0]; rs.append(r); ws.append(len(g))
            if b=="sopbench":
                # within should_succeed strata
                rr=[spearmanr(h[c],h.beta)[0] for _,h in g.groupby("ss") if len(h)>10 and h[c].nunique()>1]; rss.append(np.mean(rr))
        out.append(dict(feat=c,bench=b,mean_rho=np.nanmean(rs),wmean=np.average(np.nan_to_num(rs),weights=ws),per_domain=" ".join(f"{x:+.2f}" for x in rs),within_label=np.nanmean(rss) if rss else np.nan))
o=pd.DataFrame(out); pd.set_option("display.width",200); print(o.round(3).to_string())
o.to_csv("screen_results.csv",index=False)
