import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import pandas as pd, numpy as np, json
from scipy.special import logit
from scipy.stats import spearmanr, rankdata
from ceil import load
t=load("tau2")
m=pd.read_csv("../ideas9/gap/out/tau2/obs_new_domain.csv.gz")
d=m.drop_duplicates(["seed","fold","case_id"]).assign(l=lambda x:-logit(x.g1_clad_gap.clip(1e-4,1-1e-4))).groupby("case_id").l.mean()
t["pred"]=t.case_id.map(d)
EXT=ZR.DATA + "/tau2-domains/tasks"
idm=json.load(open(ZR.REPO + "/data/tau2/id_map.json"))
inv={v:k for k,v in idm.items()}
gold={}
for dom in ["airline","retail","telecom"]:
    for i,task in enumerate(json.load(open(f"{EXT}/{dom}_tasks.json"))):
        ac=(task.get("evaluation_criteria") or {}).get("actions") or []
        gold[(dom,str(task["id"]))]=(len(ac),sum(1 for a in ac if not a["name"].startswith("get") and not a["name"].startswith("find") and not a["name"].startswith("list") and not a["name"].startswith("search") and not a["name"].startswith("calculate")))
def g(cid):
    k=inv[cid]; dom,tid=k.split(":",1); return gold.get((dom,tid),(np.nan,np.nan))
t["gold_n"]=t.case_id.map(lambda c:g(c)[0]); t["gold_w"]=t.case_id.map(lambda c:g(c)[1])
for dom,gg in t.groupby("domain"):
    print(dom, "rho pred",round(spearmanr(gg.pred,gg.b).correlation,3),"gold_n",round(spearmanr(gg.gold_n,gg.b).correlation,3),"gold_w",round(spearmanr(gg.gold_w,gg.b).correlation,3), "len", round(spearmanr(gg.text.str.len(),gg.b).correlation,3))
    gg=gg.assign(rr=rankdata(gg.b)/len(gg)-rankdata(gg.pred)/len(gg))
    if dom!="telecom":
        print(" most UNDER-predicted (hard, predicted easy):")
        for _,r in gg.sort_values("rr").tail(6).iterrows(): print("   b=%.1f pred=%.2f gold=%s/%s | %s"%(r.b,r.pred,r.gold_n,r.gold_w,r.text.split("Customer scenario:")[1][:600].replace("\n"," ")))
        print(" most OVER-predicted (easy, predicted hard):")
        for _,r in gg.sort_values("rr").head(5).iterrows(): print("   b=%.1f pred=%.2f gold=%s/%s | %s"%(r.b,r.pred,r.gold_n,r.gold_w,r.text.split("Customer scenario:")[1][:600].replace("\n"," ")))
