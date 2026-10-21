import numpy as np, pandas as pd
res = pd.read_pickle("res.pkl"); log = pd.read_pickle("log.pkl")
for b,(g,s) in log.items():
    print(b, "loaded", len(g), "skipped", s)
def mean(d): 
    v=[x for x in d.values() if x==x]; return np.mean(v) if v else np.nan
rows={}
for r in res:
    rows.setdefault(r["feat"],{})[r["bench"]]=r
out=[]
for f,bd in rows.items():
    o=dict(feat=f, family=next(iter(bd.values()))["family"], track=next(iter(bd.values()))["track"])
    allw={}; allraw={}; beat=0; test=0
    for b in ["sopbench","tau2"]:
        if b not in bd: continue
        r=bd[b]
        o[f"{b}_raw"]=mean(r["raw"]); o[f"{b}_within"]=mean(r["within"]); o[f"{b}_btw"]=mean(r["between"])
        o[f"{b}_varw"]=r["var_within"]; o[f"{b}_z"]=r.get("fe_z",np.nan); o[f"{b}_zero"]=r["zero_within"]
        o[f"{b}_null95"]=r["null95_mean"]
        o[f"{b}_beatmean"]= (abs(o[f"{b}_within"])>r["null95_mean"]) if r["null95_mean"]==r["null95_mean"] and o[f"{b}_within"]==o[f"{b}_within"] else False
        for d,v in r["within"].items():
            if v==v:
                allw[(b,d)]=v; test+=1
                nd=r["null95_dom"][d]
                if nd==nd and abs(v)>nd: beat+=1
        for d,v in r["raw"].items():
            if v==v: allraw[(b,d)]=v
        if b=="sopbench":
            o["sop_within_label"]=mean(r["within_label"])
    if "tauk_banking" in bd: o["bank_raw"]=mean(bd["tauk_banking"]["raw"])
    pooled=np.mean(list(allw.values())) if allw else np.nan
    praw=np.mean(list(allraw.values())) if allraw else np.nan
    o["w_mean"]=pooled; o["raw_mean"]=praw
    sgn=np.sign(pooled) if pooled==pooled else 0
    o["n_dom"]=len(allw); o["sign_agree_w"]=sum(1 for v in allw.values() if np.sign(v)==sgn)
    o["worst_w"]=min(sgn*v for v in allw.values()) if allw else np.nan
    sr=np.sign(praw) if praw==praw else 0
    o["sign_agree_raw"]=sum(1 for v in allraw.values() if np.sign(v)==sr); o["n_dom_raw"]=len(allraw)
    o["worst_raw"]=min(sr*v for v in allraw.values()) if allraw else np.nan
    o["beat_dom"]=beat; o["test_dom"]=test
    out.append(o)
df=pd.DataFrame(out); df.to_pickle("summary.pkl"); df.to_csv("summary.csv",index=False)
print(len(df))
pd.set_option("display.width",250,"display.max_rows",500,"display.max_columns",40)
print(df.sort_values("w_mean",key=abs,ascending=False)[["feat","family","track","n_dom","w_mean","sign_agree_w","worst_w","beat_dom","sopbench_within","tau2_within","sopbench_z","tau2_z","sopbench_varw","tau2_varw","raw_mean"]].head(60).round(3).to_string())
