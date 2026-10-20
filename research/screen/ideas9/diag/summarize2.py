import numpy as np, pandas as pd
res = pd.read_pickle("res.pkl")
def m(d):
    v=[x for x in d.values() if x==x]; return (np.mean(v) if v else np.nan)
rows={}
for r in res: rows.setdefault(r["feat"],{})[r["bench"]]=r
out=[]
for f,bd in rows.items():
    o=dict(feat=f, family=next(iter(bd.values()))["family"], track=next(iter(bd.values()))["track"])
    if "sopbench" in bd:
        r=bd["sopbench"]; w=r["within"]
        vals={d:v for d,v in w.items() if v==v}
        o["sop_nd"]=len(vals); o["sop_raw"]=m(r["raw"]); o["sop_w"]=m(w); o["sop_wl"]=m(r["within_label"]); o["sop_btw"]=m(r["between"])
        o["sop_varw"]=r["var_within"]; o["sop_zero"]=r["zero_within"]; o["sop_z"]=r.get("fe_z",np.nan); o["sop_slope"]=r.get("fe_slope",np.nan)
        o["sop_nullm"]=r["null95_mean"]
        s=np.sign(o["sop_w"]) if o["sop_w"]==o["sop_w"] else 0
        o["sop_agree"]=sum(1 for v in vals.values() if np.sign(v)==s)
        o["sop_worst"]=min(s*v for v in vals.values()) if vals else np.nan
        o["sop_beat"]=sum(1 for d,v in vals.items() if abs(v)>r["null95_dom"][d])
        bv=[v for v in r["between"].values() if v==v]; sb=np.sign(o["sop_btw"]) if o["sop_btw"]==o["sop_btw"] else 0
        o["sop_btw_agree"]=sum(1 for v in bv if np.sign(v)==sb); o["sop_btw_nd"]=len(bv); o["sop_btw_worst"]=min(sb*v for v in bv) if bv else np.nan
        o["sop_btw_z"]=r.get("btw_z",np.nan)
        rv=[v for v in r["raw"].values() if v==v]; sr=np.sign(o["sop_raw"]) if o["sop_raw"]==o["sop_raw"] else 0
        o["sop_raw_agree"]=sum(1 for v in rv if np.sign(v)==sr); o["sop_raw_nd"]=len(rv)
        o["sop_raw_worst"]=min(sr*v for v in rv) if rv else np.nan
    if "tau2" in bd:
        r=bd["tau2"]
        o["t2_raw"]=m(r["raw"]); rv=[v for v in r["raw"].values() if v==v]
        o["t2_nd"]=len(rv)
        o["t2_air"]=r["raw"].get("airline"); o["t2_ret"]=r["raw"].get("retail"); o["t2_tel"]=r["raw"].get("telecom")
        o["t2_telw"]=r["within"].get("telecom"); o["t2_telw_beat"]=(abs(r["within"].get("telecom",np.nan))>r["null95_dom"].get("telecom",np.nan)) if r["within"].get("telecom")==r["within"].get("telecom") else False
        o["t2_telw_z"]=r.get("fe_z",np.nan)
    if "tauk_banking" in bd:
        o["bank_raw"]=m(bd["tauk_banking"]["raw"])
    out.append(o)
df=pd.DataFrame(out)
df.to_pickle("summary2.pkl"); df.to_csv("summary2.csv",index=False)
print(df.shape)
# null scale
r0=[r for r in res if r["bench"]=="sopbench"]
nd=pd.DataFrame([r["null95_dom"] for r in r0 if r["null95_dom"]]); print("null95 |rho| per sop domain (median over features)"); print(nd.median().round(3).to_dict())
print("null95 of mean:", np.nanmedian([r["null95_mean"] for r in r0]).round(3))
t0=[r for r in res if r["bench"]=="tau2"]
print("tau2 telecom null95", np.nanmedian([r["null95_dom"].get("telecom",np.nan) for r in t0]).round(3))
print("sop zero within:", int(df.sop_zero.sum()), "of", df.sop_nd.notna().sum())
print(df[df.sop_zero==True].feat.tolist())
