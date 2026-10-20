import numpy as np, pandas as pd
pd.set_option("display.width",320,"display.max_rows",500,"display.max_columns",50,"future.no_silent_downcasting",True)
df=pd.read_pickle("summary2.pkl")
sop=df[df.sop_nd.notna()].copy()
sop["zero"]=sop.sop_zero.astype(bool)
live=sop[~sop.zero].copy()
A=live
A["w_pass"]=(A.sop_nd>=5)&(A.sop_varw>=0.05)&(A.sop_w.abs()>A.sop_nullm)&(A.sop_z.abs()>=3.5)&(A.sop_z.abs()<1e6)&(A.sop_beat>=3)&(A.sop_agree>=A.sop_nd-1)&(A.sop_worst>-0.05)
sg=np.sign(A.sop_w)
A["t2_same"]=np.where(A.t2_raw.notna(), np.sign(A.t2_raw)==sg, np.nan)
A["tel_same"]=np.where(A.t2_telw.notna(), np.sign(A.t2_telw)==sg, np.nan)
A["bank_same"]=np.where(A.bank_raw.notna(), np.sign(A.bank_raw)==sg, np.nan)
cand=A[A.w_pass].copy()
cand["key"]=cand.sop_w.round(4).astype(str)+"|"+cand.sop_z.round(2).astype(str)
cols=["feat","family","track","sop_w","sop_wl","sop_z","sop_agree","sop_nd","sop_beat","sop_worst","sop_varw","sop_raw","t2_raw","t2_telw","bank_raw","t2_same","tel_same","bank_same"]
print("CASE-LEVEL CANDIDATES (sop within pass):",len(cand))
print(cand.sort_values("sop_worst",ascending=False)[cols].round(3).to_string())
# procedure-level
B=live.copy()
B["btw_pass"]=(B.sop_btw.abs()>=0.25)&(B.sop_btw_agree>=B.sop_btw_nd-2)&(B.sop_btw_nd>=5)
B["inner"]=B.w_pass
pl=B[B.btw_pass & ~B.w_pass].copy()
bc=["feat","family","track","sop_btw","sop_btw_agree","sop_btw_nd","sop_btw_z","sop_btw_worst","sop_w","sop_varw","sop_raw","t2_raw","bank_raw"]
print("\nPROCEDURE-LEVEL (between strong, within not passing):",len(pl))
print(pl.sort_values("sop_btw",key=abs,ascending=False)[bc].round(3).head(45).to_string())
# raw tau2 strong
T=df[df.t2_raw.notna()].copy()
print("\nTAU2 raw top |rho| (3 domains):")
T["t2_agree"]=[sum(np.sign(x)==np.sign(r) for x in (a,b_,c) if x==x) for r,a,b_,c in zip(T.t2_raw,T.t2_air,T.t2_ret,T.t2_tel)]
print(T.sort_values("t2_raw",key=abs,ascending=False)[["feat","family","track","t2_raw","t2_air","t2_ret","t2_tel","t2_agree","t2_telw","sop_raw","sop_w","bank_raw"]].head(40).round(3).to_string())
# dead
dead=live[(live.sop_w.abs()<live.sop_nullm)&((live.sop_btw.abs()<0.15)|live.sop_btw.isna())&((live.t2_raw.abs()<0.15)|live.t2_raw.isna())]
print("\nDEAD count",len(dead),"of",len(df)); print(dead.groupby("family").feat.apply(lambda s:", ".join(s.str.split(".").str[-1]+"("+s.str.split(".").str[0]+")")).to_string()[:6000])
print("\nZERO within-procedure variance on SOPBench:",int(sop.zero.sum()))
print(sop[sop.zero].groupby("family").feat.apply(list).to_string())
live.to_pickle("live.pkl"); cand.to_pickle("cand.pkl")
# family-level counts
fam=live.groupby("family").agg(n=("feat","size"),beat_mean=("sop_w",lambda s:int((s.abs()>live.loc[s.index,"sop_nullm"]).sum())),z3=("sop_z",lambda s:int(((s.abs()>=3.5)&(s.abs()<1e6)).sum())),max_abs_w=("sop_w",lambda s:s.abs().max()),mean_abs_w=("sop_w",lambda s:s.abs().mean()),mean_abs_wl=("sop_wl",lambda s:s.abs().mean()),mean_abs_btw=("sop_btw",lambda s:s.abs().mean()))
print(fam.round(3).to_string())
