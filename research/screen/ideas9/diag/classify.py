import numpy as np, pandas as pd
pd.set_option("display.width",300,"display.max_rows",500,"display.max_columns",50)
df=pd.read_pickle("summary2.pkl")
s=df[df.sop_nd.notna() & ~df.sop_zero.fillna(True)].copy()
s["w_pass"]=(s.sop_w.abs()>s.sop_nullm)&(s.sop_z.abs()>=3.0)&(s.sop_beat>=3)&(s.sop_agree>=s.sop_nd-1)&(s.sop_worst>-0.05)
cols=["feat","family","track","sop_w","sop_wl","sop_z","sop_agree","sop_nd","sop_beat","sop_worst","sop_varw","sop_raw","sop_btw","t2_raw","t2_air","t2_ret","t2_tel","t2_telw","t2_telw_beat","bank_raw"]
print("within-signal candidates (SOPBench):",s.w_pass.sum())
print(s[s.w_pass].sort_values("sop_worst",ascending=False)[cols].round(3).to_string())
print("\nTop |sop_w| not passing:")
print(s[~s.w_pass].sort_values("sop_w",key=abs,ascending=False)[cols].head(25).round(3).to_string())
print("\nsop within |z|>=3 count", (s.sop_z.abs()>=3).sum(), "of", len(s), "; expected by chance at 475 tests ~", round(len(s)*0.0027,1))
