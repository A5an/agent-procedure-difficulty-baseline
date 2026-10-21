import numpy as np, pandas as pd
pd.set_option("display.width",300,"display.max_rows",500,"display.max_columns",50,"future.no_silent_downcasting",True)
live=pd.read_pickle("live.pkl")
c=["feat","family","track","sop_nd","sop_w","sop_wl","sop_z","sop_agree","sop_beat","sop_worst","sop_varw","sop_raw","t2_raw","t2_telw","t2_telw_z","bank_raw"]
x=live[(live.sop_nd>=3)&(live.sop_varw>=0.02)&(live.sop_z.abs()<1e6)]
print("TOP by |within x label| (sop_wl):"); print(x.sort_values("sop_wl",key=abs,ascending=False)[c].head(25).round(3).to_string())
print("\nADELE (ad.*) all:"); print(live[live.feat.str.startswith("ad.")].sort_values("sop_w",key=abs,ascending=False)[c].round(3).to_string())
print("\nRUBRIC rubp.*:"); print(live[live.feat.str.startswith("rubp.")][c].round(3).to_string())
print("\nPLAN plan3.*:"); print(live[live.feat.str.startswith("plan3.")][c].round(3).to_string())
print("\nCMP cons/cmpp/cp/unit :"); print(live[live.feat.str.contains("^(cons|cmpp|cp|unit)\\.",regex=True)][c].round(3).head(60).to_string())
print("\nact./txt./len/ht/surf:"); print(live[live.feat.str.contains("^(act|txt|len|ht|ent)\\.",regex=True)][c].round(3).to_string())
