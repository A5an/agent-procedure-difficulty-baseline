import numpy as np, pandas as pd, sys
from scipy.stats import spearmanr
sys.path.insert(0,".")
import a5
b = pd.read_csv(a5.REPO/"data/tau2/irt/1d_1pl/items.csv", index_col=0)["b"]; T=a5.TASKS["tau2"]
ms=["adele","A","A_adele","c_lad","c_lad_A","A_dm","A_w","A_all","A_len","A_raw"]
rows=[]
for sch in ["new_procedures","new_domain"]:
    be=pd.read_csv(f"out_tau2/tau2/betas_{sch}.csv"); be["dom"]=be.case_id.map(T.domain); be["b"]=be.case_id.map(b)
    for dom in ["airline","retail","telecom"]:
        r={"scheme":sch,"dom":dom}
        for m in ms:
            x=be[(be.method==m)&(be.dom==dom)]
            r[m]=np.mean([spearmanr(g.beta_hat,g.b)[0] for _,g in x.groupby("seed")])
        rows.append(r)
    # excluding telecom: size-weighted mean of airline, retail pairs
    for lab,ds in [("no_telecom",["airline","retail"]),("telecom_only",["telecom"])]:
        w=np.array([ (T.domain==d).sum()**2 for d in ds],float); sub=[r for r in rows if r["scheme"]==sch and r["dom"] in ds]
        rows.append({"scheme":sch,"dom":lab,**{m:float(np.average([r[m] for r in sub],weights=w)) for m in ms}})
R=pd.DataFrame(rows).round(3); pd.set_option("display.width",250); print(R.to_string()); R.to_csv("final_model_domain_rho.csv",index=False)
