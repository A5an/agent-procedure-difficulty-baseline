import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import os, sys, json; os.environ["NOB"]="1"
sys.argv=["x","none"]
S = ZR.SCREEN
exec(open("run.py").read().split('if mode == "pre"')[0].replace("mode = sys.argv[1]","mode='x'").replace("sys.argv = [sys.argv[0], \"main\"]",""))
import numpy as np
rows=[]
for scheme in ["new_procedures","new_domain"]:
    for tr,te in folds(scheme):
        A=fs.stage_a(tr,te)
        for c in te:
            g=fs.gold_stats(fs._GOLD[c],fs.DOM[c]); rows.append(dict(scheme=scheme,dom=fs.DOM[c],A_len=A.loc[c,"A_len"],A_w=A.loc[c,"A_writes"],A_all=A.loc[c,"A_alllen"],Z=fs.PLAN3.loc[c,"plan_calls"],Zw=fs.PLAN3.loc[c,"plan_writes"],gl=g[0],gw=g[1],ga=g[2]))
d=pd.DataFrame(rows); res={}
def r(x,y): return None if x.nunique()<2 or y.nunique()<2 else round(float(spearmanr(x,y)[0]),3)
for sch,d0 in d.groupby("scheme"):
    for dom,x in list(d0.groupby("dom"))+[("all",d0)]:
        res[f"{sch}|{dom}"]=dict(n=len(x),zero_calls_vs_len=r(x.Z,x.gl),A_len_vs_len=r(x.A_len,x.gl),zero_writes_vs_writes=r(x.Zw,x.gw),A_w_vs_writes=r(x.A_w,x.gw),A_all_vs_all=r(x.A_all,x.ga),mean_gold=round(x.gl.mean(),2),mean_zero=round(x.Z.mean(),2),mean_A=round(x.A_len.mean(),2))
json.dump(res,open("validation_A.json","w"),indent=1)
for k,v in res.items(): print(k,v)
# paired bootstrap A vs c_lad on tau2 betas against IRT difficulty
b=pd.read_csv(fs.REPO/"data/tau2/irt/1d_1pl/items.csv",index_col=0)["b"]
rng=np.random.RandomState(0)
for sch in ["new_procedures","new_domain"]:
    be=pd.read_csv(f"out_A/tau2/betas_{sch}.csv"); P=be.pivot_table(index="case_id",columns="method",values="beta_hat")
    P["b"]=b.reindex(P.index); P["dom"]=[fs.DOM[c] for c in P.index]; P["proc"]=[fs.PROC[c] for c in P.index]
    procs=P.proc.unique()
    def rho(Q,m): return np.mean([spearmanr(-Q[m][Q.dom==dm],Q.b[Q.dom==dm])[0] if False else spearmanr(Q[m][Q.dom==dm],Q.b[Q.dom==dm])[0] for dm in Q.dom.unique() if (Q.dom==dm).sum()>3])
    for m in ["A","A_adele","c_lad_A"]:
        diffs=[]
        for _ in range(300):
            ps=rng.choice(procs,len(procs)); Q=pd.concat([P[P.proc==p] for p in ps])
            diffs.append(rho(Q,m)-rho(Q,"c_lad"))
        print(sch,m,"minus c_lad point",round(rho(P,m)-rho(P,"c_lad"),3),"CI",np.round(np.percentile(diffs,[2.5,97.5]),3))
