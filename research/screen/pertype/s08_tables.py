import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import pandas as pd, numpy as np, json
R=ZR.REPO
f=pd.read_csv("flags.csv.gz"); t=pd.read_csv(f"{R}/data/sopbench/tasks.csv"); ty=pd.read_csv("types.csv")
print("runs",len(f),"agents",f.agent.nunique(),"cases",f.case_id.nunique(),"no run",(~f.has_run).sum(),"saved eval",f.has_saved.sum(),"recomputed only",(f.has_run&~f.has_saved).sum())
g=f[f.has_run]; s=g[g.has_saved]
for k in ['dirgraph_satisfied','action_called_correctly','constraint_not_violated','success']:
    print(" agree saved vs recomputed",k,round((s[k].astype(bool)==s['saved_'+k].astype(bool)).mean(),5),"n_disagree",int((s[k].astype(bool)!=s['saved_'+k].astype(bool)).sum()))
m=g.merge(t[["case_id","should_succeed"]],on="case_id")
a=pd.read_csv("sub_abilities.csv",index_col=0)
rows=[]
for ag,x in m.groupby("agent"):
    c=x[x.should_succeed==1]; r=x[x.should_succeed==0]
    th=a.loc[ag,"sop_success"]
    rows.append(dict(agent=ag,carry_checks=c.dirgraph_satisfied.mean(),carry_action=c.action_called_correctly.mean(),carry_success=c.success.mean(),refuse_success=r.success.mean(),overall=x.success.mean(),theta_success=th,theta_checks=a.loc[ag,"sop_checks"],theta_action=a.loc[ag,"sop_action"]))
d=pd.DataFrame(rows).set_index("agent")
items=pd.read_csv(f"{R}/data/sopbench/irt/1d_1pl/items.csv",index_col=0).b
# 1D IRT expected success on carry-out cases
tt=t.set_index("case_id"); co=tt.index[tt.should_succeed==1]
d["irt_expected_carry"]=[np.mean(1/(1+np.exp(-(a.loc[ag,"sop_success"]-items[co].values)))) for ag in d.index]
d["gap_checks_minus_action"]=d.carry_checks-d.carry_action
d["rank_checks"]=d.theta_checks.rank(ascending=False); d["rank_action"]=d.theta_action.rank(ascending=False); d["rank_success"]=d.theta_success.rank(ascending=False)
d["rank_shift"]=d.rank_checks-d.rank_action
d.round(3).to_csv("per_agent_subcheck.csv")
pd.set_option("display.width",250); pd.set_option("display.max_columns",30)
print(d.round(2)[["carry_checks","carry_action","carry_success","refuse_success","irt_expected_carry","rank_checks","rank_action","rank_shift"]].sort_values("carry_action"))
print("mean carry-out checks/action",d.carry_checks.mean(),d.carry_action.mean(), "refuse", d.refuse_success.mean())
print("allow share by type"); 
z=ty.merge(t[["case_id","should_succeed"]],on="case_id")
for c in ["fam","tree","dom"]: print(z.groupby(c).should_succeed.mean().round(2).to_dict())
