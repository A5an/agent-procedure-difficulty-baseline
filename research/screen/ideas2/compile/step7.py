import sys, os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import json, collections, numpy as np, sop, common, execu
from scipy.stats import spearmanr
import analyze as an
D=os.path.dirname(os.path.abspath(__file__))+"/"
files=["programs_t0.json","programs_t1_s1.json","programs_t1_s2.json"]
progs=[json.load(open(D+f)) for f in files]
cs=sop.load_cases()
dec={};succ={}
for t in cs:
    k=common.key_of(t); ds=[];ss=[]
    for p in progs:
        ev,log,err=execu.execute(p[k]["code"],t["user_known"],t)
        ds.append(int(any(l["tool_name"]==t["user_goal"] for l in log))); ss.append(int(ev["success"]))
    dec[t["case_id"]]=ds; succ[t["case_id"]]=ss
json.dump({"dec":dec,"succ":succ},open(D+"step7_raw.json","w"))
out=[]
dis=np.array([len(set(dec[c]))>1 for c in an.ids]); sdis=np.array([len(set(succ[c]))>1 for c in an.ids])
out.append(f"Cases where the 3 base-prompt programs (t0, two samples at t=1) disagree on whether to call the target action: {dis.mean():.3f} ({dis.sum()}/830); disagree on success: {sdis.mean():.3f}")
succ_mean=np.array([np.mean(succ[c]) for c in an.ids]); 
out.append("Mean success of the three programs: "+", ".join(f"{np.mean([succ[c][i] for c in an.ids]):.3f}" for i in range(3)))
# programs agree but wrong?
agree_wrong=np.mean([(len(set(succ[c]))==1 and succ[c][0]==0) for c in an.ids]); out.append(f"All three agree and all fail: {agree_wrong:.3f}; all three agree and succeed: {np.mean([(len(set(succ[c]))==1 and succ[c][0]==1) for c in an.ids]):.3f}")
ps=sorted(set(an.proc)); d_p=[];a_p=[];s_p=[];n=[]
for p in ps:
    m=an.proc==p
    d_p.append(dis[m].mean()); a_p.append(1-np.nanmean(an.M[an.KEEP][:,m])); s_p.append(1-succ_mean[m].mean()); n.append(m.sum())
d_p=np.array(d_p);a_p=np.array(a_p);s_p=np.array(s_p);n=np.array(n)
r=spearmanr(d_p,a_p); out.append(f"Per procedure (70): Spearman(program disagreement rate, agent difficulty = 1 - mean success of 25 real agents) = {r[0]:.2f} (p={r[1]:.3f})")
r=spearmanr(s_p,a_p); out.append(f"Per procedure: Spearman(program failure rate, agent difficulty) = {r[0]:.2f} (p={r[1]:.3f})")
m5=n>=5; r=spearmanr(d_p[m5],a_p[m5]); out.append(f"Procedures with >=5 cases ({m5.sum()}): disagreement vs agent difficulty rho={r[0]:.2f} (p={r[1]:.3f})")
# case level: disagreement vs agent success within procedure? case-level agent difficulty
cd=1-np.nanmean(an.M[an.KEEP],0); r=spearmanr(dis.astype(float),cd); out.append(f"Case level (830): Spearman(disagreement, agent difficulty) = {r[0]:.2f} (p={r[1]:.3f})")
# case-level within procedure demeaned
res=[]; 
for p in ps:
    m=np.where(an.proc==p)[0]
    if len(m)>2: res.append((dis[m]-dis[m].mean(), cd[m]-cd[m].mean()))
x=np.concatenate([a for a,b in res]); y=np.concatenate([b for a,b in res]); r=spearmanr(x,y); out.append(f"Case level within procedure (demeaned): rho={r[0]:.2f} (p={r[1]:.3f})")
open(D+"step7_out.md","w").write("\n".join(out)); print("\n".join(out))
