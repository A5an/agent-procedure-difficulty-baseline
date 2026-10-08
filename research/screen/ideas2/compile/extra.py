import sys, os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import json, csv, numpy as np
import analyze as an
D=os.path.dirname(os.path.abspath(__file__))+"/"
out=[]
rng=np.random.default_rng(0)
ag=np.nanmean(an.M[an.KEEP],1); bi=an.KEEP[int(np.argmax(ag))]; out.append(f"Best of the 25 real agents: {an.subj[bi]} overall {np.nanmean(an.M[bi]):.3f}")
med=an.KEEP[int(np.argsort(ag)[len(ag)//2])]; out.append(f"Median of the 25: {an.subj[med]} {np.nanmean(an.M[med]):.3f}")
out.append(f"Always-refuse baseline (call nothing): success = share of refuse cases = {(an.should==0).mean():.3f}")
for name in ["results_A.json","results_B.json","results_AC.json","results_BC.json"]:
    d=an.load(name); c=np.array([float(d[x]["success"]) for x in an.ids])
    for tag,j in [("best",bi),("median",med)]:
        a=an.M[j]; ok=~np.isnan(a); diff=(c-a)[ok]
        bs=[diff[rng.integers(0,len(diff),len(diff))].mean() for _ in range(2000)]
        out.append(f"{name} minus {tag} agent: {diff.mean():+.3f} [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}] (paired bootstrap over cases)")
    # by perform/refuse vs best
    for lab,m in [("perform",an.should==1),("refuse",an.should==0)]:
        out.append(f"   {lab}: compiled {c[m].mean():.3f}, best agent {np.nanmean(an.M[bi][m]):.3f}")
# per procedure table
dA=an.load("results_A.json"); dC=an.load("results_AC.json"); dBC=an.load("results_BC.json")
ps=sorted(set(an.proc)); rows=[]
for p in ps:
    m=an.proc==p; cid=[x for x,mm in zip(an.ids,m) if mm]
    rows.append([p,m.sum(),int(an.should[m].sum()),round(np.mean([dA[x]["success"] for x in cid]),3),round(np.mean([dC[x]["success"] for x in cid]),3),round(np.mean([dBC[x]["success"] for x in cid]),3),round(float(np.nanmean(an.M[an.KEEP][:,m])),3),round(float(np.nanmean(an.M[:,m])),3)])
with open(D+"procedure_table.csv","w") as f:
    w=csv.writer(f); w.writerow(["procedure","n_cases","n_perform","compiled_base_A","compiled_C_A","compiled_C_B","agents25_mean","agents28_mean"]); w.writerows(rows)
cp=np.array([r[4] for r in rows]); a=np.array([r[6] for r in rows])
out.append("Procedures where compiled(C,A) is below the agent mean: "+", ".join(f"{r[0]} ({r[4]} vs {r[6]})" for r in rows if r[4]<r[6]))
out.append("Procedures with compiled(C,A) success below 0.6: "+", ".join(f"{r[0]} n={r[1]} ({r[4]})" for r in rows if r[4]<0.6))
out.append(f"Procedures with all cases passed (C,A): {sum(1 for r in rows if r[4]==1.0)} of 70; compiled success >=0.8: {sum(1 for r in rows if r[4]>=0.8)}")
open(D+"extra_out.md","w").write("\n".join(out)); print("\n".join(out))
