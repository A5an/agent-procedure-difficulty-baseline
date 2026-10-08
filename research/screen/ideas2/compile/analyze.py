import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, os, numpy as np, collections
from scipy.stats import spearmanr
HERE=os.path.dirname(os.path.abspath(__file__))
R=ZR.REPO + "/data/sopbench/"
rs=[json.loads(l) for l in open(R+"responses.jsonl")]
ids=list(rs[0]["responses"]); idx={c:i for i,c in enumerate(ids)}
M=np.array([[r["responses"].get(c,np.nan) for c in ids] for r in rs],float)  # 28 x 830
subj=[r["subject_id"] for r in rs]
EMPTY=[i for i,s in enumerate(subj) if s.split("-mode")[0] in ("gpt-5","gpt-5-mini","gemini-2.5-pro")]
KEEP=[i for i in range(28) if i not in EMPTY]
import csv
tasks={r["case_id"]:r for r in csv.DictReader(open(R+"tasks.csv"))}
dom=np.array([tasks[c]["domain"] for c in ids]); proc=np.array([tasks[c]["procedure_id"] for c in ids]); should=np.array([int(tasks[c]["should_succeed"]) for c in ids])
def load(name):
    rows=json.load(open(os.path.join(HERE,name))); d={r["case_id"]:r for r in rows}
    return d
def summarize(res,label,out):
    d=load(res); mask=np.array([c in d for c in ids]); 
    comp=np.array([float(d[c]["success"]) if c in d else np.nan for c in ids])
    out.append(f"\n### {label} (n={mask.sum()})")
    def line(name,m):
        c=np.nanmean(comp[m]); ag=np.nanmean(M[:,m],1)
        return f"| {name} | {m.sum()} | {c:.3f} | {ag[KEEP].max():.3f} | {np.median(ag[KEEP]):.3f} | {np.median(ag):.3f} | {ag.max():.3f} | {(ag[KEEP]<c).sum()}/25 | {(ag<c).sum()}/28 |"
    out.append("| subset | n | compiled | best agent (25 real) | median (25) | median (28) | best (28) | agents below compiled (25) | (28) |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    out.append(line("all",mask))
    for dname in sorted(set(dom)): out.append(line(dname,mask&(dom==dname)))
    out.append(line("perform (should succeed)",mask&(should==1))); out.append(line("refuse",mask&(should==0)))
    # per procedure
    ps=sorted(set(proc)); cp=[];ap=[];ap25=[]
    for p in ps:
        m=mask&(proc==p)
        if m.sum()==0: continue
        cp.append(np.nanmean(comp[m])); ap.append(np.nanmean(M[:,m])); ap25.append(np.nanmean(M[KEEP][:,m]))
    cp=np.array(cp);ap=np.array(ap);ap25=np.array(ap25)
    r1=spearmanr(cp,ap25); r2=spearmanr(cp,ap)
    out.append(f"\nPer procedure ({len(cp)}): compiled success above agent mean (25) on {int((cp>ap25).sum())}, equal {int((cp==ap25).sum())}, below {int((cp<ap25).sum())}. Mean procedure success: compiled {cp.mean():.3f}, agents(25) {ap25.mean():.3f}.")
    out.append(f"Spearman(compiled success, agent mean) over procedures: 25 configs rho={r1[0]:.2f} (p={r1[1]:.3f}); 28 configs rho={r2[0]:.2f} (p={r2[1]:.3f})")
    # failure types
    rows=[d[c] for c in ids if c in d]
    f=[r for r in rows if not r["success"]]
    cnt=collections.Counter()
    for r in f:
        if r["err"]: cnt["program crashed or timed out"]+=1
        elif not r["action_called_correctly"]: cnt["wrong decision: performed a case that must be refused" if not r["should"] else "wrong decision: refused a case that must be performed"]+=1
        elif not r["dirgraph_satisfied"]: cnt["right decision, required checks not called (dirgraph)"]+=1
        elif not r["constraint_not_violated"]: cnt["right decision, tool response mismatch (constraint_not_violated)"]+=1
        elif not r["database_match"]: cnt["right decision, database mismatch"]+=1
        else: cnt["other"]+=1
    out.append("Failure types ("+str(len(f))+" failures): "+"; ".join(f"{k}: {v}" for k,v in cnt.most_common()))
    dec=np.mean([r["action_called_correctly"] for r in rows]); out.append(f"Decision accuracy (action called iff must be performed): {dec:.3f}; success among decision-correct: {np.mean([r['success'] for r in rows if r['action_called_correctly']]):.3f}")
    return cp,ap25,ps
if __name__=="__main__":
    out=["Agents: all 28 configurations in responses.jsonl. Overall agent success: best %.3f, median(25 real) %.3f, median(28) %.3f"%(np.nanmean(M,1).max(),np.median(np.nanmean(M,1)[KEEP]),np.median(np.nanmean(M,1)))]
    for n,l in [("results_A.json","Variant A (user_known dict)"),("results_B.json","Variant B (parameters extracted from the user message)"),("results_AC.json","Variant A + example database in the compile prompt (C)"),("results_BC.json","Variant B + example database in the compile prompt (C)")]:
        if os.path.exists(os.path.join(HERE,n)): summarize(n,l,out)
    open(os.path.join(HERE,"analysis_out.md"),"w").write("\n".join(out)); print("\n".join(out))
