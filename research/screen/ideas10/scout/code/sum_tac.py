import json,glob,os,collections,statistics
R='data/tac_exp/evaluation/1.0.0'; T='data/tac_tasks/workspaces/tasks'
tasks=sorted(os.listdir(T)); res={}
for a in sorted(os.listdir(R)):
    res[a]={}
    for f in glob.glob(f'{R}/{a}/results/eval_*.json'):
        t=os.path.basename(f)[5:-5]
        try: d=json.load(open(f))
        except Exception: continue
        fs=d['final_score']; res[a][t]=(fs['result']/fs['total'] if fs['total'] else 0, fs['result']==fs['total'] and fs['total']>0)
# task dir names vs result names: result names often end with -image
names=set(t for a in res for t in res[a])
def norm(n): return n[:-6] if n.endswith('-image') else n
full=[a for a in res if len(res[a])>=170]
out=[f"task dirs in TheAgentCompany repo: {len(tasks)}; result names total distinct: {len(names)}; agents with >=170 tasks: {len(full)} of {len(res)}",
"task text: workspaces/tasks/<id>/task.md and checkpoints.md (gold checkpoints = leakage for test use); no separate policy; services (GitLab, RocketChat, ownCloud, Plane) are the tools",
"outcome per agent-task: final_score result/total (partial credit) ; full-success flag = result==total"]
match=sum(1 for n in names if norm(n) in set(tasks)); out.append(f"result names matching task dirs after stripping -image: {match}/{len(names)}")
for a in sorted(res):
    v=[x[0] for x in res[a].values()]; s=[x[1] for x in res[a].values()]
    out.append(f"  {a}: n={len(v)} mean_partial={statistics.mean(v):.3f} full_success={sum(s)/len(s):.3f}")
allfull=[t for t in names if all(res[a].get(t,(0,False))[1] for a in full)]
none=[t for t in names if not any(res[a].get(t,(0,False))[1] for a in full)]
out.append(f"among {len(full)} full agents: tasks solved fully by all {len(allfull)}, by none {len(none)}")
g=collections.defaultdict(list)
for t in names: g[t.split('-')[0]].append(statistics.mean(res[a][t][0] for a in full if t in res[a]))
out.append("category (task id prefix) mean partial, n: "+'; '.join(f"{k}={statistics.mean(v):.2f}({len(v)})" for k,v in sorted(g.items())))
open('data/tac_exp/SUMMARY.txt','w').write('\n'.join(out)+'\n'); print('\n'.join(out))
