import json,glob,collections,sys
root=sys.argv[1]; typekey=sys.argv[2]
res={}; types={}
for f in sorted(glob.glob(root+'/*.jsonl')):
    m=f.split('/')[-1][:-6]; res[m]={}
    for l in open(f):
        d=json.loads(l); tid=d['task_name']+'|'+str(d['meta'].get('id'))
        res[m][tid]=float(d['eval_result'].get('score',0)); types[tid]=d[typekey] if typekey in d else d['meta'].get(typekey)
tasks=sorted(types)
out=[f"source: {root}",f"models with per-task outcomes: {len(res)}",f"distinct tasks: {len(tasks)}",f"groups by {typekey}: {len(set(types.values()))} ; sizes {sorted(collections.Counter(types.values()).values())[:20]}"]
full=[m for m in res if len(res[m])>=0.95*len(tasks)]
out.append(f"models covering >=95% of tasks: {len(full)}")
import statistics
rate={t:statistics.mean(res[m][t] for m in full if t in res[m]) for t in tasks}
out.append("tasks solved by all models: %d ; by none: %d ; mixed: %d"%(sum(v==1 for v in rate.values()),sum(v==0 for v in rate.values()),sum(0<v<1 for v in rate.values())))
g=collections.defaultdict(list)
for t,v in rate.items(): g[types[t]].append(v)
out.append("group mean pass rate (n): "+'; '.join(f"{k}={statistics.mean(v):.2f}({len(v)})" for k,v in sorted(g.items())))
out.append("per-model pass: "+'; '.join(f"{m}={statistics.mean(res[m].values()):.2f}" for m in sorted(res)))
open(root+'/SUMMARY.txt','w').write('\n'.join(out)+'\n'); print('\n'.join(out))
