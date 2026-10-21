import json,glob,os,collections,statistics
R='data/mcpmark/mcpmark-v1-0905'; T='data/mcpmark_tasks/tasks'
res=collections.defaultdict(lambda: collections.defaultdict(list)); 
for f in glob.glob(R+'/*/run-*/*/meta.json'):
    p=f.split('/'); ms=p[-4]; model,svc=ms.split('__'); task=svc+'/'+p[-2]
    try: d=json.load(open(f))
    except Exception: continue
    res[model][task].append(int(bool(d['execution_result']['success'])))
tasks=set(t for m in res for t in res[m])
out=[f"models: {len(res)}; distinct (service/task): {len(tasks)}",
"task text: tasks/<service>/<difficulty>/<category>/<task>/description.md in github.com/eval-sys/mcpmark (Apache-2.0); mapping from 'desktop__music_report' to path is by last two path parts (not verified for all)",
"runs per model-task: "+str(collections.Counter(len(v) for m in res for v in res[m].values()))]
for m in sorted(res):
    v=[statistics.mean(x) for x in res[m].values()]; out.append(f"  {m}: tasks={len(v)} mean_pass={statistics.mean(v):.3f}")
full=[m for m in res if len(res[m])>=120]
mean={t:statistics.mean(statistics.mean(res[m][t]) for m in full if t in res[m]) for t in tasks}
out.append(f"models with >=120 tasks: {len(full)}; tasks solved by none (mean=0): {sum(v==0 for v in mean.values())}; by all: {sum(v==1 for v in mean.values())}")
open('data/mcpmark/SUMMARY.txt','w').write('\n'.join(out)+'\n'); print('\n'.join(out))
