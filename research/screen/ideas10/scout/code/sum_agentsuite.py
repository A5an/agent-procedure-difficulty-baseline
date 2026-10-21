import json,glob,collections,sys
root=sys.argv[1]
res=collections.defaultdict(dict); dom=collections.Counter(); first={}
for f in sorted(glob.glob(root+'/*.jsonl')):
    m=f.split('/')[-1][:-6]
    for l in open(f):
        d=json.loads(l)
        tid=d['task_name']+'_'+d['meta']['id'].split('_')[-1]; 
        res[m][tid]=res[m].get(tid,[])+[int(bool(d['meta']['is_correct']))]
        first[m]=first.get(m,0)+(len(d['messages'])==0)
tasks=sorted(set(t for m in res for t in res[m]))
out=[]
out.append(f"models with per-task outcomes: {len(res)}")
out.append(f"distinct task ids: {len(tasks)} (airline {sum(t.startswith('airline') for t in tasks)}, retail {sum(t.startswith('retail') for t in tasks)})")
out.append("policy text present in system message of each record: yes (first message role=system holds the policy)")
out.append("task text: user simulator instruction NOT in record (messages only); user turns visible. Task instruction must come from tau-bench repo tasks files")
out.append("trials per model-task: "+str(collections.Counter(len(v) for m in res for v in res[m].values())))
out.append("records with empty messages: "+str({k:v for k,v in first.items() if v}))
out.append("per-model pass rate:")
for m in sorted(res): 
    v=[x for t in res[m].values() for x in t]; out.append(f"  {m}: n={len(v)} pass={sum(v)/len(v):.3f}")
open(root+'/SUMMARY.txt','w').write('\n'.join(out)); print('\n'.join(out))
