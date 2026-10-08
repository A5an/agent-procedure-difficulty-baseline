import sys, os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import json, collections, sop, common, execu
D=os.path.dirname(os.path.abspath(__file__))+"/"
progs=json.load(open(D+'programs_C.json')); cs={t['case_id']:t for t in sop.load_cases()}
rows=json.load(open(D+'results_AC.json')); unw={c for c,_ in json.load(open(D+'unwinnable_perform_cases.json'))}
proc=sys.argv[1]; n=int(sys.argv[2])
c=0
for r in rows:
    if r['procedure']!=proc or r['success'] or r['case_id'] in unw: continue
    t=cs[r['case_id']]; k=common.key_of(t); ev,log,err=execu.execute(progs[k]['code'],t['user_known'],t)
    print("==",r['case_id'],"should",r['should'],{x:ev[x] for x in ('action_successfully_called','dirgraph_satisfied','constraint_not_violated')},err)
    print(" graph:",[n_[0] if isinstance(n_,list) else n_ for n_ in t['directed_action_graph']['nodes']])
    print(" known",json.dumps(t['user_known'])[:200])
    for l in log: print("  ",l['tool_name'],json.dumps(l['arguments'])[:100],"->",str(l['content'])[:110])
    c+=1
    if c>=n: break
