import sys, os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import json, collections, sop, common, execu
D=os.path.dirname(os.path.abspath(__file__))+"/"
progs=json.load(open(D+'programs_C.json')); cs={t['case_id']:t for t in sop.load_cases()}
rows=json.load(open(D+'results_AC.json'))
f=[r for r in rows if not r['success']]
bp=collections.Counter((r['procedure']) for r in f); print(bp.most_common(15))
def show(r):
    t=cs[r['case_id']]; k=common.key_of(t); ev,log,err=execu.execute(progs[k]['code'],t['user_known'],t)
    print("==",r['case_id'],"should",r['should'],{x:ev[x] for x in ('action_successfully_called','dirgraph_satisfied','constraint_not_violated','database_match')},err)
    print(" graph:",[n[0] if isinstance(n,list) else n for n in t['directed_action_graph']['nodes']])
    print(" known",json.dumps(t['user_known'])[:200])
    for c in log: print("  ",c['tool_name'],json.dumps(c['arguments'])[:100],"->",str(c['content'])[:100])
cat=collections.defaultdict(list)
for r in f:
    if not r['dirgraph_satisfied'] and r['action_called_correctly']: cat['dir'].append(r)
    elif not r['action_called_correctly'] and r['should']: cat['fn'].append(r)
for r in cat['dir'][:6]: show(r)
for r in cat['fn'][:8]: show(r)
