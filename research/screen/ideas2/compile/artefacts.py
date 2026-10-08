import sys, os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import json, collections, numpy as np, sop, common, execu
import analyze as an
D=os.path.dirname(os.path.abspath(__file__))+"/"
progs=json.load(open(D+'programs_C.json')); cs={t['case_id']:t for t in sop.load_cases()}
rows={r['case_id']:r for r in json.load(open(D+'results_AC.json'))}
# perform cases where the target action can never count as successfully called: tool missing or returns non-bool non-list
allsucc=an.M[:,[an.idx[c] for c in cs if cs[c]['action_should_succeed']]]
unw=[]
for c,t in cs.items():
    if not t['action_should_succeed']: continue
    ds,ai=sop.agent_view(t)
    names={a['function']['name'] for a in ai['tools']}
    if t['user_goal'] not in names: unw.append((c,'tool absent')); continue
    # run the strict reference: does a plain call of goal after prerequisites return a bool? use compiled program log
    ev,log,err=execu.execute(progs[common.key_of(t)]['code'],t['user_known'],t)
    for l in log:
        if l['tool_name']==t['user_goal']:
            v=l['content']
            if not (isinstance(v,bool) or isinstance(v,(list,tuple))): unw.append((c,'returns %s'%type(v).__name__))
cnt=collections.Counter((c.split('/')[1],w) for c,w in unw); print(len(unw),cnt)
ids=[an.idx[c] for c,_ in unw]
print("agent success on these cases: mean over 28 agents",np.nanmean(an.M[:,ids]),"max over agents of any success",np.nanmax(an.M[:,ids]))
print("compiled C success on these",np.mean([rows[c]['success'] for c,_ in unw]))
json.dump(unw,open(D+'unwinnable_perform_cases.json','w'))
