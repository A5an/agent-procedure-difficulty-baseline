import sys, os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import json, collections, sop, common, execu
D=os.path.dirname(os.path.abspath(__file__))+"/"
def causes(resfile,progfile):
    progs=json.load(open(D+progfile)); cs={t['case_id']:t for t in sop.load_cases()}
    rows=json.load(open(D+resfile)); unw={c for c,_ in json.load(open(D+'unwinnable_perform_cases.json'))}
    cat=collections.Counter(); missing=collections.Counter(); byproc=collections.defaultdict(collections.Counter); ex=collections.defaultdict(list)
    for r in rows:
        if r['success']: continue
        t=cs[r['case_id']]; ev,log,err=execu.execute(progs[r['unit']]['code'],t['user_known'],t)
        notfound=[l['tool_name'] for l in log if isinstance(l['content'],str) and l['content'].startswith('Error: Tool') ]
        if r['case_id'] in unw: c='scorer artefact (target tool absent or returns a non-boolean)'
        elif err: c='program crashed or timed out'
        elif notfound and not (set(notfound)<={t['user_goal']}): c='program called a tool that the agent does not have'
        elif not r['action_called_correctly']: c='wrong decision: refused a performable case' if r['should'] else 'wrong decision: performed a case that must be refused'
        elif not r['dirgraph_satisfied']:
            c='right decision, required prerequisite call missing (dirgraph)'
            nodes={n[0] for n in t['directed_action_graph']['nodes'] if isinstance(n,list)}
            called={l['tool_name'] for l in log}
            for m in nodes-called-{t['user_goal']}: missing[m]+=1
        else: c='right decision, tool-response mismatch or database mismatch'
        cat[c]+=1; byproc[(r['domain'],r['procedure'])][c]+=1; ex[c].append(r['case_id'])
    return cat,missing,byproc,ex
if __name__=="__main__":
    cat,missing,byproc,ex=causes(sys.argv[1],sys.argv[2])
    print(sum(cat.values())); 
    for k,v in cat.most_common(): print(v,k,ex[k][:4])
    print(missing.most_common(10))
