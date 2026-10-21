import json,glob,sys,collections
root=sys.argv[1]; out=[]
for f in sorted(glob.glob(root+'/*.json')):
    d=json.load(open(f)); sims=d['simulations']; tasks=d['tasks']
    by=collections.defaultdict(list)
    for s in sims: by[s['task_id']].append(s['reward_info']['reward'] if s.get('reward_info') else None)
    r=[x for v in by.values() for x in v if x is not None]
    out.append(f"{f.split('/')[-1]}: agent={d['info']['agent_info'].get('llm')} retrieval={d['info'].get('retrieval_config')} tasks_in_file={len(tasks)} sims={len(sims)} tasks_with_sims={len(by)} mean_reward={sum(r)/len(r):.3f}")
out.append("task text present (description/user_scenario/ticket), evaluation_criteria present (gold, leakage for test use), policy via domain dir in tau2-bench repo, KB docs via required_documents")
open(root+'/SUMMARY.txt','w').write('\n'.join(out)+'\nNOTE: same 97 banking_knowledge tasks as already in project (data/external/tau2-banking has 9 other agents). NOT fresh tasks; adds agents/retrieval configs. S3: sierra-tau-bench-public, 24+ more banking files up to 365MB each (not downloaded).\n'); print('\n'.join(out))
