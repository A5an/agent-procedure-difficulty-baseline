"""Write the valid synthetic cases (v3 E and F) as SOPBench task files, ready for run_simulation.py --data_dir. No harness change is needed:
the real case of the same constraint variant supplies constraints, constraint_parameters, directed_action_graph and user_goal, the synthetic case supplies
initial_database, user_known and user_prompt, the label comes from the env's own check functions (never from the LLM)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, os, copy, collections
import gen, gt
H=ZR.SCREEN + "/ideas2/emulate"
rows=[r for r in json.load(open(H+"/validation_rows_gen_v3.json")) if r.get('tool_runs') and r.get('code_label') is not None]
g={(x['proc'],x['variant'],x['rep']):json.loads(x['text']) for x in json.load(open(H+"/gen_v3.json"))}
out=collections.defaultdict(dict); n=0
for r in rows:
    cs,same=gen.modal_variant(r['dom'],r['proc']); c=copy.deepcopy(same[0])
    s=g[(r['proc'],r['variant'],r['rep'])][r['idx']]
    c['initial_database']=s['initial_database']; c['user_known']=s['user_known']; c['user_prompt']=s['user_prompt']
    c['action_should_succeed']=bool(r['code_label']); c['synthetic']=True
    out[r['dom']].setdefault(r['proc'],[]).append(c); n+=1
os.makedirs(H+"/synthetic_tasks",exist_ok=True)
for d,v in out.items(): json.dump(v,open(f"{H}/synthetic_tasks/{d}_tasks.json","w"))
real_keys=set(same[0].keys()); print(n,"cases written; extra key vs real:",set(c.keys())-real_keys)
# every exported case builds an agent prompt and a system through the env's own task_initializer
bad=0
for d,v in out.items():
    for p,L in v.items():
        for c in L:
            ok,err=gt.load_check(d,c,goal=p); bad+= (not ok)
print("task_initializer failures:",bad)
