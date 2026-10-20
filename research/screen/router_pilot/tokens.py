import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json,sys,os,statistics as st
sys.dont_write_bytecode=True
R=ZR.DATA + "/SOPBench-d2622008"
sys.path.insert(0,R)
from env.task import task_default_dep_full, task_initializer
DOMS=["bank","online_market","dmv","healthcare","library","hotel","university"]
tools={}
for dom in DOMS:
    d=json.load(open(f'{R}/data/{dom}_tasks.json'))
    deps=task_default_dep_full(dom,'full','structured',dependency_verb_dep_orig=True)
    k=list(d)[0]; t=d[k][0]; t['user_goal']=k
    tools[dom]=len(json.dumps(task_initializer(dom,t,*deps,None,'prompt',False,'structured')[3 and 0 or 3] if False else task_initializer(dom,t,*deps,None,'prompt',False,'structured')[2]['tools']))
MODELS=sys.argv[1:] or ["gemini-2.5-flash","gemini-2.5-pro","gpt-5","gpt-5-mini","gpt-4.1"]
def mlen(m):
    n=len(m.get('content') or '')
    if m.get('tool_calls'): n+=len(json.dumps(m['tool_calls']))
    return n
def p90(x):
    x=sorted(x); return x[int(0.9*(len(x)-1))]
out={}
for mod in MODELS:
    rows=[]
    for dom in DOMS:
        f=f'{R}/output/{dom}/ast_{mod}-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json'
        if not os.path.exists(f): continue
        for rec in json.load(open(f)):
            if not rec['interactions']: continue
            run=rec['interactions'][0]; hist=run['interaction']; prompt=len(run['prompt'])+tools[dom]
            goal=rec['task']['user_goal']; should=rec['task']['action_should_succeed']
            inp=0; outc=0; calls=0; cum=0; tgt_idx=None
            for m in hist:
                if m.get('sender') in ('user',) or m.get('tool_call_id'):
                    cum+=mlen(m)
                else:
                    calls+=1; inp+=prompt+cum; outc+=mlen(m); cum+=mlen(m)
                    if tgt_idx is None and m.get('tool_calls') and any(tc['function']['name']==goal for tc in m['tool_calls']): tgt_idx=calls
            rows.append(dict(dom=dom,calls=calls,inp=inp/4,out=outc/4,should=should,tgt=tgt_idx))
    out[mod]=rows
json.dump(out,open('episode_stats.json','w'))
for mod,rows in out.items():
    print(mod,len(rows),'calls mean %.2f'%st.mean(r['calls'] for r in rows))
    for dom in DOMS+['ALL']:
        rr=[r for r in rows if dom=='ALL' or r['dom']==dom]
        if not rr: continue
        print('  %-13s n=%3d calls %.1f  in_mean %7.0f in_p90 %7.0f  out_mean %5.0f out_p90 %5.0f  tgt_called %.2f'%(dom,len(rr),st.mean(r['calls'] for r in rr),st.mean(r['inp'] for r in rr),p90([r['inp'] for r in rr]),st.mean(r['out'] for r in rr),p90([r['out'] for r in rr]),sum(r['tgt'] is not None for r in rr)/len(rr)))
print(tools)
