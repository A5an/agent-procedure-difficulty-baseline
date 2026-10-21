import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json,sys,os,csv,statistics as st,collections
sys.dont_write_bytecode=True
R=ZR.DATA + "/SOPBench-d2622008"
DOMS=["bank","online_market","dmv","healthcare","library","hotel","university"]
TOOLS={'bank':17372,'online_market':13289,'dmv':17079,'healthcare':10925,'library':16377,'hotel':16779,'university':9606}
PRICE={'claude-sonnet-5':(2.0,10.0),'gemini-2.5-flash':(0.30,2.50),'gemini-2.5-pro':(1.25,10.0),'gemini-2.5-flash-lite':(0.10,0.40),'gpt-5':(1.25,10.0),'gpt-4.1':(2.0,8.0),'gemini-3.8-flash':(0.75,3.75)}
LAT={'claude-sonnet-5':4.0,'gemini-2.5-flash':5.0,'gemini-2.5-pro':12.0,'gemini-2.5-flash-lite':2.0,'gpt-5':25.0,'gpt-4.1':3.0}  # assumed seconds per call
sample={r['case_id'] for r in csv.DictReader(open('sample150.csv'))}
def mlen(m):
    n=len(m.get('content') or '')
    if m.get('tool_calls'): n+=len(json.dumps(m['tool_calls']))
    return n
def load(mod):
    eps={}
    for dom in DOMS:
        d=json.load(open(f'{R}/data/{dom}_tasks.json')); ids=[f'{dom}/{p}/{i}' for p,a in d.items() for i,_ in enumerate(a)]
        recs=json.load(open(f'{R}/output/{dom}/ast_{mod}-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json'))
        for cid,rec in zip(ids,recs):
            if not rec['interactions']: continue
            run=rec['interactions'][0]; hist=run['interaction']; pl=len(run['prompt'])+TOOLS[dom]
            goal=rec['task']['user_goal']; cum=0; calls=[]
            for m in hist:
                if m.get('sender')=='user' or m.get('tool_call_id'): cum+=mlen(m)
                else:
                    names=[t['function']['name'] for t in (m.get('tool_calls') or [])]
                    typ='text'
                    if goal in names: typ='final'
                    elif 'exit_conversation' in names: typ='exit'
                    elif names and all(n.startswith('internal_') for n in names): typ='check'
                    elif names: typ='lookup'
                    calls.append(dict(inp=(pl+cum)/4,out=mlen(m)/4,typ=typ)); cum+=mlen(m)
            eps[cid]=dict(dom=dom,calls=calls,carry=rec['task']['action_should_succeed'])
    return eps
THINK=150  # assumed hidden thinking tokens per call (central), cap 512 incl. thinking
def price(model,calls,think=THINK):
    pi,po=PRICE[model]; return sum(c['inp']*pi+(c['out']+think)*po for c in calls)/1e6
if __name__=='__main__':
    E={m:load(m) for m in ['gemini-2.5-flash','gemini-2.5-pro','gpt-5','gpt-4.1']}
    # step type mix
    for m in E:
        cnt=collections.Counter(c['typ'] for e in E[m].values() for c in e['calls']); t=sum(cnt.values()); print(m,{k:round(v/t,3) for k,v in cnt.items()})
    res={}
    THINKERS={'gemini-2.5-flash','gemini-2.5-pro'}
    def arm(name,fn):
        tot=dict(calls=0,inp=0,out=0,usd={0:0,500:0,1500:0},sec=0)
        for cid in sample:
            for model,c in fn(cid):
                tot['calls']+=1; tot['inp']+=c['inp']; tot['out']+=c['out']
                for th in (0,500,1500):
                    t=th if model in THINKERS else 0
                    tot['usd'][th]+=(c['inp']*PRICE[model][0]+(c['out']+t)*PRICE[model][1])/1e6
                tot['sec']+=LAT[model]
        res[name]=tot
    def alone(m): return lambda cid:[(m,c) for c in E[m][cid]['calls']]
    def mixed(a,b):
        def f(cid):
            cl=E[a][cid]['calls']; out=[]; ho=False
            for j,c in enumerate(cl):
                if not ho and c['typ']=='final':
                    ho=True
                out.append((b if ho else a,c))
            if ho: pass
            return out
        return f
    for m in ['gemini-2.5-flash','gemini-2.5-pro','gpt-5','gpt-4.1']: arm(m+' alone',alone(m))
    arm('claude-sonnet-5 alone (gpt-4.1 token profile)',lambda cid:[('claude-sonnet-5',c) for c in E['gpt-4.1'][cid]['calls']])
    arm('mixed pro->flash (A=pro checks, B=flash final)',mixed('gemini-2.5-pro','gemini-2.5-flash'))
    arm('mixed flash->pro (A=flash checks, B=pro final)',mixed('gemini-2.5-flash','gemini-2.5-pro'))
    arm('mixed gpt-5->gpt-4.1',mixed('gpt-5','gpt-4.1'))
    # extra: lookup by cheap model in mixed flash->pro (flash-lite does lookups)
    def three(cid):
        out=[]; ho=False
        for c in E['gemini-2.5-flash'][cid]['calls']:
            if not ho and c['typ']=='final': ho=True
            out.append(('gemini-2.5-pro' if ho else ('gemini-2.5-flash-lite' if c['typ']=='lookup' else 'gemini-2.5-flash'),c))
        return out
    arm('three-model flash-lite lookups + flash checks + pro final',three)
    for k,v in res.items(): print('%-60s calls %5d in %9.0f out %8.0f usd(think 0/500/1500) %.2f/%.2f/%.2f  sec %6.0f'%(k,v['calls'],v['inp'],v['out'],v['usd'][0],v['usd'][500],v['usd'][1500],v['sec']))
    json.dump({k:v for k,v in res.items()},open('cost_arms.json','w'),indent=1)
    # per-domain sample stats for flash and pro
    for m in ['gemini-2.5-flash','gemini-2.5-pro']:
        print(m)
        for d in DOMS:
            xs=[sum(c['inp'] for c in e['calls']) for k,e in E[m].items() if k in sample and e['dom']==d]
            ys=[sum(c['out'] for c in e['calls']) for k,e in E[m].items() if k in sample and e['dom']==d]
            n=[len(e['calls']) for k,e in E[m].items() if k in sample and e['dom']==d]
            print(' ',d,len(xs),round(st.mean(n),1),round(st.mean(xs)),round(st.mean(ys)))
