import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json,random,csv,sys
sys.dont_write_bytecode=True
R=ZR.DATA + "/SOPBench-d2622008/data"
DOMS=["bank","online_market","dmv","healthcare","library","hotel","university"]
cases=[]
for dom in DOMS:
    d=json.load(open(f'{R}/{dom}_tasks.json'))
    for proc,arr in d.items():
        for i,t in enumerate(arr):
            cases.append(dict(case_id=f'{dom}/{proc}/{i}',domain=dom,procedure=proc,carry=bool(t['action_should_succeed'])))
rng=random.Random(20261004)
N=75
sample=[]
for carry in (True,False):
    pool=[c for c in cases if c['carry']==carry]
    # allocate per domain: proportional to sqrt of size (keeps small domains visible), largest remainder, capped by availability
    import math
    w={d:math.sqrt(sum(1 for c in pool if c['domain']==d)) for d in DOMS}
    tot=sum(w.values()); raw={d:N*w[d]/tot for d in DOMS}
    alloc={d:int(raw[d]) for d in DOMS}
    for d in sorted(DOMS,key=lambda d:-(raw[d]-alloc[d]))[:N-sum(alloc.values())]: alloc[d]+=1
    for d in DOMS:
        p=[c for c in pool if c['domain']==d]; k=min(alloc[d],len(p))
        sample+=rng.sample(p,k)
    print('carry' if carry else 'refuse',alloc,sum(min(alloc[d],sum(1 for c in pool if c['domain']==d)) for d in DOMS))
sample.sort(key=lambda c:c['case_id'])
with open('sample150.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(sample[0])); w.writeheader(); w.writerows(sample)
print(len(sample))
