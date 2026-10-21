"""few-shot Gemini classifier. usage: fewshot.py B1|B2 [limit]"""
import sys,os,json,glob,hashlib,threading,random,re
sys.path.insert(0,'../../ideas/dryrun')
import lib
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
mode=sys.argv[1]; lim=int(sys.argv[2]) if len(sys.argv)>2 else None
S='../../'
df=pd.read_csv('data.csv')
db={}
for f in glob.glob(S+'ideas2/emulate_run/synth_all/*.json'):
    for proc,cs in json.load(open(f)).items():
        for x in cs: db[x['_cid']]=x['initial_database']
CACHE='fs_cache.jsonl';cache={};lock=threading.Lock()
if os.path.exists(CACHE):
    for l in open(CACHE): r=json.loads(l);cache[r['key']]=r['text']
syn=df[df.kind=='synth'];tgt=df[df.kind=='real']
if lim: tgt=tgt.sample(lim,random_state=0)
def ex_block(e):
    s=f'Customer message: {e.msg}\n'
    if mode=='B2': s+=f'Hidden account record at that time: {json.dumps(db[e.case_id],separators=(",",":"))}\n'
    return s+f'Correct outcome: {"refuse" if e.refuse==1 else "perform"}\n'
def prompt(r):
    s=syn[(syn.proc==r.proc)]
    sv=s[s.vi==r.vi]; 
    pool=sv if len(sv)>=8 else s
    rng=random.Random(r.case_id)
    ref=pool[pool.refuse==1];per=pool[pool.refuse==0]
    k=4
    ex=pd.concat([ref.sample(min(k,len(ref)),random_state=rng.randint(0,9999)),per.sample(min(k,len(per)),random_state=rng.randint(0,9999))])
    ex=ex.sample(frac=1,random_state=rng.randint(0,9999))
    p=("You are a customer-service agent that follows a written policy. Whether the agent must perform the requested action or must refuse depends on a hidden account record you cannot see.\n"
       f"=== POLICY ===\n{r.policy}\n\n=== SOLVED EXAMPLES (written for this policy) ===\n")
    for i,e in enumerate(ex.itertuples(),1): p+=f'--- Example {i}\n{ex_block(e)}'
    p+=f'\n=== NEW CASE ===\nCustomer message: {r.msg}\n\nBased on the examples, how likely is it that the correct outcome is refuse? Answer with only a number between 0 and 1.'
    return p
def run(r):
    pr=prompt(r);key=hashlib.sha256((mode+pr).encode()).hexdigest()
    if key in cache: t=cache[key]
    else:
        t,_=lib.generate_t(pr,0.0,1500,'low')
        with lock:
            cache[key]=t; open(CACHE,'a').write(json.dumps({'key':key,'text':t})+'\n')
    m=re.search(r'[01](?:\.\d+)?|\.\d+',t or '')
    return r.case_id,float(m.group()) if m else None
rows=list(tgt.itertuples())
with ThreadPoolExecutor(6) as ex: out=list(ex.map(run,rows))
pd.DataFrame(out,columns=['case_id','p']).to_csv(f'fs_{mode}.csv',index=False)
print(len(out),sum(o[1] is None for o in out))
