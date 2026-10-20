"""Steps 2-3: Monte Carlo over prior-built fake databases with stated facts fixed; run the compiled program; features.
Priors use ONLY: the domain's shipped default_data (value pools, ranges, formats, record shapes) and, as policy text numbers,
the case's constraint_parameters (these thresholds are printed in the agent's policy text). Never the case's initial_database."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, json, re, copy, math, random, datetime, importlib, signal, hashlib
sys.path.insert(0,ZR.SCREEN + "/ideas2/compile")
import sop
from execu import Tools, Timeout, _alarm
from env.variables import domain_keys
HERE=os.path.dirname(os.path.abspath(__file__))
NDRAW=50; P_EXIST=0.75
MODE=os.environ.get("SIM_MODE","v0")   # v0 = pre-registered prior; v1 = post hoc "known entity" prior
KEEP={"interaction_time","interaction_date","current_date","current_time"}   # "now" of the environment
ISO=re.compile(r"^(\d{4})-(\d{2})-(\d{2})(T\d{2}:\d{2}:\d{2})?$")
LONG=re.compile(r"^([A-Z][a-z]+) (\d{1,2})(st|nd|rd|th), (\d{4})$")
MONTHS=["January","February","March","April","May","June","July","August","September","October","November","December"]
def suf(n): return "th" if 10<=n%100<=20 else {1:"st",2:"nd",3:"rd"}.get(n%10,"th")
def toks(s):
    s=re.sub(r"([a-z])([A-Z])",r"\1_\2",str(s)).lower()
    out=set()
    for w in re.split(r"[^a-z0-9]+",s):
        if not w: continue
        if len(w)>3 and w.endswith("s") and not w.endswith("ss"): w=w[:-1]
        out.add(w)
    return out
GENERIC={"id","number","key","name","of","the","time","date","status","type"}
_dd={}
def default(d):
    if d not in _dd:
        mod=importlib.import_module(f"env.domains.{d}.{d}"); _dd[d]=copy.deepcopy(mod.default_data)
    return _dd[d]
_pools={}
def pools(d):
    """field name -> list of default values (documentation-level prior pools)"""
    if d in _pools: return _pools[d]
    P={}
    def walk(x,k):
        if isinstance(x,dict):
            for kk,v in x.items(): walk(v,kk)
        elif isinstance(x,list):
            for v in x: walk(v,k)
        else: P.setdefault(k,[]).append(x)
    walk(default(d),"")
    _pools[d]=P; return P
def parse_date(s):
    m=ISO.match(s)
    if m:
        y,mo,dd=int(m[1]),int(m[2]),int(m[3])
        try: return ("iso",datetime.date(y,mo,dd),m[4])
        except ValueError: return None
    m=LONG.match(s)
    if m and m[1] in MONTHS:
        try: return ("long",datetime.date(int(m[4]),MONTHS.index(m[1])+1,int(m[2])),None)
        except ValueError: return None
    return None
def fmt_date(kind,dt,tail):
    if kind=="iso": return dt.isoformat()+(tail or "")
    return f"{MONTHS[dt.month-1]} {dt.day}{suf(dt.day)}, {dt.year}"
def draw_leaf(key,v,pool,rng):
    if key in KEEP: return v
    if isinstance(v,bool): return rng.random()<0.5
    if isinstance(v,(int,float)):
        nums=[x for x in pool if isinstance(x,(int,float)) and not isinstance(x,bool)]
        lo,hi=min(nums),max(nums)
        if lo>=0:
            top=2*hi if hi>0 else 5
            x=rng.uniform(0,top)
        else: x=rng.uniform(2*lo,2*hi)
        if rng.random()<0.15: x=rng.choice(nums)
        return int(round(x)) if all(isinstance(n,int) for n in nums) else round(x,2)
    if isinstance(v,str):
        dp=parse_date(v)
        if dp:
            kind,dt,tail=dp
            return fmt_date(kind,dt+datetime.timedelta(days=rng.randint(-180,180)),tail)
        strs=[x for x in pool if isinstance(x,str)]
        if len(set(strs))>=2 and all(len(x)<=30 for x in strs) and key not in ("password","identification","admin_password"):
            return rng.choice(strs)
    return v
def perturb(node,key,path,pinned,pool,rng):
    if isinstance(node,dict):
        for k in list(node):
            node[k]=perturb(node[k],k,path+(k,),pinned,pool,rng)
        return node
    if isinstance(node,list):
        return [perturb(x,key,path+(i,),pinned,pool,rng) for i,x in enumerate(node)]
    if path in pinned: return node
    return draw_leaf(key,node,pool.get(key,[node]),rng)
def get(db,path):
    for p in path: db=db[p]
    return db
def dict_collections(db):
    out=[]
    def walk(x,path):
        if isinstance(x,dict):
            if x and all(isinstance(v,dict) for v in x.values()) and path: out.append(path)
            for k,v in x.items(): walk(v,path+(k,))
        elif isinstance(x,list):
            for i,v in enumerate(x): walk(v,path+(i,))
    walk(db,()); return out
def leaves(db):
    out=[]
    def walk(x,path):
        if isinstance(x,dict):
            for k,v in x.items(): walk(v,path+(k,))
        elif isinstance(x,list):
            for i,v in enumerate(x): walk(v,path+(i,))
        else: out.append(path)
    walk(db,()); return out
def lastkey(path):
    for p in reversed(path):
        if isinstance(p,str): return p
    return ""
def apply_facts(db,facts,rng,domain):
    """facts: list of (name, value, kind) kind in {'param','other'}. returns pinned set of leaf paths"""
    pinned=set(); scope=None
    cols=dict_collections(db); dcols=set(dict_collections(default(domain)))
    # 1. identifier facts: strings that name an entity
    for name,val,kind in facts:
        if not isinstance(val,str) or kind!="param": continue
        tk=toks(name)-GENERIC or toks(name)
        best=None;bs=0
        for c in cols:
            s=len(tk & toks(c[-1]))
            if scope and c[:len(scope)]==scope: s+=0.5
            if s>bs: best,bs=c,s
        if best is None and name.lower() in ("username","user_name","user","student_id","student","member","patient","customer"):
            for c in cols:
                if len(c)==1: best=c; break
        if best is None:
            cands=[c for c in cols if len(c)==1 and lastkey(c) in ("accounts","students","users","members")]
            if cands and re.search(r"(user|name|owner|holder|person|account)",name.lower()): best=cands[0]
        if best is None: continue
        try: coll=get(db,best)
        except (KeyError,IndexError,TypeError): continue
        known=False
        if MODE=="v1" and best in dcols:
            try: known=val in get(default(domain),best)
            except (KeyError,IndexError,TypeError): known=False
        exists=rng.random()<((0.95 if known else 0.30) if MODE=="v1" else P_EXIST)
        if exists:
            if val not in coll: coll[val]=copy.deepcopy(coll[rng.choice(list(coll))])
            if len(best)<=1:
                scope=best+(val,)
                if MODE=="v1" and known and rng.random()<0.5:
                    for p in leaves(get(db,scope)): pinned.add(scope+p)
        else:
            coll.pop(val,None)
            if not coll: coll["__placeholder__"]=copy.deepcopy(default(domain)[best[0]][next(iter(default(domain)[best[0]]))]) if len(best)==1 else {}
    # 2. leaf facts
    L=leaves(db)
    for name,val,kind in facts:
        tk=toks(name)
        def match(p):
            lk=toks(lastkey(p))
            if not lk: return False
            if kind=="param": return lk==tk
            return lk==tk or (lk<=tk and (lk-GENERIC))
        cand=[p for p in L if match(p) and p not in pinned]
        if not cand: continue
        under=[p for p in cand if scope and p[:len(scope)]==scope]
        if under: tgt=under
        elif rng.random()<P_EXIST: tgt=[rng.choice(cand)]
        else: continue
        for p in tgt:
            parent=get(db,p[:-1]); old=parent[p[-1]]
            v=val
            if isinstance(old,(int,float)) and not isinstance(old,bool):
                if not isinstance(val,(int,float)) or isinstance(val,bool): continue
            if isinstance(old,str) and isinstance(val,(int,float)): v=str(val)
            parent[p[-1]]=v; pinned.add(p)
    return pinned
def draw_db(domain,facts,rng):
    db=copy.deepcopy(default(domain))
    pinned=apply_facts(db,facts,rng,domain)
    return perturb(db,"",(),pinned,pools(domain),rng)
def facts_of(f):
    out=[]
    if not f: return out
    for k,v in (f.get("params") or {}).items():
        if v is not None and v!="": out.append((k,v,"param"))
    for o in (f.get("other") or []):
        if isinstance(o,dict) and o.get("field") and o.get("value") not in (None,""): out.append((str(o["field"]),o["value"],"other"))
    return out
def run_once(code,params,t,db,allowed):
    inn,full,descr=sop.defaults(t["domain"])
    try: ds=domain_keys[t["domain"]](db,inn,t["constraint_parameters"])
    except BaseException as e: return None,[],f"init {e.__class__.__name__}"
    tools=Tools(ds,allowed); err=None
    try:
        g={"__builtins__":__builtins__}; exec(code,g)
        signal.signal(signal.SIGALRM,_alarm); signal.alarm(3)
        try: g["handle"](copy.deepcopy(params),tools)
        finally: signal.alarm(0)
    except Timeout: err="timeout"
    except BaseException as e: err=f"{e.__class__.__name__}"
    return True,tools.log,err
def case_features(t,unit_code,f,ndraw=NDRAW):
    rng=random.Random(int(hashlib.sha1(t["case_id"].encode()).hexdigest()[:8],16))
    facts=facts_of(f); params=dict((f or {}).get("params") or {})
    _,ai=sop.agent_view(t); allowed={a["function"]["name"] for a in ai["tools"]}
    goal=t["user_goal"]; outs=[]
    for _ in range(ndraw):
        db=draw_db(t["domain"],facts,rng)
        ok,log,err=run_once(unit_code,params,t,db,allowed)
        outs.append((log,err))
    called=[any(c["tool_name"]==goal for c in log) for log,_ in outs]
    errs=sum(1 for _,e in outs if e)/ndraw
    # call keys and result constancy
    res={}
    for log,_ in outs:
        for c in log:
            if c["tool_name"]==goal: continue
            k=c["tool_name"]+json.dumps(c["arguments"],sort_keys=True,default=str)
            res.setdefault(k,set()).add(json.dumps(c["content"],default=str)[:200])
    pinned_keys={k for k,v in res.items() if len(v)==1}
    n_pin=len(pinned_keys); n_pri=len(res)-n_pin
    ends=[]
    for (log,_),cl in zip(outs,called):
        if cl: continue
        lc=[c for c in log if c["tool_name"]!=goal]
        if not lc: ends.append(1); continue
        c=lc[-1]; k=c["tool_name"]+json.dumps(c["arguments"],sort_keys=True,default=str)
        ends.append(int(k in pinned_keys))
    p=sum(called)/ndraw
    H=0 if p in (0,1) else -(p*math.log2(p)+(1-p)*math.log2(1-p))
    frac_pinned_ref=(sum(ends)/len(ends)) if ends else float("nan")
    return dict(case_id=t["case_id"],p_perf=p,n_pinned=n_pin,n_prior=n_pri,frac_prior=n_pri/max(1,n_pin+n_pri),
                forced_refuse=int(p==0 and ends and sum(ends)/len(ends)>=0.9),frac_pinned_refusal=frac_pinned_ref,
                entropy=H,err_rate=errs,n_facts=len(facts),n_param_facts=len(params),no_facts=int(len(facts)==0))
