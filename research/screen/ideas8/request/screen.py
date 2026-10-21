import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys,math,collections
sys.path.insert(0,ZR.SCREEN + "/ideas")
from cpulock import cpu_lock
from load import *
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer

def ent(s):
    c=collections.Counter(s); n=len(s)
    return -sum(v/n*math.log2(v/n) for v in c.values()) if n else 0

def sop_feats():
    st,tk,it=load('sopbench'); F={}
    ids=list(st)
    msgs={i:re.search(r'Customer message: (.*?)\nAvailable tools',st[i],re.S).group(1) for i in ids}
    pol={i:re.search(r'Policy for this action:\n(.*?)\nCustomer message',st[i],re.S).group(1) for i in ids}
    # typicality within procedure (LOO mean cosine of char n-gram tfidf)
    typ={}
    byp=collections.defaultdict(list)
    for i in ids: byp[tk[i]['procedure_id']].append(i)
    for p,L in byp.items():
        if len(L)<3: 
            for i in L: typ[i]=np.nan
            continue
        v=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,4)).fit_transform([msgs[i] for i in L])
        S=(v@v.T).toarray(); np.fill_diagonal(S,np.nan)
        for k,i in enumerate(L): typ[i]=np.nanmean(S[k])
    # numeric z within procedure
    nums={i:[float(x) for x in re.findall(r'(?<![\w.])\d+(?:\.\d+)?(?![\w])',msgs[i].replace(',',''))] for i in ids}
    numz={}
    for p,L in byp.items():
        vals=[x for i in L for x in nums[i][:1]]
        for i in L:
            if len(L)>=3 and nums[i] and np.std(vals)>0: numz[i]=abs(nums[i][0]-np.mean(vals))/np.std(vals)
            else: numz[i]=np.nan
    # polic overlap: share of policy content words mentioned in message
    stop=set('the a an of to in is be and or not for with must that this by it as on are at from'.split())
    for i in ids:
        m=msgs[i]; w=m.split()
        quoted=re.findall(r'"[^"]*"|\'[^\']*\'',m)
        pw=set(x for x in re.findall(r'[a-z_]+',pol[i].lower()) if x not in stop and len(x)>3)
        mw=set(re.findall(r'[a-z_]+',m.lower()))
        toks=re.findall(r'[A-Za-z0-9_]{8,}',m)
        F[i]=dict(words=len(w),n_digit=sum(c.isdigit() for c in m),n_quoted=len(quoted),n_q=m.count('?'),
          n_sent=len(re.findall(r'[.!?]',m)),n_ask=len(re.findall(r'\b(check|confirm|let me know|verify|determine|tell me|ensure|see if|whether)\b',m.lower())),
          n_num=len(nums[i]),maxent=max([ent(t) for t in toks],default=0),n_bullets=m.count('\n-'),
          typ=typ[i],numz=numz[i],pol_overlap=len(pw&mw)/max(1,len(pw)),
          has_pw=int(bool(re.search(r'password|identification|identity|id\b',m.lower()))),
          n_caps=len(re.findall(r'\b[A-Z][a-z]+',m)), n_kv=len(re.findall(r'\b(username|password|id|code|amount|name|date|number|card)\b',m.lower())),
          fraction_nonstd_user=int('john_doe' not in m and 'alice' not in m))
    return F,tk,it

def tau_feats(b):
    st,tk,it=load(b); F={}
    for i,t in st.items():
        s=lambda k:(re.search(k+r':(.*?)(?=\n(?:Known info|Instructions|Reason for call):|\Z)',t,re.S) or [None,''])[1]
        rc,ki,ins=s('Reason for call'),s('Known info'),s('Instructions')
        F[i]=dict(w_all=len(t.split()),w_rc=len(rc.split()),w_ki=len(ki.split()),w_ins=len(ins.split()),
          n_ids=len(re.findall(r'\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{5,8}\b',t)),
          n_num=len(re.findall(r'\d+',t)),n_if=len(re.findall(r'\b[Ii]f\b',t)),n_neg=len(re.findall(r"\b(not|n't|never|no)\b|n't",t)),
          n_sent=len(re.findall(r'[.!?](\s|$)',t)),n_then=len(re.findall(r'\b(first|then|after|also|next|finally|second|else)\b',t.lower())),
          n_you_want=len(re.findall(r'[Yy]ou (want|would like|need|wish)',t)),n_dont=len(re.findall(r"[Yy]ou (don't|do not|won't|will not|refuse)",t)),
          n_paras=t.count('\n\n'),n_quotes=t.count('"'),
          has_resid=int(bool(re.search(r'reservation|order',t.lower())) and bool(re.search(r'\b[A-Z0-9]{6}\b|#W\d+',t))),
          n_dollar=t.count('$'),emo=len(re.findall(r'\b(frustrat|angry|upset|rude|annoy|impatien|mild|sad|furious|polite|happy)\w*',t.lower())),
          persist=len(re.findall(r'\b(insist|persist|demand|do not take|negotiat|coerc|pressure|threaten|escalat|manager|supervisor)\w*',t.lower())),
          lie=len(re.findall(r'\b(claim|pretend|told you|mention that|say that|fake|lie|falsely|approved)\b',t.lower())))
    return F,tk,it

def evalF(F,tk,it,name,within_proc=False):
    ids=[i for i in F if i in it]
    feats=list(F[ids[0]].keys()); out={}
    doms=sorted(set(tk[i]['domain'] for i in ids))
    for f in feats:
        rs=[];ws=[]
        for d in doms:
            L=[i for i in ids if tk[i]['domain']==d]
            x=np.array([F[i][f] for i in L],float); y=np.array([it[i] for i in L])
            ok=~np.isnan(x)
            if ok.sum()>5 and np.std(x[ok])>0: rs.append(spearmanr(x[ok],y[ok])[0]); 
            else: rs.append(np.nan)
        wp=np.nan
        if within_proc:
            ys=[];xs=[]
            byp=collections.defaultdict(list)
            for i in ids: byp[tk[i]['procedure_id']].append(i)
            for p,L in byp.items():
                if len(L)<4: continue
                x=np.array([F[i][f] for i in L],float); y=np.array([it[i] for i in L])
                ok=~np.isnan(x)
                if ok.sum()<4 or np.std(x[ok])==0: continue
                xs+=list((x[ok]-x[ok].mean())/x[ok].std()); ys+=list(y[ok]-y[ok].mean())
            if xs: wp=spearmanr(xs,ys)[0]
        out[f]=(np.nanmean(rs),rs,wp)
    print('==',name,doms)
    for f,(m,rs,wp) in sorted(out.items(),key=lambda kv:-abs(np.nan_to_num(kv[1][0]))):
        print(f'{f:20s} mean {m:+.3f}  per-dom {[round(float(r),2) for r in rs]}  within-proc {wp:+.3f}')
    return out
if __name__=='__main__':
    with cpu_lock('request'):
        F,tk,it=sop_feats(); evalF(F,tk,it,'sopbench',True)
        for b in ['tau2','tauk_banking']:
            F,tk,it=tau_feats(b); evalF(F,tk,it,b)
