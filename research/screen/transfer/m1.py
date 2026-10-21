import sys,json,numpy as np
from common import *
from sklearn.metrics import roc_auc_score
def norm_model(m):
    m=m.lower().replace('claude 4.5 sonnet','claude sonnet 4.5').replace('claude 4.6 opus','claude opus 4.6')
    m=m.replace('minimax m2.5','minimax-m2.5').replace('minimax-m2.5','minimax m2.5')
    return re.sub(r'[\s_-]+',' ',m).strip()
def tb_rows():
    rows,tasks,S,N=load_tb(); ag=[];keep=[];dropped=[]
    for i,r in enumerate(rows):
        m=r['metadata']['model']; h=r['metadata']['agent']
        if m.lower()=='multiple': dropped.append((r['subject_id'],'model=Multiple (ambiguous)'));continue
        keep.append(i); ag.append((h,norm_model(m)))
    return ag,S[keep],N[keep],dropped,[rows[i]['subject_id'] for i in keep]
SWE_MAP={ # hand-parsed from names that explicitly contain harness_model; same underlying model unified
'20231010_rag_claude2':('rag','claude2'),'20231010_rag_gpt35':('rag','gpt35'),'20240402_rag_claude3opus':('rag','claude3opus'),'20240402_rag_gpt4':('rag','gpt4'),
'20231010_rag_swellama13b':('rag','swellama13b'),'20231010_rag_swellama7b':('rag','swellama7b'),
'20240402_sweagent_claude3opus':('sweagent','claude3opus'),'20240402_sweagent_gpt4':('sweagent','gpt4'),'20240728_sweagent_gpt4o':('sweagent','gpt4o'),
'20240620_sweagent_claude3.5sonnet':('sweagent','claude35sonnet_old'),'20250225_sweagent_claude-3-7-sonnet':('sweagent','claude37sonnet'),'20250522_sweagent_claude-4-sonnet-20250514':('sweagent','claude4sonnet'),
'20241022_tools_claude-3-5-haiku':('tools','claude35haiku'),'20241022_tools_claude-3-5-sonnet-updated':('tools','claude35sonnet_new'),'20250224_tools_claude-3-7-sonnet':('tools','claude37sonnet'),
'20250522_tools_claude-4-opus':('tools','claude4opus'),'20250522_tools_claude-4-sonnet':('tools','claude4sonnet'),
'20241029_OpenHands-CodeAct-2.1-sonnet-20241022':('openhands','claude35sonnet_new'),'20250524_openhands_claude_4_sonnet':('openhands','claude4sonnet'),'20250807_openhands_gpt5':('openhands','gpt5'),
'20250716_openhands_kimi_k2':('openhands','kimik2'),'20251127_openhands_claude-opus-4-5':('openhands','claudeopus45'),
'20251120_livesweagent_gemini-3-pro-preview':('livesweagent','gemini3pro'),'20251215_livesweagent_claude-opus-4-5':('livesweagent','claudeopus45'),
'20241028_agentless-1.5_gpt4o':('agentless','gpt4o'),'20241202_agentless-1.5_claude-3.5-sonnet-20241022':('agentless','claude35sonnet_new'),
'20241016_epam-ai-run-gpt-4o':('epam','gpt4o'),'20241212_epam-ai-run-claude-3-5-sonnet':('epam','claude35sonnet_new'),'20250804_epam-ai-run-claude-4-sonnet':('epam','claude4sonnet'),
'20250603_Refact_Agent_claude-4-sonnet':('refact','claude4sonnet'),'20250611_moatless_claude-4-sonnet-20250514':('moatless','claude4sonnet'),
'20241108_autocoderover-v2.0-claude-3-5-sonnet-20241022':('autocoderover','claude35sonnet_new'),'20250122_autocoderover-v2.1-claude-3-5-sonnet-20241022':('autocoderover','claude35sonnet_new'),
}
def swe_rows():
    rs={}
    for l in open(D+'swebench_verified/responses.jsonl'):
        r=json.loads(l); rs[r['subject_id']]=r['responses']
    ids=[k for k in SWE_MAP if k in rs]; tasks=sorted(next(iter(rs.values())))
    S=np.array([[rs[k][t] for t in tasks] for k in ids],float)
    return [SWE_MAP[k] for k in ids],S,np.ones_like(S),ids

def loo(ag,S,N,verbose=False):
    A,T=S.shape
    keys=sorted(set(ag))
    models=sorted({m for _,m in ag}); harn=sorted({h for h,_ in ag})
    res=[]
    for key in keys:
        te=np.array([a==key for a in ag]); tr=~te
        h,m=key
        # eligibility: both factors present among other training agents
        if not any(a[1]==m for a,t in zip(ag,tr) if t) or not any(a[0]==h for a,t in zip(ag,tr) if t): continue
        trag=[a for a,t in zip(ag,tr) if t]
        St,Nt=S[tr],N[tr]
        # free theta
        P,_=fit(St,Nt)
        th=P['mu'][0]+P['th']; be=P['be']
        # decomposition fit
        mi={x:i for i,x in enumerate(sorted({a[1] for a in trag}))}; hi={x:i for i,x in enumerate(sorted({a[0] for a in trag}))}
        Sm=torch.tensor(St);Nm=torch.tensor(Nt)
        Q={'mu':torch.zeros(1),'tm':torch.zeros(len(mi)),'tH':torch.zeros(len(hi)),'be':torch.zeros(T)}
        for p in Q.values(): p.requires_grad_(True)
        Mi=torch.tensor([mi[a[1]] for a in trag]);Hi=torch.tensor([hi[a[0]] for a in trag])
        def L():
            z=Q['mu']+Q['tm'][Mi][:,None]+Q['tH'][Hi][:,None]-Q['be'][None,:]
            nl=-(Sm*torch.nn.functional.logsigmoid(z)+(Nm-Sm)*torch.nn.functional.logsigmoid(-z)).sum()
            return nl+0.5*LAM_THETA*((Q['tm']**2).sum()+(Q['tH']**2).sum())+0.5*LAM_BETA*(Q['be']**2).sum()
        opt=torch.optim.LBFGS(list(Q.values()),max_iter=200,line_search_fn='strong_wolfe',tolerance_grad=1e-7,tolerance_change=1e-12)
        def clo():
            opt.zero_grad();l=L();l.backward();return l
        for _ in range(5): opt.step(clo)
        Q={k:v.detach().numpy() for k,v in Q.items()}
        z_dec=Q['mu'][0]+Q['tm'][mi[m]]+Q['tH'][hi[h]]-Q['be']
        # decomposition's own beta is used for its prediction; comparators use free-fit beta (each from its own training fit)
        same=[i for i,a in enumerate(trag) if a[1]==m]
        z_mean=th.mean()-be
        z_same=th[same].mean()-be
        z_modelonly=Q['mu'][0]+Q['tm'][mi[m]]-Q['be']
        sig=lambda z:1/(1+np.exp(-z))
        for i in np.where(te)[0]:
            res.append(dict(key=key,i=i,S=S[i],N=N[i],dec=sig(z_dec),mean=sig(z_mean),same=sig(z_same),modonly=sig(z_modelonly)))
    return res

def score(res):
    out={}
    for name in ['dec','mean','same','modonly']:
        s=np.concatenate([r['S'] for r in res]);n=np.concatenate([r['N'] for r in res]);p=np.concatenate([r[name] for r in res])
        ll=nll(s,n,p)
        # pooled AUC over expanded attempts
        y=np.concatenate([np.r_[np.ones(int(a)),np.zeros(int(b-a))] for a,b in zip(s,n)])
        pp=np.concatenate([np.r_[np.full(int(a),q),np.full(int(b-a),q)] for a,b,q in zip(s,n,p)])
        auc=roc_auc_score(y,pp)
        err=np.array([ (r[name]*r['N']).sum()/r['N'].sum() - r['S'].sum()/r['N'].sum() for r in res])
        out[name]=dict(logloss=ll,auc=auc,mae_share=np.abs(err).mean(),bias=err.mean())
    return out

if __name__=='__main__':
    which=sys.argv[1]
    if which=='tb':
        ag,S,N,dropped,ids=tb_rows()
        mapping=[(i,)+a for i,a in zip(ids,ag)]
    else:
        ag,S,N,ids=swe_rows(); dropped=[]; mapping=[(i,)+a for i,a in zip(ids,ag)]
    res=loo(ag,S,N)
    sc=score(res)
    # paired per-agent logloss differences, bootstrap over agent groups
    def agent_ll(r,name): return nll(r['S'],r['N'],r[name])*r['N'].sum(), r['N'].sum()
    rng=np.random.default_rng(0)
    diffs={}
    for base in ['mean','same']:
        d=np.array([(agent_ll(r,'dec')[0]-agent_ll(r,base)[0]) for r in res]); n=np.array([r['N'].sum() for r in res])
        bs=[d[ix].sum()/n[ix].sum() for ix in (rng.integers(0,len(d),len(d)) for _ in range(2000))]
        diffs[base]=(d.sum()/n.sum(),np.percentile(bs,2.5),np.percentile(bs,97.5),float((d<0).mean()))
    json.dump(dict(n_rows=len(ag),n_factor_groups_eval=len({r['key'] for r in res}),n_eval_rows=len(res),n_models=len({m for _,m in ag}),n_harness=len({h for h,_ in ag}),
        score=sc,paired_dec_minus=diffs,dropped=dropped,mapping=mapping),open(f'm1_{which}.json','w'),indent=1,default=str)
    print(which,len(ag),len(res));print(json.dumps(sc,indent=1));print(diffs)
