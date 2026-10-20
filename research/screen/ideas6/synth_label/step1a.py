import pandas as pd,numpy as np,warnings;warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack
df=pd.read_csv('data.csv');S=df[df.kind=='synth'].reset_index(drop=True);Rl=df[df.kind=='real'].reset_index(drop=True)
dry=pd.read_csv('../../ideas/dryrun/features/dry_sopbench.csv').set_index('case_id')
Rl['dry']=dry.loc[Rl.case_id,'p_refuse'].values
Rl['dom']=Rl.proc.str.split('/').str[0]
def within(p,df):
    a=[];w=[]
    for pr,g in df.assign(p=p).groupby('proc'):
        if g.refuse.nunique()==2: a.append(roc_auc_score(g.refuse,g.p));w.append(len(g))
    return np.average(a,weights=w)
def rep(name,p):
    print(f'{name:30s} pooled {roc_auc_score(Rl.refuse,p):.3f} within-proc {within(p,Rl):.3f}')
rep('dry p_refuse (zero-shot)',Rl.dry.values)
# a) pooled tfidf word
for an,ng in [('word',(1,2)),('char_wb',(2,5))]:
    v=TfidfVectorizer(analyzer=an,ngram_range=ng,min_df=2,sublinear_tf=True).fit(pd.concat([S.msg,Rl.msg]))
    for C in [0.3,3]:
        m=LogisticRegression(C=C,max_iter=2000).fit(v.transform(S.msg),S.refuse)
        rep(f'pooled {an} C={C}',m.predict_proba(v.transform(Rl.msg))[:,1])
    # per-proc
    p=np.zeros(len(Rl))
    for pr in Rl.proc.unique():
        s=S[S.proc==pr];i=(Rl.proc==pr).values
        if len(s)<4 or s.refuse.nunique()<2: p[i]=0.5;continue
        m=LogisticRegression(C=3,max_iter=2000).fit(v.transform(s.msg),s.refuse);p[i]=m.predict_proba(v.transform(Rl.msg[i]))[:,1]
    rep(f'per-proc {an}',p)
    # pooled + proc onehot
    from sklearn.preprocessing import OneHotEncoder
    oh=OneHotEncoder(handle_unknown='ignore').fit(S[['proc']])
    m=LogisticRegression(C=3,max_iter=3000).fit(hstack([v.transform(S.msg),oh.transform(S[['proc']])]),S.refuse)
    rep(f'pooled+procOH {an}',m.predict_proba(hstack([v.transform(Rl.msg),oh.transform(Rl[['proc']])]))[:,1])
# message-only baseline: synth proc prior
pri=S.groupby('proc').refuse.mean();rep('synth proc prior',Rl.proc.map(pri).values)
rl=Rl.groupby('proc').refuse.mean();print('real proc prior AUC (oracle proc mean)',roc_auc_score(Rl.refuse,Rl.proc.map(rl)))
