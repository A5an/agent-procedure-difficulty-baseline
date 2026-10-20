import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, re, numpy as np, torch
D=ZR.REPO + "/third_party/agent-psychometrics/data/"
# fixed hyper-parameters (chosen before any held-out result was computed)
LAM_THETA=1.0   # prior precision on abilities / model / harness effects
LAM_BETA=0.25   # prior precision on task difficulty (sd 2)
LAM_LOGA=4.0    # prior precision on log discrimination (2PL)
LAM_UV=1.0      # prior precision on rank-2 factors
torch.set_default_dtype(torch.float64)

def load_tb():
    rows=[json.loads(l) for l in open(D+'terminalbench/responses_per_attempt.jsonl')]
    tasks=[json.loads(l)['task_id'] for l in open(D+'terminalbench/tasks.jsonl')]
    S=np.zeros((len(rows),len(tasks)));N=np.zeros_like(S)
    for i,r in enumerate(rows):
        for j,t in enumerate(tasks):
            c=r['responses'].get(t)
            if c: S[i,j]=c['successes'];N[i,j]=c['trials']
    return rows,tasks,S,N

def fit(S,N,kind='1pl',lamb=LAM_BETA,seed=0,iters=200):
    """penalised binomial MLE. S,N arrays (A,T), N=0 cells ignored. kind in 1pl,2pl,rank2."""
    A,T=S.shape
    S_=torch.tensor(S);N_=torch.tensor(N)
    g=torch.Generator().manual_seed(seed)
    P={'mu':torch.zeros(1),'th':torch.zeros(A),'be':torch.zeros(T)}
    if kind=='2pl': P['la']=torch.zeros(T)
    if kind=='rank2':
        P['u']=0.1*torch.randn(A,2,generator=g);P['v']=0.1*torch.randn(T,2,generator=g)
    for p in P.values(): p.requires_grad_(True)
    def logit():
        z=P['th'][:,None]-P['be'][None,:]
        if kind=='2pl': z=torch.exp(P['la'])[None,:]*z
        if kind=='rank2': z=z+P['u']@P['v'].T
        return z+P['mu']
    def loss():
        z=logit()
        nll=-(S_*torch.nn.functional.logsigmoid(z)+(N_-S_)*torch.nn.functional.logsigmoid(-z)).sum()
        pen=0.5*LAM_THETA*(P['th']**2).sum()+0.5*lamb*(P['be']**2).sum()
        if kind=='2pl': pen=pen+0.5*LAM_LOGA*(P['la']**2).sum()
        if kind=='rank2': pen=pen+0.5*LAM_UV*((P['u']**2).sum()+(P['v']**2).sum())
        return nll+pen
    opt=torch.optim.LBFGS(list(P.values()),max_iter=iters,line_search_fn='strong_wolfe',tolerance_grad=1e-7,tolerance_change=1e-12)
    def clo():
        opt.zero_grad();l=loss();l.backward();return l
    for _ in range(5): opt.step(clo)
    with torch.no_grad(): p=torch.sigmoid(logit()).numpy()
    return {k:v.detach().numpy() for k,v in P.items()},p

def nll(S,N,p,mask=None):
    p=np.clip(p,1e-6,1-1e-6)
    v=-(S*np.log(p)+(N-S)*np.log(1-p))
    if mask is not None: v=v[mask]; return v.sum()/N[mask].sum()
    return v.sum()/N.sum()
