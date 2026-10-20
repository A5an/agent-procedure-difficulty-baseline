"""A4 of checks/E_irt_ideas.md on SOPBench only (copy of scratchpad sv/exp.py, data root as argument). usage: s05_a4.py <data root> <out.json>"""
import json,numpy as np,torch
from sklearn.metrics import roc_auc_score
torch.manual_seed(0); np.random.seed(0)
import sys,json as _j
D=sys.argv[1]
def load(d):
    R=[json.loads(l) for l in open(D+d+"/responses.jsonl")]
    ids=sorted(set().union(*[set(r["responses"]) for r in R]))
    S=np.zeros((len(R),len(ids)));T=np.zeros_like(S)
    for i,r in enumerate(R):
        for j,t in enumerate(ids):
            v=r["responses"].get(t)
            if v is None: continue
            if isinstance(v,dict): S[i,j]=v["successes"];T[i,j]=v["trials"]
            else: S[i,j]=v;T[i,j]=1
    return S,T
def fit(S,T,mask,kind,lam=0.1,steps=600):
    A,N=S.shape
    th=torch.zeros(A,requires_grad=True);b=torch.zeros(N,requires_grad=True)
    la=torch.zeros(N,requires_grad=True);ga=torch.zeros(A,requires_grad=True)
    U=torch.randn(A,2)*0.1;U.requires_grad_(True);V=(torch.randn(N,2)*0.1).requires_grad_(True)
    ps=[th,b]+([la] if kind=="2pl" else [])+([ga] if kind=="agslope" else [])+([U,V] if kind=="mf" else [])
    opt=torch.optim.LBFGS(ps,lr=1,max_iter=steps,line_search_fn="strong_wolfe")
    s=torch.tensor(S);t=torch.tensor(T);m=torch.tensor(mask,dtype=torch.float32)
    def logit():
        z=th[:,None]-b[None,:]
        if kind=="2pl": z=torch.exp(la)[None,:]*z
        if kind=="agslope": z=torch.exp(ga)[:,None]*z
        if kind=="mf": z=z+U@V.T
        return z
    def loss():
        opt.zero_grad();z=logit()
        ll=(s*torch.nn.functional.logsigmoid(z)+(t-s)*torch.nn.functional.logsigmoid(-z))*m
        pen=lam*(th**2).sum()*0+0.05*(b**2).sum()+0.05*(th**2).sum()+lam*((la**2).sum()+(ga**2).sum()+(U**2).sum()+(V**2).sum())
        l=-ll.sum()/m.sum()*1.0+pen/m.sum()*10
        l.backward();return l
    opt.step(loss)
    return torch.sigmoid(logit()).detach().numpy()
def evalp(P,S,T,te):
    ll=-(S*np.log(P+1e-9)+(T-S)*np.log(1-P+1e-9))[te].sum()/T[te].sum()
    y=[];p=[]
    I,J=np.where(te&(T>0))
    for i,j in zip(I,J):
        k=int(T[i,j]);s=int(S[i,j]);y+= [1]*s+[0]*(k-s);p+=[P[i,j]]*k
    return ll,roc_auc_score(y,p)

d="sopbench"
S,T=load(d);A,N=S.shape
print("==",d,S.shape)
folds=np.random.randint(0,5,size=S.shape)
res={}
for kind in ["rates","1pl","2pl","agslope","mf"]:
    L=[];Au=[]
    for f in range(5):
        te=folds==f;tr=~te
        if kind=="rates":
            a=(S*tr).sum(1)+1;n=(T*tr).sum(1)+2;r=a/n
            c=(S*tr).sum(0)+1;m=(T*tr).sum(0)+2;q=c/m
            g=(S*tr).sum()/(T*tr).sum()
            lg=lambda p:np.log(p/(1-p))
            z=lg(r)[:,None]+lg(q)[None,:]-lg(g)
            P=1/(1+np.exp(-z))
        else: P=fit(S,T,tr,kind)
        l,a_=evalp(P,S,T,te);L.append(l);Au.append(a_)
    res[kind]=(float(np.mean(L)),float(np.mean(Au)))
    print(f"{kind:8s} logloss {np.mean(L):.4f}  AUC {np.mean(Au):.4f}",flush=True)
_j.dump(res,open(sys.argv[2],"w"))
