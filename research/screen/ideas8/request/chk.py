from screen import *
from scipy.stats import mannwhitneyu
F,tk,it=sop_feats()
ids=list(F)
lab=np.array([int(tk[i]['should_succeed']) for i in ids])
for f in ['maxent','has_pw','n_digit','fraction_nonstd_user']:
    x=np.array([F[i][f] for i in ids],float); y=np.array([it[i] for i in ids])
    ok=~np.isnan(x)
    print(f,'rho label',round(spearmanr(x[ok],lab[ok])[0],3),'rho b',round(spearmanr(x[ok],y[ok])[0],3))
    for l in (0,1):
        m=ok&(lab==l); print('  label',l,'n',m.sum(),'rho b',round(spearmanr(x[m],y[m])[0],3))
    # per-domain given label, within-proc
