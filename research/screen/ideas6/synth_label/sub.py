import pandas as pd,numpy as np
from sklearn.metrics import roc_auc_score
d=pd.read_csv('data.csv');R=d[d.kind=='real'].set_index('case_id')
dry=pd.read_csv('../../ideas/dryrun/features/dry_sopbench.csv').set_index('case_id').p_refuse
b2=pd.read_csv('fs_B2.csv').set_index('case_id').p.dropna();b1=pd.read_csv('fs_B1.csv').set_index('case_id').p
i=b2.index;y=R.loc[i,'refuse']
for n,p in [('dry',dry[i]),('B1',b1[i]),('B2',b2),('B2+dry',(b2+dry[i])/2)]: print(n,len(i),round(roc_auc_score(y,p),3))
