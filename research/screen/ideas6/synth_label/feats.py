import pandas as pd,numpy as np
S='../../'
fs=pd.read_csv('fs_B1.csv').set_index('case_id');d=pd.read_csv('data.csv');Rl=d[d.kind=='real'].set_index('case_id')
p=fs.p.clip(0.02,0.98)
rub=pd.read_csv(S+'rubric/features/rub_proc_sopbench.csv',index_col=0)
cmp_=pd.read_csv(S+'ideas3/compile_consistency/cons_sopbench.csv',index_col=0)
ids=fs.index
proc=Rl.loc[ids,'proc']
F=pd.DataFrame(index=ids)
F['sl_p']=p
F['sl_p_x_depth']=p*rub.loc[ids,'max_depth']
F['sl_p_x_refcond']=p*rub.loc[ids,'refusal_conditions']
F['sl_p_x_len']=p*cmp_.loc[ids,'len_mean']
pm=p.groupby(proc).transform('mean')
F['sl_p_procmean']=pm;F['sl_p_dev']=p-pm
F.index.name='case_id';F.to_csv('feats.csv');print(F.describe().T[['mean','std']])
