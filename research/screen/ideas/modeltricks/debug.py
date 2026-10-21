import mt, harness as H, pandas as pd, numpy as np
from mt import *
bench="sopbench"
tasks = pd.read_csv(H.RB.DATA/bench/"tasks.csv")
src = H.RB.sources(bench)
base = H.RB.predictors
H.RB.predictors = lambda s: {"adele": base(s)["adele"], **make_preds(s, H.M, bench), "adele_plain": Trick(bench,"plain","adele")}
# one fold only
orig = H.RB.splits
H.RB.splits = lambda t, sc, seed: orig(t, sc, seed)[:1]
obs, betas = H.RB.run_scheme(bench, tasks, "new_procedures", src) if False else (None,None)
import types
H.RB.N_SEEDS = 1
obs, betas = H.RB.run_scheme(bench, tasks, "new_procedures", src)
print(obs.shape)
b = betas.pivot_table(index="case_id", columns="method", values="beta_hat")
print(b.corr().round(3)); print(np.abs(b.adele-b.adele_plain).max())
print(STACK_LOG)
