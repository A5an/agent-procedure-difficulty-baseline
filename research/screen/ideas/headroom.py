"""Headroom on SOPBench: how much of within-domain difficulty is reachable from procedure-level information.
Uses existing obs files (repo results and rubric screen) only."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import numpy as np, pandas as pd, json
from scipy.special import logit
from scipy.stats import spearmanr
R = ZR.REPO
S = ZR.SCREEN
t = pd.read_csv(f'{R}/data/sopbench/tasks.csv')
t['b'] = t.case_id.map(pd.read_csv(f'{R}/data/sopbench/irt/1d_1pl/items.csv', index_col=0).b)
KEY = ["seed", "fold", "agent", "case_id"]
def diff(m, col):
    u = m.drop_duplicates(KEY)
    return u.assign(l=-logit(np.clip(u[col], 1e-4, 1-1e-4))).groupby(["seed","fold","case_id"]).l.mean().reset_index()
def pooled_rho(x, cells, a='l', b='b'):
    num = den = 0
    for _, g in x.groupby(cells):
        n = len(g)
        if n < 3: continue
        pr = n*(n-1)/2
        if g[a].nunique() > 1 and g[b].nunique() > 1:
            num += spearmanr(g[a], g[b]).correlation * pr
        den += pr
    return num/den
out = {}
for scheme in ['new_procedures', 'new_domain']:
    m = pd.read_csv(f'{R}/results/sopbench/obs_{scheme}.csv.gz')
    r2 = pd.read_csv(f'{S}/rubric/out/run/sopbench/obs_{scheme}.csv.gz')
    for c in ['lad2_adele', 'grp_proc_adele']:
        if c in r2: m = m.merge(r2[KEY + [c]], on=KEY, how='left')
    res = {}
    for col in ['adele', 'lad2_adele', 'grp_proc_adele', 'human_time', 'length']:
        if col not in m: continue
        d = diff(m, col).merge(t[['case_id','domain','procedure_id','should_succeed','b']], on='case_id')
        s0 = []
        for seed, g in d.groupby('seed'):
            # 1. protocol metric (cells fold x domain)
            a = pooled_rho(g, ['fold','domain'])
            # 2. label-conditional: cells fold x domain x allow/refuse
            c = pooled_rho(g, ['fold','domain','should_succeed'])
            # 3. procedure level: mean predicted vs mean b per procedure, within fold x domain
            p = g.groupby(['fold','domain','procedure_id'])[['l','b']].mean().reset_index()
            pl = pooled_rho(p, ['fold','domain'])
            s0.append((a, c, pl))
        res[col] = dict(zip(['rho_protocol','rho_given_label','rho_procedure_level'], np.mean(s0, axis=0).round(3).tolist()))
    # oracles on the same cells
    g = d[d.seed == 0].copy()
    g['proc_mean'] = g.groupby('procedure_id').b.transform('mean')
    g['proc_label_mean'] = g.groupby(['procedure_id','should_succeed']).b.transform('mean')
    g['label_only'] = g.should_succeed
    res['oracle_procedure_mean'] = {'rho_protocol': round(pooled_rho(g, ['fold','domain'], 'proc_mean'), 3)}
    res['oracle_procedure_x_label'] = {'rho_protocol': round(pooled_rho(g, ['fold','domain'], 'proc_label_mean'), 3)}
    res['oracle_label_only'] = {'rho_protocol': round(pooled_rho(g, ['fold','domain'], 'label_only'), 3)}
    out[scheme] = res
print(json.dumps(out, indent=1))
json.dump(out, open('headroom.json', 'w'), indent=1)
