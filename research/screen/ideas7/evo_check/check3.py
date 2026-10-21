"""Check 3: out-of-fold predictions vs smart-agent failure on SOPBench, search vs held-out domains. Evaluation only."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, numpy as np, pandas as pd
from scipy.special import logit
from scipy.stats import spearmanr
R = ZR.REPO
S = ZR.SCREEN
EVO = S + '/ideas7/evo'; HERE = S + '/ideas7/evo_check'
SEARCH = {'sopbench': ['bank', 'dmv', 'healthcare', 'hotel'], 'tau2': ['airline', 'retail']}
HELD = {'sopbench': ['library', 'online_market', 'university'], 'tau2': ['telecom']}
DROP = ['bank/cancel_credit_card', 'bank/pay_bill_with_credit_card']
SRC = {'c_lad': S + '/ideas3/combo/out_main', 'evo_f1': EVO + '/out_protocol', 'f1_refit': HERE + '/out_leak', 'f1_nomdc': HERE + '/out_leak'}
KEY = ['seed', 'fold', 'agent', 'case_id']
def diff(bench, scheme, col):
    m = pd.read_csv(f'{SRC[col]}/{bench}/obs_{scheme}.csv.gz'); u = m.drop_duplicates(KEY)
    return u.assign(l=-logit(np.clip(u[col], 1e-4, 1 - 1e-4))).groupby(['seed', 'fold', 'case_id']).l.mean().reset_index()
def target(bench):
    t = pd.read_csv(f'{R}/data/{bench}/tasks.csv')
    t['irt'] = t.case_id.map(pd.read_csv(f'{R}/data/{bench}/irt/1d_1pl/items.csv', index_col=0).b)
    if bench == 'sopbench':
        g = pd.read_csv(f'{S}/ideas5/smart_gemini/real_results.csv').set_index('case_id').success
        s = pd.read_csv(f'{S}/ideas5/smart_sonnet/real_results.csv').set_index('case_id').success
        t['smart'] = 1 - (t.case_id.map(g) + t.case_id.map(s)) / 2
        t['gem'] = 1 - t.case_id.map(g); t['son'] = 1 - t.case_id.map(s)
    return t
def rho_cells(d, cells, col, tg):
    num = den = 0
    for _, g in d.groupby(cells):
        n = len(g)
        if n < 3: continue
        pr = n * (n - 1) / 2
        if g[col].nunique() > 1 and g[tg].nunique() > 1: num += spearmanr(g[col], g[tg])[0] * pr
        den += pr
    return num / den if den else np.nan
def metric(D, cols, tg, level):
    """D: rows seed,fold,case_id,domain,procedure_id, one column per method, target tg. returns {col: mean over seeds rho}"""
    out = {c: [] for c in cols}
    for seed, g in D.groupby('seed'):
        if level == 'proc':
            g = g.groupby(['fold', 'domain', 'procedure_id'], as_index=False)[cols + [tg]].mean()
        for c in cols: out[c].append(rho_cells(g, ['fold', 'domain'], c, tg))
    return {c: float(np.nanmean(v)) for c, v in out.items()}
def boot(D, cols, tg, level, nb, rng):
    procs = {dom: sorted(g.procedure_id.unique()) for dom, g in D.groupby('domain')}
    byp = {p: g for p, g in D.groupby('procedure_id')}
    res = []
    for _ in range(nb):
        parts = []
        for dom, ps in procs.items():
            for p in rng.choice(ps, len(ps)): parts.append(byp[p])
        res.append(metric(pd.concat(parts, ignore_index=True), cols, tg, level))
    return pd.DataFrame(res)
rng = np.random.default_rng(20261007); NB = 200
lines = []
for bench, tgs in [('sopbench', ['smart', 'gem', 'son', 'irt']), ('tau2', ['irt'])]:
    t = target(bench)
    for scheme in ['new_procedures', 'new_domain']:
        cols = ['c_lad', 'evo_f1', 'f1_nomdc', 'f1_refit']
        M = None
        for c in cols:
            d = diff(bench, scheme, c).rename(columns={'l': c})
            M = d if M is None else M.merge(d, on=['seed', 'fold', 'case_id'])
        M = M.merge(t[['case_id', 'domain', 'procedure_id'] + tgs], on='case_id')
        if bench == 'sopbench': M = M[~M.procedure_id.isin(DROP)]
        for grp, doms in [('search', SEARCH[bench]), ('heldout', HELD[bench])]:
            D = M[M.domain.isin(doms)]
            for tg in tgs:
                for level in (['case', 'proc'] if bench == 'sopbench' else ['case']):
                    pt = metric(D, cols, tg, level); bt = boot(D, cols, tg, level, NB, rng)
                    s = f'{bench} {scheme} {grp} target={tg} {level}: ' + ' '.join(f'{c}={pt[c]:+.3f}' for c in cols)
                    for c in cols[1:]:
                        dd = bt[c] - bt['c_lad']; s += f' | {c}-c_lad {pt[c]-pt["c_lad"]:+.3f} [{dd.quantile(.025):+.3f},{dd.quantile(.975):+.3f}]'
                    lines.append(s); print(s, flush=True)
open(HERE + '/check3.txt', 'w').write('\n'.join(lines))
