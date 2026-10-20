"""Preliminary: how existing out-of-fold predictors rank Gemini 3.8 Flash failures on SOPBench (case and procedure level)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, ZR.SCREEN + "/ideas")
from headroom import diff, pooled_rho
from scipy.stats import spearmanr
R = ZR.REPO
S = ZR.SCREEN
t = pd.read_csv(f'{R}/data/sopbench/tasks.csv')
t['b'] = t.case_id.map(pd.read_csv(f'{R}/data/sopbench/irt/1d_1pl/items.csv', index_col=0).b)
g = pd.read_csv(f'{S}/ideas5/smart_gemini/real_results.csv')
t = t.merge(g[['case_id', 'success']].rename(columns={'success': 'g38'}), on='case_id')
t['g38_fail'] = 1 - t.g38
src = {'adele': f'{R}/results', 'lad2_adele': f'{S}/rubric/out/run', 'c_lad': f'{S}/ideas3/combo/out_main',
       'c_lad_cmpp': f'{S}/ideas4/code_plan/out_main'}
def proc_rho(d, col, tgt):
    num = den = 0
    for _, x in d.groupby('domain'):
        p = x.groupby('procedure_id')[[col, tgt]].mean()
        n = len(p)
        if n < 3 or p[col].nunique() < 2 or p[tgt].nunique() < 2: continue
        w = n * (n - 1) / 2; num += spearmanr(p[col], p[tgt]).correlation * w; den += w
    return num / den
out = {}
for scheme in ['new_procedures', 'new_domain']:
    for name, d0 in src.items():
        f = f'{d0}/sopbench/obs_{scheme}.csv.gz'
        try: m = pd.read_csv(f)
        except Exception as e: out[f'{scheme}|{name}'] = str(e)[:60]; continue
        if name not in m: out[f'{scheme}|{name}'] = 'col missing'; continue
        d = diff(m, name).merge(t, on='case_id')
        case_g = np.mean([pooled_rho(x, ['fold', 'domain'], 'l', 'g38_fail') for _, x in d.groupby('seed')])
        case_b = np.mean([pooled_rho(x, ['fold', 'domain'], 'l', 'b') for _, x in d.groupby('seed')])
        dd = d.groupby(['case_id', 'domain', 'procedure_id']).agg(l=('l', 'mean'), g=('g38_fail', 'first'), b=('b', 'first')).reset_index()
        out[f'{scheme}|{name}'] = dict(case_vs_g38=round(case_g, 3), case_vs_public=round(case_b, 3),
                                       proc_vs_g38=round(proc_rho(dd, 'l', 'g'), 3), proc_vs_public=round(proc_rho(dd, 'l', 'b'), 3))
# public difficulty as a predictor of Gemini 3.8 failure
tt = t.copy(); tt['fold'] = 0
out['public_irt_b_vs_g38'] = dict(case=round(pooled_rho(tt, ['domain'], 'b', 'g38_fail'), 3), proc=round(proc_rho(tt, 'b', 'g38_fail'), 3))
out['g38_success'] = round(t.g38.mean(), 3)
print(json.dumps(out, indent=1)); json.dump(out, open('prelim.json', 'w'), indent=1)
