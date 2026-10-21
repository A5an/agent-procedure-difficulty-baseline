"""Emulation with smart agents: does success on LLM-generated synthetic cases rank SOPBench procedures like real difficulty?"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
R = ZR.REPO + "/data/sopbench"
S = ZR.SCREEN + "/ideas5"
BROKEN = {'bank/cancel_credit_card', 'bank/pay_bill_with_credit_card'}
t = pd.read_csv(f'{R}/tasks.csv'); t['b'] = t.case_id.map(pd.read_csv(f'{R}/irt/1d_1pl/items.csv', index_col=0).b)
gr = pd.read_csv(f'{S}/smart_gemini/real_results.csv').rename(columns={'success': 'g_real'})
sr = pd.read_csv(f'{S}/smart_sonnet/real_results.csv').rename(columns={'success': 's_real'})
t = t.merge(gr[['case_id', 'g_real']], on='case_id').merge(sr[['case_id', 's_real']], on='case_id')
t = t[~t.procedure_id.isin(BROKEN)]
t['smart_fail'] = 1 - (t.g_real + t.s_real) / 2
gs = pd.read_csv(f'{S}/smart_gemini/synth_results.csv'); ss = pd.read_csv(f'{S}/smart_sonnet/synth_results.csv')
gs = gs.rename(columns={'perform_label_by_code': 'perf', 'success': 'g'}); ss = ss.rename(columns={'perform_label': 'perf', 'success': 's'})
def est(df, col):
    rows = []
    for pid, x in df.groupby('procedure_id'):
        fp = 1 - x[x.perf == 1][col].mean() if (x.perf == 1).any() else np.nan
        fr = 1 - x[x.perf == 0][col].mean() if (x.perf == 0).any() else np.nan
        e = (0.35 * fp + 0.65 * fr) if not (np.isnan(fp) or np.isnan(fr)) else (fp if np.isnan(fr) else fr)
        rows.append((pid, e, 1 - x[col].mean(), len(x)))
    return pd.DataFrame(rows, columns=['procedure_id', f'{col}_est', f'{col}_raw', f'{col}_n'])
E = est(gs, 'g').merge(est(ss, 's'), on='procedure_id')
E['both_est'] = (E.g_est + E.s_est) / 2
P = t.groupby(['domain', 'procedure_id']).agg(b=('b', 'mean'), smart_fail=('smart_fail', 'mean'), g_fail=('g_real', lambda v: 1 - v.mean()),
                                              s_fail=('s_real', lambda v: 1 - v.mean()), n=('b', 'size')).reset_index().merge(E, on='procedure_id')
def wrho(df, a, b):
    num = den = 0
    for _, x in df.groupby('domain'):
        x = x.dropna(subset=[a, b]); n = len(x)
        if n < 3 or x[a].nunique() < 2 or x[b].nunique() < 2: continue
        w = n * (n - 1) / 2; num += spearmanr(x[a], x[b]).correlation * w; den += w
    return num / den
rng = np.random.default_rng(0)
def boot(df, a, b, B=1000):
    vals = []
    for _ in range(B):
        parts = [x.sample(len(x), replace=True, random_state=int(rng.integers(1e9))) for _, x in df.groupby('domain')]
        vals.append(wrho(pd.concat(parts), a, b))
    return [round(float(np.nanpercentile(vals, 2.5)), 3), round(float(np.nanpercentile(vals, 97.5)), 3)]
out = {'n_procedures': len(P)}
for pred in ['g_est', 's_est', 'both_est']:
    for tgt in ['smart_fail', 'g_fail', 's_fail', 'b']:
        out[f'{pred} vs {tgt}'] = (round(wrho(P, pred, tgt), 3), boot(P, pred, tgt))
# fidelity: synthetic vs real for the same agent
out['fidelity gemini synth vs real'] = round(wrho(P, 'g_est', 'g_fail'), 3)
out['fidelity sonnet synth vs real'] = round(wrho(P, 's_est', 's_fail'), 3)
# case level: procedure estimate given to every case
C = t.merge(E[['procedure_id', 'both_est']], on='procedure_id')
def case_rho(df, a, b):
    num = den = 0
    for _, x in df.groupby('domain'):
        n = len(x)
        if x[a].nunique() < 2: continue
        w = n * (n - 1) / 2; num += spearmanr(x[a], x[b]).correlation * w; den += w
    return num / den
out['case-level both_est vs smart_fail'] = round(case_rho(C, 'both_est', 'smart_fail'), 3)
out['case-level both_est vs public b'] = round(case_rho(C, 'both_est', 'b'), 3)
for k, v in out.items(): print(k, v)
P.to_csv('emu_smart_proc.csv', index=False); json.dump(out, open('emu_smart.json', 'w'), indent=1)
