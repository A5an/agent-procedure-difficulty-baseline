"""Build per-case emulation features for SOPBench (no outcome of any real case is used)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, numpy as np, pandas as pd
S = ZR.SCREEN
R = ZR.REPO + "/data/sopbench"
t = pd.read_csv(f'{R}/tasks.csv')
gs = pd.read_csv(f'{S}/ideas5/smart_gemini/synth_results.csv').rename(columns={'perform_label_by_code': 'perf', 'success': 'g'})
ss = pd.read_csv(f'{S}/ideas5/smart_sonnet/synth_results.csv').rename(columns={'perform_label': 'perf', 'success': 's'})
def cls(df, col):
    r = {}
    for pid, x in df.groupby('procedure_id'):
        r[pid] = (1 - x[x.perf == 1][col].mean() if (x.perf == 1).any() else np.nan, 1 - x[x.perf == 0][col].mean() if (x.perf == 0).any() else np.nan,
                  1 - x[col].mean())
    return r
cg, cs = cls(gs, 'g'), cls(ss, 's')
rows = []
for pid in sorted(t.procedure_id.unique()):
    a, b = cg.get(pid, (np.nan,) * 3), cs.get(pid, (np.nan,) * 3)
    fp = np.nanmean([a[0], b[0]]) if not (np.isnan(a[0]) and np.isnan(b[0])) else np.nan
    fr = np.nanmean([a[1], b[1]]) if not (np.isnan(a[1]) and np.isnan(b[1])) else np.nan
    raw = np.nanmean([a[2], b[2]])
    rows.append((pid, fp, fr, raw))
P = pd.DataFrame(rows, columns=['procedure_id', 'fp_raw', 'fr_raw', 'raw'])
print('missing fp', P.fp_raw.isna().sum(), 'missing fr', P.fr_raw.isna().sum(), 'of', len(P))
# fill a missing class with the procedure's pooled failure (documented fallback)
P['fp'] = P.fp_raw.fillna(P.raw); P['fr'] = P.fr_raw.fillna(P.raw)
P['emu_est'] = 0.35 * P.fp + 0.65 * P.fr
P['emu_mean'] = P.raw
C = t[['case_id', 'procedure_id']].merge(P, on='procedure_id', how='left')
plan = pd.read_csv(f'{S}/ideas3/plan_tools/features/plan3_sopbench.csv').set_index('case_id')
dry = pd.read_csv(f'{S}/ideas/dryrun/features/dry_sopbench.csv').set_index('case_id')
rub = pd.read_csv(f'{S}/rubric/features/rub_proc_sopbench.csv').set_index('case_id')
C['p_plan'] = C.case_id.map(plan.plan_refuse).astype(float)
C['p_dry'] = C.case_id.map(dry.p_refuse).astype(float)
C['p_rub'] = C.case_id.map(rub.p_refusal).astype(float)
res = pd.DataFrame(json.load(open(f'{S}/ideas2/compile/results_A.json')))
res['perf_hat'] = [int(p in tl) for p, tl in zip(res.procedure, res.tools)]
C['p_pre'] = 1 - C.case_id.map(res.set_index('case_id').perf_hat).astype(float)
for k in ['plan', 'dry', 'rub', 'pre']:
    C[f'mix_{k}'] = C[f'p_{k}'] * C.fr + (1 - C[f'p_{k}']) * C.fp
print(C.isna().sum()[lambda s: s > 0])
C.to_csv('emu_case_features.csv', index=False)
# label diagnostics (evaluation only)
from sklearn.metrics import roc_auc_score
lab = t.set_index('case_id').should_succeed.astype(int)
for k in ['plan', 'dry', 'rub', 'pre']:
    print('AUC refuse-state', k, round(roc_auc_score(1 - lab.loc[C.case_id], C[f'p_{k}'].fillna(.5)), 3))
print(C.describe().T[['mean', 'std', 'min', 'max']])
