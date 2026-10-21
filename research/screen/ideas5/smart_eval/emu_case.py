"""Case level for smart agents: emulation estimate per procedure, and the class-specific version using the compiled pre-check (L2)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
exec(open('emu_smart.py').read().split("out = {'n_procedures'")[0])
def cls(df, col):
    r = {}
    for pid, x in df.groupby('procedure_id'):
        r[pid] = (1 - x[x.perf == 1][col].mean() if (x.perf == 1).any() else np.nan, 1 - x[x.perf == 0][col].mean() if (x.perf == 0).any() else np.nan)
    return r
cg, cs = cls(gs, 'g'), cls(ss, 's')
res = pd.DataFrame(json.load(open(ZR.SCREEN + "/ideas2/compile/results_A.json")))
res['perf_hat'] = [int(p in tl) for p, tl in zip(res.procedure, res.tools)]
C = t.merge(res[['case_id', 'perf_hat']], on='case_id').merge(E[['procedure_id', 'both_est']], on='procedure_id')
def pick(row, which):
    fp = np.nanmean([cg[row.procedure_id][0], cs[row.procedure_id][0]]); fr = np.nanmean([cg[row.procedure_id][1], cs[row.procedure_id][1]])
    lab = row.perf_hat if which == 'precheck' else row.should_succeed
    v = fp if lab == 1 else fr
    return row.both_est if np.isnan(v) else v
C['emu_precheck'] = C.apply(lambda r: pick(r, 'precheck'), axis=1)
C['emu_truelabel'] = C.apply(lambda r: pick(r, 'label'), axis=1)
def case_rho(df, a, b):
    num = den = 0
    for _, x in df.groupby('domain'):
        n = len(x)
        if x[a].nunique() < 2: continue
        w = n * (n - 1) / 2; num += spearmanr(x[a], x[b]).correlation * w; den += w
    return num / den
for a in ['both_est', 'emu_precheck', 'emu_truelabel', 'perf_hat']:
    print(a, 'vs smart_fail', round(case_rho(C, a, 'smart_fail'), 3), '| vs public b', round(case_rho(C, a, 'b'), 3))
