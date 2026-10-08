"""Quick check: compiled rule decision (read-only pre-check before any agent run) as a case feature on SOPBench."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, numpy as np, pandas as pd, sys
sys.path.insert(0, ZR.SCREEN + '/ideas')
from headroom import diff, pooled_rho, t, R
out = {}
for var in ['A', 'B']:
    res = pd.DataFrame(json.load(open(f'results_{var}.json')))
    res['perf_hat'] = [int(p in tl) for p, tl in zip(res.procedure, res.tools)]
    acc = (res.perf_hat == res.should.astype(int)).mean()
    for scheme in ['new_procedures', 'new_domain']:
        m = pd.read_csv(f'{R}/results/sopbench/obs_{scheme}.csv.gz')
        d = diff(m, 'adele').merge(t[['case_id', 'domain', 'should_succeed', 'b']], on='case_id').merge(res[['case_id', 'perf_hat']], on='case_id')
        z = lambda s: (s - s.mean()) / s.std()
        rows = {}
        for name, col in [('baseline', d.l), ('precheck_only', d.perf_hat + 1e-6 * d.l), ('label_oracle', d.should_succeed + 1e-6 * d.l),
                          ('precheck+baseline', 2.26 * d.perf_hat + d.l)]:
            dd = d.assign(x=col)
            rows[name] = round(float(np.mean([pooled_rho(g, ['fold', 'domain'], 'x') for _, g in dd.groupby('seed')])), 3)
        out[f'{var}|{scheme}'] = dict(decision_accuracy=round(acc, 3), **rows)
print(json.dumps(out, indent=1))
json.dump(out, open('precheck.json', 'w'), indent=1)
