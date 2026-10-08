import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, numpy as np, pandas as pd, sys
sys.path.insert(0, ZR.SCREEN + '/ideas')
from headroom import diff, pooled_rho, t, R
rng = np.random.default_rng(0)
out = {}
for var in ['B', 'BC']:
    res = pd.DataFrame(json.load(open(f'results_{var}.json')))
    res['perf_hat'] = [int(p in tl) for p, tl in zip(res.procedure, res.tools)]
    for scheme in ['new_procedures', 'new_domain']:
        m = pd.read_csv(f'{R}/results/sopbench/obs_{scheme}.csv.gz')
        d = diff(m, 'adele').merge(t[['case_id', 'domain', 'procedure_id', 'b']], on='case_id').merge(res[['case_id', 'perf_hat']], on='case_id')
        d = d[d.seed == d.seed.min()]
        d['x'] = 2.26 * d.perf_hat + d.l
        f = lambda g: pooled_rho(g, ['fold', 'domain'], 'x') - pooled_rho(g, ['fold', 'domain'], 'l')
        procs = d.procedure_id.unique(); by = {p: g for p, g in d.groupby('procedure_id')}
        bs = []
        for _ in range(300):
            s = rng.choice(procs, len(procs))
            g = pd.concat([by[p].assign(fold=by[p].fold.astype(str) + f'_{i}') if False else by[p] for i, p in enumerate(s)])
            bs.append(f(g))
        out[f'{var}|{scheme}'] = dict(diff=round(f(d), 3), ci=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
                                       acc=round(float((res.perf_hat == res.should.astype(int)).mean()), 3))
print(json.dumps(out, indent=1)); json.dump(out, open('precheck_ci.json', 'w'), indent=1)
