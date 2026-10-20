"""Replication of the procedure-level metric on SOPBench new procedures.
Each fold model predicts a different set of procedures, so predictions are z-scored inside each (seed, fold) over all its
test cases before procedure means are compared across folds inside a domain. Paired bootstrap over procedures, seed 0,
mean over seeds for the point estimate."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, glob, os
import numpy as np, pandas as pd
from scipy.stats import rankdata
import metrics_extra as X
from cpulock import cpu_lock

t = X.tasks_of('sopbench')
def proc_table(m, col):
    d = X.difficulty(m, col)
    d['z'] = d.groupby(['seed', 'fold']).l.transform(lambda v: (v - v.mean()) / (v.std() + 1e-9))
    d = d.merge(t[['case_id', 'domain', 'procedure_id', 'b']], on='case_id')
    return d.groupby(['seed', 'domain', 'procedure_id'])[['z', 'b']].mean().reset_index()
def rho_dom(pm):
    num = den = 0.0
    for _, g in pm.groupby('domain'):
        n = len(g)
        if n < 3: continue
        pr = n * (n - 1) / 2
        num += np.corrcoef(rankdata(g.z), rankdata(g.b))[0, 1] * pr; den += pr
    return num / den
groups = {}
for p in sorted(glob.glob('*/out/**/sopbench/obs_new_procedures.csv.gz', recursive=True)):
    groups[p.split('/')[0]] = p
groups['rubric'] = ZR.SCREEN + "/rubric/out/run/sopbench/obs_new_procedures.csv.gz"
groups['repo'] = ZR.REPO + "/results/sopbench/obs_new_procedures.csv.gz"
out = {}
rng = np.random.default_rng(20261005)
procs = t[['domain', 'procedure_id']].drop_duplicates().values.tolist()
by_dom = {}
for dmn, p in procs: by_dom.setdefault(dmn, []).append(p)
picks = [{dmn: rng.choice(ps, len(ps)) for dmn, ps in by_dom.items()} for _ in range(1000)]
with cpu_lock('proc_np'):
    for grp, path in groups.items():
        m = pd.read_csv(path)
        skip = {'seed', 'fold', 'agent', 'case_id', 'y', 'constant', 'oracle'}
        cols = [c for c in m.columns if c not in skip and m[c].notna().all()]
        tabs = {c: proc_table(m, c) for c in cols}
        base = tabs['adele']
        def boot(pm):
            s0 = pm[pm.seed == 0].set_index('procedure_id')
            vals = []
            for pk in picks:
                num = den = 0.0
                for dmn, ps in pk.items():
                    g = s0.loc[ps]; n = len(g)
                    if n < 3 or g.z.nunique() < 2: continue
                    pr = n * (n - 1) / 2
                    num += np.corrcoef(rankdata(g.z), rankdata(g.b))[0, 1] * pr; den += pr
                vals.append(num / den)
            return np.array(vals)
        bb = boot(base)
        for c in cols:
            point = float(np.mean([rho_dom(g) for _, g in tabs[c].groupby('seed')]))
            res = {'rho_proc_np': point}
            if c != 'adele':
                dd = boot(tabs[c]) - bb
                res['minus_baseline_ci'] = [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
            out[f'{grp}:{c}'] = res
            print(f'{grp}:{c:28s} {point:+.3f} {res.get("minus_baseline_ci", "")}', flush=True)
json.dump(out, open('proc_np.json', 'w'), indent=1)
