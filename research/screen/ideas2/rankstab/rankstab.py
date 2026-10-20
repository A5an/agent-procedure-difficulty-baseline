"""Is the difficulty order of SOPBench cases the same for strong and weak agents? Existing public outcomes only."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
D = ZR.REPO + "/data/sopbench"
rows = [json.loads(l) for l in open(f'{D}/responses.jsonl')]
Y = pd.DataFrame({r['subject_id']: r['responses'] for r in rows})  # cases x agents
t = pd.read_csv(f'{D}/tasks.csv').set_index('case_id')
bad = [a for a in Y.columns if any(k in a for k in ['gpt-5-mode', 'gpt-5-mini', 'gemini-2.5-pro'])]
Y = Y.drop(columns=bad)
acc = Y.mean().sort_values(ascending=False)
t = t.loc[Y.index]

def pooled(a, b, groups):
    num = den = 0
    for _, idx in groups.items():
        x, y = a.loc[idx], b.loc[idx]
        ok = x.notna() & y.notna(); x, y = x[ok], y[ok]
        n = len(x)
        if n < 3 or x.nunique() < 2 or y.nunique() < 2: continue
        w = n * (n - 1) / 2; num += spearmanr(x, y).correlation * w; den += w
    return num / den

case_groups = t.groupby('domain').groups
def proc_level(rate):
    p = rate.groupby(t.procedure_id).mean(); dom = t.groupby('procedure_id').domain.first()
    return p, dom.groupby(dom).groups

def compare(A, B):
    ra, rb = Y[A].mean(axis=1), Y[B].mean(axis=1)
    pa, g = proc_level(ra); pb, _ = proc_level(rb)
    return pooled(ra, rb, case_groups), pooled(pa, pb, g)

rng = np.random.default_rng(0)
ag = list(acc.index); k = 5
out = {'agents_used': len(ag), 'dropped': bad, 'top5': ag[:k], 'bottom5': ag[-k:], 'top5_acc': acc[:k].round(3).tolist(), 'bottom5_acc': acc[-k:].round(3).tolist()}
out['top5_vs_bottom5'] = compare(ag[:k], ag[-k:])
out['top5_vs_middle5'] = compare(ag[:k], ag[10:15])
rand = [compare(*np.split(rng.permutation(ag)[:2 * k], 2)) for _ in range(300)]
out['random5_vs_random5_mean'] = tuple(np.mean(rand, axis=0).round(3))
out['random5_vs_random5_p5_p95'] = [tuple(np.percentile(rand, 5, axis=0).round(3)), tuple(np.percentile(rand, 95, axis=0).round(3))]
top = Y[ag[:k]].mean(axis=1)
out['share_cases_all_top5_solve'] = round(float((top == 1).mean()), 3)
out['share_cases_no_top5_solve'] = round(float((top == 0).mean()), 3)
out['top5_success_by_label'] = top.groupby(t.should_succeed).mean().round(3).to_dict()
print(json.dumps(out, indent=1, default=float))
json.dump(out, open(ZR.SCREEN + "/ideas2/rankstab/rankstab.json", 'w'), indent=1, default=float)
