"""Agent tiers by overall success. SOPBench drops the 3 empty-answer configs from every tier (they stay in 'all')."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, numpy as np, pandas as pd
REPO = ZR.REPO
BAD = ['gpt-5-mode', 'gpt-5-mini', 'gemini-2.5-pro']

def load(bench):
    rows = [json.loads(l) for l in open(f'{REPO}/data/{bench}/responses.jsonl')]
    Y = {}
    for r in rows:
        v = {c: (x['successes'] / x['trials'] if isinstance(x, dict) else float(x)) for c, x in r['responses'].items()}
        Y[r['subject_id']] = pd.Series(v)
    return pd.DataFrame(Y)  # cases x agents, success rate (0..1)

def tiers(bench):
    Y = load(bench)
    ok = [a for a in Y.columns if not (bench == 'sopbench' and any(k in a for k in BAD))]
    acc = Y[ok].mean().sort_values(ascending=False)
    ag = list(acc.index)
    return {'all': list(Y.columns), 'all_clean': ag, 'top5': ag[:5], 'top8': ag[:8], 'bot5': ag[-5:], 'bot8': ag[-8:]}, Y, acc

if __name__ == '__main__':
    for b in ['sopbench', 'tau2']:
        T, Y, acc = tiers(b)
        print(b, len(Y.columns), 'agents')
        for k, v in T.items(): print(' ', k, len(v), [f'{a}:{Y[a].mean():.2f}' for a in (v if len(v) <= 8 else v[:3])])
        json.dump({k: v for k, v in T.items()}, open(f'tiers_{b}.json', 'w'), indent=1)
