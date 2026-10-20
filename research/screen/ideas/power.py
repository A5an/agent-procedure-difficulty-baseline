"""Minimum detectable difference of the pooled rho for the protocol's paired procedure bootstrap."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, os
import numpy as np
sys.path.insert(0, ZR.SCREEN)
sys.path.insert(0, ZR.SCREEN + "/ideas")
import harness as H
from cpulock import cpu_lock
from pathlib import Path
E = H.E
R = Path(ZR.REPO + "/results")
E.RESULTS = R
E.METHODS = {"constant": "", "adele": "", "lltm_adele": "", "human_time": "", "length": "", "emb_ap": "", "oracle": ""}
E.COMBOS = []
with cpu_lock('power'):
    draws = {}
    for d, s in [x.split('|') for x in E.NEW]:
        _, draws[f'{d}|{s}'] = E.evaluate(d, s)
out = {}
base = np.nanmean([draws[k]['adele'] for k in E.NEW], axis=0)
for c in ['lltm_adele', 'human_time', 'length', 'emb_ap']:
    dist = np.nanmean([draws[k][c] for k in E.NEW], axis=0)
    dd = dist - base
    sd = float(np.nanstd(dd))
    out[c] = {'sd_paired_diff': sd, 'mde80': 2.80 * sd, 'corr_with_baseline': float(np.corrcoef(dist, base)[0, 1])}
    # per scenario sd
    out[c]['sd_by_scenario'] = {k: float(np.nanstd(draws[k][c] - draws[k]['adele'])) for k in E.NEW}
out['baseline_sd'] = float(np.nanstd(base))
print(json.dumps(out, indent=1))
json.dump(out, open('power.json', 'w'), indent=1)
