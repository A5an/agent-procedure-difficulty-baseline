import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, json, pickle
sys.path.insert(0, ZR.SCREEN); sys.path.insert(0, ZR.SCREEN + "/ideas")
import numpy as np, pandas as pd
import pilot as P
from pilot import E
rows, freq, smallnames, draws = pickle.load(open(os.path.join(P.OUT, "judge_raw.pkl"), "rb"))
SC = [f"{b}|{s}" for b, s in P.SCEN]
def pt(scen, k, col):
    out = []
    for r, fr in rows[(scen,) + k].items():
        df = pd.concat(fr)
        out.append(np.mean([E.rank_arrays(g.cell.values, g[col].values, g.tgt.values) for _, g in df.groupby("seed")]))
    return float(np.mean(out))
def ci(x): return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
R = {}
keys = sorted({k[1:] for k in rows}, key=str)
for k in keys:
    rho = np.mean([pt(s, k, "pred") for s in SC]); ref = np.mean([pt(s, k, "ref") for s in SC])
    d = np.mean([draws[((s,) + k, "pred")] for s in SC], axis=0); dr = np.mean([draws[((s,) + k, "ref")] for s in SC], axis=0)
    R["|".join(map(str, k))] = dict(rho=float(rho), rho_ci=ci(d), k0_same_target=float(ref), gain=float(rho - ref), gain_ci=ci(d - dr),
                                     by_scenario=dict(zip(["sop-proc", "sop-dom", "tau2-proc", "tau2-dom"], [pt(s, k, "pred") for s in SC])))
F = {}
for b in ("sopbench", "tau2"):
    c = sum((v for (bb, _), v in freq.items() if bb == b), type(next(iter(freq.values())))())
    n = sum(c.values()); F[b] = [(a, int(v), round(v / n, 3)) for a, v in c.most_common(5)]
res = json.load(open(os.path.join(P.OUT, "results.json")))
res["judge_error"] = R; res["adaptive_k1_pilot_freq_top5"] = F; res["small_model_names"] = smallnames
json.dump(res, open(os.path.join(P.OUT, "results.json"), "w"), indent=1)
for k, v in R.items():
    print(f"{k:34s} rho {v['rho']:+.3f} [{v['rho_ci'][0]:+.3f},{v['rho_ci'][1]:+.3f}] k0 {v['k0_same_target']:+.3f} gain {v['gain']:+.3f} [{v['gain_ci'][0]:+.3f},{v['gain_ci'][1]:+.3f}]  " + " ".join(f"{x:+.2f}" for x in v['by_scenario'].values()))
print(json.dumps(F, indent=1)); print(smallnames)
