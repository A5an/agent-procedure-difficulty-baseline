import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, pickle, numpy as np
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas"); sys.path.insert(0, S + "/ideas3/tiers")
from pathlib import Path
import pandas as pd
import score as SC
from cpulock import cpu_lock
HERE = Path(__file__).parent
which = sys.argv[1]  # tau2 | all
V_T = ["A", "A_adele", "c_lad", "c_lad_A", "A_dm", "A_w", "A_all", "A_len", "A_raw"]
V_S = ["A", "A_adele", "c_lad", "c_lad_A"]
srcs = {"adele": str(HERE / "out_tau2")}
for v in V_T: srcs[v] = str(HERE / "out_tau2")
if which == "all":
    # sop columns come from out_sop; merge by giving separate names
    pass
TG = ["all_irt"]
rng = np.random.default_rng(20261006); res = {}
with cpu_lock("fewshot_A5_score"):
    for bench, outd, V in [("tau2", "out_tau2", V_T)] + ([("sopbench", "out_sop", V_S)] if which == "all" else []):
        sr = {"adele": str(HERE / outd), **{v: str(HERE / outd) for v in V}}
        W = SC.build(sr, TG, benches=(bench,))
        for sch in ["new_procedures", "new_domain"]:
            D = W[(bench, sch)].copy(); M = ["adele"] + V
            procs = sorted(D.procedure_id.unique()); code = {p: i for i, p in enumerate(procs)}; D["u"] = D.procedure_id.map(code)
            by_dom = [np.array(sorted(g.u.unique())) for _, g in D.groupby("domain")]
            pt, dr = SC.boot(D, ["fold", "domain"], M, TG, by_dom, len(procs), np.random.default_rng(rng.integers(1 << 31)))
            res[(bench, sch)] = (M, pt, dr); print(bench, sch, flush=True)
pickle.dump(res, open(HERE / f"score_{which}.pkl", "wb"))
ci = lambda d: (np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5))
lines = []
def report(name, keys, cols):
    for v in cols:
        pts = []; drs = []
        for k in keys:
            M, pt, dr = res[k]
            if v not in M: break
            pts.append(pt[:, 0]); drs.append(dr[:, :, 0])
        else:
            def sel(m): return np.mean([p[res[k][0].index(m)] for p, k in zip(pts, keys)], 0), np.mean([d[:, res[k][0].index(m)] for d, k in zip(drs, keys)], 0)
            p, d = sel(v); s = f"{name:14s} {v:9s} {p:+.3f} [{ci(d)[0]:+.3f},{ci(d)[1]:+.3f}]"
            for base in ["adele", "c_lad"]:
                if v != base and base in res[keys[0]][0]:
                    pb, db = sel(base); l, h = ci(d - db); s += f" | vs {base} {p-pb:+.3f} [{l:+.3f},{h:+.3f}]"
            lines.append(s)
for nm, ks, cols in [("tau2_np", [("tau2", "new_procedures")], ["adele"] + V_T), ("tau2_nd", [("tau2", "new_domain")], ["adele"] + V_T),
                     ("tau2_pool2", [("tau2", "new_procedures"), ("tau2", "new_domain")], ["adele"] + V_T)]:
    report(nm, ks, cols)
if which == "all":
    for nm, ks in [("sop_np", [("sopbench", "new_procedures")]), ("sop_nd", [("sopbench", "new_domain")]),
                   ("pool4", [("sopbench", "new_procedures"), ("sopbench", "new_domain"), ("tau2", "new_procedures"), ("tau2", "new_domain")])]:
        report(nm, ks, ["adele"] + V_S)
open(HERE / f"score_{which}.txt", "w").write("\n".join(lines)); print("\n".join(lines))
