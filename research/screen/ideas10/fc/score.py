"""Paired scoring with tiers/score.py machinery. usage: score_combo.py main|top5"""
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
HERE = Path(ZR.SCREEN + "/ideas10/fc"); T5 = None
mode = "main"
V = ["g1_clad_gap","c1_all","f1_cladgap_fc","f2_all_fc","f3_adele_fc","f0_fc_only"]
if mode == "main":
    srcs = {"adele": SC.R, **{v: str(HERE / "out") for v in V}}
else:
    srcs = {"adele": str(T5 / "refit/top5"), "lad2_adele": str(T5 / "refit/top5"), **{v: str(HERE / "out_top5") for v in V}}
TG = ["all_irt"]
rng = np.random.default_rng(20261005); res = {}
with cpu_lock("fc_score"):
    W = SC.build(srcs, TG)
    for bench in ["sopbench", "tau2"]:
        Ds = {s: W[(bench, s)] for s in ["new_procedures", "new_domain"] if (bench, s) in W}
        M = [m for m in srcs if all(m in D for D in Ds.values())]
        for level in ["case", "proc"]:
            if level == "proc" and bench != "sopbench": continue
            for sch, D in Ds.items():
                D = D.copy(); procs = sorted(D.procedure_id.unique()); code = {p: i for i, p in enumerate(procs)}
                D["u"] = D.procedure_id.map(code)
                by_dom = [np.array(sorted(g.u.unique())) for _, g in D.groupby("domain")]
                if level == "proc":
                    P = D.groupby(["seed", "domain", "procedure_id"], as_index=False)[M + TG].mean(); P["u"] = P.procedure_id.map(code)
                    F, cells = P, ["domain"]
                else: F, cells = D, ["fold", "domain"]
                pt, dr = SC.boot(F, cells, M, TG, by_dom, len(procs), np.random.default_rng(rng.integers(1 << 31)))
                res[(bench, sch, level)] = (M, TG, pt, dr); print(bench, sch, level, flush=True)
pickle.dump(res, open(HERE / f"fc_score_{mode}.pkl", "wb"))
ci = lambda d: (np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5))
def comb(sel, level):
    items = [res[(b, s, level)] for b, s in sel if (b, s, level) in res]
    M = items[0][0]; return M, np.mean([i[2] for i in items], 0), np.mean([i[3] for i in items], 0)
groups = {"pool4": [("sopbench", "new_procedures"), ("sopbench", "new_domain"), ("tau2", "new_procedures"), ("tau2", "new_domain")],
          "sop_np": [("sopbench", "new_procedures")], "sop_nd": [("sopbench", "new_domain")], "tau2_np": [("tau2", "new_procedures")], "tau2_nd": [("tau2", "new_domain")]}
lines = []
for level, ws in [("case", list(groups)), ("proc", ["sop_np", "sop_nd"])]:
    for w in ws:
        M, pt, dr = comb(groups[w], level)
        for tg in TG:
            j = TG.index(tg)
            for v in ["adele"] + V:
                i = M.index(v); lo, hi = ci(dr[:, i, j]); s = f"{level} {w} {tg} {v:10s} {pt[i,j]:+.3f} [{lo:+.3f},{hi:+.3f}]"
                for base in ["adele", "g1_clad_gap", "c1_all"]:
                    if v != base:
                        b = M.index(base); l2, h2 = ci(dr[:, i, j] - dr[:, b, j]); s += f" | vs {base} {pt[i,j]-pt[b,j]:+.3f} [{l2:+.3f},{h2:+.3f}]"
                lines.append(s)
open(HERE / f"fc_score_{mode}.txt", "w").write("\n".join(lines)); print("\n".join(lines))
