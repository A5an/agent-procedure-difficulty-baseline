"""Paired scoring (ablation/score.py machinery). usage: score.py"""
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
HERE = Path(S + "/ideas7/evo")
import json; V = [c for f in json.load(open(S + "/ideas7/evo/finalists.json")) for c in (f"evo_f{f['n']}", f"evo_f{f['n']}_fam")]
srcs = {"adele": SC.R, "c_lad": S + "/ideas3/combo/out_main", **{v: str(HERE / "out_protocol") for v in V}}
TG = ["all_irt"]; DROP = ["bank/cancel_credit_card", "bank/pay_bill_with_credit_card"]
rng = np.random.default_rng(20261006); res = {}
with cpu_lock("evo_score"):
    W = SC.build(srcs, TG)
    for variant in ["all", "nodrop"]:
        for bench in ["sopbench", "tau2"]:
            if variant == "nodrop" and bench != "sopbench": continue
            for level in ["case", "proc"]:
                if level == "proc" and bench != "sopbench": continue
                for sch in ["new_procedures", "new_domain"]:
                    D = W[(bench, sch)].copy()
                    if variant == "nodrop": D = D[~D.procedure_id.isin(DROP)]
                    M = [m for m in srcs if m in D]
                    procs = sorted(D.procedure_id.unique()); code = {p: i for i, p in enumerate(procs)}
                    D["u"] = D.procedure_id.map(code)
                    by_dom = [np.array(sorted(g.u.unique())) for _, g in D.groupby("domain")]
                    if level == "proc":
                        P = D.groupby(["seed", "domain", "procedure_id"], as_index=False)[M + TG].mean(); P["u"] = P.procedure_id.map(code)
                        F, cells = P, ["domain"]
                    else: F, cells = D, ["fold", "domain"]
                    pt, dr = SC.boot(F, cells, M, TG, by_dom, len(procs), np.random.default_rng(rng.integers(1 << 31)))
                    res[(variant, bench, sch, level)] = (M, TG, pt, dr); print(variant, bench, sch, level, flush=True)
pickle.dump(res, open(HERE / "score.pkl", "wb"))
ci = lambda d: (np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5))
def comb(sel, level, variant):
    items = [res[(variant, b, s, level)] for b, s in sel if (variant, b, s, level) in res]
    return items[0][0], np.mean([i[2] for i in items], 0), np.mean([i[3] for i in items], 0)
groups = {"pool4": [("sopbench", "new_procedures"), ("sopbench", "new_domain"), ("tau2", "new_procedures"), ("tau2", "new_domain")],
          "sop_np": [("sopbench", "new_procedures")], "sop_nd": [("sopbench", "new_domain")], "tau2_np": [("tau2", "new_procedures")], "tau2_nd": [("tau2", "new_domain")],
          "sop2": [("sopbench", "new_procedures"), ("sopbench", "new_domain")], "tau2both": [("tau2", "new_procedures"), ("tau2", "new_domain")]}
lines = []
plan = [("all", "case", list(groups)), ("all", "proc", ["sop_nd", "sop_np"]), ("nodrop", "case", ["sop_np", "sop_nd", "sop2"]), ("nodrop", "proc", ["sop_nd"])]
for variant, level, ws in plan:
    for w in ws:
        M, pt, dr = comb(groups[w], level, variant)
        for v in M:
            i = M.index(v); lo, hi = ci(dr[:, i, 0]); s = f"{variant} {level} {w} {v:17s} {pt[i,0]:+.3f} [{lo:+.3f},{hi:+.3f}]"
            for base in ["adele", "c_lad"]:
                if v != base and base in M:
                    b = M.index(base); l2, h2 = ci(dr[:, i, 0] - dr[:, b, 0]); s += f" | vs {base} {pt[i,0]-pt[b,0]:+.3f} [{l2:+.3f},{h2:+.3f}]"
            lines.append(s)
open(HERE / "score.txt", "w").write("\n".join(lines)); print("\n".join(lines))
