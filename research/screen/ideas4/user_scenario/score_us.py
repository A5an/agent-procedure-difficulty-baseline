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
HERE = Path(S + "/ideas4/user_scenario"); CB = S + "/ideas3/combo/out_main"
V = ["scn_adele", "cls_adele", "clad_scn", "clad_scn_cls"]
srcs = {"adele": SC.R, "lad2_adele": S + "/rubric/out/run", "c_lad": CB, **{v: str(HERE / "out") for v in V}}
TG = ["all_irt"]; rng = np.random.default_rng(20261006); res = {}
with cpu_lock("us_score"):
    W = SC.build(srcs, TG)
    for bench in ["tau2", "sopbench"]:
        for sch in ["new_procedures", "new_domain"]:
            D = W[(bench, sch)]
            M = [m for m in srcs if m in D]
            D = D.copy(); procs = sorted(D.procedure_id.unique()); code = {p: i for i, p in enumerate(procs)}; D["u"] = D.procedure_id.map(code)
            by_dom = [np.array(sorted(g.u.unique())) for _, g in D.groupby("domain")]
            pt, dr = SC.boot(D, ["fold", "domain"], M, TG, by_dom, len(procs), np.random.default_rng(rng.integers(1 << 31)))
            res[(bench, sch)] = (M, pt, dr); print(bench, sch, M, flush=True)
pickle.dump(res, open(HERE / "score.pkl", "wb"))
ci = lambda d: (np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5))
def get(b, s, m):
    M, pt, dr = res[(b, s)]; i = M.index(m); return pt[i, 0], dr[:, i, 0]
L = []
def line(tag, v, pt, dr, bases):
    lo, hi = ci(dr); s = f"{tag:12s} {v:13s} {pt:+.3f} [{lo:+.3f},{hi:+.3f}]"
    for bn, (bp, bd) in bases.items():
        l2, h2 = ci(dr - bd); s += f" | vs {bn} {pt - bp:+.3f} [{l2:+.3f},{h2:+.3f}]"
    L.append(s)
allm = ["adele", "lad2_adele", "c_lad"] + V
for tag, sch_list in [("tau2_np", ["new_procedures"]), ("tau2_nd", ["new_domain"]), ("tau2_both", ["new_procedures", "new_domain"])]:
    for v in allm:
        parts = [get("tau2", s, v) for s in sch_list]; pt = np.mean([p[0] for p in parts]); dr = np.mean([p[1] for p in parts], 0)
        bases = {}
        for bn in ["adele", "c_lad"]:
            if bn != v:
                bp = [get("tau2", s, bn) for s in sch_list]; bases[bn] = (np.mean([p[0] for p in bp]), np.mean([p[1] for p in bp], 0))
        line(tag, v, pt, dr, bases)
# pooled: sop part reused (ADeLe-only variants: baseline adele sop; ladder variants and c_lad: c_lad sop; adele: adele; lad2: lad2)
sopsrc = {"adele": "adele", "lad2_adele": "lad2_adele", "c_lad": "c_lad", "scn_adele": "adele", "cls_adele": "adele", "clad_scn": "c_lad", "clad_scn_cls": "c_lad"}
def pool(v):
    ps = [get("tau2", s, v) for s in ["new_procedures", "new_domain"]] + [get("sopbench", s, sopsrc[v]) for s in ["new_procedures", "new_domain"]]
    return np.mean([p[0] for p in ps]), np.mean([p[1] for p in ps], 0)
P = {v: pool(v) for v in allm}
for v in allm:
    bases = {bn: P[bn] for bn in ["adele", "c_lad"] if bn != v}
    line("pool4*", v, P[v][0], P[v][1], bases)
for s in ["new_procedures", "new_domain"]:
    L.append(f"per-scenario tau2 {s}: " + " ".join(f"{v}={get('tau2', s, v)[0]:+.3f}" for v in allm))
open(HERE / "score.txt", "w").write("\n".join(L)); print("\n".join(L))
