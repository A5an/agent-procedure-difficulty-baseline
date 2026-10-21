import sys, pickle, numpy as np
from pathlib import Path
HERE = Path(__file__).parent
mode = sys.argv[1]
res = pickle.load(open(HERE / f"score_{mode}.pkl", "rb"))
names = sorted({k[0] for k in res})
ci = lambda d: (np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5))
def comb(name, level, which):
    """which: 'pool4' | 'sop' | 'tau2' | 'sop_nd' ... returns M, TG, point[M,T], draws[B,M,T]"""
    sel = {"pool4": [("sopbench", "new_procedures"), ("sopbench", "new_domain"), ("tau2", "new_procedures"), ("tau2", "new_domain")],
           "sop": [("sopbench", "new_procedures"), ("sopbench", "new_domain")],
           "tau2": [("tau2", "new_procedures"), ("tau2", "new_domain")],
           "sop_np": [("sopbench", "new_procedures")], "sop_nd": [("sopbench", "new_domain")],
           "tau2_np": [("tau2", "new_procedures")], "tau2_nd": [("tau2", "new_domain")]}[which]
    items = [res[(name, b, s, level)] for b, s in sel if (name, b, s, level) in res]
    if not items: return None
    M = items[0][0]
    if any(it[0] != M for it in items): # intersect methods
        M = [m for m in M if all(m in it[0] for it in items)]
    idx = lambda it: [it[0].index(m) for m in M]
    pt = np.mean([it[2][idx(it)] for it in items], axis=0)
    dr = np.mean([it[3][:, idx(it)] for it in items], axis=0)
    return M, items[0][1], pt, dr
def table(name, level, which, base="adele", targets=None, show=None):
    r = comb(name, level, which)
    if r is None: return ""
    M, TG, pt, dr = r
    tix = [TG.index(t) for t in (targets or TG)]
    out = [f"### {name} | {level}-level | {which}  (rho [95% CI over procedures]; in brackets after '|' paired diff vs {base})"]
    out.append(f"{'method':16s}" + "".join(f"{TG[j]:>34s}" for j in tix))
    for i, m in enumerate(M):
        if show and m not in show: continue
        row = f"{m:16s}"
        for j in tix:
            lo, hi = ci(dr[:, i, j]); s = f"{pt[i,j]:+.3f} [{lo:+.2f},{hi:+.2f}]"
            if m != base and base in M:
                b = M.index(base); d = dr[:, i, j] - dr[:, b, j]; l2, h2 = ci(d)
                s += f" |{pt[i,j]-pt[b,j]:+.2f}[{l2:+.2f},{h2:+.2f}]"
            row += f"{s:>34s}"
        out.append(row)
    return "\n".join(out)
def did(name, level, which, meth, tgt_a, tgt_b, base="adele"):
    """(meth - base) on tgt_a minus (meth - base) on tgt_b"""
    M, TG, pt, dr = comb(name, level, which)
    i, b, a, c_ = M.index(meth), M.index(base), TG.index(tgt_a), TG.index(tgt_b)
    f = lambda x: (x[..., i, a] - x[..., b, a]) - (x[..., i, c_] - x[..., b, c_])
    lo, hi = ci(f(dr)); return f(pt), lo, hi
if __name__ == "__main__":
    for name in names:
        for level in ["case", "proc"]:
            for which in ["pool4", "sop", "tau2"] if level == "case" else ["sop", "tau2", "sop_nd"]:
                print(table(name, level, which)); print()
