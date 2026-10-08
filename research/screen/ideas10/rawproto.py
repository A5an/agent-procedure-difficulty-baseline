"""Protocol cells for a raw (zero-shot) score: within (seed, fold, domain) Spearman pooled by pairs, mean over seeds."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, pandas as pd, numpy as np
from scipy.stats import spearmanr
R = ZR.REPO + "/data"
OBS = ZR.SCREEN + "/ideas10/fc/out"
def proto(bench, scheme, score):
    m = pd.read_csv(f"{OBS}/{bench}/obs_{scheme}.csv.gz", usecols=["seed", "fold", "case_id"]).drop_duplicates()
    t = pd.read_csv(f"{R}/{bench}/tasks.csv")[["case_id", "domain"]]; t["b"] = t.case_id.map(pd.read_csv(f"{R}/{bench}/irt/1d_1pl/items.csv", index_col=0).b)
    m = m.merge(t, on="case_id"); m["s"] = m.case_id.map(score)
    out = []
    for sd, g in m.groupby("seed"):
        num = den = 0
        for _, h in g.groupby(["fold", "domain"]):
            h = h.dropna(subset=["s"]); n = len(h)
            if n < 3 or h.s.nunique() < 2: 
                if n >= 3: den += n * (n - 1) / 2
                continue
            w = n * (n - 1) / 2; num += spearmanr(h.s, h.b).correlation * w; den += w
        out.append(num / den)
    return float(np.mean(out))
if __name__ == "__main__":
    for name, path, col in [("S1 Sonnet P1", "fc/fc_{b}.csv", "fc_diff"), ("FCX", "fcx/fcx_{b}.csv", "fcx_diff")]:
        res = {}
        for b in ["sopbench", "tau2"]:
            try: sc = pd.read_csv(path.format(b=b)).set_index("case_id")[col]
            except Exception as e: print(name, b, "missing", e); continue
            for s in ["new_procedures", "new_domain"]: res[f"{b}|{s}"] = round(proto(b, s, sc), 3)
        if len(res) == 4: res["pool4"] = round(np.mean(list(res.values())), 3)
        print(name, res)
