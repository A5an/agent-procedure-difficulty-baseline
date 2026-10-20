"""Entropy profile features from baseline/data/<bench>/entropy_profiles.jsonl (no model calls)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, numpy as np, pandas as pd
R = ZR.REPO + "/data"
NB = 8
def feats(H):
    H = np.asarray(H, float); L = len(H)
    q = np.quantile(H, [.1, .25, .5, .75, .9])
    bins = np.array_split(H, NB); bm = [b.mean() for b in bins]
    w = max(1, min(L, 64)); cs = np.convolve(H, np.ones(w) / w, mode="valid")
    d = {"e_mean": H.mean(), "e_sd": H.std(), "e_var": H.var(), "e_q10": q[0], "e_q25": q[1], "e_q50": q[2], "e_q75": q[3], "e_q90": q[4],
         "e_maxseg": cs.max(), "e_maxseg_pos": cs.argmax() / max(1, len(cs) - 1), "e_loglen": np.log1p(L)}
    for i, m in enumerate(bm): d[f"e_bin{i}"] = m
    return d
for b in ["sopbench", "tau2"]:
    rows = [json.loads(l) for l in open(f"{R}/{b}/entropy_profiles.jsonl")]
    df = pd.DataFrame([{"case_id": r["case_id"], **feats(r["H"])} for r in rows]).set_index("case_id")
    tasks = pd.read_csv(f"{R}/{b}/tasks.csv").case_id
    print(b, df.shape, "missing", len(set(tasks) - set(df.index)), "nan", int(df.isna().sum().sum()), "len median", np.median([len(r["H"]) for r in rows]))
    df.to_csv(f"ent_{b}.csv")
