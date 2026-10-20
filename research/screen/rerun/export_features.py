"""Export cached prototype features (old unseeded code, same feature values) into rerun/features/<bench>/ in repo formats.
No recomputation: hidden states (Qwen3-8B-Base 8-bit, probe_L*.npz), entropy.csv, bpm_complexity.csv, llm_forecast.csv were cached."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import shutil, numpy as np, pandas as pd
from pathlib import Path
SRC = Path(ZR.REPO + "/data")
REPO = Path(ZR.REPO + "/data")
OUT = Path(__file__).parent / "features"
for b in ["sopbench", "tau2", "tauk_banking"]:
    o = OUT / b; o.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(REPO / b / "tasks.csv").case_id.tolist()
    for L in (18, 36):
        z = np.load(SRC / b / "features" / f"probe_L{L}.npz", allow_pickle=True)
        pos = {t: i for i, t in enumerate(z["task_ids"])}
        assert set(ids) <= set(pos), (b, L)
        np.savez_compressed(o / f"probe_L{L}.npz", task_ids=np.array(ids, dtype=object), X=z["X"][[pos[t] for t in ids]].astype(np.float32))
    for n in ["entropy", "bpm_complexity", "llm_forecast"]:
        p = SRC / b / "features" / f"{n}.csv"
        if not p.exists(): print("absent", b, n); continue
        df = pd.read_csv(p).rename(columns={"task_id": "case_id"})
        assert set(ids) <= set(df.case_id), (b, n)
        df = df.set_index("case_id").loc[ids].reset_index()
        assert df.drop(columns="case_id").notna().all().all(), (b, n)
        df.to_csv(o / f"{n}.csv", index=False)
    print(b, "ok", len(ids))
