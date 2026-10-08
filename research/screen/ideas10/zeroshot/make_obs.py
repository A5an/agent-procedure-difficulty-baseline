"""Wrap zero-shot scores as constant-over-agents probabilities so the protocol scorer can bootstrap them."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import pandas as pd, numpy as np
from scipy.special import expit
from pathlib import Path
I10 = ZR.SCREEN + "/ideas10"
for b in ["sopbench", "tau2"]:
    d = pd.read_csv(f"{I10}/fcd/d_{b}.csv").set_index("case_id").d_diff
    s = pd.read_csv(f"{I10}/fc/fc_{b}.csv").set_index("case_id").fc_diff
    for sch in ["new_procedures", "new_domain"]:
        m = pd.read_csv(f"{I10}/fc/out/{b}/obs_{sch}.csv.gz", usecols=["seed", "fold", "agent", "case_id", "y", "constant", "oracle", "adele"])
        m["D_zero"] = expit(-m.case_id.map(d).fillna(d.median())); m["S1_zero"] = expit(-m.case_id.map(s).fillna(s.median()))
        Path(f"out/{b}").mkdir(parents=True, exist_ok=True); m.to_csv(f"out/{b}/obs_{sch}.csv.gz", index=False)
print("ok")
