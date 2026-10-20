import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import pandas as pd
from pathlib import Path
R = Path(ZR.REPO + "/data")
for b in ["sopbench", "tau2", "tauk_banking"]:
    f = R / b / "features"
    a = pd.read_csv(f / "adele.csv").set_index("task_id")
    h = pd.read_csv(f / "human_time.csv").set_index("task_id")
    l = pd.read_csv(f / "length.csv").set_index("task_id")
    o = Path(__file__).parent / "features" / b; o.mkdir(parents=True, exist_ok=True)
    a.join(h, how="left").to_csv(o / "adele_ht.csv")
    a.join(h).join(l).to_csv(o / "adele_ht_len.csv")
    print(b, a.shape, a.join(h).join(l).isna().sum().sum())
