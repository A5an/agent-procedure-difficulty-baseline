import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas")
import pandas as pd, harness as H, metrics_extra as X
from cpulock import cpu_lock
from pathlib import Path
out = Path(S + "/ideas6/synth_label/out"); V = ["sl0_clad","sl1_feat_adele","sl2_clad_feat","sl3_clad_p"]
(out/"tau2").mkdir(exist_ok=True)
for sch in ["new_procedures","new_domain"]:
    d = pd.read_csv(S+f"/ideas3/combo/out_main/tau2/obs_{sch}.csv.gz")
    for v in V: d[v] = d["c_lad"]
    d.drop(columns=["c_L0","c_L1"]).to_csv(out/"tau2"/f"obs_{sch}.csv.gz", index=False)
    import shutil; shutil.copy(S+f"/ideas3/combo/out_main/tau2/betas_{sch}.csv", out/"tau2"/f"betas_{sch}.csv")
DESC = {v: "synth_label (tau2 = c_lad copy)" for v in V}; DESC["c_lad"] = "c_lad tau2 part"
with cpu_lock("synth_label"):
    H.evaluate_all(out, DESC)
    X.report(out, V)
