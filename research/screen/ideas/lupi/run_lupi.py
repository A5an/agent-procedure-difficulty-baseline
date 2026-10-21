import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,ZR.SCREEN); sys.path.insert(0,ZR.SCREEN + "/ideas")
import harness as H, metrics_extra as X, lupi_lib as L, pandas as pd, numpy as np, pickle
from cpulock import cpu_lock
OUT = ZR.SCREEN + "/ideas/lupi/out"
HERE = ZR.SCREEN + "/ideas/lupi"
RUB = ZR.SCREEN + "/rubric/out/run"
def make(s, M, bench):
    return {"x_ridge": H.ridge(s_src[bench]) if False else None}
def mk(s, M, bench):
    from pathlib import Path
    src = H.CSVFeatureSource(Path(f"{HERE}/X_{bench}.csv"), name="xfeat")
    return {"x_ridge": H.ridge(src), "lupi": L.Base(bench, "lupi"), "mix2": L.Base(bench, "mix2"), "lupi_mix2": L.Base(bench, "lupi_mix2")}
cols = ["x_ridge", "lupi", "mix2", "lupi_mix2"]
desc = {"x_ridge": "ridge on 18 ADeLe + 12 rubric", "lupi": "two-stage LUPI", "mix2": "allow/refuse mixture", "lupi_mix2": "mixture with stage-1 predictions"}
combos = [("lupi","adele"), ("mix2","adele"), ("mix2","lad2_adele")]
with cpu_lock("lupi"):
    H.run_all(OUT, mk, benches=["sopbench","tau2"], schemes=["new_procedures","new_domain"])
    pickle.dump(L.DIAG, open(f"{HERE}/diag.pkl","wb"))
    H.evaluate_all(OUT, desc, combos=[("lupi","adele"),("mix2","adele")])
    X.report(OUT, ["adele"]+cols, extra_dirs={"lad2_adele": RUB, "grp_proc_adele": RUB},
             combos=[("lupi","adele"),("mix2","adele"),("mix2","lad2_adele")])
