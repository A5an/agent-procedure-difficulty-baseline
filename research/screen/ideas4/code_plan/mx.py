import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys; S=ZR.SCREEN
sys.path.insert(0,S); sys.path.insert(0,S+"/ideas")
import metrics_extra as X
from cpulock import cpu_lock
m=sys.argv[1]; cols={"tau2":["cp_adele","c_lad_cp","c_lad_cmpp"],"sopL0":["c_lad_cmpp"],"main":["cp_adele","c_lad_cp","c_lad_cmpp"]}[m]
with cpu_lock("code_plan_mx"):
    X.report(S+"/ideas4/code_plan/out_"+m, cols, extra_dirs={"lad2_adele":S+"/rubric/out/run","c_lad":S+"/ideas3/combo/out_main"}, combos=[])
