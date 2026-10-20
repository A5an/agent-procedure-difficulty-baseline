import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json
sys.path.insert(0, ZR.SCREEN + "/ideas")
import mt
import harness as H, metrics_extra as X
from cpulock import cpu_lock
OUT = ZR.SCREEN + "/ideas/modeltricks/out"
RUB = ZR.SCREEN + "/rubric/out/run"
desc = {"wd_center_adele": "ADeLe, within-domain centered", "wd_z_adele": "ADeLe, within-domain z-scored",
        "wd_center_adele_rub": "ADeLe+rubric, within-domain centered", "xbench_adele": "ADeLe, borrow other benchmarks",
        "xbench_adele_rub": "ADeLe+rubric, borrow other benchmarks", "stack4": "NNLS stack of 4 ridges",
        "cat_adele_rub": "plain ridge ADeLe+rubric"}
combos = [("wd_center_adele_rub", "adele"), ("stack4", "adele")]
with cpu_lock("modeltricks"):
    H.run_all(OUT, mt.make_preds, benches=["sopbench", "tau2"], schemes=["new_procedures", "new_domain"])
    H.evaluate_all(OUT, desc, combos=combos)
    json.dump(mt.STACK_LOG, open(OUT + "/stack_log.json", "w"))
    X.report(OUT, ["adele"] + list(desc), extra_dirs={"lad2_adele": RUB, "grp_proc_adele": RUB},
             combos=combos + [("wd_center_adele_rub", "lad2_adele"), ("stack4", "lad2_adele")])
