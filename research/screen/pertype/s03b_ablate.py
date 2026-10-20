"""Which type system carries the agent x type gain: stand-alone logistic with one system at a time (interaction vs matched main-effect control)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, pandas as pd
import harness as H
import pertype_models as PM
from pathlib import Path
OUT = Path(ZR.SCREEN + "/pertype/run_abl"); OUT.mkdir(exist_ok=True)
def mk(src, M):
    ms = {}
    for s in ["fam", "tree", "dom"]:
        ms["main_" + s] = PM.StandAlone([s], False); ms["inter_" + s] = PM.StandAlone([s], True)
    ms["main_famtree"] = PM.StandAlone(["fam", "tree"], False); ms["inter_famtree"] = PM.StandAlone(["fam", "tree"], True)
    for m in ms.values(): m.bind(src)
    return ms
H.run_all(OUT, mk, benches=["sopbench"], schemes=["new_procedures", "new_domain"])
