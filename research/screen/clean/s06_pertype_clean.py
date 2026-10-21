"""Check (iv) on variant A: offline routing, one ability per agent (adele) vs declared per-type model logit_all
vs best agent per domain from training. Reuses pertype/pertype_models.py and the route() of pertype/s06_routing.py.
usage: s06_pertype_clean.py A run | A route"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, json
sys.path.insert(0, ZR.SCREEN)
sys.path.insert(0, ZR.SCREEN + "/pertype")
import harness as H
import pertype_models as PM
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(ZR.SCREEN + "/clean")
PT = Path(ZR.SCREEN + "/pertype")
var, mode = sys.argv[1], sys.argv[2]
root, res = HERE / f"data_{var}", HERE / f"results_{var}"
H.RB.DATA = root; H.RB.RESULTS = res
OUT = res / "pertype"
if mode == "run":
    OUT.mkdir(parents=True, exist_ok=True)
    def mk(src, M):
        m = {"logit_all": PM.TypeLogit(systems=["fam", "tree", "dom"]), "share_dom": PM.TypeShare(["dom"])}
        for x in m.values(): x.bind(src)
        return m
    H.run_all(OUT, mk, benches=["sopbench"], schemes=["new_procedures", "new_domain"])
else:
    os.chdir(PT)
    s = open(PT / "s06_routing.py").read().split("res = {}")[0]
    s = s.replace('RUN = Path("run/sopbench")', "")
    exec(s)
    tasks = pd.read_csv(root / "sopbench/tasks.csv")
    out = {}
    for sc in ["new_procedures", "new_domain"]:
        obs = pd.read_csv(OUT / "sopbench" / f"obs_{sc}.csv.gz")
        out[f"sopbench|{sc}"] = route(obs, tasks, ["adele", "share_dom", "logit_all"], "sop")
    json.dump(out, open(HERE / f"routing_{var}.json", "w"), indent=1)
    for k, r in out.items():
        print(k)
        for m, v in r.items():
            print(f"  {m:22s} {v['mean']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}]")
