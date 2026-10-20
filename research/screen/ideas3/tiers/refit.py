"""Refit baseline (adele) and lad2_adele with the IRT fitted only on one agent tier. usage: refit.py <tier> [benches]"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, json
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas"); sys.path.insert(0, S + "/rubric")
from pathlib import Path
import pandas as pd
import harness as H
from ladder import Ladder
from cpulock import cpu_lock
import tiers as TI
HERE = Path(__file__).parent
tier = sys.argv[1]
benches = sys.argv[2].split(",") if len(sys.argv) > 2 else ["sopbench", "tau2"]
DD = HERE / "data" / tier; RES = HERE / "irt_cache" / tier
for b in benches:
    T, Y, acc = TI.tiers(b)
    keep = set(T[tier]); d = DD / b; d.mkdir(parents=True, exist_ok=True)
    for f in ["tasks.csv", "features", "statements.jsonl"]:
        if not (d / f).exists(): os.symlink(H.REPO / "data" / b / f, d / f)
    (d / "irt/1d_1pl").mkdir(parents=True, exist_ok=True)
    for f in ["abilities.csv", "items.csv"]:
        if not (d / "irt/1d_1pl" / f).exists(): os.symlink(H.REPO / "data/" / b / "irt/1d_1pl" / f, d / "irt/1d_1pl" / f)
    with open(d / "responses.jsonl", "w") as o:
        for l in open(H.REPO / "data" / b / "responses.jsonl"):
            r = json.loads(l)
            if r["subject_id"] in keep: o.write(l)
H.RB.DATA = DD; H.RB.RESULTS = RES
FEAT = S + "/rubric/features"
TYPES = None
import prompts as P
TYPES = [f"n_{t}" for t in P.STEP_TYPES]; GLOB = ["max_depth", "refusal_conditions", "needs_hidden_data", "p_refusal"]
def make_preds(src, M, bench):
    p = pd.read_csv(f"{FEAT}/rub_proc_{bench}.csv").set_index("case_id")
    a = pd.read_csv(H.REPO / "data" / bench / "features" / "adele.csv"); a = a.set_index(a.columns[0])
    return {"lad2_adele": Ladder(2, TYPES, GLOB, p, extra_table=a, name="lad2_adele")}
out = HERE / "refit" / tier
with cpu_lock("tiers_refit_" + tier):
    H.run_all(out, make_preds, benches=benches, schemes=["new_procedures", "new_domain"])
print("done", tier)
