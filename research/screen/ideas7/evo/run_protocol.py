"""Full protocol harness: c_lad + finalist features, both benchmarks, both schemes. usage: run_protocol.py [debug]. Variants fixed in FINALISTS.md."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas"); sys.path.insert(0, S + "/rubric"); sys.path.insert(0, S + "/ideas7/entropy_combo")
from pathlib import Path
import pandas as pd
import harness as H
from cpulock import cpu_lock
from ladder import Ladder
import prompts as P
from famladder import FamLadder
I3 = Path(S + "/ideas3"); HERE = Path(S + "/ideas7/evo"); REPO = H.REPO
def tab(path):
    d = pd.read_csv(path); return d.set_index(d.columns[0])
TYPES = [f"n_{t}" for t in P.STEP_TYPES]; GLOB = ["max_depth", "refusal_conditions", "needs_hidden_data", "p_refusal"]
FIN = json.load(open(HERE / "finalists.json"))
def centred(bench, n):
    t = tab(HERE / f"feat_f{n}.csv"); tk = pd.read_csv(REPO / "data" / bench / "tasks.csv").set_index("case_id")
    t = t.loc[tk.index]; return t - t.groupby(tk.domain).transform("mean")
def make_preds(src, M, bench):
    sop = bench == "sopbench"
    ad = tab(REPO / "data" / bench / "features" / "adele.csv")
    rub = tab(S + f"/rubric/features/rub_proc_{bench}.csv")
    plan = tab(I3 / f"plan_tools/features/plan3_{bench}.csv")
    cmp_ = tab(I3 / "compile_consistency/cons_sopbench.csv")[["len_mean", "branch_mean"]] if sop else None
    out = {}
    for f in FIN:
        n = f["n"]; ev = centred(bench, n); ev.columns = [f"evo{n}_{c}" for c in ev.columns]
        out[f"evo_f{n}"] = Ladder(2, TYPES, GLOB, rub, extra_table=pd.concat([ad, plan] + ([cmp_] if sop else []) + [ev], axis=1), name=f"evo_f{n}")
        fams = {"ADELE": ad, "PLAN": plan, **({"CMP": cmp_} if sop else {}), "EVO": ev}
        out[f"evo_f{n}_fam"] = FamLadder(bench, rub, TYPES, GLOB, fams, f"evo_f{n}_fam", per_family=True)
    return out
if __name__ == "__main__":
    dbg = len(sys.argv) > 1
    out = HERE / ("out_debug" if dbg else "out_protocol")
    with cpu_lock("evo_protocol"):
        H.run_all(out, make_preds, benches=["sopbench"] if dbg else ["sopbench", "tau2"], schemes=["new_domain"] if dbg else ["new_procedures", "new_domain"])
    print("run done")
