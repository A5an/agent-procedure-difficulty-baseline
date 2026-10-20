"""Round 9 probe harness run, pre-registered in PREREG.md. usage: run.py main|banking"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas"); sys.path.insert(0, S + "/rubric")
from pathlib import Path
import pandas as pd
import harness as H
from cpulock import cpu_lock
from ladder import Ladder
import prompts as P
I3 = Path(S + "/ideas3"); HERE = Path(S + "/ideas9/probe"); REPO = H.REPO
def tab(path):
    d = pd.read_csv(path); return d.set_index(d.columns[0])
TYPES = [f"n_{t}" for t in P.STEP_TYPES]; GLOB = ["max_depth", "refusal_conditions", "needs_hidden_data", "p_refusal"]
def make_preds(src, M, bench):
    sop = bench == "sopbench"
    ad = tab(REPO / "data" / bench / "features" / "adele.csv")
    rub = tab(S + f"/rubric/features/rub_proc_{bench}.csv")
    pr, prc, sca, ln = [tab(HERE / f"{n}_{bench}.csv") for n in ["probe", "probec", "scal", "len"]]
    def L(name, parts): return Ladder(2, TYPES, GLOB, rub, extra_table=pd.concat(parts, axis=1), name=name)
    if bench == "tauk_banking":
        return {"q0_adele_lad": L("q0", [ad]), "q1_probe": L("q1", [ad, pr]), "q2_len": L("q2", [ad, ln])}
    plan = tab(I3 / f"plan_tools/features/plan3_{bench}.csv")
    cmp_ = tab(I3 / "compile_consistency/cons_sopbench.csv")[["len_mean", "branch_mean"]] if sop else None
    base = [ad, plan] + ([cmp_] if sop else [])
    return {"t0_clad": L("t0_clad", base), "p0_len": L("p0_len", base + [ln]), "p1_probe": L("p1_probe", base + [pr]),
            "p3_scalars": L("p3_scalars", base + [sca[["shift12", "shift16", "shift20", "shift24", "shift28", "shift32", "nll_cond", "pmi"]]]),
            "p4_centred": L("p4_centred", base + [prc])}
if __name__ == "__main__":
    which = sys.argv[1]
    with cpu_lock("probe_" + which):
        if which == "main":
            H.run_all(HERE / "out", make_preds, benches=["sopbench", "tau2"], schemes=["new_procedures", "new_domain"])
        else:
            H.run_all(HERE / "out_banking", make_preds, benches=["tauk_banking"], schemes=["random_cases"])
    print("run done")
