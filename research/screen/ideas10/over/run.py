"""Round 10 over: c_lad + GAP (+PRIOR) + OVER. usage: run.py"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas"); sys.path.insert(0, S + "/rubric"); sys.path.insert(0, S + "/ideas9/gap")
from pathlib import Path
import pandas as pd
import harness as H
from cpulock import cpu_lock
from ladder import Ladder
import prompts as P
from feats import GAP
sys.path.insert(0, str(Path(__file__).parent)); import importlib.util
spec = importlib.util.spec_from_file_location("pf", S + "/ideas10/prior/feats.py"); pf = importlib.util.module_from_spec(spec); spec.loader.exec_module(pf)
PRIOR = pf.PRIOR
I3 = Path(S + "/ideas3"); HERE = Path(__file__).parent; REPO = H.REPO
def tab(path):
    d = pd.read_csv(path); return d.set_index(d.columns[0])
TYPES = [f"n_{t}" for t in P.STEP_TYPES]; GLOB = ["max_depth", "refusal_conditions", "needs_hidden_data", "p_refusal"]
def make_preds(src, M, bench):
    sop = bench == "sopbench"
    ad = tab(REPO / "data" / bench / "features" / "adele.csv")
    rub = tab(S + f"/rubric/features/rub_proc_{bench}.csv")
    plan = tab(I3 / f"plan_tools/features/plan3_{bench}.csv")
    cmp_ = tab(I3 / "compile_consistency/cons_sopbench.csv")[["len_mean", "branch_mean"]] if sop else None
    g = tab(S + f"/ideas9/gap/gap_{bench}.csv")
    pr = tab(S + f"/ideas10/prior/prior_{bench}.csv")[PRIOR]
    ov = tab(HERE / f"over_{bench}.csv")
    base = [ad, plan] + ([cmp_] if sop else [])
    def L(name, parts): return Ladder(2, TYPES, GLOB, rub, extra_table=pd.concat(parts, axis=1), name=name)
    return {"g1_clad_gap": L("g1_clad_gap", base + [g[GAP]]),
            "pr2_cladgap_prior": L("pr2_cladgap_prior", base + [g[GAP], pr]),
            "o1_cladgap_prior_over": L("o1_cladgap_prior_over", base + [g[GAP], pr, ov]),
            "o2_cladgap_over": L("o2_cladgap_over", base + [g[GAP], ov])}
if __name__ == "__main__":
    with cpu_lock("over"):
        H.run_all(HERE / "out", make_preds, benches=["sopbench", "tau2"], schemes=["new_procedures", "new_domain"])
    print("run done")
