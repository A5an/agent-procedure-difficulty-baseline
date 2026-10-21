"""msg_facts harness run. Variants fixed before the first full run:
 f1_feat_adele  Ladder with ADeLe + msg_facts features only (SOPBench; tau2 = c_lad configuration unchanged)
 f2_clad_feat   c_lad + msg_facts features (SOPBench; tau2 = c_lad configuration unchanged)
 f3_clad_pperf  c_lad + p_perf only (single feature, SOPBench; tau2 = c_lad)"""
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
I3 = Path(S + "/ideas3"); HERE = Path(S + "/ideas6/msg_facts"); REPO = H.REPO
def tab(path):
    d = pd.read_csv(path); return d.set_index(d.columns[0])
TYPES = [f"n_{t}" for t in P.STEP_TYPES]; GLOB = ["max_depth", "refusal_conditions", "needs_hidden_data", "p_refusal"]
FEAT = ["p_perf", "n_pinned", "n_prior", "forced_refuse", "entropy"]
MF = pd.read_csv(HERE / "msg_features.csv").set_index("case_id")
for c in FEAT: MF[c] = MF[c].fillna(MF[c].mean())
def make_preds(src, M, bench):
    sop = bench == "sopbench"
    ad = tab(REPO / "data" / bench / "features" / "adele.csv")
    rub = tab(S + f"/rubric/features/rub_proc_{bench}.csv")
    plan = tab(I3 / f"plan_tools/features/plan3_{bench}.csv")
    base = [ad, plan] + ([tab(I3 / "compile_consistency/cons_sopbench.csv")[["len_mean", "branch_mean"]]] if sop else [])
    def L(name, parts): return Ladder(2, TYPES, GLOB, rub, extra_table=pd.concat(parts, axis=1) if parts else None, name=name)
    return {"f1_feat_adele": L("f1", [ad] + ([MF[FEAT]] if sop else [])) if sop else L("f1", base),
            "f2_clad_feat": L("f2", base + ([MF[FEAT]] if sop else [])),
            "f3_clad_pperf": L("f3", base + ([MF[["p_perf"]]] if sop else []))}
if __name__ == "__main__":
    out = HERE / "out"
    with cpu_lock("msg_facts"):
        H.run_all(out, make_preds, benches=["sopbench", "tau2"], schemes=["new_procedures", "new_domain"])
    print("run done")
