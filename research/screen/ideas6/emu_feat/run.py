"""emu_feat: c_lad + emulation features. Variants fixed before the run:
 e1_clad_emu    c_lad + emu_est (0.35 fp + 0.65 fr, mean of two smart agents)           L0+E
 e2_clad_mixplan c_lad + mix_plan (p_plan x fr + (1-p_plan) x fp, p_plan = plan_refuse)  L1+E
 e3_l2e_mix     rubric ladder + mix_pre only (no ADeLe/plan/cmp)                          L2+E (separate track)
 e4_clad_mixpre c_lad + mix_pre                                                           L2+E (separate track)
tau2 has no synthetic runs: e1, e2, e4 use the c_lad tau2 configuration unchanged; e3 uses the plain rubric ladder on tau2."""
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
I3 = Path(S + "/ideas3"); HERE = Path(S + "/ideas6/emu_feat"); REPO = H.REPO
def tab(path):
    d = pd.read_csv(path); return d.set_index(d.columns[0])
TYPES = [f"n_{t}" for t in P.STEP_TYPES]; GLOB = ["max_depth", "refusal_conditions", "needs_hidden_data", "p_refusal"]
EM = pd.read_csv(HERE / "emu_case_features.csv").set_index("case_id")
for c in ["emu_est", "mix_plan", "mix_pre"]:
    EM[c] = EM[c].fillna(EM.drop_duplicates("procedure_id")[c].mean())   # two broken bank procedures have no synthetic runs
def make_preds(src, M, bench):
    sop = bench == "sopbench"
    ad = tab(REPO / "data" / bench / "features" / "adele.csv")
    rub = tab(S + f"/rubric/features/rub_proc_{bench}.csv")
    plan = tab(I3 / f"plan_tools/features/plan3_{bench}.csv")
    base = [ad, plan] + ([tab(I3 / "compile_consistency/cons_sopbench.csv")[["len_mean", "branch_mean"]]] if sop else [])
    def L(name, parts): return Ladder(2, TYPES, GLOB, rub, extra_table=pd.concat(parts, axis=1) if parts else None, name=name)
    def add(c): return base + ([EM[[c]]] if sop else [])
    return {"e1_clad_emu": L("e1", add("emu_est")), "e2_clad_mixplan": L("e2", add("mix_plan")),
            "e3_l2e_mix": L("e3", [EM[["mix_pre"]]] if sop else []), "e4_clad_mixpre": L("e4", add("mix_pre"))}
if __name__ == "__main__":
    out = HERE / "out"
    with cpu_lock("emu_feat"):
        H.run_all(out, make_preds, benches=["sopbench", "tau2"], schemes=["new_procedures", "new_domain"])
    print("run done")
