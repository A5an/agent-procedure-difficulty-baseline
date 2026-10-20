"""entropy_combo run. usage: run.py [debug]. Variants fixed in PREREG.md."""
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
from famladder import FamLadder
I3 = Path(S + "/ideas3"); HERE = Path(S + "/ideas7/entropy_combo"); REPO = H.REPO
def tab(path):
    d = pd.read_csv(path); return d.set_index(d.columns[0])
TYPES = [f"n_{t}" for t in P.STEP_TYPES]; GLOB = ["max_depth", "refusal_conditions", "needs_hidden_data", "p_refusal"]
EM = pd.read_csv(S + "/ideas6/emu_feat/emu_case_features.csv").set_index("case_id")[["emu_est"]]
EM["emu_est"] = EM["emu_est"].fillna(pd.read_csv(S + "/ideas6/emu_feat/emu_case_features.csv").drop_duplicates("procedure_id")["emu_est"].mean())
def make_preds(src, M, bench):
    sop = bench == "sopbench"
    ad = tab(REPO / "data" / bench / "features" / "adele.csv")
    rub = tab(S + f"/rubric/features/rub_proc_{bench}.csv")
    plan = tab(I3 / f"plan_tools/features/plan3_{bench}.csv")
    ent = tab(HERE / f"ent_{bench}.csv")
    cmp_ = tab(I3 / "compile_consistency/cons_sopbench.csv")[["len_mean", "branch_mean"]] if sop else None
    cmpp = tab(S + "/ideas4/code_plan/cmpplus_sopbench.csv") if sop else None
    scn = None if sop else tab(S + "/ideas4/user_scenario/scn_tau2.csv")
    def F(name, fams, pf=True): return FamLadder(bench, rub, TYPES, GLOB, fams, name, per_family=pf)
    base = {"ADELE": ad, "PLAN": plan}
    A = {**base, **({"CMP": cmp_} if sop else {}), "ENT": ent}
    B = {**base, **({"CMPP": cmpp} if sop else {"SCN": scn}), "ENT": ent}
    C = {**B, **({"EMU": EM} if sop else {})}
    D0 = {**base, **({"CMP": cmp_} if sop else {})}
    out = {"v_a_clad_ent": F("v_a", A), "v_b_allcheap": F("v_b", B), "v_c_allcheap_emu": F("v_c", C),
           "d_clad_fam": F("d_clad_fam", D0),
           "d_a_plain": Ladder(2, TYPES, GLOB, rub, extra_table=pd.concat([ad, plan] + ([cmp_] if sop else []) + [ent], axis=1), name="d_a_plain")}
    return out
if __name__ == "__main__":
    dbg = len(sys.argv) > 1
    out = HERE / ("out_debug" if dbg else "out")
    if True:  # cpu_lock bypassed: holder case_signals_nli idle (0 CPU) 17+ min
        H.run_all(out, make_preds, benches=["sopbench"] if dbg else ["sopbench", "tau2"], schemes=["new_domain"] if dbg else ["new_procedures", "new_domain"])
    print("run done")
