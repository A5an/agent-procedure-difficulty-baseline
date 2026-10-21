"""small_model run. usage: run.py [debug]. Variants fixed in PREREG.md."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas"); sys.path.insert(0, S + "/rubric"); sys.path.insert(0, S + "/ideas7/entropy_combo")
from pathlib import Path
import pandas as pd
import harness as H
from cpulock import cpu_lock
from ladder import Ladder
import prompts as P
from famladder import FamLadder
I3 = Path(S + "/ideas3"); HERE = Path(S + "/ideas7/small_model"); REPO = H.REPO
def tab(path):
    d = pd.read_csv(path); return d.set_index(d.columns[0])
TYPES = [f"n_{t}" for t in P.STEP_TYPES]; GLOB = ["max_depth", "refusal_conditions", "needs_hidden_data", "p_refusal"]
SM_SOP = ["sm_margin", "sm_ent", "sm_ptgt", "sm_rank_frac"]; PRI_SOP = ["sm_pol_nll", "sm_pol_nll_max", "sm_msg_nll"]
SM_T2 = ["sm_wr_margin", "sm_ent", "sm_pmax", "sm_p_write"]
def make_preds(src, M, bench):
    sop = bench == "sopbench"
    ad = tab(REPO / "data" / bench / "features" / "adele.csv")
    rub = tab(S + f"/rubric/features/rub_proc_{bench}.csv")
    plan = tab(I3 / f"plan_tools/features/plan3_{bench}.csv")
    cmp_ = tab(I3 / "compile_consistency/cons_sopbench.csv")[["len_mean", "branch_mean"]] if sop else None
    sm = tab(HERE / f"sm_{'sopbench' if sop else 'tau2'}.csv")
    smf = sm[SM_SOP if sop else SM_T2]; pri = sm[PRI_SOP] if sop else None
    def L(name, parts): return Ladder(2, TYPES, GLOB, rub, extra_table=pd.concat(parts, axis=1), name=name)
    base = [ad, plan] + ([cmp_] if sop else [])
    def F(name, fams): return FamLadder(bench, rub, TYPES, GLOB, fams, name, per_family=True)
    fa = {"ADELE": ad, "PLAN": plan, **({"CMP": cmp_} if sop else {}), "SM": smf}
    return {"v_a_clad_sm": L("v_a", base + [smf]),
            "v_b_clad_sm_pri": L("v_b", base + [smf] + ([pri] if sop else [])),
            "v_c_fam_sm": F("v_c", fa),
            "d_sm_only": L("d_sm_only", [smf] + ([pri] if sop else []))}
if __name__ == "__main__":
    dbg = len(sys.argv) > 1
    out = HERE / ("out_debug" if dbg else "out_h")
    with cpu_lock("small_model_h"):
        H.run_all(out, make_preds, benches=["sopbench"] if dbg else ["sopbench", "tau2"], schemes=["new_domain"] if dbg else ["new_procedures", "new_domain"])
    print("run done")
