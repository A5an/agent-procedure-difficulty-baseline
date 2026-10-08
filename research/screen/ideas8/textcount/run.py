"""Round 8 follow-up: c_lad plus pure-code request counts. usage: run.py"""
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
I3 = Path(S + "/ideas3"); HERE = Path(S + "/ideas8/textcount"); REPO = H.REPO
def tab(path):
    d = pd.read_csv(path); return d.set_index(d.columns[0])
TYPES = [f"n_{t}" for t in P.STEP_TYPES]; GLOB = ["max_depth", "refusal_conditions", "needs_hidden_data", "p_refusal"]
DIG = ["dig_chars", "dig_nums", "dig_ids"]; CRED = ["cred_maxent", "cred_word"]
def make_preds(src, M, bench):
    sop = bench == "sopbench"
    ad = tab(REPO / "data" / bench / "features" / "adele.csv")
    rub = tab(S + f"/rubric/features/rub_proc_{bench}.csv")
    plan = tab(I3 / f"plan_tools/features/plan3_{bench}.csv")
    cmp_ = tab(I3 / "compile_consistency/cons_sopbench.csv")[["len_mean", "branch_mean"]] if sop else None
    txt = tab(HERE / f"txt_{bench}.csv")
    base = [ad, plan] + ([cmp_] if sop else [])
    def L(name, parts): return Ladder(2, TYPES, GLOB, rub, extra_table=pd.concat(parts, axis=1), name=name)
    return {"t0_clad": L("t0_clad", base),
            "t1_clad_dig": L("t1_clad_dig", base + [txt[DIG]]),
            "t2_clad_dig_cred": L("t2_clad_dig_cred", base + [txt[DIG + CRED]]),
            "t3_adele_dig_cred": L("t3_adele_dig_cred", [ad, txt[DIG + CRED]])}
if __name__ == "__main__":
    with cpu_lock("textcount"):
        H.run_all(HERE / "out", make_preds, benches=["sopbench", "tau2"], schemes=["new_procedures", "new_domain"])
    print("run done")
