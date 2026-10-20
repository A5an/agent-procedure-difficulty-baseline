"""case_signals harness run (SOPBench only). Variants fixed before any harness run (see PREREG in report):
 s1_nli, s2_ret, s3_act, s4_bnd: c_lad + one family; s5_all: c_lad + all four. Same Ladder (level 2, common penalty) as ideas4/ablation."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas"); sys.path.insert(0, S + "/rubric"); sys.path.insert(0, S + "/ideas7/case_signals")
from pathlib import Path
import numpy as np, pandas as pd
import harness as H
from cpulock import cpu_lock
from ladder import Ladder
import prompts as P
import skipf
I3 = Path(S + "/ideas3"); HERE = Path(S + "/ideas7/case_signals"); REPO = H.REPO
def tab(path):
    d = pd.read_csv(path); return d.set_index(d.columns[0])
TYPES = [f"n_{t}" for t in P.STEP_TYPES]; GLOB = ["max_depth", "refusal_conditions", "needs_hidden_data", "p_refusal"]
def fam(k):
    d = pd.read_csv(HERE / f"fam_{k}.csv", index_col=0)
    return d.fillna({c: d[c].median() for c in d.columns})   # fixed median fill (bnd columns), indicator bnd_has kept
class SkipLadder(Ladder):
    """Ladder whose extra table has fold-dependent skip-weighted NLI columns (CondModel refit on the training cases)."""
    def __init__(self, *a, base_extra, **k):
        super().__init__(*a, extra_table=base_extra, **k); self.base = base_extra
    def fit(self, data, train_task_ids):
        sk = skipf.build(list(train_task_ids), list(self.base.index))
        self.extra = self.base.join(sk); self._test = None
        super().fit(data, train_task_ids)
def make_preds(src, M, bench):
    ad = tab(REPO / "data" / bench / "features" / "adele.csv"); rub = tab(S + f"/rubric/features/rub_proc_{bench}.csv")
    plan = tab(I3 / f"plan_tools/features/plan3_{bench}.csv"); cmp_ = tab(I3 / "compile_consistency/cons_sopbench.csv")[["len_mean", "branch_mean"]]
    base = pd.concat([ad, plan, cmp_], axis=1); nli, ret, act, bnd = fam("nli"), fam("ret"), fam("act"), fam("bnd")
    def L(name, parts): return Ladder(2, TYPES, GLOB, rub, extra_table=pd.concat([base] + parts, axis=1), name=name)
    return {"s1_nli": SkipLadder(2, TYPES, GLOB, rub, name="s1_nli", base_extra=pd.concat([base, nli], axis=1)),
            "s2_ret": L("s2_ret", [ret]), "s3_act": L("s3_act", [act]), "s4_bnd": L("s4_bnd", [bnd]),
            "s5_all": SkipLadder(2, TYPES, GLOB, rub, name="s5_all", base_extra=pd.concat([base, nli, ret, act, bnd], axis=1)),
            "c_lad_re": Ladder(2, TYPES, GLOB, rub, extra_table=base, name="c_lad_re")}
if __name__ == "__main__":
    dbg = len(sys.argv) > 1
    out = HERE / ("out_debug" if dbg else "out")
    with cpu_lock("case_signals"):
        H.run_all(out, make_preds, benches=["sopbench"], schemes=["new_domain"] if dbg else ["new_procedures", "new_domain"])
    print("run done")
