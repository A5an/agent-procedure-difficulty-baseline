import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
sys.path.insert(0, ZR.SCREEN); sys.path.insert(0, ZR.SCREEN + "/ideas")
sys.path.insert(0, ZR.SCREEN + "/ideas3/conditions")
import harness as H, metrics_extra as X, json
from cpulock import cpu_lock
import cm
OUT = ZR.SCREEN + "/ideas3/conditions/out"
DESC = {"l0_alone": "L0 documentation only: mixture of perform (product of 1-skip) and refuse (mean over conditions), rubric p_perform",
        "l0_adele": "L0 stacked ridge ADeLe + OOF case score", "l0_agent": "L0 with agent-family skip rates",
        "l0_prior": "L0 with training perform prior instead of rubric",
        "l2_alone": "L2 compiled program decision + failing-condition skip", "l2_adele": "L2 stacked ridge", "l2_agent": "L2 per agent family",
        "l0_const": "L0 with constant skip rates (no text features)", "l2_const": "L2 with constant skip rates",
        "l2_truth": "DIAGNOSTIC privileged label and true failing conditions"}
COMBOS = [("l0_alone", "adele"), ("l2_alone", "adele")]
MAIN = ["l0_alone", "l0_adele", "l0_agent", "l0_const", "l2_alone", "l2_adele", "l2_agent", "l2_const"]
def make_preds(src, M):
    return cm.make_all(MAIN)
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    with cpu_lock("conditions"):
        if mode in ("full", "run"):
            H.run_all(OUT, make_preds, benches=["sopbench"], schemes=["new_procedures", "new_domain"])
        if mode in ("full", "eval"):
            H.evaluate_all(OUT, {k: DESC[k] for k in MAIN}, combos=COMBOS)
            r = X.report(OUT, ["adele"] + MAIN, combos=COMBOS,
                         extra_dirs={"cnt_geo": ZR.SCREEN + "/ideas2/steps/out",
                                     "cnt_logit": ZR.SCREEN + "/ideas2/steps/out",
                                     "typ_logit": ZR.SCREEN + "/ideas2/steps/out"})
