"""One harness run for the pairwise (Bradley-Terry) features. Benches sopbench, tau2; schemes new_procedures, new_domain."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
sys.path.insert(0, ZR.SCREEN)
sys.path.insert(0, ZR.SCREEN + "/ideas")
from pathlib import Path
import harness as H, metrics_extra as X
from cpulock import cpu_lock
HERE = Path(__file__).parent
BN, SC = ["sopbench", "tau2"], ["new_procedures", "new_domain"]
extra = {n: {b: str(HERE/"features"/f"{n}_{b}.csv") for b in BN} for n in ["bt", "bt_hard", "bt_cat_adele"]}
def make_preds(src, M):
    return {n: H.ridge(src[n]) for n in ["bt", "bt_hard", "bt_cat_adele"]}
DESC = {"bt": "ridge on the Bradley-Terry score (soft outcomes) alone",
        "bt_hard": "ridge on the Bradley-Terry score from hard outcomes",
        "bt_cat_adele": "one ridge on BT score + 18 ADeLe"}
COMBOS = [("bt", "adele"), ("bt_cat_adele", "adele")]
RUB = ZR.SCREEN + "/rubric/out/run"
if __name__ == "__main__":
    out = HERE/"out"/"run"
    with cpu_lock("pairwise"):
        H.run_all(out, make_preds, extra_csv=extra, benches=BN, schemes=SC)
        H.evaluate_all(out, DESC, combos=COMBOS)
    with cpu_lock("pairwise"):
        X.report(out, ["bt", "bt_hard", "bt_cat_adele"], extra_dirs={"lad2_adele": RUB, "grp_proc_adele": RUB},
                 combos=COMBOS + [("bt", "lad2_adele"), ("bt_cat_adele", "lad2_adele")])
