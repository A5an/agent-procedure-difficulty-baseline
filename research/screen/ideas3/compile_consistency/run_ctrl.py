"""POST HOC controls (not among the fixed variants): which part of the consistency block carries the ladder gain.
 lad2_adele_cx  : only program length and branch count added (complexity of the compiled code)
 lad2_adele_dis : only disagreement and error features added (dec_dis, dec_ent, tool_pair, crash, tool_err, n_fields, cond_pair, cond_n_diff)"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
sys.path.insert(0, ZR.SCREEN + "/ideas3/compile_consistency")
from run_eval import *
CX = ["len_mean", "branch_mean"]; DIS = ["dec_dis", "dec_ent", "tool_pair", "crash", "tool_err", "n_fields", "cond_pair", "cond_n_diff"]
def make_preds2(src, M, bench):
    p = pd.read_csv(RUB / "features" / f"rub_proc_{bench}.csv").set_index("case_id"); c = cons(bench)
    return {"lad2_adele_cx": Ladder(2, TYPES, GLOB, p, extra_table=pd.concat([adele(bench), c[CX]], axis=1), name="cx"),
            "lad2_adele_dis": Ladder(2, TYPES, GLOB, p, extra_table=pd.concat([adele(bench), c[DIS]], axis=1), name="dis")}
if __name__ == "__main__":
    out = HERE / "out" / "ctrl"
    with cpu_lock("compile_consistency"):
        H.run_all(out, make_preds2, extra_csv={}, benches=B, schemes=["new_procedures", "new_domain"])
        H.evaluate_all(out, {"lad2_adele_cx": "ladder + ADeLe + length and branches", "lad2_adele_dis": "ladder + ADeLe + disagreement features"})
        X.report(out, ["lad2_adele_cx", "lad2_adele_dis"], extra_dirs={"lad2_adele": str(RUB / "out" / "run")})
