import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os
sys.path.insert(0, ZR.SCREEN)
sys.path.insert(0, ZR.SCREEN + "/ideas")
sys.path.insert(0, ZR.SCREEN + "/ideas2/steps")
import harness as H
from cpulock import cpu_lock
import steppred as S

OUT = ZR.SCREEN + "/ideas2/steps/out"
DESC = {
    "cnt_geo": "null: r_a^n (n = steps), geometric law",
    "cnt_geo_leaf": "null: r_a^n (n = rule-tree leaves + 1)",
    "cnt_logit": "null: logit = a_agent - b n (LLTM, step count)",
    "cnt_logit_a": "null: agent-specific slope on step count",
    "typ_logit": "step types: logit = a_agent - sum d_t n_t (LLTM on type counts)",
    "typ_logit_a": "step types with agent x type weights (ridge)",
    "typ_prod": "product of shrunk observed step reliabilities by type",
    "cnt_prod": "null: product of observed overall step reliability r_a^n",
}


def make_preds(src, M):
    return S.make_all()


if __name__ == "__main__":
    with cpu_lock("steps"):
        H.run_all(OUT, make_preds, benches=["sopbench"], schemes=["new_procedures", "new_domain"])
        combos = [("typ_logit", "adele"), ("cnt_logit", "adele"), ("typ_logit_a", "adele"), ("typ_prod", "adele"), ("cnt_geo", "adele")]
        H.evaluate_all(OUT, DESC, combos=combos)
