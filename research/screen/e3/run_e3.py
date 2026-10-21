"""E3 with step types (PREREG.md)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ZR.SCREEN)
sys.path.insert(0, ZR.SCREEN + "/pertype")
import harness as H
import pertype_models as PM
OUT = Path(__file__).parent / "run"; OUT.mkdir(exist_ok=True)
RUB = ZR.SCREEN + "/rubric/features/concat/cat_full_{}.csv"
STEP = ["n_check_condition", "n_lookup_document", "n_ask_customer", "n_write_system", "n_irreversible_action", "n_branch", "n_refuse_or_handover"]
Q = ["q_" + c[2:] for c in STEP]
parts, extra = [], {}
for b in ("sopbench", "tau2"):
    r = pd.read_csv(RUB.format(b))
    x = np.log1p(r[STEP].astype(float)); x = (x - x.mean()) / x.std(ddof=0).replace(0, 1)
    t = pd.read_csv(H.RB.DATA / b / "tasks.csv")[["case_id", "procedure_id"]]
    d = t.merge(pd.concat([r[["case_id"]], x.set_axis(Q, axis=1)], axis=1), on="case_id", how="left")
    assert d[Q].notna().all().all(), b
    parts.append(d)
    ad = pd.read_csv(H.REPO / f"data/{b}/features/adele.csv").rename(columns={"task_id": "case_id"})
    ad.merge(d[["case_id"] + Q], on="case_id").to_csv(OUT / f"adele_steps_{b}.csv", index=False)
    extra[b] = OUT / f"adele_steps_{b}.csv"
PM.TYPES = pd.concat(parts).set_index("case_id")

_orig_get_ctx = PM.get_ctx
def get_ctx(data, train, src):
    """tau2 cells hold {successes, trials}: Y = share, M = trials gives the binomial likelihood in _fit_W."""
    key = (id(data), len(train), train[0], train[-1])
    if key in PM._cache: return PM._cache[key]
    binom = any(isinstance(v, dict) for r in data.responses.values() for v in r.values())
    if not binom: return _orig_get_ctx(data, train, src)
    saved = data.responses
    data.responses = {a: {t: 0 for t in r} for a, r in saved.items()}
    try: c = _orig_get_ctx(data, train, src)
    finally: data.responses = saved
    for i, a in enumerate(c.agents):
        r = saved.get(a, {})
        for j, t in enumerate(c.train):
            if t in r:
                v = r[t]; c.Y[i, j] = v["successes"] / v["trials"]; c.M[i, j] = float(v["trials"])
    return c
PM.get_ctx = get_ctx

def make_preds(src, M):
    ms = {"steps_mirt": PM.TypeLogit(qcols=Q), "steps_mirt_l10": PM.TypeLogit(qcols=Q, fixed_lam=10), "adele_steps": H.ridge(src["adele_steps"])}
    for m in ms.values():
        if hasattr(m, "bind"): m.bind(src)
    return ms

if __name__ == "__main__":
    benches = sys.argv[1:] or ["sopbench", "tau2"]
    H.run_all(OUT, make_preds, extra_csv={"adele_steps": extra}, benches=benches, schemes=["new_procedures", "new_domain"])
    H.evaluate_all(OUT, {"steps_mirt": "agent x step type (MIRT, inner-CV lambda)", "steps_mirt_l10": "same, lambda 10",
                         "adele_steps": "control: ADeLe + step counts -> difficulty"})
