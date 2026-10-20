"""Skip-weighted NLI columns from a CondModel fitted on given training cases (fold-dependent)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, pickle, numpy as np, pandas as pd
S = ZR.SCREEN
sys.path.insert(0, S + "/ideas3/conditions"); sys.path.insert(0, ".")
import cm, prep
import os
MATS = pickle.load(open("nli_mats.pkl", "rb")) if os.path.exists("nli_mats.pkl") else {}
COLS = ["nli_ent_topskip", "nli_con_topskip", "nli_skipw_ent", "nli_skipw_silent"]
def ok_case(c):
    u = cm.C2U.get(c); return u is not None and len(cm.UNIT_ROWS.get(u, [])) == len(prep.CASES[c]["conds"])
def build(train_cases, cases):
    m = cm.CondModel(list(train_cases)); sp, sf = m.sk("perf"), m.sk("fail")
    rows = {}
    for c in cases:
        M = MATS[c]; Ms = M[:, :-1, :]; ent = Ms[:, :, 0].max(1); con = Ms[:, :, 2].max(1); sil = (np.maximum(ent, con) < 0.5).astype(float)
        if ok_case(c):
            r = cm.UNIT_ROWS[cm.C2U[c]]; sk = 0.5 * (sp[r] + sf[r])
        else: sk = np.full(len(ent), 0.5 * (m.mu_perf + m.mu_fail))
        k = int(np.argmax(sk)); w = sk / sk.sum()
        rows[c] = dict(nli_ent_topskip=ent[k], nli_con_topskip=con[k], nli_skipw_ent=float((w * ent).sum()), nli_skipw_silent=float((w * sil).sum()))
    return pd.DataFrame.from_dict(rows, orient="index")
if __name__ == "__main__":
    print(sum(ok_case(c) for c in prep.CASES), len(prep.CASES))
