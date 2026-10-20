import sys, json, numpy as np, pandas as pd
import lib, run_llm, prompts
from checklists import CHECK
from lib import DEMANDS
U, dom = run_llm.units()
def raw(p, rep=0, temp=0.0):
    k = lib.key(p, temp, rep); return lib._C.get(k, {}).get("text")
def vecA(text):
    j = lib.parse_json(text) if text else None
    if not j: return None, None
    e, m = {}, {}
    for d in DEMANDS:
        try:
            p = np.array(j[d], float); assert len(p) == 6; p = np.clip(p, 0, None); p = p / p.sum()
            e[d] = float((p * np.arange(6)).sum()); m[d] = float(np.argmax(p))
        except Exception: e[d] = np.nan; m[d] = np.nan
    return e, m
def vecB(text):
    j = lib.parse_json(text) if text else None
    if not j: return None
    out = {}
    for d in DEMANDS:
        try:
            y = np.array([1.0 if j[f"{d}{i+1}"] in (True, "true", "True") else 0.0 for i in range(5)]); w = np.arange(1, 6)
            out[d] = float(5 * (w * y).sum() / w.sum())
        except Exception: out[d] = np.nan
    return out
def vecC(text):
    j = lib.parse_json(text) if text else None
    if not j: return None
    out = {}
    for d in DEMANDS:
        try: out[d] = float(np.clip(float(j[d]), 0, 5))
        except Exception: out[d] = np.nan
    return out
def vecINT(text):
    j = lib.parse_json(text) if text else None
    if not j: return None
    out = {}
    for d in DEMANDS:
        try: out[d] = float(np.clip(float(j[d]), 0, 5))
        except Exception: out[d] = np.nan
    return out
if __name__ == "__main__":
    A, ad, texts = run_llm.anchors(U, dom)
    def promptfor(var, b, tx):
        c = U[(b, tx)][0]
        if var == "C":
            s1, s2 = A[dom[c][:2]]; use = s2 if c in s1 else s1
            return prompts.prompt_C(tx, dom[c][1], [(texts[a], ad.loc[a].to_dict()) for a in use])
        return {"A": prompts.prompt_A, "B": prompts.prompt_B}[var](tx)
    rows = {v: {} for v in ["A", "A0", "B", "C"]}; fail = {v: 0 for v in rows}
    for (b, tx), cases in U.items():
        ta = raw(promptfor("A", b, tx)); e, m = vecA(ta)
        tb = vecB(raw(promptfor("B", b, tx))); tc = vecC(raw(promptfor("C", b, tx)))
        for v, x in [("A", e), ("A0", m), ("B", tb), ("C", tc)]:
            if x is None: fail[v] += 1; continue
            for c in cases: rows[v][c] = x
    out = lib.HERE / "features"; out.mkdir(exist_ok=True)
    for v, r in rows.items():
        df = pd.DataFrame(r).T[DEMANDS]; df.index.name = "task_id"
        miss = int(df.isna().sum().sum())
        for b in run_llm.BENCH:
            ids = pd.read_csv(lib.REPO / "data" / b / "tasks.csv").case_id
            d = df.reindex(ids)
            d = d.fillna(d.median()).fillna(0)
            d.index.name = "task_id"; d.reset_index().to_csv(out / f"{v}_{b}.csv", index=False)
        print(v, "cases", len(df), "unparsed texts", fail[v], "nan cells", miss)
