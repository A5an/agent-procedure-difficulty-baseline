"""Validation (not features): correlation of planned length / writes with gold, per tau2 domain, Gemini mean3 (plan_tools) vs Sonnet vs intersection vs avg."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
HERE = Path(__file__).parent; FEAT = HERE / "features"
REPO = Path(ZR.REPO)
TD = Path(ZR.DATA + "/tau2-domains/tasks")
types = json.load(open(HERE / "tool_types.json")); m = json.load(open(REPO / "data/tau2/id_map.json")); gold = {}
for dom in ["airline", "retail", "telecom"]:
    tk = {str(t["id"]): t for t in json.load(open(TD / f"{dom}_tasks.json"))}
    for k, c in m.items():
        if k.startswith(dom + ":"): gold[c] = (tk[k.split(":", 1)[1]]["evaluation_criteria"].get("actions") or [])
it = pd.read_csv(REPO / "data/tau2/irt/1d_1pl/items.csv", index_col=0)["b"]
T = {n: pd.read_csv(FEAT / f"{n}_tau2.csv").set_index("case_id") for n in ["son", "int", "avg", "dis"]}
T["gem"] = pd.read_csv(HERE.parent.parent / "ideas3/plan_tools/features/plan3_tau2.csv").set_index("case_id")
rows = []
for c in T["gem"].index:
    dom = c.split("/")[0]; acts = gold.get(c, []); ga = [a["name"] for a in acts if a.get("requestor", "assistant") != "user"]
    rows.append(dict(case_id=c, dom=dom, gold_len=len(ga), gold_w=sum(types.get(f"{dom}/{n}") == "write" for n in ga), b=it.get(c, np.nan)))
d = pd.DataFrame(rows).set_index("case_id")
def rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return None if np.unique(a).size < 2 or np.unique(b).size < 2 else round(float(spearmanr(a, b)[0]), 3)
res = {}
for dom, x in list(d.groupby("dom")) + [("all", d)]:
    r = {"n": len(x)}
    for n in ["gem", "son", "int", "avg"]:
        t = T[n].loc[x.index]
        r[n] = dict(calls_vs_gold_len=rho(t.plan_calls, x.gold_len), writes_vs_gold_writes=rho(t.plan_writes, x.gold_w),
                    calls_vs_b=rho(t.plan_calls, x.b), writes_vs_b=rho(t.plan_writes, x.b), mean_calls=round(float(t.plan_calls.mean()), 2), mean_writes=round(float(t.plan_writes.mean()), 2))
    t = T["dis"].loc[x.index]
    r["dis"] = dict(dis_calls_vs_b=rho(t.dis_calls, x.b), dis_writes_vs_b=rho(t.dis_writes, x.b), jacc_writes_vs_b=rho(t.dis_jaccard_writes, x.b),
                    outcome_agree_vs_b=rho(t.dis_outcome_agree, x.b), mean_outcome_agree=round(float(t.dis_outcome_agree.mean()), 3),
                    mean_jaccard_calls=round(float(t.dis_jaccard_calls.mean()), 3))
    r["gold"] = dict(mean_len=round(x.gold_len.mean(), 2), mean_w=round(x.gold_w.mean(), 2), gold_len_vs_b=rho(x.gold_len, x.b), gold_w_vs_b=rho(x.gold_w, x.b))
    res[dom] = r
json.dump(res, open(HERE / "validation6.json", "w"), indent=1)
for dom, r in res.items():
    print(dom, r["n"], "gold", r["gold"])
    for n in ["gem", "son", "int", "avg"]: print("  ", n, r[n])
    print("   dis", r["dis"])
