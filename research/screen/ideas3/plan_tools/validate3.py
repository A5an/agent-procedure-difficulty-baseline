import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, re, numpy as np, pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
import feats3 as F3, sys
sys.path.insert(0, str(Path(__file__).parent.parent / "plan_tau")); import feats as F
HERE = Path(__file__).parent
REPO = Path(ZR.REPO)
TD = Path(ZR.DATA + "/tau2-domains/tasks")
types = json.load(open(HERE / "tool_types.json"))
m = json.load(open(REPO / "data/tau2/id_map.json")); gold = {}
for dom in ["airline", "retail", "telecom"]:
    tk = {str(t["id"]): t for t in json.load(open(TD / f"{dom}_tasks.json"))}
    for k, c in m.items():
        if k.startswith(dom + ":"): gold[c] = (tk[k.split(":", 1)[1]]["evaluation_criteria"].get("actions") or [])
plans = json.load(open(HERE / "plans3.json")); it = pd.read_csv(REPO / "data/tau2/irt/1d_1pl/items.csv", index_col=0)["b"]
rows = []
for c, d in plans.items():
    dom = c.split("/")[0]; ps = [p for p in (F.parse(d[t]) for t in ("t0", "s1", "s2")) if p]
    acts = gold.get(c, []); ga = [a["name"] for a in acts if a.get("requestor", "assistant") != "user"]
    gall = [a["name"] for a in acts]
    gw = sum(types.get(f"{dom}/{n}") == "write" for n in ga)
    mp = pd.DataFrame([F3.feats(p, dom) for p in ps]).mean()
    tools_p = []; 
    for p in ps: tools_p.append({str(x.get("tool")) for x in p.get("plan", []) if isinstance(x, dict)} - {"ask_customer_to_act"})
    jac_t0 = [len(t & set(ga)) / max(1, len(t | set(ga))) for t in tools_p]
    # per-plan agent-tool counts
    rows.append(dict(case_id=c, dom=dom, plan_calls=mp.plan_calls, plan_writes=mp.plan_writes, plan_distinct=mp.plan_distinct_tools, invalid=mp.plan_invalid_tool,
        gold_agent_len=len(ga), gold_all_len=len(gall), gold_agent_writes=gw, gold_agent_distinct=len(set(ga)),
        jac=np.mean(jac_t0) if len(ga) or any(tools_p) else np.nan, empty_both=(len(ga) == 0 and all(len(t) == 0 for t in tools_p)),
        b=it.get(c, np.nan),
        plan_user_steps=np.mean([sum(str(x.get("tool")) == "ask_customer_to_act" for x in p.get("plan", []) if isinstance(x, dict)) for p in ps]),
        gold_user_len=len(gall) - len(ga)))
d = pd.DataFrame(rows).set_index("case_id")
def rho(a, b):
    a, b = a.astype(float), b.astype(float)
    return None if a.nunique() < 2 or b.nunique() < 2 else round(float(spearmanr(a, b)[0]), 3)
res = {}
for dom, x in list(d.groupby("dom")) + [("all", d)]:
    xj = x[~x.empty_both]
    res[dom] = dict(n=len(x), plan_calls_vs_gold_agent_len=rho(x.plan_calls, x.gold_agent_len), plan_calls_vs_gold_all_len=rho(x.plan_calls + 0, x.gold_all_len),
        plan_writes_vs_gold_agent_writes=rho(x.plan_writes, x.gold_agent_writes), plan_distinct_vs_gold_distinct=rho(x.plan_distinct, x.gold_agent_distinct),
        plan_user_steps_vs_gold_user_len=rho(x.plan_user_steps, x.gold_user_len),
        mean_plan_calls=round(x.plan_calls.mean(), 2), mean_gold_agent_len=round(x.gold_agent_len.mean(), 2), mean_gold_all_len=round(x.gold_all_len.mean(), 2),
        mean_plan_writes=round(x.plan_writes.mean(), 2), mean_gold_agent_writes=round(x.gold_agent_writes.mean(), 2),
        mean_jaccard_tool_names=round(float(xj.jac.mean()), 3), invalid_tool_name_frac=round(float(x.invalid.mean()), 3),
        b_vs_plan_calls=rho(x.b, x.plan_calls), b_vs_plan_writes=rho(x.b, x.plan_writes), b_vs_gold_agent_len=rho(x.b, x.gold_agent_len), b_vs_gold_all_len=rho(x.b, x.gold_all_len))
# stability: spread across the 3 plans
sp = {}
for c, dd in plans.items():
    ps = [p for p in (F.parse(dd[t]) for t in ("t0", "s1", "s2")) if p]
    f = pd.DataFrame([F3.feats(p, c.split("/")[0]) for p in ps]); sp[c] = dict(dom=c.split("/")[0], sd_calls=f.plan_calls.std(ddof=0), sd_writes=f.plan_writes.std(ddof=0), t0_calls=f.plan_calls.iloc[0], mean_calls=f.plan_calls.mean())
sp = pd.DataFrame(sp).T; sp[["sd_calls", "sd_writes", "t0_calls", "mean_calls"]] = sp[["sd_calls", "sd_writes", "t0_calls", "mean_calls"]].astype(float)
res["stability"] = {dom: dict(mean_sd_calls=round(x.sd_calls.mean(), 3), mean_sd_writes=round(x.sd_writes.mean(), 3), rho_t0_vs_mean_calls=rho(x.t0_calls, x.mean_calls)) for dom, x in sp.groupby("dom")}
# compare with plan_tau on gold agreement
old = pd.read_csv(HERE.parent / "plan_tau/features/plan_tau2.csv").set_index("case_id")
d2 = d.join(old[["plan_calls", "plan_writes"]].add_prefix("old_"))
res["old_plan_tau_vs_gold_agent_len"] = {dom: rho(x.old_plan_calls, x.gold_agent_len) for dom, x in d2.groupby("dom")}
res["old_plan_tau_writes_vs_gold_agent_writes"] = {dom: rho(x.old_plan_writes, x.gold_agent_writes) for dom, x in d2.groupby("dom")}
res["telecom_note"] = dict(distinct_texts=5, gold_agent_len_distribution=d[d.dom == "telecom"].gold_agent_len.value_counts().to_dict(), gold_user_len_mean=round(d[d.dom == "telecom"].gold_user_len.mean(), 2))
json.dump(res, open(HERE / "validation3.json", "w"), indent=1); print(json.dumps(res, indent=1))
