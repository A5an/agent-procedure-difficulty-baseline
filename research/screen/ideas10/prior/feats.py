"""prior_raw.json -> prior_<bench>.csv (9 PRIOR features, median-filled per benchmark). usage: feats.py"""
import json, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).parent
PRIOR = ["pr_n_rules", "pr_unant_sum", "pr_unant_max", "pr_n_unanticipated", "pr_default_mean", "pr_n_lowdefault",
         "pr_dec_conflict", "pr_extra_actions", "pr_tool_unant"]
def tools(p):
    return [c.get("tool") for c in (p or {}).get("calls", []) if isinstance(c, dict) and c.get("tool")]
def feats(v):
    nv = [n for n in v["naive"] if isinstance(n, dict)]; pp = v["policy"] if isinstance(v["policy"], dict) else {}
    valid = [k + 1 for k, n in enumerate(v["naive"]) if isinstance(n, dict)]; nval = max(1, len(valid))
    rules = [r for r in pp.get("rules", []) if isinstance(r, dict)]
    d = {"pr_n_rules": len(rules) if pp else np.nan}
    cov = {}
    for c in ((v.get("match") or {}).get("coverage") or []):
        try: cov[int(c.get("rule"))] = len(set(int(x) for x in c.get("covered_by", [])) & set(valid)) / nval
        except Exception: pass
    if rules and cov:
        un = [1 - cov.get(i + 1, 0.0) for i in range(len(rules))]
        d.update(pr_unant_sum=sum(un), pr_unant_max=max(un), pr_n_unanticipated=sum(1 for i in range(len(rules)) if cov.get(i + 1, 0.0) <= 1 / 6 + 1e-9))
    elif pp and not rules:
        d.update(pr_unant_sum=0.0, pr_unant_max=0.0, pr_n_unanticipated=0)
    dfl = []
    for r in rules:
        try: dfl.append(float(r.get("default")))
        except Exception: pass
    if dfl: d.update(pr_default_mean=float(np.mean(dfl)), pr_n_lowdefault=sum(1 for x in dfl if x <= 2))
    elif pp and not rules: d.update(pr_n_lowdefault=0)
    pdec = pp.get("decision")
    if pdec and nv: d["pr_dec_conflict"] = float(np.mean([n.get("decision") != pdec for n in nv]))
    ptools = set(tools(pp))
    if nv:
        d["pr_extra_actions"] = float(np.mean([sum(1 for c in n.get("calls", []) if isinstance(c, dict) and c.get("kind") == "action" and c.get("tool") not in ptools) for n in nv]))
        if pp: d["pr_tool_unant"] = float(sum(1 - np.mean([t in set(tools(n)) for n in nv]) for t in ptools))
    return d
if __name__ == "__main__":
    raw = json.load(open(HERE / "prior_raw.json"))
    rows = [{"case_id": k, "bench": v["bench"], **feats(v)} for k, v in raw.items()]
    df = pd.DataFrame(rows)
    for b, g in df.groupby("bench"):
        g = g.set_index("case_id")[PRIOR]
        print(b, len(g), "missing per column", g.isna().sum().to_dict())
        g = g.fillna(g.median())
        name = {"sopbench": "sopbench", "tau2": "tau2", "banking": "tauk_banking"}[b]
        g.to_csv(HERE / f"prior_{name}.csv")
