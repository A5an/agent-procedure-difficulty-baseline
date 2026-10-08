import json, sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "plan_tau"))
import feats as F
HERE = Path(__file__).parent; FEAT = HERE / "features"; FEAT.mkdir(exist_ok=True)
TOOLS = json.load(open(HERE / "tool_text.json"))
import re
valid = {d: set(re.findall(r'"name":"([a-z_]+)"', t)) for d, t in TOOLS.items()}
valid_all = {d: v | {"ask_customer_to_act"} for d, v in valid.items()}
def feats(p, dom):
    f = F.feats(p); plan = [x for x in (p.get("plan") or []) if isinstance(x, dict)]
    f["plan_invalid_tool"] = float(np.mean([str(x.get("tool")) not in valid_all[dom] for x in plan])) if plan else 0.0
    return f
def build():
    res = json.load(open(HERE / "plans3.json")); rows = {}; mean = {}; sd = {}; raw = {}
    for c, d in res.items():
        dom = c.split("/")[0]; ps = [F.parse(d[t]) for t in ("t0", "s1", "s2")]
        ps = [p for p in ps if p]; raw[c] = ps
        if not ps: continue
        fs = pd.DataFrame([feats(p, dom) for p in ps])
        mean[c] = fs.drop(columns="plan_invalid_tool").mean().to_dict()
        sd[c] = {"sd_" + k: v for k, v in fs[["plan_calls", "plan_writes", "plan_distinct_tools", "plan_refuse"]].std(ddof=0).to_dict().items()}
        rows[c] = dict(n=len(ps), invalid=float(fs.plan_invalid_tool.mean()))
    m = pd.DataFrame.from_dict(mean, orient="index"); s = pd.DataFrame.from_dict(sd, orient="index")
    allc = [json.loads(l)["case_id"] for l in open(F.HERE.parent.parent.parent / "ideas3/plan_tau/features/../../../../../repo/agent-procedure-difficulty-baseline/data/tau2/statements.jsonl")] if False else list(res)
    miss = [c for c in res if c not in m.index]
    m = m.reindex(res.keys()); s = s.reindex(res.keys())
    m = m.fillna(m.median()); s = s.fillna(0.0)
    m.index.name = "case_id"; m.reset_index().to_csv(FEAT / "plan3_tau2.csv", index=False)
    ms = pd.concat([m, s], axis=1); ms.index.name = "case_id"; ms.reset_index().to_csv(FEAT / "plan3sd_tau2.csv", index=False)
    pd.DataFrame(rows).T.to_csv(FEAT / "meta_tau2.csv")
    print("tasks", len(m), "parse-fail tasks", len(miss), "mean invalid tool frac", np.mean([r["invalid"] for r in rows.values()]))
if __name__ == "__main__": build()
