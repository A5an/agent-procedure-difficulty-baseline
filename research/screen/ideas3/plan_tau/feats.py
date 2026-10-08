import json, re, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).parent
FEAT = HERE / "features"; FEAT.mkdir(exist_ok=True)

def parse(t):
    if not t: return None
    t = t.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    try: return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        try: return json.loads(m.group(0)) if m else None
        except Exception: return None

def feats(p):
    plan = p.get("plan") or []
    plan = [x for x in plan if isinstance(x, dict)]
    w = [bool(x.get("writes_db")) for x in plan]
    tools = {str(x.get("tool")) for x in plan}
    out = str(p.get("outcome", "complete"))
    return dict(plan_calls=len(plan), plan_writes=sum(w), plan_reads=len(plan) - sum(w), plan_distinct_tools=len(tools),
                plan_rules=len(p.get("policy_rules") or []), plan_facts=len(p.get("facts_to_ask_customer") or []),
                plan_refuse=float(out in ("refuse", "transfer_to_human")), plan_transfer=float(out == "transfer_to_human"),
                plan_ambig=float(p.get("ambiguity", 0) or 0))

def build():
    t0 = json.load(open(HERE / "plans_t0.json"))
    rows = {}
    for k, v in t0.items():
        b, c = k.split("|", 1); p = parse(v)
        rows.setdefault(b, {})[c] = feats(p) if p else None
    t1 = json.load(open(HERE / "plans_t1.json")) if (HERE / "plans_t1.json").exists() else {}
    stab = {}
    for k, v in t1.items():
        b, c = k.split("|", 1)
        ps = [parse(x) for x in v]; ps0 = parse(t0[k])
        fs = [feats(p) for p in [ps0] + ps if p]
        if len(fs) >= 2:
            stab[c] = {kk: float(np.std([f[kk] for f in fs])) for kk in fs[0]}
            stab[c]["_n"] = len(fs)
    out = {}
    for b, d in rows.items():
        df = pd.DataFrame.from_dict({c: (f or {}) for c, f in d.items()}, orient="index")
        miss = df.isna().any(axis=1).sum()
        df = df.fillna(df.median())
        df["plan_instab"] = 0.0  # placeholder constant column removed below; instability reported separately, not a feature
        df = df.drop(columns="plan_instab")
        df.index.name = "case_id"; df.reset_index().to_csv(FEAT / f"plan_{b}.csv", index=False)
        out[b] = dict(n=len(df), parse_fail=int(miss))
    json.dump(stab, open(HERE / "stability.json", "w"))
    print(out, "stab tasks", len(stab))
if __name__ == "__main__": build()
