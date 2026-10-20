"""Turn bank_raw.json into banking records, PLAN features, F1 features. Writes bank_records.jsonl, bank_plan.csv, bank_f1.csv, bank_dry.csv."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, re
S = ZR.SCREEN
sys.path.insert(0, S + "/ideas3/plan_tau"); sys.path.insert(0, S + "/ideas7/evo")
import numpy as np, pandas as pd
import feats as PF
from pathlib import Path
HERE = Path(S + "/ideas7/evo_check"); R = Path(ZR.REPO)
raw = json.load(open(HERE / "bank_raw.json"))
STM = {json.loads(l)["case_id"]: json.loads(l)["text"] for l in open(R / "data/tauk_banking/statements.jsonl")}
import glob
DOCS = [json.load(open(f)) for f in sorted(glob.glob(S + "/ideas3/plan_tools/src_tau2/data/tau2/domains/banking_knowledge/documents/*.json"))]
TOOLS = [t.get("function", t) for t in json.load(open(HERE / "bank_tools.json"))]
SCN = ["n_goals", "n_mind_changes", "withholds_info", "policy_forbidden_request", "pressure", "n_conditional", "needs_computation", "ambiguous_reference", "n_constraints"]
def pj(s):
    if not s: return None
    try: return json.loads(s[s.find("{"):s.rfind("}") + 1])
    except Exception: return None
recs = []; planrows = {}; dryrows = {}; miss = {"plan": 0, "scn": 0, "dry": 0}
for cid, r in raw.items():
    plans = [p for p in (pj(r.get(k)) for k in ("plan_t0", "plan_s1", "plan_s2")) if p]
    miss["plan"] += 3 - len(plans)
    if plans:
        fs = pd.DataFrame([PF.feats(p) for p in plans]); planrows[cid] = fs.mean().to_dict()
    sc = pj(r.get("scn")); scn = None
    if sc:
        scn = {k: float(sc.get(k, 0) or 0) for k in SCN}; scn["n_clauses"] = float(len(sc.get("clauses", [])))
        scn["policy_clauses_touched"] = [DOCS[int(re.sub(r"\D", "", c))]["title"] for c in sc.get("clauses", []) if re.sub(r"\D", "", c) and int(re.sub(r"\D", "", c)) < len(DOCS)]
    else: miss["scn"] += 1
    d = pj(r.get("dry")); dry = {"parsed": d}
    row = dict.fromkeys(["p_refuse", "p_perform", "n_conditions", "n_violated_by_request", "n_needs_system_data", "n_lookups", "n_customer_facts", "ambiguous_request", "ambiguous_policy", "decision_entropy", "refuse_share_sampled"], np.nan)
    if d:
        p = d["outcome_probs"]; s = sum(float(p[k]) for k in ["perform", "refuse", "transfer_or_other"])
        row.update(p_refuse=float(p["refuse"]) / s, p_perform=float(p["perform"]) / s, n_conditions=len(d["conditions"]),
                   n_violated_by_request=sum(c["status"] == "violated_by_request" for c in d["conditions"]),
                   n_needs_system_data=sum(c["status"] == "needs_system_data" for c in d["conditions"]),
                   n_lookups=int(d["n_lookups"]), n_customer_facts=int(d["n_customer_facts"]), ambiguous_request=int(d["ambiguous_request"]), ambiguous_policy=int(d["ambiguous_policy"]))
    else: miss["dry"] += 1
    dryrows[cid] = row; dry.update({k: (None if pd.isna(v) else float(v)) for k, v in row.items()})
    recs.append(dict(id=cid, benchmark="tau2", domain="banking_knowledge", procedure_id=cid, procedure=cid, domain_description="banking knowledge customer service agent",
                     policy_text="", message=STM[cid], tools=TOOLS, program=None, plans=plans, dryrun=dry, conditions=[], skip_model_case_p_success=None, scenario=scn))
plan = pd.DataFrame.from_dict(planrows, orient="index").reindex(list(raw)); plan = plan.fillna(plan.median()); plan.index.name = "case_id"; plan.reset_index().to_csv(HERE / "bank_plan.csv", index=False)
with open(HERE / "bank_records.jsonl", "w") as f:
    for r in recs: f.write(json.dumps(r) + "\n")
dd = pd.DataFrame.from_dict(dryrows, orient="index"); dd.index.name = "case_id"; dd.reset_index().to_csv(HERE / "bank_dry.csv", index=False)
code = json.load(open(S + "/ideas7/evo/finalists.json"))[0]["code"]
ns = {}; exec(code, ns)
def f1(rec):
    try: return ns["features"](rec)
    except Exception as e: return None
rows = {r["id"]: f1(r) for r in recs}
bad = [k for k, v in rows.items() if v is None]
F = pd.DataFrame.from_dict({k: v for k, v in rows.items() if v is not None}, orient="index"); F.index.name = "case_id"
F.reset_index().to_csv(HERE / "bank_f1.csv", index=False)
print("missing parses", miss, "f1 crashes", len(bad)); print(F.describe().T[["mean", "std", "min", "max"]])
# sanity: the same code on tau2 records reproduces feat_f1.csv
tr = [json.loads(l) for l in open(S + "/ideas7/evo/records.jsonl")]; tr = [r for r in tr if r["benchmark"] == "tau2"][:40]
st = pd.read_csv(S + "/ideas7/evo/feat_f1.csv").set_index("case_id")
dif = max(abs(f1(r)[k] - st.loc[r["id"], k]) for r in tr for k in st.columns); print("max abs diff vs stored feat_f1 on 40 tau2 records", dif)
