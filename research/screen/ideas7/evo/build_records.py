"""Build one JSON record per case from existing files only (no LLM calls). Output: records.jsonl, targets.csv.
Never read: outcomes of cases (except the skip-model scores, see below), gold actions, constraints, action_should_succeed,
databases, smart-agent results. targets.csv (IRT difficulty b, c_lad out-of-fold prediction) is used by the frozen evaluator only
and is never put into a record.
Skip-model scores (ideas3/conditions CondModel) are fitted on agent outcomes of OTHER domains only: search domains get a model
fitted on the other three search domains, held-out domains get a model fitted on the four search domains."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, re, os
S = ZR.SCREEN
os.environ.setdefault("PYTHONHASHSEED", "0")
sys.path.insert(0, S + "/ideas3/conditions"); sys.path.insert(0, S); sys.path.insert(0, S + "/ideas")
from pathlib import Path
import numpy as np, pandas as pd
R = Path(ZR.REPO)
HERE = Path(S + "/ideas7/evo")
EXT = Path(ZR.DATA)
SEARCH_SOP = ["bank", "dmv", "healthcare", "hotel"]
def jl(p): return [json.loads(l) for l in open(p)]
def pj(s):
    try: return json.loads(s)
    except Exception: return None

# ---- static docs
stm = {}
for b in ["sopbench", "tau2"]:
    for d in jl(R / f"data/{b}/statements.jsonl"): stm[d["case_id"]] = d["text"]
tasks = {b: pd.read_csv(R / f"data/{b}/tasks.csv").set_index("case_id") for b in ["sopbench", "tau2"]}
# SOPBench tool schemas
sys.path.insert(0, str(EXT / "SOPBench-d2622008"))
import importlib
SOPTOOLS = {}
for dom in ["bank", "dmv", "healthcare", "hotel", "library", "online_market", "university"]:
    m = importlib.import_module(f"env.domains.{dom}.{dom}_assistant")
    SOPTOOLS[dom] = {a["name"]: {"name": a["name"], "description": a.get("description") or m.action_descriptions.get(a["name"], ""), "parameters": a.get("parameters", {})} for a in m.actions}
TT = json.load(open(S + "/ideas3/plan_tools/tool_text.json"))
TAUTOOLS = {d: [json.loads(l) for l in v.splitlines() if l.strip().startswith("{")] for d, v in TT.items()}
TP = EXT / "tau2-domains/tasks"
TAUPOL = {"airline": (TP / "airline_policy.md").read_text(), "retail": (TP / "retail_policy.md").read_text(),
          "telecom": "\n\n".join((TP / f).read_text() for f in ["telecom_main_policy.md", "telecom_tech_support_manual.md", "telecom_tech_support_workflow.md"])}
# programs, units
PROG = json.load(open(S + "/ideas2/compile/programs_t0.json"))
UNITS = json.load(open(S + "/ideas3/conditions/units.json")); C2U = {c: u for u, cs in UNITS.items() for c in cs}
# plans
P0 = json.load(open(S + "/ideas3/plan_tau/plans_t0.json"))
P3 = json.load(open(S + "/ideas3/plan_tools/plans3.json"))
# dry run
DRY = {}
for d in jl(S + "/ideas/dryrun/cache/calls.jsonl"):
    if d["kind"] == "main" and d["k"] == 0 and d.get("ok"): DRY[d["case_id"]] = d["parsed"]
DRYF = {b: pd.read_csv(S + f"/ideas/dryrun/features/dry_{b}.csv").set_index("case_id") for b in ["sopbench", "tau2"]}
# user scenario
SCN = pd.read_csv(S + "/ideas4/user_scenario/scn_tau2.csv").set_index("case_id")
CLM = json.load(open(S + "/ideas4/user_scenario/clause_map.json")); CL = json.load(open(S + "/ideas4/user_scenario/clauses.json"))
CLH = {f"{d}/{c['id']}": c["heading"] for d, cs in CL.items() for c in cs}
# conditions
import cm as CMOD
cd = CMOD.cd
CONDROWS = {u: g for u, g in cd.groupby("unit")}
def fit_skip(train_domains):
    tr = [c for c in tasks["sopbench"].index if tasks["sopbench"].loc[c, "domain"] in train_domains]
    return CMOD.CondModel(tr), CMOD.Pred("l0", "l0")
skipm = {}
for d in SEARCH_SOP:
    skipm[d] = fit_skip([x for x in SEARCH_SOP if x != d])
allm = fit_skip(SEARCH_SOP)
for d in ["library", "online_market", "university"]: skipm[d] = allm

def sop_record(cid):
    t = stm[cid]; row = tasks["sopbench"].loc[cid]
    m = re.match(r"Domain: ([\w ]+)\. (.*?)\nRequested action: (.*?)\nPolicy for this action:\n(.*?)\nCustomer message: (.*?)\nAvailable tools: (.*)$", t, re.S)
    dom_desc, action, policy, msg, tl = m.group(2), m.group(3), m.group(4), m.group(5), m.group(6)
    names = [x.strip() for x in tl.split(",")]
    tools = [SOPTOOLS[row.domain][n] for n in names if n in SOPTOOLS[row.domain]]
    u = C2U[cid]; g = CONDROWS.get(u)
    cm_, pr = skipm[row.domain]
    conds = []
    if g is not None:
        for ridx, r in zip(g.index, g.itertuples()):
            conds.append({"text": r.text, "step_type": r.step_type, "step_name": r.step_name, "step_tools": r.step_tools.split("|"),
                          "is_or": int(r.is_or), "skip_if_must_perform_pred": float(cm_.pred["perf"][ridx]), "skip_if_must_refuse_pred": float(cm_.pred["fail"][ridx])})
    zz = float(pr._z(cm_, [cid])[0]); z = 1 / (1 + np.exp(-zz))
    plan = P0.get("sopbench|" + cid); plan = pj(plan) if isinstance(plan, str) else plan
    dry = {"parsed": DRY.get(cid), **{k: (None if pd.isna(v) else float(v)) for k, v in DRYF["sopbench"].loc[cid].items()}}
    pg = PROG.get(u); pg = pg.get("code") if isinstance(pg, dict) else pg
    return dict(id=cid, benchmark="sopbench", domain=row.domain, procedure_id=row.procedure_id, procedure=row.procedure,
                domain_description=dom_desc, policy_text=policy, message=msg, tools=tools, program=pg,
                plans=[plan] if plan else [], dryrun=dry, conditions=conds, skip_model_case_p_success=z, scenario=None)
def tau_record(cid):
    row = tasks["tau2"].loc[cid]; dom = row.domain
    plans = []
    for k in ["t0", "s1", "s2"]:
        p = P3.get(cid, {}).get(k)
        p = pj(p) if isinstance(p, str) else p
        if p: plans.append(p)
    dry = {"parsed": DRY.get(cid), **{k: (None if pd.isna(v) else float(v)) for k, v in DRYF["tau2"].loc[cid].items()}}
    s = SCN.loc[cid]
    scn = {k: float(v) for k, v in s.items()}
    scn["policy_clauses_touched"] = [CLH.get(c, c) for c in CLM.get(cid, [])]
    return dict(id=cid, benchmark="tau2", domain=dom, procedure_id=row.procedure_id, procedure=row.procedure,
                domain_description=f"{dom} customer service agent", policy_text=TAUPOL[dom], message=stm[cid], tools=TAUTOOLS[dom], program=None,
                plans=plans, dryrun=dry, conditions=[], skip_model_case_p_success=None, scenario=scn)
recs = []
for cid in tasks["sopbench"].index: recs.append(sop_record(cid))
for cid in tasks["tau2"].index: recs.append(tau_record(cid))
with open(HERE / "records.jsonl", "w") as f:
    for r in recs: f.write(json.dumps(r) + "\n")
rows = []
for b in ["sopbench", "tau2"]:
    it = pd.read_csv(R / f"data/{b}/irt/1d_1pl/items.csv").set_index("Unnamed: 0").b
    bs = []
    for s in ["new_procedures", "new_domain"]:
        d = pd.read_csv(S + f"/ideas3/combo/out_main/{b}/betas_{s}.csv"); d = d[d.method == "c_lad"]
        bs.append(d.groupby("case_id").beta_hat.mean())
    base = (bs[0] + bs[1]) / 2
    for cid in tasks[b].index: rows.append((cid, b, tasks[b].loc[cid, "domain"], tasks[b].loc[cid, "procedure_id"], it[cid], base[cid], bs[0][cid], bs[1][cid]))
pd.DataFrame(rows, columns=["case_id", "benchmark", "domain", "procedure_id", "y_irt_b", "clad_oof", "clad_oof_np", "clad_oof_nd"]).to_csv(HERE / "targets.csv", index=False)
print(len(recs), "records")
