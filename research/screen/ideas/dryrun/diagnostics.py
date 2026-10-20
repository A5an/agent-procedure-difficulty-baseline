import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
REPO = ZR.REPO + "/data/"
TAU = ZR.DATA + "/tau2-domains/tasks/"
out = {}
def auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 and len(set(s)) > 1 else float("nan")
t = pd.read_csv(REPO + "sopbench/tasks.csv").set_index("case_id")
d = pd.read_csv("features/dry_sopbench.csv").set_index("case_id").join(t)
d["refuse"] = 1 - d.should_succeed
rub = pd.read_csv(ZR.SCREEN + "/rubric/features/rub_proc_sopbench.csv").set_index("case_id")
d["rubric_p_refusal"] = rub.p_refusal
d["any_violated"] = (d.n_violated_by_request > 0).astype(int)
S = ["p_refuse", "refuse_share_sampled", "n_violated_by_request", "any_violated", "decision_entropy", "n_lookups", "rubric_p_refusal"]
out["sop_auc"] = {s: {"overall": auc(d.refuse, d[s]),
                      "within_domain_mean": float(np.nanmean([auc(g.refuse, g[s]) for _, g in d.groupby("domain")]))} for s in S}
dom = []
for k, g in d.groupby("domain"):
    v = g[g.any_violated == 1]
    dom.append({"domain": k, "n": len(g), "refuse_rate": round(g.refuse.mean(), 3), "share_any_violated": round(g.any_violated.mean(), 3),
                "precision_of_violated_is_refuse": round(v.refuse.mean(), 3) if len(v) else None,
                "recall_of_refuse_by_violated": round(g[g.refuse == 1].any_violated.mean(), 3)})
out["sop_domain"] = dom
v = d[d.any_violated == 1]
out["sop_total"] = {"share_any_violated": float(d.any_violated.mean()), "precision": float(v.refuse.mean()),
                    "recall": float(d[d.refuse == 1].any_violated.mean()), "base_refuse_rate": float(d.refuse.mean())}
# tau2 gold-derived label: gold has no state-changing action (actions empty) or gold actions contain transfer_to_human_agents
lab = {}
for dm in ["airline", "retail", "telecom"]:
    for i, x in enumerate(json.load(open(TAU + f"{dm}_tasks.json"))):
        names = [a["name"] for a in x["evaluation_criteria"].get("actions") or []]
        lab[(dm, str(x["id"]))] = int(len(names) == 0 or "transfer_to_human_agents" in names)
t2 = pd.read_csv(REPO + "tau2/tasks.csv")
dt = pd.read_csv("features/dry_tau2.csv").set_index("case_id")
stm = {json.loads(l)["case_id"]: json.loads(l)["text"] for l in open(REPO + "tau2/statements.jsonl")}
# map case t#### -> task index within domain (position in file); airline/retail checked by id order
ids = {}
for dm in ["airline", "retail", "telecom"]:
    ts = json.load(open(TAU + f"{dm}_tasks.json"))
    for i, x in enumerate(ts):
        ids[(dm, i)] = x
rows = []
for cid in dt.index:
    dm = cid.split("/")[0]
    hit = [x for (d2, _), x in ids.items() if d2 == dm and x["user_scenario"]["instructions"]["reason_for_call"].strip() in stm[cid]
           and (x["user_scenario"]["instructions"].get("known_info") or "").strip() in stm[cid]]
    ok = len(hit) >= 1
    lb = max(lab[(dm, str(x["id"]))] for x in hit) if ok else 0
    rows.append((cid, dm, 0, ok and len({lab[(dm, str(x["id"]))] for x in hit}) == 1, lb))
r = pd.DataFrame(rows, columns=["case_id", "domain", "idx", "match", "label"]).set_index("case_id")
out["tau2_mapping_match_rate"] = float(r.match.mean())
r = r.join(dt)
out["tau2_label_positives"] = {k: int(g.label.sum()) for k, g in r.groupby("domain")}
out["tau2_n"] = {k: len(g) for k, g in r.groupby("domain")}
rr = r[r.match]
out["tau2_matched"] = {k: [int(len(g)), int(g.label.sum())] for k, g in rr.groupby("domain")}
out["tau2_auc_overall"] = {s: auc(rr.label, rr[s]) for s in ["p_refuse", "refuse_share_sampled", "n_violated_by_request"]}
sub = rr[rr.domain.isin(["airline", "retail"])]
out["tau2_auc_airline_retail_pooled"] = {s: auc(sub.label, sub[s]) for s in ["p_refuse", "refuse_share_sampled", "n_violated_by_request"]}
out["tau2_auc_by_domain"] = {dm: {s: auc(g.label, g[s]) for s in ["p_refuse", "refuse_share_sampled", "n_violated_by_request"]} for dm, g in rr.groupby("domain") if g.label.sum()}
# reliability of repeated measurement: agreement between main decision and sampled majority
json.dump(out, open("diagnostics.json", "w"), indent=1)
print(json.dumps(out, indent=1))
