import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
HERE = Path(__file__).parent
REPO = Path(ZR.REPO)
p = pd.read_csv(HERE/"features/plan_tau2.csv").set_index("case_id"); g = pd.read_csv(HERE/"gold_tau2.csv").set_index("case_id")
it = pd.read_csv(REPO/"data/tau2/irt/1d_1pl/items.csv", index_col=0)["b"]
d = p.join(g).join(it); d["dom"] = d.index.str.split("/").str[0]
res = {}
def rho(a, b):
    if a.nunique() < 2 or b.nunique() < 2: return None
    return round(float(spearmanr(a, b)[0]), 3)
for dom, x in list(d.groupby("dom")) + [("all", d)]:
    res[dom] = dict(n=len(x), plan_calls_vs_gold_len=rho(x.plan_calls, x.gold_len), plan_writes_vs_gold_writes=rho(x.plan_writes, x.gold_writes),
        plan_calls_vs_gold_agent_len=rho(x.plan_calls, x.gold_agent_len), plan_distinct_vs_gold_distinct=rho(x.plan_distinct_tools, x.gold_distinct),
        mean_plan_calls=round(x.plan_calls.mean(),2), mean_gold_len=round(x.gold_len.mean(),2), mean_plan_writes=round(x.plan_writes.mean(),2), mean_gold_writes=round(x.gold_writes.mean(),2),
        b_vs_gold_len=rho(x.b, x.gold_len), b_vs_plan_calls=rho(x.b, x.plan_calls), b_vs_plan_writes=rho(x.b, x.plan_writes),
        **{f"b_vs_{c}": rho(x.b, x[c]) for c in ["plan_rules","plan_facts","plan_refuse","plan_ambig","plan_distinct_tools"]},
        n_distinct_plan_vectors=int(x[[c for c in p.columns]].drop_duplicates().shape[0]))
# SOPBench
ps = pd.read_csv(HERE/"features/plan_sopbench.csv").set_index("case_id")
t = pd.read_csv(REPO/"data/sopbench/tasks.csv").set_index("case_id"); b = pd.read_csv(REPO/"data/sopbench/irt/1d_1pl/items.csv", index_col=0)["b"]
s = ps.join(t[["domain","should_succeed"]]).join(b)
res["sop_all"] = {c: rho(s.b, s[c]) for c in ps.columns}
res["sop_by_label_refuse_vs_plan_refuse"] = dict(plan_refuse_by_label=s.groupby("should_succeed").plan_refuse.mean().round(3).to_dict())
from sklearn.metrics import roc_auc_score
res["sop_auc_plan_refuse_vs_label_refuse"] = round(roc_auc_score(1-s.should_succeed, s.plan_refuse), 3)
res["sop_auc_plan_calls_vs_label"]=round(roc_auc_score(s.should_succeed, s.plan_calls),3)
# within-domain SOP rho
res["sop_within_domain"] = {c: round(float(np.nanmean([spearmanr(x.b, x[c])[0] for _, x in s.groupby("domain") if x[c].nunique()>1])),3) for c in ps.columns}
# stability
st = json.load(open(HERE/"stability.json")); sd = pd.DataFrame(st).T
sd["bench"] = np.where(sd.index.str.startswith(("airline","retail","telecom")), "tau2", "sopbench")
res["stability_mean_std_across_3_samples"] = sd.groupby("bench").mean(numeric_only=True).round(3).to_dict("index")
# rank stability of plan_calls itself: correlation temp0 vs mean of sampled
t1 = json.load(open(HERE/"plans_t1.json")); t0 = json.load(open(HERE/"plans_t0.json"))
import feats
rows=[]
for k,v in t1.items():
    f0=feats.feats(feats.parse(t0[k])); fs=[feats.feats(feats.parse(x)) for x in v if feats.parse(x)]
    if fs: rows.append(dict(k=k, b=k.split("|")[0], c0=f0["plan_calls"], cs=np.mean([f["plan_calls"] for f in fs]), w0=f0["plan_writes"], ws=np.mean([f["plan_writes"] for f in fs]), r0=f0["plan_refuse"], rs=np.mean([f["plan_refuse"] for f in fs])))
r=pd.DataFrame(rows)
res["stability_rho_t0_vs_mean_samples"] = {b: dict(calls=rho(x.c0,x.cs), writes=rho(x.w0,x.ws), refuse=rho(x.r0,x.rs), n=len(x)) for b,x in r.groupby("b")}
# does b correlate with instability?
sd2 = sd.join(it.rename("b_tau"), how="left").join(b.rename("b_sop"), how="left"); sd2["b"]=sd2.b_tau.fillna(sd2.b_sop)
res["instab_calls_vs_b"] = {bn: rho(x.plan_calls, x.b) for bn, x in sd2.groupby("bench")}
# telecom
tel = d[d.dom=="telecom"]; txt = {json.loads(l)["case_id"]: json.loads(l)["text"] for l in open(REPO/"data/tau2/statements.jsonl")}
tel = tel.assign(txt=[hash(txt[c]) for c in tel.index])
res["telecom"] = dict(n_texts=int(tel.txt.nunique()), plan_features_per_text=tel.groupby("txt")[["plan_calls","plan_writes"]].first().to_dict("list"),
    var_b_within_text_share=round(float(1 - tel.groupby("txt").b.transform("mean").var()/tel.b.var()),3) if False else round(float(tel.groupby("txt").b.apply(lambda v: ((v-v.mean())**2).sum()).sum()/((tel.b-tel.b.mean())**2).sum()),3),
    gold_len_vs_b=rho(tel.b, tel.gold_len), gold_writes_vs_b=rho(tel.b, tel.gold_writes))
json.dump(res, open(HERE/"validation.json","w"), indent=1); print(json.dumps(res, indent=1))
