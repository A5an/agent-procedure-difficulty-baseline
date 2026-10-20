"""150-case SOPBench sample: modern agents' failures vs public tier difficulty and vs existing predictions."""
import sys, json, pickle, numpy as np, pandas as pd
from pathlib import Path
import score as SC
from cpulock import cpu_lock
HERE = Path(__file__).parent
S = SC.S
mc = json.load(open(HERE / "modern_cases.json"))
sample = pd.read_csv(f"{S}/ideas2/harness/sample.csv")
ids = list(sample.case_id)
M = pd.DataFrame({k: pd.Series({c: 1.0 - float(v[c]) for c in ids}) for k, v in mc.items()})  # failure indicator
GEM = [k for k in M if k.startswith("gemini")]; CC2 = ["CC native", "CC guard"]; CC4 = [k for k in M if k.startswith("CC")]
MT = pd.DataFrame(index=ids)
MT["mod_gem6"] = M[GEM].mean(axis=1); MT["mod_cc2"] = M[CC2].mean(axis=1); MT["mod_cc4"] = M[CC4].mean(axis=1)
MT["mod_all8"] = M[GEM + CC2].mean(axis=1); MT["mod_all10"] = M.mean(axis=1)
MT["g38_H0"] = M["gemini-3.8-flash H0"]; MT["cc_native"] = M["CC native"]
MODT = list(MT.columns)
t, tg = SC.targets("sopbench")
PUB = ["all_irt", "fail_all_clean", "fail_top5", "fail_top8", "fail_bot5", "fail_bot8"]
base = t.loc[ids, ["domain", "procedure_id"]].join(tg.loc[ids, PUB + ["label"]]).join(MT)
base["seed"] = 0
procs = sorted(base.procedure_id.unique()); code = {p: i for i, p in enumerate(procs)}
base["u"] = base.procedure_id.map(code)
by_dom = [np.array(sorted(g.u.unique())) for _, g in base.groupby("domain")]
out = {"n_cases": len(ids), "n_procedures": len(procs), "cases_per_domain": base.domain.value_counts().to_dict(),
       "perform_share": float(base.label.mean()),
       "modern_success": {k: float(1 - M[k].mean()) for k in M}}
rng = np.random.default_rng(7)
def run(F, cells, Mc, Tc, tag):
    pt, dr = SC.boot(F, cells, Mc, Tc, by_dom, len(procs), np.random.default_rng(rng.integers(1 << 31)))
    out[tag] = {m: {t_: [float(pt[i, j]), float(np.nanpercentile(dr[:, i, j], 2.5)), float(np.nanpercentile(dr[:, i, j], 97.5))]
                    for j, t_ in enumerate(Tc)} for i, m in enumerate(Mc)}
    print(tag, flush=True)
with cpu_lock("tiers_modern"):
    run(base, ["domain"], MODT, PUB + ["label"], "modern_vs_public__domain")
    run(base, ["domain", "label"], MODT, PUB, "modern_vs_public__domain_label")
    # among modern: agreement between modern target groups
    run(base, ["domain"], ["mod_gem6", "mod_cc2"], ["mod_gem6", "mod_cc2", "mod_all8"], "modern_vs_modern")
    # existing predictions vs modern targets (SOP, two scenarios, cells = domain)
    srcs = dict(SC.SRC)
    res = {}
    for sch in ["new_procedures", "new_domain"]:
        D = SC.case_diff("sopbench", sch, srcs)
        D = D[D.case_id.isin(ids)].merge(base.drop(columns=["seed"]).reset_index(names="case_id"), on="case_id")
        for var in ["A", "B"]:
            r = pd.DataFrame(json.load(open(f"{S}/ideas2/compile/results_{var}.json")))
            r["perf_hat"] = [int(p in tl) for p, tl in zip(r.procedure, r.tools)]
            ph = r.set_index("case_id").perf_hat
            D[f"pre{var}_only"] = D.case_id.map(ph) + 1e-6 * D["adele"]; D[f"pre{var}_base"] = 2.26 * D.case_id.map(ph) + D["adele"]
        D["label_oracle"] = D["label"] + 1e-6 * D["adele"]
        res[sch] = D
    Mc = [m for m in SC.SRC if all(m in D for D in res.values())] + ["preA_only", "preA_base", "preB_only", "label_oracle"]
    Tc = PUB + MODT
    for sch, D in res.items():
        run(D, ["domain"], Mc, Tc, f"pred_vs_targets__{sch}")
json.dump(out, open(HERE / "modern_results.json", "w"), indent=1)
