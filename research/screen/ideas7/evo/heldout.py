"""Single held-out look at the finalists (FINALISTS.md must exist). Also writes feature tables for the full-protocol harness.
A: same scoring as the search, inside held-out domains. B: ridge fitted on search cases, applied to held-out cases (transfer)."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "."); import evaluator as EV
from scipy.stats import spearmanr
fin = json.load(open("finalists.json"))
ts, _ = EV.load("search"); th, _ = EV.load("heldout"); tall = pd.read_csv("targets.csv")
res = {}
for f in fin:
    n = f["n"]; code = f["code"]
    d, err = EV.run_sandbox(code, "records.jsonl")
    assert err is None, err
    ids = [r["id"] for r in d["rows"]]; names = f["names"]
    F = np.full((len(ids), len(names)), np.nan)
    for i, r in enumerate(d["rows"]):
        if r["f"]:
            for j, k in enumerate(names):
                if r["f"].get(k) is not None: F[i, j] = r["f"][k]
    df = pd.DataFrame(F, index=ids, columns=names)
    sidx = df.index.isin(ts.case_id)
    df = df.fillna(df[sidx].median()).fillna(0.0).clip(-1e6, 1e6)
    df.index.name = "case_id"; df.to_csv(f"feat_f{n}.csv")
    # A
    oa = EV.score(code, "heldout")
    # B
    S = df.loc[ts.case_id].to_numpy(); H = df.loc[th.case_id].to_numpy()
    ds, dh = ts.domain.values, th.domain.values
    Xs, Xh = EV.demean(S, ds), EV.demean(H, dh)
    ys = EV.demean(ts.y_irt_b.values[:, None], ds)[:, 0]; bs = EV.demean(ts.clad_oof.values[:, None], ds)[:, 0]
    yh = EV.demean(th.y_irt_b.values[:, None], dh)[:, 0]; bh = EV.demean(th.clad_oof.values[:, None], dh)[:, 0]
    pf, al = EV.fit_pred(Xs, ys, bs, Xh, bh, ts.procedure_id.values, np.random.default_rng(1))
    pb = (bs @ ys / (bs @ bs)) * bh
    rf, df_ = EV.within_rho(pf, yh, dh); rb, db_ = EV.within_rho(pb, yh, dh)
    sop = [k for k in df_ if k != "telecom"]
    res[n] = {"id": f["id"], "names": names, "search_score": f["search"], "fresh_score": f["fresh"],
              "A": {k: oa.get(k) for k in ["status", "reason", "score", "rho_with", "rho_base", "per_domain_gain", "n_features", "feature_names", "max_corr_with_clad"]},
              "B": {"gain": rf - rb, "rho_with": rf, "rho_base": rb, "alpha": al, "per_domain_gain": {k: df_[k] - db_[k] for k in df_},
                    "gain_sop3": float(np.mean([df_[k] - db_[k] for k in sop]))}}
    if oa.get("per_domain_gain"):
        pd_ = oa["per_domain_gain"]; res[n]["A"]["gain_sop3"] = float(np.mean([pd_[k] for k in pd_ if k != "telecom"]))
    print(n, json.dumps(res[n], indent=1))
json.dump(res, open("heldout_results.json", "w"), indent=1)
