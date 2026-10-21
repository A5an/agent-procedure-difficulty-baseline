"""Model-specific check and diagnostics (items 4 and 5). Python: repo venv."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, numpy as np, pandas as pd, warnings
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
HERE = ZR.SCREEN + "/ideas3/paper_pilot"
REPO = ZR.REPO
tasks = pd.read_csv(f"{REPO}/data/sopbench/tasks.csv").set_index("case_id")
items = pd.read_csv(f"{REPO}/data/sopbench/irt/1d_1pl/items.csv", index_col=0)
adele = pd.read_csv(f"{REPO}/data/sopbench/features/adele.csv"); adele = adele.set_index(adele.columns[0])
dom = tasks.domain
# population baseline: ADeLe ridge -> IRT difficulty b, leave-one-domain-out
b = items.b.reindex(tasks.index); sbase = pd.Series(np.nan, index=tasks.index)
for d in dom.unique():
    tr = dom != d; sc = StandardScaler().fit(adele[tr]); m = Ridge(alpha=100).fit(sc.transform(adele[tr]), b[tr])
    sbase[~tr] = -m.predict(sc.transform(adele[~tr]))
resp = {json.loads(l)["subject_id"]: json.loads(l)["responses"] for l in open(f"{REPO}/data/sopbench/responses.jsonl")}
real = json.load(open(f"{HERE}/real_h0.json"))
Y = {"f25_public830": pd.Series(resp["gemini-2.5-flash-mode_fc"]).astype(float),
     "f25_H0_150": pd.Series({k: v["success"] for k, v in real["gemini-2.5-flash"].items()}).astype(float),
     "g38_H0_150": pd.Series({k: v["success"] for k, v in real["gemini-3.8-flash"].items()}).astype(float)}
F = {mk: pd.read_csv(f"{HERE}/features/pf_{mk}_sopbench.csv").set_index("case_id") for mk in ("lite", "g38", "f25")}

def lodo(X, y, doms, C=0.3):
    out = pd.Series(np.nan, index=y.index)
    for d in doms.unique():
        te = doms == d; tr = ~te
        if y[tr].nunique() < 2: continue
        sc = StandardScaler().fit(X[tr]); Xt = np.nan_to_num(sc.transform(X[tr])); Xe = np.nan_to_num(sc.transform(X[te]))
        m = LogisticRegression(C=C, max_iter=500).fit(Xt, y[tr]); out[te] = m.decision_function(Xe)
    return out

def auc(y, s):
    return roc_auc_score(y, s) if y.nunique() == 2 else np.nan
def wd(y, s, d, lab=None):
    """mean within-domain AUC and Spearman (cells = domain, or domain x label)"""
    cells = d.astype(str) if lab is None else d.astype(str) + "|" + lab.astype(str)
    a, r = [], []
    for c in cells.unique():
        i = cells == c
        if i.sum() < 6 or y[i].nunique() < 2: continue
        a.append(roc_auc_score(y[i], s[i])); r.append(spearmanr(y[i], s[i])[0])
    return (np.mean(a) if a else np.nan, np.mean(r) if r else np.nan, len(a))
def stats(y, s, d, lab):
    return dict(auc=auc(y, s), wd_auc=wd(y, s, d)[0], wd_rho=wd(y, s, d)[1], lc_rho=wd(y, s, d, lab)[1])
def boot(y, ss, d, lab, B=500, seed=0):
    rng = np.random.default_rng(seed); ids = np.arange(len(y)); res = {k: [] for k in ss}
    Y_ = y.values; D_ = d.values; L_ = lab.values
    for _ in range(B):
        ix = rng.integers(0, len(y), len(y)); yy = pd.Series(Y_[ix]); dd = pd.Series(D_[ix]); ll = pd.Series(L_[ix])
        if yy.nunique() < 2: continue
        for k, s in ss.items():
            v = stats(yy, pd.Series(s.values[ix]), dd, ll); res[k].append([v["auc"], v["wd_rho"], v["lc_rho"]])
    return {k: np.array(v) for k, v in res.items()}

rows = []; out = {}
for yk, y in Y.items():
    ids = y.index; yy = y.copy(); d = dom[ids]; lab = tasks.should_succeed[ids]
    own = "f25" if yk.startswith("f25") else "g38"
    scores = {"base_adele": sbase[ids]}
    for mk in ("lite", "g38", "f25"):
        if yk == "g38_H0_150" and mk == "f25" or yk != "g38_H0_150" and mk == "g38" and False: pass
    for mk in ("f25", "g38", "lite"):
        X = F[mk].loc[ids]
        if "frac_bad" in X:
            scores[f"{mk}_risk_fixed"] = -X.frac_bad
        scores[f"{mk}_lr"] = lodo(X, yy, d)
        Xb = X.copy(); Xb["sbase"] = sbase[ids]; scores[f"{mk}_lr+base"] = lodo(Xb, yy, d)
    ok = pd.DataFrame(scores).notna().all(axis=1); yy = yy[ok]; d = d[ok]; lab = lab[ok]
    scores = {k: v[ok] for k, v in scores.items()}
    bt = boot(yy, scores, d, lab)
    res = {}
    for k, s in scores.items():
        v = stats(yy, s, d, lab); cb = bt[k]; bb = bt["base_adele"]
        n = min(len(cb), len(bb)); diff = cb[:n] - bb[:n]
        v.update({f"{m}_ci": [float(np.nanpercentile(cb[:, j], 2.5)), float(np.nanpercentile(cb[:, j], 97.5))] for j, m in enumerate(["auc", "wd_rho", "lc_rho"])})
        v["wd_rho_minus_base_ci"] = [float(np.nanpercentile(diff[:, 1], 2.5)), float(np.nanpercentile(diff[:, 1], 97.5))]
        v["auc_minus_base_ci"] = [float(np.nanpercentile(diff[:, 0], 2.5)), float(np.nanpercentile(diff[:, 0], 97.5))]
        res[k] = v
    out[yk] = {"n": int(len(yy)), "success_rate": float(yy.mean()), "scores": res}
    print(f"\n== {yk} n={len(yy)} success={yy.mean():.3f}")
    for k, v in res.items():
        print(f"  {k:16s} AUC {v['auc']:.3f} {np.round(v['auc_ci'],3)}  wd_auc {v['wd_auc']:.3f}  wd_rho {v['wd_rho']:+.3f} {np.round(v['wd_rho_ci'],3)}  lc_rho {v['lc_rho']:+.3f}  d_rho vs base {np.round(v['wd_rho_minus_base_ci'],3)}  d_auc {np.round(v['auc_minus_base_ci'],3)}")
json.dump(out, open(f"{HERE}/model_specific.json", "w"), indent=1)

# ---- diagnostics: plan decision vs label
parsed = json.load(open(f"{HERE}/plans_parsed.json")); diag = {}
for mk in ("lite", "g38", "f25"):
    p = parsed[mk]; ids = [c for c in tasks.index if p.get(c)]
    dec = pd.Series({c: p[c]["dec"] for c in ids}); lab = tasks.should_succeed[ids]
    dp = (dec == "perform").astype(int)
    diag[mk] = {"n": len(ids), "share_perform": float(dp.mean()), "share_refuse": float((dec == "refuse").mean()), "share_transfer": float((dec == "transfer").mean()),
                "acc_vs_label": float((dp == lab).mean()), "acc_must_perform": float(dp[lab == 1].mean()), "acc_must_refuse": float(1 - dp[lab == 0].mean()),
                "refusal_path_share_perform": float(F[mk].refusal_path.mean()),
                "auc_refusal_path_vs_refuse_label": float(roc_auc_score(1 - lab, F[mk].loc[ids].refusal_path)) if F[mk].refusal_path.nunique() > 1 else None,
                "label_share": float(lab.mean())}
tp = {mk: parsed[mk] for mk in parsed}
print("\nplan decision vs label", json.dumps(diag, indent=1))
# tau2 decisions
tdiag = {}
for mk in ("lite", "g38", "f25"):
    t = pd.read_csv(f"{HERE}/features/pf_{mk}_tau2.csv").set_index("case_id")
    tdiag[mk] = {"mean_n_calls": float(t.n_calls.mean()), "perform": float(t.dec_perform.mean()), "refuse": float(t.dec_refuse.mean()), "transfer": float(t.dec_transfer.mean()),
                 "write_before_read": float(t.write_before_read.mean()), "calls_write": float((t.n_write > 0).mean())}
print("tau2", json.dumps(tdiag, indent=1))

# ---- diagnostics: missing checks in the plan vs checks the real agent skipped
steps = pd.read_csv(f"{HERE}/../../ideas2/steps/steps.csv"); chk = steps[steps.kind.isin(["check", "fetch", "link"])].copy()
chk["alts"] = chk.tools.str.split("|")
so = pd.read_csv(f"{HERE}/../../ideas2/steps/step_outcomes.csv.gz")
so = so[so.kind.isin(["check", "fetch", "link"])]
def plan_has(mk, cid, alts):
    p = parsed[mk].get(cid)
    return None if p is None else int(any(a in p["calls"] for a in alts))
chk["plan_f25"] = [plan_has("f25", c, a) for c, a in zip(chk.case_id, chk.alts)]
chk["plan_g38"] = [plan_has("g38", c, a) for c, a in zip(chk.case_id, chk.alts)]
chk["plan_lite"] = [plan_has("lite", c, a) for c, a in zip(chk.case_id, chk.alts)]
# real verified: public f25 from step_outcomes, fresh H0 from tool calls
pub = so[so.agent == "gemini-2.5-flash-mode_fc"][["case_id", "step_id", "verified"]].rename(columns={"verified": "real_pub_f25"})
chk = chk.merge(pub, on=["case_id", "step_id"], how="left")
for key, rk in (("f25_H0", "gemini-2.5-flash"), ("g38_H0", "gemini-3.8-flash")):
    chk[f"real_{key}"] = [(int(any(a in real[rk][c]["tools"] for a in al)) if c in real[rk] else np.nan) for c, al in zip(chk.case_id, chk.alts)]
allv = so.groupby(["case_id", "step_id"]).verified.mean().rename("skip_rate_all").reset_index(); allv["skip_rate_all"] = 1 - allv.skip_rate_all
chk = chk.merge(allv, on=["case_id", "step_id"], how="left")
sd = {}
def agree(plan, realc, name):
    x = chk[[plan, realc]].dropna(); x = x.astype(float)
    if len(x) < 10: return
    omit = x[plan] == 0; skip = x[realc] == 0
    phi = np.corrcoef(omit, skip)[0, 1] if omit.nunique() > 1 and skip.nunique() > 1 else np.nan
    sd[name] = {"n_steps": int(len(x)), "plan_omits": float(omit.mean()), "real_skips": float(skip.mean()),
                "p_skip_given_plan_omits": float(skip[omit].mean()) if omit.any() else None, "p_skip_given_plan_includes": float(skip[~omit].mean()),
                "phi": float(phi)}
    # case level
    g = x.assign(c=chk.loc[x.index, "case_id"], om=omit, sk=skip).groupby("c")[["om", "sk"]].sum()
    sd[name]["case_rho_nomitted_vs_nskipped"] = float(spearmanr(g.om, g.sk)[0]) if g.om.nunique() > 1 and g.sk.nunique() > 1 else None
for pm in ("f25", "g38", "lite"):
    for rk in ("real_pub_f25", "real_f25_H0", "real_g38_H0"):
        agree(f"plan_{pm}", rk, f"plan_{pm} vs {rk}")
for pm in ("f25", "g38", "lite"):
    x = chk[[f"plan_{pm}", "skip_rate_all"]].dropna()
    sd[f"plan_{pm} omission vs skip rate over 28 agents"] = {"n_steps": int(len(x)), "mean_skip_rate_if_plan_omits": float(x[x[f"plan_{pm}"] == 0].skip_rate_all.mean()),
        "mean_skip_rate_if_plan_includes": float(x[x[f"plan_{pm}"] == 1].skip_rate_all.mean()), "spearman": float(spearmanr(1 - x[f"plan_{pm}"], x.skip_rate_all)[0])}
print("\nstep alignment", json.dumps(sd, indent=1))
json.dump({"decision": diag, "tau2": tdiag, "steps": sd}, open(f"{HERE}/diagnostics.json", "w"), indent=1)
