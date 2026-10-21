"""Extra diagnostics: per-class fidelity, case-level noise, class-mix sensitivity, empty answers, cost. No LLM calls."""
import json, sys, numpy as np, pandas as pd, glob
from scipy.stats import spearmanr
import analyze as A
runs = sys.argv[1:]
out = {}
D = pd.read_csv(f"{A.H}/proc_table.csv")
def rho(d, e, g): return round(A.pair_rho(d.dropna(subset=[e, g]), e, g), 3)
for m in ("lite", "flash"):
    if f"real_{m}_fp" not in D: continue
    for cls, c in (("perform", "fp"), ("refuse", "fr")):
        out[f"fidelity_{m}_{cls}_class_failure"] = {"within_domain": rho(D, f"synth_{m}_{c}", f"real_{m}_{c}"),
                                                     "overall": round(float(D[[f"synth_{m}_{c}", f"real_{m}_{c}"]].dropna().corr("spearman").iloc[0, 1]), 3)}
    for w in (0.2, 0.35, 0.5, 0.65, 0.8):   # sensitivity to the perform weight, applied to both synthetic and real
        D[f"s{w}"] = w * D[f"synth_{m}_fp"].fillna(D[f"synth_{m}_fr"]) + (1 - w) * D[f"synth_{m}_fr"].fillna(D[f"synth_{m}_fp"])
        out[f"weight_sensitivity_{m}_w{w}_vs_b"] = rho(D, f"s{w}", "b")
# case-level noise of the synthetic estimate: resample cases within procedure and class, recompute rho vs b
for m in ("lite", "flash"):
    rows = json.load(open(f"{A.H}/runs/synth_{m}/results.json")); d = pd.DataFrame(rows); d["fail"] = 1 - d.success; d["pid"] = d.domain + "/" + d.proc
    rng = np.random.default_rng(1); rs = []
    grp = {k: g.fail.values for k, g in d.groupby(["pid", "perform"])}
    for _ in range(300):
        est = {}
        for pid in D.procedure_id:
            fp = grp.get((pid, True)); fr = grp.get((pid, False))
            a = rng.choice(fp, len(fp)).mean() if fp is not None else np.nan; c = rng.choice(fr, len(fr)).mean() if fr is not None else np.nan
            est[pid] = A.W_PERF * a + (1 - A.W_PERF) * c if not (np.isnan(a) or np.isnan(c)) else (c if np.isnan(a) else a)
        x = D.copy(); x["e"] = x.procedure_id.map(est); rs.append(A.pair_rho(x, "e", "b"))
    out[f"case_resampling_{m}_rho_vs_b"] = {"mean": round(float(np.nanmean(rs)), 3), "ci95": [round(float(np.nanpercentile(rs, 2.5)), 3), round(float(np.nanpercentile(rs, 97.5)), 3)]}
    out[f"n_cases_per_proc_{m}"] = {"median": float(d.groupby("pid").size().median()), "min": int(d.groupby("pid").size().min())}
# empty answers / behaviour
for run in runs:
    d = pd.DataFrame(json.load(open(f"{A.H}/runs/{run}/results.json"))); d = d[d.success.notna()]
    out[f"behaviour_{run}"] = {"n": int(len(d)), "perform_success": round(float(d[d.perform].success.mean()), 3), "refuse_success": round(float(d[~d.perform].success.mean()), 3),
        "refuse_success_with_no_tool_call_and_no_text(empty answer artefact)": int(((~d.perform) & (d.success == 1) & (d.n_toolcalls == 0) & (d.n_empty == d.n_asst)).sum()),
        "share_cases_with_an_empty_assistant_message": round(float((d.n_empty > 0).mean()), 3), "called_target_perform": round(float(d[d.perform].called_target.mean()), 3),
        "called_target_refuse": round(float(d[~d.perform].called_target.mean()), 3)}
json.dump(out, open(f"{A.H}/extra_diagnostics.json", "w"), indent=1); print(json.dumps(out, indent=1))
