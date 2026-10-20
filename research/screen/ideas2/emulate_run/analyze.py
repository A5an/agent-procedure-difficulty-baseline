"""Procedure-level analysis. Primary agent (declared before results): gemini-2.5-flash-lite. Rule: within-domain pair-weighted Spearman >= 0.40
and bootstrap 95% interval lower bound > 0.23, against mean IRT difficulty."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, sys, os, numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
H = os.path.dirname(os.path.abspath(__file__))
REPO = ZR.REPO + "/data/sopbench"
EXCL = {"bank/cancel_credit_card", "bank/pay_bill_with_credit_card"}
W_PERF = 0.35
t = pd.read_csv(f"{REPO}/tasks.csv"); b = pd.read_csv(f"{REPO}/irt/1d_1pl/items.csv", index_col=0).b
t["b"] = t.case_id.map(b); t = t[~t.procedure_id.isin(EXCL)]
key = lambda d, p: f"{d}/{p}"
P = t.groupby(["domain", "procedure_id"]).agg(b=("b", "mean"), n_real=("b", "size")).reset_index()
BAD = {"gpt-5-mode_fc", "gpt-5-mini-mode_fc", "gemini-2.5-pro-mode_fc"}
R = {}
for l in open(f"{REPO}/responses.jsonl"):
    r = json.loads(l); R[r["subject_id"]] = r["responses"]
agents = [a for a in R if a not in BAD]
Y = pd.DataFrame({a: pd.Series(R[a]) for a in agents}).reindex(t.case_id)
t["pub_fail"] = (1 - Y).mean(axis=1).values
P["pub25"] = P.procedure_id.map(t.groupby("procedure_id").pub_fail.mean())
P["tpos"] = None
def pair_rho(d, e, g):
    num = den = 0
    for _, s in d.groupby("domain"):
        n = len(s)
        if n < 3 or s[e].nunique() < 2 or s[g].nunique() < 2: continue
        pr = n * (n - 1) / 2; num += spearmanr(s[e], s[g]).correlation * pr; den += pr
    return num / den if den else np.nan
def boot(d, e, g, n=4000, seed=20261005):
    rng = np.random.default_rng(seed); by = {k: s.reset_index(drop=True) for k, s in d.groupby("domain")}; out = []
    for _ in range(n):
        num = den = 0
        for k, s in by.items():
            m = len(s)
            if m < 3: continue
            x = s.iloc[rng.integers(0, m, m)]
            if x[e].nunique() < 2 or x[g].nunique() < 2: continue
            pr = m * (m - 1) / 2; num += np.corrcoef(rankdata(x[e]), rankdata(x[g]))[0, 1] * pr; den += pr
        out.append(num / den if den else np.nan)
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))
def proc_est(rows, label):
    d = pd.DataFrame([r for r in rows if r["success"] is not None]); d["fail"] = 1 - d.success
    d["procedure_id"] = d.domain + "/" + d.proc
    g = d.groupby(["procedure_id", "perform"]).fail.agg(["mean", "size"]).unstack()
    fp, fr = g["mean"].get(True), g["mean"].get(False)
    est = pd.DataFrame({"fp": fp, "fr": fr, "n_perf": g["size"].get(True), "n_ref": g["size"].get(False), "raw": d.groupby("procedure_id").fail.mean()})
    est["est"] = [W_PERF * a + (1 - W_PERF) * c if not (np.isnan(a) or np.isnan(c)) else (c if np.isnan(a) else a) for a, c in zip(est.fp, est.fr)]
    return est.rename(columns=lambda c: f"{label}_{c}")
def load(run):
    return json.load(open(f"{H}/runs/{run}/results.json"))
def report(name, d, e, g, out):
    r = pair_rho(d, e, g); lo, hi = boot(d, e, g)
    ov = spearmanr(d[e], d[g]).correlation
    out[name] = {"within_domain_rho": round(r, 3), "ci95": [round(lo, 3), round(hi, 3)], "overall_rho": round(float(ov), 3), "n_procedures": int(len(d))}
    print(f"{name:55s} rho {r:.3f} [{lo:.3f}, {hi:.3f}]  overall {ov:.3f}  n={len(d)}")
if __name__ == "__main__":
    runs = [a for a in sys.argv[1:]]   # names like synth_lite real_lite ...
    out = {}; D = P.copy(); D["procedure_id"] = D.procedure_id
    for run in runs:
        e = proc_est(load(run), run); D = D.merge(e, left_on="procedure_id", right_index=True, how="left")
    D.to_csv(f"{H}/proc_table.csv", index=False)
    print("public 25-agent mean failure vs IRT b:"); report("public25 vs b (sanity)", D, "pub25", "b", out)
    for run in runs:
        for col, lab in (("est", "0.35/0.65 estimate"), ("raw", "raw failure")):
            d = D.dropna(subset=[f"{run}_{col}"])
            report(f"{run} {lab} vs mean IRT b", d, f"{run}_{col}", "b", out)
            report(f"{run} {lab} vs public25 failure", d, f"{run}_{col}", "pub25", out)
    # fidelity: synthetic vs real, same agent
    for m in ("lite", "flash"):
        if f"synth_{m}" in runs and f"real_{m}" in runs:
            for col in ("est", "raw"):
                d = D.dropna(subset=[f"synth_{m}_{col}", f"real_{m}_{col}"])
                report(f"FIDELITY {m} synthetic vs real ({col})", d, f"synth_{m}_{col}", f"real_{m}_{col}", out)
            d = D.dropna(subset=[f"real_{m}_est"]); report(f"real {m} 0.35/0.65 est vs b (ceiling of this agent)", d, f"real_{m}_est", "b", out)
    json.dump(out, open(f"{H}/analysis_{'_'.join(runs)}.json", "w"), indent=1)
