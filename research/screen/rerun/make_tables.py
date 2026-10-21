import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, numpy as np
from pathlib import Path
H = Path(__file__).parent
ev = json.load(open(H / "out_final/eval.json")); sop = json.load(open(H / "out_sop/eval.json"))
old = json.load(open(ZR.DOCS + "/zangir-2026-09-28-baselines/methodology/screen_eval.json"))
NEW = ["sopbench|new_procedures", "sopbench|new_domain", "tau2|new_procedures", "tau2|new_domain"]
P = ev["pooled_new_process"]
def f(x): return f"{x:+.3f}"
def ci(c): return f"[{c[0]:+.3f}, {c[1]:+.3f}]"
def verdict(col, keys, pooled, E):
    b = pooled["adele"]; r = pooled[col]
    if col == "adele": return "baseline"
    d = r.get("minus_baseline_ci")
    c1 = r["rho_mean"] > b["rho_mean"] and d and d[0] > 0
    c2 = all(not (E[k][col].get("rho_minus_baseline_ci", [0, 1])[1] < 0) for k in keys)
    sk = lambda m: (np.mean([E[k][m]["brier_skill"] for k in keys]), np.mean([E[k][m]["logloss_skill"] for k in keys]))
    c4 = sk(col)[0] >= sk("adele")[0] and sk(col)[1] >= sk("adele")[1]
    worse = d and d[1] < 0
    return "better (criteria 1,2,4 met; 3,5 not testable here)" if c1 and c2 and c4 else ("worse" if worse else "not better")
rows = []
order = [c for c in P if c not in ("adele",)] 
def line(col, pooled, E, keys):
    r = pooled[col]; d = r.get("minus_baseline_ci")
    bs = np.mean([E[k][col]["brier_skill"] for k in keys]); ls = np.mean([E[k][col]["logloss_skill"] for k in keys])
    return f"| {col} | {f(r['rho_mean'])} | {ci(r['rho_ci'])} | {f(r['rho_worst'])} | " + " / ".join(f(x) for x in r["rho_by_scenario"]) + f" | {ci(d) if d else '-'} | {bs:+.3f} | {ls:+.3f} | {verdict(col, keys, pooled, E)} |"
hdr = "| method | pooled rho | 95% CI | worst scenario | rho by scenario (sop NP / sop ND / tau2 NP / tau2 ND) | paired 95% CI vs baseline | mean Brier skill | mean log-loss skill | verdict |\n|---|---|---|---|---|---|---|---|---|"
out = ["### Four new-process scenarios (SOPBench and tau2)", hdr, line("adele", P, ev, NEW)]
for c in sorted([c for c in P if c not in ("adele", "constant", "oracle")], key=lambda c: -P[c]["rho_mean"]):
    out.append(line(c, P, ev, NEW))
Ps = sop["pooled_new_process"]; K2 = NEW[:2]
out += ["", "### SOPBench only (two scenarios), for the BPM metrics", hdr.replace("sop NP / sop ND / tau2 NP / tau2 ND", "sop NP / sop ND"), line("adele", Ps, sop, K2)]
for c in ["bpm_complexity", "bpm_complexity+adele"]:
    out.append(line(c, Ps, sop, K2))
# old vs new
out += ["", "### Old preliminary number vs clean rerun (pooled mean of per-scenario rho, within-domain)", "| method | old, 4 scenarios (or SOPBench 2) | old per scenario | clean rerun | clean per scenario |", "|---|---|---|---|---|"]
M = {"entropy": "entropy", "probe_L18": "probe_L18", "probe_L36": "probe_L36", "llm_forecast": "llm_forecast", "emb_bge": "emb_bge", "amortized_irt_bge": "amortized_irt", "adele": "adele"}
for new, o in M.items():
    ov = [old[k][o]["rho_within_domain"] for k in NEW]
    out.append(f"| {new} | {f(np.mean(ov))} | {' / '.join(f(x) for x in ov)} | {f(P[new]['rho_mean'])} | {' / '.join(f(x) for x in P[new]['rho_by_scenario'])} |")
ov = [old[k]["bpm_complexity"]["rho_within_domain"] for k in NEW[:2]]
out.append(f"| bpm_complexity (SOPBench) | {f(np.mean(ov))} | {' / '.join(f(x) for x in ov)} | {f(Ps['bpm_complexity']['rho_mean'])} | {' / '.join(f(x) for x in Ps['bpm_complexity']['rho_by_scenario'])} |")
(H / "tables.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
