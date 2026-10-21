import json, numpy as np
from pathlib import Path
J = json.load(open(Path(__file__).parent / "out/all/eval.json"))
P = J["pooled_new_process"]
NEW = ["sopbench|new_procedures", "sopbench|new_domain", "tau2|new_procedures", "tau2|new_domain"]
base = "adele"
def sk(c, k): return np.mean([J[s][c][k] for s in NEW])
rows = []
for c, r in P.items():
    if c in ("constant", "oracle"): continue
    bsk, lsk = sk(c, "brier_skill"), sk(c, "logloss_skill")
    if c == base:
        v = "baseline"
    else:
        d = r["minus_baseline_ci"]
        scen_below = [s for s in NEW if J[s][c]["rho_minus_baseline_ci"][1] < 0]
        c1 = r["rho_mean"] > P[base]["rho_mean"] and d[0] > 0
        c2 = not scen_below
        c4 = bsk >= sk(base, "brier_skill") and lsk >= sk(base, "logloss_skill")
        v = "worse" if (d[1] < 0 or scen_below) else ("better (criteria 1,2,4; 3,5 untested)" if (c1 and c2 and c4) else "not better")
    sc = " / ".join(f"{x:+.3f}" for x in r["rho_by_scenario"])
    dci = "" if c == base else f"[{r['minus_baseline_ci'][0]:+.3f}, {r['minus_baseline_ci'][1]:+.3f}]"
    rows.append((r["rho_mean"], f"| {c} | {r['rho_mean']:+.3f} | [{r['rho_ci'][0]:+.3f}, {r['rho_ci'][1]:+.3f}] | {r['rho_worst']:+.3f} | {sc} | {dci} | {bsk:+.3f} | {lsk:+.3f} | {v} |"))
print("\n".join(x[1] for x in sorted(rows, key=lambda x: -x[0])))
