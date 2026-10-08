import json, numpy as np
from pathlib import Path
H = Path(__file__).parent / "out"
d = json.load(open(H / "run/eval.json")); d0 = json.load(open(H / "run_lad0/eval.json"))
SC = ["sopbench|new_procedures", "sopbench|new_domain", "tau2|new_procedures", "tau2|new_domain"]
for s in SC:
    d[s]["lad0"] = d0[s]["lad0"]
P = d["pooled_new_process"]; P["lad0"] = d0["pooled_new_process"]["lad0"]
base = {k: np.mean([d[s]["adele"][k] for s in SC]) for k in ("brier_skill", "logloss_skill", "auc_within_config")}
rows = []
order = ["adele", "rub_proc", "rub_lara", "rub_rpa", "len_ridge", "types_ridge", "cat_proc_adele", "grp_proc_adele",
         "cat_lara_adele", "grp_lara_adele", "cat_rpa_adele", "grp_rpa_adele", "cat_full", "grp_full",
         "lad0", "lad1", "lad2", "lad2_adele", "lad3"]
def verdict(m):
    if m == "adele": return "baseline"
    ci = P[m]["minus_baseline_ci"]
    sc_below = any(d[s][m].get("rho_minus_baseline_ci", [0, 1])[1] < 0 for s in SC)
    mean_ok = P[m]["rho_mean"] > P["adele"]["rho_mean"]
    bs = np.mean([d[s][m]["brier_skill"] for s in SC]); ll = np.mean([d[s][m]["logloss_skill"] for s in SC])
    if ci[1] < 0: return "worse"
    if mean_ok and ci[0] > 0 and not sc_below and bs >= base["brier_skill"] and ll >= base["logloss_skill"]:
        return "better (criteria 1-4 met, 5 unconfirmed)"
    return "not better"
lines = ["| method | pooled rho | 95% CI | worst scen. | sop-proc | sop-dom | tau2-proc | tau2-dom | paired diff vs baseline 95% CI | Brier skill | logloss skill | within-agent AUC | verdict |", "|" + "---|" * 13]
for m in order:
    r = P[m]; sc = r["rho_by_scenario"]
    diff = r.get("minus_baseline_ci", None)
    bs = np.mean([d[s][m]["brier_skill"] for s in SC]); ll = np.mean([d[s][m]["logloss_skill"] for s in SC])
    au = np.mean([d[s][m]["auc_within_config"] for s in SC])
    lines.append(f"| {m} | {r['rho_mean']:+.3f} | [{r['rho_ci'][0]:+.3f}, {r['rho_ci'][1]:+.3f}] | {r['rho_worst']:+.3f} | " +
                 " | ".join(f"{x:+.3f}" for x in sc) + " | " +
                 (f"[{diff[0]:+.3f}, {diff[1]:+.3f}]" if diff else "-") + f" | {bs:+.3f} | {ll:+.3f} | {au:.3f} | {verdict(m)} |")
(H / "results_table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
# ladder per scenario
print()
for m in ["adele", "lad0", "lad1", "lad2", "lad2_adele", "lad3"]:
    print(m, [(round(d[s][m]["logloss_skill"], 3), round(d[s][m]["auc_within_config"], 3), round(d[s][m].get("rho_within_domain", float("nan")), 3)) for s in SC])
# per-benchmark random_cases sanity (tauk_banking only scheme)
print("tauk random_cases:", {m: (round(d["tauk_banking|random_cases"][m].get("rho_within_domain", float('nan')), 3), round(d["tauk_banking|random_cases"][m]["logloss_skill"], 3)) for m in ["adele", "rub_proc", "lad1", "lad2", "lad2_adele", "lad3", "grp_proc_adele"]})
