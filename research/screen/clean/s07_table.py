import json, numpy as np
NEW=["sopbench|new_procedures","sopbench|new_domain","tau2|new_procedures","tau2|new_domain"]
meths=["adele","length","human_time","shares_adele","bb_adele","lad2_adele","grp_proc_adele","shares_adele+adele","lad2_adele+adele","grp_proc_adele+adele"]
E={v:json.load(open(f"results_{v}/eval.json")) for v in ["old","A","B"]}
def row(v,m):
    e=E[v]; p=e["pooled_new_process"].get(m)
    if p is None: return None
    br=np.mean([e[k][m]["brier_skill"] for k in NEW]); ll=np.mean([e[k][m]["logloss_skill"] for k in NEW])
    ci=p["rho_ci"]; d=p.get("minus_baseline_ci")
    return (f"{p['rho_mean']:+.3f}",f"[{ci[0]:+.3f}, {ci[1]:+.3f}]",f"{p['rho_worst']:+.3f}",
            " / ".join(f"{x:+.3f}" for x in p["rho_by_scenario"]),
            f"[{d[0]:+.3f}, {d[1]:+.3f}]" if d else "-", f"{br:+.3f}",f"{ll:+.3f}")
L=["| method | data | pooled rho | 95% CI | worst | sop-proc / sop-dom / tau2-proc / tau2-dom | paired CI vs baseline of the same data | Brier skill | log-loss skill |","|---|---|---|---|---|---|---|---|---|"]
for m in meths:
    for v,lab in [("old","28 agents"),("A","A: 25 agents"),("B","B: 28, cells removed")]:
        r=row(v,m)
        if r: L.append(f"| {m} | {lab} | "+" | ".join(r)+" |")
open("table_main.md","w").write("\n".join(L)+"\n")
# paired CI of cleaned method vs OLD baseline is not defined (different targets); skip
# sopbench-only scenario detail for baseline
for v in E:
    for k in NEW[:2]:
        r=E[v][k]["adele"]; print(v,k,{a:(round(b,3) if isinstance(b,float) else b) for a,b in r.items()})
print("\n".join(L))
