"""Paired scoring: SOPBench two scenarios from my out/, tau2 two scenarios = c_lad (copied column, so the paired difference is 0 there).
tiers/score.py bootstrap over procedures stratified by domain, 300 draws, same draws for all columns."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, pickle, numpy as np, pandas as pd
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas"); sys.path.insert(0, S + "/ideas3/tiers")
from pathlib import Path
import score as SC
import contextlib
cpu_lock = lambda n: contextlib.nullcontext()  # shared lock queue was 20+ min, light job run with nice
HERE = Path(S + "/ideas7/case_signals")
V = ["s1_nli", "s2_ret", "s3_act", "s4_bnd", "s5_all", "c_lad_re"]
srcs = {"adele": SC.R, "c_lad": S + "/ideas3/combo/out_main", **{v: str(HERE / "out") for v in V}}
BROKEN = {"bank/cancel_credit_card", "bank/pay_bill_with_credit_card"}
gr = pd.read_csv(f"{S}/ideas5/smart_gemini/real_results.csv"); sr = pd.read_csv(f"{S}/ideas5/smart_sonnet/real_results.csv")
smart = (1 - (gr.set_index("case_id").success + sr.set_index("case_id").success) / 2).rename("smart_fail")
rng = np.random.default_rng(20261007); res = {}
with cpu_lock("case_signals_score"):
    W = SC.build(srcs, ["all_irt"])
    for bench in ["sopbench", "tau2"]:
        for sch in ["new_procedures", "new_domain"]:
            D0 = W[(bench, sch)]
            for nb in ([False, True] if bench == "sopbench" else [False]):
                D = D0.copy()
                if nb: D = D[~D.procedure_id.isin(BROKEN)].copy()
                M = ["adele", "c_lad"] + V; TG = ["all_irt"]
                if bench == "tau2":
                    for v in V: D[v] = D["c_lad"]
                else: D["smart_fail"] = D.case_id.map(smart); TG = ["all_irt", "smart_fail"]
                procs = sorted(D.procedure_id.unique()); code = {p: i for i, p in enumerate(procs)}; D["u"] = D.procedure_id.map(code)
                by_dom = [np.array(sorted(g.u.unique())) for _, g in D.groupby("domain")]
                for level in ["case", "proc"]:
                    if level == "proc" and bench != "sopbench": continue
                    if level == "proc":
                        F = D.groupby(["seed", "domain", "procedure_id"], as_index=False)[M + TG].mean(); F["u"] = F.procedure_id.map(code); cells = ["domain"]
                    else: F, cells = D, ["fold", "domain"]
                    pt, dr = SC.boot(F, cells, M, TG, by_dom, len(procs), np.random.default_rng(rng.integers(1 << 31)))
                    res[(bench, sch, level, nb)] = (M, TG, pt, dr); print(bench, sch, level, nb, flush=True)
pickle.dump(res, open(HERE / "score.pkl", "wb"))
ci = lambda d: (np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5))
def get(key, m, tg):
    M, TG, pt, dr = res[key]; return pt[M.index(m), TG.index(tg)], dr[:, M.index(m), TG.index(tg)]
def pooled(level, m, tg, nb, benches):
    keys = [(b, s, level, (nb and b == "sopbench")) for b, s in [("sopbench", "new_procedures"), ("sopbench", "new_domain"), ("tau2", "new_procedures"), ("tau2", "new_domain")] if b in benches]
    pts, drs = zip(*[get(k, m, tg if (k[0] == "sopbench") else "all_irt") for k in keys]); return np.mean(pts), np.mean(drs, 0), list(pts)
lines = []
def line(label, level, benches, m, tg, nb):
    p, d, per = pooled(level, m, tg, nb, benches); lo, hi = ci(d)
    s = f"{label:10s} {m:9s} {p:+.3f} [{lo:+.3f},{hi:+.3f}] scen " + " ".join(f"{x:+.3f}" for x in per)
    for base in ["adele", "c_lad"]:
        if m != base:
            pb, db, _ = pooled(level, base, tg, nb, benches); l2, h2 = ci(d - db); s += f" | vs {base} {p-pb:+.3f} [{l2:+.3f},{h2:+.3f}]"
    lines.append(s)
models = ["adele", "c_lad"] + V
for nb in [False, True]:
    lines.append(f"=== exclude broken bank procedures (SOPBench part): {nb}")
    lines.append("-- pooled4 case level, public IRT (tau2 = c_lad)")
    for m in models: line("pool4", "case", ["sopbench", "tau2"], m, "all_irt", nb)
    lines.append("-- SOPBench only (2 scenarios), case level, public IRT")
    for m in models: line("sop2", "case", ["sopbench"], m, "all_irt", nb)
    lines.append("-- SOPBench only, case level, smart-agent real target")
    for m in models: line("sop2smart", "case", ["sopbench"], m, "smart_fail", nb)
    lines.append("-- SOPBench procedure level")
    for tg in ["all_irt", "smart_fail"]:
        for m in models: line("proc " + tg[:5], "proc", ["sopbench"], m, tg, nb)
    lines.append("-- per scenario SOPBench case level, public IRT: point, paired CI vs c_lad")
    for sch in ["new_procedures", "new_domain"]:
        k = ("sopbench", sch, "case", nb); cp, cd = get(k, "c_lad", "all_irt")
        for m in models:
            p, d = get(k, m, "all_irt"); s = f"{sch:15s} {m:9s} {p:+.3f}"
            if m != "c_lad": s += f" vs c_lad {p-cp:+.3f} [{ci(d-cd)[0]:+.3f},{ci(d-cd)[1]:+.3f}]"
            lines.append(s)
open(HERE / "score.txt", "w").write("\n".join(lines)); print("\n".join(lines))
