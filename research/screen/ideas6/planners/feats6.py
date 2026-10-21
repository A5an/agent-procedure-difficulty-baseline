"""Features from two planners (Gemini 3.8 flash, Sonnet 5.5). Tables: son_*, int_*, dis_* (+ avg_*) for tau2 and sopbench."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, sys, re, numpy as np, pandas as pd
from collections import Counter
from pathlib import Path
I3 = Path(ZR.SCREEN + "/ideas3")
sys.path.insert(0, str(I3 / "plan_tau")); import feats as F
HERE = Path(__file__).parent; FEAT = HERE / "features"; FEAT.mkdir(exist_ok=True)
def plan_list(p): return [x for x in (p.get("plan") or []) if isinstance(x, dict)]
def outc(p): return str(p.get("outcome", "complete"))
def refuse(p): return outc(p) in ("refuse", "transfer_to_human")
def toolmap(p):
    """Counter of tool names and set of tool names marked write"""
    pl = plan_list(p); c = Counter(str(x.get("tool")) for x in pl)
    w = {str(x.get("tool")) for x in pl if x.get("writes_db")}
    return c, w
def pair(g, s):
    cg, wg = toolmap(g); cs, ws = toolmap(s)
    inter = cg & cs; n_int = sum(inter.values()); w_all = wg | ws
    int_w = sum(v for t, v in inter.items() if t in wg and t in ws)  # write only if both planners mark the tool as a write
    ng, ns = sum(cg.values()), sum(cs.values())
    dis_calls = ng + ns - 2 * n_int
    nw_g = sum(v for t, v in cg.items() if t in wg); nw_s = sum(v for t, v in cs.items() if t in ws)
    dis_w = nw_g + nw_s - 2 * int_w
    union = sum((cg | cs).values())
    set_w = (wg & ws); set_wu = (wg | ws)
    f_int = dict(plan_calls=n_int, plan_writes=int_w, plan_reads=n_int - int_w, plan_distinct_tools=len(set(inter)),
                 plan_rules=np.mean([len(g.get("policy_rules") or []), len(s.get("policy_rules") or [])]),
                 plan_facts=np.mean([len(g.get("facts_to_ask_customer") or []), len(s.get("facts_to_ask_customer") or [])]),
                 plan_refuse=np.mean([refuse(g), refuse(s)]), plan_transfer=np.mean([outc(g) == "transfer_to_human", outc(s) == "transfer_to_human"]),
                 plan_ambig=np.mean([float(g.get("ambiguity", 0) or 0), float(s.get("ambiguity", 0) or 0)]))
    f_dis = dict(dis_calls=dis_calls, dis_writes=dis_w, dis_reads=dis_calls - dis_w,
                 dis_jaccard_calls=n_int / union if union else 1.0,
                 dis_jaccard_writes=len(set_w) / len(set_wu) if set_wu else 1.0,
                 dis_outcome_agree=float(outc(g) == outc(s)), dis_refuse_agree=float(refuse(g) == refuse(s)),
                 dis_write_count_gap=abs(nw_g - nw_s), dis_len_gap=abs(ng - ns), dis_ambig_gap=abs(float(g.get("ambiguity", 0) or 0) - float(s.get("ambiguity", 0) or 0)))
    return f_int, f_dis
def load_plans():
    """returns {bench: {case: (gemini_t0_plan, gemini_mean_feats_or_None, sonnet_plan)}}"""
    out = {}
    p3 = json.load(open(I3 / "plan_tools/plans3.json")); st = json.load(open(HERE / "sonnet_tau2.json"))
    d = {}
    for c, dd in p3.items():
        g = [F.parse(dd[t]) for t in ("t0", "s1", "s2")]; s = F.parse(st.get(c))
        d[c] = (g, s)
    out["tau2"] = d
    p1 = json.load(open(I3 / "plan_tau/plans_t0.json")); ss = json.load(open(HERE / "sonnet_sop.json")) if (HERE / "sonnet_sop.json").exists() else {}; d = {}
    for k, v in p1.items():
        b, c = k.split("|", 1)
        if b == "sopbench": d[c] = ([F.parse(v)], F.parse(ss.get(c)))
    out["sopbench"] = d
    return out
def build():
    plans = load_plans(); rep = {}
    for b, d in plans.items():
        if b == "sopbench" and not any(v[1] for v in d.values() if v[1] is not None): continue
        son, avg, inter, dis = {}, {}, {}, {}
        base = pd.read_csv(I3 / f"plan_tools/features/plan3_{b}.csv").set_index("case_id")
        for c, (g, s) in d.items():
            g = [x for x in g if x]
            if not g or not s: continue
            fs = F.feats(s); son[c] = fs
            avg[c] = {k: (base.loc[c, k] + fs[k]) / 2 for k in fs}
            fi, fd = pair(g[0], s); inter[c] = fi; dis[c] = fd
        miss = len(d) - len(son)
        for name, dct in (("son", son), ("avg", avg), ("int", inter), ("dis", dis)):
            df = pd.DataFrame.from_dict(dct, orient="index").reindex(list(d)); df = df.fillna(df.median()); df.index.name = "case_id"
            df.reset_index().to_csv(FEAT / f"{name}_{b}.csv", index=False)
        rep[b] = dict(n=len(d), missing_sonnet_or_gemini=miss)
    print(rep)
if __name__ == "__main__": build()
