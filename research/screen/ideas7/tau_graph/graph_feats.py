"""Steps 2-3: graph features per task from planned tools. Output features/graph_{gem,son,avg}_tau2.csv"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, re, sys, numpy as np, pandas as pd
from pathlib import Path
S = Path(ZR.SCREEN)
sys.path.insert(0, str(S / "ideas3/plan_tau")); import feats as F
T = json.load(open("tool_table.json")); TOOLS = T["tools"]; STAT = {k: set(v) for k, v in T["statable"].items()}
USER_TOOLS = ["check_status_bar", "check_network_status", "check_network_mode_preference", "set_network_mode_preference", "run_speed_test",
 "toggle_airplane_mode", "check_sim_status", "reseat_sim_card", "toggle_data", "toggle_roaming", "check_data_restriction_status",
 "toggle_data_saver_mode", "check_apn_settings", "set_apn_settings", "reset_apn_settings", "check_wifi_status", "toggle_wifi",
 "check_wifi_calling_status", "toggle_wifi_calling", "check_vpn_status", "connect_vpn", "disconnect_vpn", "check_installed_apps",
 "check_app_status", "check_app_permissions", "grant_app_permission", "can_send_mms", "reboot_device", "check_payment_request", "make_payment"]
def idtypes(dom):
    return {t for k, v in TOOLS.items() if v["domain"] == dom for t in list(v["requires_ids"].values()) + v["produces"]}
def solve(dom):
    """d[type] = min over producer tools of 1 + max d(required id types); chain[type] = set of lookup tools used. fixpoint."""
    types = idtypes(dom); d = {t: (0 if t in STAT[dom] else 99) for t in types}; ch = {t: set() for t in types}
    prod = {t: [k for k, v in TOOLS.items() if v["domain"] == dom and t in v["produces"]] for t in types}
    for _ in range(10):
        for t in types:
            if t in STAT[dom]: continue
            for k in prod[t]:
                req = set(TOOLS[k]["requires_ids"].values()) - {"any_id"}
                dk = 1 + max([d[r] for r in req] or [0])
                if dk < d[t]:
                    d[t] = dk; ch[t] = {k}.union(*[ch[r] for r in req])
    d["any_id"] = 1; ch["any_id"] = {"telecom/get_customer_by_phone"}
    return d, ch
SOL = {dom: solve(dom) for dom in STAT}
def call_info(key):
    v = TOOLS[key]; dom = v["domain"]; d, ch = SOL[dom]
    reqs = set(v["requires_ids"].values())
    depth = max([d[r] for r in reqs] or [0]); depth = min(depth, 3)
    chain = set().union(*[ch[r] for r in reqs]) if reqs else set()
    return depth, chain
def plan_feats(p, dom):
    plan = [x for x in (p.get("plan") or []) if isinstance(x, dict)]
    rows = []
    for x in plan:
        k = f"{dom}/{x.get('tool')}"
        if k not in TOOLS: continue
        v = TOOLS[k]; dep, chain = call_info(k)
        wr = (v["type"] == "write") or (v["type"] == "generic" and bool(x.get("writes_db")))
        rows.append((k, v, dep, chain, wr))
    W = [r for r in rows if r[4]]
    def agg(vals): return (float(sum(vals)), float(max(vals)) if vals else 0.0)
    wd = agg([r[2] for r in W]); lo = agg([r[1]["n_list_obj"] for r in W]); ef = agg([r[1]["n_enum_fmt"] for r in W]); cw = agg([r[1]["cross_words"] for r in W]); un = agg([r[1]["n_unstatable"] for r in W])
    union = set().union(*[r[3] for r in W]) if W else set()
    types_unst = {t for r in W for t in set(r[1]["requires_ids"].values()) if t not in STAT[dom]}
    dw_calls = float(sum(1 + r[2] for r in rows)); dw_writes = float(sum(1 + r[2] for r in W))
    user_steps = 0.0
    if dom == "telecom":
        txt = " ".join(str(x.get("purpose", "")) for x in plan) + " " + " ".join(map(str, p.get("policy_rules") or []))
        n_ask = sum(1 for x in plan if x.get("tool") == "ask_customer_to_act")
        ment = {u for u in USER_TOOLS if u in txt}
        user_steps = float(n_ask + len(ment))
        dw_calls += user_steps  # telecom: user-side steps replace the ID depth
    return dict(g_wdepth_sum=wd[0], g_wdepth_max=wd[1], g_lookups=float(len(union)), g_unstat_ids=float(len(types_unst)), g_unstat_sum=un[0], g_unstat_max=un[1],
                g_listobj_sum=lo[0], g_listobj_max=lo[1], g_nest_max=float(max([r[1]["nest_depth"] for r in W] or [0])),
                g_enum_sum=ef[0], g_enum_max=ef[1], g_cross_sum=cw[0], g_cross_max=cw[1], g_dw_calls=dw_calls, g_dw_writes=dw_writes, g_user_steps=user_steps)
def build():
    p3 = json.load(open(S / "ideas3/plan_tools/plans3.json")); sn = json.load(open(S / "ideas6/planners/sonnet_tau2.json"))
    gem, son = {}, {}
    for c, dd in p3.items():
        dom = c.split("/")[0]
        fs = [plan_feats(q, dom) for q in (F.parse(dd[t]) for t in ("t0", "s1", "s2")) if q]
        if fs: gem[c] = {k: np.mean([f[k] for f in fs]) for k in fs[0]}
        q = F.parse(sn.get(c))
        if q: son[c] = plan_feats(q, dom)
    idx = list(p3)
    G = pd.DataFrame.from_dict(gem, orient="index").reindex(idx); So = pd.DataFrame.from_dict(son, orient="index").reindex(idx)
    print("missing gem", G.isna().any(axis=1).sum(), "son", So.isna().any(axis=1).sum())
    So = So.fillna(G); G = G.fillna(So)
    A = (G + So) / 2
    for n, D in (("gem", G), ("son", So), ("avg", A)):
        D = D.copy(); D.index.name = "case_id"; D.reset_index().to_csv(f"features/graph_{n}_tau2.csv", index=False)
    print(A.groupby([i.split("/")[0] for i in A.index]).mean().T.round(2))
if __name__ == "__main__":
    print({dom: SOL[dom][0] for dom in SOL}); build()
