"""Step 2: per (agent, case, step) outcomes from the SOPBench trajectories, plus per (agent, case) action outcomes.
Outputs: step_outcomes.csv.gz (agent, case_id, step_id, verified, ordered, blocking, acted, step_correct)
         case_outcomes.csv.gz (agent, case_id, success, flags, action outcomes, failure category)
"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, glob, re, os, hashlib, math
import pandas as pd, sys, copy
ROOT = ZR.DATA + "/SOPBench-d2622008"
sys.path.insert(0, ROOT)
from env.evaluator import evaluator_function_directed_graph
REPO = ZR.REPO + "/data/sopbench"
steps = pd.read_csv("steps.csv", keep_default_na=False, na_values=[""])
steps["tools"] = steps["tools"].fillna("")
steps["tree"] = steps["tree"].fillna("")
tasks = pd.read_csv(f"{REPO}/tasks.csv")
resp = {json.loads(l)["subject_id"]: json.loads(l)["responses"] for l in open(f"{REPO}/responses.jsonl")}
AGENTS = list(resp)


def norm(v):
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().lower()
        try:
            return float(s)
        except ValueError:
            return s
    return json.dumps(v, sort_keys=True).lower()


def eq(a, b):
    a, b = norm(a), norm(b)
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) < 1e-6
    return a == b


# case lookup: (domain, goal, user_prompt, json user_known) -> case_id ; and the case data
CASES, KEY = {}, {}
for dom in sorted(tasks.domain.unique()):
    d = json.load(open(f"{ROOT}/data/{dom}_tasks.json"))
    for proc, cs in d.items():
        for i, c in enumerate(cs):
            cid = f"{dom}/{proc}/{i}"
            CASES[cid] = c
            k = (dom, proc, c["user_prompt"], json.dumps(c["user_known"], sort_keys=True), json.dumps(c["constraints_original"]))
            KEY.setdefault(k, []).append(cid)
amb = sum(len(v) > 1 for v in KEY.values())
print("ambiguous keys", amb, "of", len(KEY))

S = {cid: g.reset_index(drop=True) for cid, g in steps.groupby("case_id")}


def try_eval(x):
    try:
        return eval(x)
    except Exception:
        return x


def reevaluate(rec):
    """Same procedure as run_evaluation.py, for records that were stored without evaluations."""
    il = copy.deepcopy(rec["interactions"][0])
    interaction = il["interaction"]
    func_calls = []
    for i in range(len(interaction) - 1):
        if interaction[i].get("tool_calls", []):
            for tc in list(interaction[i]["tool_calls"]):
                if tc["function"]["name"].lower() in ["n/a", "na", "none", "null"]:
                    interaction[i]["tool_calls"].remove(tc)
            if len(interaction[i]["tool_calls"]) > 0:
                func_calls.append({"tool_name": interaction[i + 1]["tool_name"],
                                   "arguments": try_eval(interaction[i]["tool_calls"][0]["function"]["arguments"]),
                                   "content": try_eval(interaction[i + 1]["content"])})
    return evaluator_function_directed_graph(domain_str=rec["domain"], task=rec["task"], log_msg_fcall=interaction,
                                             func_calls=func_calls, results={"final_database": il["database"]},
                                             default_constraint_option="full")


def parse_calls(rec):
    calls, results = [], {}
    inter = rec["interactions"][0]["interaction"]
    for m in inter:
        if "tool_call_id" in m:
            results[m["tool_call_id"]] = m.get("content")
    pos = 0
    for m in inter:
        for tc in (m.get("tool_calls") or []):
            f = tc.get("function", {})
            raw = f.get("arguments")
            try:
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
                if not isinstance(args, dict):
                    args = None
            except Exception:
                args = None
            calls.append(dict(name=f.get("name"), args=args, res=results.get(tc.get("id")), pos=pos))
            pos += 1
    return calls


def res_true(r):
    if r is None:
        return False
    s = str(r).strip()
    return s.startswith("True") or s.startswith("(True")


import inspect
from env.variables import domain_keys
_SIG = {}


def sig_params(dom, name):
    k = (dom, name)
    if k not in _SIG:
        try:
            _SIG[k] = set(inspect.signature(getattr(domain_keys[dom], name)).parameters) - {"self", "args", "kwargs"}
        except Exception:
            _SIG[k] = None
    return _SIG[k]


def match_call(call, tools, derivable, dom):
    if call["name"] not in tools:
        return False
    if call["args"] is None:
        return False
    sp = sig_params(dom, call["name"])
    keys = [k for k in derivable if (sp is None or k in sp)]
    return all(k in call["args"] and eq(call["args"][k], derivable[k]) for k in keys)


def eval_tree(tr, calls, dom, der, first_act):
    if tr[0] == "single":
        ms = [x for x in calls if match_call(x, {tr[1]}, der, dom)]
        return len(ms) > 0, (len(ms) > 0 and (first_act is None or any(x["pos"] < first_act for x in ms)))
    rs = [eval_tree(c, calls, dom, der, first_act) for c in tr[1]]
    f = all if tr[0] == "and" else any
    return f(r[0] for r in rs), f(r[1] for r in rs)


def graph_sat(g, calls, dom, uk, first_act):
    """Evaluate the directed action graph below the goal node: every called tool must come before the goal action.
    Returns (satisfied, list of missing tool names)."""
    nodes, conns = g["nodes"], g["connections"]
    ch = {}
    for a, b in conns:
        ch.setdefault(a, []).append(b)
    def ev(i):
        n = nodes[i]
        kids = ch.get(i, [])
        if isinstance(n, str):
            rs = [ev(k) for k in kids]
            if not rs:
                return True, []
            if n == "and":
                return all(r[0] for r in rs), [m for r in rs for m in r[1]]
            ok = any(r[0] for r in rs)
            return ok, ([] if ok else [m for r in rs for m in r[1]])
        der = {k: uk[v] for k, v in n[1].items() if isinstance(v, str) and v in uk}
        ms = [x for x in calls if match_call(x, {n[0]}, der, dom) and (first_act is None or x["pos"] < first_act)]
        miss = [] if ms else [n[0]]
        rs = [ev(k) for k in kids]
        ok = bool(ms) and all(r[0] for r in rs)
        return ok, miss + [m for r in rs if not r[0] for m in r[1]]
    rs = [ev(k) for k in ch.get(0, [])]
    return all(r[0] for r in rs), [m for r in rs for m in r[1]]


def derivable_for(step, c):
    p = json.loads(step["params"]) if step["params"] else {}
    uk = c["user_known"]
    if step["kind"] == "link":
        gn = [n for n in c["directed_action_graph"]["nodes"] if not isinstance(n, str) and n[0] == step["tools"].split("|")[0]]
        p = gn[0][1] if gn else p
    return {k: uk[v] for k, v in p.items() if isinstance(v, str) and v in uk}


step_rows, case_rows, unmatched = [], [], 0
for agent in AGENTS:
    base = agent.replace("-mode_", "|mode_").split("|")
    for dom in sorted(tasks.domain.unique()):
        fn = f"{ROOT}/output/{dom}/ast_{agent}-dep_full-fmt_structured-tool_full-shuffle_False.json"
        data = json.load(open(fn))
        for rec in data:
            t = rec["task"]
            k = (dom, t["user_goal"], t["user_prompt"], json.dumps(t["user_known"], sort_keys=True), json.dumps(t["constraints_original"]))
            cids = KEY.get(k)
            if not cids:
                unmatched += 1
                continue
            cid = cids[0] if len(cids) == 1 else None
            if cid is None:  # identical prompts: disambiguate by the database
                cid = next((x for x in cids if json.dumps(CASES[x]["initial_database"], sort_keys=True) == json.dumps(t["initial_database"], sort_keys=True)), cids[0])
            c = CASES[cid]
            ev = rec["evaluations"][0] if rec.get("evaluations") else None
            if ev is None and rec["interactions"]:
                ev = reevaluate(rec)
            if ev is None:  # run produced no interaction at all
                ev = dict(success=resp[agent].get(cid_ := None) if False else 0, dirgraph_satisfied=True, constraint_not_violated=True,
                          database_match=True, no_tool_call_error=True)
            noint = int(not rec["interactions"])
            calls = [] if noint else parse_calls(rec)
            goal = t["user_goal"]
            gcalls = [x for x in calls if x["name"] == goal]
            first_act = gcalls[0]["pos"] if gcalls else None
            act_called = len(gcalls) > 0
            act_success = any(res_true(x["res"]) for x in gcalls)
            gp = c["directed_action_graph"]["nodes"][0][1]
            gder = {k2: c["user_known"][v] for k2, v in gp.items() if isinstance(v, str) and v in c["user_known"]}
            args_ok = any(x["args"] is not None and all((k2 in x["args"] and eq(x["args"][k2], v2)) or (v2 is None) for k2, v2 in gder.items()) for x in gcalls)
            gs = S[cid]
            should = int(c["action_should_succeed"])
            tree_ok = int(gs.tree_ok.iloc[0])
            # per step
            vflag = {}
            for _, st in gs.iterrows():
                if st.step_type in ("action", "or_group"):
                    continue
                if (not st.tree) or st.tree == "null":
                    vflag[st.step_id] = (None, None)  # reasoning-only step
                    continue
                der = derivable_for(st, c)
                ver, ordv = eval_tree(json.loads(st.tree), calls, dom, der, first_act)
                vflag[st.step_id] = (ver, ordv)
            for _, st in gs.iterrows():
                if st.step_type == "action":
                    continue
                if st.step_type == "or_group":
                    kids = gs[gs.group == st.step_id]
                    kids = kids[kids.step_type != "or_group"]
                    vv = [(vflag[r.step_id], r.truth) for _, r in kids.iterrows()]
                    # True group: one satisfied child checked. False group: every child checked.
                    if st.truth == 1:
                        ok = [v for v, tr in vv if tr == 1 and v[1] is not None]
                        ver = any(v[0] for v in ok) if ok else None
                        ordv = any(v[1] for v in ok) if ok else None
                    else:
                        known = [v for v, tr in vv if v[1] is not None]
                        ver = all(v[0] for v in known) if known else None
                        ordv = all(v[1] for v in known) if known else None
                    # reasoning-only children: group is judged by the decision only
                else:
                    if st.in_or:
                        continue
                    ver, ordv = vflag[st.step_id]
                blocking = int(tree_ok == 1 and st.truth == 0 and not st.in_or and not (should == 1))
                acted = int(act_success)
                decision_bad = int(blocking and acted)
                if ordv is None:
                    sc = int(not decision_bad)
                else:
                    sc = int(bool(ordv) and not decision_bad)
                step_rows.append(dict(agent=agent, case_id=cid, step_id=st.step_id, step_type=st.step_type, name=st["name"], kind=st.kind,
                                      truth=st.truth, verified=ver, ordered=ordv, blocking=blocking, acted=acted, step_correct=sc))
            gok, gmiss = graph_sat(c["directed_action_graph"], calls, dom, c["user_known"], first_act)
            # action step
            if should == 1:
                act_ok = int(act_success and args_ok)
            else:
                act_ok = int(not act_success)
            ast = gs[gs.step_type == "action"].iloc[0]
            step_rows.append(dict(agent=agent, case_id=cid, step_id=ast.step_id, step_type="action", name=goal, kind="action", truth=should,
                                  verified=act_called, ordered=act_called, blocking=0, acted=int(act_success), step_correct=act_ok))
            case_rows.append(dict(agent=agent, case_id=cid, success=int(ev["success"]), should=should, tree_ok=tree_ok, act_called=int(act_called),
                                  act_success=int(act_success), args_ok=int(args_ok), dirgraph=int(ev["dirgraph_satisfied"]),
                                  constraint_ok=int(ev["constraint_not_violated"]), db_match=int(ev["database_match"]),
                                  tool_err=int(not ev["no_tool_call_error"]), n_calls=len(calls),
                                  graph_ok=int(gok), graph_missing='|'.join(dict.fromkeys(gmiss)), noint=noint, empty=int(len([x for x in calls if x["name"] != "exit_conversation"]) == 0)))
print("unmatched", unmatched)
so = pd.DataFrame(step_rows)
co = pd.DataFrame(case_rows)
# cross-check with the repository responses
co["resp"] = [resp[a].get(c) for a, c in zip(co.agent, co.case_id)]
print("noint", co.noint.sum(), "resp NaN", int(co.resp.isna().sum()), "mismatch rows", int((co.success != co.resp).sum()), "dup", int(co.duplicated(["agent","case_id"]).sum()))
co = co[co.resp.notna()].copy()
co["success_traj"] = co.success
co["success"] = co.resp.astype(int)
print("cases", len(co), "agents", co.agent.nunique(), "success==responses", (co.success == co.resp).mean())
so.to_csv("step_outcomes.csv.gz", index=False)
co.drop(columns=["resp"]).to_csv("case_outcomes.csv.gz", index=False)
