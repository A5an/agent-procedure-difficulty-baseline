import json, glob, collections, sys
import score as S, sop_tool as T
SOPORIG = S.SOP + "/output/"
mis = n = 0
for d in ["bank", "hotel", "dmv", "library"]:
    recs = json.load(open(glob.glob(f"{SOPORIG}{d}/ast_o4-mini-high-mode_fc-*.json")[0]))
    for r in recs[:40]:
        t = r["task"]; ev0 = r["evaluations"][0]; inter = r["interactions"][0]["interaction"]
        calls = []
        for i, m in enumerate(inter[:-1]):
            if m.get("tool_calls") and "tool_name" in inter[i + 1]:
                tc = m["tool_calls"][0]["function"]
                if tc["name"] == "exit_conversation": continue
                try: a = json.loads(tc["arguments"])
                except Exception: a = {}
                calls.append({"tool_name": inter[i+1]["tool_name"], "arguments": a})
        # tools may have been called in parallel in one message: use all
        calls = []
        for i, m in enumerate(inter):
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    if tc["function"]["name"] in ("exit_conversation", "n/a", "none"): continue
                    try: a = json.loads(tc["function"]["arguments"])
                    except Exception: a = {}
                    calls.append({"tool_name": tc["function"]["name"], "arguments": a})
        tt = S.T._load_case(f"{d}/{t['user_goal']}/0"); tt = dict(t); tt["domain"] = d
        ds, allowed, cont = T.replay(tt, calls)
        fc = [{**c, "content": S.try_eval(x)} for c, x in zip(calls, cont)]
        ev = S.evaluator_function_directed_graph(d, tt, [], fc, {"final_database": ds.data}, "full")
        n += 1; mis += ev["success"] != ev0["success"]
        if ev["success"] != ev0["success"] and mis < 6: print("mismatch", d, t["user_goal"], ev0["success"], ev["success"], len(calls))
print("checked", n, "mismatch", mis)
