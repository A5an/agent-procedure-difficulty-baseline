import json, glob, os, sys, collections
H = os.path.dirname(os.path.abspath(__file__))
def collect(run, model):
    rows = []
    for f in sorted(glob.glob(f"{H}/runs/{run}/out/*/*/ast_{model}-*.json")):
        for t in json.load(open(f)):
            task = t["task"]; runs = t["interactions"]
            r = {"cid": task.get("_cid"), "domain": t["domain"], "proc": task["user_goal"], "perform": bool(task["action_should_succeed"]),
                 "variant": task.get("_variant"), "n_runs": len(runs)}
            if not runs or "evaluations" not in t or not t["evaluations"]:
                r["success"] = None; rows.append(r); continue
            ev = t["evaluations"][0]; inter = runs[0]["interaction"]
            asst = [m for m in inter if m.get("sender", "").endswith("assistant")]
            r["success"] = int(bool(ev["success"]))
            r["n_asst"] = len(asst)
            r["n_empty"] = sum(1 for m in asst if not m.get("tool_calls") and not (m.get("content") or "").strip())
            r["n_toolcalls"] = sum(len(m.get("tool_calls") or []) for m in asst)
            r["called_target"] = int(any(tc["function"]["name"] == task["user_goal"] for m in asst for tc in (m.get("tool_calls") or [])))
            r["target_ok"] = int(bool(ev.get("action_called_correctly")) ) if "action_called_correctly" in ev else None
            r["ev"] = {k: ev[k] for k in ("dirgraph_satisfied", "action_called_correctly", "constraint_not_violated", "database_match", "no_tool_call_error") if k in ev}
            rows.append(r)
    return rows
if __name__ == "__main__":
    run, model = sys.argv[1], sys.argv[2]
    rows = collect(run, model); json.dump(rows, open(f"{H}/runs/{run}/results.json", "w"))
    n = len(rows); ok = [r for r in rows if r["success"] is not None]
    print("cases", n, "scored", len(ok), "unscored", n - len(ok))
    for lab in (True, False):
        s = [r for r in ok if r["perform"] == lab]
        print("perform" if lab else "refuse", len(s), "success", round(sum(r["success"] for r in s) / max(1, len(s)), 3),
              "no tool call at all", round(sum(1 for r in s if r["n_toolcalls"] == 0) / max(1, len(s)), 3),
              "empty-only", round(sum(1 for r in s if r["n_empty"] == r["n_asst"]) / max(1, len(s)), 3),
              "called target", round(sum(r["called_target"] for r in s) / max(1, len(s)), 3))
