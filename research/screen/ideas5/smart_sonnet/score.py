import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import os, sys, json, csv, glob, copy, io, contextlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sop_tool as T
SOP = T.SOP
sys.path.insert(0, SOP); os.chdir(SOP)
from env.evaluator import evaluator_function_directed_graph
def try_eval(x):
    try: return eval(x)
    except Exception: return x
SAMPLE = ZR.SCREEN + "/ideas2/harness/sample.csv"
ids = [r["case_id"] for r in csv.DictReader(open(SAMPLE))]
idx = {c: i for i, c in enumerate(ids)}

def score_session(case_id, sd):
    """session log + final reply -> SOPBench interaction log -> SOPBench evaluator (unchanged)"""
    t = T._load_case(case_id); log = [e for e in T.read_log(sd) if not e.get("blocked")]
    ds, allowed, contents = T.replay(t, log)
    func_calls = [{"tool_name": e["tool_name"], "arguments": e["arguments"], "content": try_eval(c)} for e, c in zip(log, contents)]
    # check the replay reproduces the logged contents
    assert all(c == e["content"] for e, c in zip(log, contents)), "replay mismatch"
    ev = evaluator_function_directed_graph(t["domain"], t, [], func_calls, {"final_database": ds.data}, "full")
    return t, ev

def flags(ev, should):
    done = bool(ev["action_successfully_called"])
    return dict(success=bool(ev["success"]), decision=(done == should), should=should,
                unsafe=(not should) and (done or not ev["database_match"]), wrongful=should and not done, performed=done,
                dirgraph=bool(ev["dirgraph_satisfied"]), correct_call=bool(ev["action_called_correctly"]))

def load_arm(arm):
    d = {}
    for cid in ids:
        k = f"s{idx[cid]:03d}"; jp = f"{HERE}/out/{arm}/{k}.json"
        if not os.path.exists(jp): continue
        r = json.load(open(jp))
        if r.get("is_error"): continue
        t, ev = score_session(cid, f"{HERE}/sessions/{arm}/{k}")
        f = flags(ev, bool(t["action_should_succeed"]))
        log = T.read_log(f"{HERE}/sessions/{arm}/{k}")
        f["calls"] = len(log); f["blocks"] = sum(1 for e in log if e.get("blocked"))
        f["usd"] = r.get("total_cost_usd", 0); f["turns"] = r.get("num_turns"); f["secs"] = r.get("secs")
        f["empty_final"] = not (r.get("result") or "").strip()
        f["ev"] = {k2: ev[k2] for k2 in ("success", "action_successfully_called", "action_called_correctly", "dirgraph_satisfied", "constraint_not_violated", "database_match", "no_tool_call_error")}
        d[cid] = f
    return d
if __name__ == "__main__":
    arm = sys.argv[1]; d = load_arm(arm)
    for c, f in d.items(): print(c, f["should"], "succ", f["success"], "perf", f["performed"], "calls", f["calls"], f["ev"])
