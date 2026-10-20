"""Step 1: classify every (configuration, case) of SOPBench into an outcome type. Writes outcomes.csv."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, json, glob, re, copy
from pathlib import Path
import pandas as pd
sys.path.insert(0, ZR.SCREEN + "/ideas")
from cpulock import cpu_lock
SOP = Path(ZR.DATA + "/SOPBench-d2622008")
REPO = Path(ZR.REPO)
DOMAINS = ["bank", "dmv", "healthcare", "hotel", "library", "online_market", "university"]
sys.path.insert(0, str(REPO / "src")); 
os.chdir(SOP); sys.path.insert(0, str(SOP))
from env.evaluator import evaluator_function_directed_graph

def try_eval(x):
    try:
        return eval(x, {"__builtins__": {}}, {})
    except Exception:
        return x

def func_calls_of(interaction):
    func_calls = []
    for i in range(len(interaction) - 1):
        if interaction[i].get("tool_calls", []):
            for tool_call in interaction[i]["tool_calls"]:
                if tool_call["function"]["name"].lower() in ["n/a", "na", "none", "null"]:
                    interaction[i]["tool_calls"].remove(tool_call)
            if len(interaction[i]["tool_calls"]) > 0:
                func_calls.append({"tool_name": interaction[i + 1]["tool_name"],
                    "arguments": try_eval(interaction[i]["tool_calls"][0]["function"]["arguments"]),
                    "content": try_eval(interaction[i + 1]["content"])})
    return func_calls
os.chdir(SOP)
resp = {}
for l in open(REPO / "data/sopbench/responses.jsonl"):
    j = json.loads(l); resp[j["subject_id"]] = j["responses"]
configs = list(resp)
rows = []
with cpu_lock("risk_extract"):
    for d in DOMAINS:
        data = json.loads((SOP / "data" / f"{d}_tasks.json").read_text())
        tasks = [(f"{d}/{g}/{j}", g, t) for g, lst in data.items() for j, t in enumerate(lst)]
        for c in configs:
            f = SOP / "output" / d / f"ast_{c}-dep_full-fmt_structured-tool_full-shuffle_False.json"
            recs = json.loads(f.read_text())
            for (cid, g, task), rec in zip(tasks, recs):
                if not rec["interactions"]:
                    continue
                log = copy.deepcopy(rec["interactions"][0])
                il = log["interaction"]
                fc = func_calls_of(il)
                res = evaluator_function_directed_graph(domain_str=rec["domain"], task=rec["task"], log_msg_fcall=il,
                    func_calls=fc, results={"final_database": log["database"]}, default_constraint_option="full")
                texts = [str(m.get("content") or "").strip() for m in il if m.get("sender", "").endswith("assistant") and not m.get("tool_calls")]
                nonempty = any(t not in ("", "None") for t in texts)
                last_t = texts[-1] if texts else ""
                frac_empty = (sum(t in ("", "None") for t in texts) / len(texts)) if texts else 1.0
                target_calls = sum(1 for x in fc if x["tool_name"] == g)
                rows.append(dict(config=c, case_id=cid, domain=d, procedure=g, should=int(res["action_should_succeed"]),
                    success=int(res["success"]), saved=int(bool(rec.get("evaluations"))), n_calls=len(fc), target_calls=target_calls,
                    nonempty_text=int(nonempty), last_empty=int(last_t in ('', 'None')), frac_empty=frac_empty, n_texts=len(texts), asc=int(res["action_successfully_called"]), acc=int(res["action_called_correctly"]),
                    dbm=int(res["database_match"]), cnv=int(res["constraint_not_violated"]), dirg=int(res["dirgraph_satisfied"]),
                    ntce=int(res["no_tool_call_error"]), resp_ok=resp[c].get(cid, -1)))
df = pd.DataFrame(rows)
df.to_csv(ZR.SCREEN + "/ideas2/risk/flags.csv", index=False)
print(len(df), (df.success == df.resp_ok).mean())
