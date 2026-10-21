"""Step 1: per-check outcome table from SOPBench raw runs. Re-scores every run with SOPBench's own evaluator
(same call as the repo's build_sopbench.py) so that the 1,944 runs without saved evaluation are covered too."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import copy, glob, json, os, re, sys
from pathlib import Path
import pandas as pd
from multiprocessing import Pool
SOP = ZR.DATA + "/SOPBench-d2622008"
REPO = Path(ZR.REPO)
OUT = Path(__file__).parent
DOMAINS = ["bank", "dmv", "healthcare", "hotel", "library", "online_market", "university"]
sys.path.insert(0, SOP); os.chdir(SOP)
from env.evaluator import evaluator_function_directed_graph  # noqa

def try_eval(x):
    try: return eval(x, {"__builtins__": {}}, {})
    except Exception: return x
def func_calls_of(interaction):
    fc = []
    for i in range(len(interaction) - 1):
        if interaction[i].get("tool_calls", []):
            for tc in interaction[i]["tool_calls"]:
                if tc["function"]["name"].lower() in ["n/a", "na", "none", "null"]:
                    interaction[i]["tool_calls"].remove(tc)
            if len(interaction[i]["tool_calls"]) > 0:
                fc.append({"tool_name": interaction[i + 1]["tool_name"],
                           "arguments": try_eval(interaction[i]["tool_calls"][0]["function"]["arguments"]),
                           "content": try_eval(interaction[i + 1]["content"])})
    return fc
def cfg(path): return re.sub(r"^ast_|-dep_full.*$", "", os.path.basename(path))

def work(args):
    c, d, f = args
    tasks = json.load(open(f"{SOP}/data/{d}_tasks.json"))
    cases = [(f"{d}/{g}/{j}") for g, lst in tasks.items() for j, t in enumerate(lst)]
    recs = json.load(open(f))
    rows = []
    for cid, rec in zip(cases, recs):
        row = dict(agent=c, case_id=cid, has_run=bool(rec["interactions"]))
        saved = rec.get("evaluations") or []
        row["has_saved"] = bool(saved and "success" in saved[0])
        if saved and "success" in saved[0]:
            for k in ["dirgraph_satisfied", "action_called_correctly", "constraint_not_violated", "success"]:
                row["saved_" + k] = saved[0].get(k)
        if rec["interactions"]:
            log = copy.deepcopy(rec["interactions"][0])
            res = evaluator_function_directed_graph(domain_str=rec["domain"], task=rec["task"], log_msg_fcall=log["interaction"],
                    func_calls=func_calls_of(log["interaction"]), results={"final_database": log["database"]}, default_constraint_option="full")
            for k in ["dirgraph_satisfied", "action_called_correctly", "constraint_not_violated", "database_match", "success",
                      "no_tool_call_error", "action_successfully_called", "num_function_calls"]:
                row[k] = res.get(k)
        rows.append(row)
    return rows

if __name__ == "__main__":
    lab = json.load(open(REPO / "data/sopbench/label_check.json"))["configurations"]
    jobs = []
    for d in DOMAINS:
        for f in glob.glob(f"{SOP}/output/{d}/ast_*.json"):
            if cfg(f) in lab: jobs.append((cfg(f), d, f))
    print(len(jobs), "files")
    with Pool(8) as p: res = p.map(work, jobs)
    df = pd.DataFrame([r for x in res for r in x])
    df.to_csv(OUT / "flags.csv.gz", index=False)
    print(df.shape)
