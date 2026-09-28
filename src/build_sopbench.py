"""Build the SOPBench inputs for the Agent psychometrics baseline.

Input: the SOPBench release (github.com/Leezekun/SOPBench at commit d2622008) in
external_data/SOPBench, fetched by src/download_data.py. Writes to data/sopbench/:
  responses.jsonl   one line per configuration, {"subject_id", "responses": {case_id: 0/1}}
                    (the response-matrix format of agent-psychometrics)
  tasks.csv         one row per case: ids, domain, procedure, allow/refuse (analysis only)
  statements.jsonl  the leak-free text each case is described by (what the agent sees)
  label_check.json  recomputed labels against the success flags saved in the release

Labels: every run is re-scored with SOPBench's own evaluator
(env.evaluator.evaluator_function_directed_graph), the same call run_evaluation.py
makes, because 1,944 of the 23,240 runs in the release have no saved evaluation.

Leak-free statement: the policy section of the target action exactly as rendered in
the agent's system prompt, the customer's first message and the list of tool names.
Fields the agent never sees are not used: constraints (evaluator tree),
directed_action_graph, action_should_succeed, initial_database, user_instruction.
"""

import copy
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from common import DATA, EXTERNAL  # noqa: E402

SOP = EXTERNAL / "SOPBench"
OUT = DATA / "sopbench"
DOMAINS = ["bank", "dmv", "healthcare", "hotel", "library", "online_market", "university"]
PROMPT_SOURCE = "gpt-4o-mode_fc"  # system prompt text is taken from this configuration

sys.path.insert(0, str(SOP))
os.chdir(SOP)
from env.evaluator import evaluator_function_directed_graph  # noqa: E402


def try_eval(x):
    # run_evaluation.try_eval, with builtins removed: same result on literals,
    # falls back to the raw string on anything else, as the original does
    try:
        return eval(x, {"__builtins__": {}}, {})
    except Exception:
        return x


def func_calls_of(interaction):
    # copy of the loop in run_evaluation.py (lines 200-215)
    func_calls = []
    for i in range(len(interaction) - 1):
        if interaction[i].get("tool_calls", []):
            for tool_call in interaction[i]["tool_calls"]:
                if tool_call["function"]["name"].lower() in ["n/a", "na", "none", "null"]:
                    interaction[i]["tool_calls"].remove(tool_call)
            if len(interaction[i]["tool_calls"]) > 0:
                func_calls.append({
                    "tool_name": interaction[i + 1]["tool_name"],
                    "arguments": try_eval(interaction[i]["tool_calls"][0]["function"]["arguments"]),
                    "content": try_eval(interaction[i + 1]["content"]),
                })
    return func_calls


def config_name(path):
    return re.sub(r"^ast_|-dep_full.*$", "", os.path.basename(path))


def strip(task):
    # the release adds user_goal (the group key) and sometimes assistant_prompt at run time
    return {k: v for k, v in task.items() if k not in ("assistant_prompt", "user_goal")}


def action_section(prompt, action):
    m = re.search(r"^\* " + re.escape(action) + r":\s*$", prompt, re.M)
    if not m:
        raise ValueError(f"no policy section for {action}")
    rest = prompt[m.end():]
    n = re.search(r"^(\* \w+|###)", rest, re.M)
    return (rest[: n.start()] if n else rest).strip()


def role_description(prompt):
    m = re.search(r"### Role Description:\s*(.*?)\n\s*\n", prompt, re.S)
    if not m:
        raise ValueError("no role description")
    return m.group(1).strip()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = {}  # domain -> list of (case_id, goal, task)
    for d in DOMAINS:
        data = json.loads((SOP / "data" / f"{d}_tasks.json").read_text())
        tasks[d] = [(f"{d}/{g}/{j}", g, t) for g, lst in data.items() for j, t in enumerate(lst)]

    # configurations with a full run on every domain
    files = defaultdict(dict)
    for d in DOMAINS:
        for f in glob.glob(str(SOP / "output" / d / "ast_*.json")):
            files[config_name(f)][d] = f
    configs = []
    for c, per_dom in sorted(files.items()):
        if all(d in per_dom for d in DOMAINS) and all(
            len(json.loads(Path(per_dom[d]).read_text())) == len(tasks[d]) for d in DOMAINS
        ):
            configs.append(c)
    assert len(configs) == 28, len(configs)

    responses = {c: {} for c in configs}
    check = Counter()
    per_config_missing = Counter()
    prompts = {}
    for d in DOMAINS:
        for c in configs:
            recs = json.loads(Path(files[c][d]).read_text())
            for (case_id, goal, task), rec in zip(tasks[d], recs):
                if strip(rec["task"]) != strip(task) or rec["task"].get("user_goal", goal) != goal:
                    raise ValueError(f"order mismatch {c} {case_id}")
                if not rec["interactions"]:
                    # run never happened (5 cells: deepseek-r1 react on online_market,
                    # gemini-1.5-pro fc on hotel); left out of the matrix, not scored as failure
                    check["no_run_left_missing"] += 1
                    continue
                log = copy.deepcopy(rec["interactions"][0])
                res = evaluator_function_directed_graph(
                    domain_str=rec["domain"],
                    task=rec["task"],
                    log_msg_fcall=log["interaction"],
                    func_calls=func_calls_of(log["interaction"]),
                    results={"final_database": log["database"]},
                    default_constraint_option="full",
                )
                y = int(bool(res["success"]))
                responses[c][case_id] = y
                saved = rec.get("evaluations") or []
                if saved and "success" in saved[0]:
                    check["saved"] += 1
                    check["agree" if int(bool(saved[0]["success"])) == y else "disagree"] += 1
                else:
                    check["recomputed_only"] += 1
                    per_config_missing[c] += 1
                if c == PROMPT_SOURCE:
                    prompts[case_id] = rec["interactions"][0]["prompt"]

    with open(OUT / "responses.jsonl", "w") as f:
        for c in configs:
            f.write(json.dumps({"subject_id": c, "responses": responses[c]}) + "\n")

    with open(OUT / "tasks.csv", "w", newline="") as f, open(OUT / "statements.jsonl", "w") as g:
        w = csv.writer(f)
        w.writerow(["case_id", "domain", "procedure", "procedure_id", "should_succeed",
                    "success_rate_28", "statement_chars"])
        for d in DOMAINS:
            for case_id, goal, task in tasks[d]:
                prompt = prompts[case_id]
                tools = re.findall(r"^\* (\w+)", prompt, re.M)
                text = (
                    f"Domain: {d.replace('_', ' ')}. {role_description(prompt)}\n"
                    f"Requested action: {goal}\n"
                    f"Policy for this action:\n{action_section(prompt, goal)}\n"
                    f"Customer message: {task['user_prompt']}\n"
                    f"Available tools: {', '.join(tools)}"
                )
                obs = [responses[c][case_id] for c in configs if case_id in responses[c]]
                rate = sum(obs) / len(obs)
                w.writerow([case_id, d, goal, f"{d}/{goal}", int(task["action_should_succeed"]),
                            f"{rate:.4f}", len(text)])
                g.write(json.dumps({"case_id": case_id, "text": text}) + "\n")

    total = sum(len(v) for v in responses.values())
    summary = {
        "configurations": configs,
        "cases": sum(len(v) for v in tasks.values()),
        "runs": total,
        "overall_success": sum(sum(v.values()) for v in responses.values()) / total,
        "saved_flags": check["saved"],
        "agree_with_saved": check["agree"],
        "disagree_with_saved": check["disagree"],
        "no_saved_flag_recomputed": check["recomputed_only"],
        "no_run_left_missing": check["no_run_left_missing"],
        "no_saved_flag_by_config": dict(per_config_missing),
    }
    (OUT / "label_check.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "configurations"}, indent=2))


if __name__ == "__main__":
    main()
