"""Build the predictor inputs for the 507 tasks of ThinkingBox-Bench v1.0 and the frozen run order.

Read, per task: the user's first message (`query`) and the simulated user's instructions (`user_context`).
Read, per domain: the agent's system instructions (`bot_instructions` of the scenario), the policy knowledge base the
agent can search (`world_state.<scenario>.sources`) and the tool names.
Never read: initial_state_patch_json, expected_tool_interactions_json, rubrics_json, and the scenario keys
data_patch, golden_test_case, rubrics, rubrics_yesno (gold or hidden state).

  python build_items.py   -> items.jsonl (per task), domains.json (per scenario), order.csv"""
import json
import math
import random

import pandas as pd
import yaml

from tbpaths import DOMAINS, HF, ITEMS, ORDER, SCEN, TESTLIST

SEED = 20261015


def domain_texts(scenario_id):
    y = yaml.safe_load(open(SCEN / f"{scenario_id}.yaml"))
    kb = "\n\n".join(f"--- {s['name']} ---\n" + "\n".join(s.get("snippets", []))
                     for s in y["world_state"][scenario_id]["sources"])
    policy = y["bot_instructions"].strip() + "\n\n=== POLICY KNOWLEDGE BASE (searchable by the agent) ===\n\n" + kb
    tools = ", ".join(t["name"] for t in y["tools"])
    return policy, tools


def request_text(r):
    return (f"First message from the user:\n{r.query.strip()}\n\n"
            f"What the simulated user knows and how it behaves (not shown to the agent):\n{r.user_context.strip()}")


def main():
    tasks = pd.read_parquet(HF / "tasks.parquet", columns=["task_ref", "domain", "scenario_id", "query", "user_context"])
    release = set(yaml.safe_load(open(TESTLIST)))
    assert set(tasks.task_ref) == release and len(tasks) == 507, "task set differs from the v1.0 test list"
    dom = {s: dict(zip(("policy", "tools"), domain_texts(s))) for s in sorted(tasks.scenario_id.unique())}
    DOMAINS.write_text(json.dumps(dom, indent=1))
    with open(ITEMS, "w") as f:
        for r in tasks.itertuples():
            req = request_text(r)
            f.write(json.dumps({"task_id": r.task_ref, "domain": r.domain, "scenario_id": r.scenario_id,
                                "request": req, "log_chars": math.log(len(req))}) + "\n")
    rows = []
    for d, g in tasks.groupby("domain"):
        ids = sorted(g.task_ref)
        random.Random(f"{SEED}:{d}").shuffle(ids)
        rows += [{"task_id": t, "domain": d, "rank": i} for i, t in enumerate(ids)]
    pd.DataFrame(rows).to_csv(ORDER, index=False)
    print(len(tasks), "tasks;", tasks.domain.value_counts().to_dict())


if __name__ == "__main__":
    main()
