"""Turn the harness outputs of a stage into one row per run slot (task_id, agent, trial, success, valid).

Each (trial, task) slot takes the first valid run among the agent's output files, oldest file first (resume.merged);
a run is valid when the harness finished it without a system error and the test returned a result (tb agg counts the
other cases as errors). Slots without a valid run are written with valid = 0.
  python collect.py main   -> ../outcomes.csv      python collect.py pilot   -> runs/pilot/outcomes.csv"""
import sys
from pathlib import Path

import pandas as pd

import resume

HERE = Path(__file__).resolve().parent
AGENTS = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.8-flash"]


def rows(stage):
    for a in AGENTS:
        if not resume.files(stage, a):
            continue
        slots, _ = resume.merged(stage, a)
        for (rep, uid), r in slots:
            tr = (r or {}).get("test_result") or {}
            yield {"task_id": uid, "agent": a, "trial": rep, "success": int(bool(tr.get("result"))) if r else 0,
                   "valid": int(r is not None), "finish_reason": (r or {}).get("finish_reason")}


if __name__ == "__main__":
    stage = sys.argv[1]
    df = pd.DataFrame(list(rows(stage)))
    path = HERE.parent / "outcomes.csv" if stage == "main" else HERE / "runs" / stage / "outcomes.csv"
    df.to_csv(path, index=False)
    print(len(df), "runs,", int(df.valid.sum()), "valid ->", path)
