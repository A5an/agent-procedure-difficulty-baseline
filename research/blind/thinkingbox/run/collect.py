"""Turn the harness outputs of a stage into one row per run (task_id, agent, trial, success, valid, error).

A run is valid when the harness finished it without a system error and the test returned a result (tb agg counts the
other cases as errors). success = the test's own pass/fail. Reads the last retry file of every agent.
  python collect.py main   -> ../outcomes.csv      python collect.py pilot   -> runs/pilot/outcomes.csv"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
AGENTS = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.8-flash"]


def last_file(d, a):
    for name in (f"{a}.r2.jsonl", f"{a}.r1.jsonl", f"{a}.jsonl"):
        if (d / name).exists():
            return d / name
    return None


def rows(stage):
    d = HERE / "runs" / stage
    for a in AGENTS:
        f = last_file(d, a)
        if f is None:
            continue
        for l in f.read_text().splitlines():
            r = json.loads(l)
            tr = r.get("test_result") or {}
            valid = not r.get("is_system_error") and bool(tr) and not tr.get("is_system_error")
            err = "" if valid else str((r.get("metadata") or {}).get("error") or tr.get("line_content") or "")[:200]
            yield {"task_id": r["uid"], "agent": a, "trial": (r.get("metadata") or {}).get("repetition", 0),
                   "success": int(bool(tr.get("result"))) if valid else 0, "valid": int(valid),
                   "finish_reason": r.get("finish_reason"), "error": err}


if __name__ == "__main__":
    stage = sys.argv[1]
    df = pd.DataFrame(list(rows(stage)))
    path = HERE.parent / "outcomes.csv" if stage == "main" else HERE / "runs" / stage / "outcomes.csv"
    df.to_csv(path, index=False)
    print(len(df), "runs,", int(df.valid.sum()), "valid ->", path)
