"""Merge every output file of one agent into one ordered set of runs, and write a resume file for tb infer.

Why: tb infer --previous-results-file walks the previous file in step with the tests and drops every previous row
that does not match the current test. Rows of runs that crashed carry repetition None, so after the first crash all
later finished runs were dropped and run again. Here each (repetition, task) slot of the test order gets the first
valid run found in the agent's files, oldest file first (a rule that does not look at outcomes); slots without a
valid run get an error row with the right key, so tb re-runs exactly those.

  python resume.py <stage> <agent>      prints valid slots / all slots"""
import copy
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def valid(r):
    tr = r.get("test_result") or {}
    return not r.get("is_system_error") and bool(tr) and not tr.get("is_system_error")


def order(stage):
    reps = 1 if stage == "pilot" else 2
    tasks = yaml.safe_load((HERE / "runs" / stage / "testlist.yaml").read_text())
    return [(rep, t) for t in tasks for rep in range(reps)]


def files(stage, agent):
    d = HERE / "runs" / stage
    fs = list(d.glob(f"{agent}.jsonl")) + list(d.glob(f"{agent}.r[0-9]*.jsonl")) + list((d / "aborted").glob(f"{agent}.*.jsonl"))
    return sorted(fs, key=lambda f: f.stat().st_mtime)


def merged(stage, agent):
    """[(key, row or None)] in test order; row is the first valid run of that slot, else None."""
    best, template = {}, None
    for f in files(stage, agent):
        for l in f.read_text().splitlines():
            r = json.loads(l)
            if not valid(r):
                template = template or r
                continue
            key = ((r.get("metadata") or {}).get("repetition"), r["uid"])
            best.setdefault(key, r)
    return [(k, best.get(k)) for k in order(stage)], template


def write_resume(stage, agent, path):
    rows, template = merged(stage, agent)
    with open(path, "w") as f:
        for (rep, uid), r in rows:
            if r is None:
                if template is None:
                    break  # nothing to stand in for the missing slot: tb runs this slot and all later ones
                r = copy.deepcopy(template)
                r["uid"], r["is_system_error"], r["test_result"] = uid, True, None
                r.setdefault("metadata", {})["repetition"] = rep
            f.write(json.dumps(r) + "\n")
    return sum(r is not None for _, r in rows), len(rows)


if __name__ == "__main__":
    rows, _ = merged(sys.argv[1], sys.argv[2])
    print(sys.argv[2], "valid slots", sum(r is not None for _, r in rows), "of", len(rows))
