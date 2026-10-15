"""Run the four agents on ThinkingBox-Bench with the harness, then retry runs that ended in a system error.

  python run_tb.py pilot             last task of each domain in order.csv, 1 trial
  python run_tb.py main [agents...]  first N tasks per domain (eval_set.json), 2 trials
Needs: the ThinkingBox venv active (tb on PATH), background services running (scripts/background_tasks.sh), and
vertex_proxy.py on port 8091. Outputs: runs/<stage>/<agent>.jsonl, then .r1 and .r2 for the retry passes
(tb infer --previous-results-file copies finished rows and re-runs error rows). Env: BATCH (default 5)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from tbpaths import ORDER, TBD  # noqa: E402

AGENTS = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.8-flash"]


def tasks(stage):
    o = pd.read_csv(ORDER)
    if stage == "pilot":
        return list(o[o["rank"] == o.groupby("domain")["rank"].transform("max")].task_id)
    n = json.loads((HERE.parent / "eval_set.json").read_text())["n_per_domain"]
    return list(o[o["rank"] < n].task_id)


def errors(path):
    if not path.exists():
        return None
    return sum(1 for l in path.read_text().splitlines() if json.loads(l).get("is_system_error")
               or (json.loads(l).get("test_result") or {}).get("is_system_error", True))


def main(stage, agents):
    out = HERE / "runs" / stage
    out.mkdir(parents=True, exist_ok=True)
    tl = out / "testlist.yaml"
    tl.write_text(yaml.safe_dump(tasks(stage)))
    rep = "1" if stage == "pilot" else "2"
    for a in agents:
        cfg = HERE / "configs" / f"{stage}_{a}.yaml"
        base = ["tb", "infer", "-c", str(cfg), "--dataset", str(TBD / "dataset"), "--agent", "think",
                "--test-list", str(tl), "--repeat", rep, "-b", os.environ.get("BATCH", "5")]
        prev = out / f"{a}.jsonl"
        if not prev.exists():
            subprocess.run(base + ["-o", str(prev)], check=False)
        for k in (1, 2):
            nxt = out / f"{a}.r{k}.jsonl"
            if errors(prev) and not nxt.exists():
                subprocess.run(base + ["--previous-results-file", str(prev), "-o", str(nxt)], check=False)
            if nxt.exists():
                prev = nxt
        print(a, "final file", prev.name, "system errors left", errors(prev), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or AGENTS)
