"""Run the four agents on ThinkingBox-Bench with the harness, then retry runs that ended in a system error.

  python run_tb.py pilot             last task of each domain in order.csv, 1 trial
  python run_tb.py main [agents...]  first N tasks per domain (eval_set.json), 2 trials
Needs: the ThinkingBox venv active (tb on PATH), background services running (scripts/background_tasks.sh), and
vertex_proxy.py on port 8091. Outputs: runs/<stage>/<agent>.jsonl, then .r1, .r2, ... for up to four retry passes,
each fed with a resume file from resume.py (finished runs copied, the rest re-run). Env: BATCH (default 5)."""
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
import resume  # noqa: E402

AGENTS = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.8-flash"]


def tasks(stage):
    o = pd.read_csv(ORDER)
    if stage == "pilot":
        return list(o[o["rank"] == o.groupby("domain")["rank"].transform("max")].task_id)
    n = json.loads((HERE.parent / "eval_set.json").read_text())["n_per_domain"]
    return list(o[o["rank"] < n].task_id)


def main(stage, agents):
    out = HERE / "runs" / stage
    out.mkdir(parents=True, exist_ok=True)
    tl = out / "testlist.yaml"
    if not tl.exists():
        tl.write_text(yaml.safe_dump(tasks(stage)))
    rep = "1" if stage == "pilot" else "2"
    for a in agents:
        cfg = HERE / "configs" / f"{stage}_{a}.yaml"
        base = ["tb", "infer", "-c", str(cfg), "--dataset", str(TBD / "dataset"), "--agent", "think",
                "--test-list", str(tl), "--repeat", rep, "-b", os.environ.get("BATCH", "5")]
        first = out / f"{a}.jsonl"
        if not first.exists() or first.stat().st_size == 0:
            subprocess.run(base + ["-o", str(first)], check=False)
        for _ in range(4):  # retry passes: re-run only the slots without a valid run (see resume.py)
            ok, n = resume.write_resume(stage, a, out / f"{a}.resume.jsonl")
            print(a, "valid", ok, "of", n, flush=True)
            if ok == n:
                break
            k = 1 + max([int(f.name.split(".r")[-1].split(".")[0]) for f in out.glob(f"{a}.r[0-9]*.jsonl")] or [0])
            subprocess.run(base + ["--previous-results-file", str(out / f"{a}.resume.jsonl"),
                                   "-o", str(out / f"{a}.r{k}.jsonl")], check=False)
        print(a, "final: valid", *resume.write_resume(stage, a, out / f"{a}.resume.jsonl"), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or AGENTS)
