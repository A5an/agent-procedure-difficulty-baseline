"""Estimated human time, a reference predictor (idea from METR's time horizon, Kwa et al. 2025:
agent success falls with the log of the time a skilled human needs). Human times are not recorded
for these tasks, so the judge (llm.MODEL, temperature 0) estimates them from the same statement
the other predictors read. Feature: log minutes, data/<bench>/features/human_time.csv.

This is our adaptation, not a published predictor, so it is reported next to the baseline and is
not a baseline candidate. Answers that never parsed after three passes (2 telecom tasks in the
reference run) are filled with the median of their domain.

  python src/human_time.py <bench> [--offline] [--fresh]     options as in adele_annotate.py
"""

import json
import math
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import llm
from common import DATA, ensure_newline, read_jsonl, statements

WORKERS = 40
PROMPT = (
    "Below is a task that an employee at a company would carry out with the company's internal "
    "tools, following the company policy.\n\n{text}\n\n"
    "Estimate how many minutes a trained, careful employee who knows the tools and the policy would "
    "need to complete this task correctly, from reading the request to the last action. Answer with "
    'one JSON object and nothing after it: {{"minutes": <number>}}'
)


def parse(text):
    for s in reversed(re.findall(r"\{[^{}]*\}", text or "")):
        try:
            v = float(json.loads(s)["minutes"])
            return v if v > 0 else None
        except Exception:
            continue
    return None


def main(bench, offline, fresh):
    d = DATA / bench
    items = statements(bench)
    cache = d / "human_time.jsonl"
    if fresh and cache.exists():
        cache.rename(cache.with_name("human_time.previous.jsonl"))
    for _ in range(3):
        done = {r["case_id"] for r in read_jsonl(cache) if r.get("parsed")} if cache.exists() else set()
        todo = [r for r in items if r["case_id"] not in done]
        print(f"{len(done)} cached, {len(todo)} to estimate", flush=True)
        if not todo or offline or not llm.available():
            break
        lock = threading.Lock()

        def one(r):
            try:
                text, usage = llm.generate(PROMPT.format(text=r["text"]), max_tokens=3000)
                p = parse(text)
                return {"case_id": r["case_id"], "parsed": p is not None, "p": p, "response": text,
                        "usage": usage}
            except Exception as e:
                return {"case_id": r["case_id"], "parsed": False, "error": str(e)[:300]}

        ensure_newline(cache)
        with ThreadPoolExecutor(WORKERS) as ex, open(cache, "a") as f:
            for fut in as_completed([ex.submit(one, r) for r in todo]):
                with lock:
                    f.write(json.dumps(fut.result()) + "\n")
                    f.flush()
    rows = {r["case_id"]: r for r in read_jsonl(cache) if r.get("parsed")}
    df = pd.DataFrame([{"task_id": r["case_id"],
                        "log_minutes": math.log(rows[r["case_id"]]["p"]) if r["case_id"] in rows else None}
                       for r in items])
    miss = int(df.log_minutes.isna().sum())
    dom = df.task_id.str.split("/").str[0]
    df["log_minutes"] = df.groupby(dom).log_minutes.transform(lambda s: s.fillna(s.median()))
    (d / "features").mkdir(exist_ok=True)
    df.to_csv(d / "features" / "human_time.csv", index=False)
    print(bench, df.shape, "filled with the domain median:", miss)


if __name__ == "__main__":
    main(sys.argv[1], "--offline" in sys.argv, "--fresh" in sys.argv)
