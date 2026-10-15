"""Frozen predictor D: one direct question to Sonnet per task, nothing fitted.

The prompt is the tau2 prompt D of round 10 (research/screen/ideas10/ablate) with four wording changes for this
benchmark: "customer-service agent" -> "support agent", "the customer" -> "the user", "company database" -> "company
backend", and the agent generation names the four agents of this test. Score = logit(1 - p), p clipped to [0.02, 0.98];
larger score = harder task.

  python forecast.py test     3 tasks, printed
  python forecast.py run      all 507 -> predictions_D.csv (NW workers, default 6)"""
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from tbpaths import DOMAINS, HERE, ITEMS

PROMPT = """A tool-using LLM support agent of the 2025-2026 generation (models such as Gemini 2.5 Flash-Lite, Gemini 2.5 Flash, Gemini 2.5 Pro and Gemini 3.8 Flash) will handle the case below, in a live multi-turn conversation with the user, calling the listed tools against the company backend, which neither you nor the agent can see in advance. The case is graded strictly: the agent must follow the policy exactly (every required check or confirmation, in order, and the right final action or refusal), and the final backend state must be correct.

=== POLICY ===
{policy}
=== END POLICY ===

=== TOOLS ===
{tools}
=== END TOOLS ===

=== CUSTOMER REQUEST OR SCENARIO ===
{request}
=== END ===

Estimate the probability that a typical agent of that generation fully succeeds on this case. Return ONLY a JSON object, no prose: {{"p_success": probability between 0 and 1}}"""
CMD = ["claude", "-p", "--model", "sonnet", "--tools", "", "--setting-sources", "", "--no-session-persistence",
       "--output-format", "json"]
CACHE = HERE / "sonnet_cache.jsonl"
WD = HERE / "work"
LO, HI = 0.02, 0.98
_lock = threading.Lock()


def prompts():
    dom = json.loads(DOMAINS.read_text())
    out = {}
    for l in ITEMS.read_text().splitlines():
        it = json.loads(l)
        d = dom[it["scenario_id"]]
        out[it["task_id"]] = PROMPT.format(policy=d["policy"], tools=d["tools"], request=it["request"])
    return out


def load_cache():
    c = {}
    if CACHE.exists():
        for l in CACHE.read_text().splitlines():
            d = json.loads(l)
            c[d["k"]] = d
    return c


def call(prompt, cache):
    k = hashlib.sha256(prompt.encode()).hexdigest()
    if k in cache:
        return cache[k]
    WD.mkdir(exist_ok=True)
    for attempt in range(6):
        try:
            p = subprocess.run(CMD, input=prompt, cwd=WD, capture_output=True, text=True, timeout=900)
            res = json.loads(p.stdout)
        except Exception as e:
            res = {"is_error": True, "result": f"exc {e}"}
        if not res.get("is_error") and res.get("result"):
            rec = {"k": k, "r": res["result"], "model": ",".join(res.get("modelUsage", {}) or []),
                   "cost_eq": res.get("total_cost_usd")}
            with _lock:
                cache[k] = rec
                with open(CACHE, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            return rec
        txt = json.dumps(res)[:800].lower()
        time.sleep(min(300, 30 * (attempt + 1)) if any(w in txt for w in ("limit", "rate", "overload", "429")) else 10)
    return None


def parse(text):
    m = re.search(r'"p_success"\s*:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)', text or "")
    return float(m.group(1)) if m else None


def score(p):
    q = min(HI, max(LO, p))
    return math.log((1 - q) / q)


def main(mode):
    P = prompts()
    ids = sorted(P)
    if mode == "test":
        ids = ids[::200][:3]
    cache = load_cache()
    with ThreadPoolExecutor(int(os.environ.get("NW", "6"))) as ex:
        recs = list(ex.map(lambda t: call(P[t], cache), ids))
    rows = []
    for t, r in zip(ids, recs):
        p = parse(r["r"]) if r else None
        rows.append({"task_id": t, "p_success": p, "score_D": None if p is None else score(p),
                     "prompt_sha256": hashlib.sha256(P[t].encode()).hexdigest(), "model": r and r["model"]})
    df = pd.DataFrame(rows)
    df["filled"] = df.score_D.isna()
    df["score_D"] = df.score_D.fillna(df.score_D.median())  # no parsable answer after six attempts (PREREG)
    if mode == "test":
        print(df.to_string())
        return
    df.to_csv(HERE / "predictions_D.csv", index=False)
    print("done", len(df), "missing (median filled)", int(df.filled.sum()),
          "cost_eq", round(sum((r or {}).get("cost_eq") or 0 for r in recs), 2))


if __name__ == "__main__":
    main(sys.argv[1])
