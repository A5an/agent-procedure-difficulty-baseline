"""ADeLe demand annotation (Zhou et al., Nature 2026) with their own rubrics, prompt and parser.

The only change from the original is the judge: Gemini (llm.MODEL, temperature 0) instead of
gpt-4o-2024-05-13. `validate` checks that change on the published ADeLe battery, where every
instance already carries GPT-4o levels.

  python src/adele_annotate.py validate [n]          n battery instances x 18 demands, compared with
                                                     the battery's GPT-4o levels
  python src/adele_annotate.py annotate <bench>      data/<bench>/statements.jsonl -> features/adele.csv
                                                     and adele16.csv (without AS and MCu, the two
                                                     demands that did not replicate in validation)
Options:
  --offline   use only the cached answers (the shipped caches are complete, so this reproduces the
              reference features exactly and needs no key)
  --fresh     move the cache aside (*.previous.jsonl) and annotate everything again with your key

Caches: data/<bench>/adele_responses.jsonl and data/adele_validation/responses.jsonl, one line per
(item, demand) with the full answer. Reruns only fill gaps, so an interrupted run can be restarted.
UG (unguessability) is not used: it is defined for multiple-choice formats.
"""

import hashlib
import json
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import llm
from common import DATA, EXTERNAL, RESULTS, ensure_newline, read_jsonl, statements, use_adele

use_adele()
from adele.annotation.parsing import extract_demand_level  # noqa: E402
from adele.annotation.prompts import build_annotation_prompt  # noqa: E402
from adele.rubrics.catalog import RubricsCatalog  # noqa: E402

CATALOG = RubricsCatalog()
DEMANDS = list(CATALOG.acronyms)  # the 18 v1.0 demands
NOT_REPLICATED = ["AS", "MCu"]
# Their limit is 1,000 completion tokens for GPT-4o. Gemini counts its internal thinking (about
# 700 to 1,000 tokens here) inside the same limit, so the cap is raised to keep the answer uncut.
MAX_TOKENS = 4000
WORKERS = 80
BATTERY_URL = ("https://media.githubusercontent.com/media/Kinds-of-Intelligence-CFI/ADeLe-AIEvaluation/"
               "a93722320b5c46a371ef94c60c4333def37bbd57/ADeLe_battery_data/ADeLe_batterry_v1dot0.csv")
BATTERY_SHA256 = "b978cdf1538d5c07e01bfb4a2636293641ac9a404b3385fdb1936fe6aec5b056"
_lock = threading.Lock()


def cached(cache_path):
    if not cache_path.exists():
        return set()
    return {(r["id"], r["demand"]) for r in read_jsonl(cache_path) if r.get("ok_call")}


def run(items, cache_path, offline=False):
    """items: list of (item_id, text). Annotate every (item, demand) not yet in the cache."""
    for attempt in range(3):  # failed calls are recorded and retried in the next pass
        done = cached(cache_path)
        todo = [(i, t, d) for i, t in items for d in DEMANDS if (i, d) not in done]
        print(f"pass {attempt + 1}: {len(done)} cached, {len(todo)} to annotate", flush=True)
        if not todo:
            return
        if offline or not llm.available():
            raise SystemExit(f"{len(todo)} annotations missing from {cache_path} and no LLM call allowed "
                             "(--offline, or no GEMINI_API_KEY / GOOGLE_CLOUD_PROJECT)")
        _pass(todo, cache_path)


def _pass(todo, cache_path):
    def one(i, t, d):
        rub = CATALOG[d]
        prompt = build_annotation_prompt(rub.full_name, rub.content, t)
        try:
            text, usage = llm.generate(prompt, max_tokens=MAX_TOKENS)
            level, ok = extract_demand_level(text)
            return {"id": i, "demand": d, "level": None if np.isnan(level) else level, "parsed": ok,
                    "ok_call": True, "response": text, "usage": usage}
        except Exception as e:  # recorded, retried on the next pass
            return {"id": i, "demand": d, "ok_call": False, "error": str(e)[:300]}

    n = 0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_newline(cache_path)
    with ThreadPoolExecutor(WORKERS) as ex, open(cache_path, "a") as f:
        for fut in as_completed([ex.submit(one, *x) for x in todo]):
            r = fut.result()
            with _lock:
                f.write(json.dumps(r) + "\n")
                f.flush()
            n += 1
            if n % 500 == 0:
                print(f"{n}/{len(todo)}", flush=True)


def levels(cache_path):
    rows = {}
    for r in read_jsonl(cache_path):
        if r.get("ok_call"):
            rows[(r["id"], r["demand"])] = r.get("level")
    s = pd.Series(rows)
    s.index = pd.MultiIndex.from_tuples(s.index, names=["id", "demand"])
    return s.unstack("demand").reindex(columns=DEMANDS)


def move_aside(cache_path):
    if cache_path.exists():
        cache_path.rename(cache_path.with_name(cache_path.stem + ".previous.jsonl"))


def battery():
    """The ADeLe battery (a Git LFS file upstream), fetched once and checked by SHA-256."""
    p = EXTERNAL / "ADeLe_batterry_v1dot0.csv"
    if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != BATTERY_SHA256:
        p.parent.mkdir(parents=True, exist_ok=True)
        print("downloading the ADeLe battery (35 MB)", flush=True)
        urllib.request.urlretrieve(BATTERY_URL, p)
        assert hashlib.sha256(p.read_bytes()).hexdigest() == BATTERY_SHA256, "battery hash mismatch"
    return pd.read_csv(p)


def validate(n, offline, fresh):
    bat = battery()
    # one instance per task across the 63 tasks first, then random, fixed seeds
    per_task = bat.groupby("task", group_keys=False).apply(lambda g: g.sample(1, random_state=0))
    rest = bat.drop(per_task.index).sample(max(0, n - len(per_task)), random_state=1)
    sample = pd.concat([per_task, rest]).head(n)
    cache = DATA / "adele_validation" / "responses.jsonl"
    if fresh:
        move_aside(cache)
    # their loader maps a benchmark's "question" column to the annotated text
    run([(str(r.instance_id), str(r.question)) for r in sample.itertuples()], cache, offline)
    ours = levels(cache).loc[sample.instance_id.astype(str)]
    ref = sample.set_index(sample.instance_id.astype(str))[DEMANDS]
    rep = {}
    for d in DEMANDS:
        a, b = ours[d].astype(float), ref[d].astype(float)
        m = a.notna() & b.notna()
        rep[d] = {"n": int(m.sum()), "exact": float((a[m] == b[m]).mean()),
                  "within_1": float(((a[m] - b[m]).abs() <= 1).mean()),
                  "spearman": float(a[m].corr(b[m], method="spearman")),
                  "mean_diff_ours_minus_gpt4o": float((a[m] - b[m]).mean())}
    allm = pd.DataFrame(rep).T
    summary = {"model": llm.MODEL, "instances": len(ours),
               "exact_mean": float(allm.exact.mean()), "within_1_mean": float(allm.within_1.mean()),
               "spearman_median": float(allm.spearman.median()), "per_demand": rep,
               "parse_failures": int(ours.isna().sum().sum())}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "adele_validation.json").write_text(json.dumps(summary, indent=2))
    print(allm.round(3).to_string())
    print(json.dumps({k: v for k, v in summary.items() if k != "per_demand"}, indent=1))


def annotate(bench, offline, fresh):
    d = DATA / bench
    items = [(r["case_id"], r["text"]) for r in statements(bench)]
    cache = d / "adele_responses.jsonl"
    if fresh:
        move_aside(cache)
    run(items, cache, offline)
    lv = levels(cache)
    lv.index.name = "task_id"
    miss = int(lv.isna().sum().sum())
    lv = lv.fillna(lv.median())  # answers without a parsable level, counted and reported
    (d / "features").mkdir(exist_ok=True)
    lv.reset_index().to_csv(d / "features" / "adele.csv", index=False)
    lv.drop(columns=NOT_REPLICATED).reset_index().to_csv(d / "features" / "adele16.csv", index=False)
    print(bench, lv.shape, "unparsed cells filled with the column median:", miss)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    offline, fresh = "--offline" in sys.argv, "--fresh" in sys.argv
    if args[0] == "validate":
        validate(int(args[1]) if len(args) > 1 else 100, offline, fresh)
    else:
        annotate(args[1], offline, fresh)
