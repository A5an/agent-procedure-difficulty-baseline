"""Test-retest of the ADeLe annotation: a random sample of tasks annotated a second time with the same
judge, prompt and temperature 0 (the judge still thinks, so answers can differ), compared with the
first annotation. Output results/adele_retest.json.

  python src/adele_retest.py [--offline] [--fresh]     caches in data/adele_retest/<bench>.jsonl
"""

import json
import random
import sys

import numpy as np
import pandas as pd

import adele_annotate as A
from common import DATA, RESULTS, statements

SAMPLE = {"sopbench": 30, "tau2": 15, "tauk_banking": 10}


def main(offline, fresh):
    out = {}
    for d, n in SAMPLE.items():
        items = statements(d)
        random.Random(11).shuffle(items)
        sample = [(r["case_id"], r["text"]) for r in items[:n]]
        cache = DATA / "adele_retest" / f"{d}.jsonl"
        if fresh:
            A.move_aside(cache)
        A.run(sample, cache, offline)
        second = A.levels(cache)
        first = A.levels(DATA / d / "adele_responses.jsonl").loc[second.index]
        a, b = first.values.astype(float).ravel(), second.values.astype(float).ravel()
        ok = ~np.isnan(a) & ~np.isnan(b)
        per_dem = {k: float(pd.Series(first[k]).astype(float).corr(pd.Series(second[k]).astype(float),
                                                                   method="spearman"))
                   for k in A.DEMANDS if first[k].nunique() > 1}
        out[d] = {"n_tasks": n, "exact": float((a[ok] == b[ok]).mean()),
                  "within_1": float((abs(a[ok] - b[ok]) <= 1).mean()),
                  "median_demand_spearman": float(np.nanmedian(list(per_dem.values())))}
        print(d, out[d], flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "adele_retest.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main("--offline" in sys.argv, "--fresh" in sys.argv)
