"""How reliable is the target itself?

1. The full-data IRT refitted with two other seeds: within-domain rank agreement with the reference fit.
2. Split-half reliability: the agents split at random into two halves (3 times), IRT fitted on each
   half, difficulties compared inside each domain; Spearman-Brown gives the reliability of a
   difficulty estimated from all agents. Its square root bounds the correlation any predictor of
   difficulty can reach.

  python src/check_target.py        -> results/target_reliability.json  (about 30 IRT fits)
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import BENCHES, DATA, RESULTS
from fit_irt import fit as fit_irt


def fit(resp_path, out_dir, seed):
    for s in [seed, seed + 10, seed + 20]:  # a failed fit is retried with another seed
        try:
            fit_irt(resp_path, out_dir, s, quiet=True)
        except Exception:
            continue
        f = Path(out_dir) / "1d_1pl" / "items.csv"
        if f.exists():
            return pd.read_csv(f, index_col=0).b
    raise RuntimeError("IRT failed")


def within(a, b, dom):
    df = pd.DataFrame({"a": a, "b": b, "d": dom}).dropna()
    v, w = [], []
    for _, g in df.groupby("d"):
        if len(g) > 2:
            v.append(spearmanr(g.a, g.b).correlation)
            w.append(len(g))
    return float(np.average(v, weights=w))


def main():
    out = {}
    RESULTS.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=RESULTS))
    for d in BENCHES:
        tasks = pd.read_csv(DATA / d / "tasks.csv").set_index("case_id")
        resp = DATA / d / "responses.jsonl"
        base = pd.read_csv(DATA / d / "irt/1d_1pl/items.csv", index_col=0).b
        dom = tasks.domain.reindex(base.index)
        seeds = [within(base, fit(resp, tmp / f"{d}_s{s}", s).reindex(base.index), dom) for s in (3, 5)]
        rows = [json.loads(l) for l in open(resp)]
        halves = []
        for rep in range(3):
            idx = np.random.default_rng(rep).permutation(len(rows))
            h = len(rows) // 2
            bs = []
            for part, sel in (("a", idx[:h]), ("b", idx[h:])):
                p = tmp / f"{d}_half{rep}{part}.jsonl"
                p.write_text("".join(json.dumps(rows[i]) + "\n" for i in sel))
                bs.append(fit(p, tmp / f"{d}_half{rep}{part}", 0).reindex(base.index))
            halves.append(within(bs[0], bs[1], dom))
        r = float(np.mean(halves))
        out[d] = {"reseed_within_domain_rho": seeds, "split_half_rho": halves, "split_half_mean": r,
                  "spearman_brown_full": 2 * r / (1 + r)}
        print(d, json.dumps(out[d]), flush=True)
    (RESULTS / "target_reliability.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
