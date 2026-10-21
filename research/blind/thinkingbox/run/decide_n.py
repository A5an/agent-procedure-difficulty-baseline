"""Set N (tasks per domain) for the main run from the pilot cost, by the rule in PREREG.md: all tasks if the projected
cost is at most BUDGET USD, otherwise the largest N within BUDGET; stop if N < 40.  python decide_n.py -> ../eval_set.json"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from tbpaths import ORDER  # noqa: E402
from cost import table  # noqa: E402

BUDGET, TRIALS, MIN_N = 150.0, 2, 40


def main():
    agg = table("pilot")
    runs = pd.read_csv(HERE / "runs" / "pilot" / "outcomes.csv").groupby("agent").size()
    per_run = {a: sum(c for (s, ag, _), (c, _) in agg.items() if ag == a) / runs[a] for a in runs.index}
    sizes = pd.read_csv(ORDER).groupby("domain").size()

    def projected(n):
        return sum(per_run.values()) * TRIALS * int(sizes.clip(upper=n).sum())

    n = int(sizes.max())
    while n > 0 and projected(n) > BUDGET:
        n -= 1
    rec = {"n_per_domain": n, "tasks": int(sizes.clip(upper=n).sum()), "projected_usd": round(projected(n), 2),
           "projected_all_usd": round(projected(int(sizes.max())), 2),
           "pilot_cost_per_run_usd": {a: round(c, 4) for a, c in per_run.items()}, "budget_usd": BUDGET}
    print(json.dumps(rec, indent=1))
    if n < MIN_N:
        raise SystemExit(f"N = {n} < {MIN_N}: stop and ask before spending more")
    (HERE.parent / "eval_set.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
