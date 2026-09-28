"""Compare your results/ with reference_results/ (the numbers reported in EVALUATION_PROTOCOL.md).

  python src/compare_reference.py [tolerance]      default tolerance 0.005

Every number present in both files is compared; the largest absolute differences are listed. With
the shipped caches and the pinned environment the differences should be 0 up to floating point.
After a fresh annotation with your own key (make annotate FRESH=1) they will not be 0: the judge
is sampled again, so compare the conclusions, not the third decimal.
"""

import json
import sys

from common import RESULTS, ROOT

FILES = ["final_evaluation.json", "lobo.json", "target_reliability.json", "adele_retest.json",
         "adele_validation.json"]


def flatten(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flatten(v, f"{prefix}/{k}" if prefix else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flatten(v, f"{prefix}[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield prefix, float(x)


def main(tol):
    worst_all = 0.0
    for f in FILES:
        mine, ref = RESULTS / f, ROOT / "reference_results" / f
        if not mine.exists():
            print(f"{f:28s} not produced yet")
            continue
        a = dict(flatten(json.loads(mine.read_text())))
        b = dict(flatten(json.loads(ref.read_text())))
        common_keys = [k for k in a if k in b and a[k] == a[k] and b[k] == b[k]]  # skip NaN
        diffs = sorted(((abs(a[k] - b[k]), k) for k in common_keys), reverse=True)
        worst = diffs[0][0] if diffs else 0.0
        worst_all = max(worst_all, worst)
        status = "OK" if worst <= tol else "DIFFERS"
        print(f"{f:28s} {status:8s} {len(common_keys)} numbers, largest difference {worst:.4g}")
        for d, k in diffs[:5]:
            if d > tol:
                print(f"    {k}: yours {a[k]:.4f}, reference {b[k]:.4f}")
    sys.exit(0 if worst_all <= tol else 1)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.005)
