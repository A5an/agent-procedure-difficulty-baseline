"""Reproduce Table 5 of Agent psychometrics (held-out benchmark: SWE-bench Pro and GSO) with their
code and data, unchanged, then the same with leak-free inputs only.

  python src/reproduce_table5.py        -> results/table5/table5.csv
                                           results/table5_statement_only/table5_statement_only.csv

1. Their default features. All ten published numbers should match to three decimals.
2. Statement only: their statement-only embedding files and the `1_problem_15` judge features that
   ship with their repository (information ablation: the judge saw the problem statement only).
   The judge CSV columns are only reordered to one common order. These rows are not in the paper;
   they are the reference for prediction from task text alone.
"""

import subprocess
import sys

import pandas as pd

from common import AP_REPO, RESULTS

EMB = {  # statement-only embedding caches in their repository
    "swebench_verified": "4c9cb2565631", "swebench_pro": "6c5de746e22f",
    "gso": "eb263fe00cea", "terminalbench": "1f4437556d21",
}


def run(out, extra):
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "experiment_new_benchmarks.run_all_datasets",
           "--heldout_datasets", "swebench_pro", "gso", "--irt_device", "cpu", "--output_dir", str(out), *extra]
    with open(out / "run.log", "w") as log:
        subprocess.run(cmd, cwd=AP_REPO, check=True, stdout=log, stderr=subprocess.STDOUT)


def main():
    run(RESULTS / "table5", ["--output", str(RESULTS / "table5" / "table5.csv")])
    print(pd.read_csv(RESULTS / "table5" / "table5.csv").iloc[:, [0, 1, 8, 15, 22, 29]].to_string(index=False))

    out = RESULTS / "table5_statement_only"
    (out / "emb").mkdir(parents=True, exist_ok=True)
    (out / "judge").mkdir(parents=True, exist_ok=True)
    for d, h in EMB.items():
        link = out / "emb" / f"{d}.npz"
        link.unlink(missing_ok=True)
        link.symlink_to(AP_REPO / "embeddings" / f"embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__{h}__maxlen8192.npz")
    src = AP_REPO / "llm_judge_features" / "information_ablation" / "{}" / "1_problem_15.csv"
    order = list(pd.read_csv(str(src).format("swebench_verified")).columns)
    for d in EMB:
        df = pd.read_csv(str(src).format(d))
        assert set(df.columns) == set(order), d
        df[order].to_csv(out / "judge" / f"{d}.csv", index=False)
    run(out, ["--embeddings_path", str(out / "emb" / "{dataset}.npz"),
              "--llm_judge_features_path", str(out / "judge" / "{dataset}.csv"),
              "--output", str(out / "table5_statement_only.csv")])
    print(pd.read_csv(out / "table5_statement_only.csv").iloc[:, [0, 1, 8, 15, 22, 29]].to_string(index=False))


if __name__ == "__main__":
    main()
