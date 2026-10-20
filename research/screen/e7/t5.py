"""E7 analysis 4 (PREREG.md): Agent psychometrics Table 5 pipeline (held-out SWE-bench Pro and GSO, statement only) with our
zero-shot score D as a judge feature file. 89 SWE-bench Pro tasks are no longer in the public dataset, so they have no text
and get the median D of the benchmark (stated in the report; this can only lower D's AUC).
usage: t5.py   (repository interpreter; IRT fits are reused from results/table5_statement_only/irt_splits)"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import shutil, subprocess, sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).parent
REPO = Path(ZR.REPO)
AP = REPO / "third_party/agent-psychometrics"
T5 = REPO / "results/table5_statement_only"
BENCH = ["swebench_verified", "swebench_pro", "gso", "terminalbench"]


def files():
    for v in ("D", "J15D"):
        (HERE / "t5" / v / "judge").mkdir(parents=True, exist_ok=True)
    for b in BENCH:
        j = pd.read_csv(T5 / f"judge/{b}.csv").set_index("instance_id"); j.index = j.index.astype(str)
        s = pd.read_csv(HERE / f"scores_{b}.csv", index_col=0)["D"]; s.index = s.index.astype(str)
        d = s.reindex(j.index)
        print(b, "tasks", len(j), "with D", int(d.notna().sum()))
        d = d.fillna(s.median())
        pd.DataFrame({"instance_id": j.index, "p_fail_logit": d.values}).to_csv(HERE / f"t5/D/judge/{b}.csv", index=False)
        j.assign(p_fail_logit=d.values).reset_index().to_csv(HERE / f"t5/J15D/judge/{b}.csv", index=False)


def run(v):
    out = HERE / "t5" / v
    if not (out / "irt_splits").exists():
        shutil.copytree(T5 / "irt_splits", out / "irt_splits")
    cmd = [sys.executable, "-m", "experiment_new_benchmarks.run_all_datasets", "--heldout_datasets", "swebench_pro", "gso",
           "--irt_device", "cpu", "--output_dir", str(out), "--embeddings_path", str(T5 / "emb" / "{dataset}.npz"),
           "--llm_judge_features_path", str(out / "judge" / "{dataset}.csv"), "--output", str(out / "table.csv")]
    with open(out / "run.log", "w") as log:
        subprocess.run(cmd, cwd=AP, check=True, stdout=log, stderr=subprocess.STDOUT)
    t = pd.read_csv(out / "table.csv")
    print(v); print(t.iloc[:, [0, 1, 8, 15, 22, 29]].to_string(index=False))


if __name__ == "__main__":
    files()
    for v in ("D", "J15D"):
        run(v)
