"""Step 3: baseline pipeline (fold 1PL IRT -> ridge on 18 ADeLe levels) fitted separately to two sub-outcome targets.
 checks : dirgraph_satisfied (the required checks were made)   action : action_called_correctly (final action right)
Everything is read from flags.csv.gz (step 1). Own folder holds the response matrices, full-data IRT and fold IRT caches."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, os, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
import harness as H
from scipy.stats import spearmanr, kendalltau
RB, E = H.RB, H.E
ROOT = Path(ZR.SCREEN + "/pertype/sub")
REPO = H.REPO
TARGETS = {"sop_checks": "dirgraph_satisfied", "sop_action": "action_called_correctly"}
flags = pd.read_csv("flags.csv.gz")
flags = flags[flags.has_run]
for b, col in TARGETS.items():
    d = ROOT / "data" / b
    (d / "irt").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "data/sopbench/tasks.csv", d / "tasks.csv")
    if not (d / "features").exists():
        os.symlink(REPO / "data/sopbench/features", d / "features")
    with open(d / "responses.jsonl", "w") as f:
        for a, g in flags.groupby("agent"):
            f.write(json.dumps({"subject_id": a, "responses": {c: int(bool(v)) for c, v in zip(g.case_id, g[col])}}) + "\n")
    if not (d / "irt/1d_1pl/items.csv").exists():
        import subprocess
        subprocess.run([sys.executable, "swebench_irt/train.py", "--dims", "1", "--model", "1pl", "--data_path", str(d / "responses.jsonl"),
                        "--output_dir", str(d / "irt"), "--seed", "0"], cwd=REPO / "third_party/agent-psychometrics", check=True, capture_output=True)
RB.DATA, RB.RESULTS = ROOT / "data", ROOT / "results"
E.DATA, E.RESULTS = ROOT / "data", ROOT / "results"
mode = sys.argv[1] if len(sys.argv) > 1 else "all"
if mode in ("run", "all"):
    H.run_all(ROOT / "results", lambda src, M: {}, benches=list(TARGETS), schemes=["new_procedures", "new_domain"])
if mode in ("eval", "all"):
    E.COMBOS = []; H.BENCHES = list(TARGETS); E.BENCHES = list(TARGETS); E.DATASETS = list(TARGETS)
    out = {}
    for b in TARGETS:
        for s in ["new_procedures", "new_domain"]:
            out[f"{b}|{s}"], _ = E.evaluate(b, s)
    json.dump(out, open("sub_eval.json", "w"), indent=1)
    for k, v in out.items():
        for c in ("adele", "oracle", "constant"):
            print(k, c, {m: (round(x, 3) if isinstance(x, float) else [round(y, 3) for y in x]) for m, x in v[c].items()})
    # ability rank agreement
    ab = {b: pd.read_csv(ROOT / "data" / b / "irt/1d_1pl/abilities.csv", index_col=0).theta for b in TARGETS}
    ab["sop_success"] = pd.read_csv(REPO / "data/sopbench/irt/1d_1pl/abilities.csv", index_col=0).theta
    A = pd.DataFrame(ab)
    A.to_csv("sub_abilities.csv")
    print(A.corr(method="spearman").round(3)); print(A.corr(method="kendall").round(3))
    it = {b: pd.read_csv(ROOT / "data" / b / "irt/1d_1pl/items.csv", index_col=0).b for b in TARGETS}
    it["sop_success"] = pd.read_csv(REPO / "data/sopbench/irt/1d_1pl/items.csv", index_col=0).b
    print(pd.DataFrame(it).corr(method="spearman").round(3))
