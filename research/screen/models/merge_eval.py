"""Merge group outputs into out/all, then score with the harness evaluator (new methods + combinations)."""
import json, sys
from pathlib import Path
import pandas as pd
import harness as H

HERE = Path(__file__).parent
GROUPS = ["trees", "rank", "irt", "ridge", "agent", "shrunk"]
KEY = ["seed", "fold", "agent", "case_id", "y"]
BASE = ["constant", "oracle", "adele"]
for b in H.BENCHES:
    for s in H.SCHEMES:
        fs = [HERE / "out" / g / b / f"obs_{s}.csv.gz" for g in GROUPS]
        if not fs[0].exists():
            continue
        dfs = [pd.read_csv(f) for f in fs]
        m = dfs[0]
        for d in dfs[1:]:
            assert (d[KEY].values == m[KEY].values).all() and (d[BASE].values == m[BASE].values).all()
            m = pd.concat([m, d.drop(columns=KEY + BASE)], axis=1)
        (HERE / "out" / "all" / b).mkdir(parents=True, exist_ok=True)
        m.to_csv(HERE / "out" / "all" / b / f"obs_{s}.csv.gz", index=False)
NEW = {c: c for c in pd.read_csv(HERE / "out/all/sopbench/obs_new_procedures.csv.gz", nrows=1).columns if c not in KEY + BASE}
combos = [(c, "adele") for c in NEW]
extra = json.loads(sys.argv[1]) if len(sys.argv) > 1 else []
combos += [tuple(x) for x in extra]
H.evaluate_all(HERE / "out" / "all", NEW, combos)
