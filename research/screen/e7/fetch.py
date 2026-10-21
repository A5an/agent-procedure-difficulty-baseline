"""Fetch task texts for E7 from the Hugging Face datasets-server (JSON rows, no code from the datasets is run)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, sys, time, requests
OUT = ZR.DATA + "/e7_texts/"
SETS = {"swebench_verified": "princeton-nlp/SWE-bench_Verified", "swebench_pro": "ScaleAI/SWE-bench_Pro", "gso": "gso-bench/gso"}
KEEP = {"swebench_verified": ["instance_id", "repo", "problem_statement"],
        "swebench_pro": ["instance_id", "repo", "problem_statement"],
        "gso": ["instance_id", "repo", "api"]}
for name, ds in SETS.items():
    rows, off = [], 0
    while True:
        for k in range(6):
            r = requests.get("https://datasets-server.huggingface.co/rows", params={"dataset": ds, "config": "default", "split": "test", "offset": off, "length": 100}, timeout=120)
            if r.status_code == 200: break
            time.sleep(5 * (k + 1))
        r.raise_for_status(); js = r.json()
        batch = [{c: x["row"].get(c) for c in KEEP[name]} for x in js["rows"]]
        rows += batch; off += len(batch)
        if not batch or off >= js["num_rows_total"]: break
    with open(OUT + name + ".jsonl", "w") as f:
        for x in rows: f.write(json.dumps(x) + "\n")
    print(name, len(rows), flush=True)
