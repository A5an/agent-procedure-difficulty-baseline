"""Difficulty targets for fresh benchmarks (run only after all predictors are computed). -> targets.csv"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, glob, os, collections, numpy as np, pandas as pd
B = ZR.SCREEN + "/ideas10/scout/data"
rows = []
# TAC: per agent per task, mean of result/total over the task and its -image variant; agents with >= 170 tasks
res = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(B + "/tac_exp/evaluation/1.0.0/*/results/eval_*.json"):
    ag = f.split("/")[-3]; t = os.path.basename(f)[5:-5].replace("-image", "")
    try: d = json.load(open(f)); fs = d.get("final_score") or {}
    except Exception: continue
    if fs.get("total"): res[ag][t].append(fs["result"] / fs["total"])
ags = [a for a in res if len(res[a]) >= 170]
tasks = sorted(set(t for a in ags for t in res[a]))
for t in tasks:
    v = [np.mean(res[a][t]) for a in ags if t in res[a]]
    full = [float(np.mean(res[a][t]) >= 1.0) for a in ags if t in res[a]]
    rows.append({"key": f"tac::{t}", "bench": "tac", "n_agents": len(v), "success": float(np.mean(v)), "success_full": float(np.mean(full))})
# MCPMark: agents with >= 120 tasks, drop k2 and k2-0905 duplicates of kimi-k2-0905
res = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(B + "/mcpmark/mcpmark-v1-0905/*/run-*/*/meta.json"):
    ms = f.split("/")[-4]; model, service = ms.split("__", 1); task = f.split("/")[-2]
    try: d = json.load(open(f)); ok = bool((d.get("execution_result") or {}).get("success"))
    except Exception: continue
    res[model][f"{service}/{task}"].append(ok)
ags = [a for a in res if len(res[a]) >= 120 and a not in ("k2", "k2-0905")]
print("mcpmark agents", len(ags), sorted(ags))
for t in sorted(set(t for a in ags for t in res[a])):
    v = [np.mean(res[a][t]) for a in ags if t in res[a]]
    rows.append({"key": f"mcpmark::{t}", "bench": "mcpmark", "n_agents": len(v), "success": float(np.mean(v)), "success_full": float(np.mean(v))})
# DrafterBench: sampled cases, all 30 model files, score >= 1.0
items = {i["id"] for i in json.load(open("items.json")) if i["bench"] == "drafter"}
res = collections.defaultdict(dict)
for f in glob.glob(B + "/agentsuite_DrafterBench-fixed-trajectories/*.jsonl"):
    m = os.path.basename(f)[:-6]
    for l in open(f):
        r = json.loads(l); k = f"{r['task_name']}|{r['meta']['id']}"
        if k in items: res[m][k] = float((r.get("eval_result") or {}).get("score") or 0)
for k in sorted(items):
    v = [res[m][k] for m in res if k in res[m]]
    rows.append({"key": f"drafter::{k}", "bench": "drafter", "n_agents": len(v), "success": float(np.mean([x >= 1.0 for x in v])), "success_full": float(np.mean(v))})
pd.DataFrame(rows).to_csv("targets.csv", index=False); print(pd.DataFrame(rows).groupby("bench").agg(n=("key", "size"), agents=("n_agents", "median"), succ=("success", "mean")))
