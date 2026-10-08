"""Independent agent population on the same tau2 tasks (AgentSuite, 30 models, 1 trial each) -> newpop_tau2.csv"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, glob, os, collections, numpy as np, pandas as pd
A = ZR.SCREEN + "/ideas10/scout/data/agentsuite_tau2-bench-trajectories"
idm = json.load(open(ZR.REPO + "/data/tau2/id_map.json"))
tel = json.load(open(ZR.DATA + "/tau2-domains/tasks/telecom_tasks.json"))
tel_ids = [str(t["id"]) for t in tel]
res = collections.defaultdict(dict); unm = collections.Counter()
for f in glob.glob(A + "/*.jsonl"):
    m = os.path.basename(f)[:-6]
    for l in open(f):
        r = json.loads(l); d = r["task_name"]; i = str(r["meta"]["id"])
        if i.startswith(d + "_"): key = f"{d}:{i[len(d) + 1:]}"
        elif d == "telecom" and i.isdigit() and int(i) < len(tel_ids): key = f"telecom:{tel_ids[int(i)]}"
        else: key = f"{d}:{i}"
        cid = idm.get(key)
        if cid is None: unm[(m, d)] += 1; continue
        res[m][cid] = float((r.get("eval_result") or {}).get("score") or 0) >= 1.0
print("unmatched", sum(unm.values()), dict(list(unm.items())[:6]))
ms = [m for m in res if len(res[m]) >= 270]; print("models with >= 270 tasks", len(ms))
cases = sorted(set(c for m in ms for c in res[m]))
df = pd.DataFrame({"case_id": cases, "newpop_success": [np.mean([res[m][c] for m in ms if c in res[m]]) for c in cases],
                   "n": [sum(c in res[m] for m in ms) for c in cases]})
df.to_csv("newpop_tau2.csv", index=False); print(df.describe())
