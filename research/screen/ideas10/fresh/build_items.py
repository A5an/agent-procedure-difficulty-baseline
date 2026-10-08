"""Task texts for the fresh benchmarks (no outcomes read here except agent/task keys). -> items.json"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, glob, os, random, collections
B = ZR.SCREEN + "/ideas10/scout/data"
items = []
# TAC
for t in sorted(os.listdir(B + "/tac_tasks/workspaces/tasks")):
    p = f"{B}/tac_tasks/workspaces/tasks/{t}/task.md"
    if os.path.exists(p): items.append({"bench": "tac", "id": t, "group": t.split("-")[0], "text": open(p).read().strip()})
# MCPMark: keys from meta.json paths of the agents
keys = set()
for f in glob.glob(B + "/mcpmark/mcpmark-v1-0905/*/run-*/*/meta.json"):
    model_service = f.split("/")[-4]; task = f.split("/")[-2]
    keys.add((model_service.split("__", 1)[1], task))
desc = {}
for p in glob.glob(B + "/mcpmark_tasks/tasks/*/*/*/*/description.md"):
    parts = p.split("/"); service, cat, task = parts[-5], parts[-3], parts[-2]
    desc.setdefault((service, f"{cat}__{task}"), []).append(p)
miss = 0
for service, task in sorted(keys):
    c = desc.get((service, task)) or desc.get((service + "_webarena", task))
    if not c: miss += 1; continue
    items.append({"bench": "mcpmark", "id": f"{service}/{task}", "group": service, "text": open(c[0]).read().strip()})
print("mcpmark keys", len(keys), "matched", len(keys) - miss)
# DrafterBench: 20 per task type, seed 0, from one model file (instructions are identical across files)
recs = [json.loads(l) for l in open(B + "/agentsuite_DrafterBench-fixed-trajectories/gpt-5.jsonl")]
bytype = collections.defaultdict(list)
for r in recs: bytype[r["task_name"]].append(r)
rnd = random.Random(0)
for tt, rs in sorted(bytype.items()):
    for r in rnd.sample(rs, min(20, len(rs))):
        items.append({"bench": "drafter", "id": f"{tt}|{r['meta']['id']}", "group": tt, "text": r["messages"][0]["content"].strip()})
json.dump(items, open("items.json", "w"), indent=0)
print(collections.Counter(i["bench"] for i in items))
