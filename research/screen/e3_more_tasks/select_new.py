"""E3-B selection: every TauIndianBank task of the 50 seedk families that was never run as an evaluation task.
Flags: seed_own (a k1/k2/k5 seed of its own family: the probe generator saw it, excluded from the primary analysis),
seed_foreign (a seed of another family's foreign probes; kept, flagged). -> new_tasks.json"""
import json, sys, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent; SCR = HERE.parent
sys.path.insert(0, str(SCR / "tib"))
from common import load_tasks, family
SEL = json.load(open(SCR / "seedk" / "selection_seedk.json"))["families"]
evals = {t for v in SEL.values() for t in v["eval_tasks"]}
own = {f: set(v.get("seeds_k1", []) + v.get("seeds_k2", []) + v.get("seeds_k5", [])) for f, v in SEL.items()}
foreign = {t for v in SEL.values() for t in v.get("seeds_foreign", [])}
out = []
for t in load_tasks():
    f = family(t)
    if f not in SEL or t["id"] in evals: continue
    out.append({"id": t["id"], "family": f, "seed_own": t["id"] in own[f], "seed_foreign": t["id"] in foreign})
json.dump(out, open(HERE / "new_tasks.json", "w"), indent=1)
c = collections.Counter(x["family"] for x in out if not x["seed_own"])
print(len(out), "unused tasks;", sum(x["seed_own"] for x in out), "own seeds;", sum(x["seed_foreign"] for x in out), "foreign seeds;",
      "non-seed per family: min", min(c.values()), "median", sorted(c.values())[len(c) // 2], "max", max(c.values()), "families", len(c))
