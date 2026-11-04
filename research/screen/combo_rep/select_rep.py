"""Replication of Exp 5 (PREREG.md): the 30 TauIndianBank families not used in P0-2, 6 real tasks each, 2 seed tasks
each (other tasks of the same family), excluded probe customers = customers of every task of these families.
Pilot tasks of P0-2 are never drawn. Writes selection_rep.json."""
import json, random, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
T = HERE.parent / "tib"
sys.path.insert(0, str(T))
from common import by_family, unit_class

old = json.load(open(T / "selection.json"))
bf = by_family()
fams = sorted(f for f in bf if f not in set(old["test_families"]))
pilot = set(old["pilot_tasks"])
r = random.Random("20261104-rep")
real, seeds = {}, {}
for f in fams:
    pool = sorted(t["id"] for t in bf[f] if t["id"] not in pilot)
    pick = r.sample(pool, 8)
    real[f], seeds[f] = sorted(pick[:6]), sorted(pick[6:])
cust = sorted({t["initial_state"]["initialization_data"]["agent_data"]["active_customer"] for f in fams for t in bf[f]})
out = {"seed": "20261104-rep", "test_families": fams, "real_tasks": real, "seed_tasks": seeds,
       "classes": {f: unit_class(bf[f][0]) for f in fams}, "purpose": {f: bf[f][0]["description"]["purpose"] for f in fams},
       "pilot_tasks": old["pilot_tasks"], "excluded_customers": cust}
json.dump(out, open(HERE / "selection_rep.json", "w"), indent=1)
print(len(fams), "families;", sum(len(v) for v in real.values()), "real tasks; excluded customers", len(cust))
print("classes perform/refuse:", sum(out["classes"].values()), "/", len(fams) - sum(out["classes"].values()))
