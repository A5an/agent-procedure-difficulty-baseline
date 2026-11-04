"""Draw the 20 test families, 6 real tasks per family, and 10 pilot tasks from other families (PREREG.md)."""
import json, random
from common import HERE, by_family, unit_class

SEED = 20261102
bf = by_family()
fams = sorted(bf)
r = random.Random(SEED)
test = sorted(r.sample(fams, 20))
real = {f: sorted(t["id"] for t in r.sample(bf[f], 6)) for f in test}
others = [f for f in fams if f not in test]
pilot_fams = r.sample(others, 5)
pilot = sorted(r.choice(bf[f])["id"] for f in pilot_fams)
used_customers = sorted({t["initial_state"]["initialization_data"]["agent_data"]["active_customer"] for f in test for t in bf[f]})
out = {"seed": SEED, "test_families": test, "real_tasks": real, "classes": {f: unit_class(bf[f][0]) for f in test},
       "purpose": {f: bf[f][0]["description"]["purpose"] for f in test}, "pilot_tasks": pilot,
       "excluded_customers": used_customers}
json.dump(out, open(HERE / "selection.json", "w"), indent=1)
print("test families:", test)
print("classes:", out["classes"])
print("pilot:", pilot, "| excluded customers", len(used_customers))
