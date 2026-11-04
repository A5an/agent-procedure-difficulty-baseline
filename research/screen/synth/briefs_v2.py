"""Briefs of the hard round (PREREG_V2.md): 12 new domains, the same briefs for every generator. Seed 20261107.

Harder than the first round: 6 to 9 rules, 2 or 3 writes, at least 2 rules that check a record the customer does not
mention, a computed argument in the main write, a counting rule in half of the briefs. Cases: no 'insufficient' level
(agents almost never failed it), the harder levels twice; behaviours changes_mind and a new 'negotiates' (asks for
something refused, then for something else) twice each.

  python briefs_v2.py   -> briefs_v2.json
"""
import json, random
from pathlib import Path
from engine import TODAY

DOMAINS = ["auto insurance claims", "mortgage servicing", "business mobile accounts", "airline loyalty program",
           "hospital billing office", "city tax office", "school district transport", "car dealership service center",
           "bike share membership", "cloud hosting billing", "freight shipping", "university housing"]
rng = random.Random(20261107)
n = len(DOMAINS)
n_rules = [6, 7, 8, 9] * 3
n_calls = [2] * 6 + [3] * 6
esc = [True, False] * (n // 2)
ver = [True, False] * (n // 2)
dbc = [True, False] * (n // 2)
for x in (n_rules, n_calls, esc, ver, dbc):
    rng.shuffle(x)
base_levels = ["upfront", "on_request", "lookup", "distracting", "conflicting", "helpful_extra"]
hard = ["on_request", "lookup", "distracting", "conflicting"]
briefs = []
for i, dom in enumerate(DOMAINS):
    levels = base_levels + rng.sample(hard, 2)
    rng.shuffle(levels)
    beh = ["changes_mind", "changes_mind", "negotiates", "negotiates", "pushy", "pushy", "terse", "cooperative"]
    rng.shuffle(beh)
    perform = set(rng.sample(range(8), 4))
    cases = [{"case_id": f"c{j + 1}", "level": l, "behavior": beh[j], "cls": "perform" if j in perform else "refuse"}
             for j, l in enumerate(levels)]
    slug = "".join(ch if ch.isalnum() else "_" for ch in dom.split()[0].lower())
    briefs.append({"procedure_id": f"h{i + 1:02d}_{slug}", "domain": dom, "today": TODAY,
                   "structure": {"n_rules": n_rules[i], "n_calls": n_calls[i], "escalation": esc[i], "verification": ver[i],
                                 "min_hidden_rules": 2, "computed_arg": True, "db_count": dbc[i]},
                   "cases": cases})
json.dump(briefs, open(Path(__file__).with_name("briefs_v2.json"), "w"), indent=1)
print(len(briefs), "briefs;", sum(len(b["cases"]) for b in briefs), "cases per generator")
