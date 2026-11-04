"""Briefs for the synthetic procedures (PREREG.md): the same 24 briefs go to every generator model.

Each brief fixes the domain, the structure of the procedure and the plan of its 8 cases (information level, customer
behaviour, class). Balanced by construction: every procedure gets all 7 information levels plus one extra non-
insufficient level (rotating), two cases of each behaviour, 4 perform and 4 refuse cases. Seed 20261103.

  python briefs.py   -> briefs.json
"""
import json, random
from pathlib import Path
from engine import LEVELS, BEHAVIORS, TODAY

DOMAINS = ["mobile phone carrier", "electricity utility", "health insurance claims", "airline", "hotel chain",
           "car rental", "university registrar", "city parking permits", "gym and fitness club", "retail pharmacy",
           "home internet provider", "online clothing store", "food delivery app", "parcel courier",
           "residential property management", "employee HR self-service", "credit union", "pet insurance",
           "vehicle registration office", "coworking space", "concert and event ticketing", "dental clinic",
           "water utility", "software subscription billing"]

rng = random.Random(20261103)
n = len(DOMAINS)
n_rules = [2, 3, 4, 5, 6] * 5
n_calls = [1] * 10 + [2] * 8 + [3] * 6
esc = [True, False] * (n // 2)
ver = [True, False] * (n // 2)
for x in (n_rules, n_calls, esc, ver):
    rng.shuffle(x)
non_ins = [l for l in LEVELS if l != "insufficient"]
briefs = []
for i, dom in enumerate(DOMAINS):
    levels = LEVELS + [non_ins[i % len(non_ins)]]
    rng.shuffle(levels)
    beh = BEHAVIORS * 2
    rng.shuffle(beh)
    others = [j for j, l in enumerate(levels) if l != "insufficient"]
    perform = set(rng.sample(others, 4))
    cases = [{"case_id": f"c{j + 1}", "level": l, "behavior": beh[j], "cls": "perform" if j in perform else "refuse"}
             for j, l in enumerate(levels)]
    slug = "".join(ch if ch.isalnum() else "_" for ch in dom.split()[0].lower())
    briefs.append({"procedure_id": f"p{i + 1:02d}_{slug}", "domain": dom, "today": TODAY,
                   "structure": {"n_rules": n_rules[i], "n_calls": n_calls[i], "escalation": esc[i], "verification": ver[i]},
                   "cases": cases})
json.dump(briefs, open(Path(__file__).with_name("briefs.json"), "w"), indent=1)
print(len(briefs), "briefs;", sum(len(b["cases"]) for b in briefs), "cases per generator")
