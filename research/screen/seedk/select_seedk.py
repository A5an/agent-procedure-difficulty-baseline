"""E1 selection (PREREG.md): the 50 families of P0-2 (../tib) and the replication (../combo_rep), their 6 evaluation tasks,
existing seeds, the new seed sets k=1, k=5 and the foreign-family control, and the 5 pilot families. No model calls.

  <tau venv python> select_seedk.py  -> selection_seedk.json"""
import json, random, sys, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tib"))
from common import load_tasks, family

s1 = json.load(open(HERE.parent / "tib" / "selection.json"))
s2 = json.load(open(HERE.parent / "combo_rep" / "selection_rep.json"))
seeds1 = json.load(open(HERE.parent / "tib" / "seeded" / "seeds.json"))["seed_tasks"]
fam_tasks = collections.defaultdict(list)
for t in load_tasks(): fam_tasks[family(t)].append(t)

F = {}
for src, s, sd in (("tib", s1, seeds1), ("rep", s2, s2["seed_tasks"])):
    for f in s["test_families"]:
        F[f] = {"source": src, "purpose": s["purpose"][f], "class": s["classes"][f], "eval_tasks": list(s["real_tasks"][f]),
                "seeds_k2": list(sd[f]),
                "tools": sorted({a["name"] for t in fam_tasks[f] for a in t["evaluation_criteria"]["actions"]})}
assert len(F) == 50
for f, d in F.items():
    d["seeds_k1"] = d["seeds_k2"][:1]
    rest = sorted(t["id"] for t in fam_tasks[f] if t["id"] not in set(d["eval_tasks"]) | set(d["seeds_k2"]))
    extra = random.Random(f"20261112-k5-{f}").sample(rest, min(3, len(rest)))
    d["seeds_k5"] = d["seeds_k2"] + extra
    assert not set(d["seeds_k5"]) & set(d["eval_tasks"])


def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


for f, d in F.items():
    pool = [g for g in sorted(F) if g != f and F[g]["class"] == d["class"] and jac(F[g]["tools"], d["tools"]) < 1 / 3]
    rule = "same class, tool-set Jaccard < 1/3"
    if not pool:
        pool = [g for g in sorted(F) if g != f and jac(F[g]["tools"], d["tools"]) < 1 / 3]; rule = "any class, tool-set Jaccard < 1/3"
    g = random.Random(f"20261112-foreign-{f}").choice(pool)
    d["foreign_family"], d["foreign_rule"], d["seeds_foreign"] = g, rule, list(F[g]["seeds_k2"])
    assert not set(d["seeds_foreign"]) & set(d["eval_tasks"])

pilot = random.Random("20261112-pilot").sample(sorted(F), 5)
out = {"seed_strings": {"k5": "20261112-k5-<family>", "foreign": "20261112-foreign-<family>", "pilot": "20261112-pilot"},
       "pilot_families": pilot, "excluded_customers": {"tib": s1["excluded_customers"], "rep": s2["excluded_customers"]},
       "families": F}
json.dump(out, open(HERE / "selection_seedk.json", "w"), indent=1)
print("pilot", pilot)
print("k5 short:", {f: len(d["seeds_k5"]) for f, d in F.items() if len(d["seeds_k5"]) < 5})
print("foreign:", {f: d["foreign_family"] for f, d in F.items()})
print("relaxed rule:", [f for f, d in F.items() if d["foreign_rule"].startswith("any")])
