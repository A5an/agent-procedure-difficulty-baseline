"""Sanity check of judge.py before the main run (3 Nov): 6 pilot specs as written, and 3 deliberately broken policies
of one pilot spec (a threshold changed, another threshold changed, one rule's sentence deleted). Output judge_sanity.out."""
import json, re
import judge, dataset

for g in ["haiku", "sonnet", "opus"]:
    for pid, s in dataset.load_specs(g, "gens_pilot").items():
        r = judge.judge(json.loads(json.dumps(s)))
        print("clean", g, pid, "ok" if r["ok"] else "FLAGGED", r["problems"][:2])
s = dataset.load_specs("sonnet", "gens_pilot")["p02_electricity"]
pol = s["policy_text"]
for name, t in {"threshold 100 -> 50": pol.replace("100.00", "50.00"), "threshold 600 -> 400": pol.replace("600.00", "400.00"),
                "rule sentence deleted": re.sub(r"[^.\n]*default[^.\n]*\.", "", pol)}.items():
    x = json.loads(json.dumps(s)); x["policy_text"] = t
    r = judge.judge(x)
    print("broken", name, "ok" if r["ok"] else "FLAGGED", r["problems"][:1])
