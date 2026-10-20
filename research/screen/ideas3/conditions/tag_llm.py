"""LLM tagging of conditions from the policy TEXT only (one gemini-3.8-flash call per distinct policy text)."""
import os, sys, json, re
os.environ.setdefault("LLM_BACKEND", "vertex")
import subprocess
if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
    os.environ["GOOGLE_CLOUD_PROJECT"] = subprocess.run(["gcloud", "config", "get-value", "project"], capture_output=True, text=True).stdout.strip()
import llmc
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from parse import st, pol
d = pd.read_csv("conds.csv")
UN = json.load(open("units.json"))
PROMPT = """You read one business policy that a customer-service agent must follow before performing an action. The policy is split into numbered conditions. For EACH condition return tags that describe how the condition is WRITTEN (not whether it holds).

Tags (all integers):
- cross_ref: 1 if the condition depends on or refers to another part of the policy, another condition, another record or a value defined elsewhere ("as defined", "their", "the same", "previously"), else 0
- date_arith: 1 if it needs date or time arithmetic or comparison (deadlines, ranges of days, overlaps, before/after the interaction time), else 0
- calc: 1 if the agent must compute a derived number (sum, difference, percentage, product, count) before comparing, else 0
- n_lookups: how many distinct records or fields the agent must fetch to check it (0 to 4)
- n_comparisons: how many separate comparisons or sub-requirements are packed in it (1 to 5)
- conditional: 1 if it has an if/otherwise or exception ("unless", "excluding", "if ... then") structure, else 0
- vague: 1 if wording is vague or open to two readings, else 0
- precondition_style: 1 if it is a generic precondition like being logged in, authenticated or the entity existing, else 0

Policy:
\"\"\"
{policy}
\"\"\"

Conditions:
{conds}

Return ONLY a JSON list with one object per condition, in order, with keys: idx, cross_ref, date_arith, calc, n_lookups, n_comparisons, conditional, vague, precondition_style."""

def job(u):
    g = d[d.unit == u].sort_values("idx")
    conds = "\n".join(f"{int(r.idx)}. {r.text}" for _, r in g.iterrows())
    txt, _ = llmc.gen(PROMPT.format(policy=pol(st[UN[u][0]]), conds=conds), tag="tags_v1", max_tokens=4000)
    m = re.search(r"\[.*\]", txt, re.S)
    try:
        return u, json.loads(m.group(0))
    except Exception:
        return u, None

if __name__ == "__main__":
    units = list(d.unit.unique())
    if len(sys.argv) > 1: units = units[:int(sys.argv[1])]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(job, units))
    out = {u: r for u, r in res}
    print("failed", [u for u, r in res if r is None], llmc.STATS)
    json.dump(out, open("tags.json", "w"))
