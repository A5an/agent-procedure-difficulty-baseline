"""Cost of the runs from the proxy ledger, by stage, agent and role (USD, Vertex list prices per 1M tokens).
Thinking tokens are billed as output; cached input at 10% of the input price.  python cost.py [stage]"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRICE = {"gemini-2.5-flash-lite": (0.10, 0.40), "gemini-2.5-flash": (0.30, 2.50), "gemini-2.5-pro": (1.25, 10.00),
         "gemini-3.8-flash": (0.75, 3.75)}


def cost(r):
    pi, po = PRICE[r["model"]]
    return ((r["in"] - r["cached"]) * pi + r["cached"] * pi * 0.1 + (r["out"] + r["reasoning"]) * po) / 1e6


def table(stage=None):
    agg = defaultdict(lambda: [0.0, 0])
    for l in (HERE / "ledger.jsonl").read_text().splitlines():
        r = json.loads(l)
        if "error" in r or not r["tag"] or (stage and not r["tag"].startswith(stage + ".")):
            continue
        st, agent, role = r["tag"].split(".", 1)[0], *r["tag"].split(".", 1)[1].rsplit(".", 1)
        agg[(st, agent, role)][0] += cost(r)
        agg[(st, agent, role)][1] += 1
    return agg


if __name__ == "__main__":
    agg = table(sys.argv[1] if len(sys.argv) > 1 else None)
    for k, (c, n) in sorted(agg.items()):
        print(f"{k[0]:6s} {k[1]:22s} {k[2]:6s} calls {n:6d}  ${c:8.3f}")
    print(f"total ${sum(c for c, _ in agg.values()):.2f}")
