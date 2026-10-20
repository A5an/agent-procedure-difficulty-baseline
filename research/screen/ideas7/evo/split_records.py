"""Split records.jsonl and targets.csv into search and heldout files (heldout = sopbench library, online_market, university and tau2 telecom)."""
import json, pandas as pd
HELD = {"library", "online_market", "university", "telecom"}
t = pd.read_csv("targets.csv"); t["heldout"] = t.domain.isin(HELD)
t[~t.heldout].drop(columns="heldout").to_csv("targets_search.csv", index=False)
t[t.heldout].drop(columns="heldout").to_csv("targets_heldout.csv", index=False)
fs, fh = open("records_search.jsonl", "w"), open("records_heldout.jsonl", "w")
for l in open("records.jsonl"):
    r = json.loads(l); (fh if r["domain"] in HELD else fs).write(l)
print((~t.heldout).sum(), t.heldout.sum())
