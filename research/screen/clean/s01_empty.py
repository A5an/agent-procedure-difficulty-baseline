"""Per-case empty-assistant-reply shares for all 28 configs; writes empty_cases.csv (agent,case_id,n_asst,n_empty,share,mostly_empty)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, glob, os, re, csv
SOP=ZR.DATA + "/SOPBench-d2622008"
DOM=["bank","dmv","healthcare","hotel","library","online_market","university"]
REPO=ZR.REPO
agents=[json.loads(l)["subject_id"] for l in open(f"{REPO}/data/sopbench/responses.jsonl")]
rows=[]
for d in DOM:
    td=json.load(open(f"{SOP}/data/{d}_tasks.json"))
    ids=[f"{d}/{g}/{j}" for g,l in td.items() for j,_ in enumerate(l)]
    for a in agents:
        recs=json.load(open(f"{SOP}/output/{d}/ast_{a}-dep_full-fmt_structured-tool_full-shuffle_False.json"))
        assert len(recs)==len(ids)
        for cid,r in zip(ids,recs):
            if not r["interactions"]: continue
            n=e=0
            for m in r["interactions"][0]["interaction"]:
                if not isinstance(m,dict) or "tool_call_id" in m or m.get("sender")=="user": continue
                n+=1
                c=m.get("content"); t=m.get("tool_calls")
                if str(c) in ("","None") and str(t) in ("None","[]",""): e+=1
            rows.append((a,cid,n,e,e/n if n else 0.0,int(n>0 and e/n>=0.5)))
with open("empty_cases.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["agent","case_id","n_asst","n_empty","share","mostly_empty"]); w.writerows(rows)
import pandas as pd
df=pd.read_csv("empty_cases.csv")
print(df.groupby("agent").mostly_empty.mean().sort_values(ascending=False).head(8))
print(df.groupby("agent").mostly_empty.sum().sort_values(ascending=False).head(5))
