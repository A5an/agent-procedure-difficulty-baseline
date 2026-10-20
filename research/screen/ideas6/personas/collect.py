import sys, json
import plib as L
bench = sys.argv[1]; temp = float(sys.argv[2]); lim = int(sys.argv[3]) if len(sys.argv) > 3 else None
rows = [json.loads(l) for l in open(f"{L.REPO}/data/{bench}/statements.jsonl")]
items = {r["case_id"]: L.build_prompt(r["case_id"], r["text"]) for r in rows}
if temp > 0:   # stability subset: 30 distinct prompts per benchmark, fixed
    seen = {}; 
    for c, p in items.items(): seen.setdefault(p, c)
    ids = sorted(seen.values()); ids = ids[::max(1, len(ids)//30)][:30]; items = {c: items[c] for c in ids}
if lim: items = dict(list(items.items())[:lim])
res = L.run(items, temp=temp)
print(sum(1 for v in res.values() if v), "of", len(res), "ok")
