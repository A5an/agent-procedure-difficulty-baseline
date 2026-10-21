import sys, json
import pp_lib as L
model = sys.argv[1]; bench = sys.argv[2]; lim = int(sys.argv[3]) if len(sys.argv) > 3 else None
R = L.REPO
rows = [json.loads(l) for l in open(f"{R}/data/{bench}/statements.jsonl")]
items = {r["case_id"]: L.build_prompt(r["case_id"], r["text"]) for r in rows}
if lim: items = dict(list(items.items())[:lim] if bench == "sopbench" else list(items.items())[::max(1, len(items)//lim)][:lim])
res = L.run(model, items, limit_calls=lim)
print(sum(1 for v in res.values() if v), "of", len(res), "ok")
