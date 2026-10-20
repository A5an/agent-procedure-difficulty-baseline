import sys, json, random
import lib
items = lib.statements()
random.Random(1).shuffle(items)
sub = [x for x in items if x[0]=="sopbench"][:6] + [x for x in items if x[0]=="tau2"][:4]
lib.run(sub, kinds=("main",), workers=5)
for b,cid,t in sub:
    r = lib.one("main", cid, t)
    print(cid, r["ok"], json.dumps(r["parsed"])[:700] if r["ok"] else r.get("raw"))
