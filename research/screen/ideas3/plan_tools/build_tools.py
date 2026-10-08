import json
raw=json.load(open("tools_raw.json")); out={}
for d,v in raw.items():
    out[d]="\n".join(json.dumps(x["function"],separators=(",",":")) for x in v["agent"])
json.dump(out,open("tool_text.json","w"),indent=1)
for d,t in out.items(): print(d,len(t))
