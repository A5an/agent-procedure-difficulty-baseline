import json, sys, pathlib
src, dst, name = sys.argv[1], sys.argv[2], sys.argv[3]
last = None
for line in open(src):
    try:
        r = json.loads(line)
    except Exception:
        continue
    m = r.get("message") or {}
    if m.get("role") == "assistant":
        c = m.get("content")
        if isinstance(c, list):
            t = "\n".join(x.get("text", "") for x in c if x.get("type") == "text").strip()
        else:
            t = (c or "").strip()
        if t:
            last = t
p = pathlib.Path(dst); p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(f"# Swarm agent `{name}`, round 8 (6 Oct 2026)\n\nSaved by the main session from the agent's final message.\n\n" + (last or "(no text found)") + "\n")
print(name, len(last or ""), "chars")
