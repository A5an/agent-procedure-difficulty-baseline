"""Write FREEZE.md: sha256 of everything fixed before the first test episode (PREREG.md)."""
import hashlib, json, time
from common import HERE

FILES = ["PREREG.md", "selection.json", "probes.json", "mf_params.pt", "analyze.py", "run.py", "gen_probes.py", "common.py",
         "proxy.py"]
probes = json.load(open(HERE / "probes.json"))
lines = ["# Freeze (written by freeze.py before the first test episode)", "",
         f"Time: {time.strftime('%Y-%m-%d %H:%M:%S %z')}", f"Probes: {len(probes)}", "", "| file | sha256 |", "|---|---|"]
for f in FILES:
    lines.append(f"| {f} | {hashlib.sha256((HERE / f).read_bytes()).hexdigest()} |")
(HERE / "FREEZE.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
