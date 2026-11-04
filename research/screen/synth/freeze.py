"""Hash everything the agent episodes depend on, before the first episode on generated data (PREREG.md)."""
import hashlib, glob, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
files = sorted(glob.glob(str(HERE / "gens" / "*" / "*.json"))) + [str(HERE / f) for f in (
    "PREREG.md", "SPEC_FORMAT.md", "example_procedure.json", "briefs.json", "engine.py", "dataset.py", "gen.py", "judge.py",
    "run.py", "proxy.py", "agent_instruction.txt", "user_guidelines.md", "analyze.py", "audit.py", "monitor_transfer.py")]
lines = ["# FREEZE (synthetic procedures)", "", "sha256 of every input of the agent episodes and of the analysis code, written before the",
         "first episode on generated data.", "", "```"]
for f in files:
    lines.append(f"{hashlib.sha256(open(f, 'rb').read()).hexdigest()}  {Path(f).relative_to(HERE)}")
lines.append("```")
(HERE / "FREEZE.md").write_text("\n".join(lines) + "\n")
print(len(files), "files hashed")
