"""Paths and small helpers shared by every script."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
EXTERNAL = Path(os.environ.get("EXTERNAL_DIR", ROOT / "external_data"))
AP_REPO = ROOT / "third_party" / "agent-psychometrics"
ADELE_REPO = ROOT / "third_party" / "ADeLe-AIEvaluation"
BENCHES = ["sopbench", "tau2", "tauk_banking"]
SCHEMES = ["random_cases", "new_procedures", "new_domain"]


def use_agent_psychometrics():
    """Make the pinned agent-psychometrics code importable (their modules, unchanged)."""
    if str(AP_REPO) not in sys.path:
        sys.path.insert(0, str(AP_REPO))


def use_adele():
    """Make the pinned ADeLe code importable (their rubrics, prompt builder and parser)."""
    if str(ADELE_REPO / "src") not in sys.path:
        sys.path.insert(0, str(ADELE_REPO / "src"))


def load_env(path=ROOT / ".env"):
    """KEY=value lines from .env into os.environ (values already set in the shell win)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()


def read_jsonl(path):
    """Lines of a JSONL cache; a line cut short by an interrupted write is skipped (and redone)."""
    out = []
    for line in Path(path).read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def ensure_newline(path):
    """Close a line left open by an interrupted write before appending to a cache."""
    p = Path(path)
    if p.exists() and p.stat().st_size and p.read_bytes()[-1:] != b"\n":
        with open(p, "a") as f:
            f.write("\n")


def statements(bench):
    return [json.loads(l) for l in (DATA / bench / "statements.jsonl").read_text().splitlines()]
