"""Load the accepted synthetic procedures of one generator as tau2 tasks (PREREG.md)."""
import json, os
from pathlib import Path
import engine

HERE = Path(__file__).resolve().parent
V = os.environ.get("SYNTH_V", "")  # "" = first round, "v2" = hard round (PREREG_V2.md)
SFX = f"_{V}" if V else ""
BRIEFS = {b["procedure_id"]: b for b in json.load(open(HERE / f"briefs{SFX}.json"))}
GEN_DIR = f"gens{SFX}"


def load_specs(gen, gen_dir=None):
    gen_dir = gen_dir or GEN_DIR
    out = {}
    for p in sorted((HERE / gen_dir / gen).glob("*.json")):
        s = json.load(open(p))
        if not s.get("rejected"):
            out[s["procedure_id"]] = s
    return out


def cases(gen, gen_dir=None, specs=None):
    """Yields dicts with the tau2 task, the parsed spec, the case, its plan, the solution, measured features and the
    case database. Only cases that pass every check (dropped cases are skipped)."""
    for pid, raw in (specs or load_specs(gen, gen_dir)).items():
        spec = json.loads(json.dumps(raw))
        engine.check_spec(spec, BRIEFS[pid])
        plans = {c["case_id"]: c for c in BRIEFS[pid]["cases"]}
        dropped = set(raw.get("_check", {}).get("cases_dropped", []))
        for case in spec["cases"]:
            if case["case_id"] in dropped:
                continue
            sol, tables = engine.solve(spec, case)
            feat = engine.case_features(spec, case, sol, tables)
            plan = plans[case["case_id"]]
            uid = f"{V + '_' if V else ''}{gen}__{pid}__{case['case_id']}"
            yield {"uid": uid, "gen": gen, "pid": pid, "spec": spec, "case": case, "plan": plan, "sol": sol, "feat": feat,
                   "tables": tables, "task": engine.to_task(spec, case, plan, sol, gen, uid)}
