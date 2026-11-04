"""Deviation fix (REPORT.md): score probe episodes with the benchmark's own per-tool argument comparison.

The generator gave compare_args only for writes, so every read in a probe was compared on all arguments (for example
the exact free-text query of search_knowledge_base), which the benchmark itself never does. probes_v2.json applies, to
every probe action, the most common compare_args the benchmark uses for that tool (a scoring convention per tool, no
task content). Trajectories do not depend on compare_args, so the recorded probe episodes are re-scored, not re-run.

  python rescore.py make            -> probes_v2.json
  python rescore.py <agent> [file]  -> runs/probes_<agent>.v2.jsonl (and a check that v1 scoring reproduces the run)
"""
import collections, json, sys
from common import HERE, load_tasks
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import evaluate_simulation, EvaluationType

NONE = "__all__"


def convention():
    c = collections.defaultdict(collections.Counter)
    for t in load_tasks():
        for a in t["evaluation_criteria"]["actions"]:
            c[a["name"]][json.dumps(a["compare_args"]) if "compare_args" in a else NONE] += 1
    return {k: v.most_common(1)[0][0] for k, v in c.items()}


def make():
    conv = convention()
    probes = json.load(open(HERE / "probes.json"))
    for p in probes:
        for a in p["evaluation_criteria"]["actions"]:
            rule = conv.get(a["name"])
            if rule is None: continue                      # tool never used by the benchmark: keep as generated
            if rule == NONE: a.pop("compare_args", None)
            else: a["compare_args"] = json.loads(rule)
    json.dump(probes, open(HERE / "probes_v2.json", "w"), indent=1)
    json.dump(conv, open(HERE / "compare_convention.json", "w"), indent=1)
    print("probes_v2.json written;", {k: v for k, v in conv.items()})


def score(task, rec, msgs):
    sim = SimulationRun.model_validate({"id": "r", "task_id": task["id"], "start_time": "", "end_time": "", "duration": 0.0,
                                        "termination_reason": rec["termination"].split(".")[-1].lower(), "messages": msgs})
    ri = evaluate_simulation(simulation=sim, task=Task.model_validate(task), evaluation_type=EvaluationType.ALL,
                             solo_mode=False, domain="indian_banking")
    return float(ri.reward)


def rescore(agent):
    v1 = {p["id"]: p for p in json.load(open(HERE / "probes.json"))}
    v2 = {p["id"]: p for p in json.load(open(HERE / "probes_v2.json"))}
    out, mism = [], 0
    for l in open(HERE / "runs" / f"probes_{agent}.jsonl"):
        r = json.loads(l)
        if r.get("reward") is None: continue
        msgs = json.load(open(HERE / "runs" / "msgs" / f"probes_{agent}" / f"{r['task_id']}.json"))
        r1 = score(v1[r["task_id"]], r, msgs)
        mism += int(r1 != r["reward"])
        r2 = score(v2[r["task_id"]], r, msgs)
        out.append({**r, "reward_v1_rescored": r1, "reward": r2})
    with open(HERE / "runs" / f"probes_{agent}.v2.jsonl", "w") as f:
        for r in out: f.write(json.dumps(r) + "\n")
    print(agent, "episodes", len(out), "v1 rescoring mismatches", mism,
          "success v1", sum(r["reward_v1_rescored"] for r in out) / max(1, len(out)), "v2", sum(r["reward"] for r in out) / max(1, len(out)))


if __name__ == "__main__":
    make() if sys.argv[1] == "make" else rescore(sys.argv[1])
